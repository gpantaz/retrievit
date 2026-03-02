from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXMLP,
    GPTNeoXPreTrainedModel,
    eager_attention_forward,
)
from transformers.utils import logging

from retrievit.datamodels import CausalLMOutputWithPastWithCorrect
from retrievit.trainer.evaluator import Evaluator

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.get_logger(__name__)


class GPTNeoXNoPEAttention(nn.Module):
    def __init__(self, config, layer_idx=None) -> None:
        super().__init__()
        self.config = config
        self.head_size = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        # partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        # self.rotary_ndims = int(self.head_size * partial_rotary_factor)
        self.scaling = self.head_size**-0.5
        self.is_causal = True
        self.layer_idx = layer_idx

        self.query_key_value = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: dict[str, Any],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)

        qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        # cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Cache QKV values
        if layer_past is not None:
            cache_kwargs = {
                # "sin": sin,
                # "cos": cos,
                # "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            key_states, value_states = layer_past.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":  # noqa: SLF001
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]  # noqa: SLF001

        # Compute attention
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        # Reshape outputs and final projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)

        return attn_output, attn_weights


class GPTNeoXLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = GPTNeoXNoPEAttention(config, layer_idx)
        self.mlp = GPTNeoXMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = False,
        layer_past: Cache | None = None,
        output_attentions: bool | None = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: dict[str, Any],
    ):
        attn_output, attn_weights = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_output = self.post_attention_dropout(attn_output)

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GPTNeoXModel(GPTNeoXPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [GPTNeoXLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.rotary_emb = GPTNeoXRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(  # type: ignore[report]
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        converted_head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # Flex Attention converts it to a separate mask
        if head_mask is not None:
            converted_head_mask = (
                ~converted_head_mask.bool() * torch.finfo(inputs_embeds.dtype).min
            )
            converted_head_mask = converted_head_mask.to(dtype=self.dtype, device=self.device)
        head_mask = converted_head_mask

        hidden_states = self.emb_dropout(inputs_embeds)

        # create position embeddings to be shared across the decoder layers
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = None

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                layer_past=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = outputs[0]

            if output_attentions:
                all_attentions = (*all_attentions, outputs[1])

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value) -> None:
        self.embed_in = value


class TransformerNoPE(GPTNeoXPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"embed_out.weight": "gpt_neox.embed_in.weight"}
    _tp_plan = {"embed_out": "colwise_rep"}
    _pp_plan = {"embed_out": (["hidden_states"], ["logits"])}

    def __init__(self, config, tokenizer) -> None:
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.config = config
        self.evaluator = Evaluator(config=config, tokenizer=tokenizer)

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings) -> None:
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: dict[str, Any],
    ) -> CausalLMOutputWithPastWithCorrect:
        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.embed_out(hidden_states[:, slice_indices, :])

        loss = None
        correct = None
        correct_per_position = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

            # Compute correct predictions only during evaluation
            if not self.training:
                correct, correct_per_position = self.evaluator(
                    input_ids=input_ids,
                    labels=labels,
                    logits=logits,
                    task_id=kwargs.get("task"),
                    raw_target=kwargs.get("raw_target"),
                    bp=loss.item() <= 0.05,
                )

        return CausalLMOutputWithPastWithCorrect(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            correct=correct,
            correct_per_position=correct_per_position,
        )

    def evaluate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        max_new_tokens: int = 5,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Evaluate the model on a given input and labels."""
        current_input_ids = input_ids.clone().to(self.device)
        current_attention_mask = attention_mask.clone() if attention_mask is not None else None
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                outputs = self.forward(
                    input_ids=current_input_ids, attention_mask=current_attention_mask
                )
                logits = outputs.logits[:, -1, :]
                predicted_id = torch.argmax(logits, dim=-1, keepdim=True)
                current_input_ids = torch.cat([current_input_ids, predicted_id], dim=1)
                # add a one-column to the attention mask
                if current_attention_mask is not None:
                    ones = current_attention_mask.new_ones((current_attention_mask.size(0), 1))
                    current_attention_mask = torch.cat([current_attention_mask, ones], dim=1)

        return current_input_ids
