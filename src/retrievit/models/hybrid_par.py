import math
from functools import partial
from typing import Any

import torch
from mamba_ssm.models.mixer_seq_simple import create_block
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.utils.generation import GenerationMixin
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPTNeoXConfig, PretrainedConfig
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXMLP,
    GPTNeoXRotaryEmbedding,
)
from transformers.processing_utils import Unpack

from retrievit.datamodels import CausalLMOutputWithPastWithCorrect
from retrievit.trainer.evaluator import Evaluator

# try:
#     from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
) -> None:
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer adapted from GPT-NeoX."""

    def __init__(self, config: PretrainedConfig | dict[str, Any], layer_idx: int) -> None:
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        # self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
        # eps=config.layer_norm_eps)
        if config.rms_norm:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = GPTNeoXAttention(config, layer_idx)
        self.mlp = GPTNeoXMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = False,
        layer_past: Cache | None = None,
        # output_attentions: bool | None = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the Transformer Decoder Layer."""
        residual_pre_ln = hidden_states
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
        residual_post_attn = attn_output

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

        # outputs = (hidden_states,)
        # if output_attentions:
        #     outputs += (attn_weights,)

        return (hidden_states, residual_pre_ln, residual_post_attn)


class HybridBlock(nn.Module):
    """Hybrid Parallel Block combining Mamba and Transformer layers."""

    def __init__(
        self,
        layer_idx: int,
        d_model: int,
        d_intermediate: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        transformer_config=None,
        factory_kwargs=None,
        is_reverse: bool = False,
    ) -> None:
        super().__init__()
        self.mamba_block = create_block(
            d_model,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=layer_idx,
            **factory_kwargs,
        )
        self.transformer = TransformerDecoderLayer(transformer_config, layer_idx=layer_idx)

        # self.gate_fn = nn.Sigmoid()

        # 0.5493061443340549
        # self.beta_param = torch.nn.Parameter(torch.zeros(0.5493061443340549))

        self.gate_fn = nn.Tanh()
        # self.beta_param = torch.nn.Parameter(torch.zeros(d_model))
        self.beta_param = torch.nn.Parameter(torch.zeros(1))
        self.is_reverse = is_reverse

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position=None,
    ) -> torch.Tensor:
        # hidden_states, present, (attn_weights)
        transformer_outputs, _, _ = self.transformer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            head_mask=None,
            use_cache=use_cache,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        mamba_outputs, _ = self.mamba_block(hidden_states)

        gated = self.gate_fn(self.beta_param)
        # hidden_states = (1 - gated) * transformer_outputs + gated * mamba_outputs
        # breakpoint()
        if self.is_reverse:
            return transformer_outputs + gated * mamba_outputs
        return mamba_outputs + gated * transformer_outputs


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        transformer_config: GPTNeoXConfig,
        ssm_cfg: dict[str, Any] | None = None,
        attn_layer_idx: list[int] | None = None,
        attn_cfg: dict[str, Any] | None = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg: dict[str, Any] | None = None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        is_reverse: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.transformer_config = transformer_config
        self.is_reverse = is_reverse

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm and (layer_norm_fn is None or rms_norm_fn is None):
            raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        blocks = [
            HybridBlock(
                i,
                d_model,
                d_intermediate,
                ssm_cfg,
                attn_layer_idx,
                attn_cfg,
                norm_epsilon,
                rms_norm,
                initializer_cfg,
                fused_add_norm,
                residual_in_fp32,
                device,
                dtype,
                transformer_config,
                factory_kwargs,
                is_reverse,
            )
            for i in range(n_layer)
        ]

        self.layers = nn.ModuleList(blocks)

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.rotary_emb = GPTNeoXRotaryEmbedding(config=transformer_config)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs: dict[str, Any],
    ) -> dict[int, Any]:
        """Allocate inference cache for each layer."""
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inference_params: dict | None = None,
        **mixer_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Forward pass for the model."""
        hidden_states = self.embedding(input_ids)

        cache_position = torch.arange(
            0,
            hidden_states.shape[1],
            device=hidden_states.device,
        )
        position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.transformer_config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        residual = None
        for layer in self.layers:
            residual = hidden_states
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

            # TODO: Note sure if this is needed
            hidden_states = hidden_states + residual

        # for layer in self.layers:
        #     hidden_states, residual = layer(
        #         hidden_states, residual, inference_params=inference_params, **mixer_kwargs
        #     )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        return hidden_states


class HybridPar(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: PretrainedConfig,
        initializer_cfg: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config
        self.transformer_config = GPTNeoXConfig(**config.transformer)
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            transformer_config=self.transformer_config,
            **factory_kwargs,
            is_reverse=self.config.reverse if hasattr(self.config, "reverse") else False,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

        self.evaluator = Evaluator(config=config, tokenizer=tokenizer)
        self.vocab_size = self.lm_head.weight.shape[0]

    def tie_weights(self) -> None:
        """Tie the weights of the LM head to the input embeddings."""
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs: dict[str, Any],
    ) -> Any:
        """Allocate inference cache for the backbone model."""
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,  # noqa: ARG002
        inference_params: dict | None = None,
        num_last_tokens: int = 0,
        **mixer_kwargs: dict[str, Any],
    ) -> CausalLMOutputWithPastWithCorrect:
        """Forward pass for the model."""
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inference_params=inference_params,
            **mixer_kwargs,
        )
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        lm_logits = self.lm_head(hidden_states)
        loss = None
        correct = None
        correct_per_position = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # Compute correct predictions only during evaluation
            if not self.training:
                correct, correct_per_position = self.evaluator(
                    input_ids=input_ids,
                    labels=labels,
                    logits=lm_logits,
                    task_id=mixer_kwargs.get("task"),
                    raw_target=mixer_kwargs.get("raw_target"),
                )

        return CausalLMOutputWithPastWithCorrect(
            loss=loss,
            logits=lm_logits,
            correct=correct,
            correct_per_position=correct_per_position,
        )

    def get_input_embeddings(self) -> torch.nn.Module:
        """Get input embeddings.

        This is required at some point during training to find the device of the model.
        """
        return self.backbone.embedding

    def evaluate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        max_new_tokens: int = 5,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Evaluate the model on a given input and labels."""
        device = self.get_input_embeddings().weight.device
        current_input_ids = input_ids.clone().to(device)
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
