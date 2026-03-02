from typing import Any

import torch
from transformers import GPTNeoXForCausalLM
from transformers.cache_utils import Cache

from retrievit.datamodels import CausalLMOutputWithPastWithCorrect
from retrievit.trainer.evaluator import Evaluator


class Transformer(GPTNeoXForCausalLM):
    """Custom GPTNeoXForCausalLM model."""

    def __init__(self, config, tokenizer) -> None:  # noqa: ANN001
        super().__init__(config)
        self.config = config
        self.evaluator = Evaluator(config=config, tokenizer=tokenizer)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: dict[str, Any],
    ) -> tuple | CausalLMOutputWithPastWithCorrect:
        """Forward pass for the model."""
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
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> torch.Tensor:
        """Evaluate the model on a given input and labels."""
        current_input_ids = input_ids.clone().to(self.device)
        current_attention_mask = attention_mask.clone() if attention_mask is not None else None
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                outputs = self.forward(
                    input_ids=current_input_ids, attention_mask=current_attention_mask
                )
                logits = outputs.logits[:, -1, :]  # type: ignore[report]
                predicted_id = torch.argmax(logits, dim=-1, keepdim=True)
                current_input_ids = torch.cat([current_input_ids, predicted_id], dim=1)
                # add a one-column to the attention mask
                if current_attention_mask is not None:
                    ones = current_attention_mask.new_ones((current_attention_mask.size(0), 1))
                    current_attention_mask = torch.cat([current_attention_mask, ones], dim=1)

        return current_input_ids
