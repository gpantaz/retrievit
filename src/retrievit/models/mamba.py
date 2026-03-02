from typing import Any

import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch.nn import CrossEntropyLoss

from retrievit.datamodels import CausalLMOutputWithPastWithCorrect
from retrievit.trainer.evaluator import Evaluator


class Mamba(MambaLMHeadModel):
    """Custom MambaForCausalLM model."""

    def __init__(self, config, tokenizer) -> None:  # noqa: ANN001
        super().__init__(config)
        self.config = config
        self.evaluator = Evaluator(config=config, tokenizer=tokenizer)
        self.vocab_size = self.lm_head.weight.shape[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,  # noqa: ARG002
        inference_params: dict | None = None,
        num_last_tokens: int = 0,
        **mixer_kwargs: dict[str, Any],
    ) -> CausalLMOutputWithPastWithCorrect:
        """Forward pass for the model."""
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
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
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> torch.Tensor:
        """Evaluate the model on a given input and labels."""
        device = self.get_input_embeddings().weight.device
        current_input_ids = input_ids.clone().to(device)
        current_attention_mask = attention_mask.clone() if attention_mask is not None else None
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids=current_input_ids)
                logits = outputs.logits[:, -1, :]
                predicted_id = torch.argmax(logits, dim=-1, keepdim=True)
                current_input_ids = torch.cat([current_input_ids, predicted_id], dim=1)
                # add a one-column to the attention mask
                if current_attention_mask is not None:
                    ones = current_attention_mask.new_ones((current_attention_mask.size(0), 1))
                    current_attention_mask = torch.cat([current_attention_mask, ones], dim=1)

        return current_input_ids
