from typing import Any

import torch
from loguru import logger

from retrievit.datamodels.datamodels import SpecialTokens, Task


class Evaluator:
    """Compute the performance of a model for all tasks."""

    def __init__(self, config, tokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer
        logger.warning(
            "The evaluator works only for single-task batches and batches without padding."
        )
        self.special_tokens = SpecialTokens()

    def __call__(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        task_id: torch.Tensor,
        raw_target: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute correct predictions."""
        # All examples have the same task
        task = Task.get_task(task_id[0])
        if task == Task.n_gram_retrieval:
            return self.correct_ngram_task_from_labels(
                input_ids=input_ids, logits=logits, raw_target=raw_target
            )

        if task == Task.position_retrieval:
            return self.correct_position_retrieval_task_from_labels(
                input_ids=input_ids, logits=logits
            )
        raise ValueError(f"Task {task} not supported.")

    def correct_ngram_task_from_labels(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        raw_target: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute correct predictions for n-gram retrieval task."""
        predicted_tokens = torch.argmax(logits, dim=-1)
        # A sequence has the form: <s> seq <r> <s1> <s2> ... <sn> <s>
        # We need to get the query tokens from the input_ids
        target_seq = input_ids[:, -(self.config.n_gram_size) :]

        # We need to get the query tokens from the predicted tokens that are shifted by 1
        predicted_seq = predicted_tokens[:, -(self.config.n_gram_size + 1) : -1]

        correct = torch.sum(torch.all(target_seq == predicted_seq, dim=-1))
        correct_per_position = torch.zeros(self.config.seq_len).to(
            device=target_seq.device, dtype=target_seq.dtype
        )
        zipped = zip(target_seq, predicted_seq, raw_target, strict=False)
        for target_token, predicted_token, example_raw_target in zipped:
            target_position = example_raw_target["target_pos"]
            correct_per_position[target_position] += int(
                torch.all(target_token == predicted_token).item()
            )

        correct = torch.sum(torch.all(predicted_seq == target_seq, dim=-1))
        return correct, correct_per_position

    def correct_position_retrieval_task_from_labels(
        self, input_ids: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute correct predictions for position retrieval task."""
        predicted_tokens = torch.argmax(logits, dim=-1)

        target_token_positions = input_ids[:, -2]
        predicted_token_positions = predicted_tokens[:, -3]

        correct_per_position = torch.zeros(
            self.config.seq_len, dtype=input_ids.dtype, device=input_ids.device
        )
        for target_token, predicted_token in zip(
            target_token_positions, predicted_token_positions, strict=False
        ):
            # Check if the predicted token is correct
            correct = int(target_token == predicted_token.item())
            # Now get the position of the target token and increment the correct count
            pos_in_seq = self.tokenizer.get_token_int_from_token_id(target_token.item())
            correct_per_position[pos_in_seq] += correct

        correct = torch.sum(predicted_token_positions == target_token_positions)
        return correct, correct_per_position
