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
        bp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute correct predictions."""
        # All examples have the same task
        task = Task.get_task(task_id[0])
        if task == Task.copy:
            return self.correct_tokens_copy_task(input_ids=input_ids, logits=logits)

        if task == Task.selective_copy:
            return self.correct_selective_copy_task_from_labels(
                input_ids=input_ids, labels=labels, logits=logits
            )
        if task == Task.n_gram_retrieval:
            return self.correct_ngram_task_from_labels(
                input_ids=input_ids, logits=logits, raw_target=raw_target
            )

        if task == Task.token_retrieval:
            return self.correct_token_task_from_labels(input_ids=input_ids, logits=logits)

        if task == Task.position_retrieval:
            return self.correct_position_retrieval_task_from_labels(
                input_ids=input_ids, logits=logits
            )
        raise ValueError(f"Task {task} not supported.")

    def correct_tokens_copy_task(
        self, input_ids: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute correct predictions for copy task."""
        predicted_tokens = torch.argmax(logits, dim=-1)
        # A sequence has the form: seq <c> <seq>
        # We need to get the seq from the input ids
        target_seq = input_ids[:, : input_ids.shape[1] // 2]

        # We need to get the seq from the predicted tokens that are shifted by 1
        predicted_seq = predicted_tokens[:, (predicted_tokens.shape[1] // 2) : -1]

        correct_per_position = torch.zeros_like(target_seq[0])
        for target_token, predicted_token in zip(target_seq, predicted_seq, strict=True):
            correct_per_position += target_token == predicted_token

        correct = torch.sum(torch.all(predicted_seq == target_seq, dim=-1))
        return correct, correct_per_position

    def correct_selective_copy_task_from_labels(
        self, input_ids: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute correct predictions for selective copy task."""
        predicted_tokens = torch.argmax(logits, dim=-1)
        # A sequence has the form: <s> seq w/ white tokens <c> seq w/o white tokens
        # We need to get the seq w/o white tokens from the input ids
        target_seq_len = labels[0][labels[0] > 0].shape[0]
        target_seq = input_ids[:, -target_seq_len:]

        # We need to get the seq from the predicted tokens that are shifted by 1
        predicted_seq = predicted_tokens[:, -(target_seq_len + 1) : -1]

        correct_per_position = torch.zeros(
            target_seq_len, device=input_ids.device, dtype=input_ids.dtype
        )
        for target_token, predicted_token in zip(target_seq, predicted_seq, strict=False):
            correct_per_position += target_token == predicted_token

        correct = torch.sum(torch.all(predicted_seq == target_seq, dim=-1))
        return correct, correct_per_position

    def correct_ngram_task_from_labels(
        self, input_ids: torch.Tensor, logits: torch.Tensor, raw_target=None
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

    def correct_token_task_from_labels(
        self, input_ids: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute correct predictions for token retrieval task."""
        predicted_tokens = torch.argmax(logits, dim=-1)
        # We need to get the seq from the predicted tokens that are shifted by 1
        predicted_tokens = predicted_tokens[:, -1]

        query_token_id = self.config.query_token_id
        correct = 0
        # Remove the bos/eos token plus the query token
        correct_per_position = torch.zeros(self.config.seq_len).to(device=input_ids.device)
        for inp_ids, predicted_token in zip(input_ids, predicted_tokens, strict=False):
            # A sequence has the form: <s> <s1> <q> <s2> <s3> ... <sn> <s>
            # We need to get the token after the <q> from the input_ids
            # The '2' here is the token_id of the <q>, see the tokenizer.py
            target_token_position = (inp_ids == query_token_id).nonzero(as_tuple=True)[0]
            # The target token is the token after the <q>
            target_token = inp_ids[target_token_position + 1]
            correct_example = int(target_token == predicted_token)
            correct += correct_example
            # The target token position is shifted by 2 because we removed the bos and the query token
            correct_per_position[target_token_position - 2] += correct_example

        correct = torch.tensor(correct).to(device=input_ids.device, dtype=input_ids.dtype)
        return correct, correct_per_position.to(dtype=input_ids.dtype)

    # def correct_selective_copy_task_from_labels(
    #     self, input_ids: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor, pb: bool = False
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Compute correct predictions for selective copy task."""
    #     predicted_tokens = torch.argmax(logits, dim=-1)
    #     # A sequence has the form: <s> padded seq <c> <seq> <s>
    #     # We need to get the seq from the input ids
    #     target_seq_len = labels[0][labels[0] > 0].shape[0]
    #     target_seq = input_ids[:, -(target_seq_len + 1) : -1]

    #     # We need to get the seq from the predicted tokens that are shifted by 1
    #     predicted_seq = predicted_tokens[:, -(target_seq_len + 2) : -2]

    #     correct = torch.sum(torch.all(target_seq == predicted_seq, dim=-1))
    #     correct_per_position = torch.zeros(
    #         target_seq_len, device=input_ids.device, dtype=input_ids.dtype
    #     )

    #     for target_token, predicted_token in zip(target_seq, predicted_seq, strict=False):
    #         correct_per_position += target_token == predicted_token

    #     correct = torch.sum(torch.all(predicted_seq == target_seq, dim=-1))
    #     return correct, correct_per_position

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

    # def correct_multimodal_retrieval_from_labels(
    #     self,
    #     input_ids: torch.Tensor,
    #     labels: torch.Tensor,
    #     logits: torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Compute correct predictions for multimodal retrieval."""
    #     predicted_tokens = torch.argmax(logits, dim=-1)
    #     if self.config.prefix:
    #         # labels = [-100, -100, ..., -100, target_label, 0]
    #         # predicted_tokens = [0, 0, ..., predicted_label, 0, w/e]
    #         target_token_positions = labels[:, -2]
    #         predicted_token_positions = predicted_tokens[:, -3]
    #     else:
    #         # labels = [-100, -100, ..., -100, target_label, 0]
    #         # predicted_tokens = [0, 0, ..., predicted_label, 0, w/e]
    #         target_token_positions = labels[:, -2]
    #         predicted_token_positions = predicted_tokens[:, -3]

    #     correct = torch.sum(predicted_token_positions == target_token_positions)
    #     correct_per_position = torch.zeros(self.config.seq_len)
    #     for target_token, predicted_token in zip(
    #         target_token_positions, predicted_token_positions, strict=False
    #     ):
    #         correct = int(target_token == predicted_token.item())
    #         correct_per_position[target_token.item() - 2] = (
    #             correct_per_position[target_token.item() - 2] + correct
    #         )

    #     correct = torch.sum(predicted_token_positions == target_token_positions)
    #     return correct, correct_per_position
