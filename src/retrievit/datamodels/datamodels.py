from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from pydantic import BaseModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.mamba.modeling_mamba import MambaCausalLMOutput


class Task(Enum):
    """Task for the an instance."""

    copy = "copy"
    n_gram_retrieval = "n_gram_retrieval"
    token_retrieval = "token_retrieval"  # noqa: S105
    selective_copy = "selective_copy"
    position_retrieval = "position_retrieval"

    @classmethod
    def get_index(cls, task: Any) -> int:
        """Get task index."""
        return list(cls).index(task)

    @classmethod
    def get_task(cls, index: int) -> Any:
        """Get task."""
        return list(cls)[index]


class Instance(BaseModel):
    """Dataset instance."""

    task: Task
    context: str
    query: str | None = None
    metadata: str | None = None

    class Config:
        """Updated config."""

        arbitrary_types_allowed = True


@dataclass
class DatasetItemCollateFn:
    """Used to determine what to do in the collate function for element in an example."""

    input_ids = "pad"
    labels = "pad"
    attention_mask = "pad"
    task = "stack"


@dataclass
class DatasetItem:
    """Output for the dataset reader."""

    input_ids: torch.Tensor
    task: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    raw_target: list[int] | None = None


@dataclass
class DatasetPadding:
    """Padding values used by collate."""

    input_ids: int = 0
    attention_mask: int = 0
    labels: int = -100
    task: int = -1


@dataclass
class SpecialTokens:
    """Special tokens used by the tokenizer."""

    copy_token: str = "<copy>"  # noqa: S105
    out_token: str = "<out>"  # noqa: S105
    query_token: str = "<query>"  # noqa: S105
    white_token: str = "<white>"  # noqa: S105
    special_token_format: str = "<s{index}>"  # noqa: S105
    position_token_format: str = "<p{index}>"  # noqa: S105
    padding_token: str = "<pad>"  # noqa: S105
    bos_token: str = "<bos>"  # noqa: S105


@dataclass
class CausalLMOutputWithPastWithCorrect(CausalLMOutputWithPast):
    """CausalLMOutputWithPast with correct number of exaples per batch."""

    correct: torch.Tensor | None = None
    correct_per_position: torch.Tensor | None = None


@dataclass
class MambaCausalLMOutputWithCorrect(MambaCausalLMOutput):
    """MambaCausalLMOutput with correct number of exaples per batch."""

    correct: torch.Tensor | None = None
    correct_per_position: torch.Tensor | None = None
