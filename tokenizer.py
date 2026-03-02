import torch

from retrievit.datamodels.datamodels import SpecialTokens, Task

return_type = torch.Tensor | list[int]


class Tokenizer:
    """Simple tokenizer class."""

    def __init__(self, vocab_tokens: list[str]) -> None:
        self.stoi = {s: i for i, s in enumerate(vocab_tokens)}
        self.itos = {i: s for s, i in self.stoi.items()}
        # This is just a workaround for padding vocab embedding to multiples
        self.unk_token = "<UNK>"  # noqa: S105

    def __len__(self) -> int:
        """Get the vocabulary size."""
        return len(self.stoi)

    def __call__(self, text: list[str], return_tensors: bool = True) -> return_type:
        """Convert a list of tokens into token ids.

        Note that even though the text is a list of strings, each string is expected to be a single token.
        """
        return self.encode(text, return_tensors=return_tensors)

    def encode(self, text: list[str], return_tensors: bool = True) -> return_type:
        """Convert a list of tokens into token ids."""
        token_ids = [self.stoi[token] for token in text]
        if return_tensors:
            return torch.tensor(token_ids)
        return token_ids

    def decode(
        self, token_ids: list[int] | torch.Tensor, return_as_str: bool = True
    ) -> str | list[str]:
        """Convert token ids back into a list of tokens."""
        if isinstance(token_ids, torch.Tensor):
            tokens = [self.itos.get(token_id.item(), self.unk_token) for token_id in token_ids]
        else:
            tokens = [self.itos.get(token_id, self.unk_token) for token_id in token_ids]

        return " ".join(tokens) if return_as_str else tokens

    def get_token_int_from_token_id(self, token_id: int) -> int:
        """Get token id from token string.

        If the token id is mapped to a chara like <p42>, return 42
        """
        token = self.itos[token_id]
        return int(token.replace("<", "").replace(">", "")[1:])


def build_ngram_retrieval_vocab(vocab_size: int) -> list[str]:
    """Build vocabulary for n-gram retrieval task."""
    special_tokens = SpecialTokens()

    base_vocab_tokens = [
        special_tokens.special_token_format.format(index=token_id)
        for token_id in range(vocab_size)
    ]

    return [
        special_tokens.padding_token,
        special_tokens.bos_token,
        special_tokens.query_token,
        *base_vocab_tokens,
    ]


def build_position_retrieval_vocab(vocab_size: int) -> list[str]:
    """Build vocabulary for position retrieval task."""
    special_tokens = SpecialTokens()

    base_vocab_tokens = [
        special_tokens.special_token_format.format(index=token_id)
        for token_id in range(vocab_size)
    ]

    base_vocab_position_tokens = [
        special_tokens.position_token_format.format(index=token_id)
        for token_id in range(vocab_size)
    ]

    return [
        special_tokens.padding_token,
        special_tokens.bos_token,
        special_tokens.query_token,
        *base_vocab_tokens,
        *base_vocab_position_tokens,
    ]


def build_vocab_for_task(
    task: str, vocab_size: int | None = None, position_vocab_size: int | None = None
) -> list[str]:
    """Build vocabulary for a specific task."""
    if task == Task.n_gram_retrieval.value:
        return build_ngram_retrieval_vocab(vocab_size)
    if task == Task.position_retrieval.value:
        return build_position_retrieval_vocab(position_vocab_size)
    raise ValueError(f"Unknown task: {task}")


def test_ngram_retrieval() -> None:
    """Test the n-gram retrieval."""
    task = "n_gram_retrieval"
    vocab_size = 200
    tokenizer = Tokenizer(build_vocab_for_task(task=task, vocab_size=vocab_size))

    example = ["<s5>", "<s0>", "<s1>", "<s2>", "<query>", "<s0>", "<s1>"]
    token_ids = tokenizer(example)
    print("Token IDs:", token_ids)
    decoded_tokens = tokenizer.decode(token_ids, return_as_str=False)
    print("Decoded Tokens:", decoded_tokens)


def test_position_retrieval() -> None:
    """Test the position retrieval."""
    task = "position_retrieval"
    position_vocab_size = 20
    tokenizer = Tokenizer(build_vocab_for_task(task=task, position_vocab_size=position_vocab_size))

    example = ["<s5>", "<s0>", "<s1>", "<s2>", "<query>", "<s0>", "<p0>"]
    token_ids = tokenizer(example)
    print("Token IDs:", token_ids)
    decoded_tokens = tokenizer.decode(token_ids, return_as_str=False)
    print("Decoded Tokens:", decoded_tokens)
