import os
import random
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import transformers
from loguru import logger
from tokenizer import Tokenizer, build_vocab_for_task
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.trainer_utils import set_seed

from retrievit.callbacks.callbacks import (
    CustomWandbCallback,
    EarlyStoppingCallback,
    UploadEmbeddingCallback,
)
from retrievit.datamodels.datamodels import DatasetItem, DatasetPadding, SpecialTokens, Task
from retrievit.datasets.collate import Collate
from retrievit.models.hybrid_par import HybridPar
from retrievit.models.hybrid_par_corrector import HybridParCorrector
from retrievit.models.hybrid_seq import HybridSeq
from retrievit.models.mamba import Mamba
from retrievit.models.transformer import Transformer
from retrievit.models.transformer_nope import TransformerNoPE
from retrievit.trainer import CustomTrainer
from retrievit.utils.compute_ngrams import compute_ngrams
from retrievit.utils.count_model_parameters import compute_trainable_params
from retrievit.utils.huggingface import upload_file_to_hub
from retrievit.utils.io import read_json

ModelType = Transformer | Mamba | HybridSeq | HybridPar | TransformerNoPE | HybridParCorrector


def random_int(min_value: int, max_value: int) -> int:
    """Randomly sample a value between min_value and max_value."""
    return random.randint(min_value, max_value)  # noqa: S311


class RetrievetSamplingDataset(Dataset[DatasetItem]):
    """Retrievet dataset with sampling."""

    def __init__(
        self,
        task: Task,
        dataset_size: int,
        vocab_size: int,
        seq_len: int,
        retrieval_n_gram_size: int,
        retrieval_query_n_gram_size: int,
        tokenizer: Tokenizer,
        needs_attention_mask: bool = False,
        is_prefix: bool = False,
        min_seq_len: int | None = None,
        is_eval: bool = False,
        duplicate_n_gram_bins: int = 4,
        duplicate_n_grams: bool = False,
        position_retrieval_varlen: bool = False,
    ) -> None:
        """Initialize the dataset."""
        self.task = task
        self.dataset_size = dataset_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.retrieval_n_gram_size = retrieval_n_gram_size
        self.retrieval_query_n_gram_size = retrieval_query_n_gram_size
        if task == Task.n_gram_retrieval and retrieval_query_n_gram_size >= retrieval_n_gram_size:
            raise ValueError("Query n-gram size should be less than the n-gram size.")

        self.tokenizer = tokenizer
        self.prepare_instance = {
            Task.n_gram_retrieval.value: self.prepare_n_gram_retrieval_instance
            if not duplicate_n_grams
            else self.prepare_corrupted_n_gram_retrieval_instance,
            Task.position_retrieval.value: self.prepare_position_retrieval_instance_varlen
            if position_retrieval_varlen
            else self.prepare_position_retrieval_instance,
        }

        self.special_tokens = SpecialTokens()

        self._needs_attention_mask = needs_attention_mask
        self._is_prefix = is_prefix
        self.min_seq_len = min_seq_len
        self.is_eval = is_eval
        self.duplicate_n_gram_bins = duplicate_n_gram_bins

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.dataset_size

    def __getitem__(self, idx: int) -> DatasetItem:
        """Get the item at the index."""
        return self.prepare_instance[self.task]()

    def prepare_position_retrieval_instance_varlen(self) -> DatasetItem:
        """Prepare an instance of the position retrieval task."""
        seq_len = self.seq_len - 1
        if self.min_seq_len is not None:
            seq_len = random_int(self.min_seq_len, self.seq_len - 1)

        full_sequence = [
            self.special_tokens.special_token_format.format(index=token_id)
            for token_id in range(self.vocab_size)
        ]
        target_token_idx = random.randint(0, len(full_sequence) - 1)  # noqa: S311
        query_token = full_sequence[target_token_idx]
        seq_without_target = (
            full_sequence[:target_token_idx] + full_sequence[target_token_idx + 1 :]
        )

        sequence = random.choices(seq_without_target, k=seq_len)  # noqa: S311
        sequence.append(query_token)
        random.shuffle(sequence)
        target_pos = sequence.index(query_token)

        if self._is_prefix:
            input_sequence = [
                self.special_tokens.query_token,
                query_token,
                self.special_tokens.bos_token,
                *sequence,
            ]
        else:
            input_sequence = [
                self.special_tokens.bos_token,
                *sequence,
                self.special_tokens.query_token,
                query_token,
            ]

        target_position_token = self.special_tokens.position_token_format.format(index=target_pos)
        target_sequence = [target_position_token, self.special_tokens.bos_token]

        input_ids, attention_mask, labels = self.prepare_inputs_targets(
            inputs=input_sequence, targets=target_sequence
        )

        return DatasetItem(
            input_ids=input_ids,
            labels=labels,
            task=self._get_task_as_tensor(Task.position_retrieval),
            attention_mask=attention_mask,
        )

    def prepare_position_retrieval_instance(self) -> DatasetItem:
        """Prepare an instance of the position retrieval task."""
        seq_len = self.seq_len
        full_sequence = [
            self.special_tokens.special_token_format.format(index=token_id)
            for token_id in range(seq_len)
        ]
        random.shuffle(full_sequence)

        target_pos = random_int(0, seq_len - 1)

        if self._is_prefix:
            input_sequence = [
                self.special_tokens.query_token,
                full_sequence[target_pos],
                self.special_tokens.bos_token,
                *full_sequence,
            ]
        else:
            input_sequence = [
                self.special_tokens.bos_token,
                *full_sequence,
                self.special_tokens.query_token,
                full_sequence[target_pos],
            ]

        target_position_token = self.special_tokens.position_token_format.format(index=target_pos)
        target_sequence = [target_position_token, self.special_tokens.bos_token]

        input_ids, attention_mask, labels = self.prepare_inputs_targets(
            inputs=input_sequence, targets=target_sequence
        )

        return DatasetItem(
            input_ids=input_ids,
            labels=labels,
            task=self._get_task_as_tensor(Task.position_retrieval),
            attention_mask=attention_mask,
        )

    def prepare_n_gram_retrieval_instance(self) -> DatasetItem:
        """Prepare an instance of the n-gram retrieval task."""
        while True:
            full_sequence = [
                self.special_tokens.special_token_format.format(index=token_id)
                for token_id in range(self.vocab_size)
            ]
            seq_len = self.seq_len
            if self.min_seq_len is not None:
                seq_len = random_int(self.min_seq_len, self.seq_len)

            sequence = random.choices(full_sequence, k=seq_len)  # noqa: S311
            target_pos = random.randint(0, seq_len - self.retrieval_n_gram_size)  # noqa: S311

            query_sequence = sequence[target_pos : target_pos + self.retrieval_query_n_gram_size]

            n_grams = compute_ngrams(sequence, self.retrieval_query_n_gram_size)
            n_gram_counter = Counter(n_grams)

            if n_gram_counter[tuple(query_sequence)] == 1:
                break

        full_query_sequence = sequence[target_pos : target_pos + self.retrieval_n_gram_size]
        target_query_sequence = full_query_sequence[self.retrieval_query_n_gram_size :]

        if self._is_prefix:
            input_sequence = [
                self.special_tokens.query_token,
                *query_sequence,
                self.special_tokens.bos_token,
                *sequence,
                self.special_tokens.bos_token,
                self.special_tokens.query_token,
            ]
        else:
            input_sequence = [
                self.special_tokens.bos_token,
                *sequence,
                self.special_tokens.bos_token,
                self.special_tokens.query_token,
                *query_sequence,
                self.special_tokens.query_token,
            ]

        input_ids, attention_mask, labels = self.prepare_inputs_targets(
            inputs=input_sequence, targets=target_query_sequence
        )

        return DatasetItem(
            input_ids=input_ids,
            labels=labels,
            task=self._get_task_as_tensor(Task.n_gram_retrieval),
            attention_mask=attention_mask,
            raw_target={"target": target_query_sequence, "target_pos": target_pos},
        )

    def prepare_inputs_targets(
        self, inputs: list[int], targets: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Prepare inputs and targets."""
        full_sequence = [inputs] if self.is_eval else [inputs, targets]

        tokens = [self.tokenizer(seq, return_tensors=True) for seq in full_sequence]
        input_ids = torch.concatenate(tokens, dim=-1)

        masked_tokens = sum([len(seq) for seq in tokens[:-1]])
        labels = None
        if not self.is_eval:
            labels = torch.concatenate(
                [torch.ones(masked_tokens, dtype=torch.long) * -100, tokens[-1]]
            )

        attention_mask = torch.ones_like(input_ids) if self._needs_attention_mask else None

        return (input_ids, attention_mask, labels)

    def prepare_corrupted_n_gram_retrieval_instance(self) -> DatasetItem:
        """Prepare a corrupted n-gram retrieval instance."""
        full_sequence = [
            self.special_tokens.special_token_format.format(index=token_id)
            for token_id in range(self.vocab_size)
        ]
        seq = deepcopy(full_sequence)
        random.shuffle(seq)
        n_grams = compute_ngrams(seq, self.retrieval_query_n_gram_size)
        random_ngram_idx = random.randint(0, len(n_grams) - 1)  # noqa: S311
        query_ngram = n_grams[random_ngram_idx]
        all_target_ngrams = []
        all_sequences = []
        all_target_positions = []
        for bucket in range(self.duplicate_n_gram_bins):
            while True:
                seq_len = (
                    self.seq_len // self.duplicate_n_gram_bins - self.retrieval_query_n_gram_size
                )

                sequence = random.choices(full_sequence, k=seq_len)  # noqa: S311
                target_ngram_size = self.retrieval_n_gram_size - self.retrieval_query_n_gram_size
                target_pos = random.randint(0, seq_len - target_ngram_size - 1)  # noqa: S311
                target_ngram = sequence[target_pos : target_pos + target_ngram_size]
                sequence = [
                    *sequence[:target_pos],
                    *query_ngram,
                    *sequence[target_pos:],
                ]
                n_grams = compute_ngrams(sequence, self.retrieval_n_gram_size)

                n_gram_counter = Counter(n_grams)

                # We should exit if the overall ngram is unique AND the target_ngram has not been
                # previously selected
                if (
                    n_gram_counter[tuple(query_ngram + tuple(target_ngram))] == 1
                    and target_ngram not in all_target_ngrams
                ):
                    all_target_ngrams.append(target_ngram)
                    all_sequences.append(sequence)
                    all_target_positions.append(
                        target_pos + bucket * (self.seq_len // self.duplicate_n_gram_bins)
                    )
                    break

        if self._is_prefix:
            input_sequence = [
                self.special_tokens.query_token,
                *query_ngram,
                self.special_tokens.bos_token,
                *[x for xs in all_sequences for x in xs],
                self.special_tokens.bos_token,
                self.special_tokens.query_token,
            ]
        else:
            input_sequence = [
                self.special_tokens.bos_token,
                *[x for xs in all_sequences for x in xs],
                self.special_tokens.bos_token,
                self.special_tokens.query_token,
                *query_ngram,
                self.special_tokens.query_token,
            ]

        input_ids = self.tokenizer(input_sequence)
        attention_mask = torch.ones_like(input_ids) if self._needs_attention_mask else None

        return DatasetItem(
            input_ids=input_ids,
            labels=None,
            task=self._get_task_as_tensor(Task.n_gram_retrieval),
            attention_mask=attention_mask,
            raw_target={
                "query_ngram": query_ngram,
                "all_target_ngrams": all_target_ngrams,
                "all_target_positions": all_target_positions,
            },
        )

    def _get_task_as_tensor(self, task: Task) -> torch.Tensor:
        """Convert the given task to a Tensor."""
        return torch.tensor([Task.get_index(task)], dtype=torch.long)


@dataclass
class ModelArguments:
    """Model arguments."""

    model_class: Literal[
        "transformer",
        "mamba",
        "hybrid_seq",
        "hybrid_par",
        "transformer_nope",
        "hybrid_par_corrector",
    ] = field(default="transformer")
    model_config: str = field(default="configs/model/transformer.json")


@dataclass
class DataArguments:
    """Data arguments."""

    task: str = field(default=Task.copy.value)

    seq_len: int = field(default=50)
    min_seq_len: int | None = field(default=None)
    vocab_size: int = field(default=30)

    retrieval_n_gram_size: int = field(default=8)
    retrieval_query_n_gram_size: int = field(default=4)

    train_dataset_size: int = field(default=1000000)
    validation_dataset_size: int = field(default=10000)

    # If the datataset has the prefix format the query is placed before the sequence
    is_prefix: bool = False

    ngram_retrieval_test_seq_len: list[int] = field(
        default_factory=lambda: [100, 125, 150, 175, 200, 300, 400]
    )

    duplicate_n_gram_bins: int = field(default=4)
    position_retrieval_varlen: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 10])


@dataclass
class TrainArgs(transformers.TrainingArguments):
    """Training arguments."""

    output_dir: str = field(default="storage/models/transformer-160m")
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=64)
    gradient_accumulation_steps: int = field(default=4)
    logging_steps: int = field(default=1)
    save_strategy: str = field(default="steps")
    save_steps: float = field(default=0.01)
    num_train_epochs: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.05)
    lr_scheduler_type: str = field(default="linear")
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    save_total_limit: int = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    log_level: str = field(default="debug")
    save_safetensors: bool = field(default=True)
    eval_strategy: str = field(default="steps")
    eval_steps: float = field(default=0.01)
    seed: int = field(default=12345)
    # data_seed: int = field(default=12345)
    dataloader_num_workers: int = field(default=4)
    logging_nan_inf_filter: bool = field(default=False)
    early_stopping_patience: int = field(default=1000)
    early_stopping_threshold: float = field(default=0.95)
    metric_for_best_model: str = field(default="eval_accuracy")
    run_name: str = field(default="copy-transformer160m")
    project_name: str = field(default="retrievit")
    # Use `none` if using a custom wandb callback
    # The reason is that if we use `all` the default behavior is to add all available callbacks
    report_to: Literal["all", "none"] = field(default="none")

    # Workaround to push_to_hub arg from Trainer
    upload_embeddings_after_training: bool = field(default=False)
    upload_embeddings_during_training: bool = field(default=False)
    upload_full_model_after_training: bool = field(default=False)
    hf_repo_id: str = field(default="gpantaz/retrievit")

    do_test_extrapolation: bool = field(default=False)
    do_test_duplicate: bool = field(default=False)


def build_tokenizer(data_args: DataArguments) -> Tokenizer:
    """Build tokenizer."""
    tokenizer = Tokenizer(
        build_vocab_for_task(
            task=data_args.task,
            vocab_size=data_args.vocab_size,
            position_vocab_size=data_args.seq_len,
        )
    )
    return tokenizer


def build_model(
    model_args: ModelArguments,
    is_prefix: bool,
    seq_len: int,
    n_gram_size: int,
    query_token_id: int | None = None,
    tokenizer: Tokenizer | None = None,
) -> ModelType:
    """Build model."""
    raw_config = read_json(model_args.model_config)
    config = transformers.PretrainedConfig(**raw_config)
    config.is_prefix = is_prefix
    config.seq_len = seq_len
    config.n_gram_size = n_gram_size
    config.query_token_id = query_token_id
    config.vocab_size = len(tokenizer)

    if model_args.model_class == "transformer":
        model = Transformer(config=config, tokenizer=tokenizer)

    elif model_args.model_class == "mamba":
        model = Mamba(config=config, tokenizer=tokenizer)

    elif model_args.model_class == "hybrid_seq":
        model = HybridSeq(config=config, tokenizer=tokenizer)

    elif model_args.model_class == "hybrid_par":
        model = HybridPar(config=config, tokenizer=tokenizer)

    elif model_args.model_class == "transformer_nope":
        model = TransformerNoPE(config=config, tokenizer=tokenizer)

    elif model_args.model_class == "hybrid_par_corrector":
        model = HybridParCorrector(config=config, tokenizer=tokenizer)

    else:
        raise ValueError(f"Model class {model_args.model_class} is not supported.")

    logger.info(model)
    # Do not count the embedding matrix as part of the trainable parameters
    # The reason is that each model has a different vocabulary size but in the end we only
    # train the special tokens
    compute_trainable_params(model, exclude="embed")
    return model


def train() -> None:
    """Train."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainArgs))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    logger.info(model_args)
    logger.info(data_args)
    logger.info(train_args)

    # Set the seed for reproducibility, not sure if this is needed as the Trainer also sets the seed
    set_seed(train_args.seed)

    tokenizer = build_tokenizer(data_args=data_args)

    model = build_model(
        model_args=model_args,
        is_prefix=data_args.is_prefix,
        seq_len=data_args.seq_len,
        n_gram_size=data_args.retrieval_n_gram_size - data_args.retrieval_query_n_gram_size,
        query_token_id=tokenizer.stoi.get(SpecialTokens().query_token, None),
        tokenizer=tokenizer,
    )

    train_dataset = RetrievetSamplingDataset(
        task=data_args.task,
        dataset_size=data_args.train_dataset_size,
        vocab_size=data_args.vocab_size,
        min_seq_len=data_args.min_seq_len,
        seq_len=data_args.seq_len,
        retrieval_n_gram_size=data_args.retrieval_n_gram_size,
        retrieval_query_n_gram_size=data_args.retrieval_query_n_gram_size,
        tokenizer=tokenizer,
        needs_attention_mask=model_args.model_class != "mamba",
        is_prefix=data_args.is_prefix,
        position_retrieval_varlen=data_args.position_retrieval_varlen,
    )

    eval_dataset = RetrievetSamplingDataset(
        task=data_args.task,
        dataset_size=data_args.validation_dataset_size,
        vocab_size=data_args.vocab_size,
        seq_len=data_args.seq_len,
        retrieval_n_gram_size=data_args.retrieval_n_gram_size,
        retrieval_query_n_gram_size=data_args.retrieval_query_n_gram_size,
        tokenizer=tokenizer,
        needs_attention_mask=model_args.model_class != "mamba",
        is_prefix=data_args.is_prefix,
        position_retrieval_varlen=data_args.position_retrieval_varlen,
    )

    collate_fn = Collate(padding=DatasetPadding(), padding_side="right")

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=train_args.early_stopping_patience,
        early_stopping_threshold=train_args.early_stopping_threshold,
    )

    target_seq_len_map = {
        Task.position_retrieval.value: data_args.seq_len,
        Task.n_gram_retrieval.value: data_args.seq_len,
    }

    target_seq_len = target_seq_len_map[data_args.task]

    custom_wandb_callback = CustomWandbCallback(seq_len=target_seq_len)

    # Set the environment variables
    # https://docs.wandb.ai/guides/integrations/huggingface
    if train_args.project_name is not None:
        os.environ["WANDB_PROJECT"] = train_args.project_name

    # This is used to log the performance per position in the sequence during training
    train_args.seq_len = target_seq_len

    callbacks = [early_stopping_callback, custom_wandb_callback]
    if train_args.upload_embeddings_during_training:
        callbacks.append(
            UploadEmbeddingCallback(
                output_dir=train_args.output_dir,
                path_in_repo=Path(data_args.task, train_args.run_name),
                repo_id=train_args.hf_repo_id,
            )
        )

    trainer = CustomTrainer(
        args=train_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=None,
        callbacks=callbacks,
    )

    trainer.train()

    if train_args.upload_embeddings_after_training:
        output_dir = train_args.output_dir

        Path(output_dir).mkdir(exist_ok=True)
        torch.save(model.get_input_embeddings().state_dict(), Path(output_dir, "embeddings.pt"))

        upload_file_to_hub(
            Path(output_dir, "embeddings.pt"),
            path_in_repo=Path(data_args.task, train_args.run_name, "embeddings.pt"),
            repo_id=train_args.hf_repo_id,
            repo_type="model",
        )

    if train_args.upload_full_model_after_training:
        output_dir = train_args.output_dir
        Path(output_dir).mkdir(exist_ok=True)
        torch.save(model.state_dict(), Path(output_dir, "model.pt"))

        upload_file_to_hub(
            Path(output_dir, "model.pt"),
            path_in_repo=Path(data_args.task, train_args.run_name, "model.pt"),
            repo_id=train_args.hf_repo_id,
            repo_type="model",
        )

    if data_args.task == Task.n_gram_retrieval.value:
        if train_args.do_test_extrapolation:
            test_n_gram_extrapolation(
                model=model,
                tokenizer=tokenizer,
                collate_fn=collate_fn,
                logger_callback=custom_wandb_callback,
                ngram_retrieval_test_seq_len=data_args.ngram_retrieval_test_seq_len,
                vocab_size=data_args.vocab_size,
                retrieval_n_gram_size=data_args.retrieval_n_gram_size,
                retrieval_query_n_gram_size=data_args.retrieval_query_n_gram_size,
                is_prefix=data_args.is_prefix,
                needs_attention_mask=model_args.model_class != "mamba",
                batch_size=train_args.per_device_eval_batch_size,
                num_workers=train_args.dataloader_num_workers,
            )

        elif train_args.do_test_duplicate:
            test_n_gram_duplicate(
                model=model,
                tokenizer=tokenizer,
                collate_fn=collate_fn,
                logger_callback=custom_wandb_callback,
                seq_len=data_args.ngram_retrieval_test_seq_len,
                duplicate_n_gram_bins=data_args.duplicate_n_gram_bins,
                vocab_size=data_args.vocab_size,
                retrieval_n_gram_size=data_args.retrieval_n_gram_size,
                retrieval_query_n_gram_size=data_args.retrieval_query_n_gram_size,
                is_prefix=data_args.is_prefix,
                needs_attention_mask=model_args.model_class != "mamba",
                batch_size=train_args.per_device_eval_batch_size,
                num_workers=train_args.dataloader_num_workers,
            )


def test_n_gram_extrapolation(
    model: ModelType,
    tokenizer: Tokenizer,
    collate_fn: Collate,
    logger_callback: CustomWandbCallback,
    ngram_retrieval_test_seq_len: list[int],
    vocab_size: int,
    retrieval_n_gram_size: int,
    retrieval_query_n_gram_size: int,
    needs_attention_mask: bool = True,
    is_prefix: bool = False,
    batch_size: int = 128,
    num_workers: int = 4,
) -> None:
    """Test the model's extrapolation capabilities on the n-gram retrieval task."""
    accuracies = []
    for seq_len in ngram_retrieval_test_seq_len:
        test_dataset = RetrievetSamplingDataset(
            task=Task.n_gram_retrieval,
            dataset_size=500000,
            vocab_size=vocab_size,
            seq_len=seq_len,
            retrieval_n_gram_size=retrieval_n_gram_size,
            retrieval_query_n_gram_size=retrieval_query_n_gram_size,
            tokenizer=tokenizer,
            needs_attention_mask=needs_attention_mask,
            is_prefix=is_prefix,
            is_eval=True,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        correct = 0
        total = 0
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            outputs = model.evaluate(**batch, max_new_tokens=seq_len)
            for output, raw_target in zip(outputs, batch["raw_target"], strict=True):
                target = raw_target["target"]
                predicted_str = tokenizer.decode(output[-seq_len:], return_as_str=False)
                correct += int(predicted_str == target)
                total += 1

        accuracy = correct / total
        logger.info(f"Test accuracy: {accuracy:.4f}")
        accuracies.append(accuracy)
        logger_callback._wandb.log({f"test_accuracy_{seq_len}": accuracy})  # noqa: SLF001


def test_n_gram_duplicate(
    model: ModelType,
    tokenizer: Tokenizer,
    collate_fn: Collate,
    logger_callback: CustomWandbCallback,
    seq_len: int,
    duplicate_n_gram_bins: list[int],
    vocab_size: int,
    retrieval_n_gram_size: int,
    retrieval_query_n_gram_size: int,
    needs_attention_mask: bool = True,
    is_prefix: bool = False,
    batch_size: int = 128,
    num_workers: int = 4,
) -> None:
    """Test the model's ability to handle duplicate n-grams."""
    for n_gram_bins in duplicate_n_gram_bins:
        test_dataset = RetrievetSamplingDataset(
            task=Task.n_gram_retrieval,
            dataset_size=500000,
            vocab_size=vocab_size,
            seq_len=seq_len,
            retrieval_n_gram_size=retrieval_n_gram_size,
            retrieval_query_n_gram_size=retrieval_query_n_gram_size,
            tokenizer=tokenizer,
            needs_attention_mask=needs_attention_mask,
            is_prefix=is_prefix,
            is_eval=True,
            duplicate_n_grams=True,
            duplicate_n_gram_bins=n_gram_bins,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        counts_dict = defaultdict(int)
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            outputs = model.evaluate(**batch, max_new_tokens=3)
            for output, raw_target in zip(outputs, batch["raw_target"], strict=True):
                target_duplicate_n_gram_bins = raw_target["all_target_ngrams"]
                predicted_str = tokenizer.decode(output[-3:], return_as_str=False)
                missed = 1
                for idx, target_bucket in enumerate(target_duplicate_n_gram_bins):
                    if predicted_str == target_bucket:
                        counts_dict[idx] += 1
                        missed = 0
                        break
                counts_dict["missed"] += missed

        total_examples = len(test_dataset)
        for target_n_gram in range(duplicate_n_gram_bins):
            perc = round(counts_dict[target_n_gram] / total_examples * 100, 2)
            logger.info(
                f"Duplicate test - {target_n_gram}: {counts_dict[target_n_gram]} ({perc}%)"
            )
            logger_callback._wandb.log(  # noqa: SLF001
                {
                    f"{target_n_gram}_maxduplicate_n_gram_bins{duplicate_n_gram_bins}": counts_dict[
                        target_n_gram
                    ]
                }
            )
        perc = round(counts_dict["missed"] / total_examples * 100, 2)
        logger.info(f"Duplicate test - missed: {counts_dict['missed']} ({perc}%)")
        logger_callback._wandb.log(  # noqa: SLF001
            {f"missed_maxduplicate_n_gram_bins{duplicate_n_gram_bins}": counts_dict["missed"]}
        )


if __name__ == "__main__":
    train()
