from copy import copy
from pathlib import Path

import torch
import transformers
import wandb
from transformers.integrations import WandbCallback, rewrite_logs
from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging

from retrievit.utils.huggingface import upload_file_to_hub

logger = logging.get_logger(__name__)


class CustomWandbCallback(WandbCallback):
    """A callback that logs metrics, media, model checkpoints to W&B."""

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len

    def setup(self, args, state, model, **kwargs) -> None:  # noqa: ANN001, ANN003
        """Setup the W&B run."""
        args.secret_field = 42
        super().setup(args, state, model, **kwargs)
        if state.is_world_process_zero:
            columns = [f"correct_pos{idx}" for idx in range(self.seq_len)]
            columns = ["global_step", *columns]
            self.eval_success_table = wandb.Table(columns=columns)

        # Log a custom field here used to group the runs in W&B
        # This is a hack because you cannot remove a run from a wandb group once it is set:
        # https://github.com/wandb/wandb/issues/9842
        if "-lr" in self._wandb.run.name:  # type: ignore[report]
            run_name = self._wandb.run.name  # type: ignore[report]
            group_parts = run_name.split("-")  # type: ignore[report]
            group_fields = []
            for part in group_parts:
                if part.startswith("lr"):
                    group_fields.append("".join(part.split("lr")[1]))

                elif not part.startswith("seed"):
                    group_fields.append(part)

            group_by_lr_and_seed_field = "-".join(group_fields)  # type: ignore[report]
            self._wandb.config.update(
                {"group_by_lr_and_seed_field": group_by_lr_and_seed_field}, allow_val_change=True
            )

    def on_predict(self, args, state, control, metrics, **kwargs) -> None:  # noqa: ANN001, ARG002, ANN003
        """Log prediction metrics to W&B."""
        if self._wandb is None:
            return

        if not self._initialized:
            self.setup(args, state, **kwargs)

        if state.is_world_process_zero:
            metrics = rewrite_logs(metrics)
            if any("correct" in key for key in metrics):  # type: ignore[report]
                correct_pos_keys = [key for key in metrics if "correct_pos" in key]
                # We need to use the test sequence length to get the correct keys because during
                # the test sequence length can be different from the training sequence length
                test_seq_len = len(correct_pos_keys)
                metrics_without_correct_keys = {
                    f"{k}_{test_seq_len}": v for k, v in metrics.items() if "correct" not in k
                }
                correct = [metrics[f"test/correct_pos{idx}"] for idx in range(test_seq_len)]  # type: ignore[report]
                columns = [f"correct_pos{idx}" for idx in range(test_seq_len)]
                test_success_table = wandb.Table(columns=columns)
                test_success_table.add_data(*correct)
                self._wandb.log(
                    {f"test/success_table_{test_seq_len}": copy(test_success_table)}, commit=False
                )
                self._wandb.log({**metrics_without_correct_keys})

            else:
                self._wandb.log(metrics)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs) -> None:  # noqa: ANN001, ARG002, ANN003
        """Log metrics to W&B."""
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if self._wandb is None:
            return

        if not self._initialized:
            self.setup(args, state, model)

        if state.is_world_process_zero:
            for k, v in logs.items():  # type: ignore[report]
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v  # type: ignore[report]

            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}  # type: ignore[report]
            non_scalar_logs = rewrite_logs(non_scalar_logs)

            beta_params = self.get_beta_params_from_model(model)
            non_scalar_logs.update(beta_params)

            if any("correct" in key for key in logs):  # type: ignore[report]
                logs_without_correct_keys = {k: v for k, v in logs.items() if "correct" not in k}  # type: ignore[report]
                correct = [logs[f"eval_correct_pos{idx}"] for idx in range(self.seq_len)]  # type: ignore[report]
                self.eval_success_table.add_data(state.global_step, *correct)
                self._wandb.log(
                    {"eval/success_table": copy(self.eval_success_table)}, commit=False
                )
                self._wandb.log(
                    {
                        **logs_without_correct_keys,
                        "train/global_step": state.global_step,
                    }
                )

            else:
                self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step})

    def get_beta_params_from_model(self, model: torch.nn.Module) -> dict[str, float]:
        """Get the beta parameters from the model."""
        layer_beta_values = {}
        if not hasattr(model, "backbone"):
            return layer_beta_values

        for idx, layer in enumerate(model.backbone.layers):  # type: ignore[report]
            if not hasattr(layer, "beta_param"):
                continue

            layer_beta_param = layer.beta_param.detach()
            is_vector = len(layer_beta_param.shape) > 1
            if is_vector:
                layer_beta_value = torch.norm(layer_beta_param.beta_param).item()
            else:
                layer_beta_value = abs(layer.gate_fn(layer_beta_param).item())

            layer_beta_values[f"layer_{idx + 1}"] = layer_beta_value

        return layer_beta_values


class ForceStopMaxStepsCallback(TrainerCallback):
    """A callback that stops training when a maximum number of steps is reached."""

    def __init__(self, max_steps: int) -> None:
        super().__init__()

        self.max_steps = max_steps
        self.max_steps_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs) -> None:  # noqa: ANN001, ARG002, ANN003
        """Check if the maximum number of steps has been reached."""
        if state.global_step >= self.max_steps:
            logger.info(f"Max steps {self.max_steps} reached. Stopping training.")
            control.should_training_stop = True


class EarlyStoppingCallback(transformers.EarlyStoppingCallback):
    """Early stopping callback."""

    def check_metric_value(self, args, state, control, metric_value) -> None:  # noqa: ANN001, ARG002
        """Check the metric value."""
        # We want to stop at any point during training if the metric is greater than the threshold
        if metric_value >= self.early_stopping_threshold:
            logger.info(f"Early stopping the training as the metric value is {metric_value}")
            self.early_stopping_patience_counter = self.early_stopping_patience


class UploadEmbeddingCallback(TrainerCallback):
    """Upload model embeddings to the Hugging Face Hub."""

    def __init__(self, output_dir: Path, path_in_repo: Path, repo_id: str) -> None:
        super().__init__()

        Path(output_dir).mkdir(exist_ok=True)
        self.output_dir = output_dir
        self.path_in_repo = path_in_repo
        self.repo_id = repo_id

    def on_evaluate(self, args, state, control, logs=None, model=None, **kwargs) -> None:  # noqa: ANN001, ARG002, ANN003
        """Upload the model embeddings to the Hugging Face Hub."""
        if state.is_local_process_zero:
            embeddings_name = f"embeddings_{state.global_step}.pt"
            embeddings_filepath = Path(self.output_dir, embeddings_name)

            torch.save(
                model.get_input_embeddings().state_dict(),  # type: ignore[report]
                embeddings_filepath,
            )

            upload_file_to_hub(
                embeddings_filepath,
                path_in_repo=Path(self.path_in_repo, embeddings_name),  # type: ignore[report]
                repo_id=self.repo_id,
                repo_type="model",
            )
