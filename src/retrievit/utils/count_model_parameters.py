import torch
from loguru import logger


def compute_trainable_params(model: torch.nn.Module, exclude: str | None = None) -> None:
    """Compute trainable parameters."""
    if exclude is not None:
        params = [parameter for name, parameter in model.named_parameters() if exclude not in name]
    else:
        params = [parameter for name, parameter in model.named_parameters()]

    model_parameters = filter(lambda p: p.requires_grad, params)
    train_params = sum([p.numel() for p in model_parameters])
    logger.info(
        f"{sum([p.numel() for p in model.parameters()])} params and {train_params} trainable params"
    )