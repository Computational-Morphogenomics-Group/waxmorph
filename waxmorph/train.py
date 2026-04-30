"""Backward-compatible PyTorch training exports."""

from .torch.train import TrainConfig, TrainResult, train

__all__ = ["TrainConfig", "TrainResult", "train"]
