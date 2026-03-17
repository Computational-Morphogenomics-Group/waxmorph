"""Backward-compatible re-export — actual implementation in waxmorph.torch.train."""

from .torch.train import TrainConfig, TrainResult, train

__all__ = ["TrainConfig", "TrainResult", "train"]
