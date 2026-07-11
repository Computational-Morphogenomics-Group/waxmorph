"""PyTorch training re-exports for the default :mod:`waxmorph.train` path."""

from .torch.train import TrainConfig, TrainResult, train

__all__ = ["TrainConfig", "TrainResult", "train"]
