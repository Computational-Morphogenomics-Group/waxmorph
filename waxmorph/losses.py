"""Backward-compatible re-export — actual implementation in waxmorph.torch.losses."""

from .torch.losses import (
    SAMPLES_LOSS_DEFAULTS,
    chamfer_distance,
    make_samples_loss,
    squared_loss,
)

__all__ = [
    "SAMPLES_LOSS_DEFAULTS",
    "chamfer_distance",
    "make_samples_loss",
    "squared_loss",
]
