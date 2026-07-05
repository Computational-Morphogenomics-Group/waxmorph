"""Backward-compatible re-export of the PyTorch training surface.

Forwards :class:`TrainConfig`, :class:`TrainResult`, and :func:`train` from
:mod:`waxmorph.torch.train` (the default backend) so the legacy
``from waxmorph.train import train`` path keeps working. Edit behavior in
:mod:`waxmorph.torch.train`, not here, and mirror it in the JAX parity backend
:mod:`waxmorph.jax.train`.
"""

from .torch.train import TrainConfig, TrainResult, train

__all__ = ["TrainConfig", "TrainResult", "train"]
