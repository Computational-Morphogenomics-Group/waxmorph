"""Backward-compatible re-export — actual implementation in waxmorph.torch.mlp."""

from .torch.mlp import _ACTIVATIONS, MLP

__all__ = ["MLP", "_ACTIVATIONS"]
