"""PyTorch MLP re-exports for the default :mod:`waxmorph.mlp` path."""

from .torch.mlp import _ACTIVATIONS, MLP

__all__ = ["MLP", "_ACTIVATIONS"]
