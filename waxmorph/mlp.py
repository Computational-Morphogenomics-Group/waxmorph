"""Backward-compatible PyTorch MLP exports."""

from .torch.mlp import _ACTIVATIONS, MLP

__all__ = ["MLP", "_ACTIVATIONS"]
