"""Backward-compatible PyTorch MLP exports.

This module re-exports :class:`waxmorph.torch.mlp.MLP` and its
``_ACTIVATIONS`` registry for callers that import from ``waxmorph.mlp``. The
``_ACTIVATIONS`` mapping is re-exported so tests and callers can introspect the
supported activation names without reaching into the backend module.
"""

from .torch.mlp import _ACTIVATIONS, MLP

__all__ = ["MLP", "_ACTIVATIONS"]
