"""Backward-compatible re-exports of :class:`waxmorph.torch.mlp.MLP` and its
``_ACTIVATIONS`` registry for callers importing from ``waxmorph.mlp``.

``_ACTIVATIONS`` is re-exported so callers can introspect supported activation
names without reaching into the backend module.
"""

from .torch.mlp import _ACTIVATIONS, MLP

__all__ = ["MLP", "_ACTIVATIONS"]
