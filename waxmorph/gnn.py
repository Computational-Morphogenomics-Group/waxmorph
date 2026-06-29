"""Backward-compatible PyTorch graph-network simulator exports.

Thin re-export shim so ``from waxmorph import GNS`` and ``waxmorph.gnn`` keep
resolving to the default PyTorch backend. Re-exports
:class:`waxmorph.torch.gnn.GNS` and :class:`waxmorph.torch.gnn.GraphNetworkBlock`
unchanged; edit behavior in :mod:`waxmorph.torch.gnn` (and mirror in
:mod:`waxmorph.jax.gnn`), not here.

See Also:
    :mod:`waxmorph.torch.gnn`: Implementation that this shim forwards to.
    :mod:`waxmorph.jax.gnn`: JAX/Equinox parity backend.
"""

from .torch.gnn import GNS, GraphNetworkBlock

__all__ = ["GNS", "GraphNetworkBlock"]
