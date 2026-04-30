"""Backward-compatible PyTorch graph-network simulator exports.

This module re-exports :class:`waxmorph.torch.gnn.GNS` and
:class:`waxmorph.torch.gnn.GraphNetworkBlock` for callers that import from
``waxmorph.gnn``.
"""

from .torch.gnn import GNS, GraphNetworkBlock

__all__ = ["GNS", "GraphNetworkBlock"]
