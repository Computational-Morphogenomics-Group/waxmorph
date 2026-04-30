"""Backward-compatible PyTorch graph construction exports.

This module re-exports the default graph builders from
:mod:`waxmorph.torch.graph` for callers that import from ``waxmorph.graph``.
"""

from .torch.graph import (
    _wp_to_torch,
    build_edge_features,
    build_edge_index,
    build_graph,
    build_node_features,
)

__all__ = [
    "_wp_to_torch",
    "build_edge_features",
    "build_edge_index",
    "build_graph",
    "build_node_features",
]
