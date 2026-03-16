"""Backward-compatible re-export — actual implementation in waxmorph.torch.graph."""

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
