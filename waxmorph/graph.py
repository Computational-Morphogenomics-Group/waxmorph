"""Graph construction surface.

Re-exports the PyTorch implementation (:mod:`waxmorph.torch.graph`) for callers
importing from ``waxmorph.*``. PyTorch is the default backend; reach the JAX
parity builders explicitly via :mod:`waxmorph.jax.graph`.
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
