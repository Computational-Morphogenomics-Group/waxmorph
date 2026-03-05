"""WaxMorph -- Differentiable morphogenesis and shape-assembly on NVIDIA Warp."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("waxmorph")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .gnn import GNS, GraphNetworkBlock
from .graph import (
    build_edge_index,
    build_edge_features,
    build_graph,
    build_node_features,
)
from .losses import chamfer_distance, squared_loss
from .mlp import MLP

__all__ = [
    "__version__",
    "GNS",
    "GraphNetworkBlock",
    "MLP",
    "build_edge_index",
    "build_edge_features",
    "build_graph",
    "build_node_features",
    "chamfer_distance",
    "squared_loss",
]
