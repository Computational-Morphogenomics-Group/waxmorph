"""Differentiable morphogenesis and shape assembly on NVIDIA Warp.

The top-level package exports the default PyTorch graph-network simulator,
graph construction utilities, shape losses, and training entry point. JAX
variants live under :mod:`waxmorph.jax`.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("waxmorph")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .gnn import GNS, GraphNetworkBlock
from .graph import (
    build_edge_features,
    build_edge_index,
    build_graph,
    build_node_features,
)
from .losses import SAMPLES_LOSS_DEFAULTS, chamfer_distance, make_samples_loss, squared_loss
from .mlp import MLP
from .train import TrainConfig, TrainResult, train

__all__ = [
    "GNS",
    "MLP",
    "GraphNetworkBlock",
    "TrainConfig",
    "TrainResult",
    "__version__",
    "build_edge_features",
    "build_edge_index",
    "build_graph",
    "build_node_features",
    "SAMPLES_LOSS_DEFAULTS",
    "chamfer_distance",
    "make_samples_loss",
    "squared_loss",
    "train",
]
