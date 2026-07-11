"""Default PyTorch backend.

Import :mod:`waxmorph.jax` explicitly for JAX. The backends differ in loss factories,
graph return shapes, and Warp gradient bridges.
"""

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
    "SAMPLES_LOSS_DEFAULTS",
    "GraphNetworkBlock",
    "TrainConfig",
    "TrainResult",
    "build_edge_features",
    "build_edge_index",
    "build_graph",
    "build_node_features",
    "chamfer_distance",
    "make_samples_loss",
    "squared_loss",
    "train",
]
