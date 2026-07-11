"""Explicit JAX/Equinox backend; top-level :mod:`waxmorph` imports use PyTorch.

JAX uses padded static graph buffers, Optax, explicit PRNG keys, and custom VJPs on CUDA
for Warp physics. Its loss factory provides OTT Sinkhorn divergence; PyTorch provides
GeomLoss families.
"""

from .gnn import GNS, GraphNetworkBlock
from .graph import (
    build_edge_features,
    build_edge_index,
    build_graph,
    build_node_features,
)
from .losses import chamfer_distance, make_sinkhorn_loss, squared_loss
from .mlp import MLP
from .train import TrainConfig, TrainResult, train

__all__ = [
    "GNS",
    "MLP",
    "GraphNetworkBlock",
    "TrainConfig",
    "TrainResult",
    "build_edge_features",
    "build_edge_index",
    "build_graph",
    "build_node_features",
    "chamfer_distance",
    "make_sinkhorn_loss",
    "squared_loss",
    "train",
]
