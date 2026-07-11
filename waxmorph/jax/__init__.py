"""Explicit JAX/Equinox backend; top-level :mod:`waxmorph` imports use PyTorch.

JAX uses padded static graph buffers, Optax, explicit PRNG keys, and CUDA-only custom VJPs
for Warp physics. Its loss factory provides OTT Sinkhorn divergence rather than PyTorch's
broader GeomLoss family.
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
