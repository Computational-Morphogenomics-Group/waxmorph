"""PyTorch (default) backend for the waxMorph GNS learning pipeline.

Top-level package imports resolve here (``from waxmorph import GNS, train, build_graph``),
and ``waxmorph.{gnn,graph,losses,mlp,train}`` are thin shims forwarding to these
implementations. Gradients flow through :mod:`waxmorph.torch.warp_autograd` over the
shared Warp physics core.

Parity: :mod:`waxmorph.jax` mirrors this surface as an Equinox/Optax parity backend; keep
the two in sync when changing graph construction, losses, or training. The torch losses
expose ``make_samples_loss`` (geomloss Sinkhorn) where jax exposes ``make_sinkhorn_loss``.
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
