"""PyTorch (default) backend for the waxMorph GNS learning pipeline.

This is the backend resolved by the top-level package imports: ``from waxmorph import GNS,
train, build_graph`` lands here, and the modules ``waxmorph.{gnn,graph,losses,mlp,train}``
are thin shims forwarding to these implementations. It provides the autograd-trained
graph-network simulator over the shared Warp physics core (gradients flow through
:mod:`waxmorph.torch.warp_autograd`).

Parity: :mod:`waxmorph.jax` mirrors this surface as an Equinox/Optax parity backend; keep
the two in sync when changing graph construction, losses, or training. The torch losses
expose ``make_samples_loss`` (geomloss Sinkhorn) where jax exposes ``make_sinkhorn_loss``.

See Also:
    waxmorph.jax: The JAX/Equinox parity backend.
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
