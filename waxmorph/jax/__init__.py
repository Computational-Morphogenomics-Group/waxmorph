"""JAX/Equinox (parity) backend for the waxMorph GNS learning pipeline.

This is the parity backend, reached only via explicit ``from waxmorph.jax import ...``; the
default top-level imports resolve to :mod:`waxmorph.torch`. It mirrors that backend's
public surface using Equinox modules, an Optax optimiser, and static-shape compilation,
and trains the graph-network simulator over the same shared Warp physics core (gradients
flow through :mod:`waxmorph.jax.warp_autograd` via a custom VJP).

Parity: :mod:`waxmorph.torch` is the template -- keep the two in sync when changing graph
construction, losses, or training, and do not weaken the Torch<->JAX parity assertions.
The one genuine divergence in this surface is losses: jax exposes ``make_sinkhorn_loss``
(ott-jax) where torch exposes ``make_samples_loss`` (geomloss) plus ``SAMPLES_LOSS_DEFAULTS``.

See Also:
    waxmorph.torch: The PyTorch default backend (the documentation template).
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
