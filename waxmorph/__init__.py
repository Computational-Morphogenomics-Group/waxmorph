"""Differentiable morphogenesis and shape assembly on NVIDIA Warp.

waxMorph does joint forward *simulation* and inverse *learning* of biophysical shape
assembly, modelling tissue as interacting spheroidal agents (position, radius, polarity,
gene/morphogen state, cell type).

Mental model: a two-backend design over a shared Warp physics core, plus rendering. The
differentiable physics lives in Warp kernels (:mod:`~waxmorph.simulator`,
:mod:`~waxmorph.emulator`) and is made differentiable to the learning framework by the
``warp_autograd`` bridge, which records kernel launches on a :class:`warp.Tape` and replays
it in reverse for gradients. On top of that physics sit two interchangeable learning
backends: **PyTorch is the default** -- the names re-exported here resolve to
:mod:`waxmorph.torch`, so ``from waxmorph import GNS, train, build_graph`` gives the torch
implementations -- while :mod:`waxmorph.jax` is a **parity backend** reached only via
explicit ``from waxmorph.jax import ...`` (Optax optimiser, static-shape compilation).
Topology construction is shared (:mod:`waxmorph._graph_core`) so both backends build
identical graphs; rendering lives in :mod:`waxmorph.render`.

This module is a thin convenience layer: every name it binds is a re-export of the
corresponding :mod:`waxmorph.torch` symbol (the modules ``waxmorph.{gnn,graph,losses,mlp,
train}`` are themselves backward-compat shims forwarding into ``torch/``). To change
learning behaviour, edit :mod:`waxmorph.torch` and mirror it in :mod:`waxmorph.jax`; do
not add logic here.

The re-export surface (``__all__``) groups into:

* **Models** -- ``GNS`` (graph-network simulator), ``GraphNetworkBlock`` (one
  message-passing block), ``MLP`` (the per-update network).
* **Graph construction** -- ``build_graph`` and its parts ``build_edge_index``,
  ``build_edge_features``, ``build_node_features``.
* **Losses** -- ``squared_loss``, ``chamfer_distance``, ``make_samples_loss`` and its
  ``SAMPLES_LOSS_DEFAULTS`` (geomloss Sinkhorn config; the jax twin exposes
  ``make_sinkhorn_loss`` instead).
* **Training** -- ``train`` plus its ``TrainConfig`` / ``TrainResult`` dataclasses.
* **Metadata** -- ``__version__``.
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
