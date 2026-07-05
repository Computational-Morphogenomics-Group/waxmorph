"""Backward-compatible shim re-exporting the default (PyTorch) shape losses.

``from waxmorph.losses import ...`` resolves to the torch backend.
``SAMPLES_LOSS_DEFAULTS`` documents every accepted :class:`geomloss.SamplesLoss`
keyword and its default. See :mod:`waxmorph.torch.losses` for full docs and
:mod:`waxmorph.jax.losses` for the JAX parity backend.
"""

from .torch.losses import (
    SAMPLES_LOSS_DEFAULTS,
    chamfer_distance,
    make_samples_loss,
    squared_loss,
)

__all__ = [
    "SAMPLES_LOSS_DEFAULTS",
    "chamfer_distance",
    "make_samples_loss",
    "squared_loss",
]
