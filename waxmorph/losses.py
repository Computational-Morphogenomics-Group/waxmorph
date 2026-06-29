"""Backward-compatible shim re-exporting the default (PyTorch) shape losses.

Forwards :func:`waxmorph.torch.losses.squared_loss`,
:func:`waxmorph.torch.losses.chamfer_distance`,
:func:`waxmorph.torch.losses.make_samples_loss`, and the
``SAMPLES_LOSS_DEFAULTS`` table so ``from waxmorph.losses import ...`` keeps
resolving to the torch backend. ``SAMPLES_LOSS_DEFAULTS`` is part of the public
surface (it documents every accepted :class:`geomloss.SamplesLoss` keyword and
its default) and is intentionally re-exported. See :mod:`waxmorph.torch.losses`
for the full documentation and :mod:`waxmorph.jax.losses` for the JAX parity
backend.
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
