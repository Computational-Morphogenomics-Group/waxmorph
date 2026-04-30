"""Loss functions for non-growing tissue emulation.

Provides two losses as described in the WaxMorph writeup:

1. Squared loss (Frobenius norm squared) — when cells have a discrete
   one-to-one assignment between predicted and target positions.
2. Two-sided Chamfer distance — when no such assignment exists and
   rows need not correspond between predicted and target.
3. ``make_samples_loss`` — thin wrapper around ``geomloss.SamplesLoss``
   supporting all loss types (sinkhorn, hausdorff, energy, gaussian,
   laplacian) via a parameter dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from geomloss import SamplesLoss


def squared_loss(X_pred: torch.Tensor, X_target: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius norm between predicted and target positions.

    .. math::
        \\mathcal{L} = \\lVert X^f - X^T \\rVert_F^2

    Args:
        X_pred: Predicted positions with shape ``[N, 3]``.
        X_target: Target positions with shape ``[N, 3]`` in the same row order.

    Returns:
        Scalar tensor containing the squared Frobenius norm.
    """
    return (X_pred - X_target).pow(2).sum()


def chamfer_distance(X_pred: torch.Tensor, X_target: torch.Tensor) -> torch.Tensor:
    """Two-sided Chamfer distance between predicted and target point clouds.

    .. math::
        \\mathcal{L} = \\frac{1}{N} \\left[
            \\sum_i \\min_j \\lVert X^f_i - X^T_j \\rVert
          + \\sum_j \\min_i \\lVert X^f_i - X^T_j \\rVert
        \\right]

    Args:
        X_pred: Predicted positions with shape ``[N, 3]``.
        X_target: Target positions with shape ``[M, 3]``. ``M`` may differ
            from ``N``.

    Returns:
        Scalar tensor containing the two-sided Chamfer distance normalized by
        ``N``.
    """
    # [N, M]
    diff = X_pred.unsqueeze(1) - X_target.unsqueeze(0)
    dist = diff.norm(dim=-1)

    # pred -> target: for each predicted point, nearest target
    min_pred_to_target = dist.min(dim=1).values.sum()

    # target -> pred: for each target point, nearest predicted
    min_target_to_pred = dist.min(dim=0).values.sum()

    n = X_pred.size(0)
    return (min_pred_to_target + min_target_to_pred) / n


# ---------------------------------------------------------------------------
# geomloss.SamplesLoss wrapper
# ---------------------------------------------------------------------------

# All parameters accepted by geomloss.SamplesLoss with their defaults
# (matching https://www.kernel-operations.io/geomloss/api/pytorch-api.html).
SAMPLES_LOSS_DEFAULTS: dict[str, Any] = {
    "loss": "sinkhorn",
    "p": 2,
    "blur": 0.05,
    "reach": None,
    "diameter": None,
    "scaling": 0.5,
    "truncate": None,
    "cost": None,
    "kernel": None,
    "cluster_scale": None,
    "debias": True,
    "potentials": False,
    "verbose": False,
    "backend": "auto",
}


def make_samples_loss(params: dict[str, Any] | None = None, **kwargs: Any) -> SamplesLoss:
    """Create a ``geomloss.SamplesLoss`` from a parameter dict.

    Args:
        params: Optional ``SamplesLoss`` keyword arguments. Missing keys fall
            back to geomloss defaults listed in ``SAMPLES_LOSS_DEFAULTS``.
        **kwargs: Additional overrides merged on top of ``params``.

    Returns:
        Configured :class:`geomloss.SamplesLoss` instance.

    Raises:
        ImportError: If ``geomloss`` is not installed.
    """
    from geomloss import SamplesLoss

    merged: dict[str, Any] = {}
    if params is not None:
        merged.update(params)
    merged.update(kwargs)

    # Only forward keys that SamplesLoss actually accepts
    filtered = {k: v for k, v in merged.items() if k in SAMPLES_LOSS_DEFAULTS and v is not None}

    # hausdorff requires an explicit kernel function; default to energy_kernel
    if filtered.get("loss") == "hausdorff" and filtered.get("kernel") is None:
        from geomloss.kernel_samples import energy_kernel

        filtered["kernel"] = energy_kernel

    return SamplesLoss(**filtered)
