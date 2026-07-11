"""PyTorch losses for row-aligned or unordered point clouds.

``squared_loss`` requires row correspondence. Chamfer and GeomLoss samples are
permutation-invariant; JAX offers OTT Sinkhorn, while PyTorch offers GeomLoss families.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from waxmorph._validation import _validate_chamfer_shapes, _validate_same_shape

if TYPE_CHECKING:
    from geomloss import SamplesLoss


def squared_loss(X_pred: torch.Tensor, X_target: torch.Tensor) -> torch.Tensor:
    r"""Squared Frobenius loss for row-aligned clouds.

    .. math::
        \mathcal{L} = \lVert X^f - X^T \rVert_F^2

    Rows are fixed correspondences and shapes must match. Use :func:`chamfer_distance` or
    :func:`make_samples_loss` for unordered rows.

    Examples:
        >>> x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> y = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> print(squared_loss(x, y))
        tensor(1.)
    """
    _validate_same_shape("squared_loss", X_pred.shape, X_target.shape)
    return (X_pred - X_target).pow(2).sum()


def chamfer_distance(X_pred: torch.Tensor, X_target: torch.Tensor) -> torch.Tensor:
    r"""Sum of directional mean nearest-neighbor distances.

    .. math::

        \mathcal{L} = \frac{1}{N}\sum_i \min_j \lVert X^f_i - X^T_j \rVert
        + \frac{1}{M}\sum_j \min_i \lVert X^f_i - X^T_j \rVert

    The loss is permutation-invariant, permits unequal cloud sizes, and uses Euclidean
    distances. Nearest-neighbor ties are nonsmooth; ``torch.min`` routes the
    derivative through its selected index. Inputs must be nonempty rank-2 arrays with equal
    feature width.

    Examples:
        >>> x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> y = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> print(chamfer_distance(x, y))
        tensor(1.)
    """
    _validate_chamfer_shapes(X_pred.shape, X_target.shape)
    diff = X_pred.unsqueeze(1) - X_target.unsqueeze(0)
    dist = diff.norm(dim=-1)
    return dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean()


SAMPLES_LOSS_DEFAULTS: dict[str, Any] = {
    "loss": "sinkhorn",
    "p": 2,
    "blur": 0.05,
    "reach": None,
    "diameter": None,
    "scaling": 0.5,
    "truncate": 5,
    "cost": None,
    "kernel": None,
    "cluster_scale": None,
    "debias": True,
    "potentials": False,
    "verbose": False,
    "backend": "auto",
}


def make_samples_loss(params: dict[str, Any] | None = None, **kwargs: Any) -> SamplesLoss:
    """Return a lazily imported GeomLoss loss for unordered clouds.

    ``params`` and ``kwargs`` merge with ``kwargs`` taking precedence. Unspecified options
    use GeomLoss defaults, explicit ``None`` is retained, and unknown keys raise
    :class:`TypeError`. Supported families are Sinkhorn, Gaussian, Laplacian, energy, and
    Hausdorff. Hausdorff uses ``energy_kernel`` by default, including GeomLoss's legacy
    import fallback. JAX's factory provides OTT Sinkhorn divergence.
    """
    merged = dict(params) if params is not None else {}
    merged.update(kwargs)

    unknown = sorted(str(key) for key in merged if key not in SAMPLES_LOSS_DEFAULTS)
    if unknown:
        raise TypeError(f"Unknown SamplesLoss parameters: {', '.join(unknown)}")

    from geomloss import SamplesLoss

    if merged.get("loss") == "hausdorff" and merged.get("kernel") is None:
        try:
            from geomloss.kernel_samples import energy_kernel
        except ImportError:
            from geomloss._legacy.kernel_samples import energy_kernel

        merged["kernel"] = energy_kernel

    return SamplesLoss(**merged)
