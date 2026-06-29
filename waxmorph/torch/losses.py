"""Shape-matching losses for non-growing tissue emulation (PyTorch backend).

The emulator deforms an unordered population of agents toward a target
morphology, so the loss must compare two point sets rather than two indexed
arrays. The three exposed losses span the two regimes of shape matching.

Provides:

1. ``squared_loss`` (Frobenius norm squared) — pick when each predicted cell
   has a *known* target identity and row order is meaningful, so position
   ``i`` of the prediction must match position ``i`` of the target.
2. ``chamfer_distance`` (two-sided, normalized) — pick for *unordered* point
   clouds sampled from shapes, the realistic biological case where row order
   carries no meaning and the two clouds may differ in size.
3. ``make_samples_loss`` — thin wrapper around :class:`geomloss.SamplesLoss`
   exposing the GeomLoss distributional-distance families used in waxMorph
   (debiased Sinkhorn ~ 2-Wasserstein, MMD/energy and Gaussian or
   Laplacian kernels, Hausdorff). Like :func:`chamfer_distance`, these treat
   the inputs as unordered samples from two shapes; Sinkhorn is the default.

See Also:
    waxmorph.jax.losses: JAX/OTT parity backend. ``squared_loss`` and
        ``chamfer_distance`` mirror these one-to-one; the JAX side exposes
        :func:`waxmorph.jax.losses.make_sinkhorn_loss` (Sinkhorn-only via
        ``ott-jax``) in place of the broader :func:`make_samples_loss`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from geomloss import SamplesLoss


def squared_loss(X_pred: torch.Tensor, X_target: torch.Tensor) -> torch.Tensor:
    r"""Squared Frobenius norm between row-aligned predicted and target positions.

    Pick this only when each predicted cell has a known target identity and row
    order is meaningful, so prediction row ``i`` is supposed to land on target
    row ``i``. For unordered point clouds sampled from shapes (the realistic
    biological case) use :func:`chamfer_distance` or :func:`make_samples_loss`,
    which are invariant to row permutations.

    .. math::
        \mathcal{L} = \lVert X^f - X^T \rVert_F^2

    Args:
        X_pred: Predicted positions with shape ``[N, 3]``.
        X_target: Target positions with shape ``[N, 3]`` in the same row order.

    Returns:
        Scalar :class:`torch.Tensor` containing the squared Frobenius norm.

    See Also:
        waxmorph.jax.losses.squared_loss: JAX twin with identical semantics.

    Examples:
        >>> x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> y = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> print(squared_loss(x, y))
        tensor(1.)
    """
    return (X_pred - X_target).pow(2).sum()


def chamfer_distance(X_pred: torch.Tensor, X_target: torch.Tensor) -> torch.Tensor:
    r"""Two-sided Chamfer distance between unordered predicted and target clouds.

    Pick this (or :func:`make_samples_loss`) for unordered point clouds sampled
    from shapes — the realistic biological case — where row order carries no
    meaning and the two clouds may differ in size. Each point is matched to its
    nearest neighbor in the other cloud and both directions are summed, so the
    loss is permutation-invariant. This is the symmetric Chamfer term used in
    waxMorph; unlike the normalized Chamfer distance (NCD) used for evaluation,
    the per-point terms here use Euclidean distances rather than squared
    distances.

    The nearest-neighbor ``min`` is non-smooth at ties; autograd takes the
    subgradient of whichever neighbor index is selected.

    .. math::

        \mathcal{L} = \frac{1}{N} \left[
        \sum_i \min_j \lVert X^f_i - X^T_j \rVert
        + \sum_j \min_i \lVert X^f_i - X^T_j \rVert
        \right]

    Args:
        X_pred: Predicted positions with shape ``[N, 3]``.
        X_target: Target positions with shape ``[M, 3]``. ``M`` may differ
            from ``N``.

    Returns:
        Scalar :class:`torch.Tensor` containing the two-sided Chamfer distance
        normalized by ``N``.

    See Also:
        waxmorph.jax.losses.chamfer_distance
            JAX twin with identical semantics.
        make_samples_loss
            GeomLoss families (Sinkhorn/MMD/Hausdorff) for the same
            unordered-cloud regime.

    Examples:
        >>> x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> y = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> print(chamfer_distance(x, y))
        tensor(1.)
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
# (matching the geomloss SamplesLoss API).
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
    """Build a GeomLoss distributional distance for unordered point clouds.

    Use this for the realistic biological case where predicted and target cells
    are unordered samples from two shapes, so the loss must compare empirical
    measures rather than indexed rows. The returned callable is invariant to row
    permutations and tolerates differing cloud sizes. The ``loss`` key selects
    the GeomLoss family:

    - ``"sinkhorn"`` (default): debiased Sinkhorn divergence, a fast entropic
      approximation of the 2-Wasserstein distance (``blur`` is the entropic
      regularization; ``debias=True`` makes self-distance ~ 0).
    - ``"gaussian"`` / ``"laplacian"`` / ``"energy"``: maximum mean discrepancy
      (MMD) under the corresponding kernel.
    - ``"hausdorff"``: Hausdorff divergence (a kernel function is required, so a
      missing ``kernel`` defaults to ``energy_kernel``).

    Args:
        params: Optional :class:`geomloss.SamplesLoss` keyword arguments.
            Missing keys fall back to geomloss defaults listed in
            ``SAMPLES_LOSS_DEFAULTS``.
        **kwargs: Additional overrides merged on top of ``params``.

    Returns:
        Configured :class:`geomloss.SamplesLoss` instance, callable as
        ``loss(X_pred, X_target)``.

    Raises:
        ImportError: If ``geomloss`` is not installed.

    See Also:
        waxmorph.jax.losses.make_sinkhorn_loss
            JAX counterpart, but Sinkhorn-only (via ``ott-jax``) rather than
            the full GeomLoss family.
        chamfer_distance
            Lightweight unordered-cloud loss with no extra dependencies.
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
        try:
            from geomloss.kernel_samples import energy_kernel
        except ImportError:
            from geomloss._legacy.kernel_samples import energy_kernel

        filtered["kernel"] = energy_kernel

    return SamplesLoss(**filtered)
