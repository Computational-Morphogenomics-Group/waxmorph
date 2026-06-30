"""Shape-matching losses for non-growing tissue emulation (JAX backend).

The emulator deforms an unordered population of agents toward a target
morphology, so the loss must compare two point sets rather than two indexed
arrays. This is the JAX/OTT parity backend of :mod:`waxmorph.torch.losses`;
``squared_loss`` and ``chamfer_distance`` mirror the torch side one-to-one.

Provides:

1. ``squared_loss`` (Frobenius norm squared) — pick when each predicted cell
   has a *known* target identity and row order is meaningful, so position
   ``i`` of the prediction must match position ``i`` of the target.
2. ``chamfer_distance`` (two-sided, normalized) — pick for *unordered* point
   clouds sampled from shapes, the realistic biological case where row order
   carries no meaning and the two clouds may differ in size.
3. ``make_sinkhorn_loss`` — factory returning a debiased Sinkhorn divergence
   callable using ``ott-jax``. It is the JAX counterpart of the PyTorch
   :func:`waxmorph.torch.losses.make_samples_loss`, but Sinkhorn-only: the
   broader GeomLoss MMD/Hausdorff families are not exposed on this backend.

See Also:
    waxmorph.torch.losses: PyTorch default backend with the same
        ``squared_loss`` and ``chamfer_distance``, plus
        :func:`waxmorph.torch.losses.make_samples_loss` covering the full
        GeomLoss family (Sinkhorn/MMD/Hausdorff).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp


def squared_loss(X_pred: jnp.ndarray, X_target: jnp.ndarray) -> jnp.ndarray:
    r"""Squared Frobenius norm between row-aligned predicted and target positions.

    Pick this only when each predicted cell has a known target identity and row
    order is meaningful, so prediction row ``i`` is supposed to land on target
    row ``i``. For unordered point clouds sampled from shapes (the realistic
    biological case) use :func:`chamfer_distance` or :func:`make_sinkhorn_loss`,
    which are invariant to row permutations.

    .. math::
        \mathcal{L} = \lVert X^f - X^T \rVert_F^2

    Args:
        X_pred: Predicted positions with shape ``[N, 3]``.
        X_target: Target positions with shape ``[N, 3]`` in the same row order.

    Returns:
        Scalar JAX array containing the squared Frobenius norm.

    See Also:
        waxmorph.torch.losses.squared_loss: PyTorch twin with identical
            semantics.

    Examples:
        >>> x = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> y = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> print(float(squared_loss(x, y)))
        1.0
    """
    return jnp.sum((X_pred - X_target) ** 2)


def chamfer_distance(X_pred: jnp.ndarray, X_target: jnp.ndarray) -> jnp.ndarray:
    r"""Two-sided Chamfer distance between unordered predicted and target clouds.

    Pick this (or :func:`make_sinkhorn_loss`) for unordered point clouds sampled
    from shapes — the realistic biological case — where row order carries no
    meaning and the two clouds may differ in size. Each point is matched to its
    nearest neighbor in the other cloud and both directions are summed, so the
    loss is permutation-invariant. This is the symmetric Chamfer term used in
    waxMorph; unlike the normalized Chamfer distance (NCD) used for evaluation,
    the per-point terms here use Euclidean distances rather than squared
    distances.

    The nearest-neighbor ``min`` is non-smooth at ties; autodiff takes the
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
        Scalar JAX array containing the two-sided Chamfer distance normalized
        by ``N``.

    See Also:
        waxmorph.torch.losses.chamfer_distance: PyTorch twin with identical
            semantics.
        make_sinkhorn_loss: Sinkhorn divergence for the same unordered-cloud
            regime.

    Examples:
        >>> x = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> y = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> print(float(chamfer_distance(x, y)))
        1.0
    """
    # [N, M]
    diff = X_pred[:, None, :] - X_target[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)

    # pred -> target: for each predicted point, nearest target
    min_pred_to_target = dist.min(axis=1).sum()

    # target -> pred: for each target point, nearest predicted
    min_target_to_pred = dist.min(axis=0).sum()

    n = X_pred.shape[0]
    return (min_pred_to_target + min_target_to_pred) / n


def make_sinkhorn_loss(
    blur: float = 0.05,
    *,
    p: int = 2,
    cost_fn: Any | None = None,
    **solve_kwargs: Any,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Build a debiased Sinkhorn divergence loss for unordered point clouds.

    Use this for the realistic biological case where predicted and target cells
    are unordered samples from two shapes. The Sinkhorn divergence is a fast
    entropic approximation of the 2-Wasserstein distance and is the default
    shape loss in waxMorph. It is permutation-invariant and the
    returned callable is differentiable through ``ott-jax``.

    Uses ``ott.tools.sinkhorn_divergence`` to compute the three-term debiased
    Sinkhorn divergence, matching the behavior of :class:`geomloss.SamplesLoss`
    with ``debias=True`` (the PyTorch default).

    This is the JAX counterpart of :func:`waxmorph.torch.losses.make_samples_loss`,
    but Sinkhorn-only: the torch wrapper additionally exposes the GeomLoss
    MMD/Hausdorff families, which are not provided on this backend.

    The debiased divergence is:

    .. math::
        S_\\varepsilon(\\alpha, \\beta)
        = \\mathrm{OT}_\\varepsilon(\\alpha, \\beta)
        - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(\\alpha, \\alpha)
        - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(\\beta, \\beta)

    This ensures :math:`S_\\varepsilon(\\alpha, \\alpha) \\approx 0`.

    Args:
        blur: Entropic regularization parameter, passed to OTT as ``epsilon``
            (the analogue of the geomloss ``blur`` knob).
        p: Ground-cost exponent selecting the OTT cost when ``cost_fn`` is not
            given: ``p=2`` uses squared Euclidean cost
            (:class:`ott.geometry.costs.SqEuclidean`, the OTT default and the
            torch-side default) and ``p=1`` uses Euclidean cost
            (:class:`ott.geometry.costs.Euclidean`). Other exponents require an
            explicit ``cost_fn``.
        cost_fn: Optional explicit OTT :class:`~ott.geometry.costs.CostFn`. When
            given it overrides ``p``.
        **solve_kwargs: Additional Sinkhorn solver options forwarded to OTT as
            ``solve_kwargs`` (e.g. ``threshold``, ``max_iterations``).

    Note:
        The divergence is always the debiased three-term form, matching the
        torch default ``geomloss.SamplesLoss(debias=True)``; this backend does
        not expose a debias toggle. The geomloss ``scaling`` (multiscale
        annealing) and ``reach`` (unbalanced OT) knobs have no direct OTT mapping
        here and are not exposed.

    Returns:
        Callable ``loss_fn(X_pred, X_target)`` returning a scalar
        :class:`jax.Array` Sinkhorn divergence.

    Raises:
        ImportError: If ``ott-jax`` is not installed.

    See Also:
        waxmorph.torch.losses.make_samples_loss
            PyTorch counterpart exposing the full GeomLoss family
            (Sinkhorn/MMD/Hausdorff); this factory is Sinkhorn-only.
        chamfer_distance
            Lightweight unordered-cloud loss with no extra dependencies.
    """
    from ott.geometry import costs, pointcloud
    from ott.tools import sinkhorn_divergence as sd

    if cost_fn is None:
        if p == 1:
            cost_fn = costs.Euclidean()
        elif p == 2:
            cost_fn = costs.SqEuclidean()
        else:
            raise ValueError(
                f"make_sinkhorn_loss maps only p in {{1, 2}} to a built-in OTT "
                f"cost (got p={p!r}); pass an explicit `cost_fn` for other "
                "exponents."
            )

    def loss_fn(X_pred: jnp.ndarray, X_target: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the configured Sinkhorn divergence between point clouds."""
        divergence, _ = sd.sinkhorn_divergence(
            pointcloud.PointCloud,
            X_pred,
            X_target,
            cost_fn=cost_fn,
            epsilon=blur,
            solve_kwargs=solve_kwargs if solve_kwargs else {},
        )
        return divergence

    return loss_fn
