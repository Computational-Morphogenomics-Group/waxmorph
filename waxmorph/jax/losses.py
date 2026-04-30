"""Loss functions for non-growing tissue emulation (JAX).

Provides:

1. Squared loss (Frobenius norm squared) — when cells have a discrete
   one-to-one assignment between predicted and target positions.
2. Two-sided Chamfer distance — when no such assignment exists and
   rows need not correspond between predicted and target.
3. ``make_sinkhorn_loss`` — factory returning a Sinkhorn divergence
   callable using ``ott-jax``, replacing the PyTorch ``geomloss`` wrapper.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def squared_loss(X_pred: jnp.ndarray, X_target: jnp.ndarray) -> jnp.ndarray:
    """Squared Frobenius norm between predicted and target positions.

    .. math::
        \\mathcal{L} = \\lVert X^f - X^T \\rVert_F^2

    Args:
        X_pred: Predicted positions with shape ``[N, 3]``.
        X_target: Target positions with shape ``[N, 3]`` in the same row order.

    Returns:
        Scalar JAX array containing the squared Frobenius norm.
    """
    return jnp.sum((X_pred - X_target) ** 2)


def chamfer_distance(X_pred: jnp.ndarray, X_target: jnp.ndarray) -> jnp.ndarray:
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
        Scalar JAX array containing the two-sided Chamfer distance normalized
        by ``N``.
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
    **kwargs: Any,
) -> callable:
    """Create a debiased Sinkhorn divergence loss function using ``ott-jax``.

    Uses ``ott.tools.sinkhorn_divergence`` to compute the three-term debiased
    Sinkhorn divergence, matching the behavior of ``geomloss.SamplesLoss``
    with ``debias=True`` (the PyTorch default).

    The debiased divergence is:

    .. math::
        S_\\varepsilon(\\alpha, \\beta)
        = \\mathrm{OT}_\\varepsilon(\\alpha, \\beta)
        - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(\\alpha, \\alpha)
        - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(\\beta, \\beta)

    This ensures :math:`S_\\varepsilon(\\alpha, \\alpha) \\approx 0`.

    Args:
        blur: Entropic regularization parameter, passed to OTT as ``epsilon``.
        **kwargs: Additional keyword arguments forwarded in ``solve_kwargs``.

    Returns:
        Callable ``loss_fn(X_pred, X_target)`` returning a scalar Sinkhorn
        divergence.

    Raises:
        ImportError: If ``ott-jax`` is not installed.
    """
    from ott.geometry import pointcloud
    from ott.tools import sinkhorn_divergence as sd

    def loss_fn(X_pred: jnp.ndarray, X_target: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the configured Sinkhorn divergence between point clouds."""
        divergence, _ = sd.sinkhorn_divergence(
            pointcloud.PointCloud,
            X_pred,
            X_target,
            epsilon=blur,
            solve_kwargs=kwargs if kwargs else {},
        )
        return divergence

    return loss_fn
