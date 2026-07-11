"""JAX losses for ordered arrays and unordered point clouds.

``squared_loss`` uses row correspondence; Chamfer and Sinkhorn support unordered clouds.
The OTT factory provides debiased Sinkhorn divergence; PyTorch also provides GeomLoss
families.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from waxmorph._validation import _validate_chamfer_shapes, _validate_same_shape


def squared_loss(X_pred: jnp.ndarray, X_target: jnp.ndarray) -> jnp.ndarray:
    r"""Squared Frobenius norm for row-aligned arrays of equal shape.

    Use :func:`chamfer_distance` or :func:`make_sinkhorn_loss` for unordered rows.

    .. math::
        \mathcal{L} = \lVert X^f - X^T \rVert_F^2

    Raises:
        ValueError: If the input shapes differ.

    Examples:
        >>> x = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> y = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> print(float(squared_loss(x, y)))
        1.0
    """
    _validate_same_shape("squared_loss", X_pred.shape, X_target.shape)
    return jnp.sum((X_pred - X_target) ** 2)


def chamfer_distance(X_pred: jnp.ndarray, X_target: jnp.ndarray) -> jnp.ndarray:
    r"""Two directional mean distances between unordered, possibly unequal clouds.

    Euclidean nearest-neighbor terms provide permutation invariance and exchange symmetry:

    .. math::

        \mathcal{L} = \frac{1}{N}\sum_i \min_j \lVert X^f_i - X^T_j \rVert
        + \frac{1}{M}\sum_j \min_i \lVert X^f_i - X^T_j \rVert

    JAX splits a ``min`` cotangent equally among tied minima; the derivative remains
    nonsmooth because the tied set changes under perturbation. Coincident distances use a
    finite zero-gradient branch.

    Raises:
        ValueError: If either cloud is empty, has another rank, or has a different width.

    Examples:
        >>> x = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> y = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> print(float(chamfer_distance(x, y)))
        1.0
    """
    _validate_chamfer_shapes(X_pred.shape, X_target.shape)
    diff = X_pred[:, None, :] - X_target[None, :, :]
    sq = jnp.sum(diff * diff, axis=-1)
    dist = jnp.where(sq > 0.0, jnp.sqrt(jnp.where(sq > 0.0, sq, 1.0)), 0.0)
    return dist.min(axis=1).mean() + dist.min(axis=0).mean()


def make_sinkhorn_loss(
    blur: float = 0.05,
    *,
    p: int = 2,
    cost_fn: Any | None = None,
    **solve_kwargs: Any,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    r"""Build OTT's debiased entropic-OT divergence for unordered clouds.

    .. math::
        S_\varepsilon(\alpha,\beta)=\mathrm{OT}_\varepsilon(\alpha,\beta)
        -\tfrac12\mathrm{OT}_\varepsilon(\alpha,\alpha)
        -\tfrac12\mathrm{OT}_\varepsilon(\beta,\beta).

    ``blur`` sets ``epsilon = blur ** p``. With ``cost_fn=None``, ``p=1`` selects Euclidean
    cost and ``p=2`` selects half squared Euclidean cost; other exponents use an explicit
    cost. An explicit cost replaces that selection, while ``p`` still sets the epsilon
    exponent. ``solve_kwargs`` are forwarded to OTT.

    The factory provides debiased Sinkhorn divergence with OTT options. PyTorch's GeomLoss
    factory additionally provides MMD, Hausdorff, ``scaling``, and ``reach`` contracts.
    Numerical values follow each solver's options. OTT is imported lazily.

    Raises:
        ImportError: If importing ``ott-jax`` fails.
    """
    from ott.geometry import costs, pointcloud
    from ott.tools import sinkhorn_divergence as sd

    if cost_fn is None:
        if p == 1:
            cost_fn = costs.Euclidean()
        elif p == 2:
            cost_fn = costs.PNormP(2)
        else:
            raise ValueError(
                f"make_sinkhorn_loss maps only p in {{1, 2}} to a built-in OTT "
                f"cost (got p={p!r}); pass an explicit `cost_fn` for other "
                "exponents."
            )

    def loss_fn(X_pred: jnp.ndarray, X_target: jnp.ndarray) -> jnp.ndarray:
        return sd.sinkhorn_divergence(
            pointcloud.PointCloud,
            X_pred,
            X_target,
            cost_fn=cost_fn,
            epsilon=blur**p,
            solve_kwargs=solve_kwargs,
        )[0]

    return loss_fn
