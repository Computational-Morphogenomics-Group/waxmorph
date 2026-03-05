"""Loss functions for non-growing tissue emulation.

Provides two losses as described in the WaxMorph writeup:

1. Squared loss (Frobenius norm squared) — when cells have a discrete
   one-to-one assignment between predicted and target positions.
2. Two-sided Chamfer distance — when no such assignment exists and
   rows need not correspond between predicted and target.
"""

import torch


def squared_loss(X_pred: torch.Tensor, X_target: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius norm between predicted and target positions.

    .. math::
        \\mathcal{L} = \\lVert X^f - X^T \\rVert_F^2

    Parameters
    ----------
    X_pred : torch.Tensor ``[N, 3]``
        Predicted positions.
    X_target : torch.Tensor ``[N, 3]``
        Target positions (same ordering as predicted).

    Returns
    -------
    loss : scalar tensor
    """
    return (X_pred - X_target).pow(2).sum()


def chamfer_distance(X_pred: torch.Tensor, X_target: torch.Tensor) -> torch.Tensor:
    """Two-sided Chamfer distance between predicted and target point clouds.

    .. math::
        \\mathcal{L} = \\frac{1}{N} \\left[
            \\sum_i \\min_j \\lVert X^f_i - X^T_j \\rVert
          + \\sum_j \\min_i \\lVert X^f_i - X^T_j \\rVert
        \\right]

    Parameters
    ----------
    X_pred : torch.Tensor ``[N, 3]``
        Predicted positions.
    X_target : torch.Tensor ``[M, 3]``
        Target positions (may differ in count from predicted).

    Returns
    -------
    loss : scalar tensor
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
