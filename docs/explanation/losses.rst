Shape losses
============

Shape losses compare predicted positions with target point clouds.

When cell identity is known
---------------------------

``squared_loss`` applies when each predicted cell has a known target identity
and row order is meaningful: the loss compares cell :math:`i` against target
:math:`i` directly. Input shapes must match exactly. Use it for experiments
that track cell identities.

When the target is an unordered cloud
-------------------------------------

For biological shapes sampled from meshes, the rows of the predicted and target
clouds are unordered, so the loss must be a distributional distance between
point sets. ``chamfer_distance`` and ``make_samples_loss`` apply here.

``chamfer_distance`` sums the mean nearest-neighbor distance in each direction,
so swapping the two clouds gives the same value even when their point counts
differ:

.. math::

   d_{\mathrm{Ch}}(X,Y) = \frac{1}{|X|}\sum_{x\in X}\min_{y\in Y}\lVert x-y\rVert_2
   + \frac{1}{|Y|}\sum_{y\in Y}\min_{x\in X}\lVert x-y\rVert_2.

Both clouds must be nonempty rank-2 arrays with the same feature width.

Training sums the selected distance over supervised frames,

.. math::

   L_{\text{shape}} = \sum_{k=1}^{K} d\big(X_{\tau_k}, \tilde{X}_{\tau_k}\big),

which reduces to :math:`d(X_T, \tilde{X}_T)` for a single terminal target.
PyTorch's GeomLoss-backed ``make_samples_loss`` exposes maximum mean
discrepancy, Hausdorff divergence, and debiased Sinkhorn divergence. JAX's
``make_sinkhorn_loss`` exposes debiased OTT Sinkhorn divergence. Numerical
values follow each backend's cost, solver, and option contract. In the
PyTorch factory, omitted ``truncate`` uses ``5`` while explicit ``None`` is
preserved.

Regularizing the trajectory
---------------------------

The training regularizer penalizes the learned position head before prescribed
mechanics,

.. math::

   L_{\text{reg}} = \sum_t
   \lVert \Delta t_{\mathrm{GNS}}\,\mathrm{GNS}_{X,t} \rVert_F^2.

``lambda_reg`` supplies its weight, so the optimized loss is
:math:`L=L_{\mathrm{shape}}+\lambda_{\mathrm{reg}}L_{\mathrm{reg}}`.
This regularizer measures learned displacement; the physics update governs
prescribed mechanical displacement.
