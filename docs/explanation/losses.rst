Shape losses
============

The shape loss measures how close a predicted cell population is to a target
morphology. :mod:`waxmorph.losses` provides both families.

When cell identity is known
---------------------------

``squared_loss`` applies when each predicted cell has a known target identity
and row order is meaningful — the loss compares cell :math:`i` against target
:math:`i` directly. Use it only when the experiment tracks cell identities.

When the target is an unordered cloud
-------------------------------------

For biological shapes sampled from meshes, the rows of the predicted and target
clouds are unordered, so the loss must be a distributional distance between
point sets. ``chamfer_distance`` and ``make_samples_loss`` apply here.

``chamfer_distance`` sums the mean nearest-neighbor distance in each direction,
so exchanging clouds with different point counts does not change the value:

.. math::

   d_{\mathrm{Ch}}(X,Y) = \frac{1}{|X|}\sum_{x\in X}\min_{y\in Y}\lVert x-y\rVert_2
   + \frac{1}{|Y|}\sum_{y\in Y}\min_{x\in X}\lVert x-y\rVert_2.

Both clouds must be nonempty rank-2 arrays with the same feature width.

The shape loss sums a distributional distance over the supervised goal frames,

.. math::

   L_{\text{shape}} = \sum_{k=1}^{K} d\big(X_{\tau_k}, \tilde{X}_{\tau_k}\big),

reducing to :math:`d(X_T, \tilde{X}_T)` for a single terminal target. The
GeomLoss-backed ``make_samples_loss`` exposes maximum mean discrepancy,
Hausdorff divergence, and debiased Sinkhorn divergence. The default is the
Sinkhorn divergence, a fast approximation of the 2-Wasserstein distance between
the empirical measures of the two clouds.

Regularizing the trajectory
---------------------------

Supervising only a few goal frames leaves the in-between motion underconstrained.
A regularizer penalizes large position changes between consecutive frames,

.. math::

   L_{\text{reg}} = \lambda \sum_{t=1}^{T-1}
   \lVert X_t - X_{t+1} \rVert_F^2,

with strength :math:`\lambda` (the ``lambda_reg`` knob) and
:math:`\lVert \cdot \rVert_F` the Frobenius norm. It discourages trajectories
that satisfy the goals only at the supervised time points. The total trajectory
loss optimized by the emulator is :math:`L = L_{\text{shape}} + L_{\text{reg}}`.
