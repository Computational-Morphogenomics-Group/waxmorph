Forward and inverse modes
=========================

Forward simulation and inverse emulation share the spheroidal cell state
described in :doc:`cell_state`, and each defines its own dynamics. Coordinates,
radii, timesteps, and coefficients are model-scale values. External calibration
maps them to SI or biological scales when an application requires that
interpretation.

Mechanics and diffusion in each mode
------------------------------------

Both modes use local pair interactions and graph-style molecular diffusion.
The emulator keeps a fixed agent count and uses one uniform sticky-sphere force.
For stabilized
distance :math:`d=\lVert x_i-x_j\rVert_2+\epsilon_n`, direction
:math:`u=(x_i-x_j)/d`, and :math:`s=r_i+r_j`,

.. math::

   F_{ij} = \left[2\max(s-\epsilon_d-d,0)
   -0.5\max(s+\epsilon_d-d,0)\mathbb{1}[d>s-\epsilon_d]\right]u,

where :math:`\epsilon_d=0.01` in model units. The emulator's Euler update adds
this force. The simulator subtracts a potential-gradient buffer, uses
``K_REP=3``, type-dependent attraction and ranges, and ``EPS_DIST=0.25`` for
contact-local polarity and chemistry terms.

For emulator concentrations, the graph Laplacian is

.. math::

   (L_G c)_i=\sum_{j:(i,j)\in E}(c_i-c_j), \qquad
   c_i' = \max(c_i-D_{emu}\Delta t\,(L_Gc)_i,0).

Simulator chemistry stores activator and inhibitor abundances and converts them
to concentrations using cell volume.

Forward mode: prescribed case study
-----------------------------------

:mod:`waxmorph.simulator` implements a polarized epithelial-mesenchymal
aggregate with type-dependent mechanics, activator-inhibitor chemistry, growth,
and division. The case study defines a prescribed model; application-specific
calibration connects it to a biological system.

With :math:`a=A/(V+\epsilon)`, :math:`i=I/(V+\epsilon)`, and contact fluxes
:math:`\ell_A=\sum_j(a_j-a_i)` and :math:`\ell_I=\sum_j(i_j-i_i)`, one chemistry
step is

.. math::

   \Phi_A &= \min\!\left(\frac{a^2}{i+\epsilon},
                 \frac{a^2}{i^2+\epsilon}\right), \\
   A' &= \operatorname{clip}\!\left(
          \frac{A+\Delta t\,\gamma(\chi D_I\ell_A+\Phi_A)}
               {1+\Delta t\,\gamma},0,10^4\right), \\
   I' &= \operatorname{clip}\!\left(
          \frac{I+\Delta t\,\gamma(D_I\ell_I+a^2)}
               {1+\Delta t\,\gamma},0,10^4\right).

Optional cell-type masking selects the cells that receive reaction and damping.
The remaining cells follow an explicit diffusion path, and every output stays
within the clamp bounds. The reacting-cell update includes the production cap
and semi-implicit damping.

Physical radii below their targets grow monotonically up to those targets;
radii at or above target remain unchanged. Mesenchymal equilibrium radii are capped
at ``R_max``. Epithelial equilibrium radii copy through while physical radii
approach ``R_ref``. Division uses bounded atomic slot reservation, so the active
count remains within preallocated capacity. Both daughters split chemical
abundance; mesenchymal daughters rescale radius for half volume, while
epithelial daughters retain the parent radius.

Inverse mode: learned emulator with a fixed agent count
-------------------------------------------------------

The emulator infers a local rollout rule from source and target point clouds.
It combines GNS updates with its own sticky-sphere mechanics and graph diffusion
for a fixed particle population. The top-level API uses PyTorch;
:mod:`waxmorph.jax` provides a counterpart with backend-specific graph tuples,
loss families, PRNG, checkpoint, and runtime contracts.

Each model-evaluation step:

1. Builds a learned contact graph from the current state.
2. Predicts local position, polarity, and concentration increments with the GNS.
3. Applies the learned increments.
4. Applies configured differentiable mechanics and diffusion corrections.
5. Accumulates shape loss at selected target frames.
