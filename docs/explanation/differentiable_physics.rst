Differentiable physics
======================

The learned emulator works because the prescribed physics is differentiable:
gradients of the shape loss flow back through the soft-sphere mechanics and the
graph diffusion, not only through the graph network. This page explains the two
ideas that make that possible — the frozen-topology gradient approximation and
the Warp tape that records and replays the physics.

Frozen topology within a step
-----------------------------

Each emulation step applies the learned GNS updates and then the prescribed
constraints. The constraints are the same primitives used everywhere in the
framework: the soft-sphere force pushes apart overlapping cells, and each latent
molecule diffuses through the shared graph Laplacian,

.. math::

   x_i^{t+1} \leftarrow x_i^{t+1}
   + \Delta t \sum_{j:\,(i,j)\in E^{t+1}} f^{\text{soft}}_{ij}, \qquad
   c_{i,M}^{t+1} \leftarrow c_{i,M}^{t+1}
   - D_{\text{emu}}\,(L_G c)_{i,M}^{t+1}\,\Delta t .

The constraints run ``n_substeps`` times per learned update on a faster time
scale, keeping the trajectory biophysically coherent.

The differentiable path in :mod:`waxmorph.emulator` records these pairwise
kernels while *freezing the neighbor topology for the step*. The edge set
:math:`E^{t+1}` is rebuilt from the provisional positions, then held fixed while
the constraint forces and diffusion are computed and differentiated. As in
:doc:`graphs_and_locality`, gradients flow through the continuous quantities —
distances, forces, concentrations — but not through the discrete appearance or
disappearance of an edge. Freezing the topology per step is what lets the
pairwise physics be expressed as a fixed computation that an autodiff engine can
record and reverse.

The Warp tape
-------------

The bridge that makes Warp differentiable to the learning framework lives in
:mod:`waxmorph.torch.warp_autograd` and :mod:`waxmorph.jax.warp_autograd`. On
the forward pass it records every kernel launch on a :class:`warp.Tape`. On the
backward pass it replays that tape in reverse to propagate gradients: the torch
side wraps this in a :class:`torch.autograd.Function`, and the jax side exposes
it as a custom VJP. Either way, the physics corrections become a differentiable
node in the surrounding computation graph, and the shape-loss gradient reaches
the GNS parameters through the physics rather than around it.

This design also keeps the constraints extensible. Because the physics enters
through a recorded tape behind a standard autograd interface, you can add a new
prescribed constraint as a Warp kernel and expose it through the same bridge,
without rewriting how gradients reach the network.

A note on gradient evaluation
-----------------------------

The forward simulator can obtain its mechanical updates either from
analytically derived gradients or from Warp's automatic differentiation of the
same scalar potentials. The two routes agree up to the baseline variation
caused by nondeterministic GPU execution — floating-point atomic reductions and
random memory-access order — which is why waxMorph offers both and treats
autodiff as a drop-in alternative to the closed-form forces.
