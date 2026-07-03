Emulator differentiability
======================

The learned emulator works by making the prescribed biophysics differentiable.
Gradients of the shape loss flow back through the soft-sphere mechanics and the
graph diffusion.

Spatial adjacencies within an update step
-----------------------------

Each emulation step applies the learned GNS updates and then the prescribed
constraints. The constraints run ``n_substeps`` times per learned update and can
be ran on a faster time scale, keeping the trajectory biophysically coherent.

The differentiable path in :mod:`waxmorph.emulator` records these pairwise
kernels while *freezing the neighbor topology for the step*. This is feasible for small enough step sizes
:math:`\Delta t` and large enough trajectory length :math:`T` as deformations induced by mechanics will be smooth. The edge set :math:`E^{t+1}`
is rebuilt from the provisional positions, then held fixed while the constraint forces and diffusion are computed and differentiated. As in
:doc:`graphs_and_locality`, gradients flow through the cell states, but not through the discrete appearance or
disappearance of an edge. This choice stems from using the spatial adjacency graph as opposed to
a fully connected graph, reducing computational complexity drastically.


The Warp tape
-------------

The bridge that makes Warp differentiable to the learning framework lives in
:mod:`waxmorph.torch.warp_autograd` and :mod:`waxmorph.jax.warp_autograd`. On
the forward pass it records every kernel launch on a :class:`warp.Tape`. On the
backward pass it replays that tape in reverse to propagate gradients: the torch
side wraps this in a :class:`torch.autograd.Function`, and the jax side exposes
it as a custom VJP. Either way, the physics corrections become a differentiable
node in the surrounding computation graph, and the shape-loss gradient reaches
the GNS parameters through the physics.

This design is purposely preferred to keep the constraints extensible. As the physics enters
through a recorded tape behind a standard autograd interface, new
prescribed constraints can be added as a Warp kernel and exposed through the same bridge,
without rewriting how gradients reach the network.

Simulator differentiability
======================

The forward simulator can obtain its mechanical updates either from
analytically derived gradients or from Warp's automatic differentiation of the
same scalar potentials.

.. note::

   The two routes agree up to the baseline variation caused by nondeterministic
   GPU execution — floating-point atomic reductions and random memory-access
   order — which is why waxMorph offers both and treats autodiff as a drop-in
   alternative to the closed-form forces.
