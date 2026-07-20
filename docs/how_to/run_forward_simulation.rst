Running forward simulations: the epithelial-mesenchymal case study
==================================================================

Use the forward simulator for the prescribed epithelial-mesenchymal case study;
the learned emulator supports source-target shape assembly. Values are in model
units; applications may supply external calibration for SI or biological
interpretation.
The active particle prefix can grow to a preallocated ``max_particles``
capacity. The :doc:`forward simulator tutorial <../tutorials/forward_simulator>`
walks through a complete run.

1. Allocate fixed-capacity Warp arrays for centers ``X``, radii ``R``,
   equilibrium radii ``R_eq``, polarities ``P``, activator ``A``, inhibitor
   ``I``, and cell types ``CT``.
2. Relax geometry with ``simulator.mech_step_sticky``. The
   ``mech_step_sticky_implicit`` variant uses local ``warp.grad`` for polarity,
   thickness, alignment, and optional WNT terms; soft-sphere derivatives remain
   explicit.
3. Pattern the activator and inhibitor fields by reaction-diffusion with
   ``simulator.chem_step``, parameterized by ``chi`` (activator diffusivity
   relative to the inhibitor), ``gamma`` (reaction rate), and ``D_inhib``
   (inhibitor diffusivity).
4. Grow cells with ``simulator.growth_step``. Radii below their targets increase
   up to those targets; mesenchymal equilibrium radii are capped at ``R_max`` and
   epithelial equilibrium radii copy through.
5. Count neighbors and divide cells with ``simulator.count_neighbors_step``,
   ``simulator.division_decision``, and ``simulator.division_logic``.
6. Write frames from the live Warp state with
   ``render.WarpMovieRenderer.write_frame_from_state`` (see
   :doc:`render_trajectories`).

Mechanics, chemistry, and growth write caller-supplied next-state arrays; growth
also advances mesenchymal RNG keys. Division mutates shared state arrays and
reserves unique daughter slots within capacity. Mesenchymal daughter
radii conserve half volume; epithelial daughter radii copy.

.. tip::

   Lower the particle count and the step count when you are validating a new
   environment.

:doc:`../explanation/forward_and_inverse` and
:doc:`../explanation/differentiable_physics` give the equations behind these
kernels: the soft-sphere force, the graph Laplacian, the activator-inhibitor
reaction-diffusion system, the growth Hill function, and the division rules.
