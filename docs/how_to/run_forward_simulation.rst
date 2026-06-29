Run a forward simulation
========================

Reach for the forward simulator when your question concerns a specified
mechanistic model rather than a learned shape-assembly rule. It integrates
prescribed mechanochemical dynamics and can grow the active particle count up
to a preallocated ``max_particles`` capacity. The ``simulation_with_autodiff``
tutorial walks through a complete run; the steps below are the recipe.

1. Allocate fixed-capacity Warp arrays for centers ``X``, radii ``R``,
   equilibrium radii ``R_eq``, polarities ``P``, activator ``A``, inhibitor
   ``I``, and cell types ``CT``.
2. Relax geometry under the soft-sphere potential with
   ``simulator.mech_step_sticky``, or with the autodiff-consistent
   ``simulator.mech_step_sticky_implicit``.
3. Pattern the activator and inhibitor fields by reaction-diffusion with
   ``simulator.chem_step``, parameterized by ``chi`` (the activator diffusivity
   relative to the inhibitor, the spatial characteristic), ``gamma`` (the
   reaction rate), and ``D_inhib`` (the inhibitor diffusivity).
4. Grow cells with ``simulator.growth_step``, parameterized by the growth Hill
   exponent ``alpha_grow`` and the switch concentration ``ell_sw``.
5. Count neighbors and divide cells with ``simulator.count_neighbors_step``,
   ``simulator.division_decision``, and ``simulator.division_logic``.
6. Write frames from the live Warp state with
   ``render.WarpMovieRenderer.write_frame_from_state`` (see
   :doc:`render_trajectories`).

Inputs are Warp arrays and scalar parameters: time steps, diffusivities, growth
constants, and division thresholds. Outputs are the state arrays, updated
in place, and any rendered frames you request.

The tutorial includes long runs with tens of thousands of particles over many
steps. Drop the particle count and the number of steps when you are validating
a new environment.

For the equations behind these kernels — the soft-sphere force, the graph
Laplacian, the activator-inhibitor reaction-diffusion system, the growth Hill
function, and the division rules — see :doc:`../explanation/forward_and_inverse`
and :doc:`../explanation/differentiable_physics`.
