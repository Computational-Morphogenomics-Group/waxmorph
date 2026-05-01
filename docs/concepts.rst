Core Concepts
=============

Spheroidal agents
-----------------

WaxMorph treats each biological unit as a spheroidal agent rather than as a
voxel, pixel, or mesh element. In the current implementation a cell state is
usually represented by:

``X``
   Center position with shape ``[N, 3]``.

``P``
   Polarity vector with shape ``[N, 3]``. Polarity affects edge features and,
   in the Warp simulator, epithelial geometry constraints.

``R``
   Radius with shape ``[N]``. Radii define contact neighborhoods and the
   characteristic length scale for packing and mechanics.

``G``
   Gene or morphogen state with shape ``[N, num_genes]``. These values become
   node features for the emulator and can diffuse over the contact graph.

``CT``
   Optional integer cell type labels. The forward simulator uses cell type to
   distinguish epithelial and mesenchymal interaction parameters.

Forward simulation
------------------

The low-level forward simulator lives in :mod:`waxmorph.simulator`. It is built
on NVIDIA Warp kernels and currently encodes sticky-sphere mechanics,
epithelial polarity and thickness constraints, mesenchymal polarity alignment,
reaction-diffusion-like gene dynamics, and stochastic division utilities.

Forward simulation is most useful when you want to ask: "Given this explicit
biophysical rule set and these initial conditions, what tissue trajectory is
produced?" It can also generate training trajectories for learned emulators.

Emulation and inverse learning
------------------------------

The learned emulator asks the inverse question: "Which local update rule could
move this initial spheroidal tissue toward these target morphologies?" The
default emulator is a Graph Network-based Simulator in :mod:`waxmorph.gnn`
backed by PyTorch. A JAX/Equinox implementation is available under
:mod:`waxmorph.jax`.

At each rollout step the training loop:

1. Builds a contact graph from the current state.
2. Predicts local updates with the GNS model.
3. Applies learned updates to positions, polarities, and gene state.
4. Optionally applies differentiable Warp mechanics and diffusion corrections.
5. Accumulates shape loss at one or more supervised target frames.

This separation is intentional. The GNS learns local update fields, while Warp
corrections keep the rollout tied to contact mechanics and graph diffusion.

Graphs and locality
-------------------

Graph construction is implemented in :mod:`waxmorph.graph`,
:mod:`waxmorph.torch.graph`, and :mod:`waxmorph.jax.graph`. For PyTorch inputs,
feature construction remains differentiable, but the edge topology is built
from a detached snapshot of positions and radii. This means gradients flow
through distances, angles, genes, and model parameters, but not through the
discrete event of an edge appearing or disappearing within a single step.

This is a standard compromise in differentiable particle systems: the
continuous state is optimized while the contact graph is treated as fixed over
the current local update.

Shape losses
------------

:mod:`waxmorph.losses` provides losses for comparing predicted and target point
clouds.

Use ``squared_loss`` when each predicted cell has a known target identity and
row order matters. Use ``chamfer_distance`` or ``make_samples_loss`` when the
target is an unordered point cloud sampled from a tissue shape. For biological
shapes sampled from meshes, unordered losses are often the more realistic
starting point unless the experiment tracks cell identities.
