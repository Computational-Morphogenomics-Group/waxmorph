Core Concepts
=============

Spheroidal agents
-----------------

waxMorph represents tissues as three-dimensional populations of spheroidal
cellular agents rather than as voxels, pixels, or mesh elements. Each agent
carries position, volume, polarity, molecular state, and cell type. In the
current implementation a cell state is represented by:

``x``
   Center position with shape ``[N, 3]``.

``p``
   Polarity vector with shape ``[N, 3]``. Polarity enters the edge features and,
   in the Warp simulator, the epithelial geometry constraints.

``r``
   Radius with shape ``[N]``. Radii induce the contact neighborhoods and set the
   characteristic length scale for packing and mechanics.

``c``
   Signaling-molecule concentrations with shape ``[N, num_molecules]``. These
   values form the node features for the emulator and diffuse over the contact
   graph.

``ct``
   Optional integer cell-type labels. The forward simulator uses cell type to
   distinguish epithelial and mesenchymal interaction parameters.

Forward simulation
------------------

The forward simulator resides in :mod:`waxmorph.simulator`. It is implemented as
NVIDIA Warp kernels and encodes sticky-sphere mechanics, epithelial polarity and
thickness constraints, mesenchymal polarity alignment, reaction-diffusion
dynamics for the signaling-molecule concentrations, and stochastic division.

Forward simulation answers the mechanistic question: given an explicit
biophysical rule set and a set of initial conditions, which tissue trajectory is
produced? It also generates training trajectories for the learned emulator.

Emulation and inverse design
----------------------------

The learned emulator addresses the inverse-design question: which local update
rule moves an initial spheroidal tissue toward a set of target morphologies? The
default emulator is a graph-network-based simulator (GNS) in :mod:`waxmorph.gnn`,
backed by PyTorch. A JAX/Equinox implementation is available under
:mod:`waxmorph.jax`.

At each rollout step the training loop proceeds as follows:

1. A contact graph is constructed from the current state.
2. Local updates are predicted with the GNS.
3. The learned updates are applied to positions, polarities, and
   signaling-molecule concentrations.
4. Differentiable Warp mechanics and graph-Laplacian diffusion corrections are
   applied.
5. A shape loss is accumulated at one or more supervised target frames.

This separation is deliberate. The graph-network processor learns local,
neighbor-dependent update fields, while the differentiable physical constraints
keep the rollout tied to contact mechanics and graph diffusion, guiding
tissue-scale assembly under biophysical constraints.

Graphs and locality
-------------------

Graph construction is implemented in :mod:`waxmorph.graph`,
:mod:`waxmorph.torch.graph`, and :mod:`waxmorph.jax.graph`. Local neighborhoods
are induced by spatial proximity and rebuilt as the tissue deforms. For PyTorch
inputs, feature construction remains differentiable, but the edge topology is
built from a detached snapshot of positions and radii. Gradients therefore flow
through distances, angles, signaling-molecule concentrations, and model
parameters, but not through the discrete event of an edge appearing or
disappearing within a single step.

This is a standard approximation in differentiable particle systems: the
continuous state is optimized while the contact graph is treated as fixed over
the current local update.

Shape losses
------------

:mod:`waxmorph.losses` provides losses for comparing predicted and target point
clouds.

``squared_loss`` applies when each predicted cell has a known target identity and
row order is meaningful. ``chamfer_distance`` and ``make_samples_loss`` apply
when the target is an unordered point cloud sampled from a tissue shape. For
biological shapes sampled from meshes, the unordered losses are the more
realistic choice unless the experiment tracks cell identities.
