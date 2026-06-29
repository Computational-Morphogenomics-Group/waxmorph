Forward and inverse modes
=========================

waxMorph runs the same cellular representation in two directions. In the
**forward** mode it prescribes every rule and integrates the resulting
dynamics; in the **inverse** mode it prescribes the known physics and learns
the rest of the local update rule. Both modes share the cell state of
:doc:`cell_state` and the physical primitives below.

Shared physical primitives
--------------------------

Both modes are built from three primitives that act over the contact graph
(:doc:`graphs_and_locality`): overdamped mechanics, a soft-sphere pairwise
force, and graph-based molecular diffusion.

Cell motion is overdamped, so velocity is proportional to net force,
:math:`\nu_i \dot{x}_i = f^{\text{net}}_i`, with :math:`\nu_i` an effective
friction coefficient absorbed into the mechanical parameters and the time
scale. Continuous quantities integrate with forward Euler, and polarity vectors
are renormalized to unit length after each update.

A soft-sphere force imposes volume exclusion and short-range adhesion. For a
neighbor pair :math:`(i, j)` with center distance :math:`d_{ij}` and unit
direction :math:`\mathbf{d}_{ij}` pointing from :math:`j` to :math:`i`,

.. math::

   f^{\text{soft}}_{ij} = \Big[ k_{\text{rep}}\,\max(r_i + r_j - \varepsilon - d_{ij},\,0)
   - k_{\text{att}}\,\max(r_i + r_j + \varepsilon - d_{ij},\,0)\,
   \mathbb{1}[d_{ij} > r_i + r_j - \varepsilon] \Big]\,\mathbf{d}_{ij},

where the repulsive branch acts for :math:`d_{ij} < r_i + r_j - \varepsilon`
and a short-range attractive branch acts over the band
:math:`[r_i + r_j - \varepsilon,\, r_i + r_j + \varepsilon]`.

Molecular transport is diffusion over the contact graph. For a concentration
field :math:`c`, the discrete graph Laplacian at cell :math:`i` sums the
differences to its neighbors,

.. math::

   (L_G c)_i = \sum_{j:\,(i,j)\in E(t)} (c_i - c_j).

Each mode layers its own rules on top of these primitives.

The forward mode
----------------

The forward simulator (:mod:`waxmorph.simulator`) is appropriate when the
scientific question concerns a specified mechanistic model rather than a learned
shape-assembly rule. It answers the mechanistic question directly: given an
explicit biophysical rule set and a set of initial conditions, which tissue
trajectory results? It also generates cheap training trajectories for the
learned emulator.

The implemented case study is a polarized epithelial-mesenchymal aggregate
coupled to a two-component activator-inhibitor reaction-diffusion system. On top
of the shared primitives it adds cell-type-specific potentials, growth, and
division: epithelial polarity and thickness potentials maintain the monolayer,
mesenchymal polarity potentials align cells and orient them up the activator
gradient, and activator-dependent growth drives division. The activator
:math:`A` and inhibitor :math:`I` abundances evolve by reaction and diffusion,

.. math::

   \dot{A}_i = \gamma\Big[ -\chi D_{\text{inhib}} (L_G c_A)_i
   + \tfrac{c_{i,A}^2}{c_{i,I}} - c_{i,A} \Big], \qquad
   \dot{I}_i = \gamma\big[ -D_{\text{inhib}} (L_G c_I)_i + c_{i,A}^2 - c_{i,I} \big],

with :math:`\gamma` the reaction time scale, :math:`D_{\text{inhib}}` the
inhibitor diffusivity, and :math:`0 \le \chi \le 1` the relative activator
diffusivity. Mesenchymal cells grow toward an activator-driven equilibrium
radius through a Hill function and divide with a radius-dependent probability;
the active particle count grows up to the preallocated capacity.

The inverse mode
----------------

The learned emulator is appropriate when you have source and target
morphologies and want to infer a local rollout rule. It answers the
inverse-design question: which local update rule moves an initial spheroidal
tissue toward a set of target morphologies? It prescribes the known
physics — soft-sphere mechanics and graph-based diffusion — and learns the
neighbor-dependent updates to position, polarity, and latent molecular state so
that an initial population assembles the prescribed target volumes.

The default emulator is a graph-network-based simulator (GNS) in
:mod:`waxmorph.gnn`, backed by PyTorch, with a JAX/Equinox parity
implementation under :mod:`waxmorph.jax`. Each rollout step proceeds as:

1. Construct a contact graph from the current state.
2. Predict local updates with the GNS.
3. Apply the learned updates to positions, polarities, and
   signaling-molecule concentrations.
4. Apply the differentiable Warp mechanics and graph-Laplacian diffusion
   corrections.
5. Accumulate a shape loss at one or more supervised target frames.

This separation is deliberate. The graph-network processor learns local,
neighbor-dependent update fields, while the prescribed physical constraints keep
the rollout tied to contact mechanics and graph diffusion — guiding
tissue-scale assembly under biophysical constraints rather than letting the
network drift away from physically plausible configurations.
