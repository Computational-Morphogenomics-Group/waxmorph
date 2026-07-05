"""Scalar constants shared by the Warp physics core across both backends.

These values are baked into ``@wp.kernel`` / ``@wp.func`` source at compile time, so
defining them once here keeps the simulator, emulator, and graph construction bit-for-bit
consistent across the torch and jax backends: the same contact slack that decides which
agents are neighbours during a simulation step is the same one that builds the GNS contact
graph, and the same HashGrid resolution accelerates every pairwise kernel.

Force-law constants (``K_REP``, ``K_ATT_*``, adhesion cutoffs, growth/division rates)
are intentionally *not* here: they are physics, not numerics, and differ between the full
:mod:`~waxmorph.simulator` and the simplified :mod:`~waxmorph.emulator`, so each module
owns its own copies. :mod:`~waxmorph.simulator` likewise shadows ``EPS_DIST`` with a
larger module-local value tuned for its growing, sticky-sphere mechanics; the value below
is the one used by the emulator and by contact-graph construction.

The constants fall into three groups:

* **Geometry** -- ``FOUR_THIRDS_PI``, the sphere-volume prefactor :math:`4\\pi/3`.
* **Adjacency / contact slack** -- ``EPS_DIST``, the tolerance :math:`\\varepsilon`
  added to the radii sum when deciding contact, edges, and HashGrid query radii.
* **Numerical-stability epsilons** -- ``EPS_DEN``, ``EPS_NORM``, ``RAND_EPS``: small
  :math:`\\varepsilon` floors that keep divisions, normalisations, and RNG draws finite
  and differentiable at degenerate (zero-distance / boundary) configurations.
"""

# --- Geometry ---
# Sphere-volume prefactor 4*pi/3, multiplied by r**3 to turn an agent radius into a
# volume proxy (e.g. concentration normalisation, growth bookkeeping). Precomputed as a
# literal so the Warp kernels never recompute pi at launch time.
FOUR_THIRDS_PI: float = 4.1887902047863905  # 4*pi/3, sphere-volume prefactor

# --- Adjacency / contact slack ---
# Tolerance epsilon added to the sum of two radii when testing contact: agents i and j are
# adjacent when dist(i, j) <= r_i + r_j + EPS_DIST. The same slack widens the HashGrid
# query radius (2*r_max + EPS_DIST) so no contacting pair is missed, and is the default
# eps_dist for building the GNS contact graph -- keeping simulated and learned topology
# identical. The boundary is a hard threshold (non-smooth); gradients are not taken w.r.t.
# this discrete neighbour selection. Simulator.py overrides this with a larger local value.
EPS_DIST: float = 1e-2  # contact slack on adjacency tests

# --- HashGrid spatial acceleration ---
# Number of cells per axis (128**3 total) for the Warp HashGrid used to find contacting
# pairs in O(n) instead of O(n**2). Shared so the simulator, emulator, and training-time
# neighbour queries all bin space identically; affects performance only, never the physics.
HASH_GRID_DIM: int = 128

# --- Numerical stability ---
# Denominator floor for safe division num / (den + EPS_DEN): prevents inf/NaN when a
# computed denominator collapses to zero, while leaving the gradient well defined.
EPS_DEN: float = 1e-9  # denominator-stability epsilon
# Distance floor added to wp.length(...) before dividing by it to normalise a direction
# vector: at zero separation the raw norm and its gradient are undefined, so this fixes a
# finite (arbitrary but stable) direction and keeps the backward pass through the Tape sane.
EPS_NORM: float = 1e-9  # norm-stability epsilon
# RNG clamp keeping uniform draws inside (RAND_EPS, 1 - RAND_EPS) before feeding them to
# log() for Gumbel / division sampling, so the exact endpoints 0 and 1 never produce -inf.
RAND_EPS: float = 1e-7  # RNG-clamp epsilon
# Norm floor for polarity renormalisation P / max(||P||, EPS_POLARITY) after a GNS update.
# Shared so the torch and jax backends renormalise polarity identically: it is the default
# eps of torch.nn.functional.normalize, which the jax twin must pass explicitly (it previously
# used 1e-9, a silent divergence). Active only for degenerate near-zero polarity vectors;
# polarities start unit-norm and the per-step delta is small, so it rarely binds.
EPS_POLARITY: float = 1e-12  # polarity-renormalisation norm floor (torch F.normalize default)
