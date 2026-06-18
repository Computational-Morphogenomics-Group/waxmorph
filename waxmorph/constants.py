"""Shared constants used across simulation, emulation, and graph construction.

Force-specific constants (K_REP, K_ATT_*, etc.) remain local to their
respective modules since they are intentionally different between the
full simulator and the simplified emulator.

Paper-symbol notes (for cross-referencing the writeup notation):

- ``FOUR_THIRDS_PI`` is the sphere-volume prefactor 4π/3.
- ``EPS_DIST`` is the contact-slack tolerance ε on adjacency/contact tests.
- ``EPS_DEN``, ``EPS_NORM``, ``RAND_EPS`` are numerical-stability epsilons ε.
"""

# Geometry
FOUR_THIRDS_PI: float = 4.1887902047863905  # paper 4π/3, sphere-volume prefactor

# Adjacency / contact detection
EPS_DIST: float = 1e-2  # paper ε, contact slack on adjacency tests

# HashGrid spatial acceleration
HASH_GRID_DIM: int = 128

# Numerical stability
EPS_DEN: float = 1e-9  # paper ε, denominator-stability epsilon
EPS_NORM: float = 1e-9  # paper ε, norm-stability epsilon
RAND_EPS: float = 1e-7  # paper ε, RNG-clamp epsilon
