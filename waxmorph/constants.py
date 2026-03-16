"""Shared constants used across simulation, emulation, and graph construction.

Force-specific constants (K_REP, K_ATT_*, etc.) remain local to their
respective modules since they are intentionally different between the
full simulator and the simplified emulator.
"""

# Geometry
FOUR_THIRDS_PI: float = 4.1887902047863905

# Adjacency / contact detection
EPS_DIST: float = 1e-2

# HashGrid spatial acceleration
HASH_GRID_DIM: int = 128

# Numerical stability
EPS_DEN: float = 1e-9
EPS_NORM: float = 1e-9
RAND_EPS: float = 1e-7
