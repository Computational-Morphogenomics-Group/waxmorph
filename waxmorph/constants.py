"""Numerical constants shared by graph construction and Warp kernels where imported.

``EPS_DIST=0.01`` is the emulator and learned-graph slack; the simulator uses its local
``EPS_DIST=0.25``.
"""

FOUR_THIRDS_PI: float = 4.1887902047863905

# Emulator and learned-graph contact slack; discrete topology is detached from gradients.
EPS_DIST: float = 1e-2

# Warp HashGrid cells per axis; this affects search performance, not physical parameters.
HASH_GRID_DIM: int = 128

# Additive offset for denominators that may reach zero.
EPS_DEN: float = 1e-9
# Regularizes near-zero norms and force-law denominators.
EPS_NORM: float = 1e-9
# Keeps uniform draws away from logarithm singularities at zero and one.
RAND_EPS: float = 1e-7
# Matches the torch.nn.functional.normalize default used for polarity updates.
EPS_POLARITY: float = 1e-12
