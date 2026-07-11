Architecture
============

waxMorph couples a prescribed forward simulator, a learned emulator with a
fixed agent count, and trajectory renderers. The simulator and emulator share
spheroidal state conventions and provide purpose-built physics implementations.

Simulation and emulation
------------------------

* :mod:`waxmorph.simulator` implements cell-type-dependent mechanics,
  activator-inhibitor chemistry, growth, and division. Division grows an active
  prefix up to caller-owned capacity.
* :mod:`waxmorph.emulator` implements uniform sticky-sphere mechanics and
  graph-Laplacian diffusion for fixed-size learned rollouts. Neighbor discovery
  stays outside autodiff; each differentiated operation uses a frozen pair list.

Each path uses its own constants and update conventions. Simulator contact-local
polarity and chemistry use ``EPS_DIST=0.25`` alongside type-dependent forces.
Emulator contacts use ``EPS_DIST=0.01`` and one pair-force law.
:mod:`waxmorph._graph_core` separately builds detached learned-graph topology
shared by the Torch and JAX graph APIs.

Learning backends
-----------------

The top-level ``gnn``, ``graph``, ``train``, ``losses``, and ``mlp`` modules
re-export :mod:`waxmorph.torch`; ``from waxmorph import GNS, train, build_graph``
therefore selects PyTorch. :mod:`waxmorph.jax` is an explicit behavioral
counterpart with backend-specific interfaces and runtime contracts.

JAX uses explicit PRNG keys, Optax, optional fixed-capacity graph padding, and a
four-value graph return. Its Warp bridge uses custom VJPs on CUDA over
per-pair kernels. PyTorch uses mutable optimizers, unpadded graph returns, and a
Warp Tape behind :class:`torch.autograd.Function`. Each backend also provides
its own checkpoint format and loss families.

Rendering
---------

:mod:`waxmorph.render` provides Matplotlib static views, PyVista interactive
glyphs, and ``WarpMovieRenderer`` for USD stages or headless OpenGL videos.
``write_frame_from_numpy`` renders learned rollouts;
``write_frame_from_state`` renders live Warp state.
