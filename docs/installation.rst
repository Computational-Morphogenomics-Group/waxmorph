Installation
============

Install the base package with pip:

.. code-block:: bash

   pip install waxmorph

The base install includes NumPy 1.24+, SciPy 1.11+, PyTorch 2+, tqdm 4.66+, and
Warp 1.10+. It provides the forward simulator and the default PyTorch graph,
GNS, losses, and training APIs. Extras add optional loss, backend, mesh, and
rendering dependencies.

Add mesh and rendering support:

.. code-block:: bash

   pip install "waxmorph[simulation]"

Add GeomLoss and PyKeOps; this extra currently includes ``simulation``:

.. code-block:: bash

   pip install "waxmorph[learning]"

Add the explicit JAX/Equinox backend:

.. code-block:: bash

   pip install "waxmorph[jax]"

Use JAX with CUDA 12:

.. code-block:: bash

   pip install "waxmorph[jax-cuda]"

Set up a development checkout with development, documentation, JAX, learning,
simulation, and style dependencies:

.. code-block:: bash

   git clone https://github.com/Computational-Morphogenomics-Group/waxmorph.git
   cd waxmorph
   pip install -e ".[all]"
   pre-commit install

``all`` contains the CPU JAX stack. GPU development adds ``jax-cuda``:

.. code-block:: bash

   pip install -e ".[all,jax-cuda]"

What each extra pulls in
------------------------

``simulation``
   PyVista, trimesh, VTK, OpenGL support, and ``usd-core`` — mesh processing and
   movie-rendering paths.

``learning``
   GeomLoss and PyKeOps for optional PyTorch point-cloud losses, plus
   ``simulation``.

``jax``
   JAX, Equinox, Optax, and ott-jax, plus ``simulation``.

``jax-cuda``
   CUDA 12 JAX support plus ``jax``.

``all``
   Development, documentation, JAX, learning, and simulation dependencies.
   ``jax-cuda`` supplies CUDA JAX support.

Hardware notes
--------------

The forward simulator and PyTorch/Warp training bridge accept supported CPU or
CUDA devices; examples default to CUDA. JAX training with mechanics or diffusion
uses custom VJPs on CUDA. JAX CPU training uses ``mech_steps=0`` and
``diff_steps=0``. ``WarpMovieRenderer`` defaults to CUDA; its OpenGL
backend also requires a working graphics and video stack.

.. note::

   USD export uses Warp's USD renderer and the ``usd-core`` package installed by
   the ``simulation`` extra.
