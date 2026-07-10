Installation
============

Install the base package with pip:

.. code-block:: bash

   pip install waxmorph

The base install includes the default PyTorch API, Warp runtime, SciPy graph
construction, and training progress support. Extras add the following workflows.

Run the forward simulator and the PyTorch learning stack:

.. code-block:: bash

   pip install "waxmorph[simulation,learning]"

Add the JAX/Equinox parity backend on CPU:

.. code-block:: bash

   pip install "waxmorph[jax]"

Use JAX with CUDA 12:

.. code-block:: bash

   pip install "waxmorph[jax-cuda]"

Set up a development checkout with every extra and the style hooks:

.. code-block:: bash

   git clone https://github.com/Computational-Morphogenomics-Group/waxmorph.git
   cd waxmorph
   pip install -e ".[all]"
   pre-commit install

What each extra pulls in
------------------------

``simulation``
   PyVista, trimesh, VTK, OpenGL support, and ``usd-core`` — mesh processing and
   movie-rendering paths.

``learning``
   GeomLoss and PyKeOps — optional optimal-transport losses for the default
   PyTorch backend.

``jax``
   JAX, Equinox, Optax, and ott-jax — the parity backend.

``all``
   Everything above.

Hardware notes
--------------

The simulation kernels and the OpenGL and USD movie renderers run on NVIDIA
Warp and target CUDA. The graph and data utilities run on CPU, but the worked
examples are GPU-oriented.

.. note::

   USD export uses Warp's USD renderer and the ``usd-core`` package installed by
   the ``simulation`` extra.
