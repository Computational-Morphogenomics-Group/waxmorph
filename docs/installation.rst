Installation
============

Install the base package with pip:

.. code-block:: bash

   pip install waxmorph

The extras gate the heavier dependencies, so pick the combination that matches
your workflow.

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
   NVIDIA Warp, PyVista, and trimesh — the forward simulator and the principal
   movie-rendering paths.

``learning``
   PyTorch, GeomLoss, and pykeops — the default graph-network backend and its
   optimal-transport losses.

``jax``
   JAX, Equinox, Optax, and ott-jax — the parity backend.

``all``
   Everything above, for a full development environment.

Hardware notes
--------------

The simulation kernels and the OpenGL and USD movie renderers run on NVIDIA
Warp and target CUDA. The graph and data utilities run on CPU, but the worked
examples are GPU-oriented.

.. note::

   USD export uses Warp's USD renderer; where that backend is unavailable,
   install a USD Python package such as ``usd-core`` or ``usd-exchange``.
