Installation
============

.. code-block:: bash

   pip install waxmorph

The base install (NumPy 1.24+, SciPy 1.11+, PyTorch 2+, tqdm 4.66+, Warp 1.18+) provides the
forward simulator and the PyTorch graph, GNS, loss, and training APIs.

Warp's PyPI wheels need an NVIDIA R580+ driver for CUDA. With an older driver (R525+), install
Warp's ``+cu12`` wheel from its `GitHub Releases <https://github.com/NVIDIA/warp/releases>`_.

Extras
------

Install an extra with ``pip install "waxmorph[<extra>]"``.

``simulation``
   Mesh sampling and rendering: trimesh, Matplotlib, PyVista, VTK, pyglet, imageio, hsluv, and
   ``usd-core``.
``learning``
   GeomLoss and PyKeOps for ``make_samples_loss``, pytorch-optimizer for SOAP, plus
   ``simulation``.
``jax``
   JAX, Equinox, Optax, ott-jax, and chex and jaxtyping for the bundled SOAP_JAX optimizer
   (``waxmorph.jax.soap_jax``), plus ``simulation``.
``jax-cuda``
   CUDA 12 JAX, plus ``jax``.
``all``
   ``dev``, ``docs``, ``jax``, ``learning``, and ``simulation``, with CPU JAX.

Development
-----------

.. code-block:: bash

   git clone https://github.com/Computational-Morphogenomics-Group/waxmorph.git
   cd waxmorph
   pip install -e ".[all]"
   pre-commit install

GPU JAX development installs ``".[all,jax-cuda]"`` instead.

Hardware
--------

The simulator and the PyTorch training bridge run on CPU or CUDA; examples default to CUDA.
JAX training with mechanics or diffusion requires CUDA, so JAX CPU training sets
``mech_steps=0`` and ``diff_steps=0``. ``WarpMovieRenderer`` defaults to CUDA; its OpenGL
backend also needs a working graphics and video stack.
