Training APIs
=============

The top-level :mod:`waxmorph.train` module is a backward-compatible alias for
the default PyTorch training API. Backend-specific training entry points remain
available from :mod:`waxmorph.torch.train` and :mod:`waxmorph.jax.train`.

.. toctree::
   :maxdepth: 1

   waxmorph.train
   waxmorph.torch.train
   waxmorph.jax.train
