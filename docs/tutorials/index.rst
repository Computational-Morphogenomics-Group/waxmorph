Tutorials
=========

Each tutorial runs a complete workflow from initial state to rendered
trajectory. Together they cover both framework modes across both learning
backends.

Start with the forward simulator to see the mechanochemical primitives in
action, then train a learned emulator to assemble a target morphology. The two
emulator tutorials run the same workflow on the PyTorch and JAX backends;
compare them to understand the backend differences described in
:doc:`../explanation/architecture`.

.. toctree::
   :maxdepth: 1

   Forward simulator <forward_simulator>
   Learned emulator (PyTorch) <learned_emulator>
   Learned emulator (JAX) <learned_emulator_jax>
