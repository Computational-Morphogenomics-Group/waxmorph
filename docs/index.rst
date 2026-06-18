waxmorph
========

waxMorph is a joint framework for forward simulation and inverse design of
biophysical shape assembly. Tissues are represented as populations of
interacting three-dimensional spheroidal agents, spatial adjacency graphs
encode which agents may exert mechanical or biochemical influence on one
another, and graph-network-based emulators are trained to reproduce target
morphologies or target morphology sequences.

The framework supports two connected modes of use:

* **Forward simulation**: mechanochemical trajectories are generated from
  explicit rules for adhesion, volume exclusion, polarity, reaction-diffusion,
  and division.
* **Inverse design / learned emulation**: a graph-network-based simulator
  (GNS) learns local, neighbor-dependent update rules that transform an initial
  spheroidal cell population into one or more target morphologies, while
  differentiable physical constraints guide tissue-scale assembly.

The central modelling assumption is locality. Each cellular agent carries a
position, volume, polarity vector, signaling molecule concentrations, and an
optional cell type; interactions are evaluated over contact neighborhoods
rather than over a global image grid. Local neighborhoods are induced by
spatial proximity and rebuilt as tissues deform. The same representation
therefore applies both to morphologies obtained from segmented shapes or meshes
and to mechanistic models in which local rules of tissue organization are
specified or learned.

Where to start
--------------

A first introduction is provided in :doc:`getting_started`, followed by
:doc:`concepts` for the vocabulary used throughout the framework. Adapting the
framework to a new biological system typically requires :doc:`workflows`
together with the :doc:`api` reference.

.. toctree::
   :maxdepth: 2

   getting_started
   concepts
   workflows
   api
   contributing
   changelog
   references
