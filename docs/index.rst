waxmorph
========

WaxMorph is a joint framework for forward simulation and inverse learning of
biophysical shape assembly. It represents tissues as interacting spheroidal
agents, uses local contact graphs to describe who can mechanically or
biochemically influence whom, and provides graph-network emulators that can be
trained to reproduce target shapes or target shape sequences.

The package is intended for two connected modes of use:

* **Forward simulation**: generate mechanochemical trajectories from explicit
  rules for adhesion, repulsion, polarity, reaction-diffusion, and division.
* **Inverse learning / emulation**: train graph neural networks to learn local
  update rules that transform an initial spheroidal cell population into one or
  more target morphologies while retaining differentiable physics corrections.

The central modelling assumption is local: each cell-like agent carries a
position, radius, polarity vector, gene or morphogen state, and optional cell
type; interactions are evaluated on contact neighborhoods rather than on a
global image grid. This makes the same representation natural for biologists
who start from segmented shapes or meshes, and for biophysicists who want to
write down or learn local rules of tissue organization.

Where to start
--------------

New users should begin with :doc:`getting_started`, then read
:doc:`concepts` for the vocabulary used across the package. Users adapting the
package to a new biological system will usually need :doc:`workflows` and then
the :doc:`api` reference.

.. toctree::
   :maxdepth: 2

   getting_started
   concepts
   workflows
   api
   contributing
   changelog
   references
