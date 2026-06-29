Cells as spheroidal agents
==========================

waxMorph represents a tissue as a three-dimensional population of spheroidal
cellular agents — not as voxels, pixels, or mesh elements. Every cell carries
its own position, volume, polarity, molecular state, and an optional cell type,
so the model reasons about cells directly rather than about a discretized image
of the tissue.

Each cell :math:`i` carries a center position :math:`x_i \in \mathbb{R}^3`, a
radius :math:`r_i \in \mathbb{R}_+`, and a unit polarity vector
:math:`p_i \in \mathbb{R}^3` with :math:`\lVert p_i \rVert_2 = 1`. The cell
volume follows from the radius,

.. math::

   v_i = \tfrac{4}{3}\pi r_i^3,

and a molecule's concentration is its abundance divided by :math:`v_i`. The
tissue state is the collection of all cells, :math:`S(t) = \{s_i(t)\}`, over the
:math:`N(t)` cells present at time :math:`t`.

In code, a minimal state comprises four arrays.

``X``
   Center positions, shape ``[N, 3]``.

``P``
   Polarity vectors, shape ``[N, 3]``. Polarity enters the edge features and,
   in the forward simulator, the epithelial geometry constraints.

``R``
   Radii, shape ``[N]``. Radii induce the contact neighborhoods and set the
   characteristic length scale for packing and mechanics.

``c``
   Signaling-molecule concentrations, shape ``[N, num_molecules]``. These form
   the node features for the emulator and diffuse over the contact graph.

``CT``
   Optional integer cell-type labels. The forward simulator uses cell type to
   distinguish epithelial and mesenchymal interaction parameters.

Each mode instantiates the molecular state to suit its question. The forward
simulator equips every cell with a two-component activator-inhibitor pair, an
equilibrium radius, and a discrete cell type. The learned emulator equips every
cell with a vector of latent signaling molecules and a fixed radius. The shared
backbone — position, radius, polarity, and a molecular state over a contact
graph — is the same in both.
