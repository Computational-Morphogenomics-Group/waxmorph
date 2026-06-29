waxMorph
========

**waxMorph builds tissue shape from the bottom up.** It treats a tissue as a
population of interacting spheroidal cells and lets you do two things with the
same representation: run a forward mechanochemical simulation, or learn the
local update rule that assembles an initial cell population into a target
morphology.

Why use it: shape assembly is local. Each cell knows only its contact
neighbors, so waxMorph evaluates mechanics, signaling, and diffusion over a
spatial adjacency graph that it rebuilds as the tissue deforms — the same
representation whether you sample cells from a mesh, prescribe a mechanistic
model, or train a graph network to reproduce an observed morphology.

New here? Install the package (:doc:`installation`), then work through the
:doc:`tutorials/index` to run a forward simulation and train a learned
emulator end to end. Reach for the :doc:`how_to/index` guides when you have a
specific task, the :doc:`explanation/index` pages to understand the modelling
choices, and the :doc:`reference/index` for the API.

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   tutorials/index
   how_to/index
   explanation/index
   reference/index
   contributing
   changelog
   references
