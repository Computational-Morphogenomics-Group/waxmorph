:sd_hide_title: true

waxMorph
========

.. raw:: html

   <div class="wm-hero">
     <img class="wm-hero-logo wm-logo-light dark-light" src="_static/logo-1024w.png"
          alt="waxMorph — interacting cells resolving into a learned graph network">
     <img class="wm-hero-logo wm-logo-dark dark-light" src="_static/logo-dark.png"
          alt="waxMorph — interacting cells resolving into a learned graph network">
     <p class="wm-tagline">A differentiable cell-based framework for three-dimensional morphogenesis.</p>
     <p class="wm-sub">
       Cells build tissues through local exchanges of force and information, yet
       the rules governing these interactions are difficult to infer from sparse
       observations. waxMorph represents tissue as interacting spheroidal cells
       and makes the governing mechanochemical dynamics differentiable, so the
       same model can be run forward to simulate prescribed shape programs and
       inverse to reconstruct continuous morphogenetic trajectories from static
       tissue volumes.
     </p>
     <div class="wm-cta">
       <a class="wm-btn wm-btn-primary" href="installation.html">Install</a>
       <a class="wm-btn wm-btn-ghost" href="tutorials/index.html">Tutorials</a>
       <a class="wm-btn wm-btn-ghost"
          href="https://github.com/waxmorph/waxmorph">GitHub</a>
     </div>
   </div>

   <figure class="wm-showcase">
     <img src="_static/images/fig1.png"
          alt="waxMorph pipeline: forward simulation and inverse learning of shape assembly">
     <figcaption>
       waxMorph represents tissue as interacting polarized spheroids, couples
       forward and inverse models of morphogenesis over that representation, and
       recovers spatially organized latent signals.
     </figcaption>
   </figure>

Explore
-------

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: wm-cards

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc
      :class-card: wm-card

      Install the package and pick the ``simulation``, ``learning``, or ``jax``
      extras for your workflow.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc
      :class-card: wm-card

      End-to-end notebooks: run a forward simulation and train a learned
      emulator from a mesh pair.

   .. grid-item-card:: Gallery
      :link: gallery
      :link-type: doc
      :class-card: wm-card

      Play reconstructed morphogenesis trajectories in an interactive 3D
      viewer, with a timeline and a free-flying camera.

   .. grid-item-card:: How-to guides
      :link: how_to/index
      :link-type: doc
      :class-card: wm-card

      Task-focused recipes for people who already know waxMorph and want to get
      something specific done.

   .. grid-item-card:: Explanation
      :link: explanation/index
      :link-type: doc
      :class-card: wm-card

      The modelling choices: cell state, locality, differentiable
      physics, forward and inverse modes.

   .. grid-item-card:: API reference
      :link: reference/index
      :link-type: doc
      :class-card: wm-card

      The public modules, with PyTorch defaults and JAX/Equinox parity
      variants.

   .. grid-item-card:: Contributing
      :link: contributing
      :link-type: doc
      :class-card: wm-card

      Development setup, testing across both backends, and the conventions
      changes should follow.

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   tutorials/index
   gallery
   how_to/index
   explanation/index
   reference/index
   contributing
   changelog
   references
