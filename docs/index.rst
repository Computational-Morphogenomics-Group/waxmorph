:sd_hide_title: true

waxMorph
========

.. raw:: html

   <div class="wm-hero">
     <img class="wm-hero-logo wm-logo-light dark-light" src="_static/logo-1024w.png"
          alt="waxMorph: interacting cells resolving into a learned graph network">
     <img class="wm-hero-logo wm-logo-dark dark-light" src="_static/logo-dark.png"
          alt="waxMorph: interacting cells resolving into a learned graph network">
     <p class="wm-tagline">A differentiable cell-based framework for three-dimensional morphogenesis.</p>
     <p class="wm-sub">
       Cells build tissues through local exchanges of force and information.
       waxMorph represents tissue as interacting spheroidal cells and couples a
       prescribed forward simulator with a differentiable learned inverse
       emulator. The two share a state representation, and each has its own dynamics.
     </p>
     <div class="wm-cta">
       <a class="wm-btn wm-btn-primary" href="installation.html">Install</a>
       <a class="wm-btn wm-btn-ghost" href="tutorials/index.html">Tutorials</a>
       <a class="wm-btn wm-btn-ghost"
          href="https://github.com/Computational-Morphogenomics-Group/waxmorph">GitHub</a>
     </div>
   </div>

   <figure class="wm-showcase">
     <img src="_static/images/fig1.png"
          alt="waxMorph pipeline: forward simulation and inverse learning of shape assembly">
     <figcaption>
       waxMorph represents tissue as interacting polarized spheroids, runs
       forward and inverse models of morphogenesis on that representation, and
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

   .. grid-item-card:: API reference
      :link: reference/index
      :link-type: doc
      :class-card: wm-card

      Public modules with PyTorch defaults and explicit JAX/Equinox
      counterparts.

   .. grid-item-card:: Contributing
      :link: contributing
      :link-type: doc
      :class-card: wm-card

      Development setup, testing across both backends, and the conventions your
      changes should follow.

Third-party code
----------------

``waxmorph.jax.soap_jax`` is borrowed from `SOAP_JAX <https://github.com/haydn-jones/SOAP_JAX>`_
by Haydn Jones (release v0.2.2), copied without modification under its MIT license, which is
reproduced at the top of the file. All credit for that code goes to its author. SOAP was
introduced by `Vyas et al. (2024) <https://arxiv.org/abs/2409.11321>`_; the PyTorch examples
use the SOAP implementation of `pytorch-optimizer <https://github.com/kozistr/pytorch_optimizer>`_,
installed by the ``learning`` extra.

Citation
--------

If you use waxMorph in your research, please cite the preprint:

   Beker, O. and Dumitrascu, B. (2026). *Differentiable Design for Morphogenesis I:
   Simulation and Simulacra*. bioRxiv. https://doi.org/10.64898/2026.07.02.736195

.. code-block:: bibtex

   @article{Beker_2026,
     title={Differentiable Design for Morphogenesis I: Simulation and Simulacra},
     url={http://dx.doi.org/10.64898/2026.07.02.736195},
     DOI={10.64898/2026.07.02.736195},
     publisher={openRxiv},
     author={Beker, Ozgur and Dumitrascu, Bianca},
     year={2026},
     month=July
   }

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   tutorials/index
   gallery
   reference/index
   contributing
   changelog
