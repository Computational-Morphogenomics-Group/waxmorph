# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### 2026-07-10

- Chamfer loss now uses directional means, preserving argument symmetry for unequal clouds.
- Squared loss rejects unequal shapes; Chamfer rejects empty or incompatible clouds.
- Torch and JAX MLPs reject invalid depths consistently.
- GeomLoss options preserve explicit `None`, reject unknown keys, and expose `truncate=5`.
- Growth now holds radii at or above their targets, clamps overshoot, caps mesenchymal
  equilibrium radii, and copies epithelial equilibrium radii.
- Growth rejects invalid `dt`, `R_ref`, and `R_max` before launching a kernel.
- Division reserves slots with compare-and-swap; the raw counter cannot exceed capacity.
- Mechanics now preserves values and identity gradients for rows beyond `particle_count`.
- Diffusion kernels no longer accept unused positions or radii; public signatures and
  Torch/JAX results are unchanged.
- Torch and JAX graph builders now reject invalid geometry, particle counts, feature shapes,
  and edge indices.
- PyVista categorical points ignore morphogen length and reject short polarity arrays.
- Torch and JAX trainers validate configurations and convert states and targets to contiguous
  float32 before device or checkpoint setup.
- Torch checkpoint refinement preserves model and optimizer bindings, rejects incompatible
  architectures, and clears stale optimizer state.

## [0.1.0] 2026-07-03

Initial release of waxMorph.
