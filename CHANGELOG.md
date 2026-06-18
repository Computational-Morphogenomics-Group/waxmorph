# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- GNS (Graph Network-based Simulator) architecture for learned emulation
- MLP building block module
- Warp-to-PyTorch graph construction utilities
- PyTorch and JAX graph, loss, model, and training backends
- Multi-target training over zero-based rollout frames
- Mesh sequence sampling helpers for target trajectories
- Forward simulator with sticky-sphere mechanics, reaction-diffusion, and cell division
- PyVista and OpenGL rendering backends
- Sphinx documentation with ReadTheDocs support
- Pre-commit hooks with ruff
- CI/CD with GitHub Actions

### Changed

- Variable and notation alignment with the waxMorph manuscript: per-cell signaling-molecule concentrations are now `c` (was `genes`/`G`) with count `num_molecules` (was `num_genes`); the predicted molecular increment is `dc` (was `dG`); `TrainConfig.D_emu` replaces `alpha_diff` and `TrainConfig.lambda_reg` replaces `l2_lambda`; training-log keys use `best_traj_c` (was `best_traj_genes`); simulator reaction-diffusion parameters are `chi`, `gamma`, and `D_inhib`, and growth parameters are `alpha_grow` and `ell_sw`.
- Plain JAX installs now use the CPU-compatible JAX extra; CUDA JAX support is exposed through `waxmorph[jax-cuda]`.

### Removed

- Legacy single-target training inputs in favor of explicit `(frame, positions)` target lists.
