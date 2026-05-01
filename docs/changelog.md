# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- GNS (Graph Network-based Simulator) architecture for learned emulation (`waxmorph.gnn`)
- MLP building block module (`waxmorph.mlp`)
- Warp-to-PyTorch graph construction utilities (`waxmorph.graph`)
- PyTorch and JAX graph, loss, model, and training backends
- Multi-target training over zero-based rollout frames
- Mesh sequence sampling helpers for target trajectories
- Forward simulator with sticky-sphere mechanics, reaction-diffusion, and cell division
- PyVista and OpenGL rendering backends
- Sphinx documentation with ReadTheDocs support
- Pre-commit hooks with ruff
- CI/CD with GitHub Actions

### Changed

- Plain JAX installs now use the CPU-compatible JAX extra; CUDA JAX support is exposed through `waxmorph[jax-cuda]`.
- Documentation now describes non-growing emulator training as the implemented training mode.

### Removed

- Legacy single-target training inputs in favor of explicit `(frame, positions)` target lists.
