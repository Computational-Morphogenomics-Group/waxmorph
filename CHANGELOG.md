# Changelog

## 0.1.1 - 2026-07-11

### Changed

- Declare SciPy, PyTorch, tqdm, and Warp 1.10+ as core dependencies; add USD support to the
  simulation extra; preserve caller-managed Warp caches during import.
- Apply shared validation to graph, loss, MLP, rendering, and training inputs across Torch
  and JAX.
- Align histories, selected models, trajectories, and checkpoint refinement with post-update
  states; restore matching Torch optimizer state and initialize JAX Optax state from loaded
  weights.
- Prefer Warp's public JAX kernel API while retaining compatibility with Warp 1.10's
  experimental FFI.

### Fixed

- Fix Chamfer loss to use Euclidean directional means; validate squared and Chamfer input
  shapes; preserve explicit GeomLoss options and reject unknown keys.
- Fix growth and division to respect target and capacity bounds; preserve inactive emulator
  rows and their identity gradients.
- Fix categorical rendering point counts and polarity validation.

### Documentation

- Document the purpose-built simulator and fixed-agent emulator, backend-specific contracts,
  model units, CUDA acceptance, and runtime decisions in ADR 0001.

## 0.1.0 - 2026-07-03

Initial release of waxMorph.
