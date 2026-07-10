# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### 2026-07-10

- Chamfer loss now uses directional means, preserving argument symmetry for unequal clouds.
- Squared loss rejects unequal shapes; Chamfer rejects empty or incompatible clouds.
- Torch and JAX MLPs reject invalid depths consistently.
- GeomLoss options preserve explicit `None`, reject unknown keys, and expose `truncate=5`.
- Verification: 311 passed, 6 CPU-only JAX/Warp skips; CUDA-focused tests 114 passed.
- Loss gradchecks: 24 passed across Torch/JAX CPU/CUDA; maximum relative error `1.022e-8`.
- Remaining: Chamfer gradients are nonsmooth at nearest-neighbor ties.
- FAILED: JAX-CUDA Chamfer directional FD with `h=1e-6` reached `2.070e-7` at seed 12
  because device-level rounding was amplified by cancellation; the predeclared `h=1e-5`
  matrix passed without per-case tuning.
- Growth now holds radii at or above their targets, clamps overshoot, caps mesenchymal
  equilibrium radii, and copies epithelial equilibrium radii.
- Growth rejects invalid `dt`, `R_ref`, and `R_max` before launching a kernel.
- Division reserves slots with compare-and-swap; the raw counter cannot exceed capacity.
- Verification: CPU-only suite 307 passed, 20 CUDA skips; H100 suite 333 passed, 6
  JAX-CUDA skips; JAX/Warp CUDA selection 57 passed.
- Growth-rate probe (`n=4096`): CPU/CUDA range `0.800052`–`0.999939`, mean `0.899432`,
  against the configured uniform range `[0.8, 1.0]`.
- Division contention: 20 H100 runs with 4,096 contenders for one slot accepted exactly
  one slot and stopped the counter at capacity every run.
- FAILED before correction: a large growth step overshot physical and equilibrium targets
  and left epithelial equilibrium outputs unwritten; division raised the raw count from 20
  to 28 despite capacity 25.

## [0.1.0] 2026-07-03

Initial release of waxMorph.
