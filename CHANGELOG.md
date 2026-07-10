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

## [0.1.0] 2026-07-03

Initial release of waxMorph.
