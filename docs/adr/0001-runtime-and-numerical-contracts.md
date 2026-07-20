# ADR 0001: Runtime and numerical contracts

## Context

waxMorph exposes a prescribed forward simulator, a learned emulator, and Torch
and JAX learning backends. These paths share state abstractions, but each has
its own kernels, constants, interfaces, and runtime contracts. The decisions
below define how they compose numerically and what release acceptance requires.

## Decision

1. Coordinates, radii, timesteps, and coefficients use nondimensional model
   units. External calibration maps them to SI or biological scales.
2. Physical radii below their growth target increase up to that target. Radii
   at or above target pass through unchanged. Mesenchymal equilibrium radii are capped
   at `R_max`; epithelial equilibrium radii copy through while physical radii
   approach `R_ref`.
3. Chamfer loss is the sum of Euclidean directional means:

   $$
   \frac{1}{N}\sum_i\min_j\lVert x_i-y_j\rVert_2+
   \frac{1}{M}\sum_j\min_i\lVert x_i-y_j\rVert_2.
   $$

   Inputs are nonempty rank-2 arrays with equal feature width. Nearest-neighbor
   ties remain nonsmooth and follow backend-specific gradient conventions.
4. `n_epochs` counts optimizer updates. Histories and model selection describe
   the resulting models, requiring one extra evaluation. The returned model,
   best loss, epoch, and trajectory refer to the same post-update state.
5. Top-level imports eagerly expose the default Torch API. NumPy, SciPy, Torch,
   tqdm, and Warp are core dependencies; JAX, GeomLoss, and rendering stacks
   remain optional.
6. Polarity vectors supplied to graph construction must have unit norm. Edge
   features clamp their raw supplied dot product before `acos`.
7. Forward evolution rebuilds contact topology. Each differentiated graph or
   physics operation holds that topology fixed while gradients cover continuous
   state arithmetic; neighbor search and edge updates define the discrete
   topology input.
8. Numerical acceptance requires CPU and CUDA verification. CUDA checks must
   prove Torch, JAX, and Warp device placement and record execution of GPU
   atomics and both Torch/Warp and JAX/Warp gradient paths.

## Consequences

- External calibration supplies physical interpretation for model values.
- Growth passes through incoming radii at or above their targets.
- The initial model evaluation supplies the first gradient. The updated models
  form the selection candidates, and evaluation cost is one rollout greater
  than the update count.
- Differentiation follows a piecewise-smooth surrogate within each fixed
  topology.
- Eager imports provide an unconditional default API with its core dependency
  closure.
- Release acceptance combines CPU CI and CUDA verification. JAX/Warp mechanics
  and diffusion run on a CUDA-backed JAX device.
