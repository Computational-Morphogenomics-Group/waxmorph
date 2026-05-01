# WaxMorph

[![CI](https://github.com/waxmorph/waxmorph/actions/workflows/ci.yml/badge.svg)](https://github.com/waxmorph/waxmorph/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/waxmorph/badge/?version=latest)](https://waxmorph.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Joint forward simulation and inverse learning of biophysical shape assembly
with spheroidal agents.

## Overview

WaxMorph represents tissues as interacting spheroidal agents with positions,
radii, polarities, gene or morphogen states, and optional cell-type labels. The
package is organized around three connected workflows:

| Workflow | Use it when you want to | Main modules |
|---|---|---|
| **Simulation** | Run explicit mechanochemical dynamics with sticky-sphere mechanics, reaction-diffusion, growth, and division | `waxmorph.simulator` |
| **Emulation** | Train a graph-network local update rule that assembles a fixed number of agents into a target shape | `waxmorph.data`, `waxmorph.graph`, `waxmorph.gnn`, `waxmorph.train` |
| **Rendering** | Inspect states, trajectories, and learned rollouts as static views, interactive views, USD stages, or OpenGL videos | `waxmorph.render` |

The default learning path uses PyTorch-compatible top-level imports. JAX and
Equinox implementations live under `waxmorph.jax`.

## Installation

For the base package:

```bash
pip install waxmorph
```

For the usual simulation and PyTorch learning workflow:

```bash
pip install "waxmorph[simulation,learning]"
```

For the JAX/Equinox backend on CPU:

```bash
pip install "waxmorph[jax]"
```

For JAX with CUDA 12 support:

```bash
pip install "waxmorph[jax-cuda]"
```

For development:

```bash
git clone https://github.com/waxmorph/waxmorph.git
cd waxmorph
pip install -e ".[all]"
pre-commit install
```

The simulation kernels and the main movie-rendering paths are designed for
NVIDIA Warp and are normally run on CUDA. Small graph and data utilities can be
used on CPU, but full examples are GPU-oriented. USD export uses Warp's USD
renderer; if that backend is unavailable in your environment, install a USD
Python package such as `usd-core` or `usd-exchange`.

## Quickstart: mesh-to-mesh shape assembly

This example follows the workflow in `shape_assembly.ipynb`: sample a source
mesh and target mesh, initialize a fixed population of spheroidal agents, train
a Graph Network-based Simulator (GNS), then export the best rollout as a USD
stage. The bundled meshes live in `meshes/`.

Inputs:

- `meshes/bunny.ply`: source morphology sampled into initial cell centers.
- `meshes/armadillo.ply`: target morphology sampled into training targets.
- `N_POINTS`: fixed number of agents. This emulation workflow does not add or
  remove particles during the rollout.

Outputs:

- `runs/bunny_armadillo.pt`: best PyTorch model checkpoint and training log.
- `Output/shape_assembly.usd`: USD trajectory for inspection in a USD viewer.
- `result.log["best_traj_pos"]`: NumPy trajectory with shape
  `[time, particles, 3]`.

```python
from pathlib import Path

import numpy as np
import torch
import warp as wp

from waxmorph.data import sample_mesh_pair
from waxmorph.gnn import GNS
from waxmorph.graph import build_graph
from waxmorph.losses import make_samples_loss
from waxmorph.render import WarpMovieRenderer
from waxmorph.train import TrainConfig, train

wp.init()

device = "cuda"
torch_device = torch.device(device)
num_genes = 32
n_points = 2000

data = sample_mesh_pair(
    source_path="meshes/bunny.ply",
    target_path="meshes/armadillo.ply",
    n_points=n_points,
    source_extent=10.0,
    target_extent=14.29,
    seed=0,
)

n_cells = len(data["source_pos"])
radii = np.full(n_cells, data["source_radius"], dtype=np.float32)

rng = np.random.default_rng(42)
polarities = rng.standard_normal((n_cells, 3)).astype(np.float32)
polarities /= np.linalg.norm(polarities, axis=1, keepdims=True) + 1e-9
genes = rng.random((n_cells, num_genes), dtype=np.float32)

# Probe the graph once so the model dimensions match the current state encoding.
X_probe = wp.from_numpy(data["source_pos"], dtype=wp.vec3f, device=device)
P_probe = wp.from_numpy(polarities, dtype=wp.vec3f, device=device)
R_probe = wp.from_numpy(radii, dtype=wp.float32, device=device)
G_probe = wp.from_numpy(genes, dtype=wp.float32, device=device)

node_features, edge_index, edge_features = build_graph(
    X_probe,
    P_probe,
    R_probe,
    particle_count=n_cells,
    G=G_probe,
)

model = GNS(
    node_feature_dim=node_features.shape[1],
    edge_feature_dim=edge_features.shape[1],
    node_latent_dim=256,
    edge_latent_dim=256,
    hidden_dim=256,
    num_mp_steps=5,
    num_mlp_layers=3,
    output_dims={"dX": 3, "dP": 3, "dG": num_genes},
    checkpoint_processor=True,
).to(torch_device)

config = TrainConfig(n_epochs=2000, t_rollout=100)
loss_fn = make_samples_loss(loss="sinkhorn")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

result = train(
    model,
    optimizer,
    loss_fn,
    source_pos=data["source_pos"],
    polarities=polarities,
    genes=genes,
    radii=radii,
    targets=[(config.t_rollout - 1, data["target_pos"])],
    config=config,
    save_path="runs/bunny_armadillo.pt",
    device=device,
)

trajectory = result.log["best_traj_pos"]
colors = np.full((n_cells, 3), [0.3, 0.6, 0.9], dtype=np.float32)

Path("Output").mkdir(exist_ok=True)
with WarpMovieRenderer(
    filename="Output/shape_assembly.usd",
    backend="usd",
    max_particles=n_cells,
    fps=30,
    device=device,
) as movie:
    for frame, centers in enumerate(trajectory):
        movie.write_frame_from_numpy(
            t=float(frame),
            centers=centers,
            radii=radii,
            colors=colors,
            particle_count=n_cells,
        )
```

For a shorter smoke test, reduce `n_points`, `n_epochs`, and `t_rollout`.
Training quality depends on those values, so small settings are useful for API
validation but not for judging morphology quality.

## Simulation workflow

Use explicit simulation when the scientific question is about a specified
mechanistic model rather than a learned shape-assembly rule. The
`simulation_with_autodiff.ipynb` notebook demonstrates the main components:

1. Allocate fixed-capacity Warp arrays for centers `X`, radii `R`, equilibrium
   radii `R_eq`, polarities `P`, activator `A`, inhibitor `I`, and cell types
   `CT`.
2. Relax geometry with `simulator.mech_step_sticky` or the autodiff-consistent
   `simulator.mech_step_sticky_implicit`.
3. Pattern morphogens with `simulator.chem_step`.
4. Grow cells with `simulator.growth_step`.
5. Count neighbors and divide cells with `simulator.count_neighbors_step`,
   `simulator.division_decision`, and `simulator.division_logic`.
6. Write frames from the live Warp state with
   `render.WarpMovieRenderer.write_frame_from_state`.

Inputs are Warp arrays and scalar parameters such as time steps, diffusion
rates, growth constants, and division thresholds. Outputs are updated in-place
state arrays and, when requested, rendered frames. The simulation path can grow
the active particle count up to the preallocated `max_particles` capacity.

The notebook includes long-running examples with tens of thousands of particles
and many time steps. Start with lower particle counts and fewer steps when
validating a new environment.

## Emulation workflow

Use emulation when you have source and target morphologies and want to learn a
local rollout rule. The standard PyTorch flow is:

1. Sample meshes with `sample_mesh_pair` for a single final target, or
   `sample_mesh_sequence` for intermediate target frames.
2. Initialize `source_pos`, `polarities`, `genes`, and `radii` as NumPy arrays.
3. Build graph features with `build_graph`.
4. Construct `GNS` with output heads matching the variables to update, usually
   `{"dX": 3, "dP": 3, "dG": num_genes}`.
5. Train with `train(..., targets=[(frame, target_pos), ...])`.
6. Inspect `TrainResult.log`, especially `losses_total` and `best_traj_pos`.

Frame indices in `targets` are zero-based rollout steps after updates. For
example, `(99, target_pos)` supervises the state after 100 learned updates when
`TrainConfig(t_rollout=100)`.

The JAX/Equinox version follows the same data flow with explicit backend
imports:

```python
from waxmorph.jax.gnn import GNS
from waxmorph.jax.graph import build_graph
from waxmorph.jax.losses import make_sinkhorn_loss
from waxmorph.jax.train import TrainConfig, train
```

The JAX training call also takes an Optax optimizer and optimizer state. Use
the JAX backend when your downstream analysis already depends on JAX, Equinox,
or Optax, or when static-shape compilation is important.

## Rendering workflow

`waxmorph.render` supports three rendering styles:

- `MPLInterface`: static Matplotlib inspection in simple scripts or notebooks.
- `PyVistaInterface`: interactive PyVista/VTK views of spheroids, polarities,
  morphogen colors, and cell-type colors.
- `WarpMovieRenderer`: trajectory export through Warp. Use `backend="usd"` for
  the main USD-stage path, or `backend="opengl"` for a headless video file.

For learned shape assembly, use `write_frame_from_numpy` with positions, radii,
and explicit RGB colors. For live simulations, use `write_frame_from_state`
with Warp arrays and choose `morph="A"`, `morph="I"`, or `morph="ratio"` to
color particles by morphogen state.

## Citation

If you use WaxMorph in your research, please cite:

```bibtex
@software{waxmorph,
  title  = {WaxMorph: Differentiable Morphogenesis on NVIDIA Warp},
  author = {WaxMorph Contributors},
  year   = {2026},
  url    = {https://github.com/waxmorph/waxmorph},
}
```
