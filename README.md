# waxMorph

[![CI](https://github.com/Computational-Morphogenomics-Group/waxmorph/actions/workflows/ci.yml/badge.svg)](https:/Computational-Morphogenomics-Group/waxmorph/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/waxmorph/badge/?version=latest)](https://waxmorph.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://pypi.org/project/waxmorph/"><img alt="PyPI" src="https://img.shields.io/pypi/v/waxmorph"></a>
<a href="https://pypi.org/project/waxmorph/"><img alt="Supported Python Versions" src="https://img.shields.io/pypi/pyversions/waxmorph?color=brightgreen"></a>

**A differentiable cell-based framework for three-dimensional morphogenesis.**

Cells build tissues through local exchanges of force and information, yet the
rules governing these interactions are difficult to infer from sparse
observations. waxMorph represents tissue as interacting spheroidal cells and
makes the governing mechanochemical dynamics differentiable, so the same model
can be run forward to simulate prescribed shape programs and inverse to
reconstruct continuous morphogenetic trajectories from static tissue volumes.

📖 **[Documentation](https://waxmorph.readthedocs.io/en/latest)** ·
[Tutorials](https://waxmorph.readthedocs.io/en/latest/tutorials/index.html) ·
[Gallery](https://waxmorph.readthedocs.io/en/latest/gallery.html) ·
[Explanation](https://waxmorph.readthedocs.io/en/latest/explanation/index.html) ·
[API reference](https://waxmorph.readthedocs.io/en/latest/reference/index.html)

## Overview

waxMorph represents tissues as three-dimensional populations of spheroidal
cellular agents, each carrying position, volume, polarity, signaling molecule
concentrations, and an optional cell-type label. A graph-network processor
learns local, neighbor-dependent update rules from target morphologies, while
differentiable physical constraints guide tissue-scale assembly. Local
neighborhoods are induced by spatial proximity and rebuilt as tissues deform.
The package is organized around three connected workflows:

| Workflow | Purpose | Main modules |
|---|---|---|
| **Simulation** | Forward simulation of explicit mechanochemical dynamics: soft-sphere mechanics, reaction-diffusion patterning, growth, and division | `waxmorph.simulator` |
| **Emulation** | Learned emulation in which a graph-network-based simulator (GNS) assembles a fixed population of agents into a target morphology under differentiable physics | `waxmorph.data`, `waxmorph.graph`, `waxmorph.gnn`, `waxmorph.train` |
| **Rendering** | Visualization of states, trajectories, and learned rollouts as static views, interactive views, USD stages, or OpenGL videos | `waxmorph.render` |

The default learning path resolves the PyTorch implementations through the
top-level imports. JAX and Equinox implementations provide a parity backend
under `waxmorph.jax`.

## Installation

The base package:

```bash
pip install waxmorph
```

The forward simulation and PyTorch learning workflow:

```bash
pip install "waxmorph[simulation,learning]"
```

The JAX/Equinox backend on CPU:

```bash
pip install "waxmorph[jax]"
```

JAX with CUDA 12 support:

```bash
pip install "waxmorph[jax-cuda]"
```

The development installation:

```bash
git clone https://github.com/Computational-Morphogenomics-Group/waxmorph.git
cd waxmorph
pip install -e ".[all]"
pre-commit install
```

The simulation kernels and the principal movie-rendering paths are implemented
on NVIDIA Warp and are intended to run on CUDA. The graph and data utilities are
available on CPU, but the full examples are GPU-oriented. USD export relies on
Warp's USD renderer; where that backend is unavailable, a USD Python package
such as `usd-core` or `usd-exchange` is required.

## Quickstart: mesh-to-mesh shape assembly

This example follows the workflow in `shape_assembly.ipynb`: a source mesh and a
target mesh are sampled, a fixed population of spheroidal agents is initialized,
a graph-network-based simulator (GNS) is trained, and the best rollout is
exported as a USD stage. The bundled meshes reside in `meshes/`.

Inputs:

- `meshes/bunny.ply`: source morphology sampled into initial cell centers.
- `meshes/armadillo.ply`: target morphology sampled into training targets.
- `n_points`: fixed number of agents. This learned-emulation workflow neither
  adds nor removes particles during the rollout.

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
num_molecules = 32
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
c = rng.random((n_cells, num_molecules), dtype=np.float32)

# Probe the graph once so the model dimensions match the state encoding.
X_probe = wp.from_numpy(data["source_pos"], dtype=wp.vec3f, device=device)
P_probe = wp.from_numpy(polarities, dtype=wp.vec3f, device=device)
R_probe = wp.from_numpy(radii, dtype=wp.float32, device=device)
c_probe = wp.from_numpy(c, dtype=wp.float32, device=device)

node_features, edge_index, edge_features = build_graph(
    X_probe,
    P_probe,
    R_probe,
    particle_count=n_cells,
    c=c_probe,
)

model = GNS(
    node_feature_dim=node_features.shape[1],
    edge_feature_dim=edge_features.shape[1],
    node_latent_dim=256,
    edge_latent_dim=256,
    hidden_dim=256,
    num_mp_steps=5,
    num_mlp_layers=3,
    output_dims={"dX": 3, "dP": 3, "dc": num_molecules},
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
    c=c,
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

A shorter smoke test follows from reducing `n_points`, `n_epochs`, and
`t_rollout`. Assembly quality depends on those values, so small settings serve
API validation rather than assessment of morphology quality.

## Simulation workflow

Forward simulation is appropriate when the scientific question concerns a
specified mechanistic model rather than a learned shape-assembly rule. The
`simulation_with_autodiff.ipynb` notebook demonstrates the main components:

1. Allocate fixed-capacity Warp arrays for centers `X`, radii `R`, equilibrium
   radii `R_eq`, polarities `P`, activator `A`, inhibitor `I`, and cell types
   `CT`.
2. Relax geometry under the soft-sphere potential with
   `simulator.mech_step_sticky`, or with the autodiff-consistent
   `simulator.mech_step_sticky_implicit`.
3. Pattern the activator and inhibitor fields by reaction-diffusion with
   `simulator.chem_step`, parameterized by `chi` (the spatial characteristic of
   the activator), `gamma` (the reaction rate), and `D_inhib` (the inhibitor
   diffusivity).
4. Grow cells with `simulator.growth_step`, parameterized by the growth Hill
   exponent `alpha_grow` and the switch concentration `ell_sw`.
5. Count neighbors and divide cells with `simulator.count_neighbors_step`,
   `simulator.division_decision`, and `simulator.division_logic`.
6. Write frames from the live Warp state with
   `render.WarpMovieRenderer.write_frame_from_state`.

Inputs are Warp arrays and scalar parameters such as time steps, diffusivities,
growth constants, and division thresholds. Outputs are state arrays updated
in-place and, when requested, rendered frames. The simulation path can grow the
active particle count up to the preallocated `max_particles` capacity.

The notebook includes long-running examples with tens of thousands of particles
over many time steps. Lower particle counts and fewer steps are advisable when
validating a new environment.

## Emulation workflow

Learned emulation is appropriate given source and target morphologies and the
objective of inferring a local rollout rule. The standard PyTorch flow is:

1. Sample meshes with `sample_mesh_pair` for a single final target, or
   `sample_mesh_sequence` for intermediate target frames.
2. Initialize `source_pos`, `polarities`, the signaling molecule concentrations
   `c`, and `radii` as NumPy arrays.
3. Build graph features with `build_graph`, which induces the contact graph from
   spatial proximity.
4. Construct `GNS` with output heads matching the variables to update, namely
   `{"dX": 3, "dP": 3, "dc": num_molecules}`.
5. Train with `train(..., targets=[(frame, target_pos), ...])`.
6. Inspect `TrainResult.log`, in particular `losses_total` and `best_traj_pos`.

Frame indices in `targets` are zero-based rollout steps after updates. For
example, `(99, target_pos)` supervises the state after 100 learned updates under
`TrainConfig(t_rollout=100)`. Molecule diffusion during the rollout is
controlled by `TrainConfig.D_emu`, and the squared-displacement regularization
by `TrainConfig.lambda_reg`. The corresponding concentration trajectory is
recorded in `result.log["best_traj_c"]`.

The JAX/Equinox backend follows the same data flow with explicit backend
imports:

```python
from waxmorph.jax.gnn import GNS
from waxmorph.jax.graph import build_graph
from waxmorph.jax.losses import make_sinkhorn_loss
from waxmorph.jax.train import TrainConfig, train
```

On the JAX backend, `build_graph` additionally returns the active edge count as
`(node_features, edge_index, edge_features, num_edges)`, and the training call
takes an Optax optimizer and optimizer state. The JAX backend is appropriate
when downstream analysis already depends on JAX, Equinox, or Optax, or when
static-shape compilation is required.

## Rendering workflow

`waxmorph.render` provides three rendering styles:

- `MPLInterface`: static Matplotlib inspection for scripts or notebooks.
- `PyVistaInterface`: interactive PyVista/VTK views of spheroids, polarities,
  signaling molecule concentrations, and cell-type colors.
- `WarpMovieRenderer`: trajectory export through Warp, with `backend="usd"` for
  the principal USD-stage path and `backend="opengl"` for a headless video file.

For learned shape assembly, `write_frame_from_numpy` takes positions, radii, and
explicit RGB colors. For live simulations, `write_frame_from_state` takes Warp
arrays, and `morph="A"`, `morph="I"`, or `morph="ratio"` colors particles by the
activator, inhibitor, or activator-to-inhibitor ratio of the reaction-diffusion
state.

## Citation

If you use waxMorph in your research, please cite the preprint:

> Beker, O. and Dumitrascu, B. (2026). *Differentiable Design for Morphogenesis
> I: Simulation and Simulacra*. bioRxiv.
> [https://doi.org/10.64898/2026.07.02.736195](https://www.biorxiv.org/content/10.64898/2026.07.02.736195v1)

```bibtex
@article{beker2026waxmorph,
  title     = {Differentiable Design for Morphogenesis {I}: Simulation and Simulacra},
  author    = {Beker, Ozgur and Dumitrascu, Bianca},
  journal   = {bioRxiv},
  year      = {2026},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.64898/2026.07.02.736195},
  url       = {https://www.biorxiv.org/content/10.64898/2026.07.02.736195v1},
}
```
