# WaxMorph

[![CI](https://github.com/waxmorph/waxmorph/actions/workflows/ci.yml/badge.svg)](https://github.com/waxmorph/waxmorph/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/waxmorph/badge/?version=latest)](https://waxmorph.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Joint forward simulation and inverse learning of biophysical shape assembly
with spheroidal agents.

## Overview

WaxMorph represents tissues as interacting spheroidal agents with positions,
radii, polarities, gene or morphogen states, and optional cell-type labels. It
connects explicit mechanochemical simulation with graph-network emulation, so
users can both simulate rule-based tissue dynamics and train local update rules
against target morphologies.

| Approach | Description |
|---|---|
| **Forward simulation** | Mechanochemical dynamics with sticky-sphere forces, reaction-diffusion, and cell division |
| **Emulation (non-growing)** | Learn local update rules that assemble a fixed number of cells into a target shape |
| **Multi-frame supervision** | Train against intermediate and final target morphologies during a fixed-cell rollout |

Key features:

- GPU-accelerated simulation via [NVIDIA Warp](https://nvidia.github.io/warp/)
- GNS (Graph Network-based Simulator) architecture for learned emulation via PyTorch
- Two cell types (epithelium / mesenchyme) with type-dependent mechanics
- Mesh-to-point-cloud sampling for source and target morphologies
- PyTorch and JAX graph/training implementations
- Rendering through PyVista / VTK and Warp OpenGL

## Installation

```bash
pip install waxmorph
```

With simulation and learning dependencies:

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

## Quick start

```python
import torch

from waxmorph.graph import build_graph
from waxmorph.gnn import GNS

# Build a contact graph from a spheroidal tissue state.
N = 32
num_genes = 2
X = torch.randn(N, 3)
P = torch.nn.functional.normalize(X, dim=1)
R = torch.full((N,), 0.45)
G = torch.rand(N, num_genes)

node_features, edge_index, edge_features = build_graph(
    X, P, R, particle_count=N, G=G,
)

# Predict per-cell updates with a graph-network emulator.
model = GNS(
    node_feature_dim=node_features.shape[1],
    edge_feature_dim=edge_features.shape[1],
    num_mp_steps=10,
    output_dims={"dX": 3, "dP": 3, "dG": num_genes},
)

outputs = model(node_features, edge_index, edge_features)
X_next = X + 1e-2 * outputs["dX"]
```

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
