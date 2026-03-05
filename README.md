# WaxMorph

[![CI](https://github.com/waxmorph/waxmorph/actions/workflows/ci.yml/badge.svg)](https://github.com/waxmorph/waxmorph/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/waxmorph/badge/?version=latest)](https://waxmorph.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Differentiable morphogenesis and shape-assembly on NVIDIA Warp.

## Overview

WaxMorph integrates three approaches to generating biological shapes under biophysical constraints:

| Approach | Description |
|---|---|
| **Forward simulation** | Mechanochemical dynamics with sticky-sphere forces, reaction-diffusion, and cell division |
| **Emulation (non-growing)** | Learn local update rules that assemble a fixed number of cells into a target shape |
| **Emulation (growing)** | Learn rules that grow seed cells into a target configuration with division |

Key features:

- GPU-accelerated simulation via [NVIDIA Warp](https://nvidia.github.io/warp/)
- GNS (Graph Network-based Simulator) architecture for learned emulation via PyTorch
- Two cell types (epithelium / mesenchyme) with type-dependent mechanics
- Rendering through PyVista / VTK and Warp OpenGL

## Installation

```bash
pip install waxmorph
```

With simulation and learning dependencies:

```bash
pip install "waxmorph[simulation,learning]"
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
from waxmorph.graph import build_graph
from waxmorph.gnn import GNS

# Build graph from Warp simulation state
node_features, edge_index, edge_features = build_graph(
    X, P, R, CT, particle_count=N, G=G,
)

# Create GNS model
model = GNS(
    node_feature_dim=node_features.shape[1],
    edge_feature_dim=edge_features.shape[1],
    num_mp_steps=10,
    output_dims={"dX": 3, "dP": 3, "dG": 2},
)

# Predict updates
outputs = model(node_features, edge_index, edge_features)
# outputs["dX"]: predicted position updates [N, 3]
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
