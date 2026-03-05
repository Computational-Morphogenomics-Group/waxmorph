"""Integration test: Warp state -> graph construction -> GNS forward."""

import numpy as np
import warp as wp

from waxmorph.gnn import GNS
from waxmorph.graph import build_graph

wp.init()

DEVICE = "cuda" if wp.is_device_available("cuda") else "cpu"


def _fibonacci_sphere(n):
    """Generate n approximately uniformly spaced points on a unit sphere."""
    golden = (1 + np.sqrt(5)) / 2
    indices = np.arange(n)
    theta = 2 * np.pi * indices / golden
    phi = np.arccos(1 - 2 * (indices + 0.5) / n)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def test_warp_to_gns_roundtrip():
    """Full pipeline: create Warp state, build graph, run GNS, check outputs."""
    particle_count = 80
    max_particles = 200
    num_genes = 2

    # Create positions on a sphere
    centers = np.zeros((max_particles, 3), dtype=np.float32)
    centers[:particle_count] = _fibonacci_sphere(particle_count) * 2.0

    radii = np.zeros(max_particles, dtype=np.float32)
    radii[:particle_count] = 0.5

    polarities = np.zeros((max_particles, 3), dtype=np.float32)
    polarities[:particle_count] = centers[:particle_count] / (
        np.linalg.norm(centers[:particle_count], axis=-1, keepdims=True) + 1e-9
    )

    cell_types = np.zeros(max_particles, dtype=np.uint32)
    cell_types[:particle_count] = 1  # all epithelium

    genes = np.zeros((max_particles, num_genes), dtype=np.float32)
    genes[:particle_count] = np.random.rand(particle_count, num_genes).astype(
        np.float32
    )

    X = wp.from_numpy(centers, dtype=wp.vec3f, device=DEVICE)
    P = wp.from_numpy(polarities, dtype=wp.vec3f, device=DEVICE)
    R = wp.from_numpy(radii, dtype=wp.float32, device=DEVICE)
    CT = wp.from_numpy(cell_types, dtype=wp.uint32, device=DEVICE)
    G = wp.from_numpy(genes, dtype=wp.float32, device=DEVICE)

    # Build graph
    node_feats, edge_index, edge_feats = build_graph(
        X, P, R, CT, particle_count=particle_count, G=G
    )

    assert node_feats.shape == (particle_count, 9 + num_genes)
    assert edge_index.shape[0] == 2
    assert edge_feats.shape[1] == 7
    assert edge_index.shape[1] > 0, "Should have at least some edges"

    # Run GNS
    gns = GNS(
        node_feature_dim=node_feats.shape[1],
        edge_feature_dim=edge_feats.shape[1],
        num_mp_steps=3,
        output_dims={"dX": 3, "dP": 3, "dG": num_genes},
    )

    # Move model to same device as data
    dev = node_feats.device
    gns = gns.to(dev)

    out = gns(node_feats, edge_index, edge_feats)

    assert out["dX"].shape == (particle_count, 3)
    assert out["dP"].shape == (particle_count, 3)
    assert out["dG"].shape == (particle_count, num_genes)

    # Verify gradient flow through the full pipeline
    node_feats_grad = node_feats.detach().requires_grad_(True)
    out2 = gns(node_feats_grad, edge_index, edge_feats)
    loss = sum(v.sum() for v in out2.values())
    loss.backward()
    assert node_feats_grad.grad is not None
