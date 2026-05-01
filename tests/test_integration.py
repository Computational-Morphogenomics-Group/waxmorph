"""Integration test: Warp state -> graph construction -> GNS forward."""

import importlib

import numpy as np
import pytest
import torch
import warp as wp

from waxmorph.gnn import GNS
from waxmorph.graph import build_graph
from waxmorph.torch.losses import squared_loss

torch_train_module = importlib.import_module("waxmorph.torch.train")

wp.init()

DEVICE = "cpu"

try:
    if wp.is_device_available("cuda"):
        DEVICE = "cuda"
except RuntimeError:
    DEVICE = "cpu"


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
    genes[:particle_count] = np.random.rand(particle_count, num_genes).astype(np.float32)

    X = wp.from_numpy(centers, dtype=wp.vec3f, device=DEVICE)
    P = wp.from_numpy(polarities, dtype=wp.vec3f, device=DEVICE)
    R = wp.from_numpy(radii, dtype=wp.float32, device=DEVICE)
    G = wp.from_numpy(genes, dtype=wp.float32, device=DEVICE)

    # Build graph
    node_feats, edge_index, edge_feats = build_graph(X, P, R, particle_count=particle_count, G=G)

    assert node_feats.shape == (particle_count, num_genes)
    assert edge_index.shape[0] == 2
    assert edge_feats.shape[1] == 2
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


def test_torch_graph_to_gns_preserves_gene_gradients():
    """Torch graph features should keep gene state connected to the GNN loss."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.25, 0.4, 0.0]],
        dtype=torch.float32,
    )
    polarities = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    radii = torch.full((3,), 0.5, dtype=torch.float32)
    genes = torch.randn(3, 2, dtype=torch.float32, requires_grad=True)

    node_feats, edge_index, edge_feats = build_graph(
        positions,
        polarities,
        radii,
        particle_count=3,
        G=genes,
    )

    gns = GNS(
        node_feature_dim=node_feats.shape[1],
        edge_feature_dim=edge_feats.shape[1],
        num_mp_steps=2,
        output_dims={"dX": 3, "dP": 3, "dG": genes.shape[1]},
    )

    out = gns(node_feats, edge_index, edge_feats)
    loss = out["dG"].square().sum() + out["dX"].square().sum()
    loss.backward()

    assert node_feats.grad_fn is not None
    assert genes.grad is not None
    assert torch.isfinite(genes.grad).all()


def test_torch_run_epoch_uses_live_torch_graph_inputs(monkeypatch):
    """The training rollout should call build_graph with live Torch state."""
    source_pos = np.array(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.25, 0.4, 0.0]],
        dtype=np.float32,
    )
    polarities = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    genes = np.array(
        [[0.2, -0.1], [0.0, 0.3], [-0.4, 0.5]],
        dtype=np.float32,
    )
    radii = np.full((3,), 0.5, dtype=np.float32)

    seen = {}
    real_build_graph = torch_train_module.build_graph

    def spy_build_graph(X, P, R, *args, **kwargs):
        seen["X"] = X
        seen["P"] = P
        seen["R"] = R
        seen["G"] = kwargs["G"]
        return real_build_graph(X, P, R, *args, **kwargs)

    monkeypatch.setattr(torch_train_module, "build_graph", spy_build_graph)

    model = GNS(
        node_feature_dim=genes.shape[1],
        edge_feature_dim=2,
        num_mp_steps=1,
        output_dims={"dX": 3, "dP": 3, "dG": genes.shape[1]},
    )

    config = torch_train_module.TrainConfig(
        n_epochs=1,
        t_rollout=1,
        mech_steps=0,
        diff_steps=0,
        log_every=1,
    )

    loss_shape, loss_l2, _trajectory = torch_train_module._run_epoch(
        model=model,
        config=config,
        source_pos=source_pos,
        polarities=polarities,
        genes=genes,
        R_t=torch.from_numpy(radii),
        R_wp=None,
        gx=None,
        lap_G=None,
        grid=None,
        N=source_pos.shape[0],
        targets_by_frame={0: torch.from_numpy(source_pos + 0.1)},
        X_source_t=torch.from_numpy(source_pos),
        loss_fn=squared_loss,
        torch_device=torch.device("cpu"),
        epoch_trajectory=[],
    )

    (loss_shape + loss_l2).backward()

    assert isinstance(seen["X"], torch.Tensor)
    assert isinstance(seen["P"], torch.Tensor)
    assert isinstance(seen["R"], torch.Tensor)
    assert isinstance(seen["G"], torch.Tensor)
    assert seen["X"].requires_grad is True
    assert seen["G"].requires_grad is True
    assert any(param.grad is not None for param in model.parameters())


def test_torch_run_epoch_clamps_genes_nonnegative_after_dg_update():
    """The learned gene update should not drive gene counts below zero."""
    source_pos = np.array(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.25, 0.4, 0.0]],
        dtype=np.float32,
    )
    polarities = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    genes = np.array(
        [[0.2, 0.1], [0.05, 0.3], [0.4, 0.5]],
        dtype=np.float32,
    )
    radii = np.full((3,), 0.5, dtype=np.float32)

    class FixedNegativeGeneUpdate(torch.nn.Module):
        def forward(self, node_feats, edge_index, edge_feats):
            del edge_index, edge_feats
            num_nodes, num_genes = node_feats.shape
            return {
                "dX": torch.zeros((num_nodes, 3), dtype=node_feats.dtype, device=node_feats.device),
                "dP": torch.zeros((num_nodes, 3), dtype=node_feats.dtype, device=node_feats.device),
                "dG": -torch.full(
                    (num_nodes, num_genes), 10.0, dtype=node_feats.dtype, device=node_feats.device
                ),
            }

    config = torch_train_module.TrainConfig(
        n_epochs=1,
        t_rollout=1,
        mech_steps=0,
        diff_steps=0,
        dt_gns=1.0,
        log_every=1,
    )

    loss_shape, loss_l2, trajectory = torch_train_module._run_epoch(
        model=FixedNegativeGeneUpdate(),
        config=config,
        source_pos=source_pos,
        polarities=polarities,
        genes=genes,
        R_t=torch.from_numpy(radii),
        R_wp=None,
        gx=None,
        lap_G=None,
        grid=None,
        N=source_pos.shape[0],
        targets_by_frame={0: torch.from_numpy(source_pos)},
        X_source_t=torch.from_numpy(source_pos),
        loss_fn=squared_loss,
        torch_device=torch.device("cpu"),
        epoch_trajectory=[],
    )

    assert loss_shape.item() == 0.0
    assert loss_l2.item() >= 0.0
    assert len(trajectory) == 1
    assert np.all(trajectory[0]["genes"] >= 0.0)
    assert np.allclose(trajectory[0]["genes"], 0.0)


def test_train_raises_before_optimizer_step_on_nonfinite_gradients():
    """Non-finite gradients should fail fast before they corrupt parameters."""
    source_pos = np.array(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.25, 0.4, 0.0]],
        dtype=np.float32,
    )
    polarities = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    genes = np.array(
        [[0.2, 0.1], [0.05, 0.3], [0.4, 0.5]],
        dtype=np.float32,
    )
    radii = np.full((3,), 0.5, dtype=np.float32)

    class NanGradModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.scale.register_hook(lambda grad: torch.full_like(grad, float("nan")))

        def forward(self, node_feats, edge_index, edge_feats):
            del edge_index, edge_feats
            num_nodes, num_genes = node_feats.shape
            scale = self.scale.to(node_feats.dtype)
            return {
                "dX": scale
                * torch.ones((num_nodes, 3), dtype=node_feats.dtype, device=node_feats.device),
                "dP": scale
                * torch.zeros((num_nodes, 3), dtype=node_feats.dtype, device=node_feats.device),
                "dG": scale
                * torch.zeros(
                    (num_nodes, num_genes), dtype=node_feats.dtype, device=node_feats.device
                ),
            }

    model = NanGradModel()
    initial_scale = model.scale.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = torch_train_module.TrainConfig(
        n_epochs=1,
        t_rollout=1,
        mech_steps=0,
        diff_steps=0,
        dt_gns=1.0,
        log_every=1,
    )

    with pytest.raises(ValueError, match=r"model gradients.*scale"):
        torch_train_module.train(
            model,
            optimizer,
            squared_loss,
            source_pos=source_pos,
            targets=[(0, source_pos + 0.1)],
            polarities=polarities,
            genes=genes,
            radii=radii,
            config=config,
            device="cpu",
        )

    assert torch.isfinite(model.scale).all()
    torch.testing.assert_close(model.scale.detach(), initial_scale)
