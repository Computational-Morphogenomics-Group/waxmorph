"""Multi-target trajectory matching in the PyTorch non-growing trainer."""

import importlib

import numpy as np
import pytest
import torch
import warp as wp

from waxmorph.gnn import GNS
from waxmorph.torch.losses import squared_loss

torch_train_module = importlib.import_module("waxmorph.torch.train")

wp.init()

DEVICE = "cpu"
try:
    if wp.is_device_available("cuda"):
        DEVICE = "cuda"
except RuntimeError:
    DEVICE = "cpu"


def _minimal_inputs():
    source_pos = np.array(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.25, 0.4, 0.0]],
        dtype=np.float32,
    )
    polarities = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    c = np.array(
        [[0.2, 0.1], [0.05, 0.3], [0.4, 0.5]],
        dtype=np.float32,
    )
    radii = np.full((3,), 0.5, dtype=np.float32)
    return source_pos, polarities, c, radii


def _make_model(num_molecules):
    return GNS(
        node_feature_dim=num_molecules,
        edge_feature_dim=2,
        num_mp_steps=1,
        output_dims={"dX": 3, "dP": 3, "dc": num_molecules},
    )


def test_run_epoch_accumulates_loss_across_tagged_frames():
    """Loss from multiple tagged frames should sum into loss_shape."""
    source_pos, polarities, c, radii = _minimal_inputs()

    config = torch_train_module.TrainConfig(
        n_epochs=1,
        t_rollout=3,
        mech_steps=0,
        diff_steps=0,
        dt_gns=0.0,  # freeze physics so X_t stays at source_pos
        log_every=1,
    )

    target_a = torch.from_numpy(source_pos + 0.1)
    target_b = torch.from_numpy(source_pos + 0.3)

    torch.manual_seed(0)
    model = _make_model(c.shape[1])

    loss_shape, _loss_l2, trajectory = torch_train_module._run_epoch(
        model=model,
        config=config,
        source_pos=source_pos,
        polarities=polarities,
        c=c,
        R_t=torch.from_numpy(radii),
        R_wp=None,
        f_net=None,
        lap_c=None,
        grid=None,
        N=source_pos.shape[0],
        targets_by_frame={0: target_a, 2: target_b},
        X_source_t=torch.from_numpy(source_pos),
        loss_fn=squared_loss,
        torch_device=torch.device("cpu"),
        epoch_trajectory=[],
    )

    # With dt_gns=0 and no physics, X_t does not move between steps; loss equals
    # sum of squared_loss at the two targets evaluated at source_pos.
    expected = squared_loss(torch.from_numpy(source_pos), target_a) + squared_loss(
        torch.from_numpy(source_pos), target_b
    )
    assert torch.allclose(loss_shape, expected, atol=1e-6)
    assert len(trajectory) == config.t_rollout


def test_run_epoch_skips_untagged_frames():
    """Steps whose index is not in targets_by_frame should contribute no shape loss."""
    source_pos, polarities, c, radii = _minimal_inputs()

    config = torch_train_module.TrainConfig(
        n_epochs=1,
        t_rollout=3,
        mech_steps=0,
        diff_steps=0,
        dt_gns=0.0,
        log_every=1,
    )

    torch.manual_seed(0)
    model = _make_model(c.shape[1])

    loss_shape, *_ = torch_train_module._run_epoch(
        model=model,
        config=config,
        source_pos=source_pos,
        polarities=polarities,
        c=c,
        R_t=torch.from_numpy(radii),
        R_wp=None,
        f_net=None,
        lap_c=None,
        grid=None,
        N=source_pos.shape[0],
        targets_by_frame={},
        X_source_t=torch.from_numpy(source_pos),
        loss_fn=squared_loss,
        torch_device=torch.device("cpu"),
        epoch_trajectory=[],
    )
    assert loss_shape.item() == 0.0


def test_run_epoch_frame_zero_supervises_post_step_state():
    """Frame 0 should match the first evolved state, not the initial source snapshot."""
    source_pos, polarities, c, radii = _minimal_inputs()

    class FixedStepModel(torch.nn.Module):
        def forward(self, node_feats, edge_index, edge_feats):
            del edge_index, edge_feats
            num_nodes = node_feats.shape[0]
            dX = torch.full((num_nodes, 3), 0.25, dtype=node_feats.dtype, device=node_feats.device)
            dP = torch.zeros((num_nodes, 3), dtype=node_feats.dtype, device=node_feats.device)
            dc = torch.zeros(
                (num_nodes, c.shape[1]), dtype=node_feats.dtype, device=node_feats.device
            )
            return {"dX": dX, "dP": dP, "dc": dc}

    config = torch_train_module.TrainConfig(
        n_epochs=1,
        t_rollout=1,
        mech_steps=0,
        diff_steps=0,
        dt_gns=1.0,
        log_every=1,
    )

    target_frame0 = torch.from_numpy(source_pos + 0.25)
    loss_shape, _loss_l2, trajectory = torch_train_module._run_epoch(
        model=FixedStepModel(),
        config=config,
        source_pos=source_pos,
        polarities=polarities,
        c=c,
        R_t=torch.from_numpy(radii),
        R_wp=None,
        f_net=None,
        lap_c=None,
        grid=None,
        N=source_pos.shape[0],
        targets_by_frame={0: target_frame0},
        X_source_t=torch.from_numpy(source_pos),
        loss_fn=squared_loss,
        torch_device=torch.device("cpu"),
        epoch_trajectory=[],
    )

    assert loss_shape.item() == pytest.approx(0.0)
    assert len(trajectory) == 1
    np.testing.assert_allclose(trajectory[0]["pos"], target_frame0.numpy(), atol=1e-6)


def test_train_rejects_frame_out_of_range():
    source_pos, polarities, c, radii = _minimal_inputs()
    model = _make_model(c.shape[1])
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    config = torch_train_module.TrainConfig(n_epochs=1, t_rollout=2, log_every=1)

    with pytest.raises(ValueError, match="outside"):
        torch_train_module.train(
            model,
            opt,
            squared_loss,
            source_pos=source_pos,
            targets=[(5, source_pos)],
            polarities=polarities,
            c=c,
            radii=radii,
            config=config,
            device="cpu",
        )


def test_train_rejects_missing_targets():
    source_pos, polarities, c, radii = _minimal_inputs()
    model = _make_model(c.shape[1])
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    config = torch_train_module.TrainConfig(n_epochs=1, t_rollout=1, log_every=1)

    with pytest.raises(ValueError, match="requires `targets`"):
        torch_train_module.train(
            model,
            opt,
            squared_loss,
            source_pos=source_pos,
            polarities=polarities,
            c=c,
            radii=radii,
            config=config,
            device="cpu",
        )
