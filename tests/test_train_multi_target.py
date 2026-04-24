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
    genes = np.array(
        [[0.2, 0.1], [0.05, 0.3], [0.4, 0.5]],
        dtype=np.float32,
    )
    radii = np.full((3,), 0.5, dtype=np.float32)
    return source_pos, polarities, genes, radii


def _make_model(num_genes):
    return GNS(
        node_feature_dim=num_genes,
        edge_feature_dim=2,
        num_mp_steps=1,
        output_dims={"dX": 3, "dP": 3, "dG": num_genes},
    )


def test_run_epoch_accumulates_loss_across_tagged_frames():
    """Loss from multiple tagged frames should sum into loss_shape."""
    source_pos, polarities, genes, radii = _minimal_inputs()

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
    model = _make_model(genes.shape[1])

    loss_shape, _loss_l2, trajectory = torch_train_module._run_epoch(
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
    source_pos, polarities, genes, radii = _minimal_inputs()

    config = torch_train_module.TrainConfig(
        n_epochs=1,
        t_rollout=3,
        mech_steps=0,
        diff_steps=0,
        dt_gns=0.0,
        log_every=1,
    )

    torch.manual_seed(0)
    model = _make_model(genes.shape[1])

    loss_shape, *_ = torch_train_module._run_epoch(
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
        targets_by_frame={},
        X_source_t=torch.from_numpy(source_pos),
        loss_fn=squared_loss,
        torch_device=torch.device("cpu"),
        epoch_trajectory=[],
    )
    assert loss_shape.item() == 0.0


def test_run_epoch_frame_zero_supervises_post_step_state():
    """Frame 0 should match the first evolved state, not the initial source snapshot."""
    source_pos, polarities, genes, radii = _minimal_inputs()

    class FixedStepModel(torch.nn.Module):
        def forward(self, node_feats, edge_index, edge_feats):
            del edge_index, edge_feats
            num_nodes = node_feats.shape[0]
            dX = torch.full((num_nodes, 3), 0.25, dtype=node_feats.dtype, device=node_feats.device)
            dP = torch.zeros((num_nodes, 3), dtype=node_feats.dtype, device=node_feats.device)
            dG = torch.zeros(
                (num_nodes, genes.shape[1]), dtype=node_feats.dtype, device=node_feats.device
            )
            return {"dX": dX, "dP": dP, "dG": dG}

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
        genes=genes,
        R_t=torch.from_numpy(radii),
        R_wp=None,
        gx=None,
        lap_G=None,
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


def test_train_back_compat_wraps_target_pos_into_final_frame():
    """train(..., target_pos=...) should behave like train(..., targets=[(t_rollout-1, ...)])."""
    source_pos, polarities, genes, radii = _minimal_inputs()
    target_pos = source_pos + 0.1

    config = torch_train_module.TrainConfig(
        n_epochs=2,
        t_rollout=2,
        mech_steps=0,
        diff_steps=0,
        dt_gns=0.0,
        log_every=1,
        grad_clip_norm=None,
    )

    torch.manual_seed(0)
    model_a = _make_model(genes.shape[1])
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.0)
    result_legacy = torch_train_module.train(
        model_a,
        opt_a,
        squared_loss,
        source_pos=source_pos,
        target_pos=target_pos,
        polarities=polarities,
        genes=genes,
        radii=radii,
        config=config,
        device="cpu",
    )

    torch.manual_seed(0)
    model_b = _make_model(genes.shape[1])
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.0)
    result_new = torch_train_module.train(
        model_b,
        opt_b,
        squared_loss,
        source_pos=source_pos,
        targets=[(config.t_rollout - 1, target_pos)],
        polarities=polarities,
        genes=genes,
        radii=radii,
        config=config,
        device="cpu",
    )

    np.testing.assert_allclose(
        result_legacy.log["losses_shape"], result_new.log["losses_shape"], atol=1e-6
    )
    np.testing.assert_array_equal(
        result_legacy.log["target_frames"], np.array([config.t_rollout - 1])
    )
    np.testing.assert_array_equal(result_new.log["target_frames"], np.array([config.t_rollout - 1]))


def test_train_rejects_both_targets_and_target_pos():
    source_pos, polarities, genes, radii = _minimal_inputs()
    model = _make_model(genes.shape[1])
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    config = torch_train_module.TrainConfig(n_epochs=1, t_rollout=1, log_every=1)

    with pytest.raises(ValueError, match="Pass `targets` OR `target_pos`"):
        torch_train_module.train(
            model,
            opt,
            squared_loss,
            source_pos=source_pos,
            target_pos=source_pos,
            targets=[(0, source_pos)],
            polarities=polarities,
            genes=genes,
            radii=radii,
            config=config,
            device="cpu",
        )


def test_train_rejects_frame_out_of_range():
    source_pos, polarities, genes, radii = _minimal_inputs()
    model = _make_model(genes.shape[1])
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
            genes=genes,
            radii=radii,
            config=config,
            device="cpu",
        )


def test_train_rejects_missing_targets():
    source_pos, polarities, genes, radii = _minimal_inputs()
    model = _make_model(genes.shape[1])
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    config = torch_train_module.TrainConfig(n_epochs=1, t_rollout=1, log_every=1)

    with pytest.raises(ValueError, match="requires either"):
        torch_train_module.train(
            model,
            opt,
            squared_loss,
            source_pos=source_pos,
            polarities=polarities,
            genes=genes,
            radii=radii,
            config=config,
            device="cpu",
        )
