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


def _checkpoint_model(**overrides):
    kwargs = {
        "node_feature_dim": 2,
        "edge_feature_dim": 2,
        "node_latent_dim": 8,
        "edge_latent_dim": 8,
        "hidden_dim": 8,
        "num_mp_steps": 1,
        "num_mlp_layers": 2,
        "output_dims": {"dX": 3, "dP": 3, "dc": 2},
        "activation": "relu",
        "layer_norm": True,
        "checkpoint_processor": False,
    }
    kwargs.update(overrides)
    return GNS(**kwargs)


def _train_from_checkpoint(model, optimizer, save_path, device="cpu"):
    source_pos, polarities, c, radii = _minimal_inputs()
    return torch_train_module.train(
        model,
        optimizer,
        squared_loss,
        source_pos=source_pos,
        polarities=polarities,
        c=c,
        radii=radii,
        targets=[(0, source_pos + 0.1)],
        config=torch_train_module.TrainConfig(
            n_epochs=4,
            t_rollout=1,
            mech_steps=0,
            diff_steps=0,
            lambda_reg=0.0,
            grad_clip_norm=None,
            log_every=10,
        ),
        save_path=save_path,
        device=device,
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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ),
    ],
)
def test_train_loads_checkpoint_into_caller_model_and_resets_optimizer(
    tmp_path, monkeypatch, device
):
    checkpoint_model = _checkpoint_model()
    with torch.no_grad():
        for parameter in checkpoint_model.parameters():
            parameter.fill_(0.05)
    checkpoint_state = {
        name: value.detach().clone() for name, value in checkpoint_model.state_dict().items()
    }
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_model.save(checkpoint_path)
    log_path = tmp_path / "model.pt.log.npz"
    log_path.write_bytes(b"existing log")
    checkpoint_bytes = checkpoint_path.read_bytes()
    log_bytes = log_path.read_bytes()

    model = _checkpoint_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.8, 0.95), weight_decay=0.07)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        state["stale"] = torch.tensor(1.0)

    parameter_ids = [id(parameter) for parameter in model.parameters()]
    group_options = {
        name: value for name, value in optimizer.param_groups[0].items() if name != "params"
    }
    states_at_step = []
    original_step = optimizer.step

    def capture_step(*args, **kwargs):
        state_before = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        state_count = len(optimizer.state)
        result = original_step(*args, **kwargs)
        state_after = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        states_at_step.append((state_count, state_before, state_after))
        return result

    monkeypatch.setattr(optimizer, "step", capture_step)
    result = _train_from_checkpoint(model, optimizer, checkpoint_path, device=device)

    assert result.model is model
    assert [id(parameter) for parameter in result.model.parameters()] == parameter_ids
    assert [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ] == parameter_ids
    assert states_at_step[0][0] == 0
    assert len(states_at_step) == 1
    for name, value in checkpoint_state.items():
        torch.testing.assert_close(states_at_step[0][1][name], value)
    assert any(
        not torch.equal(states_at_step[0][1][name], states_at_step[0][2][name])
        for name in checkpoint_state
    )
    assert all("stale" not in state for state in optimizer.state.values())
    assert all(int(state["step"].item()) == 1 for state in optimizer.state.values())
    assert {
        name: value for name, value in optimizer.param_groups[0].items() if name != "params"
    } == group_options
    assert result.log["config_n_epochs"] == 1
    assert all(
        result.log[name].shape == (1,) for name in ("losses_total", "losses_shape", "losses_l2")
    )
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    assert log_path.read_bytes() == log_bytes


@pytest.mark.parametrize(
    ("field", "override", "device"),
    [
        ("node_feature_dim", {"node_feature_dim": 3}, "cpu"),
        ("edge_feature_dim", {"edge_feature_dim": 3}, "cpu"),
        ("node_latent_dim", {"node_latent_dim": 9}, "cpu"),
        ("edge_latent_dim", {"edge_latent_dim": 9}, "cpu"),
        ("hidden_dim", {"hidden_dim": 9}, "cpu"),
        ("num_mp_steps", {"num_mp_steps": 2}, "cpu"),
        ("num_mlp_layers", {"num_mlp_layers": 3}, "cpu"),
        ("output_dims", {"output_dims": {"dX": 3, "dP": 3, "dc": 3}}, "cpu"),
        ("activation", {"activation": "gelu"}, "cpu"),
        ("layer_norm", {"layer_norm": False}, "cpu"),
        ("checkpoint_processor", {"checkpoint_processor": True}, "cpu"),
        pytest.param(
            "hidden_dim",
            {"hidden_dim": 10},
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ),
    ],
)
def test_train_rejects_incompatible_checkpoint_before_mutation(
    tmp_path, monkeypatch, field, override, device
):
    checkpoint_path = tmp_path / "model.pt"
    _checkpoint_model(**override).save(checkpoint_path)
    model = _checkpoint_model()
    model_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.07, momentum=0.8)
    first_parameter = next(model.parameters())
    sentinel = torch.tensor(2.0)
    optimizer.state[first_parameter]["sentinel"] = sentinel
    group_options = {
        name: value for name, value in optimizer.param_groups[0].items() if name != "params"
    }
    to_calls = []
    original_to = model.to

    def capture_to(*args, **kwargs):
        to_calls.append((args, kwargs))
        return original_to(*args, **kwargs)

    monkeypatch.setattr(model, "to", capture_to)
    with pytest.raises(ValueError, match=field):
        _train_from_checkpoint(model, optimizer, checkpoint_path, device=device)

    assert not to_calls
    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
    assert [id(parameter) for parameter in model.parameters()] == parameter_ids
    for name, value in model_state.items():
        torch.testing.assert_close(model.state_dict()[name], value)
    assert optimizer.state[first_parameter]["sentinel"] is sentinel
    assert {
        name: value for name, value in optimizer.param_groups[0].items() if name != "params"
    } == group_options
