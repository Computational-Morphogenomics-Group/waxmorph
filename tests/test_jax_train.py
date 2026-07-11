"""JAX non-growing trainer parity tests."""

import importlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from waxmorph.jax.gnn import GNS
from waxmorph.jax.losses import squared_loss

jax_train_module = importlib.import_module("waxmorph.jax.train")


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


class ZeroStepModel:
    def __call__(self, node_feats, edge_index, edge_feats, num_edges=None):
        del edge_index, edge_feats, num_edges
        num_nodes, num_molecules = node_feats.shape
        return {
            "dX": jnp.zeros((num_nodes, 3), dtype=node_feats.dtype),
            "dP": jnp.zeros((num_nodes, 3), dtype=node_feats.dtype),
            "dc": jnp.zeros((num_nodes, num_molecules), dtype=node_feats.dtype),
        }


def _make_model(num_molecules):
    return GNS(
        node_feature_dim=num_molecules,
        edge_feature_dim=2,
        num_mp_steps=1,
        output_dims={"dX": 3, "dP": 3, "dc": num_molecules},
        key=jax.random.PRNGKey(0),
    )


class _ScalarStepModel(eqx.Module):
    weight: jax.Array

    def __init__(self, value=1.0):
        self.weight = jnp.asarray(value, dtype=jnp.float32)

    def __call__(self, node_feats, edge_index, edge_feats, num_edges=None):
        del edge_index, edge_feats, num_edges
        num_nodes = node_feats.shape[0]
        return {
            "dX": jnp.broadcast_to(self.weight, (num_nodes, 3)),
            "dP": jnp.zeros((num_nodes, 3), dtype=node_feats.dtype),
            "dc": jnp.zeros_like(node_feats),
        }


def _run_batched_epoch(model, config, source_pos, polarities, c, radii, targets_by_frame):
    jax_device = jax.devices("cpu")[0]
    with jax.default_device(jax_device):
        X = jnp.asarray(source_pos)
        P = jnp.asarray(polarities)
        concentrations = jnp.asarray(c)
        R = jnp.asarray(radii)
        targets = {
            frame: jax.device_put(target, jax_device) for frame, target in targets_by_frame.items()
        }
        topologies, trajectory = jax_train_module._collect_topologies_and_trajectory(
            model=model,
            config=config,
            X=X,
            P=P,
            c=concentrations,
            R=R,
            particle_count=len(source_pos),
            device="cpu",
        )
        topology_batch = jax_train_module._stack_topologies(
            topologies,
            config=config,
            max_pairs=jax_train_module._max_edges_for_config(config, len(source_pos)),
        )
        loss_shape, loss_l2 = jax_train_module._epoch_loss_with_topology_batch(
            model=model,
            config=config,
            X_source=X,
            P_source=P,
            c_source=concentrations,
            R=R,
            particle_count=len(source_pos),
            targets_by_frame=targets,
            topology_batch=topology_batch,
            loss_fn=squared_loss,
            device="cpu",
        )
    return loss_shape, loss_l2, trajectory


class TestTreeGlobalNorm:
    """Overflow-safe global gradient norm; parity with the torch float64 clip path."""

    def test_matches_naive_on_normal_values(self):
        tree = {
            "a": jnp.array([3.0, 4.0], dtype=jnp.float32),
            "b": jnp.array([[0.0, 12.0]], dtype=jnp.float32),
        }
        # sqrt(3^2 + 4^2 + 12^2) = 13
        assert float(jax_train_module._tree_global_norm(tree)) == pytest.approx(13.0, rel=1e-6)

    def test_overflow_safe_on_huge_values(self):
        """A naive float32 sum-of-squares overflows to inf; the rescaled norm stays finite."""
        big = jnp.full((4,), 1e20, dtype=jnp.float32)
        naive = float(jnp.sqrt(jnp.sum(jnp.square(big))))
        assert not np.isfinite(naive)  # confirms the hazard the rescaling guards against
        got = float(jax_train_module._tree_global_norm({"g": big}))
        assert np.isfinite(got)
        assert got == pytest.approx(2e20, rel=1e-5)  # ||[1e20]*4|| = 2e20

    def test_zero_tree_is_zero(self):
        assert float(jax_train_module._tree_global_norm({"a": jnp.zeros((3,), jnp.float32)})) == 0.0

    def test_empty_tree_is_zero(self):
        assert float(jax_train_module._tree_global_norm({})) == 0.0


def test_batched_epoch_accumulates_loss_across_tagged_frames():
    source_pos, polarities, c, radii = _minimal_inputs()
    config = jax_train_module.TrainConfig(
        n_epochs=1,
        t_rollout=3,
        mech_steps=0,
        diff_steps=0,
        dt_gns=0.0,
        log_every=1,
    )
    target_a = jnp.asarray(source_pos + 0.1)
    target_b = jnp.asarray(source_pos + 0.3)

    loss_shape, _loss_l2, trajectory = _run_batched_epoch(
        ZeroStepModel(),
        config,
        source_pos,
        polarities,
        c,
        radii,
        {0: target_a, 2: target_b},
    )

    expected = squared_loss(jnp.asarray(source_pos), target_a) + squared_loss(
        jnp.asarray(source_pos),
        target_b,
    )
    assert float(loss_shape) == pytest.approx(float(expected), abs=1e-6)
    assert len(trajectory) == config.t_rollout + 1


def test_batched_epoch_frame_zero_supervises_post_step_state():
    source_pos, polarities, c, radii = _minimal_inputs()

    class FixedStepModel:
        def __call__(self, node_feats, edge_index, edge_feats, num_edges=None):
            del edge_index, edge_feats, num_edges
            num_nodes, num_molecules = node_feats.shape
            return {
                "dX": jnp.full((num_nodes, 3), 0.25, dtype=node_feats.dtype),
                "dP": jnp.zeros((num_nodes, 3), dtype=node_feats.dtype),
                "dc": jnp.zeros((num_nodes, num_molecules), dtype=node_feats.dtype),
            }

    config = jax_train_module.TrainConfig(
        n_epochs=1,
        t_rollout=1,
        mech_steps=0,
        diff_steps=0,
        dt_gns=1.0,
        log_every=1,
    )
    target_frame0 = jnp.asarray(source_pos + 0.25)

    loss_shape, _loss_l2, trajectory = _run_batched_epoch(
        FixedStepModel(),
        config,
        source_pos,
        polarities,
        c,
        radii,
        {0: target_frame0},
    )

    assert float(loss_shape) == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(trajectory[1]["pos"], np.asarray(target_frame0), atol=1e-6)


def test_train_uses_max_edges_factor_for_padded_topology_capacity():
    source_pos, polarities, c, radii = _minimal_inputs()
    config = jax_train_module.TrainConfig(
        n_epochs=1,
        t_rollout=1,
        mech_steps=0,
        diff_steps=0,
        dt_gns=0.0,
        grad_clip_norm=None,
        log_every=1,
        max_edges_factor=1,
    )
    optimizer = optax.sgd(learning_rate=0.0)
    model = _make_model(c.shape[1])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    with pytest.raises(ValueError, match="max_edges"):
        jax_train_module.train(
            model,
            optimizer,
            opt_state,
            squared_loss,
            source_pos=source_pos,
            targets=[(0, source_pos)],
            polarities=polarities,
            c=c,
            radii=radii,
            config=config,
            device="cpu",
        )


def test_stack_topologies_pads_unpadded_edges_and_pairs_to_bucketed_capacity():
    config = jax_train_module.TrainConfig(
        t_rollout=2,
        mech_steps=1,
        diff_steps=2,
        max_edges_factor=8,
    )
    pair_a = jax_train_module._PairTopology(
        pair_i=jnp.array([0], dtype=jnp.int32),
        pair_j=jnp.array([1], dtype=jnp.int32),
        num_pairs=jnp.array(1, dtype=jnp.int32),
    )
    pair_b = jax_train_module._PairTopology(
        pair_i=jnp.array([0, 1], dtype=jnp.int32),
        pair_j=jnp.array([1, 2], dtype=jnp.int32),
        num_pairs=jnp.array(2, dtype=jnp.int32),
    )
    topologies = (
        jax_train_module._StepTopology(
            edge_index=jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
            num_edges=jnp.array(2, dtype=jnp.int32),
            mech_pairs=(pair_a,),
            diff_pairs=(pair_b, pair_b),
        ),
        jax_train_module._StepTopology(
            edge_index=jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32),
            num_edges=jnp.array(3, dtype=jnp.int32),
            mech_pairs=(pair_b,),
            diff_pairs=(pair_a, pair_a),
        ),
    )

    batch = jax_train_module._stack_topologies(topologies, config=config, max_pairs=8)

    assert batch.edge_index.shape == (2, 2, 8)
    assert batch.mech_pair_i.shape == (2, 1, 8)
    assert batch.diff_pair_i.shape == (2, 2, 8)
    np.testing.assert_array_equal(np.asarray(batch.num_edges), np.array([2, 3]))
    np.testing.assert_array_equal(np.asarray(batch.mech_num_pairs), np.array([[1], [2]]))
    np.testing.assert_array_equal(
        np.asarray(batch.edge_index[0, :, :2]), np.array([[0, 1], [1, 0]])
    )
    np.testing.assert_array_equal(np.asarray(batch.edge_index[0, :, 2:]), np.zeros((2, 6)))
    np.testing.assert_array_equal(np.asarray(batch.mech_pair_i[0, 0, 1:]), np.zeros(7))


def test_stack_topologies_rejects_counts_over_capacity():
    config = jax_train_module.TrainConfig(t_rollout=1, mech_steps=1, diff_steps=0)
    pairs = jax_train_module._PairTopology(
        pair_i=jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        pair_j=jnp.array([1, 2, 3, 4], dtype=jnp.int32),
        num_pairs=jnp.array(4, dtype=jnp.int32),
    )
    topologies = (
        jax_train_module._StepTopology(
            edge_index=jnp.array([[0], [1]], dtype=jnp.int32),
            num_edges=jnp.array(1, dtype=jnp.int32),
            mech_pairs=(pairs,),
            diff_pairs=(),
        ),
    )

    with pytest.raises(ValueError, match="Neighbor pair list"):
        jax_train_module._stack_topologies(topologies, config=config, max_pairs=3)


def _has_jax_warp_cuda() -> bool:
    try:
        import warp as wp

        return bool(wp.is_device_available("cuda") and jax.devices("gpu"))
    except Exception:
        return False


JAX_TRAIN_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not _has_jax_warp_cuda(), reason="JAX CUDA required"),
    ),
]


@pytest.mark.parametrize("device", JAX_TRAIN_DEVICES)
@pytest.mark.parametrize("n_epochs", [1, 3])
def test_train_logs_post_update_model_rollouts(monkeypatch, device, n_epochs):
    source_pos, polarities, c, radii = _minimal_inputs()
    model = _ScalarStepModel()
    optimizer = optax.adamw(0.5, b1=0.8, b2=0.9, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    evaluations = []
    train_steps = []
    collect_topologies = jax_train_module._collect_topologies_and_trajectory
    make_train_step = jax_train_module._make_train_step

    def capture_collection(*, model, **kwargs):
        evaluations.append((float(jax.device_get(model.weight)), model.weight.device.platform))
        return collect_topologies(model=model, **kwargs)

    def capture_train_step(**kwargs):
        train_step = make_train_step(**kwargs)

        def counted_train_step(model, *args):
            updated = train_step(model, *args)
            train_steps.append(
                (
                    float(jax.device_get(model.weight)),
                    float(jax.device_get(updated[0].weight)),
                )
            )
            return updated

        return counted_train_step

    monkeypatch.setattr(
        jax_train_module,
        "_collect_topologies_and_trajectory",
        capture_collection,
    )
    monkeypatch.setattr(jax_train_module, "_make_train_step", capture_train_step)
    config = jax_train_module.TrainConfig(
        n_epochs=n_epochs,
        t_rollout=1,
        mech_steps=0,
        diff_steps=0,
        dt_gns=1.0,
        lambda_reg=0.0,
        grad_clip_norm=None,
        log_every=n_epochs + 1,
    )

    result = jax_train_module.train(
        model,
        optimizer,
        opt_state,
        squared_loss,
        source_pos=source_pos,
        polarities=polarities,
        c=c,
        radii=radii,
        targets=[(0, source_pos)],
        config=config,
        device=device,
    )

    weights = np.array([weight for weight, _platform in evaluations])
    expected_losses = source_pos.size * np.square(weights[1:])
    expected_platform = "gpu" if device == "cuda" else "cpu"
    assert len(evaluations) == n_epochs + 1
    assert len(train_steps) == n_epochs
    assert {platform for _weight, platform in evaluations} == {expected_platform}
    assert result.model.weight.device.platform == expected_platform
    assert all(before != after for before, after in train_steps)
    for name in ("losses_total", "losses_shape", "losses_l2"):
        np.testing.assert_allclose(result.log[name], expected_losses, rtol=1e-5, atol=1e-6)
    best_epoch = int(np.argmin(expected_losses))
    assert result.log["best_epoch"] == best_epoch
    assert result.log["best_loss"] == pytest.approx(expected_losses[best_epoch], rel=1e-5)
    assert float(jax.device_get(result.model.weight)) == pytest.approx(
        weights[best_epoch + 1], rel=1e-5
    )
    np.testing.assert_allclose(
        result.log["best_traj_pos"][-1],
        source_pos + weights[best_epoch + 1],
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.log["losses_total"][-1] == pytest.approx(expected_losses[-1], rel=1e-5)
    expected_keys = [
        "losses_total",
        "losses_shape",
        "losses_l2",
        "best_traj_pos",
        "best_traj_pol",
        "best_traj_c",
        "best_epoch",
        "best_loss",
        "target_frames",
        *(f"config_{name}" for name in config.__dataclass_fields__),
    ]
    assert list(result.log) == expected_keys
    assert all(result.log[name].shape == (n_epochs,) for name in expected_keys[:3])
    assert all(result.log[name].dtype == np.float64 for name in expected_keys[:3])
    assert result.log["best_traj_pos"].shape == (2, len(source_pos), 3)
    assert result.log["best_traj_pol"].shape == (2, len(source_pos), 3)
    assert result.log["best_traj_c"].shape == (2, len(source_pos), c.shape[1])
    assert result.log["target_frames"].dtype == np.int64


def test_train_rejects_nonfinite_post_update_loss():
    source_pos, polarities, c, radii = _minimal_inputs()
    model = _ScalarStepModel()
    optimizer = optax.adamw(0.5, b1=0.8, b2=0.9, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def finite_then_nan_loss(prediction, target):
        return jnp.where(
            prediction[0, 0] < 0.75,
            jnp.asarray(jnp.nan, dtype=prediction.dtype),
            squared_loss(prediction, target),
        )

    with pytest.raises(ValueError, match="post-update evaluation"):
        jax_train_module.train(
            model,
            optimizer,
            opt_state,
            finite_then_nan_loss,
            source_pos=source_pos,
            polarities=polarities,
            c=c,
            radii=radii,
            targets=[(0, source_pos)],
            config=jax_train_module.TrainConfig(
                n_epochs=1,
                t_rollout=1,
                mech_steps=0,
                diff_steps=0,
                dt_gns=1.0,
                lambda_reg=0.0,
                grad_clip_norm=None,
                log_every=2,
            ),
            device="cpu",
        )


@pytest.mark.skipif(not _has_jax_warp_cuda(), reason="JAX/Warp bridge requires CUDA")
def test_native_warp_collection_physics_matches_jax_bridge():
    from waxmorph.jax.warp_autograd import warp_diffusion_step, warp_mech_step

    with jax.default_device(jax.devices("gpu")[0]):
        x = jnp.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=jnp.float32)
        r = jnp.array([0.5, 0.5], dtype=jnp.float32)
        c = jnp.array([[1.0, 0.0], [0.0, 2.0]], dtype=jnp.float32)

    ctx = jax_train_module._make_warp_collection_context(
        r,
        c,
        particle_count=2,
        max_pairs=8,
        device="cuda",
    )
    x_wp = jax_train_module._jax_positions_to_warp(x)
    pairs, pair_i_wp, pair_j_wp, num_pairs = jax_train_module._build_warp_collection_pairs(
        x_wp,
        ctx,
    )

    native_x = jax_train_module._warp_to_jax(
        jax_train_module._native_warp_mech_step(
            x_wp,
            pair_i_wp,
            pair_j_wp,
            num_pairs,
            1e-2,
            ctx,
        ),
        x.dtype,
    )
    bridge_x = warp_mech_step(
        x,
        r,
        pairs.pair_i,
        pairs.pair_j,
        1e-2,
        num_pairs=pairs.num_pairs,
        device="cuda",
    )

    c_wp = jax_train_module._jax_c_to_warp(c)
    native_c = jax_train_module._warp_to_jax(
        jax_train_module._native_warp_diffusion_step(
            c_wp,
            pair_i_wp,
            pair_j_wp,
            num_pairs,
            0.1,
            1e-2,
            ctx,
        ),
        c.dtype,
    )
    bridge_c = warp_diffusion_step(
        c,
        pairs.pair_i,
        pairs.pair_j,
        0.1,
        1e-2,
        num_pairs=pairs.num_pairs,
        device="cuda",
    )

    assert num_pairs == 1
    np.testing.assert_allclose(np.asarray(native_x), np.asarray(bridge_x), atol=1e-6)
    np.testing.assert_allclose(np.asarray(native_c), np.asarray(bridge_c), atol=1e-6)


@pytest.mark.skipif(not _has_jax_warp_cuda(), reason="JAX/Warp bridge requires CUDA")
def test_native_collection_reuses_diffusion_pairs_and_returns_finite_trajectory():
    source_pos, polarities, c, radii = _minimal_inputs()
    config = jax_train_module.TrainConfig(
        n_epochs=1,
        t_rollout=2,
        mech_steps=1,
        diff_steps=2,
        dt_gns=0.0,
        log_every=1,
    )
    with jax.default_device(jax.devices("gpu")[0]):
        X = jnp.asarray(source_pos)
        P = jnp.asarray(polarities)
        C = jnp.asarray(c)
        R = jnp.asarray(radii)

    topologies, trajectory = jax_train_module._collect_topologies_and_trajectory(
        model=ZeroStepModel(),
        config=config,
        X=X,
        P=P,
        c=C,
        R=R,
        particle_count=source_pos.shape[0],
        device="cuda",
    )

    assert len(topologies) == config.t_rollout
    assert len(trajectory) == config.t_rollout + 1
    for frame in trajectory:
        assert np.isfinite(frame["pos"]).all()
        assert np.isfinite(frame["c"]).all()
    for topology in topologies:
        assert len(topology.diff_pairs) == config.diff_steps
        np.testing.assert_array_equal(
            np.asarray(topology.diff_pairs[0].pair_i),
            np.asarray(topology.diff_pairs[1].pair_i),
        )
        np.testing.assert_array_equal(
            np.asarray(topology.diff_pairs[0].pair_j),
            np.asarray(topology.diff_pairs[1].pair_j),
        )


def test_train_rejects_frame_out_of_range():
    source_pos, polarities, c, radii = _minimal_inputs()
    model = _make_model(c.shape[1])
    optimizer = optax.sgd(learning_rate=0.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    config = jax_train_module.TrainConfig(n_epochs=1, t_rollout=2, mech_steps=0, diff_steps=0)

    with pytest.raises(ValueError, match="outside"):
        jax_train_module.train(
            model,
            optimizer,
            opt_state,
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
    optimizer = optax.sgd(learning_rate=0.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    config = jax_train_module.TrainConfig(n_epochs=1, t_rollout=1, mech_steps=0, diff_steps=0)

    with pytest.raises(ValueError, match="requires `targets`"):
        jax_train_module.train(
            model,
            optimizer,
            opt_state,
            squared_loss,
            source_pos=source_pos,
            polarities=polarities,
            c=c,
            radii=radii,
            config=config,
            device="cpu",
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
    return GNS(**kwargs, key=jax.random.PRNGKey(0))


def _train_from_checkpoint(model, optimizer, opt_state, save_path, device="cpu"):
    source_pos, polarities, c, radii = _minimal_inputs()
    return jax_train_module.train(
        model,
        optimizer,
        opt_state,
        squared_loss,
        source_pos=source_pos,
        polarities=polarities,
        c=c,
        radii=radii,
        targets=[(0, source_pos + 0.1)],
        config=jax_train_module.TrainConfig(
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


@pytest.mark.parametrize("device", JAX_TRAIN_DEVICES)
def test_train_reinitializes_optimizer_from_compatible_checkpoint(tmp_path, monkeypatch, device):
    checkpoint_model = jax.tree.map(
        lambda value: jnp.full_like(value, 0.05) if eqx.is_array(value) else value,
        _checkpoint_model(),
    )
    checkpoint_arrays = [
        np.asarray(jax.device_get(value)).copy()
        for value in jax_train_module._array_leaves(checkpoint_model)
    ]
    checkpoint_path = tmp_path / "model.eqx"
    checkpoint_model.save(checkpoint_path)
    config_path = tmp_path / "model.eqx.json"
    log_path = tmp_path / "model.eqx.log.npz"
    log_path.write_bytes(b"existing log")
    saved_bytes = {path: path.read_bytes() for path in (checkpoint_path, config_path, log_path)}

    init_calls = []
    init_state_platforms = []

    def init_optimizer(params):
        init_calls.append(params)
        count = jnp.array(0, dtype=jnp.int32)
        init_state_platforms.append(count.device.platform)
        return {"count": count}

    def unused_update(*_args, **_kwargs):
        raise AssertionError("fake train step bypasses optimizer.update")

    optimizer = optax.GradientTransformation(init_optimizer, unused_update)
    stale_state = {"count": jnp.array(99, dtype=jnp.int32)}
    source_pos, polarities, c, _ = _minimal_inputs()
    collected_models = []
    evaluated_models = []
    step_calls = []

    def collect_topologies(*, model, **_kwargs):
        collected_models.append(model)
        trajectory = [
            {"pos": source_pos.copy(), "pol": polarities.copy(), "c": c.copy()},
            {"pos": source_pos.copy(), "pol": polarities.copy(), "c": c.copy()},
        ]
        return (object(),), trajectory

    def make_train_step(**_kwargs):
        def train_step(model, opt_state, *_args):
            updated_model = jax.tree.map(
                lambda value: value + 1e-3 if eqx.is_array(value) else value,
                model,
            )
            step_calls.append((model, opt_state, updated_model))
            first_leaf = jax_train_module._array_leaves(model)[0]

            def scalar(value, dtype):
                return jax.device_put(jnp.array(value, dtype=dtype), first_leaf.device)

            return (
                updated_model,
                {"count": scalar(1, jnp.int32)},
                scalar(1.0, jnp.float32),
                scalar(1.0, jnp.float32),
                scalar(0.0, jnp.float32),
                scalar(1.0, jnp.float32),
                scalar(True, jnp.bool_),
                scalar(True, jnp.bool_),
            )

        return train_step

    def evaluate_batch(*, model, **_kwargs):
        evaluated_models.append(model)
        first_leaf = jax_train_module._array_leaves(model)[0]
        return (
            jax.device_put(jnp.array(1.0, dtype=jnp.float32), first_leaf.device),
            jax.device_put(jnp.array(0.0, dtype=jnp.float32), first_leaf.device),
        )

    monkeypatch.setattr(jax_train_module, "_collect_topologies_and_trajectory", collect_topologies)
    monkeypatch.setattr(jax_train_module, "_stack_topologies", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(jax_train_module, "_make_train_step", make_train_step)
    monkeypatch.setattr(jax_train_module, "_epoch_loss_with_topology_batch", evaluate_batch)

    result = _train_from_checkpoint(
        _checkpoint_model(), optimizer, stale_state, checkpoint_path, device=device
    )

    assert len(init_calls) == 1
    assert len(collected_models) == 2
    assert len(step_calls) == len(evaluated_models) == 1
    assert collected_models[1] is evaluated_models[0] is result.model
    expected_platform = "gpu" if device == "cuda" else "cpu"
    assert init_state_platforms == [expected_platform]
    for collected_model in collected_models:
        assert {
            value.device.platform for value in jax_train_module._array_leaves(collected_model)
        } == {expected_platform}
    assert {value.device.platform for value in jax_train_module._array_leaves(init_calls[0])} == {
        expected_platform
    }
    for actual, expected in zip(
        jax_train_module._array_leaves(collected_models[0]), checkpoint_arrays, strict=True
    ):
        np.testing.assert_array_equal(np.asarray(jax.device_get(actual)), expected)
    for actual, expected in zip(
        jax_train_module._array_leaves(init_calls[0]), checkpoint_arrays, strict=True
    ):
        np.testing.assert_array_equal(np.asarray(jax.device_get(actual)), expected)
    assert int(jax.device_get(step_calls[0][1]["count"])) == 0
    assert step_calls[0][1]["count"].device.platform == expected_platform
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax_train_module._array_leaves(step_calls[0][0]),
            jax_train_module._array_leaves(step_calls[0][2]),
            strict=True,
        )
    )
    for actual, expected in zip(
        jax_train_module._array_leaves(result.model),
        jax_train_module._array_leaves(step_calls[0][2]),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(actual)), np.asarray(jax.device_get(expected))
        )
    assert result.log["config_n_epochs"] == 1
    assert all(
        result.log[name].shape == (1,) for name in ("losses_total", "losses_shape", "losses_l2")
    )
    assert {path: path.read_bytes() for path in saved_bytes} == saved_bytes


@pytest.mark.parametrize(
    ("field", "override"),
    [
        ("node_feature_dim", {"node_feature_dim": 3}),
        ("edge_feature_dim", {"edge_feature_dim": 3}),
        ("node_latent_dim", {"node_latent_dim": 9}),
        ("edge_latent_dim", {"edge_latent_dim": 9}),
        ("hidden_dim", {"hidden_dim": 9}),
        ("num_mp_steps", {"num_mp_steps": 2}),
        ("num_mlp_layers", {"num_mlp_layers": 3}),
        ("output_dims", {"output_dims": {"dX": 3, "dP": 3, "dc": 3}}),
        ("activation", {"activation": "gelu"}),
        ("layer_norm", {"layer_norm": False}),
        ("checkpoint_processor", {"checkpoint_processor": True}),
    ],
)
def test_train_rejects_incompatible_checkpoint_config(tmp_path, monkeypatch, field, override):
    checkpoint_path = tmp_path / "model.eqx"
    _checkpoint_model(**override).save(checkpoint_path)

    def forbidden_init(_params):
        raise AssertionError("optimizer.init must follow compatibility checks")

    optimizer = optax.GradientTransformation(forbidden_init, lambda *_args: None)
    monkeypatch.setattr(
        jax_train_module,
        "_collect_topologies_and_trajectory",
        lambda **_kwargs: pytest.fail("rollout must follow compatibility checks"),
    )

    with pytest.raises(ValueError, match=field):
        _train_from_checkpoint(_checkpoint_model(), optimizer, (), checkpoint_path)


@pytest.mark.parametrize(
    ("kind", "match"),
    [("structure", "PyTree structure"), ("shape", "shape"), ("dtype", "dtype")],
)
def test_train_rejects_incompatible_checkpoint_tree(tmp_path, monkeypatch, kind, match):
    checkpoint_path = tmp_path / "model.eqx"
    _checkpoint_model().save(checkpoint_path)
    model = _checkpoint_model()
    weight = model.node_encoder.net.layers[0].weight
    if kind == "structure":
        model = eqx.tree_at(lambda tree: tree.node_encoder.norm, model, None)
    elif kind == "shape":
        model = eqx.tree_at(
            lambda tree: tree.node_encoder.net.layers[0].weight,
            model,
            weight[:, :-1],
        )
    else:
        model = eqx.tree_at(
            lambda tree: tree.node_encoder.net.layers[0].weight,
            model,
            weight.astype(jnp.float16),
        )
    monkeypatch.setattr(
        jax_train_module,
        "_collect_topologies_and_trajectory",
        lambda **_kwargs: pytest.fail("rollout must follow compatibility checks"),
    )

    with pytest.raises(ValueError, match=match):
        _train_from_checkpoint(model, optax.sgd(0.0), (), checkpoint_path)
