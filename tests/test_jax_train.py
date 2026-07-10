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


def test_run_epoch_accumulates_loss_across_tagged_frames():
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

    loss_shape, _loss_l2, trajectory = jax_train_module._run_epoch(
        model=ZeroStepModel(),
        config=config,
        source_pos=source_pos,
        polarities=polarities,
        c=c,
        R=jnp.asarray(radii),
        particle_count=source_pos.shape[0],
        targets_by_frame={0: target_a, 2: target_b},
        loss_fn=squared_loss,
        device="cpu",
    )

    expected = squared_loss(jnp.asarray(source_pos), target_a) + squared_loss(
        jnp.asarray(source_pos),
        target_b,
    )
    assert float(loss_shape) == pytest.approx(float(expected), abs=1e-6)
    assert len(trajectory) == config.t_rollout + 1


def test_run_epoch_frame_zero_supervises_post_step_state():
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

    loss_shape, _loss_l2, trajectory = jax_train_module._run_epoch(
        model=FixedStepModel(),
        config=config,
        source_pos=source_pos,
        polarities=polarities,
        c=c,
        R=jnp.asarray(radii),
        particle_count=source_pos.shape[0],
        targets_by_frame={0: target_frame0},
        loss_fn=squared_loss,
        device="cpu",
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

    native_x = jax_train_module._warp_positions_to_jax(
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
    native_c = jax_train_module._warp_c_to_jax(
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
