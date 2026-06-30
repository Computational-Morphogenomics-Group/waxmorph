"""Tests for JAX loss functions with gradient flow through GNS."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from waxmorph.jax.gnn import GNS
from waxmorph.jax.losses import chamfer_distance, make_sinkhorn_loss, squared_loss


@pytest.fixture()
def key():
    return jax.random.PRNGKey(0)


class TestSquaredLoss:
    def test_zero_when_identical(self, key):
        x = jax.random.normal(key, (20, 3))
        assert float(squared_loss(x, x)) == pytest.approx(0.0, abs=1e-6)

    def test_positive(self):
        a = jnp.zeros((5, 3))
        b = jnp.ones((5, 3))
        loss = squared_loss(a, b)
        # ||0 - 1||_F^2 = 5 * 3 = 15
        assert float(loss) == pytest.approx(15.0, abs=1e-5)

    def test_gradient_to_pred(self, key):
        pred = jax.random.normal(key, (10, 3))
        target = jax.random.normal(jax.random.PRNGKey(1), (10, 3))
        grad = jax.grad(lambda p: squared_loss(p, target))(pred)
        assert jnp.abs(grad).sum() > 0

    def test_gradient_through_gns(self, key):
        """Squared loss gradients flow through GNS to all parameters."""
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
            key=key,
        )
        k1, k2 = jax.random.split(key)
        node_feat = jax.random.normal(k1, (20, 9))
        edge_idx = jax.random.randint(k1, (2, 60), 0, 20)
        edge_feat = jax.random.normal(k2, (60, 7))

        X_init = jax.random.normal(k1, (20, 3))
        X_target = jax.random.normal(k2, (20, 3))

        @eqx.filter_grad
        def grad_fn(model):
            out = model(node_feat, edge_idx, edge_feat)
            X_pred = X_init + out["dX"]
            return squared_loss(X_pred, X_target)

        grads = grad_fn(gns)
        leaves = jax.tree_util.tree_leaves(grads)
        array_leaves = [g for g in leaves if isinstance(g, jax.Array)]
        assert all(jnp.any(g != 0) for g in array_leaves)


class TestChamferDistance:
    def test_zero_when_identical(self, key):
        x = jax.random.normal(key, (20, 3))
        assert float(chamfer_distance(x, x)) == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self, key):
        a = jax.random.normal(key, (15, 3))
        b = jax.random.normal(jax.random.PRNGKey(1), (15, 3))
        assert float(chamfer_distance(a, b)) == pytest.approx(
            float(chamfer_distance(b, a)), abs=1e-5
        )

    def test_different_counts(self, key):
        """Chamfer distance works with different N and M."""
        a = jax.random.normal(key, (10, 3))
        b = jax.random.normal(jax.random.PRNGKey(1), (25, 3))
        loss = chamfer_distance(a, b)
        assert float(loss) > 0

    def test_gradient_to_pred(self, key):
        pred = jax.random.normal(key, (10, 3))
        target = jax.random.normal(jax.random.PRNGKey(1), (10, 3))
        grad = jax.grad(lambda p: chamfer_distance(p, target))(pred)
        assert jnp.abs(grad).sum() > 0

    def test_gradient_through_gns(self, key):
        """Chamfer loss gradients flow through GNS to all parameters."""
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
            key=key,
        )
        k1, k2 = jax.random.split(key)
        node_feat = jax.random.normal(k1, (20, 9))
        edge_idx = jax.random.randint(k1, (2, 60), 0, 20)
        edge_feat = jax.random.normal(k2, (60, 7))

        X_init = jax.random.normal(k1, (20, 3))
        X_target = jax.random.normal(k2, (20, 3))

        @eqx.filter_grad
        def grad_fn(model):
            out = model(node_feat, edge_idx, edge_feat)
            X_pred = X_init + out["dX"]
            return chamfer_distance(X_pred, X_target)

        grads = grad_fn(gns)
        leaves = jax.tree_util.tree_leaves(grads)
        array_leaves = [g for g in leaves if isinstance(g, jax.Array)]
        assert all(jnp.any(g != 0) for g in array_leaves)

    def test_known_value(self):
        """Two point clouds with a known Chamfer distance."""
        pred = jnp.array([[0.0, 0.0, 0.0]])
        target = jnp.array([[3.0, 4.0, 0.0]])
        # dist = 5.0, chamfer = (5 + 5) / 1 = 10.0
        assert float(chamfer_distance(pred, target)) == pytest.approx(10.0, abs=1e-5)


class TestMakeSinkhornLoss:
    def test_forward(self, key):
        loss_fn = make_sinkhorn_loss(blur=0.05)
        a = jax.random.normal(key, (50, 3))
        b = jax.random.normal(jax.random.PRNGKey(1), (50, 3))
        val = loss_fn(a, b)
        assert val.ndim == 0

    def test_gradient_flow(self, key):
        loss_fn = make_sinkhorn_loss(blur=0.05)
        pred = jax.random.normal(key, (30, 3))
        target = jax.random.normal(jax.random.PRNGKey(1), (30, 3))
        grad = jax.grad(lambda p: loss_fn(p, target))(pred)
        assert jnp.abs(grad).sum() > 0

    def test_near_zero_identical(self, key):
        """Debiased Sinkhorn divergence should be ~0 for identical point clouds."""
        loss_fn = make_sinkhorn_loss(blur=0.05)
        x = jax.random.normal(key, (50, 3))
        val = loss_fn(x, x)
        assert float(val) == pytest.approx(0.0, abs=1e-3)

    def test_p2_matches_default(self, key):
        """p=2 selects SqEuclidean, the OTT default, so it reproduces the default."""
        a = jax.random.normal(key, (40, 3))
        b = jax.random.normal(jax.random.PRNGKey(1), (40, 3))
        default = float(make_sinkhorn_loss(blur=0.05)(a, b))
        p2 = float(make_sinkhorn_loss(blur=0.05, p=2)(a, b))
        assert p2 == pytest.approx(default, rel=1e-6, abs=1e-6)

    def test_p1_euclidean_forward(self, key):
        """p=1 selects Euclidean ground cost and still returns a differentiable scalar."""
        loss_fn = make_sinkhorn_loss(blur=0.05, p=1)
        a = jax.random.normal(key, (40, 3))
        b = jax.random.normal(jax.random.PRNGKey(1), (40, 3))
        assert loss_fn(a, b).ndim == 0
        grad = jax.grad(lambda x: loss_fn(x, b))(a)
        assert jnp.abs(grad).sum() > 0

    def test_explicit_cost_fn_overrides_p(self, key):
        """An explicit OTT cost_fn is accepted and used in place of the p mapping."""
        from ott.geometry import costs

        loss_fn = make_sinkhorn_loss(blur=0.05, cost_fn=costs.Euclidean())
        a = jax.random.normal(key, (40, 3))
        b = jax.random.normal(jax.random.PRNGKey(1), (40, 3))
        assert loss_fn(a, b).ndim == 0

    def test_invalid_p_raises(self):
        """An unsupported p with no explicit cost_fn fails fast at factory time."""
        with pytest.raises(ValueError, match="p in"):
            make_sinkhorn_loss(p=3)

    def test_solve_kwargs_forwarded(self, key):
        """Extra keyword args are forwarded to the OTT Sinkhorn solver."""
        loss_fn = make_sinkhorn_loss(blur=0.05, threshold=1e-2, max_iterations=50)
        a = jax.random.normal(key, (30, 3))
        b = jax.random.normal(jax.random.PRNGKey(1), (30, 3))
        assert loss_fn(a, b).ndim == 0
