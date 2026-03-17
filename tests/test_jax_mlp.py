"""Tests for the JAX/Equinox MLP building block."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from waxmorph.jax.mlp import MLP


@pytest.fixture()
def key():
    return jax.random.PRNGKey(0)


def test_mlp_output_shape(key):
    mlp = MLP(input_dim=11, output_dim=128, hidden_dim=64, num_layers=3, key=key)
    x = jax.random.normal(key, (50, 11))
    assert mlp(x).shape == (50, 128)


def test_mlp_single_layer(key):
    mlp = MLP(input_dim=5, output_dim=3, num_layers=1, layer_norm=False, key=key)
    x = jax.random.normal(key, (10, 5))
    assert mlp(x).shape == (10, 3)


def test_mlp_no_layer_norm(key):
    mlp = MLP(input_dim=4, output_dim=4, num_layers=2, layer_norm=False, key=key)
    assert mlp.norm is None


def test_mlp_with_layer_norm(key):
    mlp = MLP(input_dim=4, output_dim=8, num_layers=2, layer_norm=True, key=key)
    assert isinstance(mlp.norm, eqx.nn.LayerNorm)


def test_mlp_batched_input(key):
    mlp = MLP(input_dim=7, output_dim=3, hidden_dim=32, num_layers=2, key=key)
    x = jax.random.normal(key, (10, 7))
    assert mlp(x).shape == (10, 3)


def test_mlp_invalid_activation(key):
    with pytest.raises(ValueError, match="Unknown activation"):
        MLP(input_dim=5, output_dim=3, activation="invalid", key=key)


@pytest.mark.parametrize("activation", ["relu", "silu", "gelu", "tanh"])
def test_mlp_activations(activation, key):
    mlp = MLP(input_dim=5, output_dim=3, num_layers=2, activation=activation, key=key)
    x = jax.random.normal(key, (8, 5))
    assert mlp(x).shape == (8, 3)


def test_mlp_gradient_flow(key):
    mlp = MLP(input_dim=5, output_dim=3, num_layers=2, key=key)
    x = jax.random.normal(key, (8, 5))

    @eqx.filter_grad
    def grad_fn(model):
        return jnp.sum(model(x))

    grads = grad_fn(mlp)
    leaves = jax.tree_util.tree_leaves(grads)
    array_leaves = [g for g in leaves if isinstance(g, jax.Array)]
    assert all(jnp.any(g != 0) for g in array_leaves)
