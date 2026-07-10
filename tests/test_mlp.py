"""Tests for the MLP building block."""

import numpy as np
import pytest
import torch

from waxmorph.mlp import MLP


def test_mlp_output_shape():
    mlp = MLP(input_dim=11, output_dim=128, hidden_dim=64, num_layers=3)
    x = torch.randn(50, 11)
    assert mlp(x).shape == (50, 128)


def test_mlp_single_layer():
    mlp = MLP(input_dim=5, output_dim=3, num_layers=1, layer_norm=False)
    x = torch.randn(10, 5)
    assert mlp(x).shape == (10, 3)


def test_mlp_no_layer_norm():
    mlp = MLP(input_dim=4, output_dim=4, num_layers=2, layer_norm=False)
    # Should not have LayerNorm as last module
    last = list(mlp.net.children())[-1]
    assert not isinstance(last, torch.nn.LayerNorm)


def test_mlp_with_layer_norm():
    mlp = MLP(input_dim=4, output_dim=8, num_layers=2, layer_norm=True)
    last = list(mlp.net.children())[-1]
    assert isinstance(last, torch.nn.LayerNorm)


def test_mlp_batched_input():
    mlp = MLP(input_dim=7, output_dim=3, hidden_dim=32, num_layers=2)
    x = torch.randn(4, 10, 7)  # batch of sequences
    assert mlp(x).shape == (4, 10, 3)


def test_mlp_invalid_activation():
    with pytest.raises(ValueError, match="Unknown activation"):
        MLP(input_dim=5, output_dim=3, activation="invalid")


@pytest.mark.parametrize("num_layers", [True, False, 1.0, 2.5, "2", None])
def test_mlp_rejects_non_integer_depth(num_layers):
    with pytest.raises(TypeError, match="non-boolean integer"):
        MLP(input_dim=5, output_dim=3, num_layers=num_layers)


@pytest.mark.parametrize("num_layers", [0, -1])
def test_mlp_rejects_nonpositive_depth(num_layers):
    with pytest.raises(ValueError, match="at least 1"):
        MLP(input_dim=5, output_dim=3, num_layers=num_layers)


def test_mlp_accepts_numpy_integer_depth():
    mlp = MLP(input_dim=5, output_dim=3, num_layers=np.int64(2), layer_norm=False)
    assert sum(isinstance(layer, torch.nn.Linear) for layer in mlp.net) == 2


@pytest.mark.parametrize("activation", ["relu", "silu", "gelu", "tanh"])
def test_mlp_activations(activation):
    mlp = MLP(input_dim=5, output_dim=3, num_layers=2, activation=activation)
    x = torch.randn(8, 5)
    assert mlp(x).shape == (8, 3)
