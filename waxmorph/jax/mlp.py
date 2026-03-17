"""Configurable MLP building block for GNS encoder/processor/decoder (Equinox).

Wraps ``eqx.nn.MLP`` with an optional trailing ``LayerNorm`` and a
constructor signature matching the PyTorch ``waxmorph.torch.mlp.MLP``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

_ACTIVATIONS = {
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "tanh": jnp.tanh,
}


class MLP(eqx.Module):
    """Multi-layer perceptron with configurable depth, width, activation, and normalization.

    Thin wrapper around :class:`eqx.nn.MLP` that adds an optional trailing
    ``LayerNorm`` and exposes the same constructor interface as the PyTorch
    ``waxmorph.torch.mlp.MLP``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    output_dim : int
        Dimensionality of output features.
    hidden_dim : int
        Width of each hidden layer.
    num_layers : int
        Total number of linear layers (including output projection).
        ``num_layers=1`` gives a single linear map with no hidden layers.
    activation : str
        One of ``"relu"``, ``"silu"``, ``"gelu"``, ``"tanh"``.
    layer_norm : bool
        If True, apply LayerNorm after the final linear layer.
    key : jax.random.PRNGKey
        PRNG key for weight initialization.
    """

    net: eqx.nn.MLP
    norm: eqx.nn.LayerNorm | None
    activation_name: str = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "silu",
        layer_norm: bool = True,
        *,
        key: jax.Array,
    ):
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from {list(_ACTIVATIONS)}."
            )

        self.activation_name = activation

        # eqx.nn.MLP `depth` = number of hidden layers.
        # Our `num_layers` = total linear layers = depth + 1.
        depth = max(num_layers - 1, 0)
        self.net = eqx.nn.MLP(
            in_size=input_dim,
            out_size=output_dim,
            width_size=hidden_dim,
            depth=depth,
            activation=_ACTIVATIONS[activation],
            key=key,
        )

        self.norm = eqx.nn.LayerNorm(output_dim) if layer_norm else None

    def __call__(self, x: jax.Array) -> jax.Array:
        """``(..., input_dim) -> (..., output_dim)``."""
        if x.ndim > 1:
            out = jax.vmap(self.net)(x)
            if self.norm is not None:
                out = jax.vmap(self.norm)(out)
        else:
            out = self.net(x)
            if self.norm is not None:
                out = self.norm(out)
        return out
