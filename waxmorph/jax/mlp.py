"""Equinox MLP used by the JAX graph-network simulator."""

from __future__ import annotations

from functools import partial
from numbers import Integral as _Integral

import equinox as eqx
import jax
import jax.numpy as jnp

_ACTIVATIONS = {
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
    "gelu": partial(jax.nn.gelu, approximate=False),
    "tanh": jnp.tanh,
}


class MLP(eqx.Module):
    """MLP with optional LayerNorm on its output.

    ``num_layers`` counts linear maps; one layer is affine with no activation. Hidden layers
    support ``relu``, ``silu``, ``gelu``, and ``tanh``. Callers must supply a PRNG ``key``;
    independent modules need independently split keys.

    Raises:
        TypeError: If ``num_layers`` is not a non-boolean integer.
        ValueError: If ``num_layers < 1`` or the activation is unsupported.

    Examples:
        >>> mlp = MLP(3, 2, hidden_dim=4, num_layers=1, layer_norm=False, key=jax.random.PRNGKey(0))
        >>> print(tuple(mlp(jnp.ones((5, 3))).shape))
        (5, 2)
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
        if isinstance(num_layers, bool) or not isinstance(num_layers, _Integral):
            raise TypeError("num_layers must be a non-boolean integer")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        num_layers = int(num_layers)

        self.activation_name = activation

        # eqx depth = hidden layers; our num_layers = total linear layers = depth + 1
        depth = num_layers - 1
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
        """Preserve arbitrary leading axes by vectorizing Equinox's single-vector network."""
        if x.ndim == 1:
            out = self.net(x)
            if self.norm is not None:
                out = self.norm(out)
            return out
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        out = jax.vmap(self.net)(flat)
        if self.norm is not None:
            out = jax.vmap(self.norm)(out)
        return out.reshape(*lead, out.shape[-1])
