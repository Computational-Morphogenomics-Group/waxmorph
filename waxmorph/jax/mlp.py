"""JAX/Equinox MLP building block for the graph-network simulator.

Encoder, processor, and decoder sub-networks of the GNS are all instances of
this small feed-forward block, so depth, width, activation, and normalization
are exposed as constructor arguments. This is the parity backend for the
default PyTorch twin in :mod:`waxmorph.torch.mlp`; it keeps the same
constructor interface but requires an explicit PRNG ``key`` for initialization.
"""

from __future__ import annotations

from functools import partial

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
    """Feed-forward block with configurable depth, width, activation, and normalization.

    Linear layers are interleaved with the chosen activation; the
    sigmoid-weighted linear unit (SiLU) is used throughout by default. An
    optional final LayerNorm stabilizes the output scale of GNS encoder/decoder
    blocks. A single-layer block is a plain linear map with no activation, used
    where the GNS needs an unbounded affine projection.

    Args:
        input_dim: Width of the trailing input axis.
        output_dim: Width of the trailing output axis.
        hidden_dim: Width of each hidden linear layer.
        num_layers: Total number of linear layers, including the output
            projection. ``num_layers=1`` creates a single linear map.
        activation: Activation name; one of ``"relu"``, ``"silu"``, ``"gelu"``,
            or ``"tanh"``. Defaults to ``"silu"``.
        layer_norm: Whether to append a LayerNorm over the output features.
        key: PRNG key that seeds the linear-layer weight initialization;
            keyword-only, with no default, so callers must pass a fresh key.

    Raises:
        ValueError: If ``activation`` is not one of the supported names.

    See Also:
        waxmorph.torch.mlp.MLP: Default PyTorch twin with the same interface.
            Both backends initialize linear weights and biases from the same
            distribution ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` -- Equinox via its
            ``lim = 1/sqrt(fan_in)`` uniform init and PyTorch via
            ``kaiming_uniform_(a=sqrt(5))`` (the ``nn.Linear`` default), which
            reduce to the same bound -- so initial weight scales match across
            backends. Only the sampled values differ, because the two frameworks
            draw from independent RNGs.

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

        self.activation_name = activation

        # eqx depth = hidden layers; our num_layers = total linear layers = depth + 1
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
        """Map the trailing axis of ``x`` from ``input_dim`` to ``output_dim``.

        Equinox builds single-vector networks, so batched inputs are flattened
        over all leading axes, vectorized with :func:`jax.vmap`, and reshaped
        back, matching the PyTorch twin's arbitrary-leading-axis behavior.

        Args:
            x: Input with trailing axis of width ``input_dim``; any number of
                leading (batch) axes is allowed.

        Returns:
            Output with the same leading axes as ``x`` and trailing axis of
            width ``output_dim``.
        """
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
