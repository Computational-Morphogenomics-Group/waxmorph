"""PyTorch MLP building block for the graph-network simulator.

Encoder, processor, and decoder sub-networks of the GNS are all instances of
this small feed-forward block, so depth, width, activation, and normalization
are exposed as constructor arguments. This is the default backend; the
JAX/Equinox twin in :mod:`waxmorph.jax.mlp` mirrors the same interface.
"""

import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """Feed-forward block with configurable depth, width, activation, and normalization.

    Linear layers are interleaved with the chosen activation; the
    sigmoid-weighted linear unit (SiLU) is used throughout by default. Hidden
    layers inherit PyTorch's Kaiming-uniform weight initialization (the
    ``torch.nn.Linear`` default). An optional final LayerNorm stabilizes the
    output scale of GNS encoder/decoder blocks. A single-layer block is a plain
    linear map with no activation, used where the GNS needs an unbounded affine
    projection.

    Args:
        input_dim: Width of the trailing input axis.
        output_dim: Width of the trailing output axis.
        hidden_dim: Width of each hidden linear layer.
        num_layers: Total number of linear layers, including the output
            projection. ``num_layers=1`` creates a single linear map.
        activation: Activation name; one of ``"relu"``, ``"silu"``, ``"gelu"``,
            or ``"tanh"``. Defaults to ``"silu"``.
        layer_norm: Whether to append a LayerNorm over the output features.

    Raises:
        ValueError: If ``activation`` is not one of the supported names.

    See Also:
        waxmorph.jax.mlp.MLP: JAX/Equinox twin with the same interface. Its
            Equinox linear layers use LeCun-uniform init rather than the
            Kaiming-uniform init applied here, so initial weight scales differ
            slightly across backends.

    Examples:
        >>> import torch
        >>> mlp = MLP(3, 2, hidden_dim=4, num_layers=1, layer_norm=False)
        >>> print(tuple(mlp(torch.ones(5, 3)).shape))
        (5, 2)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "silu",
        layer_norm: bool = True,
    ):
        super().__init__()

        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from {list(_ACTIVATIONS)}."
            )

        self.activation_name = activation
        act_cls = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []

        if num_layers == 1:
            # single linear map, no activation
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_cls())
            # num_layers - 2 hidden blocks (input + output projections account for the other 2)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_cls())
            layers.append(nn.Linear(hidden_dim, output_dim))

        if layer_norm:
            # normalize over output_dim, after final projection
            layers.append(nn.LayerNorm(output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Map the trailing axis of ``x`` from ``input_dim`` to ``output_dim``.

        Args:
            x: Input with trailing axis of width ``input_dim``; any number of
                leading (batch) axes is allowed.

        Returns:
            Output with the same leading axes as ``x`` and trailing axis of
            width ``output_dim``.
        """
        return self.net(x)
