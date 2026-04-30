"""Configurable MLP building block for GNS encoder/processor/decoder."""

import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """Multi-layer perceptron with configurable depth, width, activation, and normalization.

    Args:
        input_dim: Dimensionality of the final axis of the input tensor.
        output_dim: Dimensionality of the final axis of the output tensor.
        hidden_dim: Width of each hidden linear layer.
        num_layers: Total number of linear layers, including the output
            projection. ``num_layers=1`` creates a single linear map.
        activation: Activation name: ``"relu"``, ``"silu"``, ``"gelu"``, or
            ``"tanh"``.
        layer_norm: Whether to append :class:`torch.nn.LayerNorm` over
            ``output_dim``.

    Raises:
        ValueError: If ``activation`` is not supported.
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
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_cls())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_cls())
            layers.append(nn.Linear(hidden_dim, output_dim))

        if layer_norm:
            layers.append(nn.LayerNorm(output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Apply the MLP to the final axis of ``x``.

        Args:
            x: Tensor with trailing dimension ``input_dim``.

        Returns:
            Tensor with the same leading dimensions and trailing dimension
            ``output_dim``.
        """
        return self.net(x)
