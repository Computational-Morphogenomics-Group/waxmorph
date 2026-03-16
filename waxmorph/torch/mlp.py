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
        """``(..., input_dim) -> (..., output_dim)``."""
        return self.net(x)
