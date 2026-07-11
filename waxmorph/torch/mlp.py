"""PyTorch MLP used by the graph network."""

from numbers import Integral as _Integral

import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """MLP with one or more linear layers.

    ``num_layers`` counts all linear layers. Depth one is a single affine map with no
    activation; deeper networks apply the selected activation after each hidden layer.
    Optional LayerNorm follows the final projection. Activations are ``relu``, ``silu``,
    ``gelu``, and ``tanh``. Boolean, nonintegral, and nonpositive depths are rejected.

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
        if isinstance(num_layers, bool) or not isinstance(num_layers, _Integral):
            raise TypeError("num_layers must be a non-boolean integer")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        num_layers = int(num_layers)

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
        return self.net(x)
