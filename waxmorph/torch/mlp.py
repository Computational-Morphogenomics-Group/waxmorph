"""PyTorch MLP used by the graph network."""

import torch.nn as nn

from waxmorph._validation import _validate_mlp_config

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """MLP with one or more linear layers.

    ``num_layers`` counts all linear layers. Depth one is a single affine map; deeper
    networks apply the selected activation after each hidden layer.
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

        num_layers = _validate_mlp_config(activation, _ACTIVATIONS, num_layers)

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
