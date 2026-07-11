"""PyTorch Encode-Process-Decode graph network for cell-state updates.

Contacts are directed COO edges. :mod:`waxmorph.jax.gnn` is a behavioral counterpart,
not an architecture or serialization identity.
Architecture: Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with Graph
Networks" (ICML 2020).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .mlp import MLP


class GraphNetworkBlock(nn.Module):
    """Residual edge-then-node message passing.

    Each directed edge derives a message from its sender, receiver, and edge latents;
    incoming messages are summed at receivers. ``num_mlp_layers`` counts linear layers
    in each MLP, and ``layer_norm`` normalizes each MLP output.
    """

    def __init__(
        self,
        node_latent_dim: int = 128,
        edge_latent_dim: int = 128,
        hidden_dim: int = 128,
        num_mlp_layers: int = 2,
        activation: str = "silu",
        layer_norm: bool = True,
    ):
        super().__init__()
        self.edge_mlp = MLP(
            input_dim=2 * node_latent_dim + edge_latent_dim,
            output_dim=edge_latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.node_mlp = MLP(
            input_dim=node_latent_dim + edge_latent_dim,
            output_dim=node_latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(
        self,
        node_latent: torch.Tensor,
        edge_latent: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return residual updates for node and edge latents.

        ``edge_index`` is directed COO ``[2, E]``: row 0 contains senders and row 1
        receivers. Reverse contact orientations remain distinct messages.
        """
        senders, receivers = edge_index[0], edge_index[1]

        edge_input = torch.cat(
            [node_latent[senders], node_latent[receivers], edge_latent],
            dim=-1,
        )
        edge_latent_new = edge_latent + self.edge_mlp(edge_input)

        num_nodes = node_latent.size(0)
        agg = torch.zeros(
            num_nodes,
            edge_latent_new.size(-1),
            device=node_latent.device,
            dtype=node_latent.dtype,
        )
        idx = receivers.unsqueeze(-1).expand_as(edge_latent_new)
        agg.scatter_add_(0, idx, edge_latent_new)

        node_input = torch.cat([node_latent, agg], dim=-1)
        node_latent_new = node_latent + self.node_mlp(node_input)

        return node_latent_new, edge_latent_new


class GNS(nn.Module):
    """Encode-process-decode graph network.

    Independent residual message-passing blocks consume encoded node and edge features.
    Named decoders map final node latents to per-node updates; defaults are ``dX: 3``,
    ``dP: 3``, and ``dc: 2``. ``num_mlp_layers`` counts linear layers in every MLP.
    ``layer_norm`` applies to encoders and processor MLPs, not decoders.
    ``checkpoint_processor`` recomputes processor and decoder activations during backward
    to reduce saved activation memory.

    Examples:
        >>> model = GNS(1, 2, hidden_dim=4, num_mp_steps=1, output_dims={"dX": 3})
        >>> out = model(torch.ones(2, 1), torch.tensor([[0, 1], [1, 0]]), torch.ones(2, 2))
        >>> print(sorted(out), tuple(out["dX"].shape))
        ['dX'] (2, 3)
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        node_latent_dim: int = 128,
        edge_latent_dim: int = 128,
        hidden_dim: int = 128,
        num_mp_steps: int = 10,
        num_mlp_layers: int = 2,
        output_dims: dict[str, int] | None = None,
        activation: str = "relu",
        layer_norm: bool = True,
        checkpoint_processor: bool = False,
    ):
        super().__init__()
        self.checkpoint_processor = checkpoint_processor
        self._hidden_dim = hidden_dim

        if output_dims is None:
            output_dims = {"dX": 3, "dP": 3, "dc": 2}

        self.node_encoder = MLP(
            input_dim=node_feature_dim,
            output_dim=node_latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.edge_encoder = MLP(
            input_dim=edge_feature_dim,
            output_dim=edge_latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
        )

        self.processor = nn.ModuleList(
            [
                GraphNetworkBlock(
                    node_latent_dim=node_latent_dim,
                    edge_latent_dim=edge_latent_dim,
                    hidden_dim=hidden_dim,
                    num_mlp_layers=num_mlp_layers,
                    activation=activation,
                    layer_norm=layer_norm,
                )
                for _ in range(num_mp_steps)
            ]
        )

        self.decoders = nn.ModuleDict(
            {
                name: MLP(
                    input_dim=node_latent_dim,
                    output_dim=dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_mlp_layers,
                    activation=activation,
                    layer_norm=False,
                )
                for name, dim in output_dims.items()
            }
        )

    def _constructor_config(self) -> dict[str, object]:
        def output_dim(mlp: MLP) -> int:
            last_layer = mlp.net[-1]
            if isinstance(last_layer, nn.Linear):
                return last_layer.out_features
            return last_layer.normalized_shape[0]

        return {
            "node_feature_dim": self.node_encoder.net[0].in_features,
            "edge_feature_dim": self.edge_encoder.net[0].in_features,
            "node_latent_dim": output_dim(self.node_encoder),
            "edge_latent_dim": output_dim(self.edge_encoder),
            "hidden_dim": self._hidden_dim,
            "num_mp_steps": len(self.processor),
            "num_mlp_layers": sum(
                isinstance(module, nn.Linear) for module in self.node_encoder.net
            ),
            "output_dims": {name: output_dim(decoder) for name, decoder in self.decoders.items()},
            "activation": self.node_encoder.activation_name,
            "layer_norm": any(isinstance(module, nn.LayerNorm) for module in self.node_encoder.net),
            "checkpoint_processor": self.checkpoint_processor,
        }

    def save(self, path: str | Path) -> None:
        """Serialize ``{"config": ..., "state_dict": ...}`` with :func:`torch.save`."""
        torch.save(
            {
                "config": self._constructor_config(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> GNS:
        """Load the ``config``/``state_dict`` schema written by :meth:`save`.

        ``kwargs`` pass to :func:`torch.load`. ``weights_only=False`` unpickles Python
        objects and can execute code; load only trusted checkpoints.
        """
        data = torch.load(path, weights_only=False, **kwargs)
        model = cls(**data["config"])
        model.load_state_dict(data["state_dict"])
        return model

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode per-node updates from raw graph features.

        ``edge_index`` is directed COO ``[2, E]`` with senders in row 0. Each configured
        output head maps to ``[N, output_dim]``.
        """
        node_latent = self.node_encoder(node_features)
        edge_latent = self.edge_encoder(edge_features)

        for block in self.processor:
            if self.checkpoint_processor:
                node_latent, edge_latent = torch_checkpoint(
                    block,
                    node_latent,
                    edge_latent,
                    edge_index,
                    use_reentrant=False,
                )
            else:
                node_latent, edge_latent = block(node_latent, edge_latent, edge_index)

        ret_val = None
        if self.checkpoint_processor:
            ret_val = {
                name: torch_checkpoint(dec, node_latent, use_reentrant=False)
                for name, dec in self.decoders.items()
            }

        else:
            ret_val = {name: dec(node_latent) for name, dec in self.decoders.items()}
        return ret_val
