"""Graph Network-based Simulator (GNS) for morphogenesis emulation.

Implements the Encode-Process-Decode architecture from Sanchez-Gonzalez et al.
"Learning to Simulate Complex Physics with Graph Networks" (ICML 2020),
adapted for the WaxMorph cell state representation.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .mlp import MLP


class GraphNetworkBlock(nn.Module):
    """Single message-passing step: edge update -> aggregation -> node update.

    Both edge and node latents use residual connections.

    Parameters
    ----------
    node_latent_dim : int
        Width of node latent vectors.
    edge_latent_dim : int
        Width of edge latent vectors.
    hidden_dim : int
        Hidden layer width for internal MLPs.
    num_mlp_layers : int
        Depth of each internal MLP.
    activation : str
        Activation function name.
    layer_norm : bool
        Whether to apply LayerNorm in internal MLPs.
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
        """Run one message-passing round.

        Parameters
        ----------
        node_latent : ``[N, node_latent_dim]``
        edge_latent : ``[E, edge_latent_dim]``
        edge_index  : ``[2, E]``   (senders, receivers)

        Returns
        -------
        node_latent_new : ``[N, node_latent_dim]``
        edge_latent_new : ``[E, edge_latent_dim]``
        """
        senders, receivers = edge_index[0], edge_index[1]

        # --- Edge update ---
        edge_input = torch.cat(
            [node_latent[senders], node_latent[receivers], edge_latent],
            dim=-1,
        )
        edge_latent_new = edge_latent + self.edge_mlp(edge_input)

        # --- Aggregate incoming messages per receiver (sum) ---
        num_nodes = node_latent.size(0)
        agg = torch.zeros(
            num_nodes,
            edge_latent_new.size(-1),
            device=node_latent.device,
            dtype=node_latent.dtype,
        )
        idx = receivers.unsqueeze(-1).expand_as(edge_latent_new)
        agg.scatter_add_(0, idx, edge_latent_new)

        # --- Node update ---
        node_input = torch.cat([node_latent, agg], dim=-1)
        node_latent_new = node_latent + self.node_mlp(node_input)

        return node_latent_new, edge_latent_new


class GNS(nn.Module):
    """Full Encode-Process-Decode Graph Network Simulator.

    Parameters
    ----------
    node_feature_dim : int
        Raw node feature dimensionality (``9 + G`` for WaxMorph).
    edge_feature_dim : int
        Raw edge feature dimensionality (``7`` for default WaxMorph edges).
    node_latent_dim : int
        Width of node latent vectors in the processor.
    edge_latent_dim : int
        Width of edge latent vectors in the processor.
    hidden_dim : int
        Hidden layer width for all internal MLPs.
    num_mp_steps : int
        Number of message-passing iterations in the processor.
    num_mlp_layers : int
        Depth of each MLP (encoder, processor, decoder).
    output_dims : dict
        Named output heads mapping field name to dimensionality.
        Example: ``{"dX": 3, "dP": 3, "dG": 2}``.
    activation : str
        Activation function name.
    layer_norm : bool
        Whether to apply LayerNorm in encoder/processor MLPs.
    checkpoint_processor : bool
        If ``True``, wrap each processor block with
        ``torch.utils.checkpoint.checkpoint`` to trade compute for memory.
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

        if output_dims is None:
            output_dims = {"dX": 3, "dP": 3}

        # --- Encoder ---
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

        # --- Processor (M independent blocks) ---
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

        # --- Decoder (one head per output field, no LayerNorm) ---
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

    def save(self, path: str | Path) -> None:
        """Save model config and weights to a single file."""
        torch.save(
            {
                "config": {
                    "node_feature_dim": self.node_encoder.net[0].in_features,
                    "edge_feature_dim": self.edge_encoder.net[0].in_features,
                    "node_latent_dim": (
                        self.node_encoder.net[-1].out_features
                        if isinstance(self.node_encoder.net[-1], nn.Linear)
                        else self.node_encoder.net[-1].normalized_shape[0]
                    ),
                    "edge_latent_dim": (
                        self.edge_encoder.net[-1].out_features
                        if isinstance(self.edge_encoder.net[-1], nn.Linear)
                        else self.edge_encoder.net[-1].normalized_shape[0]
                    ),
                    "hidden_dim": self.node_encoder.net[0].out_features,
                    "num_mp_steps": len(self.processor),
                    "num_mlp_layers": len(
                        [m for m in self.node_encoder.net if isinstance(m, nn.Linear)]
                    ),
                    "output_dims": {
                        name: (
                            dec.net[-1].out_features
                            if isinstance(dec.net[-1], nn.Linear)
                            else dec.net[-1].normalized_shape[0]
                        )
                        for name, dec in self.decoders.items()
                    },
                    "activation": self.node_encoder.activation_name,
                    "layer_norm": any(isinstance(m, nn.LayerNorm) for m in self.node_encoder.net),
                    "checkpoint_processor": self.checkpoint_processor,
                },
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> GNS:
        """Load model from a file saved with :meth:`save`.

        Extra *kwargs* are forwarded to :func:`torch.load`
        (e.g. ``map_location="cpu"``).
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
        """Run full encode-process-decode.

        Parameters
        ----------
        node_features : ``[N, node_feature_dim]``
        edge_index    : ``[2, E]``
        edge_features : ``[E, edge_feature_dim]``

        Returns
        -------
        outputs : dict mapping field name to ``[N, output_dim]`` predicted updates.
        """
        # Encode
        node_latent = self.node_encoder(node_features)
        edge_latent = self.edge_encoder(edge_features)

        # Process
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

        # Decode
        ret_val = None
        if self.checkpoint_processor:
            ret_val = {
                name: torch_checkpoint(dec, node_latent, use_reentrant=False)
                for name, dec in self.decoders.items()
            }

        else:
            ret_val = {name: dec(node_latent) for name, dec in self.decoders.items()}
        return ret_val
