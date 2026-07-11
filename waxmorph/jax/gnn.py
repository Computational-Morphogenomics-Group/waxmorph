"""Equinox encode-process-decode GNS (Sanchez-Gonzalez et al., ICML 2020).

Static JIT shapes use padded directed contact edges; ``num_edges`` masks padding before
message aggregation. This is the behavioral counterpart of :mod:`waxmorph.torch.gnn`.
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from .mlp import MLP


class GraphNetworkBlock(eqx.Module):
    """Residual edge and node updates over directed COO messages.

    Each ``sender -> receiver`` row has a separate edge latent. Incoming real-edge messages
    are summed by receiver; ``num_edges`` masks padding. ``key`` initializes the two MLPs
    independently.
    """

    edge_mlp: MLP
    node_mlp: MLP

    def __init__(
        self,
        node_latent_dim: int = 128,
        edge_latent_dim: int = 128,
        hidden_dim: int = 128,
        num_mlp_layers: int = 2,
        activation: str = "silu",
        layer_norm: bool = True,
        *,
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key)
        self.edge_mlp = MLP(
            input_dim=2 * node_latent_dim + edge_latent_dim,
            output_dim=edge_latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
            key=k1,
        )
        self.node_mlp = MLP(
            input_dim=node_latent_dim + edge_latent_dim,
            output_dim=node_latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
            key=k2,
        )

    def __call__(
        self,
        node_latent: jax.Array,
        edge_latent: jax.Array,
        edge_index: jax.Array,
        num_edges: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Return residual node and edge latents.

        ``edge_index`` is directed COO ``[2, E]`` (senders, receivers). Entries at indices
        greater than or equal to ``num_edges`` contribute no message.
        """
        senders, receivers = edge_index[0], edge_index[1]

        edge_input = jnp.concatenate(
            [node_latent[senders], node_latent[receivers], edge_latent],
            axis=-1,
        )
        edge_latent_new = edge_latent + self.edge_mlp(edge_input)

        if num_edges is not None:
            mask = jnp.arange(edge_latent_new.shape[0]) < num_edges
            edge_latent_new = jnp.where(mask[:, None], edge_latent_new, 0.0)

        num_nodes = node_latent.shape[0]
        agg = jnp.zeros((num_nodes, edge_latent_new.shape[-1]), dtype=node_latent.dtype)
        agg = agg.at[receivers].add(edge_latent_new)

        node_input = jnp.concatenate([node_latent, agg], axis=-1)
        node_latent_new = node_latent + self.node_mlp(node_input)

        return node_latent_new, edge_latent_new


class GNS(eqx.Module):
    """Encode graph features, apply residual message passing, and decode state deltas.

    Default heads ``dX``, ``dP``, and ``dc`` are raw per-node position, polarity, and
    concentration deltas; training scales them by ``dt_gns``. ``key`` is split across all
    submodules. ``checkpoint_processor`` rematerializes processor blocks and decoder heads,
    trading compute for fewer saved activations.

    Examples:
        >>> model = GNS(1, 2, hidden_dim=4, num_mp_steps=1, output_dims={"dX": 3}, key=jax.random.PRNGKey(0))
        >>> out = model(jnp.ones((2, 1)), jnp.array([[0, 1], [1, 0]]), jnp.ones((2, 2)))
        >>> print(sorted(out), tuple(out["dX"].shape))
        ['dX'] (2, 3)
    """

    node_encoder: MLP
    edge_encoder: MLP
    processor: list[GraphNetworkBlock]
    decoders: dict[str, MLP]
    checkpoint_processor: bool = eqx.field(static=True)

    _config: dict = eqx.field(static=True)

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
        *,
        key: jax.Array,
    ):
        self.checkpoint_processor = checkpoint_processor

        if output_dims is None:
            output_dims = {"dX": 3, "dP": 3, "dc": 2}

        self._config = {
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
            "node_latent_dim": node_latent_dim,
            "edge_latent_dim": edge_latent_dim,
            "hidden_dim": hidden_dim,
            "num_mp_steps": num_mp_steps,
            "num_mlp_layers": num_mlp_layers,
            "output_dims": output_dims,
            "activation": activation,
            "layer_norm": layer_norm,
            "checkpoint_processor": checkpoint_processor,
        }

        keys = jax.random.split(key, 2 + num_mp_steps + len(output_dims))

        self.node_encoder = MLP(
            input_dim=node_feature_dim,
            output_dim=node_latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
            key=keys[0],
        )
        self.edge_encoder = MLP(
            input_dim=edge_feature_dim,
            output_dim=edge_latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            layer_norm=layer_norm,
            key=keys[1],
        )

        self.processor = [
            GraphNetworkBlock(
                node_latent_dim=node_latent_dim,
                edge_latent_dim=edge_latent_dim,
                hidden_dim=hidden_dim,
                num_mlp_layers=num_mlp_layers,
                activation=activation,
                layer_norm=layer_norm,
                key=keys[2 + i],
            )
            for i in range(num_mp_steps)
        ]

        self.decoders = {
            name: MLP(
                input_dim=node_latent_dim,
                output_dim=dim,
                hidden_dim=hidden_dim,
                num_layers=num_mlp_layers,
                activation=activation,
                layer_norm=False,
                key=keys[2 + num_mp_steps + i],
            )
            for i, (name, dim) in enumerate(output_dims.items())
        }

    def __call__(
        self,
        node_features: jax.Array,
        edge_index: jax.Array,
        edge_features: jax.Array,
        num_edges: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Decode per-node heads from a directed COO graph.

        ``num_edges`` masks padded rows. Each value has shape
        ``[N, output_dims[head]]``.
        """
        node_latent = self.node_encoder(node_features)
        edge_latent = self.edge_encoder(edge_features)

        for block in self.processor:
            if self.checkpoint_processor:
                node_latent, edge_latent = eqx.filter_checkpoint(block)(
                    node_latent,
                    edge_latent,
                    edge_index,
                    num_edges,
                )
            else:
                node_latent, edge_latent = block(
                    node_latent,
                    edge_latent,
                    edge_index,
                    num_edges,
                )

        if self.checkpoint_processor:
            return {
                name: eqx.filter_checkpoint(dec)(node_latent) for name, dec in self.decoders.items()
            }
        return {name: dec(node_latent) for name, dec in self.decoders.items()}

    def save(self, path: str | Path) -> None:
        """Write Equinox leaves to ``path`` and constructor config to ``path.json``."""
        path = Path(path)
        eqx.tree_serialise_leaves(path, self)
        with open(str(path) + ".json", "w") as f:
            json.dump(self._config, f)

    @classmethod
    def load(cls, path: str | Path) -> GNS:
        """Reconstruct from the ``path.json`` config and leaves stored at ``path``."""
        path = Path(path)
        with open(str(path) + ".json") as f:
            config = json.load(f)
        skeleton = cls(**config, key=jax.random.PRNGKey(0))
        return eqx.tree_deserialise_leaves(path, skeleton)
