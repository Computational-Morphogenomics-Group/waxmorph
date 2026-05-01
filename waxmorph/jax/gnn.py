"""Graph Network-based Simulator (GNS) for morphogenesis emulation (Equinox).

Implements the Encode-Process-Decode architecture from Sanchez-Gonzalez et al.
"Learning to Simulate Complex Physics with Graph Networks" (ICML 2020),
adapted for the WaxMorph cell state representation.
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from .mlp import MLP


class GraphNetworkBlock(eqx.Module):
    """Single message-passing step: edge update -> aggregation -> node update.

    Both edge and node latents use residual connections. The module subclasses
    :class:`equinox.Module`.

    Args:
        node_latent_dim: Width of node latent vectors.
        edge_latent_dim: Width of edge latent vectors.
        hidden_dim: Hidden layer width for internal MLPs.
        num_mlp_layers: Number of linear layers in each internal MLP.
        activation: Activation function name accepted by
            :class:`waxmorph.jax.mlp.MLP`.
        layer_norm: Whether to apply :class:`equinox.nn.LayerNorm` in internal
            MLPs.
        key: :class:`jax.Array` PRNG key used for weight initialization.
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
        """Run one message-passing round.

        Args:
            node_latent: Node latent array with shape
                ``[N, node_latent_dim]``.
            edge_latent: Edge latent array with shape
                ``[E, edge_latent_dim]``.
            edge_index: Directed COO edge array with shape ``[2, E]`` where
                row ``0`` stores senders and row ``1`` stores receivers.
            num_edges: Optional JAX scalar count of real, non-padding edges.
                When edge arrays are padded for JIT stability, padding entries
                are masked before aggregation.

        Returns:
            Pair ``(node_latent_new, edge_latent_new)`` with the same shapes
            as the corresponding inputs.
        """
        senders, receivers = edge_index[0], edge_index[1]

        # --- Edge update ---
        edge_input = jnp.concatenate(
            [node_latent[senders], node_latent[receivers], edge_latent],
            axis=-1,
        )
        edge_latent_new = edge_latent + self.edge_mlp(edge_input)

        # --- Mask out padding edges before aggregation ---
        if num_edges is not None:
            mask = jnp.arange(edge_latent_new.shape[0]) < num_edges
            edge_latent_new = jnp.where(mask[:, None], edge_latent_new, 0.0)

        # --- Aggregate incoming messages per receiver (sum) ---
        num_nodes = node_latent.shape[0]
        agg = jnp.zeros((num_nodes, edge_latent_new.shape[-1]), dtype=node_latent.dtype)
        agg = agg.at[receivers].add(edge_latent_new)

        # --- Node update ---
        node_input = jnp.concatenate([node_latent, agg], axis=-1)
        node_latent_new = node_latent + self.node_mlp(node_input)

        return node_latent_new, edge_latent_new


class GNS(eqx.Module):
    """Full Encode-Process-Decode Graph Network Simulator.

    Args:
        node_feature_dim: Raw node feature dimensionality.
        edge_feature_dim: Raw edge feature dimensionality.
        node_latent_dim: Width of node latent vectors in the processor.
        edge_latent_dim: Width of edge latent vectors in the processor.
        hidden_dim: Hidden layer width for all internal MLPs.
        num_mp_steps: Number of message-passing blocks in the processor.
        num_mlp_layers: Number of linear layers in each encoder, processor,
            and decoder MLP.
        output_dims: Mapping from output head name to per-node output
            dimensionality. Defaults to ``{"dX": 3, "dP": 3, "dG": 2}``.
        activation: Activation function name accepted by
            :class:`waxmorph.jax.mlp.MLP`.
        layer_norm: Whether to apply :class:`equinox.nn.LayerNorm` in encoder
            and processor MLPs.
        checkpoint_processor: If ``True``, checkpoint processor blocks to
            trade additional compute for lower activation memory.
        key: :class:`jax.Array` PRNG key used for weight initialization.

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

    # Store config for save/load
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
            output_dims = {"dX": 3, "dP": 3, "dG": 2}

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

        # --- Encoder ---
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

        # --- Processor (M independent blocks) ---
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

        # --- Decoder (one head per output field, no LayerNorm) ---
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
        """Run full encode-process-decode.

        Args:
            node_features: :class:`jax.Array` node features with shape
                ``[N, node_feature_dim]``.
            edge_index: Directed COO edge array with shape ``[2, E]``.
            edge_features: :class:`jax.Array` edge features with shape
                ``[E, edge_feature_dim]``.
            num_edges: Optional JAX scalar count of real, non-padding edges.

        Returns:
            Dictionary mapping each output head name to an array with shape
            ``[N, output_dim]``.
        """
        # Encode
        node_latent = self.node_encoder(node_features)
        edge_latent = self.edge_encoder(edge_features)

        # Process
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

        # Decode
        if self.checkpoint_processor:
            return {
                name: eqx.filter_checkpoint(dec)(node_latent) for name, dec in self.decoders.items()
            }
        return {name: dec(node_latent) for name, dec in self.decoders.items()}

    def save(self, path: str | Path) -> None:
        """Save model config and weights.

        Creates two files: ``path`` (weights serialized with
        :func:`equinox.tree_serialise_leaves`) and ``path.json`` (config).
        """
        path = Path(path)
        eqx.tree_serialise_leaves(path, self)
        with open(str(path) + ".json", "w") as f:
            json.dump(self._config, f)

    @classmethod
    def load(cls, path: str | Path) -> GNS:
        """Load model from files saved with :meth:`save`.

        Args:
            path: Path to the weights file. Configuration is read from
                ``path.json``.

        Returns:
            Deserialized :class:`waxmorph.jax.gnn.GNS` model.
        """
        path = Path(path)
        with open(str(path) + ".json") as f:
            config = json.load(f)
        # Need a dummy key to create the skeleton
        skeleton = cls(**config, key=jax.random.PRNGKey(0))
        return eqx.tree_deserialise_leaves(path, skeleton)
