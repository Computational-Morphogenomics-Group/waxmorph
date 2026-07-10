"""Graph Network-based Simulator (GNS) for morphogenesis emulation (Equinox).

A cell aggregate is a graph: cells are nodes carrying signaling state, contacts
are edges carrying mechanical features, and one emulation step is a learned,
neighbor-dependent update to each cell. The architecture is the
Encode-Process-Decode graph network of Sanchez-Gonzalez et al. "Learning to
Simulate Complex Physics with Graph Networks" (ICML 2020), adapted to the
waxMorph cell-state representation.

This is the JAX/Equinox parity twin of the default PyTorch backend
:mod:`waxmorph.torch.gnn`; the two must be kept architecturally identical.
Static-shape compilation requires padded edge arrays, masked before aggregation
via ``num_edges``.
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

    Both edge and node latents use residual connections, which stabilize deep
    message passing by letting each block learn a correction to the running
    embedding rather than rebuilding it from scratch.

    Notation:
        edge_mlp=f_psi (message fn), node_mlp=f_pi (node-update fn);
        node_latent=u, edge_latent=w, message=eta, node-update=zeta.

    Args:
        node_latent_dim: Width of node latent vectors.
        edge_latent_dim: Width of edge latent vectors.
        hidden_dim: Hidden layer width for internal MLPs.
        num_mlp_layers: Number of linear layers in each internal MLP.
        activation: Activation function name accepted by
            :class:`waxmorph.jax.mlp.MLP`.
        layer_norm: Whether to apply :class:`equinox.nn.LayerNorm` in internal
            MLPs.
        key: PRNG key used for weight initialization.

    See Also:
        :class:`waxmorph.torch.gnn.GraphNetworkBlock`: PyTorch parity twin.
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

        # Edge update (f_psi): message eta from [sender u, receiver u, edge w], residual.
        # Directed COO keeps i->j and j->i as separate rows, so the message MLP can
        # emit distinct directional latents for each orientation of a contact.
        edge_input = jnp.concatenate(
            [node_latent[senders], node_latent[receivers], edge_latent],
            axis=-1,
        )
        edge_latent_new = edge_latent + self.edge_mlp(edge_input)

        # Zero padding messages so they contribute 0 to the aggregation (static-shape JIT)
        if num_edges is not None:
            mask = jnp.arange(edge_latent_new.shape[0]) < num_edges
            edge_latent_new = jnp.where(mask[:, None], edge_latent_new, 0.0)

        # Aggregate messages: sum eta over incoming edges per receiver via scatter-add
        num_nodes = node_latent.shape[0]
        agg = jnp.zeros((num_nodes, edge_latent_new.shape[-1]), dtype=node_latent.dtype)
        agg = agg.at[receivers].add(edge_latent_new)

        # Node update (f_pi): zeta from [node u, aggregated msgs], residual
        node_input = jnp.concatenate([node_latent, agg], axis=-1)
        node_latent_new = node_latent + self.node_mlp(node_input)

        return node_latent_new, edge_latent_new


class GNS(eqx.Module):
    """Full Encode-Process-Decode Graph Network Simulator.

    Encode raw per-cell and per-contact features into latents, process them with
    M message-passing blocks over the spatial-adjacency graph, then decode the
    final node latents into Euler updates for the learned cell state.

    Notation:
        node_encoder=f_phi, edge_encoder=f_rho; processor runs M=num_mp_steps
        GraphNetworkBlock steps; decoders {dX:f_omega, dP:f_mu, dc:f_nu}.

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
            dimensionality. Defaults to ``{"dX": 3, "dP": 3, "dc": 2}``.
        activation: Activation function name accepted by
            :class:`waxmorph.jax.mlp.MLP`.
        layer_norm: Whether to apply :class:`equinox.nn.LayerNorm` in encoder
            and processor MLPs.
        checkpoint_processor: If ``True``, checkpoint processor blocks to
            trade additional compute for lower activation memory.
        key: PRNG key used for weight initialization.

    See Also:
        :class:`waxmorph.torch.gnn.GNS`: PyTorch parity twin (default backend).

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
            node_features: Per-cell signaling features with shape
                ``[N, node_feature_dim]``.
            edge_index: Directed COO edges with shape ``[2, E]``; row ``0`` holds
                senders and row ``1`` holds receivers.
            edge_features: Per-contact mechanical features with shape
                ``[E, edge_feature_dim]``.
            num_edges: Optional scalar count of real, non-padding edges. When edge
                arrays are padded for static-shape JIT, padding rows are masked
                before aggregation.

        Returns:
            Each output head name mapped to its decoded update with shape
            ``[N, output_dim]``.
        """
        # Encode: raw features -> latents (f_phi nodes, f_rho edges)
        node_latent = self.node_encoder(node_features)
        edge_latent = self.edge_encoder(edge_features)

        # Process: M message-passing rounds, optionally rematerialized in backward to save memory
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

        # Decode: per-head MLP maps final node latent -> output field (f_omega/f_mu/f_nu)
        if self.checkpoint_processor:
            return {
                name: eqx.filter_checkpoint(dec)(node_latent) for name, dec in self.decoders.items()
            }
        return {name: dec(node_latent) for name, dec in self.decoders.items()}

    def save(self, path: str | Path) -> None:
        """Save model config and weights.

        Creates two files: ``path`` (weights serialized with
        :func:`equinox.tree_serialise_leaves`) and ``path.json`` (config). Unlike
        the PyTorch twin, the constructor config is carried on the instance
        (``_config``) and written verbatim, so no layer-width introspection is
        needed.

        Args:
            path: Base path for the weights file; the config sidecar is written to
                ``path.json``.

        See Also:
            :meth:`waxmorph.torch.gnn.GNS.save`: PyTorch parity twin (single file).
        """
        path = Path(path)
        eqx.tree_serialise_leaves(path, self)
        with open(str(path) + ".json", "w") as f:
            json.dump(self._config, f)

    @classmethod
    def load(cls, path: str | Path) -> GNS:
        """Load model from files saved with :meth:`save`.

        Reads ``path.json`` for the config, builds an architecture skeleton (with a
        throwaway PRNG key, since the weights are overwritten), then deserializes
        the saved leaves into it.

        Args:
            path: Path to the weights file. Configuration is read from
                ``path.json``.

        Returns:
            Deserialized :class:`waxmorph.jax.gnn.GNS` model.

        See Also:
            :meth:`waxmorph.torch.gnn.GNS.load`: PyTorch parity twin.
        """
        path = Path(path)
        with open(str(path) + ".json") as f:
            config = json.load(f)
        skeleton = cls(**config, key=jax.random.PRNGKey(0))
        return eqx.tree_deserialise_leaves(path, skeleton)
