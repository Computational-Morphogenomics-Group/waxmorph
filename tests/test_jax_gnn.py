"""Tests for the JAX/Equinox GNS architecture."""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from waxmorph.jax.gnn import GNS, GraphNetworkBlock


@pytest.fixture()
def key():
    return jax.random.PRNGKey(0)


class TestGraphNetworkBlock:
    def test_output_shapes(self, key):
        block = GraphNetworkBlock(
            node_latent_dim=64, edge_latent_dim=32, hidden_dim=64, num_mlp_layers=2, key=key
        )
        N, E = 20, 50
        node_latent = jax.random.normal(key, (N, 64))
        edge_latent = jax.random.normal(key, (E, 32))
        edge_index = jax.random.randint(key, (2, E), 0, N)

        node_out, edge_out = block(node_latent, edge_latent, edge_index)

        assert node_out.shape == (N, 64)
        assert edge_out.shape == (E, 32)

    def test_residual_connection(self, key):
        block = GraphNetworkBlock(
            node_latent_dim=16, edge_latent_dim=16, hidden_dim=16, num_mlp_layers=1, key=key
        )
        N, E = 5, 10
        node_latent = jax.random.normal(key, (N, 16))
        edge_latent = jax.random.normal(key, (E, 16))
        edge_index = jax.random.randint(key, (2, E), 0, N)

        # Zero out all MLP weights and biases (only array leaves)
        def zero_arrays(tree):
            return jax.tree.map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, tree)

        block = eqx.tree_at(lambda b: b.edge_mlp, block, zero_arrays(block.edge_mlp))
        block = eqx.tree_at(lambda b: b.node_mlp, block, zero_arrays(block.node_mlp))

        node_out, edge_out = block(node_latent, edge_latent, edge_index)

        np.testing.assert_allclose(np.asarray(edge_out), np.asarray(edge_latent), atol=1e-6)
        np.testing.assert_allclose(np.asarray(node_out), np.asarray(node_latent), atol=1e-6)


class TestGNS:
    def test_forward_shapes(self, key):
        N, num_molecules = 50, 2
        E = 200
        F_node = 9 + num_molecules
        F_edge = 7

        gns = GNS(
            node_feature_dim=F_node,
            edge_feature_dim=F_edge,
            num_mp_steps=3,
            output_dims={"dX": 3, "dP": 3, "dc": num_molecules},
            key=key,
        )

        node_feat = jax.random.normal(key, (N, F_node))
        edge_idx = jax.random.randint(key, (2, E), 0, N)
        edge_feat = jax.random.normal(key, (E, F_edge))

        out = gns(node_feat, edge_idx, edge_feat)

        assert set(out.keys()) == {"dX", "dP", "dc"}
        assert out["dX"].shape == (N, 3)
        assert out["dP"].shape == (N, 3)
        assert out["dc"].shape == (N, num_molecules)

    def test_default_output_dims(self, key):
        gns = GNS(node_feature_dim=9, edge_feature_dim=7, num_mp_steps=1, key=key)
        node_feat = jax.random.normal(key, (10, 9))
        edge_idx = jax.random.randint(key, (2, 30), 0, 10)
        edge_feat = jax.random.normal(key, (30, 7))

        out = gns(node_feat, edge_idx, edge_feat)

        assert set(out.keys()) == {"dX", "dP", "dc"}

    def test_gradient_flow(self, key):
        gns = GNS(
            node_feature_dim=11,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
            key=key,
        )
        node_feat = jax.random.normal(key, (20, 11))
        edge_idx = jax.random.randint(key, (2, 60), 0, 20)
        edge_feat = jax.random.normal(key, (60, 7))

        @eqx.filter_grad
        def grad_fn(model):
            out = model(node_feat, edge_idx, edge_feat)
            return out["dX"].sum()

        grads = grad_fn(gns)
        leaves = jax.tree_util.tree_leaves(grads)
        array_leaves = [g for g in leaves if isinstance(g, jax.Array)]
        assert all(jnp.any(g != 0) for g in array_leaves)

    def test_no_edges(self, key):
        """GNS should handle graphs with zero edges gracefully."""
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
            key=key,
        )
        node_feat = jax.random.normal(key, (5, 9))
        edge_idx = jnp.zeros((2, 0), dtype=jnp.int32)
        edge_feat = jnp.zeros((0, 7))

        out = gns(node_feat, edge_idx, edge_feat)

        assert out["dX"].shape == (5, 3)

    def test_single_particle(self, key):
        """Single particle with no neighbors."""
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
            key=key,
        )
        node_feat = jax.random.normal(key, (1, 9))
        edge_idx = jnp.zeros((2, 0), dtype=jnp.int32)
        edge_feat = jnp.zeros((0, 7))

        out = gns(node_feat, edge_idx, edge_feat)

        assert out["dX"].shape == (1, 3)

    @pytest.mark.parametrize("num_mp_steps", [1, 5, 10])
    def test_variable_mp_steps(self, num_mp_steps, key):
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=num_mp_steps,
            output_dims={"dX": 3},
            key=key,
        )
        assert len(gns.processor) == num_mp_steps

        out = gns(
            jax.random.normal(key, (10, 9)),
            jax.random.randint(key, (2, 30), 0, 10),
            jax.random.normal(key, (30, 7)),
        )
        assert out["dX"].shape == (10, 3)

    def test_checkpoint_processor_same_output(self, key):
        """Checkpointed and non-checkpointed GNS produce identical outputs."""
        kwargs = dict(
            node_feature_dim=11,
            edge_feature_dim=7,
            num_mp_steps=3,
            output_dims={"dX": 3},
            node_latent_dim=32,
            edge_latent_dim=32,
            hidden_dim=32,
        )
        gns_plain = GNS(**kwargs, checkpoint_processor=False, key=key)
        # Create checkpointed version with same weights via serialization roundtrip
        import pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "model.eqx"
            gns_plain.save(path)
            # Reload with checkpoint_processor=True
            import json

            config_path = str(path) + ".json"
            with open(config_path) as f:
                config = json.load(f)
            config["checkpoint_processor"] = True
            with open(config_path, "w") as f:
                json.dump(config, f)
            gns_ckpt = GNS.load(path)

        node_feat = jax.random.normal(key, (15, 11))
        edge_idx = jax.random.randint(key, (2, 40), 0, 15)
        edge_feat = jax.random.normal(key, (40, 7))

        out_plain = gns_plain(node_feat, edge_idx, edge_feat)
        out_ckpt = gns_ckpt(node_feat, edge_idx, edge_feat)

        np.testing.assert_allclose(
            np.asarray(out_plain["dX"]), np.asarray(out_ckpt["dX"]), atol=1e-5
        )

    def test_checkpoint_processor_gradient_flow(self, key):
        """Checkpointed GNS still propagates gradients."""
        gns = GNS(
            node_feature_dim=11,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
            checkpoint_processor=True,
            key=key,
        )
        node_feat = jax.random.normal(key, (20, 11))
        edge_idx = jax.random.randint(key, (2, 60), 0, 20)
        edge_feat = jax.random.normal(key, (60, 7))

        @eqx.filter_grad
        def grad_fn(model):
            out = model(node_feat, edge_idx, edge_feat)
            return out["dX"].sum()

        grads = grad_fn(gns)
        leaves = jax.tree_util.tree_leaves(grads)
        array_leaves = [g for g in leaves if isinstance(g, jax.Array)]
        assert all(jnp.any(g != 0) for g in array_leaves)


class TestEdgePaddingMasking:
    def test_padding_does_not_affect_output(self, key):
        """Padded edges with num_edges mask should produce same output as unpadded."""
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=3,
            output_dims={"dX": 3},
            key=key,
        )
        N, E_real = 10, 30
        node_feat = jax.random.normal(key, (N, 9))
        edge_idx_real = jax.random.randint(key, (2, E_real), 0, N)
        edge_feat_real = jax.random.normal(key, (E_real, 7))

        # Output without padding
        out_no_pad = gns(node_feat, edge_idx_real, edge_feat_real)

        # Create padded version: pad to 200 edges with (0, 0) padding
        max_edges = 200
        edge_idx_padded = jnp.zeros((2, max_edges), dtype=jnp.int32)
        edge_idx_padded = edge_idx_padded.at[:, :E_real].set(edge_idx_real)
        edge_feat_padded = jnp.zeros((max_edges, 7))
        edge_feat_padded = edge_feat_padded.at[:E_real].set(edge_feat_real)

        # Output WITH padding + num_edges mask
        out_masked = gns(node_feat, edge_idx_padded, edge_feat_padded, num_edges=E_real)

        # All nodes should match (small tolerance for XLA float32 tiling differences
        # when processing 30 vs 200 edge batches through the encoder MLPs)
        np.testing.assert_allclose(
            np.asarray(out_masked["dX"]),
            np.asarray(out_no_pad["dX"]),
            atol=2e-3,
        )

    def test_padding_without_mask_contaminates(self, key):
        """Without num_edges, padding edges DO affect output (proving mask is needed)."""
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
            key=key,
        )
        N, E_real = 5, 10
        node_feat = jax.random.normal(key, (N, 9))
        edge_idx_real = jax.random.randint(key, (2, E_real), 0, N)
        edge_feat_real = jax.random.normal(key, (E_real, 7))

        out_no_pad = gns(node_feat, edge_idx_real, edge_feat_real)

        # Heavily padded version WITHOUT mask
        max_edges = 500
        edge_idx_padded = jnp.zeros((2, max_edges), dtype=jnp.int32)
        edge_idx_padded = edge_idx_padded.at[:, :E_real].set(edge_idx_real)
        edge_feat_padded = jnp.zeros((max_edges, 7))
        edge_feat_padded = edge_feat_padded.at[:E_real].set(edge_feat_real)

        out_no_mask = gns(node_feat, edge_idx_padded, edge_feat_padded)  # no num_edges

        # Node 0 output should differ (contaminated by 490 padding edges)
        diff = float(jnp.abs(out_no_mask["dX"][0] - out_no_pad["dX"][0]).max())
        assert diff > 0.01, f"Expected contamination but diff={diff}"


class TestGNSSaveLoad:
    def test_save_load_roundtrip(self, tmp_path, key):
        """Saved and loaded model produces identical output."""
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=4,
            node_latent_dim=32,
            edge_latent_dim=32,
            hidden_dim=32,
            num_mp_steps=3,
            output_dims={"dX": 3, "dP": 3, "dc": 2},
            activation="silu",
            key=key,
        )
        node_feat = jax.random.normal(key, (10, 9))
        edge_idx = jax.random.randint(key, (2, 30), 0, 10)
        edge_feat = jax.random.normal(key, (30, 4))

        out_orig = gns(node_feat, edge_idx, edge_feat)

        path = tmp_path / "model.eqx"
        gns.save(path)
        gns2 = GNS.load(path)

        out_loaded = gns2(node_feat, edge_idx, edge_feat)
        for k in out_orig:
            np.testing.assert_allclose(
                np.asarray(out_orig[k]), np.asarray(out_loaded[k]), atol=1e-6
            )

    def test_save_load_preserves_config(self, tmp_path, key):
        gns = GNS(
            node_feature_dim=5,
            edge_feature_dim=3,
            node_latent_dim=64,
            edge_latent_dim=48,
            hidden_dim=64,
            num_mp_steps=7,
            num_mlp_layers=3,
            output_dims={"dX": 3},
            activation="gelu",
            layer_norm=False,
            checkpoint_processor=True,
            key=key,
        )
        expected_config = {
            "node_feature_dim": 5,
            "edge_feature_dim": 3,
            "node_latent_dim": 64,
            "edge_latent_dim": 48,
            "hidden_dim": 64,
            "num_mp_steps": 7,
            "num_mlp_layers": 3,
            "output_dims": {"dX": 3},
            "activation": "gelu",
            "layer_norm": False,
            "checkpoint_processor": True,
        }
        path = tmp_path / "model.eqx"
        gns.save(path)
        assert {item.name for item in tmp_path.iterdir()} == {"model.eqx", "model.eqx.json"}
        saved_config = json.loads((tmp_path / "model.eqx.json").read_text())
        gns2 = GNS.load(path)

        assert gns._config == expected_config
        assert list(saved_config) == list(expected_config)
        assert saved_config == expected_config
        assert gns2._config == expected_config
        original_tree = eqx.filter(gns, eqx.is_array)
        loaded_tree = eqx.filter(gns2, eqx.is_array)
        assert jax.tree_util.tree_structure(original_tree) == jax.tree_util.tree_structure(
            loaded_tree
        )
        assert [(leaf.shape, leaf.dtype) for leaf in jax.tree_util.tree_leaves(original_tree)] == [
            (leaf.shape, leaf.dtype) for leaf in jax.tree_util.tree_leaves(loaded_tree)
        ]
        assert len(gns2.processor) == 7
        assert gns2.checkpoint_processor is True
        assert gns2.node_encoder.activation_name == "gelu"
        assert list(gns2.decoders.keys()) == ["dX"]
