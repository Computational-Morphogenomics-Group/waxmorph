"""Tests for the GNS architecture."""

import pytest
import torch

from waxmorph.gnn import GNS, GraphNetworkBlock


class TestGraphNetworkBlock:
    def test_output_shapes(self):
        block = GraphNetworkBlock(
            node_latent_dim=64, edge_latent_dim=32, hidden_dim=64, num_mlp_layers=2
        )
        N, E = 20, 50
        node_latent = torch.randn(N, 64)
        edge_latent = torch.randn(E, 32)
        edge_index = torch.randint(0, N, (2, E))

        node_out, edge_out = block(node_latent, edge_latent, edge_index)

        assert node_out.shape == (N, 64)
        assert edge_out.shape == (E, 32)

    def test_residual_connection(self):
        block = GraphNetworkBlock(
            node_latent_dim=16, edge_latent_dim=16, hidden_dim=16, num_mlp_layers=1
        )
        N, E = 5, 10
        node_latent = torch.randn(N, 16)
        edge_latent = torch.randn(E, 16)
        edge_index = torch.randint(0, N, (2, E))

        # Zero out MLP weights — residual should pass through identity
        with torch.no_grad():
            for p in block.edge_mlp.parameters():
                p.zero_()
            for p in block.node_mlp.parameters():
                p.zero_()

        node_out, edge_out = block(node_latent, edge_latent, edge_index)

        torch.testing.assert_close(edge_out, edge_latent)
        # Node residual won't be exact identity because aggregation of
        # zero-weight edge output still adds to the node input, but the
        # node MLP output itself is zero, so residual = node_latent.
        torch.testing.assert_close(node_out, node_latent)


class TestGNS:
    def test_forward_shapes(self):
        N, num_molecules = 50, 2
        E = 200
        F_node = 9 + num_molecules
        F_edge = 7

        gns = GNS(
            node_feature_dim=F_node,
            edge_feature_dim=F_edge,
            num_mp_steps=3,
            output_dims={"dX": 3, "dP": 3, "dc": num_molecules},
        )

        node_feat = torch.randn(N, F_node)
        edge_idx = torch.randint(0, N, (2, E))
        edge_feat = torch.randn(E, F_edge)

        out = gns(node_feat, edge_idx, edge_feat)

        assert set(out.keys()) == {"dX", "dP", "dc"}
        assert out["dX"].shape == (N, 3)
        assert out["dP"].shape == (N, 3)
        assert out["dc"].shape == (N, num_molecules)

    def test_default_output_dims(self):
        gns = GNS(node_feature_dim=9, edge_feature_dim=7, num_mp_steps=1)
        node_feat = torch.randn(10, 9)
        edge_idx = torch.randint(0, 10, (2, 30))
        edge_feat = torch.randn(30, 7)

        out = gns(node_feat, edge_idx, edge_feat)

        assert set(out.keys()) == {"dX", "dP", "dc"}

    def test_gradient_flow(self):
        gns = GNS(
            node_feature_dim=11,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
        )
        node_feat = torch.randn(20, 11, requires_grad=True)
        edge_idx = torch.randint(0, 20, (2, 60))
        edge_feat = torch.randn(60, 7)

        out = gns(node_feat, edge_idx, edge_feat)
        loss = out["dX"].sum()
        loss.backward()

        assert node_feat.grad is not None
        assert node_feat.grad.shape == (20, 11)

    def test_no_edges(self):
        """GNS should handle graphs with zero edges gracefully."""
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
        )
        node_feat = torch.randn(5, 9)
        edge_idx = torch.zeros(2, 0, dtype=torch.long)
        edge_feat = torch.zeros(0, 7)

        out = gns(node_feat, edge_idx, edge_feat)

        assert out["dX"].shape == (5, 3)

    def test_single_particle(self):
        """Single particle with no neighbors."""
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
        )
        node_feat = torch.randn(1, 9)
        edge_idx = torch.zeros(2, 0, dtype=torch.long)
        edge_feat = torch.zeros(0, 7)

        out = gns(node_feat, edge_idx, edge_feat)

        assert out["dX"].shape == (1, 3)

    @pytest.mark.parametrize("num_mp_steps", [1, 5, 10])
    def test_variable_mp_steps(self, num_mp_steps):
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=num_mp_steps,
            output_dims={"dX": 3},
        )
        assert len(gns.processor) == num_mp_steps

        out = gns(torch.randn(10, 9), torch.randint(0, 10, (2, 30)), torch.randn(30, 7))
        assert out["dX"].shape == (10, 3)

    def test_checkpoint_processor_same_output(self):
        """Checkpointed and non-checkpointed GNS produce identical outputs."""
        torch.manual_seed(0)
        kwargs = dict(
            node_feature_dim=11,
            edge_feature_dim=7,
            num_mp_steps=3,
            output_dims={"dX": 3},
            node_latent_dim=32,
            edge_latent_dim=32,
            hidden_dim=32,
        )
        gns_plain = GNS(**kwargs, checkpoint_processor=False)
        gns_ckpt = GNS(**kwargs, checkpoint_processor=True)
        # Share weights
        gns_ckpt.load_state_dict(gns_plain.state_dict())

        node_feat = torch.randn(15, 11)
        edge_idx = torch.randint(0, 15, (2, 40))
        edge_feat = torch.randn(40, 7)

        out_plain = gns_plain(node_feat, edge_idx, edge_feat)
        out_ckpt = gns_ckpt(node_feat, edge_idx, edge_feat)

        torch.testing.assert_close(out_plain["dX"], out_ckpt["dX"])

    def test_checkpoint_processor_gradient_flow(self):
        """Checkpointed GNS still propagates gradients."""
        gns = GNS(
            node_feature_dim=11,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
            checkpoint_processor=True,
        )
        node_feat = torch.randn(20, 11, requires_grad=True)
        edge_idx = torch.randint(0, 20, (2, 60))
        edge_feat = torch.randn(60, 7)

        out = gns(node_feat, edge_idx, edge_feat)
        loss = out["dX"].sum()
        loss.backward()

        assert node_feat.grad is not None
        assert node_feat.grad.shape == (20, 11)


class TestGNSSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        """Saved and loaded model produces identical output."""
        torch.manual_seed(42)
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=4,
            node_latent_dim=32,
            edge_latent_dim=32,
            hidden_dim=32,
            num_mp_steps=3,
            output_dims={"dX": 3, "dP": 3, "dc": 2},
            activation="silu",
        )
        node_feat = torch.randn(10, 9)
        edge_idx = torch.randint(0, 10, (2, 30))
        edge_feat = torch.randn(30, 4)

        out_orig = gns(node_feat, edge_idx, edge_feat)

        path = tmp_path / "model.pt"
        gns.save(path)
        gns2 = GNS.load(path)

        out_loaded = gns2(node_feat, edge_idx, edge_feat)
        for key in out_orig:
            torch.testing.assert_close(out_orig[key], out_loaded[key])

    def test_save_load_preserves_config(self, tmp_path):
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
        assert gns._constructor_config() == expected_config

        path = tmp_path / "model.pt"
        gns.save(path)
        payload = torch.load(path, weights_only=False, map_location="cpu")
        gns2 = GNS.load(path)

        assert list(payload) == ["config", "state_dict"]
        assert list(payload["config"]) == list(expected_config)
        assert payload["config"] == expected_config
        assert gns2._constructor_config() == expected_config
        assert len(gns2.processor) == 7
        assert gns2.checkpoint_processor is True
        assert gns2.node_encoder.activation_name == "gelu"
        assert list(gns2.decoders.keys()) == ["dX"]

    def test_save_load_preserves_depth_one_config(self, tmp_path):
        gns = GNS(
            node_feature_dim=4,
            edge_feature_dim=6,
            node_latent_dim=5,
            edge_latent_dim=7,
            hidden_dim=13,
            num_mp_steps=0,
            num_mlp_layers=1,
            activation="tanh",
            checkpoint_processor=True,
        )
        expected_config = {
            "node_feature_dim": 4,
            "edge_feature_dim": 6,
            "node_latent_dim": 5,
            "edge_latent_dim": 7,
            "hidden_dim": 13,
            "num_mp_steps": 0,
            "num_mlp_layers": 1,
            "output_dims": {"dX": 3, "dP": 3, "dc": 2},
            "activation": "tanh",
            "layer_norm": True,
            "checkpoint_processor": True,
        }
        path = tmp_path / "model.pt"

        gns.save(path)
        loaded = GNS.load(path)

        assert gns._constructor_config() == expected_config
        assert loaded._constructor_config() == expected_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory test")
class TestTBPTTMemoryScaling:
    """Verify truncated BPTT bounds memory by K, not T."""

    @staticmethod
    def _rollout_peak_memory(model, T, K, N=200, E=800, F_node=11, F_edge=7):
        """Run a simulated rollout and return peak GPU memory allocated."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        total_dX = torch.zeros(N, 3, device="cuda")
        for t in range(T):
            node_feat = torch.randn(N, F_node, device="cuda")
            edge_idx = torch.randint(0, N, (2, E), device="cuda")
            edge_feat = torch.randn(E, F_edge, device="cuda")

            out = model(node_feat, edge_idx, edge_feat)
            total_dX = total_dX + out["dX"]

            if (t + 1) % K == 0 and t < T - 1:
                total_dX = total_dX.detach()

        loss = total_dX.sum()
        loss.backward()

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return peak

    def test_memory_constant_across_t(self):
        """Peak memory must be constant (within noise) as T grows with fixed K.

        TBPTT detaches every K steps, so the autograd graph never exceeds K
        steps of history.  We test T=20, T=40, T=80 and assert peak memory
        stays at exactly the T=20 baseline.
        """
        K = 5
        model = GNS(
            node_feature_dim=11,
            edge_feature_dim=7,
            num_mp_steps=4,
            output_dims={"dX": 3},
            node_latent_dim=64,
            edge_latent_dim=64,
            hidden_dim=64,
            checkpoint_processor=True,
        ).cuda()

        # Warmup — let CUDA caching allocator stabilise
        self._rollout_peak_memory(model, T=K, K=K)
        model.zero_grad(set_to_none=True)

        peaks = {}
        for T in (20, 40, 80):
            peaks[T] = self._rollout_peak_memory(model, T=T, K=K)
            model.zero_grad(set_to_none=True)

        baseline = peaks[20]
        for T, peak in peaks.items():
            ratio = peak / baseline
            assert ratio < 1.0000001, (
                f"T={T}: peak memory {peak / 1e6:.0f} MB is {ratio:.2f}x the "
                f"T=20 baseline ({baseline / 1e6:.0f} MB). "
                f"Expected ~1.0x with TBPTT K={K}."
            )
