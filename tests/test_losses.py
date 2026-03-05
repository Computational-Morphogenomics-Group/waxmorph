"""Tests for emulation loss functions with gradient flow through GNS."""

import pytest
import torch

from waxmorph.gnn import GNS
from waxmorph.losses import chamfer_distance, squared_loss


class TestSquaredLoss:
    def test_zero_when_identical(self):
        x = torch.randn(20, 3)
        assert squared_loss(x, x).item() == pytest.approx(0.0, abs=1e-7)

    def test_positive(self):
        a = torch.zeros(5, 3)
        b = torch.ones(5, 3)
        loss = squared_loss(a, b)
        # ||0 - 1||_F^2 = 5 * 3 = 15
        assert loss.item() == pytest.approx(15.0, abs=1e-5)

    def test_gradient_to_pred(self):
        pred = torch.randn(10, 3, requires_grad=True)
        target = torch.randn(10, 3)
        loss = squared_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_gradient_through_gns(self):
        """Squared loss gradients flow through GNS to all parameters."""
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
        )
        node_feat = torch.randn(20, 9)
        edge_idx = torch.randint(0, 20, (2, 60))
        edge_feat = torch.randn(60, 7)

        X_init = torch.randn(20, 3)
        X_target = torch.randn(20, 3)

        out = gns(node_feat, edge_idx, edge_feat)
        X_pred = X_init + out["dX"]

        loss = squared_loss(X_pred, X_target)
        loss.backward()

        for name, p in gns.named_parameters():
            assert p.grad is not None, f"No grad on {name}"
            assert p.grad.abs().sum() > 0, f"Zero grad on {name}"


class TestChamferDistance:
    def test_zero_when_identical(self):
        x = torch.randn(20, 3)
        assert chamfer_distance(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self):
        a = torch.randn(15, 3)
        b = torch.randn(15, 3)
        assert chamfer_distance(a, b).item() == pytest.approx(
            chamfer_distance(b, a).item(), abs=1e-5
        )

    def test_different_counts(self):
        """Chamfer distance works with different N and M."""
        a = torch.randn(10, 3)
        b = torch.randn(25, 3)
        loss = chamfer_distance(a, b)
        assert loss.item() > 0

    def test_gradient_to_pred(self):
        pred = torch.randn(10, 3, requires_grad=True)
        target = torch.randn(10, 3)
        loss = chamfer_distance(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_gradient_through_gns(self):
        """Chamfer loss gradients flow through GNS to all parameters."""
        gns = GNS(
            node_feature_dim=9,
            edge_feature_dim=7,
            num_mp_steps=2,
            output_dims={"dX": 3},
        )
        node_feat = torch.randn(20, 9)
        edge_idx = torch.randint(0, 20, (2, 60))
        edge_feat = torch.randn(60, 7)

        X_init = torch.randn(20, 3)
        X_target = torch.randn(20, 3)

        out = gns(node_feat, edge_idx, edge_feat)
        X_pred = X_init + out["dX"]

        loss = chamfer_distance(X_pred, X_target)
        loss.backward()

        for name, p in gns.named_parameters():
            assert p.grad is not None, f"No grad on {name}"
            assert p.grad.abs().sum() > 0, f"Zero grad on {name}"

    def test_known_value(self):
        """Two point clouds with a known Chamfer distance."""
        # pred: single point at origin, target: single point at (3, 4, 0)
        pred = torch.tensor([[0.0, 0.0, 0.0]])
        target = torch.tensor([[3.0, 4.0, 0.0]])
        # dist = 5.0, chamfer = (5 + 5) / 1 = 10.0
        assert chamfer_distance(pred, target).item() == pytest.approx(10.0, abs=1e-5)
