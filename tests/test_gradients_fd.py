"""Numerical-oracle gradient regression tests (finite differences / gradcheck).

Why this file exists
--------------------
Every *other* gradient test in the suite is "gradient-alive" only: it asserts
``grad is not None`` / ``grad.abs().sum() > 0`` / ``isfinite``. Those catch a
**detached** graph but never a **wrong** gradient — a sign flip, a wrong
constant, a transposed Jacobian, or a broken custom autograd rule all pass
silently. The differentiable Warp autograd bridge (the most fragile numerical
machinery in the package) had *no* gradient-value coverage on the CI CPU lane
at all: its only correctness tests (``test_autodiff_grads.py``) are
``@requires_cuda``-gated and skip on CPU-only CI.

This module fills that hole. Each differentiable path is checked against a
**numerical oracle** — central finite differences (a directional VJP probe) or
:func:`torch.autograd.gradcheck` — so a regression in the gradient *value*
fails the test. It follows the ``numerics-verification`` skill:

* Pure-torch paths (losses, GNS) are dtype-preserving, so they are checked in
  **float64** at strict tolerance (rel err <= ~1e-6).
* Warp bridge / emulator tape steps run float32 kernels, so they are **float32
  smoke** checks at ``rtol ~= 1e-2`` (the skill's rule for hard-float32 paths).
  These produce correct, nonzero gradients on the Warp *CPU* backend (verified:
  no CPU/CUDA disagreement), so they run on the CI CPU lane too.
* ``waxmorph.graph`` feature builders cast to ``float()`` internally
  (``torch/graph.py``), so their FD checks are likewise float32 smoke.

A wrong-gradient regression produces a relative error of order 1 (a sign flip)
or a NaN — both orders of magnitude above these tolerances — so it fails
loudly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp
from torch.autograd import gradcheck

from waxmorph.gnn import GNS
from waxmorph.graph import build_edge_features, build_edge_index, build_node_features
from waxmorph.torch.losses import chamfer_distance, squared_loss
from waxmorph.torch.warp_autograd import WarpDiffusionStep, WarpMechStep

wp.init()

DEVICE = "cpu"
HAS_CUDA = False
try:
    if wp.is_device_available("cuda"):
        DEVICE = "cuda"
        HAS_CUDA = True
except RuntimeError:
    pass

# Always exercise CPU (the CI lane); add CUDA when this box has a GPU. Both
# devices are checked because the Warp CPU backend has historically diverged
# from CUDA on adjoint code — pinning both guards that seam.
DEVICES = ["cpu"] + (["cuda"] if HAS_CUDA else [])


# ---------------------------------------------------------------------------
# Shared finite-difference oracle
# ---------------------------------------------------------------------------


def _max_directional_rel_error(scalar_fn, x0, grad, *, h, n_dirs=5, seed=0):
    """Max relative error between analytic directional derivative and central FD.

    Compares ``g . v`` (analytic gradient projected on a random unit direction
    ``v``) against the central finite difference of ``scalar_fn`` along ``v``.
    A handful of random directions at a generic point catch essentially every
    wrong-gradient bug at a fraction of the cost of a full Jacobian.

    Args:
        scalar_fn: Maps a flat array (same shape as ``x0``) to a Python float.
            The caller is responsible for running it without building a graph.
        x0: Flat base point.
        grad: Flat analytic gradient at ``x0``.
        h: Central-difference step. Use ``~1e-6 * scale`` in float64,
            ``~1e-3`` for float32 paths (float32 cancellation floor).
        n_dirs: Number of random directions.
        seed: RNG seed for reproducible directions.

    Returns:
        The worst (largest) relative error over the sampled directions.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(grad, dtype=np.float64).ravel()
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    worst = 0.0
    for _ in range(n_dirs):
        v = rng.standard_normal(x0.size)
        v /= np.linalg.norm(v)
        fd = (scalar_fn(x0 + h * v) - scalar_fn(x0 - h * v)) / (2.0 * h)
        adv = float(g @ v)
        worst = max(worst, abs(adv - fd) / max(abs(adv), abs(fd), 1e-30))
    return worst


# ---------------------------------------------------------------------------
# Losses (float64, strict gradcheck + coincident NaN traps)
# ---------------------------------------------------------------------------


class TestLossGradientOracle:
    """Loss gradients vs finite differences (the oracle the grad-alive tests lack)."""

    def test_squared_loss_gradcheck(self):
        torch.manual_seed(0)
        pred = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
        target = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            squared_loss,
            (pred, target),
            eps=1e-6,
            atol=1e-6,
            rtol=1e-4,
            nondet_tol=0.0,
        )

    def test_chamfer_gradcheck_generic_points(self):
        """Chamfer's ``min`` is smooth away from ties — check at generic points.

        Distinct nearest neighbours make the argmin locally constant, so the
        subgradient autograd returns equals the true gradient and FD agrees.
        The seed is chosen so no query point is near-equidistant to two
        neighbours (asserted below).
        """
        rng = np.random.default_rng(4)
        pred_np = rng.standard_normal((6, 3)) + 0.3
        target_np = rng.standard_normal((7, 3)) - 0.2

        # Guard against a near-tie that would make the subgradient ambiguous.
        d = np.linalg.norm(pred_np[:, None, :] - target_np[None, :, :], axis=-1)
        by_target = np.sort(d, axis=1)
        by_pred = np.sort(d, axis=0)
        assert (by_target[:, 1] - by_target[:, 0]).min() > 1e-2
        assert (by_pred[1] - by_pred[0]).min() > 1e-2

        pred = torch.tensor(pred_np, dtype=torch.float64, requires_grad=True)
        target = torch.tensor(target_np, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            chamfer_distance,
            (pred, target),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
            nondet_tol=0.0,
        )

    @pytest.mark.parametrize(
        "loss_fn", [chamfer_distance, squared_loss], ids=["chamfer", "squared"]
    )
    def test_grad_finite_at_coincident_points(self, loss_fn):
        """Both input gradients are finite and zero at coincident clouds."""
        rng = np.random.default_rng(11)
        pred = torch.tensor(
            rng.standard_normal((6, 3)) + 0.3, dtype=torch.float64, requires_grad=True
        )
        target = pred.detach().clone().requires_grad_()
        grads = torch.autograd.grad(loss_fn(pred, target), (pred, target))
        for grad in grads:
            assert torch.isfinite(grad).all()
            torch.testing.assert_close(grad, torch.zeros_like(grad))


# ---------------------------------------------------------------------------
# GNS (float64 gradcheck through the whole message-passing backward)
# ---------------------------------------------------------------------------


def _small_gns(device, dtype=torch.float64):
    """A tiny GNS in eval mode for fast, deterministic gradchecks."""
    torch.manual_seed(0)
    gns = GNS(
        node_feature_dim=2,
        edge_feature_dim=2,
        node_latent_dim=8,
        edge_latent_dim=8,
        hidden_dim=8,
        num_mp_steps=2,
        num_mlp_layers=2,
        output_dims={"dX": 3, "dP": 3, "dc": 2},
        activation="silu",  # smooth: no ReLU kink to straddle
        layer_norm=True,
    )
    return gns.to(dtype).to(device).eval()


class TestGNSGradientOracle:
    """Numerically validate the full GNS backward, not just that grads exist."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_gns_gradcheck_wrt_inputs(self, device):
        """gradcheck d(output)/d(node_features, edge_features) — exercises every
        encoder / message-passing / decoder backward at once."""
        gns = _small_gns(device)
        rng = np.random.default_rng(2)
        n, e = 5, 8
        edge_index = torch.tensor(rng.integers(0, n, size=(2, e)), dtype=torch.long, device=device)
        node_feat = torch.tensor(
            rng.standard_normal((n, 2)) + 0.3,
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        edge_feat = torch.tensor(
            rng.standard_normal((e, 2)) + 0.3,
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def flat_out(nf, ef):
            out = gns(nf, edge_index, ef)
            return torch.cat([v.reshape(-1) for v in out.values()])

        assert gradcheck(
            flat_out, (node_feat, edge_feat), eps=1e-6, atol=1e-5, rtol=1e-3, nondet_tol=0.0
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_gns_parameter_gradient_matches_fd(self, device):
        """The gradient training actually consumes is the parameter gradient.

        gradcheck only probes input grads, so verify one representative weight's
        gradient against finite differences directly.
        """
        gns = _small_gns(device)
        rng = np.random.default_rng(3)
        n, e = 5, 8
        edge_index = torch.tensor(rng.integers(0, n, size=(2, e)), dtype=torch.long, device=device)
        node_feat = torch.tensor(
            rng.standard_normal((n, 2)) + 0.3, dtype=torch.float64, device=device
        )
        edge_feat = torch.tensor(
            rng.standard_normal((e, 2)) + 0.3, dtype=torch.float64, device=device
        )

        name, param = next(iter(gns.named_parameters()))
        p0 = param.detach().cpu().numpy().copy()

        def scalar(pnp):
            with torch.no_grad():
                param.copy_(
                    torch.tensor(pnp.reshape(param.shape), dtype=torch.float64, device=device)
                )
            out = gns(node_feat, edge_index, edge_feat)
            return float(sum(v.sum() for v in out.values()).item())

        param.grad = None
        out = gns(node_feat, edge_index, edge_feat)
        sum(v.sum() for v in out.values()).backward()
        grad = param.grad.detach().cpu().numpy()
        assert param.grad.abs().sum() > 0, f"{name}: dead parameter gradient"

        maxrel = _max_directional_rel_error(scalar, p0, grad, h=1e-6, seed=3)
        with torch.no_grad():  # restore
            param.copy_(torch.tensor(p0, dtype=torch.float64, device=device))
        assert maxrel < 1e-6, f"{name}: grad vs FD rel err {maxrel:.2e}"


# ---------------------------------------------------------------------------
# Graph feature builders (float32 smoke — internal .float() cast)
# ---------------------------------------------------------------------------


class TestGraphFeatureGradientOracle:
    """dist / angle edge-feature gradients vs FD, plus the exact node-feature grad."""

    def _triangle(self, device):
        # Three particles in mutual contact (radius 0.5) with generic pairwise
        # polarity angles (dot products 0.48..0.8, away from the acos clamp).
        pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.4, 0.0]], dtype=np.float32)
        pol = np.array([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [0.0, 0.8, 0.6]], dtype=np.float32)
        radii = torch.full((3,), 0.5, dtype=torch.float32, device=device)
        edge_index = build_edge_index(torch.tensor(pos, device=device), radii, particle_count=3)
        assert edge_index.shape[1] > 0, "expected contact edges"
        return pos, pol, radii, edge_index

    @pytest.mark.parametrize("device", DEVICES)
    def test_edge_features_grad_matches_fd_wrt_positions(self, device):
        pos, pol, _radii, edge_index = self._triangle(device)
        pol_t = torch.tensor(pol, device=device)

        def scalar(pos_flat):
            x = torch.tensor(
                pos_flat.reshape(3, 3), dtype=torch.float32, device=device, requires_grad=True
            )
            return build_edge_features(x, pol_t, edge_index, particle_count=3).sum(), x

        val, x = scalar(pos.ravel())
        val.backward()
        grad = x.grad.detach().cpu().numpy()

        def fwd(pos_flat):
            with torch.no_grad():
                x = torch.tensor(pos_flat.reshape(3, 3), dtype=torch.float32, device=device)
                return float(
                    build_edge_features(x, pol_t, edge_index, particle_count=3).sum().item()
                )

        maxrel = _max_directional_rel_error(fwd, pos.ravel(), grad, h=2e-3, seed=1)
        assert maxrel < 2e-2, f"edge-feature grad wrt positions vs FD rel err {maxrel:.2e}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_edge_features_grad_matches_fd_wrt_polarity(self, device):
        pos, pol, _radii, edge_index = self._triangle(device)
        pos_t = torch.tensor(pos, device=device)

        def scalar(pol_flat):
            p = torch.tensor(
                pol_flat.reshape(3, 3), dtype=torch.float32, device=device, requires_grad=True
            )
            return build_edge_features(pos_t, p, edge_index, particle_count=3).sum(), p

        val, p = scalar(pol.ravel())
        val.backward()
        grad = p.grad.detach().cpu().numpy()

        def fwd(pol_flat):
            with torch.no_grad():
                p = torch.tensor(pol_flat.reshape(3, 3), dtype=torch.float32, device=device)
                return float(
                    build_edge_features(pos_t, p, edge_index, particle_count=3).sum().item()
                )

        maxrel = _max_directional_rel_error(fwd, pol.ravel(), grad, h=2e-3, seed=2)
        assert maxrel < 2e-2, f"edge-feature (acos) grad wrt polarity vs FD rel err {maxrel:.2e}"

    def test_node_features_gradient_is_exact_identity(self):
        """Node features are the identity on concentrations, so d(sum)/dc == 1 exactly."""
        c = torch.randn(5, 3, dtype=torch.float32, requires_grad=True)
        build_node_features(c, particle_count=5).sum().backward()
        torch.testing.assert_close(c.grad, torch.ones_like(c))


# ---------------------------------------------------------------------------
# Warp autograd bridge (float32 smoke; runs on CPU — the CI lane)
# ---------------------------------------------------------------------------

_BRIDGE_N = 6
_BRIDGE_MOL = 2
_BRIDGE_DT = 0.01
_BRIDGE_D = 0.1


class TestWarpBridgeGradientOracle:
    """Bridge VJP vs finite differences — the coverage CI never had.

    ``WarpMechStep`` / ``WarpDiffusionStep`` record a ``wp.Tape`` on the forward
    and replay it on the backward. This checks the replayed gradient against FD
    of the forward. Runs on CPU (verified correct on the Warp CPU backend) so it
    guards the bridge on the CI CPU lane, and on CUDA when present.
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_warp_mech_step_inactive_tail_has_identity_vjp(self, device):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [7.0, -3.0, 1.0]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        radii = wp.full(3, value=0.5, dtype=wp.float32, device=device)
        f_net = wp.zeros(3, dtype=wp.vec3f, device=device, requires_grad=True)
        cotangent = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, -0.7, 1.2]],
            dtype=torch.float32,
            device=device,
        )

        output = WarpMechStep.apply(positions, radii, 2, 0.0, f_net, None)
        (output * cotangent).sum().backward()

        torch.testing.assert_close(positions.grad, cotangent, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_warp_mech_step_grad_matches_fd(self, device):
        rng = np.random.default_rng(0)
        n = _BRIDGE_N
        x0 = (rng.standard_normal((n, 3)) * 0.35 + 0.3).astype(np.float32)
        radii = np.full(n, 0.5, np.float32)
        cotangent = torch.tensor(rng.standard_normal((n, 3)).astype(np.float32), device=device)

        def scalar(x_flat, grad=False):
            xt = torch.tensor(
                x_flat.reshape(n, 3), dtype=torch.float32, device=device, requires_grad=grad
            )
            r_wp = wp.array(radii, dtype=wp.float32, device=device)
            f_net = wp.zeros(n, dtype=wp.vec3f, device=device, requires_grad=True)
            out = WarpMechStep.apply(xt, r_wp, n, _BRIDGE_DT, f_net, None)
            return (out * cotangent).sum(), xt

        val, xt = scalar(x0.ravel(), grad=True)
        val.backward()
        grad = xt.grad.detach().cpu().numpy()
        assert (
            np.abs(grad).sum() > 0
        ), "bridge returned a zero gradient (Warp CPU adjoint regression?)"

        def fwd(x_flat):
            with torch.no_grad():
                return float(scalar(x_flat, grad=False)[0].item())

        maxrel = _max_directional_rel_error(fwd, x0.ravel(), grad, h=1e-3, seed=1)
        assert maxrel < 2e-2, f"WarpMechStep VJP vs FD rel err {maxrel:.2e} (f32 smoke)"

    @pytest.mark.parametrize("device", DEVICES)
    def test_warp_diffusion_step_grad_matches_fd(self, device):
        rng = np.random.default_rng(0)
        n, mol = _BRIDGE_N, _BRIDGE_MOL
        x0 = (rng.standard_normal((n, 3)) * 0.35 + 0.3).astype(np.float32)
        c0 = (rng.standard_normal((n, mol)) * 0.4 + 0.3).astype(np.float32)
        radii = np.full(n, 0.5, np.float32)
        cotangent = torch.tensor(rng.standard_normal((n, mol)).astype(np.float32), device=device)
        x_wp = wp.array(x0, dtype=wp.vec3f, device=device)
        r_wp = wp.array(radii, dtype=wp.float32, device=device)

        def scalar(c_flat, grad=False):
            ct = torch.tensor(
                c_flat.reshape(n, mol), dtype=torch.float32, device=device, requires_grad=grad
            )
            out = WarpDiffusionStep.apply(ct, x_wp, r_wp, n, _BRIDGE_D, _BRIDGE_DT, None)
            return (out * cotangent).sum(), ct

        val, ct = scalar(c0.ravel(), grad=True)
        val.backward()
        grad = ct.grad.detach().cpu().numpy()
        assert (
            np.abs(grad).sum() > 0
        ), "bridge returned a zero gradient (Warp CPU adjoint regression?)"

        def fwd(c_flat):
            with torch.no_grad():
                return float(scalar(c_flat, grad=False)[0].item())

        maxrel = _max_directional_rel_error(fwd, c0.ravel(), grad, h=1e-3, seed=2)
        assert maxrel < 2e-2, f"WarpDiffusionStep VJP vs FD rel err {maxrel:.2e} (f32 smoke)"

    @pytest.mark.parametrize("device", DEVICES)
    def test_warp_diffusion_step_matches_linear_oracle(self, device):
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        concentrations = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        cotangent = torch.tensor(
            [[0.2, -0.3], [0.5, 0.7], [-0.4, 0.9], [1.1, -0.8]],
            dtype=torch.float32,
            device=device,
        )
        operator = torch.tensor(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.0, 0.1, 0.9, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        X = wp.array(positions, dtype=wp.vec3f, device=device)
        R = wp.full(4, value=0.5, dtype=wp.float32, device=device)

        output = WarpDiffusionStep.apply(concentrations, X, R, 3, 0.5, 0.2, None)
        (output * cotangent).sum().backward()

        torch.testing.assert_close(output, operator @ concentrations, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            concentrations.grad, operator.T @ cotangent, rtol=1e-5, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Torch <-> JAX gradient parity (the documented one-to-one invariant)
# ---------------------------------------------------------------------------


class TestTorchJaxGradientParity:
    """The torch and JAX losses are documented as mirroring one-to-one, but no
    cross-backend *gradient* parity test existed. These compare gradients at
    identical points."""

    def _inputs(self, seed, n=6, m=7):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((n, 3)) + 0.3).astype(np.float32), (
            rng.standard_normal((m, 3)) - 0.2
        ).astype(np.float32)

    def test_chamfer_gradient_matches_jax(self):
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        from waxmorph.jax.losses import chamfer_distance as jax_chamfer

        pred_np, target_np = self._inputs(seed=4)
        pred = torch.tensor(pred_np, requires_grad=True)
        target = torch.tensor(target_np, requires_grad=True)
        torch_grads = torch.autograd.grad(chamfer_distance(pred, target), (pred, target))
        jax_grads = jax.grad(jax_chamfer, argnums=(0, 1))(
            jnp.asarray(pred_np), jnp.asarray(target_np)
        )
        for torch_grad, jax_grad in zip(torch_grads, jax_grads, strict=True):
            np.testing.assert_allclose(
                torch_grad.numpy(), np.asarray(jax_grad), rtol=2e-4, atol=1e-5
            )

    def test_squared_loss_gradient_matches_jax(self):
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        from waxmorph.jax.losses import squared_loss as jax_squared

        rng = np.random.default_rng(5)
        pred_np = (rng.standard_normal((6, 3)) + 0.3).astype(np.float32)
        target_np = (rng.standard_normal((6, 3)) - 0.2).astype(np.float32)
        pred = torch.tensor(pred_np, requires_grad=True)
        target = torch.tensor(target_np, requires_grad=True)
        torch_grads = torch.autograd.grad(squared_loss(pred, target), (pred, target))
        jax_grads = jax.grad(jax_squared, argnums=(0, 1))(
            jnp.asarray(pred_np), jnp.asarray(target_np)
        )
        for torch_grad, jax_grad in zip(torch_grads, jax_grads, strict=True):
            np.testing.assert_allclose(
                torch_grad.numpy(), np.asarray(jax_grad), rtol=2e-4, atol=1e-5
            )

    def test_jax_chamfer_gradient_finite_at_coincident_points(self):
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        from waxmorph.jax.losses import chamfer_distance as jax_chamfer

        rng = np.random.default_rng(11)
        pred_np = (rng.standard_normal((6, 3)) + 0.3).astype(np.float32)
        grads = jax.grad(jax_chamfer, argnums=(0, 1))(jnp.asarray(pred_np), jnp.asarray(pred_np))
        for grad in grads:
            assert np.isfinite(grad).all()
            np.testing.assert_array_equal(np.asarray(grad), np.zeros_like(pred_np))
