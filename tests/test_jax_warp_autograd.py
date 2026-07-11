"""Tests for the JAX/Warp autodiff bridge."""

import sys
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

import waxmorph.jax.warp_autograd as warp_autograd
from waxmorph.jax.warp_autograd import warp_diffusion_step, warp_mech_step


def _has_jax_warp_cuda() -> bool:
    try:
        return bool(wp.is_device_available("cuda") and jax.devices("gpu"))
    except Exception:
        return False


# The JAX/Warp bridge is CUDA-only *by design* (``warp_mech_step`` raises on a
# CPU device — see ``test_bridge_rejects_cpu_device``). Beyond needing a GPU,
# these skip whenever JAX has no CUDA device — and ``tests/conftest.py`` sets
# ``JAX_PLATFORMS=cpu`` by default, which hides the GPU from JAX. So on a plain
# CI run (and even on a GPU box under the default env) every gradient test below
# silently skips: that is the "CI false confidence" this bridge historically
# carried. A GPU lane must export ``JAX_PLATFORMS`` off ``cpu`` (e.g. ``=cuda``)
# to actually exercise the bridge and its gradients. The torch twin's gradients
# are covered on the CPU lane by ``tests/test_gradients_fd.py``.
_requires_jax_cuda = pytest.mark.skipif(
    not _has_jax_warp_cuda(),
    reason="JAX/Warp bridge requires a CUDA JAX device (JAX_PLATFORMS!=cpu); see module note",
)


def test_jax_kernel_loader_prefers_public_api(monkeypatch):
    public_jax_kernel = object()
    monkeypatch.setattr(warp_autograd.wp, "jax_kernel", public_jax_kernel)
    monkeypatch.setitem(sys.modules, "warp.jax_experimental", None)
    monkeypatch.setitem(sys.modules, "warp.jax_experimental.ffi", None)
    warp_autograd._load_jax_kernel.cache_clear()
    try:
        assert warp_autograd._load_jax_kernel() is public_jax_kernel
    finally:
        warp_autograd._load_jax_kernel.cache_clear()


def test_jax_kernel_loader_supports_legacy_api(monkeypatch):
    legacy_jax_kernel = object()
    ffi = ModuleType("warp.jax_experimental.ffi")
    ffi.jax_kernel = legacy_jax_kernel
    experimental = ModuleType("warp.jax_experimental")
    experimental.ffi = ffi

    monkeypatch.delattr(warp_autograd.wp, "jax_kernel", raising=False)
    monkeypatch.setitem(sys.modules, "warp.jax_experimental", experimental)
    monkeypatch.setitem(sys.modules, "warp.jax_experimental.ffi", ffi)
    warp_autograd._load_jax_kernel.cache_clear()
    try:
        assert warp_autograd._load_jax_kernel() is legacy_jax_kernel
    finally:
        warp_autograd._load_jax_kernel.cache_clear()


def _fd_directional_rel_error(loss_fn, x0, grad, *, h=1e-3, n_dirs=4, seed=0):
    """Max rel error between the JAX VJP and central finite differences.

    ``loss_fn`` maps a float32 JAX array to a scalar; ``grad`` is ``jax.grad``'s
    output at ``x0``. Float32 kernels put this at "smoke" precision (rtol ~1e-2,
    per the numerics-verification skill), but a wrong custom VJP (sign flip,
    dropped term) yields an order-1 error that this still catches.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(grad, dtype=np.float64).ravel()
    base = np.asarray(x0, dtype=np.float64)
    worst = 0.0
    for _ in range(n_dirs):
        v = rng.standard_normal(base.size)
        v /= np.linalg.norm(v)
        xp = jnp.asarray((base.ravel() + h * v).reshape(base.shape), dtype=jnp.float32)
        xm = jnp.asarray((base.ravel() - h * v).reshape(base.shape), dtype=jnp.float32)
        fd = (float(loss_fn(xp)) - float(loss_fn(xm))) / (2.0 * h)
        worst = max(worst, abs(float(g @ v) - fd) / max(abs(float(g @ v)), abs(fd), 1e-30))
    return worst


def test_bridge_rejects_cpu_device():
    x = jnp.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=jnp.float32)
    r = jnp.array([0.5, 0.5], dtype=jnp.float32)
    pairs = jnp.array([0], dtype=jnp.int32)

    with pytest.raises(RuntimeError, match="CUDA"):
        warp_mech_step(x, r, pairs, jnp.array([1], dtype=jnp.int32), 1e-2, device="cpu")


@_requires_jax_cuda
def test_mechanics_bridge_has_jax_gradients():
    with jax.default_device(jax.devices("gpu")[0]):
        x = jnp.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=jnp.float32)
        r = jnp.array([0.5, 0.5], dtype=jnp.float32)
        pair_i = jnp.array([0], dtype=jnp.int32)
        pair_j = jnp.array([1], dtype=jnp.int32)

    def loss_fn(x_in):
        x_out = warp_mech_step(x_in, r, pair_i, pair_j, 1e-2, device="cuda")
        return jnp.sum(x_out**2)

    grad = jax.grad(loss_fn)(x)
    assert jnp.isfinite(grad).all()
    assert jnp.abs(grad).sum() > 0
    # Numerical oracle: the custom VJP must match finite differences, not merely
    # be nonzero/finite (a sign-flipped or dropped-term VJP passes grad-alive).
    maxrel = _fd_directional_rel_error(loss_fn, np.asarray(x), np.asarray(grad))
    assert maxrel < 2e-2, f"JAX mech-bridge VJP vs FD rel err {maxrel:.2e} (f32 smoke)"


@_requires_jax_cuda
def test_diffusion_bridge_has_jax_gradients():
    with jax.default_device(jax.devices("gpu")[0]):
        c = jnp.array([[1.0, 0.0], [0.0, 2.0]], dtype=jnp.float32)
        pair_i = jnp.array([0], dtype=jnp.int32)
        pair_j = jnp.array([1], dtype=jnp.int32)

    def loss_fn(c_in):
        c_out = warp_diffusion_step(c_in, pair_i, pair_j, 0.1, 1e-2, device="cuda")
        return jnp.sum(c_out**2)

    grad = jax.grad(loss_fn)(c)
    assert jnp.isfinite(grad).all()
    assert jnp.abs(grad).sum() > 0
    # Numerical oracle: the custom VJP must match finite differences, not merely
    # be nonzero/finite (a sign-flipped or dropped-term VJP passes grad-alive).
    maxrel = _fd_directional_rel_error(loss_fn, np.asarray(c), np.asarray(grad))
    assert maxrel < 2e-2, f"JAX diffusion-bridge VJP vs FD rel err {maxrel:.2e} (f32 smoke)"


@_requires_jax_cuda
def test_padded_mechanics_pairs_match_unpadded():
    with jax.default_device(jax.devices("gpu")[0]):
        x = jnp.array(
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        r = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)
        pair_i = jnp.array([0], dtype=jnp.int32)
        pair_j = jnp.array([1], dtype=jnp.int32)
        pair_i_padded = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        pair_j_padded = jnp.array([1, 0, 0, 0], dtype=jnp.int32)

    expected = warp_mech_step(x, r, pair_i, pair_j, 1e-2, device="cuda")
    actual = warp_mech_step(
        x,
        r,
        pair_i_padded,
        pair_j_padded,
        1e-2,
        num_pairs=jnp.array(1, dtype=jnp.int32),
        device="cuda",
    )
    assert jnp.allclose(actual, expected, atol=1e-6)


@_requires_jax_cuda
def test_padded_diffusion_pairs_match_unpadded_and_zero_pairs_noop():
    with jax.default_device(jax.devices("gpu")[0]):
        c = jnp.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        pair_i = jnp.array([0], dtype=jnp.int32)
        pair_j = jnp.array([1], dtype=jnp.int32)
        pair_i_padded = jnp.array([0, 0, 0], dtype=jnp.int32)
        pair_j_padded = jnp.array([1, 0, 0], dtype=jnp.int32)

    expected = warp_diffusion_step(c, pair_i, pair_j, 0.1, 1e-2, device="cuda")
    actual = warp_diffusion_step(
        c,
        pair_i_padded,
        pair_j_padded,
        0.1,
        1e-2,
        num_pairs=jnp.array(1, dtype=jnp.int32),
        device="cuda",
    )
    noop = warp_diffusion_step(
        c,
        pair_i_padded,
        pair_j_padded,
        0.1,
        1e-2,
        num_pairs=jnp.array(0, dtype=jnp.int32),
        device="cuda",
    )

    assert jnp.allclose(actual, expected, atol=1e-6)
    assert jnp.allclose(noop, c, atol=1e-6)
