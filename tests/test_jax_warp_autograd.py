"""Tests for the experimental JAX/Warp autodiff bridge."""

import jax
import jax.numpy as jnp
import pytest
import warp as wp

from waxmorph.jax.warp_autograd import warp_diffusion_step, warp_mech_step


def _has_jax_warp_cuda() -> bool:
    try:
        return bool(wp.is_device_available("cuda") and jax.devices("gpu"))
    except Exception:
        return False


def test_bridge_rejects_cpu_device():
    x = jnp.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=jnp.float32)
    r = jnp.array([0.5, 0.5], dtype=jnp.float32)
    pairs = jnp.array([0], dtype=jnp.int32)

    with pytest.raises(RuntimeError, match="CUDA"):
        warp_mech_step(x, r, pairs, jnp.array([1], dtype=jnp.int32), 1e-2, device="cpu")


@pytest.mark.skipif(not _has_jax_warp_cuda(), reason="JAX/Warp bridge requires CUDA")
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


@pytest.mark.skipif(not _has_jax_warp_cuda(), reason="JAX/Warp bridge requires CUDA")
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


@pytest.mark.skipif(not _has_jax_warp_cuda(), reason="JAX/Warp bridge requires CUDA")
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


@pytest.mark.skipif(not _has_jax_warp_cuda(), reason="JAX/Warp bridge requires CUDA")
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
