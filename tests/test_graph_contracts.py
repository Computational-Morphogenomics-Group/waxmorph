import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import waxmorph.jax.graph as jax_graph
import waxmorph.torch.graph as torch_graph


@pytest.fixture(params=("torch", "jax"))
def backend(request):
    if request.param == "torch":
        return request.param, torch_graph, torch.as_tensor
    return request.param, jax_graph, jnp.asarray


def _array(backend, values, dtype=np.float32):
    return backend[2](np.asarray(values, dtype=dtype))


def _edge_index(backend, *args, **kwargs):
    result = backend[1].build_edge_index(*args, **kwargs)
    return result if backend[0] == "torch" else result[0]


@pytest.mark.parametrize(
    ("positions", "radii", "eps_dist", "match"),
    [
        (np.zeros((2, 2)), np.ones(2), 0.0, "positions"),
        (np.zeros((0, 2)), np.ones(0), 0.0, "positions"),
        (np.zeros((2, 3)), np.ones((2, 1)), 0.0, "radii"),
        (np.zeros((2, 3)), np.ones(3), 0.0, "length"),
        (np.zeros((0, 3)), np.ones(1), 0.0, "length"),
        (np.array([[np.nan, 0.0, 0.0]]), np.ones(1), 0.0, "positions"),
        (np.zeros((1, 3)), np.array([np.inf]), 0.0, "radii"),
        (np.zeros((2, 3)), np.array([-0.1, 0.5]), 0.0, "radii"),
        (np.zeros((1, 3)), np.ones(1), -0.1, "eps_dist"),
        (np.zeros((1, 3)), np.ones(1), np.nan, "eps_dist"),
        (np.zeros((1, 3)), np.ones(1), np.inf, "eps_dist"),
    ],
)
def test_edge_index_rejects_invalid_geometry(backend, positions, radii, eps_dist, match):
    with pytest.raises(ValueError, match=match):
        _edge_index(
            backend,
            _array(backend, positions),
            _array(backend, radii),
            particle_count=0,
            eps_dist=eps_dist,
        )


@pytest.mark.parametrize("particle_count", [True, np.bool_(True), 1.5])
def test_build_graph_rejects_nonintegral_particle_count(backend, particle_count):
    state = (
        _array(backend, np.zeros((2, 3))),
        _array(backend, np.ones((2, 3))),
        _array(backend, np.ones(2)),
        _array(backend, np.ones((2, 1))),
    )

    with pytest.raises(TypeError, match="particle_count"):
        backend[1].build_graph(*state[:3], particle_count=particle_count, c=state[3])


@pytest.mark.parametrize("particle_count", [0, -1, np.int64(2)])
def test_nonpositive_and_numpy_particle_counts_use_full_input(backend, particle_count):
    c = _array(backend, [[1.0], [2.0]])

    assert backend[1].build_node_features(c, particle_count).shape == (2, 1)


@pytest.mark.parametrize("short_input", ["positions", "polarities", "radii", "concentrations"])
def test_build_graph_rejects_particle_count_beyond_input(backend, short_input):
    lengths = {name: 2 for name in ("positions", "polarities", "radii", "concentrations")}
    lengths[short_input] = 1
    x = _array(backend, np.zeros((lengths["positions"], 3)))
    p = _array(backend, np.ones((lengths["polarities"], 3)))
    r = _array(backend, np.ones(lengths["radii"]))
    c = _array(backend, np.ones((lengths["concentrations"], 1)))

    with pytest.raises(ValueError, match="particle_count"):
        backend[1].build_graph(x, p, r, particle_count=2, c=c)


def test_build_graph_requires_matching_full_lengths(backend):
    x = _array(backend, np.zeros((2, 3)))
    p = _array(backend, np.ones((2, 3)))
    r = _array(backend, np.ones(2))
    c = _array(backend, np.ones((1, 1)))

    with pytest.raises(ValueError, match="length"):
        backend[1].build_graph(x, p, r, particle_count=0, c=c)


def test_positive_particle_count_allows_different_capacities(backend):
    x = _array(backend, np.zeros((3, 3)))
    p = _array(backend, np.ones((4, 3)))
    r = _array(backend, np.ones(5))
    c = _array(backend, np.ones((6, 2)))

    node_features, *_ = backend[1].build_graph(x, p, r, particle_count=2, c=c)

    assert node_features.shape == (2, 2)


def test_geometry_validation_ignores_inactive_capacity(backend):
    x = _array(backend, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    r = _array(backend, [0.5, 0.5, -1.0])

    assert _edge_index(backend, x, r, particle_count=2).shape == (2, 2)


def test_build_graph_requires_concentrations_before_topology(backend):
    malformed_positions = _array(backend, np.zeros(2))
    p = _array(backend, np.ones((2, 3)))
    r = _array(backend, np.ones(2))

    with pytest.raises(TypeError, match=r"build_graph\(\) requires `c`"):
        backend[1].build_graph(malformed_positions, p, r)


@pytest.mark.parametrize("shape", [(), (2, 1, 1)])
def test_node_features_reject_invalid_shape(backend, shape):
    c = _array(backend, np.ones(shape))

    with pytest.raises(ValueError, match="concentrations"):
        backend[1].build_node_features(c, particle_count=0)


@pytest.mark.parametrize(
    ("positions", "polarities", "match"),
    [
        (np.zeros((2, 2)), np.ones((2, 3)), "positions"),
        (np.zeros((2, 3)), np.ones((2, 2)), "polarities"),
        (np.zeros((2, 3)), np.ones((1, 3)), "length"),
    ],
)
def test_edge_features_reject_invalid_state_shapes(backend, positions, polarities, match):
    edge_index = _array(backend, np.empty((2, 0)), dtype=np.int64)

    with pytest.raises(ValueError, match=match):
        backend[1].build_edge_features(
            _array(backend, positions),
            _array(backend, polarities),
            edge_index,
            particle_count=0,
        )


@pytest.mark.parametrize(
    ("edge_index", "dtype", "error"),
    [
        (np.zeros((3, 1)), np.int64, ValueError),
        (np.array([[0], [-1]]), np.int64, ValueError),
        (np.array([[0], [2]]), np.int64, ValueError),
        (np.array([[0.0], [1.0]]), np.float32, TypeError),
        (np.array([[False], [True]]), np.bool_, TypeError),
    ],
)
def test_edge_features_reject_invalid_indices(backend, edge_index, dtype, error):
    x = _array(backend, np.zeros((2, 3)))
    p = _array(backend, np.ones((2, 3)))

    with pytest.raises(error, match="edge_index"):
        backend[1].build_edge_features(
            x,
            p,
            _array(backend, edge_index, dtype=dtype),
            particle_count=2,
        )


def test_empty_graph_contract(backend):
    x = _array(backend, np.empty((0, 3)))
    p = _array(backend, np.empty((0, 3)))
    r = _array(backend, np.empty(0))
    c = _array(backend, np.empty((0, 2)))

    result = backend[1].build_graph(x, p, r, particle_count=0, c=c)

    assert result[0].shape == (0, 2)
    assert result[1].shape == (2, 0)
    assert result[2].shape == (0, 2)
    if backend[0] == "jax":
        assert int(result[3]) == 0


def test_jax_padded_empty_graph_contract():
    result = jax_graph.build_graph(
        jnp.empty((0, 3)),
        jnp.empty((0, 3)),
        jnp.empty((0,)),
        particle_count=0,
        c=jnp.empty((0, 2)),
        max_edges=4,
    )

    assert result[1].shape == (2, 4)
    assert result[2].shape == (4, 2)
    np.testing.assert_array_equal(result[2], 0.0)
    assert int(result[3]) == 0


def test_nonunit_polarities_use_raw_dot_product(backend):
    x = _array(backend, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    p = _array(backend, [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    edge_index = _array(backend, [[0], [1]], dtype=np.int64)

    features = backend[1].build_edge_features(x, p, edge_index, particle_count=2)

    assert float(features[0, 1]) == pytest.approx(np.arccos(0.25), abs=1e-6)


def test_jax_edge_validation_preserves_jit():
    x = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    p = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    edge_index = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    build = jax.jit(lambda edges: jax_graph.build_edge_features(x, p, edges, 2))

    assert build(edge_index).shape == (2, 2)
