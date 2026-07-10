import dataclasses
import importlib

import numpy as np
import pytest

from waxmorph._train_core import _prepare_training_inputs

_DEFAULT_TARGETS = object()


@pytest.fixture(params=("torch", "jax"))
def backend(request):
    return request.param, importlib.import_module(f"waxmorph.{request.param}.train")


def _inputs():
    return {
        "source_pos": np.array(
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.25, 0.4, 0.0]],
            dtype=np.float64,
        ),
        "polarities": np.array(
            [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.0, 4.0, 0.0]],
            dtype=np.float64,
        ),
        "c": np.array([[0.2, 0.1], [0.05, 0.3], [0.4, 0.5]], dtype=np.float64),
        "radii": np.full(3, 0.5, dtype=np.float64),
    }


def _config(module):
    return module.TrainConfig(
        n_epochs=1,
        t_rollout=2,
        mech_steps=0,
        diff_steps=0,
        log_every=1,
    )


def _call_train(backend, *, config=None, targets=_DEFAULT_TARGETS, **updates):
    name, module = backend
    values = _inputs()
    values.update(updates)
    if config is None:
        config = _config(module)
    if targets is _DEFAULT_TARGETS:
        targets = [(0, values["source_pos"].copy())]
    kwargs = dict(values, targets=targets, config=config, device="cpu")

    def loss_fn(*_):
        return None

    if name == "torch":
        return module.train(object(), object(), loss_fn, **kwargs)
    return module.train(object(), object(), object(), loss_fn, **kwargs)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("n_epochs", True, TypeError),
        ("n_epochs", 1.5, TypeError),
        ("n_epochs", 0, ValueError),
        ("t_rollout", True, TypeError),
        ("t_rollout", 1.5, TypeError),
        ("t_rollout", 0, ValueError),
        ("log_every", True, TypeError),
        ("log_every", 1.5, TypeError),
        ("log_every", 0, ValueError),
        ("mech_steps", True, TypeError),
        ("mech_steps", 1.5, TypeError),
        ("mech_steps", -1, ValueError),
        ("diff_steps", True, TypeError),
        ("diff_steps", 1.5, TypeError),
        ("diff_steps", -1, ValueError),
        ("dt_mech", -0.1, ValueError),
        ("dt_diff", np.nan, ValueError),
        ("dt_gns", np.inf, ValueError),
        ("D_emu", -1.0, ValueError),
        ("lambda_reg", np.nan, ValueError),
        ("grad_clip_norm", 0.0, ValueError),
        ("grad_clip_norm", -1.0, ValueError),
        ("grad_clip_norm", np.nan, ValueError),
        ("grad_clip_norm", np.inf, ValueError),
        ("grad_clip_norm", True, TypeError),
    ],
)
def test_train_rejects_invalid_config(backend, field, value, error):
    config = dataclasses.replace(_config(backend[1]), **{field: value})

    with pytest.raises(error, match=field):
        _call_train(backend, config=config)


@pytest.mark.parametrize(
    ("value", "error"),
    [(0, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_jax_train_rejects_invalid_max_edges_factor(value, error):
    backend = ("jax", importlib.import_module("waxmorph.jax.train"))
    config = dataclasses.replace(_config(backend[1]), max_edges_factor=value)

    with pytest.raises(error, match="max_edges_factor"):
        _call_train(backend, config=config)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("source_pos", np.empty((0, 3))),
        ("source_pos", np.zeros((3, 2))),
        ("source_pos", np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        ("source_pos", np.array([[1e300, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        ("polarities", np.zeros((3, 2))),
        ("polarities", np.zeros((2, 3))),
        ("polarities", np.array([[np.inf, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
        ("c", np.zeros(3)),
        ("c", np.zeros((2, 2))),
        ("c", np.array([[np.nan], [0.0], [0.0]])),
        ("radii", np.zeros((3, 1))),
        ("radii", np.zeros(2)),
        ("radii", np.array([0.5, -0.1, 0.5])),
        ("radii", np.array([0.5, -1e-50, 0.5])),
        ("radii", np.array([0.5, np.inf, 0.5])),
    ],
)
def test_train_rejects_invalid_state_arrays(backend, name, value):
    with pytest.raises(ValueError, match=name):
        _call_train(backend, **{name: value})


def _target(value=None):
    if value is not None:
        return value
    return _inputs()["source_pos"].copy()


@pytest.mark.parametrize(
    ("targets", "error", "match"),
    [
        (None, ValueError, "targets"),
        ([], ValueError, "targets"),
        ([(0,)], ValueError, r"targets\[0\]"),
        ([(0.0, _target())], TypeError, "frame"),
        ([(0, _target()), (0, _target())], ValueError, "Duplicate"),
        ([(-1, _target())], ValueError, "outside"),
        ([(2, _target())], ValueError, "outside"),
        ([(0, np.empty((0, 3)))], ValueError, "target"),
        ([(0, np.zeros((2, 2)))], ValueError, "target"),
        ([(0, np.zeros(3))], ValueError, "target"),
        ([(0, np.array([[np.nan, 0.0, 0.0]]))], ValueError, "target"),
    ],
)
def test_train_rejects_invalid_targets(backend, targets, error, match):
    with pytest.raises(error, match=match):
        _call_train(backend, targets=targets)


@pytest.mark.parametrize("backend_name", ("torch", "jax"))
def test_prepare_training_inputs_normalizes_float32_contiguous_without_changing_values(
    backend_name,
):
    module = importlib.import_module(f"waxmorph.{backend_name}.train")
    config = dataclasses.replace(
        _config(module),
        n_epochs=np.int64(1),
        t_rollout=np.int64(2),
        mech_steps=np.int64(0),
        diff_steps=np.int64(0),
        log_every=np.int64(1),
    )
    values = _inputs()
    values["source_pos"] = np.asfortranarray(values["source_pos"])
    values["polarities"] = np.pad(values["polarities"], ((0, 0), (0, 1)))[:, :3]
    values["c"] = np.pad(values["c"], ((0, 0), (0, 1)))[:, :2]
    values["radii"] = np.repeat(values["radii"], 2)[::2]
    target = np.pad(values["source_pos"], ((0, 0), (0, 1)))[:, :3]

    prepared = _prepare_training_inputs(
        config,
        values["source_pos"],
        values["polarities"],
        values["c"],
        values["radii"],
        ((np.int64(0), target) for _ in range(1)),
    )

    for original, normalized in zip(
        (*values.values(), target), (*prepared[:4], prepared[4][0][1]), strict=True
    ):
        assert normalized.dtype == np.float32
        assert normalized.flags.c_contiguous
        np.testing.assert_array_equal(normalized, np.asarray(original, dtype=np.float32))
    assert prepared[4][0][0] == 0
    np.testing.assert_array_equal(prepared[1], values["polarities"].astype(np.float32))


def test_prepare_training_inputs_accepts_index_protocol_and_unequal_target_size():
    class Frame:
        def __index__(self):
            return 1

    values = _inputs()
    prepared = _prepare_training_inputs(
        _config(importlib.import_module("waxmorph.torch.train")),
        **values,
        targets=((False, np.zeros((1, 3))), (Frame(), np.zeros((2, 3)))),
    )

    assert [frame for frame, _ in prepared[4]] == [0, 1]
    assert prepared[4][1][1].shape == (2, 3)


def test_train_validates_before_existing_checkpoint(backend, tmp_path):
    checkpoint = tmp_path / "invalid-checkpoint"
    checkpoint.write_bytes(b"not a checkpoint")

    with pytest.raises(ValueError, match="source_pos"):
        _call_train(backend, source_pos=np.empty((0, 3)), save_path=checkpoint)
