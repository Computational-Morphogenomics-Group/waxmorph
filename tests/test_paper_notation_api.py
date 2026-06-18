"""Guards that the public API exposes the manuscript ("waxMorph") notation.

These tests pin the paper-aligned names so the notation cannot silently
regress: the per-cell signaling-molecule concentration array is ``c`` (not
``genes``/``G``), the molecule-delta output head is ``dc`` (not ``dG``), the
diffusion/regularization hyperparameters are ``D_emu``/``lambda_reg``, and the
reaction-diffusion / growth kernels expose ``chi``/``gamma``/``D_inhib`` and
``alpha_grow``/``ell_sw``. They run on CPU (no CUDA required).
"""

import dataclasses
import inspect

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Concentration array `c` and the graph node features it produces
# ---------------------------------------------------------------------------
def test_build_graph_uses_c_parameter():
    """Public ``build_graph`` takes the concentration array as ``c`` (paper symbol)."""
    from waxmorph import build_graph

    params = inspect.signature(build_graph).parameters
    assert "c" in params, "build_graph must expose the concentration array as `c`"
    assert "G" not in params and "genes" not in params


def test_build_node_features_are_concentrations():
    """Node features equal the concentration array ``c`` (paper: v_i = c_i)."""
    torch = pytest.importorskip("torch")
    from waxmorph import build_graph

    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    p = torch.ones(3, 3)
    r = torch.tensor([0.6, 0.6, 0.6])
    c = torch.tensor([[0.2, 0.7], [0.4, 0.1], [0.8, 0.5]])

    node_features, _edge_index, _edge_features = build_graph(x, p, r, 3, c=c, eps_dist=0.0)
    assert node_features.shape == (3, 2)
    np.testing.assert_allclose(node_features.detach().numpy(), c.numpy(), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Decoder output heads: dX / dP / dc
# ---------------------------------------------------------------------------
def test_torch_gns_default_head_is_dc():
    """Default GNS output heads are dX, dP, dc (molecule delta, paper ċ)."""
    pytest.importorskip("torch")
    from waxmorph import GNS

    model = GNS(node_feature_dim=2, edge_feature_dim=2, hidden_dim=4, num_mp_steps=1)
    assert set(model.decoders.keys()) == {"dX", "dP", "dc"}
    assert "dG" not in model.decoders


def test_jax_gns_default_head_is_dc():
    """JAX parity backend uses the same dc head."""
    pytest.importorskip("equinox")
    import jax

    from waxmorph.jax.gnn import GNS as JaxGNS

    model = JaxGNS(
        node_feature_dim=2,
        edge_feature_dim=2,
        hidden_dim=4,
        num_mp_steps=1,
        key=jax.random.PRNGKey(0),
    )
    assert set(model.decoders.keys()) == {"dX", "dP", "dc"}


# ---------------------------------------------------------------------------
# Training entry point + hyperparameter names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", ["waxmorph.torch.train", "waxmorph.jax.train"])
def test_train_takes_c_and_paper_hyperparams(module):
    """train(..., c=...) and TrainConfig exposes D_emu and lambda_reg in both backends."""
    pytest.importorskip("torch" if module.endswith("torch.train") else "equinox")
    import importlib

    mod = importlib.import_module(module)
    train_params = inspect.signature(mod.train).parameters
    assert "c" in train_params and "genes" not in train_params

    field_names = {f.name for f in dataclasses.fields(mod.TrainConfig)}
    assert {"D_emu", "lambda_reg"} <= field_names
    assert "alpha_diff" not in field_names and "l2_lambda" not in field_names


# ---------------------------------------------------------------------------
# Forward simulator kernel parameter names (reaction-diffusion + growth)
# ---------------------------------------------------------------------------
def test_simulator_reaction_diffusion_param_names():
    """chem_step exposes chi, gamma, D_inhib (paper chi/gamma/D_inhib) not legacy S/T/phi."""
    pytest.importorskip("warp")
    from waxmorph import simulator

    params = inspect.signature(simulator.chem_step).parameters
    assert {"chi", "gamma", "D_inhib"} <= set(params)
    assert not ({"S", "T", "phi"} & set(params))


def test_simulator_growth_param_names():
    """growth_step exposes alpha_grow and ell_sw (paper alpha_grow / ell_sw)."""
    pytest.importorskip("warp")
    from waxmorph import simulator

    params = inspect.signature(simulator.growth_step).parameters
    assert {"alpha_grow", "ell_sw"} <= set(params)
    assert not ({"AP", "SC"} & set(params))


# ---------------------------------------------------------------------------
# Training log records the concentration trajectory under the paper key
# ---------------------------------------------------------------------------
def test_train_log_key_is_best_traj_c():
    """The torch training loop records ``best_traj_c`` (was best_traj_genes)."""
    torch = pytest.importorskip("torch")
    from waxmorph import GNS, TrainConfig, squared_loss, train

    n = 6
    rng = np.random.default_rng(0)
    source_pos = rng.standard_normal((n, 3)).astype(np.float32)
    polarities = rng.standard_normal((n, 3)).astype(np.float32)
    c = rng.random((n, 2)).astype(np.float32)
    radii = np.full(n, 0.6, dtype=np.float32)

    model = GNS(
        node_feature_dim=2,
        edge_feature_dim=2,
        hidden_dim=4,
        num_mp_steps=1,
        output_dims={"dX": 3, "dP": 3, "dc": 2},
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainConfig(n_epochs=1, t_rollout=1, mech_steps=0, diff_steps=0)

    result = train(
        model,
        optimizer,
        squared_loss,
        source_pos=source_pos,
        polarities=polarities,
        c=c,
        radii=radii,
        targets=[(0, source_pos)],
        config=config,
        device="cpu",
    )
    assert "best_traj_c" in result.log
    assert "best_traj_genes" not in result.log
