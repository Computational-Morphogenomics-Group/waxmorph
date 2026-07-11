"""JAX/Equinox training primitives for non-growing shape assembly."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from waxmorph._train_core import _prepare_training_inputs
from waxmorph.constants import EPS_POLARITY
from waxmorph.jax.gnn import GNS as JaxGNS
from waxmorph.jax.graph import (
    build_edge_features,
    build_edge_index,
    build_node_features,
)

_CAPACITY_HEADROOM = 1.15
_CAPACITY_BUCKET = 1024


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    r"""Hyperparameters for JAX non-growing shape assembly.

    ``n_epochs`` counts optimizer updates. Histories evaluate every post-update model, so
    training performs one additional rollout. Zero ``mech_steps`` or ``diff_steps`` disables
    that prescribed update. ``dt_mech`` and ``dt_diff`` are explicit-Euler steps; diffusion
    stability depends on ``D_emu * dt_diff`` and graph degree. ``dt_gns`` scales the raw
    ``dX``, ``dP``, and ``dc`` heads before state updates.

    The total loss is

    .. math::

        L=L_{shape}+\lambda_{reg}\sum_t
        \lVert \Delta t_{GNS}\,GNS_{X,t}\rVert_F^2,

    where the displacement term is measured before mechanics. ``grad_clip_norm=None``
    disables global-norm clipping. JAX adds ``max_edges_factor``: both directed-edge and
    neighbor-pair buffers are bounded by ``N * max_edges_factor``; exceeding either bound
    raises instead of recompiling to an unbounded shape.

    Examples:
        >>> cfg = TrainConfig(n_epochs=3, t_rollout=2)
        >>> print(cfg.n_epochs, cfg.t_rollout, cfg.grad_clip_norm)
        3 2 1.0
    """

    n_epochs: int = 2000
    t_rollout: int = 100
    mech_steps: int = 5
    diff_steps: int = 5
    dt_mech: float = 1e-2
    dt_diff: float = 1e-2
    dt_gns: float = 1e-2
    D_emu: float = 0.1
    lambda_reg: float = 1e-3
    grad_clip_norm: float | None = 1.0
    log_every: int = 10
    max_edges_factor: int = 10


@dataclasses.dataclass
class TrainResult:
    """Best post-update model and its run diagnostics.

    ``losses_total``, ``losses_shape``, and unweighted ``losses_l2`` each have length
    ``n_epochs`` and describe post-update rollouts. The best trajectory includes the source
    frame and has shapes ``[t_rollout+1, N, 3]`` for ``best_traj_pos`` and
    ``best_traj_pol``, and ``[t_rollout+1, N, num_molecules]`` for ``best_traj_c``.
    ``best_epoch`` is zero-based; ``best_loss`` matches that model and trajectory.
    ``target_frames`` is sorted, and every config field appears as ``config_<field>``. JAX
    therefore adds ``config_max_edges_factor`` to the common Torch log schema.
    """

    model: Any
    log: dict


@dataclasses.dataclass(frozen=True)
class _PairTopology:
    pair_i: jax.Array
    pair_j: jax.Array
    num_pairs: jax.Array


@dataclasses.dataclass(frozen=True)
class _StepTopology:
    edge_index: jax.Array
    num_edges: jax.Array
    mech_pairs: tuple[_PairTopology, ...]
    diff_pairs: tuple[_PairTopology, ...]


class _TopologyBatch(NamedTuple):
    edge_index: jax.Array
    num_edges: jax.Array
    mech_pair_i: jax.Array
    mech_pair_j: jax.Array
    mech_num_pairs: jax.Array
    diff_pair_i: jax.Array
    diff_pair_j: jax.Array
    diff_num_pairs: jax.Array


@dataclasses.dataclass(frozen=True)
class _WarpCollectionContext:
    R_wp: Any
    f_net: Any
    lap_c: Any
    grid: Any
    query_radius: float
    particle_count: int
    max_pairs: int
    device: str


def _uses_warp_bridge(config: TrainConfig) -> bool:
    return config.mech_steps > 0 or config.diff_steps > 0


def _resolve_jax_device(device: str, *, require_cuda: bool) -> jax.Device:
    device_name = str(device)
    if device_name.startswith("cuda"):
        try:
            gpu_devices = jax.devices("gpu")
        except Exception as exc:
            if require_cuda:
                raise RuntimeError(
                    "JAX/Warp differentiable physics requires a JAX GPU backend, "
                    "but JAX could not initialize one."
                ) from exc
            return jax.devices("cpu")[0]

        if gpu_devices:
            ordinal = 0
            if ":" in device_name:
                try:
                    ordinal = int(device_name.split(":", 1)[1])
                except ValueError as exc:
                    raise ValueError(f"Invalid CUDA device string: {device_name!r}") from exc
            if ordinal >= len(gpu_devices):
                if require_cuda:
                    raise RuntimeError(
                        f"Requested {device_name!r}, but JAX sees only {len(gpu_devices)} GPU(s)."
                    )
                return jax.devices("cpu")[0]
            return gpu_devices[ordinal]
        if require_cuda:
            raise RuntimeError("JAX/Warp differentiable physics requires a JAX GPU backend.")

    return jax.devices("cpu")[0]


def _validate_finite_array(name: str, array: jax.Array, *, rollout_step: int, phase: str) -> None:
    arr = np.asarray(jax.device_get(array))
    finite_mask = np.isfinite(arr)
    if finite_mask.all():
        return

    bad_indices = np.argwhere(~finite_mask)
    first_bad = tuple(int(i) for i in bad_indices[0])
    bad_value = arr[first_bad]
    raise ValueError(
        f"Non-finite values detected in {name} during {phase} at rollout step {rollout_step}: "
        f"total_bad={(~finite_mask).sum()}, first_bad_index={first_bad}, "
        f"first_bad_value={bad_value!r}"
    )


def _array_leaves(tree) -> list[jax.Array]:
    return [leaf for leaf in jax.tree_util.tree_leaves(tree) if isinstance(leaf, jax.Array)]


def _device_put_arrays(tree, device: jax.Device):
    return jax.tree.map(
        lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x,
        tree,
    )


def _validate_checkpoint_model(model: JaxGNS, checkpoint_model: JaxGNS) -> None:
    if model._config != checkpoint_model._config:
        config_names = dict.fromkeys((*model._config, *checkpoint_model._config))
        mismatches = [
            name
            for name in config_names
            if name not in model._config
            or name not in checkpoint_model._config
            or model._config[name] != checkpoint_model._config[name]
        ]
        raise ValueError(
            f"Incompatible checkpoint GNS configuration fields: {', '.join(mismatches)}."
        )

    model_tree = eqx.filter(model, eqx.is_array)
    checkpoint_tree = eqx.filter(checkpoint_model, eqx.is_array)
    if jax.tree_util.tree_structure(model_tree) != jax.tree_util.tree_structure(checkpoint_tree):
        raise ValueError("Incompatible checkpoint GNS PyTree structure.")

    model_leaves = _array_leaves(model_tree)
    checkpoint_leaves = _array_leaves(checkpoint_tree)
    for index, (model_leaf, checkpoint_leaf) in enumerate(
        zip(model_leaves, checkpoint_leaves, strict=True)
    ):
        if model_leaf.shape != checkpoint_leaf.shape:
            raise ValueError(
                f"Incompatible checkpoint GNS leaf {index} shape: "
                f"model={model_leaf.shape}, checkpoint={checkpoint_leaf.shape}."
            )
        if model_leaf.dtype != checkpoint_leaf.dtype:
            raise ValueError(
                f"Incompatible checkpoint GNS leaf {index} dtype: "
                f"model={model_leaf.dtype}, checkpoint={checkpoint_leaf.dtype}."
            )


def _tree_all_finite(tree) -> jax.Array:
    leaves = _array_leaves(tree)
    if not leaves:
        return jnp.asarray(True)

    finite = jnp.asarray(True)
    for leaf in leaves:
        finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf)))
    return finite


def _tree_global_norm(tree) -> jax.Array:
    """Compute global L2 norm without overflowing float32 squares.

    ``||g|| = m * sqrt(sum (g/m)^2)`` with ``m=max|g|`` avoids enabling JAX's global x64
    mode; all rescaled magnitudes lie in ``[0, 1]``.
    """
    leaves = _array_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    abs_max = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
    # Guard the all-zero gradient case so the rescaling divisor is never 0.
    scale = jnp.where(abs_max > 0, abs_max, jnp.asarray(1.0, dtype=abs_max.dtype))
    sq_sum = sum(jnp.sum(jnp.square(leaf / scale)) for leaf in leaves)
    return abs_max * jnp.sqrt(sq_sum)


def _clip_grads(grads, max_norm: float):
    grad_norm = _tree_global_norm(grads)
    scale = jnp.minimum(1.0, jnp.asarray(max_norm, dtype=grad_norm.dtype) / (grad_norm + 1e-6))
    return jax.tree.map(lambda g: g * scale if isinstance(g, jax.Array) else g, grads), grad_norm


def _as_numpy(array: jax.Array) -> np.ndarray:
    return np.asarray(jax.device_get(array)).copy()


def _checked_loss_values(
    loss: jax.Array,
    loss_shape: jax.Array,
    loss_l2: jax.Array,
    *,
    epoch: int,
    phase: str,
) -> tuple[float, float, float]:
    values = tuple(
        (name, float(jax.device_get(value)))
        for name, value in (("loss", loss), ("shape loss", loss_shape), ("L2 loss", loss_l2))
    )
    for name, value in values:
        if not np.isfinite(value):
            raise ValueError(f"Non-finite {name} during {phase} at epoch {epoch}: value={value!r}")
    return tuple(value for _name, value in values)


def _max_edges_for_config(config: TrainConfig, particle_count: int) -> int:
    if config.max_edges_factor <= 0:
        raise ValueError("TrainConfig.max_edges_factor must be positive.")
    return max(int(particle_count) * int(config.max_edges_factor), 1)


def _scalar_int(value: jax.Array | int) -> int:
    return int(np.asarray(jax.device_get(value)))


def _bucketed_capacity(observed_max: int, max_capacity: int, *, name: str) -> int:
    """Inflate and bucket a static buffer size to reduce XLA recompilations.

    The result lies in ``[1, max_capacity]``. Counts above the configured capacity raise
    rather than truncate.
    """
    if max_capacity <= 0:
        raise ValueError(f"{name} capacity must be positive.")
    if observed_max > max_capacity:
        raise ValueError(
            f"{name} has {observed_max} entries but max_edges_factor allows only "
            f"{max_capacity}. Increase TrainConfig.max_edges_factor."
        )
    if observed_max <= 0:
        return 1

    bucketed = int(np.ceil((observed_max * _CAPACITY_HEADROOM) / _CAPACITY_BUCKET))
    return min(max(bucketed * _CAPACITY_BUCKET, 1), max_capacity)


def _build_pair_topology_warp(
    X: jax.Array,
    R: jax.Array,
    particle_count: int,
    max_pairs: int | None,
    device: str,
    grid=None,
) -> _PairTopology:
    import warp as wp

    from waxmorph.constants import EPS_DIST, HASH_GRID_DIM
    from waxmorph.emulator import _build_neighbor_pairs_dynamic

    X_wp = wp.from_jax(X, dtype=wp.vec3f)
    R_wp = wp.from_jax(R, dtype=wp.float32)
    r_max = float(np.asarray(jax.device_get(R[:particle_count])).max())
    query_radius = 2.0 * r_max + EPS_DIST
    if grid is None:
        grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    grid.build(X_wp[:particle_count], query_radius)
    edges_i, edges_j, num_pairs = _build_neighbor_pairs_dynamic(
        X_wp,
        R_wp,
        particle_count,
        query_radius,
        grid,
        device,
    )

    if max_pairs is not None and num_pairs > max_pairs:
        raise ValueError(
            f"Neighbor pair list has {num_pairs} entries but max_edges_factor allows only "
            f"{max_pairs}. Increase TrainConfig.max_edges_factor."
        )
    if num_pairs == 0:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        return _PairTopology(pair_i=empty, pair_j=empty, num_pairs=jnp.asarray(0, jnp.int32))

    pair_i = jnp.asarray(wp.to_jax(edges_i[:num_pairs]), dtype=jnp.int32)
    pair_j = jnp.asarray(wp.to_jax(edges_j[:num_pairs]), dtype=jnp.int32)
    return _PairTopology(
        pair_i=pair_i,
        pair_j=pair_j,
        num_pairs=jnp.asarray(num_pairs, dtype=jnp.int32),
    )


def _build_pair_topology(
    X: jax.Array,
    R: jax.Array,
    particle_count: int,
    max_pairs: int | None = None,
    device: str = "cuda",
) -> _PairTopology:
    if max_pairs is not None and str(device).startswith("cuda"):
        return _build_pair_topology_warp(X, R, particle_count, max_pairs, device)

    edge_index, _num_edges = build_edge_index(X, R, particle_count=particle_count)
    edges = np.asarray(jax.device_get(edge_index))

    if edges.shape[1] == 0:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        return _PairTopology(pair_i=empty, pair_j=empty, num_pairs=jnp.asarray(0, jnp.int32))

    mask = edges[0] < edges[1]
    pair_i = jnp.asarray(edges[0, mask], dtype=jnp.int32)
    pair_j = jnp.asarray(edges[1, mask], dtype=jnp.int32)
    if max_pairs is not None and pair_i.shape[0] > max_pairs:
        raise ValueError(
            f"Neighbor pair list has {pair_i.shape[0]} entries but max_edges_factor allows only "
            f"{max_pairs}. Increase TrainConfig.max_edges_factor."
        )

    return _PairTopology(
        pair_i=pair_i,
        pair_j=pair_j,
        num_pairs=jnp.asarray(pair_i.shape[0], dtype=jnp.int32),
    )


def _make_warp_collection_context(
    R: jax.Array,
    c: jax.Array,
    *,
    particle_count: int,
    max_pairs: int,
    device: str,
) -> _WarpCollectionContext:
    import warp as wp

    from waxmorph.constants import EPS_DIST, HASH_GRID_DIM

    R_wp = wp.from_jax(R, dtype=wp.float32)
    r_max = float(np.asarray(jax.device_get(R[:particle_count])).max())
    query_radius = 2.0 * r_max + EPS_DIST
    max_particles = int(R.shape[0])
    num_molecules = int(c.shape[1]) if c.ndim > 1 else 1
    return _WarpCollectionContext(
        R_wp=R_wp,
        f_net=wp.zeros(max_particles, dtype=wp.vec3f, device=device),
        lap_c=wp.zeros((max_particles, num_molecules), dtype=wp.float32, device=device),
        grid=wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device),
        query_radius=query_radius,
        particle_count=particle_count,
        max_pairs=max_pairs,
        device=device,
    )


def _jax_positions_to_warp(X: jax.Array):
    import warp as wp

    return wp.from_jax(X, dtype=wp.vec3f)


def _jax_c_to_warp(c: jax.Array):
    import warp as wp

    c_wp = wp.from_jax(c, dtype=wp.float32)
    if c.ndim == 2:
        c_wp = c_wp.reshape((int(c.shape[0]), int(c.shape[1])))
    return c_wp


def _warp_positions_to_jax(X_wp, dtype) -> jax.Array:
    import warp as wp

    return jnp.asarray(wp.to_jax(X_wp), dtype=dtype)


def _warp_c_to_jax(c_wp, dtype) -> jax.Array:
    import warp as wp

    return jnp.asarray(wp.to_jax(c_wp), dtype=dtype)


def _build_warp_collection_pairs(
    X_wp,
    ctx: _WarpCollectionContext,
) -> tuple[_PairTopology, Any, Any, int]:
    import warp as wp

    from waxmorph.emulator import _build_neighbor_pairs_dynamic

    ctx.grid.build(X_wp[: ctx.particle_count], ctx.query_radius)
    edges_i, edges_j, num_pairs = _build_neighbor_pairs_dynamic(
        X_wp,
        ctx.R_wp,
        ctx.particle_count,
        ctx.query_radius,
        ctx.grid,
        ctx.device,
    )
    if num_pairs > ctx.max_pairs:
        raise ValueError(
            f"Neighbor pair list has {num_pairs} entries but max_edges_factor allows only "
            f"{ctx.max_pairs}. Increase TrainConfig.max_edges_factor."
        )
    if num_pairs == 0:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        topology = _PairTopology(empty, empty, jnp.asarray(0, dtype=jnp.int32))
    else:
        topology = _PairTopology(
            pair_i=jnp.asarray(wp.to_jax(edges_i[:num_pairs]), dtype=jnp.int32),
            pair_j=jnp.asarray(wp.to_jax(edges_j[:num_pairs]), dtype=jnp.int32),
            num_pairs=jnp.asarray(num_pairs, dtype=jnp.int32),
        )
    return topology, edges_i, edges_j, num_pairs


def _native_warp_mech_step(
    X_wp,
    pair_i_wp,
    pair_j_wp,
    num_pairs: int,
    dt: float,
    ctx: _WarpCollectionContext,
):
    import warp as wp

    from waxmorph.emulator import _gd_update, _sticky_sphere_grads_from_pairs

    ctx.f_net.zero_()
    X_out = wp.zeros_like(X_wp, device=ctx.device)
    if num_pairs > 0:
        wp.launch(
            _sticky_sphere_grads_from_pairs,
            dim=num_pairs,
            inputs=[X_wp, ctx.R_wp, pair_i_wp[:num_pairs], pair_j_wp[:num_pairs]],
            outputs=[ctx.f_net],
            device=ctx.device,
        )
    wp.launch(
        _gd_update,
        dim=ctx.particle_count,
        inputs=[X_wp, ctx.f_net, float(dt)],
        outputs=[X_out],
        device=ctx.device,
    )
    return X_out


def _native_warp_diffusion_step(
    c_wp,
    pair_i_wp,
    pair_j_wp,
    num_pairs: int,
    D_emu: float,
    dt: float,
    ctx: _WarpCollectionContext,
):
    import warp as wp

    from waxmorph.emulator import (
        _molecule_diffusion_laplacian_from_pairs,
        _molecule_diffusion_step_out,
    )

    ctx.lap_c.zero_()
    c_out = wp.zeros_like(c_wp, device=ctx.device)
    if num_pairs > 0:
        wp.launch(
            _molecule_diffusion_laplacian_from_pairs,
            dim=num_pairs,
            inputs=[c_wp, pair_i_wp[:num_pairs], pair_j_wp[:num_pairs]],
            outputs=[ctx.lap_c],
            device=ctx.device,
        )
    wp.launch(
        _molecule_diffusion_step_out,
        dim=(int(c_wp.shape[0]), int(c_wp.shape[1])),
        inputs=[c_wp, ctx.lap_c, float(D_emu), float(dt), ctx.particle_count],
        outputs=[c_out],
        device=ctx.device,
    )
    return c_out


@eqx.filter_jit
def _collection_gns_step(
    model,
    X: jax.Array,
    P: jax.Array,
    c: jax.Array,
    edge_index: jax.Array,
    num_edges: jax.Array,
    particle_count: int,
    dt_gns: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    node_feats = build_node_features(c, particle_count)
    edge_feats = build_edge_features(X, P, edge_index, particle_count)
    out = model(node_feats, edge_index, edge_feats, num_edges=num_edges)
    dX = out["dX"] * dt_gns
    dP = out["dP"] * dt_gns
    dc = out["dc"] * dt_gns

    X_next = X + dX
    P_next = P + dP
    P_next = P_next / jnp.maximum(jnp.linalg.norm(P_next, axis=-1, keepdims=True), EPS_POLARITY)
    c_next = jnp.maximum(c + dc, jnp.asarray(0.0, dtype=c.dtype))
    return X_next, P_next, c_next, dX, dP, dc


def _collect_topologies_and_trajectory(
    *,
    model,
    config: TrainConfig,
    X: jax.Array,
    P: jax.Array,
    c: jax.Array,
    R: jax.Array,
    particle_count: int,
    device: str,
) -> tuple[tuple[_StepTopology, ...], list[dict[str, np.ndarray]]]:
    """Roll out without gradients while recording each step's frozen topology.

    Collection rebuilds directed GNS edges and mechanics/diffusion pairs from the evolving
    positions. :func:`_stack_topologies` pads them to bucketed static shapes, then
    :func:`_epoch_loss_with_topology_batch` replays state values differentiably through
    ``jax.lax.scan``; adjacency and pair selection remain outside autodiff. CUDA collection
    uses native Warp pair and physics kernels when prescribed physics is enabled.

    The trajectory contains ``t_rollout + 1`` detached host snapshots, including the source;
    the topology tuple contains one entry per update step.
    """
    topologies: list[_StepTopology] = []
    trajectory = [{"pos": _as_numpy(X), "pol": _as_numpy(P), "c": _as_numpy(c)}]
    max_edges = _max_edges_for_config(config, particle_count)
    max_pairs = max_edges
    use_native_warp = str(device).startswith("cuda") and _uses_warp_bridge(config)
    warp_ctx = (
        _make_warp_collection_context(
            R,
            c,
            particle_count=particle_count,
            max_pairs=max_pairs,
            device=device,
        )
        if use_native_warp
        else None
    )

    for t in range(config.t_rollout):
        _validate_finite_array("X", X, rollout_step=t, phase="pre-graph build")
        _validate_finite_array("P", P, rollout_step=t, phase="pre-graph build")
        _validate_finite_array("c", c, rollout_step=t, phase="pre-graph build")

        edge_index, num_edges = build_edge_index(
            X,
            R,
            particle_count=particle_count,
            max_edges=max_edges,
        )
        X_next, P_next, c_next, dX, dP, dc = _collection_gns_step(
            model,
            X,
            P,
            c,
            edge_index,
            num_edges,
            particle_count,
            config.dt_gns,
        )
        num_edges_int = _scalar_int(num_edges)
        edge_index_for_replay = edge_index[:, :num_edges_int]

        _validate_finite_array("dX", dX, rollout_step=t, phase="gns output")
        _validate_finite_array("dP", dP, rollout_step=t, phase="gns output")
        _validate_finite_array("dc", dc, rollout_step=t, phase="gns output")

        X = jax.lax.stop_gradient(X_next)
        P = jax.lax.stop_gradient(P_next)
        c = jax.lax.stop_gradient(c_next)

        _validate_finite_array("X", X, rollout_step=t, phase="post-gns update")
        _validate_finite_array("P", P, rollout_step=t, phase="post-gns update")
        _validate_finite_array("c", c, rollout_step=t, phase="post-gns update")

        mech_pairs: list[_PairTopology] = []
        if warp_ctx is not None and config.mech_steps > 0:
            X_wp = _jax_positions_to_warp(X)
            for _ in range(config.mech_steps):
                pairs, pair_i_wp, pair_j_wp, num_pairs = _build_warp_collection_pairs(
                    X_wp,
                    warp_ctx,
                )
                mech_pairs.append(pairs)
                X_wp = _native_warp_mech_step(
                    X_wp,
                    pair_i_wp,
                    pair_j_wp,
                    num_pairs,
                    config.dt_mech,
                    warp_ctx,
                )
            X = jax.lax.stop_gradient(_warp_positions_to_jax(X_wp, X.dtype))
        else:
            for _ in range(config.mech_steps):
                pairs = _build_pair_topology(
                    X,
                    R,
                    particle_count,
                    max_pairs=max_pairs,
                    device=device,
                )
                mech_pairs.append(pairs)
                from waxmorph.jax.warp_autograd import warp_mech_step

                X = jax.lax.stop_gradient(
                    warp_mech_step(
                        X,
                        R,
                        pairs.pair_i,
                        pairs.pair_j,
                        config.dt_mech,
                        num_pairs=pairs.num_pairs,
                        device=device,
                    )
                )

        _validate_finite_array("X", X, rollout_step=t, phase="post-mechanics")

        diff_pairs: list[_PairTopology] = []
        if warp_ctx is not None and config.diff_steps > 0:
            X_wp = _jax_positions_to_warp(X)
            c_wp = _jax_c_to_warp(c)
            pairs, pair_i_wp, pair_j_wp, num_pairs = _build_warp_collection_pairs(
                X_wp,
                warp_ctx,
            )
            for _ in range(config.diff_steps):
                diff_pairs.append(pairs)
                c_wp = _native_warp_diffusion_step(
                    c_wp,
                    pair_i_wp,
                    pair_j_wp,
                    num_pairs,
                    config.D_emu,
                    config.dt_diff,
                    warp_ctx,
                )
            c = jax.lax.stop_gradient(_warp_c_to_jax(c_wp, c.dtype))
        else:
            diff_pairs_for_step = None
            for _ in range(config.diff_steps):
                if diff_pairs_for_step is None:
                    diff_pairs_for_step = _build_pair_topology(
                        X,
                        R,
                        particle_count,
                        max_pairs=max_pairs,
                        device=device,
                    )
                diff_pairs.append(diff_pairs_for_step)
                from waxmorph.jax.warp_autograd import warp_diffusion_step

                c = jax.lax.stop_gradient(
                    warp_diffusion_step(
                        c,
                        diff_pairs_for_step.pair_i,
                        diff_pairs_for_step.pair_j,
                        config.D_emu,
                        config.dt_diff,
                        num_pairs=diff_pairs_for_step.num_pairs,
                        device=device,
                    )
                )

        _validate_finite_array("c", c, rollout_step=t, phase="post-diffusion")

        topologies.append(
            _StepTopology(
                edge_index=edge_index_for_replay,
                num_edges=num_edges,
                mech_pairs=tuple(mech_pairs),
                diff_pairs=tuple(diff_pairs),
            )
        )
        trajectory.append({"pos": _as_numpy(X), "pol": _as_numpy(P), "c": _as_numpy(c)})

    return tuple(topologies), trajectory


def _empty_pair_stack(t_rollout: int, num_steps: int, max_pairs: int) -> jax.Array:
    return jnp.zeros((t_rollout, num_steps, max_pairs), dtype=jnp.int32)


def _max_edge_count(topologies: tuple[_StepTopology, ...]) -> int:
    if not topologies:
        return 0
    return max(_scalar_int(topology.num_edges) for topology in topologies)


def _max_pair_count(topologies: tuple[_StepTopology, ...]) -> int:
    observed = 0
    for topology in topologies:
        for pairs in (*topology.mech_pairs, *topology.diff_pairs):
            observed = max(observed, _scalar_int(pairs.num_pairs))
    return observed


def _stack_pair_topologies(
    topologies: tuple[_StepTopology, ...],
    *,
    attr: str,
    num_steps: int,
    max_pairs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    t_rollout = len(topologies)
    if num_steps == 0:
        empty_pairs = _empty_pair_stack(t_rollout, 0, max_pairs)
        empty_counts = jnp.zeros((t_rollout, 0), dtype=jnp.int32)
        return empty_pairs, empty_pairs, empty_counts

    pair_i_array = np.zeros((t_rollout, num_steps, max_pairs), dtype=np.int32)
    pair_j_array = np.zeros((t_rollout, num_steps, max_pairs), dtype=np.int32)
    num_pairs_array = np.zeros((t_rollout, num_steps), dtype=np.int32)
    for t, topology in enumerate(topologies):
        pairs_for_step = getattr(topology, attr)
        if len(pairs_for_step) != num_steps:
            raise ValueError(
                f"Internal topology error: expected {num_steps} {attr} entries, "
                f"got {len(pairs_for_step)}."
            )
        for step, pairs in enumerate(pairs_for_step):
            count = _scalar_int(pairs.num_pairs)
            if count > max_pairs:
                raise ValueError(
                    f"Neighbor pair list has {count} entries but max_edges_factor allows only "
                    f"{max_pairs}. Increase TrainConfig.max_edges_factor."
                )
            if pairs.pair_i.shape[0] < count or pairs.pair_j.shape[0] < count:
                raise ValueError(
                    f"Neighbor pair list count is {count}, but fewer entries were provided."
                )
            num_pairs_array[t, step] = count
            if count > 0:
                pair_i_array[t, step, :count] = np.asarray(
                    jax.device_get(pairs.pair_i[:count]),
                    dtype=np.int32,
                )
                pair_j_array[t, step, :count] = np.asarray(
                    jax.device_get(pairs.pair_j[:count]),
                    dtype=np.int32,
                )

    return (
        jnp.asarray(pair_i_array, dtype=jnp.int32),
        jnp.asarray(pair_j_array, dtype=jnp.int32),
        jnp.asarray(num_pairs_array, dtype=jnp.int32),
    )


def _stack_topologies(
    topologies: tuple[_StepTopology, ...],
    *,
    config: TrainConfig,
    max_pairs: int,
) -> _TopologyBatch:
    """Pad variable topologies to bucketed shapes for ``jax.lax.scan`` replay.

    ``num_edges`` and ``num_pairs`` preserve real counts so the replay masks zero-filled
    slots. ``max_pairs`` enforces the ``max_edges_factor`` budget.
    """
    edge_capacity = _bucketed_capacity(
        _max_edge_count(topologies),
        max_pairs,
        name="Graph",
    )
    pair_capacity = _bucketed_capacity(
        _max_pair_count(topologies),
        max_pairs,
        name="Neighbor pair list",
    )
    mech_pair_i, mech_pair_j, mech_num_pairs = _stack_pair_topologies(
        topologies,
        attr="mech_pairs",
        num_steps=config.mech_steps,
        max_pairs=pair_capacity,
    )
    diff_pair_i, diff_pair_j, diff_num_pairs = _stack_pair_topologies(
        topologies,
        attr="diff_pairs",
        num_steps=config.diff_steps,
        max_pairs=pair_capacity,
    )
    edge_index_array = np.zeros((len(topologies), 2, edge_capacity), dtype=np.int32)
    num_edges_array = np.zeros((len(topologies),), dtype=np.int32)
    for t, topology in enumerate(topologies):
        count = _scalar_int(topology.num_edges)
        if count > edge_capacity:
            raise ValueError(
                f"Graph has {count} edges but max_edges_factor allows only {edge_capacity}. "
                "Increase TrainConfig.max_edges_factor."
            )
        if topology.edge_index.shape[1] < count:
            raise ValueError(
                f"Graph edge count is {count}, but only {topology.edge_index.shape[1]} "
                "entries were provided."
            )
        num_edges_array[t] = count
        if count > 0:
            edge_index_array[t, :, :count] = np.asarray(
                jax.device_get(topology.edge_index[:, :count]),
                dtype=np.int32,
            )
    return _TopologyBatch(
        edge_index=jnp.asarray(edge_index_array, dtype=jnp.int32),
        num_edges=jnp.asarray(num_edges_array, dtype=jnp.int32),
        mech_pair_i=mech_pair_i,
        mech_pair_j=mech_pair_j,
        mech_num_pairs=mech_num_pairs,
        diff_pair_i=diff_pair_i,
        diff_pair_j=diff_pair_j,
        diff_num_pairs=diff_num_pairs,
    )


def _apply_rollout_step_from_batch(
    *,
    model,
    config: TrainConfig,
    X: jax.Array,
    P: jax.Array,
    c: jax.Array,
    R: jax.Array,
    topology: _TopologyBatch,
    particle_count: int,
    device: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    node_feats = build_node_features(c, particle_count)
    edge_feats = build_edge_features(X, P, topology.edge_index, particle_count)

    out = model(node_feats, topology.edge_index, edge_feats, num_edges=topology.num_edges)
    dX = out["dX"] * config.dt_gns
    dP = out["dP"] * config.dt_gns
    dc = out["dc"] * config.dt_gns

    X = X + dX
    P = P + dP
    P = P / jnp.maximum(jnp.linalg.norm(P, axis=-1, keepdims=True), EPS_POLARITY)
    c = jnp.maximum(c + dc, jnp.asarray(0.0, dtype=c.dtype))

    if config.mech_steps > 0 or config.diff_steps > 0:
        from waxmorph.jax.warp_autograd import warp_diffusion_step, warp_mech_step

        for step in range(config.mech_steps):
            X = warp_mech_step(
                X,
                R,
                topology.mech_pair_i[step],
                topology.mech_pair_j[step],
                config.dt_mech,
                num_pairs=topology.mech_num_pairs[step],
                device=device,
            )

        for step in range(config.diff_steps):
            c = warp_diffusion_step(
                c,
                topology.diff_pair_i[step],
                topology.diff_pair_j[step],
                config.D_emu,
                config.dt_diff,
                num_pairs=topology.diff_num_pairs[step],
                device=device,
            )

    return X, P, c, jnp.sum(dX**2)


def _epoch_loss_with_topology_batch(
    *,
    model,
    config: TrainConfig,
    X_source: jax.Array,
    P_source: jax.Array,
    c_source: jax.Array,
    R: jax.Array,
    particle_count: int,
    targets_by_frame: dict[int, jax.Array],
    topology_batch: _TopologyBatch,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
    device: str,
) -> tuple[jax.Array, jax.Array]:
    def scan_step(carry, topology):
        X, P, c = carry
        X, P, c, step_l2 = _apply_rollout_step_from_batch(
            model=model,
            config=config,
            X=X,
            P=P,
            c=c,
            R=R,
            topology=topology,
            particle_count=particle_count,
            device=device,
        )
        return (X, P, c), (X, step_l2)

    (_X, _P, _c), (positions, step_l2) = jax.lax.scan(
        scan_step,
        (X_source, P_source, c_source),
        topology_batch,
    )
    loss_l2 = jnp.sum(step_l2)
    loss_shape = jnp.asarray(0.0, dtype=X_source.dtype)
    for frame, target in targets_by_frame.items():
        loss_shape = loss_shape + loss_fn(positions[int(frame)], target)
    return loss_shape, loss_l2


def _zero_grads_if_nonfinite(grads, finite: jax.Array):
    return jax.tree.map(
        lambda g: jnp.where(finite, g, jnp.zeros_like(g)) if isinstance(g, jax.Array) else g,
        grads,
    )


def _make_train_step(
    *,
    config: TrainConfig,
    optimizer: optax.GradientTransformation,
    targets_by_frame: dict[int, jax.Array],
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
    particle_count: int,
    device: str,
):
    @eqx.filter_jit
    def train_step(
        model,
        opt_state: optax.OptState,
        X_source: jax.Array,
        P_source: jax.Array,
        c_source: jax.Array,
        R: jax.Array,
        topology_batch: _TopologyBatch,
    ):
        def loss_fn_inner(candidate_model):
            loss_shape, loss_l2 = _epoch_loss_with_topology_batch(
                model=candidate_model,
                config=config,
                X_source=X_source,
                P_source=P_source,
                c_source=c_source,
                R=R,
                particle_count=particle_count,
                targets_by_frame=targets_by_frame,
                topology_batch=topology_batch,
                loss_fn=loss_fn,
                device=device,
            )
            return loss_shape + (loss_l2 * config.lambda_reg), (loss_shape, loss_l2)

        (loss, (loss_shape, loss_l2)), grads = eqx.filter_value_and_grad(
            loss_fn_inner, has_aux=True
        )(model)

        grads_finite = _tree_all_finite(grads)
        grad_norm = _tree_global_norm(grads)
        grads_for_update = _zero_grads_if_nonfinite(grads, grads_finite)
        if config.grad_clip_norm is not None:
            grads_for_update, grad_norm = _clip_grads(grads_for_update, config.grad_clip_norm)

        updates, opt_state = optimizer.update(
            grads_for_update,
            opt_state,
            eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)
        params_finite = _tree_all_finite(model)
        return model, opt_state, loss, loss_shape, loss_l2, grad_norm, grads_finite, params_finite

    return train_step


def train(
    model: JaxGNS,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    source_pos: np.ndarray,
    polarities: np.ndarray,
    c: np.ndarray,
    radii: np.ndarray,
    targets: list[tuple[int, np.ndarray]] | None = None,
    config: TrainConfig | None = None,
    save_path: str | Path | None = None,
    device: str = "cuda",
) -> TrainResult:
    """Train a GNS model for non-growing shape assembly (JAX backend).

    Each epoch performs one update. Histories, the selected model, and its trajectory all
    describe post-update states; evaluating ``n_epochs`` updates requires ``n_epochs + 1``
    collect-and-replay rollouts. Collection freezes data-dependent topology outside autodiff;
    replay differentiates features and state updates through fixed, padded buffers.

    Inputs are validated before conversion to contiguous float32. State arrays share a
    nonzero particle axis. Source polarities are used as supplied for the first graph, whose
    angle feature assumes unit vectors; later learned updates apply normalization. Target
    frame ``0`` means the state after the first rollout update, and valid unique frames lie in
    ``[0, t_rollout)``. The supplied ``opt_state`` is not returned.

    A new ``save_path`` receives the best model's two-file checkpoint and log. Only trusted
    existing checkpoints should be loaded: config, PyTree, leaf shapes, and dtypes must match;
    saved weights replace the supplied model, Optax state is reinitialized, and exactly one
    refinement update runs without overwriting existing files.

    Warp mechanics or diffusion requires a CUDA-backed JAX device and uses custom VJPs;
    CPU training is available only when both prescribed step counts are zero. See
    :class:`TrainResult` for the log schema.

    Raises:
        TypeError: If config types, array dtypes, or target frames are invalid.
        RuntimeError: If Warp-backed differentiable physics is requested on a
            device where JAX cannot provide a GPU backend.
        ValueError: If config bounds, state arrays, radii, targets, or checkpoint
            structure are invalid.
    """
    if config is None:
        config = TrainConfig()

    source_pos, polarities, c, radii, targets = _prepare_training_inputs(
        config, source_pos, polarities, c, radii, targets
    )

    N = len(source_pos)
    jax_device = _resolve_jax_device(device, require_cuda=_uses_warp_bridge(config))
    if save_path is not None and os.path.exists(save_path):
        with jax.default_device(jax_device):
            checkpoint_model = JaxGNS.load(save_path)
        _validate_checkpoint_model(model, checkpoint_model)
        model = _device_put_arrays(checkpoint_model, jax_device)
        with jax.default_device(jax_device):
            opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        opt_state = _device_put_arrays(opt_state, jax_device)
        config = dataclasses.replace(config, n_epochs=1)
    else:
        model = _device_put_arrays(model, jax_device)
        opt_state = _device_put_arrays(opt_state, jax_device)

    with jax.default_device(jax_device):
        X_source = jnp.asarray(source_pos, dtype=jnp.float32)
        P_source = jnp.asarray(polarities, dtype=jnp.float32)
        c_source = jnp.asarray(c, dtype=jnp.float32)
        R = jnp.asarray(radii, dtype=jnp.float32)

    targets_by_frame: dict[int, jax.Array] = {}
    for frame, pos in targets:
        with jax.default_device(jax_device):
            targets_by_frame[frame] = jnp.asarray(pos, dtype=jnp.float32)

    losses_total = []
    losses_shape = []
    losses_l2 = []
    best_loss = float("inf")
    best_model = None
    best_trajectory = None
    best_epoch = 0
    best_shape_loss = 0.0
    best_l2_loss = 0.0
    max_pairs = _max_edges_for_config(config, N)
    train_step = _make_train_step(
        config=config,
        optimizer=optimizer,
        targets_by_frame=targets_by_frame,
        loss_fn=loss_fn,
        particle_count=N,
        device=device,
    )

    def collect_model(candidate_model):
        with jax.default_device(jax_device):
            topologies, trajectory = _collect_topologies_and_trajectory(
                model=candidate_model,
                config=config,
                X=X_source,
                P=P_source,
                c=c_source,
                R=R,
                particle_count=N,
                device=device,
            )
            topology_batch = _stack_topologies(
                topologies,
                config=config,
                max_pairs=max_pairs,
            )
        return topology_batch, trajectory

    def record_evaluation(epoch, candidate_model, trajectory, values):
        nonlocal best_loss, best_model, best_trajectory, best_epoch
        nonlocal best_shape_loss, best_l2_loss
        epoch_loss, epoch_shape_loss, epoch_l2_loss = values
        losses_total.append(epoch_loss)
        losses_shape.append(epoch_shape_loss)
        losses_l2.append(epoch_l2_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_shape_loss = epoch_shape_loss
            best_l2_loss = epoch_l2_loss
            best_model = candidate_model
            best_trajectory = trajectory
            best_epoch = epoch

        if (epoch + 1) % config.log_every == 0:
            print(
                f"Epoch {epoch + 1:4d}/{config.n_epochs}  "
                f"best={best_loss:.4f}  shape={best_shape_loss:.4f}  l2={best_l2_loss:.4f}"
            )

    topology_batch, epoch_trajectory = collect_model(model)
    for epoch in trange(config.n_epochs):
        pre_update_model = model
        (
            model,
            opt_state,
            loss,
            loss_shape,
            loss_l2,
            grad_norm,
            grads_finite,
            params_finite,
        ) = train_step(
            model,
            opt_state,
            X_source,
            P_source,
            c_source,
            R,
            topology_batch,
        )

        loss_values = _checked_loss_values(
            loss,
            loss_shape,
            loss_l2,
            epoch=0 if epoch == 0 else epoch - 1,
            phase="pre-update evaluation" if epoch == 0 else "post-update evaluation",
        )
        if not bool(jax.device_get(grads_finite)):
            raise ValueError(
                f"Non-finite values detected in model gradients during post-backward "
                f"at epoch {epoch}."
            )
        if config.grad_clip_norm is not None and not bool(jnp.isfinite(grad_norm)):
            raise ValueError(
                f"Non-finite gradient norm after clipping at epoch {epoch}: "
                f"grad_norm={float(grad_norm)!r}"
            )
        if not bool(jax.device_get(params_finite)):
            raise ValueError(
                f"Non-finite values detected in model parameters during post-optimizer step "
                f"at epoch {epoch}."
            )

        if epoch > 0:
            record_evaluation(
                epoch - 1,
                pre_update_model,
                epoch_trajectory,
                loss_values,
            )
        if epoch + 1 < config.n_epochs:
            topology_batch, epoch_trajectory = collect_model(model)

    topology_batch, final_trajectory = collect_model(model)
    with jax.default_device(jax_device):
        final_shape_loss, final_l2_loss = _epoch_loss_with_topology_batch(
            model=model,
            config=config,
            X_source=X_source,
            P_source=P_source,
            c_source=c_source,
            R=R,
            particle_count=N,
            targets_by_frame=targets_by_frame,
            topology_batch=topology_batch,
            loss_fn=loss_fn,
            device=device,
        )
        final_loss = final_shape_loss + (final_l2_loss * config.lambda_reg)
    final_values = _checked_loss_values(
        final_loss,
        final_shape_loss,
        final_l2_loss,
        epoch=config.n_epochs - 1,
        phase="post-update evaluation",
    )
    record_evaluation(
        config.n_epochs - 1,
        model,
        final_trajectory,
        final_values,
    )
    model = best_model

    traj_pos = np.stack([f["pos"] for f in best_trajectory])
    traj_pol = np.stack([f["pol"] for f in best_trajectory])
    traj_c = np.stack([f["c"] for f in best_trajectory])

    log = {
        "losses_total": np.array(losses_total),
        "losses_shape": np.array(losses_shape),
        "losses_l2": np.array(losses_l2),
        "best_traj_pos": traj_pos,
        "best_traj_pol": traj_pol,
        "best_traj_c": traj_c,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "target_frames": np.array(sorted(targets_by_frame.keys()), dtype=np.int64),
    }
    for field in dataclasses.fields(config):
        log[f"config_{field.name}"] = getattr(config, field.name)

    if save_path is not None and not os.path.exists(save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        np.savez_compressed(f"{save_path}.log.npz", **log)

    return TrainResult(model=model, log=log)
