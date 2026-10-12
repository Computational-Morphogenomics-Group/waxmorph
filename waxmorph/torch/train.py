"""PyTorch training primitives for shape assembly with a fixed agent count."""

from __future__ import annotations

import copy
import dataclasses
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp
from tqdm import trange

from waxmorph._train_core import _backtrack, _prepare_training_inputs
from waxmorph.constants import EPS_POLARITY, HASH_GRID_DIM
from waxmorph.emulator import _query_radius
from waxmorph.torch.gnn import GNS
from waxmorph.torch.graph import build_graph
from waxmorph.torch.warp_autograd import WarpDiffusionStep, WarpMechStep


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    r"""Rollout and optimizer settings for a fixed agent count.

    ``n_epochs`` counts optimizer updates; histories evaluate each resulting model, requiring
    one extra rollout. Each rollout has ``t_rollout`` learned steps. ``mech_steps`` and
    ``diff_steps`` are prescribed substep counts; zero selects the identity path for that
    correction.
    ``dt_gns`` scales all learned deltas. ``dt_mech`` and ``dt_diff`` are explicit step sizes
    whose stable range depends on the active forces and graph; nonfinite states are rejected.
    Diffusion applies :math:`c\leftarrow\max(c-D_{emu}L_Gc\,\Delta t_{diff},0)`.

    The optimized loss is :math:`L=L_{shape}+\lambda_{reg}L_{reg}`, where

    .. math::

        L_{reg}=\sum_t\lVert\Delta t_{GNS}\,GNS_{X,t}\rVert_F^2.

    The regularizer measures learned displacement before the mechanics correction.
    ``grad_clip_norm`` caps the global L2 gradient norm; ``None`` keeps the computed gradients.
    ``trainer="epochs"`` runs all ``n_epochs`` updates; ``trainer="patience"`` stops earlier once
    ``patience`` updates in a row lower the best loss by at most ``convergence_threshold``.
    An update that raises the loss by more than ``max_increase`` is shrunk by backtracking
    (factor 0.8, at most 15 times, one rollout each), which removes loss spikes; ``None`` keeps
    every update.
    A progress bar shows each epoch's loss terms and the best loss. ``compile`` runs the GNS through
    :func:`torch.compile` with dynamic shapes: the first epoch compiles (tens of seconds), later
    epochs run faster, and results match eager mode up to floating-point rounding. JAX adds
    ``max_edges_factor`` for its static edge capacity and always compiles.

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
    trainer: str = "epochs"
    patience: int = 100
    convergence_threshold: float = 1e-3
    max_increase: float | None = 0.3
    compile: bool = False


@dataclasses.dataclass
class TrainResult:
    """Best post-update model and its training log.

    The supplied optimizer state is restored alongside ``model``. ``losses_total``,
    ``losses_shape``, and ``losses_l2`` hold one post-update entry per update; shape
    loss sums supervised frames and L2 is the unweighted learned-displacement regularizer.
    ``best_traj_pos`` and
    ``best_traj_pol`` have shape ``[t_rollout + 1, N, 3]``; ``best_traj_c`` has shape
    ``[t_rollout + 1, N, C]``. ``best_epoch`` is zero-based; other metadata keys are
    ``best_loss``, sorted ``target_frames``, and one ``config_<field>`` per
    :class:`TrainConfig` field. Torch logs ``config_compile``; JAX logs
    ``config_max_edges_factor``.
    """

    model: Any
    log: dict


def _nonfinite_details(tensor: torch.Tensor) -> str | None:
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return None

    nonfinite_mask = ~finite_mask
    bad_indices = nonfinite_mask.nonzero(as_tuple=False)
    first_bad = tuple(int(i) for i in bad_indices[0].tolist())
    bad_value = tensor[first_bad].detach().cpu().item()
    return (
        f"total_bad={int(nonfinite_mask.sum().item())}, first_bad_index={first_bad}, "
        f"first_bad_value={bad_value!r}"
    )


def _validate_finite_tensor(
    name: str, tensor: torch.Tensor, *, rollout_step: int, phase: str
) -> None:
    if (details := _nonfinite_details(tensor)) is not None:
        raise ValueError(
            f"Non-finite values detected in {name} during {phase} "
            f"at rollout step {rollout_step}: {details}"
        )


def _validate_finite_tensors(checks, *, rollout_step: int) -> None:
    """Apply :func:`_validate_finite_tensor` to ``(name, tensor, phase)`` checks after one sync."""
    if not bool(torch.stack([torch.isfinite(t).all() for _, t, _ in checks]).all()):
        for name, tensor, phase in checks:
            _validate_finite_tensor(name, tensor, rollout_step=rollout_step, phase=phase)


def _raise_on_nonfinite_named_tensors(
    kind: str,
    named_tensors,
    *,
    epoch: int,
    phase: str,
) -> None:
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        if (details := _nonfinite_details(tensor)) is not None:
            raise ValueError(
                f"Non-finite values detected in model {kind} during {phase} at epoch {epoch}: "
                f"parameter={name!r}, {details}"
            )


def _format_gradient_stats(stats, *, limit: int = 5) -> str:
    if not stats:
        return "none"

    return "; ".join(
        "{}: norm={:.6g}, max_abs={:.6g}, shape={}, dtype={}, device={}".format(*item)
        for item in sorted(stats, key=lambda item: item[1], reverse=True)[:limit]
    )


def _clip_grad_norm_stable(named_parameters, max_norm: float, *, epoch: int) -> torch.Tensor:
    """Clip gradients using float64 norm accumulation to avoid fp32 overflow."""
    grad_entries = [
        (name, param.grad) for name, param in list(named_parameters) if param.grad is not None
    ]
    if not grad_entries:
        return torch.tensor(0.0, dtype=torch.float64)

    norm_sq_total = torch.zeros((), dtype=torch.float64)
    stats = []
    for name, grad in grad_entries:
        if (details := _nonfinite_details(grad)) is not None:
            raise ValueError(
                "Non-finite gradient detected during clipping at epoch "
                f"{epoch}: parameter={name!r}, {details}"
            )

        grad_detached = grad.detach()
        grad64 = grad_detached.to(dtype=torch.float64)
        norm_sq = grad64.square().sum().cpu()
        grad_norm = torch.sqrt(norm_sq).item()
        max_abs = grad_detached.abs().max().detach().cpu().item() if grad.numel() else 0.0
        norm_sq_total = norm_sq_total + norm_sq
        stats.append(
            (name, grad_norm, max_abs, tuple(grad.shape), str(grad.dtype), str(grad.device))
        )

    total_norm = torch.sqrt(norm_sq_total)
    if not torch.isfinite(total_norm):
        raise ValueError(
            "Non-finite stable gradient norm during clipping at epoch "
            f"{epoch}: grad_norm={total_norm.item()!r}; "
            f"top_gradients={_format_gradient_stats(stats)}"
        )

    clip_coef = float(max_norm) / (total_norm.item() + 1e-6)
    if clip_coef < 1.0:
        with torch.no_grad():
            for _name, grad in grad_entries:
                grad.mul_(clip_coef)

    return total_norm


def _run_epoch(
    *,
    model,
    config,
    source_pos,
    polarities,
    c,
    R_t,
    R_wp,
    f_net,
    grid,
    query_radius=None,
    N,
    targets_by_frame,
    X_source_t,
    loss_fn,
    torch_device,
    epoch_trajectory,
):
    """Roll out one differentiable epoch on ``torch_device``.

    Features remain attached to the Torch graph, while each step's COO topology comes from a
    detached state snapshot. Each Warp bridge call owns a tape; Torch autograd composes the
    calls through the rollout, and the diffusion substeps of a step share one pair list.
    ``targets_by_frame`` uses post-step indices, so frame 0 follows the first learned and
    prescribed updates. Returns the summed supervised shape loss, the unweighted
    ``sum_t ||dt_gns * GNS_X,t||^2`` measured before mechanics, and the detached
    source-plus-``t_rollout`` trajectory.
    """
    X_t = X_source_t.clone().requires_grad_(True)
    c_t = torch.from_numpy(c.copy()).to(torch_device).requires_grad_(True)
    P_t = torch.from_numpy(polarities.copy()).to(torch_device)

    loss_l2 = torch.tensor(0.0, device=torch_device)
    loss_shape = torch.tensor(0.0, device=torch_device)
    frames = []

    _validate_finite_tensors(
        [("X_t", X_t, "epoch start"), ("P_t", P_t, "epoch start"), ("c_t", c_t, "epoch start")],
        rollout_step=0,
    )

    for _t in range(config.t_rollout):
        node_feats, edge_index, edge_feats = build_graph(
            X_t,
            P_t,
            R_t,
            particle_count=N,
            c=c_t,
        )

        out = model(node_feats, edge_index, edge_feats)
        dX = out["dX"] * config.dt_gns
        dP = out["dP"] * config.dt_gns
        dc = out["dc"] * config.dt_gns

        loss_l2 = loss_l2 + dX.square().sum()

        X_t = X_t + dX
        P_t = torch.nn.functional.normalize(P_t + dP, dim=-1, eps=EPS_POLARITY)
        c_t = torch.clamp_min(c_t + dc, 0.0)
        _validate_finite_tensors(
            [
                ("dX", dX, "gns output"),
                ("dP", dP, "gns output"),
                ("dc", dc, "gns output"),
                ("X_t", X_t, "post-gns update"),
                ("P_t", P_t, "post-gns update"),
                ("c_t", c_t, "post-gns update"),
            ],
            rollout_step=_t,
        )

        for _ in range(config.mech_steps):
            X_t = WarpMechStep.apply(X_t, R_wp, N, config.dt_mech, f_net, grid, query_radius)
        _validate_finite_tensor("X_t", X_t, rollout_step=_t, phase="post-mechanics")

        if config.diff_steps:
            X_wp_diff = wp.from_torch(X_t.detach().contiguous(), dtype=wp.vec3f)
            c_t = WarpDiffusionStep.apply(
                c_t,
                X_wp_diff,
                R_wp,
                N,
                config.D_emu,
                config.dt_diff,
                grid,
                query_radius,
                config.diff_steps,
            )
        _validate_finite_tensor("c_t", c_t, rollout_step=_t, phase="post-diffusion")

        frames.append((X_t.detach(), P_t.detach(), c_t.detach()))

        if _t in targets_by_frame:
            loss_shape = loss_shape + loss_fn(X_t, targets_by_frame[_t])

    pos, pol, conc = (torch.stack(frame).cpu().numpy() for frame in zip(*frames, strict=True))
    epoch_trajectory.extend(
        {"pos": p, "pol": q, "c": r} for p, q, r in zip(pos, pol, conc, strict=True)
    )
    return loss_shape, loss_l2, epoch_trajectory


def train(
    model: GNS,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
    """Train through learned, mechanics, and diffusion rollout steps.

    Optimizes ``L_shape + lambda_reg * sum_t ||dt_gns * GNS_X,t||^2``. Histories and
    selection use post-update rollouts; the supplied optimizer is restored to the selected
    model. Feature gradients remain live; each graph build and Warp substep freezes its own
    contact or pair topology.

    State and target arrays become C-contiguous ``float32``. Positions and polarities have
    shape ``[N, 3]``, concentrations ``[N, C]``, and radii ``[N]`` with common nonzero ``N``;
    input polarities are used as supplied for the first graph, whose angle feature assumes
    unit vectors; later learned updates apply normalization. ``loss_fn`` maps predicted
    ``[N, 3]`` and target ``[M, 3]`` positions to a scalar. Targets are unique
    ``(frame, [M, 3])`` pairs with ``0 <= frame < t_rollout``. The source precedes indexed
    frames; frame 0 supervises the state after the first complete rollout step.

    ``device`` must be accepted by Torch and Warp. A new ``save_path`` receives the best GNS
    checkpoint and compressed log. An existing trusted checkpoint must match the supplied
    model; its weights load into that object, optimizer state starts fresh, and one refinement
    update preserves both existing files. See :class:`TrainResult` for the log schema.
    The JAX counterpart requires CUDA for its Warp custom VJPs and adds a static edge-capacity
    setting.
    """
    if config is None:
        config = TrainConfig()

    source_pos, polarities, c, radii, targets = _prepare_training_inputs(
        config, source_pos, polarities, c, radii, targets
    )
    stops_early = {"epochs": False, "patience": True}[config.trainer]

    wp_device = device
    torch_device = torch.device(device)
    N = len(source_pos)

    if save_path is not None and os.path.exists(save_path):
        checkpoint_model = GNS.load(save_path, map_location=torch_device).to(torch_device)
        model_config = model._constructor_config()
        checkpoint_config = checkpoint_model._constructor_config()
        mismatches = [
            name for name, value in model_config.items() if checkpoint_config[name] != value
        ]
        if mismatches:
            details = ", ".join(
                f"{name} (model={model_config[name]!r}, checkpoint={checkpoint_config[name]!r})"
                for name in mismatches
            )
            raise ValueError(f"Incompatible checkpoint GNS configuration: {details}.")
        model = model.to(torch_device)
        model.load_state_dict(checkpoint_model.state_dict())
        optimizer.state.clear()
        del checkpoint_model
        config = dataclasses.replace(config, n_epochs=1)
    else:
        model = model.to(torch_device)

    f_net = wp.zeros(N, dtype=wp.vec3f, device=wp_device)
    # Contact search reads host radii; fixed radii give one query radius for every substep.
    R_t = torch.from_numpy(radii.copy())
    R_wp = wp.from_numpy(radii.copy(), dtype=wp.float32, device=wp_device)
    query_radius = _query_radius(R_wp, N)

    targets_by_frame: dict[int, torch.Tensor] = {}
    for frame, pos in targets:
        targets_by_frame[frame] = torch.from_numpy(pos).to(torch_device)

    X_source_t = torch.from_numpy(source_pos).to(torch_device)

    losses_total = []
    losses_shape = []
    losses_l2 = []
    best_loss = float("inf")
    best_model_state = None
    best_optimizer_state = None
    best_trajectory = None
    best_epoch = 0

    grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=wp_device)
    # The rollout calls the compiled wrapper; it shares parameters with `model`.
    forward = torch.compile(model, dynamic=True) if config.compile else model

    def evaluate_model(*, epoch, phase):
        epoch_trajectory = [{"pos": source_pos.copy(), "pol": polarities.copy(), "c": c.copy()}]
        loss_shape, loss_l2, epoch_trajectory = _run_epoch(
            model=forward,
            config=config,
            source_pos=source_pos,
            polarities=polarities,
            c=c,
            R_t=R_t,
            R_wp=R_wp,
            f_net=f_net,
            grid=grid,
            query_radius=query_radius,
            N=N,
            targets_by_frame=targets_by_frame,
            X_source_t=X_source_t,
            loss_fn=loss_fn,
            torch_device=torch_device,
            epoch_trajectory=epoch_trajectory,
        )
        loss = loss_shape + (loss_l2 * config.lambda_reg)
        for name, value in (("loss", loss), ("shape loss", loss_shape), ("L2 loss", loss_l2)):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"Non-finite {name} during {phase} at epoch {epoch}: "
                    f"value={value.detach().cpu().item()!r}"
                )
        return loss, loss_shape, loss_l2, epoch_trajectory

    params = list(model.parameters())

    def evaluate_candidate(before, after, epoch, step_size):
        if step_size < 1:
            with torch.no_grad():
                for param, start, end in zip(params, before, after, strict=True):
                    param.copy_(start.lerp(end, step_size))
        evaluation = evaluate_model(epoch=epoch, phase="post-update evaluation")
        return evaluation[0].item(), evaluation

    optimizer.zero_grad(set_to_none=True)
    loss, loss_shape, loss_l2, epoch_trajectory = evaluate_model(
        epoch=0, phase="pre-update evaluation"
    )
    epoch_loss, stalled, step_size = loss.item(), 0, 1.0

    progress = trange(config.n_epochs, desc="Training", unit="epoch", dynamic_ncols=True)
    for epoch in progress:
        loss.backward()
        _raise_on_nonfinite_named_tensors(
            "gradients",
            ((name, param.grad) for name, param in model.named_parameters()),
            epoch=epoch,
            phase="post-backward",
        )
        if config.grad_clip_norm is not None:
            _clip_grad_norm_stable(model.named_parameters(), config.grad_clip_norm, epoch=epoch)

        before = [param.detach().clone() for param in params]
        optimizer.step()
        _raise_on_nonfinite_named_tensors(
            "parameters",
            model.named_parameters(),
            epoch=epoch,
            phase="post-optimizer step",
        )
        optimizer.zero_grad(set_to_none=True)
        after = [param.detach().clone() for param in params]

        with torch.set_grad_enabled(epoch + 1 < config.n_epochs):
            epoch_loss, evaluation, step_size = _backtrack(
                partial(evaluate_candidate, before, after, epoch),
                epoch_loss,
                config.max_increase,
                step_size,
            )
        loss, loss_shape, loss_l2, epoch_trajectory = evaluation

        epoch_shape_loss = loss_shape.item()
        epoch_l2_loss = loss_l2.item()
        losses_total.append(epoch_loss)
        losses_shape.append(epoch_shape_loss)
        losses_l2.append(epoch_l2_loss)

        stalled = 0 if best_loss - epoch_loss > config.convergence_threshold else stalled + 1
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_trajectory = epoch_trajectory
            best_epoch = epoch

        progress.set_postfix(
            loss=f"{epoch_loss:.4f}",
            shape=f"{epoch_shape_loss:.4f}",
            l2=f"{epoch_l2_loss:.4f}",
            best=f"{best_loss:.4f}",
            best_epoch=best_epoch,
            patience=f"{stalled}/{config.patience}" if stops_early else "off",
            refresh=False,
        )
        if stops_early and stalled >= config.patience:
            break
    progress.close()

    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)

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
