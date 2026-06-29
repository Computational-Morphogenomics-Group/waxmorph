"""PyTorch training primitives for non-growing shape assembly."""

from __future__ import annotations

import copy
import dataclasses
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp
from tqdm import trange

from waxmorph.constants import HASH_GRID_DIM
from waxmorph.torch.gnn import GNS
from waxmorph.torch.graph import build_graph
from waxmorph.torch.warp_autograd import WarpDiffusionStep, WarpMechStep


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    r"""Hyperparameters for PyTorch non-growing shape assembly training.

    Mirrors :class:`waxmorph.jax.train.TrainConfig`; shared fields carry the
    same meaning across backends. The JAX twin adds ``max_edges_factor``
    because static-shape compilation needs a compile-time edge-count bound.

    Attributes:
        n_epochs: Number of optimization epochs (AdamW updates). Each epoch
            replays a full ``t_rollout``-step trajectory. Convergence typically
            occurs near 200-300 epochs; the default 2000 is a generous ceiling.
        t_rollout: Number of emulation (Euler) steps unrolled per epoch, i.e.
            the trajectory length ``T``. The reference configuration uses
            ``T = 100`` (the default here).
        mech_steps: Soft-sphere mechanics substeps applied after each learned
            update, run on a faster time scale so the trajectory stays
            biophysically coherent. Typically a handful (default 5); ``0``
            disables the mechanics constraint.
        diff_steps: Graph-Laplacian diffusion substeps applied after each
            learned update, analogous to ``mech_steps``. Default 5; ``0``
            disables the diffusion constraint.
        dt_mech: Forward-Euler step :math:`\Delta t` for the soft-sphere
            mechanics correction. Small (default 1e-2) to keep the explicit
            integration stable; the effective per-rollout displacement is
            ``mech_steps * dt_mech``.
        dt_diff: Forward-Euler step :math:`\Delta t` for the graph-Laplacian
            diffusion correction (default 1e-2). Larger values approach the
            CFL-style stability limit of explicit diffusion and can blow up.
        dt_gns: Multiplier on the raw GNS deltas before they are added to the
            state, i.e. the learned-update Euler step ``dt`` (default 1e-2).
            Keeps initial (near-random) network outputs from moving particles
            far in a single step.
        D_emu: Signaling-molecule diffusion coefficient ``D_emu`` in the
            graph-diffusion update ``c -= D_emu * (L_G c) * dt_diff`` (default
            0.1). With ``dt_diff`` it sets how fast latent fields homogenize;
            ``D_emu * dt_diff`` near/above the inverse max node degree risks
            instability.
        lambda_reg: Strength :math:`\lambda` of the squared-displacement
            regularizer that penalizes large frame-to-frame motion,

            .. math::

                L_{reg} = \lambda \sum_{t} \lVert X_t - X_{t+1} \rVert_F^2,

            with :math:`\lVert \cdot \rVert_F` the Frobenius norm (summed over
            the per-step ``dX`` deltas). It discourages trajectories that satisfy
            the goals only at the supervised target frames by spreading motion
            smoothly across the rollout. Small (default 1e-3): too large
            over-smooths and prevents reaching the target, too small permits
            erratic jumps.
        grad_clip_norm: Maximum global gradient L2 norm; gradients are rescaled
            when they exceed it. ``None`` disables clipping. Default 1.0.
        log_every: Epoch interval used for progress logging.

    See Also:
        waxmorph.jax.train.TrainConfig: JAX/Equinox parity backend.

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


@dataclasses.dataclass
class TrainResult:
    """Result returned by :func:`waxmorph.torch.train.train`.

    Mirrors :class:`waxmorph.jax.train.TrainResult`; the ``log`` keys are
    identical across backends.

    Attributes:
        model: Best :class:`torch.nn.Module` found during training, or the
            latest model if no finite improvement was recorded.
        log: Diagnostics for the run. Per-epoch loss histories (each a 1-D
            array of length ``n_epochs``):

            - ``losses_total``: total loss ``L_shape + lambda_reg * L_reg``.
            - ``losses_shape``: shape (distributional) loss at the target
              frames only.
            - ``losses_l2``: unweighted squared-displacement term
              ``sum||dX||^2`` (multiply by ``lambda_reg`` for its loss
              contribution).

            Best trajectory (the rollout from the best-loss epoch; leading axis
            is ``t_rollout + 1`` because frame 0 is the source state):

            - ``best_traj_pos``: positions, shape ``[t_rollout+1, N, 3]``.
            - ``best_traj_pol``: polarities, shape ``[t_rollout+1, N, 3]``.
            - ``best_traj_c``: concentrations, shape
              ``[t_rollout+1, N, num_molecules]``.

            Metadata:

            - ``best_epoch``: zero-based index of the best epoch.
            - ``best_loss``: total loss at that epoch.
            - ``target_frames``: sorted supervised frame indices.
            - ``config_<field>``: one entry per :class:`TrainConfig` field
              (e.g. ``config_n_epochs``, ``config_lambda_reg``), recording the
              hyperparameters used.

    See Also:
        waxmorph.jax.train.TrainResult: JAX/Equinox parity backend.
    """

    model: Any
    log: dict


def _validate_finite_numpy(name: str, arr: np.ndarray) -> None:
    """Fail fast on invalid host inputs before training starts."""
    finite_mask = np.isfinite(arr)
    if finite_mask.all():
        return

    bad_indices = np.argwhere(~finite_mask)
    first_bad = tuple(int(i) for i in bad_indices[0])
    bad_value = arr[first_bad]
    raise ValueError(
        f"Non-finite values detected in {name} before training: "
        f"total_bad={(~finite_mask).sum()}, first_bad_index={first_bad}, "
        f"first_bad_value={bad_value!r}"
    )


def _validate_finite_tensor(
    name: str, tensor: torch.Tensor, *, rollout_step: int, phase: str
) -> None:
    """Raise a contextual error before invalid tensors reach later pipeline stages."""
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return

    bad_indices = (~finite_mask).nonzero(as_tuple=False)
    first_bad = tuple(int(i) for i in bad_indices[0].tolist())
    bad_value = tensor[first_bad].detach().cpu().item()
    raise ValueError(
        f"Non-finite values detected in {name} during {phase} at rollout step {rollout_step}: "
        f"total_bad={int((~finite_mask).sum().item())}, first_bad_index={first_bad}, "
        f"first_bad_value={bad_value!r}"
    )


def _raise_on_nonfinite_named_tensors(
    kind: str,
    named_tensors,
    *,
    epoch: int,
    phase: str,
) -> None:
    """Fail fast on invalid parameter tensors or gradients."""
    for name, tensor in named_tensors:
        if tensor is None:
            continue

        finite_mask = torch.isfinite(tensor)
        if bool(finite_mask.all()):
            continue

        bad_indices = (~finite_mask).nonzero(as_tuple=False)
        first_bad = tuple(int(i) for i in bad_indices[0].tolist())
        bad_value = tensor[first_bad].detach().cpu().item()
        raise ValueError(
            f"Non-finite values detected in model {kind} during {phase} at epoch {epoch}: "
            f"parameter={name!r}, total_bad={int((~finite_mask).sum().item())}, "
            f"first_bad_index={first_bad}, first_bad_value={bad_value!r}"
        )


def _format_gradient_stats(stats, *, limit: int = 5) -> str:
    """Format the largest gradient tensors for clipping diagnostics."""
    if not stats:
        return "none"

    ordered = sorted(stats, key=lambda item: item["norm"], reverse=True)
    entries = []
    for item in ordered[:limit]:
        entries.append(
            "{name}: norm={norm:.6g}, max_abs={max_abs:.6g}, "
            "shape={shape}, dtype={dtype}, device={device}".format(**item)
        )
    return "; ".join(entries)


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
        finite_mask = torch.isfinite(grad)
        if not bool(finite_mask.all()):
            bad_indices = (~finite_mask).nonzero(as_tuple=False)
            first_bad = tuple(int(i) for i in bad_indices[0].tolist())
            bad_value = grad[first_bad].detach().cpu().item()
            raise ValueError(
                "Non-finite gradient detected during clipping at epoch "
                f"{epoch}: parameter={name!r}, total_bad={int((~finite_mask).sum().item())}, "
                f"first_bad_index={first_bad}, first_bad_value={bad_value!r}"
            )

        grad_detached = grad.detach()
        if grad_detached.numel() == 0:
            norm_sq = torch.zeros((), dtype=torch.float64)
            grad_norm = 0.0
            max_abs = 0.0
        else:
            grad64 = grad_detached.to(dtype=torch.float64)
            norm_sq = grad64.square().sum().cpu()
            grad_norm = torch.sqrt(norm_sq).item()
            max_abs = grad_detached.abs().max().detach().cpu().item()
        norm_sq_total = norm_sq_total + norm_sq
        stats.append(
            {
                "name": name,
                "norm": grad_norm,
                "max_abs": max_abs,
                "shape": tuple(grad.shape),
                "dtype": str(grad.dtype),
                "device": str(grad.device),
            }
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
    lap_c,
    grid,
    N,
    targets_by_frame,
    X_source_t,
    loss_fn,
    torch_device,
    epoch_trajectory,
):
    """Run one training epoch with differentiable physics.

    Positions and concentrations stay on the :mod:`torch.autograd` computation
    graph throughout. Physics corrections are applied via WarpMechStep /
    WarpDiffusionStep autograd functions, so gradients flow through the full
    trajectory. This is the single-pass torch counterpart of the JAX two-phase
    collect-then-replay design (see :func:`waxmorph.jax.train._run_epoch`):
    torch keeps the whole tape live, so no frozen-topology cache is needed.

    ``targets_by_frame`` maps rollout-step index -> target position tensor.
    A shape loss is accumulated at every tagged post-update state; frame ``0``
    therefore supervises the state after the first rollout update, not the
    initial source state.

    Args:
        model: Graph Network Simulator producing per-step ``dX``/``dP``/``dc``.
        config: Training hyperparameters; only the rollout/step fields are read.
        source_pos: Initial host positions, used only to seed diagnostics; the
            live differentiable state is read from ``X_source_t``.
        polarities: Initial host polarities, shape ``[N, 3]``; copied to device.
        c: Initial host concentrations, shape ``[N, num_molecules]``; copied to
            device and made a leaf requiring grad.
        R_t: Particle radii on the torch device, shape ``[N]``.
        R_wp: Particle radii as a Warp array, consumed by the mechanics step.
        f_net: Preallocated Warp net-force scratch buffer, shape ``[N]``.
        lap_c: Preallocated Warp graph-Laplacian scratch buffer, shape
            ``[N, num_molecules]``.
        grid: Reused Warp ``HashGrid`` for neighbor queries.
        N: Active particle count.
        targets_by_frame: Map from zero-based post-update rollout step to the
            device-resident target position tensor supervised at that step.
        X_source_t: Initial positions on the torch device; cloned into the
            leaf that starts the differentiable rollout.
        loss_fn: Distributional shape loss mapping ``([N, 3], [M, 3])`` to a
            scalar.
        torch_device: Device the rollout tensors live on.
        epoch_trajectory: Mutable list seeded with the source frame; one
            detached snapshot dict is appended per rollout step.

    Returns:
        Tuple ``(loss_shape, loss_l2, epoch_trajectory)`` where ``loss_shape``
        is the summed shape loss over supervised frames, ``loss_l2`` is the
        unweighted ``sum||dX||^2`` regularizer term, and ``epoch_trajectory``
        is the same list passed in, now holding ``t_rollout + 1`` frames.
    """
    X_t = X_source_t.clone().requires_grad_(True)
    c_t = torch.from_numpy(c.copy()).to(torch_device).requires_grad_(True)
    P_t = torch.from_numpy(polarities.copy()).to(torch_device)

    loss_l2 = torch.tensor(0.0, device=torch_device)
    loss_shape = torch.tensor(0.0, device=torch_device)

    _validate_finite_tensor("X_t", X_t, rollout_step=0, phase="epoch start")
    _validate_finite_tensor("P_t", P_t, rollout_step=0, phase="epoch start")
    _validate_finite_tensor("c_t", c_t, rollout_step=0, phase="epoch start")

    for _t in range(config.t_rollout):
        _validate_finite_tensor("X_t", X_t, rollout_step=_t, phase="pre-graph build")
        _validate_finite_tensor("P_t", P_t, rollout_step=_t, phase="pre-graph build")
        _validate_finite_tensor("c_t", c_t, rollout_step=_t, phase="pre-graph build")

        # Build graph from the live Torch state. Feature gradients remain
        # connected to X_t / P_t / c_t; edge_index is rebuilt from a snapshot.
        node_feats, edge_index, edge_feats = build_graph(
            X_t,
            P_t,
            R_t,
            particle_count=N,
            c=c_t,
        )

        # GNS forward (differentiable)
        out = model(node_feats, edge_index, edge_feats)
        dX = out["dX"] * config.dt_gns
        dP = out["dP"] * config.dt_gns
        dc = out["dc"] * config.dt_gns
        _validate_finite_tensor("dX", dX, rollout_step=_t, phase="gns output")
        _validate_finite_tensor("dP", dP, rollout_step=_t, phase="gns output")
        _validate_finite_tensor("dc", dc, rollout_step=_t, phase="gns output")

        loss_l2 = loss_l2 + dX.square().sum()

        # Apply GNS deltas (stays on PyTorch graph)
        X_t = X_t + dX
        P_t = torch.nn.functional.normalize(P_t + dP, dim=-1)
        c_t = torch.clamp_min(c_t + dc, 0.0)
        _validate_finite_tensor("X_t", X_t, rollout_step=_t, phase="post-gns update")
        _validate_finite_tensor("P_t", P_t, rollout_step=_t, phase="post-gns update")
        _validate_finite_tensor("c_t", c_t, rollout_step=_t, phase="post-gns update")

        # Physics correction (differentiable via autograd functions)
        for _ in range(config.mech_steps):
            X_t = WarpMechStep.apply(X_t, R_wp, N, config.dt_mech, f_net, grid)
        _validate_finite_tensor("X_t", X_t, rollout_step=_t, phase="post-mechanics")

        for _ in range(config.diff_steps):
            X_wp_diff = wp.from_torch(X_t.detach().contiguous(), dtype=wp.vec3f)
            c_t = WarpDiffusionStep.apply(
                c_t, X_wp_diff, R_wp, lap_c, N, config.D_emu, config.dt_diff, grid
            )
        _validate_finite_tensor("c_t", c_t, rollout_step=_t, phase="post-diffusion")

        epoch_trajectory.append(
            {
                "pos": X_t.detach().cpu().numpy().copy(),
                "pol": P_t.detach().cpu().numpy().copy(),
                "c": c_t.detach().cpu().numpy().copy(),
            }
        )

        # Accumulate shape loss at every rollout step tagged with a target
        if _t in targets_by_frame:
            loss_shape = loss_shape + loss_fn(X_t, targets_by_frame[_t])

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
    """Train a GNS model for non-growing shape assembly (PyTorch backend).

    Learns neighbor-dependent updates that assemble an initial particle cloud
    into the supervised target morphologies, then composes the prescribed
    soft-sphere mechanics and graph diffusion on top so the trajectory stays
    biophysically coherent. Optimizes ``L = L_shape + lambda_reg * L_reg`` with
    the supplied optimizer (typically AdamW), tracking the best-loss epoch.

    Args:
        model: Graph Network Simulator model.
        optimizer: :class:`torch.optim.Optimizer`.
        loss_fn: Shape loss function mapping predicted positions with shape
            ``[N, 3]`` and target positions with shape ``[M, 3]`` to a scalar.
        source_pos: Initial particle positions with shape ``[N, 3]``.
        polarities: Initial polarity vectors with shape ``[N, 3]``.
        c: Initial signaling-molecule concentrations with shape ``[N, num_molecules]``.
        radii: Particle radii with shape ``[N]``.
        targets: ``(frame, positions)`` supervision pairs. Frame indices are
            zero-based rollout steps measured *after* the per-step updates, so
            frame ``0`` supervises the state after the first rollout update, not
            the initial source state, and the highest usable index is
            ``t_rollout - 1``. Frames must lie in ``[0, t_rollout)`` and must be
            unique.
        config: Training hyperparameters. Defaults to
            :class:`waxmorph.torch.train.TrainConfig`.
        save_path: Optional path where the best model and log are saved. If the
            path already exists it is treated as a checkpoint to resume from:
            the model is loaded and ``config.n_epochs`` is forced to ``1`` (a
            single refinement/inference pass), and no file is overwritten on
            exit. Saving happens only when ``save_path`` did not already exist.
        device: Warp and :class:`torch.device` string such as ``"cuda"`` or
            ``"cpu"``.

    Returns:
        Best model and full training log; see :class:`TrainResult` for the
        ``log`` keys.

    Raises:
        ValueError: If targets are missing, duplicated, out of range, or contain
            non-finite values.

    See Also:
        waxmorph.jax.train.train: JAX/Equinox parity backend. The torch path is
            the default because Warp autodiff integrates through
            :class:`torch.autograd.Function`; the JAX path needs compile-time
            shapes and an edge-count bound (more memory).
    """
    if config is None:
        config = TrainConfig()

    if targets is None:
        raise ValueError("train() requires `targets`.")

    _validate_finite_numpy("source_pos", source_pos)
    _validate_finite_numpy("polarities", polarities)
    _validate_finite_numpy("c", c)
    _validate_finite_numpy("radii", radii)

    # Detect devices
    wp_device = device
    torch_device = torch.device(device)
    model = model.to(torch_device)

    N = len(source_pos)
    max_particles = N
    num_molecules = c.shape[1]

    # Checkpoint detection
    if save_path is not None and os.path.exists(save_path):
        model = GNS.load(save_path, map_location=torch_device).to(torch_device)
        config = dataclasses.replace(config, n_epochs=1)

    # Warp scratch buffers
    f_net = wp.zeros(max_particles, dtype=wp.vec3f, device=wp_device)
    R_t = torch.from_numpy(radii.copy()).to(torch_device)
    R_wp = wp.from_numpy(radii.copy(), dtype=wp.float32, device=wp_device)
    lap_c = wp.zeros((max_particles, num_molecules), dtype=wp.float32, device=wp_device)

    # Per-frame target tensors (device-resident)
    targets_by_frame: dict[int, torch.Tensor] = {}
    for frame, pos in targets:
        frame_int = int(frame)
        if not (0 <= frame_int < config.t_rollout):
            raise ValueError(f"Target frame {frame_int} outside [0, {config.t_rollout}).")
        if frame_int in targets_by_frame:
            raise ValueError(f"Duplicate target frame {frame_int}.")
        _validate_finite_numpy(f"targets[frame={frame_int}]", pos)
        targets_by_frame[frame_int] = torch.from_numpy(pos).to(torch_device)

    X_source_t = torch.from_numpy(source_pos).to(torch_device)

    # Tracking
    losses_total = []
    losses_shape = []
    losses_l2 = []
    best_loss = float("inf")
    best_model_state = None
    best_trajectory = None
    best_epoch = 0
    best_shape_loss = 0.0
    best_l2_loss = 0.0

    # Pre-allocate hash grid
    grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=wp_device)

    for epoch in trange(config.n_epochs):
        optimizer.zero_grad(set_to_none=True)

        epoch_trajectory = [{"pos": source_pos.copy(), "pol": polarities.copy(), "c": c.copy()}]

        loss_shape, loss_l2, epoch_trajectory = _run_epoch(
            model=model,
            config=config,
            source_pos=source_pos,
            polarities=polarities,
            c=c,
            R_t=R_t,
            R_wp=R_wp,
            f_net=f_net,
            lap_c=lap_c,
            grid=grid,
            N=N,
            targets_by_frame=targets_by_frame,
            X_source_t=X_source_t,
            loss_fn=loss_fn,
            torch_device=torch_device,
            epoch_trajectory=epoch_trajectory,
        )

        loss = loss_shape + (loss_l2 * config.lambda_reg)

        loss.backward()
        _raise_on_nonfinite_named_tensors(
            "gradients",
            ((name, param.grad) for name, param in model.named_parameters()),
            epoch=epoch,
            phase="post-backward",
        )
        if config.grad_clip_norm is not None:
            grad_norm = _clip_grad_norm_stable(
                model.named_parameters(), config.grad_clip_norm, epoch=epoch
            )
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise ValueError(
                    f"Non-finite gradient norm after clipping at epoch {epoch}: "
                    f"grad_norm={grad_norm.detach().cpu().item()!r}"
                )
        optimizer.step()
        _raise_on_nonfinite_named_tensors(
            "parameters",
            model.named_parameters(),
            epoch=epoch,
            phase="post-optimizer step",
        )
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = loss.item()
        epoch_shape_loss = loss_shape.item()
        epoch_l2_loss = loss_l2.item()

        losses_total.append(epoch_loss)
        losses_shape.append(epoch_shape_loss)
        losses_l2.append(epoch_l2_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_shape_loss = epoch_shape_loss
            best_l2_loss = epoch_l2_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_trajectory = epoch_trajectory
            best_epoch = epoch

        if (epoch + 1) % config.log_every == 0:
            print(
                f"Epoch {epoch + 1:4d}/{config.n_epochs}  "
                f"best={best_loss:.4f}  shape={best_shape_loss:.4f}  l2={best_l2_loss:.4f}"
            )

    # Restore best model
    model.load_state_dict(best_model_state)

    # Build log dict
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

    # Save
    if save_path is not None and not os.path.exists(save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        np.savez_compressed(f"{save_path}.log.npz", **log)

    return TrainResult(model=model, log=log)
