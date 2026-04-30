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
    """Hyperparameters for PyTorch non-growing shape assembly training.

    Attributes:
        n_epochs: Number of optimization epochs.
        t_rollout: Number of rollout steps per epoch.
        mech_steps: Number of sticky-sphere mechanics corrections per rollout.
        diff_steps: Number of gene diffusion corrections per rollout.
        dt_mech: Euler step size for mechanics corrections.
        dt_diff: Euler step size for diffusion corrections.
        dt_gns: Scale applied to model-predicted deltas.
        alpha_diff: Gene diffusion coefficient.
        l2_lambda: Weight applied to squared model displacement regularization.
        grad_clip_norm: Optional maximum gradient norm.
        log_every: Epoch interval used for progress logging.
    """

    n_epochs: int = 2000
    t_rollout: int = 100
    mech_steps: int = 5
    diff_steps: int = 5
    dt_mech: float = 1e-2
    dt_diff: float = 1e-2
    dt_gns: float = 1e-2
    alpha_diff: float = 0.1
    l2_lambda: float = 1e-3
    grad_clip_norm: float | None = 1.0
    log_every: int = 10


@dataclasses.dataclass
class TrainResult:
    """Result returned by :func:`waxmorph.torch.train.train`.

    Attributes:
        model: Best :class:`torch.nn.Module` found during training, or the
            latest model if no finite improvement was recorded.
        log: Dictionary containing loss histories, best-epoch metadata, and
            the best trajectory.
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


def _run_epoch(
    *,
    model,
    config,
    source_pos,
    polarities,
    genes,
    R_t,
    R_wp,
    gx,
    lap_G,
    grid,
    N,
    targets_by_frame,
    X_source_t,
    loss_fn,
    torch_device,
    epoch_trajectory,
):
    """Run one training epoch with differentiable physics.

    Positions and genes stay on the :mod:`torch.autograd` computation graph throughout.
    Physics corrections are applied via WarpMechStep / WarpDiffusionStep
    autograd functions, so gradients flow through the full trajectory.

    ``targets_by_frame`` maps rollout-step index -> target position tensor.
    A shape loss is accumulated at every tagged post-update state; frame ``0``
    therefore supervises the state after the first rollout update, not the
    initial source state.
    """
    X_t = X_source_t.clone().requires_grad_(True)
    G_t = torch.from_numpy(genes.copy()).to(torch_device).requires_grad_(True)
    P_t = torch.from_numpy(polarities.copy()).to(torch_device)

    loss_l2 = torch.tensor(0.0, device=torch_device)
    loss_shape = torch.tensor(0.0, device=torch_device)

    _validate_finite_tensor("X_t", X_t, rollout_step=0, phase="epoch start")
    _validate_finite_tensor("P_t", P_t, rollout_step=0, phase="epoch start")
    _validate_finite_tensor("G_t", G_t, rollout_step=0, phase="epoch start")

    for _t in range(config.t_rollout):
        _validate_finite_tensor("X_t", X_t, rollout_step=_t, phase="pre-graph build")
        _validate_finite_tensor("P_t", P_t, rollout_step=_t, phase="pre-graph build")
        _validate_finite_tensor("G_t", G_t, rollout_step=_t, phase="pre-graph build")

        # Build graph from the live Torch state. Feature gradients remain
        # connected to X_t / P_t / G_t; edge_index is rebuilt from a snapshot.
        node_feats, edge_index, edge_feats = build_graph(
            X_t,
            P_t,
            R_t,
            particle_count=N,
            G=G_t,
        )

        # GNS forward (differentiable)
        out = model(node_feats, edge_index, edge_feats)
        dX = out["dX"] * config.dt_gns
        dP = out["dP"] * config.dt_gns
        dG = out["dG"] * config.dt_gns
        _validate_finite_tensor("dX", dX, rollout_step=_t, phase="gns output")
        _validate_finite_tensor("dP", dP, rollout_step=_t, phase="gns output")
        _validate_finite_tensor("dG", dG, rollout_step=_t, phase="gns output")

        loss_l2 = loss_l2 + dX.square().sum()

        # Apply GNS deltas (stays on PyTorch graph)
        X_t = X_t + dX
        P_t = torch.nn.functional.normalize(P_t + dP, dim=-1)
        G_t = torch.clamp_min(G_t + dG, 0.0)
        _validate_finite_tensor("X_t", X_t, rollout_step=_t, phase="post-gns update")
        _validate_finite_tensor("P_t", P_t, rollout_step=_t, phase="post-gns update")
        _validate_finite_tensor("G_t", G_t, rollout_step=_t, phase="post-gns update")

        # Physics correction (differentiable via autograd functions)
        for _ in range(config.mech_steps):
            X_t = WarpMechStep.apply(X_t, R_wp, N, config.dt_mech, gx, grid)
        _validate_finite_tensor("X_t", X_t, rollout_step=_t, phase="post-mechanics")

        for _ in range(config.diff_steps):
            X_wp_diff = wp.from_torch(X_t.detach().contiguous(), dtype=wp.vec3f)
            G_t = WarpDiffusionStep.apply(
                G_t, X_wp_diff, R_wp, lap_G, N, config.alpha_diff, config.dt_diff, grid
            )
        _validate_finite_tensor("G_t", G_t, rollout_step=_t, phase="post-diffusion")

        epoch_trajectory.append(
            {
                "pos": X_t.detach().cpu().numpy().copy(),
                "pol": P_t.detach().cpu().numpy().copy(),
                "genes": G_t.detach().cpu().numpy().copy(),
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
    genes: np.ndarray,
    radii: np.ndarray,
    target_pos: np.ndarray | None = None,
    targets: list[tuple[int, np.ndarray]] | None = None,
    config: TrainConfig | None = None,
    save_path: str | Path | None = None,
    device: str = "cuda",
) -> TrainResult:
    """Train a GNS model for non-growing shape assembly (PyTorch backend).

    Args:
        model: Graph Network Simulator model.
        optimizer: :class:`torch.optim.Optimizer`.
        loss_fn: Shape loss function mapping predicted positions with shape
            ``[N, 3]`` and target positions with shape ``[M, 3]`` to a scalar.
        source_pos: Initial particle positions with shape ``[N, 3]``.
        polarities: Initial polarity vectors with shape ``[N, 3]``.
        genes: Initial gene concentrations with shape ``[N, num_genes]``.
        radii: Particle radii with shape ``[N]``.
        target_pos: Legacy single-target input, equivalent to
            ``targets=[(t_rollout - 1, target_pos)]``.
        targets: Optional ``(frame, positions)`` supervision pairs. Frame ``0``
            supervises the state after the first rollout update. Frames must
            lie in ``[0, t_rollout)`` and must be unique.
        config: Training hyperparameters. Defaults to
            :class:`waxmorph.torch.train.TrainConfig`.
        save_path: Optional path where the best model and log are saved.
        device: Warp and :class:`torch.device` string such as ``"cuda"`` or
            ``"cpu"``.

    Returns:
        Best model and full training log.

    Raises:
        ValueError: If targets are missing, duplicated, out of range, mutually
            exclusive with ``target_pos``, or contain non-finite values.
    """
    if config is None:
        config = TrainConfig()

    if targets is None and target_pos is None:
        raise ValueError("train() requires either `targets` or `target_pos`.")
    if targets is not None and target_pos is not None:
        raise ValueError("Pass `targets` OR `target_pos`, not both.")

    if targets is None:
        targets = [(config.t_rollout - 1, target_pos)]

    _validate_finite_numpy("source_pos", source_pos)
    _validate_finite_numpy("polarities", polarities)
    _validate_finite_numpy("genes", genes)
    _validate_finite_numpy("radii", radii)

    # Detect devices
    wp_device = device
    torch_device = torch.device(device)
    model = model.to(torch_device)

    N = len(source_pos)
    max_particles = N
    num_genes = genes.shape[1]

    # Checkpoint detection
    if save_path is not None and os.path.exists(save_path):
        model = GNS.load(save_path, map_location=torch_device).to(torch_device)
        config = dataclasses.replace(config, n_epochs=1)

    # Warp scratch buffers
    gx = wp.zeros(max_particles, dtype=wp.vec3f, device=wp_device)
    R_t = torch.from_numpy(radii.copy()).to(torch_device)
    R_wp = wp.from_numpy(radii.copy(), dtype=wp.float32, device=wp_device)
    lap_G = wp.zeros((max_particles, num_genes), dtype=wp.float32, device=wp_device)

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

        epoch_trajectory = [
            {"pos": source_pos.copy(), "pol": polarities.copy(), "genes": genes.copy()}
        ]

        loss_shape, loss_l2, epoch_trajectory = _run_epoch(
            model=model,
            config=config,
            source_pos=source_pos,
            polarities=polarities,
            genes=genes,
            R_t=R_t,
            R_wp=R_wp,
            gx=gx,
            lap_G=lap_G,
            grid=grid,
            N=N,
            targets_by_frame=targets_by_frame,
            X_source_t=X_source_t,
            loss_fn=loss_fn,
            torch_device=torch_device,
            epoch_trajectory=epoch_trajectory,
        )

        loss = loss_shape + (loss_l2 * config.l2_lambda)

        loss.backward()
        _raise_on_nonfinite_named_tensors(
            "gradients",
            ((name, param.grad) for name, param in model.named_parameters()),
            epoch=epoch,
            phase="post-backward",
        )
        if config.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
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
    traj_genes = np.stack([f["genes"] for f in best_trajectory])

    log = {
        "losses_total": np.array(losses_total),
        "losses_shape": np.array(losses_shape),
        "losses_l2": np.array(losses_l2),
        "best_traj_pos": traj_pos,
        "best_traj_pol": traj_pol,
        "best_traj_genes": traj_genes,
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
