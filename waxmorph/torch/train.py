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

from waxmorph.emulator import diffusion_step, mech_step_sticky
from waxmorph.torch.gnn import GNS
from waxmorph.torch.graph import build_graph


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    n_epochs: int = 1000
    t_rollout: int = 200
    mech_steps: int = 5
    diff_steps: int = 5
    dt_mech: float = 1e-2
    dt_diff: float = 1e-2
    dt_gns: float = 1e-2
    alpha_diff: float = 0.1
    l2_lambda: float = 1e-3
    log_every: int = 10


@dataclasses.dataclass
class TrainResult:
    model: Any
    log: dict


def train(
    model: GNS,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    source_pos: np.ndarray,
    target_pos: np.ndarray,
    polarities: np.ndarray,
    genes: np.ndarray,
    radii: np.ndarray,
    config: TrainConfig | None = None,
    save_path: str | Path | None = None,
    device: str = "cuda",
) -> TrainResult:
    """Train a GNS model for non-growing shape assembly (PyTorch backend).

    Parameters
    ----------
    model : GNS
        Graph Network Simulator model.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer (e.g. AdamW).
    loss_fn : callable
        Shape loss function mapping (predicted [N,3], target [M,3]) -> scalar.
    source_pos : ndarray [N, 3]
        Initial particle positions.
    target_pos : ndarray [M, 3]
        Target positions for shape loss.
    polarities : ndarray [N, 3]
        Initial unit polarity vectors.
    genes : ndarray [N, num_genes]
        Initial gene concentrations.
    radii : ndarray [N]
        Particle radii.
    config : TrainConfig, optional
        Training hyperparameters (defaults used if None).
    save_path : str or Path, optional
        If set, saves best model and log to disk.
    device : str
        Warp device string ("cuda" or "cpu").

    Returns
    -------
    TrainResult
        Best model and full training log.
    """
    if config is None:
        config = TrainConfig()

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
    R = wp.from_numpy(radii.copy(), dtype=wp.float32, device=wp_device)
    lap_G = wp.zeros((max_particles, num_genes), dtype=wp.float32, device=wp_device)

    # Target tensor
    X_target = torch.from_numpy(target_pos).to(torch_device)
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

    for epoch in trange(config.n_epochs):
        optimizer.zero_grad()

        # Reset Warp state
        X_current = wp.from_numpy(source_pos.copy(), dtype=wp.vec3f, device=wp_device)
        G_current = wp.from_numpy(genes.copy(), dtype=wp.float32, device=wp_device)
        P_current = wp.from_numpy(polarities.copy(), dtype=wp.vec3f, device=wp_device)

        total_dX = torch.zeros(N, 3, device=torch_device)
        loss_l2 = torch.tensor(0.0, requires_grad=True)

        epoch_trajectory = [
            {"pos": source_pos.copy(), "pol": polarities.copy(), "genes": genes.copy()}
        ]

        for _t in range(config.t_rollout):
            node_feats, edge_index, edge_feats = build_graph(
                X_current,
                P_current,
                R,
                particle_count=N,
                G=G_current,
            )

            out = model(node_feats, edge_index, edge_feats)

            dX = out["dX"] * config.dt_gns
            dP = out["dP"] * config.dt_gns
            dG = out["dG"] * config.dt_gns

            total_dX = total_dX + dX

            loss_l2 = loss_l2 + dX.square().sum()
            loss_l2 = loss_l2 + dP.square().sum()
            loss_l2 = loss_l2 + dG.square().sum()

            # Detach to numpy for Warp state update
            dX_np = dX.detach().cpu().numpy()
            dP_np = dP.detach().cpu().numpy()
            dG_np = dG.detach().cpu().numpy()

            # Update positions
            x_np = X_current.numpy()
            x_np += dX_np
            X_current = wp.from_numpy(x_np, dtype=wp.vec3f, device=wp_device)

            # Update polarities (normalize)
            p_np = P_current.numpy()
            p_np += dP_np
            p_norms = np.linalg.norm(p_np, axis=-1, keepdims=True)
            p_np /= np.maximum(p_norms, 1e-9)
            P_current = wp.from_numpy(p_np, dtype=wp.vec3f, device=wp_device)

            # Update genes
            g_np = G_current.numpy()
            g_np += dG_np
            G_current = wp.from_numpy(g_np, dtype=wp.float32, device=wp_device)

            # Physics steps
            for _ in range(config.mech_steps):
                mech_step_sticky(X_current, R, N, config.dt_mech, gx)
            for _ in range(config.diff_steps):
                diffusion_step(X_current, R, G_current, lap_G, N, config.alpha_diff, config.dt_diff)

            epoch_trajectory.append(
                {
                    "pos": X_current.numpy().copy(),
                    "pol": P_current.numpy().copy(),
                    "genes": G_current.numpy().copy(),
                }
            )

        # Loss
        X_pred = X_source_t + total_dX
        loss_shape = loss_fn(X_pred, X_target)
        loss = loss_shape + (loss_l2 * config.l2_lambda)

        loss.backward()
        optimizer.step()

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
