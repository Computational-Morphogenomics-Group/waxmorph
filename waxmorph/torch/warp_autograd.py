"""PyTorch autograd functions that bridge Warp tape differentiation.

Each function wraps a Warp physics step so that PyTorch's autograd can
chain backward through it.  Forward records kernel launches on a
``wp.Tape``; backward replays the tape in reverse to propagate gradients.
"""

from __future__ import annotations

import torch
import warp as wp

from waxmorph.emulator import (
    diffusion_step_differentiable,
    mech_step_sticky_differentiable,
)


class WarpMechStep(torch.autograd.Function):
    """Differentiable sticky-sphere mechanics step.

    Forward: apply repulsion + adhesion forces and Euler-update positions.
    Backward: replay ``wp.Tape`` to propagate ``dL/dX_out → dL/dX_in``.
    """

    @staticmethod
    def forward(
        ctx,
        X_torch: torch.Tensor,
        R_wp: wp.array,
        particle_count: int,
        dt: float,
        gx_wp: wp.array,
        grid: wp.HashGrid | None,
    ) -> torch.Tensor:
        X_wp = wp.from_torch(X_torch.detach().contiguous(), dtype=wp.vec3f)
        X_wp.requires_grad = True

        tape = wp.Tape()
        X_out = mech_step_sticky_differentiable(tape, X_wp, R_wp, particle_count, dt, gx_wp, grid)

        ctx.tape = tape
        ctx.X_wp = X_wp
        ctx.X_out = X_out

        return wp.to_torch(X_out).view(-1, 3)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ctx.X_out.grad = wp.from_torch(grad_output.detach().contiguous(), dtype=wp.vec3f)
        ctx.tape.backward()
        grad_input = wp.to_torch(ctx.X_wp.grad).view(-1, 3).clone()
        ctx.tape.zero()
        return grad_input, None, None, None, None, None


class WarpDiffusionStep(torch.autograd.Function):
    """Differentiable gene diffusion step.

    Forward: graph-Laplacian diffusion of gene concentrations.
    Backward: replay ``wp.Tape`` to propagate ``dL/dG_out → dL/dG_in``.
    """

    @staticmethod
    def forward(
        ctx,
        G_torch: torch.Tensor,
        X_wp: wp.array,
        R_wp: wp.array,
        lap_G_wp: wp.array,
        particle_count: int,
        alpha: float,
        dt: float,
        grid: wp.HashGrid | None,
    ) -> torch.Tensor:
        n, num_genes = G_torch.shape
        G_wp = wp.from_torch(G_torch.detach().contiguous().view(-1), dtype=wp.float32)
        G_wp = G_wp.reshape((n, num_genes))
        G_wp.requires_grad = True

        tape = wp.Tape()
        G_out = diffusion_step_differentiable(
            tape, X_wp, R_wp, G_wp, lap_G_wp, particle_count, alpha, dt, grid
        )

        ctx.tape = tape
        ctx.G_wp = G_wp
        ctx.G_out = G_out
        ctx.shape = (n, num_genes)

        return wp.to_torch(G_out).view(n, num_genes)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        n, num_genes = ctx.shape
        ctx.G_out.grad = wp.from_torch(
            grad_output.detach().contiguous().view(-1), dtype=wp.float32
        ).reshape((n, num_genes))
        ctx.tape.backward()
        grad_input = wp.to_torch(ctx.G_wp.grad).view(n, num_genes).clone()
        ctx.tape.zero()
        return grad_input, None, None, None, None, None, None, None
