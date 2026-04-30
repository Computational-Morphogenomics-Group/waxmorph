"""PyTorch autograd functions that bridge Warp tape differentiation.

Each function wraps a Warp physics step so that :mod:`torch.autograd` can
chain backward through it.  Forward records kernel launches on a
:class:`warp.Tape`; backward replays the tape in reverse to propagate
gradients.
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
    Backward: replay :class:`warp.Tape` to propagate ``dL/dX_out → dL/dX_in``.
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
        """Apply a Warp mechanics step during the PyTorch forward pass.

        Args:
            ctx: PyTorch :class:`torch.autograd.Function` context.
            X_torch: Position tensor with shape ``[N, 3]``.
            R_wp: Warp radius array.
            particle_count: Number of active particles.
            dt: Mechanics Euler step size.
            gx_wp: Scratch Warp force buffer.
            grid: Optional reusable :class:`warp.HashGrid`.

        Returns:
            Updated position tensor with shape ``[N, 3]``.
        """
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
        """Replay the Warp tape and return the gradient with respect to positions."""
        ctx.X_out.grad = wp.from_torch(grad_output.detach().contiguous(), dtype=wp.vec3f)
        ctx.tape.backward()
        grad_input = wp.to_torch(ctx.X_wp.grad).view(-1, 3).clone()
        ctx.tape.zero()
        return grad_input, None, None, None, None, None


class WarpDiffusionStep(torch.autograd.Function):
    """Differentiable gene diffusion step.

    Forward: graph-Laplacian diffusion of gene concentrations.
    Backward: replay :class:`warp.Tape` to propagate ``dL/dG_out → dL/dG_in``.
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
        """Apply a Warp diffusion step during the PyTorch forward pass.

        Args:
            ctx: PyTorch :class:`torch.autograd.Function` context.
            G_torch: Gene concentration tensor with shape ``[N, num_genes]``.
            X_wp: Warp position array used for neighbor topology.
            R_wp: Warp radius array.
            lap_G_wp: Scratch Warp Laplacian buffer.
            particle_count: Number of active particles.
            alpha: Diffusion coefficient.
            dt: Diffusion Euler step size.
            grid: Optional reusable :class:`warp.HashGrid`.

        Returns:
            Updated gene concentration tensor with shape ``[N, num_genes]``.
        """
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
        """Replay the Warp tape and return the gradient with respect to genes."""
        n, num_genes = ctx.shape
        ctx.G_out.grad = wp.from_torch(
            grad_output.detach().contiguous().view(-1), dtype=wp.float32
        ).reshape((n, num_genes))
        ctx.tape.backward()
        grad_input = wp.to_torch(ctx.G_wp.grad).view(n, num_genes).clone()
        ctx.tape.zero()
        return grad_input, None, None, None, None, None, None, None
