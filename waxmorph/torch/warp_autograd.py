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
        f_net_wp: wp.array,
        grid: wp.HashGrid | None,
    ) -> torch.Tensor:
        """Apply a Warp mechanics step during the PyTorch forward pass.

        Args:
            ctx: PyTorch :class:`torch.autograd.Function` context.
            X_torch: Position tensor with shape ``[N, 3]``.
            R_wp: Warp radius array.
            particle_count: Number of active particles.
            dt: Mechanics Euler step size.
            f_net_wp: Scratch Warp net-force buffer.
            grid: Optional reusable :class:`warp.HashGrid`.

        Returns:
            Updated position tensor with shape ``[N, 3]``.
        """
        # zero-copy view of torch positions as a grad-tracked Warp array
        X_wp = wp.from_torch(X_torch.detach().contiguous(), dtype=wp.vec3f)
        X_wp.requires_grad = True

        # record pairwise force kernels on the tape (topology frozen for the step)
        tape = wp.Tape()
        X_out = mech_step_sticky_differentiable(
            tape, X_wp, R_wp, particle_count, dt, f_net_wp, grid
        )

        # stash tape + endpoints for backward replay
        ctx.tape = tape
        ctx.X_wp = X_wp
        ctx.X_out = X_out

        return wp.to_torch(X_out).view(-1, 3)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Replay the Warp tape and return the gradient with respect to positions."""
        # seed output adjoint, replay tape in reverse to accumulate input adjoint
        ctx.X_out.grad = wp.from_torch(grad_output.detach().contiguous(), dtype=wp.vec3f)
        ctx.tape.backward()
        # clone before zeroing: grad buffer is reused once the tape is cleared
        grad_input = wp.to_torch(ctx.X_wp.grad).view(-1, 3).clone()
        ctx.tape.zero()
        return grad_input, None, None, None, None, None


class WarpDiffusionStep(torch.autograd.Function):
    """Differentiable signaling-molecule diffusion step.

    Forward: graph-Laplacian diffusion of signaling-molecule concentrations.
    Backward: replay :class:`warp.Tape` to propagate ``dL/dc_out → dL/dc_in``.
    """

    @staticmethod
    def forward(
        ctx,
        c_torch: torch.Tensor,
        X_wp: wp.array,
        R_wp: wp.array,
        lap_c_wp: wp.array,
        particle_count: int,
        D_emu: float,
        dt: float,
        grid: wp.HashGrid | None,
    ) -> torch.Tensor:
        """Apply a Warp diffusion step during the PyTorch forward pass.

        Args:
            ctx: PyTorch :class:`torch.autograd.Function` context.
            c_torch: Concentration tensor with shape ``[N, num_molecules]``.
            X_wp: Warp position array used for neighbor topology.
            R_wp: Warp radius array.
            lap_c_wp: Scratch Warp Laplacian buffer.
            particle_count: Number of active particles.
            D_emu: Diffusion coefficient.
            dt: Diffusion Euler step size.
            grid: Optional reusable :class:`warp.HashGrid`.

        Returns:
            Updated concentration tensor with shape ``[N, num_molecules]``.
        """
        n, num_molecules = c_torch.shape
        # flatten then reshape: Warp 2d arrays need a contiguous 1d source
        c_wp = wp.from_torch(c_torch.detach().contiguous().view(-1), dtype=wp.float32)
        c_wp = c_wp.reshape((n, num_molecules))
        c_wp.requires_grad = True

        # record graph-Laplacian diffusion kernels on the tape
        tape = wp.Tape()
        c_out = diffusion_step_differentiable(
            tape, X_wp, R_wp, c_wp, lap_c_wp, particle_count, D_emu, dt, grid
        )

        ctx.tape = tape
        ctx.c_wp = c_wp
        ctx.c_out = c_out
        ctx.shape = (n, num_molecules)

        return wp.to_torch(c_out).view(n, num_molecules)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Replay the Warp tape and return the gradient with respect to concentrations."""
        n, num_molecules = ctx.shape
        # seed output adjoint (flatten/reshape to match the 2d Warp buffer)
        ctx.c_out.grad = wp.from_torch(
            grad_output.detach().contiguous().view(-1), dtype=wp.float32
        ).reshape((n, num_molecules))
        ctx.tape.backward()
        # clone before zeroing: grad buffer is reused once the tape is cleared
        grad_input = wp.to_torch(ctx.c_wp.grad).view(n, num_molecules).clone()
        ctx.tape.zero()
        return grad_input, None, None, None, None, None, None, None
