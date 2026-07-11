"""Torch autograd adapters for Warp mechanics and diffusion.

Each forward call owns a fresh :class:`warp.Tape` and a frozen neighbor list. Backward seeds
the output adjoint and replays that tape; only the documented Torch state receives a gradient.
Torch uses the input's Warp-compatible device. JAX instead uses CUDA-only custom VJPs; the
bridges do not promise identical gradients.
"""

from __future__ import annotations

import torch
import warp as wp

from waxmorph.emulator import (
    diffusion_step_differentiable,
    mech_step_sticky_differentiable,
)


class WarpMechStep(torch.autograd.Function):
    r"""Sticky-sphere Euler step differentiating positions only.

    Forward gives Warp a zero-copy view of contiguous detached Torch storage, discovers pairs
    outside a fresh tape, and records :math:`X_{out}=X+\Delta t\,F(X,R)` on that frozen list.
    Backward returns :math:`dL/dX`; radii and scratch objects receive no Torch gradient.
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
        # Warp shares storage only with the contiguous detached tensor.
        X_wp = wp.from_torch(X_torch.detach().contiguous(), dtype=wp.vec3f)
        X_wp.requires_grad = True

        tape = wp.Tape()
        X_out = mech_step_sticky_differentiable(
            tape, X_wp, R_wp, particle_count, dt, f_net_wp, grid
        )

        ctx.tape = tape
        ctx.X_wp = X_wp
        ctx.X_out = X_out

        return wp.to_torch(X_out).view(-1, 3)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ctx.X_out.grad = wp.from_torch(grad_output.detach().contiguous(), dtype=wp.vec3f)
        ctx.tape.backward()
        # Clone before tape.zero() clears the shared gradient buffer in place.
        grad_input = wp.to_torch(ctx.X_wp.grad).view(-1, 3).clone()
        ctx.tape.zero()
        return grad_input, None, None, None, None, None


class WarpDiffusionStep(torch.autograd.Function):
    r"""Graph diffusion step differentiating concentrations only.

    On frozen pairs, forward records
    :math:`c_{out}=\max(c-D_{emu}L_Gc\,\Delta t,0)` independently per molecule.
    Positions and radii determine topology but receive no gradient. Warp views contiguous
    flattened Torch storage reshaped to ``[N, C]``.
    """

    @staticmethod
    def forward(
        ctx,
        c_torch: torch.Tensor,
        X_wp: wp.array,
        R_wp: wp.array,
        particle_count: int,
        D_emu: float,
        dt: float,
        grid: wp.HashGrid | None,
    ) -> torch.Tensor:
        n, num_molecules = c_torch.shape
        c_wp = wp.from_torch(c_torch.detach().contiguous().view(-1), dtype=wp.float32)
        c_wp = c_wp.reshape((n, num_molecules))
        c_wp.requires_grad = True

        tape = wp.Tape()
        c_out = diffusion_step_differentiable(
            tape, X_wp, R_wp, c_wp, particle_count, D_emu, dt, grid
        )

        ctx.tape = tape
        ctx.c_wp = c_wp
        ctx.c_out = c_out
        ctx.shape = (n, num_molecules)

        return wp.to_torch(c_out).view(n, num_molecules)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        n, num_molecules = ctx.shape
        ctx.c_out.grad = wp.from_torch(
            grad_output.detach().contiguous().view(-1), dtype=wp.float32
        ).reshape((n, num_molecules))
        ctx.tape.backward()
        # Clone before tape.zero() clears the shared gradient buffer in place.
        grad_input = wp.to_torch(ctx.c_wp.grad).view(n, num_molecules).clone()
        ctx.tape.zero()
        return grad_input, None, None, None, None, None, None
