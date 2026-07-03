"""Warp mechanics and graph-Laplacian diffusion for the differentiable path.

The differentiable counterpart of :mod:`waxmorph.simulator`: it advances a
*non-growing* tissue (fixed particle count) under the same shared physical
primitives -- soft-sphere mechanics and graph-based molecular diffusion --
but exposes them as gradient-carrying steps for the inverse-design emulator.
Each step *freezes the neighbour topology*: neighbour
discovery runs once, outside the tape, and the resulting pair list is held
constant while the pairwise kernels are recorded on a :class:`warp.Tape`.
``tape.backward()`` then replays those launches in reverse to propagate
gradients with respect to the deformable input (positions or concentrations).

Because topology is frozen per step, gradients flow through the recorded
arithmetic only; they do *not* flow through neighbour selection, the hash-grid
build, or the (constant) radii. The simulator twin uses the same force law and
graph Laplacian but is not differentiable and supports a growing particle
count; see the See Also notes on the individual entry points for the exact
sign-convention difference.
"""

import warp as wp

from .constants import EPS_DIST, EPS_NORM, HASH_GRID_DIM

############################################################
############################################################
############################################################

# HELPERS

############################################################
############################################################
############################################################

# Soft-sphere coefficients for the emulator path: the k_rep and k_att terms in
# the soft-sphere force.
# Intentionally distinct from the simulator's type-dependent values
# (waxmorph.simulator.K_REP / K_ATT_*): the emulator uses a single uniform
# radius and one attraction strength for every pair, so a single k_att suffices.
K_REP = 2.0
K_ATT = 0.5


@wp.func
def _sticky_sphere_forces(
    x_i: wp.vec3f,
    x_j: wp.vec3f,
    r_i: wp.float32,
    r_j: wp.float32,
):
    r"""Warp implementation of the soft-sphere force for one pair.

    Volume exclusion with short-range adhesion between neighbouring agents
    ``i`` and ``j``. With center distance :math:`d_{ij}=\lVert x_i-x_j\rVert_2`
    and unit direction :math:`\hat d_{ij}=(x_i-x_j)/d_{ij}` (here ``u``),

    .. math::

        F_{ij} = \Bigl[\, k_{\mathrm{rep}}\,\max(r_i+r_j-\varepsilon-d_{ij},\,0)
                 - k_{\mathrm{att}}\,\max(r_i+r_j+\varepsilon-d_{ij},\,0)\,
                   \mathbb{1}[d_{ij} > r_i+r_j-\varepsilon] \,\Bigr]\,\hat d_{ij},

    with :math:`k_{\mathrm{rep}}` = ``K_REP``, :math:`k_{\mathrm{att}}` =
    ``K_ATT`` and :math:`\varepsilon` = ``EPS_DIST``. The repulsive branch acts
    for :math:`d_{ij} < r_i+r_j-\varepsilon` (``drep``) and a short-range
    attractive branch acts over :math:`[r_i+r_j-\varepsilon,\,r_i+r_j+\varepsilon]`
    (up to ``datr``). By Newton's third law the reaction on ``j`` is
    :math:`F_{ji}=-F_{ij}`.

    Sign convention (load-bearing for the differentiable path): the returned
    ``f_ij`` is a *force*, i.e. ``+u`` times the magnitude, equal to
    :math:`-\nabla_{x_i} U`. It is therefore consumed with a ``+`` sign in
    :func:`_gd_update` (gradient *ascent* on positions). The simulator twin
    instead returns the potential *gradient* ``fmag * -u`` and applies it with
    a ``-`` sign in ``gd_update``; the two double-negations cancel, so both
    paths integrate the identical physics.

    The ``max`` clamps and the ``dist > drep`` indicator are non-smooth: Warp
    differentiates them with the one-sided (subgradient) convention, taking the
    derivative as zero on the inactive side of each branch.

    Args:
        x_i: Center of agent ``i``.
        x_j: Center of agent ``j``.
        r_i: Radius of agent ``i``.
        r_j: Radius of agent ``j``.

    Returns:
        Pair ``(f_ij, f_ji)`` of equal-and-opposite forces on agents ``i`` and
        ``j``, each a ``vec3f`` with ``f_ji == -f_ij``.

    See Also:
        :func:`waxmorph.simulator.sticky_sphere_forces`: non-differentiable,
            type-dependent twin that returns the potential gradient (``-u``
            convention) and adds epithelial/mesenchymal adhesion terms.
    """
    d = x_i - x_j
    dist = wp.length(d) + EPS_NORM
    u = d / dist

    rs = r_i + r_j
    drep = rs - EPS_DIST
    datr = rs + EPS_DIST

    f_rep = K_REP * wp.max(drep - dist, wp.float32(0.0))
    f_att = K_ATT * wp.max(datr - dist, wp.float32(0.0)) * wp.float32(dist > drep)

    fmag = f_rep - f_att
    # f_ij is a FORCE (-grad U): +u * magnitude, applied with '+' in _gd_update.
    f_ij = fmag * u
    return f_ij, -f_ij


@wp.kernel
def _build_neighbor_pairs(
    grid: wp.uint64,
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    query_radius: wp.float32,
    edges_i: wp.array(dtype=wp.int32),
    edges_j: wp.array(dtype=wp.int32),
    edge_count: wp.array(dtype=wp.int32),
    max_edges: wp.int32,
):
    r"""Emit the frozen neighbour pair list ``(i, j)`` with ``i < j``.

    Spatial-adjacency graph for one step: a pair is recorded when the agents
    are in mechanical/diffusive contact,
    :math:`\lVert x_i-x_j\rVert_2 \le r_i+r_j+\varepsilon` (``threshold``),
    matching the range of the soft-sphere force. Each unordered pair is emitted
    once (``j <= i`` is skipped) and appended atomically; ``edge_count`` keeps
    counting even past ``max_edges`` so the caller can detect an overflow and
    retry with a larger buffer.

    This runs *outside* any :class:`warp.Tape`: the pair list it produces is the
    topology that the differentiable kernels treat as constant for the step, so
    no gradient flows through this discovery step.

    Args:
        grid: Built hash grid over the active positions.
        X: Agent center positions.
        R: Agent radii.
        query_radius: Hash-grid query radius (the contact threshold).
        edges_i: Output buffer for the lower endpoint of each pair.
        edges_j: Output buffer for the upper endpoint of each pair.
        edge_count: Single-element counter of pairs found (may exceed
            ``max_edges``).
        max_edges: Capacity of ``edges_i`` / ``edges_j``; writes past it are
            suppressed.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x_i = X[i]
    r_i = R[i]

    for j in wp.hash_grid_query(grid, x_i, query_radius):
        if j <= i:
            continue
        dist = wp.length(x_i - X[j]) + EPS_NORM
        threshold = r_i + R[j] + EPS_DIST
        if dist > threshold:
            continue
        idx = wp.atomic_add(edge_count, 0, 1)
        if idx < max_edges:
            edges_i[idx] = i
            edges_j[idx] = j


@wp.kernel
def _sticky_sphere_grads_from_pairs(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    edges_i: wp.array(dtype=wp.int32),
    edges_j: wp.array(dtype=wp.int32),
    f_net: wp.array(dtype=wp.vec3f),
):
    """Accumulate the net soft-sphere force per agent from a frozen pair list.

    Sums the per-pair soft-sphere forces (via :func:`_sticky_sphere_forces`)
    into ``f_net``, the net force :math:`f^{\\mathrm{net}}_i=\\sum_{j} F_{ij}` that
    drives the overdamped Euler position update.

    Launched with ``dim=num_edges`` over the precomputed pairs, so it issues no
    hash-grid queries. Every operation it performs (array reads, arithmetic,
    ``atomic_add`` scatter) is differentiable in Warp, which is why it is the
    part recorded on the tape: ``tape.backward()`` replays this scatter to map
    ``dL/df_net`` back to ``dL/dX``. Topology is fixed because the pair indices
    are inputs, not queried here.

    Args:
        X: Agent center positions.
        R: Agent radii (constant; not differentiated).
        edges_i: Lower endpoints of the frozen pairs.
        edges_j: Upper endpoints of the frozen pairs.
        f_net: Net-force accumulator, scattered into by both endpoints.
    """
    e = wp.tid()
    i = edges_i[e]
    j = edges_j[e]
    force_i, force_j = _sticky_sphere_forces(X[i], X[j], R[i], R[j])
    wp.atomic_add(f_net, i, force_i)
    wp.atomic_add(f_net, j, force_j)


@wp.kernel
def _gd_update(
    X: wp.array(dtype=wp.vec3f),
    f_net: wp.array(dtype=wp.vec3f),
    lr: wp.float32,
    X_next: wp.array(dtype=wp.vec3f),
):
    r"""Forward-Euler position update for overdamped motion (one step).

    Integrates :math:`\nu_i\dot x_i = f^{\mathrm{net}}_i` with the friction
    coefficient absorbed into ``lr``:

    .. math::

        x_i \leftarrow x_i + \Delta t\, f^{\mathrm{net}}_i .

    Note:
        ``f_net`` is a *force* (:math:`-\nabla_{x_i} U`), so the update adds it
        with a ``+`` sign. The simulator twin instead stores the potential
        *gradient* and subtracts it (``X_next = X - lr * f_net``); the opposite
        sign convention between the two stems entirely from the force-vs-gradient
        choice in :func:`_sticky_sphere_forces` and yields identical dynamics.

    Args:
        X: Current agent positions.
        f_net: Net force per agent.
        lr: Euler step size (``dt`` with friction folded in).
        X_next: Output buffer for the updated positions.

    See Also:
        :func:`waxmorph.simulator.gd_update`: non-differentiable twin using the
            ``-`` (gradient-descent) sign convention.
    """
    i = wp.tid()
    # f_net is a FORCE, hence '+': the simulator twin stores -grad U and uses '-'.
    X_next[i] = X[i] + lr * f_net[i]


def _build_neighbor_pairs_dynamic(
    X: wp.array,
    R: wp.array,
    particle_count: int,
    query_radius: float,
    grid: "wp.HashGrid",
    device,
) -> tuple[wp.array, wp.array, int]:
    """Build the frozen neighbour pair list, growing the buffer on overflow.

    Host-side driver for :func:`_build_neighbor_pairs`. It is the single source
    of the per-step topology shared by the differentiable mechanics and
    diffusion paths, both of which launch pair kernels with ``dim=num_edges``.
    Starting from a generous guess (``20 * particle_count``), it relaunches
    with the realized count whenever the first pass overflows, so the returned
    ``num_edges`` never exceeds the buffer capacity and the downstream launch
    dimension can never index past ``edges_i`` / ``edges_j``.

    Runs entirely outside any :class:`warp.Tape`; the topology it returns is
    treated as constant for the step (no gradient flows through pair discovery).

    Args:
        X: Agent center positions.
        R: Agent radii.
        particle_count: Number of active particles to build the grid over.
        query_radius: Hash-grid query radius (the contact threshold).
        grid: Built :class:`warp.HashGrid` over the active positions.
        device: Warp device on which the buffers are allocated.

    Returns:
        Tuple ``(edges_i, edges_j, num_edges)``: the lower- and upper-endpoint
        index buffers (each of length ``max(num_edges, 20 * particle_count)``)
        and the number of valid pairs stored in their leading ``num_edges``
        entries.
    """
    max_edges = max(particle_count * 20, 1)

    while True:
        edges_i = wp.zeros(max_edges, dtype=wp.int32, device=device)
        edges_j = wp.zeros(max_edges, dtype=wp.int32, device=device)
        edge_count = wp.zeros(1, dtype=wp.int32, device=device)

        wp.launch(
            _build_neighbor_pairs,
            dim=particle_count,
            inputs=[
                wp.uint64(grid.id),
                X,
                R,
                query_radius,
                edges_i,
                edges_j,
                edge_count,
                max_edges,
            ],
            device=device,
        )
        num_edges = int(edge_count.numpy()[0])

        if num_edges <= max_edges:
            return edges_i, edges_j, num_edges

        max_edges = num_edges


############################################################
############################################################
############################################################

# DIFFUSION

############################################################
############################################################
############################################################


@wp.kernel
def _molecule_diffusion_laplacian_from_pairs(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    c: wp.array2d(dtype=wp.float32),
    edges_i: wp.array(dtype=wp.int32),
    edges_j: wp.array(dtype=wp.int32),
    lap_c: wp.array2d(dtype=wp.float32),
):
    r"""Accumulate the graph Laplacian per molecule from frozen pairs.

    Warp implementation of the discrete graph Laplacian of the contact graph,

    .. math::

        (L_G c)_i = \sum_{j:(i,j)\in E} (c_i - c_j),

    evaluated independently for every molecule channel ``g``. Edges are
    iterated once each (``i < j``); the symmetric contribution
    :math:`c_i - c_j` is scattered to ``i`` and its negation to ``j``, so both
    endpoints accumulate the correct signed flux. (The kernel computes
    ``flux = c[j] - c[i]`` and adds it to ``i``, i.e. ``-(c_i - c_j)``, which is
    the *minus* Laplacian; the downstream Euler step folds this sign into a
    ``+`` update -- see :func:`_molecule_diffusion_step_out`.)

    Launched with ``dim=num_edges`` over the precomputed pairs, so it issues no
    hash-grid queries; all reads, arithmetic, and ``atomic_add`` scatters are
    differentiable in Warp. This is the part of the diffusion operator recorded
    on the tape, letting ``tape.backward()`` carry ``dL/dlap_c`` back to
    ``dL/dc``. The edge set is fixed for the step (the frozen-topology
    approximation), so the gradient ignores how concentrations would have
    reshaped the neighbour graph.

    Args:
        X: Agent positions (unused here; topology is carried by the pairs).
        R: Agent radii (unused here; present for launch-signature symmetry).
        c: Concentration field, shape ``[N, num_molecules]``.
        edges_i: Lower endpoints of the frozen pairs.
        edges_j: Upper endpoints of the frozen pairs.
        lap_c: Output Laplacian accumulator, shape ``[N, num_molecules]``.
    """
    e = wp.tid()
    i = edges_i[e]
    j = edges_j[e]

    num_molecules = c.shape[1]
    for g in range(num_molecules):
        flux = c[j, g] - c[i, g]
        wp.atomic_add(lap_c, i, g, flux)
        wp.atomic_add(lap_c, j, g, -flux)


@wp.kernel
def _molecule_diffusion_step_out(
    c: wp.array2d(dtype=wp.float32),
    lap_c: wp.array2d(dtype=wp.float32),
    D_emu: wp.float32,
    dt: wp.float32,
    particle_count: wp.int32,
    c_out: wp.array2d(dtype=wp.float32),
):
    r"""Out-of-place forward-Euler diffusion update for one molecule channel.

    Integrates graph diffusion of the latent signaling molecules,

    .. math::

        c_{i,M} \leftarrow c_{i,M} - D_{\mathrm{emu}}\,(L_G c)_{i,M}\,\Delta t,

    with :math:`D_{\mathrm{emu}}` = ``D_emu``. Here ``lap_c`` already holds the
    *minus* Laplacian :math:`-(L_G c)` (see
    :func:`_molecule_diffusion_laplacian_from_pairs`), so the ``+`` sign in the
    code realizes the ``-`` sign of the update above. Inactive slots
    (``i >= particle_count``) are copied through unchanged.

    The result is clamped to nonnegative concentrations via ``wp.max(v, 0)``;
    Warp differentiates this clamp with the one-sided convention (gradient zero
    where the clamp is active). Writing to a distinct ``c_out`` keeps the step
    tape-safe -- replaying an in-place update would corrupt the recorded
    adjoints.

    Args:
        c: Input concentrations, shape ``[N, num_molecules]``.
        lap_c: Minus-Laplacian buffer, shape ``[N, num_molecules]``.
        D_emu: Diffusion coefficient.
        dt: Diffusion Euler step size.
        particle_count: Number of active particles; rows at or beyond it are
            passed through.
        c_out: Output concentrations (must alias a different array than ``c``).
    """
    i, g = wp.tid()
    if i >= particle_count:
        c_out[i, g] = c[i, g]
        return

    v = c[i, g] + dt * D_emu * lap_c[i, g]
    c_out[i, g] = wp.max(v, wp.float32(0.0))


############################################################
############################################################
############################################################

# RUNNERS

############################################################
############################################################
############################################################


def mech_step_sticky_differentiable(
    tape: "wp.Tape",
    X: wp.array,
    R: wp.array,
    particle_count: int,
    dt: float,
    f_net: wp.array,
    grid: "wp.HashGrid | None" = None,
) -> wp.array:
    """Record one differentiable soft-sphere mechanics step on a Warp tape.

    Advances positions by the overdamped Euler update driven by the net
    soft-sphere force, exposing the step to autodiff. Neighbour
    discovery (hash-grid build + :func:`_build_neighbor_pairs_dynamic`) runs
    *before* and *outside* ``tape``, so the contact graph is frozen for the
    step. Only the two pair-based kernels -- force accumulation
    (:func:`_sticky_sphere_grads_from_pairs`) and the Euler position update
    (:func:`_gd_update`) -- are recorded, so ``tape.backward()`` propagates
    ``dL/dX_out -> dL/dX_in`` and nothing else.

    Gradient does *not* flow through: neighbour selection, the hash-grid build,
    the (constant) radii ``R``, or the frozen edge indices. It is the caller's
    job to seed ``X_out.grad`` and replay the tape (see
    :class:`waxmorph.torch.warp_autograd.WarpMechStep`).

    Args:
        tape: :class:`warp.Tape` on which the force and update kernels are
            recorded.
        X: Position array, shape ``[N, 3]``; the differentiated input.
        R: Radius array, length ``N`` (constant; defines contact range).
        particle_count: Number of active particles.
        dt: Mechanics Euler step size (friction folded in).
        f_net: Scratch net-force buffer, zeroed and grad-enabled here.
        grid: Optional reusable :class:`warp.HashGrid`; one is allocated when
            ``None``.

    Returns:
        New position array ``X_out`` with shape ``[N, 3]`` and
        ``requires_grad=True``, ready for tape-based backpropagation.

    See Also:
        :func:`waxmorph.simulator.mech_step_sticky`: non-differentiable,
            growing-tissue twin that also updates polarity and cell-type-
            dependent adhesion.
        :func:`diffusion_step_differentiable`: the diffusion counterpart with
            the same frozen-topology, tape-recording structure.
    """
    f_net.zero_()
    f_net.requires_grad = True

    device = X.device
    r_max = float(R.numpy()[:particle_count].max())
    query_radius = 2.0 * r_max + EPS_DIST
    if grid is None:
        grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    grid.build(X[:particle_count], query_radius)

    # Step 1: Build neighbor pairs OUTSIDE the tape (frozen topology)
    edges_i, edges_j, num_edges = _build_neighbor_pairs_dynamic(
        X,
        R,
        particle_count,
        query_radius,
        grid,
        device,
    )

    # Step 2: Compute forces from pairs and update positions ON the tape
    X_out = wp.zeros_like(X, device=device, requires_grad=True)

    with tape:
        if num_edges > 0:
            wp.launch(
                _sticky_sphere_grads_from_pairs,
                dim=num_edges,
                inputs=[X, R, edges_i[:num_edges], edges_j[:num_edges]],
                outputs=[f_net],
                device=device,
            )
        wp.launch(
            _gd_update,
            dim=particle_count,
            inputs=[X, f_net, dt],
            outputs=[X_out],
            device=device,
        )

    return X_out


def diffusion_step_differentiable(
    tape: "wp.Tape",
    X: wp.array,
    R: wp.array,
    c: wp.array,
    particle_count: int,
    D_emu: float = 0.1,
    dt: float = 1e-2,
    grid: "wp.HashGrid | None" = None,
) -> wp.array:
    """Record one differentiable graph-diffusion step on a Warp tape.

    Diffuses each latent signaling molecule over the contact graph via the
    shared graph Laplacian and a forward-Euler update, exposing the step to
    autodiff. Topology (built from positions ``X``) is discovered
    *outside* ``tape`` and held frozen; only the Laplacian accumulation
    (:func:`_molecule_diffusion_laplacian_from_pairs`) and the Euler step
    (:func:`_molecule_diffusion_step_out`) are recorded, so ``tape.backward()``
    propagates ``dL/dc_out -> dL/dc_in`` through the full diffusion operator.

    Gradient does *not* flow through: the positions ``X`` (used only to build
    the frozen graph), the radii ``R``, neighbour selection, or the edge
    indices. Each molecule channel diffuses independently. The caller seeds
    ``c_out.grad`` and replays the tape (see
    :class:`waxmorph.torch.warp_autograd.WarpDiffusionStep`).

    Args:
        tape: :class:`warp.Tape` on which the Laplacian and update kernels are
            recorded.
        X: Position array, shape ``[N, 3]``; defines neighbour topology only
            (not differentiated).
        R: Radius array, length ``N`` (constant; defines contact range).
        c: Concentration array, shape ``[N, num_molecules]``; the
            differentiated input.
        particle_count: Number of active particles.
        D_emu: Diffusion coefficient :math:`D_{\\mathrm{emu}}`.
        dt: Diffusion Euler step size.
        grid: Optional reusable :class:`warp.HashGrid`; one is allocated when
            ``None``.

    Returns:
        New concentration array ``c_out`` with shape ``[N, num_molecules]`` and
        ``requires_grad=True``, ready for tape-based backpropagation.

    See Also:
        :func:`waxmorph.simulator.reaction_diffs`: non-differentiable
            graph-Laplacian (activator/inhibitor) diffusion twin in the forward
            simulator.
        :func:`mech_step_sticky_differentiable`: the mechanics counterpart with
            the same frozen-topology, tape-recording structure.
    """
    device = X.device
    lap_c = wp.zeros_like(c, device=device, requires_grad=True)

    r_max = float(R.numpy()[:particle_count].max())
    query_radius = 2.0 * r_max + EPS_DIST
    if grid is None:
        grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    grid.build(X[:particle_count], query_radius)

    # Step 1: Build neighbor pairs OUTSIDE the tape (frozen topology)
    edges_i, edges_j, num_edges = _build_neighbor_pairs_dynamic(
        X,
        R,
        particle_count,
        query_radius,
        grid,
        device,
    )

    # Step 2: Laplacian + Euler step ON the tape
    num_molecules = int(c.shape[1])
    c_out = wp.zeros_like(c, device=device, requires_grad=True)

    with tape:
        if num_edges > 0:
            wp.launch(
                _molecule_diffusion_laplacian_from_pairs,
                dim=num_edges,
                inputs=[X, R, c, edges_i[:num_edges], edges_j[:num_edges]],
                outputs=[lap_c],
                device=device,
            )
        wp.launch(
            _molecule_diffusion_step_out,
            dim=(int(c.shape[0]), num_molecules),
            inputs=[c, lap_c, float(D_emu), float(dt), particle_count],
            outputs=[c_out],
            device=device,
        )

    return c_out


# Clear Warp's compiled-kernel caches at import time so the kernels above are
# (re)generated against the current Warp version and constants. This guards
# against stale cached binaries silently shadowing edits to the kernel sources
# or to shared values in waxmorph.constants. (LTO cache first, then the kernel
# cache.) The simulator twin performs the same import-time clear.
wp.clear_lto_cache()
wp.clear_kernel_cache()
