"""Differentiable Warp mechanics and graph diffusion for fixed-size tissue.

Each step discovers neighbors before :class:`warp.Tape` recording and replays a
frozen pair list. The gradient contract covers recorded state arithmetic.
Mechanics records radii in its pair arithmetic when they require gradients;
package bridges expose gradients for their documented inputs. The forward simulator
provides type-dependent forces, chemistry, growth, and division.
"""

import warp as wp

from .constants import EPS_DIST, EPS_NORM, HASH_GRID_DIM

# The emulator uses one attraction coefficient per pair; the simulator selects by cell type.
K_REP = 2.0
K_ATT = 0.5


@wp.func
def _sticky_sphere_forces(
    x_i: wp.vec3f,
    x_j: wp.vec3f,
    r_i: wp.float32,
    r_j: wp.float32,
):
    r"""Uniform soft-sphere force for one pair.

    For :math:`q_{ij}=x_i-x_j`,
    :math:`d_{ij}=\lVert q_{ij}\rVert_2+EPS_NORM`, and
    :math:`u_{ij}=q_{ij}/d_{ij}`,

    .. math::

        F_{ij} = \Bigl[\, k_{\mathrm{rep}}\,\max(r_i+r_j-\varepsilon-d_{ij},\,0)
                 - k_{\mathrm{att}}\,\max(r_i+r_j+\varepsilon-d_{ij},\,0)\,
                   \mathbb{1}[d_{ij} > r_i+r_j-\varepsilon] \,\Bigr]\,u_{ij},

    Here :math:`k_{\mathrm{rep}}=K_REP`, :math:`k_{\mathrm{att}}=K_ATT`, and
    :math:`\varepsilon=EPS_DIST`. The result is treated as a force and added by
    :func:`_gd_update`; the simulator's descent buffer is subtracted. Max clamps
    use Warp's pathwise derivative; backward treats the strict attraction gate
    as constant.
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
    r"""Emit each contact :math:`i<j` once outside the tape.

    Contacts satisfy
    :math:`\lVert x_i-x_j\rVert_2+EPS_NORM\le r_i+r_j+EPS_DIST`. ``edge_count``
    may exceed write capacity; the host retries at realized capacity and preserves all pairs.
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
    """Differentiably scatter equal-and-opposite forces over frozen pairs."""
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
    r"""Overdamped Euler update with friction absorbed into ``lr``.

    .. math::

        x_i \leftarrow x_i + \Delta t\, f^{\mathrm{net}}_i .

    ``f_net`` uses the force convention, hence the positive sign. The simulator
    subtracts its descent buffer.
    """
    i = wp.tid()
    X_next[i] = X[i] + lr * f_net[i]


def _build_neighbor_pairs_dynamic(
    X: wp.array,
    R: wp.array,
    particle_count: int,
    query_radius: float,
    grid: "wp.HashGrid",
    device,
) -> tuple[wp.array, wp.array, int]:
    """Build frozen pairs outside the tape, retrying at realized capacity."""
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


@wp.kernel
def _molecule_diffusion_laplacian_from_pairs(
    c: wp.array2d(dtype=wp.float32),
    edges_i: wp.array(dtype=wp.int32),
    edges_j: wp.array(dtype=wp.int32),
    lap_c: wp.array2d(dtype=wp.float32),
):
    r"""Accumulate the minus graph Laplacian from frozen pairs.

    .. math::

        (L_G c)_i = \sum_{j:(i,j)\in E} (c_i - c_j),

    The kernel scatters :math:`c_j-c_i` to ``i`` and its negation to ``j`` for
    each molecule. Frozen pair indices hold topology constant during differentiation.
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
    r"""Out-of-place Euler diffusion with a nonnegative clamp.

    .. math::

        c_{i,M} \leftarrow c_{i,M} - D_{\mathrm{emu}}\,(L_G c)_{i,M}\,\Delta t,

    ``lap_c`` stores :math:`-L_Gc`, so the implementation adds its contribution.
    Inactive rows copy through. ``c_out`` requires storage distinct from ``c``
    so tape replay preserves adjoints; the clamp has zero gradient where it
    clips to zero.
    """
    i, g = wp.tid()
    if i >= particle_count:
        c_out[i, g] = c[i, g]
        return

    v = c[i, g] + dt * D_emu * lap_c[i, g]
    c_out[i, g] = wp.max(v, wp.float32(0.0))


def mech_step_sticky_differentiable(
    tape: "wp.Tape",
    X: wp.array,
    R: wp.array,
    particle_count: int,
    dt: float,
    f_net: wp.array,
    grid: "wp.HashGrid | None" = None,
) -> wp.array:
    """Record mechanics with frozen topology.

    Pair discovery remains outside the tape; radii participate in recorded force
    arithmetic. Inactive position rows retain identity gradients. The caller
    seeds ``X_out.grad`` and replays ``tape``.
    """
    f_net.zero_()
    f_net.requires_grad = True

    device = X.device
    r_max = float(R.numpy()[:particle_count].max())
    query_radius = 2.0 * r_max + EPS_DIST
    if grid is None:
        grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    grid.build(X[:particle_count], query_radius)

    edges_i, edges_j, num_edges = _build_neighbor_pairs_dynamic(
        X,
        R,
        particle_count,
        query_radius,
        grid,
        device,
    )

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
            dim=X.shape[0],
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
    """Record graph diffusion with concentration gradients.

    Positions, radii, and pair discovery remain outside the tape. Each molecule
    channel diffuses independently over the frozen topology. The caller seeds
    ``c_out.grad`` and replays ``tape``.
    """
    device = X.device
    lap_c = wp.zeros_like(c, device=device, requires_grad=True)

    r_max = float(R.numpy()[:particle_count].max())
    query_radius = 2.0 * r_max + EPS_DIST
    if grid is None:
        grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    grid.build(X[:particle_count], query_radius)

    edges_i, edges_j, num_edges = _build_neighbor_pairs_dynamic(
        X,
        R,
        particle_count,
        query_radius,
        grid,
        device,
    )

    num_molecules = int(c.shape[1])
    c_out = wp.zeros_like(c, device=device, requires_grad=True)

    with tape:
        if num_edges > 0:
            wp.launch(
                _molecule_diffusion_laplacian_from_pairs,
                dim=num_edges,
                inputs=[c, edges_i[:num_edges], edges_j[:num_edges]],
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
