import warp as wp

from .constants import EPS_DIST, EPS_NORM, HASH_GRID_DIM

############################################################
############################################################
############################################################

# HELPERS

############################################################
############################################################
############################################################

# Emulator-specific force constants (intentionally different from simulator).
K_REP = 2.0
K_ATT = 0.5


@wp.func
def _sticky_sphere_forces(
    x_i: wp.vec3f,
    x_j: wp.vec3f,
    r_i: wp.float32,
    r_j: wp.float32,
):
    """Return pairwise mechanics gradients for repulsion+adhesion potential.

    This is the local pair contribution to ``-grad U_t`` in the writeup
    mechanics equation.
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
def _sticky_sphere_grads(
    grid: wp.uint64,
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    query_radius: wp.float32,
    gx: wp.array(dtype=wp.vec3f),
):
    """Accumulate net mechanics gradient per cell from all unordered pairs."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x_i = X[i]
    r_i = R[i]

    for j in wp.hash_grid_query(grid, x_i, query_radius):
        if j <= i:
            continue
        force_i, force_j = _sticky_sphere_forces(x_i, X[j], r_i, R[j])
        wp.atomic_add(gx, i, force_i)
        wp.atomic_add(gx, j, force_j)


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
    """Build unordered pair list (i < j) from hash-grid neighbors."""
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
    gx: wp.array(dtype=wp.vec3f),
):
    """Compute forces from precomputed neighbor pairs (tape-compatible).

    Launched with ``dim=num_edges``. No hash grid queries inside — all
    operations (reads, arithmetic, atomic_add) are differentiable in Warp.
    """
    e = wp.tid()
    i = edges_i[e]
    j = edges_j[e]
    force_i, force_j = _sticky_sphere_forces(X[i], X[j], R[i], R[j])
    wp.atomic_add(gx, i, force_i)
    wp.atomic_add(gx, j, force_j)


@wp.kernel
def _gd_update(
    X: wp.array(dtype=wp.vec3f),
    gx: wp.array(dtype=wp.vec3f),
    lr: wp.float32,
    X_next: wp.array(dtype=wp.vec3f),
):
    """Euler step for positions: ``X_next = X + dt * force``."""
    i = wp.tid()
    X_next[i] = X[i] + lr * gx[i]


def _build_neighbor_pairs_dynamic(
    X: wp.array,
    R: wp.array,
    particle_count: int,
    query_radius: float,
    grid: "wp.HashGrid",
    device,
) -> tuple[wp.array, wp.array, int]:
    """Build neighbor pairs with a retry if the initial buffer overflows.

    The differentiable mechanics and diffusion paths launch pair-based kernels
    with ``dim=num_edges``. If the temporary pair buffers are undersized, using
    the raw edge count as the launch dimension would index past those buffers.
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
def _gene_diffusion_laplacian_from_pairs(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    G: wp.array2d(dtype=wp.float32),
    edges_i: wp.array(dtype=wp.int32),
    edges_j: wp.array(dtype=wp.int32),
    lap_G: wp.array2d(dtype=wp.float32),
):
    """Accumulate graph-Laplacian from precomputed neighbor pairs (tape-compatible).

    Launched with ``dim=num_edges``. No hash grid queries — all operations
    (reads, arithmetic, atomic_add) are differentiable in Warp.
    """
    e = wp.tid()
    i = edges_i[e]
    j = edges_j[e]

    num_genes = G.shape[1]
    for g in range(num_genes):
        flux = G[j, g] - G[i, g]
        wp.atomic_add(lap_G, i, g, flux)
        wp.atomic_add(lap_G, j, g, -flux)


@wp.kernel
def _gene_diffusion_step_out(
    G: wp.array2d(dtype=wp.float32),
    lap_G: wp.array2d(dtype=wp.float32),
    alpha: wp.float32,
    dt: wp.float32,
    particle_count: wp.int32,
    G_out: wp.array2d(dtype=wp.float32),
):
    """Out-of-place Euler step (tape-safe: G and G_out must be distinct arrays)."""
    i, g = wp.tid()
    if i >= particle_count:
        G_out[i, g] = G[i, g]
        return

    v = G[i, g] + dt * alpha * lap_G[i, g]
    G_out[i, g] = wp.max(v, wp.float32(0.0))


############################################################
############################################################
############################################################

# RUNNERS

############################################################
############################################################
############################################################


def mech_step_sticky(
    X: wp.array,
    R: wp.array,
    particle_count: int,
    dt: float,
    gx: wp.array,
    grid: "wp.HashGrid | None" = None,
):
    """Run one sticky-sphere mechanics step and return position gradients."""

    gx.zero_()

    device = X.device
    r_max = float(R.numpy()[:particle_count].max())
    query_radius = 2.0 * r_max + EPS_DIST
    if grid is None:
        grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    grid.build(X[:particle_count], query_radius)

    wp.launch(
        _sticky_sphere_grads,
        dim=particle_count,
        inputs=[wp.uint64(grid.id), X, R, query_radius],
        outputs=[gx],
        device=device,
    )

    wp.launch(_gd_update, dim=particle_count, inputs=[X, gx, dt], outputs=[X], device=device)


def mech_step_sticky_differentiable(
    tape: "wp.Tape",
    X: wp.array,
    R: wp.array,
    particle_count: int,
    dt: float,
    gx: wp.array,
    grid: "wp.HashGrid | None" = None,
) -> wp.array:
    """Tape-recorded mechanics step. Returns a new position array.

    Neighbor discovery happens outside the tape (frozen topology).
    Force computation from precomputed pairs and position update are
    recorded on the tape so that ``tape.backward()`` propagates
    ``dL/dX_out → dL/dX_in``.
    """
    gx.zero_()
    gx.requires_grad = True

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
                outputs=[gx],
                device=device,
            )
        wp.launch(
            _gd_update,
            dim=particle_count,
            inputs=[X, gx, dt],
            outputs=[X_out],
            device=device,
        )

    return X_out


def diffusion_step_differentiable(
    tape: "wp.Tape",
    X: wp.array,
    R: wp.array,
    G: wp.array,
    lap_G: wp.array,
    particle_count: int,
    alpha: float = 0.1,
    dt: float = 1e-2,
    grid: "wp.HashGrid | None" = None,
) -> wp.array:
    """Tape-recorded diffusion step. Returns a new gene array.

    Neighbor discovery happens outside the tape (frozen topology).
    Laplacian computation from precomputed pairs and the Euler update
    are recorded on the tape so that ``tape.backward()`` propagates
    ``dL/dG_out → dL/dG_in`` through the full diffusion operator.
    """
    lap_G.zero_()
    lap_G.requires_grad = True

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

    # Step 2: Laplacian + Euler step ON the tape
    num_genes = int(G.shape[1])
    G_out = wp.zeros_like(G, device=device, requires_grad=True)

    with tape:
        if num_edges > 0:
            wp.launch(
                _gene_diffusion_laplacian_from_pairs,
                dim=num_edges,
                inputs=[X, R, G, edges_i[:num_edges], edges_j[:num_edges]],
                outputs=[lap_G],
                device=device,
            )
        wp.launch(
            _gene_diffusion_step_out,
            dim=(int(G.shape[0]), num_genes),
            inputs=[G, lap_G, float(alpha), float(dt), particle_count],
            outputs=[G_out],
            device=device,
        )

    return G_out


wp.clear_lto_cache()
wp.clear_kernel_cache()
