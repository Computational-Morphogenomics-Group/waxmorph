import warp as wp

from .constants import EPS_DIST, EPS_NORM

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


@wp.kernel(enable_backward=False)
def _sticky_sphere_grads(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    gx: wp.array(dtype=wp.vec3f),
):
    """Accumulate net mechanics gradient per cell from all unordered pairs."""
    i, j = wp.tid()
    if j <= i:
        return

    x_i, x_j = X[i], X[j]
    r_i, r_j = R[i], R[j]
    force_i, force_j = _sticky_sphere_forces(x_i, x_j, r_i, r_j)
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


@wp.kernel
def _plan_divisions_from_hard_sample(
    hard_division: wp.array2d(dtype=wp.float32),
    div_count: wp.array(dtype=wp.int32),
    div_slots: wp.array(dtype=wp.int32),
    max_cells: wp.int32,
):
    """Reserve daughter slots for selected parents under capacity limits."""
    i = wp.tid()
    if hard_division[i, 0] <= wp.float32(0.5):
        div_slots[i] = wp.int32(-1)
        return

    child = wp.atomic_add(div_count, 0, 1)
    if child >= max_cells:
        div_slots[i] = wp.int32(-1)
        return

    div_slots[i] = child


@wp.kernel
def _apply_division_geometry(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    P: wp.array(dtype=wp.vec3f),
    div_slots: wp.array(dtype=wp.int32),
    division_direction: wp.array2d(dtype=wp.float32),
):
    """Place daughters using normalized direction and tangent separation."""
    i = wp.tid()
    child = div_slots[i]
    if child == wp.int32(-1):
        return

    v = wp.vec3f(
        division_direction[i, 0],
        division_direction[i, 1],
        division_direction[i, 2],
    )
    if wp.length(v) <= wp.float32(EPS_NORM):
        v = P[i]
    if wp.length(v) <= wp.float32(EPS_NORM):
        v = wp.vec3f(1.0, 0.0, 0.0)
    u = wp.normalize(v)

    r = R[i]
    x = X[i]
    # Keep parent fixed; place daughter tangent to parent (center offset = 2R).
    sep = wp.float32(2.0) * r
    X[child] = x + u * sep
    R[child] = r
    P[child] = P[i]


@wp.kernel
def _apply_division_gene_split(
    genes: wp.array2d(dtype=wp.float32),
    div_slots: wp.array(dtype=wp.int32),
):
    """Conserve gene mass by splitting parent feature values evenly."""
    i, g = wp.tid()
    child = div_slots[i]
    if child == wp.int32(-1):
        return

    v = genes[i, g]
    genes[i, g] = wp.float32(0.5) * v
    genes[child, g] = wp.float32(0.5) * v


@wp.kernel
def _apply_division_reward(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    P: wp.array(dtype=wp.vec3f),
    div_slots: wp.array(dtype=wp.int32),
    gx: wp.array(dtype=wp.vec3f),
    force_tol: float,
    mesh_id: wp.uint64,
    max_dist: float,
    rewards: wp.array(dtype=wp.float32),
):
    """Division rewards per parent, after taking actions."""
    i = wp.tid()
    child = div_slots[i]

    if child == -1:
        return

    new_pos = X[child]
    q = wp.mesh_query_point_sign_normal(mesh_id, new_pos, max_dist)
    is_inside = wp.bool(q.sign <= wp.float32(0.0))
    # force_equilibrium = wp.bool(wp.length(gx[i]) <= force_tol)

    if is_inside:
        rewards[i] = wp.float32(3.0)
    else:
        rewards[i] = wp.float32(-3.0)


@wp.kernel
def _apply_gene_deltas(
    genes: wp.array2d(dtype=wp.float32),
    delta_genes: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    particle_count: wp.int32,
):
    """Euler update for genes with nonnegative clamp."""
    i, g = wp.tid()
    if i >= particle_count:
        return

    v = genes[i, g] + dt * delta_genes[i, g]
    genes[i, g] = wp.clamp(v, wp.float32(0.0), wp.float32(1e4))


@wp.kernel
def _apply_polarity_normalized_deltas(
    polarities: wp.array(dtype=wp.vec3f),
    delta_polarities: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    particle_count: wp.int32,
):
    """Euler update for polarity vectors followed by unit normalization."""
    i = wp.tid()
    if i >= particle_count:
        return

    p = polarities[i]
    p_next = wp.vec3f(
        p[0] + dt * delta_polarities[i, 0],
        p[1] + dt * delta_polarities[i, 1],
        p[2] + dt * delta_polarities[i, 2],
    )
    n = wp.length(p_next) + wp.float32(EPS_NORM)
    polarities[i] = p_next / n


@wp.kernel
def _collect_f32(
    values: wp.array(dtype=wp.float32),
    total: wp.array(dtype=wp.float32),
):
    """Parallel sum reduction helper for stage total reward."""
    i = wp.tid()
    wp.atomic_add(total, 0, values[i])


def _ensure_capacity(capacity: int, minimum: int) -> int:
    """Clamp capacity to at least the requested minimum."""
    cap = int(capacity)
    min_cap = int(minimum)
    if cap < min_cap:
        return min_cap
    return cap


def apply_policy_deltas(
    G: wp.array,
    P: wp.array,
    delta_genes: wp.array,
    delta_polarities: wp.array,
    particle_count: int,
    dt: float,
) -> None:
    """Apply policy-driven gene and polarity updates in-place."""
    count = int(particle_count)
    if count <= 0:
        return

    wp.launch(
        _apply_gene_deltas,
        dim=(int(G.shape[0]), int(G.shape[1])),
        inputs=[G, delta_genes, float(dt), count],
        device=G.device,
    )

    wp.launch(
        _apply_polarity_normalized_deltas,
        dim=int(P.shape[0]),
        inputs=[P, delta_polarities, float(dt), count],
        device=P.device,
    )


############################################################
############################################################
############################################################

# DIFFUSION

############################################################
############################################################
############################################################


@wp.func
def _adj_weight(dist: wp.float32, ri: wp.float32, rj: wp.float32) -> wp.bool:
    """Neighborhood predicate matching the simulator adjacency."""
    return wp.bool((dist - (ri + rj)) <= EPS_DIST)


@wp.kernel
def _gene_diffusion_laplacian(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    G: wp.array2d(dtype=wp.float32),
    lap_G: wp.array2d(dtype=wp.float32),
):
    """Accumulate graph-Laplacian diffusion for all gene channels.

    For each adjacent pair (i, j) and each gene g, computes the flux
    ``G[j, g] - G[i, g]`` and accumulates symmetrically into ``lap_G``.
    """
    i, j = wp.tid()

    if wp.int32(j) <= wp.int32(i):
        return

    dist = wp.length(X[i] - X[j]) + EPS_NORM
    if not _adj_weight(dist, R[i], R[j]):
        return

    num_genes = G.shape[1]
    for g in range(num_genes):
        flux = G[j, g] - G[i, g]
        wp.atomic_add(lap_G, i, g, flux)
        wp.atomic_add(lap_G, j, g, -flux)


@wp.kernel
def _gene_diffusion_step(
    G: wp.array2d(dtype=wp.float32),
    lap_G: wp.array2d(dtype=wp.float32),
    alpha: wp.float32,
    dt: wp.float32,
    particle_count: wp.int32,
):
    """Euler step: ``G[i,g] += dt * alpha * lap_G[i,g]``, clamped >= 0."""
    i, g = wp.tid()
    if i >= particle_count:
        return

    v = G[i, g] + dt * alpha * lap_G[i, g]
    G[i, g] = wp.max(v, wp.float32(0.0))


def diffusion_step(
    X: wp.array,
    R: wp.array,
    G: wp.array,
    lap_G: wp.array,
    particle_count: int,
    alpha: float = 0.1,
    dt: float = 1e-2,
) -> None:
    """Run one graph-Laplacian diffusion step for all gene channels.

    Parameters
    ----------
    X : wp.array(dtype=wp.vec3f)
        Positions.
    R : wp.array(dtype=wp.float32)
        Radii.
    G : wp.array2d(dtype=wp.float32), shape ``[max_particles, num_genes]``
        Gene concentrations (updated in-place).
    lap_G : wp.array2d(dtype=wp.float32), same shape as G
        Scratch buffer for the Laplacian (zeroed internally).
    particle_count : int
        Number of active particles.
    alpha : float
        Diffusion coefficient.
    dt : float
        Time step.
    """
    lap_G.zero_()

    wp.launch(
        _gene_diffusion_laplacian,
        dim=(particle_count, particle_count),
        inputs=[X, R, G],
        outputs=[lap_G],
    )

    num_genes = int(G.shape[1])
    wp.launch(
        _gene_diffusion_step,
        dim=(int(G.shape[0]), num_genes),
        inputs=[G, lap_G, float(alpha), float(dt), particle_count],
    )


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
):
    """Run one sticky-sphere mechanics step and return position gradients."""

    gx.zero_()

    wp.launch(
        _sticky_sphere_grads,
        dim=(particle_count, particle_count),
        inputs=[X, R],
        outputs=[gx],
    )

    wp.launch(_gd_update, dim=particle_count, inputs=[X, gx, dt], outputs=[X])


def divide_cells(
    X: wp.array,
    R: wp.array,
    P: wp.array,
    G: wp.array,
    hard_division: wp.array,
    division_direction: wp.array,
    max_particles: int,
    particle_count: int,
    num_genes: int,
    force_tol: float,
    mesh_id: wp.uint64,
    max_dist: float,
    gx: wp.array,
    rewards: wp.array,
):

    cap = _ensure_capacity(max_particles, particle_count)

    div_count = wp.full(1, value=particle_count, dtype=wp.int32)
    div_slots = wp.full(particle_count, value=-1, dtype=wp.int32)

    wp.launch(
        _plan_divisions_from_hard_sample,
        dim=particle_count,
        inputs=[hard_division, div_count, div_slots, cap],
    )

    wp.launch(
        _apply_division_geometry,
        dim=particle_count,
        inputs=[X, R, P, div_slots, division_direction],
    )

    wp.launch(
        _apply_division_gene_split,
        dim=(particle_count, num_genes),
        inputs=[G, div_slots],
    )

    # gx.zero_()

    # wp.launch(
    #     _sticky_sphere_grads,
    #     dim=(particle_count, particle_count),
    #     inputs=[X, R],
    #     outputs=[gx],
    # )

    rewards.zero_()

    wp.launch(
        _apply_division_reward,
        dim=particle_count,
        inputs=[X, R, P, div_slots, gx, force_tol, mesh_id, max_dist],
        outputs=[rewards],
    )

    return min([max_particles, div_count.numpy().item()])


wp.clear_lto_cache()
wp.clear_kernel_cache()
