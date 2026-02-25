import warp as wp

############################################################
############################################################
############################################################

            # CONSTANTS / HELPERS

############################################################
############################################################
############################################################
FOUR_THIRDS_PI = 4.1887902047863905
EPS_DIST = 2e-1
EPS_DEN = 1e-9
EPS_NORM = 1e-9
RAND_EPS = 1e-7

K_REP = 3.0

K_ATT_EE = 1.5 # epi-epi strong cohesion
K_ATT_MM = 0.15  # mes-mes medium
K_ATT_EM = 0.15  # epi-mes weak (interface tension)

ATR_EE = 0.14
ATR_MM = 0.1
ATR_EM = 0.06

D_SHIFT_EM = 0.08

@wp.func
def safe_div(num: wp.float32, den: wp.float32) -> wp.float32:
    return num / (den + EPS_DEN)

@wp.func
def volume_from_radius(r: wp.float32) -> wp.float32:
    return FOUR_THIRDS_PI * r * r * r

@wp.func
def adj_weight(dist: wp.float32, ri: wp.float32, rj: wp.float32) -> wp.bool:
    return wp.bool((dist - (ri + rj)) <= EPS_DIST)


@wp.func
def probs(p: wp.float32, ref: wp.float32):
    return (p ** 20.) / ((p ** 20.) + (ref ** 20.))

@wp.func
def softmax2d(p: wp.vec2f):
    ex = wp.exp(p.x)
    ey = wp.exp(p.y)
    inv_sum = 1.0 / (ex + ey)
    return wp.vec2f(ex * inv_sum, ey * inv_sum)

@wp.func
def gumbel(
    key: wp.uint32
):
    u = wp.randf(key, 0., 1.)
    u = wp.clamp(u, RAND_EPS, 1.0 - RAND_EPS)
    return wp.randu(key), -wp.log(-wp.log(u))


@wp.kernel
def fill_key_array(
    key_array: wp.array(dtype=wp.uint32),
) -> None:

    i = wp.tid()
    key_array[i] = wp.rand_init(i)


def gen_key_array(size: int, device: str = "cuda") -> wp.array:

    key_array = wp.zeros(shape=size, dtype=wp.uint32, device=device)
    wp.launch(fill_key_array, dim=key_array.size, inputs=[key_array], device=device)
    return key_array


############################################################
############################################################
############################################################

            # MECHANICS

############################################################
############################################################
############################################################
    

@wp.func 
def sticky_sphere_forces(x_i: wp.vec3f, x_j: wp.vec3f, r_i: wp.float32, r_j: wp.float32, ct_i: wp.uint32, ct_j: wp.uint32): 
    d = x_i - x_j
    dist = wp.length(d) + EPS_NORM
    u = d / dist

    rs = r_i + r_j

    # defaults: mes-mes
    k_att = K_ATT_MM
    atr = ATR_MM
    d_shift = wp.float32(0.0)

    if (ct_i == wp.uint32(1)) and (ct_j == wp.uint32(1)):
        k_att = K_ATT_EE
        atr = ATR_EE
    elif ct_i != ct_j:
        k_att = K_ATT_EM
        atr = ATR_EM
        d_shift = D_SHIFT_EM  # creates interfacial tension

    # preferred contact distance
    d0 = rs + d_shift

    # outside interaction range -> no force
    if dist > d0 + atr:
        return wp.vec3f(0.0), wp.vec3f(0.0)

    # repulsion when too close (dist < d0)
    delta = wp.max(d0 - dist, wp.float32(0.0))

    # attraction only in adhesive band (d0 < dist < d0+atr)
    gamma = wp.max((d0 + atr) - dist, wp.float32(0.0)) * wp.float32(dist > d0)

    fmag = K_REP * delta - k_att * gamma

    F_ij = fmag * -u
    return F_ij, -F_ij


@wp.func
def epi_polarity_grads(
    x_i: wp.vec3f,
    x_j: wp.vec3f,
    p_i: wp.vec3f,
    p_j: wp.vec3f,
):
    # Aligning polarity to be perpendicular to connections 
    
    # d, ||d||, u = d/||d||
    d = x_i - x_j
    dist = wp.length(d) + EPS_NORM
    u = d / dist

    # a = p_i·u, b = p_j·u
    a = wp.dot(p_i, u)
    b = wp.dot(p_j, u)

    # grads wrt polarities
    grad_p_i = a * u
    grad_p_j = b * u

    # dU/du = a p_i + b p_j
    g_u = a * p_i + b * p_j

    # (I - uu^T) g_u = g_u - u (u·g_u)
    proj = g_u - u * wp.dot(u, g_u)

    # d u / d d = (1/||d||) (I - uu^T)
    grad_d = proj / dist

    # d = x_i - x_j
    grad_x_i = grad_d
    grad_x_j = -grad_d

    return grad_x_i, grad_x_j, grad_p_i, grad_p_j


@wp.func
def mes_polarity_grads(p_i: wp.vec3f, p_j: wp.vec3f):
    # U = -0.5 (p_i·p_j)^2
    c = wp.dot(p_i, p_j)
    grad_p_i = -c * p_j
    grad_p_j = -c * p_i
    return grad_p_i, grad_p_j


@wp.kernel(enable_backward=False)
def sticky_sphere_grads(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    P: wp.array(dtype=wp.vec3f),
    CT: wp.array(dtype=wp.uint32),
    gx: wp.array(dtype=wp.vec3f),
    gp: wp.array(dtype=wp.vec3f),
):
    i, j = wp.tid()

    # Skip self + ensure each unordered pair is counted once
    if wp.int32(j) <= wp.int32(i):
        return

    x_i, x_j = X[i], X[j]
    r_i, r_j = R[i], R[j]
    c_i, c_j = CT[i], CT[j]
    p_i, p_j = P[i], P[j]

    # Forces
    grad_x_i_f, grad_x_j_f = sticky_sphere_forces(x_i, x_j, r_i, r_j, c_i, c_j)
    wp.atomic_add(gx, i, grad_x_i_f)
    wp.atomic_add(gx, j, grad_x_j_f)

    # Polarity Neighbors
    dist = wp.norm_l2(x_i - x_j)
    w = adj_weight(dist, r_i, r_j)

    if not wp.bool(w):
        return

    # Polarities - epithelium
    if (c_i == wp.uint32(1)) and (c_j == wp.uint32(1)):
        grad_x_i, grad_x_j, grad_p_i, grad_p_j = epi_polarity_grads(x_i, x_j, p_i, p_j)

        # Match magnitudes so movement doesn't blink
        for k in range(3):
            v_i = wp.abs(grad_x_i_f[k])
            v_j = wp.abs(grad_x_j_f[k])
            grad_x_i[k] = wp.clamp(grad_x_i[k], -1.0 * v_i, 1.0 * v_i)
            grad_x_j[k] = wp.clamp(grad_x_j[k], -1.0 * v_j, 1.0 * v_j)

        wp.atomic_add(gx, i, grad_x_i)
        wp.atomic_add(gx, j, grad_x_j)
        wp.atomic_add(gp, i, grad_p_i)
        wp.atomic_add(gp, j, grad_p_j)

    # Polarities - mesenchyme
    if (c_i == wp.uint32(0)) and (c_j == wp.uint32(0)):
        grad_p_i, grad_p_j = mes_polarity_grads(p_i, p_j)
        wp.atomic_add(gp, i, grad_p_i)
        wp.atomic_add(gp, j, grad_p_j)


@wp.kernel
def gd_update(
    X: wp.array(dtype=wp.vec3f),   # (N, 3)
    gx: wp.array(dtype=wp.vec3f),  # (N, 3) gradient of loss w.r.t. x
    lr: wp.float32,
    X_next: wp.array(dtype=wp.vec3f),   # (N, 3)
):

    i = wp.tid()
    X_next[i] = X[i] - lr * gx[i]


@wp.kernel
def gd_update_normalized(
    P: wp.array(dtype=wp.vec3f),   # (N, 3)
    gp: wp.array(dtype=wp.vec3f),  # (N, 3) gradient of loss w.r.t. x
    lr: wp.float32,
    P_next: wp.array(dtype=wp.vec3f),   # (N, 3)
):

    i = wp.tid()
    P_next[i] = wp.normalize(P[i] - lr * gp[i])  # Normals pointing both inward / outward, but direction is used only

    

def mech_step_sticky(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    P: wp.array(dtype=wp.vec3f),
    CT: wp.array(dtype=wp.uint32),
    particle_count: wp.int32,
    dt: float,
    X_next: wp.array(dtype=wp.vec3f),
    P_next: wp.array(dtype=wp.vec3f),
    device: str = "cuda",
    grad_consist: bool = False,
):
    
    # Set up gradients
    gx = wp.zeros_like(X, device=device)
    gp = wp.zeros_like(P, device=device)

    # 2D launch required for i,j indexing. 
    wp.launch(
        sticky_sphere_grads,
        dim=(particle_count, particle_count),
        inputs=[X, R, P, CT],
        outputs=[gx, gp],
        device=device,
    )

    
    if grad_consist:
        gx.mark_write()
    

    wp.launch(
        gd_update,
        dim=particle_count,
        inputs=[X, gx, dt, X_next],
        device=device,
    )

    wp.launch(
        gd_update_normalized,
        dim=particle_count,
        inputs=[P, gp, dt, P_next],
        device=device,
    )

    
    if grad_consist:
        X.mark_read()
        P.mark_read()
        gx.mark_read()
        gp.mark_read()
        
        X_next.mark_write()
        P_next.mark_write()


    return gx
    



############################################################
############################################################
############################################################

            # DIFFUSION

############################################################
############################################################
############################################################

@wp.kernel
def reaction_diffs(
    X: wp.array(dtype=wp.vec3f),  # (N, 3)
    R: wp.array(dtype=wp.float32),      # (N,)
    A: wp.array(dtype=wp.float32),         # (N,)
    I: wp.array(dtype=wp.float32),         # (N,)
    CT: wp.array(dtype=wp.uint32),
    lapA: wp.array(dtype=wp.float32),       # (N,) out (accum)
    lapI: wp.array(dtype=wp.float32),       # (N,) out (accum)
):

    i,j = wp.tid()

    if wp.int32(j) <= wp.int32(i):
        return

    if (CT[i] == wp.uint32(1)) or (CT[j] == wp.uint32(1)):
        return

    Ri, Rj = R[i], R[j]
    Vi, Vj = volume_from_radius(Ri), volume_from_radius(Rj)
    cAi, cAj = safe_div(A[i], Vi), safe_div(A[j], Vj)
    cIi, cIj = safe_div(I[i], Vi), safe_div(I[j], Vj)

    dist = wp.norm_l2(X[i] - X[j])

    w = adj_weight(dist, Ri, Rj)

    # Prune non neighbors
    if not wp.bool(w):
        return

    # Pairwise flux contribution: w * (c_j - c_i)
    dA = wp.float32(w) * (cAj - cAi)
    dI = wp.float32(w) * (cIj - cIi)

    # Symmetric accumulation: +d to i, -d to j

    lapA[i] += dA
    lapA[j] += -dA

    lapI[i] += dI
    lapI[j] += -dI


@wp.kernel
def reaction_step(
    A: wp.array(dtype=wp.float32),      # (N,)
    I: wp.array(dtype=wp.float32),      # (N,)
    R: wp.array(dtype=wp.float32),
    lapA: wp.array(dtype=wp.float32),   # (N,)
    lapI: wp.array(dtype=wp.float32),   # (N,)
    CT: wp.array(dtype=wp.uint32),
    S: wp.array(dtype=wp.float32),
    T: wp.array(dtype=wp.float32),
    phi: wp.float32,
    dt: wp.float32,
    A_next: wp.array(dtype=wp.float32),      # (N,)
    I_next: wp.array(dtype=wp.float32),      # (N,)
):

    i = wp.tid()
    
    if (CT[i] == wp.uint32(1)):
        return
    
    V = volume_from_radius(R[i])
    
    # Reaction terms (on concentrations)
    cA = safe_div(A[i], V)
    cI = safe_div(I[i], V)
    
    cA2 = cA * cA
    ciI = cI
   
    prodA_lin  = safe_div(cA2, ciI)
    prodA_quad = safe_div(cA2, ciI * ciI)
    
    prodA = prodA_lin
    if wp.float32(prodA_quad) < wp.float32(prodA_lin):
        prodA = prodA_quad
    
    prodI = cA2

    # Diffusion with graph laplacian
    diffA = (S[0] * phi) * lapA[i]
    diffI = phi * lapI[i]

    # Explicit Euler step with temporal scaling
    zAi = A[i] + dt * T[0] * (diffA + prodA)
    zIi = I[i] + dt * T[0] * (diffI + prodI)

    # Linear damping 
    inv = 1.0 / (1.0 + dt * T[0])
    Ai_next = zAi * inv
    Ii_next = zIi * inv

    A_next[i] = wp.clamp(0., Ai_next, 1e4)
    I_next[i] = wp.clamp(0., Ii_next, 1e4)


def chem_step(
    A: wp.array,
    I: wp.array,
    X: wp.array,
    R: wp.array,
    lapA: wp.array,
    lapI: wp.array,
    CT: wp.array,
    S: wp.array,
    T: wp.array,
    phi: float,
    dt: float,
    particle_count: int,
    A_next: wp.array,
    I_next: wp.array,
    device: str = "cuda",
    grad_consist: bool = True,
):

    # Cache laplacian
    wp.launch(
        reaction_diffs,
        dim=(particle_count, particle_count),
        inputs=[X, R, A, I, CT],
        outputs=[lapA, lapI],
        device=device,
    )

    if grad_consist:
        lapA.mark_write()
        lapI.mark_write()

    # Update molecules
    wp.launch(
        reaction_step,
        dim=particle_count,
        inputs=[
            A, I, R, lapA, lapI, CT, S, T,
            phi, dt
        ],
        outputs=[
            A_next, I_next
        ],
        device=device,
    )

    # Gradient consistency checks
    if grad_consist:
        lapA.mark_read()
        lapI.mark_read()
        A.mark_read()
        I.mark_read()
        
        A_next.mark_write()
        I_next.mark_write()


# ############################################################
# ############################################################
# ############################################################

#             # GROWTH

# ############################################################
# ############################################################
# ############################################################

@wp.kernel
def count_neighbors(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    CT: wp.array(dtype=wp.uint32),
    n_tot: wp.array(dtype=wp.int32),
    n_epi: wp.array(dtype=wp.int32),
    n_mes: wp.array(dtype=wp.int32),
):
    i, j = wp.tid()
    
    if wp.int32(j) <= wp.int32(i):
        return

    xi, xj = X[i], X[j]
    ri, rj = R[i], R[j]

    dist = wp.norm_l2(xi - xj)
    w = adj_weight(dist, ri, rj)
    
    if not wp.bool(w):
        return

    # total counts
    wp.atomic_add(n_tot, i, 1)
    wp.atomic_add(n_tot, j, 1)

    # type-specific (neighbor type)
    if CT[j] == wp.uint32(1):
        wp.atomic_add(n_epi, i, 1)
    else:
        wp.atomic_add(n_mes, i, 1)

    if CT[i] == wp.uint32(1):
        wp.atomic_add(n_epi, j, 1)
    else:
        wp.atomic_add(n_mes, j, 1)


def growth_step(
    R: wp.array(dtype=wp.float32),
    R_eq: wp.array(dtype=wp.float32),
    A: wp.array(dtype=wp.float32),
    CT: wp.array(dtype=wp.uint32),
    keys: wp.array(dtype=wp.uint32),
    AP: wp.array(dtype=wp.float32),
    SC: wp.array(dtype=wp.float32),
    dt: float,
    R_ref: wp.float32,
    R_max: wp.float32,
    particle_count: int,
    R_next: wp.array(dtype=wp.float32),
    R_eq_next: wp.array(dtype=wp.float32), 
    device: str = "cuda",
    grad_consist: bool = True,
    
) -> None:
    wp.launch(
        growth_step_inner,
        dim=particle_count,
        inputs=[R, R_eq, A, CT, keys, AP, SC, dt, R_ref, R_max, R_next, R_eq_next],
        device=device,
    )

    # Gradient consistency checks
    if grad_consist:
        R.mark_read()
        R_eq.mark_read()
        
        R_next.mark_write()
        R_eq_next.mark_write()


@wp.kernel
def growth_step_inner(
    R: wp.array(dtype=wp.float32),
    R_eq: wp.array(dtype=wp.float32),
    A: wp.array(dtype=wp.float32),
    CT: wp.array(dtype=wp.uint32),
    keys: wp.array(dtype=wp.uint32),
    AP: wp.array(dtype=wp.float32),
    SC: wp.array(dtype=wp.float32),
    dt: wp.float32,
    R_ref: wp.float32,
    R_max: wp.float32,
    R_next: wp.array(dtype=wp.float32),
    R_eq_next: wp.array(dtype=wp.float32),
) -> None:
    i = wp.tid()

    if CT[i] == wp.uint32(0):
        r_i_0, r_eq_i = R[i], wp.min(R_eq[i], R_max)
        
        V = volume_from_radius(R[i])
        num = safe_div(A[i], V) ** AP[0]
        
        key = keys[i]
        lam = wp.randf(key, 0.8, 1.0)
        key = wp.randu(key)
        frac = safe_div(num, (SC[0]**AP[0]) + num)
    
        R_eq_next[i] = r_eq_i + frac * lam * dt
        R_next[i] = r_i_0 + ((1.0 - safe_div(r_i_0, r_eq_i)) ** 2.0) * dt
        keys[i] = key

    if CT[i] == wp.uint32(1):
        r_i_1 = R[i]
        R_next[i] = r_i_1 + ((1.0 - safe_div(r_i_1, R_ref)) ** 2.0) * dt


@wp.func
def st_gumbel_softmax_bernoulli(
    p: wp.float32,
    key: wp.uint32,
    t: wp.float32,
    tmax: wp.float32,
    tau: wp.float32,
):

    p = wp.clamp(p, RAND_EPS, 1.0 - RAND_EPS)

    # # Gumbel softmax
    key, g0 = gumbel(key)
    key, g1 = gumbel(key)

    # # Anneal temp
    k = wp.log(tau / 0.1) / tmax
    tau = wp.max(0.1, tau * wp.exp(-k * t))

    p0 = (g1 + wp.log(1. - p)) / tau
    p1 = (g0 + wp.log(p)) / tau
    

    v = softmax2d(wp.vec2f(p0, p1))
    s = wp.dot(v, wp.vec2f(0. , 1.))

    s_straight = wp.int32(wp.argmax(v))

    return key, s_straight, s



@wp.kernel
def division_losses(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    CT: wp.array(dtype=wp.uint32),
    keys: wp.array(dtype=wp.uint32),

    n_epi: wp.array(dtype=wp.int32),
    n_mes: wp.array(dtype=wp.int32),

    div_count: wp.array(dtype=wp.int32),
    div_slots: wp.array(dtype=wp.int32),

    R_div_ref: wp.float32,
    p_epi: wp.float32,
    epi_max_neighbors: wp.int32,

    t: wp.float32,
    tmax: wp.float32,
    tau: wp.float32,
    max_particles: wp.int32,
):
    parent = wp.tid()

    key = keys[parent]

    p = wp.float32(0.0)

    if CT[parent] == wp.uint32(1):

        if n_mes[parent] <= wp.int32(0):
            keys[parent] = key
            return
        
        if n_epi[parent] >=  wp.int32(epi_max_neighbors):
            keys[parent] = key
            return

        p = p_epi

    else:


        p = probs(R[parent], R_div_ref)

    # Sample division (Gumbel-ST)
    key, s_hard, s_soft = st_gumbel_softmax_bernoulli(p, key, t, tmax, tau)
    keys[parent] = key

    if s_hard == wp.int32(0):
        return

    # Reserve child slot (unique)
    child = wp.atomic_add(div_count, 0, 1)

    if wp.int32(child) >= wp.int32(max_particles):
        print("capacity exceeded")
        return

    div_slots[parent] = child
    
@wp.kernel
def division_logic(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    R_eq: wp.array(dtype=wp.float32),
    A: wp.array(dtype=wp.float32),
    I: wp.array(dtype=wp.float32),
    P: wp.array(dtype=wp.vec3f),
    CT: wp.array(dtype=wp.uint32),
    div_slots: wp.array(dtype=wp.int32),

):
    
    parent = wp.tid()
    
    child = div_slots[parent]

    if child == -1:
        return

    # Split chemical states evenly
    a_p, i_p = A[parent], I[parent]
    A[parent] = 0.5 * a_p
    A[child] = 0.5 * a_p
    I[parent] = 0.5 * i_p
    I[child] = 0.5 * i_p

    # Randomly generate new cell
    ct = CT[parent]
    CT[child] = ct

    # Half volume, cuberoot 2 r
    r = R[parent]
    
    # Epithelium extends, mesenchyme splits in half volume
    if ct == wp.uint32(0):
        r = r / wp.cbrt(2.0)
    
    R[child] = r
    R_eq[child] = r
    R[parent] = r
    R_eq[parent] = r

    # Pass parent polarity
    v = P[parent]
    P[child] = v

    

    # # Polarized division - divide on perpendicular surface
    a = wp.vec3f(1.0, 0.0, 0.0)
    if wp.abs(v[0]) > wp.float32(0.9):
        a = wp.vec3f(0.0, 1.0, 0.0)

    u = wp.normalize(wp.cross(v, a))
    x = X[parent]
    sep = 1.02 * r
    X[parent] = x + u * sep
    X[child] = x - u * sep



# Reload signal
wp.clear_kernel_cache()
wp.clear_lto_cache()