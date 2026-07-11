"""Prescribed Warp mechanics, chemistry, growth, and division in model units.

Division grows the active prefix up to caller-owned capacity. Mechanics stores a
potential gradient and applies ``X_next = X - lr * f_net``; the differentiable
emulator stores a physical force and adds it. The paths share abstractions and
provide purpose-built kernels and cell-type behavior. ``CT`` encodes
mesenchymal as ``0`` and epithelial as ``1``.
"""

from math import isfinite as _isfinite

import warp as wp

from .constants import EPS_DEN, EPS_NORM, FOUR_THIRDS_PI, HASH_GRID_DIM, RAND_EPS

EPS_DIST = 25e-2


K_REP = 3.0

K_ATT_EE = 1.5
K_ATT_MM = 0.15
K_ATT_EM = 0.15


ATR_EE_CUTOFF = 0.25
ATR_EE = 0.14
ATR_MM = 0.1
ATR_EM = 0.06


D_SHIFT_EM = 0.08

K_THICK_EE = 6.0
H_THICK_EE = 0.08


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
    r"""Size-driven mesenchymal division probability.

    .. math::

        p_i = \frac{r_i^{20}}{r_i^{20} + r_{\mathrm{ref}}^{20}}.

    The fixed Hill exponent makes the transition steep near ``ref``.
    """
    return (p**20.0) / ((p**20.0) + (ref**20.0))


@wp.func
def softmax2d(p: wp.vec2f):
    m = wp.max(p.x, p.y)
    ex = wp.exp(p.x - m)
    ey = wp.exp(p.y - m)
    inv_sum = 1.0 / (ex + ey)
    return wp.vec2f(ex * inv_sum, ey * inv_sum)


@wp.func
def gumbel(key: wp.uint32):
    """Draw one Gumbel variate while advancing the per-particle key."""
    u = wp.randf(key, 0.0, 1.0)
    u = wp.clamp(u, RAND_EPS, 1.0 - RAND_EPS)
    return wp.randu(key), -wp.log(-wp.log(u))


@wp.kernel
def fill_key_array(
    key_array: wp.array(dtype=wp.uint32),
) -> None:

    i = wp.tid()
    key_array[i] = wp.rand_init(i)


def gen_key_array(size: int, device: str = "cuda") -> wp.array:
    """Initialize one deterministic RNG key per array index."""

    key_array = wp.zeros(shape=size, dtype=wp.uint32, device=device)
    wp.launch(fill_key_array, dim=key_array.size, inputs=[key_array], device=device)
    return key_array


@wp.func
def sticky_sphere_forces(
    x_i: wp.vec3f,
    x_j: wp.vec3f,
    r_i: wp.float32,
    r_j: wp.float32,
    ct_i: wp.uint32,
    ct_j: wp.uint32,
):
    r"""Cell-type soft-sphere potential gradient for one pair.

    For :math:`q=x_i-x_j`, :math:`d=\lVert q\rVert_2+EPS_NORM`,
    :math:`u=q/d`, and
    :math:`d_0=r_i+r_j+\mathrm{d\_shift}`,

    .. math::

        f_{ij}=\left[k_{\mathrm{rep}}\max(d_0-d,0)
        -f_{\mathrm{att}}(d)\right](-u).

    For EE, :math:`\delta=\max(d-d_0,0)` and attraction is
    :math:`k_{\mathrm{att}}[\delta+\delta^3/(ATR_{EE}^2+EPS_NORM)]` until its
    outer support boundary :math:`d\le d_0+ATR_{EE\_CUTOFF}`. MM and EM use
    :math:`k_{\mathrm{att}}\max(d_0+ATR_{pair}-d,0)\mathbb{1}[d>d_0]`, where
    :math:`ATR_{pair}` is ``ATR_MM`` or ``ATR_EM``. ``D_SHIFT_EM`` offsets the
    heterotypic preferred distance. The result is a potential-gradient buffer
    subtracted by :func:`gd_update`.
    """
    d = x_i - x_j
    dist = wp.length(d) + EPS_NORM
    u = d / dist

    rs = r_i + r_j

    k_att = K_ATT_MM
    atr = ATR_MM
    d_shift = 0.0

    is_ee = (ct_i == wp.uint32(1)) and (ct_j == wp.uint32(1))
    is_em = ct_i != ct_j

    if is_ee:
        k_att = K_ATT_EE
        atr = ATR_EE

    elif is_em:
        k_att = K_ATT_EM
        atr = ATR_EM
        d_shift = D_SHIFT_EM

    d0 = rs + d_shift

    t = dist - d0
    comp = wp.max(-t, wp.float32(0.0))
    ext = wp.max(t, wp.float32(0.0))

    f_rep = K_REP * comp

    f_att = 0.0

    if is_ee:

        invL2 = wp.float32(1.0) / (atr * atr + EPS_NORM)
        f_att = k_att * (ext + ext * ext * ext * invL2)

        if dist > d0 + ATR_EE_CUTOFF:
            return wp.vec3f(0.0), wp.vec3f(0.0)

    else:
        if dist > d0 + atr:
            return wp.vec3f(0.0), wp.vec3f(0.0)

        att_ramp = wp.max((d0 + atr) - dist, wp.float32(0.0)) * wp.float32(dist > d0)
        f_att = k_att * att_ramp

    fmag = f_rep - f_att
    F_ij = fmag * -u
    return F_ij, -F_ij


@wp.func
def epi_polarity_grads(
    x_i: wp.vec3f,
    x_j: wp.vec3f,
    p_i: wp.vec3f,
    p_j: wp.vec3f,
):
    r"""Implemented epithelial-polarity derivative mapping for one pair.

    .. math::

        s=\lVert x_i-x_j\rVert_2+EPS_NORM,\qquad
        u=(x_i-x_j)/s.

    With :math:`a=p_i^Tu`, :math:`b=p_j^Tu`, and :math:`g=ap_i+bp_j`, the
    polarity outputs are :math:`au` and :math:`bu`; the position output is
    :math:`[g-u(u^Tg)]/s`, with the opposite sign on ``j``.
    """

    d = x_i - x_j
    dist = wp.length(d) + EPS_NORM
    u = d / dist

    a = wp.dot(p_i, u)
    b = wp.dot(p_j, u)

    grad_p_i = a * u
    grad_p_j = b * u

    g_u = a * p_i + b * p_j

    proj = g_u - u * wp.dot(u, g_u)

    grad_d = proj / dist

    grad_x_i = grad_d
    grad_x_j = -grad_d

    return grad_x_i, grad_x_j, grad_p_i, grad_p_j


@wp.func
def epi_thickness_grads(x_i: wp.vec3f, x_j: wp.vec3f, p_i: wp.vec3f, p_j: wp.vec3f):
    r"""Analytic position gradient of the epithelial thickness penalty.

    .. math::

        U=\tfrac12 K_{\mathrm{thick}}
        \max(|(x_i-x_j)^Tn|-H_{\mathrm{thick}},0)^2.

    ``n`` is the sign-aligned polarity sum divided by its norm plus ``EPS_NORM``
    and is held fixed with respect to position. The hinge contributes zero
    inside its slack band.
    """
    d = x_i - x_j

    if wp.dot(p_i, -p_j) > wp.dot(p_i, p_j):
        p_j = -p_j

    s = p_i + p_j
    ns = wp.length(s) + EPS_NORM
    n = s / ns
    dn = wp.dot(d, n)

    adn = wp.abs(dn)

    excess = wp.max(adn - H_THICK_EE, wp.float32(0.0))

    sign = wp.float32(1.0)

    if dn < wp.float32(0.0):
        sign = wp.float32(-1.0)

    grad_x_i = K_THICK_EE * excess * sign * n
    grad_x_j = -grad_x_i

    return grad_x_i, grad_x_j


@wp.func
def mes_polarity_grads(p_i: wp.vec3f, p_j: wp.vec3f):
    r"""Analytic gradient of unoriented mesenchymal polarity alignment.

    .. math::

        U=-\tfrac12(p_i^Tp_j)^2,\qquad
        \nabla_{p_i}U=-(p_i^Tp_j)p_j,
        \quad \nabla_{p_j}U=-(p_i^Tp_j)p_i.
    """

    c = wp.dot(p_i, p_j)
    grad_p_i = -c * p_j
    grad_p_j = -c * p_i
    return grad_p_i, grad_p_j


@wp.func
def mes_wnt_polarity_grads(p_i: wp.vec3f, u_to_higher_wnt: wp.vec3f, w_higher: wp.float32):
    r"""Analytic mesenchymal-polarity gradient toward higher WNT.

    .. math::

        U=-\tfrac12 c_{j,A}(p_i^Tu_{ji})^2,\qquad
        \nabla_{p_i}U=-c_{j,A}(p_i^Tu_{ji})u_{ji}.

    Callers use :math:`u_{ji}=(x_j-x_i)/(\lVert x_j-x_i\rVert+EPS_NORM)`;
    positions and concentrations are held fixed.
    """
    c = wp.dot(p_i, u_to_higher_wnt)
    return -w_higher * c * u_to_higher_wnt


@wp.kernel(enable_backward=False)
def sticky_sphere_grads(
    grid: wp.uint64,
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    P: wp.array(dtype=wp.vec3f),
    CT: wp.array(dtype=wp.uint32),
    query_radius: wp.float32,
    f_net: wp.array(dtype=wp.vec3f),
    f_pol: wp.array(dtype=wp.vec3f),
):
    r"""Accumulate type-dependent mechanics and polarity gradients.

    Epithelial polarity position-gradient components are clipped to
    :math:`\pm\max(|f_{soft}|,10^{-3})`; thickness gradients retain their full values.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    x_i = X[i]
    r_i = R[i]
    c_i = CT[i]
    p_i = P[i]

    for j in wp.hash_grid_query(grid, x_i, query_radius):
        if j <= i:
            continue

        x_j = X[j]
        r_j = R[j]
        c_j = CT[j]
        p_j = P[j]

        grad_x_i_f, grad_x_j_f = sticky_sphere_forces(x_i, x_j, r_i, r_j, c_i, c_j)
        wp.atomic_add(f_net, i, grad_x_i_f)
        wp.atomic_add(f_net, j, grad_x_j_f)

        dist = wp.norm_l2(x_i - x_j)
        w = adj_weight(dist, r_i, r_j)

        if not wp.bool(w):
            continue

        if (c_i == wp.uint32(1)) and (c_j == wp.uint32(1)):
            grad_x_i_p, grad_x_j_p, grad_p_i, grad_p_j = epi_polarity_grads(x_i, x_j, p_i, p_j)
            grad_x_i_t, grad_x_j_t = epi_thickness_grads(x_i, x_j, p_i, p_j)

            for k in range(3):
                v_i = wp.max(wp.abs(grad_x_i_f[k]), wp.float32(1e-3))
                v_j = wp.max(wp.abs(grad_x_j_f[k]), wp.float32(1e-3))

                grad_x_i_p[k] = wp.clamp(grad_x_i_p[k], -1.0 * v_i, 1.0 * v_i)
                grad_x_j_p[k] = wp.clamp(grad_x_j_p[k], -1.0 * v_j, 1.0 * v_j)

            wp.atomic_add(f_net, i, grad_x_i_p)
            wp.atomic_add(f_net, j, grad_x_j_p)
            wp.atomic_add(f_net, i, grad_x_i_t)
            wp.atomic_add(f_net, j, grad_x_j_t)
            wp.atomic_add(f_pol, i, grad_p_i)
            wp.atomic_add(f_pol, j, grad_p_j)

        if (c_i == wp.uint32(0)) and (c_j == wp.uint32(0)):
            grad_p_i, grad_p_j = mes_polarity_grads(p_i, p_j)
            wp.atomic_add(f_pol, i, grad_p_i)
            wp.atomic_add(f_pol, j, grad_p_j)


@wp.kernel(enable_backward=False)
def sticky_sphere_wnt_grads(
    grid: wp.uint64,
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    P: wp.array(dtype=wp.vec3f),
    CT: wp.array(dtype=wp.uint32),
    WNT: wp.array(dtype=wp.float32),
    query_radius: wp.float32,
    gp: wp.array(dtype=wp.vec3f),
):
    """Accumulate optional WNT-driven mesenchymal polarity gradients."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    x_i = X[i]
    r_i = R[i]
    c_i = CT[i]
    p_i = P[i]
    w_i = safe_div(WNT[i], volume_from_radius(r_i))

    for j in wp.hash_grid_query(grid, x_i, query_radius):
        if j <= i:
            continue

        x_j = X[j]
        r_j = R[j]
        dist = wp.norm_l2(x_i - x_j)
        w = adj_weight(dist, r_i, r_j)

        if not wp.bool(w):
            continue

        c_j = CT[j]
        p_j = P[j]
        w_j = safe_div(WNT[j], volume_from_radius(r_j))
        dist_safe = dist + EPS_NORM

        if (c_i == wp.uint32(0)) and (w_j > w_i):
            u_i_to_j = (x_j - x_i) / dist_safe
            wp.atomic_add(gp, i, mes_wnt_polarity_grads(p_i, u_i_to_j, w_j))

        if (c_j == wp.uint32(0)) and (w_i > w_j):
            u_j_to_i = (x_i - x_j) / dist_safe
            wp.atomic_add(gp, j, mes_wnt_polarity_grads(p_j, u_j_to_i, w_i))


@wp.kernel
def gd_update(
    X: wp.array(dtype=wp.vec3f),
    f_net: wp.array(dtype=wp.vec3f),
    lr: wp.float32,
    X_next: wp.array(dtype=wp.vec3f),
):
    r"""Apply :math:`X_{next}=X-lr\,f_{net}` for a potential-gradient buffer."""

    i = wp.tid()
    X_next[i] = X[i] - lr * f_net[i]


@wp.kernel
def gd_update_normalized(
    P: wp.array(dtype=wp.vec3f),
    f_pol: wp.array(dtype=wp.vec3f),
    lr: wp.float32,
    P_next: wp.array(dtype=wp.vec3f),
):
    """Apply polarity descent and renormalize nonzero vectors."""

    i = wp.tid()
    P_next[i] = wp.normalize(P[i] - lr * f_pol[i])


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
    grid: "wp.HashGrid | None" = None,
    *,
    wnt: "wp.array | None" = None,
):
    """Advance prescribed mechanics using explicit pair derivatives.

    The active contact graph is rebuilt with a Warp HashGrid. Optional WNT adds
    mesenchymal polarity alignment; epithelial polarity position gradients use
    the componentwise soft-force clamp documented by :func:`sticky_sphere_grads`.
    ``grad_consist`` emits replay markers.
    """

    f_net = wp.zeros_like(X, device=device)
    f_pol = wp.zeros_like(P, device=device)

    r_max = float(R.numpy()[:particle_count].max())
    query_radius = 2.0 * r_max + ATR_EE_CUTOFF
    if grid is None:
        grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    grid.build(X[:particle_count], query_radius)

    wp.launch(
        sticky_sphere_grads,
        dim=particle_count,
        inputs=[wp.uint64(grid.id), X, R, P, CT, query_radius],
        outputs=[f_net, f_pol],
        device=device,
    )

    if wnt is not None:
        wp.launch(
            sticky_sphere_wnt_grads,
            dim=particle_count,
            inputs=[wp.uint64(grid.id), X, R, P, CT, wnt, query_radius],
            outputs=[f_pol],
            device=device,
        )

    if grad_consist:
        f_net.mark_write()

    wp.launch(
        gd_update,
        dim=particle_count,
        inputs=[X, f_net, dt, X_next],
        device=device,
    )

    wp.launch(
        gd_update_normalized,
        dim=particle_count,
        inputs=[P, f_pol, dt, P_next],
        device=device,
    )

    if grad_consist:
        X.mark_read()
        P.mark_read()
        f_net.mark_read()
        f_pol.mark_read()

        X_next.mark_write()
        P_next.mark_write()

    return f_net


@wp.func
def epi_polarity_potential(
    x_i: wp.vec3f,
    x_j: wp.vec3f,
    p_i: wp.vec3f,
    p_j: wp.vec3f,
) -> wp.float32:
    r"""Epithelial polarity potential.

    .. math::

        U=\tfrac12(p_i\cdot u)^2+\tfrac12(p_j\cdot u)^2,
        \qquad u=\frac{x_i-x_j}{\lVert x_i-x_j\rVert_2+EPS_NORM}.
    """
    d = x_i - x_j
    dist = wp.length(d) + EPS_NORM
    u = d / dist
    a = wp.dot(p_i, u)
    b = wp.dot(p_j, u)
    return wp.float32(0.5) * a * a + wp.float32(0.5) * b * b


@wp.func
def epi_thickness_potential(
    x_i: wp.vec3f,
    x_j: wp.vec3f,
    p_i: wp.vec3f,
    p_j: wp.vec3f,
) -> wp.float32:
    r"""Epithelial thickness potential.

    .. math::

        U=\tfrac12K_{\mathrm{thick}}
        \max(|d\cdot n|-H_{\mathrm{thick}},0)^2.

    ``n`` is the sign-aligned polarity sum divided by its norm plus ``EPS_NORM``.
    """
    d = x_i - x_j
    p_j_use = p_j
    if wp.dot(p_i, -p_j) > wp.dot(p_i, p_j):
        p_j_use = -p_j
    s = p_i + p_j_use
    ns = wp.length(s) + EPS_NORM
    n = s / ns
    dn = wp.dot(d, n)
    adn = wp.abs(dn)
    excess = wp.max(adn - H_THICK_EE, wp.float32(0.0))
    return wp.float32(0.5) * K_THICK_EE * excess * excess


@wp.func
def mes_polarity_potential(p_i: wp.vec3f, p_j: wp.vec3f) -> wp.float32:
    r"""Unoriented mesenchymal polarity potential.

    .. math:: U=-\tfrac12(p_i\cdot p_j)^2.
    """
    c = wp.dot(p_i, p_j)
    return wp.float32(-0.5) * c * c


@wp.func
def mes_wnt_polarity_potential(
    p_i: wp.vec3f,
    u_to_higher_wnt: wp.vec3f,
    w_higher: wp.float32,
) -> wp.float32:
    r"""Mesenchymal polarity potential toward higher WNT.

    .. math:: U=-\tfrac12c_{j,A}(p_i^Tu_{ji})^2.

    Callers supply :math:`u=(x_j-x_i)/(\lVert x_j-x_i\rVert+EPS_NORM)`.
    """
    c = wp.dot(p_i, u_to_higher_wnt)
    return wp.float32(-0.5) * w_higher * c * c


@wp.kernel(enable_backward=False)
def sticky_sphere_grads_implicit(
    grid: wp.uint64,
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    P: wp.array(dtype=wp.vec3f),
    CT: wp.array(dtype=wp.uint32),
    query_radius: wp.float32,
    f_net: wp.array(dtype=wp.vec3f),
    f_pol: wp.array(dtype=wp.vec3f),
):
    """Use :func:`warp.grad` for polarity and thickness potentials.

    Soft-sphere gradients remain explicit. Epithelial polarity position-gradient
    components use the same soft-force clamp as :func:`sticky_sphere_grads`.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    x_i = X[i]
    r_i = R[i]
    c_i = CT[i]
    p_i = P[i]

    for j in wp.hash_grid_query(grid, x_i, query_radius):
        if j <= i:
            continue

        x_j = X[j]
        r_j = R[j]
        c_j = CT[j]
        p_j = P[j]

        grad_x_i_f, grad_x_j_f = sticky_sphere_forces(x_i, x_j, r_i, r_j, c_i, c_j)
        wp.atomic_add(f_net, i, grad_x_i_f)
        wp.atomic_add(f_net, j, grad_x_j_f)

        dist = wp.norm_l2(x_i - x_j)
        w = adj_weight(dist, r_i, r_j)

        if not wp.bool(w):
            continue

        if (c_i == wp.uint32(1)) and (c_j == wp.uint32(1)):
            grad_x_i_p, grad_x_j_p, grad_p_i, grad_p_j = wp.grad(epi_polarity_potential)(
                x_i, x_j, p_i, p_j
            )
            grad_x_i_t, grad_x_j_t, _, _ = wp.grad(epi_thickness_potential)(x_i, x_j, p_i, p_j)

            for k in range(3):
                v_i = wp.max(wp.abs(grad_x_i_f[k]), wp.float32(1e-3))
                v_j = wp.max(wp.abs(grad_x_j_f[k]), wp.float32(1e-3))

                grad_x_i_p[k] = wp.clamp(grad_x_i_p[k], -1.0 * v_i, 1.0 * v_i)
                grad_x_j_p[k] = wp.clamp(grad_x_j_p[k], -1.0 * v_j, 1.0 * v_j)

            wp.atomic_add(f_net, i, grad_x_i_p)
            wp.atomic_add(f_net, j, grad_x_j_p)
            wp.atomic_add(f_net, i, grad_x_i_t)
            wp.atomic_add(f_net, j, grad_x_j_t)
            wp.atomic_add(f_pol, i, grad_p_i)
            wp.atomic_add(f_pol, j, grad_p_j)

        if (c_i == wp.uint32(0)) and (c_j == wp.uint32(0)):
            grad_p_i, grad_p_j = wp.grad(mes_polarity_potential)(p_i, p_j)
            wp.atomic_add(f_pol, i, grad_p_i)
            wp.atomic_add(f_pol, j, grad_p_j)


@wp.kernel(enable_backward=False)
def sticky_sphere_wnt_grads_implicit(
    grid: wp.uint64,
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    P: wp.array(dtype=wp.vec3f),
    CT: wp.array(dtype=wp.uint32),
    WNT: wp.array(dtype=wp.float32),
    query_radius: wp.float32,
    gp: wp.array(dtype=wp.vec3f),
):
    """Accumulate WNT alignment with a local :func:`warp.grad` derivative."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    x_i = X[i]
    r_i = R[i]
    c_i = CT[i]
    p_i = P[i]
    w_i = safe_div(WNT[i], volume_from_radius(r_i))

    for j in wp.hash_grid_query(grid, x_i, query_radius):
        if j <= i:
            continue

        x_j = X[j]
        r_j = R[j]
        dist = wp.norm_l2(x_i - x_j)
        w = adj_weight(dist, r_i, r_j)

        if not wp.bool(w):
            continue

        c_j = CT[j]
        p_j = P[j]
        w_j = safe_div(WNT[j], volume_from_radius(r_j))
        dist_safe = dist + EPS_NORM

        if (c_i == wp.uint32(0)) and (w_j > w_i):
            u_i_to_j = (x_j - x_i) / dist_safe
            grad_p_i, _grad_u_i, _grad_w_i = wp.grad(mes_wnt_polarity_potential)(p_i, u_i_to_j, w_j)
            wp.atomic_add(gp, i, grad_p_i)

        if (c_j == wp.uint32(0)) and (w_i > w_j):
            u_j_to_i = (x_i - x_j) / dist_safe
            grad_p_j, _grad_u_j, _grad_w_j = wp.grad(mes_wnt_polarity_potential)(p_j, u_j_to_i, w_i)
            wp.atomic_add(gp, j, grad_p_j)


def mech_step_sticky_implicit(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    P: wp.array(dtype=wp.vec3f),
    CT: wp.array(dtype=wp.uint32),
    particle_count: wp.int32,
    dt: float,
    X_next: wp.array(dtype=wp.vec3f),
    P_next: wp.array(dtype=wp.vec3f),
    device: str = "cuda",
    grid: "wp.HashGrid | None" = None,
    *,
    wnt: "wp.array | None" = None,
):
    """Advance mechanics with local autodiff for polarity and thickness potentials."""

    f_net = wp.zeros_like(X, device=device)
    f_pol = wp.zeros_like(P, device=device)

    r_max = float(R.numpy()[:particle_count].max())
    query_radius = 2.0 * r_max + ATR_EE_CUTOFF
    if grid is None:
        grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    grid.build(X[:particle_count], query_radius)

    wp.launch(
        sticky_sphere_grads_implicit,
        dim=particle_count,
        inputs=[wp.uint64(grid.id), X, R, P, CT, query_radius],
        outputs=[f_net, f_pol],
        device=device,
    )

    if wnt is not None:
        wp.launch(
            sticky_sphere_wnt_grads_implicit,
            dim=particle_count,
            inputs=[wp.uint64(grid.id), X, R, P, CT, wnt, query_radius],
            outputs=[f_pol],
            device=device,
        )

    wp.launch(
        gd_update,
        dim=particle_count,
        inputs=[X, f_net, dt, X_next],
        device=device,
    )

    wp.launch(
        gd_update_normalized,
        dim=particle_count,
        inputs=[P, f_pol, dt, P_next],
        device=device,
    )

    return f_net


@wp.kernel
def reaction_diffs(
    grid: wp.uint64,
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    query_radius: wp.float32,
    A: wp.array(dtype=wp.float32),
    I: wp.array(dtype=wp.float32),
    lapA: wp.array(dtype=wp.float32),
    lapI: wp.array(dtype=wp.float32),
):
    """Accumulate symmetric contact-graph diffusion for activator and inhibitor."""

    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x_i = X[i]
    Ri = R[i]
    Vi = volume_from_radius(Ri)
    cAi = safe_div(A[i], Vi)
    cIi = safe_div(I[i], Vi)

    for j in wp.hash_grid_query(grid, x_i, query_radius):
        if j <= i:
            continue

        Rj = R[j]
        dist = wp.norm_l2(x_i - X[j])
        w = adj_weight(dist, Ri, Rj)

        if not wp.bool(w):
            continue

        Vj = volume_from_radius(Rj)
        cAj = safe_div(A[j], Vj)
        cIj = safe_div(I[j], Vj)

        dA = wp.float32(w) * (cAj - cAi)
        dI = wp.float32(w) * (cIj - cIi)

        wp.atomic_add(lapA, i, dA)
        wp.atomic_add(lapA, j, -dA)
        wp.atomic_add(lapI, i, dI)
        wp.atomic_add(lapI, j, -dI)


@wp.kernel
def reaction_step(
    A: wp.array(dtype=wp.float32),
    I: wp.array(dtype=wp.float32),
    R: wp.array(dtype=wp.float32),
    lapA: wp.array(dtype=wp.float32),
    lapI: wp.array(dtype=wp.float32),
    chi: wp.array(dtype=wp.float32),
    gamma: wp.array(dtype=wp.float32),
    D_inhib: wp.float32,
    dt: wp.float32,
    A_next: wp.array(dtype=wp.float32),
    I_next: wp.array(dtype=wp.float32),
):
    r"""Advance the activator-inhibitor system by one bounded discrete step.

    With regularized concentrations :math:`a=A/(V+EPS_DEN)`,
    :math:`i=I/(V+EPS_DEN)`, and contact fluxes
    :math:`\ell_A=\sum_j(a_j-a_i)`, :math:`\ell_I=\sum_j(i_j-i_i)`,

    .. math::

        A_i' = \operatorname{clip}\!\left(
        \frac{A_i+dt\,\gamma[\chi D_I\ell_A+
        \min(\frac{a_i^2}{i_i+EPS_DEN},
        \frac{a_i^2}{i_i^2+EPS_DEN})]}{1+dt\,\gamma},0,10^4\right),

        I_i' = \operatorname{clip}\!\left(
        \frac{I_i+dt\,\gamma[D_I\ell_I+a_i^2]}{1+dt\,\gamma},0,10^4\right).

    The production cap, semi-implicit linear damping, and output clamp differ
    from an uncapped explicit reaction-diffusion ODE.
    """

    i = wp.tid()

    V = volume_from_radius(R[i])

    cA = safe_div(A[i], V)
    cI = safe_div(I[i], V)

    cA2 = cA * cA
    ciI = cI

    prodA_lin = safe_div(cA2, ciI)
    prodA_quad = safe_div(cA2, ciI * ciI)

    prodA = prodA_lin
    if wp.float32(prodA_quad) < wp.float32(prodA_lin):
        prodA = prodA_quad

    prodI = cA2

    diffA = (chi[0] * D_inhib) * lapA[i]
    diffI = D_inhib * lapI[i]

    zAi = A[i] + dt * gamma[0] * (diffA + prodA)
    zIi = I[i] + dt * gamma[0] * (diffI + prodI)

    inv = 1.0 / (1.0 + dt * gamma[0])
    Ai_next = zAi * inv
    Ii_next = zIi * inv

    A_next[i] = wp.clamp(Ai_next, 0.0, 1e4)
    I_next[i] = wp.clamp(Ii_next, 0.0, 1e4)


@wp.kernel
def reaction_step_masked(
    A: wp.array(dtype=wp.float32),
    I: wp.array(dtype=wp.float32),
    R: wp.array(dtype=wp.float32),
    lapA: wp.array(dtype=wp.float32),
    lapI: wp.array(dtype=wp.float32),
    chi: wp.array(dtype=wp.float32),
    gamma: wp.array(dtype=wp.float32),
    D_inhib: wp.float32,
    dt: wp.float32,
    CT: wp.array(dtype=wp.uint32),
    reaction_cell_type: wp.uint32,
    A_next: wp.array(dtype=wp.float32),
    I_next: wp.array(dtype=wp.float32),
):
    """Apply nonlinear reaction to the selected cell type and diffusion to the remaining cells.

    All cells diffuse; unmatched cells receive pure diffusion. Matched cells use
    the same production cap, semi-implicit damping, and output clamp as
    :func:`reaction_step`.
    """

    i = wp.tid()

    diffA = (chi[0] * D_inhib) * lapA[i]
    diffI = D_inhib * lapI[i]

    if CT[i] != reaction_cell_type:
        Ai_next = A[i] + dt * gamma[0] * diffA
        Ii_next = I[i] + dt * gamma[0] * diffI
        A_next[i] = wp.clamp(Ai_next, 0.0, 1e4)
        I_next[i] = wp.clamp(Ii_next, 0.0, 1e4)
        return

    V = volume_from_radius(R[i])

    cA = safe_div(A[i], V)
    cI = safe_div(I[i], V)

    cA2 = cA * cA
    ciI = cI

    prodA_lin = safe_div(cA2, ciI)
    prodA_quad = safe_div(cA2, ciI * ciI)

    prodA = prodA_lin
    if wp.float32(prodA_quad) < wp.float32(prodA_lin):
        prodA = prodA_quad

    prodI = cA2

    zAi = A[i] + dt * gamma[0] * (diffA + prodA)
    zIi = I[i] + dt * gamma[0] * (diffI + prodI)

    inv = 1.0 / (1.0 + dt * gamma[0])
    Ai_next = zAi * inv
    Ii_next = zIi * inv

    A_next[i] = wp.clamp(Ai_next, 0.0, 1e4)
    I_next[i] = wp.clamp(Ii_next, 0.0, 1e4)


def chem_step(
    A: wp.array,
    I: wp.array,
    X: wp.array,
    R: wp.array,
    lapA: wp.array,
    lapI: wp.array,
    chi: wp.array,
    gamma: wp.array,
    D_inhib: float,
    dt: float,
    particle_count: int,
    A_next: wp.array,
    I_next: wp.array,
    device: str = "cuda",
    grad_consist: bool = True,
    grid: "wp.HashGrid | None" = None,
    *,
    CT: "wp.array | None" = None,
    reaction_cell_type: int | None = None,
):
    """Run contact diffusion followed by masked or unmasked reaction.

    ``reaction_cell_type`` restricts nonlinear reaction while retaining
    diffusion for every cell and requires ``CT``. ``grad_consist`` emits replay
    markers. ``lapA`` and ``lapI`` must be zeroed by the caller because pair
    fluxes accumulate atomically.
    """

    if reaction_cell_type is not None and CT is None:
        raise ValueError("CT must be provided when reaction_cell_type is set")

    r_max = float(R.numpy()[:particle_count].max())
    query_radius = 2.0 * r_max + EPS_DIST
    if grid is None:
        grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    grid.build(X[:particle_count], query_radius)

    wp.launch(
        reaction_diffs,
        dim=particle_count,
        inputs=[wp.uint64(grid.id), X, R, query_radius, A, I],
        outputs=[lapA, lapI],
        device=device,
    )

    if grad_consist:
        lapA.mark_write()
        lapI.mark_write()

    if reaction_cell_type is None:
        wp.launch(
            reaction_step,
            dim=particle_count,
            inputs=[A, I, R, lapA, lapI, chi, gamma, D_inhib, dt],
            outputs=[A_next, I_next],
            device=device,
        )
    else:
        wp.launch(
            reaction_step_masked,
            dim=particle_count,
            inputs=[
                A,
                I,
                R,
                lapA,
                lapI,
                chi,
                gamma,
                D_inhib,
                dt,
                CT,
                wp.uint32(reaction_cell_type),
            ],
            outputs=[A_next, I_next],
            device=device,
        )

    if grad_consist:
        lapA.mark_read()
        lapI.mark_read()
        A.mark_read()
        I.mark_read()

        A_next.mark_write()
        I_next.mark_write()


@wp.kernel
def count_neighbors(
    grid: wp.uint64,
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    CT: wp.array(dtype=wp.uint32),
    query_radius: wp.float32,
    n_tot: wp.array(dtype=wp.int32),
    n_epi: wp.array(dtype=wp.int32),
    n_mes: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x_i = X[i]
    r_i = R[i]

    for j in wp.hash_grid_query(grid, x_i, query_radius):
        if j <= i:
            continue

        r_j = R[j]
        dist = wp.norm_l2(x_i - X[j])
        w = adj_weight(dist, r_i, r_j)

        if not wp.bool(w):
            continue

        wp.atomic_add(n_tot, i, 1)
        wp.atomic_add(n_tot, j, 1)

        if CT[j] == wp.uint32(1):
            wp.atomic_add(n_epi, i, 1)
        else:
            wp.atomic_add(n_mes, i, 1)

        if CT[i] == wp.uint32(1):
            wp.atomic_add(n_epi, j, 1)
        else:
            wp.atomic_add(n_mes, j, 1)


def count_neighbors_step(
    X: wp.array,
    R: wp.array,
    CT: wp.array,
    particle_count: int,
    n_tot: wp.array,
    n_epi: wp.array,
    n_mes: wp.array,
    device: str = "cuda",
    grid: "wp.HashGrid | None" = None,
) -> None:
    """Atomically accumulate into caller-zeroed neighbor counters."""
    r_max = float(R.numpy()[:particle_count].max())
    query_radius = 2.0 * r_max + EPS_DIST
    if grid is None:
        grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    grid.build(X[:particle_count], query_radius)

    wp.launch(
        count_neighbors,
        dim=particle_count,
        inputs=[wp.uint64(grid.id), X, R, CT, query_radius],
        outputs=[n_tot, n_epi, n_mes],
        device=device,
    )


def growth_step(
    R: wp.array(dtype=wp.float32),
    R_eq: wp.array(dtype=wp.float32),
    A: wp.array(dtype=wp.float32),
    CT: wp.array(dtype=wp.uint32),
    keys: wp.array(dtype=wp.uint32),
    alpha_grow: wp.array(dtype=wp.float32),
    ell_sw: wp.array(dtype=wp.float32),
    dt: float,
    R_ref: wp.float32,
    R_max: wp.float32,
    particle_count: int,
    R_next: wp.array(dtype=wp.float32),
    R_eq_next: wp.array(dtype=wp.float32),
    device: str = "cuda",
    grad_consist: bool = True,
) -> None:
    r"""Apply bounded growth in model units.

    Mesenchymal equilibrium radii use a stochastic rate in ``[0.8, 1)`` and

    .. math::

        \dot r_i^{eq}=\lambda_i\frac{a_i^{\alpha}}
        {\ell_{sw}^{\alpha}+a_i^{\alpha}+EPS_DEN},\qquad
        \dot r_i=(1-r_i/r_i^{eq})^2.

    Here :math:`a_i=A_i/(V_i+EPS_DEN)`; the physical-radius ratio is regularized
    by the same denominator constant. Physical radii below their target advance
    toward it and stop at that bound. Mesenchymal keys advance in place.
    Mesenchymal targets are capped at ``R_max``; epithelial targets copy through
    and physical radii approach ``R_ref``. ``dt`` and ``R_max`` must be finite and
    nonnegative; ``R_ref`` must be finite and positive.
    """
    if not _isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")
    if not _isfinite(R_max) or R_max < 0:
        raise ValueError("R_max must be finite and nonnegative")
    if not _isfinite(R_ref) or R_ref <= 0:
        raise ValueError("R_ref must be finite and positive")

    wp.launch(
        growth_step_inner,
        dim=particle_count,
        inputs=[R, R_eq, A, CT, keys, alpha_grow, ell_sw, dt, R_ref, R_max, R_next, R_eq_next],
        device=device,
    )

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
    alpha_grow: wp.array(dtype=wp.float32),
    ell_sw: wp.array(dtype=wp.float32),
    dt: wp.float32,
    R_ref: wp.float32,
    R_max: wp.float32,
    R_next: wp.array(dtype=wp.float32),
    R_eq_next: wp.array(dtype=wp.float32),
) -> None:
    i = wp.tid()

    if CT[i] == wp.uint32(0):
        r_i = R[i]
        target = wp.min(R_eq[i], R_max)

        V = volume_from_radius(R[i])
        num = safe_div(A[i], V) ** alpha_grow[0]

        key = keys[i]
        lam = wp.randf(key, 0.8, 1.0)
        key = wp.randu(key)

        frac = safe_div(num, (ell_sw[0] ** alpha_grow[0]) + num)

        R_eq_next[i] = wp.min(target + frac * lam * dt, R_max)
        if target > 0.0 and r_i < target:
            increment = ((1.0 - safe_div(r_i, target)) ** 2.0) * dt
            R_next[i] = wp.min(r_i + increment, target)
        else:
            R_next[i] = r_i
        keys[i] = key

    if CT[i] == wp.uint32(1):
        r_i = R[i]
        R_eq_next[i] = R_eq[i]
        if R_ref > 0.0 and r_i < R_ref:
            increment = ((1.0 - safe_div(r_i, R_ref)) ** 2.0) * dt
            R_next[i] = wp.min(r_i + increment, R_ref)
        else:
            R_next[i] = r_i


@wp.func
def st_gumbel_softmax_bernoulli(
    p: wp.float32,
    key: wp.uint32,
    t: wp.float32,
    tmax: wp.float32,
    tau: wp.float32,
):
    """Return hard Gumbel-Softmax Bernoulli sample and soft relaxation.

    ``p`` is clamped to ``[RAND_EPS, 1-RAND_EPS]``. For positive ``tau`` and
    ``tmax``, the sampled temperature is
    ``max(0.1, tau * exp(-log(tau / 0.1) * t / tmax))``.
    :func:`division_decision` bases division on the hard sample; the soft relaxation remains
    available to callers.
    """

    p = wp.clamp(p, RAND_EPS, 1.0 - RAND_EPS)

    key, g0 = gumbel(key)
    key, g1 = gumbel(key)

    k = wp.log(tau / 0.1) / tmax
    tau = wp.max(0.1, tau * wp.exp(-k * t))

    p0 = (g1 + wp.log(1.0 - p)) / tau
    p1 = (g0 + wp.log(p)) / tau

    v = softmax2d(wp.vec2f(p0, p1))
    s = wp.dot(v, wp.vec2f(0.0, 1.0))

    s_straight = wp.int32(wp.argmax(v))

    return key, s_straight, s


@wp.func
def _reserve_division_slot(
    div_count: wp.array(dtype=wp.int32), max_particles: wp.int32
) -> wp.int32:
    current = div_count[0]
    while current < max_particles:
        observed = wp.atomic_cas(div_count, wp.int32(0), current, current + wp.int32(1))
        if observed == current:
            return current
        current = observed
    return wp.int32(-1)


@wp.kernel
def division_decision(
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
    r"""Sample divisions and atomically reserve bounded daughter slots.

    Mesenchymal eligibility follows the radius Hill probability. Epithelial
    eligibility uses ``p_epi`` beside mesenchyme and below ``epi_max_neighbors``
    epithelial neighbors. Accepted parents receive unique indices below
    ``max_particles``; the counter stays within capacity. ``div_count[0]`` must
    identify the first free slot, launched ``div_slots`` entries must start at
    ``-1``, and state arrays used by the division kernels must have at least
    ``max_particles`` rows. Eligible sampled cells advance their keys in place.
    """
    parent = wp.tid()

    key = keys[parent]

    p = wp.float32(0.0)

    if CT[parent] == wp.uint32(1):
        if n_mes[parent] <= wp.int32(0):
            keys[parent] = key
            return

        if n_epi[parent] >= wp.int32(epi_max_neighbors):
            keys[parent] = key
            return

        p = p_epi

    else:
        p = probs(R[parent], R_div_ref)

    key, s_hard, _s_soft = st_gumbel_softmax_bernoulli(p, key, t, tmax, tau)
    keys[parent] = key

    if s_hard == wp.int32(0):
        return

    child = _reserve_division_slot(div_count, max_particles)
    if child < wp.int32(0):
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
    """Split accepted parents along a polarity-perpendicular surface.

    Both daughters inherit type and polarity and receive half the chemical
    abundance. Mesenchymal physical radii conserve half-volume; epithelial
    physical radii copy. Both equilibrium radii reset to the resulting daughter
    radius. Under the unit-polarity caller contract, each center moves
    ``1.02 * daughter_radius`` along a perpendicular axis, giving separation
    ``2.04 * daughter_radius``.
    """

    parent = wp.tid()

    child = div_slots[parent]

    if child == -1:
        return

    a_p, i_p = A[parent], I[parent]
    A[parent] = 0.5 * a_p
    A[child] = 0.5 * a_p
    I[parent] = 0.5 * i_p
    I[child] = 0.5 * i_p

    ct = CT[parent]
    CT[child] = ct

    r = R[parent]

    if ct == wp.uint32(0):
        r = r / wp.cbrt(2.0)

    R[child] = r
    R_eq[child] = r
    R[parent] = r
    R_eq[parent] = r

    v = P[parent]
    P[child] = v

    a = wp.vec3f(1.0, 0.0, 0.0)
    if wp.abs(v[0]) > wp.float32(0.9):
        a = wp.vec3f(0.0, 1.0, 0.0)

    u = wp.normalize(wp.cross(v, a))

    x = X[parent]

    sep = 1.02 * r
    X[parent] = x + u * sep
    X[child] = x - u * sep


@wp.kernel
def division_logic_mes_polarity(
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    R_eq: wp.array(dtype=wp.float32),
    A: wp.array(dtype=wp.float32),
    I: wp.array(dtype=wp.float32),
    P: wp.array(dtype=wp.vec3f),
    CT: wp.array(dtype=wp.uint32),
    div_slots: wp.array(dtype=wp.int32),
):
    """Split mesenchymal daughters along polarity and epithelial daughters across it.

    Inheritance, chemistry, radii, and separation distance match
    :func:`division_logic`; mesenchymal separation follows polarity.
    """

    parent = wp.tid()

    child = div_slots[parent]

    if child == -1:
        return

    a_p, i_p = A[parent], I[parent]
    A[parent] = 0.5 * a_p
    A[child] = 0.5 * a_p
    I[parent] = 0.5 * i_p
    I[child] = 0.5 * i_p

    ct = CT[parent]
    CT[child] = ct

    r = R[parent]

    if ct == wp.uint32(0):
        r = r / wp.cbrt(2.0)

    R[child] = r
    R_eq[child] = r
    R[parent] = r
    R_eq[parent] = r

    v = P[parent]
    P[child] = v

    u = wp.normalize(v)
    if ct == wp.uint32(1):

        a = wp.vec3f(1.0, 0.0, 0.0)
        if wp.abs(v[0]) > wp.float32(0.9):
            a = wp.vec3f(0.0, 1.0, 0.0)

        u = wp.normalize(wp.cross(v, a))

    x = X[parent]

    sep = 1.02 * r
    X[parent] = x + u * sep
    X[child] = x - u * sep
