"""Warp kernels for the forward mechanochemical simulator.

Think of this module as the *forward mode* of waxMorph: every rule (mechanical
potentials, activator-inhibitor reaction-diffusion, growth, division) is
prescribed, and the kernels integrate the resulting dynamics on the GPU. Unlike
the differentiable emulator, the active particle count is allowed to *grow*
through division up to a preallocated ``max_particles`` capacity, which makes
this path well suited to generating cheap training data and to running specified
mechanistic models. The case study modeled here is a polarized
epithelial-mesenchymal spheroid coupled to a two-component Turing system.

Sign-convention note: here the accumulated ``f_net`` is treated as a *descent
gradient* of the prescribed potential, so :func:`gd_update` integrates with a
``-`` sign (``X_next = X - lr * f_net``). The differentiable
:mod:`waxmorph.emulator` instead stores a physical force (``-grad U``) in its
``f_net`` and integrates with a ``+`` sign; the extra unary minus in
:func:`sticky_sphere_forces` is what reconciles the two conventions.

See Also:
    waxmorph.emulator: the non-growing, differentiable counterpart that shares
        the soft-sphere force and graph-Laplacian diffusion.
"""

import warp as wp

from .constants import EPS_DEN, EPS_NORM, FOUR_THIRDS_PI, HASH_GRID_DIM, RAND_EPS

############################################################
############################################################
############################################################

# CONSTANTS / HELPERS

############################################################
############################################################
############################################################

EPS_DIST = 25e-2

# Soft-sphere force constants. The single repulsion coefficient k_rep is
# shared by every pair; the attraction coefficient k_att and the attraction-band
# width are set per cell-type pair to encode differential interface tension.
# EE = epithelial-epithelial, MM = mesenchymal-mesenchymal,
# EM = epithelial-mesenchymal (heterotypic).

K_REP = 3.0  # k_rep: soft-sphere repulsion coefficient

K_ATT_EE = 1.5  # k_att for epi-epi: strong cohesion holds the monolayer together
K_ATT_MM = 0.15  # k_att for mes-mes: medium cohesion within the core
K_ATT_EM = 0.15  # k_att for epi-mes: weak heterotypic adhesion (interface tension)

# Attraction-band widths eps. ATR_EE_CUTOFF is the wider extent of the
# stiffening EE adhesion spring; ATR_* are the band widths for the ramped
# (non-EE) attraction branch.
ATR_EE_CUTOFF = 0.25  # outer cutoff of the EE stiffening-spring attraction
ATR_EE = 0.14  # EE characteristic length scaling the cubic stiffening term
ATR_MM = 0.1  # mes-mes attraction-band width
ATR_EM = 0.06  # epi-mes attraction-band width (short range)

# Heterotypic preferred-distance offset: EM pairs sit slightly farther apart than
# r_i + r_j, sharpening the epithelial-mesenchymal boundary.
D_SHIFT_EM = 0.08

K_THICK_EE = 6.0  # kappa_thk: epithelial-thickness stiffness
H_THICK_EE = 0.08  # h_thk: epithelial-thickness offset / slack


@wp.func
def safe_div(num: wp.float32, den: wp.float32) -> wp.float32:
    """Numerically stable divide."""
    return num / (den + EPS_DEN)


@wp.func
def volume_from_radius(r: wp.float32) -> wp.float32:
    """Sphere volume proxy used for concentration normalization."""
    return FOUR_THIRDS_PI * r * r * r


@wp.func
def adj_weight(dist: wp.float32, ri: wp.float32, rj: wp.float32) -> wp.bool:
    """Neighborhood predicate for contact/adjacency graph edges."""
    return wp.bool((dist - (ri + rj)) <= EPS_DIST)


@wp.func
def probs(p: wp.float32, ref: wp.float32):
    r"""Per-step division probability for a size-driven mesenchymal cell.

    .. math::
        p^{\mathrm{Div\text{-}Mes}}_i = \frac{r_i^{\alpha_{\mathrm{div}}}}
        {r_i^{\alpha_{\mathrm{div}}} + r_{\mathrm{ref}}^{\alpha_{\mathrm{div}}}}

    Args:
        p: Current cell radius :math:`r_i`.
        ref: Reference radius :math:`r_{\mathrm{ref}}`.

    Returns:
        Division probability in ``[0, 1]``.

    Note:
        The hard-coded exponent ``20`` is the division Hill exponent
        :math:`\alpha_{\mathrm{div}}`. Such a steep exponent makes the
        probability switch on sharply as the radius approaches the maximum,
        approximating a size threshold while staying smooth.
    """
    return (p**20.0) / ((p**20.0) + (ref**20.0))


@wp.func
def softmax2d(p: wp.vec2f):
    """Two-logit softmax used by Gumbel-Softmax sampling."""
    m = wp.max(p.x, p.y)
    ex = wp.exp(p.x - m)
    ey = wp.exp(p.y - m)
    inv_sum = 1.0 / (ex + ey)
    return wp.vec2f(ex * inv_sum, ey * inv_sum)


@wp.func
def gumbel(key: wp.uint32):
    """Draw one Gumbel variate and advance RNG key."""
    u = wp.randf(key, 0.0, 1.0)
    u = wp.clamp(u, RAND_EPS, 1.0 - RAND_EPS)
    return wp.randu(key), -wp.log(-wp.log(u))


@wp.kernel
def fill_key_array(
    key_array: wp.array(dtype=wp.uint32),
) -> None:
    """Initialize per-particle random keys deterministically by index."""

    i = wp.tid()
    key_array[i] = wp.rand_init(i)


def gen_key_array(size: int, device: str = "cuda") -> wp.array:
    """Allocate and fill RNG key array for stochastic kernels."""

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
def sticky_sphere_forces(
    x_i: wp.vec3f,
    x_j: wp.vec3f,
    r_i: wp.float32,
    r_j: wp.float32,
    ct_i: wp.uint32,
    ct_j: wp.uint32,
):
    r"""Soft-sphere pairwise force with cell-type-dependent adhesion.

    Specialized to three cell-type pairings. The repulsive branch is always a
    linear spring in compression; the attractive branch differs by pairing.

    .. math::
        f^{\mathrm{soft}}_{ij} = \Big[
        k_{\mathrm{rep}}\,\max(r_i + r_j - \varepsilon - d_{ij},\, 0)
        - k_{\mathrm{att}}\,\max(r_i + r_j + \varepsilon - d_{ij},\, 0)\,
        \mathbb{1}[d_{ij} > r_i + r_j - \varepsilon]
        \Big]\,\hat{\mathbf{u}}_{ij}

    Three pairings (see the module constants) specialize the attractive branch:

    - **EE (epithelial-epithelial):** a *stiffening* adhesion spring whose
      magnitude is :math:`k_{\mathrm{att}}(\delta + \delta^3/\ell^2)` over the
      extension :math:`\delta = d_{ij} - d_0`, with :math:`\ell` = ``ATR_EE``;
      it is truncated beyond an outer cutoff :math:`d_0 + \text{ATR\_EE\_CUTOFF}`.
    - **MM / EM (non-EE):** the *ramped* attraction, linear in
      :math:`(d_0 + \varepsilon) - d_{ij}` and gated to act only in extension
      (:math:`d_{ij} > d_0`); zero beyond :math:`d_0 + \varepsilon`.

    Here the preferred (zero-force) center distance is
    :math:`d_0 = r_i + r_j + \text{d\_shift}`. ``d_shift`` is zero except for
    heterotypic EM pairs (``D_SHIFT_EM``), which prefer to sit slightly farther
    apart to sharpen the tissue interface.

    Args:
        x_i: Center of cell :math:`i`.
        x_j: Center of cell :math:`j`.
        r_i: Radius of cell :math:`i`.
        r_j: Radius of cell :math:`j`.
        ct_i: Cell type of :math:`i` (``1`` = epithelial, ``0`` = mesenchymal).
        ct_j: Cell type of :math:`j`.

    Returns:
        The pair ``(F_ij, -F_ij)``: the force on cell :math:`i` and its
        Newton's-third-law reaction on cell :math:`j` (:math:`f_{ji} = -f_{ij}`).

    Note:
        The returned force is built as ``fmag * -u`` where ``u`` points from
        :math:`j` toward :math:`i`. The extra unary minus turns the physical
        force into the *descent gradient* expected by :func:`gd_update`
        (which subtracts it); the differentiable :mod:`waxmorph.emulator`
        instead returns ``fmag * u`` and adds it. This non-smooth piecewise
        law uses Warp's subgradient at the clamp/cutoff break points.

    See Also:
        waxmorph.emulator._sticky_sphere_forces: the differentiable twin with a
            single attraction branch and the opposite integration sign.
    """
    d = x_i - x_j
    dist = wp.length(d) + EPS_NORM
    u = d / dist

    rs = r_i + r_j

    # defaults: mes-mes
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

    # preferred distance
    d0 = rs + d_shift

    # compression/extension relative to preferred distance
    t = dist - d0
    comp = wp.max(-t, wp.float32(0.0))
    ext = wp.max(t, wp.float32(0.0))

    # repulsion (always from compression)
    f_rep = K_REP * comp

    # attraction
    f_att = 0.0

    if is_ee:
        # EE: stiffening spring in extension
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
    r"""Analytic gradients of the epithelial-polarity potential.

    Closed-form analytic gradient of the pairwise epithelial-polarity term
    with respect to both positions and both polarities. The potential is
    minimized when neighboring epithelial polarities are perpendicular to their
    displacement vector, orienting polarity normal to the sheet.

    .. math::
        U_{\mathrm{Epi\text{-}Pol}} = \tfrac{1}{2}(\mathbf{p}_i^\top
        \hat{\mathbf{u}})^2 + \tfrac{1}{2}(\mathbf{p}_j^\top \hat{\mathbf{u}})^2

    with :math:`\hat{\mathbf{u}} = (\mathbf{x}_i - \mathbf{x}_j)/d_{ij}`. Writing
    :math:`a = \mathbf{p}_i^\top\hat{\mathbf{u}}`,
    :math:`b = \mathbf{p}_j^\top\hat{\mathbf{u}}`, the polarity gradients are
    :math:`\nabla_{\mathbf{p}_i}U = a\,\hat{\mathbf{u}}`,
    :math:`\nabla_{\mathbf{p}_j}U = b\,\hat{\mathbf{u}}`, and the position
    gradient applies the projector :math:`(I - \hat{\mathbf{u}}
    \hat{\mathbf{u}}^\top)/d_{ij}` to :math:`a\mathbf{p}_i + b\mathbf{p}_j`
    (with :math:`\nabla_{\mathbf{x}_j} = -\nabla_{\mathbf{x}_i}`).

    Args:
        x_i: Center of epithelial cell :math:`i`.
        x_j: Center of epithelial cell :math:`j`.
        p_i: Unit polarity of cell :math:`i`.
        p_j: Unit polarity of cell :math:`j`.

    Returns:
        ``(grad_x_i, grad_x_j, grad_p_i, grad_p_j)`` for the pair.

    See Also:
        epi_polarity_potential: the scalar potential differentiated here, used
            by the autodiff path via :func:`warp.grad`.
    """
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
def epi_thickness_grads(x_i: wp.vec3f, x_j: wp.vec3f, p_i: wp.vec3f, p_j: wp.vec3f):
    r"""Analytic gradients of the epithelial-thickness potential.

    Closed-form analytic position gradient of the thickness penalty, which
    penalizes displacement of neighboring epithelial cells along the local sheet
    normal and so discourages cell stacking.

    .. math::
        U_{\mathrm{Epi\text{-}Thk}} = \tfrac{1}{2}\kappa_{\mathrm{thk}}
        \big[\max(|(\mathbf{x}_i - \mathbf{x}_j)^\top \hat{\mathbf{n}}_{ij}|
        - h_{\mathrm{thk}},\, 0)\big]^2

    The local normal :math:`\hat{\mathbf{n}}_{ij}` is the normalized sum of the
    sign-aligned polarities (so the two polarities point the same way). The
    gradient is

    .. math::
        \nabla_{\mathbf{x}_i}U_{\mathrm{Epi\text{-}Thk}} = \kappa_{\mathrm{thk}}
        \max(|\cdot| - h_{\mathrm{thk}},\, 0)\,
        \operatorname{sign}((\mathbf{x}_i-\mathbf{x}_j)^\top\hat{\mathbf{n}}_{ij})
        \,\hat{\mathbf{n}}_{ij}

    with :math:`\nabla_{\mathbf{x}_j} = -\nabla_{\mathbf{x}_i}`. The hinge is
    flat inside the slack band, so the pair contributes the zero subgradient when
    :math:`|(\mathbf{x}_i-\mathbf{x}_j)^\top\hat{\mathbf{n}}_{ij}| \le
    h_{\mathrm{thk}}`. Here :math:`\kappa_{\mathrm{thk}}` = ``K_THICK_EE`` and
    :math:`h_{\mathrm{thk}}` = ``H_THICK_EE``; the interpolated normal is held
    fixed when differentiating with respect to position.

    Args:
        x_i: Center of epithelial cell :math:`i`.
        x_j: Center of epithelial cell :math:`j`.
        p_i: Unit polarity of cell :math:`i`.
        p_j: Unit polarity of cell :math:`j`.

    Returns:
        ``(grad_x_i, grad_x_j)`` for the pair.

    See Also:
        epi_thickness_potential: the scalar potential differentiated here, used
            by the autodiff path via :func:`warp.grad`.
    """
    d = x_i - x_j
    # Pick correct normal (inward / outward, not explicit in our opt scheme)
    if wp.dot(p_i, -p_j) > wp.dot(p_i, p_j):
        p_j = -p_j

    s = p_i + p_j
    ns = wp.length(s) + EPS_NORM
    n = s / ns
    dn = wp.dot(d, n)

    # normal separation (signed)
    adn = wp.abs(dn)

    # hinge: only penalize if |dn| exceeds thickness slack h
    excess = wp.max(adn - H_THICK_EE, wp.float32(0.0))

    # d/d(dn) 0.5*k*excess^2 = k*excess*sign(dn)
    sign = wp.float32(1.0)

    if dn < wp.float32(0.0):
        sign = wp.float32(-1.0)

    grad_x_i = K_THICK_EE * excess * sign * n
    grad_x_j = -grad_x_i

    return grad_x_i, grad_x_j


@wp.func
def mes_polarity_grads(p_i: wp.vec3f, p_j: wp.vec3f):
    r"""Analytic gradients of the mesenchymal-polarity alignment potential.

    Closed-form analytic polarity gradient of the pairwise alignment term,
    which aligns the polarity *axes* of neighboring mesenchymal cells without
    distinguishing orientation (the inner product is squared).

    .. math::
        U_{\mathrm{Mes\text{-}Pol}} = -\tfrac{1}{2}(\mathbf{p}_i^\top
        \mathbf{p}_j)^2

    Writing :math:`c = \mathbf{p}_i^\top\mathbf{p}_j`, the gradients are
    :math:`\nabla_{\mathbf{p}_i}U = -c\,\mathbf{p}_j` and
    :math:`\nabla_{\mathbf{p}_j}U = -c\,\mathbf{p}_i`.

    Args:
        p_i: Unit polarity of mesenchymal cell :math:`i`.
        p_j: Unit polarity of mesenchymal cell :math:`j`.

    Returns:
        ``(grad_p_i, grad_p_j)`` for the pair.

    See Also:
        mes_polarity_potential: the scalar potential differentiated here, used
            by the autodiff path via :func:`warp.grad`.
    """
    # U = -0.5 (p_i·p_j)^2
    c = wp.dot(p_i, p_j)
    grad_p_i = -c * p_j
    grad_p_j = -c * p_i
    return grad_p_i, grad_p_j


@wp.func
def mes_wnt_polarity_grads(p_i: wp.vec3f, u_to_higher_wnt: wp.vec3f, w_higher: wp.float32):
    r"""Analytic gradient of the WNT-aligned mesenchymal-polarity potential.

    Closed-form analytic polarity gradient of the term that turns a
    mesenchymal cell's polarity toward whichever neighbor carries higher
    activator/WNT concentration (the simulator's ``Mes-Align`` potential). Only
    the gradient with respect to :math:`\mathbf{p}_i` is returned; positions and
    concentrations are held fixed.

    .. math::
        U_{\mathrm{Mes\text{-}Align}} = -\tfrac{1}{2}\,c_{j,A}\,
        (\mathbf{p}_i^\top \hat{\mathbf{u}}_{ji})^2,\qquad
        \nabla_{\mathbf{p}_i}U = -c_{j,A}\,(\mathbf{p}_i^\top
        \hat{\mathbf{u}}_{ji})\,\hat{\mathbf{u}}_{ji}

    where :math:`\hat{\mathbf{u}}_{ji}` points toward the higher-WNT neighbor.

    Args:
        p_i: Unit polarity of mesenchymal cell :math:`i`.
        u_to_higher_wnt: Unit vector from :math:`i` toward the higher-WNT
            neighbor :math:`\hat{\mathbf{u}}_{ji}`.
        w_higher: Activator/WNT concentration of that neighbor
            :math:`c_{j,A}`, used as the per-pair weight.

    Returns:
        The gradient with respect to :math:`\mathbf{p}_i`.

    See Also:
        mes_wnt_polarity_potential: the scalar potential differentiated here,
            used by the autodiff path via :func:`warp.grad`.
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
    f_net: wp.array(dtype=wp.vec3f),  # net mechanical force on positions
    f_pol: wp.array(dtype=wp.vec3f),  # polarity gradient/torque accumulator
):
    """Accumulate mechanics and polarity gradients over all unordered pairs."""
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

        # Forces
        grad_x_i_f, grad_x_j_f = sticky_sphere_forces(x_i, x_j, r_i, r_j, c_i, c_j)
        wp.atomic_add(f_net, i, grad_x_i_f)
        wp.atomic_add(f_net, j, grad_x_j_f)

        # Polarity Neighbors
        dist = wp.norm_l2(x_i - x_j)
        w = adj_weight(dist, r_i, r_j)

        if not wp.bool(w):
            continue

        # Polarities - epithelium
        if (c_i == wp.uint32(1)) and (c_j == wp.uint32(1)):
            grad_x_i_p, grad_x_j_p, grad_p_i, grad_p_j = epi_polarity_grads(x_i, x_j, p_i, p_j)
            grad_x_i_t, grad_x_j_t = epi_thickness_grads(x_i, x_j, p_i, p_j)

            # # Match magnitudes so movement doesn't blink
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

        # Polarities - mesenchyme
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
    """Accumulate optional WNT-driven mesenchymal polarity gradients only."""
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
    X: wp.array(dtype=wp.vec3f),  # (N, 3)
    f_net: wp.array(dtype=wp.vec3f),  # (N, 3) net mechanical force on positions
    lr: wp.float32,
    X_next: wp.array(dtype=wp.vec3f),  # (N, 3)
):
    """Forward-Euler position update by descent on the prescribed potential.

    Integrates ``X_next = X - lr * f_net``. The accumulated ``f_net`` is treated
    here as a *descent gradient* of the prescribed potential, so the update
    subtracts it. This is the opposite integration sign from
    :func:`waxmorph.emulator._gd_update`, whose ``f_net`` holds a physical force
    (``-grad U``) and is therefore *added*.

    See Also:
        waxmorph.emulator._gd_update: the differentiable-path twin (``+`` sign).
    """

    i = wp.tid()
    X_next[i] = X[i] - lr * f_net[i]


@wp.kernel
def gd_update_normalized(
    P: wp.array(dtype=wp.vec3f),  # (N, 3)
    f_pol: wp.array(dtype=wp.vec3f),  # (N, 3) polarity gradient/torque accumulator
    lr: wp.float32,
    P_next: wp.array(dtype=wp.vec3f),  # (N, 3)
):
    """Euler polarity update with normalization back to unit-like vectors."""

    i = wp.tid()
    P_next[i] = wp.normalize(
        P[i] - lr * f_pol[i]
    )  # Normals pointing both inward / outward, but direction is used only


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
    """Advance positions and polarities by one prescribed-potential mechanics step.

    Accumulates the soft-sphere force and the cell-type-specific polarity/thickness
    potential gradients over all neighbor pairs, then applies forward-Euler
    updates to positions (descent on the potential) and to renormalized
    polarities. Uses analytically derived gradients; see
    :func:`mech_step_sticky_implicit` for the :func:`warp.grad` autodiff twin.

    Args:
        X: Position array with dtype ``wp.vec3f``, shape ``(N, 3)``.
        R: Radius array with dtype ``wp.float32``, shape ``(N,)``.
        P: Unit-polarity array with dtype ``wp.vec3f``, shape ``(N, 3)``.
        CT: Cell-type array with epithelial cells encoded as ``1``.
        particle_count: Number of active particles.
        dt: Mechanical Euler step :math:`\\Delta t_{\\mathrm{mech}}`.
        X_next: Output positions, shape ``(N, 3)``.
        P_next: Output renormalized polarities, shape ``(N, 3)``.
        device: Warp device.
        grad_consist: Emit gradient-consistency read/write marks for differentiable
            replay when ``True``.
        grid: Optional reusable :class:`warp.HashGrid`.
        wnt: Optional WNT/activator abundance array; when given, adds the
            WNT-aligned mesenchymal polarity gradients.

    Returns:
        The net position-force buffer ``f_net``, dtype ``wp.vec3f``, shape
        ``(N, 3)`` (a descent gradient; see :func:`gd_update`).

    See Also:
        mech_step_sticky_implicit: the autodiff (:func:`warp.grad`) twin.
        waxmorph.emulator: the differentiable, non-growing mechanics path.
    """

    # Set up gradients
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


############################################################
############################################################
############################################################

# MECHANICS - AUTODIFF

############################################################
############################################################
############################################################


@wp.func
def epi_polarity_potential(
    x_i: wp.vec3f,
    x_j: wp.vec3f,
    p_i: wp.vec3f,
    p_j: wp.vec3f,
) -> wp.float32:
    """Scalar potential enforcing epithelial polarity perpendicular to connection axis.

    .. math::
        U = \\tfrac{1}{2}(\\mathbf{p}_i \\cdot \\hat{\\mathbf{u}})^2
          + \\tfrac{1}{2}(\\mathbf{p}_j \\cdot \\hat{\\mathbf{u}})^2

    where :math:`\\hat{\\mathbf{u}} = (\\mathbf{x}_i - \\mathbf{x}_j) / \\|\\cdot\\|`.

    See Also:
        epi_polarity_grads: the analytic-gradient twin of this potential.
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
    """Scalar potential penalizing excessive normal-direction separation.

    .. math::
        U = \\tfrac{1}{2} K_{\\mathrm{thick}}
            \\max(|\\mathbf{d} \\cdot \\hat{\\mathbf{n}}| - H, 0)^2

    where :math:`\\hat{\\mathbf{n}} = (\\mathbf{p}_i + \\mathbf{p}_j') / \\|\\cdot\\|`
    (sign-corrected so polarities point the same way), :math:`K` =
    ``K_THICK_EE``, and :math:`H` = ``H_THICK_EE``.

    See Also:
        epi_thickness_grads: the analytic-gradient twin of this potential.
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
    """Scalar potential for mesenchymal polarity alignment.

    .. math::
        U = -\\tfrac{1}{2}(\\mathbf{p}_i \\cdot \\mathbf{p}_j)^2

    See Also:
        mes_polarity_grads: the analytic-gradient twin of this potential.
    """
    c = wp.dot(p_i, p_j)
    return wp.float32(-0.5) * c * c


@wp.func
def mes_wnt_polarity_potential(
    p_i: wp.vec3f,
    u_to_higher_wnt: wp.vec3f,
    w_higher: wp.float32,
) -> wp.float32:
    r"""Scalar WNT-aligned potential for mesenchymal polarity (torque only).

    Aligns a mesenchymal cell's polarity toward the neighbor of higher
    activator/WNT concentration (the simulator's ``Mes-Align`` term).

    .. math::
        U = -\tfrac{1}{2}\,c_{j,A}\,(\mathbf{p}_i^\top \hat{\mathbf{u}}_{ji})^2

    where :math:`\hat{\mathbf{u}}_{ji}` points toward the higher-WNT neighbor and
    :math:`c_{j,A}` (``w_higher``) is that neighbor's concentration.

    See Also:
        mes_wnt_polarity_grads: the analytic-gradient twin of this potential.
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
    f_net: wp.array(dtype=wp.vec3f),  # net mechanical force on positions
    f_pol: wp.array(dtype=wp.vec3f),  # polarity gradient/torque accumulator
):
    """Autodiff-based counterpart of ``sticky_sphere_grads`` using :func:`warp.grad`."""
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

        # Forces
        grad_x_i_f, grad_x_j_f = sticky_sphere_forces(x_i, x_j, r_i, r_j, c_i, c_j)
        wp.atomic_add(f_net, i, grad_x_i_f)
        wp.atomic_add(f_net, j, grad_x_j_f)

        # Polarity Neighbors
        dist = wp.norm_l2(x_i - x_j)
        w = adj_weight(dist, r_i, r_j)

        if not wp.bool(w):
            continue

        # Polarities - epithelium
        if (c_i == wp.uint32(1)) and (c_j == wp.uint32(1)):
            grad_x_i_p, grad_x_j_p, grad_p_i, grad_p_j = wp.grad(epi_polarity_potential)(
                x_i, x_j, p_i, p_j
            )
            grad_x_i_t, grad_x_j_t, _, _ = wp.grad(epi_thickness_potential)(x_i, x_j, p_i, p_j)

            # Match magnitudes so movement doesn't blink
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

        # Polarities - mesenchyme
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
    """Autodiff counterpart of ``sticky_sphere_wnt_grads``."""
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
    """Mechanics step matching ``mech_step_sticky`` but using :func:`warp.grad` locally."""

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


############################################################
############################################################
############################################################

# DIFFUSION

############################################################
############################################################
############################################################


@wp.kernel
def reaction_diffs(
    grid: wp.uint64,
    X: wp.array(dtype=wp.vec3f),  # (N, 3)
    R: wp.array(dtype=wp.float32),  # (N,)
    query_radius: wp.float32,
    A: wp.array(dtype=wp.float32),  # (N,)
    I: wp.array(dtype=wp.float32),  # (N,)
    lapA: wp.array(dtype=wp.float32),  # (N,) out (accum)
    lapI: wp.array(dtype=wp.float32),  # (N,) out (accum)
):
    """Accumulate graph-laplacian diffusion terms for A/I channels."""

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

        # Prune non neighbors
        if not wp.bool(w):
            continue

        Vj = volume_from_radius(Rj)
        cAj = safe_div(A[j], Vj)
        cIj = safe_div(I[j], Vj)

        # Pairwise flux contribution: w * (c_j - c_i)
        dA = wp.float32(w) * (cAj - cAi)
        dI = wp.float32(w) * (cIj - cIi)

        # Symmetric accumulation: +d to i, -d to j
        wp.atomic_add(lapA, i, dA)
        wp.atomic_add(lapA, j, -dA)
        wp.atomic_add(lapI, i, dI)
        wp.atomic_add(lapI, j, -dI)


@wp.kernel
def reaction_step(
    A: wp.array(dtype=wp.float32),  # (N,)
    I: wp.array(dtype=wp.float32),  # (N,)
    R: wp.array(dtype=wp.float32),
    lapA: wp.array(dtype=wp.float32),  # (N,)
    lapI: wp.array(dtype=wp.float32),  # (N,)
    chi: wp.array(dtype=wp.float32),  # chi: spatial characteristic of activator
    gamma: wp.array(dtype=wp.float32),  # gamma: reaction rate
    D_inhib: wp.float32,  # D_inhib: inhibitor diffusivity scaling
    dt: wp.float32,
    A_next: wp.array(dtype=wp.float32),  # (N,)
    I_next: wp.array(dtype=wp.float32),  # (N,)
):
    r"""Advance the activator-inhibitor (Turing) reaction-diffusion by one step.

    For cell :math:`i` the abundances evolve as

    .. math::
        \dot{A}_i = \gamma\Big[-\chi D_{\mathrm{inhib}}(L_G c_A)_i
        + \frac{c_{i,A}^2}{c_{i,I}} - c_{i,A}\Big],\qquad
        \dot{I}_i = \gamma\Big[-D_{\mathrm{inhib}}(L_G c_I)_i + c_{i,A}^2
        - c_{i,I}\Big]

    with :math:`\chi` = ``chi``, :math:`\gamma` = ``gamma``,
    :math:`D_{\mathrm{inhib}}` = ``D_inhib``, and :math:`L_G` the shared graph
    Laplacian, here supplied precomputed in ``lapA``/``lapI``.

    Two implementation choices stabilize the explicit integration:

    - **Production cap.** Activator self-production uses the smaller of the
      standard form :math:`c_{i,A}^2/c_{i,I}` and a quadratic-denominator variant
      :math:`c_{i,A}^2/c_{i,I}^2`, so that high inhibitor caps production and the
      field cannot blow up.
    - **Linear damping.** The linear decay term :math:`-c` is applied
      implicitly as the post-step factor :math:`1/(1 + \Delta t\,\gamma)`, which
      is unconditionally stable for that term.

    Both channels are finally clamped to ``[0, 1e4]`` to bound the explicit step.
    """

    i = wp.tid()

    V = volume_from_radius(R[i])

    # Reaction terms (on concentrations)
    cA = safe_div(A[i], V)
    cI = safe_div(I[i], V)

    cA2 = cA * cA
    ciI = cI

    # Activator self-production, saturating: take the smaller of the two
    # forms so high inhibitor caps production and the field cannot blow up.
    prodA_lin = safe_div(cA2, ciI)
    prodA_quad = safe_div(cA2, ciI * ciI)

    prodA = prodA_lin
    if wp.float32(prodA_quad) < wp.float32(prodA_lin):
        prodA = prodA_quad

    prodI = cA2

    # Diffusion with graph laplacian
    diffA = (chi[0] * D_inhib) * lapA[i]
    diffI = D_inhib * lapI[i]

    # Explicit Euler step with temporal scaling
    zAi = A[i] + dt * gamma[0] * (diffA + prodA)
    zIi = I[i] + dt * gamma[0] * (diffI + prodI)

    # Linear damping
    inv = 1.0 / (1.0 + dt * gamma[0])
    Ai_next = zAi * inv
    Ii_next = zIi * inv

    A_next[i] = wp.clamp(Ai_next, 0.0, 1e4)
    I_next[i] = wp.clamp(Ii_next, 0.0, 1e4)


@wp.kernel
def reaction_step_masked(
    A: wp.array(dtype=wp.float32),  # (N,)
    I: wp.array(dtype=wp.float32),  # (N,)
    R: wp.array(dtype=wp.float32),
    lapA: wp.array(dtype=wp.float32),  # (N,)
    lapI: wp.array(dtype=wp.float32),  # (N,)
    chi: wp.array(dtype=wp.float32),  # chi: spatial characteristic of activator
    gamma: wp.array(dtype=wp.float32),  # gamma: reaction rate
    D_inhib: wp.float32,  # D_inhib: inhibitor diffusivity scaling
    dt: wp.float32,
    CT: wp.array(dtype=wp.uint32),
    reaction_cell_type: wp.uint32,
    A_next: wp.array(dtype=wp.float32),  # (N,)
    I_next: wp.array(dtype=wp.float32),  # (N,)
):
    r"""Advance the activator-inhibitor (Turing) system on one masked cell type.

    Cell-type-restricted variant of :func:`reaction_step` for surface-patterning
    experiments: graph-Laplacian diffusion runs on every cell, but the
    nonlinear reaction terms of the activator-inhibitor system are applied only
    to cells whose type matches ``reaction_cell_type``. Non-reacting cells take a
    pure-diffusion update. The same production cap and the same implicit linear
    damping :math:`1/(1 + \Delta t\,\gamma)` as :func:`reaction_step` are used on
    the reacting cells, and both channels are clamped to ``[0, 1e4]``.

    See Also:
        reaction_step: the unmasked counterpart (reaction on all cells).
    """

    i = wp.tid()

    # Diffusion with graph laplacian. The reaction rate gamma is intentionally
    # kept consistent with reaction_step so masked and unmasked calls use the
    # same effective diffusion coefficient.
    diffA = (chi[0] * D_inhib) * lapA[i]
    diffI = D_inhib * lapI[i]

    if CT[i] != reaction_cell_type:
        Ai_next = A[i] + dt * gamma[0] * diffA
        Ii_next = I[i] + dt * gamma[0] * diffI
        A_next[i] = wp.clamp(Ai_next, 0.0, 1e4)
        I_next[i] = wp.clamp(Ii_next, 0.0, 1e4)
        return

    V = volume_from_radius(R[i])

    # Reaction terms (on concentrations)
    cA = safe_div(A[i], V)
    cI = safe_div(I[i], V)

    cA2 = cA * cA
    ciI = cI

    # Activator self-production, saturating: take the smaller of the two
    # forms so high inhibitor caps production and the field cannot blow up.
    prodA_lin = safe_div(cA2, ciI)
    prodA_quad = safe_div(cA2, ciI * ciI)

    prodA = prodA_lin
    if wp.float32(prodA_quad) < wp.float32(prodA_lin):
        prodA = prodA_quad

    prodI = cA2

    # Explicit Euler step with temporal scaling
    zAi = A[i] + dt * gamma[0] * (diffA + prodA)
    zIi = I[i] + dt * gamma[0] * (diffI + prodI)

    # Linear damping on reacting cells only.
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
    chi: wp.array,  # chi: spatial characteristic of activator
    gamma: wp.array,  # gamma: reaction rate
    D_inhib: float,  # D_inhib: inhibitor diffusivity scaling
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
    r"""Run one chemistry stage: graph-Laplacian diffusion then reaction update.

    First accumulates the activator/inhibitor graph Laplacians over
    hash-grid neighbors, then advances the activator-inhibitor (Turing) system by
    one explicit step via :func:`reaction_step` (or :func:`reaction_step_masked`
    when a single reacting cell type is requested).

    Args:
        A: Activator abundance array, shape ``(N,)``.
        I: Inhibitor abundance array, shape ``(N,)``.
        X: Position array with dtype ``wp.vec3f``, shape ``(N, 3)``.
        R: Radius array with dtype ``wp.float32``, shape ``(N,)``.
        lapA: Scratch/output activator Laplacian buffer, shape ``(N,)``.
        lapI: Scratch/output inhibitor Laplacian buffer, shape ``(N,)``.
        chi: Single-element array holding :math:`\chi`, the relative activator
            diffusivity (spatial characteristic).
        gamma: Single-element array holding the reaction rate :math:`\gamma`.
        D_inhib: Inhibitor diffusivity :math:`D_{\mathrm{inhib}}`.
        dt: Reaction-diffusion Euler step :math:`\Delta t_{\mathrm{chem}}`.
        particle_count: Number of active particles.
        A_next: Output activator abundances, shape ``(N,)``.
        I_next: Output inhibitor abundances, shape ``(N,)``.
        device: Warp device.
        grad_consist: Emit gradient-consistency read/write marks when ``True``.
        grid: Optional reusable :class:`warp.HashGrid`.
        CT: Cell-type array; required when ``reaction_cell_type`` is set.
        reaction_cell_type: If given, restrict the nonlinear reaction to this
            cell type (diffusion still runs everywhere); otherwise react on all
            cells.

    Returns:
        None. Results are written in place to ``A_next`` and ``I_next``.

    Raises:
        ValueError: If ``reaction_cell_type`` is set but ``CT`` is ``None``.

    See Also:
        reaction_step: the unmasked explicit reaction-diffusion update.
        reaction_step_masked: the cell-type-restricted reaction update.
    """

    if reaction_cell_type is not None and CT is None:
        raise ValueError("CT must be provided when reaction_cell_type is set")

    r_max = float(R.numpy()[:particle_count].max())
    query_radius = 2.0 * r_max + EPS_DIST
    if grid is None:
        grid = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    grid.build(X[:particle_count], query_radius)

    # Cache laplacian
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

    # Update molecules. By default this is the unmasked chemistry
    # path. The masked path is opt-in for surface-patterning experiments where
    # only one cell type should run the local reaction-diffusion updates
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
    grid: wp.uint64,
    X: wp.array(dtype=wp.vec3f),
    R: wp.array(dtype=wp.float32),
    CT: wp.array(dtype=wp.uint32),
    query_radius: wp.float32,
    n_tot: wp.array(dtype=wp.int32),
    n_epi: wp.array(dtype=wp.int32),
    n_mes: wp.array(dtype=wp.int32),
):
    """Count local neighborhood totals and type-specific neighbors."""
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
    """Count neighbors using :class:`warp.HashGrid` acceleration.

    Args:
        X: Position array with dtype ``wp.vec3f``.
        R: Radius array with dtype ``wp.float32``.
        CT: Cell-type array with epithelial cells encoded as ``1``.
        particle_count: Number of active particles.
        n_tot: Pre-zeroed output array for total neighbor counts.
        n_epi: Pre-zeroed output array for epithelial neighbor counts.
        n_mes: Pre-zeroed output array for mesenchymal neighbor counts.
        device: Warp device.
        grid: Optional reusable :class:`warp.HashGrid`.
    """
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
    alpha_grow: wp.array(dtype=wp.float32),  # alpha_grow: growth Hill exponent
    ell_sw: wp.array(dtype=wp.float32),  # ell_sw: switch concentration
    dt: float,
    R_ref: wp.float32,
    R_max: wp.float32,
    particle_count: int,
    R_next: wp.array(dtype=wp.float32),
    R_eq_next: wp.array(dtype=wp.float32),
    device: str = "cuda",
    grad_consist: bool = True,
) -> None:
    r"""Dispatch one activator-driven growth update for all particles.

    Launches :func:`growth_step_inner`, which advances the mesenchymal Hill
    growth rule for the equilibrium radius and relaxes the physical radius toward
    it:

    .. math::
        \dot{r}^{\mathrm{eq}}_i = \lambda_{\mathrm{ref},i}\,
        \frac{c_{i,A}^{\alpha_{\mathrm{grow}}}}
        {\ell_{\mathrm{sw}}^{\alpha_{\mathrm{grow}}}
        + c_{i,A}^{\alpha_{\mathrm{grow}}}},\qquad
        \dot{r}_i = \Big(1 - \frac{r_i}{r^{\mathrm{eq}}_i}\Big)^2

    with :math:`\lambda_{\mathrm{ref},i}\sim\mathcal{U}(0.8, 1)` drawn per step,
    :math:`\ell_{\mathrm{sw}}` = ``ell_sw``, and
    :math:`\alpha_{\mathrm{grow}}` = ``alpha_grow``. Epithelial cells
    instead relax deterministically toward ``R_ref``.

    Args:
        R: Physical-radius array, shape ``(N,)``.
        R_eq: Equilibrium (target) radius array, shape ``(N,)``.
        A: Activator abundance array, shape ``(N,)``.
        CT: Cell-type array with epithelial cells encoded as ``1``.
        keys: Per-particle RNG keys, advanced in place.
        alpha_grow: Single-element array holding the growth Hill exponent.
        ell_sw: Single-element array holding the growth Hill switch concentration.
        dt: Growth Euler step :math:`\Delta t_{\mathrm{grow}}`.
        R_ref: Epithelial reference radius for deterministic relaxation.
        R_max: Cap applied to the equilibrium radius before growth.
        particle_count: Number of active particles.
        R_next: Output physical radii, shape ``(N,)``.
        R_eq_next: Output equilibrium radii, shape ``(N,)``.
        device: Warp device.
        grad_consist: Emit gradient-consistency read/write marks when ``True``.

    Returns:
        None. Results are written in place to ``R_next`` and ``R_eq_next``.
    """
    wp.launch(
        growth_step_inner,
        dim=particle_count,
        inputs=[R, R_eq, A, CT, keys, alpha_grow, ell_sw, dt, R_ref, R_max, R_next, R_eq_next],
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
    alpha_grow: wp.array(dtype=wp.float32),  # alpha_grow: growth Hill exponent
    ell_sw: wp.array(dtype=wp.float32),  # ell_sw: switch concentration
    dt: wp.float32,
    R_ref: wp.float32,
    R_max: wp.float32,
    R_next: wp.array(dtype=wp.float32),
    R_eq_next: wp.array(dtype=wp.float32),
) -> None:
    """Per-particle growth rule conditioned on cell type and local chemistry."""
    i = wp.tid()

    # Mesenchyme: activator-driven growth of the target (equilibrium) radius.
    if CT[i] == wp.uint32(0):
        r_i_0, r_eq_i = R[i], wp.min(R_eq[i], R_max)

        V = volume_from_radius(R[i])
        num = safe_div(A[i], V) ** alpha_grow[0]

        key = keys[i]
        lam = wp.randf(key, 0.8, 1.0)  # per-step stochastic growth rate
        key = wp.randu(key)
        # Hill switch on activator concentration (half-max at ell_sw)
        frac = safe_div(num, (ell_sw[0] ** alpha_grow[0]) + num)

        R_eq_next[i] = r_eq_i + frac * lam * dt
        R_next[i] = r_i_0 + ((1.0 - safe_div(r_i_0, r_eq_i)) ** 2.0) * dt  # relax r -> r_eq
        keys[i] = key

    # Epithelium: deterministic relaxation toward the reference radius R_ref.
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
    """Straight-through Gumbel-Softmax Bernoulli sample with annealed temperature.

    Draws a hard 0/1 division decision from probability ``p`` while keeping a soft
    relaxed value for gradient flow (straight-through estimator). The softmax
    temperature ``tau`` is annealed geometrically from its initial value toward a
    floor of ``0.1`` over the horizon ``tmax`` (the rate ``k`` is set so the floor
    is reached at ``t == tmax``), sharpening samples toward true Bernoulli draws as
    the simulation progresses. The ``0.1`` floor keeps the relaxed softmax
    numerically well conditioned and prevents the temperature from collapsing to
    zero.
    """

    p = wp.clamp(p, RAND_EPS, 1.0 - RAND_EPS)

    # # Gumbel softmax
    key, g0 = gumbel(key)
    key, g1 = gumbel(key)

    # # Anneal temp toward the 0.1 floor (k chosen so the floor is hit at t=tmax)
    k = wp.log(tau / 0.1) / tmax
    tau = wp.max(0.1, tau * wp.exp(-k * t))

    p0 = (g1 + wp.log(1.0 - p)) / tau
    p1 = (g0 + wp.log(p)) / tau

    v = softmax2d(wp.vec2f(p0, p1))
    s = wp.dot(v, wp.vec2f(0.0, 1.0))

    s_straight = wp.int32(wp.argmax(v))

    return key, s_straight, s


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
    r"""Sample which parents divide and reserve daughter slots atomically.

    Mesenchymal cells divide with the size-driven Hill probability
    :math:`p = r^{\alpha_{\mathrm{div}}}/(r^{\alpha_{\mathrm{div}}}
    + r_{\mathrm{ref}}^{\alpha_{\mathrm{div}}})` (:func:`probs`). Epithelial cells
    do not undergo activator-driven growth; one is instead eligible to divide
    (probability ``p_epi``) only when it has at least one mesenchymal neighbor and
    fewer than ``epi_max_neighbors`` epithelial neighbors, which lets the
    monolayer expand as the enclosed mesenchyme grows. The hard 0/1 decision is
    drawn through :func:`st_gumbel_softmax_bernoulli`, and accepted parents claim a
    unique child slot via an atomic counter (capped at ``max_particles``).
    """
    parent = wp.tid()
    # Position is currently not used directly by this decision kernel.
    _ = X[parent]

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

    # Sample division (Gumbel-ST)
    key, s_hard, _s_soft = st_gumbel_softmax_bernoulli(p, key, t, tmax, tau)
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
    """Apply state transitions for accepted divisions (perpendicular-surface axis).

    Each accepted parent is replaced by two daughters of the same type and
    polarity, with abundances split equally and radii rescaled to conserve volume
    (mesenchyme; epithelium copies the parent radius). Daughters are separated
    along the surface perpendicular to the polarity.

    See Also:
        division_logic_mes_polarity: variant that separates mesenchymal daughters
            *along* the polarity axis instead.
    """

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

    # Daughter inherits the parent's cell type (no de novo / random type).
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
    # Place daughters on opposite sides of the parent center, separated by
    # slightly more than one daughter radius (1.02*r leaves a small gap so the
    # repulsive branch of the soft-sphere force does not immediately fire).
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
    """Apply divisions with mesenchymal daughters separated along polarity.

    Like :func:`division_logic` (same equal split of abundances, type and
    polarity inheritance, and volume-conserving radii), except mesenchymal
    daughters are placed *along* the parent polarity axis; epithelial daughters
    still split on the perpendicular surface to preserve the monolayer.

    See Also:
        division_logic: variant that always separates daughters on the
            perpendicular surface.
    """

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

    # Daughter inherits the parent's cell type (no de novo / random type).
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

    # Mesenchyme divides *along* the polarity axis; epithelium divides on the
    # perpendicular surface to keep the monolayer intact.
    u = wp.normalize(v)
    if ct == wp.uint32(1):
        # Polarized epithelial division remains on the perpendicular surface.
        a = wp.vec3f(1.0, 0.0, 0.0)
        if wp.abs(v[0]) > wp.float32(0.9):
            a = wp.vec3f(0.0, 1.0, 0.0)

        u = wp.normalize(wp.cross(v, a))

    x = X[parent]
    # Place daughters on opposite sides of the parent center, separated by
    # slightly more than one daughter radius (1.02*r leaves a small gap so the
    # repulsive branch of the soft-sphere force does not immediately fire).
    sep = 1.02 * r
    X[parent] = x + u * sep
    X[child] = x - u * sep
