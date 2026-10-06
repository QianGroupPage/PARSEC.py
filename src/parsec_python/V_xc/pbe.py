"""PBE exchange and correlation (spin-unpolarized, and collinear spin-polarized) on the real-space grid
(isolated clusters with zero padding, or periodic cells with wraparound).

PBE is a generalized-gradient approximation (GGA).  Its energy has the
discrete form

``E_xc[n] = h**3 * sum_i f(n_i, sigma_i)``,

where ``sigma_i = |D n|_i**2`` and ``D`` is the same-order centered first
derivative used on the active finite-difference grid.  The potential returned
here is the *discrete variational derivative* of that exact sum,

``v_xc = df/dn + D.T @ (2 * df/dsigma * D n)``.

Using the transpose of the discrete gradient is deliberate.  Merely inserting
the continuum expression ``-div(2*f_sigma*grad(n))`` with a separately chosen
divergence stencil need not be the derivative of the energy actually summed by
the code, especially beside the zero-density exterior of a finite cluster.

Densities are in electrons/bohr**3.  Pointwise helper values are evaluated in
Hartree atomic units, while :func:`pbe` converts energies and potentials to the
Rydberg convention used throughout this solver.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.ndimage import correlate1d

from ..Grid import RealSpaceGrid
from .ca_lda import SpinPolarizedXCResult, XCResult


# Original PBE constants.  GAMMA is (1-ln(2))/pi**2.
_KAPPA = 0.804
_MU = 0.2195149727645171
_BETA = 0.06672455060314922
_GAMMA = 0.031090690869654895
_CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
_KF_COEFFICIENT = (3.0 * np.pi * np.pi) ** (1.0 / 3.0)
_DENSITY_THRESHOLD = 1.0e-20


def first_derivative_coefficients(expansion_order: int) -> np.ndarray:
    """Return centered ``d/dx`` weights for a requested even order.

    With ``M=expansion_order/2`` and grid spacing ``h``,

    ``(D f)_i = h**(-1) * sum_j d_j f_(i+j) + O(h**(2*M))``,

    ``d_j = (-1)**(j+1) * (M!)**2 /
            (j * (M-j)! * (M+j)!)`` for ``j=1,...,M``,

    and ``d_-j=-d_j``.  The returned weights are dimensionless and ordered
    from ``-M`` through ``+M``.
    """

    expansion_order = int(expansion_order)
    if expansion_order < 2 or expansion_order > 20 or expansion_order % 2:
        raise ValueError("expansion_order must be an even integer from 2 to 20")
    width = expansion_order // 2
    weights = np.zeros(2 * width + 1, dtype=np.float64)
    for shell in range(1, width + 1):
        value = (
            (-1.0) ** (shell + 1)
            * factorial(width) ** 2
            / (
                shell
                * factorial(width - shell)
                * factorial(width + shell)
            )
        )
        weights[width + shell] = value
        weights[width - shell] = -value
    return weights


def _pw92_unpolarized(rs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return PW92 ``epsilon_c`` and ``d epsilon_c/d r_s`` in Hartree."""

    # Perdew-Wang 1992 unpolarized parameters (their compact Eq. 10 form).
    a = _GAMMA
    alpha1 = 0.21370
    beta1, beta2, beta3, beta4 = 7.5957, 3.5876, 1.6382, 0.49294
    sqrt_rs = np.sqrt(rs)
    polynomial = (
        beta1 * sqrt_rs
        + beta2 * rs
        + beta3 * rs * sqrt_rs
        + beta4 * rs * rs
    )
    q = 2.0 * a * polynomial
    log_term = np.log1p(1.0 / q)
    epsilon = -2.0 * a * (1.0 + alpha1 * rs) * log_term

    polynomial_prime = (
        0.5 * beta1 / sqrt_rs
        + beta2
        + 1.5 * beta3 * sqrt_rs
        + 2.0 * beta4 * rs
    )
    q_prime = 2.0 * a * polynomial_prime
    log_prime = -q_prime / (q * (q + 1.0))
    derivative = -2.0 * a * (
        alpha1 * log_term + (1.0 + alpha1 * rs) * log_prime
    )
    return epsilon, derivative


def pbe_energy_partials(
    density: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate unpolarized PBE ``f``, ``df/dn`` and ``df/dsigma``.

    ``f`` is the exchange-correlation energy per volume, not energy per
    electron.  All three returned quantities use Hartree atomic units.  The
    derivatives hold the other independent GGA variable fixed.  Values below
    a tiny density threshold follow the conventional zero-XC vacuum branch.
    """

    density = np.asarray(density, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if density.shape != sigma.shape:
        raise ValueError("PBE density and squared gradient must have the same shape")
    if np.any(density < -1.0e-14):
        raise ValueError("PBE requires a nonnegative density")
    if np.any(sigma < -1.0e-14):
        raise ValueError("PBE requires a nonnegative squared density gradient")

    energy_density = np.zeros_like(density)
    derivative_density = np.zeros_like(density)
    derivative_sigma = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    if not np.any(positive):
        return energy_density, derivative_density, derivative_sigma

    n = density[positive]
    sig = np.maximum(sigma[positive], 0.0)
    n13 = np.cbrt(n)
    n43 = n * n13
    kf = _KF_COEFFICIENT * n13

    # Exchange: f_x = -C_x*n^(4/3)*F_x(s), with
    # s^2 = sigma/[4*k_F^2*n^2].
    s2 = sig / (4.0 * kf * kf * n * n)
    exchange_denominator = 1.0 + (_MU / _KAPPA) * s2
    enhancement = 1.0 + _KAPPA - _KAPPA / exchange_denominator
    enhancement_prime = _MU / (exchange_denominator * exchange_denominator)
    exchange_energy = -_CX * n43 * enhancement
    exchange_dn = -_CX * n13 * (
        (4.0 / 3.0) * enhancement
        - (8.0 / 3.0) * s2 * enhancement_prime
    )
    ds2_dsigma = 1.0 / (4.0 * kf * kf * n * n)
    exchange_dsigma = -_CX * n43 * enhancement_prime * ds2_dsigma

    # Correlation: PW92 LDA plus the PBE gradient correction H.
    rs = np.cbrt(3.0 / (4.0 * np.pi * n))
    epsilon_lda, epsilon_lda_rs = _pw92_unpolarized(rs)
    epsilon_lda_n = -rs * epsilon_lda_rs / (3.0 * n)

    beta_over_gamma = _BETA / _GAMMA
    exponential_minus_one = np.expm1(-epsilon_lda / _GAMMA)
    a_parameter = beta_over_gamma / exponential_minus_one
    a_epsilon = (
        beta_over_gamma
        * np.exp(-epsilon_lda / _GAMMA)
        / (_GAMMA * exponential_minus_one * exponential_minus_one)
    )

    # t^2 = |grad n|^2/(4*k_s^2*n^2), k_s^2=4*k_F/pi.
    dt2_dsigma = np.pi / (16.0 * kf * n * n)
    t2 = sig * dt2_dsigma
    at = a_parameter * t2
    numerator = beta_over_gamma * t2 * (1.0 + at)
    denominator = 1.0 + at + at * at
    q = numerator / denominator
    h = _GAMMA * np.log1p(q)

    numerator_t = beta_over_gamma * (1.0 + 2.0 * at)
    denominator_t = a_parameter + 2.0 * a_parameter * at
    q_t = (
        numerator_t * denominator - numerator * denominator_t
    ) / (denominator * denominator)
    numerator_a = beta_over_gamma * t2 * t2
    denominator_a = t2 + 2.0 * a_parameter * t2 * t2
    q_a = (
        numerator_a * denominator - numerator * denominator_a
    ) / (denominator * denominator)
    h_q = _GAMMA / (1.0 + q)
    h_t = h_q * q_t
    h_a = h_q * q_a

    t2_n = -(7.0 / 3.0) * t2 / n
    h_n = h_t * t2_n + h_a * a_epsilon * epsilon_lda_n
    h_sigma = h_t * dt2_dsigma
    correlation_energy = n * (epsilon_lda + h)
    correlation_dn = epsilon_lda + h + n * (epsilon_lda_n + h_n)
    correlation_dsigma = n * h_sigma

    energy_density[positive] = exchange_energy + correlation_energy
    derivative_density[positive] = exchange_dn + correlation_dn
    derivative_sigma[positive] = exchange_dsigma + correlation_dsigma
    return energy_density, derivative_density, derivative_sigma


def _is_periodic(grid) -> bool:
    return hasattr(grid, "cell") and hasattr(grid, "points_per_axis")


def _to_box(values: np.ndarray, grid) -> np.ndarray:
    """Cartesian-box view of a grid vector.

    A periodic grid already stores every point of its orthorhombic cell in
    row-major order, so the box is a plain reshape and derivatives wrap
    around.  An isolated cluster grid is embedded in a zero-valued box.
    """

    if _is_periodic(grid):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (grid.size,):
            raise ValueError("field does not match the periodic real-space grid")
        return values.reshape(grid.shape)
    return _embed_active(values, grid)


def _from_box(box: np.ndarray, grid) -> np.ndarray:
    if _is_periodic(grid):
        return np.ascontiguousarray(box.reshape(-1))
    return _gather_active(box, grid)


def _embed_active(values: np.ndarray, grid: RealSpaceGrid) -> np.ndarray:
    """Embed one active-grid vector into its zero-valued Cartesian box."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape != (grid.size,):
        raise ValueError("field does not match the active real-space grid")
    full = np.zeros(grid.shape, dtype=np.float64)
    local = grid.integer_coordinates - grid.index_min
    full[local[:, 0], local[:, 1], local[:, 2]] = values
    return full


def _gather_active(values: np.ndarray, grid: RealSpaceGrid) -> np.ndarray:
    local = grid.integer_coordinates - grid.index_min
    return np.ascontiguousarray(values[local[:, 0], local[:, 1], local[:, 2]])


def pbe(
    valence_density: np.ndarray,
    grid: RealSpaceGrid,
    core_density: np.ndarray | None = None,
) -> XCResult:
    """Evaluate unpolarized PBE and its exact discrete grid derivative.

    On an isolated cluster grid the density is extended by zero outside the
    active domain, matching the orbital finite-domain convention.  Centered
    derivatives retain their full stencil beside the boundary; missing samples
    therefore contribute zero instead of triggering a one-sided or
    renormalized formula.  On a periodic grid (``PeriodicRealSpaceGrid``) the
    stencil wraps around the cell instead, so the same code serves bulk
    calculations; wraparound makes the discrete gradient an exactly
    antisymmetric circulant operator, so the transpose used for the
    potential is still the exact derivative of the summed energy.
    """

    valence = np.asarray(valence_density, dtype=np.float64)
    if valence.shape != (grid.size,):
        raise ValueError("PBE density does not match the active real-space grid")
    if core_density is None:
        density = valence
    else:
        core = np.asarray(core_density, dtype=np.float64)
        if core.shape != valence.shape:
            raise ValueError("core and valence densities must have the same shape")
        density = valence + core
    if np.any(density < -1.0e-14):
        raise ValueError("PBE requires a nonnegative density")

    weights = first_derivative_coefficients(grid.settings.expansion_order)
    weights = weights / grid.spacing
    # Periodic cells wrap the stencil around; clusters zero-pad outside.
    boundary = "wrap" if _is_periodic(grid) else "constant"
    full_density = _to_box(density, grid)
    gradients = np.empty((3, grid.size), dtype=np.float64)
    sigma = np.zeros(grid.size, dtype=np.float64)
    for axis in range(3):
        derivative_box = correlate1d(
            full_density,
            weights,
            axis=axis,
            mode=boundary,
            cval=0.0,
        )
        gradient = _from_box(derivative_box, grid)
        gradients[axis] = gradient
        sigma += gradient * gradient

    energy_hartree, fn_hartree, fsigma_hartree = pbe_energy_partials(
        density, sigma
    )
    potential_hartree = fn_hartree.copy()
    for axis in range(3):
        flux_box = _to_box(2.0 * fsigma_hartree * gradients[axis], grid)
        # For the antisymmetric centered stencil D.T = -D.  Applying the
        # reversed weights is the exact transpose of the zero-padded action.
        adjoint_box = correlate1d(
            flux_box,
            -weights,
            axis=axis,
            mode=boundary,
            cval=0.0,
        )
        potential_hartree += _from_box(adjoint_box, grid)

    # Convert Hartree -> Rydberg only at the public boundary.
    energy_density = 2.0 * energy_hartree
    potential = 2.0 * potential_hartree
    epsilon = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    epsilon[positive] = energy_density[positive] / density[positive]
    return XCResult(
        potential=potential,
        energy_per_electron=epsilon,
        energy_density=energy_density,
        total_energy=float(grid.volume_element * np.sum(energy_density)),
    )


# ---------------------------------------------------------------------------
# Spin-polarized (collinear) PBE
# ---------------------------------------------------------------------------

# PW92 parameter sets (A, alpha1, beta1..beta4): eps_c(rs, zeta=0),
# eps_c(rs, zeta=1), and minus the spin stiffness alpha_c(rs).  The
# unpolarized set reuses _GAMMA as A so that the zeta=0 limit reproduces
# :func:`pbe_energy_partials` to round-off.
_PW92_SETS = {
    "unpolarized": (_GAMMA, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    "polarized": (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    "stiffness": (0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
}
_FZ_SECOND_DERIVATIVE = 1.709921
_FZ_DENOMINATOR = 2.0 ** (4.0 / 3.0) - 2.0
_COMPLEX_STEP = 1.0e-30
_SPIN_DENSITY_FLOOR = 1.0e-16


def _pw92_g(rs, parameters):
    """PW92 ``G(rs)`` (their Eq. 10); safe for complex-step arguments."""

    a, alpha1, beta1, beta2, beta3, beta4 = parameters
    sqrt_rs = rs ** 0.5
    polynomial = (
        beta1 * sqrt_rs + beta2 * rs + beta3 * rs * sqrt_rs + beta4 * rs * rs
    )
    return -2.0 * a * (1.0 + alpha1 * rs) * np.log1p(1.0 / (2.0 * a * polynomial))


def _pbe_spin_energy_density(n_up, n_down, s_uu, s_ud, s_dd):
    """Spin-polarized PBE energy per volume in Hartree.

    ``s_uu, s_ud, s_dd`` are ``grad n_a . grad n_b``.  Exchange follows the
    exact spin-scaling relation ``E_x[n_up, n_down] = (E_x[2 n_up] +
    E_x[2 n_down])/2``; correlation is PW92 ``eps_c(rs, zeta)`` plus the PBE
    gradient term ``H(rs, zeta, t)`` with the ``phi(zeta)`` spin scaling.
    Written with operations that are analytic in complex arguments so that
    the partial derivatives can be taken by complex-step differentiation.
    """

    total = n_up + n_down

    exchange = 0.0
    for density, sigma in ((n_up, s_uu), (n_down, s_dd)):
        n2 = 2.0 * density
        n13 = n2 ** (1.0 / 3.0)
        kf = _KF_COEFFICIENT * n13
        s2 = (4.0 * sigma) / (4.0 * kf * kf * n2 * n2)
        enhancement = 1.0 + _KAPPA - _KAPPA / (1.0 + (_MU / _KAPPA) * s2)
        exchange = exchange + 0.5 * (-_CX * n2 * n13 * enhancement)

    rs = (3.0 / (4.0 * np.pi * total)) ** (1.0 / 3.0)
    zeta = (n_up - n_down) / total
    f_zeta = (
        (1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0) - 2.0
    ) / _FZ_DENOMINATOR
    zeta4 = zeta ** 4
    epsilon_0 = _pw92_g(rs, _PW92_SETS["unpolarized"])
    epsilon_1 = _pw92_g(rs, _PW92_SETS["polarized"])
    alpha_c = -_pw92_g(rs, _PW92_SETS["stiffness"])
    epsilon_lda = (
        epsilon_0
        + alpha_c * f_zeta / _FZ_SECOND_DERIVATIVE * (1.0 - zeta4)
        + (epsilon_1 - epsilon_0) * f_zeta * zeta4
    )

    phi = 0.5 * ((1.0 + zeta) ** (2.0 / 3.0) + (1.0 - zeta) ** (2.0 / 3.0))
    phi3 = phi * phi * phi
    kf_total = _KF_COEFFICIENT * total ** (1.0 / 3.0)
    ks2 = 4.0 * kf_total / np.pi
    sigma_total = s_uu + 2.0 * s_ud + s_dd
    t2 = sigma_total / (4.0 * phi * phi * ks2 * total * total)
    beta_over_gamma = _BETA / _GAMMA
    a_parameter = beta_over_gamma / np.expm1(-epsilon_lda / (_GAMMA * phi3))
    at = a_parameter * t2
    q = beta_over_gamma * t2 * (1.0 + at) / (1.0 + at + at * at)
    h = _GAMMA * phi3 * np.log1p(q)
    return exchange + total * (epsilon_lda + h)


def pbe_spin_energy_partials(n_up, n_down, s_uu, s_ud, s_dd):
    """Spin-polarized PBE ``f`` and its five partial derivatives (Hartree).

    Returns ``(f, df/dn_up, df/dn_down, df/ds_uu, df/ds_ud, df/ds_dd)`` with
    ``s_ab = grad n_a . grad n_b``.  The derivatives are exact to machine
    precision (complex-step differentiation of the analytic energy), not
    finite differences.  Densities are clamped to a tiny positive floor;
    entries whose total density is below the usual PBE vacuum threshold
    return zeros.
    """

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (n_up, n_down, s_uu, s_ud, s_dd)
    ]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("PBE spin densities and gradients must share one shape")
    if np.any(arrays[0] < -1.0e-14) or np.any(arrays[1] < -1.0e-14):
        raise ValueError("PBE requires nonnegative spin densities")
    if np.any(arrays[2] < -1.0e-14) or np.any(arrays[4] < -1.0e-14):
        raise ValueError("PBE requires nonnegative squared spin-density gradients")

    outputs = [np.zeros(shape) for _ in range(6)]
    active = (arrays[0] + arrays[1]) > _DENSITY_THRESHOLD
    if not np.any(active):
        return tuple(outputs)

    n_u = np.maximum(arrays[0][active], _SPIN_DENSITY_FLOOR)
    n_d = np.maximum(arrays[1][active], _SPIN_DENSITY_FLOOR)
    suu = np.maximum(arrays[2][active], 0.0)
    sud = arrays[3][active]
    sdd = np.maximum(arrays[4][active], 0.0)
    base = (n_u, n_d, suu, sud, sdd)

    outputs[0][active] = _pbe_spin_energy_density(*base)
    for index in range(5):
        perturbed = [value.astype(np.complex128) for value in base]
        perturbed[index] = perturbed[index] + 1j * _COMPLEX_STEP
        outputs[index + 1][active] = (
            _pbe_spin_energy_density(*perturbed).imag / _COMPLEX_STEP
        )
    return tuple(outputs)


def pbe_spin_polarized(
    density_up: np.ndarray,
    density_down: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
) -> SpinPolarizedXCResult:
    """Collinear spin-polarized PBE and its exact discrete grid derivative.

    Uses the same centered stencil, zero-padded (cluster) or wraparound
    (periodic) boundary treatment and exact-transpose potential construction
    as :func:`pbe`.  With ``s_ab = D n_a . D n_b`` the potential of spin
    channel ``a`` is

    ``v_a = df/dn_a + D.T @ (2 f_aa D n_a + f_ab D n_b)``,

    the derivative of the summed energy ``h**3 * sum_i f``.  A frozen NLCC
    ``core_density`` is split evenly between the channels, as in the LSDA
    branch (it enters the gradients too).  Results are in Rydberg.
    """

    up = np.asarray(density_up, dtype=np.float64)
    down = np.asarray(density_down, dtype=np.float64)
    if up.shape != (grid.size,) or down.shape != (grid.size,):
        raise ValueError("PBE spin densities do not match the real-space grid")
    if core_density is not None:
        core = np.asarray(core_density, dtype=np.float64)
        if core.shape != up.shape:
            raise ValueError("core and valence densities must have the same shape")
        up = up + 0.5 * core
        down = down + 0.5 * core
    if np.any(up < -1.0e-14) or np.any(down < -1.0e-14):
        raise ValueError("PBE requires nonnegative spin densities")

    weights = first_derivative_coefficients(grid.settings.expansion_order)
    weights = weights / grid.spacing
    boundary = "wrap" if _is_periodic(grid) else "constant"

    def gradient(field):
        box = _to_box(field, grid)
        return [
            _from_box(
                correlate1d(box, weights, axis=axis, mode=boundary, cval=0.0),
                grid,
            )
            for axis in range(3)
        ]

    grad_up = gradient(up)
    grad_down = gradient(down)
    s_uu = sum(g * g for g in grad_up)
    s_dd = sum(g * g for g in grad_down)
    s_ud = sum(a * b for a, b in zip(grad_up, grad_down))

    f, f_up, f_down, f_uu, f_ud, f_dd = pbe_spin_energy_partials(
        up, down, s_uu, s_ud, s_dd
    )

    def divergence_adjoint(fluxes):
        total = np.zeros(grid.size, dtype=np.float64)
        for axis in range(3):
            adjoint_box = correlate1d(
                _to_box(fluxes[axis], grid),
                -weights,
                axis=axis,
                mode=boundary,
                cval=0.0,
            )
            total += _from_box(adjoint_box, grid)
        return total

    potential_up = f_up + divergence_adjoint(
        [2.0 * f_uu * gu + f_ud * gd for gu, gd in zip(grad_up, grad_down)]
    )
    potential_down = f_down + divergence_adjoint(
        [2.0 * f_dd * gd + f_ud * gu for gu, gd in zip(grad_up, grad_down)]
    )

    energy_density = 2.0 * f
    return SpinPolarizedXCResult(
        potential_up=2.0 * potential_up,
        potential_down=2.0 * potential_down,
        energy_density=energy_density,
        total_energy=float(grid.volume_element * np.sum(energy_density)),
    )


__all__ = [
    "first_derivative_coefficients",
    "pbe",
    "pbe_energy_partials",
    "pbe_spin_energy_partials",
    "pbe_spin_polarized",
]
