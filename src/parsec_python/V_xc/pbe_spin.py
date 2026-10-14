"""Spin-polarized PBE exchange and correlation on the real-space grid.

The energy is the discrete sum

``E_xc = h**3 * sum_i f(n_up, n_down, sigma_uu, sigma_dd, sigma)_i``

with ``sigma_ss = |D n_s|**2`` for each spin channel, ``sigma = |D n|**2``
for the total density ``n = n_up + n_down``, and ``D`` the centered first
derivative of :mod:`parsec_python.V_xc.pbe`.  Exchange obeys the exact spin
scaling

``E_x[n_up, n_down] = (E_x[2 n_up] + E_x[2 n_down]) / 2``,

so it reads ``n_s`` and ``sigma_ss`` only.  Correlation is the Perdew-Wang
energy of the spin-polarized uniform gas plus the gradient correction

``H = gamma phi**3 ln[1 + (beta/gamma) t**2 (1 + A t**2)
                         / (1 + A t**2 + A**2 t**4)]``,

``phi = [(1+zeta)**(2/3) + (1-zeta)**(2/3)] / 2``,
``t**2 = sigma / (4 phi**2 k_s**2 n**2)``,
``A = (beta/gamma) / [exp(-e_c/(gamma phi**3)) - 1]``,

and reads ``n``, ``zeta = (n_up - n_down)/n`` and ``sigma``.

As for the unpolarized functional, the potentials returned here are the
derivatives of that discrete sum,

``v_s = df/dn_s + D.T @ (2 df/dsigma_ss D n_s + 2 df/dsigma D n)``,

and not the continuum expression with second derivatives of the density
that ``exc_spn.f90`` evaluates through ``spn_pbe.f90``.  For equal spin
densities both potentials are the unpolarized one of
:func:`parsec_python.V_xc.pbe.pbe`.

Two more points differ from ``spn_pbe.f90``.  The Fortran sets the exchange
energy of a grid point to zero where either spin density is below 1e-18,
which removes the whole exchange energy of a channel-less state such as the
hydrogen atom; here each channel contributes wherever its own density is
above the threshold of the unpolarized functional.  And the unpolarized
Perdew-Wang parameter ``A`` has the digits of the unpolarized Python
functional, ``(1 - ln 2)/pi**2``, where the Fortran has 0.0310907; the two
parameters that only a polarized density reads are those of the Fortran.

Each spin channel sees its own density plus half of the frozen core density
of a nonlinear core correction.  Pointwise helpers work in Hartree atomic
units; :func:`pbe_spin` returns Rydberg.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import correlate1d

from ..Grid import RealSpaceGrid
from .ca_lda_spin import SpinXCResult, spin_densities_with_core
from .pbe import (
    _BETA,
    _DENSITY_THRESHOLD,
    _GAMMA,
    _KF_COEFFICIENT,
    _embed_active,
    _exchange_partials,
    _gather_active,
    _pw92_g,
    _pw92_unpolarized,
    first_derivative_coefficients,
)


# 2**(4/3) - 2 and f''(0) = 8/(9 (2**(4/3) - 2)) of the spin interpolation,
# with the digits of spn_corpbe.
_SPIN_GAM = 0.5198420997897463295344212145565
_SPIN_FZZ = 8.0 / (9.0 * _SPIN_GAM)
# spn_corpbe keeps d phi/d zeta finite at |zeta| = 1 with this number.
_SPIN_ETA = 1.0e-12


def pw92_spin_correlation(
    rs: np.ndarray, zeta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perdew-Wang correlation of the spin-polarized uniform gas, in Hartree.

    Returns the energy per electron ``e_c(r_s, zeta)`` and its derivatives
    with respect to ``r_s`` and ``zeta`` (their Eqs. 8, A2 and A3):

    ``e_c = e_u + alpha_c f(zeta) (1 - zeta**4) / f''(0)
            + (e_p - e_u) f(zeta) zeta**4``.
    """

    unpolarized, unpolarized_rs = _pw92_unpolarized(rs)
    polarized, polarized_rs = _pw92_g(
        rs, 0.01554535, 0.20548, (14.1189, 6.1977, 3.3662, 0.62517)
    )
    # Minus the spin stiffness alpha_c.
    stiffness, stiffness_rs = _pw92_g(
        rs, 0.0168869, 0.11125, (10.357, 3.6231, 0.88026, 0.49671)
    )
    plus = np.maximum(1.0 + zeta, 0.0)
    minus = np.maximum(1.0 - zeta, 0.0)
    z4 = zeta**4
    f = (plus ** (4.0 / 3.0) + minus ** (4.0 / 3.0) - 2.0) / _SPIN_GAM
    energy = (
        unpolarized * (1.0 - f * z4)
        + polarized * f * z4
        - stiffness * f * (1.0 - z4) / _SPIN_FZZ
    )
    energy_rs = (
        unpolarized_rs * (1.0 - f * z4)
        + polarized_rs * f * z4
        - stiffness_rs * f * (1.0 - z4) / _SPIN_FZZ
    )
    f_zeta = (4.0 / 3.0) * (np.cbrt(plus) - np.cbrt(minus)) / _SPIN_GAM
    energy_zeta = 4.0 * zeta**3 * f * (
        polarized - unpolarized + stiffness / _SPIN_FZZ
    ) + f_zeta * (
        z4 * polarized - z4 * unpolarized - (1.0 - z4) * stiffness / _SPIN_FZZ
    )
    return energy, energy_rs, energy_zeta


def pbe_spin_energy_partials(
    density_up: np.ndarray,
    density_down: np.ndarray,
    sigma_up: np.ndarray,
    sigma_down: np.ndarray,
    sigma_total: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Evaluate spin-polarized PBE ``f`` and its five partial derivatives.

    Returns ``f``, ``df/dn_up``, ``df/dn_down``, ``df/dsigma_uu``,
    ``df/dsigma_dd`` and ``df/dsigma`` in Hartree atomic units, each
    derivative with the other four variables fixed.  ``sigma_up`` and
    ``sigma_down`` are the squared gradients of the two spin densities and
    ``sigma_total`` that of their sum.  ``f`` is energy per volume.
    """

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (density_up, density_down, sigma_up, sigma_down, sigma_total)
    ]
    up, down, sig_up, sig_down, sig_total = arrays
    if any(value.shape != up.shape for value in arrays):
        raise ValueError("PBE spin densities and squared gradients must share a shape")
    if np.any(up < -1.0e-14) or np.any(down < -1.0e-14):
        raise ValueError("PBE requires nonnegative spin densities")
    if any(np.any(value < -1.0e-14) for value in (sig_up, sig_down, sig_total)):
        raise ValueError("PBE requires nonnegative squared density gradients")
    up = np.maximum(up, 0.0)
    down = np.maximum(down, 0.0)

    energy = np.zeros_like(up)
    d_up = np.zeros_like(up)
    d_down = np.zeros_like(up)
    d_sigma_up = np.zeros_like(up)
    d_sigma_down = np.zeros_like(up)
    d_sigma_total = np.zeros_like(up)

    # Exchange: each channel is an unpolarized density 2 n_s with squared
    # gradient 4 sigma_ss, and carries half of that energy.
    for spin_density, spin_sigma, d_density, d_sigma in (
        (up, sig_up, d_up, d_sigma_up),
        (down, sig_down, d_down, d_sigma_down),
    ):
        doubled = 2.0 * spin_density
        occupied = doubled > _DENSITY_THRESHOLD
        if not np.any(occupied):
            continue
        exchange, exchange_dn, exchange_dsigma = _exchange_partials(
            doubled[occupied], np.maximum(4.0 * spin_sigma[occupied], 0.0)
        )
        energy[occupied] += 0.5 * exchange
        d_density[occupied] += exchange_dn
        d_sigma[occupied] += 2.0 * exchange_dsigma

    density = up + down
    positive = density > _DENSITY_THRESHOLD
    if not np.any(positive):
        return energy, d_up, d_down, d_sigma_up, d_sigma_down, d_sigma_total

    n = density[positive]
    sig = np.maximum(sig_total[positive], 0.0)
    zeta = (up[positive] - down[positive]) / n
    kf = _KF_COEFFICIENT * np.cbrt(n)
    rs = np.cbrt(3.0 / (4.0 * np.pi * n))
    epsilon, epsilon_rs, epsilon_zeta = pw92_spin_correlation(rs, zeta)
    epsilon_n = -rs * epsilon_rs / (3.0 * n)

    plus = np.maximum(1.0 + zeta, 0.0)
    minus = np.maximum(1.0 - zeta, 0.0)
    phi = 0.5 * (plus ** (2.0 / 3.0) + minus ** (2.0 / 3.0))
    phi_zeta = (
        ((1.0 + zeta) ** 2 + _SPIN_ETA) ** (-1.0 / 6.0)
        - ((1.0 - zeta) ** 2 + _SPIN_ETA) ** (-1.0 / 6.0)
    ) / 3.0
    phi3 = phi * phi * phi

    beta_over_gamma = _BETA / _GAMMA
    exponent = -epsilon / (_GAMMA * phi3)
    exponential_minus_one = np.expm1(exponent)
    a_parameter = beta_over_gamma / exponential_minus_one
    # dA/d(exponent), then the two variables the exponent depends on.
    a_exponent = (
        -beta_over_gamma
        * np.exp(exponent)
        / (exponential_minus_one * exponential_minus_one)
    )
    a_epsilon = -a_exponent / (_GAMMA * phi3)
    a_phi = a_exponent * 3.0 * epsilon / (_GAMMA * phi3 * phi)

    # t^2 = sigma/(4*phi^2*k_s^2*n^2), k_s^2 = 4*k_F/pi.
    dt2_dsigma = np.pi / (16.0 * phi * phi * kf * n * n)
    t2 = sig * dt2_dsigma
    at = a_parameter * t2
    numerator = beta_over_gamma * t2 * (1.0 + at)
    denominator = 1.0 + at + at * at
    q = numerator / denominator
    log_term = np.log1p(q)
    h = _GAMMA * phi3 * log_term

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
    h_q = _GAMMA * phi3 / (1.0 + q)
    h_t = h_q * q_t
    h_a = h_q * q_a
    # H depends on phi through its prefactor, through A and through t^2.
    h_phi = (
        3.0 * _GAMMA * phi * phi * log_term
        + h_a * a_phi
        - h_t * 2.0 * t2 / phi
    )

    t2_n = -(7.0 / 3.0) * t2 / n
    # Derivatives of e_c + H at fixed zeta and at fixed n.
    per_electron = epsilon + h
    per_electron_n = epsilon_n + h_t * t2_n + h_a * a_epsilon * epsilon_n
    per_electron_zeta = (
        epsilon_zeta + h_a * a_epsilon * epsilon_zeta + h_phi * phi_zeta
    )
    common = per_electron + n * per_electron_n

    energy[positive] += n * per_electron
    d_up[positive] += common + (1.0 - zeta) * per_electron_zeta
    d_down[positive] += common - (1.0 + zeta) * per_electron_zeta
    d_sigma_total[positive] = n * h_t * dt2_dsigma
    return energy, d_up, d_down, d_sigma_up, d_sigma_down, d_sigma_total


def pbe_spin(
    valence_up: np.ndarray,
    valence_down: np.ndarray,
    grid: RealSpaceGrid,
    core_density: np.ndarray | None = None,
) -> SpinXCResult:
    """Evaluate spin-polarized PBE and its exact discrete grid derivatives.

    The densities are extended by zero outside the active cluster and the
    centered derivatives keep their full stencil beside the boundary, as in
    :func:`parsec_python.V_xc.pbe.pbe`.
    """

    up, down = spin_densities_with_core(
        valence_up, valence_down, core_density, functional="PBE"
    )
    if up.shape != (grid.size,):
        raise ValueError("PBE densities do not match the active real-space grid")

    weights = first_derivative_coefficients(grid.settings.expansion_order)
    weights = weights / grid.spacing

    def gradient(values: np.ndarray) -> np.ndarray:
        box = _embed_active(values, grid)
        components = np.empty((3, grid.size), dtype=np.float64)
        for axis in range(3):
            components[axis] = _gather_active(
                correlate1d(box, weights, axis=axis, mode="constant", cval=0.0),
                grid,
            )
        return components

    gradient_up = gradient(up)
    gradient_down = gradient(down)
    gradient_total = gradient_up + gradient_down
    (
        energy_hartree,
        f_up,
        f_down,
        f_sigma_up,
        f_sigma_down,
        f_sigma_total,
    ) = pbe_spin_energy_partials(
        up,
        down,
        np.einsum("ij,ij->j", gradient_up, gradient_up),
        np.einsum("ij,ij->j", gradient_down, gradient_down),
        np.einsum("ij,ij->j", gradient_total, gradient_total),
    )

    def potential(f_density, f_sigma, spin_gradient) -> np.ndarray:
        values = f_density.copy()
        for axis in range(3):
            flux = _embed_active(
                2.0 * f_sigma * spin_gradient[axis]
                + 2.0 * f_sigma_total * gradient_total[axis],
                grid,
            )
            # D.T = -D for the antisymmetric centered stencil.
            values += _gather_active(
                correlate1d(flux, -weights, axis=axis, mode="constant", cval=0.0),
                grid,
            )
        return values

    # Hartree -> Rydberg at the public boundary.
    energy_density = 2.0 * energy_hartree
    return SpinXCResult(
        potential_up=2.0 * potential(f_up, f_sigma_up, gradient_up),
        potential_down=2.0 * potential(f_down, f_sigma_down, gradient_down),
        energy_density=energy_density,
        total_energy=float(grid.volume_element * np.sum(energy_density)),
    )


__all__ = ["pbe_spin", "pbe_spin_energy_partials", "pw92_spin_correlation"]
