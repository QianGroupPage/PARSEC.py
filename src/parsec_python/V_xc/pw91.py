"""PW91 generalized-gradient exchange and correlation, PARSEC's ``pw``.

This is the ``Correlation_Type = pw`` (``pw91``, ``pwgga``) functional of
``exc_nspn.f90`` (lines 182-193, ``pw91`` of ``pbe.f90``) and ``exc_spn.f90``
(lines 286-288, ``spn_pbe.f90``) [J. P. Perdew et al., Phys. Rev. B 46, 6671
(1992)].  Energies per electron (Hartree):

Exchange of an unpolarized density ``d`` (``exchpw91``, spn_pbe.f90:441-500),
``s = |grad d|/(2 k_F d)``, ``k_F = (3 pi**2 d)**(1/3)``:

``e_x = a_x d**(1/3) F(s)``,
``F = [1 + a1 s asinh(a s) + (a2 - a3 exp(-a4 s**2)) s**2]
      / [1 + a1 s asinh(a s) + b1 s**4]``,

and for two spin densities the exact spin scaling
``E_x[n_up, n_down] = (E_x[2 n_up] + E_x[2 n_down]) / 2``.

Correlation: the local PW92 ``e_c(r_s, zeta)`` (``corlsd``,
:func:`parsec_python.V_xc.pw92_lda.pw92_lsd_correlation`) plus
``H = H0 + H1`` (``corpw91``, spn_pbe.f90:591-705) with
``g = [(1+zeta)**(2/3) + (1-zeta)**(2/3)]/2``,
``t = |grad n|/(2 g k_s n)``, ``k_s = (4 k_F/pi)**(1/2)``:

``H0 = g**3 (beta/delta) ln[1 + delta t**2 (1 + B t**2)/(1 + B t**2 + B**2 t**4)]``,
``B = delta / [exp(-delta e_c/(g**3 beta)) - 1]``,
``H1 = nu [C_c(r_s) - C_c0 - 3 C_x/7] g**3 t**2 exp(-100 g**4 (k_s/k_F)**2 t**2)``,

``beta = nu C_c0``, ``delta = 2 alpha/beta``.

The potential is not translated from the Fortran.  ``pbe.f90`` and
``spn_pbe.f90`` evaluate the pointwise EASYPBE formula, which needs the
Laplacian of the density and ``grad n . grad |grad n|`` besides the
gradient, and in the unpolarized ``pw91`` its reduced gradient ``u`` carries
an extra factor two (pbe.f90:413; ``spn_pbe`` has it right), which changes
that potential only.  Here, as for PBE (:mod:`parsec_python.V_xc.pbe`), the
energy is the discrete sum ``h**3 sum_i f(n_i, sigma_i)`` with
``sigma = |D n|**2`` of the centered first-derivative stencil, and the
potential is its exact derivative (:func:`~parsec_python.V_xc.pbe.discrete_gga`,
:func:`~parsec_python.V_xc.pbe_spin.discrete_spin_gga`).  The energy
density is that of the Fortran; the two potentials agree in the continuum
limit.

Further deviations, the same as for PBE: densities below 1e-20 carry no
energy (Fortran: 1e-18); a spin channel contributes its exchange wherever
its own density is above that threshold (``spn_pbe`` drops the exchange of
a point unless both spin densities exceed 1e-18); and ``d g/d zeta`` is
regularized with ``eta = 1e-12`` as in ``spn_corpbe`` (``corpw91`` has no
``eta`` and is singular at ``|zeta| = 1``).

Each spin channel sees half of a frozen nonlinear-core density; the
unpolarized functional sees all of it.  :func:`pw91` and :func:`pw91_spin`
return Rydberg.
"""

from __future__ import annotations

import numpy as np

from .ca_lda import XCResult
from .ca_lda_spin import SpinXCResult, spin_densities_with_core
from .pbe import (
    _DENSITY_THRESHOLD,
    _KF_COEFFICIENT,
    _gga_density,
    discrete_gga,
)
from .pbe_spin import discrete_spin_gga
from .pw92_lda import _AX, pw92_lsd_correlation


# exchpw91 (spn_pbe.f90:441-500).
_A = 7.7956
_A1 = 0.19645
_A2 = 0.27430
_A3 = 0.15084
_A4 = 100.0
_B1 = 0.004
# corpw91 (spn_pbe.f90:591-705).
_XNU = 15.75592
_CC0 = 0.004235
_CX = -0.001667212
_ALF = 0.09
_C1 = 0.002568
_C2 = 0.023266
_C3 = 7.389e-6
_C4 = 8.723
_C5 = 0.472
_C6 = 7.389e-2
_H1_EXPONENT = 100.0
# (k_s/k_F)**2 = 4/(pi k_F) = 0.663436444 r_s.
_KS2_OVER_KF2 = 0.663436444
_BET = _XNU * _CC0
_DELT = 2.0 * _ALF / _BET
# spn_corpbe's regularization of d g/d zeta (corpw91 has none).
_SPIN_ETA = 1.0e-12


def _exchange_partials(n: np.ndarray, sig: np.ndarray):
    """PW91 exchange ``f_x = n e_x`` of an unpolarized density and its partials.

    Returns ``(f_x, df_x/dn, df_x/dsigma)`` in Hartree for positive ``n``.
    With ``f_s = (1/s) dF/ds`` of ``exchpw91``,
    ``df_x/dn = (4/3) a_x n**(1/3) (F - s**2 f_s)`` and
    ``df_x/dsigma = a_x n**(4/3) f_s / (8 k_F**2 n**2)``.
    """

    n13 = np.cbrt(n)
    kf = _KF_COEFFICIENT * n13
    scale = 1.0 / (4.0 * kf * kf * n * n)
    s2 = sig * scale
    s = np.sqrt(s2)
    p0 = 1.0 / np.sqrt(1.0 + _A * _A * s2)
    p1 = np.arcsinh(_A * s)
    p2 = np.exp(-_A4 * s2)
    p3 = 1.0 / (1.0 + _A1 * s * p1 + _B1 * s2 * s2)
    p4 = 1.0 + _A1 * s * p1 + (_A2 - _A3 * p2) * s2
    enhancement = p3 * p4
    p5 = _B1 * s2 - (_A2 - _A3 * p2)
    p6 = _A1 * s * (p1 + _A * s * p0)
    p7 = 2.0 * (_A2 - _A3 * p2) + 2.0 * _A3 * _A4 * s2 * p2 - 4.0 * _B1 * s2 * enhancement
    fs = p3 * (p3 * p5 * p6 + p7)
    prefactor = _AX * n13
    energy = n * prefactor * enhancement
    derivative_n = (4.0 / 3.0) * prefactor * (enhancement - s2 * fs)
    derivative_sigma = 0.5 * n * prefactor * fs * scale
    return energy, derivative_n, derivative_sigma


def _gradient_correction(rs, g, gz, ec, ecrs, eczet, t2):
    """``H`` of ``corpw91`` and the derivatives the variational potential needs.

    Returns ``H``, ``hrs = -(r_s/3) dH/dr_s``, ``ht = (1/t) dH/dt`` and
    ``hz = dH/dzeta`` (all at fixed ``t``), with the names of the Fortran.
    """

    g3 = g * g * g
    g4 = g3 * g
    pon = -_DELT * ec / (g3 * _BET)
    b = _DELT / np.expm1(pon)
    b2 = b * b
    t4 = t2 * t2
    t6 = t4 * t2
    q4 = 1.0 + b * t2
    q5 = 1.0 + b * t2 + b2 * t4
    q6 = _C1 + _C2 * rs + _C3 * rs * rs
    q7 = 1.0 + _C4 * rs + _C5 * rs * rs + _C6 * rs * rs * rs
    cc = -_CX + q6 / q7
    r1 = _H1_EXPONENT * _KS2_OVER_KF2 * rs * g4
    coeff = cc - _CC0 - 3.0 * _CX / 7.0
    r2 = _XNU * coeff * g3
    r3 = np.exp(-r1 * t2)
    h0 = g3 * (_BET / _DELT) * np.log1p(_DELT * q4 * t2 / q5)
    h1 = r3 * r2 * t2

    ccrs = (_C2 + 2.0 * _C3 * rs) / q7 - q6 * (
        _C4 + 2.0 * _C5 * rs + 3.0 * _C6 * rs * rs
    ) / (q7 * q7)
    rsthrd = rs / 3.0
    r4 = rsthrd * ccrs / coeff
    fac = _DELT / b + 1.0
    bg = -3.0 * b2 * ec * fac / (_BET * g4)
    bec = b2 * fac / (_BET * g3)
    q8 = q5 * q5 + _DELT * q4 * q5 * t2
    q9 = 1.0 + 2.0 * b * t2
    h0b = -_BET * g3 * b * t6 * (2.0 + b * t2) / q8
    h0rs = -rsthrd * h0b * bec * ecrs
    h0z = 3.0 * gz * h0 / g + h0b * (bg * gz + bec * eczet)
    h0t = 2.0 * _BET * g3 * q9 / q8
    h1rs = r3 * r2 * t2 * (-r4 + r1 * t2 / 3.0)
    h1z = gz * r3 * r2 * t2 * (3.0 - 4.0 * r1 * t2) / g
    h1t = 2.0 * r3 * r2 * (1.0 - r1 * t2)
    return h0 + h1, h0rs + h1rs, h0t + h1t, h0z + h1z


def pw91_energy_partials(
    density: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpolarized PW91 ``f``, ``df/dn`` and ``df/dsigma`` in Hartree.

    ``f = n (e_x + e_c + H)`` is the energy per volume of ``pw91``
    (pbe.f90:384-443) at ``zeta = 0``, ``g = 1``; ``sigma = |grad n|**2``.
    With ``t**2 = pi sigma/(16 k_F n**2)``,

    ``df_c/dn = e_c - (r_s/3) e_c' + H + hrs - (7/6) t**2 ht``,
    ``df_c/dsigma = pi ht / (32 k_F n)``.
    """

    density = np.asarray(density, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if density.shape != sigma.shape:
        raise ValueError("PW91 density and squared gradient must have the same shape")
    if np.any(density < -1.0e-14):
        raise ValueError("PW91 requires a nonnegative density")
    if np.any(sigma < -1.0e-14):
        raise ValueError("PW91 requires a nonnegative squared density gradient")

    energy_density = np.zeros_like(density)
    derivative_density = np.zeros_like(density)
    derivative_sigma = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    if not np.any(positive):
        return energy_density, derivative_density, derivative_sigma

    n = density[positive]
    sig = np.maximum(sigma[positive], 0.0)
    exchange, exchange_dn, exchange_dsigma = _exchange_partials(n, sig)
    correlation, correlation_dn, correlation_dsigma = _correlation_partials(n, sig)

    energy_density[positive] = exchange + correlation
    derivative_density[positive] = exchange_dn + correlation_dn
    derivative_sigma[positive] = exchange_dsigma + correlation_dsigma
    return energy_density, derivative_density, derivative_sigma


def _correlation_partials(n: np.ndarray, sig: np.ndarray):
    """PW91 correlation ``f_c = n (e_c + H)`` of an unpolarized density and its partials.

    The correlation half of :func:`pw91_energy_partials` (Hartree) for positive
    ``n`` and nonnegative ``sig``.
    """

    kf = _KF_COEFFICIENT * np.cbrt(n)
    rs = np.cbrt(3.0 / (4.0 * np.pi * n))
    zeta = np.zeros_like(n)
    ec, ecrs, eczet = pw92_lsd_correlation(rs, zeta)
    dt2_dsigma = np.pi / (16.0 * kf * n * n)
    t2 = sig * dt2_dsigma
    # zeta = 0: g = 1 and d g/d zeta = 0.
    h, hrs, ht, _ = _gradient_correction(rs, np.ones_like(n), zeta, ec, ecrs, eczet, t2)
    return (
        n * (ec + h),
        ec - rs * ecrs / 3.0 + h + hrs - (7.0 / 6.0) * t2 * ht,
        0.5 * n * ht * dt2_dsigma,
    )


def pw91_spin_energy_partials(
    density_up: np.ndarray,
    density_down: np.ndarray,
    sigma_up: np.ndarray,
    sigma_down: np.ndarray,
    sigma_total: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Spin-polarized PW91 ``f`` and its five partial derivatives (Hartree).

    The variables are those of
    :func:`parsec_python.V_xc.pbe_spin.pbe_spin_energy_partials`: the two
    spin densities, their squared gradients and the squared gradient of the
    total density.  ``f`` is the energy per volume of ``spn_pbe``'s PW91.
    With ``t**2 = pi sigma/(16 g**2 k_F n**2)`` and
    ``pref = dH/dzeta - (g'/g) t**2 ht`` (at fixed ``r_s`` and ``sigma``),

    ``df_c/dn_s = e_c - (r_s/3) e_c' + H + hrs - (7/6) t**2 ht
                  +- (1 -+ zeta)(de_c/dzeta + pref)``,
    ``df_c/dsigma = pi ht / (32 g**2 k_F n)``.
    """

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (density_up, density_down, sigma_up, sigma_down, sigma_total)
    ]
    up, down, sig_up, sig_down, sig_total = arrays
    if any(value.shape != up.shape for value in arrays):
        raise ValueError("PW91 spin densities and squared gradients must share a shape")
    if np.any(up < -1.0e-14) or np.any(down < -1.0e-14):
        raise ValueError("PW91 requires nonnegative spin densities")
    if any(np.any(value < -1.0e-14) for value in (sig_up, sig_down, sig_total)):
        raise ValueError("PW91 requires nonnegative squared density gradients")
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

    correlation, correlation_up, correlation_down, correlation_sigma = _spin_correlation_partials(
        density[positive], up[positive], down[positive], sig_total[positive]
    )
    energy[positive] += correlation
    d_up[positive] += correlation_up
    d_down[positive] += correlation_down
    d_sigma_total[positive] = correlation_sigma
    return energy, d_up, d_down, d_sigma_up, d_sigma_down, d_sigma_total


def _spin_correlation_partials(n, up, down, sigma_total):
    """PW91 correlation of two spin densities: ``f_c``, ``df_c/dn_up``, ``df_c/dn_down``, ``df_c/dsigma``.

    The correlation half of :func:`pw91_spin_energy_partials` (Hartree) at
    points whose total density ``n`` is above the vacuum threshold.
    """

    sig = np.maximum(sigma_total, 0.0)
    zeta = (up - down) / n
    kf = _KF_COEFFICIENT * np.cbrt(n)
    rs = np.cbrt(3.0 / (4.0 * np.pi * n))
    ec, ecrs, eczet = pw92_lsd_correlation(rs, zeta)
    plus = np.maximum(1.0 + zeta, 0.0)
    minus = np.maximum(1.0 - zeta, 0.0)
    g = 0.5 * (plus ** (2.0 / 3.0) + minus ** (2.0 / 3.0))
    gz = (
        ((1.0 + zeta) ** 2 + _SPIN_ETA) ** (-1.0 / 6.0)
        - ((1.0 - zeta) ** 2 + _SPIN_ETA) ** (-1.0 / 6.0)
    ) / 3.0
    dt2_dsigma = np.pi / (16.0 * g * g * kf * n * n)
    t2 = sig * dt2_dsigma
    h, hrs, ht, hz = _gradient_correction(rs, g, gz, ec, ecrs, eczet, t2)
    pref = hz - gz * t2 * ht / g
    common = ec - rs * ecrs / 3.0 + h + hrs - (7.0 / 6.0) * t2 * ht
    return (
        n * (ec + h),
        common + (1.0 - zeta) * (eczet + pref),
        common - (1.0 + zeta) * (eczet + pref),
        0.5 * n * ht * dt2_dsigma,
    )


def pw91(
    valence_density: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
) -> XCResult:
    """Unpolarized PW91 and its exact discrete grid derivative (Rydberg).

    The discretization is that of :func:`parsec_python.V_xc.pbe.pbe`: the
    centered first-derivative stencil, zero padding outside an isolated
    cluster and wraparound on a periodic grid.
    """

    density = _gga_density(valence_density, core_density, grid, "PW91")
    energy_hartree, potential_hartree = discrete_gga(
        density, grid, pw91_energy_partials
    )
    energy_density = 2.0 * energy_hartree
    epsilon = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    epsilon[positive] = energy_density[positive] / density[positive]
    return XCResult(
        potential=2.0 * potential_hartree,
        energy_per_electron=epsilon,
        energy_density=energy_density,
        total_energy=float(grid.volume_element * np.sum(energy_density)),
    )


def pw91_spin(
    valence_up: np.ndarray,
    valence_down: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
) -> SpinXCResult:
    """Spin-polarized PW91 and its exact discrete grid derivatives (Rydberg)."""

    up, down = spin_densities_with_core(
        valence_up, valence_down, core_density, functional="PW91"
    )
    if up.shape != (grid.size,):
        raise ValueError("PW91 densities do not match the real-space grid")
    return discrete_spin_gga(up, down, grid, pw91_spin_energy_partials)


__all__ = [
    "pw91",
    "pw91_energy_partials",
    "pw91_spin",
    "pw91_spin_energy_partials",
]
