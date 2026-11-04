"""Perdew-Wang 1992 local (spin) density functional, PARSEC's ``pl``.

This is the ``Correlation_Type = pl`` (``pw92``, ``pwlda``) branch of
``exc_nspn.f90`` (lines 271-286) and ``exc_spn.f90`` (lines 301-341): Dirac
exchange and the Perdew-Wang parametrization of the correlation energy of
the uniform electron gas [J. P. Perdew and Y. Wang, Phys. Rev. B 45, 13244
(1992)],

``e_c(r_s, zeta) = e_u (1 - f zeta**4) + e_p f zeta**4
                   - a_m f (1 - zeta**4) / f''(0)``,

``f(zeta) = [(1+zeta)**(4/3) + (1-zeta)**(4/3) - 2] / gamma``,

where ``e_u``, ``e_p`` and ``a_m`` (minus the spin stiffness) all have the
form ``G(r_s)`` of their Eq. 10 (:func:`parsec_python.V_xc.pbe._pw92_g`).
It is the functional Fortran PARSEC uses for a spin-polarized calculation
without a ``Correlation_Type`` line.

The constants are those of the Fortran source, with its digits: ``A`` of the
unpolarized gas is 0.0310907 (not ``(1 - ln 2)/pi**2`` as in the PBE of
:mod:`parsec_python.V_xc.pbe`), ``gamma = 0.5198421`` and
``f''(0) = 1.709921``.  The same LSD correlation (``corlsd`` of
``spn_pbe.f90``) is the local part of PW91 (:mod:`parsec_python.V_xc.pw91`).

Each spin channel sees its density plus half of a frozen nonlinear-core
density (``rho_hart.F90``).  As in :mod:`parsec_python.V_xc.ca_lda_spin`,
the spin functional is evaluated wherever the total density is positive;
``exc_spn.f90`` evaluates it only where both spin densities are positive and
leaves the densities themselves in its potential array elsewhere.  Densities
are in electrons/bohr^3, energies and potentials in Rydberg.
"""

from __future__ import annotations

import numpy as np

from .ca_lda import XCResult
from .ca_lda_spin import SpinXCResult, spin_densities_with_core
from .pbe import _pw92_g


# exc_spn.f90 / spn_pbe.f90 (corlsd) constants, with the digits of the source.
_AX = -0.738558766382022405884230032680836
_GAM = 0.5198421
_FZZ = 1.709921
# (A, alpha1, beta1..beta4) of G(r_s): unpolarized gas, fully polarized gas,
# minus the spin stiffness.
_UNPOLARIZED = (0.0310907, 0.21370, (7.5957, 3.5876, 1.6382, 0.49294))
_POLARIZED = (0.01554535, 0.20548, (14.1189, 6.1977, 3.3662, 0.62517))
_STIFFNESS = (0.0168869, 0.11125, (10.357, 3.6231, 0.88026, 0.49671))
# exc_nspn.f90: a0 = (4/(9 pi))**(1/3), unpolarized Dirac exchange potential
# -2/(pi a0 r_s) in Rydberg.
_A0 = (4.0 / (9.0 * np.pi)) ** (1.0 / 3.0)
_TWO_OVER_PI_A0 = 2.0 / (np.pi * _A0)


def dirac_exchange_potential(rs: np.ndarray) -> np.ndarray:
    """Unpolarized Dirac exchange potential ``-2/(pi a0 r_s)`` in Rydberg.

    The exchange energy per electron is three quarters of it.  This is the
    ``vx`` of every LDA branch of ``exc_nspn.f90``.
    """

    return -_TWO_OVER_PI_A0 / rs


def pw92_correlation(rs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``e_c`` of the unpolarized gas and ``d e_c/d r_s``, in Hartree.

    ``gcor2`` of ``pbe.f90`` (lines 356-378) with the constants of
    ``exc_nspn.f90``.
    """

    a, alpha1, betas = _UNPOLARIZED
    return _pw92_g(np.asarray(rs, dtype=np.float64), a, alpha1, betas)


def pw92_lsd_correlation(
    rs: np.ndarray, zeta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spin-polarized PW92 ``e_c``, ``d e_c/d r_s`` and ``d e_c/d zeta`` (Hartree).

    ``corlsd`` of ``spn_pbe.f90`` (lines 506-551), the same algebra as the
    ``pl`` branch of ``exc_spn.f90``, with ``gamma = 0.5198421`` and
    ``f''(0) = 1.709921``.  At ``zeta = 0`` the result is exactly that of
    :func:`pw92_correlation`.
    """

    rs = np.asarray(rs, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    unpolarized, unpolarized_rs = _pw92_g(rs, _UNPOLARIZED[0], _UNPOLARIZED[1], _UNPOLARIZED[2])
    polarized, polarized_rs = _pw92_g(rs, _POLARIZED[0], _POLARIZED[1], _POLARIZED[2])
    stiffness, stiffness_rs = _pw92_g(rs, _STIFFNESS[0], _STIFFNESS[1], _STIFFNESS[2])
    # Roundoff must not carry 1 +- zeta below zero under the roots.
    plus = np.maximum(1.0 + zeta, 0.0)
    minus = np.maximum(1.0 - zeta, 0.0)
    f = (plus ** (4.0 / 3.0) + minus ** (4.0 / 3.0) - 2.0) / _GAM
    z4 = zeta**4
    energy = (
        unpolarized * (1.0 - f * z4)
        + polarized * f * z4
        - stiffness * f * (1.0 - z4) / _FZZ
    )
    energy_rs = (
        unpolarized_rs * (1.0 - f * z4)
        + polarized_rs * f * z4
        - stiffness_rs * f * (1.0 - z4) / _FZZ
    )
    f_zeta = (4.0 / 3.0) * (np.cbrt(plus) - np.cbrt(minus)) / _GAM
    energy_zeta = 4.0 * zeta**3 * f * (
        polarized - unpolarized + stiffness / _FZZ
    ) + f_zeta * (
        z4 * polarized - z4 * unpolarized - (1.0 - z4) * stiffness / _FZZ
    )
    return energy, energy_rs, energy_zeta


def _total_density(valence_density, core_density, functional: str) -> np.ndarray:
    valence = np.asarray(valence_density, dtype=np.float64)
    if core_density is None:
        density = valence
    else:
        core = np.asarray(core_density, dtype=np.float64)
        if core.shape != valence.shape:
            raise ValueError("core and valence densities must have the same shape")
        density = valence + core
    if np.any(density < -1.0e-14):
        raise ValueError(f"{functional} requires a nonnegative density")
    return density


def _local_result(density, potential, energy_density, volume_element) -> XCResult:
    epsilon = np.zeros_like(density)
    positive = density > 0.0
    epsilon[positive] = energy_density[positive] / density[positive]
    return XCResult(
        potential=potential,
        energy_per_electron=epsilon,
        energy_density=energy_density,
        total_energy=float(np.sum(energy_density) * volume_element),
    )


def pw92_lda(
    valence_density: np.ndarray,
    volume_element: float,
    core_density: np.ndarray | None = None,
) -> XCResult:
    """Unpolarized PW92 LDA on a uniform grid (``exc_nspn.f90``, ``pl``).

    With ``v_x = -2/(pi a0 r_s)`` (Rydberg) and ``e_c``, ``e_c'`` of
    :func:`pw92_correlation` (Hartree),

    ``V_xc = v_x + 2 (e_c - r_s e_c'/3)``,
    ``E_xc = dV * sum rho (3/4 v_x + 2 e_c)``,

    evaluated at ``rho = rho_valence + rho_core``; points with ``rho <= 0``
    carry no potential and no energy.
    """

    density = _total_density(valence_density, core_density, "PW92-LDA")
    potential = np.zeros_like(density)
    energy_density = np.zeros_like(density)
    positive = density > 0.0
    if np.any(positive):
        rho = density[positive]
        rs = np.cbrt((0.75 / np.pi) / rho)
        exchange = dirac_exchange_potential(rs)
        correlation, correlation_rs = pw92_correlation(rs)
        potential[positive] = exchange + 2.0 * (correlation - rs * correlation_rs / 3.0)
        energy_density[positive] = rho * (0.75 * exchange + 2.0 * correlation)
    return _local_result(density, potential, energy_density, volume_element)


def pw92_lda_spin(
    valence_up: np.ndarray,
    valence_down: np.ndarray,
    volume_element: float,
    core_density: np.ndarray | None = None,
) -> SpinXCResult:
    """Spin-polarized PW92 LSDA on a uniform grid (``exc_spn.f90``, ``pl``).

    ``e_x(d) = a_x d**(1/3)`` (Hartree, ``a_x = -(3/4)(3/pi)**(1/3)``) is
    the exchange per electron of an unpolarized gas of density ``d``; with
    ``e_c``, ``e_c'`` and ``de_c/dzeta`` of :func:`pw92_lsd_correlation`,

    ``V_s = 2 [(4/3) e_x(2 rho_s) + e_c - r_s e_c'/3 - zeta de_c/dzeta
               +- de_c/dzeta]`` (upper sign up),
    ``E_xc = dV * sum 2 [rho_up e_x(2 rho_up) + rho_down e_x(2 rho_down)
                         + rho e_c]``.
    """

    up, down = spin_densities_with_core(
        valence_up, valence_down, core_density, functional="PW92-LSDA"
    )
    potential_up = np.zeros_like(up)
    potential_down = np.zeros_like(up)
    energy_density = np.zeros_like(up)
    density = up + down
    positive = density > 0.0
    if np.any(positive):
        rho_up = up[positive]
        rho_down = down[positive]
        rho = density[positive]
        exchange_up = _AX * np.cbrt(2.0 * rho_up)
        exchange_down = _AX * np.cbrt(2.0 * rho_down)
        rs = np.cbrt((0.75 / np.pi) / rho)
        zeta = (rho_up - rho_down) / rho
        correlation, correlation_rs, correlation_zeta = pw92_lsd_correlation(rs, zeta)
        common = correlation - rs * correlation_rs / 3.0 - zeta * correlation_zeta
        potential_up[positive] = 2.0 * (
            (4.0 / 3.0) * exchange_up + common + correlation_zeta
        )
        potential_down[positive] = 2.0 * (
            (4.0 / 3.0) * exchange_down + common - correlation_zeta
        )
        energy_density[positive] = 2.0 * (
            rho_up * exchange_up + rho_down * exchange_down + correlation * rho
        )
    return SpinXCResult(
        potential_up=potential_up,
        potential_down=potential_down,
        energy_density=energy_density,
        total_energy=float(np.sum(energy_density) * volume_element),
    )


__all__ = [
    "dirac_exchange_potential",
    "pw92_correlation",
    "pw92_lda",
    "pw92_lda_spin",
    "pw92_lsd_correlation",
]
