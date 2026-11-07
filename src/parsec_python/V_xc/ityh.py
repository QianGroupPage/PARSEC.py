"""Short-range B88 exchange of Iikura, Tsuneda, Yanai and Hirao (the DFT exchange of CAM-B3LYP).

A GGA exchange per spin channel ``e_s = -(1/2) rho**(4/3) K_s`` is attenuated
for the interaction ``erfc(omega r)/r`` as the uniform gas would be at the
local Fermi momentum the GGA implies (H. Iikura, T. Tsuneda, T. Yanai,
K. Hirao, J. Chem. Phys. 115, 3540 (2001), Eqs. 9-11):

``e_s^SR = e_s F(a)``, ``a = omega / (2 k_s)``, ``k_s = sqrt(9 pi / K_s) rho**(1/3)``,
``F(a) = 1 - (8/3) a [sqrt(pi) erf(1/(2a)) + 2a (e - 1) - 4 a**3 (e - 1) - a]``,
``e = exp(-1/(4 a**2))``

(the short-range attenuation of the erf kernel, as in Tawada et al.,
J. Chem. Phys. 120, 8425 (2004)).  ``K_s = 2 X F_B88(x)`` with the per-spin
Slater constant ``X = (3/2)(3/(4 pi))**(1/3)`` and the B88 enhancement
``F_B88 = 1 + (beta/X) x**2 / (1 + 6 beta x asinh x)``
(:mod:`parsec_python.V_xc.becke_lyp`); for ``F_B88 = 1``, ``k_s`` is the
Fermi momentum of ``2 rho``.

For ``a >= 1.35`` the bracket cancels against the one; there ``F`` is its
expansion in ``1/a`` to ``a**-16``,
``F = -(4/3) sum_{k=1..8} beta_k (2a)**(-2k)``,
``beta_k = (-1)**k [2/(k! (2k+1)) - 1/(k+1)! - 1/(2 (k+2)!)]``
(``1/(36 a**2) - 1/(960 a**4) + ...``), derived here; libxc's
``GGA_X_ITYH`` switches at the same ``a`` with the same order.  The
derivative is ``dF/da = -(8/3)[bracket - 12 a**3 (e - 1) - 3a]``.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial

import numpy as np
from scipy.special import erf

from .becke_lyp import b88_spin_channel

_X = 1.5 * (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
_SQRT_PI = np.sqrt(np.pi)
_SERIES_CUTOFF = 1.35
_SERIES_ORDER = 8


def _series_coefficients() -> tuple[float, ...]:
    """``c_k`` of ``F = sum_k c_k a**(-2k)``, ``k = 1 .. 8``, from the exact rationals ``beta_k``."""

    coefficients = []
    for k in range(1, _SERIES_ORDER + 1):
        beta = (-1) ** k * (
            Fraction(2, factorial(k) * (2 * k + 1)) - Fraction(1, factorial(k + 1)) - Fraction(1, 2 * factorial(k + 2))
        )
        coefficients.append(float(Fraction(-4, 3) * beta / Fraction(4) ** k))
    return tuple(coefficients)


_SERIES = _series_coefficients()


def erf_attenuation(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``F(a)`` and ``dF/da`` of the module docstring for ``a > 0``."""

    a = np.asarray(a, dtype=np.float64)
    value = np.empty_like(a)
    derivative = np.empty_like(a)
    near = a < _SERIES_CUTOFF
    if np.any(near):
        x = a[near]
        e_minus_one = np.expm1(-1.0 / (4.0 * x * x))
        bracket = _SQRT_PI * erf(1.0 / (2.0 * x)) + 2.0 * x * (e_minus_one - (2.0 * x * x * e_minus_one + 0.5))
        value[near] = 1.0 - (8.0 / 3.0) * x * bracket
        derivative[near] = -(8.0 / 3.0) * (bracket - 12.0 * x * x * x * e_minus_one - 3.0 * x)
    if np.any(~near):
        x = a[~near]
        inverse2 = 1.0 / (x * x)
        total = np.zeros_like(x)
        slope = np.zeros_like(x)
        # Horner in 1/a**2: F = sum c_k u**k, dF/da = -(2/a) sum k c_k u**k.
        for k in range(_SERIES_ORDER, 0, -1):
            total = (total + _SERIES[k - 1]) * inverse2
            slope = (slope + k * _SERIES[k - 1]) * inverse2
        value[~near] = total
        derivative[~near] = -2.0 * slope / x
    return value, derivative


def ityh_b88_spin_channel(rho, sigma, omega):
    """Short-range B88 of one spin density ``rho > 0``: ``(e_s, de_s/drho, de_s/dsigma)`` (Hartree)."""

    energy, d_rho, d_sigma = b88_spin_channel(rho, sigma)
    # F_B88 = e_s / (-X rho**(4/3)); a = (omega/2) sqrt(2 X F_B88 / (9 pi)) rho**(-1/3).
    rho13 = np.cbrt(rho)
    lda = -_X * rho * rho13
    enhancement = energy / lda
    # d F_B88 = (d e_s - F_B88 d lda) / lda.
    d_enhancement_rho = (d_rho - enhancement * (4.0 / 3.0) * (-_X * rho13)) / lda
    d_enhancement_sigma = d_sigma / lda
    a = 0.5 * float(omega) * np.sqrt(2.0 * _X * enhancement / (9.0 * np.pi)) / rho13
    attenuation, slope = erf_attenuation(a)
    a_rho = a * (-1.0 / (3.0 * rho) + 0.5 * d_enhancement_rho / enhancement)
    a_sigma = a * 0.5 * d_enhancement_sigma / enhancement
    return (
        energy * attenuation,
        d_rho * attenuation + energy * slope * a_rho,
        d_sigma * attenuation + energy * slope * a_sigma,
    )


def ityh_b88_exchange_partials(n, sigma, omega):
    """Short-range B88 ``(e, de/dn, de/dsigma)`` of a positive unpolarized density: ``2 e_s(n/2, sigma/4)``."""

    energy, d_rho, d_sigma = ityh_b88_spin_channel(0.5 * n, 0.25 * sigma, omega)
    return 2.0 * energy, d_rho, 0.5 * d_sigma


__all__ = ["erf_attenuation", "ityh_b88_exchange_partials", "ityh_b88_spin_channel"]
