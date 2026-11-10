"""Slater and Becke 1988 exchange, VWN and Lee-Yang-Parr correlation with their first derivatives.

The semilocal pieces of the B3 hybrids (B3LYP, B3LYP5, B3PW91, CAM-B3LYP) as
the self-consistent field needs them: the energy per volume and its partial
derivatives with respect to the densities and the squared gradients
(Hartree atomic units).  The energies are those of
:mod:`parsec_python.V_xc.b3lyp`, which evaluates them for the post-SCF
``Energy_Refinement: b3lyp`` and is left untouched; the constants are taken
from it.

Exchange (unpolarized density ``n``, ``sigma = |grad n|**2``; spin channels
by the exact spin scaling, :func:`~parsec_python.V_xc.semilocal_components.spin_scaled_exchange`):

* Slater: ``e = -C_x n**(4/3)``, ``C_x = (3/4)(3/pi)**(1/3)``;
* B88 (A. D. Becke, Phys. Rev. A 38, 3098 (1988), Eq. 8) per spin density
  ``rho`` with ``x = |grad rho| / rho**(4/3)``:
  ``e_s = -rho**(4/3) [c_s + beta g(x)]``, ``g = x**2 / (1 + 6 beta x asinh x)``,
  ``c_s = (3/2)(3/(4 pi))**(1/3)``, ``beta = 0.0042``; the unpolarized
  energy is ``2 e_s(n/2, sigma/4)``.  Then
  ``de_s/drho = -(4/3) rho**(1/3) [c_s + beta (g - x g')]`` and
  ``de_s/dsigma_ss = -beta (g'/x) / (2 rho**(4/3))`` with
  ``g'/x = (2 D - x D') / D**2``, ``D = 1 + 6 beta x asinh x``,
  ``D' = 6 beta (asinh x + x / sqrt(1 + x**2))``.

Correlation:

* VWN (S. H. Vosko, L. Wilk, M. Nusair, Can. J. Phys. 58, 1200 (1980),
  Eq. 4.4) with ``x = sqrt(r_s)``, ``X = x**2 + b x + c``,
  ``Q = sqrt(4 c - b**2)`` per parameter set, and
  ``d eps/dx = A [2/x - (2x + b)/X - 4b/((2x+b)**2 + Q**2)
  - (b x0/X(x0)) (2/(x - x0) - (2x + b)/X - 4(b + 2 x0)/((2x+b)**2 + Q**2))]``,
  ``dx/dn = -x/(6 n)``.  ``"rpa"``: ``eps = eps_P + (eps_F - eps_P) f(zeta)``
  (libxc ``LDA_C_VWN_RPA``); ``"vwn5"``: their Eq. 4.15 with the spin
  stiffness, ``eps = eps_P + alpha_c f (1 - zeta**4)/f''(0) + (eps_F - eps_P) f zeta**4``
  (libxc ``LDA_C_VWN``);
* LYP in the gradient-only form of Miehlich, Savin, Stoll and Preuss
  (Chem. Phys. Lett. 157, 200 (1989), Eq. 2; see
  :func:`parsec_python.V_xc.b3lyp.lyp_correlation_energy_density_spin`),
  with ``u = n**(-1/3)``, ``D = 1 + d u``,
  ``omega = exp(-c u) u**11 / D``, ``delta = c u + d u / D`` and
  ``d omega/dn = -omega (11 - delta)/(3n)``,
  ``d delta/dn = -(delta - d**2 u**2 / D**2)/(3n)``.  The unpolarized form is
  the closed-shell reduction of the same equation.

The spin variables are libxc's, ``s_ab = grad n_a . grad n_b``.
"""

from __future__ import annotations

import numpy as np

from .b3lyp import (
    _B88_BETA,
    _LYP_A,
    _LYP_B,
    _LYP_C,
    _LYP_CF,
    _LYP_D,
    _SLATER_CX,
    _VWN_F_SECOND,
    _VWN_FERROMAGNETIC,
    _VWN_PARAMETERS,
    _VWN_SPIN_STIFFNESS,
)

# B88 per-spin Slater constant (3/2)(3/(4 pi))**(1/3).
_B88_CS = 1.5 * (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
# 2**(11/3) C_F of the spin LYP.
_LYP_CF_SPIN = 2.0 ** (11.0 / 3.0) * _LYP_CF
_FZ_DENOMINATOR = 2.0 ** (4.0 / 3.0) - 2.0


# ---------------------------------------------------------------------------
# Exchange (unpolarized cores on positive densities)
# ---------------------------------------------------------------------------


def slater_exchange_partials(n, sigma, omega=None):
    """Slater exchange ``(e, de/dn, de/dsigma)`` of a positive unpolarized density (Hartree)."""

    n13 = np.cbrt(n)
    return -_SLATER_CX * n * n13, -(4.0 / 3.0) * _SLATER_CX * n13, np.zeros_like(n)


def b88_spin_channel(rho, sigma):
    """B88 exchange of one spin density ``rho > 0`` with ``sigma = |grad rho|**2``: ``(e_s, de_s/drho, de_s/dsigma)``."""

    rho13 = np.cbrt(rho)
    rho43 = rho * rho13
    x = np.sqrt(sigma) / rho43
    asinh = np.arcsinh(x)
    denominator = 1.0 + 6.0 * _B88_BETA * x * asinh
    g = x * x / denominator
    denominator_prime = 6.0 * _B88_BETA * (asinh + x / np.sqrt(1.0 + x * x))
    g_prime_over_x = (2.0 * denominator - x * denominator_prime) / (denominator * denominator)
    energy = -rho43 * (_B88_CS + _B88_BETA * g)
    d_rho = -(4.0 / 3.0) * rho13 * (_B88_CS + _B88_BETA * (g - x * x * g_prime_over_x))
    d_sigma = -_B88_BETA * g_prime_over_x / (2.0 * rho43)
    return energy, d_rho, d_sigma


def b88_exchange_partials(n, sigma, omega=None):
    """B88 exchange ``(e, de/dn, de/dsigma)`` of a positive unpolarized density: ``2 e_s(n/2, sigma/4)``."""

    energy, d_rho, d_sigma = b88_spin_channel(0.5 * n, 0.25 * sigma)
    return 2.0 * energy, d_rho, 0.5 * d_sigma


# ---------------------------------------------------------------------------
# VWN
# ---------------------------------------------------------------------------


def _vwn_epsilon_partials(x, parameters):
    """VWN Eq. 4.4 for one set ``(A, x0, b, c)`` and ``d eps/dx``, with ``x = sqrt(r_s)``."""

    a, x0, b, c = parameters
    big_x = x * x + b * x + c
    big_x0 = x0 * x0 + b * x0 + c
    q = np.sqrt(4.0 * c - b * b)
    two_x_b = 2.0 * x + b
    arctangent = np.arctan(q / two_x_b)
    ratio = b * x0 / big_x0
    epsilon = a * (
        np.log(x * x / big_x)
        + (2.0 * b / q) * arctangent
        - ratio * (np.log((x - x0) * (x - x0) / big_x) + (2.0 * (b + 2.0 * x0) / q) * arctangent)
    )
    # (2x + b)**2 + Q**2 = 4 X.
    four_x = 4.0 * big_x
    derivative = a * (
        2.0 / x
        - two_x_b / big_x
        - 4.0 * b / four_x
        - ratio * (2.0 / (x - x0) - two_x_b / big_x - 4.0 * (b + 2.0 * x0) / four_x)
    )
    return epsilon, derivative


def _vwn_x(n):
    return np.sqrt(np.cbrt(3.0 / (4.0 * np.pi * n)))


def vwn_correlation_partials(n, sigma, omega=None, *, parametrization: str = "rpa"):
    """Unpolarized VWN correlation ``(e, de/dn, 0)`` (paramagnetic fit) of a positive density."""

    x = _vwn_x(n)
    epsilon, epsilon_x = _vwn_epsilon_partials(x, _VWN_PARAMETERS[parametrization])
    return n * epsilon, epsilon - x * epsilon_x / 6.0, np.zeros_like(n)


def vwn_spin_correlation_partials(n, up, down, *, parametrization: str = "rpa"):
    """Spin VWN correlation ``(e, e_up, e_down, 0, 0, 0)`` where ``n = up + down > 0`` (module docstring)."""

    x = _vwn_x(n)
    zeta = (up - down) / n
    plus = np.maximum(1.0 + zeta, 0.0)
    minus = np.maximum(1.0 - zeta, 0.0)
    f = (plus ** (4.0 / 3.0) + minus ** (4.0 / 3.0) - 2.0) / _FZ_DENOMINATOR
    f_zeta = (4.0 / 3.0) * (np.cbrt(plus) - np.cbrt(minus)) / _FZ_DENOMINATOR
    paramagnetic, paramagnetic_x = _vwn_epsilon_partials(x, _VWN_PARAMETERS[parametrization])
    ferromagnetic, ferromagnetic_x = _vwn_epsilon_partials(x, _VWN_FERROMAGNETIC[parametrization])
    difference = ferromagnetic - paramagnetic
    difference_x = ferromagnetic_x - paramagnetic_x
    if parametrization == "rpa":
        epsilon = paramagnetic + difference * f
        epsilon_x = paramagnetic_x + difference_x * f
        epsilon_zeta = difference * f_zeta
    else:
        stiffness, stiffness_x = _vwn_epsilon_partials(x, _VWN_SPIN_STIFFNESS)
        z3 = zeta**3
        z4 = zeta * z3
        epsilon = paramagnetic + stiffness * f * (1.0 - z4) / _VWN_F_SECOND + difference * f * z4
        epsilon_x = paramagnetic_x + stiffness_x * f * (1.0 - z4) / _VWN_F_SECOND + difference_x * f * z4
        epsilon_zeta = (
            stiffness * (f_zeta * (1.0 - z4) - 4.0 * z3 * f) / _VWN_F_SECOND
            + difference * (f_zeta * z4 + 4.0 * z3 * f)
        )
    common = epsilon - x * epsilon_x / 6.0
    zero = np.zeros_like(n)
    return (
        n * epsilon,
        common + (1.0 - zeta) * epsilon_zeta,
        common - (1.0 + zeta) * epsilon_zeta,
        zero,
        zero,
        zero,
    )


# ---------------------------------------------------------------------------
# LYP
# ---------------------------------------------------------------------------


def lyp_correlation_partials(n, sigma, omega=None):
    """Closed-shell LYP ``(e, de/dn, de/dsigma)`` of a positive density (module docstring).

    ``e = -A n/D - A B P [C_F n - u**5 sigma k]`` with ``P = exp(-c u)/D``,
    ``k = 1/24 + 7 delta/72``; ``dP/dn = P delta/(3n)``.
    """

    u = 1.0 / np.cbrt(n)
    denominator = 1.0 + _LYP_D * u
    delta = _LYP_C * u + _LYP_D * u / denominator
    delta_n = -(delta - (_LYP_D * u / denominator) ** 2) / (3.0 * n)
    p = np.exp(-_LYP_C * u) / denominator
    p_n = p * delta / (3.0 * n)
    u5 = u**5
    k = 1.0 / 24.0 + 7.0 * delta / 72.0
    bracket = _LYP_CF * n - u5 * sigma * k
    bracket_n = _LYP_CF - sigma * (-(5.0 / 3.0) * u5 * k / n + u5 * (7.0 / 72.0) * delta_n)
    ab = _LYP_A * _LYP_B
    energy = -_LYP_A * n / denominator - ab * p * bracket
    d_n = -_LYP_A * (1.0 / denominator + _LYP_D * u / (3.0 * denominator * denominator)) - ab * (
        p_n * bracket + p * bracket_n
    )
    d_sigma = ab * p * u5 * k
    return energy, d_n, d_sigma


def lyp_spin_correlation_partials(n, a, b, s_aa, s_ab, s_bb):
    """LYP of two spin densities ``(e, e_a, e_b, e_saa, e_sab, e_sbb)`` where ``n = a + b > 0``.

    The Miehlich form of :func:`parsec_python.V_xc.b3lyp.lyp_correlation_energy_density_spin`,
    ``e = -4A a b/(n D) - A B omega W``, rewritten with
    ``|grad n|**2 = s_aa + 2 s_ab + s_bb`` so that the gradient terms
    do not cancel for an almost empty channel:

    ``W = a b Br + G``,
    ``Br = 2**(11/3) C_F (a**(8/3) + b**(8/3)) + (47 - 7 delta)/9 s_ab
         + (1/9 - delta/3)(s_aa + s_bb) - (delta - 11)/9 (a s_aa + b s_bb)/n``,
    ``G = -(4/3) n**2 s_ab - a**2 s_bb - b**2 s_aa``.
    """

    u = 1.0 / np.cbrt(n)
    denominator = 1.0 + _LYP_D * u
    delta = _LYP_C * u + _LYP_D * u / denominator
    delta_n = -(delta - (_LYP_D * u / denominator) ** 2) / (3.0 * n)
    omega_lyp = np.exp(-_LYP_C * u) / denominator * u**11
    omega_n = -omega_lyp * (11.0 - delta) / (3.0 * n)
    ab = a * b
    cf = _LYP_CF_SPIN
    weighted = (a * s_aa + b * s_bb) / n
    same = s_aa + s_bb
    mixed = (47.0 - 7.0 * delta) / 9.0
    diagonal = 1.0 / 9.0 - delta / 3.0
    eleven = (delta - 11.0) / 9.0
    bracket = cf * (a ** (8.0 / 3.0) + b ** (8.0 / 3.0)) + mixed * s_ab + diagonal * same - eleven * weighted
    n2 = n * n
    g = -(4.0 / 3.0) * n2 * s_ab - a * a * s_bb - b * b * s_aa
    w = ab * bracket + g
    # d bracket/d(a or b) at fixed sigmas: through n (delta) and the explicit terms.
    common_bracket = delta_n * (-(7.0 / 9.0) * s_ab - same / 3.0 - weighted / 9.0) + eleven * weighted / n
    bracket_a = (8.0 / 3.0) * cf * a ** (5.0 / 3.0) + common_bracket - eleven * s_aa / n
    bracket_b = (8.0 / 3.0) * cf * b ** (5.0 / 3.0) + common_bracket - eleven * s_bb / n
    g_a = -(8.0 / 3.0) * n * s_ab - 2.0 * a * s_bb
    g_b = -(8.0 / 3.0) * n * s_ab - 2.0 * b * s_aa
    w_a = b * bracket + ab * bracket_a + g_a
    w_b = a * bracket + ab * bracket_b + g_b

    nd = n * denominator
    nd_n = denominator - _LYP_D * u / 3.0
    first = -4.0 * _LYP_A * ab / nd
    first_n = 4.0 * _LYP_A * ab * nd_n / (nd * nd)
    prefactor = _LYP_A * _LYP_B
    energy = first - prefactor * omega_lyp * w
    e_a = -4.0 * _LYP_A * b / nd + first_n - prefactor * (omega_n * w + omega_lyp * w_a)
    e_b = -4.0 * _LYP_A * a / nd + first_n - prefactor * (omega_n * w + omega_lyp * w_b)
    e_sab = -prefactor * omega_lyp * (ab * mixed - (4.0 / 3.0) * n2)
    e_saa = -prefactor * omega_lyp * (ab * (diagonal - eleven * a / n) - b * b)
    e_sbb = -prefactor * omega_lyp * (ab * (diagonal - eleven * b / n) - a * a)
    return energy, e_a, e_b, e_saa, e_sab, e_sbb


__all__ = [
    "b88_exchange_partials",
    "b88_spin_channel",
    "lyp_correlation_partials",
    "lyp_spin_correlation_partials",
    "slater_exchange_partials",
    "vwn_correlation_partials",
    "vwn_spin_correlation_partials",
]
