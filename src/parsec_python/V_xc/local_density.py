"""Slater X-alpha, Wigner and Hedin-Lundqvist local density functionals.

These are the ``Correlation_Type`` values ``xa``, ``wi`` and ``hl`` of the
unpolarized ``exc_nspn.f90`` (lines 198-245).  With
``r_s = [3/(4 pi rho)]**(1/3)`` and the Dirac exchange potential
``v_x = -2/(pi a0 r_s)``, ``a0 = (4/(9 pi))**(1/3)`` (Rydberg), whose energy
per electron is ``3 v_x / 4``:

* X-alpha [J. C. Slater, Phys. Rev. 81, 385 (1951)], ``alpha = 1`` as the
  Fortran hard-codes it (``alpha = 2/3`` is pure Dirac exchange):
  ``V = (3/2) alpha v_x``, ``e = (3/4) V``.
* Wigner's interpolation [E. Wigner, Phys. Rev. 46, 1002 (1934)] with
  ``a_w = -0.875529``, ``b_w = 7.8`` (Rydberg):
  ``e = 3 v_x/4 + a_w/(r_s + b_w)``,
  ``V = v_x + a_w (4 r_s/3 + b_w)/(r_s + b_w)**2``.
* Hedin-Lundqvist [L. Hedin and B. I. Lundqvist, J. Phys. C 4, 2064
  (1971)] with ``A = 21``, ``C = 0.045`` (Rydberg), ``x = r_s/A``:
  ``e = 3 v_x/4 - C [(1 + x**3) ln(1 + 1/x) + x/2 - x**2 - 1/3]``,
  ``V = v_x - C ln(1 + 1/x)``.

Each ``V`` is ``d(rho e)/d rho``.  The functionals are evaluated at
``rho = rho_valence + rho_core`` (frozen nonlinear-core density), and points
with ``rho <= 0`` carry no potential and no energy.

Fortran has no spin-polarized form of them: ``exc_spn.f90`` falls through
all its branches for these labels and returns the densities as potentials
and zero energy.  The input parser therefore refuses them together with spin
polarization or non-collinear magnetism.

One point of the Fortran is not translated.  Its Wigner energy multiplies
the exchange part by the order of the symmetry group of the irreducible
wedge but not the correlation part (``exc_nspn.f90``:225), so with symmetry
the correlation energy it reports is divided by that order.  This package
works on the complete grid and integrates the whole energy density with
weight one, which is the energy of the functional; the potential, and hence
the self-consistent density, is the same in both.  A Fortran run with
``Ignore_Symmetry: .true.`` (a group of order one) reports this energy.

Densities are in electrons/bohr^3, energies and potentials in Rydberg.
"""

from __future__ import annotations

import numpy as np

from .ca_lda import XCResult
from .pw92_lda import _local_result, _total_density, dirac_exchange_potential


#: exc_nspn.f90: Slater's alpha (hard-coded).
XALPHA_ALPHA = 1.0
# Wigner parameters (Rydberg).
_WIGNER_A = -0.875529
_WIGNER_B = 7.8
# Hedin-Lundqvist parameters (A = 1/21 enters as a factor, C in Rydberg).
_HL_INVERSE_A = 1.0 / 21.0
_HL_C = 0.045


def _evaluate(valence_density, volume_element, core_density, name, pointwise) -> XCResult:
    density = _total_density(valence_density, core_density, name)
    potential = np.zeros_like(density)
    energy_density = np.zeros_like(density)
    positive = density > 0.0
    if np.any(positive):
        rho = density[positive]
        rs = np.cbrt((0.75 / np.pi) / rho)
        point_potential, point_energy = pointwise(rs)
        potential[positive] = point_potential
        energy_density[positive] = rho * point_energy
    return _local_result(density, potential, energy_density, volume_element)


def _xalpha_point(rs):
    potential = 1.5 * XALPHA_ALPHA * dirac_exchange_potential(rs)
    return potential, 0.75 * potential


def _wigner_point(rs):
    exchange = dirac_exchange_potential(rs)
    shifted = rs + _WIGNER_B
    potential = exchange + _WIGNER_A * (4.0 * rs / 3.0 + _WIGNER_B) / (shifted * shifted)
    return potential, 0.75 * exchange + _WIGNER_A / shifted


def _hedin_lundqvist_point(rs):
    exchange = dirac_exchange_potential(rs)
    x = rs * _HL_INVERSE_A
    logarithm = np.log1p(1.0 / x)
    potential = exchange - _HL_C * logarithm
    energy = 0.75 * exchange - _HL_C * (
        (1.0 + x * x * x) * logarithm + 0.5 * x - x * x - 1.0 / 3.0
    )
    return potential, energy


def xalpha_lda(
    valence_density: np.ndarray,
    volume_element: float,
    core_density: np.ndarray | None = None,
) -> XCResult:
    """Slater X-alpha exchange (``alpha = 1``), PARSEC's ``xa``."""

    return _evaluate(valence_density, volume_element, core_density, "X-alpha", _xalpha_point)


def wigner_lda(
    valence_density: np.ndarray,
    volume_element: float,
    core_density: np.ndarray | None = None,
) -> XCResult:
    """Dirac exchange with Wigner's correlation, PARSEC's ``wi``."""

    return _evaluate(valence_density, volume_element, core_density, "Wigner", _wigner_point)


def hedin_lundqvist_lda(
    valence_density: np.ndarray,
    volume_element: float,
    core_density: np.ndarray | None = None,
) -> XCResult:
    """Dirac exchange with Hedin-Lundqvist correlation, PARSEC's ``hl``."""

    return _evaluate(
        valence_density, volume_element, core_density, "Hedin-Lundqvist", _hedin_lundqvist_point
    )


__all__ = ["XALPHA_ALPHA", "hedin_lundqvist_lda", "wigner_lda", "xalpha_lda"]
