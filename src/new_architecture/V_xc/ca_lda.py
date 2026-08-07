"""Spin-unpolarized CA/PZ local-density exchange and correlation.

This is the scalar ``Correlation_Type=ca`` branch of PARSEC's
``exc_nspn.f90``.  At every real-space grid point it evaluates

``rho_bar = rho_valence + rho_core``

and returns both the energy per electron ``epsilon_xc(rho_bar)`` and its
density derivative

``V_xc = d[rho_bar*epsilon_xc(rho_bar)]/d rho_bar``.

The optional core density is the frozen nonlinear-core correction (NLCC), not
an additional set of Kohn--Sham electrons.  ``rho_valence`` is the total
spin-summed density ``rho_up + rho_down``, not a one-spin density.  Densities
are in electrons/bohr^3 and all energies and potentials are in Rydberg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class XCResult:
    """Pointwise CA-LDA fields and their real-space integral.

    ``potential[i]`` is ``V_xc(r_i)`` and is the quantity placed on the
    diagonal of the Kohn--Sham Hamiltonian.  ``energy_per_electron`` is
    ``epsilon_xc`` in Rydberg/electron; multiplying it by the density gives
    ``energy_density`` in Rydberg/bohr^3.  ``total_energy`` is in Rydberg and
    is the uniform-grid integral
    ``volume_element*sum(energy_density)``.
    """

    potential: np.ndarray
    energy_per_electron: np.ndarray
    energy_density: np.ndarray
    total_energy: float


def ca_lda(
    valence_density: np.ndarray,
    volume_element: float,
    core_density: np.ndarray | None = None,
) -> XCResult:
    """Evaluate the CA/PZ local-density functional on a uniform grid.

    For the total spin-summed density ``rho_bar`` define the Wigner--Seitz
    radius

    ``r_s = [3/(4*pi*rho_bar)]^(1/3)``.

    The function evaluates the exchange and Perdew--Zunger correlation
    parameterizations point by point, then returns

    ``epsilon_xc = epsilon_x + epsilon_c``

    ``V_xc = V_x + V_c
           = d[rho_bar*epsilon_xc]/d rho_bar``

    ``E_xc = volume_element*sum(rho_bar*epsilon_xc)``.

    ``valence_density`` is the density obtained from occupied Kohn--Sham
    states.  When an NLCC ``core_density`` is supplied, PARSEC evaluates both
    ``V_xc`` and ``E_xc`` at ``rho_bar = rho_valence + rho_core``.  The core
    remains frozen: it is excluded from the electron count and Hartree source,
    and total-energy ``rho*V_xc`` terms still use valence density only.
    Because ``rho_core`` is fixed, differentiating ``E_xc`` with respect to
    ``rho_valence`` gives the same pointwise derivative evaluated at
    ``rho_bar``.
    """
    valence = np.asarray(valence_density, dtype=float)
    # A nonlinear core correction changes the density seen locally by XC.
    # It does not change the self-consistent valence density itself.
    if core_density is None:
        density = valence
    else:
        core = np.asarray(core_density, dtype=float)
        if core.shape != valence.shape:
            raise ValueError("core and valence densities must have the same shape")
        density = valence + core
    if np.any(density < -1.0e-14):
        raise ValueError("CA-LDA requires a nonnegative density")

    # PARSEC warns about negative density and assigns zero XC to nonpositive
    # points.  Here a materially negative value is treated as invalid, while
    # tiny roundoff-level negative values follow the same zero branch.
    potential = np.zeros_like(density)
    epsilon = np.zeros_like(density)
    positive = density > 0.0
    if np.any(positive):
        rho = density[positive]
        # r_s is the radius of a sphere containing one electron at density
        # rho: (4*pi/3)*r_s^3*rho = 1.
        rs = (0.75 / (np.pi * rho)) ** (1.0 / 3.0)

        # Unpolarized Dirac exchange in Rydberg units.  If
        # epsilon_x = -(3/2)/(pi*a0*r_s), then
        # V_x = d(rho*epsilon_x)/d rho = (4/3)*epsilon_x.
        a0 = (4.0 / (9.0 * np.pi)) ** (1.0 / 3.0)
        exchange_potential = -2.0 / (np.pi * a0 * rs)
        exchange_epsilon = 0.75 * exchange_potential

        correlation_epsilon = np.empty_like(rs)
        correlation_potential = np.empty_like(rs)

        # Perdew--Zunger's low-density branch, r_s >= 1.  The numerical
        # coefficients are already doubled from Hartree to Rydberg.  V_c is
        # obtained from the thermodynamic derivative
        # V_c = epsilon_c - (r_s/3)*d epsilon_c/d r_s.
        low_density = rs >= 1.0
        if np.any(low_density):
            r = rs[low_density]
            sqrt_r = np.sqrt(r)
            g = -0.2846
            b1 = 1.0529
            b2 = 0.3334
            ec = g / (1.0 + b1 * sqrt_r + b2 * r)
            vc = (ec * ec / g) * (
                1.0 + (7.0 / 6.0) * b1 * sqrt_r + (4.0 / 3.0) * b2 * r
            )
            correlation_epsilon[low_density] = ec
            correlation_potential[low_density] = vc

        # Logarithmic high-density branch, r_s < 1, with the same analytic
        # derivative used for the correlation potential.
        high_density = ~low_density
        if np.any(high_density):
            r = rs[high_density]
            log_r = np.log(r)
            c1, c2, c3, c4, c5 = 0.0622, 0.096, 0.004, 0.0232, 0.0192
            ec = c1 * log_r - c2 + (c3 * log_r - c4) * r
            vc = ec - (c1 + (c3 * log_r - c5) * r) / 3.0
            correlation_epsilon[high_density] = ec
            correlation_potential[high_density] = vc

        # Both arrays are pointwise functions of the same total density.
        epsilon[positive] = exchange_epsilon + correlation_epsilon
        potential[positive] = exchange_potential + correlation_potential

    # This is rho_bar*epsilon_xc, not merely epsilon_xc.  Multiplication by
    # the uniform cell volume performs h^3 real-space quadrature.  PARSEC also
    # applies symmetry multiplicities to a reduced wedge; this Python solver
    # stores the complete active grid, so no additional multiplicity appears.
    energy_density = density * epsilon
    return XCResult(
        potential=potential,
        energy_per_electron=epsilon,
        energy_density=energy_density,
        total_energy=float(np.sum(energy_density) * volume_element),
    )


__all__ = ["XCResult", "ca_lda"]
