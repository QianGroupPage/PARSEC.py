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
class SpinPolarizedXCResult:
    """Pointwise spin-polarized CA-LDA (LSDA) fields and their integral.

    ``potential_up``/``potential_down`` are ``V_xc,up(r_i)``/``V_xc,down(r_i)``,
    the quantities placed on the diagonal of each spin channel's Kohn--Sham
    Hamiltonian.  ``total_energy`` is in Rydberg and is the uniform-grid
    integral of the pointwise exchange-correlation energy density.
    """

    potential_up: np.ndarray
    potential_down: np.ndarray
    energy_density: np.ndarray
    total_energy: float


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


def ca_lda_spin_polarized(
    density_up: np.ndarray,
    density_down: np.ndarray,
    volume_element: float,
    core_density: np.ndarray | None = None,
) -> SpinPolarizedXCResult:
    """Evaluate the spin-polarized (LSDA) CA/PZ functional, PARSEC's ``exc_spn.f90``.

    For spin densities ``rho_up``, ``rho_down`` define the total density
    ``rho_bar = rho_up + rho_down`` (plus any frozen NLCC ``core_density``,
    split evenly so the total matches the unpolarized convention) and the
    relative spin polarization ``zeta = (rho_up - rho_down)/rho_bar``.
    Exchange is evaluated on each spin channel directly via the exact
    spin-scaling relation; Perdew-Zunger correlation is evaluated at the
    unpolarized (``zeta=0``) and fully polarized (``zeta=1``) limits and
    combined with the standard interpolation

    ``f(zeta) = [(1+zeta)^(4/3) + (1-zeta)^(4/3) - 2] / gamma``,
    ``gamma = 2^(4/3) - 2``,

    ``ec(zeta) = ec_u + f(zeta)*(ec_p - ec_u)``.

    Potentials are obtained from the same thermodynamic derivative used by
    the unpolarized branch, differentiated per spin channel.  This mirrors
    ``exc_spn.f90``'s ``icorr == 'ca'`` branch exactly, including its
    asymmetric-looking (but numerically equivalent to the unpolarized
    high-density branch) regrouping of the Perdew-Zunger coefficients.
    """
    rho_up = np.asarray(density_up, dtype=float)
    rho_down = np.asarray(density_down, dtype=float)
    if rho_up.shape != rho_down.shape:
        raise ValueError("spin densities must have the same shape")
    if core_density is not None:
        core = np.asarray(core_density, dtype=float)
        if core.shape != rho_up.shape:
            raise ValueError("core and valence densities must have the same shape")
        # Split the frozen core density evenly between channels, matching
        # PARSEC's unpolarized convention of adding rho_core to the total
        # density seen by XC while leaving the spin split (zeta) determined
        # by the valence density alone.
        rho_up = rho_up + 0.5 * core
        rho_down = rho_down + 0.5 * core

    potential_up = np.zeros_like(rho_up)
    potential_down = np.zeros_like(rho_down)
    energy_density = np.zeros_like(rho_up)

    positive = (rho_up > 0.0) & (rho_down > 0.0)
    if np.any(positive):
        rh1 = rho_up[positive]
        rh2 = rho_down[positive]
        total = rh1 + rh2

        # Dirac exchange, Rydberg units, evaluated directly on each spin
        # channel via the exact spin-scaling relation (exc_spn.f90's ax).
        ax = -0.738558766382022405884230032680836
        ex1 = ax * (2.0 * rh1) ** (1.0 / 3.0)
        vx1 = (4.0 / 3.0) * ex1
        ex2 = ax * (2.0 * rh2) ** (1.0 / 3.0)
        vx2 = (4.0 / 3.0) * ex2

        rs = (0.75 / np.pi / total) ** (1.0 / 3.0)
        zeta = (rh1 - rh2) / total
        gamma = 0.5198421
        f = ((1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0) - 2.0) / gamma
        fz = (
            (4.0 / 3.0)
            * ((1.0 + zeta) ** (1.0 / 3.0) - (1.0 - zeta) ** (1.0 / 3.0))
            / gamma
        )

        # Perdew-Zunger correlation: unpolarized ("u") and fully polarized
        # ("s", zeta=1) branches, each with their own high/low-density forms.
        c1, c2, c3, c4 = 0.0622, 0.0960, 0.0040, 0.0232
        g, b1, b2 = -0.2846, 1.0529, 0.3334
        as_, bs, cs, ds = 0.0311, -0.0538, 0.0014, -0.0096
        gs, b1s, b2s = -0.1686, 1.3981, 0.2611

        eu = np.empty_like(rs)
        vc_u = np.empty_like(rs)
        es = np.empty_like(rs)
        vc_s = np.empty_like(rs)

        high_density = rs < 1.0
        if np.any(high_density):
            r = rs[high_density]
            lrs = np.log(r)
            eu[high_density] = c1 * lrs - c2 + (c3 * lrs - c4) * r
            vc_u[high_density] = eu[high_density] - (
                c1 + (c3 * lrs + c3 - c4) * r
            ) / 3.0
            es[high_density] = as_ * lrs + bs + (cs * lrs + ds) * r
            vc_s[high_density] = es[high_density] - (
                as_ + (cs * lrs + cs + ds) * r
            ) / 3.0

        low_density = ~high_density
        if np.any(low_density):
            r = rs[low_density]
            srs = np.sqrt(r)
            eu[low_density] = g / (1.0 + b1 * srs + b2 * r)
            vc_u[low_density] = (
                eu[low_density]
                * eu[low_density]
                * (1.0 + (7.0 / 6.0) * b1 * srs + (4.0 / 3.0) * b2 * r)
                / g
            )
            es[low_density] = gs / (1.0 + b1s * srs + b2s * r)
            vc_s[low_density] = (
                es[low_density]
                * es[low_density]
                * (1.0 + (7.0 / 6.0) * b1s * srs + (4.0 / 3.0) * b2s * r)
                / gs
            )

        vc_down = vc_u + f * (vc_s - vc_u) - (es - eu) * (1.0 + zeta) * fz
        vc_up = vc_u + f * (vc_s - vc_u) + (es - eu) * (1.0 - zeta) * fz
        ec = eu + f * (es - eu)

        potential_up[positive] = 2.0 * vx1 + vc_up
        potential_down[positive] = 2.0 * vx2 + vc_down
        energy_density[positive] = 2.0 * (rh1 * ex1 + rh2 * ex2) + ec * total

    return SpinPolarizedXCResult(
        potential_up=potential_up,
        potential_down=potential_down,
        energy_density=energy_density,
        total_energy=float(np.sum(energy_density) * volume_element),
    )


__all__ = ["XCResult", "SpinPolarizedXCResult", "ca_lda", "ca_lda_spin_polarized"]
