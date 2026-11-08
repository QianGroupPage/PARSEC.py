"""Asymptotically corrected CA-LDA potentials: van Leeuwen-Baerends and Casida-Salahub.

These are the ``Correlation_Type`` values ``lb`` and ``cs`` of
``exc_nspn.f90`` (lines 288-340) and ``exc_spn.f90`` (lines 394-493).  The
energy is the CA/PZ (``ca``) exchange-correlation energy; only the potential
is changed, by the gradient correction of R. van Leeuwen and E. J. Baerends
[Phys. Rev. A 49, 2421 (1994)],

``V_s = V_s^CA - c_s``,
``c_s = 2 beta rho_s**(1/3) x_s**2 / (1 + 3 beta x_s asinh x_s)`` (Rydberg),
``x_s = |grad rho_s| / rho_s**(4/3)``, ``beta = 0.05``,

for each spin channel ``s``; without spin polarization ``rho_s = rho/2``,
that is ``x = 2**(1/3) |grad rho|/rho**(4/3)`` and
``c = 2 beta (rho**(1/3)/2**(1/3)) x**2/(1 + 3 beta x asinh x)``.  The
Casida-Salahub form [M. E. Casida and D. R. Salahub, J. Chem. Phys. 113,
8918 (2000)] caps the correction at the ionization energy difference
``Delta`` (``Ion_Energy_Diff``, Rydberg),

``c_s -> min(c_s, Delta)``, i.e. ``V_s = max(V_s^CA - c_s, V_s^CA - Delta)``,

so that the potential is the CA one shifted down by ``Delta`` inside the
molecule and the LB94 one in its tail.  ``Delta = 0`` (the Fortran default)
gives the CA potential.

The potential is not the derivative of the energy, so an SCF with it has no
variational total energy; Fortran reports the CA energy evaluated with it,
and so does this package.

The gradient is that of ``gradmvs`` (hpotcg.F90:363-473),
:func:`parsec_python.V_xc.pbe.density_gradient`: centered differences of
order ``Expansion_Order``, zero outside an isolated cluster domain,
wraparound on a periodic grid.  The densities include the frozen core
density (all of it without spin, half per channel with spin), and the
correction is applied where the density of the channel is positive.
Densities are in electrons/bohr^3, potentials in Rydberg.
"""

from __future__ import annotations

import numpy as np

from .ca_lda import XCResult, ca_lda
from .ca_lda_spin import SpinXCResult, ca_lda_spin, spin_densities_with_core
from .pbe import density_gradient
from .pw92_lda import _total_density


#: exc_nspn.f90 / exc_spn.f90: beta of the LB94 correction.
LB94_BETA = 0.05


def _channel_correction(
    density: np.ndarray,
    gradient: np.ndarray,
    spin_scale: float,
    cap: float | None,
) -> np.ndarray:
    """``c`` of one channel; ``spin_scale = 2**(1/3)`` for an unpolarized density."""

    correction = np.zeros_like(density)
    positive = density > 0.0
    if np.any(positive):
        rho = density[positive]
        magnitude = np.sqrt(np.einsum("ij,ij->j", gradient[:, positive], gradient[:, positive]))
        rho13 = np.cbrt(rho)
        x = spin_scale * magnitude / (rho * rho13)
        values = (
            2.0 * LB94_BETA * (rho13 / spin_scale) * x * x
            / (1.0 + 3.0 * LB94_BETA * x * np.arcsinh(x))
        )
        if cap is not None:
            values = np.minimum(values, cap)
        correction[positive] = values
    return correction


def lb94_potential_correction(
    density: np.ndarray,
    grid,
    ionization_energy_difference: float | None = None,
) -> np.ndarray:
    """The correction ``c`` (Rydberg) subtracted from the unpolarized CA potential.

    ``density`` is the total density seen by XC (valence plus core).  With
    ``ionization_energy_difference`` (Rydberg) the correction is capped
    there (Casida-Salahub).
    """

    density = np.asarray(density, dtype=np.float64)
    return _channel_correction(
        density,
        density_gradient(density, grid),
        2.0 ** (1.0 / 3.0),
        ionization_energy_difference,
    )


def lb94_spin_potential_corrections(
    density_up: np.ndarray,
    density_down: np.ndarray,
    grid,
    ionization_energy_difference: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The corrections ``c_up``, ``c_down`` (Rydberg) of two spin densities.

    The densities include half of any frozen core density each.
    """

    corrections = []
    for density in (density_up, density_down):
        density = np.asarray(density, dtype=np.float64)
        corrections.append(
            _channel_correction(
                density,
                density_gradient(density, grid),
                1.0,
                ionization_energy_difference,
            )
        )
    return corrections[0], corrections[1]


def lb94(
    valence_density: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
    ionization_energy_difference: float | None = None,
) -> XCResult:
    """CA-LDA energy with the LB94 (or, capped, Casida-Salahub) potential."""

    density = _total_density(valence_density, core_density, "LB94")
    local = ca_lda(valence_density, grid.volume_element, core_density)
    correction = lb94_potential_correction(density, grid, ionization_energy_difference)
    return XCResult(
        potential=local.potential - correction,
        energy_per_electron=local.energy_per_electron,
        energy_density=local.energy_density,
        total_energy=local.total_energy,
    )


def lb94_spin(
    valence_up: np.ndarray,
    valence_down: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
    ionization_energy_difference: float | None = None,
) -> SpinXCResult:
    """CA-LSDA energy with the spin LB94 (or Casida-Salahub) potentials."""

    up, down = spin_densities_with_core(
        valence_up, valence_down, core_density, functional="LB94"
    )
    local = ca_lda_spin(valence_up, valence_down, grid.volume_element, core_density)
    correction_up, correction_down = lb94_spin_potential_corrections(
        up, down, grid, ionization_energy_difference
    )
    return SpinXCResult(
        potential_up=local.potential_up - correction_up,
        potential_down=local.potential_down - correction_down,
        energy_density=local.energy_density,
        total_energy=local.total_energy,
    )


__all__ = [
    "LB94_BETA",
    "lb94",
    "lb94_potential_correction",
    "lb94_spin",
    "lb94_spin_potential_corrections",
]
