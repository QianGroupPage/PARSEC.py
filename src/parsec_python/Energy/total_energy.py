"""PARSEC ``totnrg`` energy bookkeeping for the isolated case."""

from __future__ import annotations

import numpy as np

from ..models import EnergyBreakdown


def total_energy(
    eigenvalues: np.ndarray,
    occupations: np.ndarray,
    density: np.ndarray,
    input_effective_potential: np.ndarray,
    ionic_potential: np.ndarray,
    output_hartree_potential: np.ndarray,
    output_xc_potential: np.ndarray,
    exchange_correlation_energy: float,
    ion_ion_energy: float,
    volume_element: float,
    alpha_z_energy: float = 0.0,
) -> EnergyBreakdown:
    """Evaluate the input-potential/new-density PARSEC energy expression."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    occupations = np.asarray(occupations, dtype=float)
    density = np.asarray(density, dtype=float)
    arrays = (
        input_effective_potential,
        ionic_potential,
        output_hartree_potential,
        output_xc_potential,
    )
    if occupations.shape != eigenvalues.shape:
        raise ValueError("eigenvalues and occupations must have the same shape")
    if any(np.asarray(value).shape != density.shape for value in arrays):
        raise ValueError("all potentials must match the density")

    band_energy = float(2.0 * np.dot(occupations, eigenvalues)) + float(alpha_z_energy)
    old_hxc = np.asarray(input_effective_potential) - np.asarray(ionic_potential)
    old_hxc_integral = float(volume_element * np.dot(density, old_hxc))
    hartree_integral = float(
        volume_element * np.dot(density, output_hartree_potential)
    )
    vxc_integral = float(volume_element * np.dot(density, output_xc_potential))
    electron_ion = float(volume_element * np.dot(density, ionic_potential))
    electronic = float(
        band_energy
        - old_hxc_integral
        + 0.5 * hartree_integral
        + exchange_correlation_energy
    )
    return EnergyBreakdown(
        eigenvalue=band_energy,
        hartree=0.5 * hartree_integral,
        integral_vxc_rho=vxc_integral,
        exchange_correlation=float(exchange_correlation_energy),
        electron_ion=electron_ion,
        ion_ion=float(ion_ion_energy),
        electronic=electronic,
        total=electronic + float(ion_ion_energy),
    )


def spin_total_energy(
    eigenvalues: tuple[np.ndarray, np.ndarray],
    occupations: tuple[np.ndarray, np.ndarray],
    densities: tuple[np.ndarray, np.ndarray],
    input_effective_potentials: tuple[np.ndarray, np.ndarray],
    ionic_potential: np.ndarray,
    output_hartree_potential: np.ndarray,
    output_xc_potentials: tuple[np.ndarray, np.ndarray],
    exchange_correlation_energy: float,
    ion_ion_energy: float,
    volume_element: float,
) -> EnergyBreakdown:
    """Evaluate ``totnrg.f90`` for two spin channels, up first.

    A state of a channel holds one electron, so the band energy is
    ``sum_s sum_i f_s,i e_s,i`` without the factor two of the unpolarized
    expression.  The Hartree and the electron-ion terms read the total
    density; the input-potential and the ``V_xc`` integrals are summed over
    the channels, each with its own density and potential.
    """

    ionic_potential = np.asarray(ionic_potential, dtype=float)
    hartree = np.asarray(output_hartree_potential, dtype=float)
    band_energy = 0.0
    old_hxc_integral = 0.0
    vxc_integral = 0.0
    for values, weights, density, input_potential, xc_potential in zip(
        eigenvalues,
        occupations,
        densities,
        input_effective_potentials,
        output_xc_potentials,
    ):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        density = np.asarray(density, dtype=float)
        if weights.shape != values.shape:
            raise ValueError("eigenvalues and occupations must have the same shape")
        if any(
            np.asarray(field).shape != density.shape
            for field in (input_potential, xc_potential, ionic_potential, hartree)
        ):
            raise ValueError("all potentials must match the density")
        band_energy += float(np.dot(weights, values))
        old_hxc_integral += float(
            volume_element
            * np.dot(density, np.asarray(input_potential) - ionic_potential)
        )
        vxc_integral += float(volume_element * np.dot(density, xc_potential))
    total_density = np.asarray(densities[0], dtype=float) + np.asarray(
        densities[1], dtype=float
    )
    hartree_integral = float(volume_element * np.dot(total_density, hartree))
    electron_ion = float(volume_element * np.dot(total_density, ionic_potential))
    electronic = float(
        band_energy
        - old_hxc_integral
        + 0.5 * hartree_integral
        + exchange_correlation_energy
    )
    return EnergyBreakdown(
        eigenvalue=band_energy,
        hartree=0.5 * hartree_integral,
        integral_vxc_rho=vxc_integral,
        exchange_correlation=float(exchange_correlation_energy),
        electron_ion=electron_ion,
        ion_ion=float(ion_ion_energy),
        electronic=electronic,
        total=electronic + float(ion_ion_energy),
    )


__all__ = ["spin_total_energy", "total_energy"]
