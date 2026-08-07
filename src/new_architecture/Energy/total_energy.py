"""PARSEC ``totnrg`` energy bookkeeping for the scalar isolated case."""

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

    band_energy = float(2.0 * np.dot(occupations, eigenvalues))
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


__all__ = ["total_energy"]
