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


def total_energy_no_degeneracy(
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
    """:func:`total_energy` for orbitals holding at most one electron.

    Used by :mod:`~parsec_python.SCF.self_consistent_soc`, whose
    ``2*n_grid``-length spinor eigenvectors already have spin folded in (no
    separate spin-degeneracy factor of two on the band energy, unlike the
    ordinary scalar path's two-electron orbitals) but still share one
    spin-unpolarized ``V_eff``/density (unlike
    :func:`total_energy_spin_polarized`'s two separate potential channels).
    """
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

    band_energy = float(np.dot(occupations, eigenvalues)) + float(alpha_z_energy)
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


def total_energy_spin_polarized(
    eigenvalues_up: np.ndarray,
    occupations_up: np.ndarray,
    eigenvalues_down: np.ndarray,
    occupations_down: np.ndarray,
    density_up: np.ndarray,
    density_down: np.ndarray,
    input_effective_potential_up: np.ndarray,
    input_effective_potential_down: np.ndarray,
    ionic_potential: np.ndarray,
    output_hartree_potential: np.ndarray,
    output_xc_potential_up: np.ndarray,
    output_xc_potential_down: np.ndarray,
    exchange_correlation_energy: float,
    ion_ion_energy: float,
    volume_element: float,
    alpha_z_energy: float = 0.0,
) -> EnergyBreakdown:
    """Spin-resolved analog of :func:`total_energy`.

    Each spin channel's orbitals hold at most one electron (no spin
    degeneracy), so unlike the unpolarized ``2*dot(occ, eig)`` band energy,
    here the two channels' band energies simply add.  The Hartree potential
    is spin-independent and dotted against the total density; the local
    ionic and Hxc-double-counting terms are dotted against each channel's own
    density and then summed, matching Fortran PARSEC's spin-resolved
    ``totnrg``.
    """
    eigenvalues_up = np.asarray(eigenvalues_up, dtype=float)
    eigenvalues_down = np.asarray(eigenvalues_down, dtype=float)
    occupations_up = np.asarray(occupations_up, dtype=float)
    occupations_down = np.asarray(occupations_down, dtype=float)
    density_up = np.asarray(density_up, dtype=float)
    density_down = np.asarray(density_down, dtype=float)
    if occupations_up.shape != eigenvalues_up.shape:
        raise ValueError("spin-up eigenvalues and occupations must have the same shape")
    if occupations_down.shape != eigenvalues_down.shape:
        raise ValueError("spin-down eigenvalues and occupations must have the same shape")
    per_channel_arrays = (
        density_up,
        density_down,
        input_effective_potential_up,
        input_effective_potential_down,
        ionic_potential,
        output_hartree_potential,
        output_xc_potential_up,
        output_xc_potential_down,
    )
    if any(np.asarray(value).shape != density_up.shape for value in per_channel_arrays):
        raise ValueError("all spin-resolved fields must share the density's shape")

    band_energy = (
        float(np.dot(occupations_up, eigenvalues_up))
        + float(np.dot(occupations_down, eigenvalues_down))
        + float(alpha_z_energy)
    )
    old_hxc_up = np.asarray(input_effective_potential_up) - np.asarray(ionic_potential)
    old_hxc_down = np.asarray(input_effective_potential_down) - np.asarray(ionic_potential)
    old_hxc_integral = float(
        volume_element
        * (np.dot(density_up, old_hxc_up) + np.dot(density_down, old_hxc_down))
    )
    density_total = density_up + density_down
    hartree_integral = float(
        volume_element * np.dot(density_total, output_hartree_potential)
    )
    vxc_integral = float(
        volume_element
        * (
            np.dot(density_up, output_xc_potential_up)
            + np.dot(density_down, output_xc_potential_down)
        )
    )
    electron_ion = float(volume_element * np.dot(density_total, ionic_potential))
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


__all__ = ["total_energy", "total_energy_no_degeneracy", "total_energy_spin_polarized"]
