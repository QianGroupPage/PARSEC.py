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
    band_energy_weight: float = 1.0,
) -> EnergyBreakdown:
    """Evaluate the input-potential/new-density PARSEC energy expression.

    ``band_energy_weight`` scales the ``2*dot(occ,eig)`` band-energy term
    only (everything else is a density-dotted integral and is already
    correct once ``density`` itself is built consistently).  The default 1
    is the ordinary single-Gamma-point case; k-point sampling
    (:mod:`~parsec_python.SCF.kpoints`) passes each k-point's Brillouin-zone
    weight here, since ``eigenvalues``/``occupations`` there are pooled
    across k-points without that weight otherwise applied.
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

    band_energy = float(
        2.0 * band_energy_weight * np.dot(occupations, eigenvalues)
    ) + float(alpha_z_energy)
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
    band_energy_weight: float = 1.0,
) -> EnergyBreakdown:
    """:func:`total_energy` for orbitals holding at most one electron.

    Used by :mod:`~parsec_python.SCF.self_consistent_soc`, whose
    ``2*n_grid``-length spinor eigenvectors already have spin folded in (no
    separate spin-degeneracy factor of two on the band energy, unlike the
    ordinary scalar path's two-electron orbitals) but still share one
    spin-unpolarized ``V_eff``/density (unlike
    :func:`total_energy_spin_polarized`'s two separate potential channels).
    ``band_energy_weight`` is :func:`total_energy`'s same per-k-point
    Brillouin-zone weight, for :mod:`~parsec_python.SCF.kpoints_soc`'s
    periodic self-consistent SOC (default 1: the ordinary Gamma-only case).
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

    band_energy = float(
        band_energy_weight * np.dot(occupations, eigenvalues)
    ) + float(alpha_z_energy)
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


def total_energy_no_degeneracy_spin_polarized(
    eigenvalues: np.ndarray,
    occupations: np.ndarray,
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
    band_energy_weight: float = 1.0,
) -> EnergyBreakdown:
    """:func:`total_energy_no_degeneracy` with :func:`total_energy_spin_polarized`'s
    spin-resolved Hxc bookkeeping.

    Used by self-consistent SOC combined with collinear spin polarization
    (``SCF.self_consistent_soc_spin_polarized``): the pooled spinor
    eigenvalues already hold at most one electron each (no factor of two on
    the band energy, matching :func:`total_energy_no_degeneracy`), but the
    density and its Hxc double-counting term are spin-resolved (matching
    :func:`total_energy_spin_polarized`) since ``V_xc,up != V_xc,down`` once
    the spinor density's up/down projections are unequal.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    occupations = np.asarray(occupations, dtype=float)
    density_up = np.asarray(density_up, dtype=float)
    density_down = np.asarray(density_down, dtype=float)
    if occupations.shape != eigenvalues.shape:
        raise ValueError("eigenvalues and occupations must have the same shape")
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

    band_energy = float(
        band_energy_weight * np.dot(occupations, eigenvalues)
    ) + float(alpha_z_energy)
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


def total_energy_noncollinear(
    eigenvalues: np.ndarray,
    occupations: np.ndarray,
    density: np.ndarray,
    magnetization: np.ndarray,
    input_potential: np.ndarray,
    input_field: np.ndarray,
    ionic_potential: np.ndarray,
    output_hartree_potential: np.ndarray,
    output_xc_potential: np.ndarray,
    output_xc_field: np.ndarray,
    exchange_correlation_energy: float,
    ion_ion_energy: float,
    volume_element: float,
    alpha_z_energy: float = 0.0,
    band_energy_weight: float = 1.0,
) -> EnergyBreakdown:
    """Total energy for a non-collinear spinor calculation.

    The Hxc potential is ``V_avg * 1 + B . sigma``, so the double-counting
    and ``Integral V_xc rho`` terms carry both the charge and the
    magnetization parts: ``int (n V_avg + m . B)``.  With ``m`` along z and
    ``B = (V_up - V_down)/2 z`` this equals
    :func:`total_energy_no_degeneracy_spin_polarized` exactly (``n_up V_up +
    n_down V_down = n V_avg + m_z B_z``).  Each pooled eigenvalue holds at
    most one electron (no factor of two), as for the other spinor energies.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    occupations = np.asarray(occupations, dtype=float)
    density = np.asarray(density, dtype=float)
    magnetization = np.asarray(magnetization, dtype=float)
    input_field = np.asarray(input_field, dtype=float)
    output_xc_field = np.asarray(output_xc_field, dtype=float)
    if occupations.shape != eigenvalues.shape:
        raise ValueError("eigenvalues and occupations must have the same shape")
    scalar_fields = (
        input_potential,
        ionic_potential,
        output_hartree_potential,
        output_xc_potential,
    )
    if any(np.asarray(value).shape != density.shape for value in scalar_fields):
        raise ValueError("all scalar fields must share the density's shape")
    vector_fields = (magnetization, input_field, output_xc_field)
    if any(value.shape != (3, density.size) for value in vector_fields):
        raise ValueError("magnetization and B fields must have shape (3, n_grid)")

    band_energy = float(band_energy_weight * np.dot(occupations, eigenvalues)) + float(
        alpha_z_energy
    )
    old_hxc_integral = float(
        volume_element
        * (
            np.dot(density, np.asarray(input_potential) - np.asarray(ionic_potential))
            + np.sum(magnetization * input_field)
        )
    )
    hartree_integral = float(
        volume_element * np.dot(density, output_hartree_potential)
    )
    vxc_integral = float(
        volume_element
        * (
            np.dot(density, output_xc_potential)
            + np.sum(magnetization * output_xc_field)
        )
    )
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


__all__ = [
    "total_energy",
    "total_energy_no_degeneracy",
    "total_energy_spin_polarized",
    "total_energy_no_degeneracy_spin_polarized",
    "total_energy_noncollinear",
]
