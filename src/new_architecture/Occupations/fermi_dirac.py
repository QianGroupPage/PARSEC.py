"""Fermi occupations and real-space density construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BOLTZMANN_RYDBERG_PER_KELVIN = 6.33327186e-6


@dataclass(frozen=True)
class OccupationResult:
    fermi_level: float
    occupations: np.ndarray
    electron_count: float


def _fermi_function(eigenvalues: np.ndarray, chemical_potential: float, kbt: float) -> np.ndarray:
    exponent = (eigenvalues - chemical_potential) / kbt
    result = np.empty_like(exponent)
    result[exponent >= 35.0] = 0.0
    result[exponent <= -35.0] = 1.0
    middle = np.abs(exponent) < 35.0
    result[middle] = 1.0 / (np.exp(exponent[middle]) + 1.0)
    return result


def fermi_occupations(
    eigenvalues: np.ndarray,
    electron_count: float,
    temperature_kelvin: float = 80.0,
    *,
    count_tolerance: float = 1.0e-12,
    max_iterations: int = 100,
    degeneracy_tolerance: float = 1.0e-12,
) -> OccupationResult:
    """Occupy spin-degenerate states with PARSEC's ``f_i in [0,1]`` convention."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 1 or eigenvalues.size == 0:
        raise ValueError("eigenvalues must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("eigenvalues must be finite")
    if np.any(np.diff(eigenvalues) < 0):
        raise ValueError("eigenvalues must be sorted")
    if (
        not np.isfinite(electron_count)
        or electron_count < 0
        or electron_count > 2.0 * eigenvalues.size
    ):
        raise ValueError("electron count exceeds the available spin-degenerate states")
    if (
        not np.isfinite(count_tolerance)
        or count_tolerance <= 0
        or not np.isfinite(degeneracy_tolerance)
        or degeneracy_tolerance < 0
    ):
        raise ValueError("occupation tolerances must be finite and nonnegative")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(temperature_kelvin):
        raise ValueError("temperature must be finite")
    target = 0.5 * float(electron_count)

    if temperature_kelvin == 0:
        occupations = np.zeros_like(eigenvalues)
        remaining = target
        frontier_start = 0
        frontier_stop = 0
        index = 0
        while index < eigenvalues.size and remaining > count_tolerance:
            scale = max(1.0, abs(float(eigenvalues[index])))
            stop = index + 1
            while (
                stop < eigenvalues.size
                and abs(float(eigenvalues[stop] - eigenvalues[index]))
                <= degeneracy_tolerance * scale
            ):
                stop += 1
            group_size = stop - index
            group_occupation = min(1.0, remaining / group_size)
            occupations[index:stop] = group_occupation
            remaining -= group_occupation * group_size
            frontier_start, frontier_stop = index, stop
            index = stop
        if target <= 0:
            fermi_level = float(eigenvalues[0])
        elif np.all(occupations >= 1.0 - count_tolerance):
            fermi_level = float(eigenvalues[-1])
        elif np.any(
            (occupations[frontier_start:frontier_stop] > count_tolerance)
            & (occupations[frontier_start:frontier_stop] < 1.0 - count_tolerance)
        ):
            fermi_level = float(eigenvalues[frontier_start])
        else:
            fermi_level = float(
                0.5
                * (
                    eigenvalues[frontier_stop - 1]
                    + eigenvalues[frontier_stop]
                )
            )
        return OccupationResult(
            fermi_level=fermi_level,
            occupations=occupations,
            electron_count=float(2.0 * np.sum(occupations)),
        )
    if temperature_kelvin < 0:
        raise ValueError("negative-temperature file occupations are not supported")
    if target == 0:
        occupations = np.zeros_like(eigenvalues)
        return OccupationResult(float(eigenvalues[0]), occupations, 0.0)
    if target >= eigenvalues.size:
        raise ValueError(
            "finite-temperature occupations require at least one buffered unoccupied state"
        )

    kbt = BOLTZMANN_RYDBERG_PER_KELVIN * temperature_kelvin
    lower = float(eigenvalues[0] - 36.0 * kbt)
    upper = float(eigenvalues[-1] + 36.0 * kbt)
    occupations = _fermi_function(eigenvalues, 0.5 * (lower + upper), kbt)
    for _ in range(max_iterations):
        chemical_potential = 0.5 * (lower + upper)
        occupations = _fermi_function(eigenvalues, chemical_potential, kbt)
        count = float(np.sum(occupations))
        if abs(count - target) <= count_tolerance:
            break
        if count < target:
            lower = chemical_potential
        else:
            upper = chemical_potential
    else:
        raise RuntimeError("Fermi-level bisection did not reach the electron-count tolerance")

    return OccupationResult(
        fermi_level=float(0.5 * (lower + upper)),
        occupations=occupations,
        electron_count=float(2.0 * np.sum(occupations)),
    )


def density_from_orbitals(
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    volume_element: float,
) -> np.ndarray:
    """Build ``rho = 2/dV * sum_i f_i |psi_i|**2``.

    Columns of ``wavefunctions`` must have Euclidean norm one, matching PARSEC's
    eigensolver convention.
    """
    wavefunctions = np.asarray(wavefunctions, dtype=float)
    occupations = np.asarray(occupations, dtype=float)
    if wavefunctions.ndim != 2:
        raise ValueError("wavefunctions must have shape (grid_points, states)")
    if occupations.shape != (wavefunctions.shape[1],):
        raise ValueError("occupation count does not match the wavefunction columns")
    if volume_element <= 0:
        raise ValueError("volume_element must be positive")
    return (2.0 / volume_element) * np.sum(
        wavefunctions * wavefunctions * occupations[None, :], axis=1
    )


__all__ = [
    "BOLTZMANN_RYDBERG_PER_KELVIN",
    "OccupationResult",
    "density_from_orbitals",
    "fermi_occupations",
]
