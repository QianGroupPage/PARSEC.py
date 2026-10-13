"""Occupations of a spin-polarized calculation (``flevel.f90``).

Both spin channels share one Fermi level.  A state of a channel holds one
electron, so the occupations ``f`` lie in ``[0, 1]`` and

``sum_i f_up,i + sum_i f_down,i = N_e``,

where the unpolarized convention has ``2 sum_i f_i = N_e``.  The electrons
of a channel, ``totel`` in the Fortran, are the sum of its occupations; the
net magnetic moment in Bohr magnetons is their difference.

The level is found by the bisection of ``flevel.f90``, statement by
statement: it starts at the highest eigenvalue of both channels, with the
lowest and highest eigenvalues as bounds, and stops when the electron count
is within ``1e-12``.  Between two levels of a gapped system the level is
not unique, and this is the value PARSEC reports.

A negative ``Fermi_Temp`` takes the occupations of both channels from
``occup.in`` (``initial.f90``): a section per channel, up first.  The n-th
occupation of a section goes to the state of that channel with the n-th
lowest eigenvalue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .fermi_dirac import BOLTZMANN_RYDBERG_PER_KELVIN, fermi_occupations
from .fixed import ELECTRON_COUNT_TOLERANCE


#: ``small``, ``difmax`` and ``iterlim`` of ``flevel.f90``.
COUNT_TOLERANCE = 1.0e-12
EXPONENT_CUTOFF = 35.0
MAXIMUM_BISECTIONS = 100
#: Electron count accepted from a bisection whose level double precision can
#: no longer refine (many levels within kT of the Fermi level), where
#: ``flevel.f90`` stops with "Fermi level not found".
STALLED_COUNT_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class SpinOccupationResult:
    """Occupations of the two channels, up first.

    ``fermi_level`` is ``nan`` for fixed occupations, for which none is
    searched.  ``electrons`` are the sums of the occupations per channel.
    """

    fermi_level: float
    occupations: tuple[np.ndarray, np.ndarray]
    electrons: tuple[float, float]

    @property
    def electron_count(self) -> float:
        return float(self.electrons[0] + self.electrons[1])

    @property
    def net_magnetic_moment(self) -> float:
        """Electrons up minus electrons down, in Bohr magnetons."""

        return float(self.electrons[0] - self.electrons[1])


def _channels(
    eigenvalues_up: np.ndarray, eigenvalues_down: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    channels = []
    for name, values in (("up", eigenvalues_up), ("down", eigenvalues_down)):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                f"eigenvalues of the {name} channel must be a nonempty "
                "one-dimensional array"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("eigenvalues must be finite")
        if np.any(np.diff(values) < 0):
            raise ValueError("eigenvalues must be sorted")
        channels.append(values)
    return channels[0], channels[1]


def _fermi_function(eigenvalues: np.ndarray, level: float, beta: float) -> np.ndarray:
    """One below ``-35``, zero from ``35`` on, Fermi-Dirac between."""

    exponent = (eigenvalues - level) * beta
    occupations = np.zeros_like(exponent)
    occupations[exponent < -EXPONENT_CUTOFF] = 1.0
    middle = (exponent >= -EXPONENT_CUTOFF) & (exponent < EXPONENT_CUTOFF)
    occupations[middle] = 1.0 / (np.exp(exponent[middle]) + 1.0)
    return occupations


def _result(level: float, up: np.ndarray, down: np.ndarray) -> SpinOccupationResult:
    return SpinOccupationResult(
        fermi_level=float(level),
        occupations=(up, down),
        electrons=(float(np.sum(up)), float(np.sum(down))),
    )


def spin_fermi_occupations(
    eigenvalues_up: np.ndarray,
    eigenvalues_down: np.ndarray,
    electron_count: float,
    temperature_kelvin: float = 80.0,
) -> SpinOccupationResult:
    """Occupy two spin channels up to their common Fermi level.

    At zero temperature the states of both channels are filled in ascending
    order of energy, a degenerate frontier sharing what is left equally, as
    :func:`parsec_python.Occupations.fermi_occupations` does for one
    channel.
    """

    up, down = _channels(eigenvalues_up, eigenvalues_down)
    electron_count = float(electron_count)
    states = up.size + down.size
    if not np.isfinite(electron_count) or not 0.0 <= electron_count <= states:
        raise ValueError(
            f"{electron_count:g} electrons do not fit the {states} states "
            "of the two spin channels"
        )
    if not np.isfinite(temperature_kelvin) or temperature_kelvin < 0:
        raise ValueError(
            "the Fermi temperature must be finite and nonnegative; a negative "
            "one selects fixed occupations"
        )
    if temperature_kelvin == 0:
        # One list of single-electron states: the unpolarized routine fills
        # states that hold two electrons each, so it is given twice the count.
        merged = np.concatenate((up, down))
        order = np.argsort(merged, kind="stable")
        filled = fermi_occupations(merged[order], 2.0 * electron_count, 0.0)
        occupations = np.empty_like(merged)
        occupations[order] = filled.occupations
        return _result(filled.fermi_level, occupations[: up.size], occupations[up.size :])

    # flevel.f90: tryd = 1/tempkry, beta = tryd/tfermi.
    beta = (1.0 / BOLTZMANN_RYDBERG_PER_KELVIN) / float(temperature_kelvin)
    lower = float(min(up[0], down[0]))
    upper = float(max(up[-1], down[-1]))
    level = upper
    previous = None
    for _ in range(MAXIMUM_BISECTIONS):
        count = float(
            np.sum(_fermi_function(up, level, beta))
            + np.sum(_fermi_function(down, level, beta))
        )
        if abs(electron_count - count) < COUNT_TOLERANCE:
            break
        if level == previous and abs(electron_count - count) < STALLED_COUNT_TOLERANCE:
            break
        previous = level
        if count < electron_count:
            lower = level
        else:
            upper = level
        level = (upper + lower) * 0.5
    else:
        raise RuntimeError(
            "Fermi level not found: the computed states of the two spin "
            f"channels cannot hold {electron_count:g} electrons with an empty "
            "state above them; increase States_Num"
        )
    return _result(
        level, _fermi_function(up, level, beta), _fermi_function(down, level, beta)
    )


def check_spin_fixed_occupations(
    occupations_up: Sequence[float],
    occupations_down: Sequence[float],
    electron_count: float,
) -> None:
    """Refuse occupations that do not hold the electrons of the system.

    ``initial.f90`` stops with "fixed occupation inconsistent with stated
    number of electrons" when the sum over both channels differs from the
    electron count by more than ``1e-5``.
    """

    held = float(
        np.sum(np.asarray(occupations_up, dtype=float))
        + np.sum(np.asarray(occupations_down, dtype=float))
    )
    if abs(held - float(electron_count)) > ELECTRON_COUNT_TOLERANCE:
        raise ValueError(
            f"the fixed occupations of the two spin channels hold {held:.6f} "
            f"electrons, but the system has {float(electron_count):.6f}"
        )


def spin_fixed_occupations(
    eigenvalues_up: np.ndarray,
    eigenvalues_down: np.ndarray,
    electron_count: float,
    occupations_up: Sequence[float],
    occupations_down: Sequence[float],
) -> SpinOccupationResult:
    """Give each channel its occupations of ``occup.in`` in order of energy."""

    up, down = _channels(eigenvalues_up, eigenvalues_down)
    values_up = np.array(occupations_up, dtype=np.float64)
    values_down = np.array(occupations_down, dtype=np.float64)
    for name, values, eigenvalues in (
        ("up", values_up, up),
        ("down", values_down, down),
    ):
        if values.shape != eigenvalues.shape:
            raise ValueError(
                f"{values.size} fixed occupations were given for the "
                f"{eigenvalues.size} computed states of the {name} channel"
            )
    check_spin_fixed_occupations(values_up, values_down, electron_count)
    return _result(float("nan"), values_up, values_down)


def spin_density_from_orbitals(
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    volume_element: float,
) -> np.ndarray:
    """Build the density of one channel, ``rho = 1/dV * sum_i f_i |psi_i|**2``.

    ``newrho.F90`` divides its factor ``2/h**3`` by the number of spin
    channels.  Columns of ``wavefunctions`` have Euclidean norm one.
    """

    wavefunctions = np.asarray(wavefunctions, dtype=float)
    occupations = np.asarray(occupations, dtype=float)
    if wavefunctions.ndim != 2:
        raise ValueError("wavefunctions must have shape (grid_points, states)")
    if occupations.shape != (wavefunctions.shape[1],):
        raise ValueError("occupation count does not match the wavefunction columns")
    if volume_element <= 0:
        raise ValueError("volume_element must be positive")
    return (1.0 / volume_element) * np.sum(
        wavefunctions * wavefunctions * occupations[None, :], axis=1
    )


__all__ = [
    "SpinOccupationResult",
    "check_spin_fixed_occupations",
    "spin_density_from_orbitals",
    "spin_fermi_occupations",
    "spin_fixed_occupations",
]
