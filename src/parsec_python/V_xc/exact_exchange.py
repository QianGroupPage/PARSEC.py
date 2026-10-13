"""Exact (Hartree-Fock) exchange energy of real-space orbitals.

For real orbitals ``psi_i`` with occupations ``f_i`` in ``[0, 1]`` per spin
(the convention of :mod:`parsec_python.Occupations`: one stands for a doubly
occupied orbital), the exchange energy of the spin-unpolarized one-particle
density matrix is, in Hartree atomic units,

``E_x = - sum_ij f_i f_j (ij|ij)``,
``(ij|ij) = integral integral rho_ij(r) rho_ij(r') / |r-r'| d^3r d^3r'``,

with the pair density ``rho_ij = psi_i psi_j``.  Each integral is evaluated
as the solver evaluates the Hartree energy: one Poisson solve with open
boundary conditions gives the potential ``v_ij`` of ``rho_ij`` and the
integral is the grid sum of ``rho_ij v_ij``.  No Coulomb kernel is truncated
or screened and no Poisson solver is defined here: the caller passes the one
of its system, see :func:`reference_pair_poisson`.

Only the orbitals with a nonzero occupation enter, so a closed shell of
``N`` orbitals needs ``N (N + 1) / 2`` Poisson solves.  For integer
occupations this is the Hartree-Fock exchange energy of the determinant.
For fractional occupations it is the exchange energy of the ensemble density
matrix ``sum_i f_i |i><i|`` of each spin channel (the Fermi-smeared
generalization used with hybrid functionals for metals); it is not the
energy of a spin-polarized open shell.
:class:`ExactExchangeResult` reports whether that happened.  The orbitals of
a spin-polarized calculation, two sets with one electron per state, go to
:func:`spin_exact_exchange_energy`.

The orbitals are the columns of the solver's wavefunction array: Euclidean
normalized grid vectors ``q = sqrt(h**3) psi``.  Energies and potentials are
in Rydberg, like every energy of the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp

from ..Grid import RealSpaceGrid
from ..models import HartreeSettings


# An orbital whose occupation is below this bound is not occupied: with the
# Coulomb integrals of valence orbitals (below a few Hartree) it would change
# the exchange energy by less than 1e-11 Hartree.
OCCUPATION_THRESHOLD = 1.0e-12
# Occupations this close to zero or one count as integers in the report.
FRACTIONAL_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class ExactExchangeResult:
    """Exact exchange energy and what its evaluation took.

    ``energy`` is in Rydberg.  ``orbitals`` holds the columns of the
    wavefunction array that entered; ``pair_solves`` is the number of Poisson
    solves, ``poisson_iterations`` the sum of their conjugate-gradient
    iterations where the solver reports them.  ``fractional`` is true when an
    occupation differs from zero and one by more than
    :data:`FRACTIONAL_TOLERANCE`; ``largest_fractional_deviation`` is the
    largest ``min(f, 1 - f)``.
    """

    energy: float
    orbitals: tuple[int, ...]
    pair_solves: int
    poisson_iterations: int
    poisson_seconds: float
    seconds: float
    fractional: bool
    largest_fractional_deviation: float


def occupied_columns(
    occupations: np.ndarray,
    threshold: float = OCCUPATION_THRESHOLD,
) -> np.ndarray:
    """Return the indices of the orbitals that carry electrons.

    Occupations must lie in ``[0, 1]`` (one is a doubly occupied orbital);
    anything else is refused rather than clipped.
    """

    occupations = np.asarray(occupations, dtype=np.float64)
    if occupations.ndim != 1:
        raise ValueError("occupations must be a one-dimensional array")
    if not np.all(np.isfinite(occupations)):
        raise ValueError("occupations must be finite")
    if np.any(occupations < -1.0e-10) or np.any(occupations > 1.0 + 1.0e-10):
        raise ValueError(
            "exact exchange needs occupations between 0 and 1 per spin "
            "(1 = doubly occupied orbital); found "
            f"{float(occupations.min()):.6g} .. {float(occupations.max()):.6g}"
        )
    return np.flatnonzero(occupations > threshold)


def exact_exchange_energy(
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    volume_element: float,
    solve_pair_potential: Callable[[np.ndarray], object],
    *,
    occupation_threshold: float = OCCUPATION_THRESHOLD,
    progress: Callable[[int, int, float], None] | None = None,
) -> ExactExchangeResult:
    """Accumulate ``-sum_ij f_i f_j (ij|ij)`` over the occupied orbitals.

    Parameters
    ----------
    wavefunctions
        Real array of shape ``(grid points, states)`` whose columns are
        Euclidean-normalized orbitals on the full active grid.
    occupations
        One occupation per column, between zero and one.
    volume_element
        ``h**3`` of the grid in bohr**3.
    solve_pair_potential
        ``solve_pair_potential(rho)`` returns the open-boundary potential of
        the density ``rho`` (electrons/bohr**3) in Rydberg,
        ``2 * integral rho(r')/|r-r'|``, as an array or as an object with a
        ``potential`` attribute (a
        :class:`parsec_python.Hartree.HartreeResult`).  It must be linear in
        ``rho``: a solver that adds a static term to its boundary values,
        such as the atomic tail of the total density, is not one.
    progress
        Optional ``progress(done, total, seconds)`` called after each solve.
    """

    started = perf_counter()
    if np.iscomplexobj(wavefunctions):
        raise NotImplementedError("exact exchange is implemented for real orbitals")
    if not isinstance(wavefunctions, np.ndarray):
        raise TypeError(
            "exact exchange needs the orbitals as a host array on the full "
            "grid; materialize the wavefunctions of the result first"
        )
    if wavefunctions.ndim != 2:
        raise ValueError("wavefunctions must have shape (grid points, states)")
    occupations = np.asarray(occupations, dtype=np.float64)
    if occupations.shape != (wavefunctions.shape[1],):
        raise ValueError("one occupation per wavefunction column is required")
    if not np.isfinite(volume_element) or volume_element <= 0.0:
        raise ValueError("volume_element must be positive")

    columns = occupied_columns(occupations, occupation_threshold)
    weights = np.clip(occupations[columns], 0.0, 1.0)
    deviation = float(np.max(np.minimum(weights, 1.0 - weights), initial=0.0))
    # Each orbital is copied once: contiguous float64 columns of the array of
    # the result, which may hold many empty states as well.
    orbitals = [
        np.ascontiguousarray(wavefunctions[:, column], dtype=np.float64)
        for column in columns
    ]
    for column, orbital in zip(columns, orbitals):
        norm = float(orbital @ orbital)
        if not abs(norm - 1.0) < 1.0e-6:
            raise ValueError(
                f"orbital {int(column) + 1} is not normalized on the grid "
                f"(sum of squares {norm:.8f}); exact exchange needs the "
                "Euclidean-normalized full-grid orbitals of the solver"
            )

    count = len(orbitals)
    total_pairs = count * (count + 1) // 2
    inverse_volume = 1.0 / volume_element
    energy = 0.0
    iterations = 0
    poisson_seconds = 0.0
    done = 0
    for first in range(count):
        for second in range(first + 1):
            # rho_ij = psi_i psi_j = q_i q_j / h**3.
            pair_density = orbitals[first] * orbitals[second]
            pair_density *= inverse_volume
            solve_started = perf_counter()
            solved = solve_pair_potential(pair_density)
            poisson_seconds += perf_counter() - solve_started
            potential = np.asarray(getattr(solved, "potential", solved))
            if potential.shape != pair_density.shape:
                raise ValueError("the pair potential does not match the grid")
            iterations += int(getattr(solved, "iterations", 0) or 0)
            # (ij|ij) in Rydberg: h**3 * sum rho_ij v_ij with v in Rydberg is
            # twice the Hartree-unit integral, and E_x[Ry] = 2 E_x[Ha].
            integral = volume_element * float(pair_density @ potential)
            multiplicity = 1.0 if first == second else 2.0
            energy -= multiplicity * weights[first] * weights[second] * integral
            done += 1
            if progress is not None:
                progress(done, total_pairs, perf_counter() - started)
    return ExactExchangeResult(
        energy=float(energy),
        orbitals=tuple(int(column) for column in columns),
        pair_solves=total_pairs,
        poisson_iterations=iterations,
        poisson_seconds=poisson_seconds,
        seconds=perf_counter() - started,
        fractional=deviation > FRACTIONAL_TOLERANCE,
        largest_fractional_deviation=deviation,
    )


def spin_exact_exchange_energy(
    wavefunctions: Sequence[np.ndarray],
    occupations: Sequence[np.ndarray],
    volume_element: float,
    solve_pair_potential: Callable[[np.ndarray], object],
    *,
    occupation_threshold: float = OCCUPATION_THRESHOLD,
    progress: Callable[[int, int, float], None] | None = None,
) -> ExactExchangeResult:
    """Exact exchange energy of the orbitals of two spin channels.

    Only orbitals of the same spin exchange.  With occupations ``f`` in
    ``[0, 1]`` that count one electron per state,

    ``E_x = -(1/2) sum_s sum_ij f_s,i f_s,j (ij|ij)_s``

    in Hartree atomic units: each channel contributes half of what
    :func:`exact_exchange_energy` gives for its orbitals and occupations,
    and two identical channels give the unpolarized energy.  For one
    electron the exchange energy cancels the Hartree energy.

    ``wavefunctions`` and ``occupations`` hold the two channels, up first.
    The returned ``orbitals`` number the columns of the up channel followed
    by those of the down channel; the counts and times are sums over both.
    """

    if len(wavefunctions) != 2 or len(occupations) != 2:
        raise ValueError("the orbitals and occupations of two spin channels are needed")
    started = perf_counter()
    pair_counts = []
    for channel_occupations in occupations:
        occupied = occupied_columns(channel_occupations, occupation_threshold).size
        pair_counts.append(occupied * (occupied + 1) // 2)
    total_pairs = sum(pair_counts)
    results: list[ExactExchangeResult] = []
    for index in range(2):
        offset = sum(pair_counts[:index])

        def channel_progress(done: int, _total: int, _seconds: float) -> None:
            progress(offset + done, total_pairs, perf_counter() - started)

        results.append(
            exact_exchange_energy(
                wavefunctions[index],
                occupations[index],
                volume_element,
                solve_pair_potential,
                occupation_threshold=occupation_threshold,
                progress=None if progress is None else channel_progress,
            )
        )
    up_states = int(np.shape(occupations[0])[0])
    return ExactExchangeResult(
        energy=0.5 * (results[0].energy + results[1].energy),
        orbitals=results[0].orbitals
        + tuple(up_states + column for column in results[1].orbitals),
        pair_solves=results[0].pair_solves + results[1].pair_solves,
        poisson_iterations=results[0].poisson_iterations + results[1].poisson_iterations,
        poisson_seconds=results[0].poisson_seconds + results[1].poisson_seconds,
        seconds=perf_counter() - started,
        fractional=results[0].fractional or results[1].fractional,
        largest_fractional_deviation=max(
            results[0].largest_fractional_deviation,
            results[1].largest_fractional_deviation,
        ),
    )


def pair_hartree_settings(settings: HartreeSettings) -> HartreeSettings:
    """Hartree settings of a system as a pair density needs them.

    The multipole order, the boundary method and the conjugate-gradient
    controls are kept.  The atomic tail is switched off: it is the potential
    of the valence charges of the atoms, a property of the total density.
    """

    return replace(settings, atomic_tail="off", boundary_tolerance=None)


def reference_pair_poisson(
    grid: RealSpaceGrid,
    negative_laplacian: sp.spmatrix,
    settings: HartreeSettings,
) -> Callable[[np.ndarray], object]:
    """Pair-density Poisson solver built on :func:`parsec_python.Hartree.solve_hartree`.

    Every solve builds the multipole (sphere) or direct (box) boundary values
    of the pair density at the order of ``settings`` and starts the conjugate
    gradients from zero.
    """

    # Imported here: Hartree imports the grid and the Laplacian, not V_xc.
    from ..Hartree import solve_hartree

    pair_settings = pair_hartree_settings(settings)

    def solve(pair_density: np.ndarray):
        return solve_hartree(pair_density, grid, negative_laplacian, pair_settings)

    return solve


__all__ = [
    "FRACTIONAL_TOLERANCE",
    "OCCUPATION_THRESHOLD",
    "ExactExchangeResult",
    "exact_exchange_energy",
    "occupied_columns",
    "pair_hartree_settings",
    "reference_pair_poisson",
    "spin_exact_exchange_energy",
]
