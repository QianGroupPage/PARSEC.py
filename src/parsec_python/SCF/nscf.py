"""Non-self-consistent run on a list of k-points (Fortran ``calc_nscf.F90``).

``nscf_kpoints`` turns PARSEC's run into a non-self-consistent one
(``usrinputfile.F90`` l.1056-1118): the potential and density of a previous
run are read from ``parsec.dat`` (``Restart_Run`` is required, error 114),
the SCF loop is skipped (``parsec.F90`` l.1098, ``goto 80``) and
``calc_nscf`` solves, once, the eigenproblem of the stored potential at
every NSCF k-point with ``nscf_states_num`` states, from scratch (the solver
is created anew, ``nec = -1``), then ``flevel`` computes the occupations
with a *fixed* Fermi level (``flevel.f90`` l.161-165) and prints the
eigenvalues of every k-point.  No ``parsec.dat`` or ``eigen.dat`` is written.

This module translates that run on top of the k-point machinery of the band
structure (:func:`parsec_python.SCF.bands.solve_kpoint_bands`): the
Hamiltonian at a k-point is the Bloch Hamiltonian of the band structure
(kinetic ``-2i k.grad + k^2``, Bloch-phase projectors) with the restart
potential, and it is diagonalized from scratch by the SCF's first
eigensolver, as ``calc_nscf`` does.

The Fermi level.  ``flevel``'s NSCF branch takes ``elec_st%efermi`` as it
stands.  Only ``Output_GW`` reads ``nscf_fermi_level`` into it
(``usrinputfile.F90`` l.2535); without ``Output_GW`` nothing ever sets it,
so PARSEC uses the zero of an uninitialized (static) variable: "Fermi level
at 0.0000 [Ry]", and every printed occupation is that of ``E_F = 0`` in
PARSEC's potential zero (checked with the Fortran binary, fcc Si: only the
lowest state at Gamma is occupied).  The occupations are therefore
meaningless there.  This port uses instead the Fermi level of the SCF run
that wrote ``parsec.dat``, recomputed from the levels the file stores (the
``flevel`` bisection of :func:`fermi_level_from_levels` on the stored
eigenvalues with the stored k-point weights, ``Fermi_Temp`` and the
electron count of the input); :func:`run_nscf` takes any Fermi level, so
PARSEC's ``E_F = 0`` can be asked for explicitly.  The eigenvalues - the
result of the run - do not depend on it.

Spin-orbit coupling.  ``calc_nscf`` builds its eigenstates with
``parallel%mxwd`` rows per grid point, so the NSCF run of a spin-orbit
``parsec.dat`` (``stype(2) = 2``, which ``restart_run.F90`` turns into
``is_so``, ``mxwd = 2``) solves the spinor Bloch Hamiltonian
(``zmatvec_so``: ``vnew(:,1)`` on the up and ``vnew(:,2)`` on the down
component, the Bloch-phase nonlocal and spin-orbit projectors that
``nonloc`` rebuilds for the NSCF k-points), ``nscf_states_num`` spinor
states per k-point (not doubled: ``elec_st%nstate = nscf%nstate``).
``flevel`` then counts one electron per spinor state (``nspin/mxwd = 1``,
``xele * nspin / 2 = xele``) and ``eigval.F90`` gives every state its
``<sigma_z>`` (``elec_st%magmom``, printed as ``Spin_z [hbar]``).  A
non-collinear file (``stype(2)`` 3 or 4) stores no magnetization direction
and restarts nothing (refused, as for ``Restart_Run``).

Energies are in Ry, k-points Cartesian in 1/bohr.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..Occupations.fermi_dirac import (
    FERMI_COUNT_TOLERANCE,
    FERMI_EXPONENT_CUTOFF,
    FERMI_MAX_BISECTIONS,
)

#: ``tempkry`` of ``const.f90``: Kelvin to Rydberg.
KELVIN_TO_RYDBERG = 6.33327186e-06
#: ``difmax`` of ``flevel.f90``: beyond ``|beta (E - E_F)|`` of 36.7 an
#: occupation is exactly one or zero; ``small``, the electron count of the
#: bisection; ``iterlim``.  PARSEC 1.5's values (``utilities/flevel.f90:43-51``:
#: 36.7, 1e-10, 1000; the previous tree had 35, 1e-12, 100).
_DIFMAX = FERMI_EXPONENT_CUTOFF
_SMALL = FERMI_COUNT_TOLERANCE
_ITERATION_LIMIT = FERMI_MAX_BISECTIONS


class FermiLevelError(RuntimeError):
    """``flevel.f90`` error 422: the bisection found no Fermi level."""


def fermi_level_from_levels(
    eigenvalues: Sequence[Sequence[np.ndarray]],
    k_weights: Sequence[float],
    electron_count: float,
    temperature_kelvin: float,
    *,
    spinors: bool = False,
) -> float:
    """The Fermi level (Ry) of ``flevel.f90``'s bisection (l.218-280).

    ``eigenvalues[s][k]`` are the levels of spin channel ``s`` (one, or up
    and down) at k-point ``k``, which carries the weight ``k_weights[k]``
    (normalized to one, as ``kpwt``).  The levels of every k-point of a
    channel are pooled and sorted; with one channel the weighted occupations
    count electron pairs (``xele/2``), with two single electrons, and with
    one channel of ``spinors`` single electrons too (``xele * nspin/2`` with
    ``nspin = 2``, ``mxwd = 2``).  The bisection starts from the bounds
    ``[min, max]`` of all levels with the guess ``max`` and stops when the
    count is within 1e-10 of the target (PARSEC 1.5).
    """

    channels = len(eigenvalues)
    if channels not in (1, 2):
        raise ValueError("eigenvalues must hold one or two spin channels")
    weights = np.asarray(k_weights, dtype=float)
    pooled = []
    for channel in eigenvalues:
        if len(channel) != weights.size:
            raise ValueError("every spin channel needs the levels of every k-point")
        values = np.concatenate([np.asarray(levels, dtype=float) for levels in channel])
        level_weights = np.concatenate(
            [np.full(np.asarray(levels).size, weight) for levels, weight in zip(channel, weights)]
        )
        order = np.argsort(values, kind="stable")
        pooled.append((values[order], level_weights[order]))
    if spinors and channels != 1:
        raise ValueError("spinor levels form one channel")
    target = float(electron_count) * (2 if spinors else channels) / 2.0
    beta = 1.0 / KELVIN_TO_RYDBERG / float(temperature_kelvin)
    lowest = min(float(values.min()) for values, _ in pooled)
    highest = max(float(values.max()) for values, _ in pooled)
    fermi = highest
    for _ in range(_ITERATION_LIMIT):
        # The terms in the order of the Fortran loop (channel, then the
        # sorted levels), summed one after the other as there; a channel's
        # loop ends at its first level difmax kT above the guess.
        terms = []
        for values, level_weights in pooled:
            scaled = (values - fermi) * beta
            above = scaled >= _DIFMAX
            kept = int(np.argmax(above)) if np.any(above) else scaled.size
            scaled, level_weights = scaled[:kept], level_weights[:kept]
            fraction = np.ones_like(scaled)
            middle = scaled >= -_DIFMAX
            fraction[middle] = 1.0 / (np.exp(scaled[middle]) + 1.0)
            terms.append(fraction * level_weights)
        joined = np.concatenate(terms)
        count = float(np.cumsum(joined)[-1]) if joined.size else 0.0
        if abs(target - count) < _SMALL:
            return float(fermi)
        if count < target:
            lowest = fermi
        else:
            highest = fermi
        fermi = (highest + lowest) * 0.5
    raise FermiLevelError(
        f"Fermi level not found (flevel.f90 error 422); last guess {fermi:.4f} Ry"
    )


def fixed_fermi_occupations(
    eigenvalues: np.ndarray, fermi_level: float, temperature_kelvin: float
) -> np.ndarray:
    """``flevel.f90`` l.282-317 at a given Fermi level: one, zero or Fermi-Dirac.

    The occupation of a state is a fraction of its capacity (two electrons
    without spin, one per channel with spin), as PARSEC prints it.
    """

    beta = 1.0 / KELVIN_TO_RYDBERG / float(temperature_kelvin)
    scaled = (np.asarray(eigenvalues, dtype=float) - float(fermi_level)) * beta
    occupations = np.zeros_like(scaled)
    occupations[scaled < -_DIFMAX] = 1.0
    middle = (scaled >= -_DIFMAX) & (scaled < _DIFMAX)
    occupations[middle] = 1.0 / (np.exp(scaled[middle]) + 1.0)
    return occupations


def restart_fermi_level(data, electron_count: float, temperature_kelvin: float) -> float:
    """The Fermi level of the run that wrote ``data`` (a :class:`ParsecDat`).

    The ``flevel`` bisection of :func:`fermi_level_from_levels` on the
    eigenvalues the file stores for every spin channel and k-point, with its
    k-point weights (one weight for a cluster or a Gamma-point file); the
    levels of a spinor file (``stype(2) > 1``) hold one electron each.
    """

    weights = (
        np.ones(1)
        if data.periodic is None
        else np.asarray(data.periodic.k_weights, dtype=float)
    )
    weights = weights / float(np.sum(weights))
    levels = [[block.energies for block in channel] for channel in data.levels]
    return fermi_level_from_levels(
        levels,
        weights,
        electron_count,
        temperature_kelvin,
        spinors=int(getattr(data, "wavefunction_kind", 0) or 0) > 1,
    )


@dataclass
class NSCFResult:
    """The levels of a non-self-consistent run.

    ``eigenvalues`` and ``occupations`` have shape ``(channel, k, state)``
    (``channels`` names the first axis: ``("all",)`` or ``("up", "down")``),
    ``k_points`` (Cartesian, 1/bohr) and ``weights`` (normalized) are those
    of ``nscf_kpoints``, ``fermi_level`` the fixed Fermi level (Ry) of the
    occupations, ``residual_norms`` the largest ``|H v - e v|`` of the
    states of each ``(channel, k)`` and ``attempts`` the diagonalizations
    (see :mod:`parsec_python.SCF.bands`).  ``electrons`` is the weighted
    electron count of each channel at that Fermi level (``totel`` of
    ``flevel.f90``).  ``wavefunctions[channel][k]`` are the ``(n_grid,
    n_states)`` Bloch factors ``u_nk`` when the run kept them
    (``keep_wavefunctions``, for ``Output_GW``), ``None`` otherwise.
    ``spin_z`` (shape ``(k, state)``) is ``<sigma_z>`` of every spinor state
    of a spin-orbit run (``elec_st%magmom``), ``None`` without spinors.
    """

    channels: tuple[str, ...]
    k_points: np.ndarray
    weights: np.ndarray
    eigenvalues: np.ndarray
    occupations: np.ndarray
    fermi_level: float
    temperature_kelvin: float
    residual_norms: np.ndarray
    attempts: np.ndarray
    electrons: tuple[float, ...]
    wavefunctions: list | None = None
    spin_z: np.ndarray | None = None

    @property
    def spin_orbit_moment(self) -> float | None:
        """``flevel.f90`` l.337-345: ``sum_k sum_n <sigma_z>_nk f_nk w_k``."""

        if self.spin_z is None:
            return None
        return float(np.sum(self.spin_z * self.occupations[0] * self.weights[:, None]))


def run_nscf(
    system,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    potentials: Sequence[np.ndarray],
    number_of_states: int,
    fermi_level: float,
    callback: Callable[[int, int, np.ndarray, float, int], None] | None = None,
    workers: int | str | None = None,
    backend: str | None = None,
    keep_wavefunctions: bool = False,
    spinors: bool = False,
) -> NSCFResult:
    """``calc_nscf``: one eigensolve per k-point in the fixed ``potentials``.

    ``potentials`` holds the local potential (Ry, on ``system.grid``) of
    each spin channel: one, or up and down (``vnew`` of ``parsec.dat``).
    With ``spinors`` (a spin-orbit ``parsec.dat``) the two potentials are
    those of the up and down spinor components, and every k-point is one
    spinor Bloch problem with the spin-orbit projectors (see the module
    docstring).
    ``number_of_states`` is ``nscf_states_num``; ``fermi_level`` (Ry) fixes
    the occupations (see the module docstring).  ``callback``, ``workers``
    and ``backend`` are those of
    :func:`parsec_python.SCF.bands.run_band_structure`.
    ``keep_wavefunctions`` keeps the eigenvectors (``Output_GW`` writes them).
    """

    from .bands import solve_kpoint_bands
    from .single_point import refuse_hubbard_and_dispersion

    refuse_hubbard_and_dispersion(system, "run_nscf", dispersion=False)
    k_points = np.asarray(k_points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if k_points.ndim != 2 or k_points.shape[1] != 3 or weights.shape != (k_points.shape[0],):
        raise ValueError("k_points must be (n, 3) and weights (n,)")
    if spinors:
        if len(potentials) != 2:
            raise ValueError("a spin-orbit NSCF run takes the two spinor-component potentials")
        up, down = (np.asarray(value, dtype=np.float64) for value in potentials)
        if system.input.scf.spin_polarized:
            # zmatvec_so: V_up on the up, V_down on the down component.
            mode = "soc_spin_polarized"
            named = {"effective_potential": 0.5 * (up + down), "xc_delta": 0.5 * (up - down)}
        else:
            # The one-potential spin-orbit loop stores the same potential twice.
            mode, named = "soc", {"effective_potential": 0.5 * (up + down)}
    elif len(potentials) == 1:
        mode, named = "scalar", {"effective_potential": np.asarray(potentials[0])}
    elif len(potentials) == 2:
        mode = "spin_polarized"
        named = {
            "potential_up": np.asarray(potentials[0]),
            "potential_down": np.asarray(potentials[1]),
        }
    else:
        raise ValueError("potentials must hold one or two spin channels")
    if int(number_of_states) + system.input.eigensolver.subspace_buffer >= system.grid.size:
        raise ValueError("the grid is too small for nscf_states_num and the eigensolver buffer")
    bands = solve_kpoint_bands(
        system,
        k_points,
        mode=mode,
        potentials=named,
        number_of_states=int(number_of_states),
        callback=callback,
        workers=workers,
        backend=backend,
        **({"keep_vectors": True} if keep_wavefunctions else {}),
    )
    temperature = float(system.input.scf.fermi_temperature_kelvin)
    occupations = fixed_fermi_occupations(bands.eigenvalues, fermi_level, temperature)
    electrons = tuple(
        float(np.sum(occupations[channel] * weights[:, None]))
        for channel in range(occupations.shape[0])
    )
    return NSCFResult(
        channels=bands.channels,
        k_points=k_points,
        weights=weights,
        eigenvalues=bands.eigenvalues,
        occupations=occupations,
        fermi_level=float(fermi_level),
        temperature_kelvin=temperature,
        residual_norms=bands.residual_norms,
        attempts=bands.attempts,
        electrons=electrons,
        wavefunctions=bands.vectors,
        spin_z=(
            None
            if not spinors or bands.spin_expectation is None
            else np.asarray(bands.spin_expectation[0, :, :, 2], dtype=float)
        ),
    )


__all__ = [
    "FermiLevelError",
    "KELVIN_TO_RYDBERG",
    "NSCFResult",
    "fermi_level_from_levels",
    "fixed_fermi_occupations",
    "restart_fermi_level",
    "run_nscf",
]
