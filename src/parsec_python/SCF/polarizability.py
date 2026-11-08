"""Static polarizability from seven finite-field SCF loops (``polar.f90``).

With ``Polarizability`` PARSEC (``parsec.F90`` lines 1050-1722) runs seven SCF
loops of the same structure, with a uniform field of strength ``F`` along

``0, +x, -x, +y, -y, +z, -z``

(Vasiliev et al., Phys. Rev. Lett. 78, 4805 (1997)).  After each loop
``polar`` records the total energy and the dipole, and prepares the next
loop:

* the local ionic potential gets the field of the next direction ``d``,
  ``V_ion <- V_ion + 2 F (d - d_old) . r`` [Ry] with ``r`` measured from the
  origin of the grid (:func:`parsec_python.V_ion.electric_field_potential`;
  the number ``F`` acts as a field in Hartree atomic units, see there);
* ``pot%vnew = pot%vold``: the first eigenproblem of the next loop is posed
  with the *input* potential of the last iteration, which still holds the
  old field; the new field enters through the output potential of that
  first iteration;
* the wavefunctions are kept and ``solver%firstfilt`` is set again, so the
  next loop starts with a SUBSPACE filter of the saved subspace.

No convergence is required: a loop that ends at ``Max_Iter`` is used as it
is.  From the seven loops, with energies in Hartree and dipoles in e bohr,

``alpha_ii(E) = (2 E_0 - E(+i) - E(-i)) / F^2``
``alpha_ii(mu) = (mu_i(+i) - mu_i(-i)) / (2 F)``

converted to cubic angstrom with ``cubangs = 0.529177**3`` of ``const.f90``.
The energy has no term of the nuclei in the field, as in PARSEC.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from ..Energy.dipole import DipoleMoment, cluster_dipole_moment
from ..V_ion import electric_field_potential
from .single_point import SCFWarmStart, run_scf, state_for_new_potential

#: The field directions of the seven runs, in ``polar.f90``'s order.
FIELD_DIRECTIONS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)

#: ``cubangs`` of ``const.f90``: bohr^3 -> angstrom^3 with angs = 0.529177.
CUBIC_ANGSTROM_PER_CUBIC_BOHR = 0.529177**3


@dataclass(frozen=True)
class FieldRun:
    """One of the seven SCF loops: its field direction, energy and dipole.

    ``total_energy`` is in Ry; ``dipole`` the charge and dipole of the
    converged density (:mod:`parsec_python.Energy.dipole`).
    """

    direction: tuple[float, float, float]
    total_energy: float
    dipole: DipoleMoment
    converged: bool
    iterations: int


@dataclass(frozen=True)
class PolarizabilityResult:
    """The seven runs and the diagonal polarizability ``polar.f90`` writes."""

    field: float
    runs: tuple[FieldRun, ...]
    atom_count: int

    def _require_complete(self) -> None:
        if len(self.runs) != len(FIELD_DIRECTIONS):
            raise ValueError("the polarizability needs all seven field runs")

    @property
    def energies_hartree(self) -> np.ndarray:
        """``enrg`` of ``polar.f90``: the total energies in Hartree."""

        return np.array([run.total_energy for run in self.runs]) / 2.0

    @property
    def dipoles(self) -> np.ndarray:
        """``dplx, dply, dplz``: the dipoles in e bohr, shape ``(runs, 3)``."""

        return np.array([run.dipole.dipole for run in self.runs])

    @property
    def from_energy(self) -> np.ndarray:
        """``alfaXX, alfaYY, alfaZZ`` from the energies, in angstrom^3."""

        self._require_complete()
        energy = self.energies_hartree
        inverse = 1.0 / self.field
        return np.array(
            [
                inverse * inverse * (2.0 * energy[0] - energy[1] - energy[2]),
                inverse * inverse * (2.0 * energy[0] - energy[3] - energy[4]),
                inverse * inverse * (2.0 * energy[0] - energy[5] - energy[6]),
            ]
        ) * CUBIC_ANGSTROM_PER_CUBIC_BOHR

    @property
    def from_dipole(self) -> np.ndarray:
        """``alfaXX, alfaYY, alfaZZ`` from the dipoles, in angstrom^3."""

        self._require_complete()
        dipole = self.dipoles
        inverse = 1.0 / self.field
        return np.array(
            [
                inverse * ((dipole[1, 0] - dipole[2, 0]) / 2.0),
                inverse * ((dipole[3, 1] - dipole[4, 1]) / 2.0),
                inverse * ((dipole[5, 2] - dipole[6, 2]) / 2.0),
            ]
        ) * CUBIC_ANGSTROM_PER_CUBIC_BOHR

    @property
    def charge(self) -> float:
        """The computed charge of the last run (``elec_st%charge``)."""

        return float(self.runs[-1].dipole.charge)


def field_system(system, base_ionic_potential: np.ndarray, field: float, direction):
    """``system`` with the field ``field`` along ``direction`` in its ionic potential."""

    if not np.any(np.asarray(direction, dtype=np.float64)):
        return replace(system, ionic_potential=np.asarray(base_ionic_potential))
    return replace(
        system,
        ionic_potential=np.asarray(base_ionic_potential)
        + electric_field_potential(system.grid.coordinates, field, direction),
    )


def warm_start_for_next_field(result) -> SCFWarmStart:
    """``pot%vnew = pot%vold`` and ``firstfilt`` of ``polar.f90``/``parsec.F90``.

    The input potential of the last iteration (with the old field), the
    last Hartree potential as the start of the first Poisson solve and the
    saved subspace, marked for a new SCF loop.  A spin-polarized result
    gives the pairs of its two channels.
    """

    if getattr(result, "k_points", None) is not None and hasattr(result, "density_up"):
        # The collinear spin k-point loop: the input potentials of its last
        # iteration, one per channel (scf_input_potentials, kept by
        # kpoint_scf_runner), are pot%vold(:,1:2).
        return SCFWarmStart(
            input_potential=tuple(
                np.asarray(potential) for potential in result.scf_input_potentials
            ),
            hartree_potential=np.asarray(result.scf_hartree_potential),
            eigensolver_state=None,
        )
    if getattr(result, "k_points", None) is not None:
        # A k-point loop keeps no subspace; the potential of the last
        # iteration (scf_input_potential, kept by kpoint_scf_runner) is
        # pot%vold.
        last = getattr(result, "scf_input_potential", None)
        return SCFWarmStart(
            input_potential=np.asarray(
                result.input_effective_potential if last is None else last
            ),
            hartree_potential=np.asarray(result.hartree_potential),
            eigensolver_state=None,
        )
    channels = getattr(result, "spin", None)
    if channels is not None:
        return SCFWarmStart(
            input_potential=tuple(
                np.asarray(channel.input_effective_potential) for channel in channels
            ),
            hartree_potential=np.asarray(result.hartree_potential),
            eigensolver_state=tuple(
                state_for_new_potential(channel.eigensolver_state) for channel in channels
            ),
        )
    return SCFWarmStart(
        input_potential=np.asarray(result.input_effective_potential),
        hartree_potential=np.asarray(result.hartree_potential),
        eigensolver_state=state_for_new_potential(result.eigensolver_state),
    )


def result_density(result) -> np.ndarray:
    """The total density of a result (``density``, or the sum of the two channels)."""

    density = getattr(result, "density", None)
    if density is not None:
        return np.asarray(density)
    return np.asarray(result.density_up) + np.asarray(result.density_down)


def run_polarizability(
    system,
    field: float,
    *,
    scf_runner: Callable[..., object] = run_scf,
    callback: Callable[[object], None] | None = None,
    before_run: Callable[[int, tuple[float, float, float]], None] | None = None,
    after_run: Callable[[int, object, object, FieldRun], None] | None = None,
    warm_start_builder: Callable[[object], object] | None = warm_start_for_next_field,
):
    """Run the seven field loops of ``polar.f90`` on a prepared isolated system.

    ``scf_runner`` is :func:`parsec_python.SCF.run_scf` or a callable with
    its keywords ``callback`` and ``warm_start``.  ``before_run(index,
    direction)`` is called before each loop and ``after_run(index, system,
    result, run)`` after it, for reporting.  ``warm_start_builder`` turns a
    result into the warm start of the next loop (``pot%vnew = pot%vold``);
    ``None`` starts every loop from the initial density, which the drivers
    without a warm start (spin-orbit, non-collinear, periodic-style spin
    loop) need: the converged loops are the same, their iteration counts not.
    Returns the :class:`PolarizabilityResult`, the system of the last loop
    (with the field along ``-z``) and the result of that loop.
    """

    # A wire or slab is accepted as usrinputfile.F90 accepts pbc%per <= 2
    # (error 156 only for a bulk cell): the field 2 F d.r is applied on the
    # grid as it is, r measured from the centre of the cell, which along a
    # periodic direction is a saw-tooth potential with its step at the cell
    # face; the dipole of dipole.f90 is taken over the cell.  For a k-point
    # SCF scf_runner is kpoint_scf_runner.
    if getattr(system.input, "periodic_cell", None) is not None and not getattr(
        system.grid, "is_confined_periodic", False
    ):
        raise ValueError(
            "the polarizability is computed for isolated systems, wires and slabs "
            "only (PARSEC error 156 for a bulk cell)"
        )
    field = float(field)
    if not np.isfinite(field) or field == 0.0:
        raise ValueError("the polarizability needs a nonzero finite field")
    base = np.asarray(system.ionic_potential, dtype=np.float64)
    runs: list[FieldRun] = []
    warm_start = None
    run_system = system
    result = None
    for index, direction in enumerate(FIELD_DIRECTIONS):
        if before_run is not None:
            before_run(index, direction)
        run_system = field_system(system, base, field, direction)
        result = scf_runner(run_system, callback=callback, warm_start=warm_start)
        record = FieldRun(
            direction=direction,
            total_energy=float(result.energies.total),
            dipole=cluster_dipole_moment(
                result_density(result),
                run_system.grid,
                run_system.atoms,
                run_system.pseudopotentials,
            ),
            converged=bool(result.converged),
            iterations=int(result.iterations),
        )
        runs.append(record)
        if after_run is not None:
            after_run(index, run_system, result, record)
        warm_start = None if warm_start_builder is None else warm_start_builder(result)
    return (
        PolarizabilityResult(field=field, runs=tuple(runs), atom_count=len(system.atoms)),
        run_system,
        result,
    )


def kpoint_scf_runner(k_points, weights, *, iteration_state_callback=None):
    """An ``scf_runner`` of :func:`run_polarizability` for a k-point SCF.

    :func:`parsec_python.SCF.run_scf_kpoints` started from the input
    potential of the warm start (``pot%vnew = pot%vold``); the loop keeps
    its last input potential for the next field.  The k-point loop has no
    subspace to continue from, so every loop starts with its first filter.
    A spin-polarized system runs
    :func:`parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized` from the
    two channel potentials of the warm start (the first loop from the spin
    densities of the initial moments).  ``iteration_state_callback`` goes to
    every loop (``Save_Intermediate_*``).
    """

    from .kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized

    state = (
        {} if iteration_state_callback is None else {"iteration_state_callback": iteration_state_callback}
    )

    def run(system, *, callback=None, warm_start=None):
        if system.input.scf.spin_polarized:
            return run_scf_kpoints_spin_polarized(
                system,
                k_points,
                weights,
                callback=callback,
                keep_wavefunctions=True,
                initial_potentials=(
                    None
                    if warm_start is None
                    else tuple(np.asarray(potential) for potential in warm_start.input_potential)
                ),
                **state,
            )
        initial = None if warm_start is None else np.asarray(warm_start.input_potential)
        return run_scf_kpoints(
            system,
            k_points,
            weights,
            callback=callback,
            initial_potential=initial,
            keep_wavefunctions=True,
            **state,
        )

    return run


__all__ = [
    "CUBIC_ANGSTROM_PER_CUBIC_BOHR",
    "FIELD_DIRECTIONS",
    "FieldRun",
    "PolarizabilityResult",
    "field_system",
    "kpoint_scf_runner",
    "result_density",
    "run_polarizability",
    "warm_start_for_next_field",
]
