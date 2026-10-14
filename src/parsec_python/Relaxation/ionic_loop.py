"""The ionic loop of ``parsec.F90``: SCF, forces, move the atoms, repeat.

One pass of the loop solves the electronic structure of the current
structure, evaluates the atomic forces and hands both to the mover
(:func:`parsec_python.Relaxation.domove` for a relaxation).  When the mover
returns a new structure, everything that depends on the atom positions is
rebuilt on the same grid and the next SCF loop continues from the potential
and the wavefunctions of the previous one.

The loop itself is independent of how those four operations are executed:
an :class:`IonicBackend` supplies them, so the readable NumPy path and the
accelerated paths run the same control flow.  What happens with the numbers
(``parsec.out``, ``atom.cor``, restart files, trajectories) is left to an
:class:`IonicObserver`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Callable, Sequence

import numpy as np

from ..Forces import (
    atom_symmetry_operations,
    operations_preserving,
    symmetrize_positions,
)
from ..models import Atom
from .backend import IonicBackend, reference_backend, release_orbitals
from .domove import MoveReport, MovementState, RelaxationSettings, domove
from .files import RYDBERG_TO_EV, RelaxRestart


class IonicObserver:
    """Receives the events of an ionic loop; every method may be overridden.

    ``step`` is ``move%num`` of PARSEC: 0 for the input structure.
    """

    def scf_start(self, step: int, system) -> None:
        """An SCF loop is about to start for structure ``step``."""

    def scf_iteration(self, step: int, iteration) -> None:
        """One SCF iteration of structure ``step`` finished."""

    def scf_finish(self, step: int, system, result, seconds: float) -> None:
        """The SCF loop of structure ``step`` returned."""

    def forces(self, step: int, system, forces) -> None:
        """The forces of structure ``step`` are available."""

    def movement(
        self,
        step: int,
        record: "IonicStepRecord",
        report: MoveReport | None,
        state: MovementState,
    ) -> None:
        """The mover was called after structure ``step`` (``None``: not called)."""

    def finish(self, outcome: "IonicOutcome") -> None:
        """The loop ended."""


@dataclass(frozen=True)
class IonicStepRecord:
    """One evaluated structure.  Energies in Ry, forces in Ry/bohr."""

    step: int
    symbols: tuple[str, ...]
    positions: np.ndarray
    energy: float
    energy_per_atom_ev: float
    converged: bool
    scf_iterations: int
    forces: object
    maximum_force: float
    seconds: float


@dataclass
class IonicOutcome:
    """What an ionic loop returns.

    ``system`` and ``result`` belong to the last structure that was
    computed.  ``positions`` are the coordinates a continuation should start
    from: those of the last computed structure when the run stopped on its
    force criterion, otherwise the structure the mover asked for next.

    ``timings`` holds, for every structure, the wall seconds of its parts:
    ``scf`` (the SCF loop with its per-iteration reports), ``forces``,
    ``report`` (what the observer did with the result, the forces and the
    movement: log and files), ``move`` (the mover) and ``update`` (rebuilding
    what depends on the atom positions and handing over the wavefunctions).
    """

    records: list[IonicStepRecord] = field(default_factory=list)
    system: object | None = None
    result: object | None = None
    positions: np.ndarray | None = None
    minimum_force_found: bool = False
    done: bool = False
    movements_exhausted: bool = False
    scf_failed: bool = False
    error: int = 0
    state: MovementState | None = None
    timings: list[dict[str, float]] = field(default_factory=list)


def run_relaxation(
    system,
    settings: RelaxationSettings,
    backend: IonicBackend,
    *,
    movable: Sequence[bool] | None = None,
    observer: IonicObserver | None = None,
    restart: RelaxRestart | None = None,
    manual_structures: Sequence[np.ndarray] | None = None,
    warm_start: bool = True,
    symmetrize: bool = True,
    output_level: int = 1,
) -> IonicOutcome:
    """Relax a prepared system; the control flow of ``parsec.F90``.

    ``movable`` marks the atoms that move (all of them by default, and
    always all of them in a ``manual`` run).  ``restart`` supplies the
    optimizer state of a previous run; the caller has already placed the
    atoms at its coordinates.  ``warm_start=False`` starts every SCF loop
    from superposed atomic densities, for testing.

    With ``symmetrize`` the point-group operations of the input structure
    that respect the movable flags are found once, as PARSEC finds its
    operations once, and every structure is kept on them: forces are averaged
    over the group, and so are the positions the mover returns, which removes
    the round-off that would otherwise accumulate.  A ``manual`` run uses no
    symmetry, as in PARSEC.

    With ``Movement_Num`` equal to ``M`` at most ``M + 1`` structures are
    computed.  The mover is called after each of them; the run ends when the
    largest force on a movable atom is below ``Force_Min``, when the
    optimizer stops, or when ``M`` movements have been used.
    """

    watch = IonicObserver() if observer is None else observer
    atom_count = len(system.atoms)
    mask = (
        np.ones(atom_count, dtype=bool)
        if movable is None or settings.method == "manual"
        else np.asarray(movable, dtype=bool)
    )
    if mask.size != atom_count:
        raise ValueError("one movable flag per atom is required")
    if not mask.any():
        raise ValueError("a relaxation needs at least one movable atom")

    operations = None
    if symmetrize and settings.method != "manual":
        operations = operations_preserving(
            atom_symmetry_operations(system.atoms), mask
        )

    state = MovementState()
    outcome = IonicOutcome(state=state)
    carried = None
    while True:
        step = state.number
        started = time.perf_counter()
        watch.scf_start(step, system)
        result = backend.run_scf(
            system,
            callback=lambda iteration, step=step: watch.scf_iteration(step, iteration),
            warm_start=carried,
        )
        scf_done = time.perf_counter()
        watch.scf_finish(step, system, result, scf_done - started)
        forces_started = time.perf_counter()
        forces = backend.compute_forces(
            system, result, symmetrize=operations is not None, operations=operations
        )
        forces_done = time.perf_counter()
        watch.forces(step, system, forces)
        clock = {
            "step": float(step),
            "scf": scf_done - started,
            "forces": forces_done - forces_started,
            "report": forces_started - scf_done + time.perf_counter() - forces_done,
            "move": 0.0,
            "update": 0.0,
        }
        outcome.timings.append(clock)

        positions = np.array([atom.position for atom in system.atoms], dtype=np.float64)
        symbols = tuple(atom.symbol for atom in system.atoms)
        energy = float(result.energies.total)
        if not (np.isfinite(energy) and np.all(np.isfinite(forces.total))):
            # L-BFGS-B takes an infinite energy for a converged minimum.
            raise FloatingPointError(
                f"the total energy or a force of structure {step} is not finite: "
                "the SCF loop of this structure failed, and no movement was made from it"
            )
        moving = np.asarray(forces.total)[mask]
        record = IonicStepRecord(
            step=step,
            symbols=symbols,
            positions=positions,
            energy=energy,
            energy_per_atom_ev=RYDBERG_TO_EV * energy / atom_count,
            converged=bool(result.converged),
            scf_iterations=int(result.iterations),
            forces=forces,
            maximum_force=float(
                np.max(np.linalg.norm(moving, axis=1), initial=0.0)
            ),
            seconds=time.perf_counter() - started,
        )
        outcome.records.append(record)
        outcome.system, outcome.result = system, result
        outcome.positions = positions
        outcome.scf_failed = outcome.scf_failed or not result.converged

        if settings.max_movements == 0:
            watch.movement(step, record, None, state)
            break
        move_started = time.perf_counter()
        report = domove(
            state,
            settings,
            positions,
            mask,
            forces.total,
            energy,
            manual_structures=manual_structures,
            restart=restart if step == 0 else None,
            output_level=output_level,
        )
        if report.moved and operations is not None:
            report = replace(
                report, positions=symmetrize_positions(report.positions, operations)
            )
        move_done = time.perf_counter()
        clock["move"] = move_done - move_started
        watch.movement(step, record, report, state)
        clock["report"] += time.perf_counter() - move_done
        outcome.minimum_force_found = report.minimum_force_found
        outcome.done = report.done
        outcome.error = report.error
        if report.minimum_force_found or report.done:
            break
        outcome.positions = report.positions
        state.number += 1
        if state.number > settings.max_movements:
            outcome.movements_exhausted = True
            break

        atoms = tuple(
            Atom(atom.symbol, position)
            for atom, position in zip(system.atoms, report.positions)
        )
        # The eigensolver keeps the subspace the next loop continues from;
        # views of it held by the result of this structure would only pin
        # device memory.  The result of the last structure keeps its own.
        update_started = time.perf_counter()
        release_orbitals(result)
        moved_system = backend.update_geometry(system, atoms)
        carried = (
            backend.warm_start(system, result, moved_system) if warm_start else None
        )
        system = moved_system
        clock["update"] = time.perf_counter() - update_started

    watch.finish(outcome)
    return outcome


__all__ = [
    "IonicBackend",
    "IonicObserver",
    "IonicOutcome",
    "IonicStepRecord",
    "reference_backend",
    "run_relaxation",
]
