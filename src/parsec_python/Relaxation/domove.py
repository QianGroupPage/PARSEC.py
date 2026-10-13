"""One ionic step of a structural relaxation (``domove.f90``).

PARSEC minimizes the total energy over the coordinates of the movable atoms
with one of three schemes:

``simple``
    steepest descent with a displacement that saturates between ``Min_Step``
    and ``Max_Step``;
``bfgs``
    limited-memory BFGS (L-BFGS-B 3.0 of Nocedal and Morales, driven by
    reverse communication), optionally with every coordinate confined to
    ``Max_Step`` around its initial value;
``manual``
    no minimization: the structures are read from ``manual.dat``.

:func:`domove` is called once after every SCF calculation with the energy and
the forces of the current structure and returns the next one.  It does no
file I/O: the texts PARSEC writes to ``parsec.out``, ``atom.cor`` and
``relax_restart.dat`` are returned or produced by the formatters of
:mod:`parsec_python.Relaxation.files`.

Units are PARSEC's: bohr, Rydberg and Ry/bohr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..models import DEFAULT_RELAXATION_MAXIMUM_STEP, RelaxationSettings
from .files import RelaxRestart
from .lbfgsb import LBFGSBState, setulb


STEEPEST_DESCENT = "simple"
BFGS = "bfgs"
MANUAL = "manual"
METHODS = (STEEPEST_DESCENT, BFGS, MANUAL)

#: ``xtol`` of ``domove.f90``, passed to L-BFGS-B as both ``factr`` and
#: ``pgtol``: the optimizer itself practically never declares convergence;
#: the force criterion of :func:`domove` does.
LBFGSB_TOLERANCE = 1.0e-15

#: Error flags of ``domove.f90``.
ERROR_LBFGSB = 541
ERROR_LBFGSB_ABNORMAL = 542


@dataclass
class MovementState:
    """``type(movement)`` of PARSEC: what a relaxation carries between steps.

    ``number`` is ``move%num``, the index of the structure whose forces are
    being used (0 for the input structure).  ``current`` holds the
    coordinates of the movable atoms, shape ``(movable atoms, 3)`` in bohr;
    its flattened row-major order is the variable order of L-BFGS-B, which is
    the column-major order of the Fortran ``rcur(3, nmovable)``.
    """

    number: int = 0
    current: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    bound_step: float = -1.0
    lbfgsb: LBFGSBState | None = None
    minimum_force_found: bool = False
    done: bool = False


@dataclass(frozen=True)
class MoveReport:
    """Outcome of one :func:`domove` call.

    ``positions`` are the coordinates of all atoms for the next SCF
    calculation; they equal the input positions when the relaxation ended in
    this call.  ``moved`` says whether a new structure was produced.
    ``maximum_force`` is the largest force magnitude over the movable atoms.
    ``messages`` are the lines PARSEC writes to ``parsec.out`` and
    ``atom_cor_messages`` those it adds to ``atom.cor`` besides the
    energy line and the coordinates.  ``error`` is ``ierr`` of
    ``domove.f90``: zero, 541 or 542.  ``optimizer_output`` holds what the
    L-BFGS-B routines wrote in this call as ``(unit, text)``: unit 8 is
    ``iterate.dat`` and unit 9 the report PARSEC sends to ``out.*``.
    """

    positions: np.ndarray
    maximum_force: float
    moved: bool
    minimum_force_found: bool
    done: bool
    task: str = ""
    messages: tuple[str, ...] = ()
    atom_cor_messages: tuple[str, ...] = ()
    error: int = 0
    optimizer_output: tuple[tuple[int, str], ...] = ()


def steepest_descent_displacement(
    negative_forces: np.ndarray, settings: RelaxationSettings
) -> np.ndarray:
    """Displacements of the ``simple`` scheme, component by component.

    With ``g`` the negative of a force component, ``domove.f90`` moves by

    ``-sign(g)*(stepmin + (stepmax-stepmin)*(1 - exp(-c*(|g| - fmin))))``,

    ``c = log((stepmax-stepmin)/(stepmax-50*stepmin))/(1000*fmin)``,

    so that a component of size ``fmin`` moves by ``stepmin`` and a huge one
    by ``stepmax``.  A component that is exactly zero does not move.
    """

    stepmax = settings.maximum_step
    stepmin = settings.minimum_step
    fmin = abs(settings.minimum_force)
    coefficient = np.log((stepmax - stepmin) / (stepmax - 50.0 * stepmin)) / (
        1000.0 * settings.minimum_force
    )
    gradient = np.asarray(negative_forces, dtype=np.float64)
    magnitude = np.abs(gradient)
    step = stepmin + (stepmax - stepmin) * (
        1.0 - np.exp(-coefficient * (magnitude - fmin))
    )
    return np.where(gradient == 0.0, 0.0, -np.sign(gradient) * step)


def _print_level(output_level: int) -> int:
    """``iprint`` of ``domove.f90`` for a PARSEC output level."""

    return 100 if output_level >= 3 else 1


def _drive_lbfgsb(
    state: MovementState,
    energy: float,
    gradient: np.ndarray,
    output_level: int,
    sink,
) -> str:
    """Call ``setulb`` until it asks for a new energy and gradient, or stops.

    Returns the leading word of the task: ``FG``, ``CONV``, ``WARN``,
    ``ERROR`` or ``ABNO``.  ``state.current`` is updated in place.
    """

    solver = state.lbfgsb
    count = state.current.size
    bounded = 2 if state.bound_step > 0.0 else 0
    kinds = np.full(count, bounded, dtype=np.int64)
    coordinates = state.current.reshape(-1)
    while True:
        setulb(
            solver,
            coordinates,
            state.lower,
            state.upper,
            kinds,
            float(energy),
            # setulb overwrites the gradient when it restores an iterate.
            np.array(gradient, dtype=np.float64),
            LBFGSB_TOLERANCE,
            LBFGSB_TOLERANCE,
            iprint=_print_level(output_level),
            output=sink,
        )
        task = solver.task
        if task[:2] == "FG":
            return "FG"
        for word in ("CONV", "WARN", "ERROR", "ABNO"):
            if task.startswith(word):
                return word
        # NEW_X and any other task: call again, as domove.f90 does.


def domove(
    state: MovementState,
    settings: RelaxationSettings,
    positions: np.ndarray,
    movable: Sequence[bool],
    forces: np.ndarray,
    total_energy: float,
    *,
    manual_structures: Sequence[np.ndarray] | None = None,
    restart: RelaxRestart | None = None,
    output_level: int = 1,
) -> MoveReport:
    """Move the atoms once; translate ``domove.f90`` statement by statement.

    ``positions`` and ``forces`` have shape ``(atoms, 3)``; ``movable`` marks
    the atoms that move (``clust%mvat``).  ``total_energy`` is in Ry.
    ``manual_structures[k]`` holds the movable coordinates of structure
    ``k + 1`` for the ``manual`` scheme.  ``restart`` is the content of
    ``relax_restart.dat`` of a previous run (``Relax_Restart``); its BFGS
    block is used only on the first call, when ``state.number`` is zero, and
    the caller has already placed the atoms at its coordinates.
    ``output_level`` is PARSEC's ``Output_Level``, which sets the print
    level of the optimizer.

    The caller increments ``state.number`` after a call that moved the atoms
    and stops when the report says ``minimum_force_found`` or ``done``, or
    when the number exceeds ``Movement_Num``, exactly as ``parsec.F90``.
    """

    positions = np.array(positions, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)
    mask = np.asarray(movable, dtype=bool)
    if positions.shape != forces.shape or positions.shape[0] != mask.size:
        raise ValueError("positions, forces and movable flags do not match")
    count = int(mask.sum())

    # Negative forces of the movable atoms and the largest force magnitude.
    negative = -forces[mask]
    maximum = float(
        np.max(np.sqrt(np.einsum("ij,ij->i", negative, negative)), initial=0.0)
    )

    if maximum < settings.force_threshold:
        state.minimum_force_found = True
        return MoveReport(
            positions=positions,
            maximum_force=maximum,
            moved=False,
            minimum_force_found=True,
            done=state.done,
            messages=(" Local minimum of total energy was found ",),
            atom_cor_messages=("Local minimum of total energy was found ",),
        )

    optimizer_output: list[tuple[int, str]] = []
    if state.number == 0:
        state.current = positions[mask].copy()
        if settings.method == BFGS:
            state.bound_step = settings.bfgs_bound
            saved = None if restart is None else restart.lbfgsb
            if saved is not None:
                # RESTART LBFGS FROM PREVIOUS RUN: the number of corrections,
                # the bounds and the whole optimizer state come from the
                # file; whether the bounds act is decided by this input.
                if saved.n != 3 * count:
                    raise ValueError(
                        "relax_restart.dat was written for another number of "
                        "movable atoms"
                    )
                # An unrestricted run saves zero bounds.  Made to act, they
                # confine every coordinate to zero: the Fortran then moves
                # all atoms to the origin.
                if (
                    state.bound_step > 0.0
                    and restart.bound_step is not None
                    and restart.bound_step <= 0.0
                ):
                    raise ValueError(
                        "relax_restart.dat was written by an unrestricted BFGS "
                        "run and holds no bounds: Max_Step cannot restrict its "
                        "continuation"
                    )
                state.lower = np.array(restart.lower, dtype=np.float64)
                state.upper = np.array(restart.upper, dtype=np.float64)
                state.lbfgsb = saved.copy()
            else:
                flat = state.current.reshape(-1)
                if state.bound_step > 0.0:
                    state.lower = flat - state.bound_step
                    state.upper = flat + state.bound_step
                else:
                    state.lower = np.zeros(3 * count)
                    state.upper = np.zeros(3 * count)
                state.lbfgsb = LBFGSBState.start(3 * count, settings.bfgs_corrections)
                # The first call only initializes: it returns FG_START.
                setulb(
                    state.lbfgsb,
                    flat,
                    state.lower,
                    state.upper,
                    np.full(3 * count, 2 if state.bound_step > 0.0 else 0, dtype=np.int64),
                    float(total_energy),
                    negative.reshape(-1).copy(),
                    LBFGSB_TOLERANCE,
                    LBFGSB_TOLERANCE,
                    iprint=_print_level(output_level),
                    output=lambda unit, text: optimizer_output.append((unit, text)),
                )

    messages: list[str] = []
    atom_cor: list[str] = []
    task = ""
    if settings.method == STEEPEST_DESCENT:
        state.current = state.current + steepest_descent_displacement(negative, settings)
    elif settings.method == BFGS:
        word = _drive_lbfgsb(
            state,
            total_energy,
            negative.reshape(-1),
            output_level,
            lambda unit, text: optimizer_output.append((unit, text)),
        )
        task = state.lbfgsb.task
        if word != "FG":
            state.done = True
            error = 0
            if word == "CONV":
                messages += ["LBFGS minimization converged ", task]
                atom_cor.append("Local minimum of total energy was found")
            elif word == "WARN":
                messages += ["LBFGS minimization warning", task]
            elif word == "ERROR":
                messages += ["ERROR in LBFGS routines : ", task]
                error = ERROR_LBFGSB
            else:
                messages += ["Abnormal exit from LBFGS routines : ", task]
                error = ERROR_LBFGSB_ABNORMAL
            messages.append("LBFGS is terminating")
            atom_cor.append("LBFGS is terminating")
            return MoveReport(
                positions=positions,
                maximum_force=maximum,
                moved=False,
                minimum_force_found=False,
                done=True,
                task=task,
                messages=tuple(messages),
                atom_cor_messages=tuple(atom_cor),
                error=error,
                optimizer_output=tuple(optimizer_output),
            )
    else:
        # Manual: the next structure comes from manual.dat, unless the
        # results just reported were those of the last one.
        if state.number < settings.max_movements:
            if manual_structures is None or state.number >= len(manual_structures):
                raise ValueError(
                    "manual.dat does not hold a structure for movement "
                    f"{state.number + 1}"
                )
            structure = np.asarray(manual_structures[state.number], dtype=np.float64)
            if structure.shape != (count, 3):
                raise ValueError("a manual.dat structure has the wrong number of atoms")
            state.current = structure.copy()

    positions[mask] = state.current
    return MoveReport(
        positions=positions,
        maximum_force=maximum,
        moved=True,
        minimum_force_found=False,
        done=False,
        task=task,
        messages=tuple(messages),
        atom_cor_messages=tuple(atom_cor),
        optimizer_output=tuple(optimizer_output),
    )


__all__ = [
    "BFGS",
    "DEFAULT_RELAXATION_MAXIMUM_STEP",
    "ERROR_LBFGSB",
    "ERROR_LBFGSB_ABNORMAL",
    "LBFGSB_TOLERANCE",
    "MANUAL",
    "METHODS",
    "MoveReport",
    "MovementState",
    "RelaxationSettings",
    "STEEPEST_DESCENT",
    "domove",
    "steepest_descent_displacement",
]
