"""What a command-line driver does for a relaxation, on any backend.

The readable and the accelerated drivers differ in how they prepare a system
and in the :class:`IonicBackend` they pass; reading a restart structure,
deciding about symmetry, running the loop with its recorder and reporting a
single point with forces are the same for both and live here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Callable

import numpy as np

from ..Forces import atom_symmetry_operations, operations_preserving
from ..V_ion import center_cluster_geometry
from ..models import Atom
from .files import (
    RelaxRestart,
    parse_manual_structures,
    parse_relax_restart,
    parse_xyz_frames,
)
from .ionic_loop import IonicBackend, IonicOutcome, run_relaxation
from .recorder import RELAX_RESTART, RelaxationRecorder


MANUAL_FILE = "manual.dat"


def restart_structure(translation, directory: Path, source: Path | None = None):
    """Apply ``Relax_Restart`` (or an explicit restart file) to a translation.

    Returns ``(translation, restart, notes)``.  With ``Relax_Restart`` the
    coordinates of ``relax_restart.dat`` beside the input replace those of
    the input, as in ``initial.f90``, and its BFGS block, if it has one,
    continues the optimizer; a missing file is reported and the relaxation
    starts from scratch.  ``source`` names another file: a
    ``relax_restart.dat`` of PARSEC or of this package, or an XYZ file whose
    last frame gives the coordinates (the optimizer then starts afresh).

    The coordinates of a ``relax_restart.dat`` are used as they are: they
    were recentred by the run that wrote them, and the state of its optimizer
    refers to them.  An XYZ file stands for the ``Atom_Coord`` blocks of the
    input: its structure is recentred as theirs is (not in a ``manual`` run),
    so that a file of any program can be given.
    """

    settings = translation.relaxation
    notes: list[str] = []
    if source is None:
        if settings is None or not settings.restart:
            return translation, None, notes
        source = Path(directory) / RELAX_RESTART
        if not source.is_file():
            notes.append(
                f"WARNING! {RELAX_RESTART} not found; "
                "start structural relaxation from scratch"
            )
            return translation, None, notes
    source = Path(source)
    problem = translation.problem
    atom_count = len(problem.atoms)
    text = source.read_text(encoding="utf-8", errors="replace")
    restart = None
    if source.suffix.lower() == ".xyz":
        labels, positions = parse_xyz_frames(text)[-1]
        if len(labels) != atom_count:
            raise ValueError(
                f"{source.name} holds {len(labels)} atoms, the input {atom_count}"
            )
        # The coordinates are assigned by position in the file, so the file
        # must list the atoms as the input does: type by type.
        for index, (label, atom) in enumerate(zip(labels, problem.atoms)):
            specification = problem.pseudopotentials.get(atom.symbol)
            element = getattr(specification, "element_symbol", None) or atom.symbol
            if label.lower() not in {atom.symbol.lower(), element.lower()}:
                raise ValueError(
                    f"{source.name}: atom {index + 1} is {label}, the input has "
                    f"{atom.symbol} there; an XYZ file must list the atoms in the "
                    "order of the input"
                )
        recenter = problem.recenter_geometry
    else:
        restart = parse_relax_restart(text, atom_count)
        positions = restart.positions
        recenter = False
    atoms = tuple(
        Atom(atom.symbol, position) for atom, position in zip(problem.atoms, positions)
    )
    notes.append(
        "WARNING: atom coordinates in input were replaced by coordinates in "
        f"{source.name}"
    )
    if restart is not None and restart.lbfgsb is not None and settings is not None:
        if settings.method == "bfgs":
            notes.append(" RESTART LBFGS FROM PREVIOUS RUN")
        else:
            restart = replace(restart, lbfgsb=None)
    translation = replace(
        translation, problem=replace(problem, atoms=atoms, recenter_geometry=recenter)
    )
    return translation, restart, notes


def symmetry_decision(translation) -> tuple[bool, str | None]:
    """Whether the calculation may use the point group, and why not.

    PARSEC uses no symmetry with ``Ignore_Symmetry``, for manual movements
    and in molecular dynamics.  One more case is refused here: movable flags
    that a symmetry operation of the structure does not respect, because
    such a relaxation leaves the group it started in.
    """

    if translation.ignore_symmetry:
        return False, None
    if not translation.use_symmetry:
        return False, None
    if translation.relaxation is None:
        return True, None
    problem = translation.problem
    atoms = (
        center_cluster_geometry(problem.atoms)
        if problem.recenter_geometry
        else tuple(problem.atoms)
    )
    operations = atom_symmetry_operations(atoms)
    if len(operations_preserving(operations, translation.movable)) != len(operations):
        return False, (
            "Move_Flag fixes atoms that a symmetry operation of the structure "
            "exchanges with movable ones; the relaxation uses no symmetry."
        )
    return True, None


def manual_structures(translation, directory: Path):
    """Structures of a ``manual`` run from ``manual.dat``, or ``None``."""

    settings = translation.relaxation
    if settings is None or settings.method != "manual" or settings.max_movements == 0:
        return None
    path = Path(directory) / MANUAL_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Minimization: manual with Movement_Num {settings.max_movements} "
            f"needs {MANUAL_FILE} beside the input"
        )
    return parse_manual_structures(
        path.read_text(encoding="utf-8", errors="replace"),
        len(translation.problem.atoms),
        settings.max_movements,
    )


def run_recorded_relaxation(
    translation,
    system,
    backend: IonicBackend,
    reporter,
    write: Callable[[str], None],
    *,
    directory: Path | None,
    input_directory: Path | None = None,
    restart: RelaxRestart | None = None,
    symmetrize: bool = True,
) -> IonicOutcome:
    """Run the relaxation of ``translation`` with the standard recorder.

    ``directory`` receives ``atom.cor``, ``relax_restart.dat`` and the XYZ
    trajectory.  ``manual.dat`` is read from ``input_directory``, by default
    the same one.
    """

    settings = translation.relaxation
    source = directory if input_directory is None else input_directory
    elements = [
        translation.problem.pseudopotentials[atom.symbol].element_symbol or atom.symbol
        for atom in system.atoms
    ]
    recorder = RelaxationRecorder(
        reporter=reporter,
        write=write,
        settings=settings,
        movable=translation.movable,
        directory=directory,
        elements=elements,
    )
    return run_relaxation(
        system,
        settings,
        backend,
        movable=translation.movable,
        observer=recorder,
        restart=restart,
        manual_structures=(
            None if source is None else manual_structures(translation, source)
        ),
        symmetrize=symmetrize,
        output_level=translation.output_level,
    )


def run_single_point_with_forces(
    system,
    backend: IonicBackend,
    reporter,
    *,
    symmetrize: bool = True,
):
    """One SCF loop and its forces, reported; returns ``(result, forces)``."""

    started = time.perf_counter()
    result = backend.run_scf(system, callback=reporter.iteration, warm_start=None)
    elapsed = time.perf_counter() - started
    forces = backend.compute_forces(
        system, result, symmetrize=symmetrize, operations=None
    )
    reporter.finish(result, elapsed, forces=forces)
    return result, forces


def relaxation_header_lines(translation) -> list[str]:
    """What ``usrinputfile.F90`` prints about the minimization."""

    settings = translation.relaxation
    if settings is None:
        return [" No minimization!", ""]
    lines = [" Minimization data:", " ------------------"]
    if settings.method == "simple":
        lines += [
            " Simple minimization!",
            " This scheme is not robust!",
        ]
    elif settings.method == "bfgs":
        lines.append(" Limited-memory BFGS minimization!")
        if settings.bfgs_bound > 0.0:
            lines += [
                " Restricted BFGS",
                " Maximum displacement of atoms from initial position is "
                f"{settings.bfgs_bound:.6f}",
                "  bohrs along each cartesian direction.",
            ]
    else:
        lines.append(" atoms move according to info in manual.dat")
    lines.append(f" Maximum number of movements  = {settings.max_movements:4d}")
    if settings.method == "simple":
        lines += [
            f" Maximum step size  = {settings.maximum_step:8.5f} [au]",
            f" Mininum step size  = {settings.minimum_step:8.5f} [au]",
        ]
    if settings.method != "manual":
        lines.append(f" Minimum force  = {settings.minimum_force:8.5f} [Ry/au]")
    if settings.restart:
        lines += [
            " Restart relaxation, information on previous",
            " runs read from relax_restart.dat file.",
        ]
    if settings.method == "bfgs":
        lines.append(
            f" Number of corrections used in LBFGS update = {settings.bfgs_corrections:4d}"
        )
    movable = int(np.count_nonzero(np.asarray(translation.movable, dtype=bool)))
    lines.append(f" Movable atoms = {movable:6d} of {len(translation.movable):6d}")
    extrapolation = getattr(translation, "potential_extrapolation", "none")
    if extrapolation != "none":
        # Not a line of PARSEC, which always starts from the last potential.
        lines.append(f" Potential extrapolation after a move: {extrapolation}")
    return [*lines, ""]


__all__ = [
    "MANUAL_FILE",
    "manual_structures",
    "relaxation_header_lines",
    "restart_structure",
    "run_recorded_relaxation",
    "run_single_point_with_forces",
    "symmetry_decision",
]
