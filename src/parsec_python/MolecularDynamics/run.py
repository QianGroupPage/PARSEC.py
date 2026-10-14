"""A molecular dynamics run with the electronic structure of this package.

:func:`run_recorded_molecular_dynamics` is the atom movement loop of
``parsec.F90`` for molecular dynamics (see :mod:`.loop`) with a real SCF
calculation behind every structure: the forces of the structure go to
:func:`moldyn_step`, the atoms are moved on the unchanged grid, and the next
SCF loop continues from the potential and the wavefunctions of the previous
one.  It writes the files of a PARSEC run (``md_nrg.dat``, ``md_mech.dat``,
``atom.cor``, ``movie.xyz``) step by step and reports every structure
through the text reporter of the run.

As in PARSEC, molecular dynamics uses no point-group symmetry, and the mean
force is removed before the integrator sees the forces.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import time
from typing import Callable, Sequence

import numpy as np

from ..Relaxation.backend import IonicBackend, release_orbitals
from ..models import Atom
from .atomic_masses import atomic_masses
from .md_files import (
    MD_NRG_HEADER,
    MoldynRecords,
    format_md_nrg_line,
    format_parsec_out_summary,
    initial_records,
    parse_md_restart,
    restart_molecular_dynamics,
    step_records,
)
from .moldyn import (
    MolecularDynamicsState,
    MoldynStep,
    moldyn_step,
    start_molecular_dynamics,
)


MD_NRG = "md_nrg.dat"
MD_MECH = "md_mech.dat"
MD_RESTART = "mdinit.dat"
ATOM_COR = "atom.cor"
MOVIE = "movie.xyz"


@dataclass(frozen=True)
class MolecularDynamicsRecord:
    """One computed structure.  Energies in Ry, forces in Ry/bohr."""

    index: int
    positions: np.ndarray
    energy: float
    converged: bool
    scf_iterations: int
    maximum_force: float
    seconds: float
    step: MoldynStep | None


@dataclass
class MolecularDynamicsOutcome:
    """What a molecular dynamics run returns.

    ``system``, ``result`` and ``forces`` belong to the last structure that
    was computed; ``state`` is the integrator state after the last step and
    ``positions`` the structure it would compute next.
    """

    records: list[MolecularDynamicsRecord] = field(default_factory=list)
    system: object | None = None
    result: object | None = None
    forces: object | None = None
    state: MolecularDynamicsState | None = None
    positions: np.ndarray | None = None
    scf_failed: bool = False
    files: tuple[Path, ...] = ()


def restart_structure(translation, input_directory: Path):
    """Apply ``Restart_mode`` to a translation before its system is prepared.

    Returns ``(translation, restart, notes)``.  The coordinates of
    ``mdinit.dat`` beside the input replace those of the input.  They are
    recentred for the first SCF calculation as ``initial.f90`` recentres
    them, which the preparation of the system does; the integrator itself
    continues from the coordinates of the file.
    """

    settings = translation.molecular_dynamics
    if settings is None or not settings.restart:
        return translation, None, []
    path = Path(input_directory) / MD_RESTART
    if not path.is_file():
        raise FileNotFoundError(f"Restart_mode needs {MD_RESTART} beside the input")
    problem = translation.problem
    restart = parse_md_restart(
        path.read_text(encoding="utf-8", errors="replace"), len(problem.atoms)
    )
    atoms = tuple(
        Atom(atom.symbol, position)
        for atom, position in zip(problem.atoms, restart.current_positions)
    )
    translation = replace(
        translation, problem=replace(problem, atoms=atoms, recenter_geometry=True)
    )
    return translation, restart, [
        f" Molecular dynamics restarted from {MD_RESTART}, step {restart.frame}"
    ]


class _Files:
    """The four output files of a run, appended to as the run proceeds."""

    def __init__(self, directory: Path | None) -> None:
        self.directory = None if directory is None else Path(directory)
        self.started: list[Path] = []

    def append(self, records: MoldynRecords) -> None:
        if self.directory is None:
            return
        for name, text in (
            (MD_NRG, records.md_nrg),
            (MD_MECH, records.md_mech),
            (ATOM_COR, records.atom_cor),
            (MOVIE, records.movie_xyz),
        ):
            path = self.directory / name
            if path not in self.started:
                path.write_text("", encoding="utf-8", newline="\n")
                self.started.append(path)
            if text:
                with path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(text)


def seeded_random_numbers() -> str:
    """``PARSEC_MD_RANDOM_NUMBERS``: ``gfortran`` (default) or ``numpy``.

    Without ``test_rng`` PARSEC draws from the ``random_number`` of its
    compiler, started from ``rng_seed`` when that is positive.  A run with
    such a seed draws here what a PARSEC built by gfortran 8 or later draws,
    so that the two give the same trajectory; ``numpy`` takes NumPy's
    generator with that seed instead.  A seed of zero means the clock in
    PARSEC and fresh entropy here.
    """

    import os

    value = os.environ.get("PARSEC_MD_RANDOM_NUMBERS", "gfortran").strip().lower() or "gfortran"
    if value not in {"gfortran", "numpy"}:
        raise ValueError("PARSEC_MD_RANDOM_NUMBERS must be gfortran or numpy")
    return value


def _seeded_compiler(settings, random) -> str | None:
    if random is not None or settings.use_test_rng or settings.rng_seed <= 0:
        return None
    return "gfortran" if seeded_random_numbers() == "gfortran" else None


def run_recorded_molecular_dynamics(
    translation,
    system,
    backend: IonicBackend,
    reporter,
    write: Callable[[str], None],
    *,
    directory: Path | None,
    restart=None,
    random=None,
    elements: Sequence[str] | None = None,
) -> MolecularDynamicsOutcome:
    """Run the molecular dynamics of ``translation`` on a prepared system.

    ``restart`` is what :func:`restart_structure` returned.  ``random`` is a
    NumPy generator or a seed for the initial velocities and the Langevin
    force where the input selects neither ``test_rng`` nor ``rng_seed``; with
    ``rng_seed`` alone the numbers are those of :func:`seeded_random_numbers`.
    ``directory`` receives the files of the run; ``None`` writes none.
    """

    settings = translation.molecular_dynamics
    species = [atom.symbol for atom in system.atoms]
    if elements is None:
        elements = [
            translation.problem.pseudopotentials[symbol].element_symbol or symbol
            for symbol in species
        ]
    masses = atomic_masses(elements)
    positions = np.array([atom.position for atom in system.atoms], dtype=np.float64)
    if restart is not None:
        state = restart_molecular_dynamics(settings, restart, masses)
    else:
        state = start_molecular_dynamics(settings, positions, masses, translation.movable)
    source = settings.random_source(random, compiler=_seeded_compiler(settings, random))

    write(format_parsec_out_summary(settings).rstrip("\n"))
    files = _Files(directory)
    files.append(initial_records(state, species, positions))
    outcome = MolecularDynamicsOutcome(state=state, positions=positions)
    carried = None
    index = 0
    while True:
        started = time.perf_counter()
        result = backend.run_scf(system, callback=reporter.iteration, warm_start=carried)
        elapsed = time.perf_counter() - started
        forces = backend.compute_forces(system, result, symmetrize=False, operations=None)
        reporter.finish(result, elapsed, forces=forces)
        energy = float(result.energies.total)
        outcome.system, outcome.result, outcome.forces = system, result, forces
        outcome.scf_failed = outcome.scf_failed or not result.converged
        step = None
        if settings.step_num != 0:
            step = moldyn_step(settings, state, forces.total, energy, random=source)
            files.append(step_records(state, step, species))
            write(
                "\n".join(
                    [
                        "",
                        " Molecular dynamics step:",
                        " " + MD_NRG_HEADER.rstrip("\n").replace("\n", "\n "),
                        " " + format_md_nrg_line(step).rstrip("\n"),
                        "",
                    ]
                )
            )
        outcome.records.append(
            MolecularDynamicsRecord(
                index=index,
                positions=np.array(
                    [atom.position for atom in system.atoms], dtype=np.float64
                ),
                energy=energy,
                converged=bool(result.converged),
                scf_iterations=int(result.iterations),
                maximum_force=float(forces.maximum),
                seconds=time.perf_counter() - started,
                step=step,
            )
        )
        if step is None:
            break
        state = step.state
        outcome.state = state
        outcome.positions = np.asarray(step.positions, dtype=np.float64)
        if step.finished:
            break
        atoms = tuple(
            Atom(atom.symbol, position)
            for atom, position in zip(system.atoms, step.positions)
        )
        release_orbitals(result)
        moved = backend.update_geometry(system, atoms)
        carried = backend.warm_start(system, result, moved)
        system = moved
        index += 1
    outcome.files = tuple(files.started)
    if files.started:
        write(" Files: " + ", ".join(path.name for path in files.started))
    return outcome


__all__ = [
    "ATOM_COR",
    "MD_MECH",
    "MD_NRG",
    "MD_RESTART",
    "MOVIE",
    "MolecularDynamicsOutcome",
    "MolecularDynamicsRecord",
    "restart_structure",
    "run_recorded_molecular_dynamics",
]
