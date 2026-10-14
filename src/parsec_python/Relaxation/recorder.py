"""Files and log sections of a structural relaxation.

:class:`RelaxationRecorder` is the :class:`IonicObserver` the command-line
drivers use.  It writes what a PARSEC user expects beside the input,

``atom.cor``
    every structure with its energy and largest force,
``relax_restart.dat``
    the structure the next SCF calculation uses and, for BFGS, the state of
    the optimizer, rewritten after every move,

adds an extended-XYZ trajectory with energies and forces, and sends the SCF
iterations, the force tables and the movement lines of each step to the text
reporter of the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..Output.force_output import movement_report_lines, relaxation_summary_lines
from ..models import RelaxationSettings
from .domove import MoveReport, MovementState
from .files import (
    RelaxRestart,
    format_atom_cor_energy,
    format_atom_cor_final,
    format_atom_cor_structure,
    format_relax_restart,
    format_xyz_frame,
)
from .ionic_loop import IonicObserver, IonicOutcome, IonicStepRecord


ATOM_COR = "atom.cor"
ITERATE = "iterate.dat"
RELAX_RESTART = "relax_restart.dat"
TRAJECTORY = "relaxation.xyz"


class RelaxationRecorder(IonicObserver):
    """Write the files of a relaxation and report its steps.

    ``reporter`` is the text reporter of the run (``iteration`` and
    ``finish`` are called for every structure) and ``write`` its line sink.
    ``directory`` receives the files; ``None`` writes none.  ``elements``
    are the chemical symbols for the XYZ file where a species label is not
    one.
    """

    def __init__(
        self,
        *,
        reporter,
        write: Callable[[str], None],
        settings: RelaxationSettings,
        movable: Sequence[bool],
        directory: Path | None,
        elements: Sequence[str] | None = None,
    ) -> None:
        self.reporter = reporter
        self.write = write
        self.settings = settings
        self.movable_atoms = int(np.count_nonzero(np.asarray(movable, dtype=bool)))
        self.directory = None if directory is None else Path(directory)
        self.elements = None if elements is None else list(elements)
        self._scf: tuple[object, float] | None = None
        self.files: list[Path] = []

    # ----- files ---------------------------------------------------------

    def _path(self, name: str) -> Path | None:
        return None if self.directory is None else self.directory / name

    def _start(self, name: str, text: str) -> None:
        path = self._path(name)
        if path is None:
            return
        path.write_text(text, encoding="utf-8", newline="\n")
        self.files.append(path)

    def _append(self, name: str, text: str) -> None:
        path = self._path(name)
        if path is None:
            return
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    # ----- events --------------------------------------------------------

    def scf_start(self, step: int, system) -> None:
        if step == 0:
            symbols = [atom.symbol for atom in system.atoms]
            positions = np.array([atom.position for atom in system.atoms])
            # initial.f90 opens atom.cor with the input structure.
            self._start(ATOM_COR, format_atom_cor_structure(0, symbols, positions))
            self._start(TRAJECTORY, "")
            return
        self.write(
            "\n".join(
                [
                    "",
                    f" Structure of movement {step:4d}: the SCF loop continues from",
                    " the potential and the wavefunctions of the previous structure.",
                    "",
                ]
            )
        )

    def scf_iteration(self, step: int, iteration) -> None:
        self.reporter.iteration(iteration)

    def scf_finish(self, step: int, system, result, seconds: float) -> None:
        self._scf = (result, seconds)

    def forces(self, step: int, system, forces) -> None:
        result, seconds = self._scf
        self.reporter.finish(result, seconds, forces=forces)

    def movement(
        self,
        step: int,
        record: IonicStepRecord,
        report: MoveReport | None,
        state: MovementState,
    ) -> None:
        self._append(
            TRAJECTORY,
            format_xyz_frame(
                record.symbols,
                record.positions,
                step=step,
                energy=record.energy,
                forces=record.forces.total,
                elements=self.elements,
            ),
        )
        if report is None:
            self.write(
                "\n".join(movement_report_lines(step, record.seconds, called_mover=None))
            )
            return
        self.write(
            "\n".join(movement_report_lines(step, record.seconds, report.messages))
        )
        # What the L-BFGS-B routines print: PARSEC sends unit 9 to its
        # per-process output and unit 8 to iterate.dat.
        for unit, text in report.optimizer_output:
            if unit == 8:
                if self._path(ITERATE) is not None and self._path(ITERATE) not in self.files:
                    self._start(ITERATE, "")
                self._append(ITERATE, text)
            else:
                self.write(text.rstrip("\n"))
        text = format_atom_cor_energy(record.energy_per_atom_ev, report.maximum_force)
        text += "".join(f" {message}\n" for message in report.atom_cor_messages)
        if report.moved:
            # domove.f90 saves the restart information after every move and
            # announces the next structure unless the movements are used up.
            bfgs = self.settings.method == "bfgs"
            restart = RelaxRestart(
                positions=report.positions,
                iteration=step,
                movable_atoms=self.movable_atoms,
                lbfgsb=state.lbfgsb if bfgs else None,
                lower=state.lower if bfgs else None,
                upper=state.upper if bfgs else None,
                bound_step=state.bound_step if bfgs else None,
            )
            self._start(RELAX_RESTART, format_relax_restart(restart))
            if step < self.settings.max_movements:
                text += format_atom_cor_structure(
                    step + 1, record.symbols, report.positions
                )
        self._append(ATOM_COR, text)

    def finish(self, outcome: IonicOutcome) -> None:
        lines = relaxation_summary_lines(
            movements_exhausted=outcome.movements_exhausted,
            minimum_force_found=outcome.minimum_force_found,
            manual=self.settings.method == "manual",
            scf_failed=outcome.scf_failed,
        )
        if outcome.records:
            last = outcome.records[-1]
            lines += [
                f" Structures computed            : {len(outcome.records):8d}",
                f" Total energy of the last one   : {last.energy:20.8f} [Ry]",
                f" Largest force on a movable atom: {last.maximum_force:12.6f} [Ry/bohr]",
                "",
            ]
        if outcome.timings:
            parts = (
                ("scf", "SCF loops"),
                ("forces", "forces"),
                ("move", "mover"),
                ("update", "rebuilding for moved atoms"),
                ("report", "log and files"),
            )
            lines.append(" Wall time of the ionic steps [sec]:")
            lines += [
                f"   {label:<27s}: {sum(item[key] for item in outcome.timings):11.2f}"
                for key, label in parts
            ]
            lines.append("")
        if self.files:
            lines.append(
                " Files: " + ", ".join(path.name for path in dict.fromkeys(self.files))
            )
            lines.append("")
        self.write("\n".join(lines))
        text = ""
        if outcome.movements_exhausted:
            text += " NOTICE: Maximum number of movements used.\n"
        if outcome.positions is not None:
            text += format_atom_cor_final(outcome.positions)
        self._append(ATOM_COR, text)


__all__ = ["ATOM_COR", "ITERATE", "RELAX_RESTART", "RelaxationRecorder", "TRAJECTORY"]
