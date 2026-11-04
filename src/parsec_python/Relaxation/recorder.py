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

The files are formatted and written by a host thread of their own, in the
order of the events, while the next SCF loop runs: the state of the
optimizer in ``relax_restart.dat`` alone is some 10^5 numbers in the layout
of gfortran for a cluster of a thousand atoms.  What the thread writes is
taken from copies, and :meth:`RelaxationRecorder.finish` waits for it, so the
files are those of a recorder without the thread
(``PARSEC_IONIC_FILE_THREAD=off``, which writes them where the events
happen).  The lines of the log are written at once, in order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import time
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
from .lbfgsb import RestartBlockFormatter


ATOM_COR = "atom.cor"
ITERATE = "iterate.dat"
RELAX_RESTART = "relax_restart.dat"
TRAJECTORY = "relaxation.xyz"


def file_thread_setting() -> str:
    """``PARSEC_IONIC_FILE_THREAD``: ``on`` (default) or ``off``."""

    value = os.environ.get("PARSEC_IONIC_FILE_THREAD", "on").strip().lower() or "on"
    if value in {"1", "true", "yes"}:
        value = "on"
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in {"on", "off"}:
        raise ValueError("PARSEC_IONIC_FILE_THREAD must be on or off")
    return value


def _snapshot(state):
    """A copy of an L-BFGS-B state that later calls of the optimizer do not change."""

    if state is None:
        return None
    return replace(
        state,
        lsave=state.lsave.copy(),
        isave=state.isave.copy(),
        dsave=state.dsave.copy(),
        wa=state.wa.copy(),
        iwa=state.iwa.copy(),
    )


def _copy(array):
    return None if array is None else np.array(array, dtype=np.float64, copy=True)


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
        warm_starts: bool = True,
    ) -> None:
        self.warm_starts = warm_starts
        self.reporter = reporter
        self.write = write
        self.settings = settings
        self.movable_atoms = int(np.count_nonzero(np.asarray(movable, dtype=bool)))
        self.directory = None if directory is None else Path(directory)
        self.elements = None if elements is None else list(elements)
        self._scf: tuple[object, float] | None = None
        self.files: list[Path] = []
        self._writer: ThreadPoolExecutor | None = None
        self._pending: list = []
        self._held: list = []
        if self.directory is not None and file_thread_setting() == "on":
            self._writer = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="parsec-ionic-files"
            )
        # On the file thread the formatter yields the interpreter lock to the
        # SCF loop now and then; without the thread it need not.
        self._block = RestartBlockFormatter(
            pause=None if self._writer is None else (lambda: time.sleep(0))
        )

    # ----- files ---------------------------------------------------------

    def _path(self, name: str) -> Path | None:
        return None if self.directory is None else self.directory / name

    def _later(self, work: Callable[[], None]) -> None:
        """Run ``work`` on the file thread, after the work given before it.

        The work of a movement is held until the next SCF loop begins
        (:meth:`_release`): the rebuild for the moved atoms, which comes
        first, runs on the host too, and the thread would compete with it.
        """

        if self._writer is None:
            work()
            return
        self._held.append(work)

    def _release(self) -> None:
        """Hand the held work to the file thread; raise what the thread raised."""

        if self._writer is None:
            return
        still = []
        for future in self._pending:
            if future.done():
                future.result()
            else:
                still.append(future)
        held, self._held = self._held, []
        still.extend(self._writer.submit(work) for work in held)
        self._pending = still

    def flush(self) -> None:
        """Wait until every file is written; raise what the file thread raised."""

        self._release()
        pending, self._pending = self._pending, []
        for future in pending:
            future.result()

    def _start(self, name: str, text) -> None:
        """Write file ``name`` anew with ``text``, a string or a function returning one."""

        path = self._path(name)
        if path is None:
            return
        self.files.append(path)

        def work():
            # A whole file at every moment: a run stopped while the thread
            # writes keeps the previous relax_restart.dat.
            partial = path.with_name(path.name + ".partial")
            partial.write_text(text() if callable(text) else text, encoding="utf-8", newline="\n")
            os.replace(partial, path)

        self._later(work)

    def _append(self, name: str, text) -> None:
        """Append ``text``, a string or a function returning one, to file ``name``."""

        path = self._path(name)
        if path is None:
            return

        def work():
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(text() if callable(text) else text)

        self._later(work)

    # ----- events --------------------------------------------------------

    def scf_start(self, step: int, system) -> None:
        # The files of the last movement are written while this SCF loop runs.
        self._release()
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
                    *(
                        (
                            f" Structure of movement {step:4d}: the SCF loop continues from",
                            " the potential and the wavefunctions of the previous structure.",
                        )
                        if self.warm_starts
                        else (f" Structure of movement {step:4d}: the SCF loop starts afresh.",)
                    ),
                    "",
                ]
            )
        )

    def scf_criterion(self, step: int, criterion: float, again: bool) -> None:
        if again:
            self.write(
                "\n".join(
                    [
                        "",
                        " The forces of this structure meet Force_Min at a looser SCF",
                        f" criterion: the SCF loop continues to {criterion:.2E} Ry.",
                        "",
                    ]
                )
            )
        else:
            self.write(
                f" SCF criterion of structure {step:4d}: {criterion:.2E} Ry"
                " (Relaxation_SCF_Upscale)"
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
        # The record is not changed after this step: the thread formats it.
        self._append(
            TRAJECTORY,
            lambda: format_xyz_frame(
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
            # The optimizer goes on changing its state: the thread formats a copy.
            restart = RelaxRestart(
                positions=_copy(report.positions),
                iteration=step,
                movable_atoms=self.movable_atoms,
                lbfgsb=_snapshot(state.lbfgsb) if bfgs else None,
                lower=_copy(state.lower) if bfgs else None,
                upper=_copy(state.upper) if bfgs else None,
                bound_step=state.bound_step if bfgs else None,
            )
            self._start(
                RELAX_RESTART, lambda: format_relax_restart(restart, block=self._block)
            )
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
        self.flush()
        if self._writer is not None:
            self._writer.shutdown(wait=True)
            self._writer = None


__all__ = ["ATOM_COR", "ITERATE", "RELAX_RESTART", "RelaxationRecorder", "TRAJECTORY"]
