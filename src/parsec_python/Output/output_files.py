"""The data files of a run: ``parsec.dat``, ``eigen.dat`` and cube files.

``parsec.F90`` writes ``parsec.dat`` (``wfnsave``) and ``eigen.dat``
(``eigensave``) after every SCF loop when ``Output_Data`` is true, the
default, and after every SCF iteration with
``Save_Intermediate_Charge_Density`` / ``Save_Intermediate_Eigenvalues``;
``export_grid_data`` writes the cube files of ``Output_Grid_Data`` at the end
of the run.  :class:`RunOutputFiles` does the same for a Python run, in the
directory of the input.

Python choices, stated in the run log where they apply:

* ``parsec.dat`` is written once, after the last SCF loop of the run (PARSEC
  rewrites it after each structure of a relaxation; the last file is the
  same).  ``eigen.dat`` gets one block per structure, as in PARSEC.
* Python keeps its fields on the full grid, where PARSEC writes the
  symmetry wedge, so the wavefunctions of the default selection (the
  occupied states, PARSEC's ``nsave = -1``) can be many times larger than
  PARSEC's.  Where they would exceed
  :data:`DEFAULT_WAVEFUNCTION_LIMIT_BYTES` the default file holds no
  wavefunctions (a restart does not read them); ``Save_Wfn_Bands`` or
  ``Output_All_States`` in the input always write what they ask for.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from ..models import ExchangeLoopIteration, OutputFileSettings, WavefunctionSelection
from .cube import CUBE_FILE_STEMS, grid_data_fields, write_parsec_cube
from .eigen_dat import (
    EigenDatFile,
    eigen_dat_block,
    iteration_eigen_channels,
    result_eigen_channels,
)
from .parsec_dat import parsec_dat_from_result, wavefunction_bytes, write_parsec_dat

#: Largest wavefunction payload of the default ``parsec.dat`` (bytes).
DEFAULT_WAVEFUNCTION_LIMIT_BYTES = 512 * 2**20


def _fortran_integer(value: int) -> str:
    """An integer as gfortran's list-directed output prints it."""

    return f"{int(value):12d}"


def output_header_lines(settings: OutputFileSettings, *, kpoints: bool = False) -> list[str]:
    """What ``usrinputfile.F90`` prints about the keywords the input gives.

    A run that gives none of them prints nothing, so that its log keeps the
    lines it had.
    """

    given = settings.keywords
    lines: list[str] = []
    if "output_data" in given and not settings.output_data:
        lines += [
            " NOTICE: Raw Data output has been completely disabled",
            " NOTICE: All other output flags but GW will be ingnored",
        ]
    selection = settings.wavefunctions
    if selection.explicit:
        if selection.mode == "all":
            lines.append(" Writing all calculated wave functions to parsec.dat file")
        elif selection.indices:
            lines.append(
                f" Writing {_fortran_integer(len(selection.indices))} "
                "selected wave functions to parsec.dat file:"
            )
            numbers = list(selection.indices)
            for start in range(0, len(numbers), 10):
                lines.append("".join(f"{value:6d}" for value in numbers[start : start + 10]))
        else:
            lines.append(" No wave functions are written to  parsec.dat file")
    if settings.intermediate_density:
        lines += [" Writing charge density to parsec.dat file after each step of SCF", ""]
    texts = {
        "charge_density": " Writing total charge density to external file",
        "v_external": " Writing external potential to external file",
        "v_hartree": " Writing hartree potential to external file",
        "v_external+v_hartree": (
            " Writing sum of external and hartree potentials to external file"
        ),
    }
    if "output_grid_data" in given:
        for name in settings.grid_data:
            lines += [texts[name], ""]
    if "output_eigenvalues" in given or kpoints:
        if settings.eigenvalues:
            flag = 1 + (1 if settings.all_states else 0) + (2 if settings.intermediate_eigenvalues else 0)
            lines.append(" Generating Eigenvalue Output File!")
            if settings.all_states:
                lines += [" Writing all calculated eigenvalues to eigen.dat", ""]
            if settings.intermediate_eigenvalues:
                lines += [" Writing eigenvalues to eigen.dat after each step of SCF", ""]
            lines += [f" outevflag = {_fortran_integer(flag)}", ""]
        else:
            lines += [" No eigenvalue output File!", f" outevflag = {_fortran_integer(0)}", ""]
    return lines


class RunOutputFiles:
    """Writes the data files of one run as the calculation reports them.

    ``write`` takes lines for the run log.  ``kpoint_count`` is the number
    of k-points of a k-point run (zero otherwise).  Call :meth:`start`
    before the first SCF loop, :meth:`iteration` and :meth:`scf_finished`
    as the reporter sees iterations and loops (:class:`ReporterWithFiles`
    does), :meth:`iteration_fields` after every iteration where
    ``Save_Intermediate_Charge_Density`` asks for it, and :meth:`finish`
    with the last result.
    """

    def __init__(
        self,
        settings: OutputFileSettings,
        directory: str | Path,
        *,
        write: Callable[[str], None],
        kpoint_count: int = 0,
        wavefunction_limit_bytes: int = DEFAULT_WAVEFUNCTION_LIMIT_BYTES,
    ) -> None:
        self.settings = settings
        self.directory = Path(directory)
        self.write = write
        self.kpoint_count = int(kpoint_count)
        self.limit = int(wavefunction_limit_bytes)
        self.selection: WavefunctionSelection = settings.wavefunctions
        self.move = 0
        self._eigen: EigenDatFile | None = None
        self._notes: list[str] = []
        self._unsupported: str | None = None

    # ----- files ---------------------------------------------------------

    @property
    def parsec_dat_path(self) -> Path:
        return self.directory / "parsec.dat"

    @property
    def eigen_dat_path(self) -> Path:
        return self.directory / "eigen.dat"

    def cube_paths(self) -> dict[str, Path]:
        return {
            name: self.directory / f"{CUBE_FILE_STEMS[name]}.cube"
            for name in dict.fromkeys(self.settings.grid_data)
        }

    def paths(self) -> list[Path]:
        """Every file this run may write (to keep them off its inputs)."""

        paths = []
        if self.settings.writes_parsec_dat:
            paths.append(self.parsec_dat_path)
        if self.settings.writes_eigen_dat:
            paths.append(self.eigen_dat_path)
        paths.extend(self.cube_paths().values())
        return paths

    # ----- plan ----------------------------------------------------------

    def plan_wavefunctions(
        self,
        grid_size: int,
        number_of_states: int,
        spin_count: int,
        *,
        complex_values: bool = False,
    ) -> bool:
        """Settle the wavefunctions of ``parsec.dat``; ``True`` if it holds any.

        The default selection (the occupied states) gives way to none where
        all states of the run would take more than the limit, a bound the
        run knows before its SCF loop.
        """

        selection = self.settings.wavefunctions
        if not self.settings.writes_parsec_dat:
            self.selection = replace(selection, mode="list", indices=())
            return False
        if selection.mode == "occupied" and not selection.explicit:
            size = wavefunction_bytes(grid_size, number_of_states, spin_count, complex_values)
            if size > self.limit:
                self.selection = WavefunctionSelection(mode="list", indices=(), explicit=False)
                self._notes.append(
                    "parsec.dat holds no wavefunctions: the occupied states of this "
                    f"full grid could take up to {size / 2**20:.0f} MiB (limit "
                    f"{self.limit / 2**20:.0f} MiB); give Save_Wfn_Bands or "
                    "Output_All_States to write them"
                )
                return False
        if selection.mode == "list":
            if selection.indices and max(selection.indices) > number_of_states:
                raise ValueError(
                    f"Save_Wfn_Bands asks for state {max(selection.indices)}, but "
                    f"States_Num is {number_of_states}"
                )
            self.selection = selection
            return bool(selection.indices)
        self.selection = selection
        return True

    @property
    def selection_has_states(self) -> bool:
        """Whether the planned ``parsec.dat`` holds wavefunctions."""

        if not self.settings.writes_parsec_dat or self._unsupported is not None:
            return False
        return self.selection.mode != "list" or bool(self.selection.indices)

    def refuse_parsec_dat(self, reason: str) -> None:
        """This calculation writes no ``parsec.dat``; the log says why."""

        self._unsupported = reason

    # ----- events --------------------------------------------------------

    def start(self) -> None:
        if self.settings.writes_eigen_dat:
            self._eigen = EigenDatFile(self.eigen_dat_path, kpoints=self.kpoint_count > 0)

    @property
    def wants_iteration_fields(self) -> bool:
        """Whether the SCF loop is to hand every iteration to :meth:`iteration_fields`.

        ``Save_Intermediate_Charge_Density`` (``parsec.dat`` after every
        iteration), and ``Save_Intermediate_Eigenvalues`` of a k-point run,
        whose iteration records hold the first k-point only.
        """

        if self.settings.writes_parsec_dat and self.settings.intermediate_density:
            return True
        return bool(
            self.settings.writes_eigen_dat
            and self.settings.intermediate_eigenvalues
            and self.kpoint_count > 0
        )

    def iteration(self, step) -> None:
        if isinstance(step, ExchangeLoopIteration):
            # An outer iteration of a hybrid's Fock loop: no levels of its own.
            # (A spin-polarized record has eigenvalues_up/_down and no
            # eigenvalues: it is not one.)
            return
        # A k-point run writes its levels from iteration_fields (all k-points).
        if (
            self._eigen is not None
            and self.settings.intermediate_eigenvalues
            and self.kpoint_count == 0
        ):
            self._eigen.append(
                eigen_dat_block(
                    iteration_eigen_channels(step),
                    move=self.move,
                    all_states=self.settings.all_states,
                    kpoint_count=0,
                )
            )

    def scf_finished(self, result) -> None:
        has_levels = (
            getattr(result, "eigenvalues", None) is not None
            or getattr(result, "eigenvalues_up", None) is not None
        )
        if self._eigen is not None and has_levels and not self.settings.intermediate_eigenvalues:
            channels, kpoint_count = result_eigen_channels(result)
            self._eigen.append(
                eigen_dat_block(
                    channels,
                    move=self.move,
                    all_states=self.settings.all_states,
                    kpoint_count=kpoint_count,
                )
            )
        self.move += 1

    def iteration_fields(self, state, grid, problem=None) -> None:
        """``parsec.dat`` (and the k-point levels of ``eigen.dat``) of one
        finished SCF iteration, in the order of ``parsec.F90`` l.1381-1402.

        ``state`` is a :class:`parsec_python.SCF.single_point.SCFIterationFields`
        or :class:`parsec_python.SCF.single_point.SpinorIterationFields`.
        """

        if self.settings.writes_parsec_dat and self.settings.intermediate_density:
            try:
                self._write_parsec_dat(state, grid, problem, report=False)
            except ValueError as error:
                # A loop that keeps no fields of the file (an accelerated
                # caller's device spinors): said once at the end of the run.
                self._notes.append(f"parsec.dat is not written after the SCF iterations: {error}")
        if (
            self._eigen is not None
            and self.settings.intermediate_eigenvalues
            and self.kpoint_count > 0
        ):
            channels, kpoint_count = result_eigen_channels(state)
            self._eigen.append(
                eigen_dat_block(
                    channels,
                    move=self.move,
                    all_states=self.settings.all_states,
                    kpoint_count=kpoint_count,
                )
            )

    def finish(self, result, grid, problem=None) -> list[Path]:
        """Write ``parsec.dat`` and the cube files of the finished run."""

        written: list[Path] = []
        for note in dict.fromkeys(self._notes):
            self.write(f" NOTE: {note}")
        if self.settings.writes_parsec_dat:
            reason = self._unsupported
            if reason is None:
                try:
                    written.append(self._write_parsec_dat(result, grid, problem, report=True))
                except ValueError as error:
                    # A result without the fields of the file (a stand-in
                    # for the SCF, or a driver that keeps no potential).
                    reason = str(error)
            if reason is not None:
                self.write(f" WARNING: parsec.dat is not written: {reason}")
        if self._eigen is not None:
            self.write(f" eigen.dat: {self._eigen.path}")
            written.append(self._eigen.path)
        if self.settings.grid_data:
            fields = grid_data_fields(result)
            for name, path in self.cube_paths().items():
                stem = CUBE_FILE_STEMS[name]
                if name not in fields:
                    self.write(
                        f" WARNING: {stem}.cube is not written: this calculation "
                        "keeps no such field"
                    )
                    continue
                self.write(f" Exporting {stem} grid data to file")
                write_parsec_cube(path, stem, fields[name], grid)
                self.write(f" Finished exporting {stem}")
                written.append(path)
        return written

    def _write_parsec_dat(self, result, grid, problem, *, report: bool) -> Path:
        mp_grid = getattr(problem, "monkhorst_pack_dimensions", None) if problem is not None else None
        data, notes = parsec_dat_from_result(
            result,
            grid,
            self.selection,
            periodic_cell=getattr(problem, "periodic_cell", None) if problem is not None else None,
            monkhorst_pack_grid=mp_grid,
            monkhorst_pack_shift=(
                getattr(problem, "monkhorst_pack_shift", None) if problem is not None else None
            ),
        )
        path = write_parsec_dat(self.parsec_dat_path, data)
        if report:
            for note in notes:
                self.write(f" NOTE: {note}")
            saved = sum(int(block.state_numbers.size) for spin in data.wavefunctions for block in spin)
            self.write(
                f" parsec.dat: {path} (potential and density of {data.spin_count} "
                f"spin channel(s) on {data.ndim} points, {saved} wavefunction(s))"
            )
        return path


class ReporterWithFiles:
    """A run reporter that also hands iterations and SCF loops to the files.

    Every other attribute is that of the wrapped reporter, so the ionic
    loops, which call ``iteration`` and ``finish`` of the reporter they are
    given, write ``eigen.dat`` blocks with the movement number of PARSEC.
    """

    def __init__(self, reporter, files: RunOutputFiles) -> None:
        object.__setattr__(self, "_reporter", reporter)
        object.__setattr__(self, "_files", files)

    def __getattr__(self, name):
        return getattr(self._reporter, name)

    def __setattr__(self, name, value) -> None:
        setattr(self._reporter, name, value)

    def iteration(self, step) -> None:
        self._reporter.iteration(step)
        self._files.iteration(step)

    def finish(self, result, *args, **keywords) -> None:
        self._reporter.finish(result, *args, **keywords)
        self._files.scf_finished(result)

    def iteration_spin_polarized(self, step) -> None:
        self._reporter.iteration_spin_polarized(step)
        self._files.iteration(step)

    def finish_spin_polarized(self, result, *args, **keywords) -> None:
        self._reporter.finish_spin_polarized(result, *args, **keywords)
        self._files.scf_finished(result)

    def iteration_self_consistent_soc(self, step) -> None:
        self._reporter.iteration_self_consistent_soc(step)
        self._files.iteration(step)

    def finish_self_consistent_soc(self, result, *args, **keywords) -> None:
        self._reporter.finish_self_consistent_soc(result, *args, **keywords)
        self._files.scf_finished(result)


__all__ = [
    "DEFAULT_WAVEFUNCTION_LIMIT_BYTES",
    "ReporterWithFiles",
    "RunOutputFiles",
    "output_header_lines",
]
