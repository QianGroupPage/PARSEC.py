"""Text files of a structural relaxation, in PARSEC's layouts and in XYZ.

``atom.cor``
    the structure of every step with its energy and largest force
    (``initial.f90``, ``domove.f90``, end of ``parsec.F90``);
``relax_restart.dat``
    coordinates of all atoms and, for BFGS, the complete optimizer state
    (``domove.f90``);
``manual.dat``
    the structures of a ``manual`` run (``domove.f90`` reads it);
extended XYZ
    one frame per step with energy and forces, for other programs.

Every function here formats or parses text; none opens a file.  Lengths are
in bohr and energies in Rydberg unless a name says otherwise.

Fortran writes ``relax_restart.dat`` with list-directed output.  The numbers
written here use the same layout with 17 significant digits, so that either
code reads the file of the other without loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .lbfgsb import LBFGSBState, format_restart_block, list_directed, parse_restart_block


#: ``angs`` of ``const.f90``.
BOHR_TO_ANGSTROM = 0.529177
#: ``rydberg`` of ``const.f90``.
RYDBERG_TO_EV = 13.6058


# --------------------------------------------------------------- atom.cor


def _species_blocks(symbols: Sequence[str]) -> list[tuple[str, int]]:
    """Consecutive runs of equal species labels, as PARSEC's atom types."""

    blocks: list[tuple[str, int]] = []
    for symbol in symbols:
        if blocks and blocks[-1][0] == symbol:
            blocks[-1] = (symbol, blocks[-1][1] + 1)
        else:
            blocks.append((symbol, 1))
    return blocks


def format_atom_cor_structure(
    step: int, symbols: Sequence[str], positions: np.ndarray
) -> str:
    """``Step #`` block of ``atom.cor`` (``initial.f90``/``domove.f90``).

    One line per atom type with its name and atom count, then the
    coordinates in bohr with six decimals.
    """

    positions = np.asarray(positions, dtype=np.float64)
    lines = [f"Step # {step:3d}"]
    cursor = 0
    for symbol, count in _species_blocks(symbols):
        # write(66,*) name(ity),'    ',natmi(ity) with a two-character name.
        lines.append(f" {symbol:<2s}    {count:12d}")
        for row in positions[cursor : cursor + count]:
            lines.append("".join(f"   {value:10.6f}" for value in row))
        cursor += count
    return "\n".join(lines) + "\n"


def format_atom_cor_energy(energy_per_atom_ev: float, maximum_force: float) -> str:
    """Energy line ``domove.f90`` writes for the structure just computed."""

    return (
        f"Total Energy = {energy_per_atom_ev:12.5f} [eV/at], "
        f"Max force ={maximum_force:9.5f} [Ry/au]\n\n"
    )


def format_atom_cor_final(positions: np.ndarray) -> str:
    """Closing block of ``atom.cor`` (end of ``parsec.F90``)."""

    lines = ["", "  Final coordinates for next run "]
    for row in np.asarray(positions, dtype=np.float64):
        lines.append("".join(f"  {value:11.6f}" for value in row))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------- manual.dat


def parse_manual_structures(
    text: str, movable_atoms: int, structures: int
) -> list[np.ndarray]:
    """Read the structures of a ``manual`` run as ``domove.f90`` reads them.

    Each structure is ``movable_atoms`` records of three coordinates in bohr.
    Between two structures the Fortran skips exactly one record, whatever it
    holds.  A list-directed read skips blank records, so blank lines inside
    a structure are harmless.  ``structures`` is ``Movement_Num``.
    """

    records = text.splitlines()
    cursor = 0
    result: list[np.ndarray] = []
    for index in range(structures):
        rows = []
        for _ in range(movable_atoms):
            values: list[float] = []
            while len(values) < 3:
                if cursor >= len(records):
                    raise ValueError(
                        f"manual.dat ends inside structure {index + 1}"
                    )
                values.extend(_numbers(records[cursor]))
                cursor += 1
            rows.append(values[:3])
        result.append(np.asarray(rows, dtype=np.float64))
        if index < structures - 1:
            cursor += 1
    return result


# ------------------------------------------------------ relax_restart.dat


def _numbers(record: str) -> list[float]:
    """Numbers of one list-directed record (blanks or commas between them)."""

    values = []
    for token in record.replace(",", " ").split():
        if token.startswith("!") or token.startswith("#"):
            break
        values.append(float(token.replace("D", "E").replace("d", "e")))
    return values


def _coordinates(record: str) -> list[float] | None:
    """The three coordinates a record starts with, or ``None`` without them."""

    try:
        values = _numbers(record)
    except ValueError:
        return None
    return values[:3] if len(values) >= 3 else None


@dataclass(frozen=True)
class RelaxRestart:
    """Content of ``relax_restart.dat``.

    ``positions`` are the coordinates of all atoms the next SCF calculation
    is to use, ``iteration`` is ``move%num`` at the time of writing and
    ``movable_atoms`` the number of atoms that move.  A BFGS run adds the
    state of the optimizer (``lbfgsb``, see
    :class:`parsec_python.Relaxation.lbfgsb.LBFGSBState`), the bounds of its
    variables and ``bound_step``, the ``move%stepmax`` written beside the
    number of corrections (``-1`` for an unrestricted run).
    """

    positions: np.ndarray
    iteration: int
    movable_atoms: int
    lbfgsb: LBFGSBState | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    bound_step: float | None = None


def format_relax_restart(restart: RelaxRestart) -> str:
    """Write ``relax_restart.dat`` as ``domove.f90`` does."""

    lines = ["  Atomic coordinates follow (all atoms)"]
    for row in np.asarray(restart.positions, dtype=np.float64):
        lines.append(list_directed(*(float(value) for value in row)))
    lines.append(" Iteration number, number of movable atoms")
    lines.append(list_directed(int(restart.iteration), int(restart.movable_atoms)))
    text = "\n".join(lines) + "\n"
    if restart.lbfgsb is not None:
        text += "  LBFGS work variables follow\n"
        text += format_restart_block(
            restart.lbfgsb,
            restart.lower,
            restart.upper,
            -1.0 if restart.bound_step is None else restart.bound_step,
        )
    return text


def parse_relax_restart(text: str, atoms: int) -> RelaxRestart:
    """Read ``relax_restart.dat`` written by PARSEC or by this package.

    ``atoms`` is the number of atoms of the calculation; the file itself
    does not hold it.  The BFGS block is read when present.
    """

    records = text.splitlines()
    if len(records) < atoms + 3:
        raise ValueError("relax_restart.dat is too short for this structure")
    # The file does not say for how many atoms it was written.  It fits when
    # the records of the atoms hold coordinates and the one after them does
    # not: that is the header of the counters.
    rows = [_coordinates(records[1 + index]) for index in range(atoms)]
    if None in rows or _coordinates(records[atoms + 1]) is not None:
        raise ValueError(
            "relax_restart.dat does not hold three coordinates for each of the "
            f"{atoms} atoms of this structure"
        )
    positions = np.asarray(rows, dtype=np.float64)
    counters = _numbers(records[atoms + 2])
    if len(counters) < 2:
        raise ValueError("relax_restart.dat lacks the iteration and atom counters")
    iteration, movable = int(counters[0]), int(counters[1])
    cursor = atoms + 3
    if cursor >= len(records) or "LBFGS" not in records[cursor].upper():
        return RelaxRestart(positions, iteration, movable)
    state, lower, upper, bound_step = parse_restart_block(
        "\n".join(records[cursor + 1 :]), 3 * movable
    )
    return RelaxRestart(
        positions,
        iteration,
        movable,
        lbfgsb=state,
        lower=lower,
        upper=upper,
        bound_step=bound_step,
    )


# ------------------------------------------------------------------- XYZ


def format_xyz_frame(
    symbols: Sequence[str],
    positions: np.ndarray,
    *,
    step: int,
    energy: float | None = None,
    forces: np.ndarray | None = None,
    elements: Sequence[str] | None = None,
) -> str:
    """One frame of an extended XYZ trajectory.

    Positions are written in angstrom.  With ``energy`` (Ry) the comment line
    carries it in eV and in Ry; with ``forces`` (Ry/bohr) three more columns
    hold them in eV/angstrom.  ``elements`` replaces species labels that are
    not element symbols.
    """

    positions = np.asarray(positions, dtype=np.float64) * BOHR_TO_ANGSTROM
    names = list(symbols if elements is None else elements)
    properties = "species:S:1:pos:R:3"
    if forces is not None:
        properties += ":forces:R:3"
        scaled = np.asarray(forces, dtype=np.float64) * (RYDBERG_TO_EV / BOHR_TO_ANGSTROM)
    comment = [f"step={step}"]
    if energy is not None:
        comment.append(f"energy={energy * RYDBERG_TO_EV:.10f}")
        comment.append(f"energy_Ry={energy:.10f}")
    comment.append(f'Properties={properties}')
    comment.append('pbc="F F F"')
    lines = [str(len(names)), " ".join(comment)]
    for index, name in enumerate(names):
        row = f"{name:<3s}" + "".join(f" {value:18.10f}" for value in positions[index])
        if forces is not None:
            row += "".join(f" {value:18.10f}" for value in scaled[index])
        lines.append(row)
    return "\n".join(lines) + "\n"


def parse_xyz_frames(text: str) -> list[tuple[list[str], np.ndarray]]:
    """Read every frame of an XYZ file: labels and positions in bohr."""

    records = text.splitlines()
    frames: list[tuple[list[str], np.ndarray]] = []
    cursor = 0
    while cursor < len(records):
        if not records[cursor].strip():
            cursor += 1
            continue
        try:
            count = int(records[cursor].split()[0])
        except ValueError as error:
            raise ValueError(
                f"XYZ line {cursor + 1} should hold the number of atoms"
            ) from error
        body = records[cursor + 2 : cursor + 2 + count]
        if len(body) != count:
            raise ValueError("XYZ file ends inside a frame")
        labels, rows = [], []
        for record in body:
            fields = record.split()
            if len(fields) < 4:
                raise ValueError(f"XYZ atom line is too short: {record!r}")
            labels.append(fields[0])
            rows.append([float(value) for value in fields[1:4]])
        frames.append((labels, np.asarray(rows, dtype=np.float64) / BOHR_TO_ANGSTROM))
        cursor += 2 + count
    if not frames:
        raise ValueError("the XYZ file holds no structure")
    return frames


def join_lines(lines: Iterable[str]) -> str:
    """Join report lines with a final newline."""

    return "\n".join(lines) + "\n"


__all__ = [
    "BOHR_TO_ANGSTROM",
    "RYDBERG_TO_EV",
    "RelaxRestart",
    "format_atom_cor_energy",
    "format_atom_cor_final",
    "format_atom_cor_structure",
    "format_relax_restart",
    "format_xyz_frame",
    "join_lines",
    "parse_manual_structures",
    "parse_relax_restart",
    "parse_xyz_frames",
]
