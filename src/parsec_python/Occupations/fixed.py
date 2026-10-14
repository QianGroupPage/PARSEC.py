"""Occupations fixed by the user: PARSEC's ``occup.in``.

A negative ``Fermi_Temp`` tells PARSEC to search no Fermi level.
``initial.f90`` reads the occupation of every state from the file
``occup.in`` of the run directory, and ``flevel.f90`` assigns them at every
SCF step to the computed states in ascending order of energy: the state with
the n-th lowest eigenvalue, whatever its symmetry representation, receives
the n-th occupation.

The file holds one section per spin channel::

    title line                  skipped
    ifmax                       states 1..ifmax are fully occupied
    m                           number of occupations set by hand
    j  occupation               m records: state index and its occupation

Everything is read with list-directed statements, so text after the numbers
of a record is a comment.  ``initial.f90`` reads ``nspin`` sections, hence
only the first one in a calculation without spin polarization: whatever
follows the first section is not read.  There an occupation of one stands
for the two electrons of a doubly occupied orbital, the convention of
:func:`fermi_occupations`.  A spin-polarized calculation reads two
sections, the up channel first, and an occupation of one stands for one
electron (:mod:`parsec_python.Occupations.spin`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from .fermi_dirac import OccupationResult


#: ``small`` of ``initial.f90``: largest accepted difference between the
#: electrons the fixed occupations hold and those of the system.
ELECTRON_COUNT_TOLERANCE = 1.0e-5

_INTEGER_RE = re.compile(r"[-+]?\d+")
_REAL_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?")


@dataclass(frozen=True)
class OccupationFile:
    """Occupations of the first spin section of an ``occup.in``.

    ``occupations`` has one value per state.  ``unread_lines`` counts the
    nonblank lines after that section: further spin sections, which a
    calculation without spin polarization does not read.
    """

    occupations: tuple[float, ...]
    unread_lines: int
    source: Path


@dataclass(frozen=True)
class SpinOccupationFile:
    """Occupations of the two spin sections of an ``occup.in``, up first.

    An occupation of one stands for one electron there.  ``unread_lines``
    counts the nonblank lines after the second section.
    """

    occupations_up: tuple[float, ...]
    occupations_down: tuple[float, ...]
    unread_lines: int
    source: Path


def read_occupation_file(path: str | Path, number_of_states: int) -> OccupationFile:
    """Read the occupations a spin-unpolarized PARSEC run takes from ``path``.

    ``number_of_states`` is ``States_Num``: states beyond ``ifmax`` are empty
    unless set by hand.  An index outside ``1..number_of_states`` writes
    beyond the occupation array in ``initial.f90`` and is refused here.
    """

    sections, unread, source = _read_sections(path, number_of_states, 1)
    return OccupationFile(sections[0], unread, source)


def read_spin_occupation_file(
    path: str | Path, number_of_states: int
) -> SpinOccupationFile:
    """Read the occupations a spin-polarized PARSEC run takes from ``path``.

    ``initial.f90`` reads one section per spin channel, the up channel
    first, each with its title line.  Every channel has ``number_of_states``
    states.
    """

    sections, unread, source = _read_sections(path, number_of_states, 2)
    return SpinOccupationFile(sections[0], sections[1], unread, source)


def _read_sections(
    path: str | Path, number_of_states: int, count: int
) -> tuple[list[tuple[float, ...]], int, Path]:
    """Read the first ``count`` spin sections of an ``occup.in``."""

    source = Path(path)
    states = int(number_of_states)
    if states != number_of_states or states < 1:
        raise ValueError("number_of_states must be a positive integer")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(
            f"a negative Fermi_Temp reads the occupations from {source}, "
            f"which cannot be read: {error}"
        ) from error
    position = 0

    def record(count: int, what: str) -> list[str]:
        # ``read(90,*)`` with ``count`` items: as many records as the list
        # needs, and the rest of the last one is dropped.
        nonlocal position
        found: list[str] = []
        while len(found) < count:
            if position >= len(lines):
                raise ValueError(f"{source} ends before {what}")
            found.extend(lines[position].replace(",", " ").split())
            position += 1
        return found[:count]

    def integer(token: str, what: str) -> int:
        if _INTEGER_RE.fullmatch(token) is None:
            raise ValueError(f"{source}: expected an integer for {what}, found {token!r}")
        return int(token)

    if not lines:
        raise ValueError(f"{source} is empty")
    sections: list[tuple[float, ...]] = []
    for section in range(count):
        # The title line is passed whatever it holds.
        if position >= len(lines):
            raise ValueError(
                f"{source} ends before the section of spin channel "
                f"{section + 1}; a spin-polarized calculation reads one "
                "section per channel"
            )
        position += 1
        highest_full = integer(
            record(1, "the number of fully occupied states")[0],
            "the number of fully occupied states",
        )
        if not 0 <= highest_full <= states:
            raise ValueError(
                f"{source}: {highest_full} fully occupied states do not fit "
                f"States_Num = {states}"
            )
        occupations = [1.0] * highest_full + [0.0] * (states - highest_full)
        overrides = integer(
            record(1, "the number of occupations set by hand")[0],
            "the number of occupations set by hand",
        )
        for _ in range(overrides):
            index_token, value_token = record(2, "an occupation set by hand")
            index = integer(index_token, "a state index")
            if not 1 <= index <= states:
                raise ValueError(
                    f"{source}: state index {index} lies outside 1..{states} "
                    "(States_Num)"
                )
            if _REAL_RE.fullmatch(value_token) is None:
                raise ValueError(
                    f"{source}: expected an occupation for state {index}, "
                    f"found {value_token!r}"
                )
            value = float(value_token.replace("D", "E").replace("d", "e"))
            if not np.isfinite(value):
                raise ValueError(
                    f"{source}: the occupation of state {index} is not finite"
                )
            occupations[index - 1] = value
        sections.append(tuple(occupations))
    unread = sum(1 for line in lines[position:] if line.strip())
    return sections, unread, source


def check_fixed_occupations(
    occupations: Sequence[float], electron_count: float
) -> None:
    """Refuse occupations that do not hold the electrons of the system.

    ``initial.f90`` stops with "fixed occupation inconsistent with stated
    number of electrons" when ``2*sum(occupations)`` differs from the
    electron count by more than ``1e-5``.
    """

    held = 2.0 * float(np.sum(np.asarray(occupations, dtype=float)))
    if abs(held - float(electron_count)) > ELECTRON_COUNT_TOLERANCE:
        raise ValueError(
            f"the fixed occupations hold {held:.6f} electrons, but the system "
            f"has {float(electron_count):.6f}"
        )


def fixed_occupations(
    eigenvalues: np.ndarray,
    electron_count: float,
    occupations: Sequence[float],
) -> OccupationResult:
    """Occupy the states in ascending order of energy with given values.

    ``occupations[n]`` goes to the n-th lowest eigenvalue, as the branch of
    ``flevel.f90`` for a negative Fermi temperature does.  No Fermi level is
    searched, and the result reports ``nan`` for it.
    """

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 1 or eigenvalues.size == 0:
        raise ValueError("eigenvalues must be a nonempty one-dimensional array")
    if np.any(np.diff(eigenvalues) < 0):
        raise ValueError("eigenvalues must be sorted")
    values = np.array(occupations, dtype=float)
    if values.shape != eigenvalues.shape:
        raise ValueError(
            f"{values.size} fixed occupations were given for "
            f"{eigenvalues.size} computed states"
        )
    check_fixed_occupations(values, electron_count)
    return OccupationResult(
        fermi_level=float("nan"),
        occupations=values,
        electron_count=float(2.0 * np.sum(values)),
    )


__all__ = [
    "ELECTRON_COUNT_TOLERANCE",
    "OccupationFile",
    "SpinOccupationFile",
    "check_fixed_occupations",
    "fixed_occupations",
    "read_occupation_file",
    "read_spin_occupation_file",
]
