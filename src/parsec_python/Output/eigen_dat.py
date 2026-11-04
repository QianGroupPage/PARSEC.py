"""PARSEC's ``eigen.dat``: one line per state, one block per SCF loop.

``initial.f90`` l.532-541 opens the file and writes its header;
``eigensave.f90`` appends a block at the end of every SCF loop (or after
every SCF iteration with ``Save_Intermediate_Eigenvalues``).  A block lists,
for each spin channel and k-point, the states ``1..nstate`` (with
``Output_All_States``) or ``1..ifmax`` (the highest state with a non-zero
occupation), in ascending order of energy::

    (i5,2x,f18.10,2x,f12.4,2x,f12.4,2x,i2,2x,i5,2x,a2)          no k-points
    (i5,2x,f18.10,2x,f12.4,2x,f12.4,2x,i2,2x,i5,2x,i5,2x,a2)    k-points

state, eigenvalue [Ry], eigenvalue [eV] (``rydberg = 13.6058``),
occupation, representation, k-point, ``move%num`` and the spin label
(blank, ``up`` or ``dn``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .parsec_dat import highest_occupied

#: ``rydberg`` of ``const.f90``.
RYDBERG_EV = 13.6058

_HEADER = " State     Eigenvalue [Ry]    Eigenvalue [eV]"
_HEADER_GAMMA = "    Occupation   Repr.   Movement   spin"
_HEADER_K = "    Occupation   Repr.   k-point  Movement   spin"


def eigen_dat_header(kpoints: bool) -> str:
    """The header line as gfortran's list-directed ``write(61,*)`` prints it."""

    return " " + _HEADER + (_HEADER_K if kpoints else _HEADER_GAMMA)


def _fortran_integer(value: int, width: int) -> str:
    text = f"{int(value):d}"
    return "*" * width if len(text) > width else text.rjust(width)


def _fortran_fixed(value: float, width: int, decimals: int) -> str:
    text = f"{float(value):{width}.{decimals}f}"
    return "*" * width if len(text) > width else text


def eigen_dat_line(
    state: int,
    energy_ry: float,
    occupation: float,
    representation: int,
    move: int,
    spin_label: str = "  ",
    kpoint: int | None = None,
) -> str:
    """One state in format 11 (``kpoint`` ``None``) or 12 of ``eigensave.f90``."""

    parts = [
        _fortran_integer(state, 5),
        _fortran_fixed(energy_ry, 18, 10),
        _fortran_fixed(energy_ry * RYDBERG_EV, 12, 4),
        _fortran_fixed(occupation, 12, 4),
        _fortran_integer(representation, 2),
    ]
    if kpoint is not None:
        parts.append(_fortran_integer(kpoint, 5))
    parts.append(_fortran_integer(move, 5))
    parts.append(f"{spin_label:2.2s}")
    return "  ".join(parts)


@dataclass(frozen=True)
class EigenChannel:
    """The levels of one spin channel: k-major pooled arrays."""

    eigenvalues: np.ndarray
    occupations: np.ndarray
    representations: np.ndarray


def eigen_dat_block(
    channels: Sequence[EigenChannel],
    *,
    move: int,
    all_states: bool,
    kpoint_count: int = 0,
) -> list[str]:
    """The lines ``eigensave.f90`` appends for one set of levels.

    ``kpoint_count`` is ``elec_st%nkpt``: zero for a calculation without
    k-points (format 11), the number of k-points otherwise (format 12).
    Channels are up then down where there are two (labels ``up``/``dn``);
    states are sorted by energy within each k-point.
    """

    lines: list[str] = []
    blocks = max(int(kpoint_count), 1)
    for index, channel in enumerate(channels):
        label = "  " if len(channels) == 1 else ("up", "dn")[index]
        eigenvalues = np.asarray(channel.eigenvalues, dtype=float)
        occupations = np.asarray(channel.occupations, dtype=float)
        representations = np.asarray(channel.representations, dtype=int)
        if representations.size != eigenvalues.size:
            representations = np.ones(eigenvalues.size, dtype=int)
        per_k = eigenvalues.size // blocks
        sorted_blocks = []
        for k in range(blocks):
            part = slice(k * per_k, (k + 1) * per_k)
            order = np.argsort(eigenvalues[part], kind="stable")
            sorted_blocks.append(
                (eigenvalues[part][order], occupations[part][order], representations[part][order])
            )
        count = (
            per_k
            if all_states
            else highest_occupied([occupation for _e, occupation, _r in sorted_blocks])
        )
        for k, (energies, occupation, representation) in enumerate(sorted_blocks, start=1):
            for state in range(count):
                lines.append(
                    eigen_dat_line(
                        state + 1,
                        energies[state],
                        occupation[state],
                        representation[state],
                        move,
                        label,
                        kpoint=k if kpoint_count else None,
                    )
                )
    return lines


def result_eigen_channels(result) -> tuple[tuple[EigenChannel, ...], int]:
    """The channels of a Python SCF result and its number of k-points.

    Spin-polarized results list the up channel first.  Spin-orbit and
    non-collinear results have one channel of spinor states, as PARSEC's
    ``nspin/mxwd = 1``.
    """

    k_points = getattr(result, "k_points", None)
    kpoint_count = 0 if k_points is None else int(np.asarray(k_points).shape[0])
    channels = getattr(result, "spin", None)
    if channels is not None:
        return (
            tuple(
                EigenChannel(c.eigenvalues, c.occupations, getattr(c, "representations", ()))
                for c in channels
            ),
            kpoint_count,
        )
    if hasattr(result, "eigenvalues_up"):
        return (
            (
                EigenChannel(result.eigenvalues_up, result.occupations_up, ()),
                EigenChannel(result.eigenvalues_down, result.occupations_down, ()),
            ),
            kpoint_count,
        )
    return (
        (
            EigenChannel(
                result.eigenvalues,
                result.occupations,
                getattr(result, "representations", ()),
            ),
        ),
        kpoint_count,
    )


def iteration_eigen_channels(step) -> tuple[EigenChannel, ...]:
    """The levels of one SCF iteration record (first k-point only).

    A :class:`parsec_python.models.SCFIteration` of a spin-polarized loop
    holds the up states followed by the down states.
    """

    if hasattr(step, "eigenvalues_up"):
        return (
            EigenChannel(step.eigenvalues_up, step.occupations_up, ()),
            EigenChannel(step.eigenvalues_down, step.occupations_down, ()),
        )
    eigenvalues = np.asarray(step.eigenvalues, dtype=float)
    occupations = np.asarray(step.occupations, dtype=float)
    representations = np.asarray(getattr(step, "representations", ()), dtype=int)
    if getattr(step, "spin_polarized", False):
        half = eigenvalues.size // 2
        rep = representations if representations.size == eigenvalues.size else np.ones(eigenvalues.size, dtype=int)
        return (
            EigenChannel(eigenvalues[:half], occupations[:half], rep[:half]),
            EigenChannel(eigenvalues[half:], occupations[half:], rep[half:]),
        )
    return (EigenChannel(eigenvalues, occupations, representations),)


class EigenDatFile:
    """``eigen.dat`` of one run: the header at the start, blocks appended."""

    def __init__(self, path: str | Path, *, kpoints: bool) -> None:
        self.path = Path(path)
        self.kpoints = bool(kpoints)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="ascii", newline="\n") as stream:
            stream.write(eigen_dat_header(self.kpoints) + "\n")

    def append(self, lines: Sequence[str]) -> None:
        if not lines:
            return
        with self.path.open("a", encoding="ascii", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")


__all__ = [
    "EigenChannel",
    "EigenDatFile",
    "RYDBERG_EV",
    "eigen_dat_block",
    "eigen_dat_header",
    "eigen_dat_line",
    "iteration_eigen_channels",
    "result_eigen_channels",
]
