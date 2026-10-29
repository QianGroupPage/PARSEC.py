"""PARSEC-shaped text for atomic forces and ionic movements.

The tables follow ``forceion.f90``, ``forcloc.f90`` and ``forcnloc.f90``:
positions in bohr and forces in Ry/bohr with six decimals.  ``Output_Level``
selects how much of the breakdown is printed, as ``ipr`` does in PARSEC; the
final table of total forces, the net force and the torque are always there.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


_TABLE_HEADER = (
    "-atom-  ----x----   ----y----   ----z-----   ----Fx----  ----Fy----   ----Fz---",
    "                       [bohr]                             [Ry/bohr]            ",
    "",
)


def _table(positions: np.ndarray, forces: np.ndarray) -> list[str]:
    lines = list(_TABLE_HEADER)
    for index, (position, force) in enumerate(zip(positions, forces), start=1):
        lines.append(
            f"{index:4d}  "
            + "".join(f"{value:11.6f} " for value in position)
            + "  "
            + "".join(f"{value:11.6f} " for value in force)
        )
    return lines


def force_report_lines(forces, output_level: int = 1, *, periodic: bool = False) -> list[str]:
    """Lines of the force section for one structure.

    ``forces`` is a :class:`parsec_python.Forces.ForceResult`.  For a periodic cell the report is Fortran's: the local
    forces say that the Ewald forces are included (there is no separate ion-ion table), and no torque is reported.
    """

    positions = forces.positions
    lines: list[str] = []
    if output_level >= 1 and not periodic:
        lines += [
            "",
            " Forces from ion-ion interaction:",
            " ================================",
            *_table(positions, forces.ion_ion),
            "",
        ]
    if output_level >= 2:
        lines += [
            " Format is: atom #, followed by x,y,z components of",
            " ion, self-consistency, and core-correction forces",
        ]
        for index in range(positions.shape[0]):
            ionic = forces.local.ionic[index]
            correction = forces.local.self_consistency[index]
            core = forces.local.core_correction[index]
            lines.append(
                f"at{index + 1:4d} "
                + "".join(f"{value:11.6f} " for value in ionic)
                + " "
                + "".join(f"{value:11.6f} " for value in correction)
            )
            lines.append("       " + "".join(f"{value:11.6f} " for value in core))
        lines.append("")
    if output_level >= 1:
        lines += [
            " Coordinates and local force components",
            " ======================================",
            *([" (Ewald forces included)"] if periodic else []),
            *_table(positions, forces.local_with_ion_ion),
            "",
            "",
            " Reporting non-local force components",
            " ====================================",
            *_table(positions, forces.nonlocal_pseudopotential),
        ]
    lines += ["", " Coordinates, forces, torque", " ============================", ""]
    if forces.symmetry_operations > 1:
        if output_level == 1:
            lines += [
                " coordinates and tot. forces (no symmetrization)",
                *_table(positions, forces.unsymmetrized),
            ]
        lines.append("  symmetrizing forces according to symmetry operations ")
    if output_level >= 1:
        lines += [
            " coordinates and symmetrized tot. forces",
            *_table(positions, forces.symmetrized),
            "",
        ]
    lines.append(
        "             tot.  (net) force [Ry/bohr]:   "
        + "".join(f"{value:11.6f} " for value in forces.net_force)
    )
    lines += [
        "",
        " coordinates and tot. forces (after setting net force to zero)",
        *_table(positions, forces.total),
        "",
    ]
    if not periodic:
        lines += [
            " tot. torque [Ry/radian] :  " + "".join(f"{value:11.6f} " for value in forces.torque),
            "",
        ]
    return lines


def movement_report_lines(
    step: int,
    seconds: float,
    messages: Sequence[str] = (),
    *,
    called_mover: str | None = "domove",
) -> list[str]:
    """Lines ``parsec.F90`` prints around one ionic movement."""

    lines = [
        "",
        f"Time for movement {step:4d} is [sec]: {seconds:10.2f}",
        "",
        " " + "=" * 65,
        "",
    ]
    if called_mover is not None:
        lines += ["", f"  calling {called_mover}", ""]
        lines += [f" {message}" for message in messages]
        lines += ["", f"  finished {called_mover}", ""]
    return lines


def relaxation_summary_lines(
    *,
    movements_exhausted: bool,
    minimum_force_found: bool,
    manual: bool,
    scf_failed: bool,
) -> list[str]:
    """Closing statements of a relaxation (end of ``parsec.F90``)."""

    lines: list[str] = []
    if movements_exhausted:
        lines += [" NOTICE: Maximum number of movements used.", ""]
    if not manual:
        lines += [
            "  Minimum force is satisfied "
            if minimum_force_found
            else "  WARNING: Minimum force is NOT satisfied!!",
            "",
        ]
    if scf_failed:
        lines += ["  WARNING: SCF convergence not achieved", ""]
    return lines


__all__ = [
    "force_report_lines",
    "movement_report_lines",
    "relaxation_summary_lines",
]
