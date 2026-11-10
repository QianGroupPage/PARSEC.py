"""PARSEC-shaped text for atomic forces and ionic movements.

The tables follow ``forceion.f90``, ``forcloc.f90`` and ``forcnloc.f90``:
positions in bohr and forces in Ry/bohr with six decimals.  ``Output_Level``
selects how much of the breakdown is printed, as ``ipr`` does in PARSEC; the
final table of total forces, the net force and the torque are always there.
For a periodic cell (``ForceResult.periodic``) the ion-ion table holds the
Ewald forces (``ewald_sum.f90``), the local table carries the line
"(Ewald forces included)" and there is no torque (``forcnloc.f90`` with
``is_pbc``); ``forpbc.f90`` prints no breakdown of the local terms.
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

    ``forces`` is a :class:`parsec_python.Forces.ForceResult`.  ``periodic`` is the report of a crystal of Andrew
    Grieder's routines, Fortran's: the local forces say that the Ewald forces are included (there is no separate
    ion-ion table), and no torque is reported.  Forces of the Fortran-derived routines say ``forces.periodic``
    themselves: no torque, and no table of the local terms at ``Output_Level`` 2.
    """

    positions = forces.positions
    cell = periodic or bool(getattr(forces, "periodic", False))
    lines: list[str] = []
    if output_level >= 1 and getattr(forces, "point_charge", None) is not None:
        # forcept.f90 prints the ion-ion forces with those of the point
        # charges added, before forceion.f90 prints the same table again.
        lines += [
            "",
            " Forces from ion/point charge interaction:",
            " (combined forces)",
            " =========================================",
            *_table(positions, forces.ion_ion),
            "",
        ]
    if output_level >= 1 and not periodic:
        lines += [
            "",
            " Forces from ion-ion interaction:",
            " ================================",
            *_table(positions, forces.ion_ion),
            "",
        ]
    gaussian = getattr(forces.local, "gaussian", None)
    if output_level >= 2 and gaussian is not None:
        # forcloc_wire.f90 / forcloc_slab.f90 (format 20 there).
        lines += [
            " Format is: atom #, followed by x,y,z components of",
            " ion, self-consistency, core-correction forces, and",
            " Gaussian forces.",
        ]
        for index in range(positions.shape[0]):
            lines.append(
                f"at{index + 1:4d} "
                + "".join(f"{value:11.6f} " for value in forces.local.ionic[index])
                + " "
                + "".join(f"{value:11.6f} " for value in forces.local.self_consistency[index])
            )
            lines.append(
                "       "
                + "".join(f"{value:11.6f} " for value in forces.local.core_correction[index])
                + " "
                + "".join(f"{value:11.6f} " for value in gaussian[index])
            )
        lines.append("")
    elif output_level >= 2 and (periodic or not cell):
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
    spin_orbit = getattr(forces, "spin_orbit", None)
    if output_level >= 1:
        nonlocal_rows = _table(positions, forces.nonlocal_pseudopotential)
        if spin_orbit is not None:
            # forcnloc.f90 prints this between the table header and the rows
            # (the rows are the Kleinman-Bylander term alone).
            nonlocal_rows = [
                *nonlocal_rows[: len(_TABLE_HEADER)],
                " Adding spin-orbit contribution to forces",
                *nonlocal_rows[len(_TABLE_HEADER) :],
            ]
        lines += [
            " Coordinates and local force components",
            " ======================================",
            *([" (Ewald forces included)"] if cell else []),
            *_table(positions, forces.local_with_ion_ion),
            "",
            "",
            " Reporting non-local force components",
            " ====================================",
            *nonlocal_rows,
        ]
        if spin_orbit is not None:
            # Python addition: Fortran adds forcso's term to the total
            # without a table of its own.
            lines += [
                "",
                " Reporting spin-orbit force components",
                " =====================================",
                *_table(positions, spin_orbit),
            ]
        # Python additions: PARSEC's LDA+U forces are zero (forcnloc_u) and
        # efvdw.f90 prints the TS forces to standard output only.
        hubbard = getattr(forces, "hubbard", None)
        if hubbard is not None:
            lines += [
                "",
                " Reporting LDA+U force components",
                " ================================",
                *_table(positions, hubbard),
            ]
        dispersion = getattr(forces, "dispersion", None)
        if dispersion is not None:
            lines += [
                "",
                " Reporting TS-vdW force components",
                " =================================",
                *_table(positions, dispersion),
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
    kept = bool(getattr(forces, "net_force_kept", False))
    lines += [
        "",
        " coordinates and tot. forces (after setting net force to zero)",
        # FDET keeps the net force; forcnloc.F90 prints the table without it all the same.
        *_table(
            positions,
            forces.total
            if not kept
            else np.asarray(forces.symmetrized) - np.asarray(forces.net_force) / max(len(positions), 1),
        ),
    ]
    if periodic:
        # Andrew Grieder's report of a crystal: no torque
        lines += [""]
    elif not cell:
        lines += [
            "",
            " tot. torque [Ry/radian] :  "
            + "".join(f"{value:11.6f} " for value in forces.torque),
            "",
        ]
    if kept:
        # forcnloc.F90:869-875.
        lines.append(" Net forces not removed in FDET.")
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
    stopped_by_scf_failure: bool = False,
) -> list[str]:
    """Closing statements of a relaxation (end of ``parsec.F90``).

    ``stopped_by_scf_failure``: the warning PARSEC 1.5 prints when it ends
    the movement after an SCF loop that did not converge
    (``parsec.F90:2124-2137``).
    """

    lines: list[str] = []
    if stopped_by_scf_failure:
        lines += scf_failure_stop_lines()
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


def scf_failure_stop_lines() -> list[str]:
    """``parsec.F90:2129-2134`` of PARSEC 1.5: relaxation or molecular dynamics ended by an SCF failure."""

    return [
        "",
        " WARNING !!! atom movement is terminated",
        " No point to continue without convergence",
        "",
    ]


__all__ = [
    "force_report_lines",
    "movement_report_lines",
    "relaxation_summary_lines",
    "scf_failure_stop_lines",
]
