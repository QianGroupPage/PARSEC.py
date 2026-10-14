"""Shared helpers for tests against Fortran PARSEC with old Martins files.

``data/martins_old_parsec_reference.json`` holds, for a few small clusters,
the input Fortran PARSEC was run with (``Pseudopotential_Format:
martins_old`` for one species and, for SiNa4, occupations from ``occup.in``)
and what its ``parsec.out`` reports: the pseudopotential summary, the sizes
of the nonlocal blocks, the last eigenvalues and occupations, the energy
terms and every force table.  The force tables have the layout of
``data/forces_parsec_reference.json``, so that
:func:`parsec_python.tests.force_reference.assert_matches_fortran` compares
them.

The helpers write such an input with its files into a directory.  One case
uses a new-format file rewritten in the old layout by
:func:`write_martins_old`; Fortran PARSEC ran on the file this function
writes.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from parsec_python.Pseudopotential import (
    ParsecPseudopotential,
    read_parsec_pseudopotential,
)


DATA = Path(__file__).resolve().parent / "data"


def load_cases() -> dict:
    return json.loads((DATA / "martins_old_parsec_reference.json").read_text())["cases"]


def write_martins_old(
    potential: ParsecPseudopotential,
    path: Path,
    *,
    column_order: tuple[int, ...] | None = None,
) -> None:
    """Write the tables of ``potential`` in the old Martins layout.

    The header ``ns lll icore``, the angular momentum + 1 of every column,
    then the potentials in Rydberg (not ``r*V``), the reduced radial
    functions and, with a core correction, the core charge as a volume
    density, each with its radii.  ``column_order`` lists the angular momenta
    in the order of the columns; the default is ascending.
    """

    channels = sorted(potential.channel_potentials)
    if column_order is not None:
        if sorted(column_order) != channels:
            raise ValueError("column_order must list every channel once")
        channels = list(column_order)
    core = potential.has_nonlinear_core_correction
    lines = [
        f"  {potential.radii.size}  {len(channels)}  {1 if core else 0}",
        "  " + "  ".join(str(angular_momentum + 1) for angular_momentum in channels),
    ]
    for table in (potential.channel_potentials, potential.radial_wavefunctions):
        for row, radius in enumerate(potential.radii):
            values = "  ".join(f"{table[l][row]:.16E}" for l in channels)
            lines.append(f"  {radius:.16E}  {values}")
    if core:
        for radius, value in zip(potential.radii, potential.core_density):
            lines.append(f"  {radius:.16E}  {value:.16E}")
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")


def write_case(case: dict, directory: Path, *, parsec_boundary: bool = True) -> Path:
    """Write ``parsec.in`` and every file of ``case`` into ``directory``.

    ``parsec_boundary`` adds ``Hartree_Boundary_Tolerance: off``, with which
    this package uses the Hartree boundary values of PARSEC.
    """

    directory = Path(directory)
    lines = list(case["input"])
    if parsec_boundary:
        lines.append("Hartree_Boundary_Tolerance: off")
    (directory / "parsec.in").write_text("\n".join(lines) + "\n")
    for target, source in case["files"].items():
        if isinstance(source, dict):
            write_martins_old(
                read_parsec_pseudopotential(DATA / source["martins_old_from"]),
                directory / target,
            )
        else:
            shutil.copyfile(DATA / source, directory / target)
    return directory / "parsec.in"


def support_sizes(system) -> list[int]:
    """Grid points inside the projector sphere of every atom of ``system``.

    The counterpart of "Sizes of all non-local blocks" in ``parsec.out``.
    """

    return [
        int(
            np.count_nonzero(
                np.linalg.norm(system.grid.coordinates - atom.position, axis=1)
                <= system.pseudopotentials[atom.symbol].projector_support_radius
            )
        )
        for atom in system.atoms
    ]
