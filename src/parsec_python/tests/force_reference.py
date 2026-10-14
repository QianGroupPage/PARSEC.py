"""Shared helpers for tests against forces of Fortran PARSEC.

``data/forces_parsec_reference.json`` holds, for a few small structures of low
symmetry, the input Fortran PARSEC was run with and the force tables of its
``parsec.out``.  The helpers here write such an input with its
pseudopotential files into a directory and compare a
:class:`parsec_python.Forces.ForceResult` with the Fortran numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np


DATA = Path(__file__).resolve().parent / "data"

#: Fortran prints six decimals: two correctly rounded numbers differ by at
#: most 1e-6, and the two SCF loops stop at slightly different densities.
PRINT_TOLERANCE = 1.5e-6


def load_cases() -> dict:
    return json.loads((DATA / "forces_parsec_reference.json").read_text())["cases"]


def write_case(case: dict, directory: Path, *, parsec_boundary: bool = True) -> Path:
    """Write ``parsec.in`` and the pseudopotentials of ``case``.

    ``parsec_boundary`` adds ``Hartree_Boundary_Tolerance: off``, with which
    this package uses the Hartree boundary values of PARSEC.
    """

    directory = Path(directory)
    lines = list(case["input"])
    if parsec_boundary:
        lines.append("Hartree_Boundary_Tolerance: off")
    (directory / "parsec.in").write_text("\n".join(lines) + "\n")
    for symbol, name in case["pseudopotentials"].items():
        shutil.copyfile(DATA / name, directory / f"{symbol}_POTRE.DAT")
    return directory / "parsec.in"


def assert_matches_fortran(test, case: dict, energy: float, forces) -> None:
    """Compare energy and every force table with the Fortran numbers."""

    test.assertAlmostEqual(energy, case["energy"], delta=5.0e-7)
    tables = {
        "positions": forces.positions,
        "ion_ion": forces.ion_ion,
        "ionic": forces.local.ionic,
        "sc": forces.local.self_consistency,
        "core": forces.local.core_correction,
        "nonlocal": forces.nonlocal_pseudopotential,
        "total": forces.total,
        "net": forces.net_force,
    }
    for name, values in tables.items():
        np.testing.assert_allclose(
            values,
            np.asarray(case[name]),
            rtol=0,
            atol=PRINT_TOLERANCE,
            err_msg=f"force table {name!r}",
        )
