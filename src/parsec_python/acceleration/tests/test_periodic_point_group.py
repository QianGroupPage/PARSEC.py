"""The point group of a bulk cell at the Gamma point in the accelerated SCF (``PARSEC_PERIODIC_SYMMETRY``).

A cell whose atoms and periodic grid keep Cartesian signed permutations about the origin of the cell solves its
orbitals in the sectors of that group, as a cluster does (the weighted sectors of a larger subgroup where they solve
less).  Its SCF must be the SCF of the full grid: the same energies and levels, in one process and against the
switch that keeps the full grid (``off``).
"""

from __future__ import annotations

import atexit
from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration import driver
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.tests.test_driver_cli import PERIODIC_SILICON, periodic_hydrogen
from parsec_python.models import Atom


def silicon(functional: str, spacing: float = 0.57) -> object:
    """The conventional cube of ``examples/3d_Si`` (8 atoms, an atom at the origin: Td about it), SCF to 1e-9."""

    changes = {
        "Grid_Spacing": f"{spacing}",
        "Expansion_Order": "8",
        "States_Num": "22",
        "Convergence_Criterion": "1.0e-9",
        "Diag_Tolerance": "1.0e-10",
        "Correlation_Type": functional,
    }
    lines = []
    for line in PERIODIC_SILICON.read_text().splitlines():
        keyword = line.split(":", 1)[0].strip()
        lines.append(f" {keyword}: {changes[keyword]}" if ":" in line and keyword in changes else line)
    # The pseudopotential is read when the problem is prepared: the folder stays until the tests end.
    directory = Path(tempfile.mkdtemp(prefix="parsec_pg_"))
    atexit.register(shutil.rmtree, directory, True)
    shutil.copy(PERIODIC_SILICON.parent / "Si_POTRE.DAT", directory)
    path = directory / "parsec.in"
    path.write_text("\n".join(lines) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_parsec_input(path).problem


def _solve(problem, setting: str):
    with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SYMMETRY": setting}), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = driver.prepare_single_point(problem, backend="cupy")
        result = driver.run_scf(system)
    return result, dict(result.backend.details)


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class PeriodicPointGroupSCFTests(unittest.TestCase):
    def assert_same_scf(self, left, right, label, *, energy=1.0e-8, level=1.0e-7) -> None:
        self.assertTrue(left.converged, f"{label}: the sectors did not converge")
        self.assertTrue(right.converged, f"{label}: the full grid did not converge")
        for field in ("total", "exchange_correlation", "hartree", "eigenvalue"):
            self.assertAlmostEqual(
                getattr(left.energies, field), getattr(right.energies, field), delta=energy, msg=f"{label} {field}"
            )
        occupied = int(np.count_nonzero(np.asarray(right.occupations) > 1.0e-6))
        np.testing.assert_allclose(
            np.sort(left.eigenvalues)[:occupied], np.sort(right.eigenvalues)[:occupied], rtol=0, atol=level,
            err_msg=label,
        )

    def test_silicon_in_the_sectors_of_td(self) -> None:
        for functional in ("ca", "pbe"):
            problem = silicon(functional)
            sectors, details = _solve(problem, "auto")
            self.assertIn("24 signed permutations", details["symmetry_detection"], functional)
            self.assertIn("weights 1, 1, 3, 3", details["symmetry_detection"], functional)
            full, full_details = _solve(problem, "off")
            self.assertTrue(full_details["symmetry_detection"].startswith("skipped: periodic cell"), functional)
            self.assert_same_scf(sectors, full, f"Si {functional}")

    def test_a_hydrogen_atom_in_the_sectors_of_oh(self) -> None:
        base = periodic_hydrogen()
        problem = replace(
            base,
            scf=replace(base.scf, convergence_criterion=1.0e-9, max_iterations=80),
            eigensolver=replace(base.eigensolver, tolerance=1.0e-10),
        )
        sectors, details = _solve(problem, "auto")
        self.assertIn("48 signed permutations", details["symmetry_detection"])
        full, _ = _solve(problem, "off")
        self.assert_same_scf(sectors, full, "H Oh")

    def test_a_cell_without_a_point_group_keeps_the_identity(self) -> None:
        problem = silicon("ca", spacing=0.64)
        atoms = list(problem.atoms)
        atoms[3] = Atom(atoms[3].symbol, tuple(np.asarray(atoms[3].position) + np.array((0.21, -0.13, 0.07))))
        system = driver.prepare_single_point(replace(problem, atoms=atoms), backend="cupy")
        details = dict(system.backend_info.details)
        self.assertIn("identity only", details["symmetry_detection"])
        self.assertEqual(details["detected_symmetry_group_order"], "1")


if __name__ == "__main__":
    unittest.main()
