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


def _solve(problem, setting: str, translations: str = "auto"):
    environment = {"PARSEC_PERIODIC_SYMMETRY": setting, "PARSEC_PERIODIC_TRANSLATIONS": translations}
    with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
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
            # the point group alone (the face-centred cube also has the translations by half the cell:
            # test_periodic_translations)
            sectors, details = _solve(problem, "auto", translations="off")
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


def displaced_silicon(distance: float = 0.04):
    """:func:`silicon` with the four neighbours of the atom at the origin moved ``distance`` bohr outward along their
    bonds: Td about the origin stays, the face-centring translations go, and the atoms have forces."""

    problem = silicon("pbe")
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    side = np.diag(lattice)
    atoms = []
    for atom in problem.atoms:
        position = np.asarray(atom.position, dtype=np.float64)
        bond = position - side * np.round(position / side)  # from the atom at the origin, nearest image
        if 0.0 < np.linalg.norm(bond) < 0.5 * side[0] * np.sqrt(3.0) / 2.0 + 1.0e-6 and np.allclose(
            np.abs(bond), np.abs(bond[0])
        ):
            position = position + distance * bond / np.linalg.norm(bond)
        atoms.append(Atom(atom.symbol, tuple(position)))
    return replace(problem, atoms=tuple(atoms))


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class PeriodicSectorForceTests(unittest.TestCase):
    """Forces of a cell solved in the sectors of its point group (weighted representations) against the full grid."""

    def test_forces_in_the_sectors_of_a_cell_equal_those_of_the_full_grid(self) -> None:
        from parsec_python.acceleration.Forces.sector_symmetry import sector_force_operations, sector_reduction
        from parsec_python.Forces.symmetry import symmetrize_forces

        problem = displaced_silicon()
        moved = sum(
            1 for a, b in zip(silicon("pbe").atoms, problem.atoms) if not np.array_equal(a.position, b.position)
        )
        self.assertEqual(moved, 4)
        tables, groups = {}, {}
        for setting in ("auto", "off"):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SYMMETRY": setting}), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = driver.prepare_single_point(problem, backend="cupy")
                system.materialize_final_wavefunctions = False
                reduction = sector_reduction(system)
                if setting == "auto":
                    self.assertIsNotNone(reduction, "the cell was not solved in the sectors of a point group")
                    self.assertTrue(np.any(np.asarray(reduction.sector_weights) > 1))
                    groups[setting] = sector_force_operations(system)
                result = driver.run_scf(system)
                self.assertTrue(result.converged)
                tables[setting] = driver.compute_forces(system, result, symmetrize=False)
        sectors, full = tables["auto"], tables["off"]
        self.assertEqual(len(groups["auto"]), 24)  # Td about the origin
        self.assertGreater(float(np.abs(full.total).max()), 1.0e-3)
        np.testing.assert_allclose(sectors.nonlocal_pseudopotential, full.nonlocal_pseudopotential, rtol=0, atol=2e-6)
        np.testing.assert_allclose(sectors.total, full.total, rtol=0, atol=2e-6)
        np.testing.assert_allclose(symmetrize_forces(sectors.total, groups["auto"]), sectors.total, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
