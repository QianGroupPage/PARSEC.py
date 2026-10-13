"""Forces of the readable path against Fortran PARSEC, end to end.

One small water molecule of low symmetry is solved to self-consistency on
the full grid and every force table is compared with the ``parsec.out`` of
Fortran PARSEC for the same input.  The accelerated package repeats the
comparison for more structures (core correction, p and d projectors, cubic
splines), where a calculation takes a second.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python.Forces import compute_forces
from parsec_python.Input import parse_parsec_input
from parsec_python.SCF import (
    prepare_single_point,
    run_scf,
    update_geometry,
    warm_start_for_moved_atoms,
)
from parsec_python.models import Atom
from parsec_python.tests.force_reference import (
    assert_matches_fortran,
    load_cases,
    write_case,
)


class ReadablePathForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = load_cases()["water_lda_micro"]
        with tempfile.TemporaryDirectory() as directory:
            translation = parse_parsec_input(write_case(cls.case, Path(directory)))
            cls.system = prepare_single_point(translation.problem)
        cls.result = run_scf(cls.system)
        cls.forces = compute_forces(cls.system, cls.result)

    def test_forces_match_fortran_parsec(self) -> None:
        self.assertTrue(self.result.converged)
        assert_matches_fortran(self, self.case, self.result.energies.total, self.forces)
        np.testing.assert_allclose(self.forces.total.sum(axis=0), 0.0, atol=1e-12)
        self.assertEqual(self.forces.symmetry_operations, 1)

    def test_moved_atoms_continue_from_the_previous_loop(self) -> None:
        shift = np.array([[0.0, 0.02, 0.0], [0.03, -0.01, 0.01], [-0.02, -0.01, 0.0]])
        atoms = tuple(
            Atom(atom.symbol, atom.position + delta)
            for atom, delta in zip(self.system.atoms, shift)
        )
        moved = update_geometry(self.system, atoms)
        self.assertIs(moved.grid, self.system.grid)
        self.assertIs(moved.negative_laplacian, self.system.negative_laplacian)
        warm = run_scf(
            moved,
            warm_start=warm_start_for_moved_atoms(self.system, self.result, moved),
        )
        cold = run_scf(moved)
        self.assertTrue(warm.converged and cold.converged)
        self.assertLess(warm.iterations, cold.iterations)
        self.assertAlmostEqual(warm.energies.total, cold.energies.total, delta=2e-8)
        np.testing.assert_allclose(
            compute_forces(moved, warm).total,
            compute_forces(moved, cold).total,
            rtol=0,
            atol=5e-7,
        )
        # The moved structure has a different energy: the loop did not just
        # return what it was handed.
        self.assertGreater(abs(warm.energies.total - self.result.energies.total), 1e-4)

    def test_atoms_must_stay_inside_the_domain(self) -> None:
        atoms = list(self.system.atoms)
        atoms[1] = Atom("H", (0.0, 0.0, self.system.grid.settings.radius + 0.1))
        with self.assertRaisesRegex(ValueError, "outside the domain"):
            update_geometry(self.system, atoms)
        with self.assertRaisesRegex(ValueError, "number, order and species"):
            update_geometry(self.system, atoms[:2])


if __name__ == "__main__":
    unittest.main()
