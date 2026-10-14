"""SiNa4 of the Fortran example ``examples/tests/relaxation``, end to end.

The input structure of the example (Si from an old Martins file, Na from a
new one, occupations from ``occup.in``) is solved to self-consistency on the
readable path and compared with the ``parsec.out`` of Fortran PARSEC for the
same input: total energy, energy terms, eigenvalues, occupations and every
force table.  A second calculation, a displaced cation, has a core radius
inside the Si projector and a hole spread by hand over three levels, so that
the radius of the input and the occupations of the file both decide the
energy.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python.Forces import compute_forces
from parsec_python.Input import parse_parsec_input
from parsec_python.SCF import prepare_single_point, run_scf
from parsec_python.tests.force_reference import assert_matches_fortran
from parsec_python.tests.martins_old_reference import (
    load_cases,
    support_sizes,
    write_case,
)


class _FortranCase:
    """Solve one case of the fixture once for the tests of a class."""

    case_name: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.case = load_cases()[cls.case_name]
        with tempfile.TemporaryDirectory() as directory:
            cls.translation = parse_parsec_input(write_case(cls.case, Path(directory)))
            cls.system = prepare_single_point(cls.translation.problem)
        cls.result = run_scf(cls.system)
        cls.forces = compute_forces(cls.system, cls.result)

    def assert_energies_and_levels(self) -> None:
        self.assertTrue(self.result.converged)
        self.assertEqual(support_sizes(self.system), self.case["nonlocal_block_sizes"])
        # Fortran prints eight decimals; both loops stop below 1e-8 Ry.
        self.assertAlmostEqual(
            self.result.energies.total, self.case["energy"], delta=3.0e-8
        )
        for name, value in self.case["energies"].items():
            self.assertAlmostEqual(
                getattr(self.result.energies, name), value, delta=2.0e-7, msg=name
            )
        np.testing.assert_allclose(
            self.result.eigenvalues, self.case["eigenvalues"], rtol=0, atol=2.0e-7
        )
        # Four decimals are printed.
        np.testing.assert_allclose(
            self.result.occupations, self.case["occupations"], rtol=0, atol=5.01e-5
        )
        self.assertTrue(np.isnan(self.result.fermi_level))

    def assert_forces(self) -> None:
        assert_matches_fortran(
            self, self.case, self.result.energies.total, self.forces
        )


class RelaxationExampleStructureTests(_FortranCase, unittest.TestCase):
    case_name = "sina4"

    def test_energy_terms_levels_and_occupations(self) -> None:
        self.assert_energies_and_levels()
        self.assertEqual(self.system.electron_count, 8.0)
        np.testing.assert_array_equal(
            self.result.occupations, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.assertAlmostEqual(
            float(self.system.grid.integrate(self.result.density)), 8.0, delta=1e-10
        )

    def test_forces_match_fortran_parsec(self) -> None:
        self.assert_forces()
        # The structure has the four operations of D2, as in Fortran.
        self.assertEqual(self.forces.symmetry_operations, 4)
        np.testing.assert_allclose(self.forces.total[0], 0.0, atol=1e-10)
        # The example's own atom.cor reports "Max force = 0.37621" for this
        # first step, from an SCF loop that stops at the default criterion
        # of 2e-4 Ry.
        self.assertAlmostEqual(
            float(np.max(np.linalg.norm(self.forces.total, axis=1))), 0.37621, delta=5.0e-5
        )


class DisplacedCationTests(_FortranCase, unittest.TestCase):
    case_name = "sina4_cation_fractional"

    def test_energy_terms_levels_and_occupations(self) -> None:
        self.assert_energies_and_levels()
        self.assertEqual(self.system.electron_count, 7.0)
        self.assertEqual(self.system.pseudopotentials["Si"].projector_support_radius, 1.9)
        np.testing.assert_array_equal(
            self.result.occupations,
            [1.0] + [0.833333333333] * 3 + [0.0] * 4,
        )
        # The three levels that share the hole are well separated.
        self.assertGreater(self.result.eigenvalues[3] - self.result.eigenvalues[1], 0.02)

    def test_forces_match_fortran_parsec(self) -> None:
        self.assert_forces()
        self.assertEqual(self.forces.symmetry_operations, 1)


if __name__ == "__main__":
    unittest.main()
