"""The default periodic grid of an input file is the Fortran one: ``adjust_step``'s point counts and ``grid%shift``.

The expected numbers are what the Fortran code did on the inputs of the comparison runs (``parsec_tests/pbc_forces`` and
``slab``): a plain ``Grid_Spacing 0.4`` in a 9.6 bohr cell gave 22 points and 9.6/24 gave 24; the cells of the probe
run gave 16, 20, 24, 26, 26, 30, 34 and 40 points for 18, 20, 24, 26, 28, 30, 36 and 40 requested; the 6.0093 bohr
in-plane cell of MoS2 gave 14 (spacing 0.41) and 12 (0.5); and the grid is half a step off the origin unless
``Ignore_Symmetry`` is set (a cluster run showed the same rule).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python.Grid import build_periodic_grid, fortran_grid_shift, fortran_points_per_axis
from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.models import PeriodicCell, PeriodicGridSettings

_NA = Path(__file__).resolve().parents[1] / "data" / "Na_POTRE.DAT"


class FortranPointCountTests(unittest.TestCase):
    def test_counts_are_those_of_the_fortran_runs(self) -> None:
        np.testing.assert_array_equal(fortran_points_per_axis([9.6, 9.6, 9.6], 0.4), [22, 22, 22])
        np.testing.assert_array_equal(fortran_points_per_axis([9.6, 9.6, 9.6], 9.6 / 24), [24, 24, 24])
        wanted = (18, 20, 24, 26, 28, 30, 36, 40)
        got = [int(fortran_points_per_axis([9.6], 9.6 / n)[0]) for n in wanted]
        self.assertEqual(got, [16, 20, 24, 26, 26, 30, 34, 40])
        np.testing.assert_array_equal(fortran_points_per_axis([6.0093, 6.0093], 0.41), [14, 14])
        np.testing.assert_array_equal(fortran_points_per_axis([6.0093, 6.0093], 0.5), [12, 12])

    def test_the_count_is_even_and_the_step_is_not_below_the_spacing(self) -> None:
        for length in (5.3, 7.77, 10.0, 12.345):
            for spacing in (0.3, 0.45, 0.6, 0.71):
                n = int(fortran_points_per_axis([length], spacing)[0])
                self.assertEqual(n % 2, 0)
                self.assertGreaterEqual(length / n, spacing * (1.0 - 1.0e-12))
                self.assertLess(length / (n + 2), spacing * (1.0 + 1.0e-12))  # the largest such count

    def test_a_side_shorter_than_two_steps_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            fortran_points_per_axis([0.7], 0.6)


class FortranShiftTests(unittest.TestCase):
    def test_half_a_step_unless_the_vectors_are_not_orthogonal(self) -> None:
        self.assertEqual(fortran_grid_shift(5.0 * np.eye(3), False), (0.5, 0.5, 0.5))
        self.assertEqual(fortran_grid_shift(5.0 * np.eye(3), True), (0.0, 0.0, 0.0))
        self.assertEqual(fortran_grid_shift(5.4 * (np.ones((3, 3)) - np.eye(3)), False), (0.0, 0.0, 0.0))
        hexagonal = np.array([[5.0, 0.0, 0.0], [-2.5, 4.33, 0.0], [0.0, 0.0, 8.0]])
        self.assertEqual(fortran_grid_shift(hexagonal, False), (0.0, 0.0, 0.5))  # only c is orthogonal to the others


class PeriodicGridBuilderTests(unittest.TestCase):
    def test_points_are_at_the_shifted_positions(self) -> None:
        lattice = np.array([[6.0, 0.0, 0.0], [1.5, 7.2, 0.0], [0.4, 0.9, 8.4]])
        settings = PeriodicGridSettings(spacing=0.6, expansion_order=6, shift=(0.5, 0.25, 0.75))
        grid = build_periodic_grid(PeriodicCell(lattice), settings)
        expected = (grid.integer_coordinates + np.array([0.5, 0.25, 0.75])) @ (lattice / grid.points_per_axis[:, None])
        np.testing.assert_allclose(grid.coordinates, expected)
        np.testing.assert_allclose(grid.physical_coordinates(grid.integer_coordinates[:5]), expected[:5])

    def test_orthorhombic_cell_with_a_shift_and_a_common_step(self) -> None:
        grid = build_periodic_grid(PeriodicCell(np.diag([6.0, 7.2, 8.4])), PeriodicGridSettings(0.6, 6, shift=(0.5, 0.5, 0.5)))
        self.assertTrue(grid.is_orthorhombic)
        np.testing.assert_allclose(grid.coordinates[0], [0.3, 0.3, 0.3])

    def test_the_fortran_rule_allows_different_steps_in_a_rectangular_cell(self) -> None:
        cell = PeriodicCell(np.diag([9.6, 6.0093, 7.3]))
        grid = build_periodic_grid(cell, PeriodicGridSettings(0.41, 6, points_rule="fortran"))
        np.testing.assert_array_equal(grid.points_per_axis, [22, 14, 16])
        self.assertFalse(grid.is_orthorhombic)  # steps 0.436, 0.429, 0.456: the general (metric) operators
        self.assertAlmostEqual(grid.spacing, 6.0093 / 14)
        with self.assertRaises(ValueError):
            build_periodic_grid(cell, PeriodicGridSettings(0.41, 6))  # the "round" rule wants one common spacing

    def test_operators_on_a_shifted_grid_with_unequal_steps(self) -> None:
        cell = PeriodicCell(np.diag([9.6, 6.0093, 7.3]))
        grid = build_periodic_grid(cell, PeriodicGridSettings(0.41, 8, shift=(0.5, 0.5, 0.5), points_rule="fortran"))
        laplacian, gradient = build_negative_laplacian(grid), build_gradient(grid)
        reciprocal = 2.0 * np.pi * np.linalg.inv(cell.lattice_vectors).T
        for m in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 1)]:
            k = np.asarray(m) @ reciprocal
            phase = grid.coordinates @ k + 0.3
            values = np.cos(phase)
            np.testing.assert_allclose(laplacian @ values, (k @ k) * values, atol=1.0e-5 * (k @ k))
            for axis in range(3):
                np.testing.assert_allclose(gradient[axis] @ values, -k[axis] * np.sin(phase), atol=1.0e-5 * np.linalg.norm(k))


_INPUT = """
Periodic_System: .true.
Boundary_Conditions bulk
begin Cell_Shape
{cell}
end Cell_Shape
Lattice_Vector_Scale: 1.0
Grid_Spacing: {spacing}
Expansion_Order: 6
Correlation_Type: ca
States_Num: 3
{extra}
Atom_Types_Num: 1
Coordinate_Unit: cartesian_bohr
Atom_Type: Na
Pseudopotential_Format: Martins_new
Local_Component: s
begin Atom_Coord
  1.0 1.0 1.0
end Atom_Coord
"""


class ParserDefaultTests(unittest.TestCase):
    def _grid(self, cell: str, spacing, extra: str = ""):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "parsec.in").write_text(_INPUT.format(cell=cell, spacing=spacing, extra=extra))
            (folder / "Na_POTRE.DAT").write_bytes(_NA.read_bytes())
            problem = parse_parsec_input(folder / "parsec.in").problem
        return problem.grid, build_periodic_grid(problem.periodic_cell, problem.grid)

    def test_an_orthorhombic_input_gets_the_fortran_grid(self) -> None:
        settings, grid = self._grid("9.6 0 0\n0 9.6 0\n0 0 9.6", 0.4)
        self.assertEqual((settings.shift, settings.points_rule), ((0.5, 0.5, 0.5), "fortran"))
        np.testing.assert_array_equal(grid.points_per_axis, [22, 22, 22])  # what the Fortran code chose

    def test_ignore_symmetry_puts_a_point_on_the_origin(self) -> None:
        settings, grid = self._grid("9.6 0 0\n0 9.6 0\n0 0 9.6", repr(9.6 / 24), "Ignore_Symmetry: .true.")
        self.assertEqual(settings.shift, (0.0, 0.0, 0.0))
        np.testing.assert_array_equal(grid.points_per_axis, [24, 24, 24])
        np.testing.assert_allclose(grid.coordinates[0], 0.0)

    def test_an_fcc_input_has_no_shift(self) -> None:
        settings, _ = self._grid("0 2.7 2.7\n2.7 0 2.7\n2.7 2.7 0", 0.45)
        self.assertEqual(settings.shift, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
