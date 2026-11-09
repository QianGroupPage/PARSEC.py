"""The irreducible k-point wedge on the default routines and the space-group machinery behind it (2026-11-08).

* translations snapped to simple fractions, so that ``adjust_step`` settles for coordinates with round-off;
* the subgroup of the crystal's operations that map a grid onto itself (diamond at 18 points per cubic edge:
  ``Td`` about an atom, which with time reversal reduces a mesh as ``Oh``);
* the integer affine density tables (the table of the rounded images, bit for bit) and their array-backend average;
* the parser: ``Use_Fortran_Routines: .false.`` with a ``Monkhorst_Pack_Shift`` block reduces a bulk single point.
"""

from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Grid.kpoints import irreducible_monkhorst_pack, kpoint_sampling, mesh_preserving_operations
from parsec_python.Grid.pbc import adjusted_steps, build_periodic_grid
from parsec_python.Grid.point_group import symmetrize_density
from parsec_python.Grid.space_group import (
    _nint_array,
    find_grid_space_group,
    find_space_group,
    periodic_grid_symmetry,
    snapped_translation,
)
from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings

_A = 3.567 / 0.529177210903
_BASIS = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
_C_POTRE = Path(__file__).resolve().parents[4] / "examples" / "3d_diamond_benchmark" / "C_POTRE.DAT"


def _diamond(n: int, decimals: int = 10):
    """The 8 n^3 atoms of n^3 cubic cells and the lattice, coordinates rounded as an input file writes them."""

    atoms = []
    for i, j, k in itertools.product(range(n), repeat=3):
        for shift in (0.0, 0.25):
            for basis in _BASIS:
                atoms.append(
                    Atom("C", [round((c + b + shift) * _A, decimals) for c, b in zip((i, j, k), basis)])
                )
    return np.diag([round(n * _A, 10)] * 3), atoms


def _grid(lattice, points_per_edge: int, n: int = 1):
    settings = PeriodicGridSettings(
        spacing=_A / points_per_edge * (1.0 - 1.0e-9), expansion_order=8, shift=(0.5, 0.5, 0.5), points_rule="fortran"
    )
    return build_periodic_grid(PeriodicCell(lattice_vectors=lattice), settings)


def _rounded_images(grid, group) -> np.ndarray:
    """The table of the point-by-point rounding (the construction before the affine maps)."""

    points = np.asarray(grid.integer_coordinates, dtype=np.int64)
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    shift = np.asarray(grid.shift, dtype=np.float64)
    fractional = (points.astype(np.float64) + shift) / counts
    rows = []
    for matrix, translation in zip(group.rotation_matrices, group.translation_vectors):
        mapped = (fractional - translation) @ matrix.T.astype(np.float64)
        target = _nint_array(mapped * counts - shift).astype(np.int64)
        rows.append(grid.rows_for_integer_coordinates(target % counts))
    return np.array(rows)


class SnappedTranslationTests(unittest.TestCase):
    def test_round_off_is_snapped_and_exact_values_keep_their_bits(self) -> None:
        self.assertEqual(snapped_translation(0.2499999999962911), 0.25)
        self.assertEqual(snapped_translation(0.25000000000148354), 0.25)
        self.assertEqual(snapped_translation(0.08333333333341579), 1.0 / 12.0)
        third = 0.33333333333333337  # one ulp above the double of 1/3: round-off of an exact input, kept
        self.assertEqual(snapped_translation(third), third)
        self.assertEqual(snapped_translation(0.27), 0.27)  # no fraction of q <= 48 within 1e-8
        self.assertEqual(snapped_translation(0.5), 0.5)

    def test_fortran_grid_of_rounded_diamond_settles(self) -> None:
        for n in (1, 2):
            with self.subTest(n=n):
                lattice, atoms = _diamond(n)
                group = find_space_group(lattice, atoms)
                lengths = [float(lattice[0, 0])] * 3
                points, _ = adjusted_steps(lengths, _A / 18 * (1.0 - 1.0e-9), group.commensurate_translations())
                self.assertEqual(points, [16 * n] * 3)


class GridSpaceGroupTests(unittest.TestCase):
    def test_diamond_at_18_points_keeps_td_and_at_16_the_whole_group(self) -> None:
        lattice, atoms = _diamond(1)
        grid = _grid(lattice, 18)
        group = find_grid_space_group(lattice, atoms, grid.points_per_axis, grid.shift)
        self.assertEqual(group.order, 24)
        self.assertEqual(float(np.max(np.abs(group.translation_vectors))), 0.0)
        for matrix in group.rotation_matrices:
            # Td about the atom at the origin: signed permutations with an even number of minus signs
            self.assertEqual(int(np.prod(matrix.sum(axis=1))), 1)
        fortran = _grid(lattice, 16)
        self.assertEqual(find_grid_space_group(lattice, atoms, fortran.points_per_axis, fortran.shift).order, 48)

    def test_wedges_of_td_and_time_reversal_are_those_of_oh(self) -> None:
        lattice, atoms = _diamond(2)
        grid = _grid(lattice, 18)
        group = find_grid_space_group(lattice, atoms, grid.points_per_axis, grid.shift)
        full = find_space_group(lattice, atoms)
        for mesh, expected in ((2, [1, 3, 3, 1]), (3, [1, 6, 12, 8]), (4, [1, 6, 3, 12, 12, 3, 8, 12, 6, 1])):
            with self.subTest(mesh=mesh):
                td = irreducible_monkhorst_pack(lattice, (mesh,) * 3, (0.0,) * 3, group.reciprocal_rotations)
                oh = irreducible_monkhorst_pack(lattice, (mesh,) * 3, (0.0,) * 3, full.reciprocal_rotations)
                self.assertEqual(td.counts.tolist(), expected)
                np.testing.assert_array_equal(td.fractional, oh.fractional)

    def test_a_shifted_mesh_keeps_the_operations_that_map_it(self) -> None:
        lattice, atoms = _diamond(1)
        group = find_space_group(lattice, atoms)
        self.assertIs(mesh_preserving_operations(group, (2, 2, 2), (0.0, 0.0, 0.0)), group)
        self.assertIs(mesh_preserving_operations(group, (2, 2, 2), (0.5, 0.5, 0.5)), group)
        kept = mesh_preserving_operations(group, (2, 2, 2), (0.5, 0.0, 0.0))
        self.assertEqual(kept.order, 16)
        for matrix in kept.reciprocal_rotations:
            self.assertEqual(abs(int(matrix[0, 0])), 1)


class AffineTableTests(unittest.TestCase):
    def test_tables_are_the_rounded_images(self) -> None:
        lattice, atoms = _diamond(1)
        for edge, group in (
            (18, None),
            (16, find_space_group(lattice, atoms)),
        ):
            grid = _grid(lattice, edge)
            group = group or find_grid_space_group(lattice, atoms, grid.points_per_axis, grid.shift)
            with self.subTest(edge=edge, order=group.order):
                symmetry = periodic_grid_symmetry(grid, group)
                self.assertIsNotNone(symmetry.affine_maps)
                np.testing.assert_array_equal(symmetry.images, _rounded_images(grid, group))

    def test_oblique_cell(self) -> None:
        # diamond in the primitive fcc cell on the default grid (steps a_i/n_i, no shift along oblique vectors):
        # integer matrices that are no permutations of the axes
        fcc = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * _A
        atoms = [Atom("C", (0.0, 0.0, 0.0)), Atom("C", tuple(np.full(3, 0.25 * _A)))]
        settings = PeriodicGridSettings(spacing=0.4, expansion_order=4, shift=(0.0, 0.0, 0.0), points_rule="fortran")
        grid = build_periodic_grid(PeriodicCell(lattice_vectors=fcc), settings)
        group = find_grid_space_group(fcc, atoms, grid.points_per_axis, grid.shift)
        self.assertGreater(group.order, 1)
        self.assertTrue(any(np.count_nonzero(matrix) > 3 for matrix in group.rotation_matrices))
        symmetry = periodic_grid_symmetry(grid, group)
        np.testing.assert_array_equal(symmetry.images, _rounded_images(grid, group))

    def test_array_backend_average_is_the_hosts(self) -> None:
        from parsec_python.acceleration.SCF.periodic_symmetry import PeriodicDensitySymmetrizer

        lattice, atoms = _diamond(1)
        grid = _grid(lattice, 18)
        group = find_grid_space_group(lattice, atoms, grid.points_per_axis, grid.shift)
        symmetry = periodic_grid_symmetry(grid, group)
        density = np.random.default_rng(3).random(grid.size)
        expected, report = symmetrize_density(density, symmetry, grid.volume_element)
        for table in (False, True):
            with self.subTest(table=table):
                source = symmetry
                if table:
                    from dataclasses import replace

                    source = replace(symmetry, affine_maps=None, counts=None)
                symmetric, ours = PeriodicDensitySymmetrizer(source).symmetrize_density(density, grid.volume_element)
                np.testing.assert_array_equal(symmetric, expected)
                self.assertAlmostEqual(ours.maximum_deviation_percent, report.maximum_deviation_percent, places=12)
                self.assertAlmostEqual(ours.charge_after, report.charge_after, places=10)


_DIAMOND_INPUT = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  {edge:.10f} 0.0 0.0
  0.0 {edge:.10f} 0.0
  0.0 0.0 {edge:.10f}
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: {spacing:.14f}
Expansion_Order: 8
Correlation_Type: pbe
States_Num: 22
{extra}
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
{mesh} {mesh} {mesh}
end Monkhorst_Pack_Grid
begin Monkhorst_Pack_Shift
0.0 0.0 0.0
end Monkhorst_Pack_Shift
Atom_Types_Num: 1
Atom_Type: C
Pseudopotential_Format: martins_new
Local_Component: d
Cubic_Spline: true
begin Atom_Coord
{atoms}end Atom_Coord
"""


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        (self.directory / "C_POTRE.DAT").write_bytes(_C_POTRE.read_bytes())

    def parse(self, mesh: int, extra: str = "Use_Fortran_Routines: .false."):
        lattice, atoms = _diamond(1, decimals=12)
        text = _DIAMOND_INPUT.format(
            edge=lattice[0, 0],
            spacing=_A / 18 * (1.0 - 1.0e-9),
            extra=extra,
            mesh=mesh,
            atoms="".join(f"  {a.position[0]:.12f} {a.position[1]:.12f} {a.position[2]:.12f}\n" for a in atoms),
        )
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        return parse_parsec_input(path)

    def test_default_routines_reduce_the_mesh_on_the_grid_of_grid_spacing(self) -> None:
        for mesh, points in ((2, 4), (3, 4), (4, 10)):
            with self.subTest(mesh=mesh):
                translation = self.parse(mesh)
                problem = translation.problem
                self.assertFalse(translation.routines.fortran)
                self.assertEqual(problem.monkhorst_pack_dimensions, (mesh,) * 3)
                self.assertEqual(problem.monkhorst_pack_shift, (0.0, 0.0, 0.0))
                self.assertEqual(problem.space_group.order, 24)
                self.assertTrue(problem.scf.symmetrize_density)
                k_points, weights = kpoint_sampling(problem)
                self.assertEqual(len(k_points), points)
                self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=14)
                self.assertTrue(any("24 of the 48 operations" in line for line in translation.warnings))

    def test_the_prepared_system_averages_the_density_with_the_group(self) -> None:
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        problem = self.parse(2).problem
        system = prepare_periodic_single_point(problem)
        self.assertIsNotNone(system.density_symmetry)
        self.assertEqual(system.density_symmetry.order, 24)
        density = np.asarray(system.initial_density)
        symmetric, _ = symmetrize_density(density, system.density_symmetry, system.grid.volume_element)
        self.assertLess(float(np.max(np.abs(symmetric - density))), 1.0e-10)  # the atomic densities are symmetric
        # without a group (time reversal alone) these routines average nothing, as before
        unsymmetrized = self.parse(2, "Use_Fortran_Routines: .false.\nSymmetrize_Charge_Density: .false.").problem
        self.assertIsNone(prepare_periodic_single_point(unsymmetrized).density_symmetry)

    def test_time_reversal_alone_and_the_full_mesh(self) -> None:
        unsymmetrized = self.parse(2, "Use_Fortran_Routines: .false.\nSymmetrize_Charge_Density: .false.").problem
        self.assertIsNone(unsymmetrized.space_group)
        self.assertFalse(unsymmetrized.scf.symmetrize_density)
        # every point of a Gamma-centred 2^3 mesh is its own time-reversal partner
        self.assertEqual(len(kpoint_sampling(unsymmetrized)[0]), 8)
        moving = self.parse(2, "Use_Fortran_Routines: .false.\nSkip_Force: .false.").problem
        self.assertIsNone(moving.monkhorst_pack_dimensions)
        self.assertEqual(len(moving.kpoints), 8)

    def listed(self, extra: str):
        """The 2x2x2 wedge as a weighted Kpoint_List (Gamma, X, M, R with 1, 3, 3, 1)."""

        listing = (
            "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n"
            " 0.0 0.0 0.0 1.0\n 0.0 0.0 0.5 3.0\n 0.0 0.5 0.5 3.0\n 0.5 0.5 0.5 1.0\nend Kpoint_List"
        )
        translation = self.parse(2, extra)
        text = (self.directory / "parsec.in").read_text(encoding="utf-8")
        mesh = text[text.index("Kpoint_Method: mp"): text.index("end Monkhorst_Pack_Shift") + len("end Monkhorst_Pack_Shift")]
        (self.directory / "parsec.in").write_text(text.replace(mesh, listing), encoding="utf-8")
        del translation
        return parse_parsec_input(self.directory / "parsec.in")

    def test_a_weighted_list_runs_as_given_and_averages_on_request(self) -> None:
        plain = self.listed("Use_Fortran_Routines: .false.")
        self.assertFalse(plain.routines.fortran)
        self.assertIsNone(plain.problem.space_group)
        self.assertFalse(plain.problem.scf.symmetrize_density)
        np.testing.assert_allclose(kpoint_sampling(plain.problem)[1] * 8, [1.0, 3.0, 3.0, 1.0])
        self.assertTrue(any(line.startswith("Kpoint_List with unequal weights") for line in plain.warnings))
        averaged = self.listed("Use_Fortran_Routines: .false.\nSymmetrize_Charge_Density: .true.")
        self.assertFalse(averaged.routines.fortran)
        self.assertEqual(averaged.problem.space_group.order, 24)
        self.assertTrue(averaged.problem.scf.symmetrize_density)
        # without the parameter such a list keeps the Fortran-derived routines (a default, as a shifted mesh)
        default = self.listed("")
        self.assertTrue(default.routines.by_default)
        # forces of a weighted list (a wedge) are completed over the space group by the Fortran-derived routines
        forces = self.listed("Use_Fortran_Routines: .false.\nCalculate_Force: .true.")
        self.assertTrue(forces.routines.fortran)
        self.assertIn("Kpoint_List of unequal weights", forces.routines.describe())

    def test_without_the_parameter_the_shift_block_keeps_the_fortran_derived_routines(self) -> None:
        translation = self.parse(2, "")
        self.assertTrue(translation.routines.fortran)
        # adjust_step with the snapped quarter translations: 16 points per edge (no longer "do not settle")
        self.assertEqual(len(kpoint_sampling(translation.problem)[0]), 4)


if __name__ == "__main__":
    unittest.main()
