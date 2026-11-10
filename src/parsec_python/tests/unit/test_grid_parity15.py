"""PARSEC 1.5's grid rules in this port (its parity round, 2026-11-04).

* The orthorhombic grid of the Fortran-derived routines (``Use_Fortran_Routines``) is Fortran's: ``adjust_step``'s
  even counts whose step is not below the input spacing, commensurate with the space group's fractional
  translations, and the points ``(i + shift) h`` with the shift 0.5 unless ``Ignore_Symmetry`` (before:
  ``round(L/h)`` points from the origin).
* ``adjust_step`` takes 1.5's slack ``parsec_tol`` (``grid/inipbc.f90:586-588``).
* The ``Grid_Shift`` block (``io/usrinputfile.F90:1369-1404``) and ``Foldblock_Side_Length``
  (``usrinputfile.F90:798-817``, ``grid/grid_partition.f90:525-607``).
* Periodic atoms are translated to fractional coordinates about ``[-1/2, 1/2]`` at start (``back_to_cell``,
  ``scf/initial.f90:322-326``) on the Fortran-derived routines.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.Grid.cell import back_to_cell
from parsec_python.Grid.pbc import (
    GeneralPeriodicGrid,
    PeriodicRealSpaceGrid,
    adjust_step,
    build_periodic_grid,
    fortran_grid_counts,
    fortran_points_per_axis,
    periodic_grid_shift,
)
from parsec_python.Input.parsec_input import (
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
)
from parsec_python.models import GridSettings, PeriodicCell, PeriodicGridSettings

_DATA = Path(__file__).resolve().parents[1] / "data"

_CLUSTER = """Boundary_Sphere_Radius: 7.0
Grid_Spacing: 0.4
States_Num: 4
Correlation_Type: ca
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
 0.0 0.0 0.0
 0.0 0.0 1.4
end Atom_Coord
"""

_BULK = """Periodic_System: .true.
Boundary_Conditions: bulk
Use_Fortran_Routines: {routines}
begin Cell_Shape
{cell}
end Cell_Shape
Grid_Spacing: {h}
Expansion_Order: {order}
Correlation_Type: ca
States_Num: 20
{extra}
Atom_Types_Num: 1
Atom_Type: {species}
Local_Component: {local}
Coordinate_Unit: lattice_vectors
begin Atom_Coord
{atoms}
end Atom_Coord
"""

_DIAMOND = """  0.00  0.00  0.00
  0.00  0.50  0.50
  0.50  0.00  0.50
  0.50  0.50  0.00
  0.25  0.25  0.25
  0.25  0.75  0.75
  0.75  0.25  0.75
  0.75  0.75  0.25"""

_MP222 = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid"


def _bulk(extra="", *, routines=".true.", cell="  10.26  10.26  10.26", h="0.5", order="12", species="Si",
          local="p", atoms=_DIAMOND) -> str:
    return _BULK.format(extra=extra, routines=routines, cell=cell, h=h, order=order, species=species, local=local,
                        atoms=atoms)


class _ParserCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)
        for name in ("H_POTRE.DAT", "Si_POTRE.DAT"):
            (self.directory / name).write_bytes((_DATA / name).read_bytes())

    def parse(self, text: str):
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        return parse_parsec_input(path)

    def grid(self, translation):
        problem = translation.problem
        return build_periodic_grid(problem.periodic_cell, problem.grid)


class AdjustStepTests(unittest.TestCase):
    def test_the_slack_of_parsec_15_keeps_an_even_multiple(self) -> None:
        # 1/24 is a rounding below 0.4/9.6: the previous tree's test (and Andrew Grieder's count) give 22
        self.assertEqual(adjust_step(9.6, 0.4)[0], 24)
        self.assertEqual(int(fortran_points_per_axis(np.array([9.6]), 0.4)[0]), 22)
        # far from an even multiple nothing changes: the largest even count whose step is not below h
        self.assertEqual(adjust_step(10.26, 0.5), (20, 10.26 / 20))
        self.assertEqual(adjust_step(10.0, 0.6)[0], 16)

    def test_translations_make_the_step_commensurate(self) -> None:
        # diamond's quarter translations: a multiple of four points (TierII/3d_Si_diamond_sc: 48 at 0.2)
        self.assertEqual(adjust_step(10.26, 0.2, translations=(0.25,))[0], 48)
        self.assertEqual(adjust_step(10.26, 0.2)[0], 50)

    def test_a_direction_without_a_commensurate_step_takes_parsec_15s_loop(self) -> None:
        # inipbc.f90:131-252: nint(L/h) points (odd or even) with the step L/n along that direction; the smaller
        # step is the next pass's input, so the other directions can get more points (each alone: 24, 6, 36)
        cases = (
            ((14.7239, 3.07, 21.3785), 0.55, ((0.75,), (0.125, 0.25), (1.0 / 3.0,)), [28, 6, 36]),
            ((4.5, 24.79, 24.9995), 0.35, ((0.125, 1.0 / 3.0), (1.0 / 6.0,), (0.5,)), [13, 66, 72]),
        )
        for sides, spacing, translations, counts in cases:
            with self.subTest(sides=sides):
                alone = [adjust_step(side, spacing, translations=taus)[0] for side, taus in zip(sides, translations)]
                settings = PeriodicGridSettings(
                    spacing=spacing, fortran_routines=True, expansion_order=4, fractional_translations=translations
                )
                points, steps = fortran_grid_counts(PeriodicCell(np.diag(sides)), settings)
                self.assertEqual(points.tolist(), counts)
                self.assertNotEqual(alone, counts)
                # L/n where it was reset; x L (within parsec_tol of L/n) where a step is commensurate
                np.testing.assert_allclose(steps, np.asarray(sides) / points, rtol=0.0, atol=1.0e-15)
                grid = build_periodic_grid(PeriodicCell(np.diag(sides)), settings)
                self.assertEqual(grid.points_per_axis.tolist(), counts)


class OrthorhombicGridTests(unittest.TestCase):
    def test_counts_and_points_are_fortrans(self) -> None:
        cell = PeriodicCell(np.diag([10.26, 10.26, 10.26]))
        grid = build_periodic_grid(cell, PeriodicGridSettings(spacing=0.5, fortran_routines=True))
        self.assertIs(type(grid), PeriodicRealSpaceGrid)
        self.assertEqual(grid.points_per_axis.tolist(), [20, 20, 20])  # round(L/h) was 21
        self.assertEqual(grid.spacing, 10.26 / 20)
        shifted = build_periodic_grid(
            cell, PeriodicGridSettings(spacing=0.5, fortran_routines=True, shift=(0.25, 0.0, 0.5))
        )
        np.testing.assert_array_equal(
            shifted.coordinates, (shifted.integer_coordinates + np.array([0.25, 0.0, 0.5])) * (10.26 / 20)
        )
        np.testing.assert_array_equal(shifted.shift, [0.25, 0.0, 0.5])
        np.testing.assert_array_equal(
            shifted.physical_coordinates(np.array([[3, 4, 5]])), [[(3.25) * 0.513, 4 * 0.513, 5.5 * 0.513]]
        )

    def test_one_step_is_the_orthorhombic_grid_and_steps_that_differ_the_general_grid(self) -> None:
        rectangle = build_periodic_grid(
            PeriodicCell(np.diag([10.26, 10.26, 15.39])), PeriodicGridSettings(spacing=0.5, fortran_routines=True)
        )
        self.assertIs(type(rectangle), PeriodicRealSpaceGrid)  # 20 20 30 points of 0.513 bohr
        self.assertEqual(rectangle.points_per_axis.tolist(), [20, 20, 30])
        # 10 / 10 / 10.5 at 0.5: round(L/h) gave one spacing (21 points), Fortran 20 points of 0.525 bohr
        general = build_periodic_grid(
            PeriodicCell(np.diag([10.0, 10.0, 10.5])), PeriodicGridSettings(spacing=0.5, fortran_routines=True)
        )
        self.assertIsInstance(general, GeneralPeriodicGrid)
        self.assertEqual(general.points_per_axis.tolist(), [20, 20, 20])
        np.testing.assert_allclose(general.spacings, [0.5, 0.5, 0.525])

    def test_the_shift_rule(self) -> None:
        cube = np.diag([6.0, 6.0, 6.0])
        self.assertEqual(periodic_grid_shift(cube, ignore_symmetry=False), (0.5, 0.5, 0.5))
        self.assertEqual(periodic_grid_shift(cube, ignore_symmetry=True), (0.0, 0.0, 0.0))
        self.assertEqual(
            periodic_grid_shift(cube, ignore_symmetry=True, requested=(0.5, -0.25, 0.0)), (0.5, 0.25, 0.0)
        )
        hexagonal = np.array([[3.0, 0.0, 0.0], [-1.5, 2.6, 0.0], [0.0, 0.0, 5.0]])
        self.assertEqual(
            periodic_grid_shift(hexagonal, ignore_symmetry=True, requested=(0.5, 0.5, 0.5)), (0.0, 0.0, 0.5)
        )


class ShiftedGridSymmetryTests(unittest.TestCase):
    """The density symmetrization of a shifted grid maps every point onto its image ``M (r - t)``."""

    def test_the_point_map_is_the_image_on_shifted_and_unshifted_grids(self) -> None:
        from parsec_python.Grid.space_group import find_space_group, periodic_grid_symmetry
        from parsec_python.models import Atom

        cube = 10.26 * np.eye(3)
        fractions = (
            (0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0),
            (0.25, 0.25, 0.25), (0.25, 0.75, 0.75), (0.75, 0.25, 0.75), (0.75, 0.75, 0.25),
        )
        atoms = [Atom("Si", np.array(fraction) @ cube) for fraction in fractions]
        group = find_space_group(cube, atoms)
        self.assertEqual(group.order, 48)
        for shift in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)):
            settings = PeriodicGridSettings(
                fortran_routines=True, spacing=0.64, expansion_order=6, shift=shift,
                fractional_translations=group.commensurate_translations(),
            )
            grid = build_periodic_grid(PeriodicCell(lattice_vectors=cube), settings)
            self.assertIs(type(grid), PeriodicRealSpaceGrid)
            mapping = periodic_grid_symmetry(grid, group)
            fractional = grid.coordinates @ np.linalg.inv(cube)
            for operation, (matrix, translation) in enumerate(
                zip(group.rotation_matrices, group.translation_vectors)
            ):
                with self.subTest(shift=shift, operation=operation):
                    expected = (fractional - translation) @ matrix.T.astype(float)
                    difference = expected - fractional[mapping.images[operation]]
                    difference -= np.rint(difference)
                    # one step would be 1/16; before the fix a shift of 0.5 was off by exactly one step
                    self.assertLess(float(np.max(np.abs(difference))), 1.0e-12)


class ParserGridTests(_ParserCase):
    def test_the_fortran_routines_take_fortrans_shift_and_translations(self) -> None:
        # diamond with k-points: the whole space group (no symchk), its quarter translations: 48 points at 0.2
        translation = self.parse(_bulk(_MP222, h="0.2"))
        grid = self.grid(translation)
        self.assertIs(type(grid), PeriodicRealSpaceGrid)
        self.assertEqual(grid.points_per_axis.tolist(), [48, 48, 48])
        np.testing.assert_array_equal(grid.shift, [0.5, 0.5, 0.5])
        self.assertIsNone(translation.problem.space_group)  # found for adjust_step, not kept
        # Andrew Grieder's grid: the largest even count without translations
        his = self.parse(_bulk(_MP222, h="0.2", routines=".false."))
        self.assertEqual(self.grid(his).points_per_axis.tolist(), [50, 50, 50])
        # at the Gamma point the 8-atom cube fails symchk: the identity, no translation (as Fortran)
        gamma = self.grid(self.parse(_bulk(h="0.2")))
        self.assertEqual(gamma.points_per_axis.tolist(), [50, 50, 50])
        # Ignore_Symmetry: no group, no shift
        plain = self.grid(self.parse(_bulk(_MP222 + "\nIgnore_Symmetry: .true.", h="0.2")))
        self.assertEqual(plain.points_per_axis.tolist(), [50, 50, 50])
        np.testing.assert_array_equal(plain.shift, [0.0, 0.0, 0.0])

    def test_grid_shift_of_a_bulk_cell(self) -> None:
        block = "begin Grid_Shift\n 0.5 -0.25 0\nend Grid_Shift\nIgnore_Symmetry: .true."
        for routines in (".true.", ".false."):
            with self.subTest(routines=routines):
                translation = self.parse(_bulk(block, routines=routines))
                self.assertEqual(translation.routines.fortran, routines == ".true.")
                np.testing.assert_array_equal(self.grid(translation).shift, [0.5, 0.25, 0.0])
                self.assertTrue(any("Grid Shift is read from input" in note for note in translation.warnings))

    def test_grid_shift_of_a_cluster_and_its_checks(self) -> None:
        translation = self.parse(_CLUSTER.format(extra="Ignore_Symmetry: .true.\nbegin Grid_Shift\n0.5 0.5 0.5\nend Grid_Shift"))
        self.assertEqual(translation.problem.grid.shift, (0.5, 0.5, 0.5))  # 1.5: with Ignore_Symmetry too
        self.assertEqual(self.parse(_CLUSTER.format(extra="Ignore_Symmetry: .true.")).problem.grid.shift, (0.0, 0.0, 0.0))
        two = self.parse(_CLUSTER.format(extra="begin Grid_Shift\n0.25 0 0\n0.5 0.5 0.5\nend Grid_Shift"))
        self.assertEqual(two.problem.grid.shift, (0.25, 0.0, 0.0))
        self.assertTrue(any("only needs one line" in note for note in two.warnings))
        with self.assertRaises(ParsecInputError):
            self.parse(_CLUSTER.format(extra="begin Grid_Shift\n0.5 0.5\nend Grid_Shift"))
        with self.assertRaises(UnsupportedParsecOptionError):
            self.parse(_CLUSTER.format(extra="begin Grid_Shift\n1.0 0 0\nend Grid_Shift"))
        # the density symmetrization needs points that the point group maps onto points (M s - s integral); the
        # parser's test is the run's (density_point_group), which the accepted shifts pass
        from parsec_python.SCF.single_point import density_point_group
        from parsec_python.V_ion import center_cluster_geometry

        symmetrized = "Symmetrize_Charge_Density: .true.\nbegin Grid_Shift\n{shift}\nend Grid_Shift"
        for shift in ("0 0 0", "0.5 0.5 0"):  # H2 along z: x <-> y needs s_x = s_y, reflections 0 or 1/2
            with self.subTest(shift=shift):
                problem = self.parse(_CLUSTER.format(extra=symmetrized.format(shift=shift))).problem
                self.assertTrue(problem.scf.symmetrize_density)
                group = density_point_group(build_cluster_grid(problem.grid), center_cluster_geometry(problem.atoms))
                self.assertEqual(group.order, 16)
        for shift in ("0.25 0.25 0", "0.5 0 0"):
            with self.subTest(shift=shift), self.assertRaises(UnsupportedParsecOptionError):
                self.parse(_CLUSTER.format(extra=symmetrized.format(shift=shift)))

    def test_foldblock_side_length_two_fills_a_cluster_with_blocks(self) -> None:
        # TierI/0d_H2 of PARSEC 1.5: 25600 points (22575 without the blocks), the symmetry switched off
        translation = self.parse(_CLUSTER.format(extra="Foldblock_Side_Length: 2"))
        self.assertTrue(translation.ignore_symmetry)
        self.assertFalse(translation.use_symmetry)
        self.assertEqual(translation.problem.grid.shift, (0.0, 0.0, 0.0))
        self.assertEqual(translation.problem.grid.foldblock_side_length, 2)
        self.assertTrue(any("SYMMETRY IS AUTOMATICALLY SWITCHED OFF" in note for note in translation.warnings))
        grid = build_cluster_grid(translation.problem.grid)
        self.assertEqual(grid.size, 25600)
        plain = build_cluster_grid(GridSettings(spacing=0.4, radius=7.0, shift=(0.0, 0.0, 0.0)))
        self.assertEqual(plain.size, 22575)
        # whole blocks [2m, 2m + 1] holding every point of the sphere; the rows find their points
        blocks = {tuple(point) for point in np.floor_divide(grid.integer_coordinates, 2)}
        self.assertEqual(len(blocks) * 8, grid.size)
        self.assertTrue({tuple(p) for p in plain.integer_coordinates} <= {tuple(p) for p in grid.integer_coordinates})
        np.testing.assert_array_equal(grid.rows_for_integer_coordinates(grid.integer_coordinates), np.arange(grid.size))
        np.testing.assert_array_equal(grid.coordinates, grid.integer_coordinates * 0.4)
        self.assertNotIn("foldblock", repr(plain.settings))  # the text of every earlier grid (cache keys)
        self.assertEqual(self.parse(_CLUSTER.format(extra="Foldblock_Side_Length: 1")).problem.grid.foldblock_side_length, 1)

    def test_foldblock_side_length_of_periodic_cells_and_what_is_refused(self) -> None:
        bulk = self.parse(_bulk("Foldblock_Side_Length: 2"))
        self.assertTrue(bulk.ignore_symmetry)
        self.assertEqual(self.grid(bulk).points_per_axis.tolist(), [20, 20, 20])
        np.testing.assert_array_equal(self.grid(bulk).shift, [0.0, 0.0, 0.0])
        self.assertTrue(any("fill the periodic cell" in note for note in bulk.warnings))
        for name, text in (
            ("odd half order", _bulk("Foldblock_Side_Length: 2", order="6")),
            ("side 3", _CLUSTER.format(extra="Foldblock_Side_Length: 3")),
            ("side 0", _CLUSTER.format(extra="Foldblock_Side_Length: 0")),
        ):
            with self.subTest(name):
                with self.assertRaises(UnsupportedParsecOptionError):
                    self.parse(text)

    def test_periodic_atoms_are_translated_about_the_cell_centre(self) -> None:
        atoms = "  0.75  0.10  0.20\n  0.30  0.40  0.60"
        ours = self.parse(_bulk("Ignore_Symmetry: .true.", atoms=atoms, species="H", local="s"))
        np.testing.assert_array_equal(ours.problem.atoms[0].position, [0.75 * 10.26 - 10.26, 0.10 * 10.26, 0.20 * 10.26])
        np.testing.assert_array_equal(ours.problem.atoms[1].position, [0.30 * 10.26, 0.40 * 10.26, 0.60 * 10.26 - 10.26])
        his = self.parse(_bulk("Ignore_Symmetry: .true.", atoms=atoms, species="H", local="s", routines=".false."))
        np.testing.assert_array_equal(his.problem.atoms[0].position, [0.75 * 10.26, 0.10 * 10.26, 0.20 * 10.26])
        # Recenter_Atoms_In_Aperiodic_Directions .false. skips the start-up back_to_cell (initial.f90:202)
        kept = self.parse(
            _bulk(
                "Ignore_Symmetry: .true.\nRecenter_Atoms_In_Aperiodic_Directions: .false.",
                atoms=atoms, species="H", local="s",
            )
        )
        np.testing.assert_array_equal(kept.problem.atoms[0].position, [0.75 * 10.26, 0.10 * 10.26, 0.20 * 10.26])

    def test_back_to_cell(self) -> None:
        lattice = np.diag([10.0, 8.0, 6.0])
        inside = np.array([1.0, -3.9, 2.9])
        self.assertIs(back_to_cell(inside, lattice), inside)
        np.testing.assert_array_equal(back_to_cell(np.array([7.0, 9.0, -3.5]), lattice), [-3.0, 1.0, 2.5])
        # fractional coordinates in [-1/2, 1/2): +1/2 becomes -1/2, -1/2 stays (Fortran's nint would swap the two),
        # so a written input reads back to the same positions
        np.testing.assert_array_equal(back_to_cell(np.array([5.0, -4.0, 0.0]), lattice), [-5.0, -4.0, 0.0])
        once = back_to_cell(np.array([5.0, 4.0, 3.0]), lattice)
        np.testing.assert_array_equal(once, [-5.0, -4.0, -3.0])
        self.assertIs(back_to_cell(once, lattice), once)
        # a wire: the confined directions stay
        np.testing.assert_array_equal(back_to_cell(np.array([7.0, 9.0, -3.5]), lattice, 1), [-3.0, 9.0, -3.5])


if __name__ == "__main__":
    unittest.main()
