"""The routines of a run: Andrew Grieder's or the Fortran-derived ones, chosen by Use_Fortran_Routines.

``parsec_python.Input.routines`` decides once per input; the parser records the choice on the translation and on the
periodic grid settings, and every periodic routine follows the grid's choice, so both kinds of calculation run in one
process.  Without the parameter each feature takes its default (Andrew Grieder's routines; the Fortran-derived ones
for slabs, a shifted grid of k-points of a bulk cell, the forces of a bulk cell at the Gamma point and a bulk cell whose
space group sets Fortran's grid); ``.false.`` selects his wherever his code accepts the input, ``.true.`` the
Fortran-derived ones.  An input his routines do not compute runs on the Fortran-derived ones
by itself, with the reason in a warning and in parsec.out.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Grid.pbc import (
    GeneralPeriodicGrid,
    PeriodicRealSpaceGrid,
    SlabRealSpaceGrid,
    build_periodic_grid,
    uses_fortran_routines,
)
from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.Input.routines import (
    GAMMA_FORCES_FEATURE,
    SYMMETRIC_GRID_FEATURE,
    bulk_default_features,
    routines_report_line,
)
from parsec_python.models import PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.V_ion.ewald import local_potential_uses_spline

_DATA = Path(__file__).resolve().parents[1] / "data"

_BULK = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
{cell}
end Cell_Shape
Grid_Spacing: 0.5
Expansion_Order: 6
Ignore_Symmetry: .true.
Correlation_Type: ca
States_Num: 4
Convergence_Criterion: 1.0e-7
Max_Iter: 60
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  {a}
  2.9  3.1  2.9
end Atom_Coord
"""

_SLAB = """Periodic_System: {periodic}
Boundary_Conditions: slab
begin Cell_Shape
  5.0  0.0  0.0
  0.0  5.0  0.0
end Cell_Shape
Boundary_Sphere_Radius: 5.0
Grid_Spacing: 0.45
Expansion_Order: 6
Correlation_Type: ca
States_Num: 4
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.10  -0.10   0.65
 -0.40  -0.20  -0.75
end Atom_Coord
"""

_CUBE = "  6.0  6.0  6.0"
_FCC = "  0.0  3.0  3.0\n  3.0  0.0  3.0\n  3.0  3.0  0.0"

#: Two H atoms in the diamond structure of an fcc cell (space group Fd-3m, fractional translations of a quarter):
#: Fortran's grid has a multiple of four points along each lattice vector (8 at the spacing 0.42 for |a| = 4.24),
#: Andrew Grieder's grid without a space group the largest even number (10).
_DIAMOND = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  0.0  3.0  3.0
  3.0  0.0  3.0
  3.0  3.0  0.0
end Cell_Shape
Coordinate_Unit: lattice_vectors
Grid_Spacing: 0.42
Expansion_Order: 4
Correlation_Type: ca
States_Num: 4
Convergence_Criterion: 1.0e-7
Max_Iter: 60
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.00  0.00  0.00
  0.25  0.25  0.25
end Atom_Coord
"""
_MP222 = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid"
_MP111 = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n1 1 1\nend Monkhorst_Pack_Grid"


def _bulk(extra: str = "", *, cell: str = _CUBE, a: str = "2.2  3.0  3.1") -> str:
    return _BULK.format(extra=extra, cell=cell, a=a)


def _slab(extra: str = "", *, periodic: str = ".true.") -> str:
    return _SLAB.format(extra=extra, periodic=periodic)


#: Andrew Grieder's routines where his code accepts the input (his slab and his shifted mesh are not the default).
_HIS = "Use_Fortran_Routines: .false."
_SHEETS = (
    "add_charged_sheets: .true.\nnum_charged_sheets: 1\nbegin sheet_coord\n 0.0 0.0 3.0\nend sheet_coord\nsheet_charge: 0.1"
)
_WEIGHTED = "Kpoint_Method: manual\nbegin Kpoint_List\n0.0 0.0 0.0 1.0\n0.25 0.0 0.0 3.0\nend Kpoint_List"
_SHIFTED_BULK = (
    "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid\n"
    "begin Monkhorst_Pack_Shift\n0.5 0.5 0.5\nend Monkhorst_Pack_Shift"
)


class _ParserCase(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        (self.directory / "H_POTRE.DAT").write_bytes((_DATA / "H_POTRE.DAT").read_bytes())

    def parse(self, text: str):
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        return parse_parsec_input(path)


class RoutineChoiceTests(_ParserCase):
    def test_a_periodic_input_takes_the_default_routines(self) -> None:
        translation = self.parse(_bulk())
        self.assertFalse(translation.routines.fortran)
        grid = translation.problem.grid
        self.assertFalse(grid.fortran_routines)
        self.assertEqual(grid.points_rule, "fortran")  # Andrew Grieder's grid: Fortran's adjust_step
        self.assertIn("points_rule='fortran'", repr(grid))  # his routines name themselves in cache keys
        self.assertIn("default (Andrew Grieder's)", routines_report_line(translation))
        self.assertFalse(any("Fortran-derived" in warning for warning in translation.warnings))

    def test_use_fortran_routines_selects_the_routines_of_parity_dev(self) -> None:
        translation = self.parse(_bulk("Use_Fortran_Routines: .true."))
        self.assertTrue(translation.routines.fortran)
        self.assertTrue(translation.routines.requested)
        grid = translation.problem.grid
        self.assertTrue(grid.fortran_routines)
        # the text of the settings is that of the Fortran parity round (its cache keys hash it)
        self.assertNotIn("fortran_routines", repr(grid))
        self.assertNotIn("points_rule", repr(grid))
        self.assertIn("Use_Fortran_Routines: .true.", routines_report_line(translation))

    def test_a_cluster_keeps_the_default_without_a_note(self) -> None:
        text = (_DATA / "H2_parsec.in").read_text(encoding="utf-8")
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        translation = parse_parsec_input(path)
        self.assertFalse(translation.routines.fortran)
        self.assertIsNone(routines_report_line(translation))

    def test_both_routines_build_their_own_grid_in_one_process(self) -> None:
        cell = PeriodicCell(3.0 * (np.ones((3, 3)) - np.eye(3)))
        default = build_periodic_grid(cell, PeriodicGridSettings(spacing=0.4, expansion_order=6, points_rule="fortran"))
        fortran = build_periodic_grid(cell, PeriodicGridSettings(spacing=0.4, expansion_order=6, fortran_routines=True))
        self.assertIs(type(default), PeriodicRealSpaceGrid)
        self.assertFalse(default.is_orthorhombic)
        self.assertIsInstance(fortran, GeneralPeriodicGrid)
        self.assertFalse(uses_fortran_routines(default))
        self.assertTrue(uses_fortran_routines(fortran))
        # the local potential's interpolation follows the grid: his spline, or the species' Cubic_Spline
        species = SpeciesPotential(_DATA / "H_POTRE.DAT", 0)
        self.assertTrue(local_potential_uses_spline(species, default))
        self.assertFalse(local_potential_uses_spline(species, fortran))

    def test_the_orthorhombic_grid_of_the_fortran_routines_reads_the_shift(self) -> None:
        # PARSEC 1.5's parity round: Fortran's points (i + shift) h (before, from the origin whatever the shift said)
        cell = PeriodicCell(np.diag([6.0, 6.0, 6.0]))
        settings = PeriodicGridSettings(spacing=0.5, expansion_order=6, shift=(0.5, 0.5, 0.5), fortran_routines=True)
        grid = build_periodic_grid(cell, settings)
        self.assertIs(type(grid), PeriodicRealSpaceGrid)
        np.testing.assert_array_equal(grid.shift, 0.5)
        self.assertEqual(float(grid.coordinates[0, 0]), 0.25)
        default = build_periodic_grid(cell, PeriodicGridSettings(spacing=0.5, expansion_order=6, shift=(0.5, 0.5, 0.5)))
        self.assertEqual(float(default.coordinates[0, 0]), 0.25)
        np.testing.assert_array_equal(grid.coordinates, default.coordinates)
        unshifted = build_periodic_grid(cell, PeriodicGridSettings(spacing=0.5, expansion_order=6, fortran_routines=True))
        self.assertEqual(float(unshifted.coordinates[0, 0]), 0.0)

    def test_the_parser_gives_the_fortran_routines_fortrans_shift_and_counts(self) -> None:
        # Fortran's shift: 0.5 without Ignore_Symmetry, the Grid_Shift block with it (PARSEC 1.5)
        symmetric = self.parse(_bulk("Use_Fortran_Routines: .true.").replace("Ignore_Symmetry: .true.\n", ""))
        self.assertEqual(symmetric.problem.grid.shift, (0.5, 0.5, 0.5))
        shifted = self.parse(_bulk("Use_Fortran_Routines: .true.\nbegin Grid_Shift\n0.5 -0.25 0\nend Grid_Shift"))
        self.assertEqual(shifted.problem.grid.shift, (0.5, 0.25, 0.0))
        # the largest even count whose step is not below the spacing (6.0/0.55: 10, round(L/h) was 11)
        coarse = self.parse(_bulk("Use_Fortran_Routines: .true.").replace("Grid_Spacing: 0.5", "Grid_Spacing: 0.55"))
        grid = build_periodic_grid(coarse.problem.periodic_cell, coarse.problem.grid)
        self.assertIs(type(grid), PeriodicRealSpaceGrid)
        self.assertEqual(grid.points_per_axis.tolist(), [10, 10, 10])

    def test_a_bulk_cell_keeps_the_text_of_its_cell(self) -> None:
        cell = PeriodicCell(np.diag([6.0, 6.0, 6.0]))
        self.assertNotIn("dimensions", repr(cell))
        slab = PeriodicCell(np.diag([5.0, 5.0, 10.0]), dimensions=2)
        self.assertIn("dimensions=2", repr(slab))


class FallBackTests(_ParserCase):
    """Inputs the default routines do not compute run on the Fortran-derived ones, with the reason."""

    CASES = (
        ("wire", _SLAB.replace("Boundary_Conditions: slab", "Boundary_Conditions: wire").replace(
            "  5.0  0.0  0.0\n  0.0  5.0  0.0", "  7.0  0.0  0.0").format(extra="", periodic=".true."), "wire"),
        # a slab's default is the Fortran-derived routines: its refusals are those of Use_Fortran_Routines: .false.
        ("slab without Periodic_System", _slab(_HIS, periodic=".false."), "Periodic_System"),
        ("slab forces", _slab("Skip_Force: .false.\n" + _HIS), "of a slab"),
        ("slab relaxation", _slab("Minimization: bfgs\n" + _HIS), "of a slab"),
        ("slab k-points along z", _slab("Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid\n" + _HIS), "along z"),
        ("charged sheets", _slab(_SHEETS + "\n" + _HIS), "sheet"),
        # a weighted list of a bulk single point runs on his routines with .false. since 2026-11-08 (without the
        # parameter it is a default of the Fortran-derived ones, ThreeWayChoiceTests); its forces need theirs
        ("weighted k-points with forces", _bulk(_WEIGHTED + "\nSkip_Force: .false.\n" + _HIS), "unequal weights"),
        ("spin with k-points and forces", _bulk("Spin_Polarization: .true.\nKpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid\nSkip_Force: .false."), "spin-polarized cell with k-points"),
        ("spin PBE forces", _bulk("Spin_Polarization: .true.\nSkip_Force: .false.").replace("Correlation_Type: ca", "Correlation_Type: pbe"), "spin-polarized PBE"),
        ("Double_Grid_Order 2", _bulk("Double_Grid_Order: 2"), "Double_Grid_Order"),
        ("Super_Cell", _bulk("begin Super_Cell\n1 1 2\nend Super_Cell"), "super_cell"),
        ("Create_DOS", _bulk("Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid\nCreate_DOS: .true."), "create_dos"),
        ("PW91", _bulk().replace("Correlation_Type: ca", "Correlation_Type: pw91"), "pw91"),
        ("PL by default", _bulk("Spin_Polarization: .true.").replace("Correlation_Type: ca\n", ""), "Correlation_Type"),
        ("a grid of n <= 2M", _bulk().replace("Expansion_Order: 6", "Expansion_Order: 12"), "points"),
        ("Periodic_Local_Potential", _bulk("Periodic_Local_Potential: g_space"), "periodic_local_potential"),
        ("Zero_Average_Vxc", _bulk("Zero_Average_Vxc: .true."), "zero_average_vxc"),
        ("an atom outside the cell", _bulk(a="-0.4  3.0  3.1"), "outside the cell"),
        ("ARPACK", _bulk("Eigensolver: arpack"), "arpack"),
        ("DIAGLA", _bulk("Eigensolver: diagla"), "diagla"),
        ("Broyden mixing", _bulk("Mixing_Method: broyden"), "broyden"),
        ("Dynamic_Diag_Tol", _bulk("Dynamic_Diag_Tol: .true."), "Dynamic_Diag_Tol"),
        ("Old_Interpolation_Format", _bulk("Old_Interpolation_Format: .true."), "Old_Interpolation_Format"),
        ("Electric_Field", _bulk("Electric_Field: 0.001"), "Electric_Field"),
        ("Restart_Run", _bulk("Restart_Run: .true."), "Restart_Run"),
    )

    def test_each_entry_falls_back_with_its_reason(self) -> None:
        for name, text, reason in self.CASES:
            with self.subTest(name):
                translation = self.parse(text)
                self.assertTrue(translation.routines.fortran, name)
                self.assertIsNot(translation.routines.requested, True)
                self.assertIn(reason.lower(), "; ".join(translation.routines.reasons).lower())
                self.assertTrue(any("Fortran-derived routines" in warning for warning in translation.warnings))
                line = routines_report_line(translation)
                self.assertIn("chosen automatically", line)
                if translation.problem.periodic_cell is not None and hasattr(translation.problem.grid, "fortran_routines"):
                    self.assertTrue(translation.problem.grid.fortran_routines)

    def test_the_default_routines_keep_what_they_compute(self) -> None:
        for name, text in (
            ("Gamma", _bulk()),
            ("oblique cell", _bulk(cell=_FCC, a="0.0  0.0  0.0").replace("Expansion_Order: 6", "Expansion_Order: 4")),
            ("k-points", _bulk("Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid")),
            ("forces with k-points", _bulk(_MP222 + "\nSkip_Force: .false.")),
            ("Gamma forces with .false.", _bulk("Skip_Force: .false.\n" + _HIS)),
            ("spin forces with .false.", _bulk("Spin_Polarization: .true.\nSkip_Force: .false.\n" + _HIS)),
            ("slab", _slab(_HIS)),
            ("slab shifted grid of k-points", _slab("Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2\nend Monkhorst_Pack_Grid\nbegin Monkhorst_Pack_Shift\n0.5 0.5\nend Monkhorst_Pack_Shift\n" + _HIS)),
            # qian/slab_boundary's full mesh of equal weights also for a bulk cell (qian/pbc_forces refused the block)
            ("bulk shifted grid of k-points", _bulk(_SHIFTED_BULK + "\n" + _HIS)),
        ):
            with self.subTest(name):
                translation = self.parse(text)
                self.assertFalse(translation.routines.fortran, translation.routines.reasons)



class ThreeWayChoiceTests(_ParserCase):
    """Use_Fortran_Routines not given, .false. or .true. (the user's decision of 2 November 2026)."""

    def test_without_the_parameter_each_feature_takes_its_default(self) -> None:
        bulk = self.parse(_bulk()).routines
        self.assertFalse(bulk.fortran)
        self.assertIsNone(bulk.requested)
        for name, text, feature in (
            ("slab", _slab(), "slabs"),
            ("shifted grid of a bulk cell", _bulk(_SHIFTED_BULK), "shifted Monkhorst-Pack grids of bulk cells"),
            ("weighted k-points of a bulk cell", _bulk(_WEIGHTED), "Kpoint_Lists of unequal weights of bulk cells"),
        ):
            with self.subTest(name):
                translation = self.parse(text)
                choice = translation.routines
                self.assertTrue(choice.fortran)
                self.assertIsNone(choice.requested)
                self.assertTrue(choice.by_default)
                self.assertEqual(choice.defaults, (feature,))
                self.assertTrue(translation.problem.grid.fortran_routines)
                line = routines_report_line(translation)
                self.assertIn(f"Fortran-derived, the default for {feature}", line)
                self.assertIn("Use_Fortran_Routines: .false. selects Andrew Grieder's", line)
                # a chosen default, not a refusal: no warning
                self.assertIsNone(choice.warning())
                self.assertFalse(any("Fortran-derived routines" in warning for warning in translation.warnings))

    def test_false_selects_his_routines_where_his_code_accepts_the_input(self) -> None:
        for name, text in (
            ("bulk", _bulk(_HIS)),
            ("slab", _slab(_HIS)),
            ("shifted bulk grid", _bulk(_SHIFTED_BULK + "\n" + _HIS)),
            ("weighted k-points of a bulk cell", _bulk(_WEIGHTED + "\n" + _HIS)),
        ):
            with self.subTest(name):
                translation = self.parse(text)
                self.assertFalse(translation.routines.fortran)
                self.assertIs(translation.routines.requested, False)
                self.assertFalse(translation.problem.grid.fortran_routines)
                self.assertIn("Routines: Andrew Grieder's (Use_Fortran_Routines: .false.)", routines_report_line(translation))
        # an input his routines refuse still falls back, with the warning
        refused = self.parse(_slab("Skip_Force: .false.\n" + _HIS))
        self.assertTrue(refused.routines.fortran)
        self.assertIs(refused.routines.requested, False)
        self.assertIsNotNone(refused.routines.warning())

    def test_true_selects_the_fortran_derived_routines_everywhere(self) -> None:
        for name, text in (("bulk", _bulk()), ("slab", _slab()), ("shifted bulk grid", _bulk(_SHIFTED_BULK))):
            with self.subTest(name):
                translation = self.parse(text + "\nUse_Fortran_Routines: .true.\n")
                self.assertTrue(translation.routines.fortran)
                self.assertIs(translation.routines.requested, True)
                self.assertEqual(translation.routines.defaults, ())
                self.assertIn("Use_Fortran_Routines: .true.", routines_report_line(translation))

    def test_a_shifted_bulk_grid_is_fortrans_wedge_by_default_and_his_mesh_with_false(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling

        default = self.parse(_bulk(_SHIFTED_BULK))
        self.assertTrue(default.routines.fortran)
        self.assertEqual(tuple(default.problem.monkhorst_pack_dimensions), (2, 2, 2))
        # his routines reduce the mesh of a bulk single point too (2026-11-08): with Ignore_Symmetry by time
        # reversal alone, the 8 points (+-1/4)^3 in pairs; moving atoms keep the full mesh
        his = self.parse(_bulk(_SHIFTED_BULK + "\n" + _HIS))
        self.assertFalse(his.routines.fortran)
        self.assertEqual(tuple(his.problem.monkhorst_pack_dimensions), (2, 2, 2))
        self.assertEqual(his.problem.monkhorst_pack_shift, (0.5, 0.5, 0.5))
        self.assertIsNone(his.problem.space_group)
        k_points, weights = kpoint_sampling(his.problem)
        self.assertEqual(len(k_points), 4)
        np.testing.assert_allclose(weights, 0.25)
        moving = self.parse(_bulk(_SHIFTED_BULK + "\nSkip_Force: .false.\n" + _HIS)).problem
        self.assertIsNone(moving.monkhorst_pack_dimensions)
        self.assertEqual(len(moving.kpoints), 8)
        np.testing.assert_allclose(moving.kpoint_weights, 1.0 / 8.0)

    def test_a_slab_is_the_fortran_derived_slab_by_default_and_his_with_false(self) -> None:
        default = self.parse(_slab())
        self.assertTrue(getattr(_grid_of(default), "is_confined_periodic", False))
        his = self.parse(_slab(_HIS))
        self.assertEqual(his.problem.periodic_cell.dimensions, 2)
        self.assertIsInstance(_grid_of(his), SlabRealSpaceGrid)

    def test_a_label_his_routines_do_not_read_in_a_slab_is_the_slab_default(self) -> None:
        translation = self.parse(_slab(_SHEETS))
        self.assertTrue(translation.routines.by_default)
        self.assertIsNone(translation.routines.warning())
        self.assertIn("would not compute", routines_report_line(translation))


class BulkDefaultTests(_ParserCase):
    """The bulk features whose default is the Fortran-derived routines (the user's decision of 4 November 2026)."""

    def assert_default(self, translation, feature: str) -> None:
        choice = translation.routines
        self.assertTrue(choice.fortran)
        self.assertIsNone(choice.requested)
        self.assertEqual(choice.defaults, (feature,))
        self.assertIsNone(choice.warning())
        self.assertTrue(translation.problem.grid.fortran_routines)
        self.assertIn(f"Fortran-derived, the default for {feature}", routines_report_line(translation))

    def test_gamma_point_forces_relaxation_and_dynamics_are_the_fortran_derived_routines(self) -> None:
        for name, extra in (
            ("forces", "Skip_Force: .false."),
            ("relaxation", "Minimization: bfgs"),
            (
                "molecular dynamics",
                "Molecular_Dynamics: .true.\nCooling_Method: stair\nTinit: 300.0\nT_Final: 300.0\nT_Step: 0.0\n"
                "Time_Step: 40.0\nStep_Num: 2",
            ),
            ("spin-polarized forces", "Spin_Polarization: .true.\nSkip_Force: .false."),
        ):
            with self.subTest(name):
                extra = extra + "\nCubic_Spline: .true."  # the local potential of both sets of routines
                self.assert_default(self.parse(_bulk(extra)), GAMMA_FORCES_FEATURE)
                his = self.parse(_bulk(extra + "\n" + _HIS))
                self.assertFalse(his.routines.fortran)
                self.assertFalse(his.problem.grid.fortran_routines)
                self.assertEqual(self.parse(_bulk(extra + "\nUse_Fortran_Routines: .true.")).routines.defaults, ())

    def test_gamma_point_forces_stay_his_where_the_grids_differ(self) -> None:
        # the same run without forces is on his grid with his local potential: a run with forces keeps both, so the
        # energy does not move
        cube = _bulk("Skip_Force: .false.\nCubic_Spline: .true.")
        self.assert_default(self.parse(cube), GAMMA_FORCES_FEATURE)  # 12 points of 0.5 bohr from the origin in both
        # Since PARSEC 1.5's parity round both orthorhombic grids are Fortran's: without Ignore_Symmetry both are
        # half a step off the origin, and 6.0/0.55 gives 10 points (the largest even count) in both.
        for name, text in (
            ("a cube without Ignore_Symmetry", cube.replace("Ignore_Symmetry: .true.\n", "")),
            ("the largest even count", cube.replace("Grid_Spacing: 0.5", "Grid_Spacing: 0.55")),
            ("the same Grid_Shift", cube + "\nbegin Grid_Shift\n0.25 0.25 0.25\nend Grid_Shift"),
        ):
            with self.subTest(name):
                translation = self.parse(text)
                self.assert_default(translation, GAMMA_FORCES_FEATURE)
        for name, text in (
            # linear tables: his local potential is the cubic spline, theirs the linear table
            ("linear tables on the same grid", cube.replace("\nCubic_Spline: .true.", "")),
            # 9.6/0.4: 24 points in theirs (PARSEC 1.5's slack of adjust_step), 22 in his (1/24 is a rounding below
            # 0.4/9.6)
            ("other counts", _bulk("Skip_Force: .false.\nCubic_Spline: .true.", cell="  9.6  9.6  9.6").replace(
                "Grid_Spacing: 0.5", "Grid_Spacing: 0.4")),
            # one common spacing within 1e-6 in theirs, a step per side in his
            ("steps that differ in the last bits", _bulk("Skip_Force: .false.", cell="  6.0  6.0  6.0000001")),
        ):
            with self.subTest(name):
                translation = self.parse(text)
                self.assertFalse(translation.routines.fortran, translation.routines.describe())
                self.assertFalse(translation.problem.grid.fortran_routines)

    def test_gamma_point_forces_stay_his_where_his_path_is_faster(self) -> None:
        for name, text in (
            # the Fortran-derived routines would hand these to the reference solver, his stay on the accelerated path
            ("the Gamma point as a one-point grid", _bulk(_MP111 + "\nSkip_Force: .false.")),
            ("an oblique cell", _bulk("Skip_Force: .false.", cell=_FCC, a="0.0  0.0  0.0").replace(
                "Expansion_Order: 6", "Expansion_Order: 4")),
            # k-point forces: neither set is faster throughout (the user asked for the Gamma point)
            ("k-points", _bulk(_MP222 + "\nSkip_Force: .false.")),
            ("an energy without forces", _bulk()),
        ):
            with self.subTest(name):
                translation = self.parse(text)
                self.assertFalse(translation.routines.fortran, translation.routines.describe())
        # spinor orbitals: his forces are 30-150 times faster per evaluation
        cell = PeriodicCell(np.diag([6.0, 6.0, 6.0]))
        settings = PeriodicGridSettings(spacing=0.5, expansion_order=6)
        common = dict(
            kpoints=False, gamma_only=False, forces=True, spin_polarized=False, ignore_symmetry=True, search_info=1,
            spline_species=True,
        )
        self.assertEqual(bulk_default_features((), cell, settings, spinors=False, **common), (GAMMA_FORCES_FEATURE,))
        self.assertEqual(bulk_default_features((), cell, settings, spinors=True, **common), ())

    def test_a_space_group_that_sets_fortrans_grid_is_the_fortran_derived_routines(self) -> None:
        translation = self.parse(_DIAMOND.format(extra=_MP222))
        self.assert_default(translation, SYMMETRIC_GRID_FEATURE)
        self.assertEqual(tuple(_grid_of(translation).points_per_axis), (8, 8, 8))  # Fortran's grid
        his = self.parse(_DIAMOND.format(extra=_MP222 + "\n" + _HIS))
        self.assertFalse(his.routines.fortran)
        self.assertEqual(tuple(_grid_of(his).points_per_axis), (10, 10, 10))

    def test_a_space_group_stays_his_where_his_path_is_faster_or_the_grid_is_the_same(self) -> None:
        for name, extra in (
            ("Gamma point (his accelerated oblique path)", ""),
            ("Gamma point as a one-point grid", _MP111),
            ("forces (the readable local force on Fortran's general grid)", _MP222 + "\nSkip_Force: .false."),
            ("Ignore_Symmetry", _MP222 + "\nIgnore_Symmetry: .true."),
        ):
            with self.subTest(name):
                translation = self.parse(_DIAMOND.format(extra=extra))
                self.assertFalse(translation.routines.fortran, translation.routines.describe())
        # a spacing whose largest even count is a multiple of four: both grids are the same
        same = self.parse(_DIAMOND.format(extra=_MP222).replace("Grid_Spacing: 0.42", "Grid_Spacing: 0.5"))
        self.assertFalse(same.routines.fortran, same.routines.describe())


class NoOpLabelTests(_ParserCase):
    """Labels whose value does nothing in a periodic input choose no routines (since 2026-11-05).

    PARSEC 1.5's bak_complete_parsec.in writes ``potential_field F`` and
    ``recenter_atoms_in_aperiodic_directions T`` in every run; before the fix either line moved a bulk input from
    Andrew Grieder's routines to the Fortran-derived ones (examples/3d_Si: +1.34 mRy).
    """

    NO_OPS = (
        "Potential_Field: .false.",
        "Write_Potential_Field: .false.",
        "AFM_Scan: .false.",
        "Kinetic_Energy_Functional: tf",
        "FDET_Kinetic_Form: parsec1.5",
        "Recenter_Atoms_In_Aperiodic_Directions: .true.",
        # .false. keeps the input's coordinates, which Andrew Grieder's routines do for a bulk cell anyway.
        "Recenter_Atoms_In_Aperiodic_Directions: .false.",
    )

    def test_no_op_values_keep_the_default_routines_and_the_problem(self) -> None:
        # The second atom lies at 3.1 of 6.0 along y: the start-up back_to_cell of the Fortran-derived routines
        # would move it to -2.9, Andrew Grieder's routines keep it.
        base = self.parse(_bulk())
        self.assertFalse(base.routines.fortran)
        for line in self.NO_OPS:
            with self.subTest(line=line):
                translation = self.parse(_bulk(line))
                self.assertFalse(translation.routines.fortran)
                self.assertEqual(repr(translation.problem), repr(base.problem))
                self.assertFalse(any("Fortran-derived" in warning for warning in translation.warnings))

    def test_every_fdet_label_is_neutral(self) -> None:
        from parsec_python.Input.parsec_input import FDET_BLOCKS, FDET_KEYWORDS
        from parsec_python.Input.routines import NEUTRAL_BLOCK_KEYS, NEUTRAL_KEYS, SMALL15_SHARED_KEYS

        self.assertLessEqual(set(FDET_KEYWORDS), NEUTRAL_KEYS)
        self.assertLessEqual(set(FDET_BLOCKS), NEUTRAL_BLOCK_KEYS)
        self.assertIn("recenter_atoms_in_aperiodic_directions", SMALL15_SHARED_KEYS)

    def test_fdet_itself_is_still_refused_for_a_periodic_input(self) -> None:
        from parsec_python.Input.parsec_input import UnsupportedParsecOptionError

        with self.assertRaisesRegex(UnsupportedParsecOptionError, "periodic"):
            self.parse(_bulk("Potential_Field: .true."))
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "clusters only"):
            self.parse(_bulk("Write_Potential_Field: .true."))

    def test_use_fortran_routines_still_honours_the_recentring(self) -> None:
        moved = self.parse(_bulk("Use_Fortran_Routines: .true."))
        kept = self.parse(_bulk("Use_Fortran_Routines: .true.\nRecenter_Atoms_In_Aperiodic_Directions: .false."))
        self.assertTrue(moved.routines.fortran and kept.routines.fortran)
        self.assertNotEqual(repr(moved.problem.atoms), repr(kept.problem.atoms))
        # .false. keeps the coordinates of the input, as Andrew Grieder's routines do.
        self.assertEqual(repr(kept.problem.atoms), repr(self.parse(_bulk()).problem.atoms))
        # An atom outside the cell, which his routines refuse, runs on the Fortran-derived ones for that reason;
        # .false. keeps it where the input puts it there.
        outside = self.parse(_bulk("Recenter_Atoms_In_Aperiodic_Directions: .false.", a="3.0  3.0  6.5"))
        self.assertTrue(outside.routines.fortran)
        self.assertTrue(any("outside the cell" in reason for reason in outside.routines.reasons))
        self.assertEqual(outside.problem.atoms[0].position.tolist(), [3.0, 3.0, 6.5])


def _state(translation) -> dict:
    """What a run takes from a translation: every field but its notes, path, bak text and timing (the record of the
    output labels written, ``OutputFileSettings.keywords``, prints nothing at the defaults)."""

    import dataclasses

    state = {}
    for field in dataclasses.fields(translation):
        if field.name in {"source", "warnings", "complete_input", "domain_rule_seconds"}:
            continue
        value = getattr(translation, field.name)
        if field.name == "output_files" and value is not None:
            value = dataclasses.replace(value, keywords=frozenset())
        state[field.name] = repr(value)
    return state


class ValueAwareRoutineTests(_ParserCase):
    """A label written at a value that does nothing chooses no routines (routines.NO_EFFECT; the default since
    2026-11-05).  The pairs are those of the sweep of every scalar line of PARSEC
    1.5's reference bak_complete_parsec.in files (and this port's own) on a bulk cell of Andrew Grieder's routines."""

    #: (label, value) that do nothing: the bak lines of 1.5 at their defaults, and the labels read only with another.
    NO_OPS = (
        ("add_charged_sheets", ".false."), ("add_charged_sheets", "F"), ("num_charged_sheets", "0"),
        ("add_point_charges", ".false."), ("add_point_charges", "F"), ("point_typ_num", "0"),
        ("alpha_filter", "0.000000000000000E+000"), ("beta1_filter", "0.000000000000000E+000"),
        ("core_filter", "0.000000000000000E+000"),
        ("calc_hartree_g", ".false."), ("create_dos", "F"), ("dos_pnum", "1000"), ("dos_pnum", "500"),
        ("ylm_dos_l", "-1"), ("ylm_dos_l", "2"),
        ("external_mag_field", "0.000000000000000E+000"),
        ("hf_one", "0"), ("hf_two", "0"), ("hf_three", "0"), ("hf_one", "2"), ("van_der_waals", "F"),
        ("ldaplusu_u", "0.000000000000000E+000"), ("ldaplusu_j", "0.000000000000000E+000"),
        ("mixing_en_like", "0"), ("mixing_group_size", "1"), ("mixing_memory_expand_factor", "2.00000000000000"),
        ("mixing_preferred_type", "1"), ("mixing_restart_factor", "0.300000000000000"),
        ("nscf_fermi_level", "0.000000000000000E+000"), ("nscf_kpoint_unit", "cartesian_inverse_bohr"),
        ("nscf_states_num", "16"),
        ("orb_mag", ".false."), ("orb_mag", ".true."), ("output_gw", "F"), ("polarizability", "F"),
        ("restart_from_wfndat", "F"), ("restart_from_wfndat", ".true."),
        ("symmetrize_charge_density", "F"),
        ("output_data", "T"), ("output_eigenvalues", "F"), ("save_intermediate_eigenvalues", ".false."),
        ("save_intermediate_charge_density", "F"), ("output_file_name", "parsec.out"), ("output_grid_data", "none"),
    )
    #: (label, value) that act: they keep choosing the Fortran-derived routines, and the warning names the value.
    ACTING = (
        ("calc_hartree_g", ".true."), ("mixing_restart_factor", "0.1"), ("mixing_group_size", "2"),
        ("core_filter", "0.5"), ("symmetrize_charge_density", "T"), ("output_data", ".false."),
        ("output_eigenvalues", "T"), ("save_intermediate_charge_density", "T"), ("save_intermediate_eigenvalues", "T"),
        ("output_file_name", "other.out"),
    )

    def test_every_no_op_value_keeps_the_default_routines_and_the_run(self) -> None:
        from parsec_python.Input.routines import NO_EFFECT

        base = self.parse(_bulk())
        self.assertFalse(base.routines.fortran)
        covered = set()
        for label, value in self.NO_OPS:
            with self.subTest(label=label, value=value):
                translation = self.parse(_bulk() + f"\n{label} {value}\n")
                self.assertFalse(translation.routines.fortran, translation.routines.reasons)
                self.assertEqual(_state(translation), _state(base))
                covered.add(label)
        # every entry of the table is checked here
        self.assertEqual(covered, set(NO_EFFECT))

    def test_every_acting_value_still_chooses_the_fortran_derived_routines(self) -> None:
        for label, value in self.ACTING:
            with self.subTest(label=label, value=value):
                translation = self.parse(_bulk() + f"\n{label} {value}\n")
                self.assertTrue(translation.routines.fortran)
                self.assertTrue(
                    any(f"{label} = {value}" in reason for reason in translation.routines.reasons),
                    translation.routines.reasons,
                )

    def test_a_no_op_value_with_its_switch_on_acts(self) -> None:
        # DOS_Pnum is read with Create_DOS, Restart_From_Wfndat with Restart_Run: then the value counts.
        from parsec_python.Input.routines import has_no_effect

        class Item:
            def __init__(self, value):
                self.value = value

        self.assertTrue(has_no_effect("dos_pnum", {"dos_pnum": [Item("500")]}, {}))
        self.assertFalse(has_no_effect("dos_pnum", {"dos_pnum": [Item("500")], "create_dos": [Item("T")]}, {}))
        self.assertTrue(has_no_effect("dos_pnum", {"dos_pnum": [Item("1000")], "create_dos": [Item("T")]}, {}))
        self.assertFalse(
            has_no_effect(
                "restart_from_wfndat", {"restart_from_wfndat": [Item("T")], "restart_run": [Item(".true.")]}, {}
            )
        )
        self.assertFalse(has_no_effect("nscf_states_num", {"nscf_states_num": [Item("8")]}, {"nscf_kpoints": [1]}))
        # species labels: every species at zero
        self.assertFalse(has_no_effect("ldaplusu_u", {"ldaplusu_u": [Item("0"), Item("4.0 eV")]}, {}))
        self.assertFalse(has_no_effect("core_filter", {"core_filter": [Item("0")], "alpha_filter": [Item("2")]}, {}))
        # a value the predicate does not understand acts
        self.assertFalse(has_no_effect("add_point_charges", {"add_point_charges": [Item("maybe")]}, {}))

    def test_output_eigenvalues_with_kpoints_and_the_super_cell_blocks(self) -> None:
        # With k-points eigen.dat is written anyway: Output_Eigenvalues true changes nothing (PARSEC 1.5 writes
        # "output_eigenvalues T" in the bak file of every k-point run).  At the Gamma point it acts.
        kpoints = _MP222
        base = self.parse(_bulk(kpoints))
        written = self.parse(_bulk(kpoints) + "\nOutput_Eigenvalues: .true.\n")
        self.assertEqual(written.routines.fortran, base.routines.fortran)
        self.assertEqual(_state(written), _state(base))
        self.assertTrue(self.parse(_bulk() + "\nOutput_Eigenvalues: .true.\n").routines.fortran)
        # Super_Cell 1 1 1 and Super_Cell_Vac 0 0 0 leave the problem as it is, but parsec.out then prints the
        # replication lines of PARSEC and bak_complete_parsec.in expands its k-point lists: they are not in
        # NO_EFFECT and keep choosing the Fortran-derived routines.
        cell = self.parse(_bulk() + "\nbegin Super_Cell\n1 1 1\nend Super_Cell\n")
        self.assertTrue(cell.routines.fortran)
        fortran = self.parse(_bulk("Use_Fortran_Routines: .true."))
        cell_fortran = self.parse(_bulk("Use_Fortran_Routines: .true.") + "\nbegin Super_Cell\n1 1 1\nend Super_Cell\n")
        self.assertEqual(repr(cell_fortran.problem), repr(fortran.problem))
        self.assertEqual(cell_fortran.super_cell, (1, 1, 1))

    def test_a_bak_complete_parsec_in_of_parsec15_keeps_the_routines_of_its_input(self) -> None:
        # The lines PARSEC 1.5 writes for every bulk run, at their values of a real run (TierII/3d_Fe_bcc).
        lines = (
            "add_point_charges F\nadd_charged_sheets F\ncalc_hartree_g F\ncreate_dos F\ndos_pnum 1000\n"
            "external_mag_field 0.000000000000000E+000\nhf_one 0\nhf_two 0\nhf_three 0\nmixing_en_like 0\n"
            "mixing_group_size 1\nmixing_memory_expand_factor 2.00000000000000\nmixing_preferred_type 1\n"
            "mixing_restart_factor 0.300000000000000\norb_mag F\noutput_gw F\npolarizability F\n"
            "potential_field F\nrecenter_atoms_in_aperiodic_directions T\nrestart_from_wfndat F\n"
            "symmetrize_charge_density F\nvan_der_waals F\noutput_data T\noutput_eigenvalues F\n"
            "save_intermediate_charge_density F\nsave_intermediate_eigenvalues F\n"
            "alpha_filter 0.000000000000000E+000\nbeta1_filter 0.000000000000000E+000\n"
            "core_filter 0.000000000000000E+000\nldaplusu_u 0.000000000000000E+000\nldaplusu_j 0.000000000000000E+000"
        )
        base = self.parse(_bulk())
        bak = self.parse(_bulk() + "\n" + lines + "\n")
        self.assertFalse(bak.routines.fortran, bak.routines.reasons)
        self.assertEqual(_state(bak), _state(base))


def _grid_of(translation):
    """The grid of a periodic translation (a wire or slab of the Fortran-derived routines builds its own)."""
    from parsec_python.SCF.pbc import prepare_periodic_single_point

    return prepare_periodic_single_point(translation.problem).grid


class DefectFixTests(_ParserCase):
    def test_slab_lattice_vector_coordinates_keep_z_cartesian(self) -> None:
        # qian/slab_boundary multiplied the third fractional coordinate by the box height 2R
        text = _slab().replace("Grid_Spacing: 0.45", "Grid_Spacing: 0.45\nCoordinate_Unit: lattice_vectors").replace(
            "  0.10  -0.10   0.65\n -0.40  -0.20  -0.75", "  0.10  0.20   0.35\n  0.30  0.40  -0.35"
        )
        translation = self.parse(text + "\n" + _HIS)  # his slab (the default slab is the Fortran-derived one)
        self.assertFalse(translation.routines.fortran)
        positions = np.array([atom.position for atom in translation.problem.atoms])
        np.testing.assert_allclose(positions[:, 2], [0.35, -0.35])  # bohr, not 0.35 * 2R
        np.testing.assert_allclose(positions[0, :2], [0.5, 1.0])

    def test_spin_pbe_on_an_oblique_grid_of_the_default_routines(self) -> None:
        from parsec_python.V_xc import pbe_spin, pbe_spin_polarized

        grid = build_periodic_grid(
            PeriodicCell(4.0 * (np.ones((3, 3)) - np.eye(3))),
            PeriodicGridSettings(spacing=0.45, expansion_order=6, points_rule="fortran"),
        )
        self.assertFalse(grid.is_orthorhombic)
        centre = grid.coordinates.mean(axis=0)
        r2 = np.sum((grid.coordinates - centre) ** 2, axis=1)
        up = 0.02 * np.exp(-r2 / 6.0) + 0.001
        down = 0.015 * np.exp(-r2 / 5.0) + 0.001
        analytic = pbe_spin(up, down, grid)
        reference = pbe_spin_polarized(up, down, grid)
        # the two forms of spin PBE agree to round-off of their derivatives (3e-3 Ry before the fix)
        self.assertAlmostEqual(analytic.total_energy, reference.total_energy, delta=1.0e-9)

    def test_parsec_dat_carries_the_shift_of_his_grid(self) -> None:
        # a Restart_Run or NSCF (Fortran-derived routines) then refuses the file of a shifted default grid
        from parsec_python.Output.parsec_dat import grid_shift

        cell = PeriodicCell(np.diag([6.0, 6.0, 6.0]))
        default = build_periodic_grid(cell, PeriodicGridSettings(spacing=0.5, expansion_order=6, shift=(0.5, 0.5, 0.5)))
        np.testing.assert_array_equal(grid_shift(default), [0.5, 0.5, 0.5])
        fortran = build_periodic_grid(cell, PeriodicGridSettings(spacing=0.5, expansion_order=6, fortran_routines=True))
        np.testing.assert_array_equal(grid_shift(fortran), [0.0, 0.0, 0.0])
        shifted = build_periodic_grid(
            cell, PeriodicGridSettings(spacing=0.5, expansion_order=6, shift=(0.5, 0.5, 0.5), fortran_routines=True)
        )
        np.testing.assert_array_equal(grid_shift(shifted), [0.5, 0.5, 0.5])

    def test_a_slab_grid_is_his_slab_grid(self) -> None:
        translation = self.parse(_slab(_HIS))
        cell = translation.problem.periodic_cell
        self.assertEqual(cell.dimensions, 2)
        grid = build_periodic_grid(cell, translation.problem.grid)
        self.assertIsInstance(grid, SlabRealSpaceGrid)


if __name__ == "__main__":
    unittest.main()
