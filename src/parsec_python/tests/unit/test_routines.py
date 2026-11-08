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

    def test_the_orthorhombic_grid_of_the_fortran_routines_ignores_the_shift(self) -> None:
        cell = PeriodicCell(np.diag([6.0, 6.0, 6.0]))
        settings = PeriodicGridSettings(spacing=0.5, expansion_order=6, shift=(0.5, 0.5, 0.5), fortran_routines=True)
        grid = build_periodic_grid(cell, settings)
        self.assertIs(type(grid), PeriodicRealSpaceGrid)
        np.testing.assert_array_equal(grid.shift, 0.0)
        self.assertEqual(float(grid.coordinates[0, 0]), 0.0)
        default = build_periodic_grid(cell, PeriodicGridSettings(spacing=0.5, expansion_order=6, shift=(0.5, 0.5, 0.5)))
        self.assertEqual(float(default.coordinates[0, 0]), 0.25)

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
        ("weighted k-points", _bulk("Kpoint_Method: manual\nbegin Kpoint_List\n0.0 0.0 0.0 1.0\n0.25 0.0 0.0 3.0\nend Kpoint_List"), "unequal weights"),
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
        default = self.parse(_bulk(_SHIFTED_BULK)).problem
        self.assertEqual(tuple(default.monkhorst_pack_dimensions), (2, 2, 2))
        his = self.parse(_bulk(_SHIFTED_BULK + "\n" + _HIS)).problem
        self.assertIsNone(his.monkhorst_pack_dimensions)
        self.assertEqual(len(his.kpoints), 8)
        np.testing.assert_allclose(his.kpoint_weights, 1.0 / 8.0)

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
        for name, text in (
            # linear tables: his local potential is the cubic spline, theirs the linear table
            ("linear tables on the same grid", cube.replace("\nCubic_Spline: .true.", "")),
            # without Ignore_Symmetry his grid is half a step off the origin, Fortran's grid
            ("a cube without Ignore_Symmetry", cube.replace("Ignore_Symmetry: .true.\n", "")),
            # 6.0/0.55: 11 points in theirs, 10 (the largest even count) in his
            ("other counts", cube.replace("Grid_Spacing: 0.5", "Grid_Spacing: 0.55")),
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

    def test_a_slab_grid_is_his_slab_grid(self) -> None:
        translation = self.parse(_slab(_HIS))
        cell = translation.problem.periodic_cell
        self.assertEqual(cell.dimensions, 2)
        grid = build_periodic_grid(cell, translation.problem.grid)
        self.assertIsInstance(grid, SlabRealSpaceGrid)


if __name__ == "__main__":
    unittest.main()
