"""Which symmetry operations reach PARSEC's ``adjust_step``, and the grid when none is commensurate.

``symmetries.f90`` / ``symgen.f90`` / ``inipbc.f90:327-429`` as the parser applies them
(:func:`parsec_python.Input.parsec_input._periodic_symmetry`, :func:`parsec_python.Grid.pbc.adjust_step`):

* a translation no grid is commensurate with (fcc Si2, the second atom at 0.27 on the three-fold
  axis: C3v, symmetry on, h 0.6): Fortran keeps its 12 operations; the previous Fortran tree
  kept the step at the input spacing (12 x 0.600 bohr in a 7.2549 bohr cell), PARSEC 1.5 and
  this port take nint(L/h) = 12 points with the step L/12 = 0.605 bohr, the grid of Fortran's own
  run of the input with Ignore_Symmetry;
* operations that do not form a group modulo the lattice (``symchk``, run at the Gamma point on a
  grid shifted by 0.5): Fortran keeps the identity alone (the diamond cube stretched to 15.39 bohr
  along z: 20 x 20 x 30 points, not the 28 planes of the (1/4, 1/4, 1/4) translation);
* molecular dynamics and manual movements: no search (diamond fcc at h 0.5: 14^3, not the 12^3
  of a single point).

Fortran numbers: runs of the Fortran binary on these inputs (Fortran parity round),
with ``tests/data/Si_POTRE.DAT``.
"""


# With both sets of routines: the periodic inputs of this module set Use_Fortran_Routines: .true. (the
# Fortran-derived routines); the numbers checked here are those of the Fortran parity round.
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Grid.pbc import adjust_step, build_periodic_grid
from parsec_python.Input.parsec_input import UnsupportedParsecOptionError, parse_parsec_input

_SILICON = Path(__file__).resolve().parents[1] / "data" / "Si_POTRE.DAT"

_FCC_SI2 = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
Grid_Spacing: {spacing}
Expansion_Order: 6
Ignore_Symmetry: .false.
Fermi_Temp: 300.0
Max_Iter: 80
Mixing_Method: Anderson
Mixing_Param: 0.3
begin Cell_Shape
  0.0  0.5  0.5
  0.5  0.0  0.5
  0.5  0.5  0.0
end Cell_Shape
Lattice_Vector_Scale: 10.26 bohr
Coordinate_Unit: lattice_vectors
States_Num: 8
Correlation_Type: ca
Atom_Types_Num: 1
Atom_Type: Si
Pseudopotential_Format: Martins_new
Local_Component: p
begin Atom_Coord
  0.00  0.00  0.00
  {second}  {second}  {second}
end Atom_Coord
Skip_force: .false.
Convergence_Criterion: 1.0e-8
{extra}
"""

_STRETCHED_DIAMOND = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  10.26  10.26  15.39
end Cell_Shape
Coordinate_Unit: lattice_vectors
Grid_Spacing: 0.5
Expansion_Order: 6
States_Num: 24
Correlation_Type: ca
Atom_Types_Num: 1
Atom_Type: Si
Pseudopotential_Format: Martins_new
Local_Component: p
begin Atom_Coord
  0.00  0.00  0.00
  0.00  0.50  0.50
  0.50  0.00  0.50
  0.50  0.50  0.00
  0.25  0.25  0.25
  0.25  0.75  0.75
  0.75  0.25  0.75
  0.75  0.75  0.25
end Atom_Coord
Ignore_Symmetry: .false.
{extra}
"""

_FORTRAN_EMULATION = (
    "Periodic_Local_Potential: g_space\nPeriodic_Alpha_Term: fortran\nZero_Average_Vxc: .true."
)


class _InputCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        shutil.copy(_SILICON, self.directory / "Si_POTRE.DAT")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def parse(self, text: str):
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        return parse_parsec_input(path)


class AdjustStepWithoutCommensurateStepTests(unittest.TestCase):
    def test_parsecs_count_with_the_step_that_spans_the_cell(self) -> None:
        length = 10.26 / np.sqrt(2.0)
        # 0.27 divides no even count near L/h: PARSEC 1.5's nint(L/h) points, step L/n.
        self.assertEqual(adjust_step(length, 0.6, translations=(0.27, 0.27)), (12, length / 12))
        # Fortran's nint, not the largest even count below L/h: L/h = 13.5 gives 14.
        self.assertEqual(adjust_step(8.1, 0.6, translations=(0.27,)), (14, 8.1 / 14))
        # A commensurate translation still decides; no translation keeps the plain rule.
        for translations, count in (((0.25,), 12), ((), 14)):
            points, step = adjust_step(length, 0.5, translations=translations)
            self.assertEqual(points, count)
            self.assertAlmostEqual(step, length / count, delta=1e-15)
        with self.assertRaises(ValueError):
            adjust_step(0.5, 0.6, translations=(0.27,))
        with self.assertRaises(ValueError):
            adjust_step(1.0, 0.6)


class IncommensurateTranslationTests(_InputCase):
    def test_c3v_fcc_si2_gets_parsecs_twelve_points(self) -> None:
        translation = self.parse(_FCC_SI2.format(spacing=0.6, second=0.27, extra=""))
        problem = translation.problem
        grid = build_periodic_grid(problem.periodic_cell, problem.grid)
        self.assertEqual(grid.points_per_axis.tolist(), [12, 12, 12])
        np.testing.assert_allclose(grid.spacings, 10.26 / np.sqrt(2.0) / 12.0, rtol=0, atol=1e-15)
        # The six operations with the 0.27 translation stay in Fortran's group (12 operations).
        self.assertEqual(len(problem.grid.fractional_translations[0]), 6)
        self.assertTrue(
            any("0.27000 0.27000 0.27000" in note and "nint(L/h) points" in note for note in translation.warnings)
        )
        # The density cannot be averaged over operations that do not map the grid onto itself.
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "no grid spacing is commensurate"):
            self.parse(
                _FCC_SI2.format(spacing=0.6, second=0.27, extra="Symmetrize_Charge_Density: .true.")
            )

    def test_c3v_fcc_si2_gives_the_energy_and_forces_of_parsecs_twelve_point_grid(self) -> None:
        from parsec_python.Forces import compute_forces
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.SCF.single_point import run_scf

        problem = self.parse(
            _FCC_SI2.format(spacing=0.6, second=0.27, extra=_FORTRAN_EMULATION)
        ).problem
        system = prepare_periodic_single_point(problem)
        result = run_scf(system)
        self.assertTrue(result.converged)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_FORCE_SLOPES": "tabulated"}):
            forces = compute_forces(system, result).total
        # Fortran with Ignore_Symmetry (12^3, step 0.605): -14.65875932 Ry and 0.101997 Ry/bohr
        # along (1,1,1) on atom 1; this port is 1.1e-6 Ry and 4e-7 Ry/bohr from it.  Fortran with
        # the symmetry on keeps the step 0.600 (-14.65161709 Ry, 0.095996 Ry/bohr).
        self.assertAlmostEqual(result.energies.total, -14.65875932, delta=3.0e-6)
        np.testing.assert_allclose(forces[0], [0.101997] * 3, rtol=0, atol=2.0e-6)


class SymmetryCheckTests(_InputCase):
    def test_stretched_diamond_keeps_the_identity_and_fortrans_thirty_planes(self) -> None:
        from parsec_python.Grid.symmetry_analysis import symmetry_check
        from parsec_python.Grid.space_group import find_space_group

        translation = self.parse(_STRETCHED_DIAMOND.format(extra=""))
        problem = translation.problem
        # 16 operations, eight with (1/4, 1/4, 1/4): not a group modulo the lattice.
        group = find_space_group(problem.periodic_cell.lattice_vectors, problem.atoms)
        self.assertEqual(group.order, 16)
        self.assertTrue(symmetry_check(group).error)
        self.assertFalse(any(problem.grid.fractional_translations))
        grid = build_periodic_grid(problem.periodic_cell, problem.grid)
        self.assertEqual(grid.points_per_axis.tolist(), [20, 20, 30])  # Fortran; 28 planes before
        # one step along the three sides: the orthorhombic grid of Fortran's counts (PARSEC 1.5's parity round;
        # Fortran's general grid before, the same points)
        np.testing.assert_allclose(grid.spacing, 0.513, rtol=0, atol=1e-12)
        np.testing.assert_allclose(grid.shift, 0.5)
        # Symmetrize_Charge_Density averages over the identity alone, as in Fortran.
        symmetrized = self.parse(_STRETCHED_DIAMOND.format(extra="Symmetrize_Charge_Density: .true."))
        self.assertIsNone(symmetrized.problem.space_group)
        self.assertTrue(symmetrized.problem.scf.symmetrize_density)
        # With k-points symchk is not run (info = 1): the whole group, its translations, 28 planes.
        kpoints = self.parse(
            _STRETCHED_DIAMOND.format(
                extra="Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid"
            )
        )
        grid = build_periodic_grid(kpoints.problem.periodic_cell, kpoints.problem.grid)
        self.assertEqual(grid.points_per_axis.tolist(), [20, 20, 28])

    def test_molecular_dynamics_searches_no_symmetry(self) -> None:
        dynamics = (
            "Molecular_Dynamics: .true.\nCooling_Method: stair\nTinit: 300.0\nT_Final: 300.0\n"
            "T_Step: 0.0\nTime_Step: 20.0\nStep_num: 1"
        )
        for extra, points in (("", 12), (dynamics, 14), ("Minimization: manual", 14)):
            with self.subTest(extra=extra):
                problem = self.parse(_FCC_SI2.format(spacing=0.5, second=0.25, extra=extra)).problem
                grid = build_periodic_grid(problem.periodic_cell, problem.grid)
                self.assertEqual(grid.points_per_axis.tolist(), [points] * 3)  # Fortran 12^3 / 14^3


if __name__ == "__main__":
    unittest.main()
