"""PARSEC 1.5 keywords of the forces and ionic loops (parity round of 2026-11-04).

``Calculate_Force`` (``usrinputfile.F90:4151``), ``Use_Old_Forces``
(``parsec.F90:2436,2477``), ``Write_Relax_Restart`` (``relaxation/domove.f90:390``),
``Recenter_Atoms_In_Aperiodic_Directions`` (``usrinputfile.F90:3653-3657``,
``scf/initial.f90:196-327``) and ``Mol_RNG_Seed`` (``usrinputfile.F90:4741``),
read as the parser translates them.  This port keeps two of its defaults
(forces on request, relax_restart.dat written); the keyword
written in the input gives PARSEC 1.5's behaviour.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.Input.parsec_input import ParsecInputError, UnsupportedParsecOptionError

DATA = Path(__file__).resolve().parents[1] / "data"

#: A water molecule whose bounding box is 0.4034 bohr off the origin along y
#: (tests/unit/test_restart_coordinates.py).
WATER = """\
Boundary_Conditions: cluster
Boundary_Sphere_Radius: 10.4 bohr
Grid_Spacing: 0.57 bohr
Coordinate_Unit: Cartesian_Ang
States_Num: 8
Atom_Types_Num: 2
Atom_Type: H
Local_Component: s
begin Atom_Coord
   0.000000  -0.488456   0.950000
   0.000000  -0.488456  -0.950000
end Atom_Coord
Atom_Type: O
Local_Component: s
begin Atom_Coord
   0.000000   0.061544   0.000000
end Atom_Coord
Correlation_Type: ca
Ignore_Symmetry: .true.
{extra}
"""


class _InputCase(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(scratch.cleanup)
        self.directory = Path(scratch.name)
        shutil.copy(DATA / "water_H_POTRE.DAT", self.directory / "H_POTRE.DAT")
        shutil.copy(DATA / "water_O_POTRE.DAT", self.directory / "O_POTRE.DAT")

    def parse(self, extra: str = ""):
        path = self.directory / "parsec.in"
        path.write_text(WATER.format(extra=extra), encoding="utf-8")
        return parse_parsec_input(path)


class CalculateForceTests(_InputCase):
    def test_forces_of_a_single_point_on_request(self) -> None:
        # This port's default (kept on 2026-11-04): no forces for a single point;
        # 1.5 computes them in every run (Calculate_Force defaults to .true.).
        self.assertFalse(self.parse().compute_forces)
        self.assertTrue(self.parse("Calculate_Force: .true.").compute_forces)
        self.assertFalse(self.parse("Calculate_Force: .false.").compute_forces)
        self.assertTrue(self.parse("Skip_Force: .false.").compute_forces)
        # agreeing pairs are accepted
        self.assertTrue(self.parse("Calculate_Force: .true.\nSkip_Force: .false.").compute_forces)
        self.assertFalse(self.parse("Calculate_Force: .false.\nSkip_Force: .true.").compute_forces)

    def test_contradictions_and_unimplemented_combinations_are_refused(self) -> None:
        for extra in ("Calculate_Force: .true.\nSkip_Force: .true.",
                      "Calculate_Force: .false.\nSkip_Force: .false."):
            with self.subTest(extra=extra), self.assertRaises(ParsecInputError):
                self.parse(extra)
        # 1.5 would move the atoms with the ion-ion forces alone
        for extra in ("Minimization: bfgs", "Molecular_Dynamics: .true."):
            with self.subTest(extra=extra), self.assertRaises(UnsupportedParsecOptionError):
                self.parse(extra + "\nCalculate_Force: .false.")
        self.assertTrue(self.parse("Minimization: bfgs\nCalculate_Force: .true.").compute_forces)

    def test_use_old_forces_is_accepted_and_changes_nothing(self) -> None:
        plain = self.parse("Calculate_Force: .true.")
        old = self.parse("Calculate_Force: .true.\nUse_Old_Forces: .true.")
        self.assertEqual(repr(old.problem), repr(plain.problem))
        self.assertTrue(any("Use_Old_Forces" in note for note in old.warnings))
        self.assertFalse(any("Use_Old_Forces" in note for note in self.parse("Use_Old_Forces: .false.").warnings))


class WriteRelaxRestartTests(_InputCase):
    def test_default_writes_the_file_and_false_does_not(self) -> None:
        default = self.parse("Minimization: bfgs").relaxation
        self.assertTrue(default.write_restart)
        self.assertNotIn("write_restart", repr(default))
        self.assertTrue(self.parse("Minimization: bfgs\nWrite_Relax_Restart: .true.").relaxation.write_restart)
        off = self.parse("Minimization: bfgs\nWrite_Relax_Restart: .false.").relaxation
        self.assertFalse(off.write_restart)
        self.assertIn("write_restart=False", repr(off))
        single = self.parse("Write_Relax_Restart: .false.")
        self.assertIsNone(single.relaxation)
        self.assertTrue(any("Write_Relax_Restart" in note for note in single.warnings))


class RelaxRestartFormatTests(unittest.TestCase):
    def test_the_file_of_parsec15_is_read_and_written_back_unchanged(self) -> None:
        # relaxation/domove.f90:385-432 of 1.5: the layout of the previous tree
        # (isave(1:44) then iwa(1:9n) where it had one loop over both).
        import json

        from parsec_python.Relaxation.files import format_relax_restart, parse_relax_restart

        data = json.loads((DATA / "relax_restart_parsec15_h2_bfgs.json").read_text(encoding="utf-8"))
        text = data["relax_restart_dat"]
        restart = parse_relax_restart(text, 2)
        self.assertEqual((restart.iteration, restart.movable_atoms), (2, 2))
        self.assertIsNotNone(restart.lbfgsb)
        self.assertEqual(format_relax_restart(restart), text)


class RecenteringTests(_InputCase):
    def test_false_keeps_the_input_coordinates(self) -> None:
        bohr = 1.0 / 0.529177
        given = np.array([[0.0, -0.488456, 0.95], [0.0, -0.488456, -0.95], [0.0, 0.061544, 0.0]]) * bohr
        default = self.parse()
        self.assertTrue(default.recenter_aperiodic)
        self.assertTrue(default.problem.recenter_geometry)
        kept = self.parse("Recenter_Atoms_In_Aperiodic_Directions: .false.")
        self.assertFalse(kept.recenter_aperiodic)
        self.assertFalse(kept.problem.recenter_geometry)
        np.testing.assert_allclose([atom.position for atom in kept.problem.atoms], given, rtol=0, atol=1e-12)
        self.assertTrue(self.parse("Recenter_Atoms_In_Aperiodic_Directions: .true.").problem.recenter_geometry)

    def test_a_molecular_dynamics_restart_keeps_them_too(self) -> None:
        from parsec_python.MolecularDynamics.run import restart_structure

        state = (
            "Step #     3\n"
            + "".join(
                "".join(f" {value:15.8E}" for value in row) + "\n"
                for row in (
                    [0.0, -0.9, 1.8, 0.0, -0.9, 1.8, 0, 0, 0, 0, 0, 0, 1.0],
                    [0.0, -0.9, -1.8, 0.0, -0.9, -1.8, 0, 0, 0, 0, 0, 0, 1.0],
                    [0.0, 0.2, 0.0, 0.0, 0.2, 0.0, 0, 0, 0, 0, 0, 0, 1.0],
                )
            )
        )
        (self.directory / "mdinit.dat").write_text(state, encoding="utf-8")
        for extra, recentred in (("", True), ("\nRecenter_Atoms_In_Aperiodic_Directions: .false.", False)):
            with self.subTest(extra=extra):
                translation = self.parse("Molecular_Dynamics: .true.\nRestart_mode: .true." + extra)
                moved, restart, _ = restart_structure(translation, self.directory)
                self.assertEqual(restart.frame, 3)
                self.assertEqual(moved.problem.recenter_geometry, recentred)


class RoutineChoiceTests(unittest.TestCase):
    def test_shared_keywords_keep_the_default_routines(self) -> None:
        from parsec_python.Input.routines import decide_routines

        # Since 2026-11-05 also the re-centring switch: .true. is the
        # default, and .false. keeps the input's coordinates, which Andrew Grieder's routines do for a
        # bulk cell anyway (tests/unit/test_routines.py NoOpLabelTests).
        for key in (
            "calculate_force",
            "use_old_forces",
            "write_relax_restart",
            "bands_ff_maxiter",
            "recenter_atoms_in_aperiodic_directions",
        ):
            with self.subTest(key=key):
                self.assertFalse(decide_routines({key: [None]}, {}).fortran)


class MolRngSeedTests(_InputCase):
    def test_the_new_name_of_rng_seed(self) -> None:
        md = "Molecular_Dynamics: .true.\nStep_num: 2\n"
        self.assertEqual(self.parse(md + "Mol_RNG_Seed: 7").molecular_dynamics.rng_seed, 7)
        self.assertEqual(self.parse(md + "rng_seed: 7").molecular_dynamics.rng_seed, 7)
        self.assertEqual(self.parse(md).molecular_dynamics.rng_seed, 0)
        with self.assertRaises(ParsecInputError):
            self.parse(md + "Mol_RNG_Seed: 7\nrng_seed: 7")


if __name__ == "__main__":
    unittest.main()
