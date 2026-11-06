"""The FDET keywords of PARSEC 1.5 (``usrinputfile.F90:2190-2232``) and the PARSEC.py switches."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from parsec_python.Input import ParsecInputError, parse_parsec_input
from parsec_python.Input.parsec_input import UnsupportedParsecOptionError

DATA = Path(__file__).resolve().parents[1] / "data"


def _write(
    directory: Path,
    extra: str = "",
    radius: str = "Boundary_Sphere_Radius: 4.0 bohr",
    symmetry: str = "Ignore_Symmetry: true",
) -> Path:
    text = f"""\
{radius}
Grid_Spacing: 0.8 bohr
Expansion_Order: 4
States_Num: 2
{symmetry}
Hartree_Boundary_Tolerance: off
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.2  0.0  1.0
  0.2  0.0  2.4
end Atom_Coord
{extra}
"""
    path = directory / "parsec.in"
    path.write_text(text, encoding="utf-8")
    (directory / "H_POTRE.DAT").write_bytes((DATA / "H_POTRE.DAT").read_bytes())
    return path


_CHARGES = """
Add_Point_Charges: true
Point_Typ_Num: 1
Pt_Chg: 1.0 e
begin Point_Coord
 -0.7 0.0 0.0
  0.7 0.0 0.0
end Point_Coord
"""


class FDETInputTest(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.directory = Path(self._folder.name)

    def tearDown(self):
        self._folder.cleanup()

    def parse(self, extra="", **keywords):
        return parse_parsec_input(_write(self.directory, extra, **keywords))

    def test_without_fdet_nothing_changes(self):
        translation = self.parse()
        self.assertIsNone(translation.problem.embedding)
        self.assertNotIn("embedding", repr(translation.problem))
        self.assertFalse(translation.write_potential_field)
        self.assertTrue(translation.problem.recenter_geometry)

    def test_keywords(self):
        translation = self.parse(
            "Potential_Field: true\nPotential_Field_Name: sample.dat\n"
            "Kinetic_Energy_Functional: PB\npbe_m: 0.25\npbe_k: 0.9\n" + _CHARGES
        )
        settings = translation.problem.embedding
        self.assertEqual(settings.field_path, (self.directory / "sample.dat").resolve())
        self.assertEqual((settings.kinetic_functional, settings.pbe_mu, settings.pbe_kappa), ("pb", 0.25, 0.9))
        self.assertEqual(settings.kinetic_form, "corrected")
        self.assertFalse(settings.tip_sphere)
        self.assertIn("embedding", repr(translation.problem))
        # never recentred, no symmetry
        self.assertFalse(translation.problem.recenter_geometry)
        self.assertFalse(translation.use_symmetry)
        self.assertTrue(any("not recentred" in warning for warning in translation.warnings))

    def test_never_recentred_whatever_the_recentring_keyword(self):
        # Recenter_Atoms_In_Aperiodic_Directions is a keyword of PARSEC 1.5; under FDET it changes nothing.
        from parsec_python.Input.parsec_input import FDET_KEYWORDS

        self.assertNotIn("recenter_atoms_in_aperiodic_directions", FDET_KEYWORDS)
        for value in ("true", "false"):
            translation = self.parse(
                f"Potential_Field: true\nRecenter_Atoms_In_Aperiodic_Directions: {value}\n" + _CHARGES
            )
            with self.subTest(value=value):
                self.assertFalse(translation.problem.recenter_geometry)
                self.assertFalse(translation.recenter_aperiodic)
        self.assertTrue(self.parse("Recenter_Atoms_In_Aperiodic_Directions: true\n").problem.recenter_geometry)

    def test_tip_sphere_needs_an_unshifted_grid(self):
        # fdet_tipsphere.F90:217-220 (error 622) tests the grid shift itself (the Grid_Shift block).
        sphere = "Potential_Field: true\nBoundary_Sphere_Around_Tip: true\n" + _CHARGES
        with self.assertRaisesRegex(ParsecInputError, "622"):
            self.parse(sphere + "begin Grid_Shift\n 0.25 0.0 0.5\nend Grid_Shift\n")
        with self.assertRaisesRegex(ParsecInputError, "622"):
            self.parse(sphere, symmetry="")
        zero = self.parse(sphere + "begin Grid_Shift\n 0 0 0\nend Grid_Shift\n", symmetry="")
        self.assertTrue(zero.problem.embedding.tip_sphere)
        folded = self.parse(sphere + "Foldblock_Side_Length 2\n")
        self.assertEqual(folded.problem.grid.foldblock_side_length, 2)
        # the classical mode takes any shift (the field of the sample must have the same, checked when it is read)
        shifted = self.parse("Potential_Field: true\nbegin Grid_Shift\n 0.25 0.0 0.5\nend Grid_Shift\n" + _CHARGES)
        self.assertEqual(shifted.problem.grid.shift, (0.25, 0.0, 0.5))

    def test_defaults_and_fortran_fall_back(self):
        settings = self.parse("Potential_Field: true\n" + _CHARGES).problem.embedding
        self.assertEqual((settings.field_path.name, settings.kinetic_functional), ("pot.dat", "tf"))
        self.assertEqual((settings.pbe_mu, settings.pbe_kappa), (0.2319, 0.8438))
        self.assertEqual(
            self.parse("Potential_Field: true\nKinetic_Energy_Functional: no\n" + _CHARGES).problem.embedding.kinetic_functional,
            "none",
        )
        # 1.5 knows pw, pb, tf, tw and no; any other word is TF ("pw91k" and "none" too).
        for word in ("pw91k", "none"):
            translation = self.parse(f"Potential_Field: true\nKinetic_Energy_Functional: {word}\n" + _CHARGES)
            self.assertEqual(translation.problem.embedding.kinetic_functional, "tf")
            self.assertTrue(any("Thomas-Fermi" in warning for warning in translation.warnings))
        form = self.parse(
            "Potential_Field: true\nKinetic_Energy_Functional: pw\nFDET_Kinetic_Form: parsec1.5\n" + _CHARGES
        ).problem.embedding.kinetic_form
        self.assertEqual(form, "parsec1.5")
        with self.assertRaises(ParsecInputError):
            self.parse("Potential_Field: true\nFDET_Kinetic_Form: other\n" + _CHARGES)

    def test_warnings_without_potential_field(self):
        translation = self.parse("Kinetic_Energy_Functional: pw\n")
        self.assertIsNone(translation.problem.embedding)
        self.assertTrue(any("without Potential_Field" in warning for warning in translation.warnings))
        warned = self.parse("Potential_Field: true\n")
        self.assertTrue(any("Add_Point_Charges" in warning for warning in warned.warnings))

    def test_refused_combinations(self):
        for extra in (
            "Spin_Polarization: true\n",
            "Correlation_Type: pb\nVan_Der_Waals: true\n",
            "Polarizability: true\n",
        ):
            with self.subTest(extra=extra), self.assertRaises(UnsupportedParsecOptionError):
                self.parse("Potential_Field: true\n" + _CHARGES + extra)
        with self.assertRaises(UnsupportedParsecOptionError):
            # (with point charges the point group is off anyway and the keyword changes nothing)
            self.parse("Potential_Field: true\nSymmetrize_Charge_Density: true\n", symmetry="")
        with self.assertRaises(ParsecInputError):
            self.parse("Potential_Field: true\n" + _CHARGES, radius="")

    def test_complete_input_reads_back_to_the_same_problem(self):
        # bak_complete_parsec.in (Generate_Complete_Parsec_In) with every FDET keyword
        import warnings

        from parsec_python.Input.complete_input import COMPLETE_INPUT_NAME
        from parsec_python.Output.complete_input import write_complete_input

        for extra in (
            "Potential_Field: true\nPotential_Field_Name: sample.dat\nKinetic_Energy_Functional: pb\n"
            "pbe_m: 0.25\npbe_k: 0.9\nFDET_Kinetic_Form: parsec1.5\nWrite_Potential_Field: true\n",
            "Potential_Field: true\nPotential_Field_Name: sample.dat\nKinetic_Energy_Functional: no\n",
            "Potential_Field: true\nPotential_Field_Name: sample.dat\nKinetic_Energy_Functional: pw\n"
            "Boundary_Sphere_Around_Tip: true\nAFM_Scan: true\nAFM_Scan_Heights: 5\nAFM_Scan_Height_Step: 0.8 bohr\n"
            "begin AFM_Scan_Lateral_Points\n 0.0 0.0\n 0.8 0.0\n 0.8 0.8\nend AFM_Scan_Lateral_Points\n"
            "AFM_Scan_Line_Range: 2 3\n",
        ):
            with self.subTest(extra=extra.splitlines()[2]):
                translation = self.parse("Generate_Complete_Parsec_In: .true.\n" + extra + _CHARGES)
                write_complete_input(translation)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    again = parse_parsec_input(self.directory / COMPLETE_INPUT_NAME)
                self.assertEqual(repr(again.problem), repr(translation.problem))
                self.assertEqual(again.write_potential_field, translation.write_potential_field)
                self.assertEqual(again.afm_scan, translation.afm_scan)
                self.assertEqual(again.relaxation, translation.relaxation)

    def test_write_potential_field(self):
        translation = self.parse("Write_Potential_Field: true\n")
        self.assertTrue(translation.write_potential_field)
        self.assertIsNone(translation.problem.embedding)
        # The run would overwrite the sample file it reads (PARSEC 1.5 does).
        with self.assertRaises(ParsecInputError):
            self.parse("Write_Potential_Field: true\nPotential_Field: true\n" + _CHARGES)
        both = self.parse("Write_Potential_Field: true\nPotential_Field: true\nPotential_Field_Name: sample.dat\n" + _CHARGES)
        self.assertTrue(both.write_potential_field)
        with self.assertRaises(UnsupportedParsecOptionError):
            self.parse("Write_Potential_Field: true\nSpin_Polarization: true\n")


if __name__ == "__main__":
    unittest.main()
