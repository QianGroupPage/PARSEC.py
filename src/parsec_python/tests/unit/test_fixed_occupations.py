"""Occupations read from ``occup.in`` (a negative ``Fermi_Temp``).

``initial.f90`` reads the file and ``flevel.f90`` assigns its values to the
states in ascending order of energy.  The files of the Fortran example
``examples/tests/relaxation`` and of a cation with a hole spread by hand are
read here and compared with the occupations Fortran PARSEC printed; the
end-to-end energies are compared in ``test_sina4_parsec_reference``.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python.Input import (
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
    summarize_translation,
)
from parsec_python.Occupations import (
    check_fixed_occupations,
    fermi_occupations,
    fixed_occupations,
    read_occupation_file,
)
from parsec_python.SCF import prepare_single_point
from parsec_python.models import SCFSettings
from parsec_python.tests.martins_old_reference import DATA, load_cases, write_case


class OccupationFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.path = Path(self._directory.name) / "occup.in"

    def read(self, text: str, states: int = 8):
        self.path.write_text(text)
        return read_occupation_file(self.path, states)

    def test_file_of_the_fortran_example(self) -> None:
        """Without spin polarization PARSEC reads the "spin up" section only."""

        result = read_occupation_file(DATA / "sina4_occup.in", 8)
        self.assertEqual(result.occupations, (1.0,) * 4 + (0.0,) * 4)
        # The printed occupations of the Fortran run.
        self.assertEqual(list(result.occupations), load_cases()["sina4"]["occupations"])
        # The "spin down" section: title, two numbers and two occupations.
        self.assertEqual(result.unread_lines, 5)
        self.assertEqual(result.source, DATA / "sina4_occup.in")

    def test_occupations_set_by_hand(self) -> None:
        result = read_occupation_file(DATA / "sina4_cation_occup.in", 8)
        self.assertEqual(
            result.occupations, (1.0,) + (0.833333333333,) * 3 + (0.0,) * 4
        )
        self.assertEqual(result.unread_lines, 0)
        printed = load_cases()["sina4_cation_fractional"]["occupations"]
        np.testing.assert_allclose(result.occupations, printed, rtol=0, atol=5.01e-5)
        # The second section of the example, as a first section.
        lines = (DATA / "sina4_occup.in").read_text().splitlines()[3:]
        half = self.read("\n".join(lines) + "\n")
        self.assertEqual(half.occupations, (1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0))
        # A hand-set value may also empty a state below the full ones.
        emptied = self.read("title\n 4\n 2\n 2 0.0\n 6 1.0d0\n")
        self.assertEqual(emptied.occupations, (1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0))

    def test_list_directed_records(self) -> None:
        # Text after the numbers of a record is not read, blank records are
        # passed, and a record may hold both numbers of a pair or one each.
        result = self.read(
            "any title 7 7 7\n\n  2   full levels\n\n 2 , by hand\n 3 0.25 first\n 4\n 0.75\n"
            "next section\n 0\n 0\n",
            states=5,
        )
        self.assertEqual(result.occupations, (1.0, 1.0, 0.25, 0.75, 0.0))
        self.assertEqual(result.unread_lines, 3)
        # The title is one record whatever it holds, also an empty one.
        self.assertEqual(self.read("\n 1\n 0\n", states=2).occupations, (1.0, 0.0))
        # No state may be occupied at all.
        self.assertEqual(self.read("t\n 0\n 0\n", states=2).occupations, (0.0, 0.0))

    def test_refusals(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be read"):
            read_occupation_file(self.path.with_name("absent.in"), 8)
        with self.assertRaisesRegex(ValueError, "is empty"):
            self.read("")
        with self.assertRaisesRegex(ValueError, "ends before the number of fully occupied"):
            self.read("title\n")
        with self.assertRaisesRegex(ValueError, "ends before the number of occupations set"):
            self.read("title\n 4\n")
        with self.assertRaisesRegex(ValueError, "ends before an occupation set by hand"):
            self.read("title\n 4\n 2\n 4 0.5\n")
        # initial.f90 would write beyond its occupation array.
        with self.assertRaisesRegex(ValueError, "9 fully occupied states do not fit"):
            self.read("title\n 9\n 0\n")
        with self.assertRaisesRegex(ValueError, "do not fit"):
            self.read("title\n -1\n 0\n")
        for index in (0, 9):
            with self.assertRaisesRegex(ValueError, "lies outside 1..8"):
                self.read(f"title\n 4\n 1\n {index} 0.5\n")
        with self.assertRaisesRegex(ValueError, "expected an integer"):
            self.read("title\n 4.0\n 0\n")
        with self.assertRaisesRegex(ValueError, "expected an integer"):
            self.read("title\n # 4\n 0\n")
        with self.assertRaisesRegex(ValueError, "expected an occupation for state 4"):
            self.read("title\n 3\n 1\n 4 half\n")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            read_occupation_file(DATA / "sina4_occup.in", 0)


class FixedOccupationTests(unittest.TestCase):
    def test_states_take_the_occupations_in_order_of_energy(self) -> None:
        eigenvalues = np.array([-1.0, -0.5, -0.5, -0.2, 0.3])
        given = (1.0, 0.75, 0.75, 0.5, 0.0)
        result = fixed_occupations(eigenvalues, 6.0, given)
        np.testing.assert_array_equal(result.occupations, given)
        self.assertEqual(result.electron_count, 6.0)
        # No Fermi level is searched (flevel.f90 prints none).
        self.assertTrue(np.isnan(result.fermi_level))
        # A Fermi level at 80 K would fill the three lowest states instead.
        np.testing.assert_allclose(
            fermi_occupations(eigenvalues, 6.0, 80.0).occupations, [1, 1, 1, 0, 0], atol=1e-12
        )
        # The values are copied, not shared with the settings.
        result.occupations[0] = 0.0
        self.assertEqual(given[0], 1.0)

    def test_refusals(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be sorted"):
            fixed_occupations(np.array([0.0, -1.0]), 2.0, (1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "3 fixed occupations were given for 2"):
            fixed_occupations(np.array([-1.0, 0.0]), 2.0, (1.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            fixed_occupations(np.array([]), 0.0, ())

    def test_electron_count_tolerance_of_initial_f90(self) -> None:
        # |2*sum - electrons| > 1e-5 stops Fortran with error 233.
        check_fixed_occupations((1.0, 1.0, 1.0, 1.0, 0.0), 8.0)
        check_fixed_occupations((1.0, 0.833333333333, 0.833333333333, 0.833333333333), 7.0)
        check_fixed_occupations((1.0, 1.0), 4.0 + 0.9e-5)
        with self.assertRaisesRegex(ValueError, "hold 4.000000 electrons, but the system has 4.000011"):
            check_fixed_occupations((1.0, 1.0), 4.0 + 1.1e-5)
        with self.assertRaisesRegex(ValueError, "hold 8.000000 electrons"):
            fixed_occupations(np.arange(4.0), 7.0, (1.0, 1.0, 1.0, 1.0))

    def test_scf_settings(self) -> None:
        settings = SCFSettings(
            fermi_temperature_kelvin=-1.0,
            number_of_states=3,
            fixed_occupations=[1, 0.5, 0],
        )
        self.assertEqual(settings.fixed_occupations, (1.0, 0.5, 0.0))
        # The occupations decide, not the sign of the temperature.
        SCFSettings(number_of_states=2, fixed_occupations=(1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "give fixed_occupations"):
            SCFSettings(fermi_temperature_kelvin=-1.0, number_of_states=3)
        with self.assertRaisesRegex(ValueError, "one finite value per state"):
            SCFSettings(number_of_states=3, fixed_occupations=(1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "one finite value per state"):
            SCFSettings(fixed_occupations=(1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "one finite value per state"):
            SCFSettings(number_of_states=2, fixed_occupations=(1.0, float("nan")))
        self.assertIsNone(SCFSettings().fixed_occupations)


class NegativeFermiTemperatureInputTests(unittest.TestCase):
    """``Fermi_Temp < 0`` in ``parsec.in`` reads ``occup.in`` beside it."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        self.cases = load_cases()

    def test_occupations_enter_the_problem(self) -> None:
        translation = parse_parsec_input(write_case(self.cases["sina4"], self.directory))
        scf = translation.problem.scf
        self.assertEqual(scf.fermi_temperature_kelvin, -1.0)
        self.assertEqual(scf.fixed_occupations, (1.0,) * 4 + (0.0,) * 4)
        self.assertTrue(
            any("occup.in: only the first section is read" in text for text in translation.warnings)
        )
        self.assertIn("occupations fixed by occup.in", summarize_translation(translation))
        system = prepare_single_point(translation.problem)
        self.assertEqual(system.electron_count, 8.0)

    def test_cation_with_a_hole_spread_by_hand(self) -> None:
        translation = parse_parsec_input(
            write_case(self.cases["sina4_cation_fractional"], self.directory)
        )
        self.assertEqual(translation.problem.scf.net_charge, 1.0)
        self.assertEqual(
            translation.problem.scf.fixed_occupations,
            (1.0,) + (0.833333333333,) * 3 + (0.0,) * 4,
        )
        self.assertFalse(any("occup.in" in text for text in translation.warnings))
        self.assertEqual(prepare_single_point(translation.problem).electron_count, 7.0)

    def test_file_must_exist_and_hold_the_electrons(self) -> None:
        path = write_case(self.cases["sina4"], self.directory)
        (self.directory / "occup.in").unlink()
        # initial.f90 error 236.
        with self.assertRaisesRegex(ParsecInputError, "occup.in"):
            parse_parsec_input(path)
        # initial.f90 error 233: three full levels for eight electrons.
        (self.directory / "occup.in").write_text("title\n 3\n 0\n")
        translation = parse_parsec_input(path)
        with self.assertRaisesRegex(ValueError, "hold 6.000000 electrons, but the system has 8"):
            prepare_single_point(translation.problem)
        # More full levels than States_Num.
        (self.directory / "occup.in").write_text("title\n 9\n 0\n")
        with self.assertRaisesRegex(ParsecInputError, "do not fit States_Num = 8"):
            parse_parsec_input(path)

    def test_positive_temperature_reads_no_file(self) -> None:
        case = dict(self.cases["sina4"])
        case["input"] = [
            "Fermi_Temp: 80" if line.startswith("Fermi_Temp") else line
            for line in case["input"]
        ]
        path = write_case(case, self.directory)
        (self.directory / "occup.in").unlink()
        translation = parse_parsec_input(path)
        self.assertIsNone(translation.problem.scf.fixed_occupations)
        self.assertIn("T=80 K", summarize_translation(translation))

    def test_spin_polarization_reads_both_sections(self) -> None:
        """With two spin channels ``initial.f90`` reads a section for each."""

        case = dict(self.cases["sina4"])
        case["input"] = list(case["input"]) + ["Spin_Polarization: .true."]
        translation = parse_parsec_input(write_case(case, self.directory))
        scf = translation.problem.scf
        self.assertTrue(scf.spin_polarized)
        self.assertEqual(scf.fixed_occupations, (1.0,) * 4 + (0.0,) * 4)
        # The "spin down" section of the example spreads one electron over
        # the levels 4 and 5.
        self.assertEqual(
            scf.fixed_occupations_down, (1.0,) * 3 + (0.5, 0.5) + (0.0,) * 3
        )
        self.assertFalse(any("occup.in" in text for text in translation.warnings))
        # One electron per state: the two sections hold the eight electrons.
        self.assertEqual(prepare_single_point(translation.problem).electron_count, 8.0)

    def test_spin_polarization_needs_both_sections(self) -> None:
        case = dict(self.cases["sina4"])
        case["input"] = list(case["input"]) + ["Spin_Polarization: .true."]
        path = write_case(case, self.directory)
        (self.directory / "occup.in").write_text("spin up\n 4\n 0\n")
        with self.assertRaisesRegex(ParsecInputError, "section of spin channel 2"):
            parse_parsec_input(path)
        # initial.f90 error 233: the two sections hold six of eight electrons.
        (self.directory / "occup.in").write_text("up\n 4\n 0\ndown\n 2\n 0\n")
        translation = parse_parsec_input(path)
        with self.assertRaisesRegex(ValueError, "hold 6.000000 electrons"):
            prepare_single_point(translation.problem)


if __name__ == "__main__":
    unittest.main()
