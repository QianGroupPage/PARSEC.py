"""The output-file keywords of PARSEC and Restart_Run through both command lines.

A small H2 cluster is run once, which writes ``parsec.dat``, ``eigen.dat``
and a cube file beside its input as PARSEC does, and then restarted from that
``parsec.dat`` (``Restart_Run``), by the reference and the accelerated
command line.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from parsec_python.Input import ParsecInputError, parse_parsec_input
from parsec_python.Input.parsec_input import UnsupportedParsecOptionError
from parsec_python.models import GRID_DATA_FIELDS
from parsec_python.Output.parsec_dat import read_parsec_dat

DATA = Path(__file__).resolve().parents[1] / "data"

H2_INPUT = """\
Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere
Boundary_Sphere_Radius: 5.0 bohr
Grid_Spacing: 0.6 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 4
Hartree_Boundary_Tolerance: off
States_Num: 3
Fermi_Temp: 500
Max_Iter: 40
Convergence_Criterion: 1.0e-5
Eigensolver: chebff
FF_Maxiter: 9
Chebdav_Degree: 40
Mixing_Method: Anderson
Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
    0.0  0.0  -0.7
    0.0  0.0   0.7
end Atom_Coord
Correlation_Type: ca
{extra}
"""


def _write_case(directory: Path, extra: str = "") -> Path:
    shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
    path = directory / "parsec.in"
    path.write_text(H2_INPUT.format(extra=extra), encoding="utf-8")
    return path


def _final_energy(log: Path) -> float:
    values = re.findall(r"Total Energy =\s+(-?\d+\.\d+) \[Ry\]", log.read_text(encoding="utf-8"))
    return float(values[-1])


def _iterations(log: Path) -> int:
    return len(re.findall(r"SCF iter #", log.read_text(encoding="utf-8")))


class OutputKeywordTests(unittest.TestCase):
    def parse(self, extra: str):
        with tempfile.TemporaryDirectory() as directory:
            return parse_parsec_input(_write_case(Path(directory), extra))

    def test_defaults_follow_usrinputfile(self) -> None:
        translation = self.parse("")
        files = translation.output_files
        self.assertTrue(files.output_data)
        self.assertEqual(files.wavefunctions.mode, "occupied")
        self.assertFalse(files.wavefunctions.explicit)
        self.assertFalse(files.eigenvalues)
        self.assertEqual(files.grid_data, ())
        self.assertEqual(files.output_file_name, "parsec.out")
        self.assertEqual(files.keywords, frozenset())
        self.assertFalse(translation.restart_run)
        self.assertTrue(translation.problem.recenter_geometry)

    def test_save_wfn_bands_forms(self) -> None:
        for block, expected in (
            ("", ()),
            ("2 5", (2, 3, 4, 5)),
            ("3", (3,)),
            ("1\n4\n2", (1, 4, 2)),
            ("1, 3", (1, 2, 3)),
        ):
            with self.subTest(block=block):
                translation = self.parse(f"begin Save_Wfn_Bands\n{block}\nend Save_Wfn_Bands")
                selection = translation.output_files.wavefunctions
                self.assertEqual(selection.mode, "list")
                self.assertEqual(selection.indices, expected)
                self.assertTrue(selection.explicit)
        with self.assertRaisesRegex(ParsecInputError, "Save_Wfn_Bands"):
            self.parse("begin Save_Wfn_Bands\n1\nx\nend Save_Wfn_Bands")
        # Output_All_States wins over the block, as in PARSEC.
        translation = self.parse(
            "Output_All_States: .true.\nbegin Save_Wfn_Bands\n2\nend Save_Wfn_Bands"
        )
        self.assertEqual(translation.output_files.wavefunctions.mode, "all")
        self.assertTrue(translation.output_files.all_states)

    def test_output_grid_data_words(self) -> None:
        self.assertEqual(self.parse("Output_Grid_Data: all").output_files.grid_data, GRID_DATA_FIELDS)
        self.assertEqual(
            self.parse("Output_Grid_Data: V_Hartree, charge_density").output_files.grid_data,
            ("v_hartree", "charge_density"),
        )
        self.assertEqual(
            self.parse("Output_Grid_Data: v_external+v_hartree, none, v_hartree").output_files.grid_data,
            ("v_external+v_hartree",),
        )
        with self.assertRaisesRegex(ParsecInputError, "Wrong input to output_grid_data: density"):
            self.parse("Output_Grid_Data: density")

    def test_other_keywords_and_refusals(self) -> None:
        translation = self.parse(
            "Output_Data: .false.\nOutput_Eigenvalues: .true.\n"
            "Save_Intermediate_Eigenvalues: .true.\nOutput_File_Name: 'h2.out'"
        )
        files = translation.output_files
        self.assertFalse(files.output_data)
        self.assertTrue(files.eigenvalues and files.intermediate_eigenvalues)
        self.assertFalse(files.writes_eigen_dat)
        self.assertEqual(files.output_file_name, "h2.out")
        restarted = self.parse("Restart_Run: .true.")
        self.assertTrue(restarted.restart_run)
        self.assertFalse(restarted.problem.recenter_geometry)
        # Restart_From_Wfndat is read with Restart_Run only (wfn.dat restart).
        self.assertFalse(self.parse("Restart_From_Wfndat: .true.").restart_from_wfndat)
        self.assertTrue(
            self.parse("Restart_Run: .true.\nRestart_From_Wfndat: .true.").restart_from_wfndat
        )
        for extra, message in (
            ("Output_GW: .true.", "Output_GW"),
            ("Restart_Run: .true.\nInitial_Density: file\nML_Density_File: x.npz", "Initial_Density"),
        ):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(UnsupportedParsecOptionError, message):
                    self.parse(extra)

    def test_restart_names_coordinates_that_a_first_run_would_centre(self) -> None:
        translation = self.parse("Restart_Run: .true.").warnings
        self.assertEqual(translation, ())
        with tempfile.TemporaryDirectory() as directory:
            path = _write_case(Path(directory), "Restart_Run: .true.")
            path.write_text(path.read_text().replace("0.0  0.0   0.7", "0.0  0.0   1.7"))
            warnings = parse_parsec_input(path).warnings
        self.assertTrue(any("Restart_Run: the atom coordinates" in item for item in warnings))


class CommandLineTests(unittest.TestCase):
    def run_cli(self, main, path: Path, *extra: str) -> int:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as error:
            code = main([str(path), "--quiet", *extra])
        self.last_error = error.getvalue()
        return code

    def check_first_run_and_restart(self, main, *extra: str) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write_case(
                directory,
                "Output_Eigenvalues: .true.\nOutput_Grid_Data: charge_density\n"
                "Output_File_Name: first.out",
            )
            self.assertEqual(self.run_cli(main, path, *extra), 0, self.last_error)
            log = directory / "first.out"
            self.assertTrue(log.is_file())
            for name in ("parsec.dat", "eigen.dat", "charge_den.cube"):
                self.assertTrue((directory / name).is_file(), name)
            text = log.read_text(encoding="utf-8")
            self.assertIn(" Generating Eigenvalue Output File!", text)
            self.assertIn(" parsec.dat: ", text)
            data = read_parsec_dat(directory / "parsec.dat")
            # The occupied states: the bonding state, and the next one where
            # the Fermi level leaves it an occupation above zero.
            saved = data.wavefunctions[0][0].state_numbers.tolist()
            self.assertEqual(saved[0], 1)
            self.assertEqual(saved, list(range(1, len(saved) + 1)))
            with np.load(directory / "parsec_python_results.npz") as archive:
                np.testing.assert_array_equal(
                    data.potentials[0], archive["next_effective_potential_ry"]
                )
                np.testing.assert_array_equal(data.densities[0], archive["density_e_per_bohr3"])
            eigen = (directory / "eigen.dat").read_text().splitlines()
            self.assertEqual(len(eigen), 1 + len(saved))
            first_energy = _final_energy(log)

            path.write_text(path.read_text() + "\nRestart_Run: .true.\n")
            self.assertEqual(
                self.run_cli(main, path, "--log", "restart.out", *extra), 0, self.last_error
            )
            restart_log = directory / "restart.out"
            text = restart_log.read_text(encoding="utf-8")
            self.assertIn(" Restarted Run - from parsec.dat", text)
            self.assertIn(" Restarting from previous run", text)
            self.assertAlmostEqual(_final_energy(restart_log), first_energy, places=6)
            # The stored potential is converged and the first solve is tight.
            self.assertLessEqual(_iterations(restart_log), 2)

    def test_reference_command_line(self) -> None:
        from parsec_python.cli import main

        self.check_first_run_and_restart(main)

    def test_accelerated_command_line(self) -> None:
        from parsec_python.acceleration.cli import main

        self.check_first_run_and_restart(main, "--backend", "scipy", "--device", "cpu")

    def test_missing_restart_file_and_protected_outputs(self) -> None:
        from parsec_python.cli import main

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write_case(directory, "Restart_Run: .true.")
            self.assertEqual(self.run_cli(main, path), 2)
            self.assertIn("PARSEC error 371", self.last_error)
            path = _write_case(directory, "Output_File_Name: parsec.dat")
            self.assertEqual(self.run_cli(main, path), 2)
            self.assertIn("refusing to overwrite", self.last_error)


if __name__ == "__main__":
    unittest.main()
