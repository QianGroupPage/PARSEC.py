"""End-to-end CLI tests for self-consistent spin-orbit coupling."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.cli import main as cli_main

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)

_BASE_INPUT = """\
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 20
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 5.0e-3

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

SO_from_scratch: .true.

Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Begin atom_coord
   0 0 0
End atom_coord

Correlation_Type: CA
"""


class SelfConsistentSOCCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)
        shutil.copy(_AU_POTRE, self.directory / "Au_POTRE.DAT")

    def _write_input(self, name: str, extra: str = "") -> Path:
        path = self.directory / name
        path.write_text(_BASE_INPUT + extra, encoding="utf-8")
        return path

    def test_full_run_reports_kramers_paired_states(self) -> None:
        input_path = self._write_input("parsec.in")
        exit_code = cli_main([str(input_path), "--quiet"])
        self.assertEqual(exit_code, 0)

        log_text = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn("Self-consistent SOC iter", log_text)
        self.assertIn("Self-consistency convergence achieved.", log_text)
        self.assertIn("<S_z>", log_text)

        archive = np.load(self.directory / "parsec_python_results.npz")
        self.assertTrue(bool(archive["converged"]))
        self.assertEqual(archive["eigenvalues_ry"].shape, (20,))
        self.assertEqual(archive["magnetic_moment_per_state"].shape, (20,))
        # Kramers pairs: consecutive eigenvalues nearly equal, moments cancel.
        eigenvalues = archive["eigenvalues_ry"]
        moments = archive["magnetic_moment_per_state"]
        for i in range(0, len(eigenvalues) - 3, 2):
            self.assertAlmostEqual(eigenvalues[i], eigenvalues[i + 1], delta=1.0e-4)
            self.assertAlmostEqual(moments[i] + moments[i + 1], 0.0, delta=0.05)

    def test_self_consistent_soc_with_spin_polarization_is_rejected(self) -> None:
        input_path = self._write_input(
            "parsec.in", extra="Spin_Polarization: .true.\n"
        )
        exit_code = cli_main([str(input_path), "--dry-run"])
        self.assertEqual(exit_code, 2)

    def test_self_consistent_soc_without_so_psp_species_is_rejected(self) -> None:
        text = _BASE_INPUT.replace("SO_PSP: .true.\n", "")
        input_path = self.directory / "parsec.in"
        input_path.write_text(text, encoding="utf-8")
        exit_code = cli_main([str(input_path), "--quiet"])
        self.assertEqual(exit_code, 2)

    def test_self_consistent_soc_with_periodic_boundary_is_accepted(self) -> None:
        """Periodic self-consistent SOC (k-point sampling + SOC) is
        supported; see run_self_consistent_soc_kpoints and
        test_periodic_self_consistent_soc_cli.py for a full run."""
        input_path = self._write_input(
            "parsec.in",
            extra=(
                "Periodic_System: .true.\n"
                "Boundary_Conditions: bulk\n"
                "begin Cell_Shape\n"
                "8.0 8.0 8.0\n"
                "end Cell_Shape\n"
            ),
        )
        exit_code = cli_main([str(input_path), "--dry-run"])
        self.assertEqual(exit_code, 0)

    def test_periodic_so_psp_without_self_consistent_soc_is_rejected(self) -> None:
        """Periodic SO_PSP is only supported together with
        SO_from_scratch/SCF_SO; the default perturbative path does not
        handle k-point sampling."""
        text = _BASE_INPUT.replace("SO_from_scratch: .true.\n", "")
        input_path = self.directory / "parsec.in"
        input_path.write_text(
            text
            + "Periodic_System: .true.\n"
            "Boundary_Conditions: bulk\n"
            "begin Cell_Shape\n"
            "8.0 8.0 8.0\n"
            "end Cell_Shape\n",
            encoding="utf-8",
        )
        exit_code = cli_main([str(input_path), "--dry-run"])
        self.assertEqual(exit_code, 2)

    def test_scf_so_alone_is_also_accepted(self) -> None:
        """SCF_SO=true is treated the same as SO_from_scratch=true (see
        models.SCFSettings.self_consistent_spin_orbit's docstring)."""
        text = _BASE_INPUT.replace(
            "SO_from_scratch: .true.\n", "SCF_SO: .true.\n"
        )
        input_path = self.directory / "parsec.in"
        input_path.write_text(text, encoding="utf-8")
        exit_code = cli_main([str(input_path), "--quiet"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
