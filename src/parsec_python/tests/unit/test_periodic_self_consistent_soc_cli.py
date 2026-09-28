"""End-to-end CLI tests for periodic self-consistent spin-orbit coupling
(Monkhorst-Pack k-point sampling combined with SOC)."""

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
Periodic_System: .true.
Boundary_Conditions: bulk

begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape

Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 20
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 60
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
   4.0 4.0 4.0
End atom_coord

Correlation_Type: CA
"""


class PeriodicSelfConsistentSOCCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)
        shutil.copy(_AU_POTRE, self.directory / "Au_POTRE.DAT")

    def _write_input(self, name: str, extra: str = "") -> Path:
        path = self.directory / name
        path.write_text(_BASE_INPUT + extra, encoding="utf-8")
        return path

    def test_gamma_only_run_reports_kramers_paired_states(self) -> None:
        """With no Kpoint_Method given, periodic self-consistent SOC falls
        back to a single Gamma point -- a TRIM point, so the Kramers-pair
        signature from the cluster case still applies."""
        input_path = self._write_input("parsec.in")
        exit_code = cli_main([str(input_path), "--quiet"])
        self.assertEqual(exit_code, 0)

        log_text = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn("Self-consistent SOC iter", log_text)
        self.assertIn("Self-consistency convergence achieved.", log_text)

        archive = np.load(self.directory / "parsec_python_results.npz")
        self.assertTrue(bool(archive["converged"]))
        self.assertEqual(archive["eigenvalues_ry"].shape, (20,))
        eigenvalues = archive["eigenvalues_ry"]
        moments = archive["magnetic_moment_per_state"]
        for i in range(0, len(eigenvalues) - 3, 2):
            self.assertAlmostEqual(eigenvalues[i], eigenvalues[i + 1], delta=1.0e-4)
            self.assertAlmostEqual(moments[i] + moments[i + 1], 0.0, delta=0.05)

    def test_monkhorst_pack_run_pools_every_kpoint(self) -> None:
        input_path = self._write_input(
            "parsec.in",
            extra=(
                "Kpoint_Method: mp\n"
                "begin Monkhorst_Pack_Grid\n"
                "2 1 1\n"
                "end Monkhorst_Pack_Grid\n"
            ),
        )
        exit_code = cli_main([str(input_path), "--quiet"])
        self.assertEqual(exit_code, 0)

        archive = np.load(self.directory / "parsec_python_results.npz")
        self.assertTrue(bool(archive["converged"]))
        # 2 k-points x 20 states each = 40 pooled eigenvalues.
        self.assertEqual(archive["eigenvalues_ry"].shape, (40,))

    def test_periodic_self_consistent_soc_with_spin_polarization_is_rejected(
        self,
    ) -> None:
        input_path = self._write_input(
            "parsec.in", extra="Spin_Polarization: .true.\n"
        )
        exit_code = cli_main([str(input_path), "--dry-run"])
        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
