"""End-to-end CLI tests for periodic Monkhorst-Pack k-point sampling."""

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

Kpoint_Method: mp
begin Monkhorst_Pack_Grid
2 2 2
end Monkhorst_Pack_Grid

Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 10
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 5.0e-3

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
Local_Component: s
Begin atom_coord
   4.0 4.0 4.0
End atom_coord

Correlation_Type: CA
"""


class KPointSamplingCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)
        shutil.copy(_AU_POTRE, self.directory / "Au_POTRE.DAT")

    def _write_input(self, name: str, extra: str = "") -> Path:
        path = self.directory / name
        path.write_text(_BASE_INPUT + extra, encoding="utf-8")
        return path

    def test_full_2x2x2_run_converges(self) -> None:
        input_path = self._write_input("parsec.in")
        exit_code = cli_main([str(input_path), "--quiet"])
        self.assertEqual(exit_code, 0)

        log_text = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn("Self-consistency convergence achieved.", log_text)

        archive = np.load(self.directory / "parsec_python_results.npz")
        self.assertTrue(bool(archive["converged"]))
        # 8 k-points (2x2x2) x 10 states each = 80 pooled eigenvalues.
        self.assertEqual(archive["eigenvalues_ry"].shape, (80,))

    def test_mp_with_spin_polarization_is_rejected(self) -> None:
        input_path = self._write_input(
            "parsec.in", extra="Spin_Polarization: .true.\n"
        )
        exit_code = cli_main([str(input_path), "--dry-run"])
        self.assertEqual(exit_code, 2)

    def test_manual_kpoint_method_is_rejected(self) -> None:
        text = _BASE_INPUT.replace("Kpoint_Method: mp", "Kpoint_Method: manual")
        input_path = self.directory / "parsec.in"
        input_path.write_text(text, encoding="utf-8")
        exit_code = cli_main([str(input_path), "--dry-run"])
        self.assertEqual(exit_code, 2)

    def test_monkhorst_pack_shift_is_rejected(self) -> None:
        input_path = self._write_input(
            "parsec.in",
            extra="begin Monkhorst_Pack_Shift\n0.5 0.5 0.5\nend Monkhorst_Pack_Shift\n",
        )
        exit_code = cli_main([str(input_path), "--dry-run"])
        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
