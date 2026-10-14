"""End-to-end CLI tests for the spin-polarized SCF + perturbative SOC path.

Uses the real ``Au_POTRE.DAT`` from Fortran PARSEC's own ``0d_AuH`` example
(already copied into ``examples/0d_AuH`` for the pseudopotential-level SOC
tests) to exercise ``parsec_python.cli.main`` the way a user actually would:
a text ``parsec.in`` on disk, run through the full input parser, SCF driver,
text reporter, and archive writer.
"""

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

States_Num: 6
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 1.0e-2

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

Spin_Polarization: .true.

Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
SO_PSP: .true.
Initial_Spin_Polarization: 0.3
Local_Component: s
Begin atom_coord
   0 0 0
End atom_coord

Correlation_Type: CA
"""


class SpinOrbitCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)
        shutil.copy(_AU_POTRE, self.directory / "Au_POTRE.DAT")

    def _write_input(self, name: str, extra: str = "") -> Path:
        path = self.directory / name
        path.write_text(_BASE_INPUT + extra, encoding="utf-8")
        return path

    def test_full_run_reports_spin_channels_and_soc_split(self) -> None:
        input_path = self._write_input("parsec.in")
        exit_code = cli_main([str(input_path), "--quiet"])
        self.assertEqual(exit_code, 0)

        log_text = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn("Spin-polarized SCF iter", log_text)
        self.assertIn("Magnetic moment N_up - N_down", log_text)
        self.assertIn("Perturbative spin-orbit correction", log_text)
        self.assertIn("Self-consistency convergence achieved.", log_text)

        archive = np.load(self.directory / "parsec_python_results.npz")
        self.assertEqual(archive["eigenvalues_up_ry"].shape, (6,))
        self.assertEqual(archive["eigenvalues_down_ry"].shape, (6,))
        self.assertEqual(archive["soc_eigenvalues_ry"].shape, (12,))
        self.assertTrue(bool(archive["converged"]))
        # Au's atomic ground state ([Xe]4f14 5d10 6s1) has one unpaired
        # electron.
        self.assertAlmostEqual(float(archive["magnetic_moment"]), 1.0, places=1)

    def test_scf_so_combined_with_spin_polarization_is_accepted(self) -> None:
        """Cluster self-consistent SOC combined with spin polarization is
        supported (run_self_consistent_soc_spin_polarized); see
        test_self_consistent_soc_spin_polarized.py and
        test_0d_AuH_fortran_reference.py for the physics-level and
        real-Fortran-reference validation."""
        input_path = self._write_input("parsec.in", extra="SCF_SO: .true.\n")
        exit_code = cli_main([str(input_path), "--dry-run"])
        self.assertEqual(exit_code, 0)

    def test_periodic_spin_polarized_is_rejected(self) -> None:
        input_path = self._write_input(
            "parsec.in",
            extra=(
                "Periodic_System: .true.\n"
                "Boundary_Conditions: bulk\n"
                "Begin Cell_Shape\n"
                "10.0 0.0 0.0\n"
                "0.0 10.0 0.0\n"
                "0.0 0.0 10.0\n"
                "End Cell_Shape\n"
            ),
        )
        exit_code = cli_main([str(input_path), "--dry-run"])
        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
