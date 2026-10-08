"""k-point-sampled runs report and archive every k-point, not just the first.

Au in an 8 bohr box on a 2x1x1 Monkhorst-Pack grid gives k = (+/-0.25, 0, 0)
in reciprocal-lattice units; the system is non-magnetic and inversion
symmetric, so time reversal makes the two k-points' levels identical.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.cli import main as cli_main

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)
_K = 2.0 * np.pi / 8.0 * 0.25

_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk

begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape

Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: {states}
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 80
Convergence_Criterion: 5.0e-3

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff
{extra}
Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
SO_PSP: {so_psp}
Local_Component: s
Initial_Spin_Polarization: 0.3
Begin atom_coord
   4.0 4.0 4.0
End atom_coord

Correlation_Type: CA

Kpoint_Method: mp
begin Monkhorst_Pack_Grid
2 1 1
end Monkhorst_Pack_Grid
"""

_STATE_ROW = re.compile(r"^\s*\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+")


class KpointOutputTests(unittest.TestCase):
    def _run(self, *, states: int, so_psp: str, extra: str = "") -> tuple[str, dict]:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shutil.copy(_AU_POTRE, directory / "Au_POTRE.DAT")
            (directory / "parsec.in").write_text(
                _INPUT.format(states=states, so_psp=so_psp, extra=extra),
                encoding="utf-8",
            )
            self.assertEqual(cli_main([str(directory / "parsec.in"), "--quiet"]), 0)
            log = (directory / "parsec.out").read_text(encoding="utf-8")
            archive = np.load(directory / "parsec_python_results.npz")
            return log, {key: archive[key] for key in archive.files}

    def _final_section(self, log: str, marker: str) -> str:
        self.assertIn(marker, log)
        return log[log.rindex(marker) :]

    def _check_archive_k_points(self, archive: dict) -> None:
        np.testing.assert_allclose(
            archive["k_points_cartesian_inv_bohr"], [[-_K, 0, 0], [_K, 0, 0]], atol=1e-12
        )
        np.testing.assert_allclose(archive["k_weights"], [0.5, 0.5])
        self.assertEqual(int(archive["n_kpoints"]), 2)

    def test_scalar_run_lists_every_kpoint(self) -> None:
        log, archive = self._run(states=10, so_psp="false")
        self._check_archive_k_points(archive)
        states = int(archive["n_states_per_kpoint"])
        self.assertEqual(archive["eigenvalues_ry"].size, 2 * states)

        final = self._final_section(log, "Converged eigenvalues by k-point")
        self.assertIn("k-point   1 of   2", final)
        self.assertIn("k-point   2 of   2", final)
        self.assertIn("frac = (-0.25000  0.00000  0.00000)", final)
        rows = [line for line in final.splitlines() if _STATE_ROW.match(line)]
        self.assertEqual(len(rows), 2 * states)
        # The listed eigenvalues are the archived ones, k-major.
        listed = np.array([float(row.split()[1]) for row in rows])
        np.testing.assert_allclose(listed, archive["eigenvalues_ry"], atol=1e-8)
        # The iteration blocks say they only show the first k-point.
        self.assertIn("first k-point only", log)

    def test_spin_polarized_run_lists_every_kpoint(self) -> None:
        log, archive = self._run(
            states=10, so_psp="false", extra="\nSpin_Polarization: .true.\n"
        )
        self._check_archive_k_points(archive)
        states = int(archive["n_states_per_kpoint"])
        self.assertEqual(archive["eigenvalues_up_ry"].size, 2 * states)
        final = self._final_section(log, "Converged eigenvalues by k-point")
        self.assertIn("k-point   2 of   2", final)
        rows = [line for line in final.splitlines() if _STATE_ROW.match(line)]
        # up and down tables for each of the two k-points
        self.assertEqual(len(rows), 4 * states)

    def test_spin_orbit_run_lists_every_kpoint_with_its_own_moments(self) -> None:
        log, archive = self._run(
            states=20, so_psp="true", extra="\nSO_from_scratch: .true.\n"
        )
        self._check_archive_k_points(archive)
        eigenvalues = archive["eigenvalues_ry"]
        moments = archive["magnetic_moment_per_state"]
        self.assertEqual(eigenvalues.shape, (40,))
        self.assertEqual(moments.shape, (40,))
        self.assertEqual(int(archive["n_states_per_kpoint"]), 20)

        # Non-magnetic and inversion symmetric: E(k) = E(-k), and each
        # Kramers pair at every k-point has opposite <S_z>.
        np.testing.assert_allclose(eigenvalues[:20], eigenvalues[20:], atol=1e-4)
        for start in (0, 20):
            for i in range(start, start + 16, 2):
                self.assertAlmostEqual(eigenvalues[i], eigenvalues[i + 1], delta=1e-4)
                self.assertAlmostEqual(moments[i] + moments[i + 1], 0.0, delta=0.05)

        final = self._final_section(log, "Converged spin-orbit-split spinor states")
        self.assertIn("k-point   1 of   2", final)
        self.assertIn("k-point   2 of   2", final)
        rows = [line for line in final.splitlines() if _STATE_ROW.match(line)]
        self.assertEqual(len(rows), 40)
        listed = np.array([float(row.split()[1]) for row in rows])
        np.testing.assert_allclose(listed, eigenvalues, atol=1e-8)

    def test_noncollinear_run_lists_every_kpoint(self) -> None:
        log, archive = self._run(
            states=20,
            so_psp="false",
            extra=(
                "\nNon_Collinear_magnetism: .true.\n"
                "begin Initial_NCL_Moment\n0 0 1\nend Initial_NCL_Moment\n"
            ),
        )
        self._check_archive_k_points(archive)
        states = int(archive["n_states_per_kpoint"])
        self.assertEqual(archive["eigenvalues_ry"].size, 2 * states)
        self.assertEqual(archive["magnetic_moment_per_state"].size, 2 * states)
        final = self._final_section(log, "Converged spin-orbit-split spinor states")
        self.assertIn("k-point   2 of   2", final)
        rows = [line for line in final.splitlines() if _STATE_ROW.match(line)]
        self.assertEqual(len(rows), 2 * states)

    def test_gamma_only_run_has_no_kpoint_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shutil.copy(_AU_POTRE, directory / "Au_POTRE.DAT")
            text = _INPUT.format(states=10, so_psp="false", extra="")
            text = text[: text.index("Kpoint_Method")]
            (directory / "parsec.in").write_text(text, encoding="utf-8")
            self.assertEqual(cli_main([str(directory / "parsec.in"), "--quiet"]), 0)
            archive = np.load(directory / "parsec_python_results.npz")
            self.assertNotIn("k_points_cartesian_inv_bohr", archive.files)
            log = (directory / "parsec.out").read_text(encoding="utf-8")
            self.assertNotIn("by k-point", log)
            self.assertNotIn("first k-point only", log)


if __name__ == "__main__":
    unittest.main()
