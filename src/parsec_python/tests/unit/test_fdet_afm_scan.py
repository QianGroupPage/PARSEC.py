"""The AFM scan of an FDET tip around its runs: the table, the folder of a part, the tip-sphere offset.

The tip-sphere offset rounds half steps away from zero, the table is written after every tip position, a
k_ts from an unconverged SCF loop is left out, and a folder belongs to one part of a scan.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from parsec_python.Embedding.afm import AFMScanPoint, AFMScanSettings, force_gradients, unconverged_gradients
from parsec_python.Embedding.tip_sphere import tip_offset
from parsec_python.Input import ParsecInputError, parse_parsec_input
from parsec_python.Output.afm_scan import SCAN_FOLDER_MARK, claim_scan_folder, scan_table_lines
from parsec_python.Relaxation import IonicBackend
from parsec_python.Relaxation.run import run_recorded_relaxation

DATA = Path(__file__).resolve().parents[1] / "data"


def _write(directory: Path, extra: str) -> Path:
    text = f"""\
Boundary_Sphere_Radius: 4.0 bohr
Grid_Spacing: 0.8 bohr
Expansion_Order: 4
States_Num: 2
Ignore_Symmetry: true
Hartree_Boundary_Tolerance: off
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.2  0.0  1.0
  0.2  0.0  2.4
end Atom_Coord
Add_Point_Charges: true
Point_Typ_Num: 1
Pt_Chg: 1.0 e
begin Point_Coord
 -0.7 0.0 0.0
  0.7 0.0 0.0
end Point_Coord
Potential_Field: true
AFM_Scan: true
AFM_Scan_Heights: 3
AFM_Scan_Height_Step: 0.8 bohr
{extra}
"""
    path = directory / "parsec.in"
    path.write_text(text, encoding="utf-8")
    (directory / "H_POTRE.DAT").write_bytes((DATA / "H_POTRE.DAT").read_bytes())
    return path


_TWO_LINES = "begin AFM_Scan_Lateral_Points\n 0.0 0.0\n {dx} 0.0\nend AFM_Scan_Lateral_Points\n"


class TipOffsetTest(unittest.TestCase):
    def test_half_steps_round_away_from_zero(self):
        # Fortran's nint (tipsphere_recenter); numpy.rint would give 6, -0, 2 and -2.
        h = 0.4
        positions = np.array([[6.0 * h, -0.5 * h, 2.0 * h], [7.0 * h, -0.5 * h, 3.0 * h]])
        np.testing.assert_array_equal(tip_offset(positions, h), [7, -1, 3])
        np.testing.assert_array_equal(tip_offset([[-2.5 * h, 0.49 * h, 5.51 * h]], h), [-3, 0, 6])


class ScanDomainTest(unittest.TestCase):
    def test_every_tip_position_is_checked_when_the_input_is_read(self):
        with tempfile.TemporaryDirectory() as folder:
            # Classical mode: the second line puts the upper H at x = 3.7 bohr, outside the sphere of 4 bohr.
            with self.assertRaises(ParsecInputError) as caught:
                parse_parsec_input(_write(Path(folder), _TWO_LINES.format(dx=3.5)))
            self.assertIn("line 2, height 3", str(caught.exception))
            self.assertIn("outside the domain", str(caught.exception))
            translation = parse_parsec_input(_write(Path(folder), _TWO_LINES.format(dx=0.8)))
            self.assertEqual(translation.relaxation.max_movements, 5)
            # The tip sphere centres the tip in its own sphere at every position.
            translation = parse_parsec_input(
                _write(Path(folder), "Boundary_Sphere_Around_Tip: true\n" + _TWO_LINES.format(dx=3.5))
            )
            self.assertEqual(translation.relaxation.max_movements, 5)


class ForceGradientTest(unittest.TestCase):
    def test_a_k_ts_needs_three_converged_energies(self):
        settings = AFMScanSettings(heights=4, height_step=0.5)
        points = [
            AFMScanPoint(0, k, displacement, 0.2 * displacement[2] ** 2, k != 3)
            for (_, k, displacement) in settings.offsets()
        ]
        gradients = force_gradients(settings, points)
        self.assertEqual(sorted(gradients), [(0, 1)])
        self.assertAlmostEqual(gradients[(0, 1)], 0.4, places=12)
        self.assertEqual(unconverged_gradients(settings, points), {(0, 2)})
        rows = [line.split() for line in scan_table_lines(settings, points) if not line.startswith("#")]
        self.assertEqual([row[5] for row in rows], ["-", "4.00000000e-01", "unconv", "-"])
        self.assertEqual([row[7] for row in rows], ["yes", "yes", "yes", "no"])


class ScanFolderTest(unittest.TestCase):
    def test_one_folder_per_part(self):
        with tempfile.TemporaryDirectory() as folder:
            lines = ((0.0, 0.0), (0.8, 0.0), (1.6, 0.0))
            part = SimpleNamespace(afm_scan=AFMScanSettings(heights=3, height_step=0.5, lateral_points=lines, line_range=(2, 3)))
            self.assertIn("lines 2-3", claim_scan_folder(part, folder))
            self.assertEqual((Path(folder) / SCAN_FOLDER_MARK).read_text(encoding="ascii"), "2-3\n")
            # The same part again (a rerun) is let through, another part or the whole scan is refused.
            self.assertIsNotNone(claim_scan_folder(part, folder))
            other = SimpleNamespace(afm_scan=AFMScanSettings(heights=3, height_step=0.5, lateral_points=lines, line_range=(1, 1)))
            whole = SimpleNamespace(afm_scan=AFMScanSettings(heights=3, height_step=0.5, lateral_points=lines))
            for refused in (other, whole):
                with self.assertRaises(ValueError) as caught:
                    claim_scan_folder(refused, folder)
                self.assertIn("needs a folder of its own", str(caught.exception))
            self.assertIsNone(claim_scan_folder(SimpleNamespace(afm_scan=None), folder))


class _Quiet:
    def __getattr__(self, name):
        return lambda *arguments, **keywords: None


class ScanTableTest(unittest.TestCase):
    def test_the_table_keeps_the_points_before_a_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            translation = parse_parsec_input(_write(directory, _TWO_LINES.format(dx=0.8)))
            calls = []

            def run_scf(system, callback=None, warm_start=None, **keywords):
                index = len(calls)
                calls.append(index)
                if index == 4:
                    raise RuntimeError("the SCF loop of the fifth tip position failed")
                height = float(np.mean([atom.position[2] for atom in system.atoms]))
                return SimpleNamespace(
                    energies=SimpleNamespace(total=-2.0 + 0.3 * (height - 1.7) ** 2),
                    converged=True,
                    iterations=1,
                    wavefunctions=None,
                )

            backend = IonicBackend(
                run_scf=run_scf,
                compute_forces=lambda system, result, **keywords: SimpleNamespace(total=np.zeros((2, 3))),
                update_geometry=lambda system, atoms: SimpleNamespace(atoms=tuple(atoms)),
                warm_start=lambda previous, result, system: None,
            )
            with self.assertRaises(RuntimeError):
                run_recorded_relaxation(
                    translation,
                    SimpleNamespace(atoms=tuple(translation.problem.atoms)),
                    backend,
                    _Quiet(),
                    lambda text="": None,
                    directory=directory,
                    symmetrize=False,
                )
            table = (directory / "afm_scan.dat").read_text(encoding="ascii").splitlines()
        rows = [line.split() for line in table if not line.startswith("#")]
        # Line 1 (three heights) and the top height of line 2 (it runs downwards) were done.
        self.assertEqual([(row[0], row[3]) for row in rows], [("1", "-0.80000"), ("1", "0.00000"), ("1", "0.80000"), ("2", "0.80000")])
        self.assertAlmostEqual(float(rows[1][5]), 0.6, places=8)
        self.assertEqual(rows[3][5], "-")


if __name__ == "__main__":
    unittest.main()
