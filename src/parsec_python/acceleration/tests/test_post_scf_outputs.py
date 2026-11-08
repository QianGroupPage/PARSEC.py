"""The band structure of a Gamma-point cell and the cluster DOS after the accelerated SCF.

:mod:`parsec_python.acceleration.Output.post_scf` runs ``bandstruc`` after the accelerated Gamma-point
SCF of a scalar periodic cell and ``Create_DOS`` after the accelerated SCF of a cluster; every other
combination of the two features stays on the reference solver by name.  The tests check the routing
rule and that the files of the accelerated CLI equal those of the reference CLI on the same input.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Input import parse_parsec_input

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

CELL = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  6.0 6.0 6.0
end Cell_Shape
Grid_Spacing: 0.75
Expansion_Order: 4
States_Num: 4
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 1.0e-7
Diag_Tolerance: 1.0e-8
Eigensolver: chebdav
Mixing_Method: Anderson
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  3.1 2.9 3.05
  1.6 1.4 1.55
end Atom_Coord
Correlation_Type: ca
{extra}
"""

CLUSTER = """\
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
Convergence_Criterion: 1.0e-7
Eigensolver: chebff
FF_Maxiter: 9
Chebdav_Degree: 40
Mixing_Method: Anderson
Ignore_Symmetry: .true.
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

BANDS = "begin bandstruc\n1 0.0 0.0 0.0  0.5 0.0 0.0  G-X\nend bandstruc\nbandstruc_points 3\n"
MP = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 1\nend Monkhorst_Pack_Grid\n"
BANDS_FEATURE = "band structure along a k-path (bandstruc)"
DOS_FEATURE = "the density of states (Create_DOS)"


def _write(directory: Path, template: str, extra: str = "") -> Path:
    shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
    path = directory / "parsec.in"
    path.write_text(template.format(extra=extra), encoding="utf-8")
    return path


def _features(template: str, extra: str = "") -> list[str]:
    from parsec_python.acceleration.cli import _reference_only_features

    with tempfile.TemporaryDirectory() as directory:
        translation = parse_parsec_input(_write(Path(directory), template, extra))
    return _reference_only_features(translation, accelerated_backend_selected=True)


def _numbers(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text().splitlines():
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError:
                pass
    return np.asarray(values)


class RoutingTests(unittest.TestCase):
    def test_gamma_cell_bands_and_cluster_dos_run_accelerated(self) -> None:
        self.assertEqual(_features(CELL, BANDS), [])
        self.assertEqual(_features(CLUSTER, "Create_DOS: .true.\n"), [])

    def test_other_combinations_stay_on_the_reference_solver_by_name(self) -> None:
        cases = (
            (CELL, MP + BANDS, BANDS_FEATURE),  # bands after a k-point SCF
            (CELL, "Spin_Polarization: .true.\n" + BANDS, BANDS_FEATURE),
            (CELL, MP + "Create_DOS: .true.\n", DOS_FEATURE),  # tetrahedra need k-points
        )
        for template, extra, feature in cases:
            with self.subTest(extra=extra):
                self.assertIn(feature, _features(template, extra))

    def test_mpi_keeps_both_on_the_reference_solver(self) -> None:
        with mock.patch.dict(os.environ, {"PARSEC_MPI": "1"}):
            self.assertIn(BANDS_FEATURE, _features(CELL, BANDS))
            self.assertIn(DOS_FEATURE, _features(CLUSTER, "Create_DOS: .true.\n"))

    def test_rule_of_the_band_structure(self) -> None:
        from parsec_python.acceleration.Output.post_scf import (
            accelerated_band_structure,
            accelerated_dos,
        )

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            plain = parse_parsec_input(_write(directory, CELL, BANDS))
            self.assertTrue(accelerated_band_structure(plain))
            self.assertFalse(accelerated_dos(plain))
            without = parse_parsec_input(_write(directory, CELL))
            self.assertFalse(accelerated_band_structure(without))
            sampled = parse_parsec_input(_write(directory, CELL, MP + BANDS))
            self.assertFalse(accelerated_band_structure(sampled))
            spin = parse_parsec_input(_write(directory, CELL, "Spin_Polarization: .true.\n" + BANDS))
            self.assertFalse(accelerated_band_structure(spin))


def _run(main, path: Path) -> tuple[int, str]:
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as error:
        code = main([str(path), "--quiet"])
    return code, error.getvalue()


class AcceleratedFilesEqualReferenceTests(unittest.TestCase):
    def _both(self, template: str, extra: str, names: tuple[str, ...]):
        from parsec_python.acceleration.cli import main as accelerated
        from parsec_python.cli import main as reference

        outputs = {}
        for label, main in (("reference", reference), ("accelerated", accelerated)):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                path = _write(directory, template, extra)
                code, error = _run(main, path)
                self.assertEqual(code, 0, error)
                if label == "accelerated":
                    # the input ran on the accelerated path, not on the reference solver
                    self.assertNotIn("running the reference parsec_python.cli solver", error)
                text = (directory / "parsec.out").read_text()
                outputs[label] = (text, {name: (directory / name).read_text() for name in names})
        return outputs

    def test_band_structure_after_the_accelerated_gamma_scf(self) -> None:
        outputs = self._both(CELL, BANDS, ("bands.dat",))
        text, files = outputs["accelerated"]
        self.assertIn("Band structure (non-self-consistent, Fortran calc_bands)", text)
        self.assertIn("Post-SCF band structure / DOS wall time [sec]", text)
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder) / "a", Path(folder) / "r"
            left.write_text(files["bands.dat"])
            right.write_text(outputs["reference"][1]["bands.dat"])
            a, r = _numbers(left), _numbers(right)
        self.assertEqual(a.shape, r.shape)
        # the same bands from potentials that agree to the SCF's round-off
        np.testing.assert_allclose(a, r, atol=1e-7)

    def test_dos_after_the_accelerated_cluster_scf(self) -> None:
        outputs = self._both(CLUSTER, "Create_DOS: .true.\nDOS_Pnum: 40\n", ("DOS",))
        text, files = outputs["accelerated"]
        self.assertIn(" Generation of DOS : ", text)
        lines = files["DOS"].splitlines()
        reference = outputs["reference"][1]["DOS"].splitlines()
        self.assertEqual(len(lines), len(reference))
        self.assertEqual(len(lines), 3 + 40 + 1)
        for mine, theirs in zip(lines, reference):
            if mine.split() and mine.split()[0][0] in "-0123456789":
                x, y = (np.asarray(line.split(), dtype=float) for line in (mine, theirs))
                np.testing.assert_allclose(x, y, atol=1e-6)
            else:
                self.assertEqual(mine, theirs)

    def test_runs_without_the_features_print_nothing_new(self) -> None:
        from parsec_python.acceleration.cli import main as accelerated

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            code, error = _run(accelerated, _write(directory, CLUSTER))
            self.assertEqual(code, 0, error)
            text = (directory / "parsec.out").read_text()
            self.assertNotIn("Post-SCF band structure / DOS", text)
            self.assertFalse((directory / "DOS").exists())


if __name__ == "__main__":
    unittest.main()
