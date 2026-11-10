"""PARSEC 1.5's FDET on the accelerated path (the shared SCF loop with device eigensolvers).

The accelerated driver runs FDET through the reference SCF loop with its
device Hamiltonian, Hartree solver and XC evaluator; the frozen field joins
the ionic potential and the non-additive kinetic terms are evaluated on the
host.  The numbers are those of PARSEC 1.5 (gfortran build, the inputs of
``tests/unit/test_fdet_end_to_end.py``): a single point with its
forces, a classical manual scan by whole grid steps (1.5 translates density
and orbitals; serial run), and a tip-sphere scan with the PW91k potential of
the 1.5 form.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.cli import _mpi_unsupported_features
from parsec_python.acceleration.cli import main as accelerated_main
from parsec_python.Input import parse_parsec_input

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
CUPY = cupy_available()

TIP = """\
Boundary_Sphere_Radius   {radius}
Grid_Spacing             0.5
Ignore_Symmetry          true
States_Num               4
Correlation_Type         ca
Convergence_Criterion    1d-7 ry
Max_Iter                 60
Hartree_Boundary_Tolerance off
Potential_Field            true
Potential_Field_Name       sample.dat
Kinetic_Energy_Functional  {kinetic}
Add_Point_Charges  true
Point_Typ_Num      1
Pt_Chg             1.0 e
begin Point_Coord
 -0.70 0.0 0.0
  0.70 0.0 0.0
end Point_Coord
Atom_Types_Num           1
Atom_Type        H
Local_Component  s
begin Atom_Coord
 0.10 0.0 1.90
 0.10 0.0 3.30
end Atom_Coord
{extra}
"""
MANUAL = " 0.10 0.0 1.40\n 0.10 0.0 2.80\n step 2\n 0.10 0.0 2.40\n 0.10 0.0 3.80\n"
SCAN = "Minimization manual\nMovement_Num 2\n"


def _write(directory: Path, *, radius="5.0", kinetic="tf", extra="") -> Path:
    path = directory / "parsec.in"
    path.write_text(TIP.format(radius=radius, kinetic=kinetic, extra=extra), encoding="utf-8")
    (directory / "H_POTRE.DAT").write_bytes((DATA / "fdet_H_POTRE.DAT").read_bytes())
    (directory / "sample.dat").write_bytes((DATA / "fdet_H2_sample_POT.DAT").read_bytes())
    (directory / "manual.dat").write_text(MANUAL, encoding="utf-8")
    return path


def _run(path: Path):
    stream = StringIO()
    with redirect_stdout(stream), redirect_stderr(stream):
        status = accelerated_main([str(path), "--quiet"])
    with np.load(path.parent / "parsec_python_results.npz") as archive:
        members = {name: archive[name] for name in archive.files}
    return status, stream.getvalue(), members, (path.parent / "parsec.out").read_text(encoding="utf-8")


class FDETRoutesTest(unittest.TestCase):
    def test_mpi_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as folder:
            translation = parse_parsec_input(_write(Path(folder)))
        self.assertIn("FDET (Potential_Field)", _mpi_unsupported_features(translation))


@unittest.skipUnless(CUPY, "CuPy and a usable GPU are required")
class DeviceKineticTest(unittest.TestCase):
    def test_device_terms_equal_the_host_terms(self):
        from parsec_python.acceleration.Embedding.device_kinetic import DeviceKineticOperators
        from parsec_python.Embedding.frozen_density import FrozenSampleField
        from parsec_python.Embedding.kinetic import KineticFunctional
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.models import GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.4, radius=5.0, shift=(0.0, 0.0, 0.0)))
        tip = np.exp(-np.linalg.norm(grid.coordinates - [0.0, 0.0, 1.0], axis=1))
        sample = 0.5 * np.exp(-np.linalg.norm(grid.coordinates - [0.0, 0.2, -2.0], axis=1))
        operators = DeviceKineticOperators(grid)
        for name in ("tf", "tw", "pw", "pb"):
            for form in ("corrected", "parsec1.5"):
                functional = KineticFunctional(name, form=form)
                host = FrozenSampleField(sample, sample, sample, sample, 1.0, functional).nonadditive_kinetic(
                    tip, None, grid
                )
                device = FrozenSampleField(sample, sample, sample, sample, 1.0, functional).nonadditive_kinetic(
                    tip, None, grid, operators=operators
                )
                with self.subTest(functional=name, form=form):
                    self.assertIsInstance(device.potential, np.ndarray)
                    self.assertAlmostEqual(device.energy, host.energy, delta=1.0e-12)
                    np.testing.assert_allclose(device.potential, host.potential, rtol=0.0, atol=1.0e-12)


@unittest.skipUnless(CUPY and native_available(), "CuPy, a usable GPU and the native extension are required")
class AcceleratedFDETTest(unittest.TestCase):
    def test_single_point_with_forces(self):
        with tempfile.TemporaryDirectory() as folder:
            status, output, archive, log = _run(_write(Path(folder), extra="Skip_Force false"))
        self.assertEqual(status, 0, output)
        self.assertNotIn("running the reference parsec_python.cli solver", output)
        self.assertIn("Acceleration backend:", log)
        self.assertAlmostEqual(float(archive["energy_total_ry"]), -6.26328247, delta=1.0e-6)
        self.assertAlmostEqual(float(archive["energy_nonadditive_kinetic_ry"]), 0.27637703, delta=1.0e-6)
        np.testing.assert_allclose(archive["forces_net"], (-1.838918, 0.0, 0.248902), atol=2.0e-6)
        self.assertIn("Net forces not removed in FDET.", log)
        self.assertIn("Isolated Sample Kinetic Energy=", log)

    def test_classical_scan_by_whole_grid_steps(self):
        # parsec.F90:2815-2899 translates density, orbitals and V_H (1.5: 16/11/12 iterations).
        with tempfile.TemporaryDirectory() as folder:
            status, output, archive, _ = _run(_write(Path(folder), extra=SCAN))
        self.assertIn(status, (0, 3), output)
        np.testing.assert_allclose(
            archive["relaxation_energies"], (-6.26328247, -6.19362185, -6.15386169), atol=1.0e-6
        )
        iterations = archive["relaxation_scf_iterations"]
        self.assertLess(int(iterations[1]), int(iterations[0]))

    def test_tip_sphere_scan(self):
        with tempfile.TemporaryDirectory() as folder:
            path = _write(Path(folder), radius="3.0", kinetic="pw",
                          extra=SCAN + "Boundary_Sphere_Around_Tip true\nFDET_Kinetic_Form parsec1.5")
            status, output, archive, log = _run(path)
        self.assertIn(status, (0, 3), output)
        self.assertIn("FDET tip sphere: spot grid offset", log)
        np.testing.assert_allclose(archive["relaxation_energies"], TIP_SPHERE_SCAN, atol=1.0e-6)

    def test_afm_scan_table(self):
        with tempfile.TemporaryDirectory() as folder:
            path = _write(
                Path(folder),
                radius="3.0",
                kinetic="pw",
                extra="Boundary_Sphere_Around_Tip true\nFDET_Kinetic_Form parsec1.5\nAFM_Scan true\n"
                "AFM_Scan_Heights 3\nAFM_Scan_Height_Step 0.5 bohr",
            )
            status, output, archive, _ = _run(path)
            table = (Path(folder) / "afm_scan.dat").read_text(encoding="ascii").splitlines()
        self.assertIn(status, (0, 3), output)
        # heights -0.5, 0, +0.5 around the input: the structures of the manual scan above
        energies = archive["relaxation_energies"]
        np.testing.assert_allclose(sorted(energies), sorted(TIP_SPHERE_SCAN_AFM), atol=1.0e-6)
        rows = [line.split() for line in table if not line.startswith("#")]
        self.assertEqual(len(rows), 3)
        below, middle, above = (float(row[4]) for row in rows)
        self.assertAlmostEqual(float(rows[1][5]), (below - 2.0 * middle + above) / 0.25, places=8)

    def test_molecular_dynamics_keeps_the_motion_of_the_net_force(self):
        # FDET keeps the net force and the dynamics integrates it (MolecularDynamics.moldyn keep_center_of_mass;
        # PARSEC 1.5 holds the centre of mass and loses 0.07 Ry in 4 steps of dt 10 here).
        with tempfile.TemporaryDirectory() as folder:
            status, output, archive, _ = _run(
                _write(Path(folder), extra="Molecular_Dynamics true\nStep_Num 4\nTime_Step 2.5\n")
            )
        self.assertIn(status, (0, 3), output)
        drift = archive["dynamics_conserved_energies"] - archive["dynamics_conserved_energies"][0]
        self.assertLess(float(np.max(np.abs(drift))), 2.0e-4)
        centre = archive["dynamics_positions"].mean(axis=1)[:, 0]
        steps = np.diff(centre)
        # F_net,x = -1.84 Ry/bohr: the centre speeds up along -x
        self.assertTrue(np.all(steps < 0.0))
        self.assertGreater(abs(steps[-1]), 3.0 * abs(steps[0]))

    def test_grid_shift_and_fold_blocks(self):
        # Grid_Shift and Foldblock_Side_Length 2 (PARSEC 1.5): the sample run writes pot.dat on its
        # grid, the classical tip maps it by grid indices; the tip sphere maps a folded tip grid on the spot field.
        # Numbers of PARSEC 1.5 (2 ranks).
        shift = "begin Grid_Shift\n 0.25 0.0 0.5\nend Grid_Shift\n"
        sample = TIP.split("Potential_Field")[0].format(radius="5.0") + (
            "Atom_Types_Num 1\nAtom_Type H\nLocal_Component s\nbegin Atom_Coord\n -0.70 0.0 0.0\n  0.70 0.0 0.0\n"
            "end Atom_Coord\nWrite_Potential_Field true\n"
        )
        for extra, sample_energy, tip_energy in (
            (shift, -2.31009077, -6.08555089),
            ("Foldblock_Side_Length 2\n", -2.58736028, -6.28823110),
        ):
            with self.subTest(grid=extra.splitlines()[0]), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                (directory / "sample").mkdir()
                (directory / "sample" / "parsec.in").write_text(sample + extra, encoding="utf-8")
                (directory / "sample" / "H_POTRE.DAT").write_bytes((DATA / "fdet_H_POTRE.DAT").read_bytes())
                status, output, archive, _ = _run(directory / "sample" / "parsec.in")
                self.assertEqual(status, 0, output)
                self.assertAlmostEqual(float(archive["energy_total_ry"]), sample_energy, delta=1.0e-7)
                path = _write(directory, extra=extra)
                (directory / "sample.dat").write_bytes((directory / "sample" / "pot.dat").read_bytes())
                status, output, archive, _ = _run(path)
                self.assertEqual(status, 0, output)
                self.assertAlmostEqual(float(archive["energy_total_ry"]), tip_energy, delta=1.0e-6)
        with tempfile.TemporaryDirectory() as folder:
            path = _write(
                Path(folder),
                radius="3.0",
                kinetic="pw",
                extra=SCAN + "Boundary_Sphere_Around_Tip true\nFDET_Kinetic_Form parsec1.5\nFoldblock_Side_Length 2",
            )
            status, output, archive, log = _run(path)
        self.assertIn(status, (0, 3), output)
        self.assertIn("without pot.dat data      137 of     1304", log)
        np.testing.assert_allclose(archive["relaxation_energies"], (-2.28053434, -2.14592784, -2.32940840), atol=1.0e-6)


# PARSEC 1.5, tip-sphere scan (tip sphere 3 bohr, pw, 2 ranks): the H2 tip at 1.9/3.3, 1.4/2.8 and
# 2.4/3.8 bohr (offsets 0 0 5, 4, 6; 45, 0 and 134 of 925 tip points without sample data).
TIP_SPHERE_SCAN = (-2.21694784, -2.08165999, -2.26364722)
TIP_SPHERE_SCAN_AFM = TIP_SPHERE_SCAN


if __name__ == "__main__":
    unittest.main()
