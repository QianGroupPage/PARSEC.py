"""NSCF, the periodic DOS, Output_GW, k-point bands and the k-point eigen.dat on the device workers.

These features of a k-point run stay on the reference solver (``acceleration/cli.py``
``_reference_only_features``): the reference k-point drivers and the band machinery hand their
eigenproblems to the device k-point workers (:mod:`parsec_python.acceleration.SCF.kpoint_parallel`,
:mod:`parsec_python.acceleration.Hamiltonian.device_bloch`) whenever a device backend is resolved, and
what they do on the host (Bloch projectors, the tetrahedra, the Ylm projections, the FFTs of the WFN
file) costs a few percent of a run (the report of the parity round).  The tests check that the NSCF
eigensolves run on the device route and that the files of the device route equal those of the host
route on the same input: ``numpy-device`` (the device code on the host, deterministic) always, and
CuPy where a GPU is visible.
"""


# With both sets of routines: the periodic inputs and grids of this module are those of the Fortran-derived
# routines (Use_Fortran_Routines: .true., PeriodicGridSettings(fortran_routines=True)); the numbers checked
# here are those of the Fortran parity round.
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
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  6.0 6.0 6.0
end Cell_Shape
Grid_Spacing: 0.75
Expansion_Order: 4
States_Num: 4
Fermi_Temp: 500.0
Max_Iter: 50
Convergence_Criterion: 1.0e-8
Diag_Tolerance: 1.0e-9
Eigensolver: chebdav
Mixing_Method: Anderson
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  3.1 2.9 2.35
  3.1 2.9 3.75
end Atom_Coord
Correlation_Type: ca
{extra}
"""

MP = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 {nz}\nend Monkhorst_Pack_Grid\n"
NSCF = """\
Restart_Run: .true.
nscf_kpoint_unit: reciprocal_lattice_vectors
nscf_states_num: 5
begin nscf_kpoints
  0.0 0.0 0.0 1.0
  0.5 0.0 0.0 3.0
  0.25 0.25 0.0 4.0
end nscf_kpoints
"""


def _cupy_usable() -> bool:
    try:
        from parsec_python.acceleration.Hamiltonian.device_bloch import resolve_kpoint_backend

        return resolve_kpoint_backend("auto").name == "cupy"
    except Exception:  # noqa: BLE001
        return False


DEVICE_BACKENDS = ("numpy-device",) + (("cupy",) if _cupy_usable() else ())


def _write(directory: Path, extra: str) -> Path:
    shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
    path = directory / "parsec.in"
    path.write_text(CELL.format(extra=extra), encoding="utf-8")
    return path


def _run(path: Path, backend: str, *extra: str) -> None:
    from parsec_python.cli import main

    environment = {"PARSEC_KPOINT_BACKEND": backend, "PARSEC_KPOINT_WORKERS": "1"}
    with mock.patch.dict(os.environ, environment):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as error:
            code = main([str(path), "--quiet", *extra])
    if code != 0:
        raise AssertionError(f"run of {path} on {backend} failed ({code}): {error.getvalue()}")


def _dos_pairs(path: Path) -> np.ndarray:
    pairs = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0][0] in "-0123456789":
            pairs.append([float(value) for value in parts])
    return np.asarray(pairs)


class CountingEigensolve:
    """Counts the k-point solves of the band machinery that the device solver makes."""

    def __enter__(self):
        from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter

        self.calls = 0
        original = DeviceBlochConverter.eigensolve

        def counted(converter, *args, **kwargs):
            self.calls += 1
            return original(converter, *args, **kwargs)

        self._patch = mock.patch.object(DeviceBlochConverter, "eigensolve", counted)
        self._patch.start()
        return self

    def __exit__(self, *exc_info):
        self._patch.stop()


class RoutedFeaturesTests(unittest.TestCase):
    def test_kpoint_post_processing_stays_on_the_reference_solver_by_name(self) -> None:
        from parsec_python.acceleration.cli import _reference_only_features

        cases = (
            (MP.format(nz=1) + NSCF, "a non-self-consistent run (nscf_kpoints)"),
            (MP.format(nz=1) + "Create_DOS: .true.\nYLM_DOS_L: 1\n", "the density of states (Create_DOS)"),
            (MP.format(nz=2) + "Output_GW: .true.\n", "the BerkeleyGW files WFN, RHO and VXC (Output_GW)"),
            (
                MP.format(nz=1) + "begin bandstruc\n1 0 0 0 0.5 0 0 GX\nend bandstruc\n",
                "band structure along a k-path (bandstruc)",
            ),
            (
                MP.format(nz=1) + "Output_Eigenvalues: .true.\nSave_Intermediate_Eigenvalues: .true.\n",
                "the k-point levels of every SCF iteration in eigen.dat (Save_Intermediate_Eigenvalues)",
            ),
        )
        for extra, feature in cases:
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as folder:
                translation = parse_parsec_input(_write(Path(folder), extra))
                features = _reference_only_features(translation, accelerated_backend_selected=True)
                self.assertIn(feature, features)
                # every one is a k-point run, which the reference drivers put on the device workers
                self.assertIn("k-point sampling (Monkhorst-Pack or explicit list)", features)


class DeviceRouteTests(unittest.TestCase):
    def test_nscf_eigensolves_run_on_the_device_and_equal_the_host_route(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            scf = _write(directory, MP.format(nz=2))
            _run(scf, "cpu")
            levels = {}
            for backend in ("cpu",) + DEVICE_BACKENDS:
                path = _write(directory, MP.format(nz=2) + NSCF)
                with CountingEigensolve() as counter:
                    _run(path, backend, "--log", f"nscf_{backend}.out")
                self.assertEqual(counter.calls, 0 if backend == "cpu" else 3, backend)
                with np.load(directory / "parsec_python_nscf.npz") as archive:
                    levels[backend] = np.asarray(archive["nscf_eigenvalues_ry"])
            for backend in DEVICE_BACKENDS:
                with self.subTest(backend=backend):
                    self.assertEqual(levels[backend].shape, (1, 3, 5))
                    np.testing.assert_allclose(levels[backend], levels["cpu"], atol=1e-6)

    def test_ylm_dos_eigen_dat_and_bands_of_a_kpoint_scf_on_the_device(self) -> None:
        extra = (
            MP.format(nz=1)
            + "Create_DOS: .true.\nDOS_Pnum: 200\nYLM_DOS_L: 1\nylm_dos_cutoff: 1.2\n"
            + "Output_Eigenvalues: .true.\nSave_Intermediate_Eigenvalues: .true.\n"
            + "Output_All_States: .true.\n"
        )
        outputs = {}
        for backend in ("cpu",) + DEVICE_BACKENDS:
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                _run(_write(directory, extra), backend)
                with np.load(directory / "parsec_python_results.npz") as archive:
                    k_count = int(archive["n_kpoints"])
                outputs[backend] = (
                    _dos_pairs(directory / "DOS"),
                    (directory / "eigen.dat").read_text().splitlines(),
                    k_count,
                )
        reference_dos, reference_eigen, k_count = outputs["cpu"]
        for backend in DEVICE_BACKENDS:
            with self.subTest(backend=backend):
                dos, eigen, device_k_count = outputs[backend]
                self.assertEqual(device_k_count, k_count)
                self.assertEqual(dos.shape, reference_dos.shape)
                # the mesh follows the lowest and highest level (eV); the values of the same
                # tetrahedra from levels and projections that agree to the SCF's convergence
                np.testing.assert_allclose(dos[:, 0], reference_dos[:, 0], atol=1e-5)
                scale = float(np.max(np.abs(reference_dos[:, 1])))
                self.assertLess(
                    float(np.sum(np.abs(dos[:, 1] - reference_dos[:, 1]))) / float(np.sum(np.abs(reference_dos[:, 1]))),
                    1e-3,
                )
                self.assertLess(float(np.max(np.abs(dos[:, 1] - reference_dos[:, 1]))), 1e-2 * scale)
                # a header and one block (k-points x 4 states) of every SCF iteration; the
                # levels of the last one to the SCF's convergence
                block = k_count * 4
                for lines in (eigen, reference_eigen):
                    self.assertEqual((len(lines) - 1) % block, 0)
                    self.assertGreater(len(lines), 1 + block)
                last = np.asarray([float(line.split()[1]) for line in eigen[-block:]])
                host_last = np.asarray([float(line.split()[1]) for line in reference_eigen[-block:]])
                np.testing.assert_allclose(last, host_last, atol=1e-6)

    def test_output_gw_of_a_kpoint_scf_on_the_device(self) -> None:
        from parsec_python.Output.berkeleygw import read_berkeleygw_file

        files = {}
        for backend in ("cpu",) + DEVICE_BACKENDS:
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                _run(_write(directory, MP.format(nz=2) + "Output_GW: .true.\n"), backend)
                files[backend] = {kind: read_berkeleygw_file(directory / kind) for kind in ("RHO", "VXC", "WFN")}
        reference = files["cpu"]
        for backend in DEVICE_BACKENDS:
            with self.subTest(backend=backend):
                for kind in ("RHO", "VXC"):
                    np.testing.assert_array_equal(files[backend][kind].gvectors, reference[kind].gvectors)
                    np.testing.assert_allclose(files[backend][kind].data, reference[kind].data, atol=1e-6)
                wfn, host = files[backend]["WFN"], reference["WFN"]
                np.testing.assert_allclose(wfn.header.energies, host.header.energies, atol=1e-6)
                levels = np.asarray(host.header.energies)[0]
                for k, (mine, theirs) in enumerate(zip(wfn.bands, host.bands)):
                    np.testing.assert_array_equal(wfn.k_gvectors[k], host.k_gvectors[k])
                    # the lowest band (H2 bonding state) is non-degenerate: equal up to a phase
                    self.assertGreater(levels[k, 1] - levels[k, 0], 1e-3)
                    overlap = abs(np.vdot(theirs[0, 0], mine[0, 0]))
                    self.assertGreater(overlap, 1.0 - 1e-8)


if __name__ == "__main__":
    unittest.main()
