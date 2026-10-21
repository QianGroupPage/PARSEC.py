"""The k-point eigenproblems with the basis distributed over MPI ranks.

``mpi_check.py`` runs under ``mpirun`` (ranks on the CPU, the device code on the host) and the results
are compared here with the single-process solver:

* the Hamiltonian applied to a block, rank by rank, equals the serial operator to round-off (halo
  rows of the stencils, projector and spin-orbit overlaps summed over the ranks);
* an SCF of each driver gives the energy of the serial run.

Skipped where mpi4py or mpirun is missing, or with ``PARSEC_TEST_MPI=0``.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:  # pragma: no cover - environment dependent
    import mpi4py  # noqa: F401

    # PARSEC_TEST_MPI=0 leaves out the tests that start mpirun (about a minute)
    HAVE_MPI = shutil.which("mpirun") is not None and os.environ.get("PARSEC_TEST_MPI", "1") != "0"
except ImportError:  # pragma: no cover
    HAVE_MPI = False

from parsec_python.acceleration.tests import test_kpoint_parallel as fixtures

_SRC = Path(__file__).resolve().parents[3]
_CRITERION = 1.0e-4


def _mpirun(ranks: int, *arguments: str) -> subprocess.CompletedProcess:
    command = [
        "mpirun", "-n", str(ranks), "--oversubscribe", sys.executable, "-m",
        "parsec_python.acceleration.tests.mpi_check", *arguments,
    ]
    environment = {**os.environ, "PYTHONPATH": f"{_SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    environment.pop("PARSEC_MPI", None)
    return subprocess.run(command, capture_output=True, text=True, env=environment, timeout=900)


def _results(process: subprocess.CompletedProcess) -> dict[str, dict]:
    """The ``RESULT`` lines of an ``scf`` run, by driver."""
    found = {}
    for line in process.stdout.splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT "):])
            found[payload["driver"]] = payload
    if not found:
        raise AssertionError(f"no result\nstdout:\n{process.stdout[-1500:]}\nstderr:\n{process.stderr[-2500:]}")
    return found


@unittest.skipUnless(HAVE_MPI, "needs mpi4py and mpirun")
class DistributedOperatorTests(unittest.TestCase):
    def test_application_matches_the_serial_operator_on_three_ranks(self) -> None:
        process = _mpirun(3, "operators")
        self.assertEqual(process.returncode, 0, process.stdout[-1500:] + process.stderr[-2500:])
        for name in ("scalar k-point", "spinor field + soc", "spinor delta + soc", "isolated non-collinear"):
            self.assertIn(f"ok {name}", process.stdout)


@unittest.skipUnless(HAVE_MPI, "needs mpi4py and mpirun")
class DistributedTaskTests(unittest.TestCase):
    def test_independent_tasks_are_dealt_out_and_shared(self) -> None:
        process = _mpirun(3, "map")
        self.assertEqual(process.returncode, 0, process.stdout[-1500:] + process.stderr[-2500:])
        self.assertIn("ok map_tasks [0, 1, 2, 3, 4, 5, 6]", process.stdout)


_CLUSTER_INPUT = """\
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.6
Expansion_Order: 6
States_Num: 20
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 80
Convergence_Criterion: 2.0e-3
Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff
Non_Collinear_magnetism: .true.
Atom_Types_Num: 1
Total_Atom_Num: 1
Atom_Type: Au
SO_PSP: .false.
Local_Component: s
Initial_Spin_Polarization: 0.3
Begin atom_coord
   0.0 0.0 0.0
End atom_coord
begin Initial_NCL_Moment
0 3 4
end Initial_NCL_Moment
Correlation_Type: CA
"""


@unittest.skipUnless(HAVE_MPI, "needs mpi4py and mpirun")
class DistributedCommandLineTests(unittest.TestCase):
    def test_only_rank_zero_writes_and_the_energy_is_the_serial_one(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "Au_POTRE.DAT").write_bytes(fixtures._AU_POTRE.read_bytes())
            (directory / "parsec.in").write_text(_CLUSTER_INPUT, encoding="utf-8")
            environment = {
                **os.environ,
                "PYTHONPATH": f"{_SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
                "PARSEC_KPOINT_BACKEND": "numpy-device",
            }
            environment.pop("PARSEC_MPI", None)
            process = subprocess.run(
                ["mpirun", "-n", "2", "--oversubscribe", sys.executable, "-m", "parsec_python.main",
                 "parsec.in", "--mpi", "--device", "cpu"],
                cwd=directory, capture_output=True, text=True, env=environment, timeout=900,
            )
            self.assertEqual(process.returncode, 0, process.stdout[-1500:] + process.stderr[-2500:])
            files = sorted(path.name for path in directory.iterdir())
            self.assertEqual(files, ["Au_POTRE.DAT", "parsec.in", "parsec.out", "parsec_python_results.npz"])
            log = (directory / "parsec.out").read_text(encoding="utf-8")
            self.assertIn("Self-consistency convergence achieved.", log)
            self.assertIn("Eigensolver residual max", log)
            self.assertEqual(process.stdout.count("MPI: 2 ranks"), 1)
            energy = float([l for l in log.splitlines() if l.startswith("Total Energy")][-1].split()[3])
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_BACKEND": "numpy-device"}):
            host = fixtures.IsolatedNonCollinearDeviceTests()._energy({"PARSEC_KPOINT_BACKEND": "cpu"})
        self.assertAlmostEqual(energy, host, delta=3.0e-3)


@functools.lru_cache(maxsize=None)
def _serial(name: str):
    driver, flags, soc = fixtures.DRIVERS[name]
    environment = {"PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1"}
    with mock.patch.dict(os.environ, environment):
        return driver(fixtures._system(soc=soc, criterion=_CRITERION, **flags), fixtures._K, fixtures._WEIGHTS)


@unittest.skipUnless(HAVE_MPI, "needs mpi4py and mpirun")
class DistributedSCFTests(unittest.TestCase):
    def test_each_driver_reproduces_the_serial_energy(self) -> None:
        names = ("scalar", "soc", "noncollinear")
        process = _mpirun(3, "scf", ",".join(names), str(_CRITERION))  # one start-up for the three drivers
        distributed = _results(process)
        for name in names:
            with self.subTest(driver=name):
                serial = _serial(name)
                self.assertTrue(distributed[name]["converged"])
                self.assertEqual(distributed[name]["ranks"], 3)
                self.assertAlmostEqual(distributed[name]["energy"], serial.energies.total, delta=5.0e-4)
                self.assertAlmostEqual(distributed[name]["fermi"], serial.fermi_level, delta=3.0e-3)

    def test_one_rank_is_the_serial_solver_through_the_mpi_code(self) -> None:
        single = _results(_mpirun(1, "scf", "noncollinear", str(_CRITERION)))["noncollinear"]
        self.assertAlmostEqual(single["energy"], _serial("noncollinear").energies.total, delta=5.0e-4)


if __name__ == "__main__":
    unittest.main()
