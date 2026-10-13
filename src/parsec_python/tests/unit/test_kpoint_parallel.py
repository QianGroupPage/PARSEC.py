"""Parallel k-point solution reproduces the serial SCF.

Each k-point (and spin channel) is an independent eigenproblem, so spreading
them over worker processes or threads must give the same eigenvalues exactly
and the same density, energy and moments to round-off (the density sums its
k-point contributions in a different order).  All five periodic drivers are
checked on a small Au cell with four k-points.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Parallel import available_cpus, fork_available, resolve_workers
from parsec_python.SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
from parsec_python.SCF.kpoints_soc import (
    run_self_consistent_soc_kpoints,
    run_self_consistent_soc_kpoints_spin_polarized,
)
from parsec_python.SCF.noncollinear import run_self_consistent_noncollinear_kpoints
from parsec_python.SCF.pbc import prepare_periodic_single_point

_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
_BOX = 8.0
_K = 2.0 * np.pi / _BOX * np.array(
    [[0.25, 0.25, 0.25], [-0.25, 0.25, 0.25], [0.25, -0.25, 0.25], [0.25, 0.25, -0.25]]
)
_WEIGHTS = np.full(4, 0.25)


def _system(**flags):
    soc = flags.pop("soc", False)
    problem = SinglePointInput(
        atoms=[Atom("Au", [_BOX / 2] * 3, [0.0, 0.0, 1.0])],
        pseudopotentials={
            "Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=soc, initial_spin_polarization=0.3)
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=_BOX * np.eye(3)),
        scf=SCFSettings(
            max_iterations=60, number_of_states=20, convergence_criterion=2.0e-3, **flags
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )
    return prepare_periodic_single_point(problem)


DRIVERS = {
    "scalar": (run_scf_kpoints, dict(), False),
    "spin_polarized": (run_scf_kpoints_spin_polarized, dict(spin_polarized=True), False),
    "soc": (run_self_consistent_soc_kpoints, dict(self_consistent_spin_orbit=True), True),
    "soc_spin_polarized": (
        run_self_consistent_soc_kpoints_spin_polarized,
        dict(spin_polarized=True, self_consistent_spin_orbit=True),
        True,
    ),
    "noncollinear": (run_self_consistent_noncollinear_kpoints, dict(noncollinear=True), False),
}


def _run(name: str, workers: str, pool: str = "process"):
    driver, flags, soc = DRIVERS[name]
    with mock.patch.dict(
        os.environ, {"PARSEC_KPOINT_WORKERS": workers, "PARSEC_KPOINT_POOL": pool}
    ):
        return driver(_system(soc=soc, **flags), _K, _WEIGHTS)


def _assert_same(case: unittest.TestCase, serial, parallel) -> None:
    eigenvalues = getattr(serial, "eigenvalues", None)
    if eigenvalues is None:  # collinear spin without spin-orbit
        np.testing.assert_allclose(parallel.eigenvalues_up, serial.eigenvalues_up, atol=1e-9)
        np.testing.assert_allclose(parallel.eigenvalues_down, serial.eigenvalues_down, atol=1e-9)
        np.testing.assert_allclose(parallel.density_up, serial.density_up, atol=1e-8)
    else:
        np.testing.assert_allclose(parallel.eigenvalues, serial.eigenvalues, atol=1e-9)
        np.testing.assert_allclose(parallel.density, serial.density, atol=1e-8)
    case.assertEqual(parallel.iterations, serial.iterations)
    case.assertAlmostEqual(parallel.energies.total, serial.energies.total, places=8)
    case.assertAlmostEqual(parallel.fermi_level, serial.fermi_level, places=9)
    if getattr(serial, "magnetic_moment_vector", None) is not None:
        np.testing.assert_allclose(
            parallel.magnetic_moment_vector, serial.magnetic_moment_vector, atol=1e-7
        )
    if getattr(serial, "state_moments", None) is not None:
        # <S_z> of a single state is gauge dependent inside a degenerate (Kramers) pair:
        # round-off in the BLAS summation order rotates the pair, so compare it loosely
        # (the gauge-invariant quantities above are compared tightly).
        np.testing.assert_allclose(parallel.state_moments, serial.state_moments, atol=1e-5)
        np.testing.assert_allclose(parallel.spinors.shape, serial.spinors.shape)


@unittest.skipUnless(fork_available(), "worker processes need fork")
class ParallelDriversMatchSerialTests(unittest.TestCase):
    def test_each_driver_with_two_worker_processes(self) -> None:
        for name in DRIVERS:
            with self.subTest(driver=name):
                _assert_same(self, _run(name, "1"), _run(name, "2"))

    def test_three_workers_leave_an_uneven_split_of_four_k_points(self) -> None:
        _assert_same(self, _run("soc", "1"), _run("soc", "3"))

    def test_thread_pool_matches_for_a_spinor_and_a_spin_channel_driver(self) -> None:
        for name in ("soc", "spin_polarized"):
            with self.subTest(driver=name):
                _assert_same(self, _run(name, "1"), _run(name, "2", pool="thread"))


class WorkerCountTests(unittest.TestCase):
    def test_auto_stays_serial_for_one_task_or_a_small_problem(self) -> None:
        self.assertEqual(resolve_workers("auto", 1), 1)
        self.assertEqual(resolve_workers("auto", 8, problem_size=1000), 1)

    @unittest.skipUnless(fork_available(), "worker processes need fork")
    def test_explicit_count_is_capped_by_the_number_of_tasks(self) -> None:
        self.assertEqual(resolve_workers(16, 4), 4)
        self.assertEqual(resolve_workers(1, 4), 1)
        big = resolve_workers("auto", 64, problem_size=10**6)
        self.assertEqual(big, max(1, min(64, available_cpus() // 2)))

    def test_environment_variable_selects_the_count(self) -> None:
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_WORKERS": "1"}):
            self.assertEqual(resolve_workers(None, 8, problem_size=10**6), 1)


if __name__ == "__main__":
    unittest.main()
