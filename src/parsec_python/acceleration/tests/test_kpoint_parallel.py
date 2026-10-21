"""Parallel k-point solution reproduces the serial SCF.

Each k-point (and spin channel) is an independent eigenproblem, so spreading
them over worker threads must give the same eigenvalues exactly
and the same density, energy and moments to round-off (the density sums its
k-point contributions in a different order).  All five periodic drivers are
checked on a small Au cell with four k-points.
"""

from __future__ import annotations

import dataclasses
import functools
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
from parsec_python.acceleration.Parallel import available_cpus, resolve_workers
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


_PREPARED: dict[bool, object] = {}


def _system(**flags):
    """A prepared Au cell with the settings given.  The preparation (potentials, projectors, Ewald) does
    not depend on them, so it is made once per pseudopotential flavour and each call puts its own
    input around it."""
    soc = flags.pop("soc", False)
    criterion = flags.pop("criterion", 2.0e-3)
    method = flags.pop("method", "chebff")
    problem = SinglePointInput(
        atoms=[Atom("Au", [_BOX / 2] * 3, [0.0, 0.0, 1.0])],
        pseudopotentials={
            "Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=soc, initial_spin_polarization=0.3)
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=_BOX * np.eye(3)),
        scf=SCFSettings(
            max_iterations=80, number_of_states=20, convergence_criterion=criterion, **flags
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method=method, tolerance=1.0e-6),
    )
    base = _PREPARED.get(soc)
    if base is None:
        base = _PREPARED[soc] = prepare_periodic_single_point(problem)
    return dataclasses.replace(base, input=problem)


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


@functools.lru_cache(maxsize=None)
def _recorded(name: str, workers: str):
    """``(result, iteration records)`` of a driver with ``workers`` workers; the runs are shared by the tests."""
    driver, flags, soc = DRIVERS[name]
    steps: list = []
    with mock.patch.dict(os.environ, {"PARSEC_KPOINT_WORKERS": workers}):
        result = driver(_system(soc=soc, **flags), _K, _WEIGHTS, callback=steps.append)
    return result, steps


def _run(name: str, workers: str):
    return _recorded(name, workers)[0]


@functools.lru_cache(maxsize=None)
def _run_reference(name: str):
    """The reference drivers with no parallel backend installed: the serial definition."""
    from parsec_python.acceleration.SCF.kpoint_parallel import install
    from parsec_python.SCF import kpoint_solver

    kpoint_solver.set_parallel_backend(None, None)
    try:
        driver, flags, soc = DRIVERS[name]
        return driver(_system(soc=soc, **flags), _K, _WEIGHTS)
    finally:
        install()


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


class ParallelDriversMatchSerialTests(unittest.TestCase):
    def test_each_driver_with_two_workers(self) -> None:
        for name in DRIVERS:
            with self.subTest(driver=name):
                _assert_same(self, _run(name, "1"), _run(name, "2"))

    def test_pooled_solver_matches_the_reference_serial_solver(self) -> None:
        for name in DRIVERS:
            with self.subTest(driver=name):
                _assert_same(self, _run_reference(name), _run(name, "2"))

    def test_three_workers_leave_an_uneven_split_of_four_k_points(self) -> None:
        _assert_same(self, _run("soc", "1"), _run("soc", "3"))


class WorkerCountTests(unittest.TestCase):
    def test_auto_stays_serial_for_one_task_or_a_small_problem(self) -> None:
        self.assertEqual(resolve_workers("auto", 1), 1)
        self.assertEqual(resolve_workers("auto", 8, problem_size=1000), 1)

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


class KPointTimingTests(unittest.TestCase):
    """The k-point drivers record a stage table and what the workers measured."""

    def _check(self, result, workers: int) -> None:
        timings = result.timings
        self.assertGreater(timings.total_seconds, 0.0)
        self.assertGreater(timings.diagonalization_seconds, 0.0)
        for stage in (
            timings.initial_hartree_seconds,
            timings.occupations_density_seconds,
            timings.hartree_seconds,
            timings.xc_seconds,
            timings.mixing_energy_seconds,
        ):
            self.assertGreaterEqual(stage, 0.0)
        staged = (
            timings.initial_hartree_seconds
            + timings.initial_xc_seconds
            + timings.diagonalization_seconds
            + timings.occupations_density_seconds
            + timings.hartree_seconds
            + timings.xc_seconds
            + timings.mixing_energy_seconds
            + timings.detail["solver_setup"]
        )
        self.assertLessEqual(staged, timings.total_seconds * 1.001)
        self.assertEqual(timings.detail["workers"], workers)
        # Bloch projectors are built on first use; the eigensolver takes the rest of the solve
        self.assertGreater(timings.detail["worker_hamiltonian_seconds"], 0.0)
        self.assertGreater(timings.detail["worker_solve_seconds"], 0.0)
        self.assertGreater(timings.detail["worker_density_seconds"], 0.0)

    def test_scalar_driver_fills_run_timings_and_iteration_seconds(self) -> None:
        result = _run("scalar", "2")
        self._check(result, 2)
        self.assertEqual(len(result.history), result.iterations)
        self.assertGreater(result.history[0].diagonalization_seconds, 0.0)
        self.assertGreater(result.history[0].total_seconds, 0.0)

    def test_spin_polarized_driver_fills_run_timings(self) -> None:
        result, iterations = _recorded("spin_polarized", "2")
        self._check(result, 2)
        self.assertTrue(all(step.total_seconds > 0.0 for step in iterations))

class GammaSpinRoutingTests(unittest.TestCase):
    """A Gamma-only periodic spin run goes through the k-point driver when a device is in use."""

    _INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape
Grid_Spacing: 0.7
Expansion_Order: 6
Spin_Polarization: .true.
States_Num: 12
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 60
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

    def _log(self, uses_device: bool) -> str:
        import tempfile

        from parsec_python.acceleration.SCF.kpoint_parallel import install
        from parsec_python.cli import main as cli_main
        from parsec_python.SCF import kpoint_solver

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "Au_POTRE.DAT").write_bytes(_AU_POTRE.read_bytes())
            (directory / "parsec.in").write_text(self._INPUT, encoding="utf-8")
            kpoint_solver.set_parallel_backend(
                kpoint_solver._parallel_solver,
                kpoint_solver._parallel_mapper,
                uses_device=lambda: uses_device,
            )
            try:
                self.assertEqual(cli_main([str(directory / "parsec.in"), "--quiet", "--no-archive"]), 0)
            finally:
                install()
            return (directory / "parsec.out").read_text(encoding="utf-8")

    def test_gamma_spin_uses_the_kpoint_driver_only_on_a_device(self) -> None:
        on_device = self._log(True)
        on_host = self._log(False)
        self.assertIn("K-point solver detail", on_device)
        self.assertNotIn("K-point solver detail", on_host)
        # the log says where the eigenproblems of the k-point driver run
        self.assertIn("Execution resources (k-point eigensolver)", on_device)
        self.assertNotIn("Execution resources (k-point eigensolver)", on_host)
        energy = [float(line.split()[3]) for line in on_device.splitlines() if "Total Energy =" in line and "iter" not in line]
        reference = [float(line.split()[3]) for line in on_host.splitlines() if "Total Energy =" in line and "iter" not in line]
        self.assertAlmostEqual(energy[-1], reference[-1], delta=2.0e-3)


class IsolatedNonCollinearDeviceTests(unittest.TestCase):
    """A non-collinear cluster hands its eigenproblem to the device k-point worker (one task)."""

    def _energy(self, environment: dict) -> float:
        import tempfile

        from parsec_python.cli import main as cli_main
        from parsec_python.tests.unit.test_noncollinear import _CLUSTER_INPUT

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "Au_POTRE.DAT").write_bytes(_AU_POTRE.read_bytes())
            (directory / "parsec.in").write_text(
                _CLUSTER_INPUT.format(extra="begin Initial_NCL_Moment\n0 3 4\nend Initial_NCL_Moment\n"),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, environment):
                self.assertEqual(cli_main([str(directory / "parsec.in"), "--quiet", "--no-archive"]), 0)
            log = (directory / "parsec.out").read_text(encoding="utf-8")
        self.log = log
        return float(
            [line for line in log.splitlines() if line.startswith("Total Energy")][-1].split()[3]
        )

    def test_device_route_reproduces_the_host_energy(self) -> None:
        host = self._energy({"PARSEC_KPOINT_BACKEND": "cpu"})
        device = self._energy({"PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1"})
        self.assertAlmostEqual(device, host, delta=2.0e-3)
        # a cluster of spinors: complex wavefunctions, and the log says where the eigensolver ran
        self.assertIn("WAVEFUNCTIONS ARE COMPLEX!", self.log)
        self.assertIn("Execution resources (k-point eigensolver)", self.log)
        self.assertIn("device-resident Ritz subspaces", self.log)


class StageTimingAndResidualTests(unittest.TestCase):
    """The spinor drivers carry a stage table; every driver reports the eigenpair residual."""

    def test_spinor_drivers_fill_the_stage_table(self) -> None:
        for name in ("soc", "soc_spin_polarized", "noncollinear"):
            with self.subTest(driver=name):
                result = _run(name, "2")
                timings = result.timings
                for stage in ("setup_potentials", "diagonalization", "density", "hartree", "xc", "mixing_energy"):
                    self.assertGreater(timings[stage], 0.0, stage)
                self.assertEqual(timings["eig_workers"], 2)
                self.assertGreater(timings["eig_worker_solve_seconds"], 0.0)

    def test_every_driver_reports_a_residual(self) -> None:
        for name in ("scalar", "spin_polarized", "soc", "noncollinear"):
            with self.subTest(driver=name):
                seen = _recorded(name, "2")[1]
                finite = [step.eigen_residual_max for step in seen if np.isfinite(step.eigen_residual_max)]
                # the first CHEBFF solve has no residual test; the later SUBSPACE steps do
                self.assertGreater(len(finite), 0)
                self.assertTrue(all(value > 0.0 for value in finite))

    def test_device_route_measures_residuals_each_solve_and_can_stop(self) -> None:
        environment = {"PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1"}
        seen = []
        with mock.patch.dict(os.environ, environment):
            result = run_scf_kpoints(_system(), _K, _WEIGHTS, callback=seen.append)
        detail = result.timings.detail  # the device route's counters
        self.assertEqual(detail["first_solves"], len(_K))
        self.assertGreater(detail["filter_seconds"], 0.0)
        self.assertGreater(detail["worker_upload_seconds"], 0.0)
        self.assertTrue(all(np.isfinite(step.eigen_residual_max) for step in seen))
        self.assertLess(seen[-1].eigen_residual_max, seen[0].eigen_residual_max)
        seen = []
        with mock.patch.dict(os.environ, {**environment, "PARSEC_KPOINT_RESIDUALS": "0"}):
            result = run_scf_kpoints(_system(), _K, _WEIGHTS, callback=seen.append)
        self.assertTrue(all(np.isnan(step.eigen_residual_max) for step in seen))
        self.assertNotIn("residual_seconds", result.timings.detail)
