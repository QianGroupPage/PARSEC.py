"""The device Bloch operator and filter (NumPy as the array backend).

The GPU path applies exactly this code with CuPy arrays, so these tests pin
its arithmetic: the operator equals the host Hamiltonians (scalar, spinor
with spin-orbit and a Zeeman field, spinor with a collinear term), the device
Chebyshev recurrence equals the CPU one (and is picked up by the CHEBFF,
CHEBDAV and SUBSPACE entry points), and a whole SCF or band run routed
through the device code gives the CPU result.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from parsec_python.acceleration.Hamiltonian.device_bloch import (
    DeviceBlochConverter,
    NumpyArrayBackend,
    resolve_kpoint_backend,
)
from parsec_python.Eigensolvers import solve_eigval
from parsec_python.Eigensolvers.chebyshev import chebff_filter, chebyshev_filter, subspace_filter
from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.Hamiltonian.kpoint_spinor_operator import KPointSpinorKohnShamHamiltonian
from parsec_python.SCF.kpoints_soc import _eigval_settings  # noqa: F401  (import check)
from parsec_python.tests.unit.test_hamiltonian_fast_paths import (
    _PeriodicFixture,
    _random_spinor,
)
from parsec_python.acceleration.tests import test_kpoint_parallel as parallel_tests


class DeviceOperatorTests(_PeriodicFixture):
    def setUp(self) -> None:
        self.converter = DeviceBlochConverter(NumpyArrayBackend())
        self.scalar = KPointKohnShamHamiltonian(
            self.laplacian, self.gradient, self.potential, self.nonlocal_k, self.k_point
        )
        rng = np.random.default_rng(2)
        self.field = 0.2 * rng.normal(size=(3, self.grid.size))
        self.delta = 0.1 * rng.normal(size=self.grid.size)

    def _spinors(self):
        return {
            "field + soc": KPointSpinorKohnShamHamiltonian(
                self.scalar, self.soc, xc_field=self.field
            ),
            "delta + soc": KPointSpinorKohnShamHamiltonian(
                self.scalar, self.soc, xc_delta=self.delta
            ),
            "plain soc": KPointSpinorKohnShamHamiltonian(self.scalar, self.soc),
            "noncollinear without soc": KPointSpinorKohnShamHamiltonian(
                self.scalar, (), xc_field=self.field
            ),
        }

    def test_scalar_operator_matches_the_host(self) -> None:
        device = self.converter.operator(self.scalar)
        block = _random_spinor(self.grid.size, 5, seed=1)[: self.grid.size]
        np.testing.assert_allclose(device.matmat(block), self.scalar.apply(block), atol=1e-11)
        vector = block[:, 0]
        np.testing.assert_allclose(device.matmat(vector), self.scalar.apply(vector), atol=1e-11)

    def test_spinor_operators_match_the_host(self) -> None:
        spinor = _random_spinor(self.grid.size, 4, seed=3)
        for name, host in self._spinors().items():
            with self.subTest(name):
                device = self.converter.operator(host)
                np.testing.assert_allclose(device.matmat(spinor), host.apply(spinor), atol=1e-10)
                self.assertEqual(device.dimension, 2 * self.grid.size)

    def test_device_operator_is_hermitian(self) -> None:
        host = self._spinors()["field + soc"]
        operator = self.converter.operator(host).as_linear_operator()
        u = _random_spinor(self.grid.size, 1, seed=4)[:, 0]
        v = _random_spinor(self.grid.size, 1, seed=5)[:, 0]
        self.assertLess(abs(np.vdot(u, operator @ v) - np.vdot(operator @ u, v)), 1e-9)

    def test_static_matrices_are_uploaded_once_per_host_object(self) -> None:
        first = self.converter.operator(self.scalar)
        second = self.converter.operator(
            KPointKohnShamHamiltonian(
                self.laplacian, self.gradient, 2.0 * self.potential, self.nonlocal_k, self.k_point
            )
        )
        self.assertIs(first.laplacian, second.laplacian)
        self.assertIs(first.projector_forward, second.projector_forward)
        self.assertFalse(np.allclose(first.diagonal, second.diagonal))  # the potential did change

    def test_chebyshev_filter_runs_on_the_device_and_matches_the_cpu(self) -> None:
        host = self._spinors()["field + soc"]
        cpu_operator = host.as_linear_operator()
        device_operator = self.converter.operator(host).as_linear_operator()
        self.assertTrue(hasattr(device_operator, "device_chebyshev_filter"))
        block = _random_spinor(self.grid.size, 6, seed=6)
        arguments = dict(degree=9, lower_bound=-1.5, upper_bound=60.0, reference_eigenvalue=-1.5)
        expected, sigma = chebyshev_filter(cpu_operator, block, return_final_sigma=True, **arguments)
        result, device_sigma = chebyshev_filter(
            device_operator, block, return_final_sigma=True, **arguments
        )
        scale = np.abs(expected).max()
        np.testing.assert_allclose(result, expected, atol=1e-9 * scale)
        self.assertAlmostEqual(device_sigma, sigma, places=12)
        # carried sigma and a single vector
        again = chebyshev_filter(device_operator, block[:, 0], initial_sigma=sigma, **arguments)
        reference = chebyshev_filter(cpu_operator, block[:, 0], initial_sigma=sigma, **arguments)
        np.testing.assert_allclose(again, reference, atol=1e-9 * np.abs(reference).max())

    def test_chebff_and_subspace_filters_use_the_device_filter(self) -> None:
        host = self._spinors()["delta + soc"]
        cpu_operator = host.as_linear_operator()
        device_operator = self.converter.operator(host).as_linear_operator()
        block = _random_spinor(self.grid.size, 8, seed=7)
        for name, call in (
            (
                "chebff",
                lambda op: chebff_filter(
                    op, block, degree=8, lower_bound=-1.0, upper_bound=50.0,
                    reference_eigenvalue=-1.0, block_size=3, reset_recurrence_per_block=False,
                ),
            ),
            (
                "subspace",
                lambda op: subspace_filter(
                    op, block, degree=8, degree_delta=2, lower_bound=-1.0, upper_bound=50.0,
                    block_size=3, reset_recurrence_per_block=False,
                ),
            ),
        ):
            with self.subTest(name):
                expected = call(cpu_operator)
                result = call(device_operator)
                np.testing.assert_allclose(result, expected, atol=1e-9 * np.abs(expected).max())

    def test_full_chebff_solve_gives_the_same_eigenvalues(self) -> None:
        from parsec_python.Eigensolvers import ChebFFSettings, EigvalSettings

        host = self._spinors()["field + soc"]
        settings = EigvalSettings(
            safety_buffer=4,
            initial_method="chebff",
            chebff=ChebFFSettings(polynomial_degree=20, filter_cycles=6, lanczos_steps=8, block_size=5),
        )
        cpu = solve_eigval(host.as_linear_operator(), 8, settings=settings)
        device = solve_eigval(
            self.converter.operator(host).as_linear_operator(), 8, settings=settings
        )
        np.testing.assert_allclose(device.eigenvalues, cpu.eigenvalues, atol=1e-7)


class BackendSelectionTests(unittest.TestCase):
    def test_cpu_is_the_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_KPOINT_BACKEND", None)
            self.assertEqual(resolve_kpoint_backend().name, "cpu")

    def test_numpy_device_is_a_device_backend(self) -> None:
        plan = resolve_kpoint_backend("numpy-device")
        self.assertEqual((plan.name, plan.uses_device), ("numpy-device", True))
        self.assertIsNotNone(plan.converter(0))

    def test_auto_falls_back_to_cpu_without_cupy_and_cupy_is_an_error(self) -> None:
        try:
            import cupy  # noqa: F401
        except Exception:
            self.assertEqual(resolve_kpoint_backend("auto").name, "cpu")
            with self.assertRaises(Exception):
                resolve_kpoint_backend("cupy")
        else:  # pragma: no cover - only on a GPU machine
            self.assertEqual(resolve_kpoint_backend("auto").name, "cupy")

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_kpoint_backend("tpu")


class DeviceBackendDriversTests(unittest.TestCase):
    """Whole SCF runs through the device code (host double, two worker threads).

    ``PARSEC_KPOINT_SOLVER=host`` is the former route (host eigensolver, device filter): the same
    computation as the CPU, so it matches to round-off.
    """

    def test_scf_drivers_match_the_cpu_with_the_host_solver(self) -> None:
        for name in ("spin_polarized", "soc", "noncollinear"):
            with self.subTest(driver=name):
                serial = parallel_tests._run(name, "1")
                driver, flags, soc = parallel_tests.DRIVERS[name]
                with mock.patch.dict(
                    os.environ,
                    {
                        "PARSEC_KPOINT_BACKEND": "numpy-device",
                        "PARSEC_KPOINT_SOLVER": "host",
                        "PARSEC_KPOINT_WORKERS": "2",
                    },
                ):
                    device = driver(
                        parallel_tests._system(soc=soc, **flags),
                        parallel_tests._K,
                        parallel_tests._WEIGHTS,
                    )
                parallel_tests._assert_same(self, serial, device)


def _converged(name: str, environment: dict, **overrides):
    driver, flags, soc = parallel_tests.DRIVERS[name]
    with mock.patch.dict(os.environ, environment):
        return driver(
            parallel_tests._system(soc=soc, criterion=2.0e-5, **flags, **overrides),
            parallel_tests._K,
            parallel_tests._WEIGHTS,
        )


class DeviceResidentSolverTests(unittest.TestCase):
    """The device-resident eigensolver (CHEBFF or CHEBDAV first solve, then SUBSPACE steps on the
    device) is a different path to the same converged SCF: compare tightly converged runs."""

    CPU = {"PARSEC_KPOINT_BACKEND": "cpu", "PARSEC_KPOINT_WORKERS": "1"}
    DEVICE = {"PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "2"}

    def _compare(self, name: str, device_environment: dict, **overrides) -> None:
        serial = _converged(name, self.CPU, **overrides)
        device = _converged(name, device_environment, **overrides)
        self.assertTrue(serial.converged and device.converged)
        self.assertAlmostEqual(device.energies.total, serial.energies.total, delta=5.0e-5)
        self.assertAlmostEqual(device.fermi_level, serial.fermi_level, delta=5.0e-4)
        if getattr(serial, "eigenvalues", None) is not None:
            np.testing.assert_allclose(device.eigenvalues, serial.eigenvalues, atol=1.0e-3)

    def test_chebff_first_solve(self) -> None:
        for name in ("scalar", "soc", "noncollinear"):
            with self.subTest(driver=name):
                self._compare(name, self.DEVICE)

    def test_chebdav_first_solve(self) -> None:
        for name in ("scalar", "soc"):
            with self.subTest(driver=name):
                self._compare(name, self.DEVICE, method="chebdav")

    def test_subspaces_kept_on_the_host_between_solves(self) -> None:
        for name in ("scalar", "soc"):
            with self.subTest(driver=name):
                self._compare(name, {**self.DEVICE, "PARSEC_KPOINT_STATE_STORAGE": "host"})

    def test_solver_stats_are_collected(self) -> None:
        from parsec_python.SCF import kpoint_solver

        with mock.patch.dict(os.environ, self.DEVICE):
            system = parallel_tests._system(criterion=2.0e-3)
            seen = {}
            original = kpoint_solver._parallel_solver

            def capture(job, **options):
                solver = original(job, **options)
                close = solver.close

                def close_after_stats():
                    seen["stats"] = solver.stats()  # the pool is gone after close
                    close()

                solver.close = close_after_stats
                return solver

            kpoint_solver.set_parallel_backend(capture, kpoint_solver._parallel_mapper)
            try:
                from parsec_python.SCF.kpoints import run_scf_kpoints

                run_scf_kpoints(system, parallel_tests._K, parallel_tests._WEIGHTS)
                stats = seen["stats"]
            finally:
                from parsec_python.acceleration.SCF.kpoint_parallel import install

                install()
        self.assertGreater(stats.get("subspace_steps", 0), 0)
        self.assertEqual(stats["first_solves"], len(parallel_tests._K))


if __name__ == "__main__":
    unittest.main()
