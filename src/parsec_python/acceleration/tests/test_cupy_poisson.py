"""Parity tests for the optional device-resident Hartree Poisson solver."""

from __future__ import annotations

from contextlib import contextmanager
import gc
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.sparse as sp

import parsec_python.acceleration.backends.cupy as cupy_runtime
from parsec_python.acceleration.backends.cupy import (
    CuPyHamiltonian,
    cupy_available,
)
from parsec_python.acceleration.Hartree.cupy_poisson import (
    CuPyPoissonSolver,
    build_boundary_corrected_rhs,
    solve_hartree_cupy,
)
from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree import (
    DirectCoulombBoundary,
    HartreeResult,
    MultipoleExpansion,
    solve_hartree,
)
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.models import GridSettings, HartreeSettings


REAL_CUDA_AVAILABLE = cupy_available()


class _FakeStream:
    def synchronize(self) -> None:
        return None


class _FakeRuntime:
    @staticmethod
    def getDeviceCount() -> int:
        return 1


class _FakeDevice:
    id = 0


def _fake_cupy():
    """Small NumPy API shim exercising the complete GPU control flow in CI."""

    return SimpleNamespace(
        __version__="poisson-test-shim",
        array=np.array,
        asarray=np.asarray,
        asnumpy=np.asarray,
        dtype=np.dtype,
        float64=np.float64,
        linalg=np.linalg,
        vdot=np.vdot,
        zeros_like=np.zeros_like,
        cuda=SimpleNamespace(
            runtime=_FakeRuntime(),
            Device=_FakeDevice,
            get_current_stream=lambda: _FakeStream(),
        ),
    )


@contextmanager
def _mock_cuda_runtime():
    previous_cupy = cupy_runtime._CUPY
    previous_sparse = cupy_runtime._CUPYX_SPARSE
    cupy_runtime._CUPY = _fake_cupy()
    cupy_runtime._CUPYX_SPARSE = sp
    try:
        yield
    finally:
        cupy_runtime._CUPY = previous_cupy
        cupy_runtime._CUPYX_SPARSE = previous_sparse


def _sphere_problem():
    grid = build_cluster_grid(
        GridSettings(
            spacing=0.8,
            radius=2.4,
            expansion_order=4,
            shift=(0.0, 0.0, 0.0),
        )
    )
    radius_squared = np.einsum(
        "ij,ij->i", grid.coordinates, grid.coordinates
    )
    density = np.exp(-0.65 * radius_squared)
    density *= 2.0 / grid.integrate(density)
    return grid, build_negative_laplacian(grid), density


def _box_problem():
    grid = build_cluster_grid(
        GridSettings(
            spacing=1.0,
            radius=1.5,
            expansion_order=2,
            shift=(0.0, 0.0, 0.0),
            domain_shape="box",
            box_lengths=(3.0, 3.0, 3.0),
        )
    )
    displaced = grid.coordinates - np.array((0.25, -0.15, 0.10))
    density = np.exp(-np.einsum("ij,ij->i", displaced, displaced))
    density *= 1.5 / grid.integrate(density)
    return grid, build_negative_laplacian(grid), density


def _assert_result_parity(
    testcase: unittest.TestCase,
    actual: HartreeResult,
    expected: HartreeResult,
    *,
    rtol: float = 2.0e-11,
    atol: float = 2.0e-12,
) -> None:
    np.testing.assert_allclose(
        actual.potential, expected.potential, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        actual.right_hand_side,
        expected.right_hand_side,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    testcase.assertEqual(actual.converged, expected.converged)
    testcase.assertEqual(actual.iterations, expected.iterations)
    testcase.assertEqual(
        actual.matrix_vector_products, expected.matrix_vector_products
    )
    testcase.assertAlmostEqual(
        actual.initial_residual_norm, expected.initial_residual_norm, 11
    )
    testcase.assertAlmostEqual(actual.residual_norm, expected.residual_norm, 10)


class TestCuPyPoissonWithoutCUDA(unittest.TestCase):
    def test_import_is_lazy_and_public_module_does_not_require_cupy(self):
        # This module was imported successfully before the optional-runtime
        # check above.  On CPU-only CI that is the behavior under test.
        self.assertTrue(callable(solve_hartree_cupy))
        self.assertTrue(callable(build_boundary_corrected_rhs))
        self.assertIsInstance(REAL_CUDA_AVAILABLE, bool)

    def test_sphere_multipole_parity_and_shared_device_csr(self):
        grid, negative_laplacian, density = _sphere_problem()
        settings = HartreeSettings(
            boundary_method="auto",
            multipole_order=5,
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-13,
            max_iterations=600,
        )
        expected = solve_hartree(
            density, grid, negative_laplacian, settings
        )

        with _mock_cuda_runtime():
            hamiltonian = CuPyHamiltonian(
                negative_laplacian, np.zeros(grid.size)
            )
            solver = CuPyPoissonSolver(hamiltonian)
            self.assertIs(
                solver.negative_laplacian,
                hamiltonian.negative_laplacian,
            )
            actual = solver.solve(density, grid, settings)

        self.assertIsInstance(actual.boundary, MultipoleExpansion)
        _assert_result_parity(self, actual, expected)
        self.assertEqual(solver.last_timings.solve_calls, 1)
        self.assertEqual(solver.timings.solve_calls, 1)
        for name in (
            "total_seconds",
            "rhs_seconds",
            "upload_seconds",
            "solve_seconds",
            "download_seconds",
        ):
            self.assertGreaterEqual(getattr(solver.last_timings, name), 0.0)
        self.assertGreaterEqual(
            solver.last_timings.total_seconds,
            solver.last_timings.rhs_seconds,
        )

    def test_small_box_direct_boundary_parity(self):
        grid, negative_laplacian, density = _box_problem()
        settings = HartreeSettings(
            boundary_method="auto",
            relative_tolerance=1.0e-11,
            absolute_tolerance=1.0e-13,
            max_iterations=300,
            direct_chunk_size=3,
        )
        expected = solve_hartree(
            density, grid, negative_laplacian, settings
        )

        with _mock_cuda_runtime():
            hamiltonian = CuPyHamiltonian(
                negative_laplacian, np.zeros(grid.size)
            )
            actual = solve_hartree_cupy(
                density, grid, hamiltonian, settings
            )

        self.assertIsInstance(actual.boundary, DirectCoulombBoundary)
        _assert_result_parity(self, actual, expected)

    def test_warm_start_and_exact_initial_exit_match_reference(self):
        grid, negative_laplacian, density = _sphere_problem()
        settings = HartreeSettings(
            multipole_order=4,
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            max_iterations=600,
        )
        first = solve_hartree(density, grid, negative_laplacian, settings)
        density_updated = density * (
            1.0 + 0.025 * grid.coordinates[:, 0] / grid.settings.radius
        )
        expected = solve_hartree(
            density_updated,
            grid,
            negative_laplacian,
            settings,
            initial_potential=first.potential,
        )

        with _mock_cuda_runtime():
            hamiltonian = CuPyHamiltonian(
                negative_laplacian, np.zeros(grid.size)
            )
            solver = CuPyPoissonSolver(hamiltonian)
            solver.solve(density, grid, settings)
            actual = solver.solve(
                density_updated,
                grid,
                settings,
                initial_potential=first.potential,
            )

            rhs, _ = build_boundary_corrected_rhs(
                density_updated, grid, settings
            )
            exact_initial = np.linalg.solve(
                negative_laplacian.toarray(), rhs
            )
            early_reference = solve_hartree(
                density_updated,
                grid,
                negative_laplacian,
                settings,
                initial_potential=exact_initial,
            )
            early_actual = solver.solve(
                density_updated,
                grid,
                settings,
                initial_potential=exact_initial,
            )

        _assert_result_parity(self, actual, expected)
        self.assertLess(actual.initial_residual_norm, first.initial_residual_norm)
        self.assertEqual(early_reference.iterations, 0)
        self.assertEqual(early_reference.matrix_vector_products, 1)
        _assert_result_parity(self, early_actual, early_reference)
        self.assertEqual(solver.timings.solve_calls, 3)
        self.assertEqual(solver.last_timings.solve_calls, 1)

    def test_nonconvergence_budget_raise_and_breakdown_semantics(self):
        grid, negative_laplacian, density = _sphere_problem()
        limited = HartreeSettings(
            multipole_order=2,
            relative_tolerance=1.0e-14,
            absolute_tolerance=0.0,
            max_iterations=1,
        )
        expected_limited = solve_hartree(
            density,
            grid,
            negative_laplacian,
            limited,
            raise_on_nonconvergence=False,
        )
        self.assertEqual(expected_limited.iterations, 0)
        self.assertEqual(expected_limited.matrix_vector_products, 2)
        breakdown_settings = HartreeSettings(
            multipole_order=2,
            relative_tolerance=1.0e-14,
            absolute_tolerance=0.0,
            max_iterations=10,
        )

        with _mock_cuda_runtime():
            hamiltonian = CuPyHamiltonian(
                negative_laplacian, np.zeros(grid.size)
            )
            solver = CuPyPoissonSolver(hamiltonian)
            actual_limited = solver.solve(
                density,
                grid,
                limited,
                raise_on_nonconvergence=False,
            )
            with self.assertRaisesRegex(
                RuntimeError, r"residual=.*matvecs=2"
            ):
                solver.solve(density, grid, limited)

            # A negative identity is intentionally not SPD.  Both solvers
            # count the attempted A@p, stop on p.T A p <= 0, then perform the
            # unconditional final true-residual product.
            invalid_operator = -sp.eye(grid.size, format="csr")
            invalid_hamiltonian = CuPyHamiltonian(
                invalid_operator, np.zeros(grid.size)
            )
            invalid_solver = CuPyPoissonSolver(invalid_hamiltonian)
            actual_breakdown = invalid_solver.solve(
                density,
                grid,
                breakdown_settings,
                raise_on_nonconvergence=False,
            )

        _assert_result_parity(self, actual_limited, expected_limited)
        self.assertAlmostEqual(
            actual_limited.residual_norm,
            float(
                np.linalg.norm(
                    actual_limited.right_hand_side
                    - negative_laplacian @ actual_limited.potential
                )
            ),
            13,
        )
        expected_breakdown = solve_hartree(
            density,
            grid,
            -sp.eye(grid.size, format="csr"),
            breakdown_settings,
            raise_on_nonconvergence=False,
        )
        self.assertFalse(actual_breakdown.converged)
        self.assertEqual(actual_breakdown.iterations, 0)
        self.assertEqual(actual_breakdown.matrix_vector_products, 3)
        self.assertAlmostEqual(
            actual_breakdown.residual_norm,
            float(
                np.linalg.norm(
                    actual_breakdown.right_hand_side
                    + actual_breakdown.potential
                )
            ),
            13,
        )
        _assert_result_parity(self, actual_breakdown, expected_breakdown)

    def test_explicit_multipole_boundary_rejects_box(self):
        grid, _operator, density = _box_problem()
        with self.assertRaisesRegex(ValueError, "not convergent"):
            build_boundary_corrected_rhs(
                density,
                grid,
                HartreeSettings(boundary_method="multipole"),
            )


@unittest.skipUnless(REAL_CUDA_AVAILABLE, "CuPy/CUDA are not available")
class TestCuPyPoissonRealCUDA(unittest.TestCase):
    def tearDown(self):
        # Operators and solvers that a test left in a reference cycle are destroyed
        # here, between the tests, and not by a collection inside the next one.
        gc.collect()

    def test_real_cuda_sphere_matches_reference(self):
        grid, negative_laplacian, density = _sphere_problem()
        settings = HartreeSettings(
            multipole_order=4,
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-12,
            max_iterations=600,
        )
        expected = solve_hartree(
            density, grid, negative_laplacian, settings
        )
        hamiltonian = CuPyHamiltonian(
            negative_laplacian, np.zeros(grid.size)
        )
        solver = CuPyPoissonSolver(hamiltonian)
        actual = solver.solve(density, grid, settings)
        _assert_result_parity(
            self, actual, expected, rtol=2.0e-8, atol=2.0e-10
        )


if __name__ == "__main__":
    unittest.main()
