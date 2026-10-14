"""CuPy CHEBDAV policy, selection, and optional CUDA parity tests.

The dispatcher tests use tiny NumPy-backed result objects so CPU-only CI can
verify the first-solver/state contract without pretending to execute CUDA
kernels.  Numerical kernel and end-to-end SCF parity are exercised only when
CuPy reports a usable CUDA device.
"""

from __future__ import annotations

from dataclasses import replace
import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

import parsec_python.acceleration.Eigensolvers.eigval as gpu_policy
import parsec_python.acceleration.backends.selection as selection_module
from parsec_python.acceleration.backends.cupy import (
    CuPyHamiltonian,
    CuPyTimingStats,
    cupy_available,
    require_cupy,
)
from parsec_python.acceleration.backends.selection import resolve_backend
from parsec_python.acceleration.driver import prepare_single_point, run_scf
from parsec_python.acceleration.Eigensolvers import run_chebdav as run_gpu_chebdav
from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState
from parsec_python.acceleration.models import BackendUnavailableError
from parsec_python.Eigensolvers import (
    ChebDavSettings,
    EigvalSettings,
    SubspaceSettings,
    run_chebdav as run_reference_chebdav,
    solve_eigval,
)
from parsec_python.Hamiltonian import KohnShamHamiltonian
from parsec_python.Input import parse_parsec_input
from parsec_python.V_ion import NonlocalProjectorOperator


GPU_AVAILABLE = cupy_available()
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SMOKE_INPUT = PACKAGE_ROOT / "tests" / "data" / "H_cli_smoke.in"


def _fake_device_operator(dimension: int = 12) -> CuPyHamiltonian:
    """Create only the state needed by the mocked eigensolver dispatcher."""

    operator = object.__new__(CuPyHamiltonian)
    operator.shape = (dimension, dimension)
    operator.timing_stats = CuPyTimingStats()
    return operator


def _fake_chebdav(operator, wanted_states, *, settings):
    del settings
    dimension = operator.shape[0]
    eigenvalues = np.arange(wanted_states, dtype=np.float64)
    vectors = np.eye(dimension, wanted_states, dtype=np.float64)
    residual_norms = np.linspace(0.01, 0.02, wanted_states)
    state = SimpleNamespace(
        operator_dimension=dimension,
        wanted_states=wanted_states,
        eigenvalues=eigenvalues,
        vectors=vectors,
        residual_norms=residual_norms,
    )
    return SimpleNamespace(
        eigenvalues=eigenvalues,
        vectors=vectors,
        residual_norms=residual_norms,
        state=state,
    )


def _fake_chebff(operator, wanted_states, *, settings):
    del settings
    dimension = operator.shape[0]
    eigenvalues = np.arange(wanted_states, dtype=np.float64) + 10.0
    vectors = np.eye(dimension, wanted_states, dtype=np.float64)
    state = SimpleNamespace(
        operator_dimension=dimension,
        wanted_states=wanted_states,
        eigenvalues=eigenvalues,
        vectors=vectors,
    )
    return SimpleNamespace(eigenvalues=eigenvalues, vectors=vectors, state=state)


def _fake_subspace(operator, state, *, settings):
    del operator, settings
    eigenvalues = np.asarray(state.eigenvalues) + 0.25
    vectors = np.asarray(state.vectors)
    next_state = DeviceSubspaceState(
        operator_dimension=state.operator_dimension,
        working_states=state.working_states,
        eigenvalues=eigenvalues,
        vectors=vectors,
        filter_lower_bound=float(eigenvalues[-1]),
        first_filter=False,
        filters_completed=state.filters_completed + 1,
    )
    return SimpleNamespace(
        eigenvalues=eigenvalues,
        vectors=vectors,
        residual_norms=np.full(state.working_states, 0.125),
        state=next_state,
    )


def _unsynchronized_call(function, /, *args, **kwargs):
    """CPU-only stand-in for the coarse CUDA timing wrapper."""

    return function(*args, **kwargs), 0.0


def _fake_download_runtime():
    return SimpleNamespace(asnumpy=np.asarray), None


class CuPyChebDavDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = EigvalSettings(
            safety_buffer=2,
            initial_method="chebdav",
        )

    def test_first_chebdav_then_compatible_subspace_retains_device_state(self):
        operator = _fake_device_operator()
        with (
            patch.object(
                gpu_policy,
                "run_chebdav",
                side_effect=_fake_chebdav,
                create=True,
            ) as chebdav,
            patch.object(
                gpu_policy,
                "run_chebff",
                side_effect=_fake_chebff,
            ) as chebff,
            patch.object(
                gpu_policy,
                "run_subspace_filter",
                side_effect=_fake_subspace,
            ) as subspace,
            patch.object(
                gpu_policy,
                "synchronized_call",
                side_effect=_unsynchronized_call,
            ),
            patch.object(
                gpu_policy,
                "require_cupy",
                side_effect=_fake_download_runtime,
            ),
        ):
            solver = gpu_policy.CuPyEigvalSolver(operator, settings=self.settings)
            first = solver.solve(3)
            first_state = solver.device_state
            later = solver.solve(3)

        chebdav.assert_called_once()
        self.assertEqual(chebdav.call_args.args[1], 5)
        chebff.assert_not_called()
        subspace.assert_called_once()
        self.assertEqual(first.solver_path, "chebdav")
        self.assertIsNotNone(first.residual_norms)
        np.testing.assert_allclose(first.residual_norms, (0.01, 0.0125, 0.015))
        self.assertEqual(first_state.initial_method, "chebdav")
        self.assertEqual(first_state.subspace.vectors.shape, (12, 5))
        self.assertEqual(later.solver_path, "subspace")
        self.assertFalse(later.restarted)
        self.assertEqual(solver.device_state.solves_completed, 2)
        self.assertEqual(solver.device_state.subspace.filters_completed, 1)

    def test_incompatible_request_restarts_chebdav_not_chebff(self):
        operator = _fake_device_operator()
        with (
            patch.object(
                gpu_policy,
                "run_chebdav",
                side_effect=_fake_chebdav,
                create=True,
            ) as chebdav,
            patch.object(
                gpu_policy,
                "run_chebff",
                side_effect=_fake_chebff,
            ) as chebff,
            patch.object(gpu_policy, "run_subspace_filter") as subspace,
            patch.object(
                gpu_policy,
                "synchronized_call",
                side_effect=_unsynchronized_call,
            ),
            patch.object(
                gpu_policy,
                "require_cupy",
                side_effect=_fake_download_runtime,
            ),
        ):
            solver = gpu_policy.CuPyEigvalSolver(operator, settings=self.settings)
            solver.solve(2)
            restarted = solver.solve(3)

        self.assertEqual(chebdav.call_count, 2)
        chebff.assert_not_called()
        subspace.assert_not_called()
        self.assertEqual(restarted.solver_path, "chebdav")
        self.assertTrue(restarted.restarted)
        self.assertEqual(
            restarted.restart_reason,
            "requested_state_count_changed",
        )
        self.assertEqual(solver.device_state.initial_method, "chebdav")

    def test_changing_initial_method_invalidates_chebdav_state(self):
        operator = _fake_device_operator()
        chebff_settings = replace(self.settings, initial_method="chebff")
        with (
            patch.object(
                gpu_policy,
                "run_chebdav",
                side_effect=_fake_chebdav,
                create=True,
            ) as chebdav,
            patch.object(
                gpu_policy,
                "run_chebff",
                side_effect=_fake_chebff,
            ) as chebff,
            patch.object(gpu_policy, "run_subspace_filter") as subspace,
            patch.object(
                gpu_policy,
                "synchronized_call",
                side_effect=_unsynchronized_call,
            ),
            patch.object(
                gpu_policy,
                "require_cupy",
                side_effect=_fake_download_runtime,
            ),
        ):
            solver = gpu_policy.CuPyEigvalSolver(operator, settings=self.settings)
            solver.solve(3)
            restarted = solver.solve(3, settings=chebff_settings)

        chebdav.assert_called_once()
        chebff.assert_called_once()
        subspace.assert_not_called()
        self.assertTrue(restarted.restarted)
        self.assertEqual(restarted.restart_reason, "initial_solver_changed")
        self.assertEqual(restarted.solver_path, "chebff")
        self.assertEqual(solver.device_state.initial_method, "chebff")

    def test_corrupt_saved_subspace_metadata_restarts_selected_solver(self):
        operator = _fake_device_operator()
        cases = (
            (
                {"operator_dimension": operator.shape[0] + 1},
                "saved_subspace_dimension_changed",
            ),
            (
                {"working_states": 4},
                "saved_subspace_state_count_changed",
            ),
        )
        for mutation, reason in cases:
            with self.subTest(reason=reason):
                with (
                    patch.object(
                        gpu_policy,
                        "run_chebdav",
                        side_effect=_fake_chebdav,
                        create=True,
                    ) as chebdav,
                    patch.object(gpu_policy, "run_chebff") as chebff,
                    patch.object(
                        gpu_policy, "run_subspace_filter"
                    ) as subspace,
                    patch.object(
                        gpu_policy,
                        "synchronized_call",
                        side_effect=_unsynchronized_call,
                    ),
                    patch.object(
                        gpu_policy,
                        "require_cupy",
                        side_effect=_fake_download_runtime,
                    ),
                ):
                    solver = gpu_policy.CuPyEigvalSolver(
                        operator, settings=self.settings
                    )
                    solver.solve(3)
                    original = solver.device_state
                    solver._state = replace(
                        original,
                        subspace=replace(original.subspace, **mutation),
                    )
                    restarted = solver.solve(3)

                self.assertEqual(chebdav.call_count, 2)
                chebff.assert_not_called()
                subspace.assert_not_called()
                self.assertTrue(restarted.restarted)
                self.assertEqual(restarted.restart_reason, reason)
                self.assertEqual(restarted.solver_path, "chebdav")


class CuPyChebDavSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = SimpleNamespace(
            eigensolver=SimpleNamespace(method="chebdav")
        )

    def test_explicit_cupy_chebdav_is_selected_when_cuda_is_available(self):
        with patch.object(
            selection_module,
            "_cupy_status",
            return_value=(True, None),
        ):
            selected = resolve_backend("cupy", self.problem)

        self.assertEqual(selected.requested, "cupy")
        self.assertEqual(selected.selected, "cupy")
        self.assertEqual(selected.finite_difference_builder, "reference")
        self.assertEqual(selected.hartree_backend, "cupy")
        self.assertEqual(selected.fallback_reasons, ())

    def test_explicit_cupy_chebdav_unavailable_does_not_fallback(self):
        with patch.object(
            selection_module,
            "_cupy_status",
            return_value=(False, "test CUDA unavailable"),
        ):
            with self.assertRaisesRegex(
                BackendUnavailableError,
                "CuPy backend was requested.*test CUDA unavailable",
            ):
                resolve_backend("cupy", self.problem)

    def test_auto_composes_cupy_chebdav_with_native_components(self):
        with (
            patch.object(
                selection_module,
                "_cupy_status",
                return_value=(True, None),
            ),
            patch.object(
                selection_module,
                "_native_status",
                return_value=(True, None),
            ) as native,
        ):
            selected = resolve_backend("auto", self.problem)

        self.assertEqual(selected.selected, "cupy")
        self.assertEqual(selected.finite_difference_builder, "native")
        self.assertEqual(selected.hartree_backend, "native")
        self.assertEqual(selected.fallback_reasons, ())
        native.assert_called_once_with()

    def test_auto_records_cuda_failure_before_native_fallback(self):
        with (
            patch.object(
                selection_module,
                "_cupy_status",
                return_value=(False, "test CUDA unavailable"),
            ),
            patch.object(
                selection_module,
                "_native_status",
                return_value=(True, None),
            ),
        ):
            selected = resolve_backend("auto", self.problem)

        self.assertEqual(selected.selected, "native")
        self.assertEqual(selected.finite_difference_builder, "native")
        self.assertEqual(selected.hartree_backend, "native")
        self.assertEqual(
            selected.fallback_reasons,
            ("CuPy skipped: test CUDA unavailable",),
        )


@unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA are not available")
class CuPyChebDavRealCudaTests(unittest.TestCase):
    def tearDown(self):
        # Solvers and prepared systems that a test left in a reference cycle are
        # destroyed here, between the tests, and not by a collection inside the
        # next one.
        gc.collect()

    def _assert_policy_state_equal(self, actual, expected):
        fields = (
            "operator_dimension",
            "wanted_states",
            "maximum_subspace_dimension",
            "truly_converged_states",
            "approximate_cleanup_used",
            "matrix_vector_products",
            "iterations_completed",
            "inner_restarts",
            "outer_restarts",
        )
        for field in fields:
            self.assertEqual(
                getattr(actual, field),
                getattr(expected, field),
                field,
            )

    def test_forced_approximate_cleanup_matches_reference_policy(self):
        cp, _ = require_cupy()
        generator = np.random.default_rng(3)
        matrix = generator.normal(size=(30, 30))
        matrix = 0.5 * (matrix + matrix.T)
        settings = ChebDavSettings(
            block_size=3,
            workspace_window=6,
            convergence_tolerance=1.0e-16,
            max_iterations=1,
            random_seed=100,
        )

        expected = run_reference_chebdav(matrix, 6, settings=settings)
        gpu_operator = CuPyHamiltonian(
            sp.csr_matrix(matrix), np.zeros(matrix.shape[0])
        )
        actual = run_gpu_chebdav(gpu_operator, 6, settings=settings)

        self._assert_policy_state_equal(actual.state, expected.state)
        self.assertTrue(actual.state.approximate_cleanup_used)
        np.testing.assert_allclose(
            cp.asnumpy(actual.eigenvalues),
            expected.eigenvalues,
            rtol=2.0e-8,
            atol=2.0e-8,
        )
        np.testing.assert_allclose(
            cp.asnumpy(actual.residual_norms),
            expected.residual_norms,
            rtol=2.0e-7,
            atol=2.0e-8,
        )

    def test_outer_restart_lifecycle_matches_reference_policy(self):
        cp, _ = require_cupy()
        generator = np.random.default_rng(0)
        matrix = generator.normal(size=(30, 30))
        matrix = 0.5 * (matrix + matrix.T)
        settings = ChebDavSettings(
            block_size=3,
            workspace_window=6,
            convergence_tolerance=1.0e-8,
            random_seed=100,
        )

        expected = run_reference_chebdav(matrix, 8, settings=settings)
        gpu_operator = CuPyHamiltonian(
            sp.csr_matrix(matrix), np.zeros(matrix.shape[0])
        )
        actual = run_gpu_chebdav(gpu_operator, 8, settings=settings)

        self._assert_policy_state_equal(actual.state, expected.state)
        self.assertGreater(actual.state.outer_restarts, 0)
        self.assertEqual(
            tuple(record.outer_restart for record in actual.iterations),
            tuple(record.outer_restart for record in expected.iterations),
        )
        np.testing.assert_allclose(
            cp.asnumpy(actual.eigenvalues),
            expected.eigenvalues,
            rtol=2.0e-8,
            atol=2.0e-8,
        )

    def test_first_chebdav_and_later_subspace_match_reference(self):
        cp, _ = require_cupy()
        diagonal = np.array(
            (-4.0, -1.0, 0.5, 2.0, 4.0, 7.0, 10.0, 15.0),
            dtype=np.float64,
        )
        kinetic = sp.diags(diagonal, format="csr")
        potential = 0.01 * np.sin(np.arange(diagonal.size))
        nonlocal_operator = NonlocalProjectorOperator(
            projectors=sp.csc_matrix((diagonal.size, 0), dtype=np.float64),
            signs=np.empty(0, dtype=np.float64),
            labels=(),
        )
        reference = KohnShamHamiltonian(
            kinetic,
            potential,
            nonlocal_operator,
        )
        settings = EigvalSettings(
            safety_buffer=0,
            initial_method="chebdav",
            chebdav=ChebDavSettings(
                polynomial_degree=15,
                convergence_tolerance=1.0e-6,
                block_size=2,
                workspace_window=6,
                lanczos_steps=5,
                random_seed=17,
            ),
            subspace=SubspaceSettings(
                polynomial_degree=8,
                degree_delta=1,
                lanczos_steps=5,
                block_size=2,
                random_seed=17,
            ),
        )
        expected_first = solve_eigval(
            reference.as_linear_operator(),
            2,
            settings=settings,
        )
        solver = gpu_policy.CuPyEigvalSolver(
            kinetic,
            potential,
            nonlocal_operator,
            settings=settings,
        )
        actual_first = solver.solve(2)

        self.assertEqual(actual_first.solver_path, "chebdav")
        self.assertIsNotNone(actual_first.residual_norms)
        self.assertIsInstance(solver.device_state.subspace.vectors, cp.ndarray)
        np.testing.assert_allclose(
            actual_first.eigenvalues,
            expected_first.eigenvalues,
            rtol=5.0e-8,
            atol=5.0e-8,
        )
        np.testing.assert_allclose(
            actual_first.residual_norms,
            expected_first.residual_norms,
            rtol=2.0e-6,
            atol=2.0e-8,
        )

        updated = potential + 0.015 * np.cos(np.arange(diagonal.size))
        expected_later = solve_eigval(
            KohnShamHamiltonian(
                kinetic,
                updated,
                nonlocal_operator,
            ).as_linear_operator(),
            2,
            settings=settings,
            state=expected_first.state,
        )
        solver.update_local_potential(updated)
        actual_later = solver.solve(2)

        self.assertEqual(actual_later.solver_path, "subspace")
        self.assertEqual(solver.device_state.solves_completed, 2)
        np.testing.assert_allclose(
            actual_later.eigenvalues,
            expected_later.eigenvalues,
            rtol=5.0e-8,
            atol=5.0e-8,
        )

    def test_one_iteration_scf_matches_reference_backend(self):
        problem = parse_parsec_input(SMOKE_INPUT).problem
        problem = replace(
            problem,
            eigensolver=replace(problem.eigensolver, method="chebdav"),
        )
        expected = run_scf(prepare_single_point(problem, backend="scipy"))
        # This test targets the full-grid CuPy CHEBDAV adapter itself.  The
        # representation-sector wrapper has separate integration coverage.
        actual_system = prepare_single_point(
            problem, backend="cupy", symmetry="off"
        )
        actual = run_scf(actual_system)

        self.assertEqual(actual.backend.selected, "cupy")
        self.assertEqual(
            actual_system.backend.eigenproblem_solver.solver.device_state.initial_method,
            "chebdav",
        )
        np.testing.assert_allclose(
            actual.eigenvalues,
            expected.eigenvalues,
            rtol=2.0e-7,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            actual.density,
            expected.density,
            rtol=2.0e-7,
            atol=2.0e-9,
        )
        self.assertAlmostEqual(actual.energies.total, expected.energies.total, 7)


if __name__ == "__main__":
    unittest.main()
