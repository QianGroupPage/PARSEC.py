"""Focused wiring tests for the component-aware default execution policy."""

from __future__ import annotations

from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import parsec_python.acceleration.driver as driver_module
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.cupy import CuPyTimingStats
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.backends.cupy_runtime import CuPyHamiltonianBackend
from parsec_python.acceleration.backends.selection import BackendSelection
from parsec_python.acceleration.driver import prepare_single_point, run_scf
from parsec_python.acceleration.models import BackendInfo, BackendStatistics
from parsec_python.Input import parse_parsec_input
from parsec_python.SCF.single_point import PreparedSinglePointSystem
from parsec_python.models import PreparationTimings


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SMOKE_INPUT = PACKAGE_ROOT / "tests" / "data" / "H_cli_smoke.in"
HYBRID_AVAILABLE = native_available() and cupy_available()


def _hybrid_selection() -> BackendSelection:
    """Return the fastest default component combination without probing runtimes."""

    return BackendSelection(
        requested="auto",
        selected="cupy",
        finite_difference_builder="native",
        hartree_backend="native",
    )


class HybridPreparationTests(unittest.TestCase):
    def tearDown(self) -> None:
        with driver_module._REFERENCE_CACHE_LOCK:
            driver_module._REFERENCE_CACHE.clear()

    def test_cuda_probe_overlaps_independent_reference_preparation(self) -> None:
        selection = _hybrid_selection()
        reference = object()
        rendezvous = Barrier(2)

        def resolve_backend(*_args):
            rendezvous.wait(timeout=5.0)
            return selection

        def prepare_reference(*_args):
            rendezvous.wait(timeout=5.0)
            return reference

        with (
            patch.object(
                driver_module,
                "resolve_backend",
                side_effect=resolve_backend,
            ),
            patch.object(
                driver_module,
                "_prepare_reference_physics",
                side_effect=prepare_reference,
            ),
            patch(
                "parsec_python.acceleration.backends.selection._native_status",
                return_value=(True, None),
            ),
            patch.dict(
                driver_module.os.environ,
                {"PARSEC_OVERLAP_CUDA_INITIALIZATION": "1"},
            ),
        ):
            selected, prepared, timing = (
                driver_module._resolve_and_prepare_reference(
                    object(), "auto"
                )
            )

        self.assertIs(selected, selection)
        self.assertIs(prepared, reference)
        self.assertEqual(
            timing["cuda_initialization_overlap"],
            "cuda_probe_with_cpu_reference_setup",
        )
        self.assertGreaterEqual(
            timing["backend_reference_overlapped_seconds"], 0.0
        )

    def test_native_finite_difference_builder_is_used_for_cupy_execution(self) -> None:
        problem = object()
        prepared = object()
        native_builder = MagicMock(name="build_native_negative_laplacian")

        with (
            patch.object(
                driver_module,
                "prepare_reference_single_point",
                return_value=prepared,
            ) as prepare_reference,
            patch(
                "parsec_python.acceleration.backends.native."
                "build_native_negative_laplacian",
                native_builder,
            ),
            # This test isolates the finite-difference routing.  Pretend the
            # installed extension predates the independent radial kernels so
            # its expectation does not vary with the locally installed wheel.
            patch(
                "parsec_python.acceleration.backends.native._load_native",
                return_value=SimpleNamespace(),
            ),
        ):
            result = driver_module._prepare_reference_physics(
                problem,
                _hybrid_selection(),
            )

        self.assertIs(result, prepared)
        prepare_reference.assert_called_once_with(
            problem,
            negative_laplacian_builder=native_builder,
        )

    def test_reference_builder_does_not_depend_on_cupy_execution(self) -> None:
        problem = object()
        prepared = object()
        selection = BackendSelection(
            requested="cupy",
            selected="cupy",
            finite_difference_builder="reference",
            hartree_backend="cupy",
        )

        with patch.object(
            driver_module,
            "prepare_reference_single_point",
            return_value=prepared,
        ) as prepare_reference:
            result = driver_module._prepare_reference_physics(problem, selection)

        self.assertIs(result, prepared)
        prepare_reference.assert_called_once_with(problem)

    def test_resident_reference_cache_reuses_only_static_preparation(self) -> None:
        translation = parse_parsec_input(SMOKE_INPUT)
        problem = translation.problem
        selection = BackendSelection(
            requested="scipy",
            selected="scipy",
            finite_difference_builder="reference",
            hartree_backend="scipy",
        )
        prepared = PreparedSinglePointSystem(
            input=problem,
            atoms=tuple(problem.atoms),
            electron_count=1.0,
            pseudopotentials={},
            grid=object(),
            negative_laplacian=object(),
            ionic_potential=np.zeros(1),
            nonlocal_operator=object(),
            initial_density=np.ones(1),
            core_density=np.zeros(1),
            ion_ion_energy=0.0,
            timings=PreparationTimings(total_seconds=2.0),
        )

        with (
            patch.object(
                driver_module,
                "prepare_reference_single_point",
                return_value=prepared,
            ) as prepare_reference,
            patch.dict(
                driver_module.os.environ,
                {
                    "PARSEC_ACCELERATED_RESIDENT": "1",
                    "PARSEC_RESIDENT_REFERENCE_CACHE_SIZE": "1",
                },
            ),
        ):
            first = driver_module._prepare_reference_physics(problem, selection)
            second = driver_module._prepare_reference_physics(problem, selection)

        self.assertIs(first, prepared)
        self.assertIsNot(second, first)
        self.assertIs(second.grid, first.grid)
        self.assertIs(second.ionic_potential, first.ionic_potential)
        self.assertLess(second.timings.total_seconds, first.timings.total_seconds)
        prepare_reference.assert_called_once_with(problem)

class HybridHartreeWiringTests(unittest.TestCase):
    @staticmethod
    def _cupy_backend_shell() -> CuPyHamiltonianBackend:
        """Build host adapter state without importing CuPy or requiring CUDA."""

        implementation = object.__new__(CuPyHamiltonianBackend)
        implementation.statistics = BackendStatistics()
        implementation.timing_stats = CuPyTimingStats()
        implementation.info = BackendInfo(
            requested="auto",
            selected="cupy",
            device="mock GPU",
            implementation="mock CuPy execution",
        )
        implementation.device_operator = object()
        implementation.eigenproblem_solver = object()
        return implementation

    def test_cupy_execution_with_native_hartree_keeps_statistics_compatible(self) -> None:
        selection = _hybrid_selection()
        hartree_settings = SimpleNamespace(
            boundary_method="auto",
            multipole_order=9,
        )
        grid = SimpleNamespace(
            settings=SimpleNamespace(domain_shape="sphere")
        )
        negative_laplacian = object()
        reference = SimpleNamespace(
            negative_laplacian=negative_laplacian,
            grid=grid,
            input=SimpleNamespace(hartree=hartree_settings),
        )
        implementation = self._cupy_backend_shell()

        native_solver = MagicMock(name="native_poisson_solver")
        native_result = MagicMock(name="native_poisson_result")
        expected_result = object()
        native_result.as_hartree_result.return_value = expected_result
        native_solver.solve.return_value = native_result
        native_boundary_builder = MagicMock(name="native_boundary_builder")

        density = np.array((0.25, 0.75), dtype=np.float64)
        initial = np.array((0.0, 0.0), dtype=np.float64)
        right_hand_side = np.array((1.0, 2.0), dtype=np.float64)
        boundary = object()

        with (
            patch.object(driver_module, "resolve_backend", return_value=selection),
            patch.object(
                driver_module,
                "_prepare_reference_physics",
                return_value=reference,
            ),
            patch.object(
                driver_module,
                "_build_backend",
                return_value=implementation,
            ),
            patch(
                "parsec_python.acceleration.Hartree.native_poisson."
                "NativePoissonSolver",
                return_value=native_solver,
            ) as solver_type,
            patch(
                "parsec_python.acceleration.Hartree.native_boundary."
                "NativeMultipoleBoundaryBuilder",
                return_value=native_boundary_builder,
            ) as boundary_builder_type,
            patch(
                "parsec_python.acceleration.backends.native.native_build_info",
                return_value={
                    "openmp_detected_processors": 32,
                    "openmp_reserved_threads": 4,
                    "openmp_max_threads": 28,
                    "openmp_thread_source": "detected_processors_minus_4",
                },
            ),
            patch.dict(
                driver_module.os.environ,
                {"PARSEC_OVERLAP_HARTREE_SETUP": "1"},
            ),
        ):
            native_boundary_builder.build.return_value = (
                right_hand_side,
                boundary,
            )
            system = driver_module.prepare_single_point(object(), backend="auto")
            result = system.solve_hartree(
                density,
                initial,
                raise_on_nonconvergence=False,
            )

        self.assertIs(result, expected_result)
        solver_type.assert_called_once_with(negative_laplacian)
        boundary_builder_type.assert_called_once_with(grid, 9)
        native_boundary_builder.build.assert_called_once_with(density)
        native_solver.solve.assert_called_once_with(
            right_hand_side,
            initial,
            hartree_settings,
            raise_on_nonconvergence=False,
        )
        native_result.as_hartree_result.assert_called_once_with(boundary)

        # CuPy's synchronization bridge understands only a CuPy solver under
        # ``poisson_solver``.  The native object has a distinct inspectable
        # attribute, while its wall timings are accumulated directly.
        self.assertIs(implementation.native_poisson_solver, native_solver)
        self.assertIs(
            implementation.native_boundary_builder,
            native_boundary_builder,
        )
        self.assertFalse(hasattr(implementation, "poisson_solver"))
        self.assertEqual(implementation.statistics.hartree_solve_calls, 1)
        self.assertGreaterEqual(implementation.statistics.hartree_rhs_seconds, 0.0)
        self.assertGreaterEqual(
            implementation.statistics.hartree_linear_solve_seconds,
            0.0,
        )
        self.assertGreaterEqual(implementation.statistics.hartree_total_seconds, 0.0)

        # This is the call made after SCF.  It must neither inspect native
        # solver event fields nor erase the manually accumulated Hartree data.
        implementation.synchronize_statistics()
        self.assertEqual(implementation.statistics.hartree_solve_calls, 1)

        details = dict(system.backend_info.details)
        self.assertEqual(details["hartree_backend"], "native")
        self.assertIn("C++17", details["finite_difference_builder"])
        self.assertIn("C++/OpenMP CG", details["hartree_implementation"])
        self.assertEqual(
            details["hartree_boundary_setup"],
            "overlapped with GPU orbital setup",
        )
        self.assertGreaterEqual(
            float(details["hartree_boundary_setup_seconds"]), 0.0
        )
        self.assertGreaterEqual(
            float(details["hartree_boundary_setup_overlapped_seconds"]), 0.0
        )
        self.assertEqual(details["native_openmp_detected_processors"], "32")
        self.assertEqual(details["native_openmp_reserved_threads"], "4")
        self.assertEqual(details["native_openmp_max_threads"], "28")


@unittest.skipUnless(
    HYBRID_AVAILABLE,
    "both parsec_accelerated_native and CuPy/CUDA are required",
)
class RealHybridAccuracyTests(unittest.TestCase):
    def test_auto_hybrid_one_iteration_matches_reference(self) -> None:
        """Exercise the actual C++/CuPy composition, not only its wiring."""

        problem = parse_parsec_input(SMOKE_INPUT).problem
        expected = run_scf(prepare_single_point(problem, backend="scipy"))
        system = prepare_single_point(
            problem, backend="auto", symmetry="off"
        )
        actual = run_scf(system)

        self.assertEqual(actual.backend.selected, "cupy")
        details = dict(actual.backend.details)
        self.assertIn("C++17", details["finite_difference_builder"])
        self.assertEqual(details["hartree_backend"], "native")
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

    def test_auto_hybrid_uses_reflection_representations(self) -> None:
        problem = parse_parsec_input(SMOKE_INPUT).problem
        expected = run_scf(prepare_single_point(problem, backend="scipy"))
        system = prepare_single_point(problem, backend="auto")
        self.assertIsNone(system.backend.device_operator)
        actual = run_scf(system)
        self.assertIsNone(system.backend.device_operator)

        details = dict(actual.backend.details)
        self.assertEqual(
            details["orbital_symmetry"],
            "CuPy real one-dimensional reflection representations",
        )
        self.assertEqual(details["orbital_symmetry_representations"], "8")
        self.assertEqual(details["symmetry_reduction_ratio"], "8")
        np.testing.assert_array_equal(
            actual.representations,
            np.asarray(actual.history[-1].representations),
        )
        self.assertTrue(np.any(actual.representations > 1))
        np.testing.assert_allclose(
            actual.eigenvalues,
            expected.eigenvalues,
            rtol=2.0e-7,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            actual.density,
            expected.density,
            rtol=2.0e-4,
            atol=5.0e-7,
        )
        self.assertAlmostEqual(actual.energies.total, expected.energies.total, 7)

    def test_explicit_cupy_symmetry_adapts_compact_scf_scalar_fields(self) -> None:
        """Full-grid CuPy Poisson/XC APIs accept compact symmetry SCF state."""

        problem = parse_parsec_input(SMOKE_INPUT).problem
        expected = run_scf(prepare_single_point(problem, backend="scipy"))
        system = prepare_single_point(problem, backend="cupy")
        actual = run_scf(system)

        details = dict(actual.backend.details)
        self.assertEqual(details["hartree_backend"], "cupy")
        self.assertGreater(int(details["symmetry_reduction_ratio"]), 1)
        np.testing.assert_allclose(
            actual.eigenvalues,
            expected.eigenvalues,
            rtol=2.0e-7,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            actual.density,
            expected.density,
            rtol=2.0e-4,
            atol=5.0e-7,
        )
        self.assertAlmostEqual(actual.energies.total, expected.energies.total, 7)

    def test_cuda_stream_scheduler_preserves_sector_results(self) -> None:
        problem = parse_parsec_input(SMOKE_INPUT).problem
        with patch.dict(
            "os.environ",
            {"PARSEC_CUPY_SECTOR_SCHEDULER": "sequential"},
        ):
            sequential = run_scf(
                prepare_single_point(problem, backend="auto")
            )
        with patch.dict(
            "os.environ",
            {"PARSEC_CUPY_SECTOR_SCHEDULER": "streams"},
        ):
            concurrent = run_scf(
                prepare_single_point(problem, backend="auto")
            )

        np.testing.assert_array_equal(
            concurrent.representations, sequential.representations
        )
        np.testing.assert_allclose(
            concurrent.eigenvalues,
            sequential.eigenvalues,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            concurrent.density,
            sequential.density,
            rtol=2.0e-12,
            atol=2.0e-13,
        )
        self.assertAlmostEqual(
            concurrent.energies.total, sequential.energies.total, 11
        )


if __name__ == "__main__":
    unittest.main()
