"""Focused wiring tests for the component-aware default execution policy."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import gc
import os
from pathlib import Path
import sys
import threading
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import DEFAULT, MagicMock, patch

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


# Switches of the Hartree boundary that replace settings of the input.  A test
# that prepares a stand-in for the input has none to replace, whatever the
# shell that runs the tests names.
_BOUNDARY_INPUT_SWITCHES = (
    "PARSEC_HARTREE_BOUNDARY",
    "PARSEC_HARTREE_BOUNDARY_TOLERANCE",
    "PARSEC_HARTREE_ATOMIC_TAIL",
    "PARSEC_HARTREE_LPOLE",
)


@contextmanager
def _stand_in_input():
    """Keep the boundary switches of the shell from an input that is a stand-in."""

    with patch.dict(driver_module.os.environ):
        for name in _BOUNDARY_INPUT_SWITCHES:
            driver_module.os.environ.pop(name, None)
        yield


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
                {
                    "PARSEC_OVERLAP_CUDA_INITIALIZATION": "1",
                    # GPU ionic sums wait for the device selection.
                    "PARSEC_IONIC_BACKEND": "native",
                },
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
            # The grid builder follows its own switch, whatever the caller set.
            patch.dict(driver_module.os.environ, {"PARSEC_FAST_GRID": "1"}),
        ):
            result = driver_module._prepare_reference_physics(
                problem,
                _hybrid_selection(),
            )

        self.assertIs(result, prepared)
        prepare_reference.assert_called_once_with(
            problem,
            negative_laplacian_builder=native_builder,
            grid_builder=driver_module.build_cluster_grid_by_slabs,
            build_atomic_tail=False,
        )

    def test_deferral_is_requested_for_cupy_execution_only(self) -> None:
        problem = object()
        native = BackendSelection(
            requested="native",
            selected="native",
            finite_difference_builder="native",
            hartree_backend="native",
        )
        for backend, selection, expected in (
            ("native", native, {}),
            (
                "auto",
                _hybrid_selection(),
                {
                    "defer_native_laplacian": True,
                    "deferred_laplacian_cache_directory": None,
                },
            ),
        ):
            with (
                self.subTest(selected=selection.selected),
                patch.object(
                    driver_module, "resolve_backend", return_value=selection
                ),
                patch.object(
                    driver_module, "_prepare_reference_physics", return_value=object()
                ) as prepare_reference,
                patch.dict(
                    driver_module.os.environ,
                    {
                        "PARSEC_OVERLAP_CUDA_INITIALIZATION": "0",
                        # GPU ionic sums add their options to the call.
                        "PARSEC_IONIC_BACKEND": "native",
                    },
                ),
            ):
                driver_module._resolve_and_prepare_reference(
                    problem, backend, defer_native_laplacian=True
                )
                prepare_reference.assert_called_once_with(
                    problem, selection, **expected
                )

    def test_overlapped_preparation_hands_another_backend_the_matrix(self) -> None:
        from parsec_python.acceleration.Laplacian import (
            DeferredNativeNegativeLaplacian,
        )

        native = BackendSelection(
            requested="auto",
            selected="native",
            finite_difference_builder="native",
            hartree_backend="native",
        )
        matrix = object()

        def materialize(descriptor):
            descriptor.materialization_seconds += 0.25
            return matrix

        def prepared_reference():
            descriptor = object.__new__(DeferredNativeNegativeLaplacian)
            descriptor.materialization_seconds = 0.0
            return PreparedSinglePointSystem(
                input=object(),
                atoms=(),
                electron_count=1.0,
                pseudopotentials={},
                grid=object(),
                negative_laplacian=descriptor,
                ionic_potential=np.zeros(1),
                nonlocal_operator=object(),
                initial_density=np.ones(1),
                core_density=np.zeros(1),
                ion_ion_energy=0.0,
                timings=PreparationTimings(
                    finite_difference_seconds=0.5, total_seconds=2.0
                ),
            )

        for installed, selection in (
            (True, native),
            (True, _hybrid_selection()),
            (False, native),
        ):
            reference = prepared_reference()
            with (
                self.subTest(cupy_installed=installed, selected=selection.selected),
                patch.object(
                    driver_module, "resolve_backend", return_value=selection
                ),
                patch.object(
                    driver_module, "_prepare_reference_physics", return_value=reference
                ) as prepare_reference,
                patch.object(
                    driver_module, "_cupy_installed", return_value=installed
                ),
                patch(
                    "parsec_python.acceleration.backends.selection._native_status",
                    return_value=(True, None),
                ),
                patch.object(
                    DeferredNativeNegativeLaplacian,
                    "materialize",
                    autospec=True,
                    side_effect=materialize,
                ),
                patch.dict(
                    driver_module.os.environ,
                    {
                        "PARSEC_OVERLAP_CUDA_INITIALIZATION": "1",
                        "PARSEC_IONIC_BACKEND": "native",
                    },
                ),
            ):
                _, prepared, timing = driver_module._resolve_and_prepare_reference(
                    object(), "auto", defer_native_laplacian=True
                )
                # The backend is not known while the reference is prepared;
                # the descriptor is asked for only where CuPy could be it.
                self.assertEqual(
                    prepare_reference.call_args.kwargs,
                    (
                        {
                            "defer_native_laplacian": True,
                            "deferred_laplacian_cache_directory": None,
                        }
                        if installed
                        else {}
                    ),
                )
                self.assertEqual(
                    timing["cuda_initialization_overlap"],
                    "cuda_probe_with_cpu_reference_setup",
                )
                if selection.selected == "cupy":
                    self.assertIs(prepared, reference)
                    continue
                # The matrix replaces the descriptor and its build time
                # joins the stage it was deferred from.
                self.assertIs(prepared.negative_laplacian, matrix)
                self.assertEqual(prepared.timings.finite_difference_seconds, 0.75)
                self.assertEqual(prepared.timings.total_seconds, 2.25)
                self.assertIs(prepared.grid, reference.grid)

    def test_resident_repeats_build_the_full_grid_matrix_once(self) -> None:
        from parsec_python.Laplacian import build_negative_laplacian

        problem = parse_parsec_input(SMOKE_INPUT).problem
        # CuPy is installed and the probe selects another backend, which
        # reads the matrix in every calculation.
        native = BackendSelection(
            requested="auto",
            selected="native",
            finite_difference_builder="native",
            hartree_backend="native",
        )
        with (
            patch.dict(
                driver_module.os.environ,
                {
                    "PARSEC_ACCELERATED_RESIDENT": "1",
                    "PARSEC_RESIDENT_REFERENCE_CACHE_SIZE": "1",
                    "PARSEC_OVERLAP_CUDA_INITIALIZATION": "1",
                    "PARSEC_IONIC_BACKEND": "native",
                },
            ),
            patch.object(driver_module, "resolve_backend", return_value=native),
            patch.object(driver_module, "_cupy_installed", return_value=True),
            patch(
                "parsec_python.acceleration.backends.selection._native_status",
                return_value=(True, None),
            ),
            # The C++ builder, whose matrix this one equals.
            patch(
                "parsec_python.acceleration.backends.native."
                "build_native_negative_laplacian",
                side_effect=build_negative_laplacian,
            ) as builder,
        ):
            prepared = [
                driver_module._resolve_and_prepare_reference(
                    problem, "auto", defer_native_laplacian=True
                )[1]
                for _ in range(3)
            ]
        builder.assert_called_once()
        self.assertEqual(len(driver_module._REFERENCE_CACHE), 1)
        for repeat in prepared[1:]:
            self.assertIs(repeat.negative_laplacian, prepared[0].negative_laplacian)
        # The build is charged to the calculation that made it.
        self.assertGreater(
            prepared[0].timings.finite_difference_seconds,
            prepared[1].timings.finite_difference_seconds,
        )

    def test_cached_reference_view_reports_for_itself_and_shares_the_matrix(
        self,
    ) -> None:
        from parsec_python.Laplacian import build_negative_laplacian
        from parsec_python.acceleration.Laplacian import (
            DeferredNativeNegativeLaplacian,
        )

        problem = parse_parsec_input(SMOKE_INPUT).problem
        with (
            patch.dict(
                driver_module.os.environ,
                {
                    "PARSEC_ACCELERATED_RESIDENT": "1",
                    "PARSEC_RESIDENT_REFERENCE_CACHE_SIZE": "1",
                },
            ),
            patch(
                "parsec_python.acceleration.backends.native."
                "build_native_negative_laplacian",
                side_effect=build_negative_laplacian,
            ) as builder,
        ):
            first, second, third = (
                driver_module._prepare_reference_physics(
                    problem, _hybrid_selection(), defer_native_laplacian=True
                ).negative_laplacian
                for _ in range(3)
            )
            for descriptor in (first, second, third):
                self.assertIsInstance(descriptor, DeferredNativeNegativeLaplacian)
                self.assertFalse(descriptor.materialized)
                self.assertIsNone(descriptor.matrix_origin)
            self.assertIsNot(second, first)
            self.assertEqual(first.reference_static_cache_status, "miss-stored")
            self.assertEqual(second.reference_static_cache_status, "hit")
            # A repeat is the first to need the matrix; a later one finds it.
            matrix = second.materialize()
            builder.assert_called_once()
            self.assertGreater(second.materialization_seconds, 0.0)
            self.assertTrue(third.materialized)
            self.assertIsNone(third.matrix_origin)
            self.assertIs(third.materialize(), matrix)
            later = driver_module._prepare_reference_physics(
                problem, _hybrid_selection(), defer_native_laplacian=True
            ).negative_laplacian
            self.assertIs(later.materialize(), matrix)
        builder.assert_called_once()
        self.assertEqual(
            [item.matrix_origin for item in (first, second, third, later)],
            [None, "built", "shared", "shared"],
        )
        self.assertEqual(third.materialization_seconds, 0.0)
        self.assertEqual(later.materialization_seconds, 0.0)
        with self.assertRaisesRegex(ValueError, "same grid"):
            DeferredNativeNegativeLaplacian(
                replace(first.grid), sharing_matrix_with=first
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

        with (
            patch.object(
                driver_module,
                "prepare_reference_single_point",
                return_value=prepared,
            ) as prepare_reference,
            patch.dict(driver_module.os.environ, {"PARSEC_FAST_GRID": "1"}),
        ):
            result = driver_module._prepare_reference_physics(problem, selection)

        self.assertIs(result, prepared)
        prepare_reference.assert_called_once_with(
            problem,
            grid_builder=driver_module.build_cluster_grid_by_slabs,
            build_atomic_tail=False,
        )

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
                    "PARSEC_FAST_GRID": "1",
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
        prepare_reference.assert_called_once_with(
            problem,
            grid_builder=driver_module.build_cluster_grid_by_slabs,
            build_atomic_tail=False,
        )

    def test_absent_cache_key_is_reported_as_text(self) -> None:
        from parsec_python.cli import save_result_archive

        digest = "a" * 64
        self.assertEqual(driver_module._reported_cache_key(digest), digest)
        absent = driver_module._reported_cache_key(None)
        self.assertIsInstance(absent, str)

        # The reference backend detects the symmetry through the same cache
        # routine, so the report of a disabled cache is produced without a GPU.
        problem = parse_parsec_input(SMOKE_INPUT).problem
        directory = Path.cwd() / ".tmp" / f"hybrid-report-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            plain = run_scf(prepare_single_point(problem, backend="scipy"))
            hashed = prepare_single_point(
                problem, backend="scipy", symmetry_cache_directory=directory
            )
            entries = sorted(item.name for item in directory.iterdir())
            # A run with symmetry off reads and writes no entry of a named cache.
            unreduced = prepare_single_point(
                problem,
                backend="scipy",
                symmetry="off",
                symmetry_cache_directory=directory,
            )
            self.assertEqual(
                sorted(item.name for item in directory.iterdir()), entries
            )
            archive = save_result_archive(directory / "result.npz", plain)
            with np.load(archive, allow_pickle=False) as stored:
                archived = dict(
                    zip(
                        stored["backend_detail_keys"].tolist(),
                        stored["backend_detail_values"].tolist(),
                    )
                )
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()
        details = dict(plain.backend.details)
        self.assertEqual(details["symmetry_geometry_cache_key"], absent)
        self.assertEqual(details["symmetry_geometry_cache_path"], "disabled")
        self.assertEqual(float(details["symmetry_geometry_hash_seconds"]), 0.0)
        # The archive keeps the report as strings that load without pickle.
        self.assertEqual(archived["symmetry_geometry_cache_key"], absent)
        self.assertEqual(
            len(dict(hashed.backend_info.details)["symmetry_geometry_cache_key"]), 64
        )
        # No cache is the default of a preparation; the report names the
        # directory of one that was used and no path that was not.
        self.assertEqual(details["symmetry_cache_directory"], "disabled")
        self.assertEqual(archived["symmetry_cache_directory"], "disabled")
        self.assertEqual(
            dict(hashed.backend_info.details)["symmetry_cache_directory"],
            str(directory.resolve()),
        )
        self.assertEqual(len(entries), 1)
        # That run kept its orbitals on the full grid and used the cache all
        # the same; "not used" is the answer of symmetry off alone.
        self.assertTrue(
            dict(hashed.backend_info.details)["orbital_symmetry"].startswith("full grid")
        )
        unreduced_details = dict(unreduced.backend_info.details)
        self.assertEqual(unreduced_details["symmetry_cache_directory"], "not used")
        self.assertNotIn(str(directory), " ".join(unreduced_details.values()))

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
            _stand_in_input(),
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


class MPIRolePreparationTests(unittest.TestCase):
    """Which components each MPI rank prepares, without CUDA or the extension."""

    def tearDown(self) -> None:
        with driver_module._REFERENCE_CACHE_LOCK:
            driver_module._REFERENCE_CACHE.clear()

    def prepare(
        self,
        mpi_context,
        linear_backend="native",
        environment=None,
        sector=None,
        ionic_devices=None,
        negative_laplacian=None,
        stencil_builder="csr",
        poisson_failure=None,
    ):
        """Prepare one rank against doubles of every heavy component.

        Four CUDA devices are visible and device 0 is current.  Settings in
        ``environment`` replace the defaults below, and ``sector`` the
        attributes of the sector eigensolver double.  ``ionic_devices`` maps
        the role the GPU ionic sums are told to the devices they report.
        ``negative_laplacian`` is the operator of the reference system and
        ``stencil_builder`` what the sector solver reports to have packed
        its stencils.  ``poisson_failure`` is raised where the reduced
        Poisson solver is constructed.
        """

        selection = _hybrid_selection()
        reduction = SimpleNamespace(
            group_order=4,
            multiplicities=np.full(8, 4),
            full_size=32,
            wedge_size=8,
            reduction_ratio=4.0,
        )
        cache_info = SimpleNamespace(
            status="disabled",
            key="content-key",
            path=None,
            hash_seconds=0.0,
            load_seconds=0.0,
            build_seconds=0.0,
            write_seconds=0.0,
            stencil_builder=stencil_builder,
        )
        decomposition = SimpleNamespace(
            sector_sizes=(8, 8, 8, 8),
            representation_count=4,
            reduction=reduction,
        )
        reference = SimpleNamespace(
            negative_laplacian=(
                object() if negative_laplacian is None else negative_laplacian
            ),
            nonlocal_operator=object(),
            core_density=object(),
            atoms=(),
            grid=SimpleNamespace(
                settings=SimpleNamespace(domain_shape="sphere"),
                volume_element=0.5,
            ),
            input=SimpleNamespace(
                hartree=SimpleNamespace(boundary_method="auto", multipole_order=9)
            ),
        )
        implementation = HybridHartreeWiringTests._cupy_backend_shell()
        implementation.eigensolver_operator = object()
        implementation.orbital_density_builder = object()
        implementation.local_potential = None
        sector_solver = SimpleNamespace(
            operator_cache_info=cache_info,
            scheduler_mode="multi-gpu",
            scheduler_workers=4,
            device_ids=(0, 1, 2, 3),
            bound_schedule="inline",
            collective_lanczos=False,
            finite_difference_storage="stencil",
            finite_difference_neighbors="shared",
            fused_projector_scatter=False,
            custom_projector_projection=False,
            projector_reduction_modes="none",
            later_filter_precision="float64",
            local_potential_storage="shared",
            assembled_sector_count=1,
            totally_symmetric_negative_laplacian=object(),
            totally_symmetric_stencil=object(),
        )
        vars(sector_solver).update(sector or {})
        boundary_builder = MagicMock(name="symmetry_boundary_builder")
        boundary_builder.cache_info = None
        doubles = SimpleNamespace(
            reference=reference,
            reduction=reduction,
            implementation=implementation,
            sector_solver=sector_solver,
            density_builder=object(),
        )
        acceleration = "parsec_python.acceleration."

        def reference_physics(*_arguments, **options):
            # As the GPU ionic sums of a reference built here would report.
            if ionic_devices is not None and "ionic_report" in options:
                devices = ionic_devices[options["ionic_sector_rank"]]
                if devices:
                    options["ionic_report"]["devices"] = devices
            return DEFAULT

        with (
            patch.object(driver_module, "resolve_backend", return_value=selection),
            patch.object(
                driver_module,
                "_prepare_reference_physics",
                return_value=reference,
                side_effect=reference_physics,
            ) as doubles.prepare_reference,
            patch.object(
                driver_module, "_build_backend", return_value=implementation
            ) as doubles.build_backend,
            patch(
                acceleration + "Symmetry.load_or_detect_reflection_reduction",
                return_value=(reduction, cache_info),
            ),
            patch(
                acceleration + "Symmetry.load_or_build_reflection_decomposition",
                return_value=(decomposition, cache_info),
            ),
            patch(
                acceleration + "Eigensolvers.CuPySymmetrySCFEigensolver",
                return_value=sector_solver,
            ) as doubles.sector_solver_type,
            patch(
                acceleration + "SCF.SymmetrySCFReducer",
                return_value=SimpleNamespace(
                    reduction=reduction,
                    mixer=object(),
                    potential_residual_metrics=object(),
                    total_energy=object(),
                ),
            ),
            patch(
                acceleration + "Occupations.CuPySymmetryDensityBuilder",
                return_value=doubles.density_builder,
            ),
            patch(
                acceleration + "V_xc.NativeCALDAEvaluator"
            ) as doubles.xc_evaluator_type,
            patch(
                acceleration + "Hartree.symmetry_poisson.SymmetryReducedPoissonSolver",
                side_effect=poisson_failure,
            ) as doubles.poisson_solver_type,
            patch(
                acceleration
                + "Hartree.native_boundary.NativeSymmetryMultipoleBoundaryBuilder",
                return_value=boundary_builder,
            ) as doubles.boundary_builder_type,
            patch(
                acceleration + "backends.native.native_build_info",
                return_value={"implemented_kernels": ("CALDAEvaluator",)},
            ),
            patch(
                acceleration + "Hartree.cupy_boundary.CuPyMultipoleBoundaryBuilder"
            ) as doubles.gpu_boundary_type,
            patch(
                acceleration + "backends.cupy.require_cupy",
                return_value=(
                    SimpleNamespace(
                        cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0))
                    ),
                    None,
                ),
            ) as doubles.require_cupy,
            patch(
                acceleration + "Eigensolvers.symmetry.cupy_device_count",
                return_value=4,
            ),
            patch.dict(
                driver_module.os.environ,
                {
                    "PARSEC_OVERLAP_CUDA_INITIALIZATION": "0",
                    "PARSEC_OVERLAP_HARTREE_SETUP": "1",
                    "PARSEC_HARTREE_LINEAR_BACKEND": linear_backend,
                    "PARSEC_HARTREE_BOUNDARY_BACKEND": "native",
                    "PARSEC_CUPY_RESIDENT_HARTREE": "0",
                    "PARSEC_CUPY_DEVICES": "0,1,2,3",
                    "PARSEC_IONIC_BACKEND": "native",
                    "PARSEC_OVERLAP_IONIC_SETUP": "1",
                    **(environment or {}),
                },
            ),
        ):
            for name in (
                "PARSEC_HARTREE_DEVICE", "PARSEC_SECTOR_STENCIL", *_BOUNDARY_INPUT_SWITCHES
            ):
                if name not in (environment or {}):
                    driver_module.os.environ.pop(name, None)
            options = {} if mpi_context is None else {"mpi_context": mpi_context}
            doubles.problem = object()
            doubles.system = driver_module.prepare_single_point(
                doubles.problem, backend="auto", **options
            )
        return doubles

    def test_stand_in_input_is_prepared_whatever_boundary_the_shell_names(self) -> None:
        # The switches of the Hartree boundary replace settings of the input
        # before the reference is prepared.  The stand-in of these tests has
        # none: a shell that names the former boundary for a run must not
        # stop them.
        shell = {
            "PARSEC_HARTREE_BOUNDARY": "legacy",
            "PARSEC_HARTREE_LPOLE": "7",
            "PARSEC_HARTREE_BOUNDARY_KERNEL": "full",
            "PARSEC_HARTREE_ATOMIC_TAIL_VALUES": "host",
        }
        with patch.dict(driver_module.os.environ, shell):
            prepared = self.prepare(None)
            prepared.prepare_reference.assert_called_once_with(
                prepared.problem,
                _hybrid_selection(),
                defer_native_laplacian=True,
                deferred_laplacian_cache_directory=None,
            )
            # The shell has its switches again.
            self.assertEqual(driver_module.os.environ["PARSEC_HARTREE_BOUNDARY"], "legacy")
            # A switch a test names itself still reaches the driver.
            with self.assertRaisesRegex(ValueError, "PARSEC_HARTREE_BOUNDARY"):
                self.prepare(None, environment={"PARSEC_HARTREE_BOUNDARY": "former"})

    def test_boundary_rebuilt_after_a_poisson_fallback_is_asked_for_as_the_first(
        self,
    ) -> None:
        # A reduced Poisson solver that cannot be built leaves the full-grid
        # one, and the boundary that the set-up thread built for the wedge
        # is replaced.  The second is asked for like the first: with the
        # tail in rows of the discarded builder, and otherwise with a tail
        # from the evaluator that one had, the device kernel.
        from parsec_python.Hartree import AtomicTail

        for carried in (
            AtomicTail(
                order=9, rows=np.zeros(0, dtype=np.int64), values=np.zeros(0),
                maximum=0.0, values_from="device kernel"),
            None,
        ):
            with self.subTest(carried=carried is not None):
                calls = []

                def build(_reference, reduction, **options):
                    calls.append((reduction, options))
                    builder = MagicMock(name=f"boundary_builder_{len(calls)}")
                    if len(calls) == 1 and carried is not None:
                        builder.boundary_tail = carried
                    return builder, reduction is not None, None, 0.0

                with (
                    patch.object(
                        driver_module, "_build_native_boundary_builder", side_effect=build
                    ),
                    patch(
                        "parsec_python.acceleration.Hartree.native_poisson."
                        "NativePoissonSolver"
                    ) as full_grid_solver,
                ):
                    prepared = self.prepare(
                        None, poisson_failure=ValueError("no reduced operator"))
                full_grid_solver.assert_called_once_with(
                    prepared.reference.negative_laplacian)
                (first_reduction, first), (second_reduction, second) = calls
                self.assertIs(first_reduction, prepared.reduction)
                self.assertIsNone(second_reduction)
                self.assertIs(first["tail_on_device"], True)
                self.assertIs(second["tail_on_device"], True)
                self.assertIs(second["boundary_tail"], carried)
                self.assertEqual(second["gpu_kernels"], first["gpu_kernels"])
                self.assertEqual(second["device_id"], first["device_id"])
                details = dict(prepared.system.backend_info.details)
                self.assertEqual(
                    details["hartree_boundary_setup"],
                    "overlap discarded after reduced-Poisson fallback",
                )
                self.assertEqual(details["hartree_symmetry"], "full grid")

    def test_root_and_serial_control_prepare_every_component(self) -> None:
        for mpi_context in (None, SimpleNamespace(rank=0, root=0, size=2)):
            with self.subTest(mpi_context=mpi_context):
                prepared = self.prepare(mpi_context)
                # The sectors take their stencils from the grid, so the
                # reference system is asked for no full-grid matrix.
                prepared.prepare_reference.assert_called_once_with(
                    prepared.problem,
                    _hybrid_selection(),
                    defer_native_laplacian=True,
                    deferred_laplacian_cache_directory=None,
                )
                prepared.poisson_solver_type.assert_called_once()
                self.assertIs(
                    prepared.poisson_solver_type.call_args.args[0],
                    prepared.sector_solver.totally_symmetric_negative_laplacian,
                )
                prepared.boundary_builder_type.assert_called_once()
                prepared.xc_evaluator_type.assert_called_once()
                self.assertIs(
                    prepared.xc_evaluator_type.call_args.args[0],
                    prepared.reference.core_density,
                )
                self.assertIs(
                    prepared.system.xc_evaluator,
                    prepared.xc_evaluator_type.return_value,
                )
                self.assertIs(
                    prepared.implementation.native_poisson_solver,
                    prepared.poisson_solver_type.return_value,
                )
                details = dict(prepared.system.backend_info.details)
                self.assertNotIn("mpi_rank_preparation", details)
                self.assertIn("hartree_cg_storage", details)
                self.assertEqual(
                    details["hartree_boundary_setup"],
                    "overlapped with GPU orbital setup",
                )

    def test_sector_worker_prepares_only_what_its_commands_use(self) -> None:
        root = self.prepare(SimpleNamespace(rank=0, root=0, size=2))
        context = SimpleNamespace(rank=1, root=0, size=2)
        worker = self.prepare(context)

        worker.prepare_reference.assert_called_once_with(
            worker.problem,
            _hybrid_selection(),
            defer_native_laplacian=True,
            deferred_laplacian_cache_directory=None,
            orbital_operators_only=True,
        )
        worker.poisson_solver_type.assert_not_called()
        worker.boundary_builder_type.assert_not_called()
        worker.xc_evaluator_type.assert_not_called()
        self.assertFalse(hasattr(worker.implementation, "native_poisson_solver"))
        self.assertFalse(hasattr(worker.implementation, "native_boundary_builder"))

        # The sector eigensolver and density builder its command loop needs
        # are constructed exactly as on the root.
        worker.sector_solver_type.assert_called_once()
        root.sector_solver_type.assert_called_once()
        worker_call = worker.sector_solver_type.call_args
        root_call = root.sector_solver_type.call_args
        self.assertIs(worker_call.kwargs["mpi_context"], context)
        self.assertEqual(
            sorted(worker_call.kwargs), sorted(root_call.kwargs)
        )
        self.assertEqual(len(worker_call.args), len(root_call.args))
        self.assertIs(worker_call.args[1], worker.reference.negative_laplacian)
        self.assertIs(worker_call.args[2], worker.reference.nonlocal_operator)
        self.assertIs(worker.system.eigenproblem_solver, worker.sector_solver)
        self.assertIs(
            worker.implementation.symmetry_eigensolver, worker.sector_solver
        )
        self.assertIs(
            worker.system.orbital_density_builder, worker.density_builder
        )
        self.assertIs(
            worker.build_backend.call_args.kwargs["defer_cupy_device_operator"],
            True,
        )

        # Entering the SCF loop there fails with the reason, not on a
        # missing field.
        for call in (
            lambda: worker.system.solve_hartree(np.zeros(8)),
            lambda: worker.system.evaluate_xc(np.zeros(8)),
        ):
            with self.assertRaisesRegex(RuntimeError, "MPI root rank only"):
                call()

        details = dict(worker.system.backend_info.details)
        self.assertIn("root rank", details["mpi_rank_preparation"])
        self.assertNotIn("hartree_cg_storage", details)
        self.assertNotIn("hartree_boundary_setup", details)
        self.assertEqual(details["hartree_backend"], "native")

    def test_gpu_cg_receives_the_packed_sector_stencil(self) -> None:
        for mpi_context in (None, SimpleNamespace(rank=0, root=0, size=2)):
            with self.subTest(mpi_context=mpi_context):
                # The native CG copies CSR buffers, so it is given a CSR.
                native = self.prepare(mpi_context)
                native_call = native.poisson_solver_type.call_args
                self.assertIs(
                    native_call.args[0],
                    native.sector_solver.totally_symmetric_negative_laplacian,
                )
                gpu = self.prepare(mpi_context, linear_backend="cupy")
                gpu.poisson_solver_type.assert_called_once()
                gpu_call = gpu.poisson_solver_type.call_args
                self.assertIs(
                    gpu_call.args[0], gpu.sector_solver.totally_symmetric_stencil
                )
                # Nothing else about the call depends on the operator form.
                self.assertIs(gpu_call.args[1], gpu.reduction)
                self.assertEqual(gpu_call.kwargs["operator_is_reduced"], True)
                self.assertEqual(native_call.kwargs["operator_is_reduced"], True)
                # The CG goes to the least loaded of the four devices, the
                # last one in both layouts.
                self.assertEqual(
                    gpu_call.kwargs["solver_factory"].keywords, {"device_id": 3}
                )

    def test_filter_precision_detail_is_the_statement_of_the_sector_eigensolver(
        self,
    ) -> None:
        key = "orbital_sector_later_filter_precision"
        prepared = self.prepare(None)
        self.assertEqual(dict(prepared.system.backend_info.details)[key], "float64")
        statement = (
            "float32 stencil/projectors/recurrence prepared for sectors 0 1 2 3; "
            "float64 Ritz and SCF"
        )
        for mpi_context in (None, SimpleNamespace(rank=0, root=0, size=2)):
            with self.subTest(mpi_context=mpi_context):
                prepared = self.prepare(
                    mpi_context, sector={"later_filter_precision": statement}
                )
                self.assertEqual(
                    dict(prepared.system.backend_info.details)[key], statement
                )

    def test_preparation_under_doubles_leaves_none_bound_in_the_gpu_ionic_module(
        self,
    ) -> None:
        from parsec_python.acceleration.backends import cupy as cupy_backend

        self.prepare(None)
        # The module as the driver has imported it: a first import inside a
        # preparation would have bound the double of ``require_cupy`` above
        # for every later GPU sum of the process.
        ionic = sys.modules["parsec_python.acceleration.V_ion.cupy_ionic"]
        self.assertIs(ionic.require_cupy, cupy_backend.require_cupy)
        self.assertIs(
            driver_module.ionic_gpu_count_setting, ionic.ionic_gpu_count_setting
        )

    def test_switches_without_a_stage_of_their_own_are_reported(self) -> None:
        names = (
            "PARSEC_SYMMETRY_FAST_MAPS",
            "PARSEC_CUPY_FILTER_GRAPH_REUSE",
            "PARSEC_CUPY_IMPLICIT_PACK_WORKERS",
        )
        shared, per_block = "one per block width and degree", "one per block of a plan"
        with patch.dict(driver_module.os.environ):
            for name in names:
                driver_module.os.environ.pop(name, None)
            for settings, storage, expected in (
                ((), "stencil", ("on", shared, "no tiles packed")),
                ((), "implicit_affine_tile_16", ("on", shared, "4")),
                (("0", "off", " 2 "), "implicit_affine_tile_16", ("off", per_block, "2")),
                (("no", "1", "3"), "mixed", ("off", shared, "3")),
                # The count that the packer reads, not the text of the setting.
                (("1", "on", "+3"), "implicit_affine_tile_16", ("on", shared, "3")),
                # No operator packs tiles under a setting that is no count.
                (("1", "on", "many"), "mixed", ("on", shared, "no tiles packed")),
            ):
                with self.subTest(settings=settings, storage=storage):
                    prepared = self.prepare(
                        SimpleNamespace(rank=0, root=0, size=2),
                        environment=dict(zip(names, settings)),
                        sector={"finite_difference_storage": storage},
                    )
                    details = dict(prepared.system.backend_info.details)
                    self.assertEqual(
                        (
                            details["symmetry_fast_maps"],
                            details["orbital_sector_filter_graphs"],
                            details["orbital_sector_tile_pack_workers"],
                        ),
                        expected,
                    )
            # A value the graph switch does not know stops the run before
            # anything is prepared, not at the first filter.
            with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_FILTER_GRAPH_REUSE"):
                self.prepare(None, environment={names[1]: "yes"})

    def test_gpu_ionic_sums_are_told_the_role_and_their_devices_are_reported(
        self,
    ) -> None:
        root = SimpleNamespace(rank=0, root=0, size=2)
        worker = SimpleNamespace(rank=1, root=0, size=2)
        # What the sums use: every device of a rank that solves sectors, one
        # otherwise, and none on a rank that builds no ionic field.
        devices = {True: (0, 1, 2, 3), False: (0,)}
        gpu = {"PARSEC_IONIC_BACKEND": "cupy"}
        with patch.dict(driver_module.os.environ):
            for name in ("PARSEC_IONIC_BACKEND", "PARSEC_IONIC_GPU_COUNT"):
                driver_module.os.environ.pop(name, None)
            # Another ionic backend: nothing is passed for the sums and
            # nothing reported.
            prepared = self.prepare(root, ionic_devices=devices)
            prepared.prepare_reference.assert_called_once_with(
                prepared.problem,
                _hybrid_selection(),
                defer_native_laplacian=True,
                deferred_laplacian_cache_directory=None,
            )
            details = dict(prepared.system.backend_info.details)
            self.assertEqual(details["ionic_gpu_count_requested"], "auto")
            self.assertEqual(details["ionic_gpu_devices"], "none")

            for mpi_context, sector_rank, used in (
                (None, False, "0"),
                (root, True, "0 1 2 3"),
            ):
                with self.subTest(mpi_context=mpi_context):
                    prepared = self.prepare(
                        mpi_context, environment=gpu, ionic_devices=devices
                    )
                    options = prepared.prepare_reference.call_args.kwargs
                    self.assertIs(options["ionic_sector_rank"], sector_rank)
                    self.assertEqual(
                        sorted(options),
                        [
                            "defer_native_laplacian",
                            "deferred_laplacian_cache_directory",
                            "ionic_field_overlap",
                            "ionic_report",
                            "ionic_sector_rank",
                        ],
                    )
                    details = dict(prepared.system.backend_info.details)
                    self.assertEqual(details["ionic_gpu_count_requested"], "auto")
                    self.assertEqual(details["ionic_gpu_devices"], used)

            prepared = self.prepare(
                worker, environment=gpu, ionic_devices={True: (), False: ()}
            )
            options = prepared.prepare_reference.call_args.kwargs
            self.assertIs(options["orbital_operators_only"], True)
            self.assertIs(options["ionic_sector_rank"], True)
            self.assertEqual(
                dict(prepared.system.backend_info.details)["ionic_gpu_devices"],
                "none",
            )

            prepared = self.prepare(
                None,
                environment={**gpu, "PARSEC_IONIC_GPU_COUNT": " 2 "},
                ionic_devices={True: (), False: (0, 1)},
            )
            details = dict(prepared.system.backend_info.details)
            self.assertEqual(details["ionic_gpu_count_requested"], "2")
            self.assertEqual(details["ionic_gpu_devices"], "0 1")

    def test_reference_physics_builds_the_gpu_ionic_sums_for_the_role(self) -> None:
        built = []

        class Builders:
            """Stands in for ``CupyIonicBuilders``; ``devices`` are those its sums use."""

            devices = (2, 3)

            def __init__(self, *, sector_rank=False):
                self.sector_rank, self.device_ids = sector_rank, ()
                built.append(self)

            def build_local_ionic_potential(self, *_arguments):
                self.device_ids = self.devices

            build_nonlocal_projectors = superpose_atomic_density = (
                build_local_ionic_potential
            )

        def reference(problem, *, local_ionic_builder, **_options):
            if isinstance(local_ionic_builder.__self__, Builders):
                local_ionic_builder()
            return prepared

        problem, prepared = object(), object()
        native = "parsec_python.acceleration.backends.native."
        with (
            patch.object(
                driver_module, "prepare_reference_single_point", side_effect=reference
            ),
            patch(native + "build_native_negative_laplacian"),
            patch(
                native + "_load_native",
                return_value=SimpleNamespace(RadialGridEvaluator=object),
            ),
            patch(
                "parsec_python.acceleration.V_ion.cupy_ionic.CupyIonicBuilders",
                Builders,
            ),
            patch.dict(driver_module.os.environ, {"PARSEC_IONIC_BACKEND": "cupy"}),
        ):
            for sector_rank in (False, True):
                report = {}
                result = driver_module._prepare_reference_physics(
                    problem,
                    _hybrid_selection(),
                    ionic_sector_rank=sector_rank,
                    ionic_report=report,
                )
                self.assertIs(result, prepared)
                self.assertIs(built[-1].sector_rank, sector_rank)
                self.assertEqual(report, {"devices": (2, 3)})
            # Sums that did not run leave the report as it was.
            with patch.object(Builders, "devices", ()):
                report = {}
                driver_module._prepare_reference_physics(
                    problem, _hybrid_selection(), ionic_report=report
                )
                self.assertEqual(report, {})
                self.assertIs(built[-1].sector_rank, False)
            # With the native ionic backend there are no such builders.
            driver_module.os.environ["PARSEC_IONIC_BACKEND"] = "native"
            report = {}
            driver_module._prepare_reference_physics(
                problem, _hybrid_selection(), ionic_sector_rank=True, ionic_report=report
            )
            self.assertEqual((report, len(built)), ({}, 3))

    def test_csr_switch_prepares_the_full_grid_matrix_first(self) -> None:
        environment = {"PARSEC_SECTOR_STENCIL": "csr"}
        root = self.prepare(None, environment=environment)
        root.prepare_reference.assert_called_once_with(
            root.problem, _hybrid_selection()
        )
        worker = self.prepare(
            SimpleNamespace(rank=1, root=0, size=2), environment=environment
        )
        worker.prepare_reference.assert_called_once_with(
            worker.problem, _hybrid_selection(), orbital_operators_only=True
        )
        with self.assertRaisesRegex(ValueError, "PARSEC_SECTOR_STENCIL"):
            self.prepare(None, environment={"PARSEC_SECTOR_STENCIL": "on"})

    def test_unknown_switch_value_is_refused_whatever_the_symmetry_mode(
        self,
    ) -> None:
        for symmetry, cache in (("off", None), ("auto", "cache"), ("auto", None)):
            with (
                self.subTest(symmetry=symmetry, cache=cache),
                _stand_in_input(),
                patch.dict(driver_module.os.environ, PARSEC_SECTOR_STENCIL="on"),
                patch.object(
                    driver_module,
                    "_resolve_and_prepare_reference",
                    side_effect=AssertionError("the reference was prepared"),
                ),
                self.assertRaisesRegex(ValueError, "PARSEC_SECTOR_STENCIL"),
            ):
                driver_module.prepare_single_point(
                    object(), symmetry=symmetry, symmetry_cache_directory=cache
                )

    def test_deferred_laplacian_is_neither_built_nor_hashed(self) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Laplacian import (
            DeferredNativeNegativeLaplacian,
        )
        from parsec_python.models import GridSettings

        grid = build_cluster_grid(
            GridSettings(spacing=0.8, radius=2.8, expansion_order=4)
        )
        absent = driver_module._reported_cache_key(None)
        for builder, reported in (
            ("direct-native", "skipped_by_direct_sector_stencil"),
            ("direct-numpy", "skipped_by_direct_sector_stencil"),
            ("cached", "skipped_by_exact_reduced_operator_cache"),
        ):
            with self.subTest(stencil_builder=builder):
                descriptor = DeferredNativeNegativeLaplacian(grid)
                with patch.object(
                    DeferredNativeNegativeLaplacian,
                    "materialize",
                    side_effect=AssertionError("the full-grid matrix was built"),
                ):
                    prepared = self.prepare(
                        SimpleNamespace(rank=0, root=0, size=2),
                        linear_backend="cupy",
                        negative_laplacian=descriptor,
                        stencil_builder=builder,
                    )
                self.assertFalse(descriptor.materialized)
                self.assertIsNone(descriptor.hashed_cache_key)
                call = prepared.sector_solver_type.call_args
                self.assertIs(call.args[1], descriptor)
                # No cache directory: the sector solver is handed no key.
                self.assertIsNone(call.kwargs["kinetic_cache_key"])
                self.assertIsNone(call.kwargs["operator_cache_directory"])
                details = dict(prepared.system.backend_info.details)
                self.assertEqual(
                    details["finite_difference_full_grid_materialization"], reported
                )
                self.assertEqual(details["orbital_operator_stencil_builder"], builder)
                self.assertEqual(details["finite_difference_provenance_key"], absent)
                self.assertEqual(
                    float(details["finite_difference_provenance_hash_seconds"]), 0.0
                )
                self.assertEqual(details["finite_difference_nnz_cache"], "disabled")

    def test_report_tells_a_built_matrix_from_a_shared_one(self) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.Laplacian import build_negative_laplacian
        from parsec_python.acceleration.Laplacian import (
            DeferredNativeNegativeLaplacian,
        )
        from parsec_python.models import GridSettings

        grid = build_cluster_grid(
            GridSettings(spacing=0.8, radius=2.8, expansion_order=4)
        )
        cached = DeferredNativeNegativeLaplacian(grid)
        with patch(
            "parsec_python.acceleration.backends.native."
            "build_native_negative_laplacian",
            side_effect=build_negative_laplacian,
        ) as builder:
            reported = []
            for _ in range(2):
                descriptor = DeferredNativeNegativeLaplacian(
                    grid, sharing_matrix_with=cached
                )
                descriptor.materialize()
                prepared = self.prepare(
                    None, linear_backend="cupy", negative_laplacian=descriptor
                )
                details = dict(prepared.system.backend_info.details)
                reported.append(
                    (
                        details["finite_difference_full_grid_materialization"],
                        float(details["finite_difference_materialization_seconds"])
                        > 0.0,
                    )
                )
        builder.assert_called_once()
        self.assertEqual(
            reported,
            [("performed", True), ("reused_from_resident_reference", False)],
        )

    def test_reported_build_workers_follow_the_assembled_sectors(self) -> None:
        from parsec_python.acceleration.Symmetry import operator_build_workers

        with patch.dict(driver_module.os.environ):
            driver_module.os.environ.pop("PARSEC_SYMMETRY_OPERATOR_WORKERS", None)
            prepared = self.prepare(SimpleNamespace(rank=1, root=0, size=4))
            details = dict(prepared.system.backend_info.details)
            # One assembled sector is packed serially although the group has four.
            self.assertEqual(operator_build_workers(1), 1)
            self.assertEqual(operator_build_workers(4), 4)
            self.assertEqual(details["orbital_operator_build_workers"], "1")

    def test_worker_reference_is_reduced_and_never_cached(self) -> None:
        problem = parse_parsec_input(SMOKE_INPUT).problem
        selection = BackendSelection(
            requested="scipy",
            selected="scipy",
            finite_difference_builder="reference",
            hartree_backend="scipy",
        )
        with patch.dict(
            driver_module.os.environ,
            {
                "PARSEC_ACCELERATED_RESIDENT": "1",
                "PARSEC_RESIDENT_REFERENCE_CACHE_SIZE": "1",
            },
        ):
            reduced = driver_module._prepare_reference_physics(
                problem, selection, orbital_operators_only=True
            )
            self.assertEqual(len(driver_module._REFERENCE_CACHE), 0)
            complete = driver_module._prepare_reference_physics(problem, selection)
            self.assertEqual(len(driver_module._REFERENCE_CACHE), 1)
            again = driver_module._prepare_reference_physics(
                problem, selection, orbital_operators_only=True
            )

        self.assertIsNone(reduced.ionic_potential)
        self.assertIsNone(reduced.initial_density)
        self.assertIsNone(reduced.core_density)
        # A cached complete system does not stand in for the reduced one.
        self.assertIsNone(again.ionic_potential)
        self.assertEqual(complete.ionic_potential.shape, (complete.grid.size,))
        self.assertEqual(reduced.grid.size, complete.grid.size)
        for name in ("indptr", "indices", "data"):
            np.testing.assert_array_equal(
                getattr(reduced.negative_laplacian, name),
                getattr(complete.negative_laplacian, name),
            )
            np.testing.assert_array_equal(
                getattr(reduced.nonlocal_operator.projectors, name),
                getattr(complete.nonlocal_operator.projectors, name),
            )


class IonicFieldOverlapTests(unittest.TestCase):
    """PARSEC_OVERLAP_IONIC_SETUP: the root-only ionic fields beside host set-up."""

    prepare = MPIRolePreparationTests.prepare
    gpu_fields = {"PARSEC_IONIC_BACKEND": "cupy"}
    overlapped = "overlapped with symmetry and orbital setup"
    # The reference preparation of every rank is asked for the deferred
    # Laplacian, and with GPU ionic fields told the role of their sums.
    deferred = {
        "defer_native_laplacian": True,
        "deferred_laplacian_cache_directory": None,
    }
    with_overlap = sorted(
        [*deferred, "ionic_sector_rank", "ionic_report", "ionic_field_overlap"]
    )

    def tearDown(self) -> None:
        with driver_module._REFERENCE_CACHE_LOCK:
            driver_module._REFERENCE_CACHE.clear()

    @staticmethod
    def _starting_the_fields(**builders):
        """Start the fields where the reference preparation does, around its double."""

        resolve = driver_module._resolve_and_prepare_reference

        def resolving(problem, backend, **options):
            selection, reference, timing = resolve(problem, backend, **options)
            options["ionic_field_overlap"].start(reference, **builders)
            return selection, reference, timing

        return patch.object(driver_module, "_resolve_and_prepare_reference", resolving)

    @staticmethod
    def _doubles_in_place():
        """Sector solver, XC evaluator and Poisson solver that ``prepare`` has patched in."""

        import parsec_python.acceleration.Eigensolvers as eigensolvers
        import parsec_python.acceleration.V_xc as xc
        from parsec_python.acceleration.Hartree import symmetry_poisson

        return (
            eigensolvers.CuPySymmetrySCFEigensolver,
            xc.NativeCALDAEvaluator,
            symmetry_poisson.SymmetryReducedPoissonSolver,
        )

    @staticmethod
    def _ionic_threads():
        return [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("parsec-ionic-setup")
        ]

    def test_overlap_needs_the_switch_and_gpu_ionic_fields(self) -> None:
        for environment, expected in (
            ({}, False),
            ({"PARSEC_IONIC_BACKEND": "native"}, False),
            ({"PARSEC_OVERLAP_IONIC_SETUP": "1"}, False),
            # Unset is on.
            ({"PARSEC_IONIC_BACKEND": "cupy"}, True),
            ({"PARSEC_IONIC_BACKEND": "CuPy", "PARSEC_OVERLAP_IONIC_SETUP": " 1 "}, True),
            ({"PARSEC_IONIC_BACKEND": "cupy", "PARSEC_OVERLAP_IONIC_SETUP": "0"}, False),
            ({"PARSEC_IONIC_BACKEND": "cupy", "PARSEC_OVERLAP_IONIC_SETUP": " Off "}, False),
            ({"PARSEC_IONIC_BACKEND": "cupy", "PARSEC_OVERLAP_IONIC_SETUP": "false"}, False),
            ({"PARSEC_IONIC_BACKEND": "cupy", "PARSEC_OVERLAP_IONIC_SETUP": "no"}, False),
        ):
            with self.subTest(environment=environment), patch.dict(
                driver_module.os.environ
            ):
                for name in ("PARSEC_IONIC_BACKEND", "PARSEC_OVERLAP_IONIC_SETUP"):
                    driver_module.os.environ.pop(name, None)
                driver_module.os.environ.update(environment)
                self.assertIs(driver_module._ionic_overlap_requested(), expected)

    def test_fields_are_built_beside_the_sector_solver_and_joined_before_xc(
        self,
    ) -> None:
        for mpi_context in (None, SimpleNamespace(rank=0, root=0, size=2)):
            with self.subTest(mpi_context=mpi_context):
                core_density = object()
                record = SimpleNamespace(
                    threads=[], builders=None, complete=None, xc_calls=None
                )

                def complete(reference, **builders):
                    record.threads.append(threading.current_thread())
                    record.builders = builders
                    solver_type, xc_type, _ = self._doubles_in_place()
                    # Returns only once the sector solver has been built: a
                    # join placed before that construction would not return.
                    pause = threading.Event()
                    for _ in range(500):
                        if solver_type.called:
                            break
                        pause.wait(timeout=0.02)
                    else:
                        raise AssertionError("the fields were joined too early")
                    # The caller is now at the join or on its way there.
                    record.xc_calls = xc_type.call_count
                    record.complete = SimpleNamespace(
                        **{**vars(reference), "core_density": core_density}
                    )
                    return record.complete

                def sum_devices():
                    # Asked on the calling thread, once the thread has ended.
                    record.asked = (
                        threading.current_thread(),
                        record.threads[0].is_alive(),
                    )
                    return (1, 2)

                with (
                    self._starting_the_fields(
                        sum_devices=sum_devices,
                        local_ionic_builder="local",
                        atomic_density_builder="density",
                    ),
                    patch.object(
                        driver_module, "complete_reference_single_point", complete
                    ),
                ):
                    prepared = self.prepare(mpi_context, environment=self.gpu_fields)

                self.assertEqual(record.xc_calls, 0)
                self.assertEqual(record.asked, (threading.main_thread(), False))
                call = prepared.prepare_reference.call_args
                self.assertEqual(call.args, (prepared.problem, _hybrid_selection()))
                self.assertEqual(sorted(call.kwargs), self.with_overlap)
                self.assertEqual(
                    record.builders,
                    {"local_ionic_builder": "local", "atomic_density_builder": "density"},
                )
                # One thread of its own, which the join has ended.
                (thread,) = record.threads
                self.assertIsNot(thread, threading.main_thread())
                self.assertTrue(thread.name.startswith("parsec-ionic-setup"))
                self.assertFalse(thread.is_alive())
                # The first reader takes the field of the completed system,
                # which is the one the prepared system carries.
                prepared.xc_evaluator_type.assert_called_once()
                self.assertIs(
                    prepared.xc_evaluator_type.call_args.args[0], core_density
                )
                self.assertIs(prepared.system.reference, record.complete)
                self.assertIsNot(prepared.system.reference, prepared.reference)
                # The sector solver was given the operators of the system
                # prepared without the fields, which the completed one shares.
                sector_call = prepared.sector_solver_type.call_args
                self.assertIs(sector_call.args[1], prepared.reference.negative_laplacian)
                self.assertIs(sector_call.args[2], prepared.reference.nonlocal_operator)
                details = dict(prepared.system.backend_info.details)
                self.assertEqual(details["ionic_setup"], self.overlapped)
                # The devices of sums that ran after the reference
                # preparation had returned.
                self.assertEqual(details["ionic_gpu_devices"], "1 2")
                seconds = float(details["ionic_setup_seconds"])
                waited = float(details["ionic_setup_wait_seconds"])
                self.assertGreater(seconds, 0.0)
                self.assertGreaterEqual(waited, 0.0)
                self.assertAlmostEqual(
                    float(details["ionic_setup_overlapped_seconds"]),
                    max(0.0, seconds - waited),
                    5,
                )

    def test_switch_off_and_host_fields_keep_the_inline_preparation(self) -> None:
        for environment in (
            {**self.gpu_fields, "PARSEC_OVERLAP_IONIC_SETUP": "0"},
            {"PARSEC_IONIC_BACKEND": "native"},
        ):
            for mpi_context in (None, SimpleNamespace(rank=0, root=0, size=2)):
                with self.subTest(environment=environment, mpi_context=mpi_context):
                    prepared = self.prepare(mpi_context, environment=environment)
                    ionic_sums = (
                        {"ionic_sector_rank": mpi_context is not None, "ionic_report": {}}
                        if environment["PARSEC_IONIC_BACKEND"] == "cupy"
                        else {}
                    )
                    prepared.prepare_reference.assert_called_once_with(
                        prepared.problem,
                        _hybrid_selection(),
                        **self.deferred,
                        **ionic_sums,
                    )
                    self.assertIs(prepared.system.reference, prepared.reference)
                    details = dict(prepared.system.backend_info.details)
                    self.assertEqual(details["ionic_setup"], "inline")
                    self.assertNotIn("ionic_setup_seconds", details)
        self.assertEqual(self._ionic_threads(), [])

    def test_preparation_that_starts_no_thread_is_not_joined(self) -> None:
        # The preparation decides: it keeps the fields in line for a resident
        # process, for instance.
        prepared = self.prepare(None, environment=self.gpu_fields)
        self.assertEqual(
            sorted(prepared.prepare_reference.call_args.kwargs), self.with_overlap
        )
        self.assertIs(prepared.system.reference, prepared.reference)
        self.assertEqual(
            dict(prepared.system.backend_info.details)["ionic_setup"], "inline"
        )

    def test_sector_worker_has_no_ionic_fields_to_overlap(self) -> None:
        worker = self.prepare(
            SimpleNamespace(rank=1, root=0, size=2), environment=self.gpu_fields
        )
        worker.prepare_reference.assert_called_once_with(
            worker.problem,
            _hybrid_selection(),
            **self.deferred,
            orbital_operators_only=True,
            ionic_sector_rank=True,
            ionic_report={},
        )
        self.assertNotIn("ionic_setup", dict(worker.system.backend_info.details))

    def test_failure_of_the_fields_is_raised_at_the_join(self) -> None:
        doubles = []

        def fail(_reference, **_builders):
            doubles.extend(self._doubles_in_place())
            raise RuntimeError("ionic kernel failed")

        with (
            self._starting_the_fields(),
            patch.object(driver_module, "complete_reference_single_point", fail),
            self.assertRaisesRegex(RuntimeError, "ionic kernel failed"),
        ):
            self.prepare(None, environment=self.gpu_fields)
        sector_solver_type, xc_evaluator_type, poisson_solver_type = doubles
        self.assertEqual(self._ionic_threads(), [])
        # The sectors were built; nothing that reads an ionic field was.
        sector_solver_type.assert_called_once()
        xc_evaluator_type.assert_not_called()
        poisson_solver_type.assert_not_called()

    def test_thread_that_is_never_joined_ends_with_its_task(self) -> None:
        # A failure of the preparation between the start and the join leaves
        # the thread to itself; it must not wait for a collection to end.
        release = threading.Event()
        overlap = driver_module._OverlappedIonicFields()
        with patch.object(
            driver_module,
            "complete_reference_single_point",
            lambda _reference: release.wait(timeout=10.0),
        ):
            overlap.start(object())
            (thread,) = self._ionic_threads()
            self.assertTrue(thread.is_alive())
            release.set()
            thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())

    class _HostFields:
        """Stands for the GPU builders: the reference routines, with their thread."""

        calls: list[tuple[str, str]] = []
        # The devices a sum says it ran on.
        devices = (2, 3)

        def __init__(self, *, sector_rank=False):
            self.sector_rank = sector_rank
            self.device_ids = ()

        def _note(self, name):
            type(self).calls.append((name, threading.current_thread().name))

        def keep_current_device(self):
            self._note("current device")

        def build_local_ionic_potential(self, *arguments):
            from parsec_python.V_ion import build_local_ionic_potential

            self._note("local")
            self.device_ids = self.devices
            return build_local_ionic_potential(*arguments)

        def superpose_atomic_density(self, *arguments, core=False):
            from parsec_python.V_ion import superpose_atomic_density

            self._note("core" if core else "valence")
            self.device_ids = self.devices
            return superpose_atomic_density(*arguments, core=core)

        def build_nonlocal_projectors(self, *arguments):
            from parsec_python.V_ion import build_nonlocal_projectors

            self._note("projectors")
            return build_nonlocal_projectors(*arguments)

    def _reference_physics(self, environment, **options):
        """Run the reference preparation of the hybrid path on the host."""

        from parsec_python.Laplacian import build_negative_laplacian

        problem = parse_parsec_input(SMOKE_INPUT).problem
        # A displaced second atom gives every field and the ion-ion energy
        # a value that depends on the atom order.
        second = replace(problem.atoms[0], position=np.array((0.9, -0.4, 0.3)))
        problem = replace(
            problem, atoms=[*problem.atoms, second], recenter_geometry=False
        )
        self._HostFields.calls = []
        native = "parsec_python.acceleration.backends.native."
        with (
            patch(
                native + "_load_native",
                return_value=SimpleNamespace(RadialGridEvaluator=object),
            ),
            patch(native + "build_native_negative_laplacian", build_negative_laplacian),
            patch("parsec_python.acceleration.V_ion.NativeIonicBuilders", self._HostFields),
            patch(
                "parsec_python.acceleration.V_ion.cupy_ionic.CupyIonicBuilders",
                self._HostFields,
            ),
            patch.dict(
                driver_module.os.environ,
                {"PARSEC_ACCELERATED_RESIDENT": "0", **environment},
            ),
        ):
            reference = driver_module._prepare_reference_physics(
                problem, _hybrid_selection(), **options
            )
        return reference, list(self._HostFields.calls)

    def test_deferred_fields_are_bitwise_the_inline_ones(self) -> None:
        inline, inline_calls = self._reference_physics(self.gpu_fields)
        self.assertEqual(
            inline_calls,
            [
                (name, "MainThread")
                for name in ("local", "projectors", "valence", "core")
            ],
        )

        # The in-line sums report their devices when the preparation returns.
        report = {}
        self._reference_physics(self.gpu_fields, ionic_report=report)
        self.assertEqual(report, {"devices": (2, 3)})
        self.assertNotIn("current device", dict(self._HostFields.calls))

        overlap = driver_module._OverlappedIonicFields()
        self.assertFalse(overlap.started)
        self.assertEqual(overlap.details(), (("ionic_setup", "inline"),))
        report = {}
        reduced, _ = self._reference_physics(
            self.gpu_fields, ionic_field_overlap=overlap, ionic_report=report
        )
        self.assertTrue(overlap.started)
        # The sums of the thread are not over when the preparation returns:
        # their devices are the overlap's to report, once it is joined.
        self.assertEqual(report, {})
        self.assertEqual(overlap.devices, ())
        # What the sector solver reads is there at once; the rest is absent.
        for name in ("ionic_potential", "initial_density", "core_density"):
            self.assertIsNone(getattr(reduced, name))
        self.assertTrue(np.isnan(reduced.ion_ion_energy))
        self.assertGreater(reduced.nonlocal_operator.projectors.shape[0], 0)
        complete = overlap.result()
        self.assertEqual(overlap.devices, (2, 3))
        calls = dict(self._HostFields.calls)
        self.assertEqual(calls["projectors"], "MainThread")
        # The thread that prepares has its current device kept for the sums.
        self.assertEqual(calls["current device"], "MainThread")
        for name in ("local", "valence", "core"):
            self.assertTrue(calls[name].startswith("parsec-ionic-setup"), calls)
        # In the order of the in-line stages, the device kept before them.
        self.assertEqual(
            [name for name, _ in self._HostFields.calls if name != "projectors"],
            ["current device", "local", "valence", "core"],
        )

        for name in ("ionic_potential", "initial_density", "core_density"):
            with self.subTest(field=name):
                left, right = getattr(complete, name), getattr(inline, name)
                self.assertEqual(left.dtype, right.dtype)
                self.assertEqual(left.tobytes(), right.tobytes())
        self.assertGreater(float(np.ptp(complete.ionic_potential)), 0.0)
        self.assertGreater(complete.ion_ion_energy, 0.0)
        self.assertEqual(complete.ion_ion_energy, inline.ion_ion_energy)
        self.assertEqual(
            complete.atomic_reference_correction, inline.atomic_reference_correction
        )
        self.assertEqual(complete.electron_count, inline.electron_count)
        for name in ("grid", "negative_laplacian", "nonlocal_operator"):
            self.assertIs(getattr(complete, name), getattr(reduced, name))
        for name in ("indptr", "indices", "data"):
            np.testing.assert_array_equal(
                getattr(complete.negative_laplacian, name),
                getattr(inline.negative_laplacian, name),
            )
            np.testing.assert_array_equal(
                getattr(complete.nonlocal_operator.projectors, name),
                getattr(inline.nonlocal_operator.projectors, name),
            )
        self.assertGreater(overlap.seconds, 0.0)
        self.assertEqual(overlap.details()[0], ("ionic_setup", self.overlapped))
        # The stage times of the fields are the thread's.  The wall time is
        # that of the preparation that left them out, not a sum of calls
        # that ran beside each other.
        self.assertEqual(reduced.timings.local_ionic_seconds, 0.0)
        self.assertGreater(complete.timings.local_ionic_seconds, 0.0)
        self.assertGreater(reduced.timings.total_seconds, 0.0)
        self.assertEqual(
            complete.timings.total_seconds, reduced.timings.total_seconds
        )
        with self.assertRaisesRegex(RuntimeError, "start once"):
            overlap.start(reduced)
        self.assertEqual(self._ionic_threads(), [])

    def test_preparation_keeps_the_fields_inline_where_they_cannot_overlap(
        self,
    ) -> None:
        resident = {
            "PARSEC_ACCELERATED_RESIDENT": "1",
            "PARSEC_RESIDENT_REFERENCE_CACHE_SIZE": "1",
        }
        for label, environment, options in (
            # A resident process stores the complete system under its key.
            ("resident", {**self.gpu_fields, **resident}, {}),
            ("host builders", {"PARSEC_IONIC_BACKEND": "native"}, {}),
            ("sector worker", self.gpu_fields, {"orbital_operators_only": True}),
        ):
            with self.subTest(case=label):
                overlap = driver_module._OverlappedIonicFields()
                reference, calls = self._reference_physics(
                    environment, ionic_field_overlap=overlap, **options
                )
                self.assertFalse(overlap.started)
                self.assertTrue(all(thread == "MainThread" for _, thread in calls))
                if options:
                    self.assertIsNone(reference.ionic_potential)
                    self.assertEqual(calls, [("projectors", "MainThread")])
                else:
                    self.assertEqual(
                        reference.ionic_potential.shape, (reference.grid.size,)
                    )
                self.assertEqual(
                    len(driver_module._REFERENCE_CACHE), int(label == "resident")
                )
                with driver_module._REFERENCE_CACHE_LOCK:
                    driver_module._REFERENCE_CACHE.clear()
        self.assertEqual(self._ionic_threads(), [])


class HartreeDevicePlacementTests(unittest.TestCase):
    """PARSEC_HARTREE_DEVICE: its values, the choice, and where it applies."""

    prepare = MPIRolePreparationTests.prepare

    @staticmethod
    def request(value):
        with patch.dict(driver_module.os.environ):
            if value is None:
                driver_module.os.environ.pop("PARSEC_HARTREE_DEVICE", None)
            else:
                driver_module.os.environ["PARSEC_HARTREE_DEVICE"] = value
            return driver_module._hartree_device_request()

    def test_request_is_auto_off_or_a_device_index(self) -> None:
        # Unset is auto.
        self.assertEqual(self.request(None), "auto")
        self.assertEqual(self.request("  "), "auto")
        self.assertEqual(self.request(" Auto "), "auto")
        self.assertIsNone(self.request(" Off "))
        self.assertEqual(self.request("0"), 0)
        self.assertEqual(self.request(" 3 "), 3)
        for invalid in ("-1", "1.5", "last", "0,1"):
            with (
                self.subTest(value=invalid),
                self.assertRaisesRegex(ValueError, "PARSEC_HARTREE_DEVICE"),
            ):
                self.request(invalid)

    def test_off_and_explicit_requests(self) -> None:
        select = driver_module._select_hartree_device
        groups = {0: (0, 1, 2, 3)}
        self.assertIsNone(select(None, (0, 1, 2, 3), groups))
        self.assertEqual(select(2, (0, 1, 2, 3), groups), 2)
        self.assertEqual(select(0, (0, 1, 2, 3), groups), 0)
        with self.assertRaisesRegex(ValueError, "PARSEC_HARTREE_DEVICE"):
            select(4, (0, 1, 2, 3), groups)
        # One device leaves nothing to choose.
        self.assertIsNone(select("auto", (2,), {0: (2,)}))

    def test_auto_takes_the_least_loaded_device(self) -> None:
        select = driver_module._select_hartree_device
        cases = (
            # One sector on the four devices of a rank (16 GPUs).
            ((0, 1, 2, 3), {0: (0, 1, 2, 3)}, 3),
            # Two sectors on pairs (8 GPUs): devices 0 and 2 own them.
            ((0, 1, 2, 3), {0: (0, 1), 2: (2, 3)}, 3),
            # Every device owns one sector (4 GPUs, one rank).
            ((0, 1, 2, 3), {0: (0,), 1: (1,), 2: (2,), 3: (3,)}, 3),
            # A sector alone on the last device carries a whole basis.
            ((0, 1, 2), {0: (0, 1), 1: (2,)}, 1),
            ((0, 1, 2, 3), {0: (0, 1), 3: (2,), 6: (3,)}, 1),
            # More sectors than devices: the first device has one more.
            ((0, 1), {0: (0,), 1: (1,), 2: (0,)}, 1),
            ((4, 5, 6), {0: (4,), 1: (5,), 2: (6,), 3: (4,)}, 6),
            # Devices without any sector.
            ((0, 1, 2, 3), {0: (0,), 1: (1,)}, 3),
            # A full-grid operator on a device that is not listed first.
            ((0, 1), {0: (1,)}, 0),
        )
        for devices, groups, expected in cases:
            with self.subTest(devices=devices, groups=groups):
                self.assertEqual(select("auto", devices, groups), expected)

    def test_auto_never_takes_the_first_sector_device(self) -> None:
        from parsec_python.acceleration.experimental.mpi_scf import (
            sector_device_groups,
        )

        select = driver_module._select_hartree_device
        plan = driver_module._planned_sector_device_groups
        # One process hands the sectors to its devices in turn.
        self.assertEqual(
            plan(4, (0, 1), None), {0: (0,), 1: (1,), 2: (0,), 3: (1,)}
        )
        for sectors in (2, 4, 8):
            for device_count in range(2, 7):
                devices = tuple(range(1, device_count + 1))
                layouts = [plan(sectors, devices, None)]
                for size in range(1, sectors + 1):
                    for rank in range(size):
                        groups = plan(
                            sectors, devices, SimpleNamespace(size=size, rank=rank)
                        )
                        self.assertEqual(
                            groups,
                            sector_device_groups(sectors, size, rank, devices),
                        )
                        layouts.append(groups)
                for groups in layouts:
                    choice = select("auto", devices, groups)
                    share = dict.fromkeys(devices, 0.0)
                    for group in groups.values():
                        for device in group:
                            share[device] += 1.0 / len(group)
                    with self.subTest(devices=devices, groups=groups):
                        self.assertNotEqual(choice, devices[0])
                        self.assertEqual(share[choice], min(share.values()))

    def placement(self, mpi_context, environment):
        """Devices handed to the GPU CG factory and to the GPU boundary."""

        prepared = self.prepare(
            mpi_context,
            linear_backend="cupy",
            environment={
                "PARSEC_HARTREE_BOUNDARY_BACKEND": "cupy",
                **environment,
            },
        )
        prepared.gpu_boundary_type.assert_called_once()
        boundary_call = prepared.gpu_boundary_type.call_args
        self.assertEqual(boundary_call.args, (prepared.reference.grid, 9))
        factory = prepared.poisson_solver_type.call_args.kwargs["solver_factory"]
        return factory.keywords["device_id"], boundary_call.kwargs["device_id"]

    def test_off_keeps_the_former_placement(self) -> None:
        # The CG on the first sector device; the boundary builder on the
        # device current in its thread, which ``None`` stands for.
        off = {"PARSEC_HARTREE_DEVICE": "off"}
        for mpi_context in (None, SimpleNamespace(rank=0, root=0, size=4)):
            with self.subTest(mpi_context=mpi_context):
                self.assertEqual(self.placement(mpi_context, off), (0, None))

    def test_auto_asks_for_a_device_only_where_a_hartree_object_is_on_a_gpu(
        self,
    ) -> None:
        rank_of_four = SimpleNamespace(rank=0, root=0, size=4)
        # CG and boundary on the host, the default: nothing to place.
        for request in ({}, {"PARSEC_HARTREE_DEVICE": "auto"}):
            with self.subTest(environment=request):
                prepared = self.prepare(rank_of_four, environment=request)
                prepared.require_cupy.assert_not_called()
                prepared.boundary_builder_type.assert_called_once()
        # Either of them on a GPU is placed.
        prepared = self.prepare(rank_of_four, linear_backend="cupy")
        self.assertEqual(
            prepared.poisson_solver_type.call_args.kwargs["solver_factory"].keywords,
            {"device_id": 3},
        )
        prepared = self.prepare(
            rank_of_four, environment={"PARSEC_HARTREE_BOUNDARY_BACKEND": "cupy"}
        )
        self.assertEqual(
            prepared.gpu_boundary_type.call_args.kwargs["device_id"], 3
        )
        # An index is checked against the devices of the process even so.
        with self.assertRaisesRegex(ValueError, "PARSEC_HARTREE_DEVICE"):
            self.prepare(rank_of_four, environment={"PARSEC_HARTREE_DEVICE": "4"})

    def test_cg_and_boundary_move_together(self) -> None:
        rank_of_four = SimpleNamespace(rank=0, root=0, size=4)
        rank_of_two = SimpleNamespace(rank=0, root=0, size=2)
        auto = {"PARSEC_HARTREE_DEVICE": "auto"}
        # Unset is auto.
        self.assertEqual(self.placement(rank_of_four, {}), (3, 3))
        self.assertEqual(self.placement(rank_of_four, auto), (3, 3))
        self.assertEqual(self.placement(rank_of_two, auto), (3, 3))
        self.assertEqual(self.placement(None, auto), (3, 3))
        self.assertEqual(
            self.placement(rank_of_four, {"PARSEC_HARTREE_DEVICE": "2"}), (2, 2)
        )
        self.assertEqual(
            self.placement(rank_of_four, {**auto, "PARSEC_CUPY_DEVICES": "1,2"}),
            (2, 2),
        )
        # A single device: nothing moves.
        self.assertEqual(
            self.placement(rank_of_four, {**auto, "PARSEC_CUPY_DEVICES": "0"}),
            (0, None),
        )
        with self.assertRaisesRegex(ValueError, "PARSEC_HARTREE_DEVICE"):
            self.placement(
                rank_of_four,
                {"PARSEC_HARTREE_DEVICE": "3", "PARSEC_CUPY_DEVICES": "0,1"},
            )

    def test_the_sector_eigensolver_is_told_the_device_of_the_hartree_objects(
        self,
    ) -> None:
        # A sector that shares its basis gives that device, the fullest of
        # the process, no work beside its own.
        rank_of_four = SimpleNamespace(rank=0, root=0, size=4)
        for environment, expected in (
            ({}, 3),
            ({"PARSEC_HARTREE_DEVICE": "2"}, 2),
            ({"PARSEC_HARTREE_DEVICE": "off"}, None),
        ):
            with self.subTest(environment=environment):
                prepared = self.prepare(
                    rank_of_four, linear_backend="cupy", environment=environment
                )
                self.assertEqual(prepared.sector_solver.hartree_device, expected)
        # A sector worker has no Hartree object, and nothing to keep free.
        prepared = self.prepare(
            SimpleNamespace(rank=1, root=0, size=4), linear_backend="cupy"
        )
        self.assertIsNone(prepared.sector_solver.hartree_device)

    def test_sector_worker_builds_no_hartree_object_to_place(self) -> None:
        # It chooses no device either: an index outside its own devices,
        # an error on the root, is not one there.
        for request, devices in (("auto", "0,1,2,3"), ("3", "0,1")):
            with self.subTest(PARSEC_HARTREE_DEVICE=request):
                prepared = self.prepare(
                    SimpleNamespace(rank=1, root=0, size=4),
                    linear_backend="cupy",
                    environment={
                        "PARSEC_HARTREE_BOUNDARY_BACKEND": "cupy",
                        "PARSEC_HARTREE_DEVICE": request,
                        "PARSEC_CUPY_DEVICES": devices,
                    },
                )
                prepared.gpu_boundary_type.assert_not_called()
                prepared.poisson_solver_type.assert_not_called()


@unittest.skipUnless(
    HYBRID_AVAILABLE,
    "both parsec_accelerated_native and CuPy/CUDA are required",
)
class RealHybridAccuracyTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Prepared systems that a test left in a reference cycle are destroyed
        # here, between the tests, and not by a collection inside the next one.
        gc.collect()

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

    def test_exported_symmetry_orbitals_reproduce_their_ritz_values(self) -> None:
        problem = parse_parsec_input(SMOKE_INPUT).problem
        system = prepare_single_point(problem, backend="auto")
        solver = system.backend.symmetry_eigensolver
        # The sector index maps and the full-grid expansion maps stay on the
        # host; the export below is their only reader.
        for name in (
            "_device_full_to_wedge",
            "_device_phases",
            "_device_sector_orbits",
            "_device_sector_scales",
        ):
            self.assertFalse(hasattr(solver, name))
        for representation, orbits in enumerate(solver._sector_orbits):
            self.assertIs(
                orbits, solver.decomposition.sector_orbit_indices(representation)
            )
        actual = run_scf(system)

        # Signed full-grid states against the full-grid reference operator:
        # a wrong phase or orbit map changes these Rayleigh quotients.
        orbitals = np.asarray(actual.wavefunctions)
        self.assertEqual(
            orbitals.shape, (system.grid.size, actual.eigenvalues.size)
        )
        np.testing.assert_allclose(
            orbitals.T @ orbitals,
            np.eye(orbitals.shape[1]),
            rtol=0.0,
            atol=1.0e-9,
        )
        hamiltonian = system.reference.hamiltonian(
            actual.input_effective_potential
        )
        np.testing.assert_allclose(
            np.einsum("ij,ij->j", orbitals, hamiltonian.apply(orbitals)),
            actual.eigenvalues,
            rtol=0.0,
            atol=1.0e-9,
        )
        rebuilt = (2.0 / system.grid.volume_element) * np.sum(
            orbitals * orbitals * actual.occupations[None, :], axis=1
        )
        np.testing.assert_allclose(
            rebuilt, actual.density, rtol=1.0e-11, atol=1.0e-13
        )

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

    def test_every_scf_step_reuses_the_static_sector_maps(self) -> None:
        from dataclasses import replace

        problem = parse_parsec_input(SMOKE_INPUT).problem
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))
        system = prepare_single_point(problem, backend="auto")
        solver = system.backend.symmetry_eigensolver
        multiplicities = solver.decomposition.reduction.multiplicities
        for orbits, scales in zip(
            solver._sector_orbits, solver._sector_scales, strict=True
        ):
            np.testing.assert_array_equal(
                scales, 1.0 / np.sqrt(multiplicities[orbits])
            )
            self.assertFalse(scales.flags.writeable)
        static_maps = (solver._sector_orbits, solver._sector_scales)
        packed = []
        build_density = system.orbital_density_builder

        def recording_builder(wavefunctions, occupations, volume_element):
            packed.append((wavefunctions.sector_orbits, wavefunctions.sector_scales))
            return build_density(wavefunctions, occupations, volume_element)

        system.orbital_density_builder = recording_builder
        actual = run_scf(system)

        self.assertGreater(actual.iterations, 1)
        self.assertEqual(len(packed), actual.iterations)
        for orbits, scales in packed:
            self.assertIs(orbits, static_maps[0])
            self.assertIs(scales, static_maps[1])
        # The density of the last step is that of the exported orbitals, and
        # the lazily expanded residual does not change the reported norms.
        orbitals = np.asarray(actual.wavefunctions)
        rebuilt = (2.0 / system.grid.volume_element) * np.sum(
            orbitals * orbitals * actual.occupations[None, :], axis=1
        )
        np.testing.assert_allclose(
            rebuilt, actual.density, rtol=1.0e-11, atol=1.0e-13
        )
        residual = (
            actual.output_effective_potential - actual.input_effective_potential
        )
        last = actual.history[-1]
        self.assertAlmostEqual(
            last.plain_residual,
            float(np.sqrt(system.grid.volume_element * np.dot(residual, residual))),
            11,
        )
        self.assertAlmostEqual(
            last.weighted_residual,
            float(
                np.sqrt(
                    system.grid.volume_element
                    * np.dot(actual.density, residual * residual)
                    / actual.electron_count
                )
            ),
            11,
        )

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

    def test_ionic_fields_built_on_their_thread_leave_the_scf_bitwise_unchanged(
        self,
    ) -> None:
        problem = parse_parsec_input(SMOKE_INPUT).problem
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))

        def run(switch):
            with patch.dict(
                driver_module.os.environ,
                {
                    "PARSEC_IONIC_BACKEND": "cupy",
                    "PARSEC_OVERLAP_IONIC_SETUP": switch,
                    "PARSEC_ACCELERATED_RESIDENT": "0",
                    "PARSEC_HARTREE_LINEAR_BACKEND": "cupy",
                },
            ):
                system = prepare_single_point(problem, backend="auto")
                return system, dict(system.backend_info.details), run_scf(system)

        inline_system, inline_details, expected = run("0")
        system, details, actual = run("1")
        if "CUDA" not in details["ionic_setup_implementation"]:
            self.skipTest("the native extension has no radial kernels")
        self.assertEqual(inline_details["ionic_setup"], "inline")
        self.assertEqual(
            details["ionic_setup"], "overlapped with symmetry and orbital setup"
        )
        self.assertGreater(float(details["ionic_setup_seconds"]), 0.0)
        # The sums of the thread are reported like the in-line ones.
        self.assertNotEqual(inline_details["ionic_gpu_devices"], "none")
        self.assertEqual(
            details["ionic_gpu_devices"], inline_details["ionic_gpu_devices"]
        )
        for name in ("ionic_potential", "initial_density", "core_density"):
            with self.subTest(field=name):
                self.assertEqual(
                    getattr(system, name).tobytes(),
                    getattr(inline_system, name).tobytes(),
                )
        self.assertEqual(system.ion_ion_energy, inline_system.ion_ion_energy)
        np.testing.assert_array_equal(actual.eigenvalues, expected.eigenvalues)
        np.testing.assert_array_equal(actual.density, expected.density)
        self.assertEqual(actual.energies, expected.energies)
        self.assertEqual(actual.iterations, expected.iterations)

    def test_gpu_cg_on_the_reused_sector_stencil_is_bitwise_the_repacked_one(
        self,
    ) -> None:
        from parsec_python.acceleration.Hartree import symmetry_poisson
        from parsec_python.acceleration.backends.cupy_stencil_major import (
            StencilMajorHostMetadata,
        )

        problem = parse_parsec_input(SMOKE_INPUT).problem
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))
        adapter = symmetry_poisson.SymmetryReducedPoissonSolver

        def repacking_adapter(operator, *args, **kwargs):
            # Reference route: the sector stencil converted to CSR, which
            # the CG backend packs itself.
            if isinstance(operator, StencilMajorHostMetadata):
                operator = operator.to_csr()
            return adapter(operator, *args, **kwargs)

        def prepare(reuse):
            if reuse:
                return prepare_single_point(problem, backend="auto")
            with patch.object(
                symmetry_poisson,
                "SymmetryReducedPoissonSolver",
                repacking_adapter,
            ):
                return prepare_single_point(problem, backend="auto")

        def hartree_solves(system):
            # Fixed host densities; the third solve uses the predictor.
            reducer = system.scalar_field_adapter
            solves = []
            for factor in (1.0, 1.02, 1.05):
                result = system.solve_hartree(
                    reducer.from_full(factor * system.initial_density)
                )
                solves.append(result)
            return reducer, solves

        with patch.dict(
            "os.environ", {"PARSEC_HARTREE_LINEAR_BACKEND": "cupy"}
        ):
            repacked_system, reused_system = prepare(False), prepare(True)
            repacked_details = dict(repacked_system.backend_info.details)
            reused_details = dict(reused_system.backend_info.details)
            self.assertEqual(
                repacked_details["hartree_cg_stencil_packing"], "packed from CSR"
            )
            self.assertEqual(
                reused_details["hartree_cg_stencil_packing"],
                "reused from the symmetry-sector stencil",
            )
            for name in (
                "hartree_cg_storage",
                "hartree_cg_coefficient_palette_size",
                "hartree_cg_device",
                "hartree_cg_device_bytes",
                "hartree_wedge_points",
            ):
                with self.subTest(detail=name):
                    self.assertEqual(reused_details[name], repacked_details[name])

            reducer, expected_solves = hartree_solves(repacked_system)
            reused_reducer, actual_solves = hartree_solves(reused_system)
            self.assertGreater(expected_solves[0].iterations, 0)
            for actual, expected in zip(actual_solves, expected_solves, strict=True):
                self.assertTrue(expected.converged)
                for name in (
                    "converged",
                    "iterations",
                    "matrix_vector_products",
                    "residual_norm",
                    "initial_residual_norm",
                ):
                    with self.subTest(diagnostic=name):
                        self.assertEqual(
                            getattr(actual, name), getattr(expected, name)
                        )
                np.testing.assert_array_equal(
                    reused_reducer.to_full(actual.potential),
                    reducer.to_full(expected.potential),
                )

            expected = run_scf(prepare(False))
            actual = run_scf(prepare(True))

        np.testing.assert_array_equal(
            actual.hartree_potential, expected.hartree_potential
        )
        np.testing.assert_array_equal(actual.eigenvalues, expected.eigenvalues)
        np.testing.assert_array_equal(actual.density, expected.density)
        self.assertEqual(actual.energies, expected.energies)
        for step, reference in zip(actual.history, expected.history, strict=True):
            self.assertEqual(step.hartree_residual, reference.hartree_residual)
            self.assertEqual(step.energies.hartree, reference.energies.hartree)

    def test_hartree_objects_on_another_device_leave_the_scf_bitwise_unchanged(
        self,
    ) -> None:
        from parsec_python.acceleration.backends.cupy import require_cupy

        cp, _ = require_cupy()
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 2:
            self.skipTest("requires at least two allocated GPUs")
        problem = parse_parsec_input(SMOKE_INPUT).problem
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))
        # CG arrays, boundary geometry and resident maps all on a GPU.
        hartree_on_gpu = {
            "PARSEC_HARTREE_LINEAR_BACKEND": "cupy",
            "PARSEC_HARTREE_BOUNDARY_BACKEND": "cupy",
            "PARSEC_CUPY_RESIDENT_HARTREE": "1",
            "PARSEC_CUPY_DEVICES": ",".join(map(str, range(device_count))),
        }

        def run(request):
            with patch.dict(driver_module.os.environ, hartree_on_gpu):
                if request is None:
                    driver_module.os.environ.pop("PARSEC_HARTREE_DEVICE", None)
                else:
                    driver_module.os.environ["PARSEC_HARTREE_DEVICE"] = request
                system = prepare_single_point(problem, backend="auto")
                details = dict(system.backend_info.details)
                return run_scf(system), details

        # The former placement: the CG on the first sector device.
        expected, default_details = run("off")
        self.assertEqual(default_details["hartree_cg_device"], "0")
        self.assertEqual(
            default_details["hartree_full_grid_transfer_policy"], "device-resident"
        )
        # Eight sectors in turn over the devices leave the last one lightest;
        # unset is auto.
        last = str(device_count - 1)
        for request, device in (
            (None, last), ("auto", last), (last, last), ("1", "1"), ("0", "0")
        ):
            with self.subTest(PARSEC_HARTREE_DEVICE=request):
                actual, details = run(request)
                self.assertEqual(details["hartree_cg_device"], device)
                self.assertEqual(details["hartree_boundary_device"], device)
                self.assertEqual(
                    details["hartree_full_grid_transfer_policy"], "device-resident"
                )
                for name in ("hartree_cg_device_bytes", "hartree_boundary_device_bytes"):
                    self.assertEqual(details[name], default_details[name])
                np.testing.assert_array_equal(
                    actual.hartree_potential, expected.hartree_potential
                )
                np.testing.assert_array_equal(
                    actual.eigenvalues, expected.eigenvalues
                )
                np.testing.assert_array_equal(actual.density, expected.density)
                self.assertEqual(actual.energies, expected.energies)
                for step, reference in zip(
                    actual.history, expected.history, strict=True
                ):
                    self.assertEqual(
                        step.hartree_residual, reference.hartree_residual
                    )
        with self.assertRaisesRegex(ValueError, "PARSEC_HARTREE_DEVICE"):
            run(str(device_count))

    def test_sector_stencils_from_the_grid_leave_the_scf_bitwise_unchanged(
        self,
    ) -> None:
        from parsec_python.acceleration.backends.native import _load_native

        if not hasattr(_load_native(), "reduce_sector_csr"):
            self.skipTest("the former route needs the native sector reduction")
        problem = parse_parsec_input(SMOKE_INPUT).problem
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))

        def run(route):
            with patch.dict(
                driver_module.os.environ,
                {
                    "PARSEC_SECTOR_STENCIL": route,
                    "PARSEC_NATIVE_SECTOR_ASSEMBLY": "1",
                    "PARSEC_ACCELERATED_RESIDENT": "0",
                },
            ):
                system = prepare_single_point(problem, backend="auto")
                details = dict(system.backend_info.details)
                return run_scf(system), details

        # The former route: full-grid matrix, native reduction, packing.
        expected, former = run("csr")
        self.assertEqual(former["orbital_operator_stencil_builder"], "csr")
        self.assertNotIn("finite_difference_full_grid_materialization", former)
        for route in ("direct", "numpy"):
            with self.subTest(PARSEC_SECTOR_STENCIL=route):
                actual, details = run(route)
                self.assertIn(
                    details["orbital_operator_stencil_builder"],
                    ("direct-numpy",)
                    if route == "numpy"
                    else ("direct-native", "direct-numpy"),
                )
                self.assertEqual(
                    details["finite_difference_full_grid_materialization"],
                    "skipped_by_direct_sector_stencil",
                )
                self.assertEqual(details["laplacian_nnz"], former["laplacian_nnz"])
                self.assertEqual(actual.iterations, expected.iterations)
                np.testing.assert_array_equal(
                    actual.hartree_potential, expected.hartree_potential
                )
                np.testing.assert_array_equal(
                    actual.eigenvalues, expected.eigenvalues
                )
                np.testing.assert_array_equal(actual.density, expected.density)
                self.assertEqual(actual.energies, expected.energies)

    def test_tiles_of_a_stencil_built_from_the_grid_leave_the_scf_bitwise_unchanged(
        self,
    ) -> None:
        problem = parse_parsec_input(SMOKE_INPUT).problem
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))

        def run(tile, fast_maps):
            with patch.dict(
                driver_module.os.environ,
                {
                    "PARSEC_SECTOR_STENCIL": "direct",
                    "PARSEC_CUPY_IMPLICIT_TILE": tile,
                    "PARSEC_SYMMETRY_FAST_MAPS": fast_maps,
                    "PARSEC_CUPY_MIXED_FILTER": "off",
                    "PARSEC_ACCELERATED_RESIDENT": "0",
                },
            ):
                system = prepare_single_point(problem, backend="auto")
                details = dict(system.backend_info.details)
                return run_scf(system), details

        # The slot-major stencil on the default maps.
        expected, slot_major = run("0", "1")
        self.assertTrue(
            slot_major["orbital_operator_stencil_builder"].startswith("direct")
        )
        self.assertEqual(slot_major["orbital_sector_tile_pack_workers"], "no tiles packed")
        self.assertEqual(slot_major["symmetry_fast_maps"], "on")
        for tile, fast_maps in (("16", "1"), ("16", "0"), ("0", "0")):
            with self.subTest(tile=tile, fast_maps=fast_maps):
                actual, details = run(tile, fast_maps)
                self.assertEqual(
                    details["orbital_operator_stencil_builder"],
                    slot_major["orbital_operator_stencil_builder"],
                )
                self.assertEqual(
                    details["orbital_sector_finite_difference_storage"],
                    "implicit_affine_tile_16"
                    if tile == "16"
                    else slot_major["orbital_sector_finite_difference_storage"],
                )
                self.assertEqual(
                    details["symmetry_fast_maps"], "on" if fast_maps == "1" else "off"
                )
                self.assertEqual(actual.iterations, expected.iterations)
                np.testing.assert_array_equal(
                    actual.eigenvalues, expected.eigenvalues
                )
                np.testing.assert_array_equal(actual.density, expected.density)
                self.assertEqual(actual.energies, expected.energies)

    def test_a_result_reports_the_filter_graphs_that_its_sectors_recorded(
        self,
    ) -> None:
        problem = parse_parsec_input(SMOKE_INPUT).problem
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))
        key = "orbital_sector_filter_graphs"
        shared, per_block = "one per block width and degree", "one per block of a plan"

        def run(graphs, reuse):
            with patch.dict(
                driver_module.os.environ,
                {
                    "PARSEC_CUPY_FILTER_GRAPHS": graphs,
                    "PARSEC_CUPY_FILTER_GRAPH_REUSE": reuse,
                    "PARSEC_CUPY_DISTRIBUTED_FILTER": "0",
                    "PARSEC_CUPY_MIXED_FILTER": "off",
                    "PARSEC_ACCELERATED_RESIDENT": "0",
                },
            ):
                system = prepare_single_point(problem, backend="auto")
                prepared = dict(system.backend_info.details)[key]
                result = run_scf(system)
                builds = sum(
                    operator.timing_stats.filter_graph_builds
                    for operator in system.backend.symmetry_eigensolver._operators
                )
                return result, prepared, dict(result.backend.details)[key], builds

        # Without the graph filter the switch of its graphs says nothing of
        # the run: the preparation names the route, the result what ran.
        for reuse, route in (("1", shared), ("0", per_block)):
            with self.subTest(PARSEC_CUPY_FILTER_GRAPHS="0", reuse=reuse):
                _result, prepared, ran, builds = run("0", reuse)
                self.assertEqual((prepared, ran, builds), (route, "none recorded", 0))
        expected, prepared, ran, former_builds = run("1", "0")
        self.assertEqual((prepared, ran), (per_block, per_block))
        actual, prepared, ran, builds = run("1", "1")
        self.assertEqual((prepared, ran), (shared, shared))
        self.assertGreater(former_builds, builds)
        self.assertGreater(builds, 0)
        np.testing.assert_array_equal(actual.eigenvalues, expected.eigenvalues)
        np.testing.assert_array_equal(actual.density, expected.density)
        self.assertEqual(actual.energies, expected.energies)

    def test_resident_repeats_on_the_full_grid_build_the_matrix_once(self) -> None:
        from parsec_python.acceleration.backends import native as native_backend

        problem = parse_parsec_input(SMOKE_INPUT).problem
        # No symmetry is left, so the orbitals stay on the full grid and the
        # CuPy backend reads the full-grid matrix in every calculation.
        problem = replace(
            problem,
            atoms=(replace(problem.atoms[0], position=(0.11, 0.23, 0.37)),),
            recenter_geometry=False,
            scf=replace(problem.scf, max_iterations=2),
        )

        def clear():
            with driver_module._REFERENCE_CACHE_LOCK:
                driver_module._REFERENCE_CACHE.clear()

        clear()
        self.addCleanup(clear)
        reported, results = [], []
        with (
            patch.dict(
                driver_module.os.environ,
                {
                    "PARSEC_ACCELERATED_RESIDENT": "1",
                    "PARSEC_RESIDENT_REFERENCE_CACHE_SIZE": "1",
                },
            ),
            patch.object(
                native_backend,
                "build_native_negative_laplacian",
                wraps=native_backend.build_native_negative_laplacian,
            ) as builder,
        ):
            for _ in range(3):
                system = prepare_single_point(problem, backend="auto")
                details = dict(system.backend_info.details)
                self.assertTrue(details["orbital_symmetry"].startswith("full grid"))
                reported.append(
                    details["finite_difference_full_grid_materialization"]
                )
                results.append(run_scf(system))
                del system
        builder.assert_called_once()
        self.assertEqual(
            reported,
            ["performed"] + ["reused_from_resident_reference"] * 2,
        )
        for repeat in results[1:]:
            np.testing.assert_array_equal(repeat.eigenvalues, results[0].eigenvalues)
            np.testing.assert_array_equal(repeat.density, results[0].density)
            self.assertEqual(repeat.energies, results[0].energies)

    def test_disabled_caches_skip_their_keys_and_keep_reports_writable(self) -> None:
        from parsec_python.cli import save_result_archive

        problem = parse_parsec_input(SMOKE_INPUT).problem
        directory = Path.cwd() / ".tmp" / f"hybrid-key-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            with patch.dict("os.environ", {"PARSEC_ACCELERATED_RESIDENT": "0"}):
                plain = run_scf(prepare_single_point(problem, backend="auto"))
                # A cache directory is the route that still hashes every key.
                hashed = run_scf(
                    prepare_single_point(
                        problem,
                        backend="auto",
                        symmetry_cache_directory=directory,
                    )
                )
            archive = save_result_archive(directory / "result.npz", plain)
            with np.load(archive, allow_pickle=False) as stored:
                archived = dict(
                    zip(
                        stored["backend_detail_keys"].tolist(),
                        stored["backend_detail_values"].tolist(),
                    )
                )
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()

        details = dict(plain.backend.details)
        hashed_details = dict(hashed.backend.details)
        absent = driver_module._reported_cache_key(None)
        for name in (
            "symmetry_geometry",
            "symmetry_representation",
            "orbital_operator",
        ):
            with self.subTest(cache=name):
                self.assertEqual(details[f"{name}_cache_key"], absent)
                self.assertEqual(details[f"{name}_cache_path"], "disabled")
                self.assertEqual(float(details[f"{name}_hash_seconds"]), 0.0)
                self.assertEqual(archived[f"{name}_cache_key"], absent)
                self.assertEqual(len(hashed_details[f"{name}_cache_key"]), 64)
        # Skipping the keys does not touch the calculation.  The members of a
        # degenerate level belong to different sectors, and their last bits
        # decide which of them are selected and in what order.  Sector labels
        # are therefore compared only for states separated from every other
        # selected one; the highest may share its level with a state that was
        # not selected.
        gaps = np.abs(plain.eigenvalues[:, None] - plain.eigenvalues[None, :])
        np.fill_diagonal(gaps, np.inf)
        separated = np.min(gaps, axis=1) > 1.0e-6
        separated[-1] = False
        self.assertTrue(np.any(separated))
        np.testing.assert_array_equal(
            plain.representations[separated], hashed.representations[separated]
        )
        np.testing.assert_allclose(
            plain.eigenvalues, hashed.eigenvalues, rtol=2.0e-13, atol=2.0e-13
        )
        np.testing.assert_allclose(
            plain.density, hashed.density, rtol=2.0e-12, atol=2.0e-13
        )
        self.assertAlmostEqual(plain.energies.total, hashed.energies.total, 11)


if __name__ == "__main__":
    unittest.main()
