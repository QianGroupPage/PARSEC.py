"""Focused wiring tests for the component-aware default execution policy."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
import gc
import os
from pathlib import Path
import sys
import threading
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, DEFAULT, MagicMock, patch

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


# A preparation hands itself to its reference stage: the clock of its stages
# and the threads it has to join.
_PREPARATION = {"preparation": ANY}


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

    def test_filter_plan_is_what_the_input_fixes_of_the_filter_work(self) -> None:
        from parsec_python.models import EigensolverSettings, SCFSettings

        eigensolver = EigensolverSettings(first_filter_degree=30, first_filter_cycles=4)

        def plan(electrons=3480.0, **scf):
            found = driver_module._filter_plan(
                SimpleNamespace(
                    input=SimpleNamespace(scf=SCFSettings(**scf), eigensolver=eigensolver),
                    electron_count=electrons,
                )
            )
            return found.states, found.buffer, found.applications

        # The states of the input with the buffer of the subspace, and the
        # applications of four first cycles of degree 30 and eight later
        # steps at degree ten.
        self.assertEqual(plan(number_of_states=1746, max_iterations=70), (1746, 6, 200))
        # Without States_Num: half the electrons and six more, as the SCF
        # resolves them; an odd count rounds up.
        self.assertEqual(plan(max_iterations=70), (1746, 6, 200))
        self.assertEqual(plan(electrons=5.0), (9, 6, 200))
        # No more later steps than the input allows.
        self.assertEqual(plan(number_of_states=12, max_iterations=1), (12, 6, 120))
        self.assertEqual(plan(number_of_states=12, max_iterations=3), (12, 6, 140))
        # A stand-in that carries no settings has no plan.
        for system in (
            SimpleNamespace(),
            SimpleNamespace(input=SimpleNamespace(hartree=object())),
            SimpleNamespace(input=SimpleNamespace(scf=SCFSettings())),
        ):
            self.assertIsNone(driver_module._filter_plan(system))

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
            stencil_kernel="former",
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
                **_PREPARATION,
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
                    **_PREPARATION,
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
            **_PREPARATION,
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
        # Both are handed what the input fixes of the filter work, which
        # the tiles of their sector stencils are decided by: nothing here,
        # where the stand-in of the system carries no SCF settings.
        self.assertIsNone(worker_call.kwargs["filter_plan"])
        self.assertIsNone(root_call.kwargs["filter_plan"])
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

    def test_sector_solver_is_handed_the_filter_plan_of_the_prepared_input(self) -> None:
        # What the input fixes of the filter work reaches the constructor of
        # the sector eigensolver on the root, on a sector worker and without
        # a sector context, made from the system the rank prepared.
        plan = object()
        for context in (
            None,
            SimpleNamespace(rank=0, root=0, size=2),
            SimpleNamespace(rank=1, root=0, size=2),
        ):
            with self.subTest(context=context), patch.object(
                driver_module, "_filter_plan", return_value=plan
            ) as planned:
                prepared = self.prepare(context)
                planned.assert_called_once_with(prepared.reference)
                self.assertIs(
                    prepared.sector_solver_type.call_args.kwargs["filter_plan"], plan
                )

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

    def test_stencil_kernel_detail_is_the_record_of_the_sector_eigensolver(
        self,
    ) -> None:
        # Which recurrence kernels the stencils of the sectors launch, also
        # where PARSEC_CUPY_STENCIL_KERNEL is unset.
        key = "orbital_sector_stencil_kernel"
        prepared = self.prepare(None)
        self.assertEqual(dict(prepared.system.backend_info.details)[key], "former")
        for record in (
            "pipelined: fast path for 25 slots",
            "pipelined: former kernels (the generated kernel could not be built: "
            "CompileException: nvrtc failed)",
        ):
            for mpi_context in (None, SimpleNamespace(rank=0, root=0, size=2)):
                with self.subTest(record=record, mpi_context=mpi_context):
                    prepared = self.prepare(
                        mpi_context, sector={"stencil_kernel": record}
                    )
                    self.assertEqual(
                        dict(prepared.system.backend_info.details)[key], record
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
                **_PREPARATION,
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
                            "preparation",
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
            root.problem, _hybrid_selection(), **_PREPARATION
        )
        worker = self.prepare(
            SimpleNamespace(rank=1, root=0, size=2), environment=environment
        )
        worker.prepare_reference.assert_called_once_with(
            worker.problem,
            _hybrid_selection(),
            orbital_operators_only=True,
            **_PREPARATION,
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
        [
            *deferred,
            "ionic_sector_rank",
            "ionic_report",
            "ionic_field_overlap",
            "preparation",
        ]
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
                        **_PREPARATION,
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
            **_PREPARATION,
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

    @staticmethod
    def _ion_ion_threads():
        return [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("parsec-ion-ion")
        ]

    def _fields_with_ion_ion(self, switch):
        """Build the fields on their thread; say who summed the ion-ion energy, and how.

        Returns the complete system, the overlap, the calls of the builders
        with the ion-ion sums among them, and the options that the
        completion of the reference package was given.
        """

        import parsec_python.SCF.single_point as reference_module

        sums, given = [], []
        reference_sum = reference_module.ion_ion_energy
        complete = driver_module.complete_reference_single_point

        def summed_by(who):
            def ion_ion(atoms, pseudopotentials):
                self._HostFields.calls.append(
                    ("ion-ion", threading.current_thread().name)
                )
                sums.append((who, atoms, pseudopotentials))
                return reference_sum(atoms, pseudopotentials)

            return ion_ion

        def completing(reference, **options):
            given.append(options)
            return complete(reference, **options)

        overlap = driver_module._OverlappedIonicFields()
        environment = dict(self.gpu_fields)
        if switch is not None:
            environment["PARSEC_OVERLAP_ION_ION_SETUP"] = switch
        with (
            patch.object(driver_module, "ion_ion_energy", summed_by("driver")),
            patch.object(reference_module, "ion_ion_energy", summed_by("reference")),
            patch.object(driver_module, "complete_reference_single_point", completing),
            patch.dict(driver_module.os.environ),
        ):
            driver_module.os.environ.pop("PARSEC_OVERLAP_ION_ION_SETUP", None)
            reduced, _ = self._reference_physics(
                environment, ionic_field_overlap=overlap
            )
            system = overlap.result()
        return SimpleNamespace(
            system=system,
            reduced=reduced,
            overlap=overlap,
            calls=list(self._HostFields.calls),
            sums=sums,
            given=given,
        )

    def test_ion_ion_energy_is_summed_on_a_thread_of_its_own_beside_the_fields(
        self,
    ) -> None:
        inline, _ = self._reference_physics(self.gpu_fields)
        for switch in (None, "1", " On "):
            with self.subTest(switch=switch):
                built = self._fields_with_ion_ion(switch)
                # One sum, by the driver's thread, of the atoms of the system.
                ((who, atoms, pseudopotentials),) = built.sums
                self.assertEqual(who, "driver")
                self.assertIs(atoms, built.reduced.atoms)
                self.assertIs(pseudopotentials, built.reduced.pseudopotentials)
                calls = dict(built.calls)
                self.assertTrue(calls["ion-ion"].startswith("parsec-ion-ion"), calls)
                self.assertTrue(calls["local"].startswith("parsec-ionic-setup"))
                # It starts once the projectors are built, never beside them.
                names = [name for name, _ in built.calls]
                self.assertLess(names.index("projectors"), names.index("ion-ion"))
                # The energy of the in-line stage, bit for bit, and the fields.
                self.assertEqual(
                    built.system.ion_ion_energy.hex(), inline.ion_ion_energy.hex()
                )
                self.assertGreater(built.system.ion_ion_energy, 0.0)
                for name in ("ionic_potential", "initial_density", "core_density"):
                    self.assertEqual(
                        getattr(built.system, name).tobytes(),
                        getattr(inline, name).tobytes(),
                    )
                # The completion took the value through its hook, after the
                # densities; the seconds of the stage are the thread's.
                (options,) = built.given
                self.assertEqual(
                    sorted(options),
                    [
                        "atomic_density_builder",
                        "build_atomic_tail",
                        "ion_ion_energy_builder",
                        "local_ionic_builder",
                    ],
                )
                details = dict(built.overlap.details())
                self.assertEqual(
                    details["ion_ion_setup"], "own thread beside the ionic fields"
                )
                self.assertGreater(built.system.timings.ion_ion_seconds, 0.0)
                self.assertEqual(
                    f"{built.system.timings.ion_ion_seconds:.6f}",
                    details["ion_ion_setup_seconds"],
                )
                # Both threads have ended at the join.
                self.assertEqual(self._ion_ion_threads(), [])
                self.assertEqual(self._ionic_threads(), [])

    def test_former_value_leaves_the_ion_ion_energy_to_the_thread_of_the_fields(
        self,
    ) -> None:
        inline, _ = self._reference_physics(self.gpu_fields)
        for switch in ("0", "off", "False", "no"):
            with self.subTest(switch=switch):
                built = self._fields_with_ion_ion(switch)
                # The stage of the reference package, on the ionic thread and
                # after the densities, as before the switch existed.
                ((who, _atoms, _pseudopotentials),) = built.sums
                self.assertEqual(who, "reference")
                self.assertEqual(
                    [name for name, _ in built.calls if name != "projectors"],
                    ["current device", "local", "valence", "core", "ion-ion"],
                )
                self.assertTrue(
                    dict(built.calls)["ion-ion"].startswith("parsec-ionic-setup")
                )
                # The completion is called as it was: no hook is handed to it.
                (options,) = built.given
                self.assertEqual(
                    sorted(options),
                    ["atomic_density_builder", "build_atomic_tail", "local_ionic_builder"],
                )
                self.assertEqual(
                    built.system.ion_ion_energy.hex(), inline.ion_ion_energy.hex()
                )
                details = dict(built.overlap.details())
                self.assertEqual(details["ion_ion_setup"], "last stage of the ionic thread")
                self.assertNotIn("ion_ion_setup_seconds", details)
                self.assertGreater(built.system.timings.ion_ion_seconds, 0.0)
                self.assertEqual(self._ion_ion_threads(), [])

    def test_switch_of_the_ion_ion_thread_is_on_unless_named_off(self) -> None:
        for value, expected in (
            (None, True), ("1", True), ("yes", True), ("", True),
            ("0", False), (" OFF ", False), ("false", False), ("No", False),
        ):
            with self.subTest(value=value), patch.dict(driver_module.os.environ):
                driver_module.os.environ.pop("PARSEC_OVERLAP_ION_ION_SETUP", None)
                if value is not None:
                    driver_module.os.environ["PARSEC_OVERLAP_ION_ION_SETUP"] = value
                self.assertIs(driver_module._ion_ion_overlap_requested(), expected)

    def test_failure_of_the_ion_ion_energy_is_raised_at_the_join_of_the_fields(
        self,
    ) -> None:
        import parsec_python.SCF.single_point as reference_module

        def coincident(_atoms, _pseudopotentials):
            raise ValueError("two nuclei occupy the same position")

        for switch in ("1", "0"):
            with self.subTest(switch=switch):
                overlap = driver_module._OverlappedIonicFields()
                with (
                    patch.object(driver_module, "ion_ion_energy", coincident),
                    patch.object(reference_module, "ion_ion_energy", coincident),
                ):
                    self._reference_physics(
                        {**self.gpu_fields, "PARSEC_OVERLAP_ION_ION_SETUP": switch},
                        ionic_field_overlap=overlap,
                    )
                    # Where the stage stands: after the three sums, at the join.
                    with self.assertRaisesRegex(ValueError, "two nuclei occupy"):
                        overlap.result()
                self.assertEqual(
                    [name for name, _ in self._HostFields.calls if name != "projectors"],
                    ["current device", "local", "valence", "core"],
                )
                self.assertEqual(self._ion_ion_threads(), [])
                self.assertEqual(self._ionic_threads(), [])

    def test_failure_of_the_fields_comes_first_and_joins_the_ion_ion_thread(
        self,
    ) -> None:
        release = threading.Event()
        ended = []

        def slow(_atoms, _pseudopotentials):
            release.wait(timeout=0.3)
            ended.append(threading.current_thread().name)
            raise ValueError("two nuclei occupy the same position")

        def fail(_reference, **_builders):
            raise RuntimeError("ionic kernel failed")

        overlap = driver_module._OverlappedIonicFields()
        with (
            patch.object(driver_module, "ion_ion_energy", slow),
            patch.object(driver_module, "complete_reference_single_point", fail),
        ):
            self._reference_physics(self.gpu_fields, ionic_field_overlap=overlap)
            # The sums fail before the stage of the ion-ion energy: theirs is
            # the error, and the second thread has ended when it is raised.
            with self.assertRaisesRegex(RuntimeError, "ionic kernel failed"):
                overlap.result()
        self.assertEqual(len(ended), 1)
        self.assertEqual(self._ion_ion_threads(), [])
        # A preparation that never reaches the join closes both.
        overlap = driver_module._OverlappedIonicFields()
        del ended[:]
        with patch.object(driver_module, "ion_ion_energy", slow):
            self._reference_physics(self.gpu_fields, ionic_field_overlap=overlap)
            overlap.close()
        self.assertEqual(len(ended), 1)
        self.assertEqual(self._ion_ion_threads(), [])
        self.assertEqual(self._ionic_threads(), [])

    def test_ion_ion_thread_sums_in_place_for_other_atoms_than_its_own(self) -> None:
        calls = []

        def energy(atoms, _pseudopotentials):
            calls.append((atoms, threading.current_thread().name))
            return float(len(atoms))

        atoms, pseudopotentials = (object(), object()), {}
        with patch.object(driver_module, "ion_ion_energy", energy):
            beside = driver_module._IonIonEnergyBeside(atoms, pseudopotentials)
            self.assertEqual(beside.energy(atoms, pseudopotentials), 2.0)
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0][1].startswith("parsec-ion-ion"))
            # Other atoms, or other pseudopotentials: not the thread's value.
            other = (object(),)
            self.assertEqual(beside.energy(other, pseudopotentials), 1.0)
            self.assertEqual(calls[1], (other, threading.current_thread().name))
            self.assertEqual(beside.energy(atoms, {}), 2.0)
            self.assertEqual(len(calls), 3)
            beside.close()
        self.assertGreaterEqual(beside.seconds, 0.0)
        self.assertEqual(self._ion_ion_threads(), [])

    class _OrbitFields(_HostFields):
        """Host builders that can be told to sum on the orbits of a reduction."""

        told: list = []

        def sum_on_orbits(self, grid, reduction, coordinates=None):
            type(self).told.append(
                (grid, reduction, coordinates, threading.current_thread().name)
            )

    def _symmetric_reference(self, positions=((0.9, -0.4, 0.3), (-0.9, 0.4, 0.3))):
        """A grid, two atoms that a twofold axis exchanges, and their reduction."""

        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Symmetry import SignedPermutationReduction

        problem = parse_parsec_input(SMOKE_INPUT).problem
        grid = build_cluster_grid(problem.grid)
        atoms = tuple(
            replace(problem.atoms[0], position=np.array(position))
            for position in positions
        )
        reference = SimpleNamespace(grid=grid, atoms=atoms)
        return reference, SignedPermutationReduction.detect(grid, atoms)

    def test_thread_sums_on_the_wedge_where_the_images_are_exact(self) -> None:
        reference, reduction = self._symmetric_reference()
        self.assertEqual(reduction.group_order, 2)
        builders = self._OrbitFields()
        type(builders).told = []
        preparation = driver_module._Preparation()
        preparation.symmetry_reduction.publish(reduction)
        with patch.dict(driver_module.os.environ):
            driver_module.os.environ.pop("PARSEC_IONIC_SUM_GRID", None)
            text = driver_module._choose_ionic_sum_grid(builders, reference, preparation)
        self.assertEqual(
            text,
            f"symmetry wedge ({reduction.wedge_size} of {reference.grid.size} points)",
        )
        ((grid, told, coordinates, _thread),) = builders.told
        self.assertIs(grid, reference.grid)
        self.assertIs(told, reduction)
        # The coordinates of the representative rows, gathered once for the
        # test of the grid and for the sums.
        self.assertTrue(coordinates.flags.c_contiguous)
        np.testing.assert_array_equal(
            coordinates, reference.grid.coordinates[reduction.representative_rows]
        )
        self.assertEqual(
            [record["stage"] for record in preparation.stages.records()],
            ["ionic_symmetry_wait", "ionic_sum_grid_test"],
        )

    def test_thread_sums_every_point_wherever_the_wedge_is_not_exact(self) -> None:
        reference, reduction = self._symmetric_reference()
        # Within the tolerance of the detector, not exact in every bit.
        nearly, accepted = self._symmetric_reference(
            ((0.9, -0.4, 0.3), (-0.9, 0.4, 0.3 + 2.0e-9))
        )
        self.assertEqual(accepted.group_order, 2)
        alone, identity = self._symmetric_reference(
            ((0.9, -0.4, 0.3), (0.2, 0.1, -0.7))
        )
        self.assertEqual(identity.group_order, 1)
        for label, system, published, builders, switch, words in (
            ("switch", reference, reduction, self._OrbitFields(), "full",
             "full grid (PARSEC_IONIC_SUM_GRID=full)"),
            ("builders", reference, reduction, self._HostFields(), None,
             "full grid (these builders sum every point)"),
            ("no reduction", reference, None, self._OrbitFields(), None,
             "full grid (no symmetry reduction)"),
            ("identity", alone, identity, self._OrbitFields(), None,
             "full grid (no symmetry operation beside the identity)"),
            ("atoms", nearly, accepted, self._OrbitFields(), None,
             "full grid (an image of an atom is not an atom of its species in every bit)"),
            ("stand-in", reference, SimpleNamespace(group_order=4), self._OrbitFields(), None,
             "full grid (the operations of the symmetry reduction are not known)"),
        ):
            with self.subTest(case=label), patch.dict(driver_module.os.environ):
                driver_module.os.environ.pop("PARSEC_IONIC_SUM_GRID", None)
                if switch is not None:
                    driver_module.os.environ["PARSEC_IONIC_SUM_GRID"] = switch
                self._OrbitFields.told = []
                preparation = driver_module._Preparation()
                preparation.symmetry_reduction.publish(published)
                self.assertEqual(
                    driver_module._choose_ionic_sum_grid(builders, system, preparation),
                    words,
                )
                self.assertEqual(self._OrbitFields.told, [])
        # The former value does not wait for a reduction that nobody has
        # published.  The choice is made on a thread, so that one that waits
        # fails this test and does not stop the tests after it.
        unpublished = driver_module._Preparation()
        texts = []
        with patch.dict(driver_module.os.environ, {"PARSEC_IONIC_SUM_GRID": "full"}):
            thread = threading.Thread(
                target=lambda: texts.append(
                    driver_module._choose_ionic_sum_grid(
                        self._OrbitFields(), reference, unpublished
                    )
                )
            )
            thread.start()
            thread.join(timeout=10.0)
            waited = thread.is_alive()
            # A thread that waits is told that there is no reduction.
            unpublished.close()
            thread.join(timeout=10.0)
        self.assertFalse(waited, "the former value waited for a reduction")
        self.assertEqual(texts, ["full grid (PARSEC_IONIC_SUM_GRID=full)"])
        # An unknown value is refused by its name.
        with (
            patch.dict(driver_module.os.environ, {"PARSEC_IONIC_SUM_GRID": "wedge"}),
            self.assertRaisesRegex(ValueError, "PARSEC_IONIC_SUM_GRID must be auto or full"),
        ):
            driver_module._choose_ionic_sum_grid(
                self._OrbitFields(), reference, driver_module._Preparation()
            )

    def test_test_of_the_images_that_fails_leaves_the_sums_on_every_point(self) -> None:
        # The wedge is an economy.  A test that cannot be made, or builders
        # that refuse the orbits, do not fail the preparation at the join
        # of the fields: every point is summed, and the report says why.
        reference, reduction = self._symmetric_reference()
        taken_back = []

        class Refusing(self._OrbitFields):
            def sum_on_orbits(fields, grid, reduction, coordinates=None):
                raise ValueError("the symmetry reduction does not match the grid")

            def sum_on_full_grid(fields):
                taken_back.append(threading.current_thread().name)

        def no_memory(*_arguments):
            raise MemoryError("Unable to allocate " + "the coordinates of the wedge " * 8)

        exact_images = "parsec_python.acceleration.Symmetry.exact_images."
        failed = "full grid (the test of the images failed: "
        for label, builders, test, words in (
            ("builders", Refusing(), None,
             failed + "ValueError: the symmetry reduction does not match the grid)"),
            ("test", self._OrbitFields(), no_memory,
             failed + "MemoryError: Unable to allocate the coordinates of the wedge the"),
        ):
            with self.subTest(case=label), patch.dict(driver_module.os.environ):
                driver_module.os.environ.pop("PARSEC_IONIC_SUM_GRID", None)
                self._OrbitFields.told = []
                del taken_back[:]
                preparation = driver_module._Preparation()
                preparation.symmetry_reduction.publish(reduction)
                with (
                    nullcontext()
                    if test is None
                    else patch(exact_images + "orbit_sum_points", test)
                ):
                    text = driver_module._choose_ionic_sum_grid(
                        builders, reference, preparation
                    )
                self.assertTrue(text.startswith(words), text)
                self.assertTrue(text.endswith(")"))
                # An entry of the backend report: no longer than the others.
                self.assertLess(len(text), 160)
                self.assertEqual(self._OrbitFields.told, [])
                # Builders that can take the orbits back were asked to.
                self.assertEqual(
                    taken_back,
                    [threading.current_thread().name] if label == "builders" else [],
                )
                self.assertEqual(
                    [record["stage"] for record in preparation.stages.records()],
                    ["ionic_symmetry_wait", "ionic_sum_grid_test"],
                )

    def test_unknown_sum_grid_is_refused_by_every_preparation(self) -> None:
        # Also by one that builds its ionic fields in line and never reads it.
        for environment in (
            {},
            self.gpu_fields,
            {**self.gpu_fields, "PARSEC_OVERLAP_IONIC_SETUP": "0"},
        ):
            with (
                self.subTest(environment=environment),
                self.assertRaisesRegex(
                    ValueError, "PARSEC_IONIC_SUM_GRID must be auto or full"
                ),
            ):
                self.prepare(
                    None, environment={**environment, "PARSEC_IONIC_SUM_GRID": "wedge"}
                )
        self.assertEqual(self._ionic_threads(), [])

    def test_thread_waits_for_the_reduction_and_is_released_by_a_preparation_that_leaves(
        self,
    ) -> None:
        reference, reduction = self._symmetric_reference()
        for published in (reduction, None, "left"):
            with self.subTest(published=published), patch.dict(driver_module.os.environ):
                driver_module.os.environ.pop("PARSEC_IONIC_SUM_GRID", None)
                self._OrbitFields.told = []
                preparation = driver_module._Preparation()
                texts = []
                thread = threading.Thread(
                    target=lambda: texts.append(
                        driver_module._choose_ionic_sum_grid(
                            self._OrbitFields(), reference, preparation
                        )
                    )
                )
                thread.start()
                thread.join(timeout=0.2)
                # Nothing is known yet: the sums have not been chosen.
                self.assertTrue(thread.is_alive())
                self.assertEqual(texts, [])
                if published == "left":
                    # A preparation on its way out says that there is none.
                    preparation.close()
                else:
                    preparation.symmetry_reduction.publish(published)
                thread.join(timeout=10.0)
                self.assertFalse(thread.is_alive())
                self.assertEqual(
                    texts[0].startswith("symmetry wedge"), published is reduction
                )
                self.assertEqual(len(self._OrbitFields.told), int(published is reduction))

    def test_detection_in_line_publishes_the_reduction_that_the_thread_waits_for(
        self,
    ) -> None:
        # The maps in line (the former value of their switch, or a reference
        # stage that starts no thread for them) and the wedge allowed: the
        # thread of the fields waits for the reduction that the calling
        # thread detects, and the calling thread waits for the fields at
        # their join.  Without the publication neither would go on.
        points = np.zeros((8, 3))
        for environment in ({"PARSEC_OVERLAP_SYMMETRY_SETUP": "0"}, {}):
            with self.subTest(environment=environment):
                self._OrbitFields.told = []
                builders = self._OrbitFields()
                handed, prepared, raised = [], [], []
                resolve = driver_module._resolve_and_prepare_reference

                def resolving(problem, backend, **options):
                    selection, reference, timing = resolve(problem, backend, **options)
                    preparation = options["preparation"]
                    handed.append(preparation)
                    options["ionic_field_overlap"].start(
                        reference,
                        stages=preparation.stages,
                        before_sums=lambda: driver_module._choose_ionic_sum_grid(
                            builders, reference, preparation
                        ),
                    )
                    return selection, reference, timing

                def prepare():
                    try:
                        with (
                            patch.object(
                                driver_module, "_resolve_and_prepare_reference", resolving
                            ),
                            patch.object(
                                driver_module,
                                "complete_reference_single_point",
                                lambda reference, **_builders: SimpleNamespace(
                                    **vars(reference)
                                ),
                            ),
                            # The stand-in reduction of ``prepare`` has no
                            # operations to test: its images count as exact.
                            patch(
                                "parsec_python.acceleration.Symmetry.exact_images."
                                "orbit_sum_points",
                                lambda _grid, _atoms, _reduction: (points, None),
                            ),
                            patch.dict(driver_module.os.environ),
                        ):
                            for name in (
                                "PARSEC_IONIC_SUM_GRID", "PARSEC_OVERLAP_SYMMETRY_SETUP"
                            ):
                                driver_module.os.environ.pop(name, None)
                            prepared.append(
                                self.prepare(
                                    None, environment={**self.gpu_fields, **environment}
                                )
                            )
                    except BaseException as error:
                        raised.append(error)

                # On a thread with a time limit: a preparation that waits
                # fails this test and does not stop the tests after it.
                thread = threading.Thread(target=prepare)
                thread.start()
                thread.join(timeout=30.0)
                waited = thread.is_alive()
                if waited:
                    # Release the fields, and with them the preparation.
                    handed[0].symmetry_reduction.publish(None)
                    thread.join(timeout=30.0)
                self.assertFalse(
                    waited, "the thread of the fields was not told the reduction"
                )
                self.assertEqual(raised, [])
                (prepared,) = prepared
                details = dict(prepared.system.backend_info.details)
                self.assertEqual(details["symmetry_setup"], "inline")
                self.assertEqual(details["ionic_sum_grid"], "symmetry wedge (8 of 32 points)")
                # The reduction that the calling thread detected, no other.
                ((grid, told, coordinates, told_on),) = self._OrbitFields.told
                self.assertIs(grid, prepared.reference.grid)
                self.assertIs(told, prepared.reduction)
                self.assertIs(coordinates, points)
                self.assertTrue(told_on.startswith("parsec-ionic-setup"))
                threads = {
                    record["stage"]: record["thread"]
                    for record in prepared.system.preparation_stages
                }
                self.assertTrue(
                    threads["ionic_symmetry_wait"].startswith("parsec-ionic-setup")
                )
                self.assertEqual(self._ionic_threads(), [])

    def test_reference_stage_lets_the_thread_choose_the_grid_of_its_sums(self) -> None:
        from parsec_python.acceleration.Symmetry import SignedPermutationReduction

        for switch, wedge in ((None, True), ("auto", True), ("full", False)):
            with self.subTest(switch=switch):
                self._OrbitFields.told = []
                overlap = driver_module._OverlappedIonicFields()
                preparation = driver_module._Preparation()
                environment = dict(self.gpu_fields)
                if switch is not None:
                    environment["PARSEC_IONIC_SUM_GRID"] = switch
                with (
                    patch.object(self, "_HostFields", self._OrbitFields),
                    patch.dict(driver_module.os.environ),
                ):
                    driver_module.os.environ.pop("PARSEC_IONIC_SUM_GRID", None)
                    reduced, _ = self._reference_physics(
                        environment,
                        ionic_field_overlap=overlap,
                        preparation=preparation,
                    )
                    # The thread that prepares detects the symmetry of the
                    # system and publishes it; the ionic thread goes on.
                    reduction = SignedPermutationReduction.detect(
                        reduced.grid, reduced.atoms
                    )
                    preparation.symmetry_reduction.publish(reduction)
                    complete = overlap.result()
                details = dict(overlap.details())
                # The two atoms of this helper have no operation in common.
                self.assertEqual(reduction.group_order, 1)
                self.assertEqual(
                    details["ionic_sum_grid"],
                    "full grid (no symmetry operation beside the identity)"
                    if wedge
                    else "full grid (PARSEC_IONIC_SUM_GRID=full)",
                )
                self.assertEqual(self._OrbitFields.told, [])
                self.assertEqual(complete.ionic_potential.shape, (reduced.grid.size,))
        # Started without a preparation the thread is told nothing, sums
        # every point and reports no grid.
        overlap = driver_module._OverlappedIonicFields()
        self._reference_physics(self.gpu_fields, ionic_field_overlap=overlap)
        overlap.result()
        self.assertNotIn("ionic_sum_grid", dict(overlap.details()))

    def test_wedge_is_for_the_sums_of_the_thread_and_for_no_later_ones(self) -> None:
        # The orbits were tested for the atoms of the system.  The builders
        # stay with the preparation: once the fields of the thread are
        # built, or have failed, they sum every point again, whatever atoms
        # a later caller gives them.
        order = []
        failing = []

        class Fields(self._OrbitFields):
            def sum_on_orbits(fields, grid, reduction, coordinates=None):
                order.append("orbits")

            def sum_on_full_grid(fields):
                order.append("every point")

            def build_local_ionic_potential(fields, *arguments):
                order.append("local")
                return super().build_local_ionic_potential(*arguments)

            def superpose_atomic_density(fields, *arguments, core=False):
                order.append("core" if core else "valence")
                if core and failing:
                    raise RuntimeError("ionic kernel failed")
                return super().superpose_atomic_density(*arguments, core=core)

        points = np.zeros((2, 3))
        for switch, fails, expected in (
            (None, False, ["orbits", "local", "valence", "core", "every point"]),
            (None, True, ["orbits", "local", "valence", "core", "every point"]),
            ("full", False, ["local", "valence", "core", "every point"]),
        ):
            with self.subTest(switch=switch, fails=fails):
                del order[:]
                failing[:] = [True] if fails else []
                overlap = driver_module._OverlappedIonicFields()
                preparation = driver_module._Preparation()
                environment = dict(self.gpu_fields)
                if switch is not None:
                    environment["PARSEC_IONIC_SUM_GRID"] = switch
                with (
                    patch.object(self, "_HostFields", Fields),
                    # The stand-in reduction has no operations to test: its
                    # images count as exact.
                    patch(
                        "parsec_python.acceleration.Symmetry.exact_images."
                        "orbit_sum_points",
                        lambda _grid, _atoms, _reduction: (points, None),
                    ),
                    patch.dict(driver_module.os.environ),
                ):
                    driver_module.os.environ.pop("PARSEC_IONIC_SUM_GRID", None)
                    reduced, _ = self._reference_physics(
                        environment,
                        ionic_field_overlap=overlap,
                        preparation=preparation,
                    )
                    preparation.symmetry_reduction.publish(
                        SimpleNamespace(
                            group_order=2, wedge_size=2, full_size=reduced.grid.size
                        )
                    )
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "ionic kernel failed"):
                            overlap.result()
                    else:
                        overlap.result()
                self.assertEqual(order, expected)
                self.assertEqual(
                    dict(overlap.details()).get("ionic_sum_grid", "").startswith(
                        "symmetry wedge"
                    ),
                    switch is None,
                )
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


class SymmetryMapsOverlapTests(unittest.TestCase):
    """PARSEC_OVERLAP_SYMMETRY_SETUP: the symmetry maps beside the projectors."""

    prepare = MPIRolePreparationTests.prepare
    _HostFields = IonicFieldOverlapTests._HostFields
    gpu_fields = IonicFieldOverlapTests.gpu_fields
    overlapped = "overlapped with the projectors"

    def tearDown(self) -> None:
        with driver_module._REFERENCE_CACHE_LOCK:
            driver_module._REFERENCE_CACHE.clear()

    @staticmethod
    def _threads():
        return [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("parsec-symmetry-setup")
        ]

    @staticmethod
    def _problem(atoms=((0.9, -0.4, 0.3), (-0.9, 0.4, 0.3)), shift=None):
        """The smoke input with two atoms that a twofold axis along z exchanges.

        ``shift`` replaces the shift of its grid: with none, grid points lie
        on the symmetry planes and their orbits are smaller.
        """

        problem = parse_parsec_input(SMOKE_INPUT).problem
        if shift is not None:
            problem = replace(problem, grid=replace(problem.grid, shift=shift))
        return replace(
            problem,
            atoms=[
                replace(problem.atoms[0], position=np.array(position))
                for position in atoms
            ],
            recenter_geometry=False,
        )

    @staticmethod
    def _preparation(switch_on=True):
        preparation = driver_module._Preparation()
        if switch_on:
            preparation.symmetry_maps = driver_module._OverlappedSymmetryMaps(
                cache_directory=None,
                reduction_known=preparation.symmetry_reduction,
                stages=preparation.stages,
            )
            preparation.join_on_exit(preparation.symmetry_maps.close)
        return preparation

    def _reference_stage(
        self, preparation, environment=None, problem=None, selection=None, **options
    ):
        """Run the reference stage of a preparation on the host.

        The native builders are stood in for by those of the reference
        package, as in ``IonicFieldOverlapTests``.  The projectors return
        only once the detection of a symmetry thread has begun, where the
        stage has one: the two run beside each other or the test stops.
        Returns the reference system and what ran, with its thread, in the
        order of the starts.
        """

        from parsec_python.Laplacian import build_negative_laplacian
        import parsec_python.acceleration.Symmetry as symmetry

        ran = []
        detecting = threading.Event()
        detect, decompose = (
            symmetry.load_or_detect_reflection_reduction,
            symmetry.load_or_build_reflection_decomposition,
        )
        maps = preparation.symmetry_maps

        def detection(*arguments, **more):
            ran.append(("reduction", threading.current_thread().name))
            detecting.set()
            return detect(*arguments, **more)

        def decomposition(*arguments, **more):
            ran.append(("decomposition", threading.current_thread().name))
            return decompose(*arguments, **more)

        class Fields(self._HostFields):
            def build_nonlocal_projectors(fields, *arguments):
                ran.append(("projectors", threading.current_thread().name))
                if maps is not None and maps.started:
                    self.assertTrue(detecting.wait(timeout=10.0))
                return super().build_nonlocal_projectors(*arguments)

        native = "parsec_python.acceleration.backends.native."
        with (
            patch(
                native + "_load_native",
                return_value=SimpleNamespace(RadialGridEvaluator=object),
            ),
            patch(native + "build_native_negative_laplacian", build_negative_laplacian),
            patch("parsec_python.acceleration.V_ion.NativeIonicBuilders", Fields),
            patch("parsec_python.acceleration.V_ion.cupy_ionic.CupyIonicBuilders", Fields),
            patch.object(symmetry, "load_or_detect_reflection_reduction", detection),
            patch.object(symmetry, "load_or_build_reflection_decomposition", decomposition),
            patch.dict(
                driver_module.os.environ,
                {"PARSEC_ACCELERATED_RESIDENT": "0", **(environment or {})},
            ),
        ):
            if "PARSEC_ORBITAL_STABILIZER_SYMMETRY" not in (environment or {}):
                driver_module.os.environ.pop("PARSEC_ORBITAL_STABILIZER_SYMMETRY", None)
            reference = driver_module._prepare_reference_physics(
                self._problem() if problem is None else problem,
                _hybrid_selection() if selection is None else selection,
                preparation=preparation,
                **options,
            )
            # The thread, if any, ends inside the doubles it was started in.
            if maps is not None:
                maps.close()
        return reference, ran

    def test_switch_is_on_unless_named_off(self) -> None:
        for value, expected in (
            (None, True), ("1", True), ("", True), ("on", True),
            ("0", False), (" Off ", False), ("false", False), ("NO", False),
        ):
            with self.subTest(value=value), patch.dict(driver_module.os.environ):
                driver_module.os.environ.pop("PARSEC_OVERLAP_SYMMETRY_SETUP", None)
                if value is not None:
                    driver_module.os.environ["PARSEC_OVERLAP_SYMMETRY_SETUP"] = value
                self.assertIs(driver_module._symmetry_overlap_requested(), expected)

    def test_maps_of_a_rank_without_ionic_fields_are_built_beside_its_projectors(
        self,
    ) -> None:
        from parsec_python.acceleration.Symmetry import (
            load_or_build_reflection_decomposition,
            load_or_detect_reflection_reduction,
        )

        for role, options in (
            ("sector worker", {"orbital_operators_only": True}),
            ("fields on their thread", {"ionic_field_overlap": True}),
        ):
            with self.subTest(role=role):
                preparation = self._preparation()
                maps = preparation.symmetry_maps
                self.assertFalse(maps.started)
                self.assertEqual(maps.details(), (("symmetry_setup", "inline"),))
                if "ionic_field_overlap" in options:
                    overlap = driver_module._OverlappedIonicFields()
                    preparation.join_on_exit(overlap.close)
                    options = {"ionic_field_overlap": overlap}
                reference, ran = self._reference_stage(
                    preparation,
                    self.gpu_fields,
                    defer_native_laplacian=True,
                    **options,
                )
                preparation.close()
                self.assertTrue(maps.started)
                threads = dict(ran)
                self.assertEqual(threads["projectors"], "MainThread")
                for stage in ("reduction", "decomposition"):
                    self.assertTrue(
                        threads[stage].startswith("parsec-symmetry-setup"), threads
                    )
                # Started on the grid and the atoms of the reference system.
                self.assertTrue(maps.serves(reference.grid, reference.atoms))
                self.assertFalse(maps.serves(replace(reference.grid), reference.atoms))
                moved = (reference.atoms[1], reference.atoms[0])
                self.assertFalse(maps.serves(reference.grid, moved))
                # The maps of the functions the calling thread calls in line.
                expected, expected_info = load_or_detect_reflection_reduction(
                    reference.grid, reference.atoms, cache_directory=None
                )
                candidate, info = maps.reduction()
                self.assertIs(type(candidate), type(expected))
                self.assertEqual(candidate.group_order, 2)
                for name in (
                    "signs", "representative_rows", "full_to_wedge", "multiplicities"
                ):
                    left, right = getattr(candidate, name), getattr(expected, name)
                    self.assertEqual(left.dtype, right.dtype)
                    self.assertEqual(left.tobytes(), right.tobytes())
                self.assertEqual((info.status, info.key), (expected_info.status, expected_info.key))
                wanted, _ = load_or_build_reflection_decomposition(
                    reference.grid, expected, reduction_key=expected_info.key,
                    cache_directory=None,
                )
                decomposition, _ = maps.decomposition(candidate)
                self.assertIs(decomposition.reduction, candidate)
                self.assertEqual(decomposition.sector_sizes, wanted.sector_sizes)
                for sector in range(wanted.representation_count):
                    np.testing.assert_array_equal(
                        decomposition.sector_orbit_indices(sector),
                        wanted.sector_orbit_indices(sector),
                    )
                # Of another reduction it has none to give.
                self.assertIsNone(maps.decomposition(expected))
                # The reduction was published when it was detected.
                self.assertIs(preparation.symmetry_reduction.wait(), candidate)
                details = dict(maps.details())
                self.assertEqual(details["symmetry_setup"], self.overlapped)
                self.assertGreater(float(details["symmetry_setup_seconds"]), 0.0)
                self.assertGreaterEqual(float(details["symmetry_setup_wait_seconds"]), 0.0)
                stages = {
                    record["stage"]: record["thread"]
                    for record in preparation.stages.records()
                }
                self.assertEqual(stages["grid"], "MainThread")
                self.assertEqual(stages["projectors"], "MainThread")
                for stage in ("symmetry_reduction", "symmetry_decomposition"):
                    self.assertTrue(stages[stage].startswith("parsec-symmetry-setup"))
                for stage in ("symmetry_reduction_wait", "symmetry_decomposition_wait"):
                    self.assertEqual(stages[stage], "MainThread")
                self.assertEqual(self._threads(), [])

    def test_thread_decomposes_only_what_the_calling_thread_would(self) -> None:
        # No operation beside the identity; a backend that solves on the full
        # grid; a stabilizer policy that keeps the orbitals off the wedge of
        # a group that fixes grid points.
        on_axis, unshifted = ((0.0, 0.0, 0.3), (0.0, 0.0, -0.9)), (0.0, 0.0, 0.0)
        scipy = BackendSelection(
            requested="native", selected="native",
            finite_difference_builder="native", hartree_backend="native",
        )
        for label, problem, selection, environment, order in (
            ("identity", self._problem(((0.9, -0.4, 0.3), (0.2, 0.1, -0.7))), None, {}, 1),
            ("full-grid backend", None, scipy, {}, 2),
            ("stabilizer off", self._problem(on_axis, unshifted), None,
             {"PARSEC_ORBITAL_STABILIZER_SYMMETRY": "off"}, 4),
            ("stabilizer unknown", self._problem(on_axis, unshifted), None,
             {"PARSEC_ORBITAL_STABILIZER_SYMMETRY": "sometimes"}, 4),
        ):
            with self.subTest(case=label):
                preparation = self._preparation()
                maps = preparation.symmetry_maps
                reference, ran = self._reference_stage(
                    preparation, environment, problem=problem, selection=selection,
                    defer_native_laplacian=True, orbital_operators_only=True,
                )
                self.assertEqual(
                    [name for name, _ in ran if name != "projectors"], ["reduction"]
                )
                candidate, _ = maps.reduction()
                self.assertEqual(candidate.group_order, order)
                self.assertIsNone(maps.decomposition(candidate))
                self.assertIs(preparation.symmetry_reduction.wait(), candidate)
        # The same atoms with the policy at its default are decomposed.
        preparation = self._preparation()
        _, ran = self._reference_stage(
            preparation, problem=self._problem(on_axis, unshifted),
            defer_native_laplacian=True, orbital_operators_only=True,
        )
        self.assertIn("decomposition", dict(ran))
        self.assertEqual(self._threads(), [])

    def test_maps_stay_in_line_where_the_reference_stage_builds_more_than_projectors(
        self,
    ) -> None:
        resident = {
            "PARSEC_ACCELERATED_RESIDENT": "1",
            "PARSEC_RESIDENT_REFERENCE_CACHE_SIZE": "1",
        }
        reference_builders = BackendSelection(
            requested="cupy", selected="cupy",
            finite_difference_builder="reference", hartree_backend="cupy",
        )
        overlap = driver_module._OverlappedIonicFields
        for label, environment, selection, options in (
            # The ionic fields in line: C++ builders, or the GPU sums.
            ("host fields", {"PARSEC_IONIC_BACKEND": "native"}, None, {}),
            ("fields in line", self.gpu_fields, None, {}),
            ("resident", {**self.gpu_fields, **resident}, None, {"ionic_field_overlap": overlap}),
            # The full-grid matrix before the projectors.
            ("matrix", self.gpu_fields, None, {"orbital_operators_only": True, "defer": False}),
            ("reference builders", {}, reference_builders, {"orbital_operators_only": True}),
        ):
            with self.subTest(case=label):
                preparation = self._preparation()
                options = dict(options)
                if "ionic_field_overlap" in options:
                    options["ionic_field_overlap"] = overlap()
                reference, ran = self._reference_stage(
                    preparation, environment, selection=selection,
                    defer_native_laplacian=options.pop("defer", True), **options,
                )
                self.assertFalse(preparation.symmetry_maps.started)
                self.assertEqual(
                    preparation.symmetry_maps.details(), (("symmetry_setup", "inline"),)
                )
                self.assertFalse(
                    preparation.symmetry_maps.serves(reference.grid, reference.atoms)
                )
                self.assertNotIn("reduction", dict(ran))
                # The grid is a stage of the clock all the same.
                self.assertIn(
                    "grid", [record["stage"] for record in preparation.stages.records()]
                )
                with driver_module._REFERENCE_CACHE_LOCK:
                    driver_module._REFERENCE_CACHE.clear()
        # Without the thread of a preparation the builders are called as
        # they were, and nobody is left waiting for a reduction.
        preparation = self._preparation(switch_on=False)
        reference, ran = self._reference_stage(
            preparation, self.gpu_fields, defer_native_laplacian=True,
            orbital_operators_only=True,
        )
        self.assertEqual(ran, [("projectors", "MainThread")])
        preparation.close()
        self.assertIsNone(preparation.symmetry_reduction.wait())
        self.assertEqual(self._threads(), [])

    def test_maps_stay_in_line_beside_the_probe_of_the_auto_backend(self) -> None:
        # Under ``auto`` without GPU ionic fields the probe of the CUDA
        # runtime runs beside the reference preparation, which does not
        # know the backend yet: it is handed no thread for the maps, even
        # as a sector worker with the sector stencils from the grid.  Where
        # the backend is resolved first, the thread is handed on.
        handed = []

        def reference_stage(_problem, _selection, **options):
            handed.append(
                (options["preparation"].symmetry_maps, sorted(options))
            )
            return object()

        worker = {"orbital_operators_only": True, "defer_native_laplacian": True}
        for label, backend, environment, in_line in (
            ("probe beside", "auto", {"PARSEC_IONIC_BACKEND": "native"}, True),
            ("GPU ionic fields", "auto", {"PARSEC_IONIC_BACKEND": "cupy"}, False),
            (
                "probe first",
                "auto",
                {"PARSEC_IONIC_BACKEND": "native", "PARSEC_OVERLAP_CUDA_INITIALIZATION": "0"},
                False,
            ),
            ("named backend", "cupy", {"PARSEC_IONIC_BACKEND": "native"}, False),
        ):
            with (
                self.subTest(case=label),
                patch.object(
                    driver_module, "resolve_backend", return_value=_hybrid_selection()
                ),
                patch.object(
                    driver_module, "_prepare_reference_physics", side_effect=reference_stage
                ),
                patch(
                    "parsec_python.acceleration.backends.selection._native_status",
                    return_value=(True, None),
                ),
                # The deferred Laplacian of a worker, whichever interpreter
                # runs the test.
                patch.object(driver_module, "_cupy_installed", return_value=True),
                patch.dict(
                    driver_module.os.environ,
                    {"PARSEC_OVERLAP_CUDA_INITIALIZATION": "1", **environment},
                ),
            ):
                preparation = self._preparation()
                maps = preparation.symmetry_maps
                _, _, timing = driver_module._resolve_and_prepare_reference(
                    object(), backend, preparation=preparation, **worker
                )
                ((given, options),) = handed
                del handed[:]
                self.assertEqual(
                    timing["cuda_initialization_overlap"],
                    "cuda_probe_with_cpu_reference_setup" if in_line else "disabled",
                )
                # A worker that defers its matrix: the stage that would
                # start the thread if it were handed one.
                self.assertIn("orbital_operators_only", options)
                self.assertIn("defer_native_laplacian", options)
                if in_line:
                    self.assertIsNone(given)
                    self.assertIsNone(preparation.symmetry_maps)
                else:
                    self.assertIs(given, maps)
                    self.assertIs(preparation.symmetry_maps, maps)
                self.assertFalse(maps.started)
                preparation.close()
        self.assertEqual(self._threads(), [])

    def test_errors_of_the_thread_are_raised_where_the_maps_are_taken(self) -> None:
        import parsec_python.acceleration.Symmetry as symmetry

        def refuse(*_arguments, **_options):
            raise ValueError("no symmetric lattice")

        # The detection: raised by its name at its join, and the reduction
        # that is published says that there is none.
        preparation = self._preparation()
        with patch.object(symmetry, "load_or_detect_reflection_reduction", refuse):
            maps = preparation.symmetry_maps
            grid = SimpleNamespace()
            maps.start(grid, (), decompose=True)
            with self.assertRaisesRegex(ValueError, "no symmetric lattice"):
                maps.reduction()
            self.assertIsNone(preparation.symmetry_reduction.wait())
            self.assertIsNone(maps.decomposition(object()))
        with self.assertRaisesRegex(RuntimeError, "start once"):
            maps.start(grid, (), decompose=True)
        # The decomposition: the reduction is there, the failure at its join.
        preparation = self._preparation()
        reduction = SimpleNamespace(group_order=2, multiplicities=np.full(3, 2))
        info = SimpleNamespace(key=None)
        with (
            patch.object(
                symmetry, "load_or_detect_reflection_reduction",
                lambda *_a, **_o: (reduction, info),
            ),
            patch.object(symmetry, "load_or_build_reflection_decomposition", refuse),
            patch.dict(driver_module.os.environ),
        ):
            driver_module.os.environ.pop("PARSEC_ORBITAL_STABILIZER_SYMMETRY", None)
            maps = preparation.symmetry_maps
            maps.start(SimpleNamespace(), (), decompose=True)
            self.assertEqual(maps.reduction(), (reduction, info))
            self.assertIs(preparation.symmetry_reduction.wait(), reduction)
            with self.assertRaisesRegex(ValueError, "no symmetric lattice"):
                maps.decomposition(reduction)
        self.assertEqual(self._threads(), [])

    def test_failure_of_the_reference_stage_comes_first_and_joins_the_thread(
        self,
    ) -> None:
        import parsec_python.acceleration.Symmetry as symmetry

        ended = []
        detect = symmetry.load_or_detect_reflection_reduction

        def slow(*arguments, **options):
            threading.Event().wait(timeout=0.3)
            ended.append(threading.current_thread().name)
            raise RuntimeError("detection failed")

        preparation = self._preparation()
        # The second atom lies outside the sphere: the reference preparation
        # refuses it after the grid, where the thread has been started.
        outside = self._problem(((0.9, -0.4, 0.3), (0.0, 0.0, 50.0)))
        with (
            patch.object(symmetry, "load_or_detect_reflection_reduction", slow),
            self.assertRaisesRegex(ValueError, "lies outside the domain"),
        ):
            try:
                self._reference_stage_without_join(preparation, outside)
            finally:
                # What prepare_single_point does on every way out.
                preparation.close()
        self.assertEqual(len(ended), 1)
        self.assertTrue(ended[0].startswith("parsec-symmetry-setup"))
        self.assertIsNone(preparation.symmetry_reduction.wait())
        self.assertIs(symmetry.load_or_detect_reflection_reduction, detect)
        self.assertEqual(self._threads(), [])

    def _reference_stage_without_join(self, preparation, problem):
        from parsec_python.Laplacian import build_negative_laplacian

        native = "parsec_python.acceleration.backends.native."
        with (
            patch(
                native + "_load_native",
                return_value=SimpleNamespace(RadialGridEvaluator=object),
            ),
            patch(native + "build_native_negative_laplacian", build_negative_laplacian),
            patch("parsec_python.acceleration.V_ion.NativeIonicBuilders", self._HostFields),
            patch.dict(driver_module.os.environ, {"PARSEC_ACCELERATED_RESIDENT": "0"}),
        ):
            return driver_module._prepare_reference_physics(
                problem, _hybrid_selection(), preparation=preparation,
                defer_native_laplacian=True, orbital_operators_only=True,
            )

    @staticmethod
    def _starting_the_maps():
        """Start the maps where the reference stage does, around its double."""

        resolve = driver_module._resolve_and_prepare_reference

        def resolving(problem, backend, **options):
            selection, reference, timing = resolve(problem, backend, **options)
            maps = options["preparation"].symmetry_maps
            if maps is not None:
                maps.start(reference.grid, reference.atoms, decompose=True)
            return selection, reference, timing

        return patch.object(driver_module, "_resolve_and_prepare_reference", resolving)

    def test_preparation_takes_the_maps_of_the_thread_and_reports_them(self) -> None:
        root = SimpleNamespace(rank=0, root=0, size=2)
        for mpi_context in (None, root, SimpleNamespace(rank=1, root=0, size=2)):
            with self.subTest(mpi_context=mpi_context), self._starting_the_maps():
                prepared = self.prepare(mpi_context)
                details = dict(prepared.system.backend_info.details)
                self.assertEqual(details["symmetry_setup"], self.overlapped)
                self.assertIn("symmetry_setup_seconds", details)
                self.assertIn("symmetry_setup_wait_seconds", details)
                # The sector solver was given the decomposition of the thread.
                decomposition = prepared.sector_solver_type.call_args.args[3]
                self.assertIs(decomposition.reduction, prepared.reduction)
                records = prepared.system.preparation_stages
                threads = {record["stage"]: record["thread"] for record in records}
                for stage in ("symmetry_reduction", "symmetry_decomposition"):
                    self.assertTrue(threads[stage].startswith("parsec-symmetry-setup"))
                for stage in ("symmetry_reduction_wait", "symmetry_decomposition_wait"):
                    self.assertEqual(threads[stage], threading.current_thread().name)
                # Each map was built once, by the thread.
                names = [record["stage"] for record in records]
                self.assertEqual(names.count("symmetry_reduction"), 1)
                self.assertEqual(names.count("symmetry_decomposition"), 1)
                self.assertEqual(self._threads(), [])

    def test_thread_whose_maps_are_not_taken_has_ended_before_the_backend_is_built(
        self,
    ) -> None:
        # A thread that was started on another grid serves nobody: the
        # calling thread builds its maps in line and reads nothing of the
        # thread, and still waits for it before the sector operators.
        acceleration = "parsec_python.acceleration."
        seen = []
        resolve = driver_module._resolve_and_prepare_reference

        def slow(*_arguments, **_options):
            # Longer than the calling thread takes to reach the backend.
            threading.Event().wait(timeout=0.3)
            seen.append(
                (threading.current_thread().name, driver_module._build_backend.called)
            )
            raise ValueError("another structure")

        def resolving(problem, backend, **options):
            selection, reference, timing = resolve(problem, backend, **options)
            maps = options["preparation"].symmetry_maps
            with patch(acceleration + "Symmetry.load_or_detect_reflection_reduction", slow):
                maps.start(SimpleNamespace(), reference.atoms, decompose=True)
            self.assertFalse(maps.serves(reference.grid, reference.atoms))
            return selection, reference, timing

        with patch.object(driver_module, "_resolve_and_prepare_reference", resolving):
            prepared = self.prepare(None)
        ((thread, backend_built),) = seen
        self.assertTrue(thread.startswith("parsec-symmetry-setup"))
        self.assertFalse(backend_built)
        prepared.build_backend.assert_called_once()
        # The maps of the calling thread, built in line: no wait for those
        # of the thread, and the sector solver has the decomposition.
        records = prepared.system.preparation_stages
        main = threading.current_thread().name
        self.assertEqual(
            [
                record["thread"] == main
                for record in records
                if record["stage"] == "symmetry_decomposition"
            ],
            [True],
        )
        self.assertNotIn(
            "symmetry_reduction_wait", [record["stage"] for record in records]
        )
        decomposition = prepared.sector_solver_type.call_args.args[3]
        self.assertIs(decomposition.reduction, prepared.reduction)
        self.assertEqual(self._threads(), [])

    def test_former_value_and_runs_without_symmetry_build_no_thread(self) -> None:
        handed = []
        resolve = driver_module._resolve_and_prepare_reference

        def resolving(problem, backend, **options):
            handed.append(options["preparation"].symmetry_maps)
            return resolve(problem, backend, **options)

        with patch.object(driver_module, "_resolve_and_prepare_reference", resolving):
            for value in ("0", "off"):
                with self.subTest(value=value):
                    prepared = self.prepare(
                        None, environment={"PARSEC_OVERLAP_SYMMETRY_SETUP": value}
                    )
                    self.assertIsNone(handed[-1])
                    details = dict(prepared.system.backend_info.details)
                    self.assertEqual(details["symmetry_setup"], "inline")
                    self.assertNotIn("symmetry_setup_seconds", details)
                    # In line, by the calling thread, after the reference stage.
                    stages = [
                        (record["stage"], record["thread"])
                        for record in prepared.system.preparation_stages
                        if record["stage"].startswith(("symmetry", "reference"))
                    ]
                    main = threading.current_thread().name
                    self.assertEqual(
                        stages,
                        [
                            ("reference_preparation", main),
                            ("symmetry_reduction", main),
                            ("symmetry_decomposition", main),
                        ],
                    )
            # By default the reference stage is handed a thread to start; one
            # that it does not start leaves the maps in line.
            prepared = self.prepare(None)
            self.assertIsInstance(handed[-1], driver_module._OverlappedSymmetryMaps)
            self.assertFalse(handed[-1].started)
            self.assertEqual(
                dict(prepared.system.backend_info.details)["symmetry_setup"], "inline"
            )
        # Without symmetry nothing is detected, and no thread waits for it.
        with patch.object(driver_module, "_resolve_and_prepare_reference", resolving):
            system = driver_module.prepare_single_point(
                parse_parsec_input(SMOKE_INPUT).problem, backend="scipy", symmetry="off"
            )
        self.assertIsNone(handed[-1])
        self.assertEqual(
            dict(system.backend_info.details)["symmetry_setup"], "not used"
        )
        self.assertNotIn(
            "symmetry_reduction",
            [record["stage"] for record in system.preparation_stages],
        )
        self.assertEqual(self._threads(), [])

    def test_detection_that_fails_on_the_thread_falls_back_as_in_line(self) -> None:
        def refuse(*_arguments, **_options):
            raise ValueError("no symmetric lattice")

        acceleration = "parsec_python.acceleration."
        resolve = driver_module._resolve_and_prepare_reference

        def resolving(problem, backend, **options):
            selection, reference, timing = resolve(problem, backend, **options)
            maps = options["preparation"].symmetry_maps
            with patch(
                acceleration + "Symmetry.load_or_detect_reflection_reduction", refuse
            ):
                maps.start(reference.grid, reference.atoms, decompose=True)
                maps.close()
            return selection, reference, timing

        with (
            patch.object(driver_module, "_resolve_and_prepare_reference", resolving),
            patch.object(driver_module, "_hartree_boundary_tail", return_value=None),
            patch(acceleration + "Hartree.native_poisson.NativePoissonSolver"),
            patch(acceleration + "Hartree.native_boundary.NativeMultipoleBoundaryBuilder"),
        ):
            prepared = self.prepare(None)
        details = dict(prepared.system.backend_info.details)
        # The words of the fallback of a detection that fails in line.
        self.assertEqual(
            details["symmetry_detection"],
            "automatic detection fell back to full grid: ValueError: no symmetric lattice",
        )
        self.assertEqual(details["symmetry_setup"], self.overlapped)
        self.assertEqual(details["orbital_symmetry"], "full grid (no nontrivial supported symmetry)")
        prepared.sector_solver_type.assert_not_called()
        self.assertEqual(self._threads(), [])


class PreparationStageTests(unittest.TestCase):
    """The clock of a preparation, and the threads it joins on the way out."""

    prepare = MPIRolePreparationTests.prepare
    _starting_the_fields = staticmethod(IonicFieldOverlapTests._starting_the_fields)

    def tearDown(self) -> None:
        with driver_module._REFERENCE_CACHE_LOCK:
            driver_module._REFERENCE_CACHE.clear()

    @staticmethod
    def _threads(prefix):
        return [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith(prefix)
        ]

    def test_clock_records_the_stages_of_every_thread_from_its_origin(self) -> None:
        stages = driver_module._PreparationStages()
        started = driver_module.perf_counter()
        stages.add("second", started + 2.0, started + 3.0)
        stages.add("first", started + 1.0, started + 5.0)
        timed = stages.timed("call", lambda value, scale=1: value * scale)
        self.assertEqual(timed(3, scale=2), 6)
        # A name may be read from the arguments of the call, and a call that
        # raises is a stage up to its failure.
        named = stages.timed(lambda *_arguments, core=False: "core" if core else "valence", dict)
        named(core=True)
        named()

        def fail():
            raise RuntimeError("stage failed")

        with self.assertRaisesRegex(RuntimeError, "stage failed"):
            stages.timed("failed", fail)()
        worker = threading.Thread(
            target=lambda: stages.add("beside", driver_module.perf_counter()),
            name="parsec-test-stage",
        )
        worker.start()
        worker.join()

        records = stages.records()
        self.assertEqual(
            sorted(record["stage"] for record in records),
            ["beside", "call", "core", "failed", "first", "second", "valence"],
        )
        self.assertEqual(
            set(records[0]), {"stage", "thread", "start_seconds", "end_seconds"}
        )
        # In the order of the starts, in seconds from the origin of the clock.
        starts = [record["start_seconds"] for record in records]
        self.assertEqual(starts, sorted(starts))
        by_name = {record["stage"]: record for record in records}
        offset = started - stages.origin
        self.assertAlmostEqual(by_name["first"]["start_seconds"], offset + 1.0, 9)
        self.assertAlmostEqual(by_name["first"]["end_seconds"], offset + 5.0, 9)
        self.assertEqual(by_name["first"]["thread"], threading.current_thread().name)
        self.assertEqual(by_name["beside"]["thread"], "parsec-test-stage")
        for record in records:
            self.assertGreaterEqual(record["end_seconds"], record["start_seconds"])

    def test_preparation_reports_when_each_of_its_stages_ran(self) -> None:
        root = SimpleNamespace(rank=0, root=0, size=2)
        for mpi_context in (None, root, SimpleNamespace(rank=1, root=0, size=2)):
            with self.subTest(mpi_context=mpi_context):
                prepared = self.prepare(mpi_context)
                records = prepared.system.preparation_stages
                names = [record["stage"] for record in records]
                worker = mpi_context is not None and mpi_context.rank != 0
                expected = {
                    "backend_resolution",
                    "reference_preparation",
                    "symmetry_reduction",
                    "symmetry_decomposition",
                    "backend_build",
                    "sector_eigensolver",
                    "xc_evaluator",
                    "hartree_solver",
                    "preparation",
                }
                if not worker:
                    expected |= {
                        "poisson_solver",
                        "hartree_boundary_setup",
                        "hartree_boundary_setup_wait",
                    }
                self.assertEqual(set(names), expected)
                self.assertEqual(len(names), len(expected))
                by_name = {record["stage"]: record for record in records}
                whole = by_name["preparation"]
                self.assertEqual(whole["start_seconds"], 0.0)
                for record in records:
                    self.assertGreaterEqual(record["start_seconds"], 0.0)
                    self.assertLessEqual(record["end_seconds"], whole["end_seconds"])
                # The stages of the calling thread in the order it runs them.
                main = [
                    record["stage"]
                    for record in records
                    if record["thread"] == threading.current_thread().name
                    and record["stage"] not in {"preparation", "poisson_solver", "hartree_boundary_setup_wait"}
                ]
                self.assertEqual(
                    main,
                    [
                        "backend_resolution",
                        "reference_preparation",
                        "symmetry_reduction",
                        "symmetry_decomposition",
                        "backend_build",
                        "sector_eigensolver",
                        "xc_evaluator",
                        "hartree_solver",
                    ],
                )
                if not worker:
                    # The boundary geometry on its thread, joined inside the
                    # stage of the Hartree solver, after the Poisson solver.
                    boundary = by_name["hartree_boundary_setup"]
                    self.assertTrue(boundary["thread"].startswith("parsec-hartree-setup"))
                    solver, wait = by_name["hartree_solver"], by_name["hartree_boundary_setup_wait"]
                    self.assertGreaterEqual(wait["start_seconds"], by_name["poisson_solver"]["end_seconds"])
                    self.assertGreaterEqual(wait["end_seconds"], boundary["end_seconds"])
                    self.assertLessEqual(wait["end_seconds"], solver["end_seconds"])
                    self.assertGreaterEqual(wait["start_seconds"], solver["start_seconds"])
                # The timeline is the prepared system's and no entry of the
                # backend report: the result archive stores every entry of
                # the report at the width of the longest, and one line of
                # all stages would widen each of them.
                details = dict(prepared.system.backend_info.details)
                self.assertNotIn("preparation_stage_seconds", details)
                for key, value in details.items():
                    # Nor does an entry list the stages under another name.
                    self.assertFalse(
                        "backend_build" in value and "sector_eigensolver" in value, key
                    )
                self.assertEqual(self._threads("parsec-hartree-setup"), [])

    def test_ionic_thread_and_its_join_are_stages(self) -> None:
        def complete(reference, **_builders):
            return SimpleNamespace(**vars(reference))

        overlap = driver_module._OverlappedIonicFields()
        stages = driver_module._PreparationStages()
        with patch.object(driver_module, "complete_reference_single_point", complete):
            overlap.start(SimpleNamespace(), stages=stages)
            overlap.result()
        by_name = {record["stage"]: record for record in stages.records()}
        self.assertEqual(set(by_name), {"ionic_fields", "ionic_fields_wait"})
        self.assertTrue(by_name["ionic_fields"]["thread"].startswith("parsec-ionic-setup"))
        self.assertEqual(
            by_name["ionic_fields_wait"]["thread"], threading.current_thread().name
        )
        self.assertGreaterEqual(
            by_name["ionic_fields_wait"]["end_seconds"],
            by_name["ionic_fields"]["end_seconds"],
        )
        # Closing after the join, or before any start, waits for nothing.
        overlap.close()
        driver_module._OverlappedIonicFields().close()

    def test_seconds_of_the_fields_leave_out_the_choice_of_their_grid(self) -> None:
        # Before its first sum the thread may wait for the symmetry
        # reduction and test the images.  The time of the fields, from which
        # the hidden time is counted, is that of the sums as it was before
        # the thread chose a grid; the choice has an entry of its own, and
        # the stage of the thread spans both.  A clock that only the two
        # doubles move makes the seconds exact.
        now = [100.0]

        def choose():
            now[0] += 5.0
            return "symmetry wedge (2 of 8 points)"

        def complete(reference, **_builders):
            now[0] += 2.0
            return reference

        overlap = driver_module._OverlappedIonicFields()
        with (
            patch.object(driver_module, "perf_counter", lambda: now[0]),
            patch.object(driver_module, "complete_reference_single_point", complete),
        ):
            stages = driver_module._PreparationStages()
            overlap.start(SimpleNamespace(), stages=stages, before_sums=choose)
            # The join finds the thread ended: it waits for nothing.
            for thread in self._threads("parsec-ionic-setup"):
                thread.join(timeout=10.0)
            overlap.result()
        self.assertEqual(overlap.seconds, 2.0)
        self.assertEqual(overlap.sum_grid_seconds, 5.0)
        details = dict(overlap.details())
        self.assertEqual(details["ionic_setup_seconds"], "2.000000")
        self.assertEqual(details["ionic_setup_wait_seconds"], "0.000000")
        self.assertEqual(details["ionic_setup_overlapped_seconds"], "2.000000")
        self.assertEqual(details["ionic_sum_grid"], "symmetry wedge (2 of 8 points)")
        self.assertEqual(details["ionic_sum_grid_choice_seconds"], "5.000000")
        (fields,) = [
            record for record in stages.records() if record["stage"] == "ionic_fields"
        ]
        self.assertEqual((fields["start_seconds"], fields["end_seconds"]), (0.0, 7.0))
        # A thread that is told nothing about a grid reports neither entry.
        overlap = driver_module._OverlappedIonicFields()
        with patch.object(driver_module, "complete_reference_single_point", complete):
            overlap.start(SimpleNamespace())
            overlap.result()
        details = dict(overlap.details())
        self.assertNotIn("ionic_sum_grid", details)
        self.assertNotIn("ionic_sum_grid_choice_seconds", details)

    def test_failed_preparation_has_joined_the_thread_of_the_ionic_fields(self) -> None:
        release = threading.Event()
        ended = []

        def complete(reference, **_builders):
            # Longer than the rest of the preparation takes to fail.
            release.wait(timeout=0.5)
            ended.append(threading.current_thread().name)
            return reference

        with (
            self._starting_the_fields(),
            patch.object(driver_module, "complete_reference_single_point", complete),
            self.assertRaisesRegex(ValueError, "PARSEC_ORBITAL_STABILIZER_SYMMETRY"),
        ):
            self.prepare(
                None,
                environment={
                    **IonicFieldOverlapTests.gpu_fields,
                    # Refused after the reference stage has started the
                    # thread and before the join in front of the XC evaluator.
                    "PARSEC_ORBITAL_STABILIZER_SYMMETRY": "sometimes",
                },
            )
        # The error is the preparation's own, and the thread has ended.
        self.assertEqual(len(ended), 1)
        self.assertEqual(self._threads("parsec-ionic-setup"), [])

    def test_failed_preparation_has_joined_the_thread_of_the_ion_ion_energy(
        self,
    ) -> None:
        ended = []

        def slow(_atoms, _pseudopotentials):
            # Longer than the fields and the rest of the preparation take.
            threading.Event().wait(timeout=0.5)
            ended.append(threading.current_thread().name)
            return 0.0

        resolve = driver_module._resolve_and_prepare_reference

        def resolving(problem, backend, **options):
            selection, reference, timing = resolve(problem, backend, **options)
            reference.pseudopotentials = {}
            options["ionic_field_overlap"].start(reference, ion_ion_beside=True)
            return selection, reference, timing

        with (
            patch.object(driver_module, "_resolve_and_prepare_reference", resolving),
            patch.object(driver_module, "ion_ion_energy", slow),
            # The fields end at once and never come to the stage of the
            # energy: nothing but the way out waits for its thread.
            patch.object(
                driver_module,
                "complete_reference_single_point",
                lambda reference, **_builders: reference,
            ),
            self.assertRaisesRegex(ValueError, "PARSEC_ORBITAL_STABILIZER_SYMMETRY"),
        ):
            self.prepare(
                None,
                environment={
                    **IonicFieldOverlapTests.gpu_fields,
                    "PARSEC_ORBITAL_STABILIZER_SYMMETRY": "sometimes",
                },
            )
        self.assertEqual(len(ended), 1)
        self.assertTrue(ended[0].startswith("parsec-ion-ion"))
        self.assertEqual(self._threads("parsec-ion-ion"), [])
        self.assertEqual(self._threads("parsec-ionic-setup"), [])

    def test_failed_preparation_has_joined_the_thread_of_the_boundary(self) -> None:
        ended = []

        def build(_reference, reduction, **_options):
            threading.Event().wait(timeout=0.3)
            ended.append(threading.current_thread().name)
            raise RuntimeError("boundary geometry failed")

        with (
            patch.object(driver_module, "_build_native_boundary_builder", side_effect=build),
            # The reduced Poisson solver of the GPU CG has no fallback: its
            # failure leaves the preparation while the boundary thread works.
            self.assertRaisesRegex(RuntimeError, "no reduced operator"),
        ):
            self.prepare(
                None,
                linear_backend="cupy",
                poisson_failure=RuntimeError("no reduced operator"),
            )
        # What the thread raised is not read: the first error is reported.
        self.assertEqual(len(ended), 1)
        self.assertTrue(ended[0].startswith("parsec-hartree-setup"))
        self.assertEqual(self._threads("parsec-hartree-setup"), [])

    def test_closing_a_preparation_joins_the_last_thread_first_and_drops_errors(self) -> None:
        preparation = driver_module._Preparation()
        order = []

        def failing():
            order.append("second")
            raise RuntimeError("join failed")

        preparation.join_on_exit(lambda: order.append("first"))
        preparation.join_on_exit(failing)
        preparation.close()
        self.assertEqual(order, ["second", "first"])
        # Nothing is joined twice.
        preparation.close()
        self.assertEqual(order, ["second", "first"])


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
        # A sector that shares its basis gives that device work beside its
        # own only where it has no other device and the sharing rule counts
        # the room.
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

        def run(switch, grid="full", **more):
            # The sums of the thread on every grid point unless a run names
            # the default: fields summed on the symmetry wedge differ from
            # the in-line ones by round-off (below).
            with patch.dict(
                driver_module.os.environ,
                {
                    "PARSEC_IONIC_BACKEND": "cupy",
                    "PARSEC_OVERLAP_IONIC_SETUP": switch,
                    "PARSEC_ACCELERATED_RESIDENT": "0",
                    "PARSEC_HARTREE_LINEAR_BACKEND": "cupy",
                    "PARSEC_IONIC_SUM_GRID": grid,
                    **more,
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
        # The ion-ion energy on its own thread by default, and as the last
        # stage of the ionic thread with the former value: the same bits.
        self.assertEqual(details["ion_ion_setup"], "own thread beside the ionic fields")
        self.assertNotIn("ion_ion_setup", inline_details)
        former_system, former_details, former = run(
            "1", PARSEC_OVERLAP_ION_ION_SETUP="0"
        )
        self.assertEqual(
            former_details["ion_ion_setup"], "last stage of the ionic thread"
        )
        # The symmetry maps beside the projectors where the fields are left
        # to their thread, and in line with the former value or with the
        # fields in line: the same bits again.
        self.assertEqual(details["symmetry_setup"], "overlapped with the projectors")
        self.assertEqual(inline_details["symmetry_setup"], "inline")
        maps_system, maps_details, maps_result = run(
            "1", PARSEC_OVERLAP_SYMMETRY_SETUP="0"
        )
        self.assertEqual(maps_details["symmetry_setup"], "inline")
        self.assertEqual(
            maps_details["detected_symmetry_group_order"],
            details["detected_symmetry_group_order"],
        )
        for built, result in (
            (system, actual),
            (former_system, former),
            (maps_system, maps_result),
        ):
            for name in ("ionic_potential", "initial_density", "core_density"):
                with self.subTest(field=name):
                    self.assertEqual(
                        getattr(built, name).tobytes(),
                        getattr(inline_system, name).tobytes(),
                    )
            self.assertEqual(built.ion_ion_energy, inline_system.ion_ion_energy)
            np.testing.assert_array_equal(result.eigenvalues, expected.eigenvalues)
            np.testing.assert_array_equal(result.density, expected.density)
            self.assertEqual(result.energies, expected.energies)
            self.assertEqual(result.iterations, expected.iterations)
        self.assertEqual(
            details["ionic_sum_grid"], "full grid (PARSEC_IONIC_SUM_GRID=full)"
        )
        self.assertNotIn("ionic_sum_grid", inline_details)

        # By default the thread sums on the symmetry wedge: one point per
        # orbit, whose value every point of the orbit takes.
        wedge_system, wedge_details, wedge_result = run("1", grid="auto")
        reduction = wedge_system.scalar_field_adapter.reduction
        rows, orbit = reduction.representative_rows, reduction.full_to_wedge
        self.assertEqual(
            wedge_details["ionic_sum_grid"],
            f"symmetry wedge ({rows.size} of {orbit.size} points)",
        )
        self.assertLess(rows.size, orbit.size)
        for name in ("ionic_potential", "initial_density", "core_density"):
            with self.subTest(wedge_field=name):
                on_orbits = getattr(wedge_system, name)
                everywhere = getattr(inline_system, name)
                self.assertEqual(on_orbits.shape, everywhere.shape)
                np.testing.assert_array_equal(on_orbits, on_orbits[rows][orbit])
                if name != "initial_density":
                    # The sum of the representative row, bit for bit; the
                    # density is normalized by the sum of all its rows.
                    np.testing.assert_array_equal(on_orbits[rows], everywhere[rows])
                # Round-off of one sum over the atoms on the other rows.
                scale = float(np.max(np.abs(everywhere)))
                self.assertLessEqual(
                    float(np.max(np.abs(on_orbits - everywhere))),
                    16 * np.finfo(float).eps * scale,
                )
        self.assertEqual(wedge_system.ion_ion_energy, inline_system.ion_ion_energy)
        self.assertEqual(wedge_result.iterations, expected.iterations)
        np.testing.assert_allclose(
            wedge_result.eigenvalues, expected.eigenvalues, rtol=0.0, atol=1.0e-11
        )
        np.testing.assert_allclose(
            wedge_result.density, expected.density, rtol=0.0, atol=1.0e-11
        )
        self.assertAlmostEqual(
            wedge_result.energies.total, expected.energies.total, delta=1.0e-11
        )

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

    def test_tiles_that_the_filter_work_chooses_leave_the_scf_bitwise_unchanged(
        self,
    ) -> None:
        from parsec_python.acceleration.Eigensolvers import symmetry
        from parsec_python.acceleration.backends import implicit_stencil

        problem = parse_parsec_input(SMOKE_INPUT).problem
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))
        names = (
            "PARSEC_CUPY_IMPLICIT_TILE",
            "PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS",
            "PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS",
        )
        kept = {
            name: value
            for name, value in driver_module.os.environ.items()
            if name not in names
        }
        handed = []
        choice = symmetry.implicit_tile_for_sector

        def recorded(rows, device_count, float32_filter=False, sectors_on_device=1, work=None):
            tile = choice(rows, device_count, float32_filter, sectors_on_device, work)
            handed.append((work, tile))
            return tile

        def run(**settings):
            del handed[:]
            with patch.dict(
                driver_module.os.environ,
                {
                    **kept,
                    "PARSEC_CUPY_MIXED_FILTER": "off",
                    "PARSEC_ACCELERATED_RESIDENT": "0",
                    **settings,
                },
                clear=True,
            ), patch.object(symmetry, "implicit_tile_for_sector", recorded):
                system = prepare_single_point(problem, backend="auto")
                details = dict(system.backend_info.details)
                return run_scf(system), details

        storage = "orbital_sector_finite_difference_storage"
        # The defaults: the driver hands every sector the filter work of the
        # input, which for a few states repays no packing.
        expected, slot_major = run()
        self.assertEqual(slot_major["orbital_sector_tile_pack_workers"], "no tiles packed")
        self.assertEqual(len(handed), 8)
        plan = driver_module._filter_plan(SimpleNamespace(input=problem, electron_count=8.0))
        for work, tile in handed:
            self.assertEqual(
                (work.applications, work.devices, tile), (plan.applications, 1, 0)
            )
            self.assertGreaterEqual(work.columns, 1)
            self.assertLessEqual(work.columns, plan.states // 8 + plan.buffer + 1)
        self.assertEqual(
            (plan.states, plan.buffer),
            (problem.scf.number_of_states, problem.eigensolver.subspace_buffer),
        )
        # Packing that costs next to nothing and tiles that save at any
        # share: the rule itself now asks for tiles, of every sector.
        with patch.object(implicit_stencil, "PACKING_SECONDS_PER_ROW", 1.0e-18), patch.object(
            implicit_stencil, "REGULAR_SHARE_WITHOUT_GAIN", -1.0
        ):
            actual, details = run()
            self.assertEqual([tile for _work, tile in handed], [16] * 8)
            self.assertEqual(details[storage], "implicit_affine_tile_16")
            # The two row limits at their former values give the choice
            # before the rule, whatever the rule would say.
            former, former_details = run(
                PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS="650000",
                PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS="1100000",
            )
            self.assertEqual([tile for _work, tile in handed], [0] * 8)
        self.assertEqual(former_details[storage], slot_major[storage])
        for result in (actual, former):
            self.assertEqual(result.iterations, expected.iterations)
            np.testing.assert_array_equal(result.eigenvalues, expected.eigenvalues)
            np.testing.assert_array_equal(result.density, expected.density)
            np.testing.assert_array_equal(
                result.hartree_potential, expected.hartree_potential
            )
            self.assertEqual(result.energies, expected.energies)

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
