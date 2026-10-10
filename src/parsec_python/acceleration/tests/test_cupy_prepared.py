"""Real-device validation of the optional prepared FP64 Hartree solve.

The packing of the stencil and what surrounds the capture of the CG graph are
checked without a device.
"""
from functools import partial
import gc
import os
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from parsec_python.acceleration.Hartree.cupy_prepared import (
    CuPyPreparedConjugateGradientBackend, CuPyPreparedPoissonSolver, pack_stencil,
    packed_stencil,
)
from parsec_python.acceleration.Hartree.native_poisson import NativePoissonSolver
from parsec_python.acceleration.Hartree.symmetry_poisson import SymmetryReducedPoissonSolver
from parsec_python.acceleration.backends.cupy import cupy_available, require_cupy
from parsec_python.acceleration.backends.cupy_stencil_major import StencilMajorHostMetadata
from parsec_python.acceleration.Symmetry import (
    AxisReflectionReduction, ReflectionRepresentationDecomposition,
)
from parsec_python.acceleration.Symmetry.operator_cache import load_or_build_reduced_operators
from parsec_python.acceleration.SCF.symmetry_fields import SymmetryScalarField
from parsec_python.acceleration.tests.test_filter_graph import _CaptureStream, _Cyclic
from parsec_python.Grid import build_cluster_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.V_ion import NonlocalProjectorOperator
from parsec_python.models import Atom, GridSettings


def totally_symmetric_sector():
    """The packed stencil the symmetry eigensolver hands to the Poisson setup.

    A zero grid shift puts orbits on the mirror planes, so the multiplicities
    differ and the reduced coefficients are not those of the full stencil.
    """
    grid = build_cluster_grid(GridSettings(spacing=.8, radius=3.3, expansion_order=4, shift=(0., 0., 0.)))
    a = build_negative_laplacian(grid)
    reduction = AxisReflectionReduction.detect(grid, (Atom("H", (0., 0., 0.)),))
    decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
    symmetric = int(np.flatnonzero(np.all(decomposition.characters == 1, axis=1))[0])
    projectors = NonlocalProjectorOperator(
        projectors=sp.csc_matrix(np.random.default_rng(3).normal(size=(grid.size, 2))),
        signs=np.array([1., -1.]), labels=((0, 0, 0), (0, 0, 1)))
    bundle = load_or_build_reduced_operators(decomposition, a, projectors, cache_directory=None,
                                             representations=(symmetric,))
    return grid, a, reduction, bundle.stencil_metadata[symmetric]


def slot_major_matvec(neighbors, codes, palette, x):
    """NumPy transcription of the CUDA matvec: every row adds its slots in order."""
    value = np.zeros(neighbors.shape[1])
    for slot in range(neighbors.shape[0]):
        valid = neighbors[slot] >= 0
        value[valid] += palette[codes[slot, valid]] * x[neighbors[slot, valid]]
    return value


class _HostBackend:
    """Stand-in for a CG backend; counts how often its CSR operator is read."""
    def __init__(self, operator):
        self.received, self.reads = operator, 0
        self.shape = tuple(operator.shape)
        self.storage_mode, self.worker_count, self.coefficient_palette_size = "host", 0, 1

    @property
    def operator(self):
        self.reads += 1
        received = self.received
        return received.to_csr() if isinstance(received, StencilMajorHostMetadata) else received


class PackingTests(unittest.TestCase):
    def test_sector_packing_is_what_packing_its_csr_again_produces(self):
        _, _, reduction, metadata = totally_symmetric_sector()
        self.assertEqual(metadata.shape, (reduction.wedge_size, reduction.wedge_size))
        self.assertGreater(len(set(reduction.multiplicities.tolist())), 1)
        # The independent route: CSR out of the packing, packed by pack_stencil.
        canonical, neighbors, codes, palette = pack_stencil(metadata.to_csr())
        reused_neighbors, reused_codes, reused_palette = packed_stencil(metadata)
        self.assertIs(reused_neighbors, metadata.neighbors)
        self.assertIs(reused_codes, metadata.coefficient_codes)
        self.assertIs(reused_palette, metadata.coefficient_palette)
        # Same layout for the kernels: C-order (slot, row), int32 and uint8.
        for reused, repacked in ((reused_neighbors, neighbors), (reused_codes, codes)):
            self.assertEqual(reused.dtype, repacked.dtype)
            self.assertEqual(reused.shape, repacked.shape)
            self.assertTrue(reused.flags.c_contiguous)
        np.testing.assert_array_equal(reused_neighbors, neighbors)
        active = neighbors >= 0
        # The same float64 in every slot, compared as bit patterns ...
        np.testing.assert_array_equal(reused_palette[reused_codes[active]].view(np.uint64),
                                      palette[codes[active]].view(np.uint64))
        # ... selected from palettes holding the same values in another order.
        np.testing.assert_array_equal(np.sort(reused_palette), palette)
        self.assertFalse(np.array_equal(reused_palette, palette))
        x = np.random.default_rng(5).normal(size=metadata.shape[0])
        expected = slot_major_matvec(neighbors, codes, palette, x)
        actual = slot_major_matvec(reused_neighbors, reused_codes, reused_palette, x)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        np.testing.assert_allclose(actual, canonical @ x, rtol=1e-13, atol=1e-13)

    def test_packed_stencil_rejects_what_pack_stencil_rejects(self):
        nonfinite = StencilMajorHostMetadata(shape=(2, 2), neighbors=np.array([[0, 1]]),
            coefficient_codes=np.zeros((1, 2)), coefficient_palette=np.array([np.inf]))
        wide = StencilMajorHostMetadata(shape=(2, 2), neighbors=np.zeros((257, 2)),
            coefficient_codes=np.zeros((257, 2)), coefficient_palette=np.array([1.]))
        for metadata in (nonfinite, wide):
            with self.assertRaises(ValueError):
                packed_stencil(metadata)

    def test_packed_operator_reaches_the_backend_and_its_csr_is_formed_on_request(self):
        _, _, reduction, metadata = totally_symmetric_sector()
        factory = lambda operator: NativePoissonSolver(operator, backend_factory=_HostBackend)
        solver = SymmetryReducedPoissonSolver(metadata, reduction, operator_is_reduced=True, solver_factory=factory)
        backend = solver.solver.backend
        self.assertIs(backend.received, metadata)
        self.assertEqual(solver.shape, metadata.shape)
        self.assertEqual(backend.reads, 0)
        reduced = solver.reduced_negative_laplacian
        self.assertEqual(backend.reads, 1)
        self.assertIs(solver.reduced_negative_laplacian, reduced)
        self.assertEqual(backend.reads, 1)
        # A CSR operator is canonicalised by the adapter and handed on as CSR.
        from_csr = SymmetryReducedPoissonSolver(metadata.to_csr(), reduction, operator_is_reduced=True,
                                                solver_factory=factory)
        self.assertIs(from_csr.solver.backend.received, from_csr.reduced_negative_laplacian)
        self.assertEqual(from_csr.solver.backend.reads, 0)
        for matrix in (reduced, solver.negative_laplacian, from_csr.negative_laplacian):
            for name in ("indptr", "indices", "data"):
                np.testing.assert_array_equal(getattr(matrix, name),
                                              getattr(from_csr.reduced_negative_laplacian, name))
        # The wedge check applies to a packed operator as well.
        small = StencilMajorHostMetadata(shape=(2, 2), neighbors=np.array([[0, 1]]),
            coefficient_codes=np.zeros((1, 2)), coefficient_palette=np.array([1.]))
        with self.assertRaisesRegex(ValueError, "symmetry wedge"):
            SymmetryReducedPoissonSolver(small, reduction, operator_is_reduced=True, solver_factory=factory)

    def test_exact_reconstruction_with_duplicates_empty_rows_and_uint16(self):
        n = 521
        # More than 256 unique coefficients exercises the uint16 path.
        a = sp.diags((np.linspace(.1, .9, n-1), np.arange(1., n+1)), (-1, 0), format="csr")
        a[7, 7] = 0
        a.eliminate_zeros()
        canonical, indices, codes, palette = pack_stencil(a)
        self.assertEqual(codes.dtype, np.uint16)
        reconstructed = np.zeros((n, n))
        for slot in range(indices.shape[0]):
            valid = indices[slot] >= 0
            reconstructed[np.flatnonzero(valid), indices[slot, valid]] += palette[codes[slot, valid]]
        np.testing.assert_array_equal(reconstructed, canonical.toarray())
        duplicate = sp.csr_matrix((np.array([1., 2.]), np.array([0, 0]), np.array([0, 2, 2])), shape=(2, 2))
        canonical, _, _, palette = pack_stencil(duplicate)
        np.testing.assert_array_equal(canonical.toarray(), [[3., 0.], [0., 0.]])
        np.testing.assert_array_equal(palette, [3.])

    def test_rejects_invalid_operators(self):
        for a in (sp.csr_matrix((0, 0)), sp.csr_matrix((2, 3)), sp.diags([np.nan])):
            with self.subTest(shape=a.shape), self.assertRaises(ValueError):
                pack_stencil(a)


class _Device:
    """Stands for the one CUDA device of the stand-in."""
    id = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class CaptureStandInTests(unittest.TestCase):
    """What surrounds the capture of the CG graph, on host memory and a stand-in stream."""

    def setUp(self):
        self.addCleanup(gc.enable if gc.isenabled() else gc.disable)
        gc.enable()
        # Every stream that a backend created, and the error with which the next ones end a capture.
        self.streams, self.end_error = [], None

    def backend(self, launch, **options):
        """A backend on a stand-in for the CuPy calls of its constructor; every kernel is ``launch``."""
        def stream(non_blocking=False):
            self.streams.append(_CaptureStream(self.end_error))
            return self.streams[-1]

        cp = SimpleNamespace(
            cuda=SimpleNamespace(Device=lambda device=None: _Device(), Stream=stream,
                                 runtime=SimpleNamespace(streamCaptureModeRelaxed=2)),
            float64=np.float64, asarray=np.asarray, empty=np.empty, zeros=np.zeros,
            RawModule=lambda code, options: SimpleNamespace(get_function=lambda name: launch))
        with patch("parsec_python.acceleration.backends.cupy.require_cupy", return_value=(cp, None)):
            return CuPyPreparedConjugateGradientBackend(sp.eye(5, format="csr"), **options)

    def test_the_graph_is_captured_with_the_collector_off_and_without_a_collection(self):
        for enabled in (True, False):
            with self.subTest(collector_enabled=enabled):
                destroyed, seen = [], []

                def launch(grid, block, args):
                    seen.append((self.streams[-1].capturing, gc.isenabled(), list(destroyed)))

                # Garbage that waits for a collection: none can have run while the collector was off.
                gc.disable()
                _Cyclic(destroyed, self.streams)
                (gc.enable if enabled else gc.disable)()
                backend = self.backend(launch, graph_iterations=8)
                stream = self.streams[-1]
                # Five launches per iteration, all of them in the one capture and with the collector off.
                self.assertEqual([(capturing, collecting) for capturing, collecting, _ in seen], [(True, False)] * 40)
                # Nothing was destroyed while the stream captured.
                self.assertEqual([found for _, _, found in seen], [seen[0][2]] * 40)
                if not enabled:
                    # Only a collection of the guard could have destroyed the garbage: it makes none.
                    self.assertEqual((seen[0][2], destroyed), ([], []))
                self.assertIs(backend.stream, stream)
                self.assertIsNotNone(backend.graph)
                self.assertEqual((stream.capturing, stream.ended), (False, 1))
                self.assertEqual(gc.isenabled(), enabled)

    def test_a_failed_capture_is_ended_and_its_error_is_raised(self):
        # With and without an error of ending the capture, as an invalidated one reports.
        for end_error in (None, RuntimeError("the capture was invalidated")):
            for enabled in (True, False):
                with self.subTest(end_error=end_error, collector_enabled=enabled):
                    (gc.enable if enabled else gc.disable)()
                    self.end_error, calls = end_error, []

                    def launch(grid, block, args):
                        calls.append(1)
                        if len(calls) == 3:
                            raise RuntimeError("injected launch failure")

                    with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
                        self.backend(launch, graph_iterations=8)
                    self.assertEqual((self.streams[-1].capturing, self.streams[-1].ended), (False, 1))
                    self.assertEqual(gc.isenabled(), enabled)

    def test_an_error_of_ending_the_capture_is_raised_and_the_capture_is_ended_once(self):
        self.end_error = RuntimeError("the capture was invalidated")
        with self.assertRaisesRegex(RuntimeError, "the capture was invalidated"):
            self.backend(lambda grid, block, args: None, graph_iterations=8)
        self.assertEqual((self.streams[-1].capturing, self.streams[-1].ended), (False, 1))
        self.assertTrue(gc.isenabled())

    def test_an_invalidated_capture_is_recorded_again(self):
        calls = []

        def launch(grid, block, args):
            calls.append(1)
            # The third launch finds its capture invalidated, once, as after the end of another thread.
            if len(calls) == 3:
                raise RuntimeError("CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: operation failed due to a previous error")

        with self.assertWarnsRegex(RuntimeWarning, r"is repeated \(attempt 2 of 4\)"):
            backend = self.backend(launch, graph_iterations=8)
        stream = self.streams[-1]
        self.assertIsNotNone(backend.graph)
        # The invalidated capture was ended, and all forty launches were recorded again.
        self.assertEqual((stream.capturing, stream.ended, len(calls)), (False, 2, 43))
        self.assertTrue(gc.isenabled())

    def test_single_iterations_are_not_captured(self):
        launches = []
        backend = self.backend(lambda grid, block, args: launches.append(1), graph_iterations=1)
        self.assertIsNone(backend.graph)
        self.assertEqual((launches, self.streams[-1].ended), ([], 0))


@unittest.skipUnless(cupy_available(), "a real CUDA device and CuPy are required")
class DevicePoissonTests(unittest.TestCase):
    def tearDown(self):
        # Solvers and their graphs that a test left in a reference cycle are destroyed here, between the tests,
        # and not by a collection inside the next one.
        gc.collect()

    def solve(self, a, rhs, initial=None, steps=8, budget=1000, rtol=1e-12, atol=1e-14):
        backend = CuPyPreparedConjugateGradientBackend(a, graph_iterations=steps)
        result = backend.solve(rhs, np.zeros_like(rhs) if initial is None else initial,
                               relative_tolerance=rtol, absolute_tolerance=atol, max_iterations=budget)
        return result, backend

    def test_non_multiple_block_size_matches_direct_solve_and_true_residual(self):
        n = 773
        a = sp.diags((-np.ones(n-1), 2.1*np.ones(n), -np.ones(n-1)), (-1, 0, 1), format="csr")
        rhs = np.random.default_rng(23).normal(size=n)
        result, backend = self.solve(a, rhs)
        self.assertTrue(result["converged"])
        self.assertFalse(result["breakdown"])
        np.testing.assert_allclose(result["solution"], spla.spsolve(a, rhs), rtol=1e-10, atol=1e-10)
        self.assertAlmostEqual(result["residual_norm"], np.linalg.norm(rhs-a@result["solution"]), delta=2e-13)
        self.assertEqual(result["matrix_vector_products"], result["iterations"]+2)
        self.assertLess(backend.last_host_polls, result["iterations"]//2)

    def test_graph_stops_at_exact_first_convergence_and_warm_start(self):
        a = sp.eye(519, format="csr")
        rhs = np.arange(519, dtype=float) / 512
        for steps in (1, 8, 32):
            result, _ = self.solve(a, rhs, steps=steps)
            self.assertTrue(result["converged"])
            self.assertEqual(result["iterations"], 1)
            self.assertEqual(result["matrix_vector_products"], 3)
            np.testing.assert_array_equal(result["solution"], rhs)
            warm, _ = self.solve(a, rhs, initial=rhs, steps=steps)
            self.assertEqual(warm["iterations"], 0)
            self.assertEqual(warm["matrix_vector_products"], 1)
            zero, _ = self.solve(a, np.zeros(519), steps=steps)
            self.assertTrue(zero["converged"])
            self.assertEqual(zero["matrix_vector_products"], 1)

    def test_graph_and_uncaptured_iterations_are_identical_including_budget_tail(self):
        a = sp.diags(np.linspace(1., 17., 527), format="csr")
        rhs = np.random.default_rng(31).normal(size=527)
        for budget in (1, 2, 7, 10, 1000):
            uncaptured, _ = self.solve(a, rhs, steps=1, budget=budget)
            for steps in (8, 32):
                captured, _ = self.solve(a, rhs, steps=steps, budget=budget)
                np.testing.assert_array_equal(captured.pop("solution"), uncaptured["solution"])
                self.assertEqual(captured, {k: v for k, v in uncaptured.items() if k != "solution"})

    def test_a_capture_invalidated_by_another_thread_is_recorded_again(self):
        cp, _ = require_cupy()
        a = sp.diags(np.linspace(1., 17., 527), format="csr")
        rhs = np.random.default_rng(37).normal(size=527)
        iterations, calls = CuPyPreparedConjugateGradientBackend._iterations, []

        def eigensolve(device):
            # Measured on an A100 with CuPy 14.2: the end of a thread that holds a cuSOLVER or a cuBLAS
            # handle invalidates an open capture on its device. The eigensolve gives this thread a handle;
            # by itself, in a thread that lives on, it leaves the capture valid.
            with cp.cuda.Device(device):
                cp.linalg.eigh(cp.eye(8))

        def disturbed(backend, count):
            calls.append(1)
            if len(calls) == 1:
                thread = Thread(target=eigensolve, args=(backend.device_id,))
                thread.start()
                thread.join()
            iterations(backend, count)

        expected, _ = self.solve(a, rhs)
        with patch.object(CuPyPreparedConjugateGradientBackend, "_iterations", disturbed):
            with self.assertWarnsRegex(RuntimeWarning, "is repeated"):
                backend = CuPyPreparedConjugateGradientBackend(a, graph_iterations=8)
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(backend.graph)
        self.assertFalse(backend.stream.is_capturing())
        actual = backend.solve(rhs, np.zeros_like(rhs), relative_tolerance=1e-12, absolute_tolerance=1e-14,
                               max_iterations=1000)
        np.testing.assert_array_equal(actual.pop("solution"), expected.pop("solution"))
        self.assertEqual(actual, expected)

    def test_a_failed_capture_is_ended_on_the_stream_of_the_backend(self):
        a = sp.eye(519, format="csr")
        iterations, built = CuPyPreparedConjugateGradientBackend._iterations, []

        def failing(backend, count):
            # One iteration is recorded before the capture fails.
            built.append(backend)
            iterations(backend, 1)
            raise RuntimeError("injected launch failure")

        enabled = gc.isenabled()
        with patch.object(CuPyPreparedConjugateGradientBackend, "_iterations", failing):
            with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
                CuPyPreparedConjugateGradientBackend(a)
        self.assertIsNone(built[0].graph)
        self.assertFalse(built[0].stream.is_capturing())
        self.assertEqual(gc.isenabled(), enabled)
        # A backend built afterwards captures its graph and solves with it.
        rhs = np.arange(519, dtype=float) / 512
        result, backend = self.solve(a, rhs)
        self.assertIsNotNone(backend.graph)
        self.assertTrue(result["converged"])
        np.testing.assert_array_equal(result["solution"], rhs)

    def test_breakdown_freezes_graph_and_does_not_produce_nan_solution(self):
        for a in (sp.csr_matrix((257, 257)), -sp.eye(257, format="csr")):
            result, _ = self.solve(a, np.ones(257))
            self.assertFalse(result["converged"])
            self.assertTrue(result["breakdown"])
            self.assertEqual(result["iterations"], 0)
            self.assertEqual(result["matrix_vector_products"], 3)
            np.testing.assert_array_equal(result["solution"], np.zeros(257))

    def test_existing_chronological_predictor_is_preserved(self):
        solver = CuPyPreparedPoissonSolver(sp.eye(19, format="csr"))
        origin = np.linspace(-.4, .7, 19)
        direction = np.linspace(.03, -.02, 19)
        with patch.dict(os.environ, {"PARSEC_HARTREE_CHRONOLOGICAL_GUESS": "1"}):
            for step in range(3):
                result = solver.solve(origin+step*direction, absolute_tolerance=1e-14)
        self.assertEqual(solver.chronological_prediction_calls, 1)
        self.assertEqual(result.iterations, 0)
        np.testing.assert_allclose(result.potential, origin+2*direction, atol=2e-16, rtol=0)

    def test_normalized_wedge_potential_and_energy_match_full_grid(self):
        grid = build_cluster_grid(GridSettings(spacing=.8, radius=3.3, expansion_order=4, shift=(0., 0., 0.)))
        a = build_negative_laplacian(grid)
        reduction = AxisReflectionReduction.detect(grid, (Atom("H", (0., 0., 0.)),))
        rhs = np.exp(-.3 * np.sum(grid.coordinates**2, axis=1))
        reference = spla.spsolve(a, rhs)
        reduced_rhs = reduction.reduce_vector(rhs)
        solver = SymmetryReducedPoissonSolver(a, reduction, solver_factory=CuPyPreparedPoissonSolver)
        result = solver.solve_reduced(reduced_rhs, return_wedge=True, relative_tolerance=1e-13, absolute_tolerance=1e-14)
        self.assertIsInstance(result.potential, SymmetryScalarField)
        full = result.potential.values[reduction.full_to_wedge]
        np.testing.assert_allclose(full, reference, rtol=2e-12, atol=2e-12)
        self.assertAlmostEqual(float(rhs@full), float(rhs@reference), delta=1e-10)
        warm = solver.solve_reduced(reduced_rhs, initial_potential=result.potential,
                                    return_wedge=True, relative_tolerance=1e-12, absolute_tolerance=1e-11)
        self.assertEqual(warm.iterations, 0)

    def test_reused_sector_stencil_solves_bitwise_like_its_repacked_csr(self):
        grid, a, reduction, metadata = totally_symmetric_sector()
        full_rhs = np.exp(-.3 * np.sum(grid.coordinates**2, axis=1))
        rhs = reduction.reduce_vector(full_rhs)
        # The backend built from the CSR packs it itself and is the reference.
        repacked = CuPyPreparedConjugateGradientBackend(metadata.to_csr())
        reused = CuPyPreparedConjugateGradientBackend(metadata)
        self.assertEqual(repacked.stencil_packing, "packed from CSR")
        self.assertEqual(reused.stencil_packing, "reused from the symmetry-sector stencil")
        self.assertIsNone(reused._operator)
        for name in ("shape", "n", "width", "blocks", "storage_mode", "coefficient_palette_size",
                     "device_bytes", "device_id", "graph_iterations"):
            with self.subTest(attribute=name):
                self.assertEqual(getattr(reused, name), getattr(repacked, name))
        cp = reused.cp
        neighbors = cp.asnumpy(reused.neighbors)
        np.testing.assert_array_equal(neighbors, cp.asnumpy(repacked.neighbors))
        self.assertEqual(reused.codes.dtype, repacked.codes.dtype)
        active = neighbors >= 0
        np.testing.assert_array_equal(
            cp.asnumpy(reused.palette)[cp.asnumpy(reused.codes)[active]].view(np.uint64),
            cp.asnumpy(repacked.palette)[cp.asnumpy(repacked.codes)[active]].view(np.uint64))
        controls = dict(relative_tolerance=1e-12, absolute_tolerance=1e-14)
        for initial in (np.zeros_like(rhs), .5*rhs):
            # Ten matvecs stop short after one graph group; the larger budget
            # converges after about eighteen iterations.
            for budget in (10, 1000):
                expected = repacked.solve(rhs, initial, max_iterations=budget, **controls)
                actual = reused.solve(rhs, initial, max_iterations=budget, **controls)
                np.testing.assert_array_equal(actual.pop("solution"), expected.pop("solution"))
                self.assertEqual(actual, expected)
        self.assertTrue(expected["converged"])
        self.assertGreater(expected["iterations"], reused.graph_iterations)
        # Its CSR is formed on request and is the canonical matrix.
        for name in ("indptr", "indices", "data"):
            np.testing.assert_array_equal(getattr(reused.operator, name), getattr(repacked.operator, name))
        self.assertIs(reused.operator, reused.operator)

        # Through the wedge adapter and the chronological predictor, as the
        # SCF calls it. The adapter given the CSR is the reference.
        from_csr = SymmetryReducedPoissonSolver(metadata.to_csr(), reduction, operator_is_reduced=True,
                                                solver_factory=CuPyPreparedPoissonSolver)
        current = SymmetryReducedPoissonSolver(metadata, reduction, operator_is_reduced=True,
                                               solver_factory=CuPyPreparedPoissonSolver)
        self.assertIs(current.solver.backend._stencil, metadata)
        for factor in (1., 1.03, 1.05):
            expected = from_csr.solve_reduced(factor*rhs, return_wedge=True, **controls)
            actual = current.solve_reduced(factor*rhs, return_wedge=True, **controls)
            np.testing.assert_array_equal(actual.potential.values, expected.potential.values)
            np.testing.assert_array_equal(actual.right_hand_side.values, expected.right_hand_side.values)
            for name in ("converged", "iterations", "matrix_vector_products", "residual_norm",
                         "initial_residual_norm", "tolerance", "breakdown"):
                with self.subTest(factor=factor, diagnostic=name):
                    self.assertEqual(getattr(actual, name), getattr(expected, name))
        self.assertEqual(current.solver.chronological_prediction_calls, 1)
        self.assertEqual(from_csr.solver.chronological_prediction_calls, 1)
        # And both are the solution of the full-grid problem.
        full = actual.potential.values[reduction.full_to_wedge]
        np.testing.assert_allclose(full, spla.spsolve(a.tocsc(), 1.05*full_rhs), rtol=2e-10, atol=2e-10)

    def test_cg_on_another_device_returns_the_same_bits(self):
        cp, _ = require_cupy()
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 2:
            self.skipTest("requires at least two allocated GPUs")
        home = int(cp.cuda.Device().id)
        other = max(device for device in range(device_count) if device != home)
        n = 773
        a = sp.diags((-np.ones(n-1), 2.1*np.ones(n), -np.ones(n-1)), (-1, 0, 1), format="csr")
        rhs = np.random.default_rng(23).normal(size=n)
        first = CuPyPreparedConjugateGradientBackend(a, device_id=home)
        moved = CuPyPreparedConjugateGradientBackend(a, device_id=other)
        self.assertEqual(moved.device_id, other)
        for array in (moved.neighbors, moved.codes, moved.palette, moved.x, moved.r, moved.p,
                      moved.ap, moved.b, moved.partial, moved.state):
            self.assertEqual(int(array.device.id), other)
        self.assertEqual(moved.device_bytes, first.device_bytes)
        controls = dict(relative_tolerance=1e-12, absolute_tolerance=1e-14)
        for budget in (10, 1000):
            expected = first.solve(rhs, np.zeros(n), max_iterations=budget, **controls)
            actual = moved.solve(rhs, np.zeros(n), max_iterations=budget, **controls)
            solution = expected.pop("solution")
            np.testing.assert_array_equal(actual.pop("solution"), solution)
            self.assertEqual(actual, expected)
        self.assertTrue(expected["converged"])
        self.assertGreater(expected["iterations"], moved.graph_iterations)
        # Device vectors of that device in and out, as the resident chain uses it.
        with cp.cuda.Device(other):
            resident = moved.solve(cp.asarray(rhs), cp.zeros(n), max_iterations=1000,
                                   device_result=True, **controls)
            resident_solution = resident.pop("solution")
            self.assertEqual(int(resident_solution.device.id), other)
            np.testing.assert_array_equal(cp.asnumpy(resident_solution), solution)
        self.assertEqual(resident, expected)
        self.assertEqual(int(cp.cuda.Device().id), home)
        np.testing.assert_allclose(solution, spla.spsolve(a.tocsc(), rhs), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
