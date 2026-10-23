"""The triangular Ritz hand-over (PARSEC_CUPY_RITZ_HANDOVER) in densities, forces, passes, solvers and SCF runs.

The small solve, the product and the Ritz step of one array are in ``test_ritz_handover``; this module checks
what reads a hand-over basis: for a fixed basis the density and the forces of its Ritz vectors to round-off,
later passes and the first solve, the deferred trim, the one place that forms Ritz vectors, a basis shared by
several devices (one GPU named two and four times) against one array, and complete SCF runs.
"""
import contextlib
import gc
import importlib
import os
import pickle
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy import cupy_available, require_cupy
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.Eigensolvers import handover
symmetry = importlib.import_module('parsec_python.acceleration.Eigensolvers.symmetry')
from parsec_python.acceleration.tests.test_ritz_handover import (
    _NAMES,
    _ROUTE,
    _clean,
    _device_count,
    _occupations,
    _operator,
    chebff,
    distributed_state,
    subspace,
)


def _orbitals(vectors, record, occupations, columns):
    """The lazy orbitals of one sector whose wedge is its grid, as the density builder is handed them."""
    from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetryOrbitals
    rows, count = int(vectors.shape[0]), len(columns)
    return CuPySymmetryOrbitals(
        scaled_wedge_vectors=None, representations=np.zeros(count, dtype=np.int32), full_to_wedge=np.arange(rows),
        device_full_to_wedge=None, phases=np.ones((1, rows)), full_size=rows,
        representation_columns=np.asarray(columns, dtype=np.int32), sector_vectors=(vectors,),
        sector_orbits=(np.arange(rows),), sector_scales=(np.ones(rows),), wedge_size=rows,
        sector_handovers=(record,))


def _handover_basis(rows, columns, seed):
    """An orthonormal basis Q, orthogonal eigenvectors Z and the Ritz vectors V = Q Z (host arrays)."""
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(rows, columns)))[0]
    z = np.linalg.qr(rng.normal(size=(columns, columns)))[0]
    return np.asfortranarray(q), np.asfortranarray(z), q @ z


class _NumpyAsCupy:
    """NumPy under the names that the density builder and the Ritz columns ask of CuPy."""

    cuda = SimpleNamespace(Device=lambda device: contextlib.nullcontext())
    asnumpy = staticmethod(np.asarray)

    def __getattr__(self, name):
        return getattr(np, name)


class HostDensityTests(unittest.TestCase):
    """The density of a hand-over basis with NumPy for CuPy: the identity, without a device."""

    def test_all_columns_less_the_corrections_are_the_density_of_the_ritz_vectors(self):
        from parsec_python.acceleration.backends import cupy as cupy_backend
        from parsec_python.acceleration.Occupations.device_density import CuPyDeviceDensityBuilder
        from parsec_python.acceleration.Occupations.symmetry_density import CuPySymmetryDensityBuilder
        shim = _NumpyAsCupy()
        q, z, v = _handover_basis(311, 23, 7)
        with _clean(), patch.object(cupy_backend, 'require_cupy', lambda: (shim, None)):
            builder = CuPySymmetryDensityBuilder(CuPyDeviceDensityBuilder(shim))
            for kind in ('fermi', 'frontier', 'holes', 'just_below_one', 'full', 'empty'):
                for count in (23, 17):
                    f = _occupations(kind, count)
                    record = handover.RitzHandover(z, 0, 23)
                    actual = builder(_orbitals(q, record, f, np.arange(count)), f, .37)
                    expected = builder(_orbitals(v, None, f, np.arange(count)), f, .37)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13*max(1., np.abs(expected).max()))
                    self.assertFalse(record.rotated or record.disable)
            self.assertEqual(builder.handover_record['calls'], 12)
            self.assertEqual(builder.handover_record['largest'], 23)


@unittest.skipUnless(_device_count() >= 1, 'a CUDA device required')
class DensityForceTests(unittest.TestCase):
    """For a fixed basis the hand-over gives the density and the forces of its Ritz vectors to round-off."""

    rows = 257

    def setUp(self):
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()

    def tearDown(self):
        gc.collect()

    def _builder(self):
        from parsec_python.acceleration.Occupations.device_density import CuPyDeviceDensityBuilder
        from parsec_python.acceleration.Occupations.symmetry_density import CuPySymmetryDensityBuilder
        return CuPySymmetryDensityBuilder(CuPyDeviceDensityBuilder(self.cp))

    def test_the_density_is_that_of_the_ritz_vectors_for_every_kind_of_occupation(self):
        cp = self.cp
        q, z, v = _handover_basis(self.rows, 31, 3)
        with _clean():
            builder = self._builder()
            for kind in ('fermi', 'frontier', 'holes', 'just_below_one', 'full', 'empty'):
                for count in (31, 25, 5):
                    f = _occupations(kind, count)
                    record = handover.RitzHandover(cp.asarray(z, order='F'), 0, 31)
                    basis = cp.array(q, order='F')
                    actual = builder(_orbitals(basis, record, f, np.arange(count)), f, .5)
                    expected = builder(_orbitals(cp.array(v, order='F'), None, f, np.arange(count)), f, .5)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13*max(1., np.abs(expected).max()))
                    # The basis is read, not written.
                    np.testing.assert_array_equal(cp.asnumpy(basis), q)
            # The exact rule: an occupation one ulp below one is corrected, one of one is not.
            f = _occupations('just_below_one', 31)
            builder(_orbitals(cp.array(q, order='F'), handover.RitzHandover(cp.asarray(z, order='F'), 0, 31), f,
                              np.arange(31)), f, .5)
            self.assertEqual(builder.handover_record['largest'], 31)
            before = builder.handover_record['largest']
            fresh = self._builder()
            fresh(_orbitals(cp.array(q, order='F'), handover.RitzHandover(cp.asarray(z, order='F'), 0, 31), f,
                            np.arange(31)), f, .5)
            self.assertEqual(fresh.handover_record['largest'], 1)
            fresh(_orbitals(cp.array(q, order='F'), handover.RitzHandover(cp.asarray(z, order='F'), 0, 31), np.ones(31),
                            np.arange(31)), np.ones(31), .5)
            self.assertEqual((fresh.handover_record['calls'], fresh.handover_record['largest'], before), (2, 1, 31))

    def test_the_correction_takes_pieces_of_rows_and_no_copy_of_the_basis(self):
        # Pieces of 40 rows of 257, the last one shorter: every piece is multiplied where the rows lie.  An
        # allocation as large as 40 rows of all columns would be a copy of the basis in pieces (on A100 it took
        # 22 GiB for 19,392 electrons on one device).
        from cupy.cuda import memory_hook
        cp = self.cp
        q, z, v = _handover_basis(self.rows, 31, 6)
        # Two empty Ritz vectors: a piece of them is far smaller than the same rows of the basis.
        f = np.ones(29)
        columns = np.arange(29)
        _full, correction, weights = handover.correction_set(f, columns, 31)
        self.assertEqual(list(correction), [29, 30])
        sizes = []

        class Sizes(memory_hook.MemoryHook):
            name = 'sizes'

            def malloc_preprocess(self, device_id, size, mem_size):
                sizes.append(int(size))
        with _clean():
            builder = self._builder().device_builder
            record = handover.RitzHandover(cp.asarray(z, order='F'), 0, 31)
            whole = handover.correction_density(builder, cp.array(q, order='F'), record, correction, weights, .5)
            basis = cp.array(q, order='F')
            with patch.object(handover, 'CORRECTION_BYTES', 8*correction.size*40), Sizes():
                pieces = handover.correction_density(builder, basis, record, correction, weights, .5)
            # A spilled sector goes up a piece at a time as well.
            with patch.object(handover, 'CORRECTION_BYTES', 8*(correction.size+31)*40):
                spilled = handover.correction_density(builder, np.asfortranarray(q), record, correction, weights, .5)
        expected = (2./.5)*np.sum((v[:, correction]**2)*weights[None, :], axis=1)
        for result in (whole, pieces, spilled):
            np.testing.assert_allclose(result, expected, rtol=0, atol=1e-13*np.abs(expected).max())
        # The last piece has 17 rows: no allocation holds even those rows of all columns.
        self.assertLess(max(sizes), 17*31*8)
        # A device array whose rows are not where a column-major matrix has them is refused, not copied.
        with self.assertRaisesRegex(ValueError, 'column-major'):
            handover.rows_times(cp, cp.array(q, order='C'), 0, 10, cp.asarray(z[:, :3], order='F'),
                                cp.empty((10, 3), order='F'))

    def test_more_corrections_than_the_limit_rotate_one_array_and_turn_the_hand_over_off(self):
        cp = self.cp
        q, z, v = _handover_basis(self.rows, 90, 4)
        f = _occupations('fermi', 10)
        with _clean():
            builder = self._builder()
            expected = builder(_orbitals(cp.array(v, order='F'), None, f, np.arange(10)), f, .5)
            # On its device: rotated in place into the Ritz vectors, which the density then reads as before.
            record = handover.RitzHandover(cp.asarray(z, order='F'), 0, 90)
            basis = cp.array(q, order='F')
            actual = builder(_orbitals(basis, record, f, np.arange(10)), f, .5)
            self.assertTrue(record.rotated and record.disable)
            np.testing.assert_allclose(cp.asnumpy(basis), v, rtol=0, atol=1e-13)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13*np.abs(expected).max())
            # On the host (a spilled sector): corrections in pieces of the limit, and the hand-over off.
            record = handover.RitzHandover(cp.asarray(z, order='F'), 0, 90)
            actual = builder(_orbitals(np.asfortranarray(q), record, f, np.arange(10)), f, .5)
            self.assertTrue(record.disable and not record.rotated)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12*np.abs(expected).max())
            self.assertEqual((builder.handover_record['fallbacks'], builder.handover_record['rotations']), (2, 1))

    def test_the_forces_of_all_columns_less_the_corrections_are_those_of_the_ritz_vectors(self):
        from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetrySCFEigensolver
        from parsec_python.acceleration.Forces.nonlocal_device import gradient_transposes, projector_force_terms
        cp = self.cp
        op = _operator(self.rows)
        rng = np.random.default_rng(12)
        transposes = gradient_transposes(tuple(sp.csr_matrix(rng.normal(size=(self.rows, 2))*.01) for _ in range(3)))
        q, z, v = _handover_basis(self.rows, 31, 5)
        owner = SimpleNamespace(_operators=[op])
        terms = CuPySymmetrySCFEigensolver._handover_force_terms
        for kind in ('fermi', 'holes', 'full'):
            f = _occupations(kind, 26)
            expected = projector_force_terms(op, cp.array(v, order='F'), f, transposes)
            for vectors in (cp.array(q, order='F'), np.asfortranarray(q)):
                record = handover.RitzHandover(cp.asarray(z, order='F'), 0, 31)
                with _clean():
                    actual = terms(owner, 0, vectors, record, f, transposes)
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14*np.abs(expected).max())
            # Twice on one basis: the same bits, nothing was written.
            basis = cp.array(q, order='F')
            record = handover.RitzHandover(cp.asarray(z, order='F'), 0, 31)
            with _clean():
                first = terms(owner, 0, basis, record, f, transposes)
                second = terms(owner, 0, basis, record, f, transposes)
            np.testing.assert_array_equal(first, second)


@unittest.skipUnless(_device_count() >= 1, 'a CUDA device required')
class PassTests(unittest.TestCase):
    """Later passes and the first solve under both values of the switch."""

    rows = 257

    def setUp(self):
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()
        self.op = _operator(self.rows)

    def tearDown(self):
        gc.collect()

    def _ritz_pairs(self, values, vectors, atol=1e-9):
        cp = self.cp
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(len(values)), atol=atol)
        applied = cp.asnumpy(self.op @ cp.array(vectors, order='F'))
        np.testing.assert_allclose(vectors.T @ applied, np.diag(values), rtol=0, atol=atol)

    def _exact(self, count):
        """The lowest ``count`` eigenvalues of the operator."""
        cp = self.cp
        dense = cp.asnumpy(self.op @ cp.eye(self.rows))
        return np.linalg.eigvalsh((dense+dense.T)/2)[:count]

    def test_three_later_passes_hand_over_and_follow_the_rotation(self):
        # The path differs from the second pass on: the lower degree of the next filter meets the span of the
        # filtered lower half, not that of the lowest Ritz vectors.  On this far from converged toy both still
        # converge, the hand-over within twice the error of the rotation.
        from parsec_python.Eigensolvers.subspace import SubspaceSettings
        cp = self.cp
        exact = self._exact(23)
        start = np.linalg.qr(np.random.default_rng(17).normal(size=(self.rows, 23)))[0]
        settings = SubspaceSettings(polynomial_degree=5, degree_delta=1)
        initial = subspace.DeviceSubspaceState(self.rows, 23, cp.linspace(.1, 1.5, 23), cp.array(start, order='F'))
        errors = []
        with _clean(**_ROUTE, PARSEC_CUPY_RITZ_HANDOVER_COLUMNS='4'):
            rotated, handed = initial, replace(initial, vectors=cp.array(start, order='F'))
            for number in range(3):
                a = subspace.run_subspace_filter(self.op, rotated, settings=settings, compute_residuals=False,
                                                 consume_state=True)
                pointer = handed.vectors.data.ptr
                b = subspace.run_subspace_filter(self.op, handed, settings=settings, compute_residuals=False,
                                                 consume_state=True, handover='triangular')
                self.assertIsNone(a.handover_pass)
                self.assertIsNone(a.state.handover)
                self.assertEqual((b.handover_pass['mode'], b.handover_pass['product'], b.vectors.data.ptr),
                                 ('triangular', 'columns', pointer))
                record = b.state.handover
                self.assertEqual((record.columns, record.pending_count, record.rotated), (23, None, False))
                if number == 0:
                    # One saved state: the same filtered columns and small solve, the values bit for bit.
                    np.testing.assert_array_equal(cp.asnumpy(b.eigenvalues), cp.asnumpy(a.eigenvalues))
                ritz_vectors = cp.asnumpy(b.vectors) @ cp.asnumpy(record.coefficients)
                self._ritz_pairs(cp.asnumpy(b.eigenvalues), ritz_vectors)
                errors.append([(cp.asnumpy(run.eigenvalues)-exact)[:17].max() for run in (a, b)])
                rotated, handed = a.state, b.state
        errors = np.array(errors)
        self.assertTrue(np.all(np.diff(errors, axis=0) < 0.), errors)
        self.assertTrue(np.all(errors[:, 1] <= 2.*errors[:, 0]), errors)

    def test_a_pending_trim_is_carried_out_by_the_next_pass_with_a_rotation(self):
        from parsec_python.Eigensolvers.subspace import SubspaceSettings
        cp = self.cp
        start = np.linalg.qr(np.random.default_rng(19).normal(size=(self.rows, 23)))[0]
        settings = SubspaceSettings(polynomial_degree=5, degree_delta=1)
        state = subspace.DeviceSubspaceState(self.rows, 23, cp.linspace(.1, 1.5, 23), cp.array(start, order='F'))
        with _clean(**_ROUTE):
            first = subspace.run_subspace_filter(self.op, state, settings=settings, compute_residuals=False,
                                                 consume_state=True, handover='triangular')
            first.state.handover.pending_count = 17
            second = subspace.run_subspace_filter(self.op, first.state, settings=settings, compute_residuals=False,
                                                  consume_state=True, handover='triangular')
        self.assertEqual((second.handover_pass['mode'], second.handover_pass['note'], second.handover_pass['deferred_trim']),
                         ('rotation', 'deferred trim', 17))
        self.assertIsNone(second.state.handover)
        self.assertEqual((second.state.working_states, second.vectors.shape), (17, (self.rows, 17)))
        self._ritz_pairs(cp.asnumpy(second.eigenvalues), cp.asnumpy(second.vectors))
        # A record rotated in place since holds Ritz vectors: the next pass takes them as such.
        third_state = replace(second.state, handover=handover.RitzHandover(None, 0, 17, rotated=True))
        with _clean(**_ROUTE):
            third = subspace.run_subspace_filter(self.op, third_state, settings=settings, compute_residuals=False,
                                                 consume_state=True, handover='triangular')
        self.assertEqual(third.handover_pass['mode'], 'triangular')

    def test_the_first_solve_hands_on_values_and_factors_and_ends_with_ritz_vectors(self):
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        cp = self.cp
        settings = ChebFFSettings(polynomial_degree=12, filter_cycles=4)
        with _clean(**_ROUTE):
            expected = chebff.run_chebff(self.op, 19, settings=settings)
            self.assertEqual(expected.handover_cycles, ())
            actual = chebff.run_chebff(self.op, 19, settings=settings, handover='triangular')
            with patch.dict(os.environ, PARSEC_CUPY_RITZ_HANDOVER_FIRST_CONDITION='0'):
                guarded = chebff.run_chebff(self.op, 19, settings=settings, handover='triangular')
            with patch.dict(os.environ, PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='0'):
                qr = chebff.run_chebff(self.op, 19, settings=settings, handover='triangular')
        self.assertEqual([item['mode'] for item in actual.handover_cycles], ['values', 'factor', 'factor', 'rotation'])
        self.assertEqual(actual.handover_cycles[-1]['note'], 'last cycle')
        self.assertEqual([item['mode'] for item in guarded.handover_cycles], ['factor']*3+['rotation'])
        self.assertEqual(guarded.handover_cycles[0]['note'], 'values not allowed')
        self.assertEqual([(item['mode'], item['note']) for item in qr.handover_cycles], [('rotation', 'QR cycles')]*4)
        # The QR cycles rotate as without the switch; the others reach the lowest values as well as the rotation.
        np.testing.assert_allclose(cp.asnumpy(qr.eigenvalues), cp.asnumpy(expected.eigenvalues), rtol=0, atol=1e-9)
        exact = self._exact(19)
        error = (cp.asnumpy(expected.eigenvalues)-exact)[:13].max()
        for run in (actual, guarded, qr):
            self.assertEqual([c.number for c in run.cycles], [1, 2, 3, 4])
            self.assertLessEqual((cp.asnumpy(run.eigenvalues)-exact)[:13].max(), 2.*error)
            self._ritz_pairs(cp.asnumpy(run.eigenvalues), cp.asnumpy(run.vectors))


@unittest.skipUnless(_device_count() >= 1, 'a CUDA device required')
class SolverTests(unittest.TestCase):
    """The eigensolver of a sector: its records, the deferred trim and the one place that forms Ritz vectors."""

    rows = 257

    def setUp(self):
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()
        self.op = _operator(self.rows)

    def tearDown(self):
        gc.collect()

    def _solver(self, capable=True, op=None):
        from parsec_python.acceleration.Eigensolvers.eigval import CuPyEigvalSolver
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        from parsec_python.Eigensolvers.eigval import EigvalSettings
        from parsec_python.Eigensolvers.subspace import SubspaceSettings
        settings = EigvalSettings(safety_buffer=0, chebff=ChebFFSettings(polynomial_degree=12, filter_cycles=3),
                                  subspace=SubspaceSettings(polynomial_degree=5, degree_delta=1))
        return CuPyEigvalSolver(self.op if op is None else op, settings=settings, retain_vectors_on_device=True,
                                compute_subspace_residuals=False, handover_capable=capable)

    def _ritz_pairs(self, values, vectors, atol=1e-9):
        cp = self.cp
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(len(values)), atol=atol)
        applied = cp.asnumpy(self.op @ cp.array(vectors, order='F'))
        np.testing.assert_allclose(vectors.T @ applied, np.diag(values), rtol=0, atol=atol)

    def test_a_trim_of_a_hand_over_waits_for_the_next_pass(self):
        cp = self.cp
        with _clean(**_ROUTE, PARSEC_CUPY_RITZ_HANDOVER='triangular'):
            solver = self._solver()
            first = solver.solve(23)
            self.assertEqual([step['mode'] for step in first.handover_step['steps']], ['values', 'factor', 'rotation'])
            self.assertFalse(first.handover_step['holds'])
            second = solver.solve(23)
            self.assertEqual((second.handover_step['path'], second.handover_step['holds']), ('subspace', True))
            self.assertEqual(second.handover_step['steps'][0]['mode'], 'triangular')
            record = second.handover
            self.assertIs(record, solver.device_state.subspace.handover)
            # The plain record travels by MPI; the record with its device array is stripped.
            pickle.loads(pickle.dumps(second.handover_step))
            self.assertIsNone(replace(second, vectors=None, state=None, handover=None).handover)
            # The trim launches nothing: the saved subspace keeps its columns and values.
            pointer = solver.device_state.subspace.vectors.data.ptr
            values = cp.asnumpy(solver.device_state.subspace.eigenvalues)
            solver.truncate_state(17)
            saved = solver.device_state
            self.assertEqual((saved.requested_states, saved.working_states, saved.subspace.working_states), (17, 17, 23))
            self.assertEqual((saved.subspace.vectors.data.ptr, record.pending_count), (pointer, 17))
            np.testing.assert_array_equal(cp.asnumpy(saved.subspace.eigenvalues), values)
            self.assertIsNone(solver._incompatibility(17, 17))
            self.assertEqual(solver._incompatibility(19, 19), 'requested_state_count_changed')
            self.assertEqual(solver.handover_record['deferred_trims'], 1)
            third = solver.solve(17)
        self.assertEqual(third.solver_path, 'subspace')
        self.assertFalse(third.restarted)
        step = third.handover_step['steps'][0]
        self.assertEqual((step['mode'], step['note'], step['deferred_trim']), ('rotation', 'deferred trim', 17))
        self.assertIsNone(third.handover)
        self.assertEqual(third.vectors.shape, (self.rows, 17))
        self._ritz_pairs(third.eigenvalues, cp.asnumpy(third.vectors))

    def test_ritz_vectors_are_formed_once_in_the_memory_that_the_result_views(self):
        cp = self.cp
        with _clean(**_ROUTE, PARSEC_CUPY_RITZ_HANDOVER='triangular'):
            solver = self._solver()
            solver.solve(23)
            result = solver.solve(23)
            q = cp.asnumpy(result.vectors)
            expected = q @ cp.asnumpy(result.handover.coefficients)
            self.assertTrue(solver.ensure_ritz_vectors())
            self.assertFalse(solver.ensure_ritz_vectors())
            self.assertTrue(result.handover.rotated)
            self.assertIsNone(solver.device_state.subspace.handover)
            np.testing.assert_allclose(cp.asnumpy(result.vectors), expected, rtol=0, atol=1e-13)
            self._ritz_pairs(result.eigenvalues, cp.asnumpy(result.vectors))
            self.assertEqual(solver.handover_record['ensure_rotations'], 1)
            # A trim pending at that moment is then the view of the leading Ritz vectors.
            solver.solve(23)
            solver.truncate_state(20)
            self.assertTrue(solver.ensure_ritz_vectors())
            self.assertEqual(solver.device_state.subspace.vectors.shape, (self.rows, 20))
            self.assertEqual(solver.device_state.subspace.working_states, 20)
            # A state spilled to the host is rotated through the device, into the same host array.
            solver.solve(20)
            state = solver.offload_state_to_host()
            host = state.subspace.vectors
            self.assertIsInstance(host, np.ndarray)
            expected = host @ cp.asnumpy(state.subspace.handover.coefficients)
            self.assertTrue(solver.ensure_ritz_vectors())
            self.assertTrue(np.shares_memory(solver.device_state.subspace.vectors, host))
            np.testing.assert_allclose(host, expected, rtol=0, atol=1e-13)
            self._ritz_pairs(cp.asnumpy(solver.device_state.subspace.eigenvalues), host)

    def test_a_solver_without_the_capability_and_the_default_keep_the_rotation(self):
        with _clean(**_ROUTE, PARSEC_CUPY_RITZ_HANDOVER='triangular'):
            solver = self._solver(capable=False)
            solver.solve(23)
            result = solver.solve(23)
        self.assertEqual((result.handover, result.handover_step), (None, None))
        with _clean(**_ROUTE):
            solver = self._solver()
            solver.solve(23)
            result = solver.solve(23)
        self.assertEqual((result.handover, result.handover_step), (None, None))
        self.assertFalse(solver.ensure_ritz_vectors())


@unittest.skipUnless(_device_count() >= 1, 'a CUDA device required')
class SharedRouteTests(unittest.TestCase):
    """A basis shared by two and four devices (one GPU named that often) against one array, under triangular."""

    rows, states = 257, 53

    def setUp(self):
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()
        self.op = _operator(self.rows)

    def tearDown(self):
        gc.collect()

    def _group(self, count):
        devices = (0,)*count
        self.op.distributed_filter_devices = devices
        return distributed_state.SectorDeviceGroup(self.op, devices)

    @staticmethod
    def _environment(**extra):
        # One GPU named several times has one stream per device id: the copies between "devices" are queued on it.
        return _clean(**_ROUTE, PARSEC_CUPY_EXCHANGE_CONCURRENT='0', PARSEC_CUPY_RITZ_HANDOVER_COLUMNS='8', **extra)

    def test_both_routes_hand_on_the_same_spans_and_densities_over_a_first_solve_and_three_passes(self):
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        from parsec_python.Eigensolvers.subspace import SubspaceSettings
        from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetrySCFEigensolver
        from parsec_python.acceleration.Forces.nonlocal_device import gradient_transposes
        from parsec_python.acceleration.Occupations.device_density import CuPyDeviceDensityBuilder
        from parsec_python.acceleration.Occupations.symmetry_density import CuPySymmetryDensityBuilder
        cp = self.cp
        settings = ChebFFSettings(polynomial_degree=12, filter_cycles=4)
        later = SubspaceSettings(polynomial_degree=5, degree_delta=1)
        rng = np.random.default_rng(12)
        transposes = gradient_transposes(tuple(sp.csr_matrix(rng.normal(size=(self.rows, 2))*.01) for _ in range(3)))
        owner = SimpleNamespace(_operators=[self.op])
        terms = CuPySymmetrySCFEigensolver._handover_force_terms
        with self._environment():
            builder = CuPySymmetryDensityBuilder(CuPyDeviceDensityBuilder(cp))
            reference = chebff.run_chebff(self.op, self.states, settings=settings, handover='triangular')
            for count in (2, 4):
                group = self._group(count)
                first = distributed_state.run_distributed_chebff(self.op, self.states, settings=settings, group=group,
                                                                 handover='triangular')
                self.assertEqual([item['mode'] for item in first.handover_cycles],
                                 [item['mode'] for item in reference.handover_cycles])
                np.testing.assert_allclose(cp.asnumpy(first.eigenvalues), cp.asnumpy(reference.eigenvalues), rtol=0,
                                           atol=5e-11)
                one = subspace.DeviceSubspaceState(self.rows, self.states, reference.eigenvalues,
                                                   reference.vectors.copy(order='F'))
                shared = subspace.DeviceSubspaceState(self.rows, self.states, first.eigenvalues, first.vectors)
                for number in range(3):
                    a = subspace.run_subspace_filter(self.op, one, settings=later, compute_residuals=False,
                                                     consume_state=True, handover='triangular')
                    b = distributed_state.run_distributed_subspace(self.op, shared, settings=later, group=group,
                                                                   handover='triangular')
                    np.testing.assert_allclose(cp.asnumpy(b.eigenvalues), cp.asnumpy(a.eigenvalues), rtol=0, atol=5e-11)
                    if number == 2:
                        # The pass that carried out the trim rotated on both routes.
                        self.assertEqual((a.handover_pass['note'], b.handover_pass['note']), ('deferred trim',)*2)
                        self.assertEqual(b.state.working_states, 47)
                        np.testing.assert_allclose(
                            np.abs(cp.asnumpy(a.vectors).T @ cp.asnumpy(group.gather(b.vectors))), np.eye(47), atol=5e-8)
                        break
                    qa, qb = cp.asnumpy(a.vectors), cp.asnumpy(group.gather(b.vectors))
                    za = cp.asnumpy(a.state.handover.coefficients)
                    # Z of a shared basis stays in the order of its factorization; the record names the row of every slot.
                    self.assertIsNotNone(b.state.handover.rows)
                    zb = cp.asnumpy(b.state.handover.coefficients)[np.asarray(b.state.handover.rows)]
                    np.testing.assert_allclose(qb.T @ qb, np.eye(self.states), atol=1e-12)
                    # The two spans that the next filter reads, the lower half and the whole, and the Ritz vectors.
                    halves = sum(block.stop-block.start for block in b.filter_blocks
                                 if block.degree == b.polynomial_degree_used-later.degree_delta)
                    for width in (halves, self.states):
                        np.testing.assert_allclose(np.linalg.svd(qa[:, :width].T @ qb[:, :width], compute_uv=False),
                                                   np.ones(width), atol=5e-9)
                    np.testing.assert_allclose(np.abs((qa @ za).T @ (qb @ zb)), np.eye(self.states), atol=5e-8)
                    self.assertEqual((b.handover_pass['mode'], b.state.handover.device), ('triangular', group.owner))
                    self.assertIsNotNone(b.handover_pass['band'])
                    # Ritz columns of the shared basis are those of the gathered one.
                    columns = np.array([0, 5, self.states-1])
                    np.testing.assert_allclose(
                        cp.asnumpy(b.vectors.ritz_columns(b.state.handover.coefficients, columns, rows=b.state.handover.rows)),
                        (qb @ zb)[:, columns], rtol=0, atol=1e-13)
                    # Densities and forces of the two routes, each from its own hand-over.
                    f = _occupations('fermi', self.states-4)
                    selected = np.arange(self.states-4)
                    dense_a = builder(_orbitals(a.vectors, a.state.handover, f, selected), f, .5)
                    dense_b = builder(_orbitals(b.vectors, b.state.handover, f, selected), f, .5)
                    np.testing.assert_allclose(dense_b, dense_a, rtol=0, atol=1e-7*np.abs(dense_a).max())
                    # In pieces of a few rows and columns, the same to round-off.
                    with patch.object(handover, 'CORRECTION_BYTES', 8*5*40), patch.object(symmetry, 'CORRECTION_BYTES', 8*2*self.rows):
                        pieces_a = builder(_orbitals(a.vectors, a.state.handover, f, selected), f, .5)
                        pieces_b = builder(_orbitals(b.vectors, b.state.handover, f, selected), f, .5)
                        pieced_forces_b = terms(owner, 0, b.vectors, b.state.handover, f, transposes)
                    np.testing.assert_allclose(pieces_a, dense_a, rtol=0, atol=1e-13*np.abs(dense_a).max())
                    np.testing.assert_allclose(pieces_b, dense_b, rtol=0, atol=1e-13*np.abs(dense_b).max())
                    forces_a = terms(owner, 0, a.vectors, a.state.handover, f, transposes)
                    forces_b = terms(owner, 0, b.vectors, b.state.handover, f, transposes)
                    np.testing.assert_allclose(forces_b, forces_a, rtol=1e-6, atol=1e-12*np.abs(forces_a).max())
                    np.testing.assert_allclose(pieced_forces_b, forces_b, rtol=1e-12, atol=1e-14*np.abs(forces_b).max())
                    records = a.state.handover, b.state.handover
                    if number == 1:
                        # A trim of both: the next pass carries it out.
                        a.state.handover.pending_count = b.state.handover.pending_count = 47
                    one, shared = a.state, b.state
                    if number:
                        # The eigenvectors of the hand-over before went when this pass started.
                        self.assertEqual([item.coefficients for item in previous], [None, None])
                    previous = records
                self.assertGreater(group.seconds['handover'], 0.)
                self.assertGreater(group.seconds['correction'], 0.)
                self.assertEqual(group.handover_counts.get('rotation (deferred trim)'), 1)

    def test_values_return_the_filtered_columns_and_the_rotation_is_todays_step(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import subspace_filter_blocks
        cp = self.cp
        rng = np.random.default_rng(23)
        host = np.linalg.qr(rng.normal(size=(self.rows, self.states)))[0] @ \
            (np.eye(self.states)+.2*rng.normal(size=(self.states, self.states)))
        with self._environment():
            for count in (2, 4):
                group = self._group(count)
                layout = group.column_layout(subspace_filter_blocks(self.states, 6, 5, 1))

                def scattered():
                    return group.scatter(cp.array(host, order='F'), layout)
                values, kept, _whitened, step = group.ritz_handover(self.op, scattered(), stages=False, handover='values',
                                                                    values_condition_limit=1e12)
                self.assertEqual(step.mode, 'values')
                np.testing.assert_array_equal(cp.asnumpy(group.gather(kept)), host)
                rotated = group.ritz(self.op, scattered(), stages=False)
                asked = group.ritz_handover(self.op, scattered(), stages=False, handover='rotation')
                np.testing.assert_array_equal(asked[0], rotated[0])
                np.testing.assert_array_equal(cp.asnumpy(group.gather(asked[1])), cp.asnumpy(group.gather(rotated[1])))
                self.assertEqual(asked[3].mode, 'rotation')
                np.testing.assert_allclose(values, rotated[0], rtol=0, atol=1e-11)
                # Two tall blocks rotate and say so.
                with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='2'):
                    two = group.ritz_handover(self.op, scattered(), stages=False, handover='triangular', halves=24)
                self.assertEqual((two[3].mode, two[3].note), ('rotation', '2 tall blocks'))
                np.testing.assert_allclose(two[0], rotated[0], rtol=0, atol=1e-11)

    def test_the_hand_over_takes_the_same_bits_under_every_pipeline_and_exchange(self):
        # PARSEC_CUPY_RITZ_PIPELINE and PARSEC_CUPY_EXCHANGE_STRIDED move no bit of the rounds; the hand-over,
        # which changes only the order of the factorization and what follows the rounds, keeps that.
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        from parsec_python.Eigensolvers.subspace import SubspaceSettings
        cp = self.cp
        settings = ChebFFSettings(polynomial_degree=12, filter_cycles=4)
        later = SubspaceSettings(polynomial_degree=5, degree_delta=1)
        for count in (2, 4):
            runs = {}
            for pipeline, strided in (('off', '0'), ('p1', '0'), ('p2', '0'), ('off', '1'), ('p2', '1')):
                with _clean(**_ROUTE, PARSEC_CUPY_RITZ_HANDOVER_COLUMNS='8', PARSEC_CUPY_RITZ_PIPELINE=pipeline,
                            PARSEC_CUPY_EXCHANGE_STRIDED=strided):
                    group = self._group(count)
                    first = distributed_state.run_distributed_chebff(self.op, self.states, settings=settings,
                                                                     group=group, handover='triangular')
                    state = subspace.DeviceSubspaceState(self.rows, self.states, first.eigenvalues, first.vectors)
                    values = [cp.asnumpy(first.eigenvalues)]
                    for _ in range(2):
                        result = distributed_state.run_distributed_subspace(self.op, state, settings=later, group=group,
                                                                            handover='triangular')
                        values.append(cp.asnumpy(result.eigenvalues))
                        state = result.state
                    self.assertEqual(group.pipeline, pipeline)
                    runs[pipeline, strided] = (values, cp.asnumpy(group.gather(state.vectors)),
                                               cp.asnumpy(state.handover.coefficients), np.asarray(state.handover.rows))
            reference = runs['off', '0']
            for key, run in runs.items():
                for mine, theirs in zip(run[0], reference[0]):
                    np.testing.assert_array_equal(mine, theirs, err_msg=str((count, key)))
                for mine, theirs in zip(run[1:], reference[1:]):
                    np.testing.assert_array_equal(mine, theirs, err_msg=str((count, key)))

    def test_a_shared_basis_forms_no_ritz_vectors_outside_a_pass(self):
        from parsec_python.acceleration.Eigensolvers.eigval import CuPyEigvalSolver
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        from parsec_python.Eigensolvers.eigval import EigvalSettings
        from parsec_python.Eigensolvers.subspace import SubspaceSettings
        settings = EigvalSettings(safety_buffer=0, chebff=ChebFFSettings(polynomial_degree=12, filter_cycles=3),
                                  subspace=SubspaceSettings(polynomial_degree=5, degree_delta=1))
        with self._environment(PARSEC_CUPY_RITZ_HANDOVER='triangular', PARSEC_CUPY_DISTRIBUTED_STATE='1'):
            self.op.distributed_filter_devices = (0, 0)
            solver = CuPyEigvalSolver(self.op, settings=settings, retain_vectors_on_device=True,
                                      compute_subspace_residuals=False, handover_capable=True)
            first = solver.solve(self.states)
            self.assertTrue(hasattr(first.vectors, 'group'))
            second = solver.solve(self.states)
            self.assertTrue(second.handover_step['holds'])
            with self.assertRaisesRegex(RuntimeError, 'PARSEC_CUPY_RITZ_HANDOVER=rotation'):
                solver.ensure_ritz_vectors()


@unittest.skipUnless(cupy_available() and native_available(), 'a real CUDA device and the native extension are required')
class SCFTests(unittest.TestCase):
    """Complete SCF runs on one device under both values of the switch."""

    def tearDown(self):
        gc.collect()

    def run_scf(self, source, steps=None, **environment):
        from parsec_python.Input import parse_parsec_input
        from parsec_python.acceleration.driver import prepare_single_point, run_scf
        problem = parse_parsec_input(Path(source)).problem
        if steps is not None:
            problem = replace(problem, scf=replace(problem.scf, max_iterations=steps))
        names = _NAMES+('PARSEC_CUPY_SECTOR_STATE_STORAGE', 'PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR')
        with patch.dict(os.environ, PARSEC_CUPY_DEVICES='0'):
            for name in names:
                os.environ.pop(name, None)
            os.environ.update(environment)
            system = prepare_single_point(problem, backend='auto')
            result = run_scf(system)
        self.details = dict(result.backend.details)
        self.statistics = result.backend_statistics
        return result.result

    def assert_same_bits(self, actual, expected):
        self.assertEqual(actual.iterations, expected.iterations)
        np.testing.assert_array_equal(actual.eigenvalues, expected.eigenvalues)
        np.testing.assert_array_equal(actual.occupations, expected.occupations)
        np.testing.assert_array_equal(actual.density, expected.density)
        np.testing.assert_array_equal(actual.hartree_potential, expected.hartree_potential)
        self.assertEqual(actual.energies.total, expected.energies.total)
        for step, reference in zip(actual.history, expected.history, strict=True):
            np.testing.assert_array_equal(step.eigenvalues, reference.eigenvalues)

    @staticmethod
    def _smoke():
        return Path(__file__).resolve().parents[2]/'tests'/'data'/'H_cli_smoke.in'

    def test_unset_is_the_rotation_bit_for_bit_and_the_hand_over_moves_the_path_only(self):
        route = dict(PARSEC_CUPY_GENERALIZED_RITZ='on', PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='1')
        unset = self.run_scf(self._smoke(), 4, **route)
        self.assertEqual(self.details['orbital_sector_ritz_handover'], 'rotation')
        self.assertEqual(self.statistics.subspace_ritz_handover_seconds, 0.)
        named = self.run_scf(self._smoke(), 4, PARSEC_CUPY_RITZ_HANDOVER='rotation', **route)
        self.assert_same_bits(unset, named)
        np.testing.assert_array_equal(unset.wavefunctions, named.wavefunctions)
        triangular = self.run_scf(self._smoke(), 4, PARSEC_CUPY_RITZ_HANDOVER='triangular', **route)
        detail = self.details['orbital_sector_ritz_handover']
        self.assertTrue(detail.startswith('triangular: first solve '), detail)
        self.assertIn('triangular', detail.split('later passes')[1])
        self.assertEqual(triangular.iterations, unset.iterations)
        self.assertLess(abs(triangular.energies.total-unset.energies.total), 1e-6)
        # The orbitals that the API exports are the Ritz vectors, which ensure_ritz_vectors formed.
        self.assertIn('Ritz vectors formed', detail)
        self.assertIsInstance(triangular.wavefunctions, np.ndarray)
        # States spilled to the host give the same bits as states kept on the device.
        spilled = self.run_scf(self._smoke(), 4, PARSEC_CUPY_RITZ_HANDOVER='triangular',
                               PARSEC_CUPY_SECTOR_STATE_STORAGE='host', PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR='pool', **route)
        kept = self.run_scf(self._smoke(), 4, PARSEC_CUPY_RITZ_HANDOVER='triangular',
                            PARSEC_CUPY_SECTOR_STATE_STORAGE='device', PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR='pool', **route)
        self.assert_same_bits(spilled, kept)
        np.testing.assert_array_equal(spilled.wavefunctions, kept.wavefunctions)

    def test_a_small_cluster_takes_the_steps_and_the_energy_of_the_rotation(self):
        source = Path(__file__).resolve().parents[4]/'examples'/'0d_benzene'/'parsec.in'
        if not source.exists():
            self.skipTest('examples/0d_benzene is not in this tree')
        profile = dict(PARSEC_CUPY_GENERALIZED_RITZ='on', PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='1',
                       PARSEC_CUPY_FILTER_GRAPHS='1', PARSEC_CUPY_FILTER_COLUMN_MAJOR='1', PARSEC_CUPY_DISTRIBUTED_FILTER='1',
                       PARSEC_CUPY_DEVICE_RANDOM='1', PARSEC_CUPY_RITZ_CONDITION='symmetric',
                       PARSEC_CUPY_RITZ_EIGH_BACKEND='cupy', PARSEC_CUPY_RITZ_ROTATION='reuse', PARSEC_CUPY_RECYCLE_STATE='1',
                       PARSEC_CUPY_STAGE_TIMING='1')
        rotation = self.run_scf(source, **profile)
        triangular = self.run_scf(source, PARSEC_CUPY_RITZ_HANDOVER='triangular', **profile)
        self.assertGreater(self.statistics.subspace_ritz_handover_seconds, 0.)
        self.assertTrue(rotation.converged and triangular.converged)
        self.assertEqual(triangular.iterations, rotation.iterations)
        self.assertLess(abs(triangular.energies.total-rotation.energies.total), 1e-6)
        for mine, theirs in zip(triangular.history, rotation.history):
            self.assertLess(abs(np.log(mine.weighted_residual/theirs.weighted_residual)), np.log(1.5))


if __name__ == '__main__':
    unittest.main()
