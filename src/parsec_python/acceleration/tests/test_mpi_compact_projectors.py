"""Shared-projector communication must reproduce the complete KB operator."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import unittest

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.experimental.mpi_domain import generalized_ritz
from parsec_python.acceleration.experimental.mpi_domain_overlap import OverlappedDistributedHamiltonian
from parsec_python.acceleration.experimental.mpi_compact_projectors import CompactProjectorHamiltonian
from parsec_python.acceleration.tests import test_mpi_domain as base_tests


class _FIFORequest:
    """Model MPI's ordered same-tag message queue, including eager sends."""
    def __init__(self, world=None, key=None, buffer=None):
        self.world, self.key, self.buffer = world, key, buffer

    def Wait(self):
        if self.world is None:
            return
        with self.world.condition:
            ready = self.world.condition.wait_for(
                lambda: self.key in self.world.messages and len(self.world.messages[self.key]),
                timeout=10)
            if not ready:
                raise TimeoutError(f'missing FIFO message {self.key}')
            self.buffer[...] = self.world.messages[self.key].popleft()


class _FIFOComm(base_tests._Comm):
    def Irecv(self, buffer, source, tag):
        return _FIFORequest(self.world, (source, self.rank, tag), buffer)

    def Isend(self, buffer, dest, tag):
        with self.world.condition:
            self.world.messages.setdefault((self.rank, dest, tag), deque()).append(
                np.array(buffer, copy=True))
            self.world.condition.notify_all()
        return _FIFORequest()


class CompactProjectorTests(unittest.TestCase):
    def setUp(self):
        base_tests.DomainTests.setUp(self)
        rows = [1, 3, 15, 17, 31, 34, 2, 16, 4, 35, 20, 0, 14, 30]
        columns = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 6]
        values = [1., -.3, .2, -.9, 1.2, .4, -.7, .6, .8, -.2, 0., .3, -.1, -.4]
        self.b = sp.coo_matrix((values, (rows, columns)), shape=(self.n, 7)).tocsr()
        self.signs = np.array([-1., 1., -1., 1., -1., 1., -1.])

    def parallel(self, action, *, size=3, checks=False):
        world = base_tests._World(size)
        owner = np.minimum(np.arange(self.n)*size//self.n, size-1)

        def run(rank):
            comm = _FIFOComm(world, rank)
            rows = np.flatnonzero(owner == rank)[::-1]
            options = dict(comm=comm, local_rows=rows, collective_checks=checks,
                           validated_widths=(1, 6, 7))
            baseline = OverlappedDistributedHamiltonian(self.meta, self.v, self.b,
                          self.signs, owner, **options)
            compact = CompactProjectorHamiltonian(self.meta, self.v, self.b,
                          self.signs, owner, **options)
            return action(baseline, compact, comm)

        with ThreadPoolExecutor(max_workers=size) as pool:
            futures = [pool.submit(run, rank) for rank in range(size)]
            return [future.result(timeout=30) for future in futures]

    def test_signed_local_shared_and_explicit_zero_columns(self):
        for checks in (False, True):
            def action(baseline, compact, comm):
                self.assertEqual(compact.projector_partition, dict(
                    total_columns=7, shared_columns=3, wholly_local_columns_global=3,
                    wholly_local_columns_this_rank=1, zero_columns=1))
                np.testing.assert_array_equal(compact.shared_projector_columns, [3, 4, 6])
                for width in (1, 6, 7):
                    x = self.x[baseline.rows, :width]
                    expected = baseline.apply(x)
                    actual = compact.apply(x)
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(compact.stats['projector_reduce_input_bytes'], 3*(1+6+7)*8)
                self.assertEqual(compact.stats['projector_full_reduce_equivalent_bytes'], 7*(1+6+7)*8)
                self.assertEqual(compact.stats['projector_reduction_saved_input_bytes'], 4*(1+6+7)*8)
            self.parallel(action, checks=checks)

    def test_ritz_values_residuals_and_wavefunctions_unchanged(self):
        def action(baseline, compact, comm):
            x = self.x[baseline.rows, :6]
            expected = generalized_ritz(baseline, x)
            actual = generalized_ritz(compact, x)
            np.testing.assert_array_equal(actual.eigenvalues, expected.eigenvalues)
            np.testing.assert_array_equal(actual.local_vectors, expected.local_vectors)
            np.testing.assert_array_equal(actual.residual_norms, expected.residual_norms)
            np.testing.assert_array_equal(actual.overlap, expected.overlap)
            np.testing.assert_array_equal(actual.projected_hamiltonian, expected.projected_hamiltonian)
        self.parallel(action)

    def test_wholly_local_columns_remove_physical_allreduce(self):
        self.b = self.b[:, :3]
        self.signs = self.signs[:3]
        def action(baseline, compact, comm):
            self.assertEqual(compact.shared_projector_count, 0)
            x = self.x[baseline.rows, :7]
            expected = baseline.apply(x)
            before = comm.sequence
            actual = compact.apply(x)
            self.assertEqual(comm.sequence-before, 0)
            self.assertEqual(compact.stats['projector_reduce_input_bytes'], 0)
            np.testing.assert_array_equal(actual, expected)
        self.parallel(action)

    def test_no_projector_columns(self):
        self.b, self.signs = sp.csr_matrix((self.n, 0)), np.empty(0)
        def action(baseline, compact, comm):
            self.assertEqual(compact.projector_partition['total_columns'], 0)
            self.assertEqual(compact.stats['projector_reduce_input_bytes'], 0)
            x = self.x[baseline.rows, :6]
            np.testing.assert_array_equal(compact.apply(x), baseline.apply(x))
        self.parallel(action)

    def test_single_rank_never_reduces_projector_coefficients(self):
        def action(baseline, compact, comm):
            self.assertEqual(compact.shared_projector_count, 0)
            self.assertEqual(compact.local_projector_count, 6)
            x = self.x[baseline.rows, :6]
            np.testing.assert_array_equal(compact.apply(x), baseline.apply(x))
            self.assertEqual(compact.stats['projector_reduce_input_bytes'], 0)
        self.parallel(action, size=1)

    def test_optional_cupy_local_projectors(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest('no CUDA device')
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f'CuPy/CUDA unavailable: {exc}')
        owner = np.zeros(self.n, dtype=np.int32)
        options = dict(xp=cp, collective_checks=False, validated_widths=(1, 6, 7))
        baseline = OverlappedDistributedHamiltonian(self.meta, self.v, self.b,
                                                    self.signs, owner, **options)
        compact = CompactProjectorHamiltonian(self.meta, self.v, self.b,
                                              self.signs, owner, **options)
        for width in (1, 6, 7):
            x = cp.asarray(self.x[:, :width], order='F')
            expected = baseline.apply(x)
            actual = compact.apply(x)
            cp.cuda.get_current_stream().synchronize()
            np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected))
        self.assertEqual(compact.stats['projector_reduce_input_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
