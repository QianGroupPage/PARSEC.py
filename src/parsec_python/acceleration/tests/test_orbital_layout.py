"""MPI-independent tests of actual Alltoallv layout packing/order."""
from concurrent.futures import ThreadPoolExecutor
import unittest

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy_stencil_major import build_stencil_major_metadata
from parsec_python.acceleration.experimental.mpi_domain import CollectiveDomainError
from parsec_python.acceleration.experimental.orbital_layout import (
    ColumnLayout, balanced_ranges, column_chebyshev_filter)
from parsec_python.acceleration.tests import test_mpi_domain as base_tests


class _LayoutComm(base_tests._Comm):
    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.exchange_calls = 0

    def allgather(self, value):
        if isinstance(value, np.ndarray):
            raise AssertionError('whole orbital arrays must not be allgathered')
        return super().allgather(value)

    def Alltoallv(self, send, receive):
        data, counts, offsets, _ = send
        target, receive_counts, receive_offsets, _ = receive
        if data.ndim != 1 or target.ndim != 1 or not data.flags.c_contiguous or not target.flags.c_contiguous:
            raise AssertionError('Alltoallv buffers must be contiguous vectors')
        packets = [data[int(offset):int(offset)+int(count)].copy()
                   for count, offset in zip(counts, offsets)]
        incoming = self._call('alltoall', packets)
        for source, packet in enumerate(incoming):
            count, offset = int(receive_counts[source]), int(receive_offsets[source])
            if len(packet) != count:
                raise AssertionError('sender/receiver Alltoallv counts disagree')
            target[offset:offset+count] = packet
        self.exchange_calls += 1


class OrbitalLayoutTests(unittest.TestCase):
    def setUp(self):
        base_tests.DomainTests.setUp(self)

    def parallel(self, global_states, action, size=3):
        world = base_tests._World(size)
        def run(rank):
            comm = _LayoutComm(world, rank)
            layout = ColumnLayout(self.meta, self.v, self.b, self.signs,
                                  global_states, comm=comm)
            return action(layout, comm)
        with ThreadPoolExecutor(max_workers=size) as pool:
            futures = [pool.submit(run, rank) for rank in range(size)]
            return [f.result(timeout=30) for f in futures]

    def test_balanced_ranges_allow_empty_tails(self):
        counts, offsets = balanced_ranges(11, 3)
        np.testing.assert_array_equal(counts, [4, 4, 3])
        np.testing.assert_array_equal(offsets, [0, 4, 8, 11])
        counts, offsets = balanced_ranges(2, 4)
        np.testing.assert_array_equal(counts, [1, 1, 0, 0])
        np.testing.assert_array_equal(offsets, [0, 1, 2, 2, 2])

    def test_uneven_roundtrip_preserves_every_global_index(self):
        matrix = np.arange(self.n*11, dtype=np.float64).reshape(self.n, 11)
        def action(layout, comm):
            columns = matrix[:, layout.column_start:layout.column_stop]
            row_output = np.empty(layout.row_shape, dtype=np.float64, order='F')
            rows = layout.columns_to_rows(columns, out=row_output)
            self.assertIs(rows, row_output)
            np.testing.assert_array_equal(rows, matrix[layout.row_start:layout.row_stop])
            column_output = np.empty(layout.column_shape, dtype=np.float64, order='F')
            restored = layout.rows_to_columns(rows, out=column_output)
            self.assertIs(restored, column_output)
            np.testing.assert_array_equal(restored, columns)
            self.assertEqual(comm.exchange_calls, 2)
            self.assertTrue(rows.flags.f_contiguous and restored.flags.f_contiguous)
        self.parallel(11, action)

    def test_empty_column_ranks_and_empty_row_ranks(self):
        for rows, columns, size in ((37, 2, 4), (3, 7, 5)):
            self.n = rows
            self.t = sp.diags((-np.ones(rows-1), 4*np.ones(rows), -np.ones(rows-1)),
                              (-1, 0, 1), shape=(rows, rows), format='csr')
            self.meta = build_stencil_major_metadata(self.t)
            self.v, self.b, self.signs = np.arange(rows)*.01, sp.csr_matrix((rows, 0)), np.empty(0)
            matrix = np.arange(rows*columns, dtype=np.float64).reshape(rows, columns)
            def action(layout, comm):
                local = matrix[:, layout.column_start:layout.column_stop]
                restored = layout.rows_to_columns(layout.columns_to_rows(local))
                np.testing.assert_array_equal(restored, local)
                np.testing.assert_allclose(layout.apply(local), self.t @ local+self.v[:, None]*local)
            self.parallel(columns, action, size=size)

    def test_column_h_has_no_mpi_and_matches_full_operator(self):
        matrix = self.x[:, :7]
        def action(layout, comm):
            local = matrix[:, layout.column_start:layout.column_stop]
            before = comm.sequence
            applied = layout.apply(local)
            self.assertEqual(comm.sequence, before)
            np.testing.assert_allclose(applied, self.h @ local, atol=3e-14, rtol=3e-14)
            rows = layout.columns_to_rows(applied)
            np.testing.assert_allclose(rows, (self.h @ matrix)[layout.row_start:layout.row_stop],
                                       atol=3e-14, rtol=3e-14)
        self.parallel(7, action)

    def test_uniform_filter_plus_two_redistributions_matches_reference(self):
        matrix = self.x[:, :7]
        lower, upper, reference, degree, initial = -4., 35., -6., 10, -.23
        half, center = .5*(upper-lower), .5*(upper+lower)
        s1 = half/(reference-center)
        previous, current, sigma = matrix.copy(), (self.h @ matrix-center*matrix)*(s1/half), initial
        for _ in range(2, degree+1):
            sn = 1/(2/s1-sigma)
            previous, current = current, sn*((2/half)*(self.h @ current-center*current)-sigma*previous)
            sigma = sn
        def action(layout, comm):
            local = matrix[:, layout.column_start:layout.column_stop]
            before = comm.sequence
            filtered, final_sigma = column_chebyshev_filter(layout, local, degree=degree,
                lower_bound=lower, upper_bound=upper, reference_eigenvalue=reference,
                initial_sigma=initial, return_final_sigma=True)
            self.assertEqual(comm.sequence-before, 2)  # Once-only validation; no recurrence MPI.
            self.assertEqual(final_sigma, sigma)
            rows = layout.columns_to_rows(filtered)
            np.testing.assert_allclose(rows, current[layout.row_start:layout.row_stop],
                                       atol=5e-13, rtol=5e-13)
            restored = layout.rows_to_columns(rows)
            np.testing.assert_array_equal(restored, filtered)
            self.assertEqual(layout.stats['local_h_applications'], degree)
        self.parallel(7, action)

    def test_bad_local_redistribution_shape_fails_collectively(self):
        def action(layout, comm):
            local = self.x[:, layout.column_start:layout.column_stop]
            if comm.rank == 1:
                local = local[:-1]
            with self.assertRaisesRegex(CollectiveDomainError, 'rank 1'):
                layout.columns_to_rows(local)
        self.parallel(7, action)

    def test_serial_layout_reference(self):
        layout = ColumnLayout(self.meta, self.v, self.b, self.signs, 7)
        matrix = np.asfortranarray(self.x[:, :7])
        np.testing.assert_array_equal(layout.rows_to_columns(layout.columns_to_rows(matrix)), matrix)
        self.assertEqual(layout.stats['redistribution_off_rank_send_bytes'], 0)

    def test_single_rank_identity_alias_copy_and_no_communication(self):
        layout = ColumnLayout(self.meta, self.v, self.b, self.signs, 7)
        matrix = np.asfortranarray(self.x[:, :7])
        def forbidden(*args, **kwargs):
            self.fail('single-rank identity must not invoke MPI')
        layout.comm.Alltoallv = forbidden
        layout.comm.allgather = forbidden
        for conversion in (layout.columns_to_rows, layout.rows_to_columns):
            self.assertIs(conversion(matrix), matrix)
            target = np.empty_like(matrix, order='F')
            self.assertIs(conversion(matrix, out=target), target)
            np.testing.assert_array_equal(target, matrix)
            target[0, 0] += 1
            self.assertNotEqual(target[0, 0], matrix[0, 0])
            c_order = np.ascontiguousarray(matrix)
            converted = conversion(c_order)
            self.assertFalse(np.shares_memory(converted, c_order))
            self.assertTrue(converted.flags.f_contiguous)
            np.testing.assert_array_equal(converted, matrix)
            single_precision = matrix.astype(np.float32, order='F')
            converted = conversion(single_precision)
            self.assertEqual(converted.dtype, np.dtype(np.float64))
            self.assertTrue(converted.flags.f_contiguous)
            self.assertFalse(np.shares_memory(converted, single_precision))
            np.testing.assert_array_equal(converted, single_precision.astype(np.float64))
            with self.assertRaises(ValueError):
                conversion(matrix, out=np.empty_like(matrix, order='C'))
        self.assertEqual(layout.stats['redistribution_send_bytes'], 0)
        self.assertEqual(layout.stats['redistribution_receive_bytes'], 0)

    def test_optional_cupy_h_and_serial_redistribution(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest('no CUDA device')
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f'CuPy/CUDA unavailable: {exc}')
        for transport in ('host', 'cuda'):
            layout = ColumnLayout(self.meta, self.v, self.b, self.signs, 7,
                                  xp=cp, transport=transport)
            local = cp.asarray(self.x[:, :7], order='F')
            np.testing.assert_allclose(cp.asnumpy(layout.apply(local)), self.h @ self.x[:, :7],
                                       atol=5e-13, rtol=5e-13)
            restored = layout.rows_to_columns(layout.columns_to_rows(local))
            np.testing.assert_array_equal(cp.asnumpy(restored), self.x[:, :7])


if __name__ == '__main__':
    unittest.main()
