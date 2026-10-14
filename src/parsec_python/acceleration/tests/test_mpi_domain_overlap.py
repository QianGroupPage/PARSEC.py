"""Optional overlap must preserve the base distributed operator exactly."""
from concurrent.futures import ThreadPoolExecutor
import unittest

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.experimental.mpi_domain import (
    DistributedHamiltonian, generalized_ritz, CollectiveDomainError)
from parsec_python.acceleration.experimental.mpi_domain_overlap import (
    OverlappedDistributedHamiltonian)
from parsec_python.acceleration.tests import test_mpi_domain as base_tests

_Comm, _World = base_tests._Comm, base_tests._World


class OverlapTests(unittest.TestCase):
    def setUp(self):
        base_tests.DomainTests.setUp(self)

    def parallel(self, action, *, size=3, round_robin=False, **options):
        world = _World(size)
        owner = (np.arange(self.n) % size if round_robin else
                 np.minimum(np.arange(self.n)*size//self.n, size-1))

        def run(rank):
            comm = _Comm(world, rank)
            rows = np.flatnonzero(owner == rank)[::-1]
            base = DistributedHamiltonian(self.meta, self.v, self.b, self.signs,
                                          owner, comm=comm, local_rows=rows)
            candidate = OverlappedDistributedHamiltonian(
                self.meta, self.v, self.b, self.signs, owner, comm=comm,
                local_rows=rows, **options)
            return action(base, candidate, comm)

        with ThreadPoolExecutor(max_workers=size) as pool:
            futures = [pool.submit(run, r) for r in range(size)]
            return [f.result(timeout=30) for f in futures]

    def test_overlap_and_fast_switches_preserve_each_operator_value(self):
        for overlap in (False, True):
            for checks in (False, True):
                def action(base, candidate, comm):
                    for width in (1, 6, 7):
                        x = self.x[base.rows, :width]
                        np.testing.assert_array_equal(candidate.apply(x), base.apply(x))
                    return candidate.stats
                stats = self.parallel(action, overlap=overlap,
                    collective_checks=checks, validated_widths=(1, 6, 7))
                self.assertTrue(all(s['interior_rows'] > 0 for s in stats))
                self.assertTrue(all(s['boundary_rows'] > 0 for s in stats))

    def test_only_boundary_rows_and_no_projectors(self):
        self.b, self.signs = sp.csr_matrix((self.n, 0)), np.empty(0)
        def action(base, candidate, comm):
            x = self.x[base.rows, :7]
            np.testing.assert_array_equal(candidate.apply(x), base.apply(x))
            self.assertEqual(candidate.stats['interior_rows'], 0)
        self.parallel(action, round_robin=True, collective_checks=False,
                      validated_widths=(7,))

    def test_serial_no_halo_and_vector(self):
        def action(base, candidate, comm):
            x = self.x[base.rows, 0]
            np.testing.assert_array_equal(candidate.apply(x), base.apply(x))
            self.assertEqual(candidate.stats['boundary_rows'], 0)
            self.assertEqual(candidate.stats['halo_send_bytes'], 0)
        self.parallel(action, size=1, collective_checks=False)

    def test_fast_path_eliminates_guard_collectives(self):
        def action(base, candidate, comm):
            x = self.x[base.rows, :6]
            before = comm.sequence
            expected = base.apply(x)
            base_calls = comm.sequence-before
            before = comm.sequence
            actual = candidate.apply(x)
            fast_calls = comm.sequence-before
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(fast_calls, 1)  # The physical KB coefficient sum only.
            self.assertGreater(base_calls, fast_calls)
            return base_calls, fast_calls
        self.parallel(action, collective_checks=False, validated_widths=(6,))

    def test_interior_is_dispatched_before_wait_then_boundary(self):
        def action(base, candidate, comm):
            events = []
            original_rows = candidate._apply_rows
            original_recv = comm.Irecv
            def rows(*args):
                events.append('interior' if args[3] is candidate.interior_rows else 'boundary')
                return original_rows(*args)
            def receive(*args, **kwargs):
                request = original_recv(*args, **kwargs)
                original_wait = request.Wait
                def wait():
                    events.append('wait')
                    return original_wait()
                request.Wait = wait
                return request
            candidate._apply_rows = rows
            comm.Irecv = receive
            candidate.apply(self.x[base.rows, :6])
            self.assertLess(events.index('interior'), events.index('wait'))
            self.assertLess(events.index('wait'), events.index('boundary'))
        self.parallel(action, collective_checks=False, validated_widths=(6,))

    def test_registering_new_width_is_explicit_and_collective(self):
        def action(base, candidate, comm):
            candidate.apply(self.x[base.rows, :1])  # First-width validation.
            with self.assertRaisesRegex(ValueError, 'was not validated'):
                candidate.apply(self.x[base.rows, :6])
            candidate.validate_width(6)
            np.testing.assert_array_equal(candidate.apply(self.x[base.rows, :6]),
                                          base.apply(self.x[base.rows, :6]))
        self.parallel(action, collective_checks=False)

    def test_fast_ritz_retains_physics_and_safety_audits(self):
        def action(base, candidate, comm):
            x = self.x[base.rows, :6]
            reference = generalized_ritz(base, x)
            actual = generalized_ritz(candidate, x)
            np.testing.assert_array_equal(actual.eigenvalues, reference.eigenvalues)
            np.testing.assert_array_equal(actual.local_vectors, reference.local_vectors)
            np.testing.assert_array_equal(actual.residual_norms, reference.residual_norms)
            singular = np.repeat(x[:, :1], 6, axis=1)
            with self.assertRaises(CollectiveDomainError):
                generalized_ritz(candidate, singular)
        self.parallel(action, collective_checks=False, validated_widths=(6,))

    def test_safe_mode_retains_collective_bad_shape_failure(self):
        def action(base, candidate, comm):
            x = self.x[base.rows, :6]
            if comm.rank == 1:
                x = x[:-1]
            with self.assertRaisesRegex(CollectiveDomainError, 'rank 1'):
                candidate.apply(x)
        self.parallel(action)

    def test_optional_cupy_subset_kernel(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest('no CUDA device')
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f'CuPy/CUDA unavailable: {exc}')
        owner = np.zeros(self.n, dtype=np.int32)
        base = DistributedHamiltonian(self.meta, self.v, self.b, self.signs,
                                      owner, xp=cp)
        for transport in ('host', 'cuda'):
            candidate = OverlappedDistributedHamiltonian(
                self.meta, self.v, self.b, self.signs, owner, xp=cp,
                transport=transport, collective_checks=False, validated_widths=(1, 6, 7))
            for width in (1, 6, 7):
                x = cp.asarray(self.x[:, :width], order='F')
                actual = candidate.apply(x)
                expected = base.apply(x)
                cp.cuda.get_current_stream().synchronize()
                np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected))
            # Exercise boundary-row dispatch independently in serial: all
            # mapped stencil indices remain local so no uninitialized halo is read.
            candidate.interior_rows = np.empty(0, dtype=np.int32)
            candidate.device_interior = cp.asarray(candidate.interior_rows)
            candidate.boundary_rows = np.arange(self.n, dtype=np.int32)
            candidate.device_boundary = cp.asarray(candidate.boundary_rows)
            x = cp.asarray(self.x[:, :7], order='F')
            np.testing.assert_array_equal(cp.asnumpy(candidate.apply(x)), cp.asnumpy(base.apply(x)))


if __name__ == '__main__':
    unittest.main()
