"""CPU simulations exercise the same collective/halo orchestration as MPI."""
import copy
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy_stencil_major import build_stencil_major_metadata
from parsec_python.acceleration.experimental.mpi_domain import (
    CollectiveDomainError, DistributedHamiltonian, build_row_domain,
    chebyshev_filter, generalized_ritz)


class _Request:
    def __init__(self, world=None, key=None, buffer=None):
        self.world, self.key, self.buffer = world, key, buffer

    def Wait(self):
        if self.world is None:
            return
        with self.world.condition:
            ready = self.world.condition.wait_for(
                lambda: self.key in self.world.messages, timeout=10)
            if not ready:
                raise TimeoutError(f'halo message missing: {self.key}')
            self.buffer[...] = self.world.messages.pop(self.key)


class _World:
    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.collectives = {}
        self.messages = {}

    def collective(self, sequence, rank, kind, value, root=0):
        with self.condition:
            state = self.collectives.setdefault(sequence, dict(kind=kind, values={}, read=0))
            if state['kind'] != kind:
                raise RuntimeError('collectives called in different order')
            state['values'][rank] = copy.deepcopy(value)
            if len(state['values']) == self.size:
                values = [state['values'][r] for r in range(self.size)]
                if kind == 'sum':
                    state['result'] = sum(values)
                elif kind == 'gather':
                    state['result'] = values
                elif kind == 'broadcast':
                    state['result'] = values[root]
                elif kind == 'alltoall':
                    state['result'] = [[values[src][dst] for src in range(self.size)]
                                       for dst in range(self.size)]
                self.condition.notify_all()
            ready = self.condition.wait_for(lambda: 'result' in state, timeout=10)
            if not ready:
                raise TimeoutError(f'collective {sequence} {kind} incomplete')
            result = state['result'][rank] if kind == 'alltoall' else state['result']
            state['read'] += 1
            if state['read'] == self.size:
                del self.collectives[sequence]
            return copy.deepcopy(result)


class _Comm:
    def __init__(self, world, rank):
        self.world, self.rank, self.size = world, rank, world.size
        self.sequence = 0

    def _call(self, kind, value, root=0):
        sequence = self.sequence
        self.sequence += 1
        return self.world.collective(sequence, self.rank, kind, value, root)

    def Allreduce(self, send, recv):
        recv[...] = self._call('sum', send)

    def allgather(self, value):
        return self._call('gather', value)

    def alltoall(self, value):
        return self._call('alltoall', value)

    def bcast(self, value, root=0):
        return self._call('broadcast', value, root)

    def Irecv(self, buffer, source, tag):
        return _Request(self.world, (source, self.rank, tag), buffer)

    def Isend(self, buffer, dest, tag):
        with self.world.condition:
            key = (self.rank, dest, tag)
            if key in self.world.messages:
                raise RuntimeError('an unconsumed halo message was overwritten')
            self.world.messages[key] = np.array(buffer, copy=True)
            self.world.condition.notify_all()
        return _Request()


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.n = 37
        self.t = sp.diags((-np.ones(36), 4*np.ones(37), -np.ones(36)),
                          (-1, 0, 1), shape=(37, 37), format='csr')
        self.meta = build_stencil_major_metadata(self.t)
        rng = np.random.default_rng(74)
        self.v = rng.random(37)
        self.b = sp.csr_matrix(rng.normal(size=(37, 4)) * (rng.random((37, 4)) < .2))
        self.signs = np.array([1., -1., 1., -1.])
        self.h = self.t.toarray() + np.diag(self.v) + self.b.toarray() @ np.diag(self.signs) @ self.b.toarray().T
        self.x = rng.normal(size=(37, 8))

    def parallel(self, size, function, *, owner=None, reverse=False):
        if owner is None:
            owner = np.minimum(np.arange(self.n) * size // self.n, size-1)
        world = _World(size)

        def run(rank):
            rows = np.flatnonzero(owner == rank)
            if reverse:
                rows = rows[::-1]
            op = DistributedHamiltonian(self.meta, self.v, self.b, self.signs,
                                        owner, comm=_Comm(world, rank), local_rows=rows)
            return rows, function(op, rank)

        with ThreadPoolExecutor(max_workers=size) as pool:
            futures = [pool.submit(run, rank) for rank in range(size)]
            return [future.result(timeout=30) for future in futures]

    def test_slot_order_and_sparse_halo(self):
        owner = np.arange(self.n) // 10
        for rank in range(4):
            rows = np.flatnonzero(owner == rank)[::-1]
            domain = build_row_domain(self.meta, owner, rank, 4, rows)
            global_indices = np.concatenate((rows, domain.ghost_rows))
            active = domain.neighbors >= 0
            restored = domain.neighbors.copy()
            restored[active] = global_indices[restored[active]]
            np.testing.assert_array_equal(restored, self.meta.neighbors[:, rows])
            self.assertLessEqual(len(domain.ghost_rows), 2)
            np.testing.assert_array_equal(domain.codes, self.meta.coefficient_codes[:, rows])

    def test_serial_and_three_rank_hamiltonian(self):
        for size in (1, 3):
            for width in (1, 6, 7):
                x = self.x[:, :width]
                pieces = self.parallel(size, lambda op, rank: (op.apply(x[op.rows]), op.stats), reverse=True)
                combined = np.empty_like(x)
                for rows, (result, stats) in pieces:
                    combined[rows] = result
                    self.assertLess(stats['halo_receive_bytes'], self.n * width * 8)
                np.testing.assert_allclose(combined, self.h @ x, rtol=2e-14, atol=2e-14)

    def test_empty_projectors_and_vector(self):
        self.b, self.signs = sp.csr_matrix((self.n, 0)), np.empty(0)
        x = self.x[:, 0]
        pieces = self.parallel(4, lambda op, rank: op.apply(x[op.rows]))
        out = np.empty_like(x)
        for rows, result in pieces:
            out[rows] = result
        np.testing.assert_allclose(out, self.t @ x + self.v * x, atol=2e-14)

    def test_generalized_ritz_global_normalization(self):
        pieces = self.parallel(3, lambda op, rank: generalized_ritz(op, self.x[op.rows]),
                               owner=np.arange(self.n) % 3, reverse=True)
        reference_values = scipy.linalg.eigh(self.x.T @ self.h @ self.x,
                                              self.x.T @ self.x, eigvals_only=True)
        q = np.empty_like(self.x)
        for rows, result in pieces:
            q[rows] = result.local_vectors
            np.testing.assert_allclose(result.eigenvalues, reference_values, atol=3e-13)
        np.testing.assert_allclose(q.T @ q, np.eye(8), atol=3e-14)
        expected_residual = np.linalg.norm(self.h @ q - q * reference_values, axis=0)
        np.testing.assert_allclose(pieces[0][1].residual_norms, expected_residual, atol=3e-13)

    def test_chebyshev_carry_sigma(self):
        lower, upper, reference, degree, initial = -4., 35., -6., 10, -.23
        half, center = .5*(upper-lower), .5*(upper+lower)
        s1 = half/(reference-center)
        previous, current, sigma = self.x.copy(), (self.h @ self.x-center*self.x)*(s1/half), initial
        for _ in range(2, degree+1):
            sn = 1/(2/s1-sigma)
            previous, current = current, sn*((2/half)*(self.h @ current-center*current)-sigma*previous)
            sigma = sn
        pieces = self.parallel(3, lambda op, rank: chebyshev_filter(
            op, self.x[op.rows], degree=degree, lower_bound=lower, upper_bound=upper,
            reference_eigenvalue=reference, initial_sigma=initial, return_final_sigma=True))
        out = np.empty_like(self.x)
        for rows, (result, final_sigma) in pieces:
            out[rows] = result
            self.assertEqual(final_sigma, sigma)
        np.testing.assert_allclose(out, current, rtol=3e-13, atol=3e-13)

    def test_rank_local_bad_shape_fails_collectively(self):
        def action(op, rank):
            try:
                op.apply(self.x[op.rows][:-1] if rank == 1 else self.x[op.rows])
            except CollectiveDomainError as exc:
                return str(exc)
            self.fail('bad shape was accepted')
        messages = self.parallel(3, action)
        self.assertEqual(len({message for _, message in messages}), 1)
        self.assertIn('rank 1', messages[0][1])

    def test_singular_overlap_fails_collectively(self):
        x = np.repeat(self.x[:, :1], 2, axis=1)
        def action(op, rank):
            try:
                generalized_ritz(op, x[op.rows])
            except CollectiveDomainError as exc:
                return str(exc)
            self.fail('singular overlap was accepted')
        messages = self.parallel(3, action)
        self.assertEqual(len({message for _, message in messages}), 1)
        self.assertIn('unsafe overlap', messages[0][1])

    def test_optional_cupy_serial_kernel_and_ritz(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest('no CUDA device')
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f'CuPy/CUDA unavailable: {exc}')
        # Kernel construction/execution failures below must fail, not skip.
        for transport in ('host', 'cuda'):
            op = DistributedHamiltonian(self.meta, self.v, self.b, self.signs,
                                        np.zeros(self.n, dtype=np.int32),
                                        xp=cp, transport=transport)
            for width in (1, 6, 7):
                local = cp.asarray(self.x[:, :width], order='F')
                result = op.apply(local)
                np.testing.assert_allclose(cp.asnumpy(result),
                                           self.h @ self.x[:, :width], atol=5e-13)
            ritz = generalized_ritz(op, cp.asarray(self.x, order='F'))
            q = cp.asnumpy(ritz.local_vectors)
            np.testing.assert_allclose(q.T @ q, np.eye(8), atol=5e-13)
            expected = scipy.linalg.eigh(self.x.T @ self.h @ self.x,
                                         self.x.T @ self.x, eigvals_only=True)
            np.testing.assert_allclose(ritz.eigenvalues, expected, atol=5e-13)


if __name__ == '__main__':
    unittest.main()
