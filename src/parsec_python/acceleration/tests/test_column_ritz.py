"""Complete Ritz must include two forward and one backward layout exchange."""
from concurrent.futures import ThreadPoolExecutor
import unittest

import numpy as np
import scipy.linalg

from parsec_python.acceleration.experimental.orbital_layout import ColumnLayout
from parsec_python.acceleration.experimental.column_ritz import complete_column_ritz
from parsec_python.acceleration.experimental.mpi_domain import CollectiveDomainError
from parsec_python.acceleration.tests import test_mpi_domain as base_tests
from parsec_python.acceleration.tests.test_orbital_layout import _LayoutComm


class CompleteRitzTests(unittest.TestCase):
    def setUp(self):
        base_tests.DomainTests.setUp(self)

    def parallel(self, matrix, callback):
        world = base_tests._World(3)
        def run(rank):
            comm = _LayoutComm(world, rank)
            layout = ColumnLayout(self.meta, self.v, self.b, self.signs,
                                  matrix.shape[1], comm=comm)
            x = matrix[:, layout.column_start:layout.column_stop]
            return callback(layout, x, comm)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run, r) for r in range(3)]
            return [f.result(timeout=30) for f in futures]

    def test_complete_ritz_three_transposes_and_physical_invariants(self):
        x = self.x[:, :7]
        values = scipy.linalg.eigh(x.T @ self.h @ x, x.T @ x, eigvals_only=True)
        def callback(layout, local, comm):
            result = complete_column_ritz(layout, local)
            self.assertEqual(comm.exchange_calls, 3)
            self.assertEqual(layout.stats['column_to_row_calls'], 2)
            self.assertEqual(layout.stats['row_to_column_calls'], 1)
            np.testing.assert_allclose(result.eigenvalues, values, atol=5e-13)
            np.testing.assert_allclose(result.overlap, x.T @ x, atol=5e-13)
            np.testing.assert_allclose(result.projected_hamiltonian, x.T @ self.h @ x, atol=5e-13)
            self.assertLess(result.orthogonality_error, 5e-13)
            return layout.column_start, layout.column_stop, result
        results = self.parallel(x, callback)
        q = np.empty_like(x)
        for start, stop, result in results:
            q[:, start:stop] = result.local_columns
        np.testing.assert_allclose(q.T @ q, np.eye(x.shape[1]), atol=5e-13)
        expected = np.linalg.norm(self.h @ q-q*values, axis=0)
        np.testing.assert_allclose(results[0][2].residual_norms, expected, atol=5e-13)

    def test_singular_input_fails_on_every_rank(self):
        x = np.repeat(self.x[:, :1], 6, axis=1)
        def callback(layout, local, comm):
            with self.assertRaisesRegex(CollectiveDomainError, 'unsafe overlap'):
                complete_column_ritz(layout, local)
        self.parallel(x, callback)

    def test_serial_identity_transposes_and_optional_residuals(self):
        layout = ColumnLayout(self.meta, self.v, self.b, self.signs, 7)
        result = complete_column_ritz(layout, self.x[:, :7], compute_residuals=False)
        self.assertIsNone(result.residual_norms)
        self.assertEqual(layout.stats['redistribution_send_bytes'], 0)
        self.assertTrue(np.all(np.isfinite(result.eigenvalues)))


if __name__ == '__main__':
    unittest.main()
