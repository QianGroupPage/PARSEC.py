"""The local Bloch term of an orthorhombic periodic grid as one kernel equals the sparse products.

:mod:`parsec_python.acceleration.Hamiltonian.bloch_stencil` reads the stencil from the matrices and
uses it only where the matrices are that stencil; the kernel (CuPy) agrees with the cuSPARSE route of
:class:`~parsec_python.acceleration.Hamiltonian.device_bloch.DeviceBlochOperator` and with the host
Hamiltonian to round-off, at a general k-point, at Gamma and with Bloch projectors.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from parsec_python import PeriodicCell, PeriodicGridSettings
from parsec_python.acceleration.Hamiltonian.bloch_stencil import AxialStencils, grid_shape_of
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Laplacian import build_gradient, build_negative_laplacian

try:  # pragma: no cover - environment dependent
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False


def _matrices(lengths, spacing: float, order: int):
    grid = build_periodic_grid(PeriodicCell(lattice_vectors=np.diag(lengths)), PeriodicGridSettings(spacing=spacing, expansion_order=order))
    return grid, build_negative_laplacian(grid), tuple(build_gradient(grid))


class AxialStencilTests(unittest.TestCase):
    def test_stencils_of_orthorhombic_grids(self) -> None:
        for lengths, order in (((8.0, 8.0, 8.0), 8), ((6.0, 7.5, 9.0), 6), ((9.0, 7.5, 7.0), 12)):
            with self.subTest(lengths=lengths, order=order):
                grid, laplacian, gradient = _matrices(lengths, 0.5, order)
                shape = grid_shape_of(gradient, grid.size)
                self.assertEqual(shape, tuple(grid.shape))
                stencils = AxialStencils.from_matrices(shape, laplacian, gradient)
                self.assertIsNotNone(stencils)
                self.assertEqual(stencils.half_width, order // 2)
                self.assertTrue(np.all(stencils.laplacian < 0.0) or np.all(stencils.laplacian[:, 0] < 0.0))

    def test_other_matrices_are_refused(self) -> None:
        grid, laplacian, gradient = _matrices((8.0, 8.0, 8.0), 0.5, 8)
        shape = tuple(grid.shape)
        changed = laplacian.tolil()
        changed[5, 5] += 1.0e-12  # one point that is not the stencil (a potential folded in, say)
        self.assertIsNone(AxialStencils.from_matrices(shape, changed.tocsr(), gradient))
        mixed = list(gradient)
        mixed[0] = (gradient[0] + 1.0e-3 * gradient[1]).tocsr()  # an oblique derivative
        self.assertIsNone(AxialStencils.from_matrices(shape, laplacian, tuple(mixed)))
        self.assertIsNone(AxialStencils.from_matrices((shape[1], shape[0], shape[2] + 1), laplacian, gradient))


@unittest.skipUnless(cupy_available(), "CuPy and a CUDA device are required")
class FusedKernelTests(unittest.TestCase):
    def _operators(self, host):
        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter

        sparse = DeviceBlochConverter(CupyArrayBackend(0))
        fused = DeviceBlochConverter(CupyArrayBackend(0))
        fused.fused_stencil = True
        return sparse.operator(host), fused.operator(host)

    def test_kernel_matches_the_sparse_products(self) -> None:
        import cupy as cp

        grid, laplacian, gradient = _matrices((5.4, 6.75, 8.1), 0.45, 8)
        potential = np.random.default_rng(0).normal(size=grid.size)
        for k_point in ((0.0, 0.0, 0.0), (0.21, -0.13, 0.4), (np.pi / 6.0, 0.0, 0.0)):
            k = np.asarray(k_point)
            host = SimpleNamespace(
                negative_laplacian=laplacian,
                gradient=gradient,
                k_point=k,
                _k_gradient=None if not np.any(k) else sum(value * matrix for value, matrix in zip(k, gradient)).tocsr(),
                _diagonal=potential + float(k @ k),
                nonlocal_operator=SimpleNamespace(projectors=sp.csr_matrix((grid.size, 0)), signs=np.zeros(0)),
            )
            sparse, fused = self._operators(host)
            self.assertIsNotNone(fused.stencil)
            self.assertIsNone(fused.laplacian)  # the stencil matrices are not uploaded
            for columns in (1, 7, 70):
                with self.subTest(k=k_point, columns=columns):
                    rng = np.random.default_rng(columns)
                    block = cp.asarray(rng.normal(size=(grid.size, columns)) + 1j * rng.normal(size=(grid.size, columns)))
                    expected = sparse.apply(block)
                    scale = float(cp.max(cp.abs(expected)))
                    self.assertLess(float(cp.max(cp.abs(fused.apply(block) - expected))), 1.0e-14 * scale * 10)

    def test_recurrence_step_and_shared_column_solves(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Eigensolvers.chebff_columns import ColumnGroup, solve_column_eigval
        from parsec_python.Eigensolvers import ChebFFSettings, EigvalSettings, SubspaceSettings
        from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
        from parsec_python.tests.unit.test_hamiltonian_fast_paths import _PeriodicFixture

        _PeriodicFixture.setUpClass()
        fx = _PeriodicFixture
        host = KPointKohnShamHamiltonian(fx.laplacian, fx.gradient, fx.potential, fx.nonlocal_k, fx.k_point)
        sparse, fused = self._operators(host)
        self.assertTrue(fused.fused_recurrence)
        self.assertFalse(sparse.fused_recurrence)
        rng = np.random.default_rng(9)
        current, previous = (
            cp.asarray(rng.normal(size=(fx.grid.size, 6)) + 1j * rng.normal(size=(fx.grid.size, 6))) for _ in range(2)
        )
        before, after = cp.asarray(rng.normal(size=6)), cp.asarray(rng.normal(size=6))
        expected = sparse.apply(current)
        expected -= 1.3 * current
        expected *= 0.7
        expected -= before[None, :] * previous
        expected *= after[None, :]
        step = fused.chebyshev_step(current, previous, 1.3, 0.7, before, after)
        self.assertLess(float(cp.max(cp.abs(step - expected))), 1.0e-12 * float(cp.max(cp.abs(expected))))
        first = fused.chebyshev_step(current, None, 1.3, 0.7)
        reference = (sparse.apply(current) - 1.3 * current) * 0.7
        self.assertLess(float(cp.max(cp.abs(first - reference))), 1.0e-12 * float(cp.max(cp.abs(reference))))
        # four solves of the shared-column eigensolver on one rank: the fused filter gives the sparse numbers
        settings = EigvalSettings(
            safety_buffer=6,
            initial_method="chebff",
            chebff=ChebFFSettings(polynomial_degree=20, filter_cycles=3),
            subspace=SubspaceSettings(polynomial_degree=12, degree_delta=3, block_size=6),
        )
        comm = SimpleNamespace(rank=0, size=1)
        values = []
        for operator in (sparse, fused):
            group = ColumnGroup(comm, cp, sync=cp.cuda.Device().synchronize, transport="host")
            state = None
            for _ in range(4):
                result = solve_column_eigval(operator, 30, settings=settings, group=group, state=state, xp=cp)
                state = result.state
            values.append(np.asarray(result.eigenvalues))
        np.testing.assert_allclose(values[1], values[0], rtol=0.0, atol=1.0e-10)

    def test_kernel_with_bloch_projectors_matches_the_host_hamiltonian(self) -> None:
        import cupy as cp

        from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
        from parsec_python.tests.unit.test_hamiltonian_fast_paths import _PeriodicFixture

        _PeriodicFixture.setUpClass()
        fx = _PeriodicFixture
        host = KPointKohnShamHamiltonian(fx.laplacian, fx.gradient, fx.potential, fx.nonlocal_k, fx.k_point)
        _, fused = self._operators(host)
        self.assertIsNotNone(fused.stencil)
        rng = np.random.default_rng(4)
        block = rng.normal(size=(fx.grid.size, 5)) + 1j * rng.normal(size=(fx.grid.size, 5))
        expected = host.apply(block)
        error = float(np.max(np.abs(cp.asnumpy(fused.apply(cp.asarray(block))) - expected)))
        self.assertLess(error, 1.0e-12 * float(np.max(np.abs(expected))))


if __name__ == "__main__":
    unittest.main()
