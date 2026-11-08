"""The reused prefix buffer of the device Gram--Schmidt loops.

``_PrefixWorkspace.project`` must give the products of the column slice
``basis[:, :column]`` of a C-ordered basis bit for bit (CuPy copies that slice
before each product, C-ordered for ``P.T @ v`` and F-ordered for ``P @ c``),
and the loops must no longer grow CuPy's memory pool by one copy per column:
CHEBDAV's literal fallback over its C-ordered workspace raised the pool of a
274,784-row sector from 3.8 to 24.5 GiB.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available

GPU_AVAILABLE = cupy_available()


def _old_project(self, basis, column, vector):
    previous = basis[:, :column]
    return vector - previous @ (previous.T @ vector)


@unittest.skipUnless(GPU_AVAILABLE, "CuPy with a CUDA device is required")
class PrefixWorkspaceTests(unittest.TestCase):
    def test_products_equal_those_of_the_slice_bit_for_bit(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Eigensolvers.orthogonalize import _PrefixWorkspace

        generator = np.random.default_rng(4)
        for rows in (7, 1000, 20000, 100003):
            for width in (3, 12, 64):
                basis = cp.asarray(generator.standard_normal((rows, width)))
                workspace = _PrefixWorkspace(cp, basis)
                for column in range(1, width):
                    strided = basis[:, column]
                    contiguous = cp.ascontiguousarray(strided) * 0.5
                    for vector in (strided, contiguous):
                        expected = _old_project(None, basis, column, vector)
                        actual = workspace.project(basis, column, vector)
                        self.assertTrue(
                            bool(cp.all(expected == actual)),
                            f"rows {rows} width {width} column {column}",
                        )
        # A slice of a wider C-ordered workspace (CHEBDAV's small sectors).
        wide = cp.asarray(generator.standard_normal((5000, 30)))
        view = wide[:, :20]
        workspace = _PrefixWorkspace(cp, view)
        for column in range(1, 20):
            self.assertTrue(
                bool(cp.all(workspace.project(view, column, view[:, column]) == _old_project(None, view, column, view[:, column])))
            )
        self.assertIsNotNone(workspace.buffer)
        # An F-ordered basis and the whole C-ordered one take the former path.
        basis = cp.asfortranarray(cp.asarray(generator.standard_normal((500, 9))))
        workspace = _PrefixWorkspace(cp, basis)
        self.assertTrue(bool(cp.all(workspace.project(basis, 5, basis[:, 5]) == _old_project(None, basis, 5, basis[:, 5]))))
        self.assertIsNone(workspace.buffer)

    def test_literal_and_speculative_loops_give_the_former_bases(self) -> None:
        import cupy as cp

        from parsec_python.Eigensolvers.lapack_random import LapackRandom
        import parsec_python.acceleration.Eigensolvers.orthogonalize as module

        generator = np.random.default_rng(5)
        rows, width, existing = 30000, 40, 25
        start = generator.standard_normal((rows, width))
        start[:, :existing] = np.linalg.qr(start[:, :existing])[0]
        # A dependent column and zero columns: the replacement branches.
        start[:, existing + 1] = start[:, 0] + 1.0e-14 * start[:, existing + 1]
        start[:, -6:] = 0.0
        outcomes = []
        for project in (_old_project, module._PrefixWorkspace.project):
            with patch.object(module._PrefixWorkspace, "project", project):
                literal = module._orthonormalize_literal(
                    cp.asarray(start), existing_columns=existing, rng=LapackRandom(), copy=False
                )
                speculative = module.orthonormalize(
                    cp.asfortranarray(cp.asarray(start[:, :existing + 1])),
                    existing_columns=0,
                    rng=LapackRandom(),
                )
            outcomes.append(
                (
                    cp.asnumpy(literal.basis),
                    literal.random_replacements,
                    cp.asnumpy(speculative.basis),
                )
            )
        (old_literal, old_count, old_speculative), (new_literal, new_count, new_speculative) = outcomes
        self.assertEqual(old_count, new_count)
        self.assertGreater(new_count, 0)
        self.assertTrue(np.array_equal(old_literal, new_literal))
        self.assertTrue(np.array_equal(old_speculative, new_speculative))

    def test_the_pool_keeps_one_buffer_instead_of_one_copy_per_column(self) -> None:
        import cupy as cp

        from parsec_python.Eigensolvers.lapack_random import LapackRandom
        import parsec_python.acceleration.Eigensolvers.orthogonalize as module

        generator = np.random.default_rng(6)
        rows, width, existing = 60000, 64, 32
        start = generator.standard_normal((rows, width))
        start[:, :existing] = np.linalg.qr(start[:, :existing])[0]
        start[:, existing:] = 0.0  # CHEBDAV's unused workspace tail
        basis_bytes = rows * width * 8
        growth = {}
        for name, project in (("former", _old_project), ("buffer", module._PrefixWorkspace.project)):
            pool = cp.cuda.MemoryPool()
            with cp.cuda.using_allocator(pool.malloc), patch.object(
                module._PrefixWorkspace, "project", project
            ):
                basis = cp.asarray(start)
                before = pool.total_bytes()
                module._orthonormalize_literal(
                    basis, existing_columns=existing, rng=LapackRandom(), copy=False
                )
                cp.cuda.get_current_stream().synchronize()
                growth[name] = pool.total_bytes() - before
                del basis
            pool.free_all_blocks()
        self.assertLess(growth["buffer"], 1.5 * basis_bytes)
        self.assertGreater(growth["former"], 4 * growth["buffer"])


if __name__ == "__main__":
    unittest.main()
