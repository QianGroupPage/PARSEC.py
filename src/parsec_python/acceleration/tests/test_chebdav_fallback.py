"""The literal fallback of CHEBDAV's block orthogonalization over a wide workspace.

The first CHEBDAV solve of the Si216 cell (238,328 rows, 476 working states,
a Davidson workspace of 488 columns) stopped with "orthogonalization exceeded
the replacement limit": an appended block that lay almost entirely in the span
of the 157 columns before it failed the audit of the block route, the literal
PARSEC Gram--Schmidt then swept the whole workspace, and its 325 columns
beyond the block, still zero, took one random replacement each, more than the
limit of 100.  Those columns no longer count towards the limit; the sweep, its
draws and every basis of a fallback that passed are unchanged.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available

GPU_AVAILABLE = cupy_available()


def _workspace(width: int, *, rows: int = 20000, existing: int = 20):
    """A C-ordered CHEBDAV-like workspace: an orthonormal prefix, a nearly
    rank-two appended block of six columns, zeros beyond."""

    generator = np.random.default_rng(0)
    basis = np.zeros((rows, width))
    basis[:, :existing] = np.linalg.qr(generator.standard_normal((rows, existing)))[0]
    mixture = basis[:, :existing] @ generator.standard_normal((existing, 6))
    low_rank = generator.standard_normal((rows, 2)) @ generator.standard_normal((2, 6))
    basis[:, existing:existing + 6] = mixture + low_rank + 1.0e-12 * generator.standard_normal((rows, 6))
    return basis, existing


@unittest.skipUnless(GPU_AVAILABLE, "CuPy with a CUDA device is required")
class ChebDavFallbackTests(unittest.TestCase):
    def test_a_dependent_block_in_a_wide_workspace_no_longer_stops(self) -> None:
        import cupy as cp

        from parsec_python.Eigensolvers.lapack_random import LapackRandom
        import parsec_python.acceleration.Eigensolvers.orthogonalize as module

        start, existing = _workspace(200)
        # The former rule counted the 174 zero columns beyond the block.
        with self.assertRaisesRegex(np.linalg.LinAlgError, "replacement limit"):
            module._orthonormalize_literal(
                cp.asarray(start), existing_columns=existing, rng=LapackRandom(), copy=False
            )
        basis = cp.asarray(start)
        result = module.orthonormalize_appended_block(
            basis, existing_columns=existing, active_columns=existing + 6, rng=LapackRandom()
        )
        self.assertEqual(result.algorithm, "parsec_selective_mgs")
        self.assertEqual(result.random_replacements, 200 - existing - 6)
        active = cp.asnumpy(basis[:, : existing + 6])
        self.assertLess(float(np.abs(active.T @ active - np.eye(existing + 6)).max()), 1.0e-12)

    def test_a_fallback_that_passed_keeps_its_basis_and_its_draws(self) -> None:
        import cupy as cp

        from parsec_python.Eigensolvers.lapack_random import LapackRandom
        import parsec_python.acceleration.Eigensolvers.orthogonalize as module

        start, existing = _workspace(60)
        former_random = LapackRandom()
        former = module._orthonormalize_literal(
            cp.asarray(start), existing_columns=existing, rng=former_random, copy=False
        )
        basis = cp.asarray(start)
        random = LapackRandom()
        result = module.orthonormalize_appended_block(
            basis, existing_columns=existing, active_columns=existing + 6, rng=random
        )
        self.assertEqual(result.random_replacements, former.random_replacements)
        self.assertTrue(np.array_equal(cp.asnumpy(basis), cp.asnumpy(former.basis)))
        self.assertEqual(random.seed, former_random.seed)

    def test_the_opt_in_skip_leaves_the_unused_columns_and_the_stream(self) -> None:
        import cupy as cp

        from parsec_python.Eigensolvers.lapack_random import LapackRandom
        import parsec_python.acceleration.Eigensolvers.orthogonalize as module

        start, existing = _workspace(200)
        basis = cp.asarray(start)
        random = LapackRandom()
        with patch.dict(os.environ, {"PARSEC_CUPY_CHEBDAV_FALLBACK_TAIL": "skip"}):
            module.orthonormalize_appended_block(
                basis, existing_columns=existing, active_columns=existing + 6, rng=random
            )
        self.assertEqual(random.seed, LapackRandom().seed)
        self.assertFalse(bool(cp.any(basis[:, existing + 6 :])))
        active = cp.asnumpy(basis[:, : existing + 6])
        self.assertLess(float(np.abs(active.T @ active - np.eye(existing + 6)).max()), 1.0e-12)
        with patch.dict(os.environ, {"PARSEC_CUPY_CHEBDAV_FALLBACK_TAIL": "drop"}):
            with self.assertRaises(ValueError):
                module.chebdav_fallback_tail()


if __name__ == "__main__":
    unittest.main()
