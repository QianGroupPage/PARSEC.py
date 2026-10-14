"""Native sector assembly versus literal projection, including fixed planes."""
import os
import unittest
from unittest.mock import patch
import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.models import Atom, GridSettings
from parsec_python.acceleration.Symmetry import (
    AxisReflectionReduction, SignedPermutationReduction, ReflectionRepresentationDecomposition,
)
from parsec_python.acceleration.backends.native import native_available, _load_native


@unittest.skipUnless(native_available(), "native extension required")
class NativeSectorTests(unittest.TestCase):
    def test_stabilizers_and_signed_permutations(self):
        self.assertTrue(hasattr(_load_native(), "reduce_sector_csr"))
        for shift in ((0.,0.,0.), (.5,.5,.5)):
            grid = build_cluster_grid(GridSettings(spacing=.65, radius=2.8, expansion_order=8, shift=shift))
            matrix = build_negative_laplacian(grid)
            for kind in (AxisReflectionReduction, SignedPermutationReduction):
                reduction = kind.detect(grid, (Atom("H",(0.,0.,0.)),))
                decomposition = ReflectionRepresentationDecomposition.build(grid,reduction)
                with patch.dict(os.environ, PARSEC_NATIVE_SECTOR_ASSEMBLY="0"):
                    expected = decomposition.reduce_operators(matrix)
                for dtype in (np.int32,np.int64):
                    matrix.indptr = matrix.indptr.astype(dtype)
                    matrix.indices = matrix.indices.astype(dtype)
                    with patch.dict(os.environ, PARSEC_NATIVE_SECTOR_ASSEMBLY="1"):
                        actual = decomposition.reduce_operators(matrix)
                    for a,b in zip(actual,expected):
                        np.testing.assert_array_equal(a.indptr,b.indptr)
                        np.testing.assert_array_equal(a.indices,b.indices)
                        np.testing.assert_allclose(a.data,b.data,rtol=0,atol=2e-14)
                        self.assertTrue(a.has_canonical_format)


if __name__ == "__main__":
    unittest.main()
