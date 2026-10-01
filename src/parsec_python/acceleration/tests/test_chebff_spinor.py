"""Tests for the correctness-first CuPy complex CHEBFF primitives.

Like ``test_cupy_spinor.py``, the module-level helpers in
``acceleration.Eigensolvers.chebff_spinor`` take the array module as an
explicit first argument so they can be run with ``numpy`` in place of
``cupy`` and checked against the already-validated reference CPU
implementations (:mod:`parsec_python.Eigensolvers.chebyshev`,
``rayleigh_ritz``) without any CUDA runtime. The top-level
``run_chebff_spinor``/``cupy_spinor_eigenproblem_solver`` orchestration
functions call ``require_cupy()`` directly and are therefore only exercised
by the GPU-skip-gated test in ``test_self_consistent_soc_gpu.py``.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.Eigensolvers.chebyshev import chebyshev_filter as reference_chebyshev_filter
from parsec_python.Eigensolvers.rayleigh_ritz import rayleigh_ritz as reference_rayleigh_ritz

from parsec_python.acceleration.Eigensolvers.chebff_spinor import (
    _chebyshev_filter,
    _lanczos_upper_bound,
    _orthonormalize,
    _rayleigh_ritz,
)


class _DenseOperator:
    """Minimal Hermitian operator exposing both ``@`` (reference) and
    ``.apply`` (cupy_spinor, called here with numpy as the array module)."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.shape = matrix.shape

    def __matmul__(self, other):
        return self.matrix @ other

    def apply(self, other):
        return self.matrix @ other


def _random_hermitian(rng: np.random.Generator, size: int) -> np.ndarray:
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return raw + raw.conj().T


class ChebyshevFilterMatchesReferenceTests(unittest.TestCase):
    def test_matches_reference_on_complex_hermitian_operator(self) -> None:
        rng = np.random.default_rng(41)
        size = 12
        operator = _DenseOperator(_random_hermitian(rng, size))
        eigenvalues = np.linalg.eigvalsh(operator.matrix)
        block = rng.normal(size=(size, 3)) + 1j * rng.normal(size=(size, 3))

        lower_bound = float(np.median(eigenvalues))
        upper_bound = float(eigenvalues[-1]) + 1.0
        reference_eigenvalue = float(eigenvalues[0])

        mine = _chebyshev_filter(
            np, operator, block.copy(), 8, lower_bound, upper_bound, reference_eigenvalue
        )
        reference = reference_chebyshev_filter(
            operator, block.copy(), 8, lower_bound, upper_bound, reference_eigenvalue
        )
        np.testing.assert_allclose(mine, reference, rtol=1e-10, atol=1e-10)


class RayleighRitzMatchesReferenceTests(unittest.TestCase):
    def test_eigenvalues_match_reference_and_basis_orthonormal(self) -> None:
        rng = np.random.default_rng(53)
        size = 10
        operator = _DenseOperator(_random_hermitian(rng, size))
        raw_basis = rng.normal(size=(size, 4)) + 1j * rng.normal(size=(size, 4))
        basis = _orthonormalize(np, raw_basis)

        gram = basis.conj().T @ basis
        np.testing.assert_allclose(gram, np.eye(4), atol=1e-10)

        mine_eigenvalues, mine_vectors = _rayleigh_ritz(np, operator, basis)
        reference = reference_rayleigh_ritz(operator, basis)
        np.testing.assert_allclose(mine_eigenvalues, reference.eigenvalues, atol=1e-10)
        np.testing.assert_allclose(
            np.abs(mine_vectors), np.abs(reference.wavefunctions), atol=1e-8
        )


class LanczosBoundSanityTests(unittest.TestCase):
    """The complex-starting-vector bound estimate does not reproduce the
    reference's real-starting-vector bound bit-for-bit (different starting
    distributions, same documented-as-empirical algorithm), so this checks
    structural correctness: a finite interval that actually contains the
    operator's full spectrum, not literal parity with the reference."""

    def test_bound_contains_the_spectrum(self) -> None:
        rng = np.random.default_rng(61)
        size = 16
        operator = _DenseOperator(_random_hermitian(rng, size))
        true_eigenvalues = np.linalg.eigvalsh(operator.matrix)

        upper_bound, lower_estimate = _lanczos_upper_bound(
            np, operator, size, steps=8, rng=rng
        )
        self.assertTrue(np.isfinite(upper_bound))
        self.assertTrue(np.isfinite(lower_estimate))
        self.assertGreaterEqual(upper_bound, float(true_eigenvalues[-1]))


if __name__ == "__main__":
    unittest.main()
