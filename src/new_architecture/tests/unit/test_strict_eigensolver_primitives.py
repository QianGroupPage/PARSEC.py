from __future__ import annotations

import unittest

import numpy as np

from new_architecture.Eigensolvers.chebyshev import (
    chebff_filter,
    chebyshev_filter,
    subspace_filter,
    subspace_filter_blocks,
)
from new_architecture.Eigensolvers.lapack_random import LapackRandom
from new_architecture.Eigensolvers.orthogonalize import orthonormalize
from new_architecture.Eigensolvers.rayleigh_ritz import rayleigh_ritz
from new_architecture.Eigensolvers.spectral_bounds import (
    lanczos_upper_bound,
)


class StrictSpectralBoundTests(unittest.TestCase):
    def test_parsec_lapack_random_array_stream(self) -> None:
        generator = LapackRandom()
        values = generator.uniform_0_1(6)
        np.testing.assert_array_equal(
            values,
            [
                0.56268588535571595,
                0.10215064555407238,
                0.46159709789862902,
                0.46980667878125004,
                0.034645433176688556,
                0.58351660993852406,
            ],
        )
        self.assertEqual(generator.seed, (2390, 344, 837, 3031))

    def test_non_beta_lanczos_clamp_and_reference_values(self) -> None:
        operator = np.diag([-2.0, 0.5, 3.0, 5.0, 8.0, 11.0])
        result = lanczos_upper_bound(
            operator,
            np.arange(1.0, 7.0),
            steps=1,
        )

        # Non-BETA lancz_bound clamps even a request for one step to four.
        self.assertEqual(result.requested_steps, 1)
        self.assertEqual(result.steps, 4)
        self.assertFalse(result.breakdown)
        np.testing.assert_allclose(
            result.ritz_values,
            [-1.1611083795103032, 3.2218362293024456, 7.391590539894991, 10.956707895699786],
            rtol=0.0,
            atol=2.0e-14,
        )
        self.assertAlmostEqual(result.raw_beta, 3.526680820239204)
        self.assertAlmostEqual(result.upper_bound, 14.48338871593899)
        self.assertAlmostEqual(result.lower_bound, -1.1611083795103032)
        self.assertAlmostEqual(result.middle, 4.897799758094742)

    def test_invariant_start_reports_lanczos_breakdown(self) -> None:
        result = lanczos_upper_bound(
            np.eye(5),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            steps=5,
        )
        self.assertTrue(result.breakdown)
        self.assertEqual(result.steps, 1)
        np.testing.assert_allclose(result.ritz_values, [1.0])
        self.assertEqual(result.raw_beta, 0.0)
        self.assertEqual(result.upper_bound, 1.0)
        self.assertEqual(result.lower_bound, 1.0)
        self.assertEqual(result.middle, 1.0)


class StrictSubspacePrimitiveTests(unittest.TestCase):
    def test_orth_normal_replaces_dependent_and_zero_columns(self) -> None:
        vectors = np.array(
            [
                [1.0, 2.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = orthonormalize(
            vectors,
            rng=np.random.default_rng(19),
        )
        np.testing.assert_allclose(
            result.basis.T @ result.basis,
            np.eye(vectors.shape[1]),
            atol=2.0e-15,
        )
        self.assertGreaterEqual(result.random_replacements, 2)
        self.assertEqual(result.zero_replacements, 1)

    def test_rayleigh_ritz_rotates_an_orthonormal_basis(self) -> None:
        operator = np.array(
            [
                [2.0, -1.0, 0.0],
                [-1.0, 3.0, 0.5],
                [0.0, 0.5, 5.0],
            ]
        )
        basis, _ = np.linalg.qr(
            np.array(
                [
                    [1.0, 2.0, 0.5],
                    [0.2, 1.0, 3.0],
                    [2.0, 0.5, 1.0],
                ]
            )
        )
        result = rayleigh_ritz(operator, basis)
        expected, _ = np.linalg.eigh(operator)
        np.testing.assert_allclose(result.eigenvalues, expected, atol=2.0e-15)
        np.testing.assert_allclose(
            result.wavefunctions.T @ result.wavefunctions,
            np.eye(3),
            atol=2.0e-15,
        )
        np.testing.assert_allclose(result.residual_norms, 0.0, atol=3.0e-15)

    def test_normalized_recurrence_matches_chebyshev_polynomial(self) -> None:
        eigenvalues = np.array([0.0, 1.0, 2.0])
        operator = np.diag(eigenvalues)
        degree = 5
        lower, upper, reference = 2.0, 6.0, 0.0
        filtered = chebyshev_filter(
            operator,
            np.eye(3),
            degree,
            lower,
            upper,
            reference,
        )

        center = 0.5 * (upper + lower)
        half_span = 0.5 * (upper - lower)
        coefficients = np.zeros(degree + 1)
        coefficients[-1] = 1.0
        factors = np.polynomial.chebyshev.chebval(
            (eigenvalues - center) / half_span,
            coefficients,
        )
        factors /= np.polynomial.chebyshev.chebval(
            (reference - center) / half_span,
            coefficients,
        )
        np.testing.assert_allclose(filtered, np.diag(factors), atol=3.0e-15)

    def test_subspace_degree_split_is_block_rounded(self) -> None:
        blocks = subspace_filter_blocks(
            vector_count=11,
            block_size=4,
            degree=10,
            degree_delta=2,
        )
        self.assertEqual(
            [(block.start, block.stop, block.degree) for block in blocks],
            [(0, 4, 8), (4, 8, 12), (8, 11, 12)],
        )

        operator = np.diag(np.linspace(-2.0, 8.0, 6))
        vectors = np.random.default_rng(5).normal(size=(6, 5))
        filtered = subspace_filter(
            operator,
            vectors,
            degree=4,
            degree_delta=1,
            lower_bound=1.0,
            upper_bound=9.0,
            block_size=2,
            reset_recurrence_per_block=True,
        )
        expected = np.empty_like(vectors)
        for block in subspace_filter_blocks(5, 2, 4, 1):
            expected[:, block.start:block.stop] = chebyshev_filter(
                operator,
                vectors[:, block.start:block.stop],
                block.degree,
                1.0,
                9.0,
                1.0,
            )
        np.testing.assert_allclose(filtered, expected, atol=2.0e-14)

    def test_fortran_cross_block_sigma_carry_is_explicit(self) -> None:
        operator = np.diag(np.linspace(-1.0, 5.0, 5))
        vectors = np.random.default_rng(8).normal(size=(5, 4))
        one_block = chebff_filter(
            operator,
            vectors,
            5,
            1.0,
            6.0,
            -1.0,
            block_size=4,
        )
        reset_blocks = chebff_filter(
            operator,
            vectors,
            5,
            1.0,
            6.0,
            -1.0,
            block_size=1,
            reset_recurrence_per_block=True,
        )
        source_quirk = chebff_filter(
            operator,
            vectors,
            5,
            1.0,
            6.0,
            -1.0,
            block_size=1,
            reset_recurrence_per_block=False,
        )
        np.testing.assert_allclose(reset_blocks, one_block, atol=2.0e-14)
        self.assertGreater(
            float(np.linalg.norm(source_quirk - one_block)),
            1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
