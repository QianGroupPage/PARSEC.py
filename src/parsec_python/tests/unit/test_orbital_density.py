"""Densities from orbital blocks (``SCF/orbital_density.py``) on any array namespace and any rows."""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.SCF.noncollinear import spinor_density_and_magnetization
from parsec_python.SCF.orbital_density import spinor_sums, weighted_square_sum


class OrbitalDensityTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(4)
        self.rows, self.states = 37, 150  # more columns than one chunk
        self.vectors = rng.normal(size=(self.rows, self.states)) + 1j * rng.normal(size=(self.rows, self.states))
        self.occupations = rng.uniform(0.0, 1.0, self.states)
        self.occupations[10:40] = 0.0  # a whole chunk without weight is skipped

    def test_weighted_square_sum_is_the_direct_formula(self) -> None:
        direct = (np.abs(self.vectors) ** 2) @ self.occupations
        np.testing.assert_allclose(weighted_square_sum(self.vectors, self.occupations), direct, rtol=1e-13)

    def test_rows_given_are_rows_returned(self) -> None:
        whole = weighted_square_sum(self.vectors, self.occupations)
        pieces = [weighted_square_sum(self.vectors[a:b], self.occupations) for a, b in ((0, 11), (11, 37))]
        np.testing.assert_allclose(np.concatenate(pieces), whole, rtol=1e-13)

    def test_spinor_density_and_magnetization_of_stacked_spinors(self) -> None:
        n = self.rows
        spinors = np.concatenate([self.vectors, self.vectors[::-1] * (0.7 + 0.1j)], axis=0)
        density, magnetization = spinor_density_and_magnetization(spinors, self.occupations, 2.0, 0.5)
        cross = np.conj(spinors[:n]) * spinors[n:]
        scale = 0.5 / 2.0
        np.testing.assert_allclose(
            density, scale * ((np.abs(spinors[:n]) ** 2 + np.abs(spinors[n:]) ** 2) @ self.occupations), rtol=1e-13
        )
        np.testing.assert_allclose(magnetization[0], scale * 2.0 * (cross.real @ self.occupations), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(magnetization[1], scale * 2.0 * (cross.imag @ self.occupations), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(
            magnetization[2],
            scale * ((np.abs(spinors[:n]) ** 2 - np.abs(spinors[n:]) ** 2) @ self.occupations),
            rtol=1e-12,
            atol=1e-13,
        )

    def test_spinor_sums_of_a_block_of_rows_are_those_rows_of_the_sums(self) -> None:
        n = self.rows
        spinors = np.concatenate([self.vectors, self.vectors * (0.3 - 0.2j)], axis=0)
        whole = spinor_sums(spinors, self.occupations)
        lo, hi = 5, 20
        block = np.concatenate([spinors[lo:hi], spinors[n + lo : n + hi]], axis=0)
        part = spinor_sums(block, self.occupations)
        for full, piece in zip(whole, part):
            np.testing.assert_allclose(piece, full[lo:hi], rtol=1e-13)


if __name__ == "__main__":
    unittest.main()
