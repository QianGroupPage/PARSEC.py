"""The bounding box of a grid, which the periodic local-force ball sums ask for once per atom."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from parsec_python.acceleration.Forces import periodic_local as local


def _grid(seed: int, points: int = 400) -> SimpleNamespace:
    return SimpleNamespace(coordinates=np.random.default_rng(seed).uniform(-3.0, 9.0, size=(points, 3)))


class GridBoundsTests(unittest.TestCase):
    def test_the_bounds_are_the_minimum_and_maximum_of_the_coordinates(self) -> None:
        grid = _grid(0)
        low, high = local._grid_bounds(grid)
        np.testing.assert_array_equal(low, grid.coordinates.min(axis=0))
        np.testing.assert_array_equal(high, grid.coordinates.max(axis=0))

    def test_a_second_request_for_the_same_grid_does_not_reduce_the_coordinates_again(self) -> None:
        grid = _grid(1)
        first = local._grid_bounds(grid)
        second = local._grid_bounds(grid)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_another_grid_gets_its_own_bounds(self) -> None:
        one, other = _grid(2), _grid(3)
        low_one, high_one = local._grid_bounds(one)
        low_other, high_other = local._grid_bounds(other)
        self.assertFalse(np.array_equal(low_one, low_other))
        np.testing.assert_array_equal(low_other, other.coordinates.min(axis=0))
        np.testing.assert_array_equal(high_other, other.coordinates.max(axis=0))
        # and asking for the first again is right, not the cache of the other
        np.testing.assert_array_equal(local._grid_bounds(one)[0], low_one)
        np.testing.assert_array_equal(local._grid_bounds(one)[1], high_one)

    def test_a_grid_rebuilt_in_the_same_place_is_not_taken_for_the_old_one(self) -> None:
        grid = _grid(4)
        local._grid_bounds(grid)
        del grid
        rebuilt = _grid(5)
        np.testing.assert_array_equal(local._grid_bounds(rebuilt)[1], rebuilt.coordinates.max(axis=0))

    def test_the_images_that_reach_the_grid_are_those_of_the_uncached_rule(self) -> None:
        grid = _grid(6)
        rng = np.random.default_rng(7)
        translations = rng.integers(-3, 4, size=(60, 3)).astype(float) * 6.0
        for _ in range(5):
            centre = rng.uniform(0.0, 6.0, size=3)
            radius = float(rng.uniform(0.5, 7.0))
            low, high = grid.coordinates.min(axis=0), grid.coordinates.max(axis=0)
            points = centre + translations
            gap = np.maximum(np.maximum(low - points, points - high), 0.0)
            expected = translations[np.sqrt(np.einsum("ij,ij->i", gap, gap)) <= radius]
            np.testing.assert_array_equal(local._images_reaching_the_grid(grid, centre, translations, radius), expected)


if __name__ == "__main__":
    unittest.main()
