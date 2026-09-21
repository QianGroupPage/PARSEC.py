from __future__ import annotations

import unittest

import numpy as np

from parsec_python.Grid import build_periodic_grid
from parsec_python.Hartree import neutralize_density, solve_periodic_hartree
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.models import HartreeSettings, PeriodicCell, PeriodicGridSettings

_TIGHT_SETTINGS = HartreeSettings(
    relative_tolerance=1.0e-10, absolute_tolerance=1.0e-14, max_iterations=5000
)


def _cubic_grid(side_length: float, spacing: float, expansion_order: int = 8):
    cell = PeriodicCell(lattice_vectors=side_length * np.eye(3))
    settings = PeriodicGridSettings(spacing=spacing, expansion_order=expansion_order)
    return build_periodic_grid(cell, settings)


class NeutralizeDensityTests(unittest.TestCase):
    def test_removes_mean(self) -> None:
        grid = _cubic_grid(6.0, 0.5)
        density = 1.0 + np.sin(grid.coordinates[:, 0])
        neutral = neutralize_density(density, grid)
        self.assertAlmostEqual(float(np.mean(neutral)), 0.0, places=12)

    def test_rejects_wrong_shape(self) -> None:
        grid = _cubic_grid(6.0, 0.5)
        with self.assertRaises(ValueError):
            neutralize_density(np.zeros(3), grid)


class SolvePeriodicHartreeTests(unittest.TestCase):
    def test_matches_discrete_eigenvalue_for_plane_wave_source(self) -> None:
        side_length = 10.0
        grid = _cubic_grid(side_length, 0.5)
        operator = build_negative_laplacian(grid)

        harmonic_index = 2
        x = grid.coordinates[:, 0]
        density = np.cos(2.0 * np.pi * harmonic_index * x / side_length)  # already zero-mean

        result = solve_periodic_hartree(density, grid, operator, settings=_TIGHT_SETTINGS)
        self.assertTrue(result.converged)

        # A cos-mode source is an exact eigenvector of the discrete operator,
        # so CG's solution must equal rhs / (that same discrete eigenvalue)
        # up to CG's own tolerance -- a check independent of finite-difference
        # truncation error, unlike comparing against the continuum eigenvalue.
        significant = np.abs(density) > 0.1
        discrete_eigenvalue = float(
            np.mean((operator @ density)[significant] / density[significant])
        )
        analytic_discrete_potential = 8.0 * np.pi * density / discrete_eigenvalue
        np.testing.assert_allclose(
            result.potential, analytic_discrete_potential, atol=1.0e-10
        )

    def test_satisfies_poisson_equation_for_arbitrary_neutral_density(self) -> None:
        grid = _cubic_grid(8.0, 0.5)
        operator = build_negative_laplacian(grid)

        rng = np.random.default_rng(0)
        density = rng.normal(size=grid.size)
        density -= density.mean()

        result = solve_periodic_hartree(density, grid, operator, settings=_TIGHT_SETTINGS)
        self.assertTrue(result.converged)

        residual = operator @ result.potential - result.right_hand_side
        self.assertLess(np.max(np.abs(residual)), 1.0e-6)

    def test_potential_has_zero_mean(self) -> None:
        grid = _cubic_grid(8.0, 0.5)
        operator = build_negative_laplacian(grid)
        rng = np.random.default_rng(1)
        density = rng.normal(size=grid.size)
        density -= density.mean()

        result = solve_periodic_hartree(density, grid, operator, settings=_TIGHT_SETTINGS)
        self.assertAlmostEqual(float(np.mean(result.potential)), 0.0, places=10)

    def test_nonzero_net_charge_is_neutralized_before_solving(self) -> None:
        # A nonneutral density solves the same problem as its neutralized
        # version: the mean component is projected out before CG ever runs.
        grid = _cubic_grid(8.0, 0.5)
        operator = build_negative_laplacian(grid)
        rng = np.random.default_rng(2)
        base = rng.normal(size=grid.size)
        neutral = base - base.mean()
        charged = neutral + 3.7  # add spurious net charge

        result_neutral = solve_periodic_hartree(
            neutral, grid, operator, settings=_TIGHT_SETTINGS
        )
        result_charged = solve_periodic_hartree(
            charged, grid, operator, settings=_TIGHT_SETTINGS
        )
        np.testing.assert_allclose(
            result_neutral.potential, result_charged.potential, atol=1.0e-9
        )

    def test_warm_start_mean_component_is_discarded(self) -> None:
        grid = _cubic_grid(8.0, 0.5)
        operator = build_negative_laplacian(grid)
        rng = np.random.default_rng(3)
        density = rng.normal(size=grid.size)
        density -= density.mean()

        cold = solve_periodic_hartree(density, grid, operator, settings=_TIGHT_SETTINGS)
        warm = solve_periodic_hartree(
            density,
            grid,
            operator,
            settings=_TIGHT_SETTINGS,
            initial_potential=cold.potential + 42.0,
        )
        np.testing.assert_allclose(warm.potential, cold.potential, atol=1.0e-9)

    def test_rejects_mismatched_operator_shape(self) -> None:
        grid = _cubic_grid(8.0, 0.5)
        wrong_operator = build_negative_laplacian(_cubic_grid(6.0, 0.5))
        density = np.zeros(grid.size)
        with self.assertRaises(ValueError):
            solve_periodic_hartree(density, grid, wrong_operator)


if __name__ == "__main__":
    unittest.main()
