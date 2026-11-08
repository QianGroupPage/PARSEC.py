"""The derivative operators of the meta-GGA accuracy benchmark (``acceleration/benchmarks/meta_gga_accuracy.py``).

The numbers of the accuracy study against Quantum ESPRESSO rest on them:

* ``fd<order>`` on the bounding box with zeros outside the domain is the
  gradient operator of the run (``grid_gradient``) for the run's order;
* spectral (FFT) derivatives of a Gaussian, centered in the domain, are exact
  to round-off where the Gaussian has decayed at the boundary;
* the spectral interpolation to a twice finer box reproduces the Gaussian and
  its integral, and its derivatives are exact there too.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.V_xc.meta_gga import grid_gradient
from parsec_python.acceleration.benchmarks.meta_gga_accuracy import (
    BoxDerivatives,
    RefinedSpectral,
)
from parsec_python.models import GridSettings


def _grid():
    return build_cluster_grid(GridSettings(spacing=0.3, radius=6.0, expansion_order=12))


def _gaussian(points, width=0.8, center=(0.1, -0.2, 0.15)):
    offset = np.asarray(points, dtype=np.float64) - np.asarray(center)
    value = np.exp(-np.sum(offset * offset, axis=1) / (2.0 * width * width))
    gradient = -(offset / (width * width)) * value[:, None]
    return value, gradient.T


class AccuracyToolOperatorTest(unittest.TestCase):
    def test_fd_of_the_run_order_is_the_run_gradient(self):
        grid = _grid()
        field = np.random.default_rng(3).standard_normal(grid.size)
        box = BoxDerivatives(grid, "fd12").gradient(field)
        code = grid_gradient(grid).apply(field)
        np.testing.assert_allclose(box, code, rtol=0.0, atol=1e-12 * np.max(np.abs(code)))

    def test_spectral_gradient_of_a_gaussian(self):
        grid = _grid()
        value, exact = _gaussian(grid.coordinates)
        spectral = BoxDerivatives(grid, "fft").gradient(value)
        self.assertLess(np.max(np.abs(spectral - exact)), 1e-9)
        # The stencil of order 12 is accurate, but not to round-off, at h = 0.3.
        stencil = BoxDerivatives(grid, "fd12").gradient(value)
        self.assertGreater(np.max(np.abs(stencil - exact)), 1e-9)

    def test_refined_interpolation_and_gradient(self):
        grid = _grid()
        value, _ = _gaussian(grid.coordinates)
        refined = RefinedSpectral(grid, 2)
        fine = refined.interpolate(value)
        origin = (np.asarray(grid.index_min) - refined.coarse.pad + np.asarray(grid.settings.shift, dtype=float)) * grid.spacing
        axes = [origin[axis] + np.arange(refined.shape[axis]) * refined.h for axis in range(3)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        exact, exact_gradient = _gaussian(points)
        self.assertLess(np.max(np.abs(fine - exact)), 1e-9)
        self.assertAlmostEqual(float(np.sum(fine) * refined.volume), float(np.sum(value) * grid.volume_element), places=10)
        self.assertLess(np.max(np.abs(refined.gradient(fine) - exact_gradient)), 1e-8)


if __name__ == "__main__":
    unittest.main()
