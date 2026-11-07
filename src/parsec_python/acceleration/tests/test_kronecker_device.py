"""The CuPy Kronecker-product Poisson solver against the NumPy reference."""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree.kronecker import KroneckerPoisson
from parsec_python.models import GridSettings


REAL_CUDA_AVAILABLE = cupy_available()


def _densities(grid, count, seed=0):
    rng = np.random.default_rng(seed)
    points = grid.coordinates
    columns = []
    for _ in range(count):
        center = rng.normal(scale=1.5, size=3)
        offset = points - center
        envelope = np.exp(-0.7 * np.einsum("ij,ij->i", offset, offset))
        columns.append(envelope * (rng.normal() + offset[:, int(rng.integers(3))]))
    return np.column_stack(columns)


@unittest.skipUnless(REAL_CUDA_AVAILABLE, "CuPy with a CUDA device is not available")
class DeviceKroneckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hartree.kronecker_device import DeviceKroneckerPoisson

        cls.cp = cp
        cls.Device = DeviceKroneckerPoisson
        cls.sphere = build_cluster_grid(GridSettings(spacing=0.45, radius=7.0, expansion_order=8))
        cls.box = build_cluster_grid(
            GridSettings(
                spacing=0.45, radius=7.0, expansion_order=6, domain_shape="box",
                box_lengths=(12.0, 11.0, 10.0),
            )
        )

    def test_device_solves_are_the_reference_solves(self) -> None:
        for grid in (self.sphere, self.box):
            densities = _densities(grid, 7)
            for kernel, omega, screening in (
                ("coulomb", None, "split"),
                ("erfc", 0.11, "split"),
                ("erf", 0.4, "split"),
                ("erfc", 0.2, "spectral"),
            ):
                host = KroneckerPoisson(grid, kernel=kernel, omega=omega, screening=screening)
                device = self.Device(host)
                expected = host.solve_batch(densities)
                scale = np.abs(expected).max()
                # Host input: host result; batches of three give the same numbers.
                result = device.solve_batch(densities, batch_size=3)
                self.assertIsInstance(result, np.ndarray)
                np.testing.assert_allclose(result, expected, atol=1e-12 * scale)
                # Device input: device result.
                on_device = device.solve_batch(self.cp.asarray(densities))
                self.assertIsInstance(on_device, self.cp.ndarray)
                np.testing.assert_allclose(self.cp.asnumpy(on_device), expected, atol=1e-12 * scale)
                single = device(densities[:, 2])
                np.testing.assert_allclose(single, expected[:, 2], atol=1e-12 * scale)

    def test_built_from_a_grid_and_memory_bound_batches(self) -> None:
        device = self.Device(self.sphere, kernel="erf", omega=0.11, memory_budget=1)
        self.assertEqual(device.batch_size, 1)
        self.assertIn("CUDA", device.description)
        densities = _densities(self.sphere, 3, seed=4)
        expected = KroneckerPoisson(self.sphere, kernel="erf", omega=0.11).solve_batch(densities)
        np.testing.assert_allclose(device.solve_batch(densities), expected, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "does not match"):
            device(np.zeros(self.sphere.size + 1))
        weighted = self.Device(self.box, kernel="erfc", omega=0.11, weights=(0.19, 0.46))
        densities = _densities(self.box, 3, seed=5)
        expected = KroneckerPoisson(
            self.box, kernel="erfc", omega=0.11, weights=(0.19, 0.46)
        ).solve_batch(densities)
        np.testing.assert_allclose(
            weighted.solve_batch(densities), expected, atol=1e-12 * np.abs(expected).max()
        )


if __name__ == "__main__":
    unittest.main()
