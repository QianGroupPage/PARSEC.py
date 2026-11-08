"""The periodic exchange kernel built on the device (one launch, and the array-module path) against the host."""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.Hartree.exchange_kernel import PeriodicExchangeKernel

REAL_CUDA_AVAILABLE = cupy_available()
FCC = 0.5 * 10.26 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
SLAB = np.array([[4.73, 0.0, 0.0], [-2.365, 4.73 * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 15.0]])


def mesh_vectors(kernel, mesh):
    return [
        sum(index[axis] / mesh[axis] * kernel.reciprocal[axis] for axis in range(3))
        for index in itertools.product(*(range(count) for count in mesh))
    ]


@unittest.skipUnless(REAL_CUDA_AVAILABLE, "CuPy with a CUDA device is not available")
class DeviceExchangeKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hartree.exchange_kernel_device import device_values

        cls.cp = cp
        cls.device_values = staticmethod(device_values)

    def test_one_launch_equals_the_host_values(self) -> None:
        cp = self.cp
        cases = [
            dict(alpha=0.25, beta=0.0),
            dict(alpha=0.0, beta=0.25, omega=0.11),
            dict(alpha=1.0, beta=-1.0, omega=0.3),
            dict(alpha=0.19, beta=0.46, omega=0.33),
        ]
        for shape, mesh in (((12, 14, 9), (2, 2, 2)), ((16, 16, 16), (3, 3, 3)), ((10, 11, 12), (4, 2, 1))):
            for coefficients in cases:
                for extrapolation in (None, True, False):
                    for divergence in ("gygi-baldereschi", "none"):
                        kernel = PeriodicExchangeKernel(
                            FCC, shape, mesh, divergence=divergence,
                            x_gamma_extrapolation=extrapolation, **coefficients,
                        )
                        vectors = mesh_vectors(kernel, mesh)
                        for q in (vectors[0], vectors[-1], -vectors[1] + kernel.reciprocal[2]):
                            host = kernel.values(q)
                            device = cp.asnumpy(self.device_values(kernel, q))
                            scale = np.max(np.abs(host))
                            np.testing.assert_allclose(device, host, rtol=1.0e-13, atol=1.0e-14 * scale)

    def test_near_mesh_q_gets_the_element(self) -> None:
        # q = k - k' of two equal k-points given to 8 digits: snapped onto the mesh, the element at G = 0.
        for extrapolation in (False, True, "coulomb"):
            kernel = PeriodicExchangeKernel(
                FCC, (12, 12, 12), (2, 2, 2), alpha=0.25, beta=0.0, x_gamma_extrapolation=extrapolation
            )
            device = self.cp.asnumpy(self.device_values(kernel, 2.0e-8 * kernel.reciprocal[0]))
            self.assertEqual(device[0, 0, 0], kernel.divergence_element)
            np.testing.assert_allclose(device, kernel.values(np.zeros(3)), rtol=1.0e-13)

    def test_out_argument_and_slab_fallback(self) -> None:
        cp = self.cp
        kernel = PeriodicExchangeKernel(FCC, (8, 8, 8), (2, 2, 2), alpha=0.25, beta=0.0)
        out = cp.zeros(kernel.shape)
        result = self.device_values(kernel, np.zeros(3), out=out)
        self.assertIs(result, out)
        with self.assertRaises(ValueError):
            self.device_values(kernel, np.zeros(3), out=cp.zeros((8, 8, 9)))
        for mode in ("padded", "truncated"):  # padded: the one launch; truncated: the array-module fallback
            slab = PeriodicExchangeKernel(
                SLAB, (12, 12, 30), (3, 3, 1), alpha=0.25, beta=-0.25, omega=0.11, periodic_axes=(0, 1),
                slab_coulomb=mode,
            )
            for q in mesh_vectors(slab, (3, 3, 1))[:4]:
                host = slab.values(q)
                device = cp.asnumpy(self.device_values(slab, q))
                np.testing.assert_allclose(device, host, rtol=1.0e-12, atol=1.0e-12 * np.max(np.abs(host)))


if __name__ == "__main__":
    unittest.main()
