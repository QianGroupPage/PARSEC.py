"""GPU DLARNV sequence, ordering and seed continuity against host reference."""

import unittest
import numpy as np
from parsec_python.acceleration.backends.cupy import cupy_available, require_cupy
from parsec_python.acceleration.Eigensolvers.lapack_random import LapackRandom


@unittest.skipUnless(cupy_available(), "CUDA unavailable")
class DeviceLapackRandomTests(unittest.TestCase):
    def test_skip_ahead_boundaries_and_host_continuation_are_bit_exact(self):
        cp, _ = require_cupy()
        for count in (0, 1, 2047, 2048, 65535, 65536, 65537, 131077):
            with self.subTest(count=count):
                gpu = LapackRandom()
                host = LapackRandom()
                actual = cp.asnumpy(gpu.device_uniform_minus_1_1(count))
                expected = host.uniform_minus_1_1(count)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(gpu.seed, host.seed)
                np.testing.assert_array_equal(gpu.uniform_0_1(19), host.uniform_0_1(19))

    def test_column_major_shapes_repeated_calls_and_nondefault_seed(self):
        cp, _ = require_cupy()
        for order in (False, True):
            gpu = LapackRandom(seed=(7, 123, 3045, 4095))
            host = LapackRandom(seed=(7, 123, 3045, 4095))
            for shape in ((37, 5), (16385, 7), (13, 3)):
                actual = gpu.device_uniform_minus_1_1(shape, column_major=order)
                np.testing.assert_array_equal(
                    cp.asnumpy(actual),
                    host.uniform_minus_1_1(shape, column_major=order),
                )
                self.assertEqual(gpu.seed, host.seed)


if __name__ == "__main__":
    unittest.main()
