"""GPU boundary parity on anisotropic sources, origin and repeated calls."""
from hashlib import sha256
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from parsec_python.Grid import build_cluster_grid
from parsec_python.models import GridSettings, HartreeSettings
from parsec_python.acceleration.backends.cupy import cupy_available, require_cupy
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.Hartree.cupy_boundary import CuPyMultipoleBoundaryBuilder, _SOURCE
from parsec_python.acceleration.Hartree.native_boundary import NativeMultipoleBoundaryBuilder
from parsec_python.acceleration.Hartree.poisson import build_hartree_problem


class BoundaryPolicyTests(unittest.TestCase):
    def test_auto_policy_retains_cached_and_unmeasured_cases(self):
        from parsec_python.acceleration.Hartree.cupy_boundary import auto_gpu_boundary_requested
        cp = MagicMock()
        cp.cuda.runtime.getDeviceProperties.return_value = {'name': b'NVIDIA A100-SXM4-80GB'}
        with patch('parsec_python.acceleration.Hartree.cupy_boundary.require_cupy', return_value=(cp, None)) as load:
            self.assertFalse(auto_gpu_boundary_requested(1000,9))
            load.assert_not_called()
            self.assertTrue(auto_gpu_boundary_requested(1031322,9))
            self.assertFalse(auto_gpu_boundary_requested(1031322,4))
            for name in (b'NVIDIA A40', b'NVIDIA H100 80GB HBM3'):
                cp.cuda.runtime.getDeviceProperties.return_value = {'name':name}
                self.assertFalse(auto_gpu_boundary_requested(1031322,9))
        with patch('parsec_python.acceleration.Hartree.cupy_boundary.require_cupy', side_effect=RuntimeError('unavailable')):
            self.assertFalse(auto_gpu_boundary_requested(1031322,9))


def former_kernel_source():
    """The former kernel text: angular arrays at ``l*10+m``."""
    return (_SOURCE.replace('double* partial, int stride)', 'double* partial)')
            .replace('l*stride+m', 'l*10+m')
            .replace('const double* q,int stride) {', 'const double* q) {')
            .replace('double* rhs,int stride) {', 'double* rhs) {')
            .replace('order,norm,q,stride);', 'order,norm,q);'))


class KernelSourceTests(unittest.TestCase):
    def test_stride_is_the_only_change_of_the_kernel_text(self):
        self.assertEqual(
            sha256(former_kernel_source().encode()).hexdigest(),
            '023a9ebb86a71339b9e94399bfb39cca5bc231d2178c04da87d2cfdc1cc4a66c')


@unittest.skipUnless(cupy_available() and native_available(), "CUDA and native extension required")
class CuPyBoundaryTests(unittest.TestCase):
    def test_orders_up_to_9_are_bitwise_the_former_kernels(self):
        cp, _ = require_cupy()
        former = former_kernel_source()
        moments = cp.RawKernel(former, "moments", options=("--std=c++11",))
        boundary_rhs = cp.RawKernel(former, "boundary_rhs", options=("--std=c++11",))
        for shift in ((0.5,0.5,0.5), (0.0,0.0,0.0)):
            grid = build_cluster_grid(GridSettings(spacing=0.5, radius=5.2,
                                                  expansion_order=8, shift=shift))
            xyz = grid.coordinates
            density = np.exp(-0.3*np.sum((xyz-np.array([0.13,0.18,-0.11]))**2, axis=1))
            density *= 4.0/grid.integrate(density)
            for order in (0, 4, 9):
                with self.subTest(shift=shift, order=order):
                    gpu = CuPyMultipoleBoundaryBuilder(grid, order)
                    rhs, boundary = gpu.build(density)
                    g = gpu.geometry
                    norm = cp.zeros(100, dtype=cp.float64)
                    norm.reshape(10,10)[:order+1,:order+1] = g['normalization'].reshape(order+1,order+1)
                    partial = cp.zeros((200, gpu.blocks), dtype=cp.float64)
                    rho = cp.asarray(density)
                    old_rhs = cp.empty_like(rho)
                    moments((gpu.blocks, order+1), (256,), (
                        np.int64(grid.size), np.int32(order), np.float64(grid.volume_element), rho,
                        *(g['source_'+name] for name in ('radius','cosine','sine','phase_real','phase_imag')),
                        norm, partial))
                    positive = partial.sum(axis=1)
                    boundary_rhs((gpu.blocks,), (256,), (
                        np.int64(grid.size), np.int32(order), rho, g['boundary_indptr'],
                        g['boundary_operator_coefficient'],
                        *(g['boundary_'+name] for name in ('radius','cosine','sine','phase_real','phase_imag')),
                        norm, positive, old_rhs))
                    np.testing.assert_array_equal(rhs, cp.asnumpy(old_rhs))
                    old = cp.asnumpy(positive).view(np.complex128).reshape(10,10)
                    for l in range(order+1):
                        for m in range(l+1):
                            self.assertEqual(boundary.moments[l,m], complex(old[l,m]))

    def check_high_order(self, shift, order):
        from parsec_python.acceleration.backends.native import native_build_info
        if order > int(native_build_info().get("maximum_multipole_order", 9)):
            self.skipTest("orders above 9 need native extension 0.6.0")
        self.check_grid(shift, order)

    def test_half_shift_order20(self):
        self.check_high_order((0.5,0.5,0.5), 20)

    def test_origin_order34(self):
        self.check_high_order((0.0,0.0,0.0), 34)

    def check_grid(self, shift, order):
        grid = build_cluster_grid(GridSettings(spacing=0.65, radius=3.5,
                                              expansion_order=8, shift=shift))
        gpu = CuPyMultipoleBoundaryBuilder(grid, order)
        cpu = NativeMultipoleBoundaryBuilder(grid, order)
        cp, _ = require_cupy()
        for offset in (0.13, -0.23):
            xyz = grid.coordinates
            density = np.exp(-0.7*np.sum((xyz-np.array([offset,0.18,-0.11]))**2, axis=1))
            density *= 4.0/grid.integrate(density)
            rhs, boundary = cpu.build(density)
            reference_rhs, reference_boundary = build_hartree_problem(
                density, grid, HartreeSettings(boundary_method="multipole", multipole_order=order))
            with cp.cuda.Stream(non_blocking=True):
                actual_rhs, actual_boundary = gpu.build(density)
            np.testing.assert_allclose(actual_rhs, rhs, rtol=3e-12, atol=3e-12)
            np.testing.assert_allclose(actual_rhs, reference_rhs, rtol=3e-12, atol=3e-12)
            for key in boundary.moments:
                np.testing.assert_allclose(actual_boundary.moments[key], boundary.moments[key],
                                           rtol=3e-11, atol=3e-11)
            points = 1.2*xyz[:37]
            np.testing.assert_allclose(actual_boundary.potential(points),
                                       reference_boundary.potential(points), rtol=3e-12, atol=3e-12)
        self.assertLess(gpu.device_storage_bytes, grid.size*200+gpu.boundary_term_count*80+4096)
        with self.assertRaises(ValueError): gpu.build(np.full(grid.size, np.nan))

    def test_builder_on_another_device_returns_the_same_bits(self):
        cp, _ = require_cupy()
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 2:
            self.skipTest('requires at least two allocated GPUs')
        home = int(cp.cuda.Device().id)
        other = max(device for device in range(device_count) if device != home)
        grid = build_cluster_grid(GridSettings(spacing=0.65, radius=3.5,
                                              expansion_order=8, shift=(0.5,0.5,0.5)))
        # Without a device the builder takes the current one.
        first = CuPyMultipoleBoundaryBuilder(grid, 9)
        moved = CuPyMultipoleBoundaryBuilder(grid, 9, device_id=other)
        self.assertEqual(first.device_id, home)
        self.assertEqual(moved.device_id, other)
        for array in (*moved.geometry.values(), moved.partial, moved.density, moved.rhs):
            self.assertEqual(int(array.device.id), other)
        self.assertEqual(moved.device_storage_bytes, first.device_storage_bytes)
        self.assertEqual(int(cp.cuda.Device().id), home)
        xyz = grid.coordinates
        for offset in (0.13, -0.23):
            density = np.exp(-0.7*np.sum((xyz-np.array([offset,0.18,-0.11]))**2, axis=1))
            density *= 4.0/grid.integrate(density)
            expected_rhs, expected_boundary = first.build(density)
            actual_rhs, actual_boundary = moved.build(density)
            np.testing.assert_array_equal(actual_rhs, expected_rhs)
            self.assertEqual(actual_boundary.moments, expected_boundary.moments)
            # A density already on that device, as the resident chain passes it.
            with cp.cuda.Device(other):
                on_other = cp.asarray(density)
                device_rhs, device_boundary = moved.build_device(on_other)
                np.testing.assert_array_equal(cp.asnumpy(device_rhs), expected_rhs)
            self.assertEqual(device_boundary.moments, expected_boundary.moments)
            self.assertEqual(int(cp.cuda.Device().id), home)
            # The same array handed over while another device is current.
            device_rhs, device_boundary = moved.build_device(on_other)
            self.assertEqual(int(device_rhs.device.id), other)
            with cp.cuda.Device(other):
                np.testing.assert_array_equal(cp.asnumpy(device_rhs), expected_rhs)
            self.assertEqual(device_boundary.moments, expected_boundary.moments)
            self.assertEqual(int(cp.cuda.Device().id), home)
        rhs, _ = NativeMultipoleBoundaryBuilder(grid, 9).build(density)
        np.testing.assert_allclose(actual_rhs, rhs, rtol=3e-12, atol=3e-12)

    def test_half_shift_order9(self):
        self.check_grid((0.5,0.5,0.5), 9)

    def test_origin_order4(self):
        self.check_grid((0.0,0.0,0.0), 4)

    def test_monopole(self):
        self.check_grid((0.0,0.0,0.0), 0)
