"""The meta-GGA terms of periodic cells on the devices of the k-point workers.

* the CuPy Bloch operator applies the host terms (``(G + ik)^H vtau (G + ik)``
  and ``2 sum_c [W_c G_c + G_c^T W_c]``), on an orthorhombic grid and on
  Fortran's general grid of an fcc cell (whose derivatives the device applies
  along the grid directions), in ``apply`` and in the Chebyshev filter;
* ``tau`` and the orbital gradient of device Bloch factors are the host ones;
* the k-point loop on the device (CuPy, and the NumPy double of the device
  code) is the host loop to 1e-9 Ry, density and orbital forms of the gradient.
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.SCF.meta_gga_periodic import _host_bloch_fields
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.tests.unit.test_meta_gga_periodic import (
    DATA,
    SILICON,
    TWO_KPOINTS,
    _empty_projectors,
    _grids,
)


@unittest.skipUnless(cupy_available(), "CuPy with a CUDA device is needed")
class DeviceBlochMetaGGATests(unittest.TestCase):
    def test_cupy_operator_is_the_host_operator(self) -> None:
        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter

        backend = CupyArrayBackend(0)
        for name, grid, _ in _grids():
            with self.subTest(grid=name):
                gradient = build_gradient(grid)
                laplacian = build_negative_laplacian(grid)
                rng = np.random.default_rng(4)
                size = grid.size
                k_point = np.array([0.17, -0.05, 0.29])
                block = rng.standard_normal((size, 5)) + 1j * rng.standard_normal((size, 5))
                for fluxes in (None, tuple(0.1 * rng.standard_normal(size) for _ in range(3))):
                    host = KPointKohnShamHamiltonian(
                        laplacian, gradient, rng.standard_normal(size), _empty_projectors(size), k_point,
                        kinetic_potential=0.5 + rng.random(size), gradient_flux=fluxes,
                    )
                    device = DeviceBlochConverter(backend).operator(host)
                    expected = host.apply(block)
                    result = backend.asnumpy(device.apply(backend.asarray(np.ascontiguousarray(block))))
                    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-11 * np.abs(expected).max())
                    filtered, _ = device.chebyshev_filter(block, 4, -1.0, 40.0, -2.0)
                    reference = block.copy()
                    previous = block.copy()
                    # the same recurrence on the host operator
                    half, center = 20.5, 19.5
                    sigma_one = half / (-2.0 - center)
                    current = (host.apply(block) - center * block) * (sigma_one / half)
                    sigma = sigma_one
                    for _ in range(2, 5):
                        following = (host.apply(current) - center * current) * (2.0 / half) - sigma * previous
                        sigma_next = 1.0 / (2.0 / sigma_one - sigma)
                        following *= sigma_next
                        previous, current, sigma = current, following, sigma_next
                    reference = current
                    np.testing.assert_allclose(filtered, reference, rtol=0, atol=1e-9 * np.abs(reference).max())

    def test_device_tau_and_gradient_are_the_host_fields(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.bloch_meta_gga_device import device_bloch_fields

        for name, grid, _ in _grids():
            with self.subTest(grid=name):
                gradient = tuple(build_gradient(grid))
                rng = np.random.default_rng(8)
                vectors = rng.standard_normal((grid.size, 4)) + 1j * rng.standard_normal((grid.size, 4))
                occupations = np.array([1.0, 0.7, 0.0, 0.2])
                k_point = np.array([0.3, 0.0, -0.11])
                tau, density_gradient = device_bloch_fields(
                    cp, gradient, cp.asarray(vectors), occupations, k_point, 0.37, with_gradient=True
                )
                host_tau, host_gradient = _host_bloch_fields(
                    gradient, vectors, occupations, k_point, 0.37, with_gradient=True
                )
                np.testing.assert_allclose(cp.asnumpy(tau), host_tau, rtol=1e-13, atol=1e-14)
                np.testing.assert_allclose(cp.asnumpy(density_gradient), host_gradient, rtol=1e-12, atol=1e-13)


class DeviceKPointLoopTests(unittest.TestCase):
    """The k-point loop of a meta-GGA on the device code = the host loop."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="parsec_meta_gga_periodic_device_"))
        shutil.copy(DATA / "Si_FHIPP.DAT", self.directory / "Si_FHIPP.DAT")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _energy(self, backend: str, form: str) -> float:
        import parsec_python.acceleration  # noqa: F401  (installs the parallel k-point solver)
        from parsec_python.Grid import kpoint_sampling
        from parsec_python.Input import parse_parsec_input
        from parsec_python.SCF import prepare_periodic_single_point
        from parsec_python.SCF.kpoints import run_scf_kpoints

        path = self.directory / f"{backend}_{form}.in"
        path.write_text(SILICON.format(extra=TWO_KPOINTS, criterion=1e-9))
        with mock.patch.dict(
            os.environ,
            {"PARSEC_KPOINT_BACKEND": backend, "PARSEC_KPOINT_WORKERS": "1", "PARSEC_META_GGA_GRADIENT": form},
        ):
            system = prepare_periodic_single_point(parse_parsec_input(path).problem)
            k_points, weights = kpoint_sampling(system.input)
            result = run_scf_kpoints(system, k_points, weights)
        self.assertTrue(result.converged)
        return float(result.energies.total)

    def test_device_loops_are_the_host_loop(self) -> None:
        backends = ["numpy-device"] + (["cupy"] if cupy_available() else [])
        for form in ("density", "orbital"):
            host = self._energy("cpu", form)
            for backend in backends:
                with self.subTest(gradient=form, backend=backend):
                    self.assertAlmostEqual(self._energy(backend, form), host, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
