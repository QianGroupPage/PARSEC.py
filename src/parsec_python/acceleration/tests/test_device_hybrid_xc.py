"""The semilocal parts of the hybrid functionals on the device (``acceleration/V_xc/device_hybrid_xc.py``).

The kernels translate the NumPy components line by line, so device and host
agree to round-off: pointwise for every registered hybrid (unpolarized and in
libxc's spin variables) on well-conditioned points, and as fields on a cluster
grid (full grid, symmetry orbits, spin channels with a core density).
Heyd's integration of the screened hole cancels digits as ``nu = omega/k_F``
grows (``V_xc/wpbe.py``: 1e-14 of a 40-digit evaluation for ``nu < 1``,
1e-11 up to 3, 1e-7 to 1e-4 between 3 and 14), and the elementary functions
of host and device differ in the last bits, so where the omega-PBE hole enters
the comparison is made at round-off where the density is above 1e-5 and to
1e-8 of the largest potential in the low-density tails.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.Grid.cluster import build_cluster_grid
from parsec_python.V_xc.hybrid import HYBRID_FUNCTIONALS, hybrid_semilocal_xc, hybrid_semilocal_xc_spin
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.models import Atom, GridSettings

POINT_TOLERANCE = 1.0e-12
FIELD_TOLERANCE = 1.0e-12
#: Potentials with the omega-PBE hole in the low-density tails (nu = omega/k_F up to 14).
TAIL_TOLERANCE = 1.0e-8
_WITH_HOLE = frozenset({"hse06", "hse03", "lc-wpbe"})


def _dense_threshold(functional) -> float:
    """A density whose stencil neighbourhood keeps ``nu = omega/k_F`` below 0.8 (a factor 100 for the stencil)."""

    omegas = [c.omega for c in functional.components if c.name == "wpbe_sr_x"]
    if not omegas:
        return 1.0e-5
    return 100.0 * (max(omegas) / 0.8) ** 3 / (3.0 * np.pi**2)


def _gaussians(coordinates, centers, width, weights):
    values = np.zeros(len(coordinates))
    for center, weight in zip(centers, weights):
        difference = coordinates - np.asarray(center)
        values += weight * np.exp(-np.sum(difference * difference, axis=1) / width)
    return values


def _relative_points(reference, value):
    reference = np.asarray(reference)
    value = np.asarray(value)
    scale = np.max(np.abs(reference))
    if scale == 0.0:
        return float(np.max(np.abs(value)))
    keep = np.abs(reference) > 1e-10 * scale
    return float(np.max(np.abs(reference[keep] - value[keep]) / np.abs(reference[keep])))


def _relative_field(reference, value):
    reference = np.asarray(getattr(reference, "values", reference))
    value = np.asarray(getattr(value, "values", value))
    return float(np.max(np.abs(reference - value)) / max(np.max(np.abs(reference)), 1e-300))


def _points():
    """Points with densities above 5e-3 (spin channels above 2.5e-3): ``nu < 0.8`` for omega up to 0.4."""

    rng = np.random.default_rng(7)
    count = 600
    n = 10.0 ** rng.uniform(-2.3, 2.0, count)
    s = np.concatenate([rng.uniform(0.0, 5.0, count - 40), 10.0 ** rng.uniform(-9, -2, 30), np.zeros(10)])
    sigma = (2.0 * (3.0 * np.pi**2 * n) ** (1.0 / 3.0) * n * s) ** 2
    total = 10.0 ** rng.uniform(-2.0, 2.0, count)
    zeta = rng.uniform(-0.5, 0.5, count)
    up = 0.5 * total * (1.0 + zeta)
    down = 0.5 * total * (1.0 - zeta)
    kf = lambda rho: (6.0 * np.pi**2 * rho) ** (1.0 / 3.0)  # noqa: E731
    s_uu = (2.0 * kf(up) * up * rng.uniform(0.0, 4.0, count)) ** 2
    s_dd = (2.0 * kf(down) * down * rng.uniform(0.0, 4.0, count)) ** 2
    s_ud = np.sqrt(s_uu * s_dd) * rng.uniform(-1.0, 1.0, count)
    return (n, sigma), (up, down, s_uu, s_ud, s_dd)


@unittest.skipUnless(cupy_available(), "CuPy is not available")
class DeviceHybridSemilocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(GridSettings(spacing=0.45, radius=5.0, expansion_order=12))
        coordinates = cls.grid.coordinates
        cls.density = _gaussians(coordinates, ((0.3, -0.2, 0.1), (-1.1, 0.8, 0.4)), 1.4, (1.1, 0.6))
        cls.up = _gaussians(coordinates, ((0.3, -0.2, 0.1),), 1.2, (0.8,))
        cls.down = _gaussians(coordinates, ((-1.1, 0.8, 0.4),), 1.6, (0.5,))
        cls.core = _gaussians(coordinates, ((0.3, -0.2, 0.1),), 0.3, (0.05,))

    def test_points(self) -> None:
        from parsec_python.acceleration.V_xc.device_hybrid_xc import device_semilocal_points

        unpolarized, polarized = _points()
        for label, functional in HYBRID_FUNCTIONALS.items():
            if label == "hf":
                continue
            host = functional.semilocal.unpolarized(*unpolarized)
            device = device_semilocal_points(functional.semilocal, *unpolarized)
            for index, (reference, value) in enumerate(zip(host, device)):
                self.assertLess(_relative_points(reference, value), POINT_TOLERANCE, (label, "unpolarized", index))
            host = functional.semilocal.polarized(*polarized)
            device = device_semilocal_points(functional.semilocal, *polarized)
            for index, (reference, value) in enumerate(zip(host, device)):
                self.assertLess(_relative_points(reference, value), POINT_TOLERANCE, (label, "polarized", index))

    def test_fields(self) -> None:
        from parsec_python.acceleration.V_xc.device_hybrid_xc import (
            DeviceHybridXCEvaluator,
            DeviceSpinHybridXCEvaluator,
        )

        for label, functional in HYBRID_FUNCTIONALS.items():
            reference = hybrid_semilocal_xc(functional, self.density, self.grid, self.core)
            value = DeviceHybridXCEvaluator(functional, self.grid, core_density=self.core)(self.density)
            if label == "hf":
                self.assertEqual(value.total_energy, 0.0)
                continue
            dense = self.density + self.core > _dense_threshold(functional)
            for name in ("potential", "energy_density", "energy_per_electron"):
                tolerance = TAIL_TOLERANCE if label in _WITH_HOLE else FIELD_TOLERANCE
                field, device_field = getattr(reference, name), getattr(value, name)
                self.assertLess(_relative_field(field, device_field), tolerance, (label, name))
                self.assertLess(
                    np.max(np.abs(field - device_field)[dense]) / np.max(np.abs(field)), FIELD_TOLERANCE, (label, name)
                )
            self.assertAlmostEqual(
                value.total_energy, reference.total_energy, delta=1e-13 * abs(reference.total_energy), msg=label
            )
            spin_reference = hybrid_semilocal_xc_spin(functional, self.up, self.down, self.grid, self.core)
            spin = DeviceSpinHybridXCEvaluator(functional, self.grid, core_density=self.core)(self.up, self.down)
            dense = 2.0 * (np.minimum(self.up, self.down) + 0.5 * self.core) > _dense_threshold(functional)
            for name in ("potential_up", "potential_down", "energy_density"):
                tolerance = TAIL_TOLERANCE if label in _WITH_HOLE else FIELD_TOLERANCE
                field, device_field = getattr(spin_reference, name), getattr(spin, name)
                self.assertLess(_relative_field(field, device_field), tolerance, (label, name))
                self.assertLess(
                    np.max(np.abs(field - device_field)[dense]) / np.max(np.abs(field)), FIELD_TOLERANCE, (label, name)
                )
            self.assertAlmostEqual(
                spin.total_energy, spin_reference.total_energy, delta=1e-13 * abs(spin_reference.total_energy), msg=label
            )

    def test_symmetry_orbits(self) -> None:
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
        from parsec_python.acceleration.Symmetry.axis_reflection import AxisReflectionReduction
        from parsec_python.acceleration.V_xc.device_hybrid_xc import DeviceHybridXCEvaluator

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.5, expansion_order=8, shift=(0.5, 0.5, 0.5)))
        atoms = (Atom("O", (0.0, 0.0, 0.3)), Atom("H", (1.2, 0.5, 0.4)), Atom("H", (-1.2, -0.5, 0.4)))
        reducer = SymmetrySCFReducer(AxisReflectionReduction.detect(grid, atoms))
        full = _gaussians(grid.coordinates, [a.position for a in atoms], 0.9, (1.0, 0.4, 0.4))
        wedge = reducer.from_full(full)
        for label in ("hse06", "b3lyp", "cam-b3lyp"):
            functional = HYBRID_FUNCTIONALS[label]
            reference = hybrid_semilocal_xc(functional, reducer.to_full(wedge), grid)
            value = DeviceHybridXCEvaluator(functional, grid, reducer=reducer)(wedge)
            tolerance = TAIL_TOLERANCE if label in _WITH_HOLE else FIELD_TOLERANCE
            self.assertLess(
                _relative_field(reducer.wedge_values(reference.potential), value.potential), tolerance, label
            )
            self.assertAlmostEqual(
                value.total_energy, reference.total_energy, delta=1e-13 * abs(reference.total_energy), msg=label
            )

    def test_refusals(self) -> None:
        from parsec_python.acceleration.V_xc.device_hybrid_xc import (
            component_parameters,
            device_hybrid_supported,
        )

        functional = HYBRID_FUNCTIONALS["pbe0"]
        self.assertTrue(device_hybrid_supported(functional, self.grid))
        with self.assertRaises(ValueError):
            component_parameters(object())
        # (The spin evaluator takes the orbits of the symmetry sectors since
        # Spin_Symmetry_Sectors: test_spin_sectors.py.)


if __name__ == "__main__":
    unittest.main()
