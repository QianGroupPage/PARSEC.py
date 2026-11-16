"""The local ionic potential of a periodic cell on the device (``acceleration/V_ion/periodic_local_device.py``).

The device sums the terms of :func:`parsec_python.acceleration.V_ion.periodic_setup.fft_local_ionic_potential` (the
structure-factor series and the short-ranged terms over the reach of every image) in another order: the potential of
the host to round-off, for cubic-spline and linear tables, both grid shifts, atoms just outside the cell, cells
smaller than the reach, two species.  ``PARSEC_PERIODIC_SETUP_DEVICE=off`` keeps the host.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.V_ion import ewald
from parsec_python.V_ion.ionic_potential import load_pseudopotentials
from parsec_python.acceleration.V_ion import periodic_local_device, periodic_setup
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential

_ROOT = Path(__file__).resolve().parents[4]
_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_SILICON = SpeciesPotential(_ROOT / "examples" / "3d_Si" / "Si_POTRE.DAT", 1)
_HYDROGEN = SpeciesPotential(_DATA / "H_POTRE.DAT", 0)
_FCC = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])


def _silicon_atoms(side: float, offset=(0.0, 0.0, 0.0)):
    sites = np.vstack([_FCC, _FCC + 0.25]) * side + np.asarray(offset)
    return [Atom("Si", tuple(site)) for site in sites]


def _potential(grid, atoms, specifications, lattice, setting):
    potentials = load_pseudopotentials(specifications, xc_functional="ca")
    with mock.patch.dict(os.environ, {periodic_local_device.SWITCH: setting}):
        return periodic_setup.fft_local_ionic_potential(grid, atoms, potentials, specifications, lattice, None, 1.0e-14)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceLocalPotentialTests(unittest.TestCase):
    def assert_same(self, grid, atoms, specifications, lattice, label) -> None:
        self.assertIsNotNone(
            periodic_local_device.device_fft_local_ionic_potential(
                grid, atoms, load_pseudopotentials(specifications, xc_functional="ca"), specifications, lattice,
                None, 1.0e-14,
            ),
            label,
        )
        host = _potential(grid, atoms, specifications, lattice, "off")
        device = _potential(grid, atoms, specifications, lattice, "auto")
        scale = float(np.max(np.abs(host)))
        np.testing.assert_allclose(device, host, rtol=0, atol=1.0e-12 * scale, err_msg=label)

    def test_silicon_cells(self) -> None:
        side = 10.26
        for points, shift, offset in ((18, 0.5, (0.0, 0.0, 0.0)), (19, 0.0, (0.3, -0.2, 0.1))):
            lattice = side * np.eye(3)
            grid = build_periodic_grid(
                PeriodicCell(lattice), PeriodicGridSettings(spacing=side / points, expansion_order=8, shift=(shift,) * 3)
            )
            self.assert_same(grid, _silicon_atoms(side, offset), {"Si": _SILICON}, lattice, f"Si {points} {shift} {offset}")

    def test_a_cell_smaller_than_the_reach_and_two_species(self) -> None:
        lattice = np.diag([5.6, 5.6, 7.0])
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=0.35, expansion_order=6))
        atoms = [Atom("Si", (0.4, 0.5, 0.6)), Atom("H", (2.9, 3.1, 1.2)), Atom("H", (5.1, 0.2, 6.6))]
        self.assert_same(grid, atoms, {"Si": _SILICON, "H": _HYDROGEN}, lattice, "two species")

    def test_linear_tables(self) -> None:
        side = 10.26
        lattice = side * np.eye(3)
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=side / 16, expansion_order=8))
        with mock.patch.object(ewald, "LOCAL_POTENTIAL_USES_CUBIC_SPLINE", False):
            self.assert_same(grid, _silicon_atoms(side), {"Si": _SILICON}, lattice, "linear")

    def test_the_switch(self) -> None:
        for value, expected in (("", "auto"), ("OFF", "off")):
            with mock.patch.dict(os.environ, {periodic_local_device.SWITCH: value}):
                self.assertEqual(periodic_local_device.setup_device_setting(), expected)
        with mock.patch.dict(os.environ, {periodic_local_device.SWITCH: "on"}):
            with self.assertRaises(ValueError):
                periodic_local_device.setup_device_setting()


if __name__ == "__main__":
    unittest.main()
