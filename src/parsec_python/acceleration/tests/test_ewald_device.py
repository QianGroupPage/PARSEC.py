"""The device Ewald ion-ion sums (acceleration/V_ion/ewald_device.py) against the host's."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.special import erfc, erfcinv

from parsec_python.acceleration.V_ion import ewald_device
from parsec_python.V_ion import ewald


def _cell(count: int, side: float, seed: int = 5, oblique: bool = False):
    rng = np.random.default_rng(seed)
    lattice = side * np.eye(3)
    if oblique:
        lattice = side * np.array([[1.0, 0.0, 0.0], [0.3, 0.95, 0.0], [0.2, 0.1, 0.9]])
    fractional = rng.uniform(0.0, 1.0, (count, 3))
    positions = fractional @ lattice
    symbols = ["C" if index % 3 else "H" for index in range(count)]
    atoms = [SimpleNamespace(symbol=symbol, position=position) for symbol, position in zip(symbols, positions)]
    potentials = {"C": SimpleNamespace(ionic_charge=4.0), "H": SimpleNamespace(ionic_charge=1.0)}
    return atoms, potentials, lattice


class CellListTests(unittest.TestCase):
    def test_the_cells_hold_the_pairs_of_the_tree(self) -> None:
        # the loop of the kernel (27 cells around each atom, the atom itself left out) replayed on the host: the
        # real-space energy and forces of the k-d tree's pairs
        for oblique in (False, True):
            atoms, potentials, lattice = _cell(37, 7.0, oblique=oblique)
            lattice, _, fractional, positions, charges = ewald._ion_ion_inputs(atoms, potentials, lattice)
            eta = ewald.ion_ion_splitting(lattice, len(atoms))
            cutoff = float(erfcinv(ewald.ION_ION_TOLERANCE) / eta)
            reference, reference_force = 0.0, np.zeros((len(atoms), 3))
            for rows, others, displacement, distance in ewald._real_space_pairs(positions, fractional, lattice, cutoff):
                products = charges[rows] * charges[others]
                screened = erfc(eta * distance)
                reference += 0.5 * float(np.sum(products * screened / distance))
                radial = products * (
                    screened / distance**2 + 2.0 * eta * np.exp(-((eta * distance) ** 2)) / (np.sqrt(np.pi) * distance)
                ) / distance
                for component in range(3):
                    reference_force[:, component] += np.bincount(
                        rows, weights=radial * displacement[:, component], minlength=len(atoms)
                    )
            images, image_atoms, _ = ewald.candidate_images(positions, fractional, lattice, cutoff)
            origin, cell, shape, order, start = ewald_device._bins(images, cutoff)
            self.assertGreaterEqual(cell, cutoff)
            sorted_images, sorted_atoms = images[order], image_atoms[order]
            total, force, pairs = 0.0, np.zeros((len(atoms), 3)), 0
            for a, position in enumerate(positions):
                home = np.clip(np.floor((position - origin) / cell).astype(int), 0, shape - 1)
                for i in range(max(home[0] - 1, 0), min(home[0] + 1, shape[0] - 1) + 1):
                    for j in range(max(home[1] - 1, 0), min(home[1] + 1, shape[1] - 1) + 1):
                        for k in range(max(home[2] - 1, 0), min(home[2] + 1, shape[2] - 1) + 1):
                            c = (i * shape[1] + j) * shape[2] + k
                            for p in range(start[c], start[c + 1]):
                                if order[p] == a:
                                    continue
                                r = position - sorted_images[p]
                                d = float(np.sqrt(r @ r))
                                if d > cutoff:
                                    continue
                                pairs += 1
                                qq = charges[a] * charges[sorted_atoms[p]]
                                s = float(erfc(eta * d))
                                total += qq * s / d
                                radial = qq * (s / d**2 + 2.0 * eta * np.exp(-((eta * d) ** 2)) / (np.sqrt(np.pi) * d)) / d
                                force[a] += radial * r
            with self.subTest(oblique=oblique):
                self.assertGreater(pairs, 10 * len(atoms))
                self.assertAlmostEqual(0.5 * total, reference, delta=1.0e-12 * abs(reference))
                np.testing.assert_allclose(force, reference_force, rtol=0, atol=1.0e-11 * np.abs(reference_force).max())

    def test_the_setting_and_small_cells_keep_the_host_sums(self) -> None:
        terms = ewald_device.DeviceEwaldTerms()
        atoms, potentials, lattice = _cell(20, 6.0)
        with mock.patch.dict(os.environ, {"PARSEC_DEVICE_EWALD": "auto"}):
            self.assertIsNone(terms._cupy(ewald_device.MINIMUM_ATOMS - 1))
        with mock.patch.dict(os.environ, {"PARSEC_DEVICE_EWALD": "off"}):
            self.assertIsNone(terms._cupy(10 * ewald_device.MINIMUM_ATOMS))
        with mock.patch.dict(os.environ, {"PARSEC_DEVICE_EWALD": "bogus"}):
            with self.assertRaises(ValueError):
                ewald_device.device_ewald_setting()
        # a device object that declines leaves the host's bits
        with mock.patch.object(ewald, "_ion_ion_device", None):
            host = ewald.ewald_ion_ion_terms(atoms, potentials, lattice, forces=True)
        declining = SimpleNamespace(real_space=lambda *args: None, reciprocal=lambda *args: None)
        with mock.patch.object(ewald, "_ion_ion_device", declining):
            again = ewald.ewald_ion_ion_terms(atoms, potentials, lattice, forces=True)
        self.assertEqual(host[0], again[0])
        np.testing.assert_array_equal(host[1], again[1])


def _cuda() -> bool:
    from parsec_python.acceleration.V_ion.periodic_device import _cupy

    return _cupy() is not None


@unittest.skipUnless(_cuda(), "no usable CUDA device")
class DeviceTests(unittest.TestCase):
    def test_the_device_sums_are_the_host_sums_to_round_off(self) -> None:
        terms = ewald_device.DeviceEwaldTerms()
        for count, side, oblique in ((300, 14.0, False), (1100, 21.0, True)):
            atoms, potentials, lattice = _cell(count, side, oblique=oblique)
            # (the acceleration package installs the device sums when it is imported: the host's without them)
            with mock.patch.object(ewald, "_ion_ion_device", None):
                host = ewald.ewald_ion_ion_terms(atoms, potentials, lattice, forces=True)
            with mock.patch.object(ewald_device, "MINIMUM_ATOMS", 0), mock.patch.object(ewald, "_ion_ion_device", terms):
                device = ewald.ewald_ion_ion_terms(atoms, potentials, lattice, forces=True)
                energy_only = ewald.ewald_ion_ion_terms(atoms, potentials, lattice)
                with mock.patch.dict(os.environ, {"PARSEC_DEVICE_EWALD": "off"}):
                    off = ewald.ewald_ion_ion_terms(atoms, potentials, lattice, forces=True)
            with self.subTest(atoms=count):
                self.assertAlmostEqual(device[0], host[0], delta=1.0e-11 * abs(host[0]))
                self.assertEqual(energy_only[0], device[0])
                self.assertIsNone(energy_only[1])
                np.testing.assert_allclose(device[1], host[1], rtol=0, atol=1.0e-10 * np.abs(host[1]).max())
                self.assertEqual(off[0], host[0])
                np.testing.assert_array_equal(off[1], host[1])

    def test_coincident_nuclei_are_refused(self) -> None:
        atoms, potentials, lattice = _cell(40, 8.0)
        atoms[7] = SimpleNamespace(symbol=atoms[7].symbol, position=np.array(atoms[3].position))
        with mock.patch.object(ewald_device, "MINIMUM_ATOMS", 0), mock.patch.object(
            ewald, "_ion_ion_device", ewald_device.DeviceEwaldTerms()
        ):
            with self.assertRaisesRegex(ValueError, "same position"):
                ewald.ewald_ion_ion_terms(atoms, potentials, lattice)


if __name__ == "__main__":
    unittest.main()
