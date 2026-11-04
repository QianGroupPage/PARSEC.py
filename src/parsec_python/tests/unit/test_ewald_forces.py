"""Periodic ion-ion forces: the derivative of the Ewald energy."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from parsec_python.Forces import ewald_ion_ion_forces
from parsec_python.models import Atom
from parsec_python.V_ion import ewald_ion_ion_energy

_POTENTIALS = {"A": SimpleNamespace(ionic_charge=4.0), "B": SimpleNamespace(ionic_charge=1.0), "C": SimpleNamespace(ionic_charge=6.0)}
_FRACTIONAL = np.array([[0.11, 0.23, 0.37], [0.58, 0.41, 0.82], [0.76, 0.93, 0.19]])
_SYMBOLS = ("A", "B", "C")
_CELLS = {
    "rectangular": np.diag([6.0, 7.0, 8.5]),
    "fcc": 5.4 * (np.ones((3, 3)) - np.eye(3)),
    "triclinic": np.array([[6.0, 0.0, 0.0], [1.7, 6.5, 0.0], [-1.1, 2.2, 7.0]]),
}


def _atoms(lattice: np.ndarray, positions: np.ndarray | None = None) -> list[Atom]:
    positions = _FRACTIONAL @ lattice if positions is None else positions
    return [Atom(symbol=symbol, position=np.array(position)) for symbol, position in zip(_SYMBOLS, positions)]


def _energy(lattice: np.ndarray, positions: np.ndarray, **kwargs) -> float:
    return ewald_ion_ion_energy(_atoms(lattice, positions), _POTENTIALS, lattice, **kwargs)


class EwaldIonIonForceTests(unittest.TestCase):
    def test_equals_the_finite_difference_of_the_energy(self) -> None:
        step = 0.01
        weights = np.array([1.0, -8.0, 8.0, -1.0]) / (12.0 * step)  # five-point central difference
        offsets = np.array([-2.0, -1.0, 1.0, 2.0]) * step
        for name, lattice in _CELLS.items():
            with self.subTest(cell=name):
                positions = _FRACTIONAL @ lattice
                analytic = ewald_ion_ion_forces(_atoms(lattice), _POTENTIALS, lattice)
                numeric = np.zeros_like(analytic)
                for atom in range(len(positions)):
                    for axis in range(3):
                        values = []
                        for offset in offsets:
                            moved = positions.copy()
                            moved[atom, axis] += offset
                            values.append(_energy(lattice, moved))
                        numeric[atom, axis] = -float(weights @ np.array(values))
                self.assertGreater(float(np.max(np.abs(analytic))), 0.05)
                np.testing.assert_allclose(analytic, numeric, atol=1.0e-7)

    def test_forces_sum_to_zero_and_do_not_depend_on_the_splitting_parameter(self) -> None:
        for name, lattice in _CELLS.items():
            with self.subTest(cell=name):
                default = ewald_ion_ion_forces(_atoms(lattice), _POTENTIALS, lattice)
                np.testing.assert_allclose(default.sum(axis=0), 0.0, atol=1.0e-10)
                for eta in (0.35, 0.6):
                    np.testing.assert_allclose(
                        ewald_ion_ion_forces(_atoms(lattice), _POTENTIALS, lattice, eta=eta), default, atol=1.0e-9
                    )

    def test_an_atom_moved_by_a_lattice_vector_has_the_same_forces(self) -> None:
        for name, lattice in _CELLS.items():
            with self.subTest(cell=name):
                positions = _FRACTIONAL @ lattice
                shifted = positions.copy()
                shifted[1] += lattice[0] - 2.0 * lattice[2]
                np.testing.assert_allclose(
                    ewald_ion_ion_forces(_atoms(lattice, shifted), _POTENTIALS, lattice),
                    ewald_ion_ion_forces(_atoms(lattice, positions), _POTENTIALS, lattice),
                    atol=1.0e-9,
                )

    def test_rocksalt_sites_feel_no_force(self) -> None:
        a = 1.0
        lattice = (a / 2.0) * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        atoms = [Atom(symbol="Na", position=np.zeros(3)), Atom(symbol="Cl", position=np.array([a / 2.0, 0.0, 0.0]))]
        potentials = {"Na": SimpleNamespace(ionic_charge=1.0), "Cl": SimpleNamespace(ionic_charge=-1.0)}
        np.testing.assert_allclose(ewald_ion_ion_forces(atoms, potentials, lattice), 0.0, atol=1.0e-10)


if __name__ == "__main__":
    unittest.main()
