"""A toy force field for the tests of ``parsec_python.MLFF`` (``MLFF_Model: python:mlff_springs.py:make_calculator``).

Springs join the atoms that lie within 1.3 times the sum of their covalent
radii in the first structure; each spring is at rest at that sum.  The
variable ``PARSEC_MLFF_TEST_PUSH`` (eV/angstrom) adds a constant force along
x on the first atom, which breaks every symmetry of the structure.
"""

import os

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import covalent_radii


class SpringCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, stiffness=10.0, **kwargs):
        super().__init__(**kwargs)
        self.stiffness = float(stiffness)
        self.pairs = None
        self.push = float(os.environ.get("PARSEC_MLFF_TEST_PUSH", "0") or 0)
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.get_positions()
        radii = covalent_radii[self.atoms.get_atomic_numbers()]
        if self.pairs is None:
            pairs = []
            for first in range(len(positions)):
                for second in range(first + 1, len(positions)):
                    rest = radii[first] + radii[second]
                    if np.linalg.norm(positions[first] - positions[second]) < 1.3 * rest:
                        pairs.append((first, second, rest))
            self.pairs = pairs
        energy = 0.0
        forces = np.zeros_like(positions)
        for first, second, rest in self.pairs:
            vector = positions[first] - positions[second]
            distance = np.linalg.norm(vector)
            stretch = distance - rest
            energy += 0.5 * self.stiffness * stretch**2
            pull = -self.stiffness * stretch * vector / distance
            forces[first] += pull
            forces[second] -= pull
        if self.push:
            energy -= self.push * positions[0, 0]
            forces[0, 0] += self.push
        self.calls += 1
        self.results = {"energy": energy, "forces": forces}


def make_calculator(device="cpu", task=None, charge=0, spin=1):
    return SpringCalculator()
