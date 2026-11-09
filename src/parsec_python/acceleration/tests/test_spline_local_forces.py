"""The local forces of ``Cubic_Spline`` species of a periodic cell by ball sums (CPU, no device needed).

``acceleration/Forces/periodic_local.py::fortran_spline_local_forces`` replaces, for the Fortran-derived routines, the
readable local forces (``Forces/periodic.py``: the whole grid for each of about 580 images of each atom, which takes
hours for a few hundred atoms on a grid of 10^5 points and was what a 192-atom molecular dynamics of cubic LLZO
waited for).  Checked here: the self-consistency and core-correction terms equal the readable ones, the ionic term is the
derivative of the FFT potential the SCF uses (central differences) and equals the readable term to the splitting
accuracy, and the inputs it takes.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from parsec_python.Forces import periodic as periodic_forces
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.V_ion import load_pseudopotentials

from ..Forces import periodic_local as fast
from ..V_ion import periodic_setup_fortran as periodic_setup

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_CUBE = PeriodicCell(lattice_vectors=5.6 * np.eye(3))
_ATOMS = (Atom("Si", np.array([0.31, 0.22, 0.13])), Atom("Si", np.array([2.9, 2.6, 3.1])))
_SETTINGS = PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4)


def _fields(grid, lattice):
    x = grid.coordinates
    b = 2.0 * np.pi * np.linalg.inv(lattice).T
    density = 0.05 + 0.02 * np.cos(x @ b[0]) + 0.015 * np.sin(x @ (b[1] + b[2]))
    residual = 0.03 * np.sin(x @ b[1]) + 0.02 * np.cos(x @ (b[0] + b[1]))
    xc = -0.4 + 0.1 * np.cos(x @ b[2])
    return density, residual, xc


def _moved(atoms, index, axis, step):
    out = list(atoms)
    position = np.array(out[index].position, dtype=float)
    position[axis] += step
    out[index] = Atom(out[index].symbol, position)
    return tuple(out)


class SplineLocalForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_periodic_grid(_CUBE, _SETTINGS)
        cls.lattice = _CUBE.lattice_vectors
        cls.density, cls.residual, cls.xc = _fields(cls.grid, cls.lattice)
        cls.species = {
            "Si": SpeciesPotential(DATA / "Si_POTRE.DAT", 2, use_spline=True),
            "Ge": SpeciesPotential(DATA / "Ge_POTRE.DAT", 1, use_spline=True),  # a nonlinear core correction
        }
        cls.potentials = load_pseudopotentials(cls.species)
        cls.atoms = _ATOMS + (Atom("Ge", np.array([1.4, 4.1, 2.2])),)

    def _readable(self):
        self.assertIsNone(periodic_forces._local_force_provider)  # the scan of the grid, not a provider
        return periodic_forces.periodic_local_forces(
            self.grid, self.atoms, self.potentials, self.species, self.lattice, self.density, self.residual, self.xc
        )

    def test_the_other_two_terms_equal_the_readable_terms(self) -> None:
        readable = self._readable()
        quick = fast.fortran_spline_local_forces(
            self.grid, self.atoms, self.potentials, self.species, self.lattice, self.density, self.residual, self.xc
        )
        np.testing.assert_allclose(quick.self_consistency, readable.self_consistency, atol=1.0e-7, rtol=0.0)
        np.testing.assert_allclose(quick.core_correction, readable.core_correction, atol=1.0e-7, rtol=0.0)
        self.assertGreater(np.abs(readable.self_consistency).max(), 1.0e-3)  # the terms are not trivially zero
        self.assertGreater(np.abs(readable.core_correction).max(), 1.0e-3)

    def test_the_ionic_term_is_the_derivative_of_the_scf_potential(self) -> None:
        quick = fast.fortran_spline_local_forces(
            self.grid, self.atoms, self.potentials, self.species, self.lattice, self.density, self.residual, self.xc
        )

        def energy(atoms):
            potential = periodic_setup.fft_local_ionic_potential(
                self.grid, atoms, self.potentials, self.species, self.lattice, None, 1.0e-14
            )
            return self.grid.volume_element * float(np.dot(self.density, potential))

        step = 1.0e-3
        for index, axis in ((0, 0), (1, 2), (2, 1)):
            with self.subTest(atom=index, axis=axis):
                numeric = -(energy(_moved(self.atoms, index, axis, step)) - energy(_moved(self.atoms, index, axis, -step))) / (
                    2.0 * step
                )
                self.assertAlmostEqual(quick.ionic[index, axis], numeric, delta=2.0e-5)
        # the readable term differentiates the exactly converged sum: the same function to the splitting accuracy
        np.testing.assert_allclose(quick.ionic, self._readable().ionic, atol=1.0e-3, rtol=0.0)

    def test_the_inputs_it_takes(self) -> None:
        self.assertTrue(fast.spline_species_takes(self.grid, self.lattice, self.species, self.atoms))
        linear = {"Si": SpeciesPotential(DATA / "Si_POTRE.DAT", 2), "Ge": self.species["Ge"]}
        self.assertFalse(fast.spline_species_takes(self.grid, self.lattice, linear, self.atoms))  # device kernel's
        general = build_periodic_grid(
            PeriodicCell(lattice_vectors=0.5 * 10.26 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])),
            PeriodicGridSettings(fortran_routines=True, spacing=0.9, expansion_order=4),
        )
        self.assertFalse(fast.spline_species_takes(general, np.eye(3), self.species, self.atoms))


if __name__ == "__main__":
    unittest.main()
