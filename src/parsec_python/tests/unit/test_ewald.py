from __future__ import annotations

from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from parsec_python.Grid import build_periodic_grid
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.V_ion import (
    build_local_ionic_potential,
    ewald_alpha_z_energy,
    ewald_ion_ion_energy,
    ewald_local_ionic_potential,
    load_pseudopotentials,
)

_H_POTRE = Path(__file__).resolve().parents[1] / "data" / "H_POTRE.DAT"

# Well-established constant for the rock-salt (NaCl) structure; see e.g.
# Kittel, "Introduction to Solid State Physics".
_NACL_MADELUNG_CONSTANT = 1.747564594633182


def _hydrogen_box(side_length: float, spacing: float = 0.5):
    specifications = {"H": SpeciesPotential(_H_POTRE, 0)}
    potentials = load_pseudopotentials(specifications, xc_functional="ca")
    cell = PeriodicCell(lattice_vectors=side_length * np.eye(3))
    grid = build_periodic_grid(
        cell, PeriodicGridSettings(spacing=spacing, expansion_order=4)
    )
    center = side_length / 2.0
    atoms = [Atom("H", [center, center, center])]
    return grid, atoms, potentials, specifications, cell


def _rocksalt(lattice_constant: float) -> tuple[np.ndarray, list[Atom], dict]:
    """Return (lattice_vectors, atoms, potentials) for a rock-salt cell.

    FCC Bravais lattice with a two-ion basis: Na+ at the origin, Cl- offset
    by (a/2, 0, 0), giving nearest-neighbor distance a/2 -- the standard
    setup used to quote the NaCl Madelung constant.
    """

    a = lattice_constant
    lattice_vectors = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    atoms = [
        Atom(symbol="Na", position=np.array([0.0, 0.0, 0.0])),
        Atom(symbol="Cl", position=np.array([a / 2.0, 0.0, 0.0])),
    ]
    potentials = {
        "Na": SimpleNamespace(ionic_charge=1.0),
        "Cl": SimpleNamespace(ionic_charge=-1.0),
    }
    return lattice_vectors, atoms, potentials


class EwaldIonIonEnergyTests(unittest.TestCase):
    def test_matches_nacl_madelung_constant(self) -> None:
        lattice_constant = 1.0
        lattice_vectors, atoms, potentials = _rocksalt(lattice_constant)

        # A tight explicit tolerance makes this the strongest available
        # correctness check: the cumulative truncation error is well below
        # the difference from any plausible formula bug.
        energy = ewald_ion_ion_energy(
            atoms, potentials, lattice_vectors, tolerance=1.0e-18
        )

        nearest_neighbor_distance = lattice_constant / 2.0
        # Rydberg convention: a factor of two on the standard Hartree-unit
        # Madelung energy -alpha_M/r0, matching ion_ion_energy's convention.
        expected = -2.0 * _NACL_MADELUNG_CONSTANT / nearest_neighbor_distance
        self.assertAlmostEqual(energy, expected, places=9)

    def test_independent_of_eta_choice(self) -> None:
        lattice_vectors, atoms, potentials = _rocksalt(1.0)

        default_energy = ewald_ion_ion_energy(
            atoms, potentials, lattice_vectors, tolerance=1.0e-18
        )
        for eta in (0.5, 1.0, 2.0, 5.0):
            with self.subTest(eta=eta):
                energy = ewald_ion_ion_energy(
                    atoms, potentials, lattice_vectors, eta=eta, tolerance=1.0e-18
                )
                self.assertAlmostEqual(energy, default_energy, places=9)

    def test_invariant_under_uniform_translation(self) -> None:
        lattice_vectors, atoms, potentials = _rocksalt(1.0)
        shift = np.array([0.37, -1.21, 2.05])
        shifted_atoms = [
            Atom(symbol=atom.symbol, position=np.asarray(atom.position) + shift)
            for atom in atoms
        ]

        energy = ewald_ion_ion_energy(atoms, potentials, lattice_vectors)
        shifted_energy = ewald_ion_ion_energy(
            shifted_atoms, potentials, lattice_vectors
        )
        self.assertAlmostEqual(energy, shifted_energy, places=8)


class EwaldLocalIonicPotentialTests(unittest.TestCase):
    _hydrogen_box = staticmethod(_hydrogen_box)

    def test_shape_independent_of_eta_choice(self) -> None:
        # Unlike ewald_ion_ion_energy, the *absolute* value here does depend
        # on eta: the dropped G=0 constant (see the docstring) scales as
        # 1/eta**2. Only the mean-subtracted shape -- the physically
        # meaningful part, since a spatially uniform potential shift is just
        # the same gauge freedom solve_periodic_hartree already has -- is
        # independent of eta.
        grid, atoms, potentials, specifications, cell = self._hydrogen_box(8.0)
        default = ewald_local_ionic_potential(
            grid, atoms, potentials, specifications, cell.lattice_vectors, tolerance=1.0e-16
        )
        default_shape = default - default.mean()
        for eta in (0.5, 1.0, 2.0):
            with self.subTest(eta=eta):
                value = ewald_local_ionic_potential(
                    grid,
                    atoms,
                    potentials,
                    specifications,
                    cell.lattice_vectors,
                    eta=eta,
                    tolerance=1.0e-16,
                )
                np.testing.assert_allclose(
                    value - value.mean(), default_shape, atol=1.0e-6
                )

    def test_shape_converges_to_isolated_potential_for_large_cell(self) -> None:
        # As the cell grows, periodic images and the (gauge-arbitrary) cell
        # average become negligible, so the mean-subtracted shape near the
        # atom should approach build_local_ionic_potential's single-copy
        # result -- the correct isolated-atom limit.
        errors = []
        for side_length in (10.0, 20.0, 40.0):
            grid, atoms, potentials, specifications, cell = self._hydrogen_box(
                side_length
            )
            periodic = ewald_local_ionic_potential(
                grid, atoms, potentials, specifications, cell.lattice_vectors
            )
            isolated = build_local_ionic_potential(grid, atoms, potentials, specifications)
            distance = np.linalg.norm(
                grid.coordinates - np.asarray(atoms[0].position), axis=1
            )
            near = distance < 3.0
            difference = (periodic - periodic.mean()) - (isolated - isolated.mean())
            errors.append(float(np.max(np.abs(difference[near]))))
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[-1], 0.05)


class EwaldAlphaZEnergyTests(unittest.TestCase):
    """Regression coverage for the alpha_Z absolute-total-energy correction.

    ``ewald_local_ionic_potential``'s returned mean is provably eta-dependent
    (see its own docstring and ``test_shape_independent_of_eta_choice``
    above). The derivation behind ``ewald_alpha_z_energy`` claims that adding
    ``ewald_alpha_z_energy(...) / electron_count`` to that mean removes the
    eta-dependence entirely -- i.e. the *corrected* mean sits at a single,
    eta-independent absolute reference. That is a strong, independent check
    on the formula (a wrong prefactor or sign would not cancel the
    eta-dependence), and is exactly how the formula was validated during
    development before being trusted.
    """

    _hydrogen_box = staticmethod(_hydrogen_box)

    def test_cancels_eta_dependence_of_local_potential_mean(self) -> None:
        grid, atoms, potentials, _specifications, cell = self._hydrogen_box(8.0)
        specifications = {"H": SpeciesPotential(_H_POTRE, 0)}
        electron_count = potentials["H"].ionic_charge

        corrected_means = []
        for eta in (0.6, 0.8, 1.0, 1.5, 2.0):
            local_potential = ewald_local_ionic_potential(
                grid,
                atoms,
                potentials,
                specifications,
                cell.lattice_vectors,
                eta=eta,
                tolerance=1.0e-16,
            )
            correction = ewald_alpha_z_energy(
                atoms, potentials, cell.lattice_vectors, electron_count, eta=eta,
                tolerance=1.0e-16,
            )
            corrected_means.append(
                float(local_potential.mean()) + correction / electron_count
            )

        for value in corrected_means[1:]:
            self.assertAlmostEqual(value, corrected_means[0], places=5)

    def test_scales_linearly_with_electron_count(self) -> None:
        _grid, atoms, potentials, _specifications, cell = self._hydrogen_box(8.0)
        one_electron = ewald_alpha_z_energy(
            atoms, potentials, cell.lattice_vectors, 1.0
        )
        five_electrons = ewald_alpha_z_energy(
            atoms, potentials, cell.lattice_vectors, 5.0
        )
        self.assertAlmostEqual(five_electrons, 5.0 * one_electron, places=12)


class EwaldSharedValidationTests(unittest.TestCase):
    """Guard-clause coverage shared by all three Ewald entry points.

    Each function repeats its own degenerate-cell/empty-atoms check (the
    validation isn't factored into a shared helper), but the behavior being
    guaranteed is identical across all three, so it is parameterized here
    with subTest instead of living as three (or two) near-identical test
    methods per function.
    """

    def test_rejects_degenerate_cell(self) -> None:
        grid, atoms, potentials, specifications, _cell = _hydrogen_box(8.0)
        rocksalt_vectors, rocksalt_atoms, rocksalt_potentials = _rocksalt(1.0)
        degenerate = np.zeros((3, 3))
        cases = {
            "ewald_ion_ion_energy": lambda: ewald_ion_ion_energy(
                rocksalt_atoms, rocksalt_potentials, degenerate
            ),
            "ewald_local_ionic_potential": lambda: ewald_local_ionic_potential(
                grid, atoms, potentials, specifications, degenerate
            ),
            "ewald_alpha_z_energy": lambda: ewald_alpha_z_energy(
                atoms, potentials, degenerate, 1.0
            ),
        }
        for name, call in cases.items():
            with self.subTest(function=name):
                with self.assertRaises(ValueError):
                    call()

    def test_empty_atoms_returns_zero(self) -> None:
        grid, _atoms, potentials, specifications, cell = _hydrogen_box(8.0)
        with self.subTest(function="ewald_ion_ion_energy"):
            self.assertEqual(
                ewald_ion_ion_energy([], potentials, cell.lattice_vectors), 0.0
            )
        with self.subTest(function="ewald_local_ionic_potential"):
            value = ewald_local_ionic_potential(
                grid, [], potentials, specifications, cell.lattice_vectors
            )
            np.testing.assert_array_equal(value, np.zeros(grid.size))
        with self.subTest(function="ewald_alpha_z_energy"):
            self.assertEqual(
                ewald_alpha_z_energy([], potentials, cell.lattice_vectors, 1.0), 0.0
            )


if __name__ == "__main__":
    unittest.main()
