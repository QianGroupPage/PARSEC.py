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

    def test_atoms_across_a_cell_face_keep_every_pair_term(self) -> None:
        # The terms decay with |r_a - r_b + T|, so pairs near opposite
        # corners need translations longer than the cutoff.  Moving the
        # crystal so that atoms cross the faces (and are translated back)
        # changes nothing: the Si8 cube with the atom on the corner gave
        # -67.20371845 Ry with |T| <= cutoff, Fortran's -67.20371829 with
        # every term; moving the crystal by -0.1 bohr along the diagonal
        # changed the former energy by 5e-8 Ry).
        side = 10.26
        fractions = np.array(
            [
                [0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
                [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
            ]
        )
        potentials = {"Si": SimpleNamespace(ionic_charge=4.0)}
        cells = {
            "cube": side * np.eye(3),
            "monoclinic": np.array([[side, 0.0, 0.0], [1.3, side, 0.0], [0.0, 0.0, side]]),
        }
        for name, lattice_vectors in cells.items():
            with self.subTest(cell=name):
                positions = fractions @ lattice_vectors
                atoms = [Atom("Si", position) for position in positions]
                energy = ewald_ion_ion_energy(atoms, potentials, lattice_vectors)
                for shift in ([0.31, 0.17, 0.05], [-0.1, -0.1, -0.1], [-1.0e-7, 2.0, 7.5]):
                    moved = positions + np.asarray(shift)
                    fractional = moved @ np.linalg.inv(lattice_vectors)
                    wrapped = (fractional - np.floor(fractional)) @ lattice_vectors
                    shifted = ewald_ion_ion_energy(
                        [Atom("Si", position) for position in wrapped],
                        potentials,
                        lattice_vectors,
                    )
                    self.assertAlmostEqual(shifted, energy, delta=1.0e-11)
        cube_energy = ewald_ion_ion_energy(
            [Atom("Si", position) for position in fractions * side],
            potentials,
            cells["cube"],
        )
        self.assertAlmostEqual(cube_energy, -67.20371829, delta=5.0e-9)


def direct_ewald_sum(atoms, potentials, lattice_vectors, *, tolerance: float = 1.0e-18, forces: bool = False):
    """The Ewald ion-ion energy (Ry) and forces (Ry/bohr) as the code summed them before 2026-11-08.

    Every lattice translation ``T`` with ``|T| <= r_c + 2 max|r_a - centroid|`` and every atom pair, every ``G``
    with ``|G| <= G_c``, ``eta = sqrt(pi)/V^(1/3)``: ``O(N^2)`` per translation, the reference the pair sums of
    :func:`parsec_python.V_ion.ewald.ewald_ion_ion_terms` are tested against.
    """
    from scipy.special import erfc, erfcinv

    lattice = np.asarray(lattice_vectors, dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    positions = np.array([np.asarray(atom.position, dtype=float) for atom in atoms])
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=float)
    eta = float(np.sqrt(np.pi) / volume ** (1.0 / 3.0))
    real_cutoff = float(erfcinv(tolerance) / eta)
    reciprocal_cutoff = float(2.0 * eta * np.sqrt(-np.log(tolerance)))
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T

    def points(basis, dual, cutoff):
        spacing = 2.0 * np.pi / np.linalg.norm(dual, axis=1)
        top = np.ceil(cutoff / spacing).astype(int) + 1
        mesh = np.meshgrid(*[np.arange(-m, m + 1) for m in top], indexing="ij")
        found = np.column_stack([axis.reshape(-1) for axis in mesh]) @ basis
        return found[np.linalg.norm(found, axis=1) <= cutoff]

    reach = 2.0 * float(np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))
    energy = 0.0
    force = np.zeros_like(positions)
    products = charges[:, None] * charges[None, :]
    for translation in points(lattice, reciprocal, real_cutoff + reach):
        displacement = positions[:, None, :] - positions[None, :, :] + translation
        distance = np.linalg.norm(displacement, axis=2)
        if np.all(translation == 0.0):
            np.fill_diagonal(distance, np.inf)
        energy += 0.5 * np.sum(products * erfc(eta * distance) / distance)
        radial = (erfc(eta * distance) / distance**2
                  + 2.0 * eta * np.exp(-((eta * distance) ** 2)) / (np.sqrt(np.pi) * distance)) / distance
        force += np.einsum("ab,abk->ak", products * radial, displacement)
    vectors = points(reciprocal, lattice, reciprocal_cutoff)
    vectors = vectors[np.any(vectors != 0.0, axis=1)]
    g_squared = np.sum(vectors**2, axis=1)
    weight = np.exp(-g_squared / (4.0 * eta**2)) / g_squared
    phases = np.exp(1j * positions @ vectors.T)
    structure = np.sum(charges[:, None] * phases, axis=0)
    energy += (2.0 * np.pi / volume) * float(np.sum(weight * np.abs(structure) ** 2))
    sine = np.imag(phases * np.conj(structure)[None, :])
    force += (4.0 * np.pi / volume) * charges[:, None] * ((sine * weight[None, :]) @ vectors)
    energy += -(eta / np.sqrt(np.pi)) * float(np.sum(charges**2))
    energy += -(np.pi / (2.0 * eta**2 * volume)) * float(np.sum(charges)) ** 2
    return (2.0 * energy, 2.0 * force) if forces else 2.0 * energy


class EwaldPairSumTests(unittest.TestCase):
    """The pair and structure-factor sums against the direct sum of every translation and atom pair."""

    _POTENTIALS = {
        "A": SimpleNamespace(ionic_charge=4.0),
        "B": SimpleNamespace(ionic_charge=1.0),
        "C": SimpleNamespace(ionic_charge=-2.5),
    }
    _CELLS = {
        "cube": 10.26 * np.eye(3),
        "oblique": np.array([[6.0, 0.0, 0.0], [1.7, 6.5, 0.0], [-1.1, 2.2, 7.0]]),
        "fcc": 5.4 * (np.ones((3, 3)) - np.eye(3)),
        "elongated": np.diag([5.0, 6.0, 23.0]),
    }

    def test_energy_and_forces_equal_the_direct_sum(self) -> None:
        from parsec_python.Forces.ion_ion import ewald_ion_ion_forces

        generator = np.random.default_rng(3)
        for name, lattice in self._CELLS.items():
            for count in (1, 3, 17):
                with self.subTest(cell=name, atoms=count):
                    # fractional coordinates outside [0, 1) too: the sums take the atoms into the cell
                    fractional = generator.uniform(-0.3, 1.3, (count, 3))
                    symbols = generator.choice(sorted(self._POTENTIALS), count)
                    atoms = [Atom(str(symbol), position) for symbol, position in zip(symbols, fractional @ lattice)]
                    direct, direct_forces = direct_ewald_sum(atoms, self._POTENTIALS, lattice, forces=True)
                    energy = ewald_ion_ion_energy(atoms, self._POTENTIALS, lattice)
                    self.assertAlmostEqual(energy, direct, delta=1.0e-13 * max(1.0, abs(direct)))
                    np.testing.assert_allclose(
                        ewald_ion_ion_forces(atoms, self._POTENTIALS, lattice), direct_forces, rtol=0.0, atol=2.0e-13
                    )

    def test_diamond_supercell_energy_per_atom(self) -> None:
        # 2 x 2 x 2 conventional cells of the bulk-diamond benchmark (64 carbon atoms, Z = 4): the energy per atom
        # of the primitive crystal, and the direct sum
        side = 3.567 / 0.529177210903
        basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        sites = np.concatenate((basis, basis + 0.25))
        cells = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=float)
        atoms = [Atom("C", (site + cell) * side) for cell in cells for site in sites]
        potentials = {"C": SimpleNamespace(ionic_charge=4.0)}
        lattice = 2.0 * side * np.eye(3)
        energy = ewald_ion_ion_energy(atoms, potentials, lattice)
        self.assertAlmostEqual(energy, direct_ewald_sum(atoms, potentials, lattice), delta=1.0e-11)
        primitive = ewald_ion_ion_energy(atoms[:8], potentials, side * np.eye(3))
        self.assertAlmostEqual(energy / 64.0, primitive / 8.0, delta=1.0e-12)


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
            # the periodic potential always uses the cubic spline of the table (a coarse one here), so the single
            # copy it is compared with does too
            spline_specifications = {"H": SpeciesPotential(_H_POTRE, 0, use_spline=True)}
            isolated = build_local_ionic_potential(grid, atoms, potentials, spline_specifications)
            distance = np.linalg.norm(
                grid.coordinates - np.asarray(atoms[0].position), axis=1
            )
            near = distance < 3.0
            difference = (periodic - periodic.mean()) - (isolated - isolated.mean())
            errors.append(float(np.max(np.abs(difference[near]))))
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[-1], 0.05)

    def test_atoms_across_a_cell_face_move_the_potential_with_them(self) -> None:
        # Moving every atom by one grid step moves the potential by one grid
        # step, also when atoms cross a face and are translated back into the
        # cell: each atom keeps every image that reaches a grid point of the
        # cell (|T| <= cutoff alone dropped some, up to 2e-6 Ry for Si8).
        specifications = {"H": SpeciesPotential(_H_POTRE, 0)}
        potentials = load_pseudopotentials(specifications, xc_functional="ca")
        side = 6.0
        cells = {
            "cube": side * np.eye(3),
            "fcc": 0.5 * 2.0 * side * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]) / np.sqrt(2.0),
        }
        fractions = np.array([[0.01, 0.02, 0.005], [0.985, 0.97, 0.99], [0.5, 0.97, 0.03]])
        for name, lattice_vectors in cells.items():
            with self.subTest(cell=name):
                cell = PeriodicCell(lattice_vectors=lattice_vectors)
                grid = build_periodic_grid(
                    cell, PeriodicGridSettings(spacing=0.5, expansion_order=4, shift=(0.0, 0.0, 0.0))
                )
                n = tuple(int(value) for value in grid.points_per_axis)
                step = lattice_vectors[0] / n[0]
                positions = fractions @ lattice_vectors
                moved = positions + step
                fractional = moved @ np.linalg.inv(lattice_vectors)
                wrapped = (fractional - np.floor(fractional)) @ lattice_vectors
                values = [
                    ewald_local_ionic_potential(
                        grid,
                        [Atom("H", position) for position in coordinates],
                        potentials,
                        specifications,
                        lattice_vectors,
                    ).reshape(n)
                    for coordinates in (positions, wrapped)
                ]
                np.testing.assert_allclose(
                    values[1], np.roll(values[0], 1, axis=0), rtol=0.0, atol=1.0e-11
                )


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
