"""The orbital subgroup of a point group with weighted one-dimensional sectors (Symmetry/orbital_groups.py).

Group theory on signed permutation matrices (CPU), and the exact check: the weighted sector spectra of a Td
model Hamiltonian on a real grid are the full-grid spectrum with its degeneracies.
"""

from __future__ import annotations

import itertools
import os
import unittest
from unittest.mock import patch

import numpy as np


def _signed_permutations(sign_rule, permutation_rule):
    operations = []
    for permutation in itertools.permutations(range(3)):
        matrix = np.zeros((3, 3), dtype=np.int64)
        for row, column in enumerate(permutation):
            matrix[row, column] = 1
        even = bool(np.linalg.det(matrix) > 0)
        for signs in itertools.product((1, -1), repeat=3):
            if sign_rule(signs) and permutation_rule(permutation, even):
                operations.append(np.diag(signs) @ matrix)
    return operations


def _even_signs(signs):
    return sum(value < 0 for value in signs) % 2 == 0


GROUPS = {
    "Td": _signed_permutations(_even_signs, lambda p, even: True),
    "Oh": _signed_permutations(lambda s: True, lambda p, even: True),
    "T": _signed_permutations(_even_signs, lambda p, even: even),
    "Th": _signed_permutations(lambda s: True, lambda p, even: even),
    "D2h": _signed_permutations(lambda s: True, lambda p, even: p == (0, 1, 2)),
    "D2": _signed_permutations(lambda s: s[0] * s[1] * s[2] == 1, lambda p, even: p == (0, 1, 2)),
    "D4h": _signed_permutations(lambda s: True, lambda p, even: p[2] == 2),
}


class CharacterTests(unittest.TestCase):
    def test_the_real_irreducible_representations(self):
        from parsec_python.acceleration.Symmetry.orbital_groups import multiplication_table, real_irreducible_characters

        expected = {"Td": [1, 1, 2, 3, 3], "Oh": [1, 1, 1, 1, 2, 2, 3, 3, 3, 3], "T": [1, 2, 3],
                    "Th": [1, 1, 2, 2, 3, 3], "D2h": [1] * 8, "D4h": [1] * 8 + [2, 2]}
        for name, dimensions in expected.items():
            with self.subTest(group=name):
                table = multiplication_table(GROUPS[name])
                representations = real_irreducible_characters(table)
                self.assertEqual(sorted(r.dimension for r in representations), dimensions)
                order = len(GROUPS[name])
                # the regular representation: sum over complex irreps of d^2 is the order
                total = sum(r.dimension ** 2 / (2 if r.complex_pair else 1) for r in representations)
                self.assertAlmostEqual(total, order)
                for rep in representations:
                    # the character of the identity (the first operation) is the dimension
                    self.assertAlmostEqual(rep.character[0], rep.dimension)

    def test_one_dimensional_characters_of_a_non_abelian_group(self):
        from parsec_python.acceleration.Symmetry.orbital_groups import multiplication_table, one_dimensional_characters

        operations = GROUPS["Td"]
        table = multiplication_table(operations)
        # Td itself has two (A1, A2); D2d (S4 along z) four
        self.assertEqual(len(one_dimensional_characters(table, range(len(operations)))), 2)
        d2d = [i for i, op in enumerate(operations) if abs(op[2, 2]) == 1]
        self.assertEqual(len(d2d), 8)
        characters = one_dimensional_characters(table, d2d)
        self.assertEqual(len(characters), 4)
        for chi in characters:
            for a in d2d:
                for b in d2d:
                    self.assertEqual(chi[int(table[a, b])], chi[a] * chi[b])


class PlanTests(unittest.TestCase):
    def test_the_chosen_subgroups_weights_and_costs(self):
        from parsec_python.acceleration.Symmetry.orbital_groups import orbital_group_plan

        expected = {
            # group: (order of H, sorted weights, cost in N M, abelian cost)
            "Td": (8, [1, 1, 3, 3], 1 / 16, 1 / 4),
            "Oh": (16, [1, 1, 1, 1, 3, 3, 3, 3], 1 / 32, 1 / 8),
            "T": (4, [1, 3], 1 / 8, 1 / 4),
            "Th": (8, [1, 1, 3, 3], 1 / 16, 1 / 8),
            "D2h": (8, [1] * 8, 1 / 8, 1 / 8),
            "D2": (4, [1] * 4, 1 / 4, 1 / 4),
        }
        for name, (order, weights, cost, abelian) in expected.items():
            with self.subTest(group=name):
                plan = orbital_group_plan(GROUPS[name])
                self.assertEqual(len(plan.subgroup), order)
                self.assertEqual(sorted(plan.weights), weights)
                self.assertAlmostEqual(plan.cost, cost)
                self.assertAlmostEqual(plan.abelian_cost, abelian)
                # the cosets cover the group once: |G| = |H| [G:H]
                self.assertEqual(len(plan.subgroup) * len(plan.cosets), len(GROUPS[name]))

    def test_every_level_counts_its_degeneracy(self):
        from parsec_python.acceleration.Symmetry.orbital_groups import (
            multiplication_table,
            orbital_group_plan,
            real_irreducible_characters,
        )

        for name in ("Td", "Oh", "T", "Th", "D4h"):
            with self.subTest(group=name):
                operations = GROUPS[name]
                table = multiplication_table(operations)
                plan = orbital_group_plan(operations)
                members = list(plan.subgroup)
                for rep in real_irreducible_characters(table):
                    # sum over solved sectors of weight x multiplicity of the sector in rep|H = dimension
                    found = 0.0
                    for chi, weight in zip(plan.characters, plan.weights):
                        multiplicity = sum(chi[h] * rep.character[h] for h in members) / len(members)
                        found += weight * round(multiplicity)
                    self.assertAlmostEqual(found, rep.dimension)


class TdSpectrumTests(unittest.TestCase):
    """A Td model Hamiltonian (Laplacian and Gaussian wells on CH4) on half-shifted and zero-shift grids."""

    def test_weighted_sector_spectra_are_the_full_spectrum(self):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        from parsec_python.Grid import build_cluster_grid
        from parsec_python.Laplacian import build_negative_laplacian
        from parsec_python.acceleration.Symmetry import (
            PointGroupReduction,
            ReflectionRepresentationDecomposition,
            SignedPermutationReduction,
            detect_point_group_reduction,
        )
        from parsec_python.models import Atom, GridSettings

        a = 1.2
        atoms = [Atom("C", (0.0, 0.0, 0.0))] + [
            Atom("H", tuple(a * np.array(signs, dtype=float)))
            for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        ]
        for shift in (0.5, 0.0):
            with self.subTest(shift=shift):
                grid = build_cluster_grid(GridSettings(spacing=0.5, radius=3.8, expansion_order=4, shift=(shift,) * 3))
                points = np.asarray(grid.physical_coordinates(grid.integer_coordinates))
                potential = np.zeros(points.shape[0])
                for atom, depth in zip(atoms, (3.0, 1.0, 1.0, 1.0, 1.0)):
                    potential -= depth * np.exp(-np.sum((points - np.asarray(atom.position)) ** 2, axis=1) / 0.8)
                hamiltonian = (0.5 * build_negative_laplacian(grid) + sp.diags(potential)).tocsr()
                abelian = SignedPermutationReduction.detect(grid, atoms)
                reduction = detect_point_group_reduction(grid, atoms, abelian)
                self.assertIsInstance(reduction, PointGroupReduction)
                self.assertEqual((abelian.group_order, reduction.group_order), (4, 8))
                self.assertEqual(reduction.full_operations.shape[0], 24)
                decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                self.assertEqual(sorted(decomposition.sector_weights.tolist()), [1, 1, 3, 3])
                count = 30
                full = np.sort(spla.eigsh(hamiltonian, k=count, which="SA", tol=1e-12)[0])
                weighted = []
                for index, matrix in enumerate(decomposition.reduce_operators(hamiltonian)):
                    values = np.linalg.eigvalsh(matrix.toarray())[:count]
                    weighted += [(value, int(decomposition.sector_weights[index])) for value in values]
                expanded = np.array([value for value, weight in sorted(weighted) for _ in range(weight)][:count])
                np.testing.assert_allclose(expanded, full, rtol=0.0, atol=1e-11)
                # the average over the cosets of a field invariant under H gives one invariant under G
                field = np.random.default_rng(1).standard_normal(reduction.wedge_size)
                symmetric = reduction.symmetrize_values(field)
                for images in reduction.coset_images:
                    np.testing.assert_allclose(symmetric[images], symmetric, atol=1e-14)
                # the partners of a weighted state: orthonormal eigenvectors of the full operator, its eigenvalue
                weighted = int(np.flatnonzero(decomposition.sector_weights == 3)[0])
                values, vectors = np.linalg.eigh(decomposition.reduce_operators(hamiltonian)[weighted].toarray())
                state = decomposition.expand_vector(vectors[:, 0], weighted)
                partners = reduction.partner_states(state[:, None], np.array([3]))
                self.assertEqual(partners.shape, (grid.size, 3))
                np.testing.assert_allclose(partners[:, 0], state, atol=0.0)
                np.testing.assert_allclose(partners.T @ partners, np.eye(3) * (state @ state), atol=1e-12)
                residual = hamiltonian @ partners - values[0] * partners
                self.assertLess(float(np.max(np.abs(residual))), 1e-10)
                # the switch keeps the group of before
                with patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP": "abelian"}):
                    self.assertIs(detect_point_group_reduction(grid, atoms, abelian), abelian)

    def test_a_structure_of_an_abelian_group_keeps_its_reduction(self):
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Symmetry import SignedPermutationReduction, detect_point_group_reduction
        from parsec_python.models import Atom, GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=3.0, expansion_order=4, shift=(0.5,) * 3))
        # water-like C2v in the xz plane: no larger group, the reduction of before
        atoms = [Atom("O", (0.0, 0.0, 0.0)), Atom("H", (0.9, 0.0, 0.6)), Atom("H", (-0.9, 0.0, 0.6))]
        abelian = SignedPermutationReduction.detect(grid, atoms)
        self.assertIs(detect_point_group_reduction(grid, atoms, abelian), abelian)


class DiagonalSubgroupTests(unittest.TestCase):
    def test_the_diagonal_members_of_d2d_are_the_reflection_group_of_the_structure(self):
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Symmetry import (
            AxisReflectionReduction,
            SignedPermutationReduction,
            detect_point_group_reduction,
            diagonal_subgroup_reduction,
        )
        from parsec_python.models import Atom, GridSettings

        a = 1.2
        atoms = [Atom("C", (0.0, 0.0, 0.0))] + [
            Atom("H", tuple(a * np.array(signs, dtype=float)))
            for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        ]
        for shift in (0.5, 0.0):
            with self.subTest(shift=shift):
                grid = build_cluster_grid(GridSettings(spacing=0.5, radius=3.8, expansion_order=4, shift=(shift,) * 3))
                reduction = detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
                diagonal = diagonal_subgroup_reduction(grid, reduction.operations)
                reference = AxisReflectionReduction.detect(grid, atoms)
                np.testing.assert_array_equal(diagonal.signs, reference.signs)
                np.testing.assert_array_equal(diagonal.representative_rows, reference.representative_rows)
                np.testing.assert_array_equal(diagonal.full_to_wedge, reference.full_to_wedge)
                np.testing.assert_array_equal(diagonal.multiplicities, reference.multiplicities)
                # every field on the wedge of D2d is invariant under the subgroup
                field = np.random.default_rng(2).standard_normal(reduction.wedge_size)
                full = field[reduction.full_to_wedge]
                for orbit in range(0, diagonal.wedge_size, 37):
                    members = np.flatnonzero(diagonal.full_to_wedge == orbit)
                    self.assertEqual(np.ptp(full[members]), 0.0)

    def test_a_group_without_diagonal_members_has_none(self):
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Symmetry import diagonal_subgroup_reduction
        from parsec_python.models import GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=3.0, expansion_order=4, shift=(0.5,) * 3))
        cyclic = np.array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]]])
        operations = np.concatenate([np.eye(3, dtype=np.int64)[None], cyclic])
        self.assertIsNone(diagonal_subgroup_reduction(grid, operations))


class SectorMarginTests(unittest.TestCase):
    """The margin of the first counts of weighted sectors, weighed against the devices."""

    def test_the_largest_fraction_that_fits(self):
        from parsec_python.acceleration.Eigensolvers.symmetry import largest_fitting_fraction

        rows = 1000

        def fit(counts):
            # two devices, one sector each, 8 bytes per row and column
            return {0: 8 * rows * counts[0], 1: 8 * rows * counts[1]}

        totals = {0: 8 * rows * 1000, 1: 8 * rows * 1000}
        low, high = [800, 760], [880, 836]
        self.assertEqual(largest_fitting_fraction(fit, totals, 1.0, low, high), 1.0)
        self.assertEqual(largest_fitting_fraction(fit, totals, 0.5, low, high), 0.0)
        fraction = largest_fitting_fraction(fit, totals, 0.84, low, high)
        # sector 0 limits: 800 + f 80 <= 840 -> f <= 0.5
        self.assertLessEqual(fraction, 0.5)
        self.assertGreater(fraction, 0.5 - 1.0 / 64)
        # weighed after one growth (2 % and the increment): sector 0 grown must fit 840
        from parsec_python.acceleration.Eigensolvers.symmetry import regrown_count

        grown = largest_fitting_fraction(fit, totals, 0.84, low, high, increment=6)
        self.assertLessEqual(regrown_count(800 + int(grown * 80), 6), 840)
        self.assertGreater(regrown_count(800 + int((grown + 1.0 / 64) * 80), 6), 840)
        # the full margin that fits is taken as it is
        self.assertEqual(largest_fitting_fraction(fit, totals, 1.0, low, high, increment=6), 1.0)

    def test_the_deficit_of_a_sector_cut_short(self):
        from parsec_python.acceleration.Eigensolvers.symmetry import bracket_deficits

        rng = np.random.default_rng(3)
        # three sectors with even level densities of different size, weights 1, 1, 3
        full = [np.sort(rng.uniform(0.0, 10.0, size)) for size in (1100, 950, 1000)]
        weights = [1, 1, 3]
        requested = 4000
        # the true cutoff and the true count of each sector below it
        merged = sorted((value, w) for values, w in zip(full, weights) for value in values)
        total, cutoff = 0, None
        for value, w in merged:
            total += w
            if total >= requested:
                cutoff = value
                break
        truth = [int(np.sum(values <= cutoff)) for values in full]
        # sector 0 was given 85 % of what it needs, the others all they need and more
        computed = [full[0][: int(0.85 * truth[0])], full[1][: truth[1] + 40], full[2][: truth[2] + 15]]
        deficits = bracket_deficits(computed, weights, requested, [0])
        missing = truth[0] - computed[0].size
        self.assertGreater(deficits[0], 0.6 * missing)
        self.assertLess(deficits[0], 1.6 * missing)
        # Ritz values of a first solve: the uppermost tenth lies above its levels and closer together;
        # the estimate takes the density of the converged middle and is not raised by them.
        squeezed = computed[0].copy()
        top = squeezed.size // 10
        squeezed[-top:] = np.linspace(squeezed[-top], squeezed[-top] + 0.2 * (squeezed[-1] - squeezed[-top]), top)
        squeezed[-top:] += 0.8 * (squeezed[-1] - squeezed[-top])
        squeezed.sort()
        raised = bracket_deficits([squeezed, computed[1], computed[2]], weights, requested, [0])
        self.assertLess(raised[0], 1.6 * missing)
        # a sector that holds its states has no deficit
        self.assertEqual(bracket_deficits(full, weights, requested, [1])[1], 0)

    def test_switches(self):
        from parsec_python.acceleration.Eigensolvers.symmetry import sector_margin_budget, sector_margin_policy

        with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_MARGIN": ""}):
            self.assertEqual(sector_margin_policy(), "auto")
        with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_MARGIN": "fixed"}):
            self.assertEqual(sector_margin_policy(), "fixed")
        with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_MARGIN": "none"}):
            with self.assertRaises(ValueError):
                sector_margin_policy()
        with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_MARGIN_BUDGET": "0.5"}):
            self.assertEqual(sector_margin_budget(), 0.5)
        with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_MARGIN_BUDGET": "1.5"}):
            with self.assertRaises(ValueError):
                sector_margin_budget()


if __name__ == "__main__":
    unittest.main()
