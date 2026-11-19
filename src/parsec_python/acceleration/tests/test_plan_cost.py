"""The cost of the weighted point-group sectors (stream B, plan cost).

The partners of the weighted states (:meth:`PointGroupReduction.partner_states`, the full-grid orbitals of a
result) come from the first operation of every left coset of the subgroup (``PARSEC_PARTNER_STATE_COSETS``): the
same states, bit for bit, as the former loop over every operation of the full group, which took 67 of the 80 s of
the SCF of Si 4x4x4 on one A100 node through the driver (6 s now).  The weighted plan of a cell solves no more rows
x states than the abelian group with translations that the cell takes otherwise, in fewer sectors.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from parsec_python.acceleration.Symmetry import axis_reflection
from parsec_python.acceleration.Symmetry.axis_reflection import (
    PointGroupReduction,
    SignedPermutationReduction,
    detect_point_group_reduction,
)
from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
from parsec_python.models import Atom


def _cluster_td():
    from parsec_python.Grid import build_cluster_grid
    from parsec_python.models import GridSettings

    a = 1.2
    atoms = [Atom("C", (0.0, 0.0, 0.0))] + [
        Atom("H", tuple(a * np.array(signs, dtype=float)))
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    ]
    grid = build_cluster_grid(GridSettings(spacing=0.5, radius=3.8, expansion_order=4, shift=(0.5,) * 3))
    return grid, atoms


def _silicon(repeats, points=10, setting="auto"):
    from parsec_python.acceleration.tests.test_periodic_translations import _supercell

    grid, atoms = _supercell(repeats, points)
    return grid, atoms, setting


def _moved_diamond():
    from parsec_python.acceleration.tests.test_orbital_groups import _periodic_model
    from parsec_python.acceleration.tests.test_periodic_symmetry_detection import _diamond

    moved, sides = _diamond((2, 1, 1), offset=0.75 * np.array((1.0, 2.0, 3.0)))
    grid, _ = _periodic_model(sides, 16, 0.0, moved, {"C": 2.0})
    return grid, moved


def _cases():
    grid, atoms = _cluster_td()
    yield "Td cluster", grid, atoms, {}
    for repeats, setting in (((1, 1, 1), "half"), ((1, 1, 1), "auto"), ((2, 1, 1), "auto"), ((2, 2, 2), "auto")):
        grid, atoms, _ = _silicon(repeats, 10, setting)
        yield f"Si {repeats} {setting}", grid, atoms, {"PARSEC_PERIODIC_TRANSLATIONS": setting}
    grid, atoms = _moved_diamond()
    yield "moved diamond 2x1x1", grid, atoms, {}


def _random_sector_states(decomposition, rng, per_sector=2):
    columns, weights = [], []
    for sector, weight in enumerate(decomposition.sector_weights):
        for _ in range(per_sector):
            values = rng.standard_normal(decomposition.sector_size(sector))
            columns.append(decomposition.expand_vector(values, sector))
            weights.append(int(weight))
    return np.column_stack(columns), np.asarray(weights, dtype=np.int64)


class PlanAgainstAbelianTranslationsTests(unittest.TestCase):
    """The weighted plan solves no more than the abelian group with translations that the cell takes otherwise.

    On A100 the time of an SCF iteration followed the sectors per device times their rows x states, the same per
    sector for both routes (Si 4x4x4: 10 weighted sectors 0.57 s per iteration, 32 abelian ones 1.71 s; Si 2x1x1,
    4x1x1 and 3x3x2, whose plans solve 3/4 of the rows x states of the abelian group with translations: 0.74 to
    0.82 of its time per iteration).  A plan with fewer sectors and no more rows x states is therefore not slower.
    """

    def test_fewer_sectors_and_no_more_rows_times_states(self) -> None:
        from parsec_python.acceleration.Symmetry.axis_reflection import periodic_abelian_reduction

        def work(grid, reduction):
            decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
            sizes = np.asarray(decomposition.sector_sizes, dtype=np.float64)
            weights = np.asarray(decomposition.sector_weights, dtype=np.float64)
            return float(np.sum(sizes) / grid.size / weights.sum()), sizes.size

        for repeats in ((1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 1, 1), (2, 2, 2)):
            with self.subTest(repeats=repeats):
                grid, atoms, _ = _silicon(repeats, 10)
                origin = SignedPermutationReduction.detect(grid, atoms)
                weighted = detect_point_group_reduction(grid, atoms, origin)
                self.assertIsInstance(weighted, PointGroupReduction)
                abelian = periodic_abelian_reduction(grid, atoms, origin)
                self.assertGreater(abelian.group_order, origin.group_order)
                weighted_work, weighted_sectors = work(grid, weighted)
                abelian_work, abelian_sectors = work(grid, abelian)
                self.assertLessEqual(weighted_work, abelian_work * (1.0 + 1.0e-9))
                self.assertLess(weighted_sectors, abelian_sectors)


class PartnerStateTests(unittest.TestCase):
    def test_the_cosets_give_the_partners_of_before_bit_for_bit(self) -> None:
        rng = np.random.default_rng(7)
        for label, grid, atoms, environment in _cases():
            with self.subTest(label), mock.patch.dict(os.environ, environment):
                reduction = detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
                self.assertIsInstance(reduction, PointGroupReduction)
                decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                if not np.any(decomposition.sector_weights > 1):
                    continue
                states, weights = _random_sector_states(decomposition, rng)
                with mock.patch.dict(os.environ, {"PARSEC_PARTNER_STATE_COSETS": "off"}):
                    before = reduction.partner_states(states, weights)
                calls = []
                original = axis_reflection._grid_row_mapping_operation

                def counted(*args, **kwargs):
                    calls.append(1)
                    return original(*args, **kwargs)

                with mock.patch.object(axis_reflection, "_grid_row_mapping_operation", counted):
                    after = reduction.partner_states(states, weights)
                self.assertEqual(after.shape, (grid.size, int(weights.sum())))
                self.assertTrue(np.array_equal(before, after), label)
                # at most one row map per left coset other than H
                self.assertLessEqual(len(calls), reduction.coset_count - 1, label)
                # orthogonal partners of the norm of their state
                column = 0
                for index, weight in enumerate(weights):
                    block = after[:, column:column + weight]
                    norm = float(states[:, index] @ states[:, index])
                    np.testing.assert_allclose(block.T @ block, norm * np.eye(weight), atol=1e-10 * norm)
                    column += int(weight)

    def test_the_candidates_are_the_first_operation_of_every_left_coset(self) -> None:
        grid, atoms, _ = _silicon((2, 2, 2))
        reduction = detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
        positions = list(range(1, reduction.full_operations.shape[0]))
        candidates = reduction._partner_candidates(positions)
        # [G:H] - 1 left cosets besides H (192 / 32 = 6)
        self.assertEqual(len(candidates), reduction.coset_count - 1)
        self.assertEqual(len(set(candidates)), len(candidates))

    def test_the_switch(self) -> None:
        with mock.patch.dict(os.environ, {"PARSEC_PARTNER_STATE_COSETS": "off"}):
            self.assertEqual(axis_reflection.partner_state_cosets_setting(), "off")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_PARTNER_STATE_COSETS", None)
            self.assertEqual(axis_reflection.partner_state_cosets_setting(), "auto")
        with mock.patch.dict(os.environ, {"PARSEC_PARTNER_STATE_COSETS": "yes"}):
            with self.assertRaises(ValueError):
                axis_reflection.partner_state_cosets_setting()


if __name__ == "__main__":
    unittest.main()
