"""CPU-only ownership, halo accounting, and untouched-operator checks."""
import unittest

import numpy as np

from parsec_python.acceleration.experimental.domain_partition import (
    hilbert_keys, make_partition, morton_keys, partition_metrics,
)


def grid_stencil(coords):
    lookup = {tuple(point): index for index, point in enumerate(coords)}
    result = np.full((7, len(coords)), -1, dtype=np.int64)
    result[0] = np.arange(len(coords))
    slot = 1
    for axis in range(3):
        for direction in (-1, 1):
            for row, point in enumerate(coords):
                other = point.copy()
                other[axis] += direction
                result[slot, row] = lookup.get(tuple(other), -1)
            slot += 1
    return result


class DomainPartitionTests(unittest.TestCase):
    def test_hilbert_complete_cubes_are_unique_and_face_adjacent(self):
        for size in (2, 4, 8):
            coords = np.indices((size,) * 3).reshape(3, -1).T
            keys = hilbert_keys(coords)
            np.testing.assert_array_equal(np.sort(keys), np.arange(size ** 3))
            path = coords[np.argsort(keys)]
            np.testing.assert_array_equal(np.abs(np.diff(path, axis=0)).sum(axis=1), 1)
            np.testing.assert_array_equal(np.sort(morton_keys(coords)), np.arange(size ** 3))

    def test_irregular_wedge_balanced_coverage_and_canonical_local_order(self):
        coords = np.indices((7, 6, 5)).reshape(3, -1).T
        coords = coords[(coords[:, 0] >= coords[:, 1]) & (coords.sum(axis=1) < 11)]
        np.random.default_rng(73).shuffle(coords)
        neighbors = grid_stencil(coords)
        for method in ("axis", "brick", "morton", "hilbert"):
            for parts in (1, 3, 16, len(coords)):
                with self.subTest(method=method, parts=parts):
                    result = make_partition(coords, neighbors, parts, method)
                    sizes = [len(rows) for rows in result.local_rows]
                    self.assertLessEqual(max(sizes) - min(sizes), 1)
                    np.testing.assert_array_equal(np.sort(np.concatenate(result.local_rows)), np.arange(len(coords)))
                    for rank, rows in enumerate(result.local_rows):
                        np.testing.assert_array_equal(result.owner[rows], rank)
                        self.assertTrue(np.all(rows[1:] > rows[:-1]))
                    shifted = make_partition(coords - np.array([11, 20, 70]), neighbors, parts, method)
                    np.testing.assert_array_equal(result.owner, shifted.owner)

    def test_missing_entries_and_duplicate_halo_references(self):
        # Rank 0 needs row 2 three times: payload must contain it only once.
        neighbors = np.array([[0, 1, 2, 3], [2, 2, 0, -1], [2, -1, 1, -1]])
        owner = np.array([0, 0, 1, 1])
        halos, stats = partition_metrics(neighbors, owner)
        np.testing.assert_array_equal(halos[0], [2])
        np.testing.assert_array_equal(halos[1], [0, 1])
        self.assertEqual(stats["recv_rows_by_source"], [[0, 1], [2, 0]])
        self.assertEqual(stats["remote_stencil_references_per_rank"], [3, 2])
        self.assertEqual(stats["boundary_rows_per_rank"], [2, 1])
        self.assertEqual(stats["total_payload_bytes_fp64_per_column"], 24)
        self.assertEqual(stats["recv_bytes_fp64_per_column"], [8, 16])
        self.assertEqual(stats["send_bytes_fp64_per_column"], [16, 8])

    def test_directed_long_edges_use_operator_connectivity_not_coordinates(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [100, 0, 0], [101, 0, 0]])
        neighbors = np.array([[0, 1, 2, 3], [3, -1, -1, -1]])
        result = make_partition(coords, neighbors, 2, "axis")
        self.assertEqual(result.metrics["recv_rows_by_source"], [[0, 1], [0, 0]])
        self.assertEqual(result.metrics["recv_neighbor_counts"], [1, 0])
        self.assertEqual(result.metrics["send_neighbor_counts"], [0, 1])
        np.testing.assert_array_equal(result.halo_rows[0], [3])

    def test_remapped_stencil_preserves_original_slot_sum_exactly(self):
        coords = np.indices((4, 5, 3)).reshape(3, -1).T
        neighbors = grid_stencil(coords)
        # Add a nongeometric symmetry-like dependency in an existing slot.
        neighbors[-1, 0] = len(coords) - 1
        rng = np.random.default_rng(19)
        coefficients = rng.normal(size=neighbors.shape)
        x = rng.normal(size=(len(coords), 5))
        reference = np.zeros_like(x)
        for slot in range(len(neighbors)):
            valid = neighbors[slot] >= 0
            reference[valid] += coefficients[slot, valid, None] * x[neighbors[slot, valid]]
        for method in ("axis", "brick", "morton", "hilbert"):
            result = make_partition(coords, neighbors, 7, method)
            actual = np.zeros_like(x)
            for rank, rows in enumerate(result.local_rows):
                stored = np.concatenate((rows, result.halo_rows[rank]))
                inverse = np.full(len(coords), -1, dtype=int)
                inverse[stored] = np.arange(len(stored))
                local_x = x[stored]
                local_y = np.zeros((len(rows), x.shape[1]))
                for slot in range(len(neighbors)):
                    sources = neighbors[slot, rows]
                    valid = sources >= 0
                    self.assertTrue(np.all(inverse[sources[valid]] >= 0))
                    local_y[valid] += coefficients[slot, rows[valid], None] * local_x[inverse[sources[valid]]]
                actual[rows] = local_y
            np.testing.assert_array_equal(actual, reference)

    def test_block_partition_does_not_pad_domain_and_reports_split_blocks(self):
        coords = np.indices((2, 2, 2)).reshape(3, -1).T[:-1]
        neighbors = grid_stencil(coords)
        for method in ("morton", "hilbert"):
            result = make_partition(coords, neighbors, 3, method, tile_size=2)
            self.assertEqual(len(result.owner), 7)
            self.assertEqual(result.metrics["split_blocks"], 1)
            self.assertEqual(result.metrics["block_partition_boundaries"], 2)
            self.assertEqual(result.metrics["row_counts"], [3, 2, 2])

    def test_one_rank_has_no_halo_and_all_rows_interior(self):
        coords = np.indices((2, 3, 4)).reshape(3, -1).T
        result = make_partition(coords, grid_stencil(coords), 1)
        self.assertEqual(result.metrics["total_payload_bytes_fp64_per_column"], 0)
        self.assertEqual(result.metrics["boundary_rows_per_rank"], [0])
        self.assertEqual(result.metrics["interior_rows_per_rank"], [24])
        self.assertEqual(result.halo_rows[0].size, 0)

    def test_validation_prevents_empty_ranks_bad_shapes_and_unsafe_indices(self):
        coords = np.array([[0, 0, 0], [1, 0, 0]])
        neighbors = np.array([[0, 1], [-1, -1]])
        for parts in (0, 3, 1.5, True):
            with self.assertRaises(ValueError):
                make_partition(coords, neighbors, parts)
        for tile in (0, -1, 2.0, True):
            with self.assertRaises(ValueError):
                make_partition(coords, neighbors, 1, tile_size=tile)
        for bad in (np.array([[0, 2]]), np.array([[0, -2]]), np.array([[0., 1.]]), np.zeros((2, 3), dtype=int)):
            with self.assertRaises(ValueError):
                make_partition(coords, bad, 1)
        for bad_coords in (coords.astype(float), np.zeros((0, 3), dtype=int), np.zeros((2, 2), dtype=int)):
            with self.assertRaises(ValueError):
                make_partition(bad_coords, neighbors, 1)
        with self.assertRaises(ValueError):
            make_partition(coords, neighbors, 1, "not-a-method")
        with self.assertRaises(ValueError):
            partition_metrics(neighbors, np.array([0, 0]), nparts=2)
        with self.assertRaises(ValueError):
            partition_metrics(neighbors, np.array([0, 0]), local_rows=(np.array([1, 0]),))

    def test_sfc_overflow_is_rejected(self):
        for keys in (hilbert_keys, morton_keys):
            with self.assertRaises(ValueError):
                keys(np.array([[1 << 21, 0, 0]]))
            with self.assertRaises(ValueError):
                keys(np.array([[-1, 0, 0]]))


if __name__ == "__main__":
    unittest.main()
