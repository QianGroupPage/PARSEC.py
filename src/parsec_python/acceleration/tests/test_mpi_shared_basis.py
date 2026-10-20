"""CPU tests of the sector basis that MPI ranks share (molecular dynamics on several nodes).

The exchanges themselves need GPUs and an MPI launch: the check script
``benchmarks/check_mpi_shared_basis.py`` runs them under srun.  Here the
assignment of the one sector, the switch of the MPI context, the row blocks
and exchange rounds, and the launch reading of the runner.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from parsec_python.Eigensolvers.lapack_random import LapackRandom
from parsec_python.acceleration.Eigensolvers.chebyshev import subspace_filter_blocks, uniform_filter_blocks
from parsec_python.acceleration.Eigensolvers.distributed_state import (
    ColumnLayout,
    SectorDeviceGroup,
    interleaved_ranges,
)
from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import (
    _RowExchange,
    _skip,
    exchange_bytes,
    row_edges,
    shared_transport,
)
from parsec_python.acceleration.benchmarks.mpi_full_scf import _ranks_per_node_of_launch
from parsec_python.acceleration.experimental.mpi_scf import MPISCFError, MPISectorContext, sector_device_groups
from parsec_python.acceleration.tests.test_mpi_domain import _Comm, _World


class SharedSectorAssignmentTests(unittest.TestCase):
    def test_every_rank_owns_the_one_sector_on_its_first_device(self):
        for rank in range(4):
            self.assertEqual(sector_device_groups(1, 4, rank, (0,), shared=True), {0: (0,)})
            self.assertEqual(sector_device_groups(1, 4, rank, (2, 3), shared=True), {0: (2,)})
        with self.assertRaisesRegex(ValueError, "only the single sector"):
            sector_device_groups(4, 4, 0, (0,), shared=True)
        # Without the switch the former rule still refuses more ranks than sectors.
        with self.assertRaises(ValueError):
            sector_device_groups(1, 4, 0, (0,))

    def test_the_context_shares_one_sector_among_several_ranks(self):
        for size, representations, shared in ((4, 1, True), (1, 1, False), (2, 4, False)):
            world = _World(size)

            def configure(rank):
                context = MPISectorContext(_Comm(world, rank))
                groups = context.configure(representations, (0,) if representations == 1 else (0, 1))
                return context.shared, context.owned_sectors, groups

            with ThreadPoolExecutor(max_workers=size) as pool:
                results = [future.result(timeout=25) for future in [pool.submit(configure, r) for r in range(size)]]
            for flag, owned, groups in results:
                self.assertEqual(flag, shared)
                if shared:
                    self.assertEqual(owned, (0,))
                    self.assertEqual(groups, {0: (0,)})

    def test_more_ranks_than_several_sectors_still_fails(self):
        world = _World(3)

        def configure(rank):
            context = MPISectorContext(_Comm(world, rank))
            with self.assertRaises(MPISCFError):
                context.configure(2, (0,))

        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in [pool.submit(configure, r) for r in range(3)]:
                future.result(timeout=25)


class ExchangeRoundTests(unittest.TestCase):
    def test_row_blocks_cover_the_rows_once(self):
        for rows, size in ((10, 4), (10007, 16), (3, 4)):
            edges = row_edges(rows, size)
            self.assertEqual(edges[0], 0)
            self.assertEqual(edges[-1], rows)
            heights = [b - a for a, b in zip(edges, edges[1:])]
            self.assertLessEqual(max(heights) - min(heights), 1)

    def test_rounds_move_every_column_of_every_rank_once(self):
        blocks = uniform_filter_blocks(203, 6, 10)
        layout = ColumnLayout(interleaved_ranges(blocks, 4))
        for width in (1, 7, 24, 1000):
            plan = _RowExchange(SimpleNamespace(size=4, rank=1), layout, 10007, width)
            moved = [0] * 4
            for step in range(plan.rounds):
                for rank, count in enumerate(plan.counts(step)):
                    moved[rank] += count
            self.assertEqual(moved, list(layout.widths))
            self.assertEqual(plan.own, row_edges(10007, 4)[2] - row_edges(10007, 4)[1])

    def test_the_later_plans_keep_whole_blocks_on_every_rank(self):
        # The layout of the first solve serves the later passes and every trimmed count.
        blocks = uniform_filter_blocks(1752, 6, 30)
        layout = ColumnLayout(interleaved_ranges(blocks, 16))
        for count in (1752, 1750, 1000, 997):
            leading = layout.leading(count)
            later = subspace_filter_blocks(count, 6, 15, 3)
            self.assertIsNotNone(SectorDeviceGroup._whole_block_partitions(leading, later), count)

    def test_skipping_the_stream_reaches_the_seed_of_the_drawn_values(self):
        drawn = LapackRandom()
        drawn.uniform_0_1(12345)
        self.assertEqual(tuple(_skip(LapackRandom().seed, 12345)), tuple(drawn.seed))


class SettingTests(unittest.TestCase):
    def test_transport_and_buffer_settings(self):
        with patch.dict(os.environ, {"MPICH_GPU_SUPPORT_ENABLED": "1"}, clear=False):
            os.environ.pop("PARSEC_MPI_SHARED_TRANSPORT", None)
            self.assertEqual(shared_transport(), "cuda")
        with patch.dict(os.environ, {"MPICH_GPU_SUPPORT_ENABLED": "0", "PARSEC_MPI_SHARED_TRANSPORT": "auto"}):
            self.assertEqual(shared_transport(), "host")
        with patch.dict(os.environ, {"PARSEC_MPI_SHARED_TRANSPORT": "fast"}):
            with self.assertRaisesRegex(ValueError, "auto, cuda or host"):
                shared_transport()
        with patch.dict(os.environ, {"PARSEC_MPI_SHARED_EXCHANGE_BYTES": "4096"}):
            self.assertEqual(exchange_bytes(), 4096)

    def test_ranks_per_node_of_the_launch(self):
        for value, expected in (("4", 4), ("4(x2)", 4), ("1", 1), ("2,1", 2), ("", 1), ("x", 1)):
            with patch.dict(os.environ, {"SLURM_NTASKS_PER_NODE": value}):
                self.assertEqual(_ranks_per_node_of_launch(), expected)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLURM_NTASKS_PER_NODE", None)
            self.assertEqual(_ranks_per_node_of_launch(), 1)


if __name__ == "__main__":
    unittest.main()
