"""The k-point pools of an MPI run reproduce the one-process solver.

The ranks are threads here (:mod:`.thread_mpi`, the collectives of MPI on host arrays, in a process of
:mod:`.pool_check` with a single-threaded BLAS) and the device code runs on the host (``numpy-device``),
so the checks run without MPI or a GPU:

* the eigensolver of a basis shared by columns (:func:`..Eigensolvers.chebff_columns.solve_column_eigval`)
  gives the eigenvalues and residuals of the one-device solver to round-off, on 1, 2 and 3 ranks, with
  exchanges in many rounds, through a first solve and SUBSPACE steps, and for a converged first solve;
* the pools (:func:`..SCF.kpoint_pools.plan_pools`) are balanced and fit the memory;
* the scalar and the collinear-spin k-point SCF on pools of one and of several ranks give the energy,
  eigenvalues and density of the one-process run.

The same solver under MPI with CuPy runs in the diamond cases of examples/3d_diamond_benchmark/run_mpi_kpoints.py.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from parsec_python.acceleration.Eigensolvers.chebff_columns import ColumnLayout, requested_spans
from parsec_python.acceleration.SCF.kpoint_pools import plan_pools, pool_memory_bytes

_SRC = Path(__file__).resolve().parents[3]


@functools.lru_cache(maxsize=None)
def _check(*arguments: str) -> dict:
    """The ``RESULT`` of a :mod:`.pool_check` run (a process of its own: single-threaded BLAS)."""
    environment = {
        **os.environ,
        "PYTHONPATH": f"{_SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        # one BLAS thread: the package imports NumPy before pool_check runs (see thread_mpi.run_ranks)
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    process = subprocess.run(
        [sys.executable, "-m", "parsec_python.acceleration.tests.pool_check", *arguments],
        capture_output=True, text=True, env=environment, timeout=1500,
    )
    for line in process.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise AssertionError(
        f"no result (exit {process.returncode})\nstdout:\n{process.stdout[-1500:]}\nstderr:\n{process.stderr[-3000:]}"
    )


class ColumnLayoutTests(unittest.TestCase):
    def test_every_column_once_and_both_degree_parts_shared(self) -> None:
        for columns, block, size in ((26, 6, 3), (36, 6, 4), (7, 6, 2), (1042, 6, 16), (5, 6, 1)):
            with self.subTest(columns=columns, block=block, size=size):
                layout = ColumnLayout(columns, block, size)
                held = np.concatenate([layout.global_columns(rank) for rank in range(size)])
                np.testing.assert_array_equal(np.sort(held), np.arange(columns))
                split = ((columns // block + 1) // 2) * block
                for part in ((0, split), (split, columns)):
                    counts = [
                        int(np.count_nonzero((layout.global_columns(r) >= part[0]) & (layout.global_columns(r) < part[1])))
                        for r in range(size)
                    ]
                    self.assertLessEqual(max(counts) - min(counts), 1)
                for rank in range(size):
                    local = 0
                    for first, offset, count in layout.spans(rank):
                        self.assertEqual(offset, local)
                        local += count
                    self.assertEqual(local, layout.widths[rank])

    def test_requested_spans_cut_the_buffer_columns(self) -> None:
        layout = ColumnLayout(26, 6, 3)
        kept = []
        for rank in range(3):
            for first, _, count in requested_spans(layout, rank, 20):
                kept.extend(range(first, first + count))
        self.assertEqual(sorted(kept), list(range(20)))


class ColumnEigensolverTests(unittest.TestCase):
    def _assert(self, report: dict, sizes) -> None:
        for size in sizes:
            with self.subTest(ranks=size):
                entry = report[str(size)]
                self.assertTrue(entry["ranks_equal"])  # every rank of a pool has the same numbers, bit for bit
                self.assertLess(entry["value_error"], 1.0e-11)
                self.assertLess(entry["norm_error"], 1.0e-11)

    def test_first_solve_and_subspace_steps_match_the_one_device_solver(self) -> None:
        self._assert(_check("eigensolver", "chebff", "1,2,3", "4", "route=rows"), (1, 2, 3))

    def test_exchanges_in_many_rounds(self) -> None:
        # rounds of 3 rows per rank: every move goes in hundreds of Alltoallv
        self._assert(_check("eigensolver", "chebff", "3", "2", str(16 * 36 * 3), "route=rows"), (3,))

    def test_smart_subspace_bound(self) -> None:
        self._assert(_check("eigensolver", "chebff-smart", "1,3", "4", "route=rows"), (1, 3))

    def test_converged_first_solve(self) -> None:
        self._assert(_check("eigensolver", "chebdav", "2", "2", "route=rows"), (2,))

    # The block route (PARSEC_KPOINT_RITZ=block): one share of the basis, the same numbers to round-off.
    def test_block_route_first_solve_and_subspace_steps(self) -> None:
        self._assert(_check("eigensolver", "chebff", "1,2,3", "4", "route=block"), (1, 2, 3))

    def test_block_route_exchanges_in_many_rounds(self) -> None:
        # rounds of 3 columns per rank: every move goes in many Alltoallv, the row block in place
        self._assert(_check("eigensolver", "chebff", "3", "2", str(16 * 3 * 2000), "route=block"), (3,))

    def test_block_route_smart_subspace_bound(self) -> None:
        self._assert(_check("eigensolver", "chebff-smart", "1,3", "4", "route=block"), (1, 3))

    def test_block_route_converged_first_solve(self) -> None:
        self._assert(_check("eigensolver", "chebdav", "2", "2", "route=block"), (2,))


def _diamond(dimension: int, working: int):
    """The memory of a diamond cell of ``dimension`` points and ``working`` states (18 points per edge)."""
    return lambda tasks, ranks, resident=None: pool_memory_bytes(dimension, working, tasks, ranks, 6, resident=resident)


class ProjectorBuildTests(unittest.TestCase):
    def test_atoms_dealt_over_the_pool_give_the_whole_build(self) -> None:
        for result in _check("projectors", "2")["ranks"]:
            self.assertEqual(result["error"], 0.0)
            self.assertTrue(result["labels"])
            self.assertTrue(result["signs"])


class PoolPlanTests(unittest.TestCase):
    """The plans of the rows route (three shares of a basis in a Rayleigh-Ritz pass), with the filter chunk of
    64 columns these numbers were set for (the pools take 32 by default)."""

    usable = 0.9 * 80e9  # an A100 of 80 GB

    def setUp(self) -> None:
        from unittest.mock import patch

        guard = patch.dict(os.environ, {"PARSEC_KPOINT_RITZ": "rows", "PARSEC_SPINOR_FILTER_COLUMNS": "64"})
        guard.start()
        self.addCleanup(guard.stop)

    def test_balance_then_more_pools(self) -> None:
        free = lambda tasks, ranks, resident=None: 0  # noqa: E731 - memory no object
        self.assertEqual(plan_pools(4, 16, free, 1.0).pools, 4)  # 4 pools of 4 ranks, one k-point each
        self.assertEqual(plan_pools(4, 4, free, 1.0).pools, 4)
        self.assertEqual(plan_pools(10, 16, free, 1.0).pools, 2)  # 5 tasks on 8 ranks beats 3 on 4
        self.assertEqual(plan_pools(1, 16, free, 1.0).pools, 1)
        self.assertEqual(plan_pools(8, 3, free, 1.0).pools, 1)
        plan = plan_pools(4, 6, free, 1.0)
        self.assertEqual(sum(plan.sizes), 6)
        self.assertFalse(plan.host_storage)
        self.assertEqual(sorted(t for pool in range(plan.pools) for t in plan.tasks_of_pool(pool)), [0, 1, 2, 3])
        self.assertEqual([plan.pool_of_rank(rank) for rank in range(6)], sorted(plan.pool_of_rank(r) for r in range(6)))

    def test_memory_takes_fewer_pools(self) -> None:
        n5, n6 = _diamond(729_000, 2018), _diamond(1_259_712, 3468)
        self.assertEqual(plan_pools(4, 16, n5, self.usable).pools, 4)
        # n = 6 on 16 A100: a pool of four ranks per k-point (69 GB of the 72 the rule allows)
        plan = plan_pools(4, 16, n6, self.usable)
        self.assertEqual((plan.pools, plan.host_storage), (4, False))
        self.assertLess(plan.bytes_per_rank, self.usable)
        self.assertEqual(plan_pools(4, 16, n6, self.usable, requested=1).pools, 1)
        with self.assertRaises(MemoryError):  # four pools of n = 6 do not fit 40 GB GPUs
            plan_pools(4, 16, n6, 0.9 * 40e9, requested=4, limit=40e9)
        # on GPUs of 72 GB four pools of n = 6 (69 GB) are left out, and run when asked for
        self.assertEqual(plan_pools(4, 16, n6, 0.9 * 72e9).pools, 2)
        self.assertEqual(plan_pools(4, 16, n6, 0.9 * 72e9, requested=4, limit=72e9).pools, 4)
        # the chunk of 32 columns, the default, leaves them room there
        from unittest.mock import patch

        with patch.dict(os.environ, {"PARSEC_SPINOR_FILTER_COLUMNS": ""}):
            self.assertEqual(plan_pools(4, 16, n6, 0.9 * 72e9).pools, 4)
        with self.assertRaises(ValueError):
            plan_pools(4, 16, n6, self.usable, requested=5)

    def test_idle_subspaces_on_the_host_where_the_device_cannot_hold_them(self) -> None:
        n7 = _diamond(2_000_376, 5500)
        plan = plan_pools(4, 16, n7, self.usable)  # n = 7 on 16 A100: one pool, two subspaces on the host
        self.assertEqual((plan.pools, plan.host_storage, plan.resident), (1, True, 2))
        self.assertLess(plan.bytes_per_rank, self.usable)
        with self.assertRaises(MemoryError):
            plan_pools(4, 16, n7, self.usable, storage="device")
        self.assertTrue(plan_pools(4, 16, _diamond(729_000, 2018), self.usable, storage="host", requested=1).host_storage)
        with self.assertRaises(MemoryError):  # n = 8 does not fit 16 A100 in any way
            plan_pools(4, 16, _diamond(2_985_984, 8198), self.usable)


class BlockPoolPlanTests(unittest.TestCase):
    """The plans of the block route (PARSEC_KPOINT_RITZ=block): one share of the basis in hand."""

    usable = 0.9 * 80e9

    def setUp(self) -> None:
        from unittest.mock import patch

        guard = patch.dict(os.environ, {"PARSEC_KPOINT_RITZ": "block"})
        guard.start()
        self.addCleanup(guard.stop)

    def test_one_share_lets_more_pools_fit(self) -> None:
        n6, n7 = _diamond(1_259_712, 3468), _diamond(2_000_376, 5500)
        with unittest.mock.patch.dict(os.environ, {"PARSEC_KPOINT_RITZ": "rows"}):
            rows_n6 = plan_pools(4, 16, n6, self.usable)
            rows_n7 = plan_pools(4, 16, n7, self.usable)
        block_n6 = plan_pools(4, 16, n6, self.usable)
        self.assertEqual(block_n6.pools, rows_n6.pools)
        self.assertLess(block_n6.bytes_per_rank, 0.6 * rows_n6.bytes_per_rank)
        # n = 7 on 16 A100: a pool of four ranks (one node) per k-point instead of one pool of 16 ranks
        self.assertEqual((rows_n7.pools, rows_n7.host_storage), (1, True))
        block_n7 = plan_pools(4, 16, n7, self.usable)
        self.assertEqual((block_n7.pools, block_n7.host_storage), (4, False))
        self.assertLess(block_n7.bytes_per_rank, self.usable)


class AutoRouteTests(unittest.TestCase):
    """PARSEC_KPOINT_RITZ=auto: the block route where the rows route needs most of a device or plans worse."""

    usable = 0.9 * 80e9

    @staticmethod
    def _plans(dimension: int, working: int, usable: float):
        from parsec_python.acceleration.SCF.kpoint_pools import choose_route

        def plan(route):
            model = lambda tasks, ranks, resident=None: pool_memory_bytes(  # noqa: E731
                dimension, working, tasks, ranks, 6, resident=resident, route=route)
            return plan_pools(4, 16, model, usable)

        try:
            rows = plan("rows")
        except MemoryError:
            rows = None
        block = plan("block")
        return rows, block, choose_route(rows, block, usable)

    def test_the_route_of_each_size(self) -> None:
        # n = 5: rows fits a third of the device in four pools and was the faster (60.7 against 64.2 s)
        rows, block, route = self._plans(729_000, 2018, self.usable)
        self.assertEqual((rows.pools, block.pools, route), (4, 4, "rows"))
        # n = 6: the same plan, rows near the whole device -> block (205 against 215 s)
        rows, block, route = self._plans(1_259_712, 3468, self.usable)
        self.assertEqual((rows.pools, block.pools, route), (4, 4, "block"))
        self.assertGreater(rows.bytes_per_rank, 0.5 * self.usable)
        # n = 7: rows in one pool with subspaces on the host, block in four pools -> block (621 against 860 s)
        rows, block, route = self._plans(2_000_376, 5500, self.usable)
        self.assertEqual((rows.pools, rows.host_storage, block.pools, route), (1, True, 4, "block"))
        # n = 8: only the block route fits
        rows, block, route = self._plans(2_985_984, 8198, self.usable)
        self.assertIsNone(rows)
        self.assertEqual(route, "block")

    def test_the_setting(self) -> None:
        from unittest.mock import patch

        from parsec_python.acceleration.Eigensolvers.chebff_columns import kpoint_ritz_route

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_KPOINT_RITZ", None)
            self.assertEqual(kpoint_ritz_route(), "auto")
        for value in ("rows", "block", "auto"):
            with patch.dict(os.environ, {"PARSEC_KPOINT_RITZ": value}):
                self.assertEqual(kpoint_ritz_route(), value)
        with patch.dict(os.environ, {"PARSEC_KPOINT_RITZ": "inplace"}):
            with self.assertRaises(ValueError):
                kpoint_ritz_route()


class PooledSCFTests(unittest.TestCase):
    def _check(self, name: str, ranks: int, pools: int, tolerance: float, route: str = "rows") -> None:
        report = _check("scf", name, str(ranks), str(pools), f"route={route}")
        serial = report["serial"]
        for result in report["pooled"]:
            self.assertTrue(result["converged"])
            self.assertEqual(result["iterations"], serial["iterations"])
            self.assertAlmostEqual(result["energy"], serial["energy"], delta=tolerance)
            self.assertAlmostEqual(result["fermi"], serial["fermi"], delta=tolerance)
        for result in report["pooled"][1:]:  # the replicated drivers agree bit for bit
            self.assertEqual(result["energy"], report["pooled"][0]["energy"])
            self.assertEqual(result["density"], report["pooled"][0]["density"])
        np.testing.assert_allclose(report["pooled"][0]["density"], serial["density"], atol=100 * tolerance)

    def test_scalar_pools_of_one_rank_give_the_one_process_numbers(self) -> None:
        # four k-points on three ranks: pools of one rank, one of them with two k-points
        self._check("scalar", ranks=3, pools=3, tolerance=1.0e-9)

    def test_scalar_one_pool_shares_every_basis(self) -> None:
        self._check("scalar", ranks=3, pools=1, tolerance=1.0e-8)

    def test_spin_two_pools_of_two_ranks(self) -> None:
        self._check("spin_polarized", ranks=4, pools=2, tolerance=1.0e-8)

    def test_block_route_scalar_one_pool_shares_every_basis(self) -> None:
        self._check("scalar", ranks=3, pools=1, tolerance=1.0e-8, route="block")

    def test_block_route_scalar_pools_of_one_rank(self) -> None:
        self._check("scalar", ranks=3, pools=3, tolerance=1.0e-9, route="block")

    def test_block_route_spin_two_pools_of_two_ranks(self) -> None:
        self._check("spin_polarized", ranks=4, pools=2, tolerance=1.0e-8, route="block")


if __name__ == "__main__":
    unittest.main()
