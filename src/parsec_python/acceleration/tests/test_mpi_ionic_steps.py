"""CPU protocol tests of the ionic-step commands of the MPI sector context.

A relaxation on several nodes adds three commands to the protocol of the
SCF loop: ``geometry`` (every rank rebuilds what it holds for moved atoms),
``new_scf_loop`` (every sector solver keeps its subspace for the next
structure) and ``forces`` (the nonlocal force terms of the sectors of each
rank, summed on the root).  The real context runs here against toy solvers
on threads that stand for the ranks.
"""
from concurrent.futures import ThreadPoolExecutor
import unittest

import numpy as np

from parsec_python.acceleration.experimental.mpi_scf import MPISCFError, MPISectorContext
from parsec_python.acceleration.tests.test_mpi_domain import _Comm, _World
from parsec_python.acceleration.tests.test_mpi_scf import _make_solver


class MPIIonicStepTests(unittest.TestCase):
    def run_protocol(self, size, root_action, prepare):
        world = _World(size)

        def rank_main(rank):
            context = MPISectorContext(_Comm(world, rank))
            solver, builder = _make_solver(context)
            prepare(rank, solver)
            if rank:
                context.worker_loop(solver, builder)
                return None
            result = root_action(solver)
            context.stop_workers()
            return result

        with ThreadPoolExecutor(max_workers=size) as pool:
            futures = [pool.submit(rank_main, rank) for rank in range(size)]
            return [future.result(timeout=25) for future in futures][0]

    def run_failure(self, prepare, root_action, message):
        world = _World(2)

        def rank_main(rank):
            context = MPISectorContext(_Comm(world, rank))
            solver, builder = _make_solver(context)
            prepare(rank, solver)
            with self.assertRaisesRegex(MPISCFError, message):
                if rank:
                    context.worker_loop(solver, builder)
                else:
                    root_action(solver)
            self.assertTrue(context.failed)

        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(rank_main, rank) for rank in range(2)]:
                future.result(timeout=25)

    def test_moved_atoms_reach_the_handler_of_every_rank(self):
        positions = np.arange(12.0).reshape(4, 3) / 7.0
        for size in (1, 2, 4):
            received = {}

            def prepare(rank, solver):
                solver.geometry_handler = lambda moved, rank=rank: received.__setitem__(rank, moved.copy())

            self.run_protocol(size, lambda solver: solver.mpi_context.execute(solver, "geometry", positions), prepare)
            self.assertEqual(sorted(received), list(range(size)))
            for moved in received.values():
                self.assertEqual(moved.dtype, np.float64)
                np.testing.assert_array_equal(moved, positions)

    def test_a_rank_without_a_geometry_handler_fails_the_command_on_every_rank(self):
        def prepare(rank, solver):
            if rank == 0:
                solver.geometry_handler = lambda moved: None

        self.run_failure(prepare, lambda solver: solver.mpi_context.execute(solver, "geometry", np.zeros((2, 3))),
                         "rank 1: RuntimeError: this rank was prepared without a geometry handler")

    def test_a_new_scf_loop_restarts_the_sector_solvers_of_every_rank(self):
        for size in (1, 2, 4):
            started = []

            def prepare(rank, solver):
                solver._mpi_latest_results = {"stale": rank}
                for local in solver._solvers:
                    if local is not None:
                        local.start_new_scf_loop = lambda sector=local.sector: started.append(sector)

            def root_action(solver):
                state = solver.start_new_scf_loop()
                # Before any solve there is no state to hand to the next loop.
                self.assertIsNone(state)
                return dict(solver._mpi_latest_results)

            self.assertEqual(self.run_protocol(size, root_action, prepare), {})
            self.assertEqual(sorted(started), [0, 1, 2, 3])

    def test_force_terms_of_the_ranks_are_summed_on_the_root(self):
        representations = np.array([0, 1, 0, 2, 3])
        columns = np.array([0, 0, 1, 0, 0])
        occupations = np.array([1.0, 1.0, 0.75, 0.5, 0.0])
        for size in (1, 2, 4):
            seen = {}

            def prepare(rank, solver):
                def local(sectors, sector_columns, weights, rank=rank, solver=solver):
                    seen[rank] = (sectors.copy(), sector_columns.copy(), weights.copy())
                    owned = np.isin(sectors, solver._owned_representations)
                    # One row per projector on every rank; each rank adds what its sectors hold.
                    return np.outer(np.arange(1.0, 7.0), [1.0, -2.0, 0.5]) * weights[owned].sum()

                solver._projector_force_terms_local = local

            total = self.run_protocol(
                size, lambda solver: solver.projector_force_terms(representations, columns, occupations), prepare)
            np.testing.assert_allclose(
                total, np.outer(np.arange(1.0, 7.0), [1.0, -2.0, 0.5]) * occupations.sum(), rtol=1e-15, atol=0.0)
            self.assertEqual(sorted(seen), list(range(size)))
            for sectors, sector_columns, weights in seen.values():
                np.testing.assert_array_equal(sectors, representations)
                np.testing.assert_array_equal(sector_columns, columns)
                np.testing.assert_array_equal(weights, occupations)
                self.assertEqual((sectors.dtype, sector_columns.dtype, weights.dtype), (np.int32, np.int32, np.float64))

    def test_force_terms_that_are_not_finite_fail_the_command_on_every_rank(self):
        def prepare(rank, solver):
            value = np.nan if rank else 1.0
            solver._projector_force_terms_local = lambda *arguments: np.full((6, 3), value)

        self.run_failure(
            prepare,
            lambda solver: solver.projector_force_terms(np.array([0]), np.array([0]), np.array([1.0])),
            "forces failed: rank 1: ValueError: local force terms must be finite")


if __name__ == "__main__":
    unittest.main()
