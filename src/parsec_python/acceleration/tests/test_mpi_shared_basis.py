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
    shared_ritz_route,
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
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_MPI_SHARED_RITZ", None)
            # auto: the route of the group, the two row blocks unless it names rounds (a periodic cell).
            self.assertEqual(shared_ritz_route(), "rows")
            self.assertEqual(shared_ritz_route("rounds"), "rounds")
            with self.assertRaisesRegex(ValueError, "rows or rounds"):
                shared_ritz_route("slabs")
        for value, default, expected in (("rounds", "rows", "rounds"), ("rows", "rounds", "rows"),
                                         ("auto", "rounds", "rounds"), ("auto", "rows", "rows")):
            with patch.dict(os.environ, {"PARSEC_MPI_SHARED_RITZ": value}):
                self.assertEqual(shared_ritz_route(default), expected)
        with patch.dict(os.environ, {"PARSEC_MPI_SHARED_RITZ": "slabs"}):
            with self.assertRaisesRegex(ValueError, "auto, rows or rounds"):
                shared_ritz_route()

    def test_ranks_per_node_of_the_launch(self):
        for value, expected in (("4", 4), ("4(x2)", 4), ("1", 1), ("2,1", 2), ("", 1), ("x", 1)):
            with patch.dict(os.environ, {"SLURM_NTASKS_PER_NODE": value}):
                self.assertEqual(_ranks_per_node_of_launch(), expected)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLURM_NTASKS_PER_NODE", None)
            self.assertEqual(_ranks_per_node_of_launch(), 1)


class _ThreadWorld:
    """Ranks as threads of one process: the collectives of :class:`MPIShardGroup` on host buffers."""

    def __init__(self, size):
        import threading

        self.size = size
        self.barrier = threading.Barrier(size, timeout=60)
        self.slots = [None] * size
        self.condition = threading.Condition()
        self.messages = {}

    def post(self, key, value):
        with self.condition:
            self.messages.setdefault(key, []).append(value)
            self.condition.notify_all()

    def take(self, key):
        with self.condition:
            if not self.condition.wait_for(lambda: self.messages.get(key), timeout=60):
                raise TimeoutError(f"no message {key}")
            return self.messages[key].pop(0)

    def share(self, rank, value):
        self.slots[rank] = value
        self.barrier.wait()
        values = list(self.slots)
        self.barrier.wait()
        return values


class _ThreadComm:
    def __init__(self, world, rank):
        self.world, self.rank, self.size = world, rank, world.size

    def bcast(self, value, root=0):
        import copy

        return copy.deepcopy(self.world.share(self.rank, value)[root])

    def allreduce(self, value):
        return sum(self.world.share(self.rank, value))

    def send(self, value, dest, tag=0):
        import copy

        self.world.post((self.rank, dest, tag), copy.deepcopy(value))

    def recv(self, source, tag=0):
        return self.world.take((source, self.rank, tag))

    def Send(self, buffer, dest, tag=0):
        self.world.post((self.rank, dest, tag), buffer.copy())

    def Recv(self, buffer, source, tag=0):
        buffer[...] = self.world.take((source, self.rank, tag))

    def Bcast(self, buffer, root=0):
        buffer[...] = self.world.share(self.rank, buffer.copy())[root]

    def Allreduce(self, send, receive):
        values = self.world.share(self.rank, send.copy())
        total = values[0].copy()
        for value in values[1:]:
            total += value
        receive[...] = total

    def Alltoallv(self, send_spec, receive_spec):
        send, (send_counts, send_offsets) = send_spec[0], send_spec[1]
        receive, (receive_counts, receive_offsets) = receive_spec[0], receive_spec[1]
        shared = self.world.share(self.rank, (send.copy(), send_counts, send_offsets))
        for source, (buffer, counts, offsets) in enumerate(shared):
            count = int(counts[self.rank])
            if count != int(receive_counts[source]):
                raise RuntimeError("send and receive counts differ")
            start = int(receive_offsets[source])
            receive[start : start + count] = buffer[int(offsets[self.rank]) : int(offsets[self.rank]) + count]


class SharedRitzRoundsTests(unittest.TestCase):
    """The Rayleigh--Ritz step of a shared basis in rounds (PARSEC_MPI_SHARED_RITZ=rounds) against the two row
    blocks, with ranks as threads on one device and the host-staged transport."""

    rows, states, size = 1201, 97, 3

    def setUp(self):
        from parsec_python.acceleration.backends.cupy import cupy_available

        if not cupy_available():
            self.skipTest("CuPy and a CUDA device are required")
        import numpy as np
        import scipy.sparse as sp

        generator = np.random.default_rng(11)
        diagonal = generator.uniform(-2.0, 6.0, self.rows)
        off = generator.uniform(-1.0, 1.0, self.rows - 1)
        self.hamiltonian = sp.diags((off, diagonal, off), (-1, 0, 1), format="csr")
        self.matrix = generator.standard_normal((self.rows, self.states))

    def run_ranks(self, route, matrix, *, per_round=None, helper=None, failing_rank=None, world=None):
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import numpy as np

        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import MPIShardGroup, SharedBasis
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import GeneralizedRitzStabilityError

        world = _ThreadWorld(self.size) if world is None else world
        blocks = uniform_filter_blocks(self.states, 6, 10)

        class Operator:
            def __init__(self, hamiltonian):
                self.effective_potential = cp.zeros(1)
                self.hamiltonian = csp.csr_matrix(hamiltonian)

            def apply_into(self, block, out):
                out[...] = self.hamiltonian @ block

        class Failing(Operator):
            def apply_into(self, block, out):
                raise RuntimeError("the operator of this rank failed")

        def run(rank):
            with cp.cuda.Device(0):
                operator = (Failing if rank == failing_rank else Operator)(self.hamiltonian)
                group = MPIShardGroup(_ThreadComm(world, rank), operator, transport="host")
                layout = group.column_layout(blocks)
                local = np.concatenate([matrix[:, c : c + n] for c, _o, n in layout.spans(rank)], axis=1)
                basis = SharedBasis(group, cp.asfortranarray(cp.asarray(local)), layout)
                try:
                    values, rotated, _whitened = group.ritz(operator, basis, stages=False)
                except GeneralizedRitzStabilityError as error:
                    return "refused", local, cp.asnumpy(error.filtered.local), layout
                return np.asarray(values), local, cp.asnumpy(rotated.local), layout

        environment = {"PARSEC_MPI_SHARED_RITZ": route}
        if helper is not None:
            environment["PARSEC_CUPY_RITZ_CONDITION_HELPER"] = helper
        if per_round is not None:
            environment["PARSEC_MPI_SHARED_EXCHANGE_BYTES"] = str(8 * self.rows * per_round)
        with patch.dict(os.environ, environment):
            if per_round is None:
                os.environ.pop("PARSEC_MPI_SHARED_EXCHANGE_BYTES", None)
            with ThreadPoolExecutor(self.size) as pool:
                return list(pool.map(run, range(self.size)))

    @staticmethod
    def whole(results):
        import numpy as np

        layout = results[0][3]
        basis = np.empty((results[0][2].shape[0], layout.columns))
        for rank, (_values, _local, rotated, _layout) in enumerate(results):
            for column, offset, count in layout.spans(rank):
                basis[:, column : column + count] = rotated[:, offset : offset + count]
        return basis

    def test_the_rounds_give_the_ritz_pairs_of_the_two_row_blocks(self):
        import numpy as np
        from scipy.linalg import eigh

        overlap = self.matrix.T @ self.matrix
        projection = self.matrix.T @ (self.hamiltonian @ self.matrix)
        expected = eigh(projection, overlap, eigvals_only=True)
        reference = self.run_ranks("rows", self.matrix)
        reference_basis = self.whole(reference)
        for per_round in (None, 5, 1):
            with self.subTest(per_round=per_round):
                results = self.run_ranks("rounds", self.matrix, per_round=per_round)
                for values, *_ in results:
                    np.testing.assert_allclose(values, reference[0][0], rtol=0.0, atol=1.0e-11 * np.abs(expected).max())
                    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1.0e-10 * np.abs(expected).max())
                basis = self.whole(results)
                # Ritz vectors of a simple spectrum: the same up to their signs.
                signs = np.sign(np.sum(basis * reference_basis, axis=0))
                np.testing.assert_allclose(basis * signs, reference_basis, atol=1.0e-9)
                np.testing.assert_allclose(basis.T @ basis, np.eye(self.states), atol=1.0e-10)

    def test_the_condition_number_on_the_next_rank_changes_no_bit(self):
        import numpy as np

        for route in ("rows", "rounds"):
            with self.subTest(route=route):
                beside = self.run_ranks(route, self.matrix, per_round=5)
                alone = self.run_ranks(route, self.matrix, per_round=5, helper="off")
                for ours, theirs in zip(beside, alone):
                    np.testing.assert_array_equal(ours[0], theirs[0])
                    np.testing.assert_array_equal(ours[2], theirs[2])

    def test_an_error_of_one_rank_goes_to_the_failure_handler(self):
        import threading

        from parsec_python.acceleration.Eigensolvers import mpi_shared_basis

        world = _ThreadWorld(self.size)
        calls, lock = [], threading.Lock()

        def handler(group, error):
            with lock:
                calls.append((group.rank, type(error).__name__, str(error)))
            # The other ranks wait at a collective; a real handler ends the job (MPI Abort).
            world.barrier.abort()

        mpi_shared_basis.set_failure_handler(handler)
        try:
            for route in ("rows", "rounds"):
                with self.subTest(route=route):
                    calls.clear()
                    world.barrier.reset()
                    with self.assertRaises(Exception):
                        self.run_ranks(route, self.matrix, per_round=5, failing_rank=1, world=world)
                    self.assertEqual(calls[0], (1, "RuntimeError", "the operator of this rank failed"))
        finally:
            mpi_shared_basis.set_failure_handler(None)

    def test_a_refused_step_leaves_the_filtered_columns_intact(self):
        import numpy as np

        # Two columns alike: an overlap far beyond the condition limit.
        matrix = self.matrix.copy()
        matrix[:, 40] = matrix[:, 3] + 1.0e-9 * matrix[:, 40]
        from parsec_python.acceleration.Eigensolvers import mpi_shared_basis

        calls = []
        # A refusal that every rank raises together is no failure of some ranks.
        mpi_shared_basis.set_failure_handler(lambda group, error: calls.append(error))
        try:
            for route in ("rows", "rounds"):
                for helper in ("auto", "off"):
                    with self.subTest(route=route, helper=helper):
                        for values, local, filtered, _layout in self.run_ranks(route, matrix, per_round=5, helper=helper):
                            self.assertEqual(values, "refused")
                            np.testing.assert_array_equal(filtered, local)
        finally:
            mpi_shared_basis.set_failure_handler(None)
        self.assertEqual(calls, [])


class ShiftedCholeskyTests(unittest.TestCase):
    """The three shifted Cholesky QR passes of the fallback of a shared basis, on the host and on the device."""

    @staticmethod
    def basis(rows=3000, columns=200, condition=1.0e11):
        import numpy as np

        generator = np.random.default_rng(3)
        left, _ = np.linalg.qr(generator.standard_normal((rows, columns)))
        right, _ = np.linalg.qr(generator.standard_normal((columns, columns)))
        return (left * np.logspace(0, -np.log10(condition), columns)) @ right.T

    def test_the_host_passes_are_those_of_before(self):
        import numpy as np

        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import _host_shifted_cholesky

        matrix = self.basis(400, 50, 1.0e6)
        overlap = matrix.T @ matrix
        # The former statements of the fallback, operation for operation.
        expected = 0.5 * (overlap + overlap.T)
        shift = 11.0 * (400 * 50 + 50 * 51) * np.finfo(float).eps
        expected = np.linalg.cholesky(expected + shift * np.linalg.norm(expected, 2) * np.eye(50))
        np.testing.assert_array_equal(_host_shifted_cholesky(overlap, 400, shifted=True), expected)

    def test_the_device_passes_orthonormalize_an_ill_conditioned_basis_in_place(self):
        from parsec_python.acceleration.backends.cupy import cupy_available

        if not cupy_available():
            self.skipTest("CuPy and a CUDA device are required")
        import cupy as cp
        import numpy as np

        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import (
            _device_shifted_cholesky,
            _solve_lower_transpose_in_place,
        )

        matrix = self.basis()
        rows = cp.asfortranarray(cp.asarray(matrix))
        pointer = rows.data.ptr
        for step in range(3):
            factor = _device_shifted_cholesky(rows.T @ rows, matrix.shape[0], shifted=step == 0)
            _solve_lower_transpose_in_place(factor, rows)
        self.assertEqual(rows.data.ptr, pointer)
        result = cp.asnumpy(rows)
        np.testing.assert_allclose(result.T @ result, np.eye(matrix.shape[1]), atol=1.0e-13)
        # The same column space.
        self.assertLess(np.linalg.norm(matrix - result @ (result.T @ matrix)) / np.linalg.norm(matrix), 1.0e-13)
        with self.assertRaises(np.linalg.LinAlgError):
            _device_shifted_cholesky(-cp.eye(4), 4, shifted=False)

    def test_the_condition_number_of_the_small_solve_comes_from_the_device(self):
        from parsec_python.acceleration.backends.cupy import cupy_available

        if not cupy_available():
            self.skipTest("CuPy and a CUDA device are required")
        import cupy as cp
        import numpy as np

        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import _device_condition_number
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import (
            GeneralizedRitzStabilityError,
            solve_whitened_ritz_on_device,
        )

        for condition in (1.0e3, 1.0e5):
            matrix = cp.asarray(self.basis(600, 40, condition))
            overlap = matrix.T @ matrix
            self.assertAlmostEqual(_device_condition_number(overlap)() / condition**2, 1.0, places=4)
        # Within the limit (1e8) the coefficients are those of the solve that estimates by itself.
        matrix = cp.asarray(self.basis(600, 40, 1.0e3))
        overlap, projection = matrix.T @ matrix, matrix.T @ (2.0 * matrix)
        ours = solve_whitened_ritz_on_device(overlap, projection, condition=_device_condition_number)
        theirs = solve_whitened_ritz_on_device(overlap, projection)
        for left, right in zip(ours, theirs):
            np.testing.assert_array_equal(cp.asnumpy(left), cp.asnumpy(right))
        # Beyond it the solve is refused.
        matrix = cp.asarray(self.basis(600, 40, 1.0e5))
        with self.assertRaises(GeneralizedRitzStabilityError):
            solve_whitened_ritz_on_device(matrix.T @ matrix, matrix.T @ matrix, condition=_device_condition_number)
        self.assertEqual(_device_condition_number(-cp.eye(3))(), float("inf"))


if __name__ == "__main__":
    unittest.main()
