"""A sector basis shared by several CUDA devices must reproduce one device."""
import gc
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy import CuPyHamiltonian, cupy_available, require_cupy


def _device_count():
    if not cupy_available():
        return 0
    cp, _ = require_cupy()
    return int(cp.cuda.runtime.getDeviceCount())


_COMMON = dict(PARSEC_CUPY_GENERALIZED_RITZ='on', PARSEC_CUPY_MIXED_FILTER='off', PARSEC_CUPY_DISTRIBUTED_FILTER='0',
               PARSEC_CUPY_FILTER_GRAPHS='1', PARSEC_CUPY_FILTER_COLUMN_MAJOR='1', PARSEC_CUPY_RITZ_ROTATION='reuse',
               PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='1', PARSEC_CUPY_STREAMING_RITZ='0')


def _hamiltonian(n, **options):
    """A small Hamiltonian with projectors on the current device."""
    return CuPyHamiltonian(sp.diags((-np.ones(n-1), 2.2*np.ones(n), -np.ones(n-1)), (-1, 0, 1)),
                           np.linspace(-.1, .1, n),
                           (sp.csr_matrix(np.random.default_rng(8).normal(size=(n, 2))*.002), np.array([1., -1.])),
                           retain_generic_laplacian=False, **options)


def _held(layout, index):
    """The columns of the basis in the block of device ``index``, in the order in which the block holds them."""
    return [column for start, stop in layout.parts[index] for column in range(start, stop)]


@unittest.skipUnless(_device_count() >= 1, 'a CUDA device required')
class SharedPoolTests(unittest.TestCase):
    """What the group of a shared basis leaves in the memory pool, on one device that stands for the group."""

    def test_a_workspace_that_a_pass_has_outgrown_goes_back_to_the_driver_before_the_larger_one_is_taken(self):
        from parsec_python.acceleration.Eigensolvers.distributed_state import SectorDeviceGroup
        cp, _ = require_cupy()
        cp.cuda.Device(0).use()
        pool, mib = cp.get_default_memory_pool(), 1 << 20
        name = 'PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE'

        def passes(*needs, **environment):
            """Megabytes that the pool of the device holds while each pass has its workspace of ``needs`` elements."""
            # The pool keeps its free lists by the handle of a stream, and the stream of a new group can be
            # given the handle of one that is gone: nothing of an earlier group or test is left for it.
            gc.collect()
            pool.free_all_blocks()
            group = SectorDeviceGroup(_hamiltonian(257), (0,))
            held = []
            with patch.dict(os.environ):
                os.environ.pop(name, None)
                os.environ.update(environment)
                for need in needs:
                    work = group._workspace_capacity(need)
                    (taken,) = group.map(lambda index, device: group._slab_workspace(cp, device, work))
                    self.assertEqual((taken.size, int(taken.device.id)), (work, 0))
                    held.append(int(pool.total_bytes()) // mib)
                    # A pass gives its workspace back before the next one asks; the thread of the device
                    # refers to the function it ran last until it runs another.
                    del taken
                    group.map(lambda index, device: None)
            return [later - held[0] for later in held]

        # Workspaces of 8 MiB, then 16 and 24.  A pass that needs no more finds its block in the free list of
        # the stream; one that needs more first returns that list to the driver, so the pool holds the larger
        # workspace in place of the smaller.
        self.assertEqual(passes(mib, mib, mib // 2, 2 * mib, 2 * mib, 3 * mib), [0, 0, 0, 8, 8, 16])
        # Where the density step empties the pools the group returns nothing: here, where no density step
        # follows a pass, the outgrown workspaces stay beside the larger ones.
        self.assertEqual(passes(mib, mib, mib // 2, 2 * mib, 2 * mib, 3 * mib, **{name: '1'}), [0, 0, 0, 16, 16, 40])


@unittest.skipUnless(_device_count() >= 2, 'at least two CUDA devices required')
class DistributedStateTests(unittest.TestCase):
    rows = 257

    def setUp(self):
        from parsec_python.acceleration.Eigensolvers.distributed_state import SectorDeviceGroup
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()
        self.op = _hamiltonian(self.rows)
        self.devices = tuple(range(min(4, _device_count())))
        self.op.distributed_filter_devices = self.devices
        self.group = SectorDeviceGroup(self.op, self.devices)

    def tearDown(self):
        # Groups, operators and filter graphs that a test left in a reference cycle are destroyed here, between
        # the tests, and not by a collection inside the next one.
        gc.collect()

    def _empty(self, *pools):
        """Hand back to the driver what ``pools`` hold on every device; the arrays taken from them are to be gone."""
        cp = self.cp
        # The thread of a device refers to the function it ran last, and through it to the blocks that function
        # used, until it runs another.  The devices share their threads among all groups.
        self.group.map(lambda index, device: None)
        # An array in a reference cycle is released only when it is collected.
        gc.collect()
        for pool in pools:
            for device in self.devices:
                with cp.cuda.Device(device):
                    pool.free_all_blocks()

    def _subspace(self, left, right, count):
        np.testing.assert_allclose(np.linalg.svd(left.T @ right, compute_uv=False), np.ones(count), atol=5e-9)

    def _ritz_vectors(self, op, values, vectors):
        """The columns of ``vectors`` are orthonormal Ritz vectors of ``op`` in the order of ``values``."""
        cp = self.cp
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(len(values)), atol=1e-9)
        # The Ritz values come from the small solve: only this shows that each column returned to its place.
        applied = cp.asnumpy(op @ cp.array(vectors, order='F'))
        np.testing.assert_allclose(vectors.T @ applied, np.diag(values), rtol=0, atol=1e-9)

    def test_layout_changes_move_every_entry_exactly(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import subspace_filter_blocks, uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import DistributedBasis
        cp, group = self.cp, self.group
        host = np.random.default_rng(31).normal(size=(self.rows, 23))
        full = cp.array(host, order='F')
        ranges = group.column_ranges(uniform_filter_blocks(23, 6, 5))
        self.assertEqual((ranges[0][0], ranges[-1][1]), (0, 23))
        basis = group.scatter(full, ranges)
        self.assertEqual(basis.shape, (self.rows, 23))
        self.assertEqual([int(block.device.id) for block in basis.blocks], list(self.devices))
        np.testing.assert_array_equal(cp.asnumpy(group.gather(basis)), host)
        # Leading-column views share memory and keep the device layout.
        np.testing.assert_array_equal(cp.asnumpy(group.gather(basis[:, :10])), host[:, :10])
        with self.assertRaises(IndexError):
            basis[:, 3:9]
        # A different filter-block layout moves whole columns between devices.
        other = group.column_ranges(subspace_filter_blocks(23, 6, 9, 3))
        moved = group.repartition(basis, other)
        self.assertEqual(moved.ranges, other)
        np.testing.assert_array_equal(cp.asnumpy(group.gather(moved)), host)
        self.assertIs(group.repartition(moved, other), moved)
        # Columns -> rows -> columns.
        edges = group.row_edges(self.rows)
        rows = group._to_rows(moved.take(), other, edges)
        self.assertEqual([tuple(block.shape) for block in rows],
                         [(edges[i+1]-edges[i], 23) for i in range(len(self.devices))])
        for index, block in enumerate(rows):
            np.testing.assert_array_equal(cp.asnumpy(block), host[edges[index]:edges[index+1], :])
        columns = DistributedBasis(group, group._to_columns(rows, other, edges))
        np.testing.assert_array_equal(cp.asnumpy(group.gather(columns)), host)

    def _exchange_round_trip(self, **environment):
        """Columns -> rows -> columns under ``environment``; every block is compared with the host matrix."""
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        cp, group = self.cp, self.group
        host = np.random.default_rng(37).normal(size=(self.rows, 23))
        ranges = group.column_ranges(uniform_filter_blocks(23, 6, 5))
        edges = group.row_edges(self.rows)
        with patch.dict(os.environ, environment):
            kept = group.scatter(cp.array(host, order='F'), ranges).take()
            pointers = [int(block.data.ptr) for block in kept]
            rows = group._to_rows(kept, ranges, edges, keep=True)
            self.assertEqual([int(block.data.ptr) for block in kept], pointers)
            given = group.scatter(cp.array(host, order='F'), ranges).take()
            again = group._to_rows(given, ranges, edges)
            self.assertEqual(given, [])
            for index, device in enumerate(self.devices):
                for block in (rows[index], again[index]):
                    self.assertEqual((int(block.device.id), block.flags.f_contiguous), (device, True))
                    np.testing.assert_array_equal(cp.asnumpy(block), host[edges[index]:edges[index+1], :])
            del again
            fresh = group._to_columns(rows, ranges, edges)
            # Supplied column blocks are overwritten where they lie.
            for device, block in zip(self.devices, kept):
                with cp.cuda.Device(device):
                    block.fill(np.nan)
                    cp.cuda.get_current_stream().synchronize()
            reused = group._to_columns(rows, ranges, edges, targets=kept)
            self.assertEqual([int(block.data.ptr) for block in reused], pointers)
            for index, (start, stop) in enumerate(ranges):
                for block in (fresh[index], reused[index]):
                    self.assertEqual((int(block.device.id), block.flags.f_contiguous), (self.devices[index], True))
                    np.testing.assert_array_equal(cp.asnumpy(block), host[:, start:stop])

    def test_chunked_exchange_moves_every_entry_exactly(self):
        # One column per chunk, two, a few, and everything at once; with both switches of the exchange as
        # they are by default, and with both off.
        away = self.rows - self.rows // len(self.devices)
        former = dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='0', PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0')
        for size in (8, 16*away, 8*700, 1 << 30):
            for switches in ({}, former):
                with self.subTest(chunk_bytes=size, switches=switches):
                    self._exchange_round_trip(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(size), **switches)

    def _assert_one_inbound_stream_per_peer(self):
        group = self.group
        self.assertEqual(set(group._inbound), set(self.devices))
        for device in self.devices:
            streams = group._inbound[device]
            self.assertEqual([stream is None for stream in streams], [other == device for other in self.devices])
            pointers = [int(stream.ptr) for stream in streams if stream is not None]
            self.assertEqual(len(set(pointers + [int(group._stream(device).ptr)])), len(self.devices))

    def test_concurrent_exchange_moves_every_entry_exactly(self):
        away = self.rows - self.rows // len(self.devices)
        self.assertEqual(self.group._inbound, {})
        for size in (8, 16*away, 1 << 30):
            with self.subTest(chunk_bytes=size):
                self._exchange_round_trip(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(size), PARSEC_CUPY_EXCHANGE_CONCURRENT='1',
                                          PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0')
        self._assert_one_inbound_stream_per_peer()

    def test_later_passes_with_concurrent_exchange_match_one_device(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        with patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400), PARSEC_CUPY_EXCHANGE_CONCURRENT='1',
                        PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0'):
            self._later_passes(self.group, self.group.column_ranges(uniform_filter_blocks(23, 6, 5)))
        self._assert_one_inbound_stream_per_peer()

    def test_four_devices_copy_their_rows_pair_by_pair_and_move_every_entry_exactly(self):
        # Written where there is one CUDA device: this test has run only with that device named four times as
        # the devices of the group, where every copy stays inside it.
        # Four devices take a chunk of the way to their rows from one partner per turn, unless that is switched
        # off or every device queues its copies on its one stream; fewer devices issue them together.  With a
        # column per chunk the blocks of 6, 6, 6 and 5 columns end with a chunk of three senders, which four
        # devices receive from all of them at once.
        away = self.rows - self.rows // len(self.devices)
        paired = 'pairs' if len(self.devices) == 4 else 'together'
        ways = ((paired, {}), (paired, dict(PARSEC_CUPY_EXCHANGE_PAIRS='1')),
                ('together', dict(PARSEC_CUPY_EXCHANGE_PAIRS='0')), ('queued', dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='0')))
        self.assertIsNone(self.group.to_rows_copies)
        for size in (8, 16*away, 1 << 30):
            for name, switches in ways:
                with self.subTest(chunk_bytes=size, switches=switches), patch.dict(os.environ):
                    # Whatever the calling shell names.
                    os.environ.pop('PARSEC_CUPY_EXCHANGE_PAIRS', None)
                    os.environ.pop('PARSEC_CUPY_EXCHANGE_CONCURRENT', None)
                    self._exchange_round_trip(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(size), **switches)
                    self.assertEqual(self.group.to_rows_copies, name)

    def test_four_devices_get_the_same_bits_whichever_way_they_copy_their_rows(self):
        # Written where there is one CUDA device: this test has run only with that device named four times as
        # the devices of the group, where every copy stays inside it.
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        group, states = self.group, 23
        blocks = uniform_filter_blocks(states, 6, 5)
        ranges = group.column_ranges(blocks)
        basis = self._filtered(group, blocks, ranges, 37)
        paired = 'pairs' if len(self.devices) == 4 else 'together'
        for backend in ('host', 'device'):
            # Two or three columns per chunk, so that every way to the rows takes several.
            common = dict(PARSEC_CUPY_RITZ_DENSE_BACKEND=backend, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400),
                          PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed', PARSEC_CUPY_EXCHANGE_CONCURRENT='1')
            for tall in (1, 2, 3):
                with self.subTest(backend=backend, tall_blocks=tall):
                    steps = {}
                    for pairs in ('1', '0'):
                        steps[pairs] = self._ritz_step(group, basis, ranges, tall, roomy=tall == 1,
                                                       PARSEC_CUPY_EXCHANGE_PAIRS=pairs, **common)
                        self.assertEqual(group.to_rows_copies, paired if pairs == '1' else 'together')
                    (values, rotated, gram), (former, vectors, given) = steps['1'], steps['0']
                    self._ritz_vectors(self.op, values, rotated)
                    # The Ritz values, the rotated basis and the Gram pair that the small solve was given.
                    np.testing.assert_array_equal(values, former)
                    np.testing.assert_array_equal(rotated, vectors)
                    for mine, theirs in zip(gram, given, strict=True):
                        np.testing.assert_array_equal(mine, theirs)

    def test_exchange_takes_one_chunk_buffer_per_device(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        cp, group = self.cp, self.group
        host = np.random.default_rng(39).normal(size=(self.rows, 96))
        ranges = group.column_ranges(uniform_filter_blocks(96, 6, 5))
        edges = group.row_edges(self.rows)
        blocks = group.scatter(cp.array(host, order='F'), ranges).take()
        # Only what the two conversions request is taken from this pool.
        pool, previous = cp.cuda.MemoryPool(), cp.cuda.get_allocator()
        rows = columns = None
        cp.cuda.set_allocator(pool.malloc)
        try:
            with patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*512)):
                rows = group._to_rows(blocks, ranges, edges, keep=True)
                columns = group._to_columns(rows, ranges, edges)
            # What the checks allocate is not to be counted.
            cp.cuda.set_allocator(previous)
            tall, chunk = group._tall, group._chunk
            self.assertEqual(chunk, 512)
            for index, device in enumerate(self.devices):
                start, stop = ranges[index]
                with cp.cuda.Device(device):
                    reserved = pool.total_bytes()
                # A row block, a column block and one buffer that served both directions, each rounded up to 512
                # bytes by the pool.  All of the column block outside the device's own rows would be several chunks.
                self.assertGreaterEqual(reserved, 8*(2*tall + chunk))
                self.assertLessEqual(reserved, 8*(2*tall + chunk) + 3*512)
                self.assertGreater((self.rows - (edges[index+1] - edges[index]))*(stop - start), 4*chunk)
                np.testing.assert_array_equal(cp.asnumpy(columns[index]), host[:, start:stop])
        finally:
            cp.cuda.set_allocator(previous)
            # The blocks go before the pool they were taken from, which then keeps nothing to free later.
            rows = columns = None
            self._empty(pool)

    def test_column_copies_move_every_entry_exactly(self):
        away = self.rows - self.rows // len(self.devices)
        for size in (8, 16*away, 1 << 30):
            for concurrent in ('0', '1'):
                with self.subTest(chunk_bytes=size, concurrent=concurrent):
                    self._exchange_round_trip(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(size), PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='1',
                                              PARSEC_CUPY_EXCHANGE_CONCURRENT=concurrent)

    def test_later_passes_with_column_copies_match_one_device(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        with patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400), PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='1',
                        PARSEC_CUPY_EXCHANGE_CONCURRENT='0'):
            self._later_passes(self.group, self.group.column_ranges(uniform_filter_blocks(23, 6, 5)))

    def test_later_passes_with_small_exchange_chunks_match_one_device(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        # Two or three columns per chunk: every exchange of a pass takes several rounds.  Both switches of the
        # exchange are off here; the other tests of later passes leave them as they are by default.
        with patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400), PARSEC_CUPY_EXCHANGE_CONCURRENT='0',
                        PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0'):
            self._later_passes(self.group, self.group.column_ranges(uniform_filter_blocks(23, 6, 5)))

    def test_random_trial_basis_is_the_single_device_stream(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.lapack_random import LapackRandom
        cp, group = self.cp, self.group
        ranges = group.column_ranges(uniform_filter_blocks(19, 6, 7))
        for flag in ('1', '0'):
            with patch.dict(os.environ, PARSEC_CUPY_DEVICE_RANDOM=flag):
                expected = LapackRandom().uniform_minus_1_1((self.rows, 19), column_major=True)
                basis = group.random_basis(LapackRandom(), self.rows, ranges)
            np.testing.assert_array_equal(cp.asnumpy(group.gather(basis)), expected)

    def test_two_later_passes_and_a_truncation_match_one_device(self):
        from parsec_python.acceleration.Eigensolvers.distributed_state import ColumnLayout
        # Balanced ranges start from one device and must move columns; fixed ranges keep the layout of the trial
        # basis, two column ranges per device.  The layout is left to follow the ranges, whatever the calling
        # shell names: balanced ones are contiguous.
        for policy in ('balanced', 'fixed'):
            with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_RANGES=policy):
                os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT', None)
                if policy == 'balanced':
                    group, initial = self.group, ColumnLayout(((0, 23),) + ((23, 23),)*(len(self.devices)-1))
                    kept, _passes = self._later_passes(group, initial)
                else:
                    group = type(self.group)(self.op, self.devices)
                    initial = self._interleaved(group)
                    # 53 columns are compared less closely than 23, as in the test of both layouts below.
                    kept, _passes = self._later_passes(group, initial, self.states, (self.states, 47), atol=2e-11)
            self.assertEqual(group.seconds['repartition'] > 0, policy == 'balanced')
            self.assertEqual(kept == initial, policy == 'fixed')

    def _later_passes(self, group, initial, columns=23, counts=(23, 17), op=None, atol=5e-12):
        """Later passes of ``counts`` leading columns, each against one device: Ritz values within ``atol``.

        Returns the layout that the first pass left and, per pass, the Ritz values and the gathered basis.
        """
        from parsec_python.acceleration.Eigensolvers.distributed_state import run_distributed_subspace
        from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState, SubspaceSettings, run_subspace_filter
        cp, op = self.cp, self.op if op is None else op
        start = np.linalg.qr(np.random.default_rng(17).normal(size=(self.rows, columns)))[0]
        settings = SubspaceSettings(polynomial_degree=5, degree_delta=1)
        single = DeviceSubspaceState(self.rows, columns, cp.linspace(.1, 1.5, columns), cp.array(start, order='F'))
        shared = replace(single, vectors=group.scatter(cp.array(start, order='F'), initial))
        kept, passes, captured = None, [], None
        with patch.dict(os.environ, **_COMMON):
            for working in counts:
                again = working == single.working_states and bool(passes)
                if working != single.working_states:
                    single = replace(single, working_states=working, eigenvalues=single.eigenvalues[:working],
                                     vectors=single.vectors[:, :working])
                    shared = replace(shared, working_states=working, eigenvalues=shared.eigenvalues[:working],
                                     vectors=shared.vectors[:, :working])
                a = run_subspace_filter(op, single, settings=settings, compute_residuals=False)
                b = run_distributed_subspace(op, shared, settings=settings, compute_residuals=False, group=group)
                self.assertEqual(b.vectors.shape, (self.rows, working))
                self.assertEqual(b.rayleigh_ritz.algorithm, 'distributed_generalized_cholesky_rayleigh_ritz')
                np.testing.assert_allclose(cp.asnumpy(b.eigenvalues), cp.asnumpy(a.eigenvalues), rtol=0, atol=atol)
                left, right = cp.asnumpy(a.vectors), cp.asnumpy(group.gather(b.vectors))
                np.testing.assert_allclose(right.T @ right, np.eye(working), atol=1e-9)
                self._subspace(left, right, working)
                self.assertEqual((b.state.filters_completed, b.state.first_filter), (a.state.filters_completed, False))
                single, shared = a.state, b.state
                kept = b.vectors.layout if kept is None else kept
                passes.append((cp.asnumpy(b.eigenvalues), right))
                # A pass with the plan of the one before finds the filter graphs that were captured for it.
                graphs = [group._worker.graphs[device].graphs for device in group.devices]
                if again:
                    self.assertEqual([now is before for now, before in zip(graphs, captured)], [True]*len(graphs))
                captured = graphs
        self.assertGreater(group.passes, 1)
        return kept, passes

    def test_first_solve_of_four_filter_blocks_matches_one_device(self):
        # 19 states: two filter blocks in a half are too few for four devices, whose ranges then stay contiguous.
        self._first_solve_matches_one_device()

    def _first_solve_matches_one_device(self, group=None, *, states=19, atol=5e-11, **environment):
        import importlib
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        from parsec_python.acceleration.Eigensolvers.distributed_state import run_distributed_chebff
        run_chebff = importlib.import_module('parsec_python.acceleration.Eigensolvers.chebff').run_chebff
        cp, group = self.cp, self.group if group is None else group
        settings = ChebFFSettings(polynomial_degree=12, filter_cycles=4)
        with patch.dict(os.environ, PARSEC_CUPY_DEVICE_RANDOM='1', **_COMMON, **environment):
            expected = run_chebff(self.op, states, settings=settings)
            actual = run_distributed_chebff(self.op, states, settings=settings, group=group)
        np.testing.assert_allclose(cp.asnumpy(actual.eigenvalues), cp.asnumpy(expected.eigenvalues), rtol=0, atol=atol)
        self._subspace(cp.asnumpy(expected.vectors), cp.asnumpy(group.gather(actual.vectors)), states)
        self.assertEqual([c.number for c in actual.cycles], [1, 2, 3, 4])
        for mine, theirs in zip(actual.cycles, expected.cycles):
            self.assertAlmostEqual(mine.lower_bound_out, theirs.lower_bound_out, delta=1e-9)
        return actual

    def test_a_group_filters_and_solves_with_the_tiles_of_its_owner_bit_for_bit_like_slot_major(self):
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import SectorDeviceGroup, run_distributed_chebff
        from parsec_python.acceleration.backends import implicit_stencil
        cp = self.cp
        groups = []
        for tile in (0, 16):
            operator = _hamiltonian(self.rows, implicit_tile=tile)
            operator.distributed_filter_devices = self.devices
            # The owner has packed its tiles by now, and nobody packs them again.
            with patch.object(implicit_stencil, 'pack_affine_tiles', side_effect=AssertionError('packed again')):
                groups.append((operator, SectorDeviceGroup(operator, self.devices)))
        (plain, slot_major), (tiled, tiles) = groups
        self.assertEqual((slot_major.stencil_storage, tiles.stencil_storage),
                         ('stencil_major_int32_neighbors_uint8_coefficient_palette', 'implicit_affine_tile_16'))
        self.assertGreater(tiles.replica_seconds, 0.0)
        packed = tiled.compact_finite_difference
        expected = cp.asnumpy(packed.neighbors), cp.asnumpy(packed.coefficient_codes)
        self.assertEqual(expected[0].ndim, 1)
        for device in self.devices[1:]:
            held = tiles._operator(device).compact_finite_difference
            self.assertEqual((int(held.neighbors.device.id), held.implicit_statistics), (device, packed.implicit_statistics))
            with cp.cuda.Device(device):
                np.testing.assert_array_equal(cp.asnumpy(held.neighbors), expected[0])
                np.testing.assert_array_equal(cp.asnumpy(held.coefficient_codes), expected[1])
        # The filter of every device, in its own column range.
        blocks = uniform_filter_blocks(23, 6, 5)
        ranges = tiles.column_ranges(blocks)
        np.testing.assert_array_equal(self._filtered(tiles, blocks, ranges, 5), self._filtered(slot_major, blocks, ranges, 5))
        # A first solve: the filters, H times the columns of every device in the Ritz steps, and all that
        # follows from them.
        settings = ChebFFSettings(polynomial_degree=12, filter_cycles=4)
        solved = []
        for operator, group in groups:
            with patch.dict(os.environ, PARSEC_CUPY_DEVICE_RANDOM='1', **_COMMON):
                result = run_distributed_chebff(operator, 19, settings=settings, group=group)
            solved.append((cp.asnumpy(result.eigenvalues), cp.asnumpy(group.gather(result.vectors))))
        np.testing.assert_array_equal(solved[1][0], solved[0][0])
        np.testing.assert_array_equal(solved[1][1], solved[0][1])
        # Two later passes, the second on fewer states: the filter graphs of a plan and H in their Ritz steps.
        later = [self._later_passes(group, ranges, op=operator)[1] for operator, group in groups]
        for (tiled_values, tiled_basis), (values, basis) in zip(later[1], later[0], strict=True):
            np.testing.assert_array_equal(tiled_values, values)
            np.testing.assert_array_equal(tiled_basis, basis)

    def test_unstable_overlap_falls_back_to_the_orthonormal_route(self):
        self._unstable_overlap_falls_back_to_the_orthonormal_route()

    def _unstable_overlap_falls_back_to_the_orthonormal_route(self, columns=17, initial=None, **environment):
        import importlib
        from parsec_python.acceleration.Eigensolvers.distributed_state import run_distributed_subspace
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import GeneralizedRitzStabilityError
        from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState, SubspaceSettings, run_subspace_filter
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp, group = self.cp, self.group
        start = np.linalg.qr(np.random.default_rng(23).normal(size=(self.rows, columns)))[0]
        settings = SubspaceSettings(polynomial_degree=5, degree_delta=1)
        single = DeviceSubspaceState(self.rows, columns, cp.linspace(.1, 1.5, columns), cp.array(start, order='F'))
        if initial is None:
            # Everything on the owner.
            initial = ((0, columns),) + ((columns, columns),)*(len(self.devices)-1)
        shared = replace(single, vectors=group.scatter(cp.array(start, order='F'), initial))

        def unstable(*_args, **_options):
            raise GeneralizedRitzStabilityError('forced')

        with patch.dict(os.environ, **_COMMON, **environment):
            expected = run_subspace_filter(self.op, single, settings=settings, compute_residuals=False)
            # Whichever small solve the environment selects for the shared basis fails.
            with patch.object(module, 'solve_whitened_ritz', unstable), \
                 patch.object(module, 'solve_whitened_ritz_on_device', unstable):
                actual = run_distributed_subspace(self.op, shared, settings=settings, compute_residuals=False, group=group)
        self.assertTrue(actual.state.generalized_ritz_failed)
        self.assertEqual(group.passes, 0)
        np.testing.assert_allclose(cp.asnumpy(actual.eigenvalues), cp.asnumpy(expected.eigenvalues), rtol=0, atol=5e-11)
        self._subspace(cp.asnumpy(expected.vectors), cp.asnumpy(group.gather(actual.vectors)), columns)
        return actual

    def test_density_is_the_sum_over_devices(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Occupations.device_density import CuPyDeviceDensityBuilder
        cp, group = self.cp, self.group
        host = np.random.default_rng(41).normal(size=(self.rows, 23))
        basis = group.scatter(cp.array(host, order='F'), group.column_ranges(uniform_filter_blocks(23, 6, 5)))
        occupations = np.random.default_rng(43).uniform(size=14)
        builder = CuPyDeviceDensityBuilder(cp)
        expected = builder(cp.array(host[:, :14], order='F'), occupations, 0.37)
        np.testing.assert_allclose(basis.density(builder, occupations, 0.37), expected, rtol=2e-14, atol=0)
        np.testing.assert_allclose(basis[:, :20].density(builder, occupations, 0.37), expected, rtol=2e-14, atol=0)

    def test_solver_shares_the_basis_unless_switched_off(self):
        from parsec_python.Eigensolvers.eigval import EigvalSettings
        from parsec_python.acceleration.Eigensolvers.distributed_state import DistributedBasis
        from parsec_python.acceleration.Eigensolvers.eigval import CuPyEigvalSolver
        cp = self.cp
        settings = EigvalSettings(initial_method='chebff', safety_buffer=0)
        results = {}
        # Unset is auto: a basis whose blocks fit is shared.
        for flag in ('0', '1', 'auto', None):
            solver = CuPyEigvalSolver(self.op, settings=settings, retain_vectors_on_device=True,
                                      compute_subspace_residuals=False)
            with patch.dict(os.environ, PARSEC_CUPY_DEVICE_RANDOM='1', PARSEC_CUPY_RECYCLE_STATE='1', **_COMMON):
                os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE', None)
                if flag is not None:
                    os.environ['PARSEC_CUPY_DISTRIBUTED_STATE'] = flag
                first = solver.solve(19)
                solver.truncate_state(17)
                second = solver.solve(17)
            self.assertEqual((first.solver_path, second.solver_path), ('chebff', 'subspace'))
            self.assertEqual(isinstance(second.vectors, DistributedBasis), flag != '0')
            self.assertEqual(tuple(second.vectors.shape), (self.rows, 17))
            results[flag] = (first.eigenvalues, second.eigenvalues)
        for flag in ('1', 'auto', None):
            np.testing.assert_allclose(results[flag][0], results['0'][0], rtol=0, atol=5e-11)
            np.testing.assert_allclose(results[flag][1], results['0'][1], rtol=0, atol=5e-11)
        self.assertEqual(self.op._sector_device_group.seconds['repartition'], 0.0)
        # A basis whose blocks would not fit is left to the owner-centred route.
        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE='auto', PARSEC_CUPY_DEVICE_RANDOM='1',
                        PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION='1e-9', **_COMMON):
            solver = CuPyEigvalSolver(self.op, settings=settings, retain_vectors_on_device=True,
                                      compute_subspace_residuals=False)
            self.assertNotIsInstance(solver.solve(19).vectors, DistributedBasis)
        # Nor is a basis that one device is to orthonormalize: a shared basis is always solved as filtered.
        with patch.dict(os.environ, {**_COMMON, 'PARSEC_CUPY_GENERALIZED_RITZ': 'off'}, PARSEC_CUPY_DEVICE_RANDOM='1'):
            os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE', None)
            solver = CuPyEigvalSolver(self.op, settings=settings, retain_vectors_on_device=True,
                                      compute_subspace_residuals=False)
            self.assertNotIsInstance(solver.solve(19).vectors, DistributedBasis)

    def test_a_pass_returns_the_basis_in_the_blocks_it_was_given(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import run_distributed_subspace
        from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState, SubspaceSettings
        cp, group = self.cp, self.group
        start = np.linalg.qr(np.random.default_rng(29).normal(size=(self.rows, 23)))[0]
        ranges = group.column_ranges(uniform_filter_blocks(23, 6, 5))
        state = DeviceSubspaceState(self.rows, 23, cp.linspace(.1, 1.5, 23), group.scatter(cp.array(start, order='F'), ranges))
        pointers = [int(block.data.ptr) for block in state.vectors.blocks]
        settings = SubspaceSettings(polynomial_degree=5, degree_delta=1)
        # Values are compared with one device in the test of two later passes; this one is about memory.
        with patch.dict(os.environ, **_COMMON):
            for _ in range(2):
                result = run_distributed_subspace(self.op, state, settings=settings, compute_residuals=False, group=group)
                self.assertEqual([int(block.data.ptr) for block in result.vectors.blocks], pointers)
                state = result.state
        vectors = cp.asnumpy(group.gather(result.vectors))
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(23), atol=1e-9)

    def _recorded_pass(self, name, **environment):
        """One later pass: its eigenvalues and host copies of the blocks each device gave to and got from ``name``."""
        import importlib
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import run_distributed_subspace
        from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState, SubspaceSettings
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp, group = self.cp, self.group
        start = np.linalg.qr(np.random.default_rng(37).normal(size=(self.rows, 23)))[0]
        ranges = group.column_ranges(uniform_filter_blocks(23, 6, 5))
        state = DeviceSubspaceState(self.rows, 23, cp.linspace(.1, 1.5, 23), group.scatter(cp.array(start, order='F'), ranges))
        original, calls = getattr(module, name), []

        def record(*blocks):
            product = original(*blocks)
            calls.append(tuple(cp.asnumpy(block) for block in (*blocks, product)))
            return product

        # The whole pass is run because replicas receive the effective potential in its filter step.
        with patch.dict(os.environ, **_COMMON, **environment), patch.object(module, name, record):
            result = run_distributed_subspace(self.op, state, settings=SubspaceSettings(polynomial_degree=5, degree_delta=1),
                                              compute_residuals=False, group=group)
        self.assertEqual(len(calls), len(self.devices))
        self.assertEqual(sum(blocks[0].shape[0] for blocks in calls), self.rows)
        return cp.asnumpy(result.eigenvalues), calls

    def test_every_device_projects_on_trailing_columns(self):
        # One slab is the former full product of a row block; 4 slabs of the 23 columns are 6, 6, 6 and 5 wide.
        # With three tall blocks: with two, H times the columns is projected a slab of its own kind at a time.
        three = dict(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='3')
        former, whole = self._recorded_pass('_lower_triangle_product', PARSEC_CUPY_RITZ_GRAM_SLABS='1', **three)
        values, slabs = self._recorded_pass('_lower_triangle_product', PARSEC_CUPY_RITZ_GRAM_SLABS='4', **three)
        for calls, slabbed in ((whole, False), (slabs, True)):
            for row_basis, row_applied, product in calls:
                reference = row_basis.T @ row_applied
                np.testing.assert_allclose(np.tril(product), np.tril(reference), rtol=1e-12, atol=1e-12*np.abs(reference).max())
                # Nothing is computed above the diagonal slabs.
                self.assertEqual(not np.triu(product, 6).any(), slabbed)
        np.testing.assert_allclose(values, former, rtol=0, atol=5e-12)

    def test_every_device_forms_the_overlap_from_trailing_columns(self):
        # The DSYRK overlap of a row block and 4 slabs on trailing columns, both against the dense overlap.
        former, dsyrk = self._recorded_pass('_symmetric_overlap', PARSEC_CUPY_RITZ_SYRK='on', PARSEC_CUPY_RITZ_GRAM_SLABS='4')
        values, slabs = self._recorded_pass('_symmetric_overlap', PARSEC_CUPY_RITZ_SYRK='slabs', PARSEC_CUPY_RITZ_GRAM_SLABS='4')
        for calls in (dsyrk, slabs):
            for row_basis, product in calls:
                reference = row_basis.T @ row_basis
                np.testing.assert_allclose(np.tril(product), np.tril(reference), rtol=1e-12, atol=1e-12*np.abs(reference).max())
        for _row_basis, product in slabs:
            self.assertFalse(np.triu(product, 6).any())
        np.testing.assert_allclose(values, former, rtol=0, atol=5e-12)

    def _filtered(self, group, blocks, ranges, seed):
        """Host copy of a filtered basis of ``group`` in the column ranges ``ranges``."""
        cp = self.cp
        start = np.linalg.qr(np.random.default_rng(seed).normal(size=(group._operator(group.owner).shape[0], blocks[-1].stop)))[0]
        with patch.dict(os.environ, PARSEC_CUPY_MIXED_FILTER='off', PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed'):
            # The replicas of the other devices receive the effective potential in this step.
            filtered = group.filter(group.scatter(cp.array(start, order='F'), ranges), blocks, 1.2, 6., 1.2, False)
        self.assertEqual(filtered.ranges, tuple(ranges))
        return cp.asnumpy(group.gather(filtered))

    def _placed(self, group, basis, layout):
        """The host matrix ``basis`` in column blocks with the capacity that ``group`` gives those of a trial basis.

        The capacity is that of the Ritz step which the environment selects when this is called.
        """
        from parsec_python.acceleration.Eigensolvers.distributed_state import ColumnLayout, DistributedBasis
        cp = self.cp
        layout = layout if isinstance(layout, ColumnLayout) else ColumnLayout(tuple(layout))
        rows = basis.shape[0]
        tall, _chunk = group._capacities(layout, group.row_edges(rows))
        blocks = []
        for index, device in enumerate(group.devices):
            with cp.cuda.Device(device):
                block = group._block(cp, (rows, layout.widths[index]), tall)
                if block.size:
                    block.set(np.asfortranarray(basis[:, _held(layout, index)]))
                    cp.cuda.get_current_stream().synchronize()
                blocks.append(block)
        return DistributedBasis(group, blocks, layout)

    def _ritz_step(self, group, basis, ranges, blocks, roomy=False, **environment):
        """``group.ritz`` of the host matrix ``basis`` with ``blocks`` tall blocks per device.

        Returns the Ritz values, the rotated basis and the two Gram matrices that the small solve was given.
        ``roomy`` column blocks have the capacity of a trial basis (:meth:`_placed`); the others are as large
        as their columns.
        """
        import importlib
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp = self.cp
        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=str(blocks), **environment):
            columns = self._placed(group, basis, ranges) if roomy else group.scatter(cp.array(basis, order='F'), ranges)
            pointers = [int(block.data.ptr) for block in columns.blocks]
            on_device = module.dense_solve_on_device()
            solver = 'solve_whitened_ritz_on_device' if on_device else 'solve_whitened_ritz'
            with patch.object(module, solver, wraps=getattr(module, solver)) as solve:
                values, rotated, _whitened = group.ritz(group._operator(group.owner), columns, stages=False)
        solve.assert_called_once()
        gram = [cp.asnumpy(part) if on_device else np.array(part) for part in solve.call_args.args]
        # The rotated basis lies in the blocks that held the filtered one.
        self.assertEqual([int(block.data.ptr) for block in rotated.blocks], pointers)
        return values, cp.asnumpy(group.gather(rotated)), gram

    def test_two_tall_blocks_match_three(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import SectorDeviceGroup, slab_columns
        cp, states = self.cp, 23
        blocks = uniform_filter_blocks(states, 6, 5)
        # A sector is given all four devices of a node, or two of them where two sectors share the node.
        for devices in sorted({self.devices, self.devices[:2]}, key=len):
            op, group = self.op, self.group
            if devices != self.devices:
                op = _hamiltonian(self.rows)
                op.distributed_filter_devices = devices
                group = SectorDeviceGroup(op, devices)
            # The columns spread over the devices, and all of them on the owner with the other devices empty.
            for ranges in (group.column_ranges(blocks), ((0, states),) + ((states, states),)*(len(devices)-1)):
                basis = self._filtered(group, blocks, ranges, 37)
                exact = basis.T @ basis, basis.T @ cp.asnumpy(op @ cp.array(basis, order='F'))
                for backend in ('host', 'device'):
                    # Two or three columns per chunk where a whole block is exchanged.
                    common = dict(PARSEC_CUPY_RITZ_DENSE_BACKEND=backend, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400),
                                  PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed')
                    former, vectors, gram = self._ritz_step(group, basis, ranges, 3, **common)
                    self._ritz_vectors(op, former, vectors)
                    # One column per slab, three, and what the streaming budget gives a basis this small.
                    for slab in (1, 3, None):
                        budget = {} if slab is None else dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*self.rows*slab))
                        with self.subTest(devices=devices, ranges=ranges, backend=backend, slab=slab):
                            with patch.dict(os.environ, budget):
                                width = slab_columns(self.rows, states, len(devices))
                            seconds, passes = dict(group.seconds), group.passes
                            values, rotated, slabbed = self._ritz_step(group, basis, ranges, 2, **common, **budget)
                            np.testing.assert_allclose(values, former, rtol=0, atol=5e-12)
                            for mine, theirs, reference in zip(slabbed, gram, exact):
                                scale = np.abs(reference).max()
                                np.testing.assert_allclose(np.tril(mine), np.tril(theirs), rtol=0, atol=1e-12*scale)
                                np.testing.assert_allclose(np.tril(mine), np.tril(reference), rtol=0, atol=1e-12*scale)
                            # Nothing is projected above the diagonal slabs.
                            self.assertFalse(np.triu(slabbed[1], width).any())
                            self._ritz_vectors(op, values, rotated)
                            self._subspace(vectors, rotated, states)
                            # Every stage of the step has taken its time.
                            self.assertEqual(group.passes, passes + 1)
                            for stage in ('apply', 'to_rows', 'gram', 'dense', 'rotate', 'to_columns'):
                                self.assertGreater(group.seconds[stage], seconds[stage])

    def test_two_tall_blocks_with_full_slabs_match_equal_ones(self):
        # Written where there is no CUDA device: this test had not run when it was added.
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        group, states = self.group, self.states
        blocks = uniform_filter_blocks(states, 6, 5)
        ranges = group.column_ranges(blocks)
        basis = self._filtered(group, blocks, ranges, 37)
        # Slabs of at most five columns in ranges of two or three filter blocks of six: the former cut leaves
        # a last slab of one to three columns in every range, the equal one none.
        common = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400), PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*self.rows*5),
                      PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed')
        results = {}
        for cut in (None, 'equal', 'full'):
            with patch.dict(os.environ):
                os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_SLABS', None)
                if cut is not None:
                    os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_SLABS'] = cut
                results[cut] = self._ritz_step(group, basis, ranges, 2, **common)
            self._ritz_vectors(self.op, results[cut][0], results[cut][1])
            # What a run reports of the step: its blocks, its cut and a widest slab within the budget.
            self.assertEqual((group.tall_blocks, group.slab_cut), (2, cut or 'equal'))
            self.assertIn(group.slab_columns, range(1, 6))
        # Equal slabs are the default.
        np.testing.assert_allclose(results[None][0], results['equal'][0], rtol=0, atol=1e-13)
        # The two cuts sum the projection over other slabs: the same Gram matrices and Ritz pairs to round-off.
        np.testing.assert_allclose(results['full'][0], results['equal'][0], rtol=0, atol=2e-11)
        self._subspace(results['full'][1], results['equal'][1], states)
        for mine, theirs in zip(results['full'][2], results['equal'][2]):
            np.testing.assert_allclose(np.tril(mine), np.tril(theirs), rtol=0, atol=1e-12*np.abs(theirs).max())

    def test_later_passes_with_two_tall_blocks_match_one_device(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        # Two columns per slab and two or three per exchange chunk: several rounds in a pass and several chunks
        # where a whole block is exchanged.  Then one column per slab with both switches of the exchange off.
        former = dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='0', PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0')
        for slab, switches in ((2, {}), (1, former)):
            with self.subTest(slab=slab, switches=switches):
                group = type(self.group)(self.op, self.devices)
                with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='2', PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400),
                                PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*self.rows*slab),
                                PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed', **switches):
                    self._later_passes(group, group.column_ranges(uniform_filter_blocks(23, 6, 5)))
                # The second pass has fewer states, and no column on the last of four devices.
                self.assertEqual((group.passes, group.seconds['repartition']), (2, 0.0))

    def test_later_passes_with_three_tall_blocks_match_one_device(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        # The former step, which the other tests of later passes take only where the calling shell names it.
        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='3', PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400),
                        PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed'):
            self._later_passes(self.group, self.group.column_ranges(uniform_filter_blocks(23, 6, 5)))
        self.assertEqual((self.group.passes, self.group.seconds['repartition']), (2, 0.0))

    def test_first_solve_with_two_tall_blocks_matches_one_device(self):
        # Four cycles, each with its own Ritz step unless its overlap fails an audit.  A cycle that leaves
        # through the orthonormal fallback gives the same eigenvalues, so the steps are counted as well: two
        # tall blocks serve every cycle that three serve.
        chunks = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400))
        three = type(self.group)(self.op, self.devices)
        self._first_solve_matches_one_device(three, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='3', **chunks)
        self._first_solve_matches_one_device(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='2', **chunks)
        self.assertGreater(self.group.passes, 0)
        self.assertEqual(self.group.passes, three.passes)

    def test_unstable_overlap_with_two_tall_blocks_falls_back_to_the_orthonormal_route(self):
        self._unstable_overlap_falls_back_to_the_orthonormal_route(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='2')

    def test_unstable_overlap_with_three_tall_blocks_falls_back_to_the_orthonormal_route(self):
        self._unstable_overlap_falls_back_to_the_orthonormal_route(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='3')

    def test_two_tall_blocks_keep_the_filtered_columns_when_the_audit_fails(self):
        import importlib
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import GeneralizedRitzStabilityError
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp, group = self.cp, self.group
        blocks = uniform_filter_blocks(23, 6, 5)
        ranges = group.column_ranges(blocks)
        basis = self._filtered(group, blocks, ranges, 41)
        columns = group.scatter(cp.array(basis, order='F'), ranges)
        pointers = [int(block.data.ptr) for block in columns.blocks]

        def unstable(*_args, **_options):
            raise GeneralizedRitzStabilityError('forced')

        # The audit fails after every slab of H X has been formed, sent and projected.
        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='2', PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400),
                        PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*self.rows*2)), \
             patch.object(module, 'solve_whitened_ritz', unstable), \
             patch.object(module, 'solve_whitened_ritz_on_device', unstable):
            with self.assertRaises(GeneralizedRitzStabilityError) as caught:
                group.ritz(self.op, columns, stages=False)
        kept = caught.exception.filtered
        self.assertEqual((kept.ranges, group.passes), (ranges, 0))
        self.assertEqual([int(block.data.ptr) for block in kept.blocks], pointers)
        np.testing.assert_array_equal(cp.asnumpy(group.gather(kept)), basis)
        self.assertGreater(min(group.seconds[stage] for stage in ('apply', 'to_rows', 'gram')), 0.)
        self.assertEqual((group.seconds['rotate'], group.seconds['to_columns']), (0., 0.))

    def test_two_tall_blocks_and_the_slab_workspace_are_the_peak(self):
        import importlib
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import SectorDeviceGroup
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import solve_whitened_ritz_on_device
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp = self.cp
        # Tall blocks of megabytes, so that the Gram matrices are small beside them: 32 states make 8 kB each.
        rows, states = 65537, 32
        op = _hamiltonian(rows)
        op.distributed_filter_devices = self.devices
        group = SectorDeviceGroup(op, self.devices)
        blocks = uniform_filter_blocks(states, 4, 5)
        ranges = group.column_ranges(blocks)
        basis = self._filtered(group, blocks, ranges, 43)
        count = len(self.devices)
        # One column per slab and per exchange chunk.  The owner estimates the condition number of the overlap
        # itself, as in the solve that is measured alone below: a helper would take a copy of the overlap and
        # the work arrays of its spectrum on another device.
        sizes = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*rows), PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*512),
                     PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed', PARSEC_CUPY_RITZ_EIGH_BACKEND='host',
                     PARSEC_CUPY_RITZ_CONDITION_HELPER='off')
        # An array must not outlive the pool it was taken from.
        pools = []

        def taken(function):
            """What ``function`` takes from a pool of its own on every device, and what it returns."""
            pool, previous = cp.cuda.MemoryPool(), cp.cuda.get_allocator()
            pools.append(pool)
            cp.cuda.set_allocator(pool.malloc)
            try:
                result = function()
            finally:
                cp.cuda.set_allocator(previous)
            reserved = []
            for device in self.devices:
                with cp.cuda.Device(device):
                    reserved.append(pool.total_bytes())
            return reserved, result

        given = []

        def recorded(raw_overlap, raw_projection):
            # Host copies: a reference to the Gram pair would keep its pool block from the rest of the step.
            given[:] = cp.asnumpy(raw_overlap), cp.asnumpy(raw_projection)
            return solve_whitened_ritz_on_device(raw_overlap, raw_projection)

        _whitened = None
        try:
            # One column per slab makes the last slab the last column of the basis, and what a slab is projected
            # on, the basis columns from its own first one onwards, is then that column alone: every device forms
            # a 1 x 1 product of two columns of the height of its row block.  CuPy computes such a product as an
            # inner product and sizes the work arrays of that itself.  With CuPy 14.2 on an A100 they are one
            # such column and 35 kB more, where the other products of the step take their results only.  What
            # they take is therefore measured, like the small solve below: the same product of two columns of
            # the tallest row block in a pool of its own.
            with cp.cuda.Device(group.owner):
                tallest = cp.ones((-(-rows // count), 2), order='F')
                inner = taken(lambda: tallest[:, 1:].T @ tallest[:, :1])[0][0]
            for backend in ('host', 'device'):
                values, reserved, solving = {}, {}, 0
                with patch.dict(os.environ, PARSEC_CUPY_RITZ_DENSE_BACKEND=backend, **sizes), \
                     patch.object(module, 'solve_whitened_ritz_on_device', recorded):
                    for tall_blocks in (3, 2):
                        columns = group.scatter(cp.array(basis, order='F'), ranges)
                        # Only what the Ritz step requests is taken from its pool: the columns exist already.
                        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=str(tall_blocks)):
                            reserved[tall_blocks], (values[tall_blocks], rotated, _whitened) = taken(
                                lambda: group.ritz(op, columns, stages=False))
                        vectors = cp.asnumpy(group.gather(rotated))
                        np.testing.assert_allclose(vectors.T @ vectors, np.eye(states), atol=1e-9)
                    if backend == 'device':
                        # cuSOLVER sizes its work arrays itself, so what the small solve takes on the owner is
                        # measured: the same solve of the same Gram pair in a pool of its own.
                        with cp.cuda.Device(group.owner):
                            overlap, projection = (cp.asarray(part) for part in given)
                            solving = taken(lambda: solve_whitened_ritz_on_device(overlap, projection))[0][0]
                        self.assertGreater(solving, 0)
                # Values are compared more closely on the small operator; this test is about memory.
                np.testing.assert_allclose(values[2], values[3], rtol=0, atol=5e-11)
                tall, chunk, work = group._tall, group._chunk, group._work
                # A chunk is one column outside the rows of a device; the workspace is one column in all rows and,
                # in the rows of the tallest row block, one column of every device.
                self.assertEqual((chunk, work), (rows - rows // count, rows + -(-rows // count)*count))
                self.assertGreaterEqual(tall, -(-rows // count)*states)
                # With the columns, two tall blocks: the step takes one row block, the slab workspace and one
                # exchange buffer, once each and for all of its exchanges, each rounded up to 512 bytes by the pool.
                held = 8*(tall + work + chunk)
                # Besides them a device takes its Gram pair, two m x m arrays, the overlap product the pair is
                # filled from, a third, and the products of single slabs, which together stay below a fourth.  What
                # comes later, the copy of the coefficients among it, is served from these blocks.
                pair = 8*2*states*states
                small = 2*pair + 3*512
                # The 1 x 1 product of the last slab comes on top, as measured above.
                for index in range(count):
                    # The owner of a device solve also takes the pair it receives and the arrays of that solve.
                    owner = pair + solving if backend == 'device' and self.devices[index] == group.owner else 0
                    message = (f'{backend} solve, device {index}: the step with two tall blocks takes {reserved[2]} '
                               f'bytes on the devices, of them {held} in row block, workspace and buffer; {small} '
                               f'more are expected for the Gram arrays, {inner} for the 1 x 1 product of the last '
                               f'slab and {owner} for the small solve, which alone takes {solving}; the step with '
                               f'three takes {reserved[3]}')
                    # All that the device may take besides the three, the small solve of the owner included, is
                    # less than a tall block less the workspace, which a third tall block in the place of the
                    # workspace would add: the bound below cannot let one pass.
                    self.assertLess(small + inner + owner, 8*(tall - work), message)
                    self.assertGreaterEqual(reserved[2][index], held, message)
                    self.assertLessEqual(reserved[2][index], held + small + inner + owner, message)
                    # H times the columns and a row block, whose pool block the second row block takes over:
                    # more than with two by a tall block less the workspace, which is more than a buffer.
                    self.assertGreaterEqual(reserved[3][index], 8*(2*tall + chunk), message)
                    self.assertLess(reserved[2][index] + 8*chunk, reserved[3][index], message)
        finally:
            # What a step returned goes before the pools of the steps, which then keep nothing to free later.
            _whitened = None
            self._empty(*pools)

    # ----- one tall block per device (PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=1) -----
    # These tests were written where there is no CUDA device: they had not run when they were added.  Their
    # host counterparts on a NumPy stand-in are in test_distributed_exchange.

    def test_one_tall_block_matches_two(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import (
            ColumnLayout, SectorDeviceGroup, interleaved_ranges)
        cp = self.cp
        # A sector is given all four devices of a node, or two of them where two sectors share the node.
        for devices in sorted({self.devices, self.devices[:2]}, key=len):
            op, group = self.op, self.group
            if devices != self.devices:
                op = _hamiltonian(self.rows)
                op.distributed_filter_devices = devices
                group = SectorDeviceGroup(op, devices)
            # 23 columns in four filter blocks and 53 in nine: 53 are compared less closely, as in the tests of
            # the two layouts below.
            for states, atol in ((23, 5e-12), (self.states, 2e-11)):
                blocks = uniform_filter_blocks(states, 6, 5)
                ranges = group.column_ranges(blocks)
                basis = self._filtered(group, blocks, ranges, 37)
                exact = basis.T @ basis, basis.T @ cp.asnumpy(op @ cp.array(basis, order='F'))
                # The columns spread over the devices in ranges of uneven widths, all of them on the owner with
                # the other devices empty, and two ranges per device where every device gets blocks of both halves.
                layouts = [ColumnLayout(ranges), ColumnLayout(((0, states),) + ((states, states),)*(len(devices)-1))]
                two_ranges = interleaved_ranges(blocks, len(devices))
                if two_ranges is not None:
                    layouts.append(ColumnLayout(two_ranges))
                for layout in layouts:
                    for backend in ('host', 'device'):
                        # Two or three columns per chunk where a whole block is exchanged.
                        common = dict(PARSEC_CUPY_RITZ_DENSE_BACKEND=backend, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400),
                                      PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed')
                        # One column per slab, three, and what the streaming budget gives a basis this small.
                        for slab in (1, 3, None):
                            budget = {} if slab is None else dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*self.rows*slab))
                            with self.subTest(devices=devices, widths=layout.widths, backend=backend, slab=slab):
                                former, vectors, gram = self._ritz_step(group, basis, layout, 2, **common, **budget)
                                self._ritz_vectors(op, former, vectors)
                                seconds, passes, separate = dict(group.seconds), group.passes, group.separate_row_blocks
                                values, rotated, relaid = self._ritz_step(group, basis, layout, 1, roomy=True, **common, **budget)
                                np.testing.assert_allclose(values, former, rtol=0, atol=atol)
                                # The small solve is given the Gram matrices in the order of the basis, whatever
                                # the order in which the row blocks of the step hold the columns.
                                for mine, theirs, reference in zip(relaid, gram, exact):
                                    scale = np.abs(reference).max()
                                    np.testing.assert_allclose(np.tril(mine), np.tril(theirs), rtol=0, atol=1e-12*scale)
                                    np.testing.assert_allclose(np.tril(mine), np.tril(reference), rtol=0, atol=1e-12*scale)
                                self._ritz_vectors(op, values, rotated)
                                self._subspace(vectors, rotated, states)
                                # Every stage of the step has taken its time.
                                self.assertEqual(group.passes, passes + 1)
                                for stage in ('apply', 'to_rows', 'gram', 'dense', 'rotate', 'to_columns'):
                                    self.assertGreater(group.seconds[stage], seconds[stage])
                                # Every block that holds columns held its row block too; a device without
                                # columns has no block and took one from the pool.
                                self.assertEqual(group.separate_row_blocks - separate, sum(not width for width in layout.widths))

    def test_one_tall_block_in_blocks_without_room_matches_two(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        group, states = self.group, 23
        blocks = uniform_filter_blocks(states, 6, 5)
        ranges = group.column_ranges(blocks)
        basis = self._filtered(group, blocks, ranges, 37)
        common = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400), PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*self.rows*2),
                      PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed')
        former, vectors, _gram = self._ritz_step(group, basis, ranges, 2, **common)
        # Blocks that are only as large as their columns, as the fallback after a failed audit leaves them:
        # the row block of a device comes from the pool unless it happens to fit the columns.
        values, rotated, _gram = self._ritz_step(group, basis, ranges, 1, **common)
        np.testing.assert_allclose(values, former, rtol=0, atol=5e-12)
        self._ritz_vectors(self.op, values, rotated)
        self._subspace(vectors, rotated, states)
        self.assertGreater(group.separate_row_blocks, 0)

    def test_later_passes_with_one_tall_block_match_one_device(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        # Two columns per slab and two or three per exchange chunk, then one column per slab with both switches
        # of the exchange off.  The basis starts in blocks that are only as large as their columns.
        former = dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='0', PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0')
        for slab, switches in ((2, {}), (1, former)):
            with self.subTest(slab=slab, switches=switches):
                group = type(self.group)(self.op, self.devices)
                with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='1', PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400),
                                PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*self.rows*slab),
                                PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed', **switches):
                    self._later_passes(group, group.column_ranges(uniform_filter_blocks(23, 6, 5)))
                self.assertEqual((group.passes, group.seconds['repartition']), (2, 0.0))

    def test_first_solve_with_one_tall_block_matches_one_device(self):
        # Four cycles, each with its own Ritz step unless its overlap fails an audit, which the count of the
        # steps would show: one tall block serves every cycle that two serve.
        chunks = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400))
        two = type(self.group)(self.op, self.devices)
        self._first_solve_matches_one_device(two, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='2', **chunks)
        self._first_solve_matches_one_device(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='1', **chunks)
        self.assertGreater(self.group.passes, 0)
        self.assertEqual(self.group.passes, two.passes)
        # The trial basis was given the room: every cycle laid the rows into the blocks of the columns.
        self.assertEqual(self.group.separate_row_blocks, 0)

    def test_unstable_overlap_with_one_tall_block_falls_back_to_the_orthonormal_route(self):
        self._unstable_overlap_falls_back_to_the_orthonormal_route(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='1')

    def test_one_tall_block_puts_the_filtered_columns_back_when_the_audit_fails(self):
        import importlib
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import GeneralizedRitzStabilityError
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp, group = self.cp, self.group
        blocks = uniform_filter_blocks(23, 6, 5)
        ranges = group.column_ranges(blocks)
        basis = self._filtered(group, blocks, ranges, 41)
        given, held = [], []

        def unstable(*_args, **_options):
            # What the column blocks hold when the small problem is to be solved.
            held[:] = [cp.asnumpy(block) for block in given]
            raise GeneralizedRitzStabilityError('forced')

        # The audit fails after every slab has left its block, with the rows of the basis in their place.
        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='1', PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400),
                        PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*self.rows*2), PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed'), \
             patch.object(module, 'solve_whitened_ritz', unstable), \
             patch.object(module, 'solve_whitened_ritz_on_device', unstable):
            columns = self._placed(group, basis, ranges)
            # The step empties the basis it is given: the blocks are read through this list.
            given[:] = columns.blocks
            pointers = [int(block.data.ptr) for block in given]
            with self.assertRaises(GeneralizedRitzStabilityError) as caught:
                group.ritz(self.op, columns, stages=False)
        kept = caught.exception.filtered
        self.assertEqual((kept.ranges, group.passes, group.separate_row_blocks), (ranges, 0, 0))
        self.assertEqual([int(block.data.ptr) for block in kept.blocks], pointers)
        # The columns were gone from their blocks, and every entry is back as it was filtered.
        for index, (start, stop) in enumerate(ranges):
            self.assertFalse(np.array_equal(held[index], basis[:, start:stop]))
        np.testing.assert_array_equal(cp.asnumpy(group.gather(kept)), basis)
        # The way back was taken without a rotation.
        self.assertGreater(min(group.seconds[stage] for stage in ('apply', 'to_rows', 'gram', 'to_columns')), 0.)
        self.assertEqual(group.seconds['rotate'], 0.)

    def test_one_tall_block_and_the_slab_workspace_are_the_peak(self):
        import importlib
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import SectorDeviceGroup
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import solve_whitened_ritz_on_device
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp = self.cp
        # Tall blocks of megabytes, so that the Gram matrices are small beside them: 32 states make 8 kB each.
        rows, states = 65537, 32
        op = _hamiltonian(rows)
        op.distributed_filter_devices = self.devices
        group = SectorDeviceGroup(op, self.devices)
        blocks = uniform_filter_blocks(states, 4, 5)
        ranges = group.column_ranges(blocks)
        basis = self._filtered(group, blocks, ranges, 43)
        count = len(self.devices)
        tallest = -(-rows // count)
        # One column per slab and per exchange chunk.  The owner estimates the condition number of the overlap
        # itself, as in the solve that is measured alone below: a helper would take a copy of the overlap and
        # the work arrays of its spectrum on another device.
        sizes = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*rows), PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*512),
                     PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed', PARSEC_CUPY_RITZ_EIGH_BACKEND='host',
                     PARSEC_CUPY_RITZ_CONDITION_HELPER='off')
        # An array must not outlive the pool it was taken from.
        pools = []

        def taken(function):
            """What ``function`` takes from a pool of its own on every device, and what it returns."""
            pool, previous = cp.cuda.MemoryPool(), cp.cuda.get_allocator()
            pools.append(pool)
            cp.cuda.set_allocator(pool.malloc)
            try:
                result = function()
            finally:
                cp.cuda.set_allocator(previous)
            reserved = []
            for device in self.devices:
                with cp.cuda.Device(device):
                    reserved.append(pool.total_bytes())
            return reserved, result

        given = []

        def recorded(raw_overlap, raw_projection):
            # Host copies: a reference to the Gram pair would keep its pool block from the rest of the step.
            given[:] = cp.asnumpy(raw_overlap), cp.asnumpy(raw_projection)
            return solve_whitened_ritz_on_device(raw_overlap, raw_projection)

        # A tall block of the step with two holds the widest column block and the tallest row block.
        tall = max(rows*max(stop - start for start, stop in ranges), tallest*states)
        _whitened = rotated = columns = None
        try:
            for backend in ('host', 'device'):
                values, reserved, solving = {}, {}, 0
                with patch.dict(os.environ, PARSEC_CUPY_RITZ_DENSE_BACKEND=backend, **sizes), \
                     patch.object(module, 'solve_whitened_ritz_on_device', recorded):
                    for tall_blocks in (2, 1):
                        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=str(tall_blocks)):
                            # The columns exist already, in blocks with the capacity of a trial basis of this
                            # step: only what the Ritz step requests is taken from its pool.
                            columns = self._placed(group, basis, ranges)
                            reserved[tall_blocks], (values[tall_blocks], rotated, _whitened) = taken(
                                lambda: group.ritz(op, columns, stages=False))
                        vectors = cp.asnumpy(group.gather(rotated))
                        np.testing.assert_allclose(vectors.T @ vectors, np.eye(states), atol=1e-9)
                    # No step with one tall block took a row block from the pool.
                    self.assertEqual(group.separate_row_blocks, 0)
                    if backend == 'device':
                        # cuSOLVER sizes its work arrays itself, so what the small solve takes on the owner is
                        # measured: the same solve of the same Gram pair in a pool of its own.
                        with cp.cuda.Device(group.owner):
                            overlap, projection = (cp.asarray(part) for part in given)
                            solving = taken(lambda: solve_whitened_ritz_on_device(overlap, projection))[0][0]
                            del overlap, projection
                        self.assertGreater(solving, 0)
                # Values are compared more closely on the small operator; this test is about memory.
                np.testing.assert_allclose(values[1], values[2], rtol=0, atol=5e-11)
                chunk, work = group._chunk, group._work
                # A chunk is one column outside the rows of a device; the workspace is one column in all rows and,
                # in the rows of the tallest row block, one column of every device.  A tall block of the step with
                # one has the room of a row of all devices per row of the tallest row block besides, and a group
                # keeps the largest capacity that it has been asked for.
                self.assertEqual((chunk, work), (rows - rows // count, rows + tallest*count))
                self.assertEqual(group._tall, tall + count*tallest)
                # The step takes the slab workspace and one exchange buffer, once each and for all of its
                # exchanges, each rounded up to 512 bytes by the pool, and no row block: the rows lie in the
                # blocks of the columns, which were there before.
                held = 8*(work + chunk)
                # Besides them a device takes its Gram pair, two m x m arrays, the overlap product the pair is
                # filled from, a third, the products of the rounds and of single slabs of the overlap, and the
                # copy of the coefficients.  The pool may serve these from blocks of other sizes than those
                # that were freed, so twice the pair more is allowed for them.
                pair = 8*2*states*states
                small = 4*pair + 8*512
                for index in range(count):
                    # The owner of a device solve also takes the pair it receives, the pair in the order of the
                    # basis with the m x m arrays it is gathered from, the coefficients in the order of the row
                    # blocks, and the arrays of that solve.  These are requested on the calling thread, whose
                    # blocks the pool keeps apart from those of the group: eight times the pair is allowed.
                    owner = 8*pair + 8*512 + solving if backend == 'device' and self.devices[index] == group.owner else 0
                    message = (f'{backend} solve, device {index}: the step with one tall block takes {reserved[1]} '
                               f'bytes on the devices, of them {held} in workspace and buffer; {small} more are '
                               f'allowed for the Gram arrays and {owner} for the small solve, which alone takes '
                               f'{solving}; a tall block is {8*tall} bytes and the step with two takes {reserved[2]}')
                    # All that the device may take besides workspace and buffer is far less than a tall block:
                    # the bound below cannot let one pass.
                    self.assertLess(small + owner, 8*tall // 2, message)
                    self.assertGreaterEqual(reserved[1][index], held, message)
                    self.assertLessEqual(reserved[1][index], held + small + owner, message)
                    # With two tall blocks the row block comes on top.
                    self.assertGreaterEqual(reserved[2][index], held + 8*tall, message)
        finally:
            # What a step was given and returned goes before the pools of the steps, which then keep nothing
            # to free later.
            _whitened = rotated = columns = None
            self._empty(*pools)

    def test_the_helper_of_the_condition_number_takes_what_its_estimate_takes_alone(self):
        # Written where there is one CUDA device: this test had not run when it was added.
        import importlib
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import SectorDeviceGroup
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import solve_whitened_ritz_on_device
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp = self.cp
        # Tall blocks of megabytes, so that the arrays of the small problem are small beside them.
        rows, states = 65537, 32
        op = _hamiltonian(rows)
        op.distributed_filter_devices = self.devices
        group = SectorDeviceGroup(op, self.devices)
        blocks = uniform_filter_blocks(states, 4, 5)
        ranges = group.column_ranges(blocks)
        basis = self._filtered(group, blocks, ranges, 43)
        sizes = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*rows), PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*512),
                     PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed', PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='1',
                     PARSEC_CUPY_RITZ_DENSE_BACKEND='device', PARSEC_CUPY_RITZ_CONDITION='symmetric')
        # An array must not outlive the pool it was taken from.
        pools = []

        def taken(function):
            """What ``function`` takes from a pool of its own on every device, and what it returns."""
            pool, previous = cp.cuda.MemoryPool(), cp.cuda.get_allocator()
            pools.append(pool)
            cp.cuda.set_allocator(pool.malloc)
            try:
                result = function()
            finally:
                cp.cuda.set_allocator(previous)
            reserved = []
            for device in self.devices:
                with cp.cuda.Device(device):
                    reserved.append(pool.total_bytes())
            return reserved, result

        given = []

        def recorded(raw_overlap, raw_projection, **beside):
            # A host copy: a reference to the overlap would keep its pool block from the rest of the step.
            given[:] = [cp.asnumpy(raw_overlap)]
            return solve_whitened_ritz_on_device(raw_overlap, raw_projection, **beside)

        _whitened = rotated = columns = None
        try:
            reserved, values, estimated = {}, {}, {}
            for policy in ('off', 'auto'):
                with patch.dict(os.environ, PARSEC_CUPY_RITZ_CONDITION_HELPER=policy, **sizes), \
                     patch.object(module, 'solve_whitened_ritz_on_device', recorded):
                    # The columns exist already, in blocks with the capacity of a trial basis of this step:
                    # only what the Ritz step requests is taken from its pool.
                    columns = self._placed(group, basis, ranges)
                    reserved[policy], (values[policy], rotated, _whitened) = taken(
                        lambda: group.ritz(op, columns, stages=False))
                estimated[policy] = group.condition_device
            # The second device of the sector estimated; the operator names no device of Hartree objects.
            helper = self.devices[1]
            self.assertEqual(estimated, dict(off=group.owner, auto=helper))
            self.assertGreater(group.condition_seconds, 0.)
            np.testing.assert_array_equal(values['auto'], values['off'])
            # cuSOLVER sizes the work array of the spectrum itself, so what the estimate takes on the helper is
            # measured: the same estimate of the same overlap, copied from the owner, in a pool of its own.
            lower = np.tril(given[0])
            with patch.dict(os.environ, **sizes), cp.cuda.Device(group.owner):
                overlap = cp.asarray(lower + np.tril(lower, -1).T)
                cp.cuda.get_current_stream().synchronize()
                alone, number = taken(lambda: group._condition_beside(helper)(overlap)())
                del overlap
            self.assertTrue(np.isfinite(number))
            self.assertGreater(alone[1], 8*states*states)
            self.assertEqual([took for index, took in enumerate(alone) if index != 1], [0]*(len(self.devices) - 1))
            # The pool may serve the arrays of the small problem from blocks of other sizes than those that
            # were freed, and the owner no longer has the blocks of the spectrum to serve its solve from: twice
            # the Gram pair more is allowed on every device.
            small = 2*8*2*states*states
            for index in range(len(self.devices)):
                beside = alone[1] if index == 1 else 0
                message = (f'device {index}: the step takes {reserved["auto"]} bytes on the devices with a helper '
                           f'and {reserved["off"]} without; the estimate alone takes {alone}')
                # With a helper a device takes what it took without one, and the helper what its estimate
                # takes alone: far less than a tall block.
                self.assertLessEqual(reserved['auto'][index], reserved['off'][index] + beside + small, message)
                self.assertLess(beside + small, 8*group._tall // 2, message)
        finally:
            # What a step was given and returned goes before the pools of the steps, which then keep nothing
            # to free later.
            _whitened = rotated = columns = None
            self._empty(*pools)

    def test_one_tall_block_takes_the_workspace_of_the_slabs_that_were_cut(self):
        # Written where there is no CUDA device: this test had not run when it was added.
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import (
            SectorDeviceGroup, block_rounds, slab_columns, slab_workspace)
        devices, states = self.devices[:2], 23
        # A group of its own, which has taken no workspace yet: a group keeps the largest one.
        op = _hamiltonian(self.rows)
        op.distributed_filter_devices = devices
        group = SectorDeviceGroup(op, devices)
        blocks = uniform_filter_blocks(states, 6, 5)
        # Two filter blocks on each device: 12 and 11 columns.
        ranges = ((0, 12), (12, states))
        basis = self._filtered(group, blocks, ranges, 37)
        common = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400), PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed')
        with patch.dict(os.environ):
            # The budget of a basis this small is half of an even share of the columns: 5.
            os.environ.pop('PARSEC_CUPY_STREAMING_RITZ_BYTES', None)
            budget = slab_columns(self.rows, states, 2, blocks=1)
            allowed = slab_workspace(self.rows, states, 2, 1)
            values, rotated, _gram = self._ritz_step(group, basis, ranges, 1, roomy=True, **common)
        self._ritz_vectors(op, values, rotated)
        # The wider block takes three rounds, of 4 columns at most where the budget allows 5.
        rounds = block_rounds([stop - start for start, stop in ranges], budget)
        widest = max(last - first for taken in rounds for first, last in taken)
        joined = max(sum(last - first for first, last in taken) for taken in rounds)
        self.assertEqual((budget, len(rounds), widest, joined), (5, 3, 4, 8))
        # H times the widest slab in all rows and, behind it, the slab on its way or the rows of a device of
        # a whole round, whichever is more: less than the workspace of slabs of the budget.
        tallest = -(-self.rows // 2)
        self.assertEqual(group._work, self.rows*widest + max(tallest*joined, self.rows*widest))
        self.assertLess(group._work, allowed)
        self.assertEqual(group.separate_row_blocks, 0)

    def test_solver_with_one_tall_block_keeps_the_rows_in_its_blocks_through_a_truncation(self):
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        from parsec_python.Eigensolvers.eigval import EigvalSettings
        from parsec_python.acceleration.Eigensolvers.distributed_state import DistributedBasis, equal_slabs_requested
        from parsec_python.acceleration.Eigensolvers.eigval import CuPyEigvalSolver
        from parsec_python.acceleration.Eigensolvers.subspace import SubspaceSettings
        # 49 states and 4 more to work with, then 43: the later passes re-lay fewer columns than the blocks of
        # the trial basis were made for.
        settings = EigvalSettings(initial_method='chebff', safety_buffer=4, chebff=ChebFFSettings(polynomial_degree=12),
                                  subspace=SubspaceSettings(polynomial_degree=5, degree_delta=1))
        values = {}
        # A shared basis with one tall block per device, left to the default and named, and with two, and
        # the basis on one device.
        for blocks in ('default', '1', '2', None):
            solver = CuPyEigvalSolver(self.op, settings=settings, retain_vectors_on_device=True,
                                      compute_subspace_residuals=False)
            with patch.dict(os.environ, PARSEC_CUPY_DEVICE_RANDOM='1', PARSEC_CUPY_RECYCLE_STATE='1',
                            PARSEC_CUPY_DISTRIBUTED_STATE='0' if blocks is None else '1',
                            PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed', **_COMMON):
                os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT', None)
                os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS', None)
                if blocks in ('1', '2'):
                    os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = blocks
                first = solver.solve(49)
                solver.truncate_state(43)
                later = [solver.solve(43) for _ in range(2)]
            self.assertEqual([result.solver_path for result in (first, *later)], ['chebff', 'subspace', 'subspace'])
            self.assertEqual(isinstance(later[-1].vectors, DistributedBasis), blocks is not None)
            values[blocks] = [result.eigenvalues for result in (first, *later)]
            if blocks in ('default', '1'):
                # The cycles of the first solve and both later passes took the generalized solve, and none of
                # them a row block from the pool.
                group = self.op._sector_device_group
                self.assertGreater(group.passes, 2)
                self.assertEqual((group.tall_blocks, group.separate_row_blocks, group.seconds['repartition']),
                                 (1, 0, 0.0))
                self.assertEqual(group.slab_cut, 'equal')
            elif blocks == '2':
                # The cut is left to the calling shell here, and only this step reads it.
                group = self.op._sector_device_group
                self.assertEqual((group.tall_blocks, group.slab_cut),
                                 (2, 'equal' if equal_slabs_requested() else 'full'))
        for blocks in ('default', '1', '2'):
            for mine, theirs in zip(values[blocks], values[None], strict=True):
                np.testing.assert_allclose(mine, theirs, rtol=0, atol=5e-11)

    # ----- two column ranges per device, the default layout (PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT) -----

    # Columns of the trial basis of these tests: nine filter blocks of six columns, the top one of five, so
    # that also each of four devices gets blocks of both halves.
    states = 53

    def _sectors(self):
        """The sector of all devices and, where there are more than two, one of two of them: (operator, group)."""
        from parsec_python.acceleration.Eigensolvers.distributed_state import SectorDeviceGroup
        sectors = [(self.op, self.group)]
        if len(self.devices) > 2:
            op = _hamiltonian(self.rows)
            op.distributed_filter_devices = self.devices[:2]
            sectors.append((op, SectorDeviceGroup(op, self.devices[:2])))
        return sectors

    def _interleaved(self, group, columns=None):
        """The layout that a trial basis of ``group`` is given by default: two column ranges per device."""
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        blocks = uniform_filter_blocks(self.states if columns is None else columns, 6, 5)
        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed'):
            os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT', None)
            layout = group.column_layout(blocks)
        self.assertEqual([len(part) for part in layout.parts], [2]*len(group.devices))
        self.assertEqual(layout.columns, blocks[-1].stop)
        return layout

    def test_interleaved_layout_is_the_default_and_every_device_gets_both_halves(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import ColumnLayout
        group, count = self.group, len(self.devices)
        trial = uniform_filter_blocks(self.states, 6, 5)
        contiguous = ColumnLayout(group.column_ranges(trial))
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT', None)
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_RANGES'] = 'fixed'
            layout = group.column_layout(trial)
            # The lower half of the nine blocks is the first four.  The top block is the last of the last device.
            lower, upper = zip(*layout.parts)
            self.assertEqual((lower[0][0], lower[-1][1], upper[0][0], upper[-1][1]), (0, 24, 24, 53))
            self.assertEqual({(start % 6, stop > start) for start, stop in lower + upper}, {(0, True)})
            # Fewer blocks in a half than devices: the ranges stay contiguous.
            few = uniform_filter_blocks(6*count + 1, 6, 5)
            self.assertEqual(group.column_layout(few), ColumnLayout(group.column_ranges(few)))
            # Named, it is the same layout; the former one has a range of neighbouring columns per device.
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT'] = 'interleaved'
            self.assertEqual(group.column_layout(trial), layout)
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT'] = 'contiguous'
            self.assertEqual(group.column_layout(trial), contiguous)
            # So have balanced ranges, which the layout follows where it is left unset.
            del os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT']
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_RANGES'] = 'balanced'
            self.assertEqual(group.column_layout(trial), contiguous)

    def test_two_column_ranges_per_device_are_filled_viewed_gathered_and_moved_exactly(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        cp = self.cp
        host = np.random.default_rng(31).normal(size=(self.rows, self.states))
        for _op, group in self._sectors():
            layout = self._interleaved(group)
            basis = group.scatter(cp.array(host, order='F'), layout)
            self.assertEqual((basis.shape, basis.layout), ((self.rows, self.states), layout))
            for index, (device, block) in enumerate(zip(group.devices, basis.blocks)):
                self.assertEqual((int(block.device.id), block.flags.f_contiguous), (device, True))
                np.testing.assert_array_equal(cp.asnumpy(block), host[:, _held(layout, index)])
            np.testing.assert_array_equal(cp.asnumpy(group.gather(basis)), host)
            with self.assertRaises(ValueError):
                basis.ranges
            with self.assertRaises(IndexError):
                basis[:, 3:9]
            # Leading columns are the leading columns of every block, in its memory: inside the top block, at the
            # end of a range, inside a range of the upper half and inside the lower half.
            for count in (self.states, 49, 42, 31, 20, 5):
                leading = basis[:, :count]
                self.assertEqual((leading.shape, leading.layout), ((self.rows, count), layout.leading(count)))
                for index, (view, block) in enumerate(zip(leading.blocks, basis.blocks)):
                    self.assertEqual(view.shape, (self.rows, leading.layout.widths[index]))
                    if view.shape[1]:
                        self.assertEqual(int(view.data.ptr), int(block.data.ptr))
                        np.testing.assert_array_equal(cp.asnumpy(view), host[:, _held(leading.layout, index)])
                np.testing.assert_array_equal(cp.asnumpy(group.gather(leading)), host[:, :count])
            # To one range per device and back: whole columns move.
            ranges = group.column_ranges(uniform_filter_blocks(self.states, 6, 5))
            moved = group.repartition(basis, ranges)
            self.assertEqual(moved.ranges, ranges)
            np.testing.assert_array_equal(cp.asnumpy(group.gather(moved)), host)
            back = group.repartition(moved, layout)
            self.assertEqual(back.layout, layout)
            for index, block in enumerate(back.blocks):
                np.testing.assert_array_equal(cp.asnumpy(block), host[:, _held(layout, index)])
            self.assertIs(group.repartition(back, layout), back)

    def test_exchange_with_two_column_ranges_per_device_moves_every_entry_exactly(self):
        cp = self.cp
        host = np.random.default_rng(37).normal(size=(self.rows, self.states))
        former = dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='0', PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0')
        for _op, group in self._sectors():
            whole = self._interleaved(group)
            edges = group.row_edges(self.rows)
            away = self.rows - self.rows // len(group.devices)
            # All columns, and leading ones up to inside the top block and inside a range of the upper half.
            for count in (self.states, 49, 31):
                layout = whole.leading(count)
                # One column per chunk, two, seven, so that chunks reach from one range into the other, and
                # everything at once; with both switches of the exchange as they are by default, and with both off.
                for size in (8, 16*away, 8*7*away, 1 << 30):
                    for switches in ({}, former):
                        with self.subTest(devices=group.devices, count=count, chunk_bytes=size, switches=switches), \
                             patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(size), **switches):
                            kept = group.scatter(cp.array(host, order='F'), whole)[:, :count].take()
                            pointers = [int(block.data.ptr) for block in kept]
                            rows = group._to_rows(kept, layout, edges, keep=True)
                            self.assertEqual([int(block.data.ptr) for block in kept], pointers)
                            # A row block holds all columns in their own order.
                            for index, device in enumerate(group.devices):
                                block = rows[index]
                                self.assertEqual((int(block.device.id), block.flags.f_contiguous), (device, True))
                                np.testing.assert_array_equal(cp.asnumpy(block), host[edges[index]:edges[index+1], :count])
                            fresh = group._to_columns(rows, layout, edges)
                            # Supplied column blocks are overwritten where they lie.
                            for device, block in zip(group.devices, kept):
                                with cp.cuda.Device(device):
                                    block.fill(np.nan)
                                    cp.cuda.get_current_stream().synchronize()
                            reused = group._to_columns(rows, layout, edges, targets=kept)
                            self.assertEqual([int(block.data.ptr) for block in reused], pointers)
                            for index, device in enumerate(group.devices):
                                for block in (fresh[index], reused[index]):
                                    self.assertEqual((int(block.device.id), block.flags.f_contiguous), (device, True))
                                    np.testing.assert_array_equal(cp.asnumpy(block), host[:, _held(layout, index)])

    def test_random_trial_basis_with_two_column_ranges_per_device_is_the_single_device_stream(self):
        from parsec_python.acceleration.Eigensolvers.lapack_random import LapackRandom
        cp = self.cp
        for _op, group in self._sectors():
            layout = self._interleaved(group)
            for flag in ('1', '0'):
                with self.subTest(devices=group.devices, device_random=flag), \
                     patch.dict(os.environ, PARSEC_CUPY_DEVICE_RANDOM=flag):
                    single, generator = LapackRandom(), LapackRandom()
                    expected = single.uniform_minus_1_1((self.rows, self.states), column_major=True)
                    basis = group.random_basis(generator, self.rows, layout)
                    self.assertEqual(basis.layout, layout)
                    for index, block in enumerate(basis.blocks):
                        np.testing.assert_array_equal(cp.asnumpy(block), expected[:, _held(layout, index)])
                    np.testing.assert_array_equal(cp.asnumpy(group.gather(basis)), expected)
                    # The stream goes on where one device would have left it.
                    self.assertEqual(tuple(generator.seed), tuple(single.seed))

    def test_density_with_two_column_ranges_per_device_is_the_sum_over_devices(self):
        from parsec_python.acceleration.Occupations.device_density import CuPyDeviceDensityBuilder
        cp = self.cp
        rng = np.random.default_rng(41)
        host = rng.normal(size=(self.rows, self.states))
        builder = CuPyDeviceDensityBuilder(cp)
        for _op, group in self._sectors():
            layout = self._interleaved(group)
            basis = group.scatter(cp.array(host, order='F'), layout)
            # All columns occupied, and leading ones: up to inside the top block, inside a range of the upper
            # half, and inside the lower half, where the last devices hold no occupied column.
            for count in (self.states, 49, 31, 14, 3):
                occupations = rng.uniform(size=count)
                expected = builder(cp.array(host[:, :count], order='F'), occupations, 0.37)
                for selected in (basis, basis[:, :count], basis[:, :min(self.states, count + 5)]):
                    with self.subTest(devices=group.devices, occupied=count, columns=selected.shape[1]):
                        np.testing.assert_allclose(selected.density(builder, occupations, 0.37), expected, rtol=1e-13, atol=0)

    def _one_device_filter(self, op, start, blocks, lower, upper, reference):
        """Host copy of ``start`` filtered on the device of ``op`` block after block, sigma carried on, without graphs."""
        from parsec_python.acceleration.Eigensolvers.chebyshev import chebyshev_filter
        cp = self.cp
        matrix = cp.array(start, order='F')
        filtered, carried = cp.empty_like(matrix), None
        for block in blocks:
            result, carried = chebyshev_filter(op, matrix[:, block.start:block.stop], block.degree, lower, upper, reference,
                                               initial_sigma=carried, return_final_sigma=True, mixed_precision=False)
            filtered[:, block.start:block.stop] = result
        return cp.asnumpy(filtered)

    def test_filter_with_two_column_ranges_per_device_matches_one_device(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import subspace_filter_blocks, uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import ColumnLayout
        cp = self.cp
        start = np.linalg.qr(np.random.default_rng(43).normal(size=(self.rows, self.states)))[0]
        # The reference lies below the interval, as in a first solve: sigma then changes from block to block,
        # and the second range of a device must not continue the recurrence of its first.  A later pass has its
        # reference on the lower bound, where sigma stays -1.
        lower, upper, reference = 1.2, 6., .4
        plans = (uniform_filter_blocks(self.states, 6, 5), subspace_filter_blocks(self.states, 6, 5, 1),
                 subspace_filter_blocks(49, 6, 5, 1), subspace_filter_blocks(31, 6, 5, 1))
        for op, group in self._sectors():
            layouts = dict(interleaved=self._interleaved(group),
                           contiguous=ColumnLayout(group.column_ranges(uniform_filter_blocks(self.states, 6, 5))))
            for blocks in plans:
                count, degrees = blocks[-1].stop, sorted({block.degree for block in blocks})
                expected = self._one_device_filter(op, start[:, :count], blocks, lower, upper, reference)
                for name, layout in layouts.items():
                    with self.subTest(devices=group.devices, columns=count, degrees=degrees, layout=name), \
                         patch.dict(os.environ, PARSEC_CUPY_MIXED_FILTER='off', PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed'):
                        basis = group.scatter(cp.array(start, order='F'), layout)[:, :count]
                        filtered = group.filter(basis, blocks, lower, upper, reference, False)
                        # Every device filtered the columns it held, where it held them.
                        self.assertEqual((filtered.layout, group.layout), (layout.leading(count),)*2)
                        np.testing.assert_allclose(cp.asnumpy(group.gather(filtered)), expected, rtol=0,
                                                   atol=1e-12*np.abs(expected).max())
            self.assertEqual(group.seconds['repartition'], 0.0)

    def test_later_passes_with_interleaved_columns_match_contiguous_ranges_and_one_device(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        trial = uniform_filter_blocks(self.states, 6, 5)
        # Two or three columns per exchange chunk and two per slab, so that chunks reach from one column range
        # into the other and slabs end with their range.
        sizes = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400), PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*self.rows*2),
                     PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed')
        for op, whole in self._sectors():
            # Two tall blocks, three and one, the default, with the small solve on the device, its default,
            # and on the host.  The basis of these passes starts in blocks that are only as large as their
            # columns, so the step with one tall block takes most of its row blocks from the pool here.
            for blocks, backend in (('2', 'device'), ('3', 'device'), ('1', 'device'), ('2', 'host'), ('3', 'host'),
                                    ('1', 'host')):
                with self.subTest(devices=whole.devices, blocks=blocks, backend=backend):
                    passes = {}
                    for name in ('interleaved', 'contiguous'):
                        group = type(whole)(op, whole.devices)
                        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=blocks,
                                        PARSEC_CUPY_RITZ_DENSE_BACKEND=backend, **sizes):
                            # Interleaved columns are the default and are left to it; contiguous ranges are named.
                            os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT', None)
                            if name == 'contiguous':
                                os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT'] = name
                            initial = group.column_layout(trial)
                            self.assertEqual([len(part) for part in initial.parts],
                                             [2 if name == 'interleaved' else 1]*len(group.devices))
                            # All columns, then 47 of them twice: the second of these passes has the plan of the first.
                            # The overlap of 53 filtered random columns has a condition number of 1e4, ten times that
                            # of the 23 of the other tests, so the Ritz values are compared four times less closely.
                            kept, passes[name] = self._later_passes(group, initial, self.states, (self.states, 47, 47), op,
                                                                    atol=2e-11)
                        # Every pass took its Ritz step with the columns where the trial basis had them.
                        self.assertEqual((kept, group.passes, group.seconds['repartition']), (initial, 3, 0.0))
                        self.assertEqual(group.layout, initial.leading(47))
                    # Each layout agrees with one device; with each other they differ by round-off too.
                    for (mine, vectors), (theirs, others) in zip(passes['interleaved'], passes['contiguous'], strict=True):
                        np.testing.assert_allclose(mine, theirs, rtol=0, atol=4e-11)
                        self._subspace(vectors, others, mine.size)

    def test_first_solve_matches_one_device(self):
        # The trial basis is drawn into two column ranges per device, the layout that it has where none is named,
        # and every cycle filters and solves it there.  The overlaps of the cycles of 53 states reach a condition
        # number of 3e5, a thousand times that of the 19 states of the other first solves, so the Ritz values are
        # compared ten times less closely.
        for blocks in ('2', '3', '1'):
            with self.subTest(blocks=blocks), patch.dict(os.environ):
                os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT', None)
                group = type(self.group)(self.op, self.devices)
                actual = self._first_solve_matches_one_device(
                    group, states=self.states, atol=5e-10, PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed',
                    PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=blocks, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8*400))
                self.assertEqual(actual.vectors.layout, self._interleaved(group))
                self.assertEqual(group.seconds['repartition'], 0.0)
                self.assertGreater(group.passes, 0)
                # With one tall block the rows of every cycle lay in the blocks of the trial basis.
                self.assertEqual(group.separate_row_blocks, 0)

    def test_unstable_overlap_with_interleaved_columns_falls_back_to_the_orthonormal_route(self):
        # The fallback gathers the columns of both ranges on the owner and hands them back to where they were.
        layout = self._interleaved(self.group)
        for blocks in ('2', '3', '1'):
            with self.subTest(blocks=blocks):
                actual = self._unstable_overlap_falls_back_to_the_orthonormal_route(
                    self.states, layout, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=blocks,
                    PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed')
                self.assertEqual(actual.vectors.layout, layout)

    def test_solver_keeps_interleaved_columns_through_a_truncation(self):
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        from parsec_python.Eigensolvers.eigval import EigvalSettings
        from parsec_python.acceleration.Eigensolvers.distributed_state import DistributedBasis
        from parsec_python.acceleration.Eigensolvers.eigval import CuPyEigvalSolver
        from parsec_python.acceleration.Eigensolvers.subspace import SubspaceSettings
        from parsec_python.acceleration.Occupations.device_density import CuPyDeviceDensityBuilder
        cp = self.cp
        # 49 states and 4 more to work with: the states that a solve returns end inside the top block.
        settings = EigvalSettings(initial_method='chebff', safety_buffer=4, chebff=ChebFFSettings(polynomial_degree=12),
                                  subspace=SubspaceSettings(polynomial_degree=5, degree_delta=1))
        builder = CuPyDeviceDensityBuilder(cp)
        occupations = np.random.default_rng(47).uniform(size=43)
        values, densities, layouts = {}, {}, {}
        # A shared basis in either layout, and the basis on one device.
        for name in ('interleaved', 'contiguous', None):
            solver = CuPyEigvalSolver(self.op, settings=settings, retain_vectors_on_device=True,
                                      compute_subspace_residuals=False)
            with patch.dict(os.environ, PARSEC_CUPY_DEVICE_RANDOM='1', PARSEC_CUPY_RECYCLE_STATE='1',
                            PARSEC_CUPY_DISTRIBUTED_STATE='0' if name is None else '1',
                            PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed', **_COMMON):
                # Interleaved columns are the default and are left to it; contiguous ranges are named.
                os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT', None)
                if name == 'contiguous':
                    os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT'] = name
                first = solver.solve(49)
                solver.truncate_state(43)
                later = [solver.solve(43) for _ in range(2)]
            self.assertEqual([result.solver_path for result in (first, *later)], ['chebff', 'subspace', 'subspace'])
            self.assertEqual(isinstance(later[-1].vectors, DistributedBasis), name is not None)
            self.assertEqual(tuple(later[-1].vectors.shape), (self.rows, 43))
            values[name] = [result.eigenvalues for result in (first, *later)]
            if name is None:
                densities[name] = builder(later[-1].vectors, occupations, 0.37)
            else:
                densities[name] = later[-1].vectors.density(builder, occupations, 0.37)
                layouts[name] = [result.vectors.layout for result in (first, *later)]
        # The 53 columns of the trial basis stay where they were laid out: 49 and then 43 of them are returned.
        whole = self._interleaved(self.group)
        self.assertEqual(layouts['interleaved'], [whole.leading(49), whole.leading(43), whole.leading(43)])
        self.assertEqual([len(part) for part in layouts['contiguous'][0].parts], [1]*len(self.devices))
        self.assertEqual(self.op._sector_device_group.seconds['repartition'], 0.0)
        for name in ('interleaved', 'contiguous'):
            for mine, theirs in zip(values[name], values[None], strict=True):
                np.testing.assert_allclose(mine, theirs, rtol=0, atol=5e-11)
            # Ritz vectors agree up to round-off over the gaps between their values, and so do their densities.
            np.testing.assert_allclose(densities[name], densities[None], rtol=1e-7, atol=0)


if __name__ == '__main__':
    unittest.main()
