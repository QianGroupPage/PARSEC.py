"""``PARSEC_CUPY_RITZ_PIPELINE`` and ``PARSEC_CUPY_EXCHANGE_STRIDED`` against the step as it was, bit for bit.

The host tests read the switches and what the rounds and the pitched copies are cut into.  The tests of the
pipelined rounds on the NumPy stand-in, whose streams hold their work back until something waits for it, are in
test_distributed_exchange.

On a CUDA device the Ritz step of a sector basis shared by one GPU named twice and four times, and by distinct
GPUs where there are two or four, is taken with every value of both switches and compared with ``off`` and the
packed exchange to the last bit: the Ritz values, the rotated basis and the Gram pair that the small solve was
given.  A GPU named several times runs the compute stream and the exchange stream of its names side by side, so
the waits between the two streams are tested there; the copies between the names stay inside the device, and the
names take their turns one after another on the one thread of the device.
"""
import gc
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy import CuPyHamiltonian, cupy_available, require_cupy
from parsec_python.acceleration.Eigensolvers import distributed_state
from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks


def _device_count():
    if not cupy_available():
        return 0
    cp, _ = require_cupy()
    return int(cp.cuda.runtime.getDeviceCount())


# The settings of the shared route as the tests of test_distributed_state give them.
_COMMON = dict(PARSEC_CUPY_GENERALIZED_RITZ='on', PARSEC_CUPY_MIXED_FILTER='off', PARSEC_CUPY_DISTRIBUTED_FILTER='0',
               PARSEC_CUPY_FILTER_GRAPHS='1', PARSEC_CUPY_FILTER_COLUMN_MAJOR='1', PARSEC_CUPY_RITZ_ROTATION='reuse',
               PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='1', PARSEC_CUPY_STREAMING_RITZ='0',
               PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed')
# Every switch that these tests set, removed first whatever the calling shell exports.
_NAMED = ('PARSEC_CUPY_RITZ_PIPELINE', 'PARSEC_CUPY_EXCHANGE_STRIDED', 'PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS',
          'PARSEC_CUPY_DISTRIBUTED_STATE_OVERLAP_COLUMNS', 'PARSEC_CUPY_RITZ_GRAM_SLABS', 'PARSEC_CUPY_RITZ_SYRK',
          'PARSEC_CUPY_RITZ_DENSE_BACKEND', 'PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_EXCHANGE_CHUNK_BYTES',
          'PARSEC_CUPY_EXCHANGE_CONCURRENT', 'PARSEC_CUPY_EXCHANGE_PAIRS', 'PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT')
# The values compared with the first, the step as it was.
_VALUES = (('off', '0'), ('p1', '0'), ('p2', '0'), ('off', '1'), ('p1', '1'), ('p2', '1'))


class SwitchTests(unittest.TestCase):
    """What the two switches accept, and what the rounds and the pitched copies are cut into: on the host."""

    def test_the_pipeline_is_off_unless_p1_or_p2_is_named(self):
        # off, the step of the code of 21 October, is the default, also where the variable is empty; p2 was the
        # default of the code published on 2026-10-22.
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_RITZ_PIPELINE', None)
            self.assertEqual(distributed_state.ritz_pipeline(), 'off')
            for value, expected in (('', 'off'), ('off', 'off'), (' P1 ', 'p1'), ('p2', 'p2'), ('OFF', 'off')):
                os.environ['PARSEC_CUPY_RITZ_PIPELINE'] = value
                self.assertEqual(distributed_state.ritz_pipeline(), expected, value)
            for value in ('on', '1', 'p3', 'pipelined'):
                os.environ['PARSEC_CUPY_RITZ_PIPELINE'] = value
                with self.assertRaisesRegex(ValueError, 'off, p1 or p2'):
                    distributed_state.ritz_pipeline()

    def test_the_exchange_is_packed_unless_pitched_copies_are_named(self):
        # The packed exchange of the code of 21 October is the default; pitched copies (1) were the default of
        # the code published on 2026-10-22.
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_EXCHANGE_STRIDED', None)
            self.assertFalse(distributed_state.strided_exchange_requested())
            for value, expected in (('1', True), ('on', True), ('TRUE', True), ('0', False), ('off', False), ('false', False)):
                os.environ['PARSEC_CUPY_EXCHANGE_STRIDED'] = value
                self.assertEqual(distributed_state.strided_exchange_requested(), expected, value)
            os.environ['PARSEC_CUPY_EXCHANGE_STRIDED'] = 'pitched'
            with self.assertRaisesRegex(ValueError, 'on or off'):
                distributed_state.strided_exchange_requested()

    def test_a_mistyped_value_stops_the_run_where_the_trial_basis_is_sized(self):
        group = distributed_state.SectorDeviceGroup.__new__(distributed_state.SectorDeviceGroup)
        group.devices, group._tall, group._chunk = (0, 1), 0, 0
        layout = distributed_state.ColumnLayout(((0, 6), (6, 12)))
        for blocks in ('1', '2', '3'):
            for name, value in (('PARSEC_CUPY_RITZ_PIPELINE', 'p3'), ('PARSEC_CUPY_EXCHANGE_STRIDED', 'yes')):
                with self.subTest(blocks=blocks, name=name), patch.dict(os.environ):
                    for other in _NAMED:
                        os.environ.pop(other, None)
                    os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = blocks
                    group._capacities(layout, [0, 50, 100])
                    os.environ[name] = value
                    with self.assertRaises(ValueError):
                        group._capacities(layout, [0, 50, 100])

    def test_every_slab_of_the_overlap_is_formed_after_the_round_that_holds_its_first_column(self):
        by_round = distributed_state.SectorDeviceGroup._overlap_by_round
        # Rounds fill the row block from its end: 64 columns at [4864, 4928) first, then 38 of 128 in front.
        spots = [(4864, 4928)] + [(4864 - 128 * (index + 1), 4864 - 128 * index) for index in range(38)]
        with patch.dict(os.environ):
            for name in _NAMED:
                os.environ.pop(name, None)
            slabs = by_round(4928, 128, spots)
            # Every slab of 128 columns lies in one round, the last one in time first.
            self.assertEqual(sorted(first for cut in slabs.values() for first, _last in cut), list(range(0, 4928, 128)))
            self.assertEqual(slabs[0], [(4864, 4928)])
            self.assertEqual(slabs[38], [(0, 128)])
            self.assertEqual(set(map(len, slabs.values())), {1})
            # Two devices: rounds of 64 columns, so a slab of 128 waits for the second of its two rounds, whose
            # spot holds its first column.
            pairs = [(3640 - 64 * index, 3704 - 64 * index) for index in range(57)] + [(0, 56)]
            slabs = by_round(3704, 128, pairs)
            for step, cut in slabs.items():
                for first, last in cut:
                    low, high = pairs[step]
                    self.assertTrue(low <= first < high, (step, first))
                    self.assertEqual(last, min(3704, first + 128))
            self.assertEqual(sum(len(cut) for cut in slabs.values()), -(-3704 // 128))
            # A count of slabs is cut as the product cuts it: three slabs of 18 columns of 53.
            os.environ['PARSEC_CUPY_RITZ_GRAM_SLABS'] = '3'
            spots = [(45, 53), (37, 45), (29, 37), (21, 29), (13, 21), (5, 13), (0, 5)]
            slabs = by_round(53, None, spots)
            self.assertEqual(sorted(cut for cuts in slabs.values() for cut in cuts), [(0, 18), (18, 36), (36, 53)])
            self.assertEqual(slabs, {6: [(0, 18)], 4: [(18, 36)], 2: [(36, 53)]})
            del os.environ['PARSEC_CUPY_RITZ_GRAM_SLABS']
            # No slabs: one slab for the whole basis, DSYRK or the full product.
            self.assertEqual(by_round(53, 53, spots), {})
            self.assertEqual(by_round(53, 128, spots), {})
            for policy in ('on', 'off'):
                os.environ['PARSEC_CUPY_RITZ_SYRK'] = policy
                self.assertEqual(by_round(53, 8, spots), {})
            os.environ['PARSEC_CUPY_RITZ_SYRK'] = 'slabs'
            self.assertEqual(len(by_round(53, 8, spots)), len({step for step in range(7) for first in range(0, 53, 8)
                                                                if spots[step][0] <= first < spots[step][1]}))

    def test_a_pitched_copy_takes_a_line_per_column_with_the_pitch_of_each_array(self):
        calls = []
        runtime = SimpleNamespace(memcpyDefault=4, memcpy2DAsync=lambda *args: calls.append(args))
        cp = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
        stream = SimpleNamespace(ptr=77)

        class View:
            """Stands for a CuPy view: its shape, strides, item size and first address."""

            def __init__(self, shape, strides, address):
                self.shape, self.strides, self.itemsize = shape, strides, 8
                self.data = SimpleNamespace(ptr=address)

        # Rows 100 to 140 of five columns of a block of 1,000 rows, into five whole columns of 40 rows.
        source = View((40, 5), (8, 8000), 1000 + 8 * 100)
        target = View((40, 5), (8, 320), 50000)
        distributed_state._pitched_copy(cp, target, source, stream)
        self.assertEqual(calls, [(50000, 320, 1800, 8000, 320, 5, 4, 77)])
        # One column: its pitch is its own length.
        calls.clear()
        distributed_state._pitched_copy(cp, View((40, 1), (8, 0), 9), View((40, 1), (8, 8), 17), stream)
        self.assertEqual(calls, [(9, 320, 17, 320, 320, 1, 4, 77)])
        # Nothing to copy, nothing called.
        calls.clear()
        distributed_state._pitched_copy(cp, View((0, 5), (8, 0), 9), View((0, 5), (8, 0), 17), stream)
        self.assertEqual(calls, [])
        # Views of other shapes, row-major views, and columns closer than their rows are refused.
        for bad, message in ((View((40, 4), (8, 8000), 1), 'one shape'), (View((40, 5), (40, 8), 1), 'column-major'),
                             (View((40, 5), (8, 16), 1), 'overlap')):
            with self.assertRaisesRegex(ValueError, message):
                distributed_state._pitched_copy(cp, target, bad, stream)

    def test_peer_access_is_asked_for_once_per_pair_and_never_of_a_device_itself(self):
        asked, enabled = [], []

        class Error(Exception):
            def __init__(self, status):
                super().__init__(status)
                self.status = status

        def enable(peer):
            enabled.append(peer)
            if peer == 2:
                raise Error(704)

        runtime = SimpleNamespace(deviceCanAccessPeer=lambda device, peer: asked.append((device, peer)) or peer != 3,
                                  deviceEnablePeerAccess=enable, CUDARuntimeError=Error)

        class Device:
            def __init__(self, device):
                self.device = device

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        cp = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime, Device=Device))
        with patch.object(distributed_state, '_PEER_ACCESS', set()):
            for device, peer in ((0, 1), (0, 1), (1, 1), (0, 2), (0, 3), (1, 0)):
                distributed_state._peer_access(cp, device, peer)
            # Already enabled by someone else is no error, another error is.
            self.assertEqual((asked, enabled), ([(0, 1), (0, 2), (0, 3), (1, 0)], [1, 2, 0]))
            runtime.deviceEnablePeerAccess = lambda peer: (_ for _ in ()).throw(Error(1))
            with self.assertRaises(Error):
                distributed_state._peer_access(cp, 2, 0)


def _hamiltonian(n):
    """The operator of test_distributed_state: a small Hamiltonian with projectors on the current device."""
    return CuPyHamiltonian(sp.diags((-np.ones(n - 1), 2.2 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1)),
                           np.linspace(-.1, .1, n),
                           (sp.csr_matrix(np.random.default_rng(8).normal(size=(n, 2)) * .002), np.array([1., -1.])),
                           retain_generic_laplacian=False)


def _named_groups():
    """The devices of the groups to test: one GPU named twice and four times, distinct GPUs where they exist."""
    groups = [(0, 0), (0, 0, 0, 0)]
    count = _device_count()
    if count >= 2:
        groups.append((0, 1))
    if count >= 4:
        groups.append((0, 1, 2, 3))
    return groups


@unittest.skipUnless(_device_count() >= 1, 'a CUDA device required')
class PipelineDeviceTests(unittest.TestCase):
    rows, states = 257, 53

    def setUp(self):
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()

    def tearDown(self):
        gc.collect()

    def _group(self, devices, rows=None):
        op = _hamiltonian(self.rows if rows is None else rows)
        op.distributed_filter_devices = devices
        return op, distributed_state.SectorDeviceGroup(op, devices)

    def _layout(self, devices, states=None):
        blocks = uniform_filter_blocks(self.states if states is None else states, 6, 5)
        ranges = distributed_state.interleaved_ranges(blocks, len(devices))
        return distributed_state.ColumnLayout(ranges)

    @staticmethod
    def _basis(rows, states, seed=71):
        """Like a filtered basis: far from orthonormal, but well conditioned."""
        rng = np.random.default_rng(seed)
        return np.linalg.qr(rng.normal(size=(rows, states)))[0] @ (np.eye(states) + .2 * rng.normal(size=(states, states)))

    def _placed(self, group, basis, layout):
        """``basis`` in column blocks with the room of a trial basis, as a run gives them."""
        cp = self.cp
        rows = basis.shape[0]
        tall, _chunk = group._capacities(layout, group.row_edges(rows))
        blocks = []
        for index, device in enumerate(group.devices):
            with cp.cuda.Device(device):
                block = group._block(cp, (rows, layout.widths[index]), tall)
                if block.size:
                    held = [column for start, stop in layout.parts[index] for column in range(start, stop)]
                    block.set(np.asfortranarray(basis[:, held]))
                    cp.cuda.get_current_stream().synchronize()
                blocks.append(block)
        return distributed_state.DistributedBasis(group, blocks, layout)

    def _environment(self, mode, strided, **settings):
        """The settings of a test: ``None`` for either switch leaves it unset, at its default."""
        environment = dict(_COMMON, PARSEC_CUPY_RITZ_PIPELINE=mode, PARSEC_CUPY_EXCHANGE_STRIDED=strided)
        environment.update(settings)
        return {name: value for name, value in environment.items() if value is not None}

    def _step(self, group, basis, layout, mode, strided, **settings):
        """One Ritz step of ``basis``: its values, the rotated basis and the Gram pair given to the small solve."""
        cp = self.cp
        with patch.dict(os.environ):
            for name in _NAMED:
                os.environ.pop(name, None)
            os.environ.update(self._environment(mode, strided, **settings))
            columns = self._placed(group, basis, layout)
            pointers = [int(block.data.ptr) for block in columns.blocks]
            on_device = distributed_state.dense_solve_on_device()
            solver = 'solve_whitened_ritz_on_device' if on_device else 'solve_whitened_ritz'
            with patch.object(distributed_state, solver, wraps=getattr(distributed_state, solver)) as solve:
                values, rotated, _whitened = group.ritz(group._operator(group.owner), columns, stages=False)
        solve.assert_called_once()
        gram = [cp.asnumpy(part) if on_device else np.array(part) for part in solve.call_args.args]
        # The rotated basis lies in the blocks that held the filtered one.
        self.assertEqual([int(block.data.ptr) for block in rotated.blocks], pointers)
        return values, cp.asnumpy(group.gather(rotated)), gram

    def _assert_same(self, mine, theirs, message):
        (values, rotated, gram), (former, vectors, given) = mine, theirs
        np.testing.assert_array_equal(values, former, err_msg=message)
        np.testing.assert_array_equal(rotated, vectors, err_msg=message)
        for part, other in zip(gram, given, strict=True):
            np.testing.assert_array_equal(part, other, err_msg=message)

    def test_one_step_gives_the_bits_of_the_step_as_it_was_with_every_value(self):
        # Slabs of two columns, so that a step takes several rounds, and chunks of two or three columns, so that
        # a packed conversion takes several steps.  The overlap in slabs of 8 columns, in a count of three, by
        # DSYRK (no slabs: p2 forms it after the rounds as p1 does) and in the default 128, one slab for 53.
        cuts = ({'PARSEC_CUPY_DISTRIBUTED_STATE_OVERLAP_COLUMNS': '8'}, {'PARSEC_CUPY_RITZ_GRAM_SLABS': '3'},
                {'PARSEC_CUPY_RITZ_SYRK': 'on'}, {})
        basis = self._basis(self.rows, self.states)
        for devices in _named_groups():
            op, group = self._group(devices)
            layout = self._layout(devices)
            for backend in ('host', 'device'):
                for cut in cuts:
                    settings = dict(cut, PARSEC_CUPY_RITZ_DENSE_BACKEND=backend,
                                    PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2),
                                    PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 400))
                    steps = {}
                    for mode, strided in _VALUES:
                        message = f'devices {devices}, {backend} solve, {cut}, {mode}, strided {strided}'
                        with self.subTest(message):
                            steps[mode, strided] = self._step(group, basis, layout, mode, strided, **settings)
                            self._assert_same(steps[mode, strided], steps['off', '0'], message)
                            # What the step says it ran.
                            self.assertEqual((group.pipeline, group.stage_clock, group.exchange_copies),
                                             (mode, 'host' if mode == 'off' else 'events',
                                              'strided' if strided == '1' else 'packed'))
            # The rounds of every step were timed, and every stage took time.
            self.assertGreater(group.rounds_seconds, 0.)
            for stage in ('apply', 'to_rows', 'gram', 'dense', 'rotate', 'to_columns'):
                self.assertGreater(group.seconds[stage], 0., stage)
            self.assertGreater(group.overlap_seconds, 0.)

    def test_the_defaults_are_the_former_step_and_p2_with_pitched_copies_gives_its_bits(self):
        # Both switches unset: the rounds one after another and the packed exchange of the code of 21 October.
        # p2 with pitched copies, the defaults of the code published on 2026-10-22, gives the same bits.
        basis = self._basis(self.rows, self.states)
        for devices in _named_groups():
            op, group = self._group(devices)
            layout = self._layout(devices)
            for cut in ({'PARSEC_CUPY_DISTRIBUTED_STATE_OVERLAP_COLUMNS': '8'}, {}):
                settings = dict(cut, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2))
                with self.subTest(devices=devices, cut=cut):
                    default = self._step(group, basis, layout, None, None, **settings)
                    self.assertEqual((group.pipeline, group.stage_clock, group.exchange_copies), ('off', 'host', 'packed'))
                    self.assertGreater(group.exchange_buffer_bytes, 0)
                    # A group of its own, which has taken no exchange buffer.
                    fresh_op, fresh = self._group(devices)
                    opted = self._step(fresh, basis, layout, 'p2', '1', **settings)
                    del fresh_op
                    self.assertEqual((fresh.pipeline, fresh.stage_clock, fresh.exchange_copies), ('p2', 'events', 'strided'))
                    self.assertEqual(fresh.exchange_buffer_bytes, 0)
                    self._assert_same(opted, default, f'devices {devices}, {cut}: p2 with pitched copies against the defaults')

    def test_a_strided_exchange_takes_no_buffer(self):
        basis = self._basis(self.rows, self.states)
        for devices in _named_groups():
            for mode in ('off', 'p2'):
                op, group = self._group(devices)
                with self.subTest(devices=devices, mode=mode):
                    self._step(group, basis, self._layout(devices), mode, '1',
                               PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2))
                    self.assertEqual((group.exchange_buffer_bytes, group.exchange_copies), (0, 'strided'))
                    self._step(group, basis, self._layout(devices), mode, '0',
                               PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2))
                    self.assertGreater(group.exchange_buffer_bytes, 0)

    def test_steps_with_two_and_three_tall_blocks_read_no_pipeline_and_keep_their_bits_strided(self):
        basis = self._basis(self.rows, self.states)
        for devices in _named_groups():
            op, group = self._group(devices)
            layout = self._layout(devices)
            for blocks in ('2', '3'):
                settings = dict(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=blocks, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 400),
                                PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 3))
                reference = None
                for mode, strided in _VALUES:
                    message = f'devices {devices}, {blocks} tall blocks, {mode}, strided {strided}'
                    with self.subTest(message):
                        cp = self.cp
                        columns = group.scatter(cp.array(basis, order='F'), layout)
                        with patch.dict(os.environ):
                            for name in _NAMED:
                                os.environ.pop(name, None)
                            os.environ.update(self._environment(mode, strided, **settings))
                            values, rotated, _whitened = group.ritz(op, columns, stages=False)
                        step = values, cp.asnumpy(group.gather(rotated))
                        self.assertEqual((group.pipeline, group.stage_clock), ('off', 'host'))
                        if reference is None:
                            reference = step
                        np.testing.assert_array_equal(step[0], reference[0], err_msg=message)
                        np.testing.assert_array_equal(step[1], reference[1], err_msg=message)

    def _passes(self, devices, mode, strided, **settings):
        """A CHEBFF first solve, then two later passes, the second on fewer states: values and gathered bases."""
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState, SubspaceSettings
        cp = self.cp
        op, group = self._group(devices)
        found = []
        with patch.dict(os.environ):
            for name in _NAMED:
                os.environ.pop(name, None)
            os.environ.update(self._environment(mode, strided, PARSEC_CUPY_DEVICE_RANDOM='1', **settings))
            first = distributed_state.run_distributed_chebff(op, 19, settings=ChebFFSettings(polynomial_degree=12,
                                                                                            filter_cycles=4), group=group)
            found.append((cp.asnumpy(first.eigenvalues), cp.asnumpy(group.gather(first.vectors))))
            start = np.linalg.qr(np.random.default_rng(17).normal(size=(self.rows, self.states)))[0]
            state = DeviceSubspaceState(self.rows, self.states, cp.linspace(.1, 1.5, self.states),
                                        group.scatter(cp.array(start, order='F'), self._layout(devices)))
            pass_settings = SubspaceSettings(polynomial_degree=5, degree_delta=1)
            for working in (self.states, 47):
                if working != state.working_states:
                    state = replace(state, working_states=working, eigenvalues=state.eigenvalues[:working],
                                    vectors=state.vectors[:, :working])
                result = distributed_state.run_distributed_subspace(op, state, settings=pass_settings,
                                                                    compute_residuals=False, group=group)
                found.append((cp.asnumpy(result.eigenvalues), cp.asnumpy(group.gather(result.vectors))))
                state = result.state
        self.assertGreater(group.passes, 2)
        return found

    def test_a_first_solve_and_later_passes_give_the_bits_of_the_step_as_it_was(self):
        settings = dict(PARSEC_CUPY_DISTRIBUTED_STATE_OVERLAP_COLUMNS='8',
                        PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2))
        for devices in _named_groups():
            reference = self._passes(devices, 'off', '0', **settings)
            for mode, strided in _VALUES[1:]:
                with self.subTest(devices=devices, mode=mode, strided=strided):
                    found = self._passes(devices, mode, strided, **settings)
                    for (values, vectors), (former, basis) in zip(found, reference, strict=True):
                        np.testing.assert_array_equal(values, former)
                        np.testing.assert_array_equal(vectors, basis)

    def test_a_failed_audit_puts_the_filtered_columns_back_bit_for_bit_with_every_value(self):
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import GeneralizedRitzStabilityError
        cp = self.cp
        basis = self._basis(self.rows, self.states, 41)

        def unstable(*_args, **_options):
            raise GeneralizedRitzStabilityError('forced')

        for devices in _named_groups():
            op, group = self._group(devices)
            layout = self._layout(devices)
            for mode, strided in _VALUES:
                with self.subTest(devices=devices, mode=mode, strided=strided), patch.dict(os.environ):
                    for name in _NAMED:
                        os.environ.pop(name, None)
                    os.environ.update(self._environment(mode, strided, PARSEC_CUPY_DISTRIBUTED_STATE_OVERLAP_COLUMNS='8',
                                                        PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2)))
                    columns = self._placed(group, basis, layout)
                    pointers = [int(block.data.ptr) for block in columns.blocks]
                    rotate = group.seconds['rotate']
                    with patch.object(distributed_state, 'solve_whitened_ritz', unstable), \
                         patch.object(distributed_state, 'solve_whitened_ritz_on_device', unstable):
                        with self.assertRaises(GeneralizedRitzStabilityError) as caught:
                            group.ritz(op, columns, stages=False)
                    kept = caught.exception.filtered
                    self.assertEqual([int(block.data.ptr) for block in kept.blocks], pointers)
                    np.testing.assert_array_equal(cp.asnumpy(group.gather(kept)), basis)
                    self.assertEqual(group.seconds['rotate'], rotate)

    def test_nothing_is_allocated_under_an_exchange_stream(self):
        cp = self.cp
        pool = cp.get_default_memory_pool()
        basis = self._basis(self.rows, self.states)
        previous = cp.cuda.get_allocator()
        for devices in _named_groups():
            op, group = self._group(devices)
            for mode, strided in _VALUES[1:3] + _VALUES[4:]:
                taken = []

                def recording(size):
                    # The exchange streams of the group live as long as the group: their handles are theirs.
                    current = int(cp.cuda.get_current_stream().ptr)
                    taken.append(current in {int(stream.ptr) for stream in group._exchange.values()})
                    return pool.malloc(size)

                with self.subTest(devices=devices, mode=mode, strided=strided):
                    cp.cuda.set_allocator(recording)
                    try:
                        self._step(group, basis, self._layout(devices), mode, strided,
                                   PARSEC_CUPY_DISTRIBUTED_STATE_OVERLAP_COLUMNS='8',
                                   PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2))
                    finally:
                        cp.cuda.set_allocator(previous)
                    self.assertEqual(len(group._exchange), len(set(devices)))
                    self.assertGreater(len(taken), 0)
                    self.assertEqual(sum(taken), 0)

    def test_the_comparison_sees_a_missing_wait_between_the_two_streams(self):
        # The exchange stream of a device waits for the leave of the slab before it sends it, and for the apply
        # before it sends H times the slab.  Each wait is dropped in turn while the compute stream is held busy
        # before that work: the bits are then no longer those of the step as it was, which shows that the
        # comparisons above would see a missing wait.
        cp = self.cp
        spin = cp.RawKernel(r'''
        extern "C" __global__ void spin(long long cycles) {
            long long start = clock64();
            while (clock64() - start < cycles) { }
        }''', 'spin')
        original_to_rows = distributed_state.SectorDeviceGroup._to_rows
        clock_type = distributed_state._RoundClock

        class Held:
            """An event whose record first holds the stream busy."""

            def __init__(self, event):
                self.event = event

            def record(self, stream=None):
                with stream if stream is not None else cp.cuda.get_current_stream():
                    spin((1,), (1,), (np.int64(20_000_000),))
                self.event.record(stream)

        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import GeneralizedRitzStabilityError
        basis = self._basis(self.rows, self.states)
        for devices in ((0, 0), (0, 0, 0, 0)):
            outcomes = []
            settings = dict(PARSEC_CUPY_DISTRIBUTED_STATE_OVERLAP_COLUMNS='8',
                            PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2))
            op, group = self._group(devices)
            reference = self._step(group, basis, self._layout(devices), 'off', '0', **settings)
            for dropped in ('leave', 'apply'):
                ends = []

                class Clock(clock_type):
                    def __init__(self, cp_module, rounds):
                        super().__init__(cp_module, rounds)
                        pairs = getattr(self, dropped)
                        setattr(self, dropped, [(Held(start), end) for start, end in pairs])
                        ends.extend(id(end) for _start, end in pairs)

                    def seconds(self, cp_module, overlap):
                        return dict(apply=0., to_rows=0., gram=0., overlap=0.)

                def without_the_wait(self, *args, after=None, **options):
                    if after is not None and id(after[0]) in ends:
                        after = None
                    return original_to_rows(self, *args, after=after, **options)

                for mode in ('p1', 'p2'):
                    # The control: the compute stream held as busy, with the wait, gives the bits of the step.
                    with self.subTest(devices=devices, held=dropped, mode=mode), \
                         patch.object(distributed_state, '_RoundClock', Clock):
                        self._assert_same(self._step(group, basis, self._layout(devices), mode, '0', **settings),
                                          reference, f'{dropped} held, {mode}')
                    with self.subTest(devices=devices, dropped=dropped, mode=mode), \
                         patch.object(distributed_state, '_RoundClock', Clock), \
                         patch.object(distributed_state.SectorDeviceGroup, '_to_rows', without_the_wait):
                        try:
                            mutated = self._step(group, basis, self._layout(devices), mode, '0', **settings)
                        except GeneralizedRitzStabilityError:
                            # Values that were not there yet can make the overlap fail its audit: seen as well.
                            outcomes.append('audit')
                            continue
                        same = (np.array_equal(mutated[0], reference[0]) and np.array_equal(mutated[1], reference[1])
                                and all(np.array_equal(a, b) for a, b in zip(mutated[2], reference[2])))
                        self.assertFalse(same, 'a missing wait went unseen')
                        outcomes.append('differ')
            # Each wait was dropped under both values, and every time the step saw it.
            self.assertEqual(len(outcomes), 4)

    def test_strided_conversions_move_every_entry_exactly(self):
        cp = self.cp
        host = np.random.default_rng(37).normal(size=(self.rows, 23))
        for devices in _named_groups():
            op, group = self._group(devices)
            edges = group.row_edges(self.rows)
            layouts = [distributed_state.ColumnLayout(group.column_ranges(uniform_filter_blocks(23, 6, 5))),
                       distributed_state.ColumnLayout(((0, 23),) + ((23, 23),) * (len(devices) - 1))]
            for layout in layouts:
                for switches in ({}, dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='0'), dict(PARSEC_CUPY_EXCHANGE_PAIRS='0'),
                                 dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES='8')):
                    with self.subTest(devices=devices, widths=layout.widths, switches=switches), patch.dict(os.environ):
                        for name in _NAMED:
                            os.environ.pop(name, None)
                        os.environ.update(switches, PARSEC_CUPY_EXCHANGE_STRIDED='1')
                        kept = group.scatter(cp.array(host, order='F'), layout).take()
                        rows = group._to_rows(kept, layout, edges, keep=True)
                        self.assertEqual(group.exchange_copies, 'strided')
                        for index, block in enumerate(rows):
                            np.testing.assert_array_equal(cp.asnumpy(block), host[edges[index]:edges[index + 1], :])
                        for device, block in zip(devices, kept):
                            with cp.cuda.Device(device):
                                block.fill(np.nan)
                                cp.cuda.get_current_stream().synchronize()
                        back = group._to_columns(rows, layout, edges, targets=kept)
                        for index, block in enumerate(back):
                            held = [column for start, stop in layout.parts[index] for column in range(start, stop)]
                            np.testing.assert_array_equal(cp.asnumpy(block), host[:, held])
                        # No buffer was taken on either way.
                        self.assertEqual(group.exchange_buffer_bytes, 0)


if __name__ == '__main__':
    unittest.main()
