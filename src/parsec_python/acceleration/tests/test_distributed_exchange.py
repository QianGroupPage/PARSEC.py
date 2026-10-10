"""The exchange and the Ritz step of a shared sector basis, checked on the host.

Chunk plans, slab plans, capacities and the memory tests are plain arithmetic.
The data movement and the Ritz step run on a NumPy stand-in for the few CuPy
calls they make.  Its streams hold their work back until they are
synchronized, so a missing synchronization shows as wrong data, and its
copies refuse to leave their allocation.  Real devices are covered by
test_distributed_state.
"""
import contextlib
import ctypes
import gc
import importlib
import itertools
import math
import os
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from parsec_python.acceleration.Eigensolvers import distributed_state, filter_graph
from parsec_python.acceleration.Eigensolvers.chebyshev import subspace_filter_blocks, uniform_filter_blocks
from parsec_python.acceleration.Eigensolvers.distributed_filter import block_partitions, starting_sigmas
from parsec_python.acceleration.Eigensolvers.lapack_random import LapackRandom

# The package also exports a function named like this submodule.
rayleigh_ritz = importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')


class _Stream:
    def __init__(self, runtime, device):
        self.runtime, self.device_id, self.pending = runtime, device, []
        runtime.streams.append(self)

    def __enter__(self):
        self.runtime.local.streams.append(self)
        return self

    def __exit__(self, *_exc):
        self.runtime.local.streams.pop()
        return False

    def queue(self, work):
        if self.runtime.local.device != self.device_id:
            raise AssertionError('a stream was used while another device was current')
        if self.runtime.eager:
            work()
        else:
            self.pending.append(work)

    def synchronize(self):
        pending, self.pending = self.pending, []
        for work in pending:
            work()


class _Device:
    def __init__(self, runtime, device):
        self.runtime, self.device = runtime, device

    def __enter__(self):
        local = self.runtime.local
        self.previous = (getattr(local, 'device', None), getattr(local, 'streams', None))
        # Work that names no stream goes to the default stream of the device, one per thread here.
        if not hasattr(local, 'defaults'):
            local.defaults = {}
        if self.device not in local.defaults:
            local.defaults[self.device] = _Stream(self.runtime, self.device)
        local.device, local.streams = self.device, [local.defaults[self.device]]
        return self

    def __exit__(self, *_exc):
        self.runtime.local.device, self.runtime.local.streams = self.previous
        return False


class _Memory:
    """Stands for the allocation that an array lies in: where it starts and its bytes."""

    def __init__(self, array):
        self.array, self.runtime = array, array.runtime
        self.ptr, stop = array.allocation
        self.size = stop - self.ptr


class _Pointer:
    def __init__(self, array, address):
        self.mem, self.ptr, self.device_id = _Memory(array), address, array.device_id

    def __add__(self, offset):
        return _Pointer(self.mem.array, self.ptr + int(offset))

    def inside(self, size):
        return self.mem.ptr <= self.ptr and self.ptr + size <= self.mem.ptr + self.mem.size

    def copy_from_device_async(self, source, size, stream=None):
        runtime = self.mem.runtime
        stream = runtime.local.streams[-1] if stream is None else stream
        if stream.device_id != self.device_id:
            raise AssertionError('an arriving copy must use a stream of the device it writes to')
        if not (self.inside(size) and source.inside(size)):
            raise AssertionError('a copy leaves its allocation')
        runtime.copies.append((self.device_id, source.device_id, stream))
        stream.queue(lambda: ctypes.memmove(self.ptr, source.ptr, size))


class _Array(np.ndarray):
    runtime = device_id = allocation = owner = None

    def __array_finalize__(self, parent):
        for name in ('runtime', 'device_id', 'allocation', 'owner'):
            setattr(self, name, getattr(parent, name, None))

    @property
    def data(self):
        return _Pointer(self, self.__array_interface__['data'][0])

    @property
    def device(self):
        return SimpleNamespace(id=self.device_id)

    def __setitem__(self, key, value):
        local = self.runtime.local
        if local.device != self.device_id or getattr(value, 'device_id', self.device_id) != self.device_id:
            raise AssertionError('a kernel needs all of its arrays on the current device')
        if isinstance(value, np.ndarray) and np.may_share_memory(self, value):
            raise AssertionError('a kernel writes into the memory that it reads')
        self.runtime.kernels.append((self.strides, getattr(value, 'strides', None)))
        local.streams[-1].queue(lambda: np.ndarray.__setitem__(self, key, value))

    def __matmul__(self, other):
        return self.runtime.matmul(self, other)

    def __iadd__(self, other):
        local = self.runtime.local
        if local.device != self.device_id or getattr(other, 'device_id', self.device_id) != self.device_id:
            raise AssertionError('a kernel needs all of its arrays on the current device')
        local.streams[-1].queue(lambda: np.add(np.asarray(self), np.asarray(other), out=np.asarray(self)))
        return self

    def set(self, values, stream=None):
        if not (self.flags.c_contiguous or self.flags.f_contiguous) or values.shape != self.shape:
            raise AssertionError('an upload fills a contiguous array of its own shape')
        values = np.array(values)
        if stream is None:
            # Complete on return, like an upload that names no stream.
            np.asarray(self)[...] = values
        else:
            stream.queue(lambda: np.copyto(np.asarray(self), values))


def _stand_in(eager=False, total_bytes=0):
    """A CuPy look-alike on host memory and the record of what was asked of it."""

    # allocated: (device, elements) of every new array; copies: (device, source device, stream) of every peer copy;
    # kernels: (target strides, source strides) of every assignment between arrays of one device;
    # live, peak: elements per device in arrays that something still refers to, now and at most.
    # returned: (device, stream, sizes that the device had allocated by then) of every release of the free blocks
    # of one stream.
    runtime = SimpleNamespace(local=threading.local(), eager=eager, streams=[], allocated=[], copies=[], kernels=[],
                              live={}, peak={}, lock=threading.Lock(), returned=[])

    def release(device, size):
        with runtime.lock:
            runtime.live[device] -= size

    def empty(shape, dtype=np.float64, order='C'):
        # NaN stands for whatever a device block holds before it is written, and the least integer in an
        # array of indices.
        unwritten = np.nan if np.dtype(dtype).kind == 'f' else np.iinfo(dtype).min
        array = np.full(shape, unwritten, dtype=dtype, order=order).view(_Array)
        start = array.__array_interface__['data'][0]
        array.runtime, array.device_id, array.allocation = runtime, runtime.local.device, (start, start + array.nbytes)
        runtime.allocated.append((array.device_id, array.size))
        with runtime.lock:
            live = runtime.live[array.device_id] = runtime.live.get(array.device_id, 0) + array.size
            runtime.peak[array.device_id] = max(runtime.peak.get(array.device_id, 0), live)
        # Views keep the array they were taken from, so this runs when the last of them is gone.
        weakref.finalize(array, release, array.device_id, array.size).atexit = False
        return array

    def ndarray(shape, dtype=np.float64, memptr=None, order='C'):
        count = int(np.prod(shape))
        if not memptr.inside(8 * count):
            raise AssertionError('a view leaves its allocation')
        array = np.ctypeslib.as_array((ctypes.c_double * count).from_address(memptr.ptr))
        array = array.reshape(shape, order=order).view(_Array)
        array.runtime, array.device_id = runtime, memptr.device_id
        array.allocation, array.owner = memptr.mem.array.allocation, memptr.mem.array
        return array

    # What follows serves the Ritz step.  A kernel reads and writes when its stream gets to it, like the
    # assignments above, so that work of one stream stays in order and a result that another stream is
    # still to produce is not there yet.

    def here(*arrays):
        if any(getattr(array, 'device_id', runtime.local.device) != runtime.local.device for array in arrays):
            raise AssertionError('a kernel needs all of its arrays on the current device')
        return runtime.local.streams[-1]

    def launch(shape, compute, *operands, order='C'):
        """Queue ``compute`` on the current stream; its value fills a new array of the current device."""
        stream = here(*operands)
        result = empty(shape, order=order)
        stream.queue(lambda: np.copyto(np.asarray(result), compute()))
        return result

    def matmul(left, right, out=None):
        def product():
            return np.asarray(left) @ np.asarray(right)
        if out is None:
            return launch((left.shape[0], right.shape[1]), product, left, right)
        here(left, right, out).queue(lambda: np.copyto(np.asarray(out), product()))
        return out

    def zeros(shape, dtype=np.float64, order='C'):
        array = empty(shape, dtype=dtype, order=order)
        np.asarray(array)[...] = 0.
        return array

    def asarray(array, dtype=None):
        if isinstance(array, _Array):
            here(array)
            return array
        # An upload has completed when it returns.  It keeps the type of what it is given: the device
        # branches of the step with one tall block upload column indices.
        array = np.asarray(array, dtype=dtype)
        uploaded = empty(array.shape, dtype=array.dtype, order='F' if np.isfortran(array) else 'C')
        np.asarray(uploaded)[...] = array
        return uploaded

    def asnumpy(array):
        # A download waits for the work queued before it on the current stream, and for no other.
        here(array).synchronize()
        return np.array(np.asarray(array))

    def laid_out(order, contiguous):
        def convert(array, dtype=np.float64):
            if getattr(array.flags, contiguous):
                return array
            return launch(array.shape, lambda: np.asarray(array), array, order=order)
        return convert

    def stack(arrays):
        return launch((len(arrays),) + arrays[0].shape, lambda: np.stack([np.asarray(array) for array in arrays]), *arrays)

    # What follows serves the small solve on the device.  The sum and the indexed read that the step with
    # one tall block makes of a triangle are plain arithmetic here, done at once: the triangle is therefore
    # complete on return, after the work that the current stream had queued before it.

    def tril(array, k=0):
        here(array).synchronize()
        triangle = empty(array.shape, dtype=array.dtype)
        np.asarray(triangle)[...] = np.tril(np.asarray(array), k)
        return triangle

    def empty_like(array):
        return empty(array.shape, dtype=array.dtype, order='F' if np.isfortran(array) else 'C')

    def free_all_blocks(stream=None):
        device = runtime.local.device
        runtime.returned.append((device, stream, [size for owner, size in runtime.allocated if owner == device]))

    runtime.matmul = matmul
    cuda = SimpleNamespace(
        Device=lambda device: _Device(runtime, device),
        Stream=lambda non_blocking=False: _Stream(runtime, runtime.local.device),
        get_current_stream=lambda: runtime.local.streams[-1],
        runtime=SimpleNamespace(memGetInfo=lambda: (total_bytes, total_bytes)),
    )
    return SimpleNamespace(
        empty=empty, ndarray=ndarray, float64=np.float64, dtype=np.dtype, cuda=cuda, zeros=zeros, asarray=asarray,
        asnumpy=asnumpy, matmul=matmul, stack=stack, tril=tril, empty_like=empty_like,
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=free_all_blocks),
        asfortranarray=laid_out('F', 'f_contiguous'), ascontiguousarray=laid_out('C', 'c_contiguous')), runtime


class _Owner:
    """Stands for the sector operator: carries a worker with one stream per device."""

    def __init__(self, cp, devices):
        streams = {}
        for device in devices:
            with cp.cuda.Device(device):
                streams[device] = cp.cuda.Stream(non_blocking=True)
        self._distributed_filter = SimpleNamespace(devices=devices, owner=devices[0], streams=streams)


class _Applies:
    """Stands for the operator of one device: ``H`` times a block is a kernel on the current stream of that device."""

    def __init__(self, runtime, matrix, device, applied):
        self.runtime, self.matrix, self.device, self.applied = runtime, matrix, device, applied

    def apply_into(self, block, target):
        local = self.runtime.local
        if not local.device == self.device == block.device_id == target.device_id:
            raise AssertionError('an operator applies to arrays of its own device while it is current')
        if block.shape != target.shape or not target.flags.f_contiguous:
            raise AssertionError('H times a block goes to a column-major array of the same shape')
        self.applied.append((self.device, block.shape[1]))
        local.streams[-1].queue(lambda: np.copyto(np.asarray(target), self.matrix @ np.asarray(block)))
        return target


class _Sector(_Owner):
    """An owner that applies the host matrix ``H``, with a replica for every other device."""

    def __init__(self, cp, runtime, devices, matrix):
        super().__init__(cp, devices)
        # (device, columns) of every application of H.
        self.applied = []
        operators = [_Applies(runtime, matrix, device, self.applied) for device in devices]
        self.apply_into = operators[0].apply_into
        self._distributed_filter.replicas = dict(zip(devices[1:], operators[1:]))


def _table(blocks, lower, upper, reference, reset, initial_sigma=None):
    """The coefficient table that the filter graphs upload for ``blocks``: one row per recurrence step."""
    half_span, center = .5 * (upper - lower), .5 * (upper + lower)
    table = np.empty((sum(block.degree for block in blocks), 4))
    filter_graph.fill_recurrence_table(table, blocks, center, half_span, half_span / (reference - center), reset,
                                       initial_sigma)
    return table


def _recurrence(apply, columns, table):
    """The filtered ``columns``: the three-term recurrence with a row of ``table`` per step."""
    previous, current = 0., columns
    for center, scale, sigma, following in table:
        previous, current = current, scale * following * (apply(current) - center * current) - sigma * following * previous
    return current


def _plan_filter(matrix, potential, columns, blocks, lower, upper, reference, reset):
    """What one device makes of all ``columns`` with the plan ``blocks``, sigma carried from block to block."""
    table, filtered, offset = _table(blocks, lower, upper, reference, reset), np.empty_like(columns), 0
    for block in blocks:
        filtered[:, block.start:block.stop] = _recurrence(
            lambda vectors: matrix @ vectors + potential[:, None] * vectors, columns[:, block.start:block.stop],
            table[offset:offset + block.degree])
        offset += block.degree
    return filtered


class _Graphs:
    """Stands for the filter graphs of one device: the recurrence of every block it is given, in place and in order."""

    def __init__(self, runtime, device, matrix, potential):
        self.runtime, self.device, self.matrix, self.potential = runtime, device, matrix, potential
        # The plan of every call: with one graph per block, a call with another key captures them again.
        self.keys = []

    def apply(self, matrix, blocks, lower, upper, reference, reset, initial_sigma=None, out=None):
        local = self.runtime.local
        if not local.device == self.device == matrix.device_id or out is not matrix:
            raise AssertionError('a device filters its own block in place while it is current')
        covered = [column for block in blocks for column in range(block.start, block.stop)]
        if covered != list(range(matrix.shape[1])) or max(block.stop - block.start for block in blocks) > 6:
            raise AssertionError('the blocks of a device lie side by side in its block')
        self.keys.append((matrix.shape[0], tuple((block.start, block.stop, block.degree) for block in blocks)))
        table, offset = _table(blocks, lower, upper, reference, reset, initial_sigma), 0

        def operator(vectors):
            # The potential is read when the stream gets here: an upload queued before it has arrived.
            return self.matrix @ vectors + np.asarray(self.potential)[:, None] * vectors

        def launch(block, steps):
            columns = np.asarray(matrix)[:, block.start:block.stop]
            columns[...] = _recurrence(operator, columns.copy(), steps)

        for block in blocks:
            local.streams[-1].queue(lambda block=block, steps=table[offset:offset + block.degree]: launch(block, steps))
            offset += block.degree
        return out


class _FilterSector(_Sector):
    """A sector whose devices also filter: ``H`` is the host matrix plus a local potential on every device.

    The potential of the other devices is unset until a filter uploads it.
    """

    def __init__(self, cp, runtime, devices, matrix, potential):
        super().__init__(cp, runtime, devices, (matrix + sp.diags(potential)).tocsr())
        worker = self._distributed_filter
        worker.graphs = {}
        for device in devices:
            with cp.cuda.Device(device):
                field = cp.empty(potential.shape)
            if device == devices[0]:
                np.asarray(field)[...] = potential
                self.effective_potential = field
            else:
                worker.replicas[device].effective_potential = field
            worker.graphs[device] = _Graphs(runtime, device, matrix, field)


def _columns(widths):
    offsets = np.concatenate(([0], np.cumsum(widths)))
    return tuple((int(start), int(stop)) for start, stop in zip(offsets[:-1], offsets[1:]))


def _layout(columns):
    """The layout of column blocks of these widths, one after the other, or the layout that is given."""
    if isinstance(columns, distributed_state.ColumnLayout):
        return columns
    return distributed_state.ColumnLayout(_columns(columns))


def _held(layout, index):
    """The columns of the basis in the block of device ``index``, in the order in which the block holds them."""
    return [column for start, stop in layout.parts[index] for column in range(start, stop)]


def _cut(length, width, equal=True):
    """Widths of the slabs of at most ``width`` columns that a range of ``length`` columns is cut into.

    As few as that takes and as equal as they can be or, unless ``equal``, full ones and what is left.
    """
    if not equal:
        return [width] * (length // width) + [length % width] * bool(length % width)
    count = -(-length // width)
    return [length * (index + 1) // count - length * index // count for index in range(count)]


def _interleaved(columns, devices, block=6):
    """The layout of two column ranges per device for a trial basis of ``columns`` columns."""
    trial = uniform_filter_blocks(columns, block, 5)
    return distributed_state.ColumnLayout(distributed_state.interleaved_ranges(trial, devices))


# Two column ranges per device: (rows, layout).
_TWO_RANGES = (
    (257, _interleaved(53, 4)),              # blocks of 18, 12, 12 and 11 columns; the top block has 5
    (257, _interleaved(23, 2)),
    (257, _interleaved(53, 4).leading(47)),  # trimmed inside the top block
    (257, _interleaved(53, 4).leading(31)),  # trimmed to a part of one upper range
    (257, _interleaved(53, 4).leading(20)),  # trimmed into the lower half: the last device keeps two columns
    (61, distributed_state.ColumnLayout((((0, 5), (14, 20)), ((5, 5), (20, 20)), ((5, 14), (20, 29))))),
    # Fewer rows than devices: the first device holds none, and two columns that are no neighbours.
    (3, distributed_state.ColumnLayout((((0, 1), (2, 3)), ((1, 1), (3, 3)), ((1, 2), (3, 3)), ((2, 2), (3, 3))))),
)


class ExchangePlanTests(unittest.TestCase):
    def test_chunk_size_comes_from_the_environment(self):
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_EXCHANGE_CHUNK_BYTES', None)
            self.assertEqual(distributed_state.exchange_chunk_bytes(), 1 << 30)
            os.environ['PARSEC_CUPY_EXCHANGE_CHUNK_BYTES'] = ' 4096 '
            self.assertEqual(distributed_state.exchange_chunk_bytes(), 4096)
            for bad in ('0', '-8', '1e9', 'auto', ''):
                os.environ['PARSEC_CUPY_EXCHANGE_CHUNK_BYTES'] = bad
                with self.assertRaises(ValueError):
                    distributed_state.exchange_chunk_bytes()

    def test_chunks_cover_every_column_once_within_the_limit(self):
        for width in (0, 1, 5, 6, 23, 607):
            for away in (0, 1, 64, 193):
                for limit in (0, 1, 192, 193, 500, 10**9):
                    chunks = distributed_state.exchange_chunks(width, away, limit)
                    if not width or not away:
                        self.assertEqual(chunks, ())
                        continue
                    covered = [column for first, last in chunks for column in range(first, last)]
                    self.assertEqual(covered, list(range(width)))
                    for first, last in chunks:
                        self.assertTrue(last - first == 1 or (last - first) * away <= limit)
                    # No chunk but the last could have taken one more column.
                    for first, last in chunks[:-1]:
                        self.assertGreater((last - first + 1) * away, limit)

    def test_concurrent_receives_are_on_unless_switched_off(self):
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_EXCHANGE_CONCURRENT', None)
            self.assertTrue(distributed_state.concurrent_exchange_requested())
            for raw, expected in (('1', True), ('on', True), (' True ', True), ('0', False), ('off', False), ('false', False)):
                os.environ['PARSEC_CUPY_EXCHANGE_CONCURRENT'] = raw
                self.assertEqual(distributed_state.concurrent_exchange_requested(), expected)
            for bad in ('2', 'auto', 'yes', ''):
                os.environ['PARSEC_CUPY_EXCHANGE_CONCURRENT'] = bad
                with self.assertRaises(ValueError):
                    distributed_state.concurrent_exchange_requested()
            # A mistyped value already stops the run where the trial basis is sized, before the first filter.
            cp, _runtime = _stand_in()
            group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
            with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_EXCHANGE_CONCURRENT'):
                group._capacities(_columns((3, 2)), [0, 4, 9])

    def test_four_devices_pair_up_unless_switched_off(self):
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_EXCHANGE_PAIRS', None)
            self.assertTrue(distributed_state.paired_exchange_requested())
            for raw, expected in (('1', True), ('On', True), (' true', True), ('0', False), ('off', False), ('False ', False)):
                os.environ['PARSEC_CUPY_EXCHANGE_PAIRS'] = raw
                self.assertEqual(distributed_state.paired_exchange_requested(), expected)
            for bad in ('2', 'auto', 'pairs', ''):
                os.environ['PARSEC_CUPY_EXCHANGE_PAIRS'] = bad
                with self.assertRaises(ValueError):
                    distributed_state.paired_exchange_requested()
            # A mistyped value already stops the run where the trial basis is sized, before the first filter,
            # also of a sector that has no four devices and would never ask.
            cp, _runtime = _stand_in()
            group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
            with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_EXCHANGE_PAIRS'):
                group._capacities(_columns((3, 2)), [0, 4, 9])
        # Three turns.  In each every device has a partner that has it as its partner, and over the three it
        # meets each of the other devices once: the twelve copies of a chunk, four per turn.
        turns = distributed_state._PAIR_TURNS
        self.assertEqual(len(turns), 3)
        for partners in turns:
            self.assertEqual([partners[partners[device]] for device in range(4)], [0, 1, 2, 3])
        for device in range(4):
            self.assertEqual(sorted(partners[device] for partners in turns), [other for other in range(4) if other != device])

    def test_capacities_bound_the_buffer_by_the_chunk(self):
        cp, _runtime = _stand_in()
        devices = (0, 1, 2, 3)
        ranges, edges = _columns((6, 6, 6, 5)), [0, 64, 128, 192, 257]
        # Tall: 257 rows of the 6 widest columns.  Outside a device's own rows: at most 257 - 64 = 193 rows.
        expected = {'8': 193, str(8 * 400): 400, str(1 << 30): 193 * 6}
        # One tall block, the default, adds the room in which a row block may end behind its columns: a row
        # of all four devices per row of the tallest row block, which has 65.
        room = 4 * 65
        with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)), patch.dict(os.environ):
            for blocks, tall in ((None, 257 * 6 + room), ('1', 257 * 6 + room), ('2', 257 * 6), ('3', 257 * 6)):
                os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS', None)
                if blocks is not None:
                    os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = blocks
                for raw, chunk in expected.items():
                    group = distributed_state.SectorDeviceGroup(_Owner(cp, devices), devices)
                    with patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=raw,
                                    PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed'):
                        self.assertEqual(group._capacities(ranges, edges), (tall, chunk), (blocks, raw))
                # Capacities never shrink: a later, smaller limit keeps the buffer that exists.
                with patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES='8', PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed'):
                    self.assertEqual(group._capacities(ranges, edges), (tall, 193 * 6), blocks)

    def test_column_copies_are_on_unless_switched_off(self):
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_EXCHANGE_COLUMN_COPIES', None)
            self.assertTrue(distributed_state.column_copies_requested())
            for raw, expected in (('1', True), ('ON', True), ('true', True), ('0', False), ('off', False), (' False', False)):
                os.environ['PARSEC_CUPY_EXCHANGE_COLUMN_COPIES'] = raw
                self.assertEqual(distributed_state.column_copies_requested(), expected)
            for bad in ('2', 'auto', 'columns', ''):
                os.environ['PARSEC_CUPY_EXCHANGE_COLUMN_COPIES'] = bad
                with self.assertRaises(ValueError):
                    distributed_state.column_copies_requested()
            # A mistyped value already stops the run where the trial basis is sized, before the first filter.
            cp, _runtime = _stand_in()
            group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
            with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_EXCHANGE_COLUMN_COPIES'):
                group._capacities(_columns((3, 2)), [0, 4, 9])

    def test_the_condition_number_has_a_helper_unless_none_is_free_or_it_is_switched_off(self):
        cp, _runtime = _stand_in()

        def helper(devices, hartree_device=None):
            owner = _Owner(cp, devices)
            if hartree_device is not None:
                owner.hartree_device = hartree_device
            return distributed_state.SectorDeviceGroup(owner, devices)._condition_helper()

        with patch.dict(os.environ):
            for name in ('PARSEC_CUPY_RITZ_CONDITION_HELPER', 'PARSEC_CUPY_RITZ_CONDITION'):
                os.environ.pop(name, None)
            self.assertEqual(distributed_state.condition_helper_policy(), 'auto')
            # The first device after the owner, which never helps itself ...
            self.assertEqual([helper(devices) for devices in ((0, 1, 2, 3), (2, 3), (3, 1, 0), (1,))], [1, 3, 1, None])
            # ... unless it holds the Hartree objects of the process: the root's sector on 16 GPUs and its
            # two on 8, of which the second has no other device.  Elsewhere the owner holds them.
            self.assertEqual([helper((0, 1, 2, 3), device) for device in (3, 1, 0)], [1, 2, 1])
            self.assertEqual((helper((0, 1), 3), helper((2, 3), 3)), (1, None))
            # ``any`` asks that device where the sector has no other, and only there.
            for raw, expected in ((' Any ', (2, 3)), ('auto', (2, None)), ('OFF', (None, None))):
                os.environ['PARSEC_CUPY_RITZ_CONDITION_HELPER'] = raw
                self.assertEqual(distributed_state.condition_helper_policy(), raw.strip().lower())
                self.assertEqual((helper((0, 1, 2, 3), 1), helper((2, 3), 3)), expected)
            os.environ['PARSEC_CUPY_RITZ_CONDITION_HELPER'] = 'any'
            self.assertEqual([helper(devices, 1) for devices in ((0, 1), (0, 1, 2), (1,))], [1, 2, None])
            # The Ritz step on real devices (test_ritz_dense) expects these devices, of a group of two, three
            # and four: a node gives it as many as it has, up to four.
            from parsec_python.acceleration.tests.test_ritz_dense import _helper_cases
            for devices in ((0, 1), (0, 1, 2), (0, 1, 2, 3)):
                for policy, hartree_device, expected in _helper_cases(devices):
                    os.environ['PARSEC_CUPY_RITZ_CONDITION_HELPER'] = policy
                    found = helper(devices, hartree_device)
                    self.assertEqual(devices[0] if found is None else found, expected, (devices, policy))
            os.environ['PARSEC_CUPY_RITZ_CONDITION_HELPER'] = 'any'
            # The host SVD of the overlap needs no device; the symmetric spectrum is the default of a device solve.
            for policy, expected in (('svd', None), ('symmetric', 1)):
                with patch.dict(os.environ, PARSEC_CUPY_RITZ_CONDITION=policy):
                    self.assertEqual(helper((0, 1, 2, 3)), expected)
            for bad in ('1', 'on', 'owner', ''):
                os.environ['PARSEC_CUPY_RITZ_CONDITION_HELPER'] = bad
                with self.assertRaises(ValueError):
                    distributed_state.condition_helper_policy()
            # A mistyped value already stops the run where the trial basis is sized, before the first filter.
            group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
            with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_RITZ_CONDITION_HELPER'):
                group._capacities(_columns((3, 2)), [0, 4, 9])
            self.assertEqual((group.condition_device, group.condition_seconds), (None, 0.))

    @patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='3')
    def test_fit_counts_three_tall_blocks_and_one_chunk_where_three_are_named(self):
        gib = 1 << 30
        cp, _runtime = _stand_in(total_bytes=160 * gib)
        rows, columns = 10 * (1 << 20), 1024  # 80 GiB of basis
        cases = (
            # devices, ranges, chunk bytes, needed GiB
            (4, 'fixed', gib, 3 * 20 + 1),
            (4, 'fixed', 8 * gib, 3 * 20 + 8),
            (4, 'fixed', 64 * gib, 3 * 20 + 15),  # never more than a block outside the device's rows
            (2, 'fixed', gib, 3 * 40 + 1),
            (4, 'balanced', gib, 4 * 25 + 1),
        )
        with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)):
            for devices, policy, chunk, needed in cases:
                for fraction, fits in (((needed + .5) / 160, True), ((needed - .5) / 160, False)):
                    with patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(chunk),
                                    PARSEC_CUPY_DISTRIBUTED_STATE_RANGES=policy,
                                    PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION=repr(fraction)):
                        self.assertEqual(distributed_state.shared_basis_fits(rows, columns, devices), fits,
                                         (devices, policy, chunk, fraction))
        # By default the blocks may take 0.85 of the device: 61 GiB fit 72 GiB, not 71 GiB.
        for total, fits in ((72, True), (71, False)):
            cp, _runtime = _stand_in(total_bytes=total * gib)
            with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)), \
                 patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(gib), PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed'):
                os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION', None)
                self.assertEqual(distributed_state.shared_basis_fits(rows, columns, 4), fits, total)

    def test_tall_blocks_are_one_unless_two_or_three_are_named(self):
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS', None)
            os.environ.pop('PARSEC_CUPY_STREAMING_RITZ_BYTES', None)
            self.assertEqual(distributed_state.tall_blocks_per_device(), 1)
            for raw, expected in (('1', 1), (' 1', 1), ('2', 2), ('3', 3), (' 3 ', 3)):
                os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = raw
                self.assertEqual(distributed_state.tall_blocks_per_device(), expected)
            for bad in ('0', '4', 'two', 'one', 'auto', ''):
                os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = bad
                with self.assertRaises(ValueError):
                    distributed_state.tall_blocks_per_device()
            # A mistyped value already stops the run where the trial basis is sized, before the first filter.
            cp, _runtime = _stand_in()
            group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
            with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'):
                group._capacities(_columns((3, 2)), [0, 4, 9])
            # So does a mistyped slab budget of the steps with slabs: one tall block, named or by default, and two.
            os.environ['PARSEC_CUPY_STREAMING_RITZ_BYTES'] = '4GiB'
            for raw in ('1', None, '2'):
                os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS', None)
                if raw is not None:
                    os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = raw
                with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_STREAMING_RITZ_BYTES'):
                    group._capacities(_columns((3, 2)), [0, 4, 9])
            # The step with three has no slabs and does not read the budget.
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = '3'
            group._capacities(_columns((3, 2)), [0, 4, 9])

    def test_slab_width_follows_the_streaming_budget_and_not_the_column_split(self):
        gib = 1 << 30
        rows = 10 * (1 << 20)  # a column is 80 MiB
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_STREAMING_RITZ_BYTES', None)
            # Columns, devices -> columns of a slab.  An eighth of the basis but at most 4 GiB: 51 columns of
            # the 80 GiB basis and 32 of the 20 GiB one, on two devices as on four.  At least 1 GiB, 12 columns,
            # unless that is more than half of an even share of the columns: 8 of 64 columns on four devices.
            for columns, devices, width in ((1024, 4, 51), (1024, 2, 51), (256, 4, 32), (256, 2, 32), (64, 2, 12),
                                            (64, 4, 8), (7, 4, 1)):
                self.assertEqual(distributed_state.slab_columns(rows, columns, devices), width, (columns, devices))
                # The slab of a device in all rows, and its rows of the slabs of all devices.
                self.assertEqual(distributed_state.slab_workspace(rows, columns, devices),
                                 rows * width + rows // devices * min(columns, devices * width), (columns, devices))
            os.environ['PARSEC_CUPY_STREAMING_RITZ_BYTES'] = str(2 * gib)
            self.assertEqual(distributed_state.slab_columns(rows, 1024, 4), 25)
            os.environ['PARSEC_CUPY_STREAMING_RITZ_BYTES'] = '8'
            self.assertEqual(distributed_state.slab_columns(rows, 1024, 4), 1)
            # The last row block is the tallest: 65 of 257 rows on four devices.
            os.environ['PARSEC_CUPY_STREAMING_RITZ_BYTES'] = str(8 * 257 * 2)
            self.assertEqual(distributed_state.slab_workspace(257, 96, 4), 257 * 2 + 65 * 8)
            # A group keeps the largest workspace it has needed, like its other capacities.
            cp, _runtime = _stand_in()
            group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1, 2, 3)), (0, 1, 2, 3))
            self.assertEqual(group._slab_capacity(257, 96), (2, 257 * 2 + 65 * 8))
            self.assertEqual(group._slab_capacity(257, 6), (1, 257 * 2 + 65 * 8))
            self.assertEqual(group._work, 257 * 2 + 65 * 8)
            # The step with one tall block keeps a slab in all rows in the second part as well, which is more
            # than the rows of one device of all slabs only where there are fewer columns than devices.
            os.environ['PARSEC_CUPY_STREAMING_RITZ_BYTES'] = str(8 * 257)
            for columns, devices in ((96, 4), (23, 4), (23, 2), (4, 4)):
                self.assertEqual(distributed_state.slab_workspace(257, columns, devices, 1),
                                 distributed_state.slab_workspace(257, columns, devices), (columns, devices))
            self.assertEqual((distributed_state.slab_workspace(257, 3, 4), distributed_state.slab_workspace(257, 3, 4, 1)),
                             (257 + 65 * 3, 257 * 2))

    def test_slabs_of_the_step_with_one_tall_block_are_an_even_share_of_a_round(self):
        gib = 1 << 30
        columns_of, workspace = distributed_state.slab_columns, distributed_state.slab_workspace

        def former(rows, columns, devices):
            # The slab that this step cut before: an eighth of the basis, at least 1 and at most 4 GiB.
            budget = min(max(rows * columns, gib), 4 * gib)
            return max(1, min(budget // (8 * rows), columns // (2 * devices)))

        # Sectors of 14,680, 19,392, 23,768, 29,576 and 39,368 electrons: rows and columns.
        sectors = ((2604846, 1842), (2873400, 2438), (3786832, 2979), (4130510, 3704), (6157900, 4928))
        with patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES'):
                os.environ.pop(name, None)
            # A round of 8 GiB on four devices: slabs of 2 GiB.  Two devices have slabs of 2 GiB as well
            # (see the test of two devices below); the other steps keep 4 GiB.
            for sector, two, four in zip(sectors, (206, 186, 141, 129, 87), (103, 93, 70, 64, 43)):
                rows = sector[0]
                self.assertEqual((columns_of(*sector, 2, blocks=1), columns_of(*sector, 4, blocks=1)), (four, four))
                self.assertEqual((columns_of(*sector, 2), columns_of(*sector, 4)), (two, two))
                # The workspace is two slabs: about 4 GiB instead of 8 on four devices.
                self.assertEqual(workspace(*sector, 4, 1), rows * four + -(-rows // 4) * 4 * four)
                self.assertLess(8 * workspace(*sector, 4, 1), 4 * gib)
                self.assertGreater(8 * workspace(*sector, 4, 2), 7.9 * gib)
            # More devices share the round further.  Three keep the 4 GiB: none has run.
            self.assertEqual((columns_of(3786832, 2979, 8, blocks=1), columns_of(3786832, 2979, 3, blocks=1)), (35, 141))

            def cut(sector, widths, width):
                """Rounds, columns of the widest slab and of the widest round, and megabytes of the workspace."""
                sizes = [[last - first for first, last in taken] for taken in distributed_state.block_rounds(widths, width)]
                cp, _runtime = _stand_in()
                group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1, 2, 3)), (0, 1, 2, 3))
                widest, joined = max(map(max, sizes)), max(map(sum, sizes))
                return len(sizes), widest, joined, 8 * group._cut_capacity(sector[0], widest, joined, 1) / 2**20

            # The cuts of the default on 16 GPUs, for the column blocks of the measured runs, against those of
            # slabs of 4 GiB: rounds, widest slab, widest round, megabytes that the workspace returns.  The
            # two that ran with a budget of 2 GiB, 23,768 and 29,576 electrons, gave back 3230 and 3474 MiB of
            # their peaks; 14,680, 19,392 and 39,368 electrons have not run with theirs.
            for sector, widths, default, before, returned in (
                    ((2604846, 1842), (462, 462, 456, 462), (5, 93, 371), (3, 154, 614), 2425),
                    ((2873400, 2431), (606, 612, 612, 601), (7, 88, 349), (4, 153, 609), 2850),
                    ((3786832, 2979), (750, 744, 744, 740), (11, 69, 273), (6, 125, 497), 3236),
                    ((4130510, 3704), (930, 924, 924, 926), (15, 62, 248), (8, 117, 465), 3466),
                    ((6157900, 4928), (1230, 1236, 1236, 1226), (29, 43, 172), (15, 83, 330), 3758)):
                now, then = cut(sector, widths, columns_of(*sector, 4, blocks=1)), cut(sector, widths, former(*sector, 4))
                self.assertEqual((now[:3], then[:3], round(then[3] - now[3])), (default, before, returned))
                # A budget of 2 GiB, with which the two ran, cuts as the default does, and one of 4 GiB as before.
                for budget, expected in ((2 * gib, now), (4 * gib, then)):
                    with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(budget)):
                        self.assertEqual(cut(sector, widths, columns_of(*sector, 4, blocks=1)), expected)
            # 10,456 electrons on 16 GPUs are cut as before, in three rounds, although their slab may be narrower.
            small = (1924792, 1314)
            self.assertEqual((columns_of(*small, 4, blocks=1), former(*small, 4)), (139, 164))
            self.assertEqual(cut(small, (330, 330, 324, 330), 139), cut(small, (330, 330, 324, 330), 164))
            shapes = [(rows, columns) for rows in (257, 10**5, 10**6, 1924792, 2873400, 3786832, 6157900)
                      for columns in (8, 64, 500, 1300, 2979, 4928)]
            # Three devices keep the slabs that they had, and two do up to 16 GiB of basis ...
            for rows, columns in shapes:
                self.assertEqual(columns_of(rows, columns, 3, blocks=1), former(rows, columns, 3), (rows, columns))
                if rows * columns <= 2 * gib:
                    self.assertEqual(columns_of(rows, columns, 2, blocks=1), former(rows, columns, 2), (rows, columns))
            # ... and a budget of 4 GiB gives four and more devices theirs, whatever the size of the basis.
            os.environ['PARSEC_CUPY_STREAMING_RITZ_BYTES'] = str(4 * gib)
            for rows, columns in shapes:
                for devices in (4, 5, 8):
                    self.assertEqual(columns_of(rows, columns, devices, blocks=1), former(rows, columns, devices),
                                     (rows, columns, devices))
            # It is their way back only.  Fewer devices took an eighth of a basis below 32 GiB per slab, and
            # take 4 GiB with it: 10,456 electrons on 8 GPUs three rounds of 220 columns for five of 132.
            self.assertEqual([(columns_of(*small, devices, blocks=1), former(*small, devices)) for devices in (2, 3)],
                             [(278, 164), (219, 164)])
            self.assertEqual([cut(small, (660, 654), width)[:2] for width in (278, 164)], [(3, 220), (5, 132)])
            # A budget that is set is the slab of every step, on any number of devices.
            for budget in (gib, 3 * gib, 6 * gib):
                os.environ['PARSEC_CUPY_STREAMING_RITZ_BYTES'] = str(budget)
                for devices in (2, 4, 8):
                    self.assertEqual(columns_of(3786832, 2979, devices, blocks=1), columns_of(3786832, 2979, devices))
        # The fit rule counts the workspace of those slabs.  On four A100 of 79.25 GiB, of which the blocks
        # may take 67.4, a sector of 39,368 electrons in a sphere of 36.1 A (226 GiB) counts 61.5 GiB, and
        # one of 6,417,000 rows (236 GiB, about 36.6 A) 63.9; with slabs of 4 GiB that one counted 67.9.
        measured, _runtime = _stand_in(total_bytes=85093777408)
        with patch.object(distributed_state, 'require_cupy', lambda: (measured, None)), patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS',
                         'PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION', 'PARSEC_CUPY_EXCHANGE_CHUNK_BYTES',
                         'PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES'):
                os.environ.pop(name, None)
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_RANGES'] = 'fixed'
            fits = distributed_state.shared_basis_fits
            self.assertEqual((fits(6157900, 4928, 4), fits(6417000, 4928, 4)), (True, True))
            with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(4 * gib)):
                self.assertEqual((fits(6157900, 4928, 4), fits(6417000, 4928, 4)), (True, False))
            # Four devices share a basis of up to 249 GiB instead of 233, two of up to 125 instead of 117.
            rows = 6 * 10**6
            self.assertEqual([fits(rows, columns, 4) for columns in (5580, 5590)], [True, False])
            self.assertEqual([fits(rows, columns, 2) for columns in (2790, 2800)], [True, False])
            with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(4 * gib)):
                self.assertEqual([fits(rows, columns, 4) for columns in (5220, 5230)], [True, False])
                self.assertEqual([fits(rows, columns, 2) for columns in (2610, 2620)], [True, False])

    def test_slabs_of_the_step_with_one_tall_block_on_two_devices_take_2_gib_at_most(self):
        gib = 1 << 30
        columns_of, workspace = distributed_state.slab_columns, distributed_state.slab_workspace
        limit = 'PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES'

        def former(rows, columns, devices=2):
            # The slab that this step cut before: an eighth of the basis, at least 1 and at most 4 GiB.
            budget = min(max(rows * columns, gib), 4 * gib)
            return max(1, min(budget // (8 * rows), columns // (2 * devices)))

        def cut(rows, widths, width, group=None):
            """Rounds, columns of the widest slab and of the widest round, and megabytes of the workspace."""
            sizes = [[last - first for first, last in taken] for taken in distributed_state.block_rounds(widths, width)]
            if group is None:
                cp, _runtime = _stand_in()
                group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
            widest, joined = max(map(max, sizes)), max(map(sum, sizes))
            return len(sizes), widest, joined, 8 * group._cut_capacity(rows, widest, joined, 1) / 2**20

        def halves(columns):
            return columns - columns // 2, columns // 2

        with patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES', limit):
                os.environ.pop(name, None)
            self.assertEqual(distributed_state.pair_slab_bytes(), 2 * gib)
            # Rows and column blocks of the sectors of 14,680, 19,392, 23,768 and 29,576 electrons on 8 GPUs
            # (36, 52, 84 and 114 GiB): the cut of the default against that of slabs of 4 GiB, as rounds,
            # widest slab and widest round, and the megabytes that the workspace returns.  The two largest ran
            # with a budget of 2 GiB and gave back 3936 and 3786 MiB of their peaks; the other two have not run.
            for rows, widths, default, before, returned in (
                    (2604846, (924, 918), (9, 103, 205), (5, 185, 369), 3259),
                    (2873400, (1218, 1213), (14, 87, 174), (7, 174, 348), 3814),
                    (3786832, (1494, 1484), (22, 68, 136), (11, 136, 271), 3929),
                    (4130510, (1854, 1850), (29, 64, 128), (15, 124, 248), 3782)):
                columns = sum(widths)
                now, then = cut(rows, widths, columns_of(rows, columns, 2, blocks=1)), cut(rows, widths, former(rows, columns))
                self.assertEqual((now[:3], then[:3], round(then[3] - now[3])), (default, before, returned))
                # The workspace of the budget is two slabs of 2 GiB; the two-block step keeps two of 4.
                self.assertLess(8 * workspace(rows, columns, 2, 1), 4 * gib)
                self.assertGreater(8 * workspace(rows, columns, 2, 2), 7.9 * gib)
                # A budget of 2 GiB, with which the two ran, cuts as the default does, and one of 4 GiB as before.
                for budget, expected in ((2 * gib, now), (4 * gib, then)):
                    with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(budget)):
                        self.assertEqual(cut(rows, widths, columns_of(rows, columns, 2, blocks=1)), expected)
                # So does the limit of two devices at 4 GiB, and at 3 GiB it cuts as a budget of 3 GiB does.
                with patch.dict(os.environ, {limit: str(4 * gib)}):
                    self.assertEqual(cut(rows, widths, columns_of(rows, columns, 2, blocks=1)), then)
                with patch.dict(os.environ, {limit: str(3 * gib)}):
                    between = columns_of(rows, columns, 2, blocks=1)
                with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(3 * gib)):
                    self.assertEqual(columns_of(rows, columns, 2, blocks=1), between)
            # The sectors of 10,456 electrons on 8 GPUs (18.8 GiB) may cut narrower slabs than the eighth of
            # the basis that they took, 2.4 GiB, and keep their five rounds of 132 columns; those of 7,120
            # electrons (8.0 GiB) keep their slab and their five of 90.
            for rows, widths, slabs, rounds in ((1924792, (660, 654), (139, 164), (5, 132, 263)),
                                                (1194720, (450, 447), (112, 112), (5, 90, 180))):
                columns = sum(widths)
                self.assertEqual((columns_of(rows, columns, 2, blocks=1), former(rows, columns)), slabs)
                self.assertEqual(cut(rows, widths, slabs[0]), cut(rows, widths, slabs[1]))
                self.assertEqual(cut(rows, widths, slabs[0])[:3], rounds)
            # The limit is the same for a basis of any size.  A column of a million rows is 8 MiB: a slab is the
            # eighth of the basis up to 2048 columns, 16 GiB, and the 256 columns of 2 GiB from there on, where
            # it grew to 512 at 4096 columns, 32 GiB.  The workspace is two slabs.
            rows, sizes = 1 << 20, (2047, 2048, 2049, 3000, 4095, 4096, 8192)
            self.assertEqual([columns_of(rows, columns, 2, blocks=1) for columns in sizes], [255] + [256] * 6)
            self.assertEqual([former(rows, columns) for columns in sizes], [255, 256, 256, 375, 511, 512, 512])
            self.assertEqual([8 * workspace(rows, columns, 2, 1) // 2**20 for columns in sizes], [4080] + [4096] * 6)
            # A slab never narrows as columns are added, so the workspace that the fit rule counts for the
            # columns of a first solve is the most that a later step of the trimmed basis holds.  A limit
            # that began at 32 GiB halved the slab there: the sector of 14,680 electrons, were it solved for
            # 1,652 columns first (32.06 GiB) and trimmed to 1,646 (31.94 GiB), counted 4093 MiB and then
            # held 6558.
            rows = 2604846
            slabs = [columns_of(rows, columns, 2, blocks=1) for columns in range(1, 4000)]
            self.assertEqual(slabs, sorted(slabs))
            held = [workspace(rows, columns, 2, 1) for columns in range(1, 4000)]
            self.assertEqual(held, sorted(held))
            cp, _runtime = _stand_in()
            group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
            counted = 8 * workspace(rows, 1652, 2, 1) / 2**20
            steps = [cut(rows, halves(columns), columns_of(rows, columns, 2, blocks=1), group) for columns in (1652, 1646)]
            self.assertEqual([step[:2] for step in steps], [(9, 92), (8, 103)])
            self.assertEqual((int(counted), [int(step[3]) for step in steps]), (4093, [3656, 4093]))
            self.assertLessEqual(steps[1][3], counted)
            # The way back is the limit at 4 GiB, which is then that of the streaming budget itself: the
            # former slabs for a basis of any size, also for that sector before and after its trim.
            shapes = [(rows, columns) for rows in (257, 10**5, 10**6, 1924792, 2604846, 2873400, 3786832, 4130510, 6157900)
                      for columns in (8, 64, 500, 1300, 1646, 1652, 1842, 2431, 2978, 3704, 4928)]
            changed = 0
            for rows, columns in shapes:
                if rows * columns <= 2 * gib:
                    self.assertEqual(columns_of(rows, columns, 2, blocks=1), former(rows, columns), (rows, columns))
                changed += columns_of(rows, columns, 2, blocks=1) != former(rows, columns)
                for most in (4 * gib, 6 * gib):
                    with patch.dict(os.environ, {limit: str(most)}):
                        self.assertEqual(columns_of(rows, columns, 2, blocks=1), former(rows, columns), (rows, columns))
                        self.assertEqual(workspace(rows, columns, 2, 1), workspace(rows, columns, 2, 2), (rows, columns))
            self.assertEqual(changed, 53)
            rows = 2604846
            with patch.dict(os.environ, {limit: str(4 * gib)}):
                self.assertEqual([cut(rows, halves(columns), columns_of(rows, columns, 2, blocks=1))[:2]
                                  for columns in (1652, 1646)], [(5, 166), (5, 165)])
            # A budget of 4 GiB is not: a basis below 32 GiB took an eighth of itself per slab and takes 4 GiB
            # with it, 10,456 electrons on 8 GPUs and that sector once it is trimmed.
            with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(4 * gib)):
                self.assertEqual((columns_of(1924792, 1314, 2, blocks=1), former(1924792, 1314)), (278, 164))
                self.assertEqual([cut(rows, halves(columns), columns_of(rows, columns, 2, blocks=1))[:2]
                                  for columns in (1652, 1646)], [(5, 166), (4, 206)])
            # A budget that is set is the slab whatever the limit says.  One device, three and more, and the
            # steps with two blocks never read it.
            rows = 1 << 20
            with patch.dict(os.environ, {limit: str(gib // 2)}):
                self.assertEqual(columns_of(rows, 4096, 2, blocks=1), 64)
                for budget in (gib, 3 * gib, 6 * gib):
                    with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(budget)):
                        self.assertEqual(columns_of(rows, 4096, 2, blocks=1), budget >> 23)
                for columns in (2047, 4095, 4096, 8192):
                    self.assertEqual(columns_of(rows, columns, 3, blocks=1), former(rows, columns, 3))
                    self.assertEqual(columns_of(rows, columns, 1, blocks=1), former(rows, columns, 1))
                    self.assertEqual(columns_of(rows, columns, 4, blocks=1), min(256, former(rows, columns, 4)))
                    self.assertEqual(columns_of(rows, columns, 2), former(rows, columns))
                self.assertEqual([columns_of(rows, columns, 2, wide=True) for columns in (4095, 4096, 8192)], [511, 512, 768])
            # A value that is no size is refused, by the step with one tall block where the trial basis is
            # sized, before the first filter; the steps with two and three do not read it.
            for bad in ('2GiB', '0', '-1', ''):
                os.environ[limit] = bad
                with self.assertRaisesRegex(ValueError, limit):
                    distributed_state.pair_slab_bytes()
                for raw, refused in ((None, True), ('1', True), ('2', False), ('3', False)):
                    os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS', None)
                    if raw is not None:
                        os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = raw
                    cp, _runtime = _stand_in()
                    group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
                    if refused:
                        with self.assertRaisesRegex(ValueError, limit):
                            group._capacities(_columns((3, 2)), [0, 4, 9])
                    else:
                        group._capacities(_columns((3, 2)), [0, 4, 9])
        # The fit rule counts the workspace of those slabs.  On two A100 of 79.25 GiB, of which the blocks may
        # take 67.4, the 114 GiB sector of 29,576 electrons counts 62.0 GiB instead of 66.0, and a sector of
        # 4,400 columns on the rows of 23,768 electrons (124 GiB), which counted 71.1, is shared at 67.0.
        measured, _runtime = _stand_in(total_bytes=85093777408)
        with patch.object(distributed_state, 'require_cupy', lambda: (measured, None)), patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS',
                         'PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION', 'PARSEC_CUPY_EXCHANGE_CHUNK_BYTES', limit):
                os.environ.pop(name, None)
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_RANGES'] = 'fixed'
            fits = distributed_state.shared_basis_fits
            self.assertEqual((fits(4130510, 3704, 2), fits(3786832, 4400, 2)), (True, True))
            for slabs_of_4_gib in ({'PARSEC_CUPY_STREAMING_RITZ_BYTES': str(4 * gib)}, {limit: str(4 * gib)}):
                with patch.dict(os.environ, slabs_of_4_gib):
                    self.assertEqual((fits(4130510, 3704, 2), fits(3786832, 4400, 2)), (True, False))
            for fraction, counted in (('0.781', False), ('0.783', True)):
                with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION=fraction):
                    self.assertEqual(fits(4130510, 3704, 2), counted)

    def test_slabs_take_the_columns_of_every_device_once_in_rounds(self):
        for widths in ((6, 6, 6, 5), (23, 0, 0, 0), (12, 11), (30, 3, 7), (5, 0, 9), (1,)):
            ranges = _columns(widths)
            for width in (1, 2, 5, 6, 40):
                # The former cut: full slabs and a narrower last one.
                rounds = distributed_state.projection_slabs(ranges, width, False)
                with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_SLABS='full'):
                    self.assertEqual(distributed_state.projection_slabs(ranges, width), rounds)
                # As many rounds as the widest range has slabs, and one range per device in each.
                self.assertEqual(len(rounds), -(-max(widths) // width))
                self.assertEqual({len(taken) for taken in rounds}, {len(widths)})
                for index, (start, stop) in enumerate(ranges):
                    taken = [rounds[step][index] for step in range(len(rounds))]
                    self.assertEqual([column for first, last in taken for column in range(first, last)],
                                     list(range(start, stop)))
                    # Full slabs, then what is left, then empty ranges at the end of the device's own.
                    sizes = [last - first for first, last in taken]
                    full = (stop - start) // width
                    self.assertEqual(sizes[:full], [width] * full)
                    self.assertEqual(sizes[full:], [(stop - start) % width][:len(sizes) - full]
                                     + [0] * max(0, len(sizes) - full - 1))
                    self.assertTrue(all(first == last == stop for first, last in taken if first == last))

    def test_slabs_are_equal_unless_full_ones_are_named(self):
        requested = distributed_state.equal_slabs_requested
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_SLABS', None)
            os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS', None)
            self.assertTrue(requested())
            for raw, expected in (('equal', True), (' Equal ', True), ('full', False), ('FULL', False)):
                os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_SLABS'] = raw
                self.assertEqual(requested(), expected)
            cp, _runtime = _stand_in()
            group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
            for bad in ('1', 'on', 'tail', 'even', ''):
                os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_SLABS'] = bad
                with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_DISTRIBUTED_STATE_SLABS'):
                    requested()
                # A mistyped value already stops the run where the trial basis is sized, before the first
                # filter, for the steps that cut slabs: one tall block, named or by default, and two.
                for blocks in (None, '2', '1'):
                    with patch.dict(os.environ, {} if blocks is None else dict(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=blocks)):
                        with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_DISTRIBUTED_STATE_SLABS'):
                            group._capacities(_columns((3, 2)), [0, 4, 9])
                # The step with three has no slabs.
                with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='3'):
                    group._capacities(_columns((3, 2)), [0, 4, 9])

    def test_equal_slabs_cut_every_range_without_a_short_last_one(self):
        layouts = [_columns(widths) for widths in ((6, 6, 6, 5), (23, 0, 0, 0), (12, 11), (30, 3, 7), (5, 0, 9), (1,))]
        layouts += [layout for _rows, layout in _TWO_RANGES]
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_SLABS', None)
            for layout in layouts:
                if not isinstance(layout, distributed_state.ColumnLayout):
                    layout = distributed_state.ColumnLayout(layout)
                for width in (1, 2, 5, 6, 40):
                    # Equal slabs are the default and what is asked for by name.
                    rounds = distributed_state.projection_slabs(layout, width)
                    self.assertEqual(distributed_state.projection_slabs(layout, width, True), rounds)
                    self.assertEqual({len(taken) for taken in rounds}, {len(layout.parts)})
                    for index, part in enumerate(layout.parts):
                        taken = [rounds[step][index] for step in range(len(rounds))]
                        slabs = [(first, last) for first, last in taken if last > first]
                        # Every range in as few slabs of at most this width as it takes, as equal as they can
                        # be, one range after the other; then empty ranges at the end of the device's last one.
                        expected = []
                        for start, stop in part:
                            for size in _cut(stop - start, width):
                                expected.append((start, start + size))
                                start += size
                        self.assertEqual(slabs, expected)
                        self.assertEqual(taken[:len(slabs)], slabs)
                        for start, stop in part:
                            sizes = [last - first for first, last in slabs if start <= first < stop]
                            if sizes:
                                self.assertEqual((len(sizes), sum(sizes)), (-(-(stop - start) // width), stop - start))
                                self.assertLessEqual(max(sizes) - min(sizes), 1)
                                self.assertLessEqual(max(sizes), width)
                        self.assertTrue(all(first == last == (part[-1][1] if part else 0) for first, last in taken[len(slabs):]))
                    # No more rounds than the former cut takes.
                    self.assertEqual(len(rounds), len(distributed_state.projection_slabs(layout, width, False)))
        # A sector of 23,768 electrons on 16 GPUs: two column ranges of 368 to 378 columns per device.  Slabs
        # of 4 GiB have 141 columns and cut the first range into 141, 141 and 90; equal slabs are three of 124,
        # and with the 212 columns of 6 GiB two of 186.
        parts = (((0, 372), (1488, 1866)), ((372, 744), (1866, 2238)), ((744, 1116), (2238, 2610)), ((1116, 1488), (2610, 2978)))
        cuts = {}
        for name, width, equal in (('full', 141, False), ('equal', 141, True), ('wider', 212, True)):
            rounds = distributed_state.projection_slabs(distributed_state.ColumnLayout(parts), width, equal)
            cuts[name] = [[last - first for first, last in (taken[index] for taken in rounds)] for index in range(4)]
        self.assertEqual(cuts['full'], [[141, 141, 90, 141, 141, 96], [141, 141, 90, 141, 141, 90], [141, 141, 90, 141, 141, 90],
                                        [141, 141, 90, 141, 141, 86]])
        self.assertEqual(cuts['equal'], [[124, 124, 124, 126, 126, 126], [124] * 6, [124] * 6, [124, 124, 124, 122, 123, 123]])
        self.assertEqual(cuts['wider'], [[186, 186, 189, 189], [186] * 4, [186] * 4, [186, 186, 184, 184]])

    def test_slabs_are_wider_where_the_blocks_fit_with_them(self):
        gib = 1 << 30
        a100 = 85093777408
        fits, columns_of, workspace = distributed_state.wider_slabs_fit, distributed_state.slab_columns, distributed_state.slab_workspace
        # Sectors of 23,768, 19,392 and 14,680 electrons: rows, columns, and columns of a slab of 4 and of 6 GiB
        # at most.  An eighth of the smallest basis is 4.5 GiB, which is then the most.
        k12, k11, k10 = (3786832, 2979), (2873400, 2438), (2604846, 1842)
        with patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_DISTRIBUTED_STATE_SLABS',
                         'PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION', 'PARSEC_CUPY_EXCHANGE_CHUNK_BYTES'):
                os.environ.pop(name, None)
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_RANGES'] = 'fixed'
            # Wider slabs are a matter of the step with two tall blocks, which is named here.
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = '2'
            for sector, narrow, wide in ((k12, 141, 212), (k11, 186, 280), (k10, 206, 230)):
                for devices in (2, 4):
                    self.assertEqual((columns_of(*sector, devices), columns_of(*sector, devices, True)), (narrow, wide))
                    # Two slabs in the workspace, of the wider ones too.
                    self.assertEqual(workspace(*sector, devices, 2, True) - workspace(*sector, devices),
                                     (sector[0] + -(-sector[0] // devices) * devices) * (wide - narrow))
            # A basis whose eighth is below 4 GiB has no wider slabs, and no device is asked.
            self.assertFalse(fits(2000000, 1200, 4))
            measured, _runtime = _stand_in(total_bytes=a100)
            with patch.object(distributed_state, 'require_cupy', lambda: (measured, None)):
                # On 16 GPUs the blocks of all three fit with the wider slabs, on 8 GPUs those of 19,392
                # electrons too: 65.2 GiB of the 67.4 that the blocks may take.  The 84 GiB basis of 23,768
                # electrons does not fit two devices with two tall blocks at all.
                self.assertEqual([fits(*sector, 4) for sector in (k12, k11, k10)], [True, True, True])
                self.assertEqual([fits(*sector, 2) for sector in (k12, k11, k10)], [False, True, True])
                self.assertFalse(distributed_state.shared_basis_fits(*k12, 2))
                # A basis that fits with slabs of 4 GiB only keeps those and is shared as before: 103 GiB on
                # four devices count 60.6 GiB with them and 64.6 with the wider ones, of 64 that may be taken.
                tight = (4200000, 3300)
                with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION=repr(64 * gib / a100)):
                    self.assertEqual((distributed_state.shared_basis_fits(*tight, 4), fits(*tight, 4)), (True, False))
                # The former cut keeps the slabs of 4 GiB, a budget that is set is the width to cut by, and
                # the step with one tall block and the one with three have no wider slabs.
                for named in (dict(PARSEC_CUPY_DISTRIBUTED_STATE_SLABS='full'), dict(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='1'),
                              dict(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='3'), dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(5 * gib))):
                    with patch.dict(os.environ, named):
                        self.assertFalse(fits(*k12, 4), named)
                # Nor has the default, which is one tall block: with it the 84 GiB basis fits two devices.
                with patch.dict(os.environ):
                    del os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS']
                    self.assertFalse(fits(*k12, 4))
                    self.assertTrue(distributed_state.shared_basis_fits(*k12, 2))
                with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(5725689984)):
                    self.assertEqual((columns_of(*k12, 4), columns_of(*k12, 4, True)), (189, 189))
            # A device that is too small for the wider slabs: 0 bytes on the stand-in.
            small, _runtime = _stand_in()
            with patch.object(distributed_state, 'require_cupy', lambda: (small, None)):
                self.assertFalse(fits(*k12, 4))
        # The workspace of a step follows the slabs that it cut, where these are narrower than the budget
        # allows: for the ranges above two slabs of 189 columns instead of 212, and 747 columns in a round.
        cp, _runtime = _stand_in()
        group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1, 2, 3)), (0, 1, 2, 3))
        rows = k12[0]
        self.assertEqual(group._cut_capacity(rows, 189, 747), rows * 189 + rows // 4 * 747)
        self.assertLess(8 * group._work, 10.7 * gib)
        with patch.dict(os.environ):
            # The widest slabs that no set budget narrows.
            os.environ.pop('PARSEC_CUPY_STREAMING_RITZ_BYTES', None)
            self.assertGreater(8 * workspace(*k12, 4, 2, True), 11.9 * gib)
        # It never shrinks, and the step with one tall block keeps a slab in all rows in the second part too.
        self.assertEqual(group._cut_capacity(rows, 100, 400), rows * 189 + rows // 4 * 747)
        self.assertEqual(distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1, 2, 3)), (0, 1, 2, 3))._cut_capacity(257, 1, 3, 1),
                         257 * 2)

    def test_a_mistyped_share_of_the_device_stops_the_run_where_the_trial_basis_is_sized(self):
        # PARSEC_CUPY_DISTRIBUTED_STATE=1 shares a basis without asking whether its blocks fit, so nothing had
        # read the share of a device that they may take.  The two-block step with equal slabs asks in every
        # pass whether they fit with the wider slabs: the share is read with the other switches, where the
        # trial basis is sized, and not after the first filter.  The default, one tall block, never asks.
        k12 = (3786832, 2979)
        measured, _runtime = _stand_in(total_bytes=85093777408)
        cp, _runtime = _stand_in()
        group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))

        def sized():
            return group._capacities(_columns((3, 2)), [0, 4, 9])

        two = dict(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='2')
        asks = ((two, True), (dict(two, PARSEC_CUPY_DISTRIBUTED_STATE_SLABS='equal'), True),
                # The former cut and the steps with one tall block, named or by default, and with three never ask.
                (dict(two, PARSEC_CUPY_DISTRIBUTED_STATE_SLABS='full'), False),
                ({}, False), (dict(PARSEC_CUPY_DISTRIBUTED_STATE_SLABS='equal'), False),
                (dict(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='1'), False),
                (dict(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='3'), False))
        with patch.object(distributed_state, 'require_cupy', lambda: (measured, None)), patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_DISTRIBUTED_STATE_SLABS',
                         'PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS', 'PARSEC_CUPY_EXCHANGE_CHUNK_BYTES'):
                os.environ.pop(name, None)
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_RANGES'] = 'fixed'
            for good in ('0.85', '1', ' 0.5 ', '1e-9'):
                os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION'] = good
                sized()
            for bad in ('0', '85%', '1.5', '-0.2', 'nan', 'most', ''):
                os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION'] = bad
                # The fit rule names the switch whatever is wrong with its value.
                with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION'):
                    distributed_state.shared_basis_fits(*k12, 4)
                for named, read in asks:
                    with self.subTest(bad=bad, named=named), patch.dict(os.environ, named):
                        if read:
                            # What a Ritz step of the largest sector would run into ...
                            with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION'):
                                distributed_state.wider_slabs_fit(*k12, 4)
                            # ... stops the run before anything is filtered.
                            with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION'):
                                sized()
                        else:
                            # A step that never asks is not stopped by a share that it does not read.
                            self.assertFalse(distributed_state.wider_slabs_fit(*k12, 4))
                            sized()
            # Whether a step that can ask does so depends on the basis and on the budget, which cuts no
            # wider slabs where it is set.  The share is read wherever the step can ask.
            with patch.dict(os.environ, two, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(5 << 30)):
                self.assertFalse(distributed_state.wider_slabs_fit(*k12, 4))
                with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION'):
                    sized()

    def test_rounds_take_the_columns_of_every_block_once_in_equal_slabs(self):
        cases = ((6, 6, 6, 5), (23, 0, 0, 0), (12, 11), (30, 3, 7), (5, 0, 9), (1,), (0, 0, 4), (750, 744, 744, 740),
                 (1218, 1213))
        for widths in cases:
            for width in (1, 2, 5, 6, 40, 141, 189, 10**6):
                rounds = distributed_state.block_rounds(widths, width)
                # As many rounds as the widest block has slabs of this width, and one range per device in each.
                self.assertEqual(len(rounds), -(-max(widths) // width))
                self.assertEqual({len(taken) for taken in rounds}, {len(widths)})
                for index, held in enumerate(widths):
                    taken = [rounds[step][index] for step in range(len(rounds))]
                    # From the end of the block to its first column, every column once.
                    self.assertEqual([last for _first, last in taken] + [0], [held] + [first for first, _last in taken])
                    # Slabs as equal as they can be, none wider than asked, and never less given than the
                    # rounds so far are of all rounds.
                    sizes = [last - first for first, last in taken]
                    self.assertLessEqual(max(sizes), width)
                    self.assertLessEqual(max(sizes) - min(sizes), 1)
                    for step in range(len(rounds)):
                        self.assertGreaterEqual(sum(sizes[:step + 1]) * len(rounds), held * (step + 1))
        # 23,768 electrons on 16 GPUs: six rounds of 125 columns of the widest block where the former slabs
        # of at most 141 columns were 141, 141 and 90 or more in each of its two column ranges.
        rounds = distributed_state.block_rounds((750, 744, 744, 740), 141)
        self.assertEqual([[last - first for first, last in taken] for taken in rounds],
                         [[125, 124, 124, 124], [125, 124, 124, 123], [125, 124, 124, 123], [125, 124, 124, 124],
                          [125, 124, 124, 123], [125, 124, 124, 123]])

    def test_whole_rounds_bring_a_multiple_of_64_columns_together(self):
        rounds_of, columns_of = distributed_state.block_rounds, distributed_state.slab_columns

        def cut(widths, width, multiple):
            """Rounds, the widest slab and the columns that the rounds bring together, first to last."""
            rounds = rounds_of(widths, width, multiple)
            return (len(rounds), max(last - first for taken in rounds for first, last in taken),
                    [sum(last - first for first, last in taken) for taken in rounds])

        with patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES',
                         'PARSEC_CUPY_RITZ_GRAM_MULTIPLE'):
                os.environ.pop(name, None)
            # Rows and column blocks of the shared sectors of the measured series, 10,456 to 29,576 electrons
            # on 8 GPUs and 10,456 to 39,368 on 16: the columns of a slab of 2 GiB, the equal cut as rounds,
            # widest slab and widest round, and the rounds of a multiple of 64 as their number, the widest
            # slab, what goes first and the width of every round after it.
            for rows, widths, budget, equal, whole in (
                    (1924792, (660, 654), 139, (5, 132, 263), (6, 128, 34, 256)),
                    (2604846, (924, 918), 103, (9, 103, 205), (10, 96, 114, 192)),
                    (2873400, (1218, 1213), 93, (14, 87, 174), (19, 66, 127, 128)),
                    (3786832, (1494, 1484), 70, (22, 68, 136), (24, 64, 34, 128)),
                    (4130510, (1854, 1850), 64, (29, 64, 128), (29, 64, 120, 128)),
                    (1924792, (330, 330, 324, 330), 139, (3, 110, 438), (4, 96, 162, 384)),
                    (2604846, (462, 462, 456, 462), 103, (5, 93, 371), (6, 80, 242, 320)),
                    (2873400, (606, 612, 612, 601), 93, (7, 88, 349), (8, 80, 191, 320)),
                    (3786832, (750, 744, 744, 740), 70, (11, 69, 273), (12, 64, 162, 256)),
                    (4130510, (930, 924, 924, 926), 64, (15, 62, 248), (20, 48, 56, 192)),
                    (5660200, (1230, 1236, 1236, 1226), 47, (27, 46, 184), (39, 32, 64, 128))):
                width = columns_of(rows, sum(widths), len(widths), blocks=1)
                before, after = cut(widths, width, 1), cut(widths, width, 64)
                self.assertEqual((width, before[:2] + (max(before[2]),)), (budget, equal))
                self.assertEqual((after[0], after[1], after[2][0], set(after[2][1:])), whole[:3] + ({whole[3]},))
                # No round of the equal cut is a multiple, but for 23 of the 29 of 29,576 electrons on 8 GPUs.
                self.assertEqual(sum(total % 64 == 0 for total in before[2]), 23 if (rows, len(widths)) == (4130510, 2) else 0)
            # The widest rounds of the same cut while a slab of four devices took 4 GiB: what the limit of 2 GiB
            # halved for 14,680, 19,392 and 39,368 electrons, whose equal rounds above have 371, 349 and 184.
            with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(4 << 30)):
                former = [max(cut(widths, columns_of(rows, sum(widths), 4, blocks=1), 1)[2]) for rows, widths in (
                    (2604846, (462, 462, 456, 462)), (2873400, (606, 612, 612, 601)), (5660200, (1230, 1236, 1236, 1226)))]
            self.assertEqual(former, [614, 609, 354])
            # The slab of a whole round can be little more than half of the widest equal one, and the rounds
            # nearly twice as many: where the equal slabs lie just below such a slab.  Blocks of 1,857 columns,
            # three more than the wider one of 29,576 electrons on 8 GPUs, had 30 rounds of 122 to 124 columns
            # and have 58 of 64 after one of 2.  The sector of the series itself is cut so where its slabs
            # may have 63 columns, which is how a run measures that cut: 57 rounds of 64 after one of 56.
            for widths, limit, whole in (((1857, 1857), None, (59, 32, 2)), ((1854, 1850), '2081777040', (58, 32, 56))):
                with patch.dict(os.environ, {} if limit is None else dict(PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=limit)):
                    width = columns_of(4130510, sum(widths), 2, blocks=1)
                before, after = cut(widths, width, 1), cut(widths, width, 64)
                self.assertEqual((width, before[:2], min(before[2]), max(before[2])), (63 if limit else 64, (30, 62), 122, 124))
                self.assertEqual((after[0], after[1], after[2][0], set(after[2][1:])), whole + ({64},))
        # Any blocks, budgets and multiples.
        ends_of = distributed_state.SectorDeviceGroup._row_block_ends
        cases = ((6, 6, 6, 5), (23, 0, 0, 0), (12, 11), (30, 3, 7), (5, 0, 9), (1,), (0, 0, 4), (40, 37), (77, 77, 77),
                 (750, 744, 744, 740), (1218, 1213), (1854, 1850), (900, 1300), (64, 64), (640, 64, 640, 640))
        for widths in cases:
            count, columns = len(widths), sum(widths)
            for width in (1, 2, 5, 6, 16, 31, 32, 40, 64, 70, 141, 189, 10**6):
                former = rounds_of(widths, width)
                # A multiple of 1 is the equal cut, which the step took before, and so is no multiple at all.
                self.assertEqual(rounds_of(widths, width, 1), former)
                self.assertEqual(former, tuple(
                    tuple((held - -(-held * (step + 1) // len(former)), held - -(-held * step // len(former)))
                          for held in widths) for step in range(len(former))))
                most = max(last - first for taken in former for first, last in taken)
                for multiple in (4, 8, 48, 64, 128):
                    rounds = rounds_of(widths, width, multiple)
                    self.assertEqual({len(taken) for taken in rounds}, {count})
                    for index, held in enumerate(widths):
                        taken = [rounds[step][index] for step in range(len(rounds))]
                        # From the end of the block to its first column, every column once, and no slab
                        # wider than the widest of the equal cut.
                        self.assertEqual([last for _first, last in taken] + [0], [held] + [first for first, _last in taken])
                        self.assertLessEqual(max(last - first for first, last in taken), most)
                    # The slab that every device gives in a whole round: the widest that makes the round a
                    # multiple within the equal slab and the narrowest block.
                    unit = multiple // math.gcd(multiple, count)
                    slab = min(most, min(widths)) // unit * unit
                    if not slab:
                        self.assertEqual(rounds, former)
                        continue
                    whole = min(widths) // slab
                    self.assertEqual([[last - first for first, last in taken] for taken in rounds[-whole:]],
                                     [[slab] * count] * whole)
                    self.assertEqual(slab * count % multiple, 0)
                    # What the blocks hold beyond those rounds goes first, in equal slabs.
                    left = [held - whole * slab for held in widths]
                    first = rounds[:-whole]
                    self.assertEqual(len(first), -(-max(left) // most))
                    for index, held in enumerate(left):
                        sizes = [last - begin for begin, last in (taken[index] for taken in first)]
                        self.assertEqual(sum(sizes), held)
                        if sizes:
                            self.assertLessEqual(max(sizes) - min(sizes), 1)
                    # The workspace of the slabs that are cut is that of the equal cut at most, but for what
                    # the row blocks of a round of equal slabs are rounded up by, less than a row of the round;
                    # neither is more than what the fit rule counts for slabs of the budget.
                    cp, _runtime = _stand_in()
                    held = []
                    for taken in (rounds, former):
                        group = distributed_state.SectorDeviceGroup(_Owner(cp, tuple(range(count))), tuple(range(count)))
                        held.append(group._cut_capacity(
                            4097, max(last - begin for step in taken for begin, last in step),
                            max(sum(last - begin for begin, last in step) for step in taken), 1))
                    self.assertLess(held[0], held[1] + count * most)
                    if width <= columns // (2 * count):
                        tallest = -(-4097 // count)
                        self.assertLessEqual(max(held), 4097 * width + max(tallest * min(columns, count * width), 4097 * width))
                    # Every row block ends inside a tall block of the step, as those of the equal cut do.
                    for rows in (257, 4097, 3786833):
                        edges = [rows * index // count for index in range(count + 1)]
                        heights = [high - low for low, high in zip(edges[:-1], edges[1:])]
                        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed',
                                        PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='1'):
                            group = distributed_state.SectorDeviceGroup(_Owner(cp, tuple(range(count))), tuple(range(count)))
                            capacity, _chunk = group._capacities(_columns(widths), edges)
                        self.assertLessEqual(max(ends_of(rounds, widths, heights, rows)), capacity,
                                             (rows, widths, width, multiple))
        # The fit rule counts the slabs of the budget whatever the multiple: it shares the same bases.
        measured, _runtime = _stand_in(total_bytes=85093777408)
        with patch.object(distributed_state, 'require_cupy', lambda: (measured, None)), patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS',
                         'PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION', 'PARSEC_CUPY_EXCHANGE_CHUNK_BYTES',
                         'PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES'):
                os.environ.pop(name, None)
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_RANGES'] = 'fixed'
            counted = {}
            for multiple in ('1', '64', '128'):
                os.environ['PARSEC_CUPY_RITZ_GRAM_MULTIPLE'] = multiple
                counted[multiple] = [
                    (columns_of(rows, columns, devices, blocks=1), distributed_state.slab_workspace(rows, columns, devices, 1),
                     distributed_state.shared_basis_fits(rows, columns, devices))
                    for rows in (2604846, 4130510, 6 * 10**6) for devices in (2, 4)
                    for columns in (1842, 2790, 2800, 3704, 5580, 5590)]
            self.assertEqual(counted['64'], counted['1'])
            self.assertEqual(counted['128'], counted['1'])
            self.assertEqual({fits for _slab, _work, fits in counted['1']}, {True, False})

    def test_a_slab_of_a_whole_round_can_be_a_chunk_and_a_few_columns_more(self):
        # The way to the rows sends every slab of a round a chunk of its columns at a time
        # (SectorDeviceGroup._to_rows): as many as a chunk holds of the rows outside the device, then what is
        # left.  Whole rounds give every device the same slab, so all four have a chunk in every step, which
        # then takes their turns.  Rows and blocks of the sectors of the measured series on four devices,
        # 10,456 to 39,368 electrons, with the columns of a chunk of 1 GiB, the slab of a whole round, and the
        # chunk steps of a pass (the slab and H times it), those of them that all four devices send and those
        # of at most 8 columns, for the equal cut and for the whole rounds; then the columns of a chunk of
        # 1,500,000,000 bytes and the steps of the whole rounds with it.  The slabs of 96, 48 and 32 columns
        # are just wider than a chunk of 1 GiB and travel as a chunk and one of 4, 5 and 1 columns; the larger
        # chunk holds each of them.
        def steps(rows, widths, rounds, limit):
            edges = [rows * index // len(widths) for index in range(len(widths) + 1)]
            count = turns = small = 0
            for taken in rounds:
                chunks = [distributed_state.exchange_chunks(last - first, rows - (edges[index + 1] - edges[index]), limit)
                          for index, (first, last) in enumerate(taken)]
                count += 2 * max(map(len, chunks))
                turns += 2 * min(map(len, chunks))
                small += 2 * sum(max(cut[step][1] - cut[step][0] for cut in chunks if step < len(cut)) <= 8
                                 for step in range(max(map(len, chunks))))
            return count, turns, small

        with patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES',
                         'PARSEC_CUPY_RITZ_GRAM_MULTIPLE', 'PARSEC_CUPY_EXCHANGE_CHUNK_BYTES',
                         'PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS', 'PARSEC_CUPY_DISTRIBUTED_STATE_RANGES'):
                os.environ.pop(name, None)
            self.assertEqual(distributed_state.exchange_chunk_bytes(), 1 << 30)
            for rows, widths, chunk, slab, equal, whole, larger, held in (
                    (1924792, (330, 330, 324, 330), 92, 96, (12, 12, 0), (14, 14, 6), 129, (8, 8, 0)),
                    (2604846, (462, 462, 456, 462), 68, 80, (20, 20, 0), (22, 22, 0), 95, (12, 12, 0)),
                    (2873400, (606, 612, 612, 601), 62, 80, (28, 28, 0), (30, 30, 0), 87, (16, 16, 0)),
                    (3786832, (750, 744, 744, 740), 47, 64, (44, 44, 0), (46, 46, 0), 66, (24, 24, 0)),
                    (4130510, (930, 924, 924, 926), 43, 48, (60, 60, 0), (78, 78, 38), 60, (40, 40, 0)),
                    (5660200, (1230, 1236, 1236, 1226), 31, 32, (108, 108, 0), (154, 154, 76), 44, (78, 78, 0))):
                away = rows - rows // 4
                width = distributed_state.slab_columns(rows, sum(widths), 4, blocks=1)
                former, rounds = (distributed_state.block_rounds(widths, width, multiple) for multiple in (1, 64))
                self.assertEqual(((1 << 30) // 8 // away, {last - first for taken in rounds[1:] for first, last in taken}),
                                 (chunk, {slab}))
                self.assertEqual((steps(rows, widths, former, (1 << 30) // 8), steps(rows, widths, rounds, (1 << 30) // 8)),
                                 (equal, whole))
                self.assertEqual((1500000000 // 8 // away, steps(rows, widths, rounds, 1500000000 // 8)), (larger, held))
            # The larger chunk is 426,258,176 bytes more in the exchange buffer of a device, which the rule that
            # shares a basis counts: 0.7238 instead of 0.7188 of a device of 85,093,777,408 bytes for the basis
            # of 39,368 electrons, of which it allows 0.85.
            counted = []
            for chunk in (1 << 30, 1500000000):
                with patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(chunk)):
                    counted.append(distributed_state.shared_block_bytes(5660200, 4928, 4))
            self.assertEqual(counted, [61162425024, 61588683200])
            self.assertEqual(counted[1] - counted[0], 1500000000 - (1 << 30))

    def test_slabs_of_the_step_with_two_tall_blocks_are_whole_multiples_too(self):
        slabs_of = distributed_state.projection_slabs
        # A sector of 23,768 electrons on 16 GPUs: two column ranges of 368 to 378 columns per device.
        parts = (((0, 372), (1488, 1866)), ((372, 744), (1866, 2238)), ((744, 1116), (2238, 2610)), ((1116, 1488), (2610, 2978)))
        layout = distributed_state.ColumnLayout(parts)

        def sizes(width, equal, multiple):
            rounds = slabs_of(layout, width, equal, multiple)
            return [[last - first for first, last in (taken[index] for taken in rounds)] for index in range(4)]

        # Equal slabs of 124 and of 186 or 189 columns become slabs of 64 and of 128 with a narrower last one
        # in every range, and the full slabs of 141 slabs of 128.
        self.assertEqual(sizes(141, True, 64), [[64] * 5 + [52] + [64] * 5 + [58], ([64] * 5 + [52]) * 2,
                                               ([64] * 5 + [52]) * 2, [64] * 5 + [52] + [64] * 5 + [48]])
        self.assertEqual(sizes(212, True, 64), [[128, 128, 116, 128, 128, 122], [128, 128, 116] * 2, [128, 128, 116] * 2,
                                               [128, 128, 116, 128, 128, 112]])
        self.assertEqual(sizes(141, False, 64), [[128, 128, 116, 128, 128, 122], [128, 128, 116] * 2, [128, 128, 116] * 2,
                                                [128, 128, 116, 128, 128, 112]])
        layouts = [_columns(widths) for widths in ((6, 6, 6, 5), (23, 0, 0, 0), (12, 11), (30, 3, 7), (5, 0, 9), (1,), (200, 77))]
        layouts += [layout for _rows, layout in _TWO_RANGES] + [parts]
        for ranges in layouts:
            ranges = ranges if isinstance(ranges, distributed_state.ColumnLayout) else distributed_state.ColumnLayout(ranges)
            for width in (1, 2, 5, 6, 40, 70, 141):
                for equal in (True, False):
                    former = slabs_of(ranges, width, equal)
                    # A multiple of 1 is the cut as it was, and so is one that no slab reaches.
                    self.assertEqual(slabs_of(ranges, width, equal, 1), former)
                    self.assertEqual(slabs_of(ranges, width, equal, 10**6), former)
                    for multiple in (4, 64):
                        rounds = slabs_of(ranges, width, equal, multiple)
                        self.assertEqual({len(taken) for taken in rounds}, {len(ranges.parts)})
                        for index, part in enumerate(ranges.parts):
                            taken = [(first, last) for first, last in (step[index] for step in rounds) if last > first]
                            before = [(first, last) for first, last in (step[index] for step in former) if last > first]
                            # Every column of the device once, in the order of its block.
                            self.assertEqual([column for first, last in taken for column in range(first, last)],
                                             [column for start, stop in part for column in range(start, stop)])
                            for start, stop in part:
                                mine = [last - first for first, last in taken if start <= first < stop]
                                theirs = [last - first for first, last in before if start <= first < stop]
                                if not mine:
                                    continue
                                # Slabs that reach the multiple, the widest of the equal ones of a range or
                                # the full one of the budget, are cut down to a whole one, and a range ends
                                # with what is left; any other range is cut as before.
                                widest = max(theirs) if equal else width
                                if widest < multiple:
                                    self.assertEqual(mine, theirs)
                                else:
                                    whole = widest - widest % multiple
                                    self.assertEqual(mine, [whole] * ((stop - start) // whole)
                                                     + [(stop - start) % whole] * bool((stop - start) % whole))
                                self.assertLessEqual(max(mine), max(theirs))

    def test_a_row_block_ends_where_no_round_writes_rows_over_columns_that_are_still_to_go(self):
        ends_of = distributed_state.SectorDeviceGroup._row_block_ends
        cases = (
            (257, (6, 6, 6, 5)), (257, (23, 0, 0, 0)), (257, (12, 11)), (257, (30, 3, 7)), (61, (5, 0, 9)),
            (3, (1, 1, 0, 1)), (258, (7, 7, 7)), (1000, (3, 40)), (64, (16, 16, 16, 16)), (4097, (8, 8, 8, 8)),
            # Sectors of 19,392 and 23,768 electrons on 16 and on 8 GPUs, trial bases and later passes.
            (2873400, (606, 612, 612, 608)), (2873400, (606, 612, 612, 601)), (2873400, (1218, 1220)),
            (3786832, (750, 744, 744, 741)), (3786832, (750, 744, 744, 740)), (3786833, (1494, 1485)),
        )
        for rows, widths in cases:
            count, columns = len(widths), sum(widths)
            edges = [rows * index // count for index in range(count + 1)]
            heights = [high - low for low, high in zip(edges[:-1], edges[1:])]
            for width in (1, 2, 3, 7, 141, 189, 10**6):
                rounds = distributed_state.block_rounds(widths, width)
                ends = ends_of(rounds, widths, heights, rows)
                totals = [sum(last - first for first, last in taken) for taken in rounds]
                for index, (held, height, end) in enumerate(zip(widths, heights, ends)):
                    # The row block lies in front of its end ...
                    self.assertGreaterEqual(end, height * columns)
                    arrived, smallest = 0, height * columns
                    for step, taken in enumerate(rounds):
                        arrived += totals[step]
                        # ... the rows of a round in front of those of the rounds before ...
                        written = (end - height * arrived, end - height * (arrived - totals[step]))
                        # ... and behind every column that a later round takes, which is still in its place.
                        for later in rounds[step + 1:]:
                            first, last = later[index]
                            if last > first:
                                self.assertLessEqual(rows * last, written[0], (rows, widths, width, index, step))
                                smallest = max(smallest, rows * last + height * arrived)
                    # No smaller end would do.
                    self.assertEqual(end, smallest, (rows, widths, width, index))
                # Every row block fits a tall block of the step with one, which has about a column more room
                # than those of the other steps: a row of all devices per row of the tallest row block.
                cp, _runtime = _stand_in()
                with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed'):
                    capacity = {}
                    for blocks in ('1', '2'):
                        group = distributed_state.SectorDeviceGroup(_Owner(cp, tuple(range(count))), tuple(range(count)))
                        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=blocks):
                            capacity[blocks], _chunk = group._capacities(_columns(widths), edges)
                self.assertEqual(capacity['1'], capacity['2'] + count * max(heights))
                self.assertLessEqual(max(ends), capacity['1'], (rows, widths, width))
        # Without that room the rows of a round would reach into the columns.  Of 17 rows and 17 columns the
        # last of four devices holds 5 each.  In slabs of one column it gives a column in each of five rounds,
        # and the others have given all of theirs after four: it then holds 16 columns of its rows next to
        # one of its own, 97 elements in a block that the other steps give 85.
        rows, widths, heights = 17, (4, 4, 4, 5), [4, 4, 4, 5]
        ends = ends_of(distributed_state.block_rounds(widths, 1), widths, heights, rows)
        self.assertEqual(ends, [68, 68, 68, 97])
        self.assertGreater(max(ends), max(rows * 5, 5 * 17))

    def test_fit_counts_one_tall_block_and_the_slab_workspace_unless_two_or_three_are_named(self):
        gib = 1 << 30
        cp, _runtime = _stand_in(total_bytes=160 * gib)
        # 80 GiB of basis in columns of 80 MiB.  A slab of the step with one tall block has the 25 columns of
        # 2 GiB, on two devices as on four.
        rows, columns = 10 * (1 << 20), 1024
        slab, column = {2: 25 * 80 / 1024, 4: 25 * 80 / 1024}, 80 / 1024
        cases = (
            # devices, ranges, needed GiB: the block with a column of room, the chunk and the workspace
            (4, 'fixed', 20 + column + 1 + 2 * slab[4]),
            (2, 'fixed', 40 + column + 1 + 2 * slab[2]),
            (4, 'balanced', 2 * 25 + column + 1 + 2 * slab[4]),
        )
        def select(blocks):
            os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS', None)
            if blocks is not None:
                os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = blocks

        with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)), patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES'):
                os.environ.pop(name, None)
            # One tall block is counted by default and where it is named.
            for blocks in (None, '1'):
                select(blocks)
                for devices, policy, needed in cases:
                    for fraction, fits in (((needed + .02) / 160, True), ((needed - .02) / 160, False)):
                        with patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(gib),
                                        PARSEC_CUPY_DISTRIBUTED_STATE_RANGES=policy,
                                        PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION=repr(fraction)):
                            self.assertEqual(distributed_state.shared_basis_fits(rows, columns, devices), fits,
                                             (blocks, devices, policy, fraction))
        # Sectors of 19,392 and of 23,768 electrons in D2 symmetry on A100 of 79.25 GiB, of which the rule
        # lets the blocks take 67.4: devices of the sector -> GiB that the rule counts with one tall block,
        # with two and with three.  The larger basis, 84 GiB, is shared by two devices with one block only,
        # and so by default.  The workspace of one block is 4 GiB less than that of two, on two devices as
        # on four: with slabs of 4 GiB one block counted 35.08 and 51.01 GiB on two.
        a100, limit = 85093777408, .85 * 85093777408 / gib
        measured, _runtime = _stand_in(total_bytes=a100)
        counted = {(2873400, 2438): {2: (31.10, 61.16, 79.29), 4: (18.05, 35.06, 40.15)},
                   (3786832, 2979): {2: (47.00, 93.01, 127.07), 4: (25.99, 50.98, 64.04)}}
        with patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_EXCHANGE_CHUNK_BYTES',
                         'PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES'):
                os.environ.pop(name, None)
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_RANGES'] = 'fixed'
            for sector, by_devices in counted.items():
                for devices, sizes in by_devices.items():
                    for blocks, size in zip((None, '1', '2', '3'), (sizes[0], *sizes)):
                        select(blocks)
                        # The bytes by their name, as the rule of the state storage reads them, ...
                        self.assertAlmostEqual(distributed_state.shared_block_bytes(*sector, devices) / gib, size,
                                               delta=.005)
                        # ... and what is counted, read on a device that is large enough for all of them.
                        with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)):
                            for margin, fits in ((.01, True), (-.01, False)):
                                with patch.dict(os.environ,
                                                PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION=repr((size + margin) / 160)):
                                    self.assertEqual(distributed_state.shared_basis_fits(*sector, devices), fits,
                                                     (sector, devices, blocks))
                        os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION', None)
                        with patch.object(distributed_state, 'require_cupy', lambda: (measured, None)):
                            self.assertEqual(distributed_state.shared_basis_fits(*sector, devices), size < limit,
                                             (sector, devices, blocks))

    def test_fit_counts_two_tall_blocks_and_the_slab_workspace_where_two_are_named(self):
        gib = 1 << 30
        cp, _runtime = _stand_in(total_bytes=160 * gib)
        rows, columns = 10 * (1 << 20), 1024  # 80 GiB of basis, slabs of 51 columns of 80 MiB
        slab = 51 * 80 / 1024
        cases = (
            # devices, ranges, needed GiB
            (4, 'fixed', 2 * 20 + 1 + 2 * slab),
            (2, 'fixed', 2 * 40 + 1 + 2 * slab),
            (4, 'balanced', 3 * 25 + 1 + 2 * slab),
        )

        def select(blocks):
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = blocks

        with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)), patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_STREAMING_RITZ_BYTES', None)
            select('2')
            for devices, policy, needed in cases:
                for fraction, fits in (((needed + .25) / 160, True), ((needed - .25) / 160, False)):
                    with patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(gib),
                                    PARSEC_CUPY_DISTRIBUTED_STATE_RANGES=policy,
                                    PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION=repr(fraction)):
                        self.assertEqual(distributed_state.shared_basis_fits(rows, columns, devices), fits,
                                         (devices, policy, fraction))
            # On two devices this basis takes 89 GiB of each with two tall blocks and 121 GiB with three:
            # at 0.7 of 160 GiB only the former fit.
            os.environ.update(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(gib), PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed',
                              PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION='0.7')
            for blocks, fits in (('2', True), ('3', False)):
                select(blocks)
                self.assertEqual(distributed_state.shared_basis_fits(rows, columns, 2), fits, blocks)
            # A sector of 19,392 electrons in D2 symmetry on two A100 of 79.25 GiB, 52.2 GiB of basis: 61 GiB
            # with two tall blocks and 79 GiB with three, and 0.85 of a device are 67 GiB.
            del os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION']
            sector, a100 = (2873400, 2438), 85093777408
            measured, _runtime = _stand_in(total_bytes=a100)
            with patch.object(distributed_state, 'require_cupy', lambda: (measured, None)):
                for blocks, fits in (('2', True), ('3', False)):
                    select(blocks)
                    self.assertEqual(distributed_state.shared_basis_fits(*sector, 2), fits, blocks)
            # Gathered on the owner next to the owner's columns, as the fallback after a failed audit holds
            # it, that basis is 78.3 GiB, more than the rule lets the blocks take: it keeps no room for that.
            self.assertGreater(8 * sector[0] * sector[1] * 3 // 2, .85 * a100)
            # Small bases get narrower slabs, so that the workspace takes no more than the block it replaces:
            # an even share of the basis, and one row of the slabs for the row block that is taller than the others.
            for rows, columns, devices in ((257, 23, 4), (257, 96, 2), (10 * (1 << 20), 64, 4), (5000, 8, 4)):
                width = distributed_state.slab_columns(rows, columns, devices)
                self.assertLessEqual(distributed_state.slab_workspace(rows, columns, devices),
                                     -(-rows * columns // devices) + devices * width, (rows, columns, devices))

    def test_a_basis_is_shared_by_default_where_it_fits_and_its_devices_can_filter_it(self):
        from parsec_python.acceleration.backends.cupy_stencil_major import CuPyStencilMajorFiniteDifference
        gib = 1 << 30
        cp, _runtime = _stand_in(total_bytes=80 * gib)
        rows = 10 * (1 << 20)  # 1024 columns are 80 GiB of basis and 25 GiB of blocks on four devices

        def sector(devices=(0, 1, 2, 3), neighbors=np.zeros((2, 8), dtype=np.int32), **projectors):
            """Stands for a sector operator; the stencil holds its slot-major neighbor table unless given another."""
            stencil = object.__new__(CuPyStencilMajorFiniteDifference)
            stencil.neighbors = neighbors
            fields = dict(shape=(rows, rows), compact_finite_difference=stencil, projector_count=3,
                          fused_projector_scatter=True, custom_projector_projection=object())
            fields.update(projectors)
            operator = SimpleNamespace(**fields)
            if devices is not None:
                operator.distributed_filter_devices = devices
            return operator

        shared = distributed_state.shared_basis_devices
        with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)), patch.dict(os.environ):
            for name in ('PARSEC_CUPY_DISTRIBUTED_STATE', 'PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION',
                         'PARSEC_CUPY_DISTRIBUTED_STATE_RANGES', 'PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS',
                         'PARSEC_CUPY_EXCHANGE_CHUNK_BYTES', 'PARSEC_CUPY_STREAMING_RITZ_BYTES',
                         'PARSEC_CUPY_GENERALIZED_RITZ', 'PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION',
                         'PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD'):
                os.environ.pop(name, None)
            # Unset is auto.
            self.assertEqual(distributed_state.distributed_state_policy(), 'auto')
            self.assertEqual(shared(sector(), 1024), (0, 1, 2, 3))
            self.assertEqual(shared(sector(devices=[2, 3]), 256), (2, 3))
            # 3300 columns need 69.4 GiB with the one tall block of the default, more than 0.85 of the 80 GiB;
            # 3200 columns fit.  With slabs of 4 GiB instead of 2 it is 3100 columns that need 69.6 GiB, and
            # 3000 that fit.
            self.assertIsNone(shared(sector(), 3300))
            self.assertEqual(shared(sector(), 3200), (0, 1, 2, 3))
            with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(4 * gib)):
                self.assertIsNone(shared(sector(), 3100))
                self.assertEqual(shared(sector(), 3000), (0, 1, 2, 3))
            # The two tall blocks of the former step are 71 GiB for 1600 columns already, and the three of the
            # step before it for 1200: a basis that only they refuse is shared by default.
            self.assertEqual(shared(sector(), 1600), (0, 1, 2, 3))
            with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='2'):
                self.assertIsNone(shared(sector(), 1600))
                self.assertEqual(shared(sector(), 1200), (0, 1, 2, 3))
            with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='3'):
                self.assertIsNone(shared(sector(), 1200))
                self.assertEqual(shared(sector(), 1024), (0, 1, 2, 3))
            # A sector on one device has nothing to share, whatever the step would hold: it never reads the count.
            with patch.object(distributed_state, 'tall_blocks_per_device', side_effect=AssertionError('read')):
                self.assertIsNone(shared(sector(devices=(2,)), 1024))
                self.assertIsNone(shared(sector(devices=None), 1024))
            # No group, or a group of one device.
            self.assertIsNone(shared(sector(devices=None), 1024))
            self.assertIsNone(shared(sector(devices=(2,)), 1024))
            # A stencil packed into implicit tiles is shared like the slot-major one: the other devices
            # are given the tiles of the owner.
            tiles = np.zeros(8, dtype=np.int32)
            self.assertEqual(shared(sector(neighbors=tiles), 1024), (0, 1, 2, 3))
            # The fit rule does not read the layout: the same columns fit and are refused, with either slabs.
            self.assertEqual(shared(sector(neighbors=tiles), 3200), (0, 1, 2, 3))
            self.assertIsNone(shared(sector(neighbors=tiles), 3300))
            with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(4 * gib)):
                self.assertEqual(shared(sector(neighbors=tiles), 3000), (0, 1, 2, 3))
                self.assertIsNone(shared(sector(neighbors=tiles), 3100))
            # Devices that could not filter the columns: a stencil held by another kernel, and projectors
            # without the fused kernels.
            other = sector()
            other.compact_finite_difference = SimpleNamespace(neighbors=np.zeros((2, 8), dtype=np.int32))
            self.assertIsNone(shared(other, 1024))
            self.assertIsNone(shared(sector(fused_projector_scatter=False), 1024))
            self.assertIsNone(shared(sector(custom_projector_projection=None), 1024))
            self.assertEqual(shared(sector(projector_count=0, custom_projector_projection=None), 1024), (0, 1, 2, 3))
            # A shared basis is always solved in its filtered form, so a basis that one device would
            # orthonormalize stays there: 3 columns are below the work threshold of that solve, 4 reach it.
            self.assertIsNone(shared(sector(), 3))
            self.assertEqual(shared(sector(), 4), (0, 1, 2, 3))
            for name, value in (('PARSEC_CUPY_GENERALIZED_RITZ', 'off'), ('PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION', 'qr'),
                                ('PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD', str(10**14))):
                with patch.dict(os.environ, {name: value}):
                    self.assertIsNone(shared(sector(), 1024), name)
                    # A caller that names no state count has only the group and the filter checked.
                    self.assertEqual(shared(sector()), (0, 1, 2, 3))
            with patch.dict(os.environ, PARSEC_CUPY_GENERALIZED_RITZ='on'):
                self.assertEqual(shared(sector(), 3), (0, 1, 2, 3))
            # An explicit 1 shares whatever has a group, an explicit 0 nothing.
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE'] = '1'
            self.assertEqual(shared(sector(), 1600), (0, 1, 2, 3))
            self.assertEqual(shared(sector(neighbors=tiles), 1024), (0, 1, 2, 3))
            with patch.dict(os.environ, PARSEC_CUPY_GENERALIZED_RITZ='off'):
                self.assertEqual(shared(sector(), 1024), (0, 1, 2, 3))
            self.assertIsNone(shared(sector(devices=(2,)), 1024))
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE'] = '0'
            self.assertIsNone(shared(sector(), 1024))
            self.assertIsNone(distributed_state.sector_device_group(sector(), 1024))

    def test_a_shared_basis_is_neither_spilled_to_the_host_nor_restored_from_it(self):
        import importlib
        eigval = importlib.import_module('parsec_python.acceleration.Eigensolvers.eigval')
        # The symmetry scheduler spills the states of large sectors when PARSEC_CUPY_SECTOR_STATE_STORAGE lets
        # it, and asks every sector solver to restore its own before the next solve.
        cp = SimpleNamespace(ndarray=np.ndarray)
        basis = object.__new__(distributed_state.DistributedBasis)
        solver = object.__new__(eigval.CuPyEigvalSolver)
        solver._state = state = SimpleNamespace(subspace=SimpleNamespace(vectors=basis))
        with patch.object(eigval, 'require_cupy', lambda: (cp, None)):
            self.assertIs(solver.offload_state_to_host(), state)
            self.assertIs(solver.restore_state_to_device(), state)
        self.assertIs(solver._state, state)
        self.assertIs(state.subspace.vectors, basis)

    def test_only_a_chebff_first_solve_asks_for_a_group_and_later_passes_follow_the_basis(self):
        import importlib
        from parsec_python.Eigensolvers.eigval import EigvalSettings
        eigval = importlib.import_module('parsec_python.acceleration.Eigensolvers.eigval')
        cp, _runtime = _stand_in()
        rows, devices = 9, (0, 1)
        with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)):
            group = distributed_state.SectorDeviceGroup(_Owner(cp, devices), devices)
            shared = distributed_state.DistributedBasis(
                group, group.map(lambda index, device: cp.empty((rows, 3 - index), order='F')))
        asked, calls, answer = [], [], [None]

        def ask(operator, columns):
            asked.append(columns)
            return answer[0]

        def call(function, operator, argument, **options):
            """Stands for the solvers: a first solve leaves a basis of its kind, a later pass keeps what it gets."""
            calls.append((function, options.get('group')))
            if function in (eigval.run_subspace_filter, eigval.run_distributed_subspace):
                return SimpleNamespace(state=argument, residual_norms=None), 0.0
            vectors = shared if function is eigval.run_distributed_chebff else np.zeros((rows, argument), order='F')
            state = SimpleNamespace(operator_dimension=rows, wanted_states=argument, eigenvalues=np.zeros(argument),
                                    vectors=vectors)
            return SimpleNamespace(state=state, residual_norms=None), 0.0

        def solver(method):
            made = object.__new__(eigval.CuPyEigvalSolver)
            made.settings = EigvalSettings(initial_method=method, safety_buffer=0)
            made.operator = SimpleNamespace(shape=(rows, rows))
            made._state = None
            made.timing_stats = SimpleNamespace(first_solve_seconds=0.0, first_solve_calls=0, subspace_solve_seconds=0.0,
                                                subspace_solve_calls=0, solve_calls=0, download_seconds=0.0)
            made.compute_subspace_residuals, made.retain_vectors_on_device = False, True
            return made

        with patch.object(eigval, 'sector_device_group', ask), patch.object(eigval, 'synchronized_call', call), \
             patch.object(eigval, 'resolve_device_stages', lambda stats: None), \
             patch.object(eigval, 'require_cupy', lambda: (SimpleNamespace(asnumpy=np.asarray), None)):
            # CHEBDAV creates no shared basis, so it asks for no group, whatever would be answered.
            answer[0] = group
            davidson = solver('chebdav')
            davidson.solve(5)
            davidson.solve(5)
            self.assertEqual((asked, calls), ([], [(eigval.run_chebdav, None), (eigval.run_subspace_filter, None)]))
            del calls[:]
            # CHEBFF asks once, for the states of its first solve.
            spread = solver('chebff')
            first = spread.solve(5)
            self.assertEqual((asked, calls), ([5], [(eigval.run_distributed_chebff, group)]))
            self.assertIs(first.state.subspace.vectors, shared)
            # A later pass stays with the devices that hold the basis. The policy is not asked again: for the
            # fewer states of a trimmed sector it could decline now.
            answer[0] = None
            spread.truncate_state(4)
            later = spread.solve(4)
            self.assertEqual((asked, calls[1:]), ([5], [(eigval.run_distributed_subspace, group)]))
            self.assertEqual((later.solver_path, tuple(later.vectors.shape)), ('subspace', (rows, 4)))
            self.assertIs(later.vectors.group, group)
            del asked[:], calls[:]
            # A basis that its first solve left on one device takes the single-device pass.
            alone = solver('chebff')
            alone.solve(5)
            answer[0] = group
            alone.solve(5)
            self.assertEqual((asked, calls), ([5], [(eigval.run_chebff, None), (eigval.run_subspace_filter, None)]))

    def test_columns_are_interleaved_unless_contiguous_ranges_are_named(self):
        requested = distributed_state.interleaved_layout_requested
        cp, _runtime = _stand_in()
        pair = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
        four = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1, 2, 3)), (0, 1, 2, 3))
        trial, wider = uniform_filter_blocks(23, 6, 5), uniform_filter_blocks(53, 6, 5)
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT', None)
            os.environ.pop('PARSEC_CUPY_DISTRIBUTED_STATE_RANGES', None)
            self.assertTrue(requested())
            # The lower half of the four blocks is the first two, the upper half the others with the top block.
            self.assertEqual(pair.column_layout(trial).parts, (((0, 6), (12, 18)), ((6, 12), (18, 23))))
            # Two blocks in a half are too few for four devices: their ranges stay contiguous.
            self.assertEqual(four.column_layout(trial).ranges, four.column_ranges(trial))
            self.assertEqual([len(part) for part in four.column_layout(wider).parts], [2] * 4)
            for raw, expected in (('interleaved', True), (' INTERLEAVED ', True), ('contiguous', False), (' Contiguous', False)):
                os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT'] = raw
                self.assertEqual(requested(), expected)
            # Named, the former layout: one range of neighbouring columns per device.
            self.assertEqual(pair.column_ranges(trial), ((0, 12), (12, 23)))
            self.assertEqual(pair.column_layout(trial).ranges, pair.column_ranges(trial))
            self.assertEqual(four.column_layout(wider).ranges, four.column_ranges(wider))
            # A mistyped value stops the run where the trial basis is laid out, before anything is filtered.
            for bad in ('1', 'on', 'interleave', 'round-robin', ''):
                os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT'] = bad
                with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT'):
                    pair.column_layout(trial)
            # Balanced ranges move columns from pass to pass, which the interleaved layout is there to avoid:
            # left unset the layout follows them, and named together with them it is refused.
            del os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT']
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_RANGES'] = 'balanced'
            self.assertFalse(requested())
            self.assertEqual(pair.column_layout(trial).ranges, pair.column_ranges(trial))
            self.assertEqual(four.column_layout(wider).ranges, four.column_ranges(wider))
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT'] = 'interleaved'
            with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_DISTRIBUTED_STATE_RANGES'):
                pair.column_layout(trial)
            os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT'] = 'contiguous'
            self.assertEqual(pair.column_layout(trial).ranges, pair.column_ranges(trial))

    def test_a_layout_names_the_columns_of_every_block(self):
        layout = distributed_state.ColumnLayout
        # A device is given as its one range or as its ranges; an empty range holds nothing.
        self.assertEqual(layout(((0, 6), (6, 11))), layout((((0, 6),), ((6, 11),))))
        self.assertEqual(layout(((0, 6), (6, 11))).ranges, ((0, 6), (6, 11)))
        two = layout((((0, 6), (12, 18)), ((6, 12), (18, 23))))
        self.assertEqual((two.widths, two.columns), ((12, 11), 23))
        with self.assertRaisesRegex(ValueError, 'one column range'):
            two.ranges
        self.assertEqual(layout(((0, 5), (5, 5), (5, 14))).widths, (5, 0, 9))
        self.assertEqual(layout(((0, 5), (), (5, 14))).widths, (5, 0, 9))
        # Where the columns of a block lie in the basis: all of them, and a few that reach into both ranges.
        self.assertEqual(two.spans(0), ((0, 0, 6), (12, 6, 6)))
        self.assertEqual(two.spans(1), ((6, 0, 6), (18, 6, 5)))
        self.assertEqual(two.spans(0, 4, 9), ((4, 4, 2), (12, 6, 3)))
        self.assertEqual(two.spans(1, 6, 8), ((18, 6, 2),))
        self.assertEqual(two.spans(1, 2, 6), ((8, 2, 4),))
        self.assertEqual(two.spans(0, 3, 3), ())
        self.assertEqual([two.offset(0, column) for column in (0, 5, 12, 17)], [0, 5, 6, 11])
        self.assertEqual([two.offset(1, column) for column in (6, 11, 18, 22)], [0, 5, 6, 10])
        with self.assertRaises(ValueError):
            two.offset(0, 6)
        # Ranges that leave a column out, hold one twice or descend on a device are refused.
        for bad in (((0, 6), (7, 11)), ((0, 6), (5, 11)), ((1, 6), (6, 11)), (((6, 11), (0, 6)),), ((6, 0), (6, 11))):
            with self.assertRaises(ValueError, msg=bad):
                layout(bad)
        # One device may still hold the later columns, as long as its own ranges ascend.
        self.assertEqual(layout(((6, 11), (0, 6))).widths, (5, 6))

    def test_interleaved_ranges_give_every_device_a_share_of_both_halves(self):
        for columns in (47, 48, 53, 54, 59, 60, 96, 97, 2431, 2438, 2972):
            for size in (6, 4):
                for devices in (2, 3, 4):
                    with self.subTest(columns=columns, size=size, devices=devices):
                        trial = uniform_filter_blocks(columns, size, 30)
                        ranges = distributed_state.interleaved_ranges(trial, devices)
                        # A later pass filters this many blocks at its lower degree, with an odd count one more
                        # than at the higher degree unless a narrower top block follows.
                        later = subspace_filter_blocks(columns, size, 15, 3)
                        low = sum(block.degree == 12 for block in later)
                        self.assertEqual(low, (columns // size + 1) // 2)
                        if min(low, len(trial) - low) < devices:
                            self.assertIsNone(ranges)
                            continue
                        layout = distributed_state.ColumnLayout(ranges)
                        self.assertEqual((layout.columns, [len(part) for part in layout.parts]), (columns, [2] * devices))
                        lower, upper = zip(*ranges)
                        # The parts of each half follow each other in the order of the devices, the top block last.
                        self.assertEqual([lower[0][0], lower[-1][1], upper[0][0], upper[-1][1]],
                                         [0, low * size, low * size, columns])
                        for half in (lower, upper):
                            self.assertEqual([stop for _start, stop in half[:-1]], [start for start, _stop in half[1:]])
                            # Whole blocks, as equal in number as they can be.
                            self.assertEqual({start % size for start, _stop in half}, {0})
                            counts = [-(-(stop - start) // size) for start, stop in half]
                            self.assertLessEqual(max(counts) - min(counts), 1)
                            self.assertGreaterEqual(min(counts), 1)
                        # Every device filters whole blocks of the later plan, also once trailing columns are cut.
                        for count in (columns, columns - 1, columns - size, columns - 2 * size - 1):
                            plan = subspace_filter_blocks(count, size, 15, 3)
                            taken = distributed_state.SectorDeviceGroup._whole_block_partitions(layout.leading(count), plan)
                            filtered = [index for runs in taken for first, last in runs for index in range(first, last)]
                            self.assertEqual(sorted(filtered), list(range(len(plan))))
                        # The leading columns of the basis are the leading columns of every block.
                        some = (0, 1, low * size - 1, low * size + 1, columns - 7)
                        for count in (range(columns + 1) if columns < 100 else some):
                            trimmed = layout.leading(count)
                            self.assertEqual(trimmed.columns, count)
                            for index in range(devices):
                                self.assertEqual(_held(trimmed, index), _held(layout, index)[:trimmed.widths[index]])
                                self.assertTrue(all(column < count for column in _held(trimmed, index)))

    def test_interleaved_ranges_balance_the_work_of_a_later_pass(self):
        # Sector bases of the size of 19,392 and 23,768 electrons: columns of the trial basis and of a later
        # pass, which has a few columns less, and the work of the busiest device over the average that the
        # interleaved ranges stay below.  The layout is that of the trial basis, so cutting many columns
        # leaves the last device with less than its share: with 6% of them gone the busiest has 7% more
        # than the average, where contiguous ranges give it 28% more.
        for first, later, devices, bound in ((2438, 2431, 4, 1.02), (2438, 2431, 2, 1.02), (2979, 2972, 4, 1.02),
                                             (2979, 2800, 4, 1.08)):
            trial = uniform_filter_blocks(first, 6, 30)
            contiguous = distributed_state.ColumnLayout(
                tuple((trial[start].start, trial[stop - 1].stop) for start, stop in block_partitions(trial, devices)))
            interleaved = distributed_state.ColumnLayout(distributed_state.interleaved_ranges(trial, devices))
            # No block is wider than those of contiguous ranges by more than a filter block.
            self.assertLessEqual(max(interleaved.widths), max(contiguous.widths) + 6)
            for degree in (15, 10):
                plan = subspace_filter_blocks(later, 6, degree, 3)
                work = {}
                for layout in (contiguous, interleaved):
                    taken = distributed_state.SectorDeviceGroup._whole_block_partitions(layout.leading(later), plan)
                    work[layout] = [sum(block.degree * (block.stop - block.start) for start, stop in runs
                                        for block in plan[start:stop]) for runs in taken]
                    self.assertEqual(sum(work[layout]), sum(block.degree * (block.stop - block.start) for block in plan))
                mean = sum(work[contiguous]) / devices
                # The filter lasts as long as its busiest device: with contiguous ranges one that holds only
                # columns of the higher degree.
                self.assertGreater(max(work[contiguous]), (1 + 2.8 / degree) * mean)
                self.assertLess(max(work[interleaved]), bound * mean)

    def test_a_starting_sigma_per_block_gives_the_carried_recurrence(self):
        lower, upper, reference = 1.2, 4.6, .7
        for plan in (uniform_filter_blocks(23, 6, 7), subspace_filter_blocks(53, 6, 9, 3), uniform_filter_blocks(5, 6, 1)):
            steps = np.concatenate(([0], np.cumsum([block.degree for block in plan])))
            for reset in (False, True):
                carried = _table(plan, lower, upper, reference, reset)
                starts = starting_sigmas(plan, lower, upper, reset, reference)
                # Unless every block starts anew, sigma moves on from block to block: the starts differ.
                if len(plan) > 2:
                    self.assertEqual(starts[1] != starts[2], not reset)
                np.testing.assert_array_equal(_table(plan, lower, upper, reference, reset, starts), carried)
                for first in range(len(plan)):
                    # Blocks that follow each other in the plan need the sigma carried into the first of them ...
                    np.testing.assert_array_equal(_table(plan[first:], lower, upper, reference, reset, starts[first]),
                                                  carried[steps[first]:])
                    # ... and a block on its own, or after one that does not precede it in the plan, its own.
                    order = [first] + [index for index in reversed(range(len(plan))) if index != first]
                    table = _table([plan[index] for index in order], lower, upper, reference, reset,
                                   [starts[index] for index in order])
                    offset = 0
                    for index in order:
                        np.testing.assert_array_equal(table[offset:offset + plan[index].degree],
                                                      carried[steps[index]:steps[index + 1]])
                        offset += plan[index].degree
            with self.assertRaisesRegex(ValueError, 'per block'):
                _table(plan, lower, upper, reference, False, [None] * (len(plan) + 1))


class ExchangeStandInTests(unittest.TestCase):
    layouts = (
        (257, (6, 6, 6, 5)),
        (257, (23, 0, 0, 0)),   # everything on the owner, as before the first balanced repartition
        (257, (12, 11)),
        (61, (5, 0, 9)),
        (3, (2, 1, 3, 2)),      # fewer rows than devices: the first device holds none
    )

    def _round_trip(self, rows, widths, *, eager=False, **environment):
        """Columns -> rows -> columns; returns the group and what each conversion asked of the stand-in.

        ``widths`` are those of column blocks that follow each other, or a layout.
        """

        cp, runtime = _stand_in(eager)
        layout = _layout(widths)
        devices = tuple(range(len(layout.parts)))
        # Blocks that follow each other are named by their ranges, as most callers name them.
        ranges = layout if layout is widths else _columns(widths)
        host = np.random.default_rng(53).normal(size=(rows, layout.columns))
        with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)), patch.dict(os.environ, environment):
            owner = _Owner(cp, devices)
            group = distributed_state.SectorDeviceGroup(owner, devices)
            edges = group.row_edges(rows)

            def place(index, device):
                block = cp.empty((rows, layout.widths[index]), order='F')
                np.asarray(block)[...] = host[:, _held(layout, index)]
                return block

            def asked():
                taken = SimpleNamespace(
                    allocated=list(runtime.allocated), copies=list(runtime.copies), kernels=list(runtime.kernels))
                del runtime.allocated[:], runtime.copies[:], runtime.kernels[:]
                return taken

            blocks = group.map(place)
            asked()
            kept = list(blocks)
            row_blocks = group._to_rows(kept, ranges, edges, keep=True)
            to_rows = asked()
            self.assertEqual(len(kept), len(devices))
            given = list(blocks)
            again = group._to_rows(given, ranges, edges)
            self.assertEqual(given, [])
            for index in range(len(devices)):
                for block in (row_blocks[index], again[index]):
                    self.assertEqual((block.device.id, block.flags.f_contiguous), (index, True))
                    np.testing.assert_array_equal(np.asarray(block), host[edges[index]:edges[index + 1], :])
            del again
            asked()
            fresh = group._to_columns(row_blocks, ranges, edges)
            to_columns = asked()
            # Supplied column blocks are overwritten where they lie.
            for block in kept:
                np.asarray(block)[...] = np.nan
            reused = group._to_columns(row_blocks, ranges, edges, targets=kept)
            for index in range(len(devices)):
                self.assertIs(reused[index], kept[index])
                for block in (fresh[index], reused[index]):
                    self.assertEqual((block.device.id, block.flags.f_contiguous), (index, True))
                    np.testing.assert_array_equal(np.asarray(block), host[:, _held(layout, index)])
            self.assertFalse(any(stream.pending for stream in runtime.streams))
            return group, to_rows, to_columns

    def _every_layout(self, **environment):
        """Round trips of all layouts, a column or a few per chunk or everything at once, both stream orders."""

        for rows, widths in self.layouts:
            for chunk in ('8', str(8 * 193), str(8 * 500), str(1 << 30)):
                for eager in (False, True):
                    with self.subTest(rows=rows, widths=widths, chunk=chunk, eager=eager):
                        self._round_trip(rows, widths, eager=eager, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=chunk, **environment)

    def test_every_entry_arrives_for_any_chunk_size_and_layout(self):
        # With both switches of the exchange as they are by default, and with both off.
        self._every_layout()
        self._every_layout(PARSEC_CUPY_EXCHANGE_CONCURRENT='0', PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0')

    def test_every_entry_arrives_with_two_column_ranges_per_device(self):
        former = dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='0', PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0')
        for rows, layout in _TWO_RANGES:
            # A column per chunk, a few, so that chunks reach from one range into the other, and everything at once.
            for chunk in ('8', str(8 * 193), str(8 * 500), str(8 * 1500), str(1 << 30)):
                for eager in (False, True):
                    for switches in ({}, former):
                        with self.subTest(rows=rows, layout=layout.parts, chunk=chunk, eager=eager, switches=switches):
                            self._round_trip(rows, layout, eager=eager, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=chunk, **switches)
        rows, layout = _TWO_RANGES[0]
        self.assertEqual(layout.parts, (((0, 6), (24, 36)), ((6, 12), (36, 42)), ((12, 18), (42, 48)), ((18, 24), (48, 53))))
        for chunk, runs in ((str(1 << 30), [2, 2, 2, 2]), (str(8 * 1500), [4, 3, 3, 3])):
            # A chunk is a run of whole columns for each range that it reaches into.  With all of a block in
            # one chunk that is two runs per peer.  With 7 columns in a chunk, the 18 columns of the first
            # device travel as columns 0:7, 7:14 and 14:18 of its block, of which the first has one column of
            # its second range; the first chunk of the other devices has one such column too.
            _group, to_rows, to_columns = self._round_trip(rows, layout, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=chunk)
            for device in range(4):
                sent = [sum((owner, source) == (other, device) for owner, source, _stream in to_rows.copies)
                        for other in range(4) if other != device]
                fetched = [sum((owner, source) == (device, other) for owner, source, _stream in to_columns.copies)
                           for other in range(4) if other != device]
                self.assertEqual((sent, fetched), ([runs[device]] * 3,) * 2)

    def test_concurrent_receives_use_one_stream_per_source(self):
        # Four devices that take no turns on the way to their rows: every device receives from all at once.
        together = dict(PARSEC_CUPY_EXCHANGE_PAIRS='0')
        self._every_layout(PARSEC_CUPY_EXCHANGE_CONCURRENT='1', PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0', **together)
        for flag in ('0', '1'):
            group, to_rows, to_columns = self._round_trip(
                257, (6, 6, 6, 5), PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 500), PARSEC_CUPY_EXCHANGE_CONCURRENT=flag,
                **together)
            self.assertEqual(group.to_rows_copies, 'together' if flag == '1' else 'queued')
            for asked in (to_rows, to_columns):
                for device in range(4):
                    streams = {}
                    for owner, source, stream in asked.copies:
                        if owner == device:
                            streams.setdefault(source, set()).add(stream)
                    self.assertEqual(sorted(streams), [other for other in range(4) if other != device])
                    used = [stream for of_source in streams.values() for stream in of_source]
                    if flag == '1':
                        # A stream of this device per source, none of them the one it computes on.
                        self.assertEqual((len(used), len(set(used))), (3, 3))
                        self.assertNotIn(group._stream(device), used)
                        self.assertEqual({stream.device_id for stream in used}, {device})
                    else:
                        self.assertEqual(set(used), {group._stream(device)})
        # The caller of a conversion reads the switch for all of its chunks; the threads of the devices never do.
        readers, read = [], distributed_state.concurrent_exchange_requested
        with patch.object(distributed_state, 'concurrent_exchange_requested',
                          lambda: readers.append(threading.current_thread()) or read()):
            self._round_trip(
                257, (6, 6, 6, 5), PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 500), PARSEC_CUPY_EXCHANGE_CONCURRENT='1')
        self.assertEqual(set(readers), {threading.current_thread()})

    def _to_rows_by_map(self, rows, widths, **environment):
        """One conversion to rows: the group, and the copies between devices that each of its maps issued.

        ``widths`` are those of column blocks that follow each other, or a layout.  A map has returned when
        every device has done its part of it; those that issued no copy between devices are left out.  The two
        switches of the order of the copies are as ``environment`` names them and unset where it names none,
        whatever the calling shell names.
        """

        cp, runtime = _stand_in()
        layout = _layout(widths)
        devices = tuple(range(len(layout.parts)))
        host = np.random.default_rng(59).normal(size=(rows, layout.columns))
        with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)), patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_EXCHANGE_PAIRS', None)
            os.environ.pop('PARSEC_CUPY_EXCHANGE_CONCURRENT', None)
            os.environ.update(environment)
            group = distributed_state.SectorDeviceGroup(_Owner(cp, devices), devices)
            edges = group.row_edges(rows)

            def place(index, device):
                block = cp.empty((rows, layout.widths[index]), order='F')
                np.asarray(block)[...] = host[:, _held(layout, index)]
                return block

            blocks = group.map(place)
            run, issued = group.map, []

            def run_and_record(function):
                first = len(runtime.copies)
                values = run(function)
                issued.append(runtime.copies[first:])
                return values

            group.map = run_and_record
            row_blocks = group._to_rows(blocks, layout, edges, keep=True)
        self.assertFalse(any(stream.pending for stream in runtime.streams))
        for index, block in enumerate(row_blocks):
            np.testing.assert_array_equal(np.asarray(block), host[edges[index]:edges[index + 1], :])
        return group, [copies for copies in issued if copies]

    def test_four_devices_receive_their_rows_from_one_partner_per_turn(self):
        turns = distributed_state._PAIR_TURNS
        others = [(device, source) for device in range(4) for source in range(4) if source != device]
        # Two columns in a chunk: the columns of every device travel in three chunks.
        rows, widths, chunk = 257, (6, 6, 6, 5), dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 500))
        # Every other layout and chunk size, in both stream orders, with the pairs on and by default.
        self._every_layout(PARSEC_CUPY_EXCHANGE_PAIRS='1', PARSEC_CUPY_EXCHANGE_CONCURRENT='1')
        for switches in ({}, dict(PARSEC_CUPY_EXCHANGE_PAIRS='1'), dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='1')):
            # Also where the calling shell names both switches off: a conversion here has those of its call only.
            with patch.dict(os.environ, PARSEC_CUPY_EXCHANGE_PAIRS='0', PARSEC_CUPY_EXCHANGE_CONCURRENT='0'):
                group, maps = self._to_rows_by_map(rows, widths, **chunk, **switches)
            # Three turns per chunk, each a map of its own: all devices are done with one before the next.
            self.assertEqual((group.to_rows_copies, len(maps)), ('pairs', 3 * 3), switches)
            for number, copies in enumerate(maps):
                partners = turns[number % 3]
                # Two pairs without a common device, both directions of each ...
                self.assertEqual(sorted((device, source) for device, source, _stream in copies),
                                 [(device, partners[device]) for device in range(4)])
                # ... on the stream that the receiving device computes on.
                self.assertEqual([stream is group._stream(device) for device, _source, stream in copies], [True] * 4)
            self.assertEqual(group._inbound, {})
        # Switched off, or with every device queueing on its one stream: all twelve copies of a chunk in one
        # map, as before.
        former = (('together', dict(PARSEC_CUPY_EXCHANGE_PAIRS='0')),
                  ('queued', dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='0')),
                  ('queued', dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='0', PARSEC_CUPY_EXCHANGE_PAIRS='1')))
        for name, switches in former:
            group, maps = self._to_rows_by_map(rows, widths, **chunk, **switches)
            self.assertEqual((group.to_rows_copies, len(maps)), (name, 3), switches)
            for copies in maps:
                self.assertEqual(sorted((device, source) for device, source, _stream in copies), others)
                self.assertEqual([stream is group._stream(device) for device, _source, stream in copies],
                                 [name == 'queued'] * 12)
                for device in range(4):
                    used = [stream for owner, _source, stream in copies if owner == device]
                    self.assertEqual(len(set(used)), 1 if name == 'queued' else 3)
                    if name == 'queued':
                        # In the order of the devices.
                        self.assertEqual([source for owner, source, _stream in copies if owner == device],
                                         [other for other in range(4) if other != device])
        # A chunk that reaches into two column ranges of its device arrives as two runs of whole columns, both
        # from the partner of the turn.  The first device holds 18 columns, the others 12 or 11, and 7 are a
        # chunk: the first chunk of every device has a column of its second range.  In the third chunk only the
        # first device sends: the three others take it in one map and not in turns (see the next test).
        rows, layout = _TWO_RANGES[0]
        chunk = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 1500))
        group, maps = self._to_rows_by_map(rows, layout, **chunk)
        self.assertEqual((group.to_rows_copies, len(maps)), ('pairs', 2 * 3 + 1))
        for number, copies in enumerate(maps[:-1]):
            partners = turns[number % 3]
            self.assertEqual({source == partners[device] and stream is group._stream(device)
                              for device, source, stream in copies}, {True})
        self.assertEqual([len(copies) for copies in maps], [8, 8, 8, 4, 4, 4, 3])
        # The same copies as without turns, those of the first two chunks in another order.
        _group, together = self._to_rows_by_map(rows, layout, PARSEC_CUPY_EXCHANGE_PAIRS='0', **chunk)
        self.assertEqual([len(copies) for copies in together], [24, 12, 3])
        self.assertEqual(sorted((device, source) for copies in maps for device, source, _stream in copies),
                         sorted((device, source) for copies in together for device, source, _stream in copies))
        # The caller of a conversion reads the switch for all of its chunks; the threads of the devices never do.
        readers, read = [], distributed_state.paired_exchange_requested
        with patch.object(distributed_state, 'paired_exchange_requested',
                          lambda: readers.append(threading.current_thread()) or read()):
            self._round_trip(257, (6, 6, 6, 5), PARSEC_CUPY_EXCHANGE_CONCURRENT='1', **chunk)
        self.assertEqual(set(readers), {threading.current_thread()})

    def test_a_chunk_that_not_all_four_devices_send_is_received_from_all_senders_at_once(self):
        # The turns were timed against the twelve copies of a chunk that every device sends to every other one.
        # Once a device has sent all its columns, what the others still send is issued as without pairs: one map
        # in which a device receives from all senders at once, a stream per source.  In turns the chunk of one
        # sender would be a single copy per map where three started together.
        turns, chunk = distributed_state._PAIR_TURNS, dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 500))
        # Two columns in a chunk.  (rows, widths, chunks that all four send, copies of each chunk that follows):
        cases = (
            (257, (23, 0, 0, 0), 0, [3] * 12),   # one sender throughout
            (257, (12, 11, 0, 0), 0, [6] * 6),   # two
            (257, (8, 8, 7, 0), 0, [9] * 4),     # three
            (257, (9, 6, 6, 5), 3, [3] * 2),     # 5, 3, 3 and 3 chunks: the last two have one sender
            (257, (8, 8, 6, 4), 2, [9, 6]),      # 4, 4, 3 and 2 chunks: three senders, then two
            (3, (2, 1, 3, 2), 0, [9]),           # all four send, and the first device has no rows to receive
        )
        for rows, widths, whole, rest in cases:
            with self.subTest(rows=rows, widths=widths):
                group, maps = self._to_rows_by_map(rows, widths, **chunk)
                other, together = self._to_rows_by_map(rows, widths, PARSEC_CUPY_EXCHANGE_PAIRS='0', **chunk)
                # The order that the switches chose, which only a chunk of four senders takes.
                self.assertEqual((group.to_rows_copies, other.to_rows_copies), ('pairs', 'together'))
                self.assertEqual([len(copies) for copies in maps], [4] * (3 * whole) + rest)
                self.assertEqual([len(copies) for copies in together], [12] * whole + rest)
                for number, copies in enumerate(maps[:3 * whole]):
                    partners = turns[number % 3]
                    self.assertEqual(sorted((device, source) for device, source, _stream in copies),
                                     [(device, partners[device]) for device in range(4)])
                    self.assertEqual({stream is group._stream(device) for device, _source, stream in copies}, {True})
                # The chunks that follow: the copies of the conversion without pairs, map for map, each on the
                # stream that its device keeps for its source.
                for copies, same in zip(maps[3 * whole:], together[whole:], strict=True):
                    self.assertEqual(sorted((device, source) for device, source, _stream in copies),
                                     sorted((device, source) for device, source, _stream in same))
                    for made, issued in ((group, copies), (other, same)):
                        self.assertEqual({stream is made._inbound[device][source] for device, source, stream in issued},
                                         {True})

    def test_only_four_devices_take_turns_and_only_on_the_way_to_their_rows(self):
        chunk = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 500))
        # Two devices, the one pair there is, three and five: the copies of a chunk are issued together on a
        # stream per source whatever the switch says.
        for widths in ((12, 11), (8, 8, 7), (5, 5, 5, 4, 4)):
            count, issued = len(widths), {}
            for pairs in ('1', '0'):
                group, maps = self._to_rows_by_map(257, widths, PARSEC_CUPY_EXCHANGE_PAIRS=pairs, **chunk)
                self.assertEqual(group.to_rows_copies, 'together', (widths, pairs))
                # Two columns in a chunk: in the first every device receives from every other one.
                self.assertEqual(len(maps[0]), count * (count - 1))
                for copies in maps:
                    for device in range(count):
                        sources = [source for owner, source, _stream in copies if owner == device]
                        used = [stream for owner, _source, stream in copies if owner == device]
                        # A stream per source, none of them the one that the device computes on.
                        self.assertEqual((len(set(sources)), len(set(used))), (len(used),) * 2)
                        self.assertNotIn(group._stream(device), used)
                issued[pairs] = [sorted((device, source) for device, source, _stream in copies) for copies in maps]
            self.assertEqual(issued['1'], issued['0'])
        # The way back of four devices: every device fetches from its three peers at once, a stream per source,
        # with the pairs on as with them off.
        for pairs in ('1', '0'):
            group, to_rows, to_columns = self._round_trip(
                257, (6, 6, 6, 5), PARSEC_CUPY_EXCHANGE_PAIRS=pairs, PARSEC_CUPY_EXCHANGE_CONCURRENT='1', **chunk)
            self.assertEqual(group.to_rows_copies, 'pairs' if pairs == '1' else 'together')
            for device in range(4):
                fetched = [(source, stream) for owner, source, stream in to_columns.copies if owner == device]
                # Three chunks from each of three peers.
                self.assertEqual(sorted(source for source, _stream in fetched),
                                 sorted([other for other in range(4) if other != device] * 3))
                streams = {source: {stream for other, stream in fetched if other == source} for source, _stream in fetched}
                self.assertEqual([len(used) for used in streams.values()], [1, 1, 1])
                used = [stream for of_source in streams.values() for stream in of_source]
                self.assertEqual(len(set(used)), 3)
                self.assertNotIn(group._stream(device), used)
                # On the way to the rows the pairs took the stream of the device itself.
                own = {stream is group._stream(device) for owner, _source, stream in to_rows.copies if owner == device}
                self.assertEqual(own, {pairs == '1'})

    def test_each_device_takes_one_buffer_of_at_most_a_chunk(self):
        rows, widths = 257, (24, 24, 24, 24)
        # Row blocks of 64, 64, 64 and 65 rows: a tall block has 65 x 96 elements and a column
        # holds at most 193 rows of other devices.  Limit in elements -> buffer in elements and
        # rounds in which the 24 columns of a device travel to or from each of its three peers.
        # The last buffer is all of a tall block outside one device, which used to be packed at once.
        tall = 65 * 96
        for limit, chunk, rounds in ((1, 193, 24), (512, 512, 12), (2000, 2000, 3), (1 << 27, tall - tall // 4, 1)):
            # Whole blocks are exchanged by the step with two tall blocks, whose capacity this is.
            group, to_rows, to_columns = self._round_trip(
                rows, widths, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * limit), PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed',
                PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='2')
            self.assertEqual((group._tall, group._chunk), (tall, chunk))
            for device in range(len(widths)):
                for asked in (to_rows, to_columns):
                    self.assertEqual(sorted(size for owner, size in asked.allocated if owner == device and size), [chunk, tall])
                    self.assertEqual(sum(owner == device for owner, _source, _stream in asked.copies), 3 * rounds)

    def test_no_buffer_outlives_its_conversion_to_rows(self):
        # When map() returns, the thread of a device still refers to the function it ran last and to all that
        # this function can reach.  A buffer held that way would be missing from the pool when the next
        # conversion asks for it, and the pool would take a second one.
        cp, _runtime = _stand_in()
        devices, ranges = (0, 1, 2, 3), _columns((6, 6, 6, 5))
        environment = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 500))
        made, last = [], []
        empty = cp.empty

        def watched(shape, **options):
            array = empty(shape, **options)
            made.append((weakref.ref(array), array.size))
            return array

        cp.empty = watched
        with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)), patch.dict(os.environ, environment):
            group = distributed_state.SectorDeviceGroup(_Owner(cp, devices), devices)
            blocks = group.map(lambda index, device: cp.empty((257, ranges[index][1] - ranges[index][0]), order='F'))
            run = group.map

            def run_and_keep(function):
                # As on the threads, only the last function stays referred to.
                last[:] = [function]
                return run(function)

            group.map = run_and_keep
            group._to_rows(blocks, ranges, group.row_edges(257), keep=True)
        buffers = [array for array, size in made if size == group._chunk]
        self.assertEqual((group._chunk, len(buffers)), (500, 4))
        self.assertEqual([array() is None for array in buffers], [True] * 4)

    def test_column_copies_run_the_fast_index_down_a_column(self):
        self._every_layout(PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='1', PARSEC_CUPY_EXCHANGE_CONCURRENT='0')
        launched = {}
        for flag in ('0', '1'):
            _group, to_rows, to_columns = self._round_trip(
                257, (6, 6, 6, 5), PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 500), PARSEC_CUPY_EXCHANGE_COLUMN_COPIES=flag)
            for asked in (to_rows, to_columns):
                # Row blocks have 64 or 65 rows and column blocks 257: a step of one element along the
                # last axis stays inside a column, a longer one crosses to the next column.
                for target, source in asked.kernels:
                    self.assertEqual((target[-1] == 8, source[-1] == 8), (flag == '1',) * 2)
            launched[flag] = (len(to_rows.kernels), len(to_columns.kernels))
        # Per device its own rows and, with two columns in a chunk, three chunks of pieces for three peers.
        self.assertEqual(launched, {'0': (4 * 10, 4 * 10), '1': (4 * 10, 4 * 10)})
        # The caller of a conversion reads the switch for all of its copies; the threads of the devices never do.
        readers, read = [], distributed_state.column_copies_requested
        with patch.object(distributed_state, 'column_copies_requested',
                          lambda: readers.append(threading.current_thread()) or read()):
            self._round_trip(
                257, (6, 6, 6, 5), PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 500), PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='1')
        self.assertEqual(set(readers), {threading.current_thread()})


class RitzStandInTests(unittest.TestCase):
    """The Ritz step with three, with two and with one tall block per device."""

    layouts = (
        (257, (6, 6, 6, 5)),
        (257, (23, 0, 0, 0)),   # everything on the owner: three devices apply H to nothing
        (257, (12, 11)),
        (257, (30, 3, 7)),      # the second device holds fewer columns than a slab can have
        (61, (5, 0, 9)),
        (3, (1, 1, 0, 1)),      # fewer rows than devices: the first device holds none
    )
    # The host solve with its defaults, whatever the calling shell exports.
    settings = dict(PARSEC_CUPY_RITZ_DENSE_BACKEND='host', PARSEC_CUPY_RITZ_EIGH_BACKEND='host', PARSEC_CUPY_RITZ_SYRK='auto',
                    PARSEC_CUPY_RITZ_GRAM_SLABS='8', PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1e8',
                    PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed')
    unset = ('PARSEC_CUPY_RITZ_CONDITION', 'PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_ROTATE_COLUMN_COPY',
             'PARSEC_CUPY_EXCHANGE_CHUNK_BYTES', 'PARSEC_CUPY_EXCHANGE_CONCURRENT', 'PARSEC_CUPY_EXCHANGE_COLUMN_COPIES',
             'PARSEC_CUPY_EXCHANGE_PAIRS', 'PARSEC_CUPY_DISTRIBUTED_STATE_SLABS', 'PARSEC_CUPY_RITZ_CONDITION_HELPER',
             'PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES', 'PARSEC_CUPY_RITZ_GRAM_MULTIPLE',
             'PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE')

    def _step(self, rows, widths, blocks, *, eager=False, unstable=False, roomy=False, hartree_device=None,
              **environment):
        """One Ritz step of a filtered basis: what went in, what came out and what the stand-in was asked.

        ``widths`` are those of column blocks that follow each other, or a layout.  ``blocks`` names the tall
        blocks per device; ``None`` leaves that to the default.  ``roomy`` column blocks have the capacity that
        the group gives the blocks of a trial basis; the others are as large as their columns.  The small
        problem is solved on the host unless ``environment`` names the device for it.  ``hartree_device`` is
        the device that the sector operator names as the holder of the Hartree objects.
        """

        cp, runtime = _stand_in(eager)
        layout = _layout(widths)
        devices = tuple(range(len(layout.parts)))
        states = layout.columns
        rng = np.random.default_rng(71)
        # Like a filtered basis: far from orthonormal, but well conditioned.
        host = np.linalg.qr(rng.normal(size=(rows, states)))[0] @ (np.eye(states) + .2 * rng.normal(size=(states, states)))
        hamiltonian = sp.diags((-np.ones(rows - 1), 2.2 + rng.uniform(-.1, .1, rows), -np.ones(rows - 1)), (-1, 0, 1)).tocsr()
        run = SimpleNamespace(host=host, hamiltonian=hamiltonian, layout=layout, solution=None, filtered=None, solved_on=None,
                              estimates=[], condition=None, runtime=runtime)
        if layout is not widths:
            run.ranges = _columns(widths)
        solve = rayleigh_ritz.solve_whitened_ritz

        def small_solve(raw_overlap, raw_projection):
            run.gram = np.array(raw_overlap), np.array(raw_projection)
            # What the column blocks hold while the small problem is solved.
            run.at_solve = [np.array(np.asarray(block)) for block in run.given]
            if unstable:
                raise rayleigh_ritz.GeneralizedRitzStabilityError('forced')
            run.solution = solve(raw_overlap, raw_projection)
            return run.solution

        def device_solve(raw_overlap, raw_projection, condition=None):
            # Stands for the small solve on the device: the host solve of the pair that the current device
            # holds once its stream has done the work queued before, and the results as arrays of that device.
            # A ``condition`` is given the mirrored overlap as an array of the device, complete, before that
            # solve, and is waited for after it whatever it raised, as the solve on a device does.
            run.solved_on = runtime.local.device
            pair = cp.asnumpy(raw_overlap), cp.asnumpy(raw_projection)
            if condition is None:
                return tuple(cp.asarray(part) for part in small_solve(*pair))
            mirrored = cp.asarray(np.tril(pair[0]) + np.tril(pair[0], -1).T)
            waiting = condition(mirrored)
            try:
                solution = small_solve(*pair)
            finally:
                run.condition = waiting()
                del mirrored
            return tuple(cp.asarray(part) for part in solution)

        def estimate(overlap):
            # Stands for the condition number that a device estimates of an overlap it holds: (device, thread,
            # elements) of every estimate, and the number of what its stream has brought by then.
            run.estimates.append((runtime.local.device, threading.current_thread().name, overlap.size))
            if overlap.device_id != runtime.local.device:
                raise AssertionError('a device estimates the condition number of its own copy of the overlap')
            return float(np.linalg.cond(cp.asnumpy(overlap)))

        def place(index, device):
            if roomy:
                block = group._block(cp, (rows, layout.widths[index]), run.capacity)
            else:
                block = cp.empty((rows, layout.widths[index]), order='F')
            np.asarray(block)[...] = host[:, _held(layout, index)]
            return block

        with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)), \
             patch.object(rayleigh_ritz, 'require_cupy', lambda: (cp, None)), \
             patch.object(distributed_state, 'solve_whitened_ritz', small_solve), \
             patch.object(distributed_state, 'solve_whitened_ritz_on_device', device_solve), \
             patch.object(distributed_state, '_device_overlap_condition', estimate), patch.dict(os.environ):
            for name in self.unset + ('PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS',):
                os.environ.pop(name, None)
            os.environ.update(self.settings, **environment)
            if blocks is not None:
                os.environ['PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS'] = str(blocks)
            run.sector = _Sector(cp, runtime, devices, hamiltonian)
            if hartree_device is not None:
                run.sector.hartree_device = hartree_device
            run.group = group = distributed_state.SectorDeviceGroup(run.sector, devices)
            run.capacity = group._capacities(layout, group.row_edges(rows))[0]
            run.given = group.map(place)
            del runtime.allocated[:]
            runtime.peak.update(runtime.live)
            # Blocks that follow each other need no layout, as before there was one.
            basis = distributed_state.DistributedBasis(group, run.given, layout if layout is widths else None)
            def held():
                # The thread of a device refers to the function it ran last until it runs another.
                group.map(lambda index, device: None)
                gc.collect()
                return dict(runtime.live)

            try:
                run.values, rotated, _whitened = group.ritz(run.sector, basis, stages=False)
            except rayleigh_ritz.GeneralizedRitzStabilityError as error:
                run.filtered, rotated = error.filtered, None
                # The fallback may run while the traceback of the error still refers to the failed step.
                run.live_in_handler = held()
            run.live = held()
        self.assertFalse(any(stream.pending for stream in runtime.streams))
        run.allocated, run.peak, run.copies = list(runtime.allocated), dict(runtime.peak), list(runtime.copies)
        if rotated is not None:
            self.assertEqual(rotated.layout, layout)
            run.returned = rotated.blocks
            run.rotated = np.empty((rows, states))
            for index, block in enumerate(rotated.blocks):
                run.rotated[:, _held(layout, index)] = np.asarray(block)
        return run

    def _assert_ritz_pairs(self, run):
        host, states = run.host, run.host.shape[1]
        overlap, projection = host.T @ host, host.T @ (run.hamiltonian @ host)
        # The small solve was given the lower triangles of the two Gram matrices ...
        for given, exact in zip(run.gram, (overlap, projection)):
            np.testing.assert_allclose(np.tril(given), np.tril(exact), rtol=0, atol=1e-12 * np.abs(exact).max())
        values, coefficients, _whitened = run.solution
        np.testing.assert_array_equal(run.values, values)
        dense = la.eigh((projection + projection.T) / 2, (overlap + overlap.T) / 2, eigvals_only=True)
        np.testing.assert_allclose(values, dense, rtol=0, atol=1e-10)
        # ... and its coefficients rotated the basis, which is back in the column blocks it came in.
        exact = host @ coefficients
        np.testing.assert_allclose(run.rotated, exact, rtol=0, atol=1e-12 * np.abs(exact).max())
        np.testing.assert_allclose(run.rotated.T @ run.rotated, np.eye(states), atol=1e-9)
        self.assertEqual([block is given for block, given in zip(run.returned, run.given)], [True] * len(run.given))
        self.assertEqual(run.group.passes, 1)
        timed = ('apply', 'to_rows', 'gram', 'dense', 'rotate', 'to_columns')
        self.assertEqual([stage for stage in timed if not run.group.seconds[stage] > 0.], [])
        # Of the Gram seconds, those of the overlap: a part of them where the step forms it in a call of its
        # own, with one and two tall blocks, so that what is left is the projection of its rounds or slabs.
        spent, gram = run.group.overlap_seconds, run.group.seconds['gram']
        self.assertTrue(spent == 0. if run.group.tall_blocks == 3 else 0. < spent < gram, (spent, gram))

    def test_the_overlap_is_one_call_of_the_gram_stage_and_every_round_another(self):
        # A clock that every reading moves on by one: a stage that hands one call to the devices then takes 1.
        # The overlap is one such call however the projection is cut, and every round or slab of the projection
        # is another: narrower slabs add to what is left of the Gram seconds and not to those of the overlap.
        # A step with three tall blocks forms both matrices in one call and counts none of it apart.
        rows, widths, taken = 257, (12, 11), {}
        for blocks, slab in ((1, 2), (1, 5), (2, 2), (2, 5), (3, 5)):
            ticks = itertools.count()
            with patch.object(distributed_state, 'perf_counter', lambda: float(next(ticks))):
                run = self._step(rows, widths, blocks, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * slab))
            self._assert_ritz_pairs(run)
            taken[blocks, slab] = run.group.overlap_seconds, run.group.seconds['gram']
        # The twelve columns of the wider block in slabs of two and of five: six rounds and three.  Beside them
        # and the overlap the stage holds the download of the pairs for the host solve and, with one tall
        # block, the zeroed pairs and their way back into the order of the basis.
        for blocks, others in ((1, 3), (2, 1)):
            self.assertEqual((taken[blocks, 2], taken[blocks, 5]), ((1., 6. + 1. + others), (1., 3. + 1. + others)))
        self.assertEqual(taken[3, 5], (0., 1.))

    def test_two_tall_blocks_give_the_ritz_pairs_of_three(self):
        for rows, widths in self.layouts:
            states, count = sum(widths), len(widths)
            # One column per chunk of an exchange, a few, and everything at once; both stream orders.
            for chunk in ('8', str(8 * 193), str(1 << 30)):
                for eager in (False, True):
                    exchange = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=chunk)
                    three = self._step(rows, widths, 3, eager=eager, **exchange)
                    self._assert_ritz_pairs(three)
                    # H times the columns of a device, all of them at once.
                    self.assertEqual(sorted(three.sector.applied),
                                     [(index, width) for index, width in enumerate(widths) if width])
                    # Slabs of one column, of three, and as wide as the streaming budget allows.
                    for slab in (1, 3, None):
                        budget = {} if slab is None else dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * slab))
                        with self.subTest(rows=rows, widths=widths, chunk=chunk, eager=eager, slab=slab):
                            two = self._step(rows, widths, 2, eager=eager, **exchange, **budget)
                            self._assert_ritz_pairs(two)
                            np.testing.assert_allclose(two.values, three.values, rtol=0, atol=1e-11)
                            np.testing.assert_allclose(np.linalg.svd(three.rotated.T @ two.rotated, compute_uv=False),
                                                       np.ones(states), atol=5e-9)
                            # Every device applied H to its own columns once, in equal slabs of at most the
                            # width of the budget, and nothing was projected above the diagonal slabs.
                            width = max(1, min(slab or states, states // (2 * count)))
                            for index, held in enumerate(widths):
                                taken = [columns for device, columns in two.sector.applied if device == index]
                                self.assertEqual(taken, _cut(held, width))
                            self.assertFalse(np.triu(two.gram[1], width).any())
                            # The former cut, full slabs and what is left, gives the same Ritz pairs to round-off.
                            if not eager and chunk == str(8 * 193):
                                full = self._step(rows, widths, 2, eager=eager, **exchange, **budget,
                                                  PARSEC_CUPY_DISTRIBUTED_STATE_SLABS='full')
                                self._assert_ritz_pairs(full)
                                np.testing.assert_allclose(full.values, two.values, rtol=0, atol=1e-11)
                                for index, held in enumerate(widths):
                                    taken = [columns for device, columns in full.sector.applied if device == index]
                                    self.assertEqual(taken, _cut(held, width, False))
                                # Its workspace is that of the budget, the other that of the slabs that were cut.
                                tallest = -(-rows // count)
                                joined = max(sum(sizes) for sizes in zip(*(
                                    _cut(held, width) + [0] * states for held in widths)))
                                self.assertEqual(full.group._work, rows * width + tallest * min(states, count * width))
                                self.assertEqual(two.group._work, rows * max(max(_cut(held, width), default=0) for held in widths)
                                                 + tallest * joined)
                                self.assertLessEqual(two.group._work, full.group._work)

    def test_the_step_takes_one_tall_block_unless_two_or_three_are_named(self):
        rows, widths = 257, (6, 6, 6, 5)
        # One column per slab, and blocks with the capacity that a trial basis is given.
        budget = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows))
        unnamed, one, two, three = runs = [self._step(rows, widths, blocks, roomy=True, **budget)
                                           for blocks in (None, 1, 2, 3)]
        for run in runs:
            self._assert_ritz_pairs(run)
        self.assertEqual([run.group.tall_blocks for run in runs], [1, 1, 2, 3])
        # H is applied a slab at a time and a slab workspace is held, except with three tall blocks.
        slabs = [(index, 1) for index, width in enumerate(widths) for _ in range(width)]
        self.assertEqual([sorted(run.sector.applied) for run in (unnamed, one, two)], [slabs] * 3)
        self.assertEqual(sorted(three.sector.applied), list(enumerate(widths)))
        # The group says how its step cut the slabs and how wide they were; three tall blocks cut none.
        self.assertEqual([(run.group.slab_cut, run.group.slab_columns) for run in runs],
                         [('equal', 1)] * 3 + [(None, None)])
        self.assertEqual([run.group._work for run in runs], [rows + 65 * 4] * 3 + [0])
        # Left to the default the blocks have the room of the step with one tall block, and the rows are laid
        # into them: no device takes a row block from the pool, where two tall blocks take one each.
        tall = rows * 6
        self.assertEqual([run.capacity for run in runs], [tall + 4 * 65] * 2 + [tall] * 2)
        self.assertEqual((unnamed.group.separate_row_blocks, one.group.separate_row_blocks), (0, 0))
        for run, row_blocks in ((unnamed, 0), (one, 0), (two, 1)):
            for device in range(len(widths)):
                self.assertEqual([size for owner, size in run.allocated if owner == device].count(run.capacity),
                                 row_blocks)
        # The projection of that step is summed round by round over all columns that have arrived, the one of
        # two tall blocks slab by slab below the diagonal.
        self.assertTrue(np.triu(unnamed.gram[1], 1).any())
        self.assertFalse(np.triu(two.gram[1], 1).any())
        np.testing.assert_array_equal(unnamed.values, one.values)
        np.testing.assert_allclose(unnamed.values, two.values, rtol=0, atol=1e-11)

    def test_the_step_with_two_tall_blocks_cuts_wider_slabs_where_they_fit(self):
        # A basis that a host can hold has no slabs of 4 GiB, so the two budgets and the answer of the fit rule
        # are stood in for: slabs of two columns, and of five where the wider ones fit.
        rows, widths, asked = 257, (12, 11), []

        def budget(rows, columns, devices, wide=False, blocks=2):
            return 5 if wide else 2

        def fit(answer):
            def fits(rows, columns, devices):
                asked.append((rows, columns, devices))
                return answer
            return fits

        runs = {}
        full = dict(PARSEC_CUPY_DISTRIBUTED_STATE_SLABS='full')
        for name, answer, blocks, named in (('wider', True, 2, {}), ('narrow', False, 2, {}), ('full', True, 2, full),
                                            ('one', True, 1, {}), ('default', True, None, {}),
                                            ('default named full', True, None, full)):
            del asked[:]
            with patch.object(distributed_state, 'slab_columns', budget), \
                 patch.object(distributed_state, 'wider_slabs_fit', fit(answer)):
                runs[name] = run = self._step(rows, widths, blocks, roomy=blocks != 2, **named)
            self._assert_ritz_pairs(run)
            # The steps that cut equal slabs for the two-block route ask once, for the basis of the step.
            self.assertEqual(asked, [(rows, 23, 2)] if name in ('wider', 'narrow') else [])
            np.testing.assert_allclose(run.values, runs['wider'].values, rtol=0, atol=1e-11)
        # What a run reports of its step: the blocks, the cut and the widest slab.  The cut is a switch of the
        # two-block step: naming the former one beside the default changes nothing and is not reported.
        self.assertEqual({name: (run.group.tall_blocks, run.group.slab_cut, run.group.slab_columns)
                          for name, run in runs.items()},
                         {'wider': (2, 'equal', 4), 'narrow': (2, 'equal', 2), 'full': (2, 'full', 2),
                          'one': (1, 'equal', 2), 'default': (1, 'equal', 2), 'default named full': (1, 'equal', 2)})
        for part in ('values', 'rotated'):
            np.testing.assert_array_equal(getattr(runs['default named full'], part), getattr(runs['default'], part))
        taken = {name: [[columns for device, columns in run.sector.applied if device == index] for index in range(2)]
                 for name, run in runs.items()}
        # Equal slabs of at most five columns, or of at most two; the former cut and the step with one tall
        # block, named or by default, keep to the narrow budget whatever would fit.
        self.assertEqual(taken['wider'], [[4, 4, 4], [3, 4, 4]])
        self.assertEqual(taken['narrow'], [[2] * 6, [1, 2, 2, 2, 2, 2]])
        self.assertEqual(taken['full'], [[2] * 6, [2] * 5 + [1]])
        self.assertEqual(taken['one'], [[2] * 6, [2, 2, 2, 2, 2, 1]])
        self.assertEqual(taken['default'], taken['one'])
        self.assertEqual((runs['default'].group.tall_blocks, runs['default'].group.separate_row_blocks), (1, 0))
        # The workspace is that of the slabs that were cut: one in all rows and the rows of a device of two.
        self.assertEqual((runs['wider'].group._work, runs['narrow'].group._work), (257 * 4 + 129 * 8, 257 * 2 + 129 * 4))

    def test_one_tall_block_cuts_the_slabs_of_its_own_budget(self):
        # A basis that a host can hold is far below any slab of gigabytes, so the round is stood in for: two
        # columns of all rows for every device of four.  Half of an even share of the columns is 5.
        rows, widths = 257, (12, 12, 12, 11)
        with patch.object(distributed_state, '_ROUND_BYTES', 8 * rows * 2 * len(widths)):
            one, unnamed, two = (self._step(rows, widths, blocks, roomy=blocks != 2) for blocks in (1, None, 2))
            # A budget that is set is the slab of both steps.
            set_one, set_two = (self._step(rows, widths, blocks, roomy=blocks == 1,
                                           PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 4)) for blocks in (1, 2))
            # Two and three devices share no round: their slabs are those of the streaming budget, which is
            # far below the limit of two here.
            fewer = [self._step(rows, shares, 1, roomy=True) for shares in ((12, 11), (12, 12, 11))]
        runs = (one, unnamed, two, set_one, set_two)
        for run in runs:
            self._assert_ritz_pairs(run)
            np.testing.assert_allclose(run.values, two.values, rtol=0, atol=1e-11)
        taken = [[[columns for device, columns in run.sector.applied if device == index]
                  for index in range(len(run.layout.parts))] for run in runs + tuple(fewer)]
        # The step with one tall block, named or by default, cuts by the share of the round; the step with
        # two keeps the slabs of the streaming budget, which no round narrows.
        self.assertEqual(taken[0], [[2] * 6] * 3 + [[2, 2, 2, 2, 2, 1]])
        self.assertEqual(taken[1], taken[0])
        self.assertEqual(taken[2], [[4, 4, 4]] * 3 + [[3, 4, 4]])
        self.assertEqual(taken[3], [[4, 4, 4]] * 3 + [[4, 4, 3]])
        self.assertEqual(taken[4], taken[2])
        self.assertEqual([run.group.slab_columns for run in runs], [2, 2, 4, 4, 4])
        # Its workspace is that of the narrower slabs: one in all rows and the 65 rows of a device of a round.
        self.assertEqual((one.group._work, set_one.group._work), (rows * 2 + 65 * 8, rows * 4 + 65 * 16))
        # Half of an even share of the columns is 5 for the two devices and 5 for the three.
        self.assertEqual(taken[5:], [[[4, 4, 4], [4, 4, 3]], [[4, 4, 4], [4, 4, 4], [4, 4, 3]]])
        self.assertEqual([run.group.slab_columns for run in fewer], [4, 4])
        for run in fewer:
            self._assert_ritz_pairs(run)

    def test_one_tall_block_on_two_devices_cuts_the_slabs_that_its_limit_allows(self):
        # The limit of two devices is stood in for: two columns of all rows.  The streaming budget allows half
        # of an even share of the columns, 11 of the 47 and 5 of the 23.
        rows, widths = 257, (24, 23)
        limit = dict(PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=str(8 * rows * 2))
        one, unnamed, two = (self._step(rows, widths, blocks, roomy=blocks != 2, **limit) for blocks in (1, None, 2))
        # A budget that is set is the slab, as before: four columns here.
        budget = self._step(rows, widths, 1, roomy=True, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 4), **limit)
        # The limit at the 4 GiB of the streaming budget is the way back: the slabs of that budget.  The default
        # cuts them as well where an eighth of the basis is less than its 2 GiB, as here.
        former = self._step(rows, widths, 1, roomy=True, PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=str(4 << 30))
        default = self._step(rows, widths, 1, roomy=True)
        # A smaller basis has the same limit, and one on three devices has none.
        others = [self._step(rows, shares, 1, roomy=True, **limit) for shares in ((12, 11), (16, 16, 15))]
        runs = (one, unnamed, two, budget, former, default)
        for run in runs + tuple(others):
            self._assert_ritz_pairs(run)
        for run in runs:
            np.testing.assert_allclose(run.values, two.values, rtol=0, atol=1e-11)
        taken = [[[columns for device, columns in run.sector.applied if device == index]
                  for index in range(len(run.layout.parts))] for run in runs + tuple(others)]
        # Twelve rounds of two columns of each device, named or by default, where the slabs of the streaming
        # budget made three rounds of eight; the step with two blocks keeps those.
        self.assertEqual(taken[0], [[2] * 12, [2] * 11 + [1]])
        self.assertEqual(taken[1], taken[0])
        self.assertEqual(taken[3], [[4] * 6, [4] * 5 + [3]])
        self.assertEqual(taken[4], [[8, 8, 8], [8, 8, 7]])
        self.assertEqual([run.group.slab_columns for run in runs], [2, 2, two.group.slab_columns, 4, 8, 8])
        self.assertGreater(two.group.slab_columns, 4)
        # The way back and the default made the same sums in the same order here.
        self.assertEqual(taken[5], taken[4])
        np.testing.assert_array_equal(default.values, former.values)
        np.testing.assert_array_equal(default.rotated, former.rotated)
        # The workspace is that of the narrower slabs: one in all rows and the 129 rows of a device of a round.
        self.assertEqual((one.group._work, budget.group._work, former.group._work),
                         (rows * 2 + 129 * 4, rows * 4 + 129 * 8, rows * 8 + 129 * 16))
        self.assertEqual(taken[6:], [[[2] * 6, [2] * 5 + [1]], [[6, 5, 5], [6, 5, 5], [5, 5, 5]]])

    def test_the_steps_cut_their_products_at_whole_multiples(self):
        # A basis that a host can hold has no slab of 64 columns, so a multiple of 8 stands for it, with
        # slabs of at most 5 columns: blocks of 24 and 23 columns in rounds of 4 of each device, after one of
        # 4 and 3, where the equal cut is five rounds of 9 or 10.
        rows, widths = 257, (24, 23)
        budget = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 5))
        multiple = dict(PARSEC_CUPY_RITZ_GRAM_MULTIPLE='8')

        def taken(run):
            return [[columns for device, columns in run.sector.applied if device == index]
                    for index in range(len(run.layout.parts))]

        for eager in (False, True):
            for roomy in (True, False):
                with self.subTest(eager=eager, roomy=roomy):
                    whole = self._step(rows, widths, 1, eager=eager, roomy=roomy, **budget, **multiple)
                    equal = self._step(rows, widths, 1, eager=eager, roomy=roomy, **budget,
                                       PARSEC_CUPY_RITZ_GRAM_MULTIPLE='1')
                    default = self._step(rows, widths, 1, eager=eager, roomy=roomy, **budget)
                    for run in (whole, equal, default):
                        self._assert_ritz_pairs(run)
                    self.assertEqual(taken(whole), [[4] * 6, [3] + [4] * 5])
                    self.assertEqual(taken(equal), [[5, 5, 5, 5, 4], [5, 5, 4, 5, 4]])
                    # What a run reports of its step: the cut, the widest slab and the widest product.
                    self.assertEqual([(run.group.slab_cut, run.group.slab_columns, run.group.projection_columns)
                                      for run in (whole, equal, default)],
                                     [('multiple', 4, 8), ('equal', 5, 10), ('equal', 5, 10)])
                    # No slab of this basis reaches 64 columns: the default is the equal cut, bit for bit.
                    self.assertEqual(taken(default), taken(equal))
                    for part in ('values', 'rotated'):
                        np.testing.assert_array_equal(getattr(default, part), getattr(equal, part))
                    for part in default.gram:
                        self.assertEqual([np.array_equal(mine, theirs) for mine, theirs in zip(default.gram, equal.gram)],
                                         [True, True])
                    # The whole rounds sum the projection from other pieces: the same Ritz pairs to round-off.
                    np.testing.assert_allclose(whole.values, equal.values, rtol=0, atol=1e-11)
                    np.testing.assert_allclose(np.linalg.svd(equal.rotated.T @ whole.rotated, compute_uv=False),
                                               np.ones(sum(widths)), atol=5e-9)
                    self.assertFalse(np.array_equal(whole.gram[1], equal.gram[1]))
                    # The workspace is that of the narrower slabs, H times one in all rows and one on its way,
                    # and the rows lie in the blocks of the columns wherever those of the equal cut do.
                    self.assertEqual((whole.group._work, equal.group._work), (rows * 4 + 129 * 8, rows * 5 + 129 * 10))
                    self.assertEqual(whole.group.separate_row_blocks, equal.group.separate_row_blocks)
                    if roomy:
                        self.assertEqual(whole.group.separate_row_blocks, 0)
        # A block that is a whole number of slabs gives nothing to the round that goes first: blocks of 24
        # and 20 columns in a round of 4 columns of the first device alone and five of 4 of each, in blocks
        # with room and without, and with every chunk size of the exchange.
        for roomy in (True, False):
            for chunk in ('8', str(8 * 193), str(1 << 30)):
                run = self._step(rows, (24, 20), 1, roomy=roomy, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=chunk, **budget, **multiple)
                self._assert_ritz_pairs(run)
                self.assertEqual(taken(run), [[4] * 6, [4] * 5])
                self.assertEqual((run.group.slab_cut, run.group.slab_columns, run.group.projection_columns), ('multiple', 4, 8))
                if roomy:
                    self.assertEqual(run.group.separate_row_blocks, 0)
        # After a failed audit the whole rounds are run back like the equal ones: every filtered column is
        # where it was, to the last bit, in the blocks that were given.  Two column ranges per device are
        # re-laid in whole rounds like one: blocks of 30 and 23 columns give 10 and 3 in two rounds first,
        # equal slabs of at most 5, and then five rounds of 4 each.
        for shares in ((24, 23), (24, 20)):
            for eager in (False, True):
                run = self._step(rows, shares, 1, eager=eager, unstable=True, roomy=True, **budget, **multiple)
                self.assertEqual((run.solution, run.group.passes, run.group.slab_cut), (None, 0, 'multiple'))
                for block, given, (start, stop) in zip(run.filtered.blocks, run.given, run.ranges):
                    self.assertIs(block, given)
                    np.testing.assert_array_equal(np.asarray(block), run.host[:, start:stop])
                self.assertEqual((run.group.seconds['rotate'], run.group.separate_row_blocks), (0., 0))
        layout = _interleaved(53, 2)
        ranged = self._step(rows, layout, 1, roomy=True, **budget, **multiple)
        self._assert_ritz_pairs(ranged)
        self.assertEqual((ranged.group.slab_cut, ranged.group.projection_columns), ('multiple', 8))
        self.assertEqual((ranged.layout.widths, taken(ranged)), ((30, 23), [[5, 5] + [4] * 5, [2, 1] + [4] * 5]))
        # Every layout of these tests under a multiple of 2, which the slabs of all of them reach: the Ritz
        # pairs of the equal cut to round-off, with the rounds that the plan names, on blocks with room and
        # without.  Blocks that hold no column keep the equal cut.
        cuts = []
        for size, shares in self.layouts:
            for roomy in (True, False):
                with self.subTest(rows=size, widths=shares, roomy=roomy):
                    one = self._step(size, shares, 1, roomy=roomy, PARSEC_CUPY_RITZ_GRAM_MULTIPLE='2')
                    before = self._step(size, shares, 1, roomy=roomy, PARSEC_CUPY_RITZ_GRAM_MULTIPLE='1')
                    self._assert_ritz_pairs(one)
                    np.testing.assert_allclose(one.values, before.values, rtol=0, atol=1e-11)
                    with patch.dict(os.environ):
                        os.environ.pop('PARSEC_CUPY_STREAMING_RITZ_BYTES', None)
                        width = distributed_state.slab_columns(size, sum(shares), len(shares), blocks=1)
                    rounds = distributed_state.block_rounds(shares, width, 2)
                    self.assertEqual(taken(one), [[last - first for first, last in (step[index] for step in rounds)
                                                   if last > first] for index in range(len(shares))])
                    self.assertEqual(one.group.separate_row_blocks, before.group.separate_row_blocks)
            cuts.append(one.group.slab_cut)
        self.assertEqual(cuts, ['multiple', 'equal', 'multiple', 'multiple', 'equal', 'equal'])
        # Four devices give 3 columns each to a round of 12, a multiple of 4, after one of 11 columns, and the
        # overlap of every row block is cut at multiples as well: slabs of 4 columns where an eighth of the
        # 47 columns is 6.
        four = (12, 12, 12, 11)
        budget = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 3))
        products = []
        lower = rayleigh_ritz._lower_triangle_product

        def recorded(left, right):
            products.append(rayleigh_ritz._gram_slab_width(int(left.shape[1])))
            return lower(left, right)

        with patch.object(rayleigh_ritz, '_lower_triangle_product', recorded):
            run = self._step(rows, four, 1, roomy=True, PARSEC_CUPY_RITZ_GRAM_MULTIPLE='4', **budget)
        self._assert_ritz_pairs(run)
        self.assertEqual(taken(run), [[3] * 4] * 3 + [[2, 3, 3, 3]])
        self.assertEqual((run.group.slab_cut, run.group.slab_columns, run.group.projection_columns), ('multiple', 3, 12))
        self.assertEqual(products, [4] * 4)
        # The step with two tall blocks projects every slab by itself: slabs of 4 columns and what is left of
        # a block, whichever cut is named, where they were five of 4 or 5 and four of 5 with one of 4 or 3.
        budget = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 5))
        for cut, former in (('equal', [[4, 5, 5, 5, 5], [4, 5, 4, 5, 5]]), ('full', [[5, 5, 5, 5, 4], [5, 5, 5, 5, 3]])):
            named = dict(PARSEC_CUPY_DISTRIBUTED_STATE_SLABS=cut, **budget)
            two = self._step(rows, widths, 2, PARSEC_CUPY_RITZ_GRAM_MULTIPLE='4', **named)
            before = self._step(rows, widths, 2, PARSEC_CUPY_RITZ_GRAM_MULTIPLE='1', **named)
            unnamed = self._step(rows, widths, 2, **named)
            for step in (two, before, unnamed):
                self._assert_ritz_pairs(step)
            self.assertEqual((taken(two), taken(before)), ([[4] * 6, [4] * 5 + [3]], former))
            self.assertEqual([(step.group.slab_cut, step.group.slab_columns, step.group.projection_columns)
                              for step in (two, before)], [(cut, 4, 4), (cut, 5, 5)])
            np.testing.assert_allclose(two.values, before.values, rtol=0, atol=1e-11)
            np.testing.assert_array_equal(unnamed.values, before.values)
            np.testing.assert_array_equal(unnamed.rotated, before.rotated)
            self.assertLessEqual(two.group._work, before.group._work)
        # A value that is no count of columns stops the run where the trial basis is sized, before the first
        # filter, whichever step is named.
        cp, _runtime = _stand_in()
        group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
        for bad in ('0', '-64', 'sixty-four', ''):
            for blocks in ('1', '2', '3'):
                with patch.dict(os.environ, PARSEC_CUPY_RITZ_GRAM_MULTIPLE=bad, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=blocks):
                    with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_RITZ_GRAM_MULTIPLE'):
                        group._capacities(_columns((3, 2)), [0, 4, 9])

    def test_a_workspace_that_a_pass_has_outgrown_goes_back_before_the_larger_one_is_taken(self):
        # The density step leaves the pools of a shared basis as they are, so the group itself returns the
        # one block that no later pass asks for: the workspace of the passes before, where one needs more.
        name = 'PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE'
        devices, rows = (0, 1, 2), 257

        def passes(*needs, **environment):
            """Passes that need these workspaces, slab by round: what the devices returned and what each pass held."""
            cp, runtime = _stand_in()
            with patch.object(distributed_state, 'require_cupy', lambda: (cp, None)), patch.dict(os.environ):
                os.environ.pop(name, None)
                os.environ.update(environment)
                group = distributed_state.SectorDeviceGroup(_Owner(cp, devices), devices)
                held = []
                for slab, joined in needs:
                    work = group._cut_capacity(rows, slab, joined, 1)
                    taken = group.map(lambda index, device: group._slab_workspace(cp, device, work))
                    self.assertEqual([(array.device_id, array.size) for array in taken],
                                     [(device, work) for device in devices])
                    held.append(work)
                    # A pass gives its workspace back before the next one asks.
                    del taken
                return group, runtime, held

        # A first pass, one that needs as much and one that needs less: the workspace is that of the first,
        # which the free list of the stream serves, and no device returns anything.
        group, runtime, held = passes((4, 8), (4, 8), (3, 6))
        self.assertEqual((held, runtime.returned, group._outgrown), ([257 * 4 * 2] * 3, [], set()))
        # A pass that needs more: every device returns the free blocks of the stream of its group, once, on
        # its own thread and before it takes the larger workspace, and the passes after it return nothing.
        group, runtime, held = passes((4, 8), (5, 10), (5, 10), (4, 8))
        small, large = 257 * 4 * 2, 257 * 5 * 2
        self.assertEqual(held, [small, large, large, large])
        self.assertEqual(sorted((device, stream is group._stream(device)) for device, stream, _taken in runtime.returned),
                         [(device, True) for device in devices])
        for device, _stream, taken in runtime.returned:
            self.assertEqual(taken, [small], device)
        self.assertEqual(group._outgrown, set())
        # Twice outgrown, twice returned.
        group, runtime, held = passes((4, 8), (5, 10), (6, 12))
        self.assertEqual([len([entry for entry in runtime.returned if entry[0] == device]) for device in devices], [2] * 3)
        # Where the density step empties the pools as before, the group returns nothing: there is nothing
        # of a pass before in them.
        group, runtime, held = passes((4, 8), (5, 10), (6, 12), **{name: '1'})
        self.assertEqual((held, runtime.returned, group._outgrown), ([small, large, 257 * 6 * 2], [], set()))
        # The steps with one and with two tall blocks take their workspace that way, one per device and
        # step; the step with three holds none.
        taken = []
        workspace = distributed_state.SectorDeviceGroup._slab_workspace

        def recorded(group, cp, device, work):
            taken.append((device, work))
            return workspace(group, cp, device, work)

        with patch.object(distributed_state.SectorDeviceGroup, '_slab_workspace', recorded):
            for blocks in (1, 2, 3):
                del taken[:]
                run = self._step(257, (12, 11), blocks, roomy=blocks == 1)
                self._assert_ritz_pairs(run)
                self.assertEqual(sorted(taken), [(0, run.group._work), (1, run.group._work)] if blocks < 3 else [])
                # A group that has taken one workspace has outgrown none.
                self.assertEqual(run.runtime.returned, [])
        # A value that is no switch stops the run where the trial basis is sized, before the first filter.
        cp, _runtime = _stand_in()
        group = distributed_state.SectorDeviceGroup(_Owner(cp, (0, 1)), (0, 1))
        for bad in ('keep', '2', ''):
            with patch.dict(os.environ, {name: bad}):
                with self.assertRaisesRegex(ValueError, name):
                    group._capacities(_columns((3, 2)), [0, 4, 9])

    def test_two_tall_blocks_with_the_former_exchange_switches(self):
        former = dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='0', PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0',
                      PARSEC_CUPY_ROTATE_COLUMN_COPY='0', PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 193))
        for rows, widths in self.layouts:
            for eager in (False, True):
                with self.subTest(rows=rows, widths=widths, eager=eager):
                    self._assert_ritz_pairs(self._step(rows, widths, 2, eager=eager, **former))

    def test_two_tall_blocks_and_the_slab_workspace_are_the_peak(self):
        rows, widths = 4097, (8, 8, 8, 8)
        # Slabs of one column and exchange chunks of at most 3073 elements, the rows of one column outside a
        # device.  Work is done at once here: a kernel still queued would keep its temporary arrays.
        sizes = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows), PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 1000))
        two, three = (self._step(rows, widths, blocks, eager=True, **sizes) for blocks in (2, 3))
        self._assert_ritz_pairs(two)
        self._assert_ritz_pairs(three)
        tall, chunk, work = two.group._tall, two.group._chunk, two.group._work
        self.assertEqual((tall, chunk, work), (1025 * 32, 3073, rows + 1025 * 4))
        self.assertEqual((three.group._tall, three.group._chunk, three.group._work), (tall, chunk, 0))
        # What a device holds besides the tall blocks, the workspace and the buffer are arrays of the order of
        # the Gram matrices and, on the stand-in only, copies of a rotation tile, which is no larger than a
        # slab: far less than a tall block.
        small = tall // 2
        self.assertLess(work + chunk + small, tall)
        for device, held in enumerate(widths):
            columns = rows * held
            taken = [size for owner, size in two.allocated if owner == device]
            # One row block, one workspace and one buffer for the whole step ...
            self.assertEqual([taken.count(size) for size in (tall, work, chunk)], [1, 1, 1])
            self.assertLessEqual(max(size for size in taken if size not in (tall, work, chunk)), rows)
            # ... which with the columns are all that is ever held at a time.
            self.assertGreaterEqual(two.peak[device], columns + tall + work + chunk)
            self.assertLessEqual(two.peak[device], columns + tall + work + chunk + small)
            # Three tall blocks: H times the columns and two row blocks from the pool, a buffer per exchange.
            taken = [size for owner, size in three.allocated if owner == device]
            self.assertEqual([taken.count(size) for size in (tall, chunk)], [3, 3])
            self.assertGreaterEqual(three.peak[device], columns + 2 * tall + chunk)
            # Afterwards only the columns are left.
            self.assertEqual((two.live[device], three.live[device]), (columns, columns))

    def test_a_failed_audit_leaves_the_filtered_columns_and_nothing_else(self):
        for rows, widths in self.layouts:
            for blocks in (2, 3):
                for eager in (False, True):
                    with self.subTest(rows=rows, widths=widths, blocks=blocks, eager=eager):
                        run = self._step(rows, widths, blocks, eager=eager, unstable=True,
                                         PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 3),
                                         PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 193))
                        self.assertIsNone(run.solution)
                        self.assertEqual((run.group.passes, run.filtered.ranges), (0, run.ranges))
                        # The blocks that were given, with every entry as it was filtered.
                        for block, given, (start, stop) in zip(run.filtered.blocks, run.given, run.ranges):
                            self.assertIs(block, given)
                            np.testing.assert_array_equal(np.asarray(block), run.host[:, start:stop])
                        # Row blocks, workspace and buffers are gone before the fallback gathers the basis.
                        self.assertEqual(run.live_in_handler, {index: rows * width for index, width in enumerate(widths)})
                        self.assertEqual(run.live, run.live_in_handler)

    def test_two_column_ranges_per_device_give_the_ritz_pairs_of_one(self):
        for rows, layout in _TWO_RANGES:
            states, count = layout.columns, len(layout.parts)
            for chunk in ('8', str(8 * 193), str(1 << 30)):
                for eager in (False, True):
                    exchange = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=chunk)
                    # Two tall blocks with slabs of one column, of three and as wide as the streaming budget
                    # allows; then three tall blocks.
                    for blocks, slab in ((2, 1), (2, 3), (2, None), (3, None)):
                        budget = {} if slab is None else dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * slab))
                        with self.subTest(rows=rows, layout=layout.parts, chunk=chunk, eager=eager, blocks=blocks, slab=slab):
                            two = self._step(rows, layout, blocks, eager=eager, **exchange, **budget)
                            self._assert_ritz_pairs(two)
                            # The same columns in blocks of the same widths that follow each other.
                            one = self._step(rows, layout.widths, blocks, eager=eager, **exchange, **budget)
                            np.testing.assert_array_equal(two.host, one.host)
                            np.testing.assert_allclose(two.values, one.values, rtol=0, atol=1e-11)
                            np.testing.assert_allclose(np.linalg.svd(one.rotated.T @ two.rotated, compute_uv=False),
                                                       np.ones(states), atol=5e-9)
                            # The row blocks hold the columns in their own order whatever the layout is.
                            for mine, theirs in zip(two.gram, one.gram):
                                np.testing.assert_allclose(np.tril(mine), np.tril(theirs), rtol=0,
                                                           atol=1e-12 * np.abs(theirs).max())
                            if blocks == 3:
                                self.assertEqual(sorted(two.sector.applied),
                                                 [(index, width) for index, width in enumerate(layout.widths) if width])
                                continue
                            # Every device applied H to its own columns once, a slab at a time.  A slab ends
                            # with its column range, so that its columns are neighbours in the basis, and
                            # nothing was projected above the diagonal slabs.
                            width = max(1, min(slab or states, states // (2 * count)))
                            for index, part in enumerate(layout.parts):
                                taken = [columns for device, columns in two.sector.applied if device == index]
                                self.assertEqual(taken, [size for start, stop in part for size in _cut(stop - start, width)])
                            self.assertFalse(np.triu(two.gram[1], width).any())

    def test_a_failed_audit_leaves_two_column_ranges_per_device_as_they_were_filtered(self):
        for rows, layout in _TWO_RANGES:
            # Two tall blocks and three.
            for blocks in (2, 3):
                for eager in (False, True):
                    with self.subTest(rows=rows, layout=layout.parts, blocks=blocks, eager=eager):
                        run = self._step(rows, layout, blocks, eager=eager, unstable=True,
                                         PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 3),
                                         PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 193))
                        self.assertIsNone(run.solution)
                        self.assertEqual((run.group.passes, run.filtered.layout), (0, layout))
                        for index, (block, given) in enumerate(zip(run.filtered.blocks, run.given)):
                            self.assertIs(block, given)
                            np.testing.assert_array_equal(np.asarray(block), run.host[:, _held(layout, index)])
                        self.assertEqual(run.live_in_handler,
                                         {index: rows * width for index, width in enumerate(layout.widths)})
                        self.assertEqual(run.live, run.live_in_handler)


    # ----- one tall block per device (PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=1) -----

    @staticmethod
    def _heights(rows, count):
        return [rows * (index + 1) // count - rows * index // count for index in range(count)]

    def _assert_rows_lay_in_the_column_blocks(self, run, rows, widths, roomy):
        """Where the step with one tall block kept the row block of every device of ``run``.

        A block that was given the capacity of a trial basis holds its row block itself, and by the time of
        the small solve it holds none of its columns as they were filtered.  An empty block, and one that is
        only as large as its columns unless the rows happen to fit them, gets a row block from the pool.
        """
        heights, states = self._heights(rows, len(widths)), sum(widths)
        ends = distributed_state.SectorDeviceGroup._row_block_ends(
            distributed_state.block_rounds(widths, distributed_state.slab_columns(rows, states, len(widths))),
            widths, heights, rows)
        separate = 0
        for index, (held, height) in enumerate(zip(widths, heights)):
            inside = ends[index] <= (run.capacity if roomy and held else rows * held)
            separate += not inside
            taken = [size for owner, size in run.allocated if owner == index]
            # A row block of its own is the one allocation of the step that has the size of a tall block.
            self.assertEqual(taken.count(run.capacity), int(not inside and height > 0), (index, roomy))
            if roomy and held:
                self.assertTrue(inside, index)
            if inside and held and height:
                self.assertFalse(np.array_equal(run.at_solve[index], run.host[:, _held(run.layout, index)]), index)
        self.assertEqual(run.group.separate_row_blocks, separate)

    def test_one_tall_block_gives_the_ritz_pairs_of_two(self):
        for rows, widths in self.layouts:
            states, count = sum(widths), len(widths)
            # One column per chunk of an exchange, a few, and everything at once; both stream orders.
            for chunk in ('8', str(8 * 193), str(1 << 30)):
                for eager in (False, True):
                    exchange = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=chunk)
                    # Slabs of one column, of three, and as wide as the streaming budget allows.
                    for slab in (1, 3, None):
                        budget = {} if slab is None else dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * slab))
                        two = self._step(rows, widths, 2, eager=eager, **exchange, **budget)
                        self._assert_ritz_pairs(two)
                        # Column blocks with the capacity of a trial basis, into which the rows are laid, and
                        # blocks that are only as large as their columns.
                        for roomy in (True, False):
                            with self.subTest(rows=rows, widths=widths, chunk=chunk, eager=eager, slab=slab, roomy=roomy):
                                one = self._step(rows, widths, 1, eager=eager, roomy=roomy, **exchange, **budget)
                                self._assert_ritz_pairs(one)
                                np.testing.assert_allclose(one.values, two.values, rtol=0, atol=1e-11)
                                np.testing.assert_allclose(np.linalg.svd(two.rotated.T @ one.rotated, compute_uv=False),
                                                           np.ones(states), atol=5e-9)
                                # Every device applied H to its own columns once, in the slabs of the rounds:
                                # as many as the widest block has slabs of the budget, and as equal as they can be.
                                width = max(1, min(slab or states, states // (2 * count)))
                                with patch.dict(os.environ, budget or {'PARSEC_CUPY_STREAMING_RITZ_BYTES': str(1 << 62)}):
                                    self.assertEqual(distributed_state.slab_columns(rows, states, count), width)
                                    rounds = distributed_state.block_rounds(widths, width)
                                    for index in range(count):
                                        taken = [columns for device, columns in one.sector.applied if device == index]
                                        self.assertEqual(taken, [last - first for first, last in
                                                                 (step[index] for step in rounds) if last > first])
                                    self._assert_rows_lay_in_the_column_blocks(one, rows, widths, roomy)

    def test_one_tall_block_with_the_former_exchange_switches(self):
        former = dict(PARSEC_CUPY_EXCHANGE_CONCURRENT='0', PARSEC_CUPY_EXCHANGE_COLUMN_COPIES='0',
                      PARSEC_CUPY_ROTATE_COLUMN_COPY='0', PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 193))
        for rows, widths in self.layouts:
            for eager in (False, True):
                for roomy in (True, False):
                    with self.subTest(rows=rows, widths=widths, eager=eager, roomy=roomy):
                        self._assert_ritz_pairs(self._step(rows, widths, 1, eager=eager, roomy=roomy, **former))

    def test_four_devices_get_the_same_bits_whichever_way_they_copy_their_rows(self):
        # One, two and three tall blocks, with two or three columns in a chunk of an exchange so that every way
        # to the rows takes several, in both stream orders; column blocks that follow each other, fewer rows
        # than devices, and two column ranges per device.  ``turns``: whether all four devices send a chunk.
        # With fewer rows than devices one of them holds no column here, and with two ranges the first device
        # holds 18 columns and the others 12 or 11, so that it still sends when they have finished.
        chunk = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 500))
        for rows, widths, turns in ((257, (6, 6, 6, 5), True), (3, (1, 1, 0, 1), False), (*_TWO_RANGES[0], True)):
            for blocks in (1, 2, 3):
                for eager in (False, True):
                    with self.subTest(rows=rows, widths=getattr(widths, 'parts', widths), blocks=blocks, eager=eager):
                        paired = self._step(rows, widths, blocks, eager=eager, **chunk)
                        together = self._step(rows, widths, blocks, eager=eager, PARSEC_CUPY_EXCHANGE_PAIRS='0', **chunk)
                        self.assertEqual((paired.group.to_rows_copies, together.group.to_rows_copies), ('pairs', 'together'))
                        self._assert_ritz_pairs(paired)
                        # The Gram pair that the small solve was given, the Ritz values and the rotated basis.
                        for one, other in zip(paired.gram + (paired.values, paired.rotated),
                                              together.gram + (together.values, together.rotated)):
                            np.testing.assert_array_equal(one, other)
                        # The same pieces crossed between the same devices.
                        self.assertEqual(sorted((device, source) for device, source, _stream in paired.copies),
                                         sorted((device, source) for device, source, _stream in together.copies))
                        # The pairs kept to the streams that the devices compute on.  The way back took a
                        # stream per source as both ways do without pairs, and so did the chunks that not all
                        # four devices send.
                        own = {paired.group._stream(device) for device in range(4)}
                        on_own = sum(stream in own for _device, _source, stream in paired.copies)
                        self.assertTrue(0 < on_own < len(paired.copies) if turns else on_own == 0)
                        own = {together.group._stream(device) for device in range(4)}
                        self.assertFalse(any(stream in own for _device, _source, stream in together.copies))

    def test_the_slab_limit_of_two_devices_and_the_turns_of_four_leave_each_other_alone(self):
        # The two switches of a shared basis that came together, in the step with one tall block.  A
        # sector on four devices takes the turns on the way to its rows with the slabs of its own round
        # budget: the limit of two devices is not read there, neither at one column nor at the former 4
        # GiB.  A sector on two devices follows that limit and is one pair, which has no turns to take
        # whatever their switch says; one on three devices reads neither.
        rows, chunk = 257, dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 500))
        limits = ({}, dict(PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=str(8 * rows)),
                  dict(PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=str(4 << 30)))
        orders = ({}, dict(PARSEC_CUPY_EXCHANGE_PAIRS='1'), dict(PARSEC_CUPY_EXCHANGE_PAIRS='0'))

        def taken(run):
            return [[columns for device, columns in run.sector.applied if device == index]
                    for index in range(len(run.layout.parts))]

        def same(run, other):
            self.assertEqual((taken(run), run.group.slab_columns, run.group._work),
                             (taken(other), other.group.slab_columns, other.group._work))
            for one, two in zip(run.gram + (run.values, run.rotated), other.gram + (other.values, other.rotated)):
                np.testing.assert_array_equal(one, two)

        for widths, slabs in (((6, 6, 6, 5), (2, 2, 2)), ((12, 11), (4, 1, 4)), ((8, 8, 7), (3, 3, 3))):
            devices = len(widths)
            first = None
            for limit, slab in zip(limits, slabs):
                plain = None
                for order in orders:
                    with self.subTest(widths=widths, limit=limit, order=order):
                        run = self._step(rows, widths, 1, roomy=True, **chunk, **limit, **order)
                        self._assert_ritz_pairs(run)
                        turns = devices == 4 and order.get('PARSEC_CUPY_EXCHANGE_PAIRS') != '0'
                        self.assertEqual((run.group.to_rows_copies, run.group.slab_columns),
                                         ('pairs' if turns else 'together', slab))
                        own = {run.group._stream(device) for device in range(devices)}
                        self.assertEqual(any(stream in own for _device, _source, stream in run.copies), turns)
                        # The order of the copies moves no bit, on any number of devices.
                        plain = plain or run
                        same(run, plain)
                        # Nor does the limit of two devices where there are three or four.
                        first = first or run
                        if devices != 2:
                            same(run, first)
        # On two devices the default is the way back where an eighth of the basis is below both limits.
        default, narrow, former = (self._step(rows, (12, 11), 1, roomy=True, **chunk, **limit) for limit in limits)
        same(default, former)
        self.assertEqual((taken(narrow)[0], taken(former)[0]), ([1] * 12, [4, 4, 4]))

    def test_whole_rounds_of_four_devices_take_the_turns_and_a_round_of_one_sender_does_not(self):
        # The rounds of whole multiples and the turns of four devices came together and act on the same
        # way to the rows.  In a whole round every device gives the same slab, so every chunk of it has four
        # senders and takes the three turns; what the blocks hold beyond those rounds goes first, and where
        # only one block holds such columns their chunk has one sender, which the three others receive from
        # at once as without turns.  A multiple of 8 stands for 64, with slabs of at most 3 columns: blocks
        # of 9, 8, 8 and 8 columns give one column of the first device and then four rounds of 2 of each,
        # and four blocks of 9 give a round of 1 of each first.  One chunk per slab, and one per column.
        rows = 257
        cut = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 3), PARSEC_CUPY_RITZ_GRAM_MULTIPLE='8')

        def taken(run):
            return [[columns for device, columns in run.sector.applied if device == index]
                    for index in range(len(run.layout.parts))]

        for widths, first, senders in (((9, 8, 8, 8), [1], 1), ((9, 9, 9, 9), [1] * 4, 4)):
            for chunk, per_slab in ((str(1 << 30), 1), (str(8 * 193), 2)):
                for eager in (False, True):
                    with self.subTest(widths=widths, chunk=chunk, eager=eager):
                        named = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=chunk, **cut)
                        paired = self._step(rows, widths, 1, eager=eager, roomy=True, **named)
                        together = self._step(rows, widths, 1, eager=eager, roomy=True, PARSEC_CUPY_EXCHANGE_PAIRS='0', **named)
                        self._assert_ritz_pairs(paired)
                        self.assertEqual(taken(paired), [first[:1] + [2] * 4] + [first[1:2] + [2] * 4] * 3)
                        self.assertEqual([(run.group.slab_cut, run.group.slab_columns, run.group.projection_columns,
                                           run.group.to_rows_copies, run.group.separate_row_blocks)
                                          for run in (paired, together)],
                                         [('multiple', 2, 8, 'pairs', 0), ('multiple', 2, 8, 'together', 0)])
                        # The order of the copies moves no bit of the Gram pair, the Ritz values or the basis.
                        for one, other in zip(paired.gram + (paired.values, paired.rotated),
                                              together.gram + (together.values, together.rotated)):
                            np.testing.assert_array_equal(one, other)
                        self.assertEqual(sorted((device, source) for device, source, _stream in paired.copies),
                                         sorted((device, source) for device, source, _stream in together.copies))
                        # The copies that kept to the stream their device computes on are those of the turns:
                        # twelve per chunk of a slab and of H times it, in the four whole rounds, and in the
                        # round that goes first only where all four devices send.
                        own = {paired.group._stream(device) for device in range(4)}
                        in_turns = sum(stream in own for _device, _source, stream in paired.copies)
                        self.assertEqual(in_turns, 12 * 2 * (4 * per_slab + (senders == 4)))
                        own = {together.group._stream(device) for device in range(4)}
                        self.assertFalse(any(stream in own for _device, _source, stream in together.copies))
                        # The equal cut of the same blocks, three rounds of 11 or 12 columns, takes the turns
                        # in every round and gives the same Ritz pairs to round-off.
                        equal = self._step(rows, widths, 1, eager=eager, roomy=True,
                                           **dict(named, PARSEC_CUPY_RITZ_GRAM_MULTIPLE='1'))
                        self.assertEqual((equal.group.slab_cut, equal.group.slab_columns, equal.group.to_rows_copies),
                                         ('equal', 3, 'pairs'))
                        np.testing.assert_allclose(paired.values, equal.values, rtol=0, atol=1e-11)

    def test_a_slab_one_column_wider_than_a_chunk_takes_the_turns_twice_and_moves_no_bit(self):
        # The slab of a whole round can be a column or a few wider than a chunk of the exchange: 32 columns
        # against 31 for 39,368 electrons on four devices with the chunk of 1 GiB.  It then travels as a chunk
        # and a small one, each from all four devices and so each in the three turns.  Here a multiple of 16
        # stands for 64, with slabs of at most 5 columns: blocks of 13, 12, 12 and 12 columns give one column
        # of the first device and then three rounds of 4 of each.  With a chunk of 3 columns every slab of
        # those rounds is two chunks, of 3 columns and of 1; a chunk of 4 holds it.  The chunk moves no bit,
        # with the turns or without: it only halves the chunk steps.
        rows, widths = 257, (13, 12, 12, 12)
        cut = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 5), PARSEC_CUPY_RITZ_GRAM_MULTIPLE='16')
        # The rows of a column outside its device: 193, and 192 for the last device.
        away = rows - rows // 4

        def taken(run):
            return [[columns for device, columns in run.sector.applied if device == index] for index in range(4)]

        for eager in (False, True):
            runs = []
            for columns, per_slab in ((3, 2), (4, 1)):
                with self.subTest(eager=eager, columns=columns):
                    named = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * away * columns), **cut)
                    self.assertEqual([len(distributed_state.exchange_chunks(4, away - last, away * columns))
                                      for last in (0, 1)], [per_slab] * 2)
                    paired = self._step(rows, widths, 1, eager=eager, roomy=True, **named)
                    together = self._step(rows, widths, 1, eager=eager, roomy=True, PARSEC_CUPY_EXCHANGE_PAIRS='0', **named)
                    self._assert_ritz_pairs(paired)
                    self.assertEqual(taken(paired), [[1] + [4] * 3] + [[4] * 3] * 3)
                    self.assertEqual([(run.group.slab_cut, run.group.slab_columns, run.group.projection_columns,
                                       run.group.to_rows_copies, run.group.separate_row_blocks)
                                      for run in (paired, together)],
                                     [('multiple', 4, 16, 'pairs', 0), ('multiple', 4, 16, 'together', 0)])
                    # Twelve copies in turns per chunk of a slab and of H times it, in the three whole rounds;
                    # the column that goes first has one sender and is received at once.
                    own = {paired.group._stream(device) for device in range(4)}
                    self.assertEqual(sum(stream in own for _device, _source, stream in paired.copies),
                                     12 * 2 * 3 * per_slab)
                    own = {together.group._stream(device) for device in range(4)}
                    self.assertFalse(any(stream in own for _device, _source, stream in together.copies))
                    runs.extend((paired, together))
            # The Gram pair that the small solve was given, the Ritz values and the rotated basis: the same
            # bits with either chunk and either order of the copies.
            self.assertEqual(len(runs), 4)
            for run in runs[1:]:
                for one, other in zip(runs[0].gram + (runs[0].values, runs[0].rotated),
                                      run.gram + (run.values, run.rotated)):
                    np.testing.assert_array_equal(one, other)

    def test_one_tall_block_and_the_slab_workspace_are_the_peak(self):
        rows, widths = 4097, (8, 8, 8, 8)
        # Slabs of one column and exchange chunks of at most 3073 elements, the rows of one column outside a
        # device.  Work is done at once here: a kernel still queued would keep its temporary arrays.
        sizes = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows), PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 1000))
        one, two = (self._step(rows, widths, blocks, eager=True, roomy=True, **sizes) for blocks in (1, 2))
        self._assert_ritz_pairs(one)
        self._assert_ritz_pairs(two)
        tall, chunk, work = two.group._tall, two.group._chunk, two.group._work
        self.assertEqual((tall, chunk, work), (1025 * 32, 3073, rows + 1025 * 4))
        # The same workspace and buffer, and a tall block with the room of one more row of all devices per
        # row of the tallest row block: 4100 elements, where a column has 4097.
        room = 4 * 1025
        self.assertEqual((one.group._tall, one.group._chunk, one.group._work), (tall + room, chunk, work))
        self.assertEqual((one.capacity, two.capacity, one.group.separate_row_blocks), (tall + room, tall, 0))
        # As with two tall blocks, what a device holds besides is far less than a tall block.
        small = tall // 2
        self.assertLess(work + chunk + small + room, tall)
        for device in range(len(widths)):
            taken = [size for owner, size in one.allocated if owner == device]
            # One workspace and one buffer for the whole step, and nothing of the size of a tall block ...
            self.assertEqual([taken.count(size) for size in (work, chunk)], [1, 1])
            self.assertLessEqual(max(size for size in taken if size not in (work, chunk)), rows)
            # ... so that the block of the columns, the workspace and the buffer are all that is ever held.
            self.assertGreaterEqual(one.peak[device], one.capacity + work + chunk)
            self.assertLessEqual(one.peak[device], one.capacity + work + chunk + small)
            # With two tall blocks a row block from the pool comes on top.
            self.assertEqual([size for owner, size in two.allocated if owner == device].count(tall), 1)
            self.assertGreaterEqual(two.peak[device], 2 * tall + work + chunk)
            self.assertLess(one.peak[device] + tall - small - room, two.peak[device])
            # Afterwards only the block of the columns is left.
            self.assertEqual((one.live[device], two.live[device]), (one.capacity, tall))

    def test_the_workspace_and_the_rotation_tile_follow_the_slabs_that_were_cut(self):
        # Blocks of 40 and 37 columns in 257 rows, of which a device holds 128 or 129.  Half of an even share
        # of the columns is 19, which is the budget unless one is named.
        rows, widths, states, tallest = 257, (40, 37), 77, 129
        rotate = distributed_state._rotate_in_place
        cases = (
            # tall blocks, cut, budget -> slabs of each device; the widest slab; the most columns of a round
            # One tall block: the wider block takes three rounds, of 14 columns at most where the budget
            # allows 19 ...
            (1, None, None, [[14, 13, 13], [13, 12, 12]], 14, 27),
            # ... and four of the 10 columns of a budget that it is a multiple of.
            (1, None, 10, [[10] * 4, [10, 9, 9, 9]], 10, 20),
            # Two tall blocks: three equal slabs of at most 14 columns, or two of the budget and what is left.
            (2, None, 15, [[13, 13, 14], [12, 12, 13]], 14, 27),
            (2, 'full', 15, [[15, 15, 10], [15, 15, 7]], 15, 30),
        )
        for blocks, cut, budget, slabs, widest, joined in cases:
            named = {} if budget is None else dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * budget))
            if cut is not None:
                named['PARSEC_CUPY_DISTRIBUTED_STATE_SLABS'] = cut
            tiles = []

            def recorded(block, coefficients, tile):
                tiles.append((block.shape[0], tile.shape))
                return rotate(block, coefficients, tile)

            with self.subTest(blocks=blocks, cut=cut, budget=budget):
                with patch.object(distributed_state, '_rotate_in_place', recorded):
                    run = self._step(rows, widths, blocks, roomy=blocks == 1, **named)
                self._assert_ritz_pairs(run)
                self.assertEqual([[columns for device, columns in run.sector.applied if device == index]
                                  for index in range(2)], slabs)
                # H times the widest slab in all rows and, behind it, the rows of a device of a whole round.
                # With one tall block the slab itself passes through the second part in all rows.
                arriving = tallest * joined
                self.assertEqual(run.group._work,
                                 rows * widest + (max(arriving, rows * widest) if blocks == 1 else arriving))
                # The tile lies where H times a slab was: as many rows of all columns as that holds.
                self.assertEqual(sorted(tiles), [(height, (rows * widest // states, states)) for height in (128, 129)])
                # The budget allows a larger workspace, and so a taller tile, unless the slabs are as wide.
                with patch.dict(os.environ, named or {'PARSEC_CUPY_STREAMING_RITZ_BYTES': str(1 << 62)}):
                    allowed = distributed_state.slab_columns(rows, states, 2)
                    by_budget = distributed_state.slab_workspace(rows, states, 2, blocks)
                self.assertEqual(allowed, budget or 19)
                if widest < allowed:
                    self.assertLess(run.group._work, by_budget)
                    self.assertLess(rows * widest // states, rows * allowed // states)
                else:
                    self.assertEqual(run.group._work, by_budget)

    def test_a_failed_audit_with_one_tall_block_puts_the_filtered_columns_back(self):
        for rows, widths in self.layouts:
            for roomy in (True, False):
                for eager in (False, True):
                    with self.subTest(rows=rows, widths=widths, roomy=roomy, eager=eager):
                        run = self._step(rows, widths, 1, eager=eager, unstable=True, roomy=roomy,
                                         PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 3),
                                         PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 193))
                        self.assertIsNone(run.solution)
                        self.assertEqual((run.group.passes, run.filtered.ranges), (0, run.ranges))
                        # The audit failed with the columns re-laid as rows inside their blocks ...
                        with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 3)):
                            self._assert_rows_lay_in_the_column_blocks(run, rows, widths, roomy)
                        # ... and the error carries the blocks that were given, with every entry as it was filtered.
                        for block, given, (start, stop) in zip(run.filtered.blocks, run.given, run.ranges):
                            self.assertIs(block, given)
                            np.testing.assert_array_equal(np.asarray(block), run.host[:, start:stop])
                        # The way back was taken without a rotation ...
                        self.assertGreater(run.group.seconds['to_columns'], 0.)
                        self.assertEqual(run.group.seconds['rotate'], 0.)
                        # ... and row blocks from the pool, workspace and buffers are gone before the fallback
                        # gathers the basis: only the blocks of the columns are left.
                        left = {index: run.capacity if roomy and width else rows * width for index, width in enumerate(widths)}
                        self.assertEqual(run.live_in_handler, left)
                        self.assertEqual(run.live, left)
                        # Putting the columns back took no memory beyond that of the step itself: the block, a
                        # row block from the pool where there is one, workspace, buffer and the Gram arrays.
                        # Work that is still queued keeps its temporary arrays, so this is read where none is.
                        work, chunk, gram = run.group._work, run.group._chunk, 4 * sum(widths) ** 2
                        for index, held in left.items() if eager else ():
                            separate = [size for owner, size in run.allocated if owner == index].count(run.capacity)
                            self.assertLessEqual(run.peak[index], held + separate * run.capacity + work + chunk + gram)

    def test_one_tall_block_with_two_column_ranges_per_device_gives_the_ritz_pairs_of_two(self):
        for rows, layout in _TWO_RANGES:
            states, count = layout.columns, len(layout.parts)
            for chunk in ('8', str(8 * 193), str(1 << 30)):
                for eager in (False, True):
                    exchange = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=chunk)
                    for slab in (1, 3, None):
                        budget = {} if slab is None else dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * slab))
                        with self.subTest(rows=rows, layout=layout.parts, chunk=chunk, eager=eager, slab=slab):
                            one = self._step(rows, layout, 1, eager=eager, roomy=True, **exchange, **budget)
                            self._assert_ritz_pairs(one)
                            two = self._step(rows, layout, 2, eager=eager, **exchange, **budget)
                            np.testing.assert_array_equal(one.host, two.host)
                            np.testing.assert_allclose(one.values, two.values, rtol=0, atol=1e-11)
                            np.testing.assert_allclose(np.linalg.svd(two.rotated.T @ one.rotated, compute_uv=False),
                                                       np.ones(states), atol=5e-9)
                            # The small solve is given the Gram matrices in the order of the basis, whatever
                            # the order in which the row blocks of the step hold the columns.
                            for mine, theirs in zip(one.gram, two.gram):
                                np.testing.assert_allclose(np.tril(mine), np.tril(theirs), rtol=0,
                                                           atol=1e-12 * np.abs(theirs).max())
                            # A slab of a round is the next columns of the block, whichever range they lie in.
                            width = max(1, min(slab or states, states // (2 * count)))
                            rounds = distributed_state.block_rounds(layout.widths, width)
                            for index in range(count):
                                taken = [columns for device, columns in one.sector.applied if device == index]
                                self.assertEqual(taken, [last - first for first, last in
                                                         (step[index] for step in rounds) if last > first])
                            # Every block that holds columns holds its row block too.
                            self.assertEqual(one.group.separate_row_blocks,
                                             sum(not held and height > 0
                                                 for held, height in zip(layout.widths, self._heights(rows, count))))

    def test_a_failed_audit_with_one_tall_block_puts_two_column_ranges_per_device_back(self):
        for rows, layout in _TWO_RANGES:
            for eager in (False, True):
                with self.subTest(rows=rows, layout=layout.parts, eager=eager):
                    run = self._step(rows, layout, 1, eager=eager, unstable=True, roomy=True,
                                     PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 3),
                                     PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 193))
                    self.assertIsNone(run.solution)
                    self.assertEqual((run.group.passes, run.filtered.layout), (0, layout))
                    for index, (block, given) in enumerate(zip(run.filtered.blocks, run.given)):
                        self.assertIs(block, given)
                        np.testing.assert_array_equal(np.asarray(block), run.host[:, _held(layout, index)])
                    left = {index: run.capacity if width else 0 for index, width in enumerate(layout.widths)}
                    self.assertEqual((run.live_in_handler, run.live), (left, left))

    def test_the_small_solve_on_the_device_is_given_the_pair_and_rotates_the_rows_as_the_host_solve(self):
        # The default of a run, which the other tests of the step leave to the host: the owner sums the parts
        # of the devices, puts the pair of the step with one tall block into the order of the basis and the
        # coefficients into that of its row blocks, and the other devices copy the coefficients from it.
        # The solve itself is stood in for by the host solve (see _step).
        device = dict(PARSEC_CUPY_RITZ_DENSE_BACKEND='device', PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 193))
        host = dict(device, PARSEC_CUPY_RITZ_DENSE_BACKEND='host')
        for rows, widths in self.layouts + _TWO_RANGES:
            layout = _layout(widths)
            for eager in (False, True):
                # One tall block with slabs of one column, of three, and as wide as the streaming budget
                # allows; then the steps with two and with three, whose owner sums and hands out the same way.
                for blocks, slab in ((1, 1), (1, 3), (1, None), (2, None), (3, None)):
                    budget = {} if slab is None else dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * slab))
                    with self.subTest(rows=rows, layout=layout.parts, eager=eager, blocks=blocks, slab=slab):
                        there = self._step(rows, widths, blocks, eager=eager, roomy=blocks == 1, **device, **budget)
                        self._assert_ritz_pairs(there)
                        here = self._step(rows, widths, blocks, eager=eager, roomy=blocks == 1, **host, **budget)
                        # The solve ran on the owner, on the pair that the host solve is given: the parts of
                        # the devices summed in their order, entry by entry.
                        self.assertEqual((there.solved_on, here.solved_on), (0, None))
                        for mine, theirs in zip(there.gram, here.gram):
                            np.testing.assert_array_equal(np.tril(mine), np.tril(theirs))
                        np.testing.assert_allclose(there.values, here.values, rtol=0, atol=1e-11)
                        np.testing.assert_allclose(np.linalg.svd(here.rotated.T @ there.rotated, compute_uv=False),
                                                   np.ones(layout.columns), atol=5e-9)
                # A failed audit of that solve leaves the step with one tall block as one of the host solve
                # does: every filtered column back in its block, and only the blocks once the error is
                # dealt with.  While it is raised, its traceback still refers to the pair on the owner.
                with self.subTest(rows=rows, layout=layout.parts, eager=eager, unstable=True):
                    run = self._step(rows, widths, 1, eager=eager, unstable=True, roomy=True, **device,
                                     PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * rows * 3))
                    self.assertEqual((run.solved_on, run.solution, run.group.passes), (0, None, 0))
                    for index, (block, given) in enumerate(zip(run.filtered.blocks, run.given)):
                        self.assertIs(block, given)
                        np.testing.assert_array_equal(np.asarray(block), run.host[:, _held(layout, index)])
                    left = [run.capacity if width else 0 for width in layout.widths]
                    self.assertEqual(run.live, dict(enumerate(left)))
                    beyond = [run.live_in_handler[index] - held for index, held in enumerate(left)]
                    self.assertEqual(beyond[1:], [0] * (len(left) - 1))
                    self.assertLessEqual(beyond[0], 2 * layout.columns ** 2)
                    self.assertGreaterEqual(beyond[0], 0)

    def test_another_device_estimates_the_condition_number_beside_the_small_solve_on_the_owner(self):
        # The small solve on the owner is given a way to have the condition number of the overlap estimated
        # meanwhile: the first other device of the sector that does not hold the Hartree objects copies the
        # mirrored overlap from the owner, on its own thread and stream, and estimates there.  The solve itself
        # is stood in for by the host solve (see _step), so the step must give the same bits with and without.
        device = dict(PARSEC_CUPY_RITZ_DENSE_BACKEND='device', PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 193))

        def from_owner(run):
            """Copies that the second device took from the owner on its stream of the group."""
            return sum(1 for target, source, stream in run.copies
                       if (target, source) == (1, 0) and stream is run.group._stream(1))

        # Every layout of the step and two with two column ranges per device; all have several devices.
        for rows, widths in self.layouts + _TWO_RANGES[:2]:
            layout = _layout(widths)
            count, states = len(layout.parts), layout.columns
            for blocks, eager in ((1, False), (1, True), (2, False), (3, False)):
                with self.subTest(rows=rows, layout=layout.parts, eager=eager, blocks=blocks):
                    common = dict(device, eager=eager, roomy=blocks == 1)
                    off = self._step(rows, widths, blocks, **common, PARSEC_CUPY_RITZ_CONDITION_HELPER='off')
                    self._assert_ritz_pairs(off)
                    self.assertEqual((off.estimates, off.condition), ([], None))
                    self.assertEqual((off.group.condition_device, off.group.condition_seconds), (0, 0.))
                    beside = self._step(rows, widths, blocks, **common)
                    self._assert_ritz_pairs(beside)
                    # One estimate, on the thread of the helper, of a copy that the helper holds.
                    self.assertEqual(beside.estimates, [(1, 'orbital-shard-1_0', states * states)])
                    self.assertEqual(beside.group.condition_device, 1)
                    self.assertGreater(beside.group.condition_seconds, 0.)
                    mirrored = np.tril(beside.gram[0]) + np.tril(beside.gram[0], -1).T
                    self.assertEqual(beside.condition, float(np.linalg.cond(mirrored)))
                    # The copy came from the owner on the stream of the helper in this group, and is the
                    # only array that the helper takes for it.
                    self.assertEqual(len(beside.copies) - len(off.copies), 1)
                    self.assertEqual(from_owner(beside) - from_owner(off), 1)
                    taken = sorted(beside.allocated)
                    for made in off.allocated:
                        taken.remove(made)
                    self.assertEqual(taken, [(0, states * states), (1, states * states)])
                    # Nothing of the step depends on where the number was estimated.
                    np.testing.assert_array_equal(beside.values, off.values)
                    np.testing.assert_array_equal(beside.rotated, off.rotated)
            # The device that holds the Hartree objects is passed over.  A sector that has no other keeps the
            # estimate on its owner, unless the setting says that this device may be asked too.
            with self.subTest(rows=rows, layout=layout.parts, hartree_device=1):
                passed = self._step(rows, widths, 1, roomy=True, hartree_device=1, **device)
                self.assertEqual(passed.group.condition_device, 2 if count > 2 else 0)
                self.assertEqual([estimate[0] for estimate in passed.estimates], [2] if count > 2 else [])
                asked = self._step(rows, widths, 1, roomy=True, hartree_device=1, **device,
                                   PARSEC_CUPY_RITZ_CONDITION_HELPER='any')
                self.assertEqual((asked.group.condition_device, [estimate[0] for estimate in asked.estimates]),
                                 (2, [2]) if count > 2 else (1, [1]))
                np.testing.assert_array_equal(asked.rotated, passed.rotated)
            # A solve that fails its audit has waited for the helper, which holds nothing afterwards.
            with self.subTest(rows=rows, layout=layout.parts, unstable=True):
                run = self._step(rows, widths, 1, unstable=True, roomy=True, **device)
                self.assertEqual((run.solution, run.group.passes, len(run.estimates)), (None, 0, 1))
                self.assertIsNotNone(run.condition)
                left = [run.capacity if width else 0 for width in layout.widths]
                self.assertEqual(run.live, dict(enumerate(left)))
                self.assertEqual([run.live_in_handler[index] - held for index, held in enumerate(left)][1:],
                                 [0] * (count - 1))


class LayoutStandInTests(unittest.TestCase):
    """A basis with two column ranges per device: how it is made, viewed, moved, weighted, filtered and solved."""

    rows = 257
    # The filter interval of these tests.  The reference lies below it, as in a first solve: sigma then
    # changes from step to step.  A later pass has its reference on the lower bound, where sigma stays -1.
    lower, upper, below = 1.2, 4.6, .7

    def _sector(self, devices, rows=None, eager=False):
        """A sector of ``devices`` on a stand-in of its own: tridiagonal ``H`` with a local potential."""
        rows = self.rows if rows is None else rows
        cp, runtime = _stand_in(eager)
        static = sp.diags((-np.ones(rows - 1), 2.2 * np.ones(rows), -np.ones(rows - 1)), (-1, 0, 1)).tocsr()
        potential = np.random.default_rng(89).uniform(-.1, .1, rows)
        sector = _FilterSector(cp, runtime, devices, static, potential)
        return SimpleNamespace(cp=cp, runtime=runtime, sector=sector, static=static, potential=potential,
                               group=distributed_state.SectorDeviceGroup(sector, devices))

    def _on(self, made, **environment):
        """The stand-in of ``made`` in place of CuPy with its first device current; the host solve with its defaults."""
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(distributed_state, 'require_cupy', lambda: (made.cp, None)))
        stack.enter_context(patch.object(rayleigh_ritz, 'require_cupy', lambda: (made.cp, None)))
        stack.enter_context(patch.dict(os.environ))
        for name in RitzStandInTests.unset + ('PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS', 'PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT',
                                               'PARSEC_CUPY_DEVICE_RANDOM'):
            os.environ.pop(name, None)
        os.environ.update(RitzStandInTests.settings, **environment)
        stack.enter_context(made.cp.cuda.Device(0))
        return stack

    @staticmethod
    def _upload(made, host):
        full = made.cp.empty(host.shape, order='F')
        np.asarray(full)[...] = host
        return full

    def _idle(self, made):
        self.assertFalse(any(stream.pending for stream in made.runtime.streams))

    def test_blocks_are_filled_from_and_gathered_to_the_columns_of_the_basis(self):
        for rows, layout in _TWO_RANGES:
            with self.subTest(rows=rows, layout=layout.parts):
                made = self._sector(tuple(range(len(layout.parts))), rows)
                host = np.random.default_rng(59).normal(size=(rows, layout.columns))
                with self._on(made):
                    basis = made.group.scatter(self._upload(made, host), layout)
                    self.assertEqual((basis.shape, basis.layout), ((rows, layout.columns), layout))
                    for index, block in enumerate(basis.blocks):
                        self.assertEqual((block.device.id, block.flags.f_contiguous), (index, True))
                        np.testing.assert_array_equal(np.asarray(block), host[:, _held(layout, index)])
                    np.testing.assert_array_equal(np.asarray(made.group.gather(basis)), host)
                    # No device holds one range of columns, and only leading columns can be selected.
                    with self.assertRaises(ValueError):
                        basis.ranges
                    with self.assertRaises(IndexError):
                        basis[:, 1:3]
                    for count in sorted({0, 1, layout.columns // 3, layout.columns // 2, layout.columns - 1, layout.columns}):
                        leading = basis[:, :count]
                        self.assertEqual((leading.shape, leading.layout), ((rows, count), layout.leading(count)))
                        for index, (view, block) in enumerate(zip(leading.blocks, basis.blocks)):
                            # The leading columns of the block, where the block has them.
                            self.assertEqual(view.shape, (rows, leading.layout.widths[index]))
                            if view.size:
                                self.assertEqual(view.data.ptr, block.data.ptr)
                            np.testing.assert_array_equal(np.asarray(view), host[:, _held(leading.layout, index)])
                        np.testing.assert_array_equal(np.asarray(made.group.gather(leading)), host[:, :count])
                self._idle(made)

    def test_columns_move_from_one_layout_to_another(self):
        rows, layout = _TWO_RANGES[0]
        made = self._sector((0, 1, 2, 3))
        host = np.random.default_rng(61).normal(size=(rows, 53))
        with self._on(made):
            group = made.group
            ranges = group.column_ranges(uniform_filter_blocks(53, 6, 5))
            basis = group.scatter(self._upload(made, host), ranges)
            self.assertEqual(basis.ranges, ranges)
            moved = group.repartition(basis, layout)
            self.assertEqual(moved.layout, layout)
            for index, block in enumerate(moved.blocks):
                self.assertEqual((block.device.id, block.flags.f_contiguous), (index, True))
                np.testing.assert_array_equal(np.asarray(block), host[:, _held(layout, index)])
            self.assertIs(group.repartition(moved, layout), moved)
            # Back to one range per device, and from leading columns in two ranges to other ranges.
            back = group.repartition(moved, ranges)
            self.assertEqual(back.ranges, ranges)
            np.testing.assert_array_equal(np.asarray(group.gather(back)), host)
            fewer = group.column_ranges(subspace_filter_blocks(31, 6, 9, 3))
            trimmed = group.repartition(moved[:, :31], fewer)
            self.assertEqual(trimmed.ranges, fewer)
            np.testing.assert_array_equal(np.asarray(group.gather(trimmed)), host[:, :31])
            again = group.repartition(trimmed, layout.leading(31))
            np.testing.assert_array_equal(np.asarray(group.gather(again)), host[:, :31])
            with self.assertRaisesRegex(ValueError, 'cover the basis'):
                group.repartition(moved, layout.leading(47))
        self.assertGreater(made.group.seconds['repartition'], 0.)
        self._idle(made)

    def test_random_trial_basis_is_the_single_device_stream(self):
        for rows, layout in _TWO_RANGES:
            with self.subTest(rows=rows, layout=layout.parts):
                made = self._sector(tuple(range(len(layout.parts))), rows)
                with self._on(made, PARSEC_CUPY_DEVICE_RANDOM='0'):
                    generator, single = LapackRandom(), LapackRandom()
                    expected = single.uniform_minus_1_1((rows, layout.columns), column_major=True)
                    basis = made.group.random_basis(generator, rows, layout)
                    self.assertEqual(basis.layout, layout)
                    for index, block in enumerate(basis.blocks):
                        self.assertEqual((block.device.id, block.flags.f_contiguous), (index, True))
                        np.testing.assert_array_equal(np.asarray(block), expected[:, _held(layout, index)])
                    np.testing.assert_array_equal(np.asarray(made.group.gather(basis)), expected)
                    # The stream goes on where one device would have left it.
                    self.assertEqual(generator.seed, single.seed)
                self._idle(made)

    def test_a_trial_basis_is_given_the_room_of_the_step_with_one_tall_block_by_default(self):
        room_of = distributed_state.SectorDeviceGroup._room
        for rows, layout in _TWO_RANGES:
            count, capacity = len(layout.parts), {}
            for blocks in ('2', '3', '1', None):
                with self.subTest(rows=rows, layout=layout.parts, blocks=blocks):
                    made = self._sector(tuple(range(count)), rows)
                    named = {} if blocks is None else dict(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=blocks)
                    with self._on(made, PARSEC_CUPY_DEVICE_RANDOM='0', **named):
                        basis = made.group.random_basis(LapackRandom(), rows, layout)
                        capacity[blocks] = made.group._tall
                        # A block with columns is one allocation of that capacity, whatever they take of it.
                        for block in basis.blocks:
                            if block.size:
                                self.assertEqual(room_of(block), capacity[blocks])
                    self._idle(made)
            # Left to the default a block has the room in which its row block may end behind its columns: a
            # row of all devices per row of the tallest row block more than the blocks of the former steps.
            room = count * -(-rows // count)
            self.assertEqual((capacity[None], capacity['1']), (capacity['2'] + room,) * 2)
            self.assertEqual(capacity['3'], capacity['2'])

    def test_density_weights_follow_the_columns_of_every_block(self):
        for rows, layout in _TWO_RANGES:
            with self.subTest(rows=rows, layout=layout.parts):
                made = self._sector(tuple(range(len(layout.parts))), rows)
                rng = np.random.default_rng(73)
                host, given = rng.normal(size=(rows, layout.columns)), []

                def builder(block, weights, volume_element):
                    given.append((block.device.id, np.array(weights)))
                    return (2. / volume_element) * np.sum(np.asarray(block) ** 2 * weights[None, :], axis=1)

                with self._on(made):
                    basis = made.group.scatter(self._upload(made, host), layout)
                    for count in sorted({1, layout.columns // 2, layout.columns - 1, layout.columns}):
                        occupations = rng.uniform(size=count)
                        expected = (2. / .37) * np.sum(host[:, :count] ** 2 * occupations[None, :], axis=1)
                        # Of the whole basis, and of leading columns of it that still hold the occupied ones.
                        for selected in (basis, basis[:, :count], basis[:, :min(layout.columns, count + 4)]):
                            del given[:]
                            np.testing.assert_allclose(selected.density(builder, occupations, .37), expected, rtol=1e-13, atol=0)
                            # Every device with occupied columns was given their occupations in the order of its block.
                            occupied = layout.leading(count)
                            self.assertEqual(sorted(device for device, _weights in given),
                                             [index for index, width in enumerate(occupied.widths) if width])
                            for device, weights in given:
                                np.testing.assert_array_equal(weights, occupations[_held(occupied, device)])
                    with self.assertRaises(ValueError):
                        basis[:, :1].density(builder, np.ones(2), .37)
                self._idle(made)

    def test_every_device_filters_its_blocks_as_the_plan_does(self):
        lower, upper, reference = self.lower, self.upper, self.below
        cases = (((0, 1), 23, (23, 17)), ((0, 1, 2, 3), 53, (53, 47, 44, 31)), ((0, 1, 2), 59, (59, 54)))
        for devices, columns, counts in cases:
            for reset in (False, True):
                # The default layout, two column ranges per device, and contiguous ranges.
                for name in (None, 'contiguous'):
                    named = {} if name is None else dict(PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT=name)
                    with self.subTest(devices=devices, reset=reset, layout=name):
                        made = self._sector(devices)
                        graphs = made.sector._distributed_filter.graphs
                        expected = np.linalg.qr(np.random.default_rng(61).normal(size=(self.rows, columns)))[0]
                        with self._on(made, **named):
                            group = made.group
                            trial = uniform_filter_blocks(columns, 6, 7)
                            layout = group.column_layout(trial)
                            self.assertEqual({len(part) for part in layout.parts}, {2 if name is None else 1})
                            basis = group.scatter(self._upload(made, expected), layout)
                            # The cycles of a first solve, every block at one degree; then later passes with a
                            # lower and a higher degree, each count of columns twice.
                            plans = [trial] * 2 + [subspace_filter_blocks(count, 6, 7, 2) for count in counts for _ in range(2)]
                            for step, plan in enumerate(plans):
                                count = plan[-1].stop
                                basis, expected = basis[:, :count], expected[:, :count]
                                basis = group.filter(basis, plan, lower, upper, reference, reset)
                                # One device with the whole plan: the same columns to the last bit.
                                expected = _plan_filter(made.static, made.potential, expected, plan, lower, upper, reference,
                                                        reset)
                                np.testing.assert_array_equal(np.asarray(group.gather(basis)), expected)
                                self.assertEqual((basis.layout, group.layout), (layout.leading(count),) * 2)
                                for index, device in enumerate(devices):
                                    # One call per pass on a device that holds columns, for all of its blocks,
                                    # and none on one that holds none.
                                    held = [layout.leading(done[-1].stop).widths[index] > 0 for done in plans[:step + 1]]
                                    self.assertEqual(len(graphs[device].keys), sum(held))
                                    # The same plan again finds the graphs that were captured for it.
                                    if step and plan == plans[step - 1] and held[-1]:
                                        self.assertEqual(graphs[device].keys[-1], graphs[device].keys[-2])
                        self.assertEqual(group.seconds['repartition'], 0.)
                        self.assertGreater(group.seconds['filter'], 0.)
                        self._idle(made)

    def test_blocks_that_are_no_whole_filter_blocks_are_laid_out_anew(self):
        # A plan with blocks of five columns for a basis that was laid out for blocks of six: in the default
        # layout, in contiguous ranges where they are named, and in those that balanced ranges bring with them.
        contiguous = (((0, 15),), ((15, 23),))
        for named, parts in (({}, (((0, 5), (10, 20)), ((5, 10), (20, 23)))),
                             (dict(PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT='contiguous'), contiguous),
                             (dict(PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='balanced'), contiguous)):
            with self.subTest(named=named):
                made = self._sector((0, 1))
                start = np.linalg.qr(np.random.default_rng(63).normal(size=(self.rows, 23)))[0]
                with self._on(made, **named):
                    group = made.group
                    basis = group.scatter(self._upload(made, start), group.column_layout(uniform_filter_blocks(23, 6, 7)))
                    plan = subspace_filter_blocks(23, 5, 7, 2)
                    filtered = group.filter(basis, plan, self.lower, self.upper, self.below, False)
                    self.assertEqual(filtered.layout, group.column_layout(plan))
                    self.assertEqual(filtered.layout.parts, parts)
                    np.testing.assert_array_equal(
                        np.asarray(group.gather(filtered)),
                        _plan_filter(made.static, made.potential, start, plan, self.lower, self.upper, self.below, False))
                self.assertGreater(group.seconds['repartition'], 0.)
                self._idle(made)

    def _passes(self, devices, counts, *, eager=False, roomy=False, random=False, in_place=True, **environment):
        """A cycle of a first solve and later passes for ``counts`` columns: what each made of the basis.

        ``roomy`` column blocks have the capacity that the group gives the blocks of a trial basis; a
        ``random`` basis is that trial basis itself, as a first solve makes it.  ``in_place`` says that no
        pass may move the basis out of the blocks it started in.
        """
        made = self._sector(devices, eager=eager)
        start = np.linalg.qr(np.random.default_rng(67).normal(size=(self.rows, counts[0])))[0]
        hamiltonian = made.static + sp.diags(made.potential)
        made.history = []
        with self._on(made, **environment):
            group = made.group
            trial = uniform_filter_blocks(counts[0], 6, 5)
            made.layout = group.column_layout(trial)
            if random:
                basis = group.random_basis(LapackRandom(), self.rows, made.layout)
            else:
                basis = group.scatter(self._upload(made, start), made.layout)
            if roomy:
                tall, _chunk = group._capacities(made.layout, group.row_edges(self.rows))

                def place(index, device):
                    block = group._block(made.cp, (self.rows, made.layout.widths[index]), tall)
                    block[...] = basis.blocks[index]
                    return block

                basis = distributed_state.DistributedBasis(group, group.map(place), made.layout)
            made.blocks = basis.blocks
            plans = [(trial, self.below)] + [(subspace_filter_blocks(count, 6, 5, 1), self.lower) for count in counts]
            for plan, reference in plans:
                basis = basis[:, :plan[-1].stop]
                before = np.array(np.asarray(group.gather(basis)))
                filtered = group.filter(basis, plan, self.lower, self.upper, reference, False)
                after = np.array(np.asarray(group.gather(filtered)))
                np.testing.assert_array_equal(
                    after, _plan_filter(made.static, made.potential, before, plan, self.lower, self.upper, reference, False))
                values, basis, _whitened = group.ritz(made.sector, filtered, stages=False)
                # A pass leaves the basis in the memory in which it found it.
                for block, first in zip(basis.blocks, made.blocks):
                    self.assertTrue(not in_place or not block.size or block.data.ptr == first.data.ptr)
                rotated = np.array(np.asarray(group.gather(basis)))
                # Orthonormal Ritz vectors of the filtered columns, in the order of their values.
                np.testing.assert_allclose(rotated.T @ rotated, np.eye(values.size), atol=1e-9)
                np.testing.assert_allclose(rotated.T @ (hamiltonian @ rotated), np.diag(values), rtol=0, atol=1e-9)
                overlap, projection = after.T @ after, after.T @ (hamiltonian @ after)
                np.testing.assert_allclose(values, la.eigh((projection + projection.T) / 2, (overlap + overlap.T) / 2,
                                                           eigvals_only=True), rtol=0, atol=1e-9)
                made.history.append(SimpleNamespace(values=values, rotated=rotated, layout=basis.layout))
        self._idle(made)
        return made

    def test_later_passes_with_a_truncation_match_contiguous_ranges(self):
        for devices, counts in (((0, 1), (23, 17, 17)), ((0, 1, 2, 3), (53, 47, 31, 31))):
            # Two tall blocks, then three; two columns per slab and two or three per exchange chunk.
            for blocks in ('2', '3'):
                for eager in (False, True):
                    sizes = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 400),
                                 PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2),
                                 PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=blocks)
                    with self.subTest(devices=devices, blocks=blocks, eager=eager):
                        # The default layout, two column ranges per device, and contiguous ranges.
                        two = self._passes(devices, counts, eager=eager, **sizes)
                        one = self._passes(devices, counts, eager=eager, PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT='contiguous',
                                           **sizes)
                        self.assertEqual(([len(part) for part in two.layout.parts], [len(part) for part in one.layout.parts]),
                                         ([2] * len(devices), [1] * len(devices)))
                        for made in (two, one):
                            # Every pass took its Ritz step with the columns where the trial basis had them.
                            self.assertEqual((made.group.passes, made.group.seconds['repartition']), (len(counts) + 1, 0.))
                            self.assertEqual([step.layout for step in made.history],
                                             [made.layout.leading(count) for count in (counts[0],) + counts])
                            # The last two passes have one plan: no device that still holds columns captures
                            # its graphs again.
                            for device, width in zip(devices, made.history[-1].layout.widths):
                                keys = made.sector._distributed_filter.graphs[device].keys
                                if width:
                                    self.assertEqual(keys[-1], keys[-2])
                        for mine, theirs in zip(two.history, one.history, strict=True):
                            np.testing.assert_allclose(mine.values, theirs.values, rtol=0, atol=1e-10)
                            np.testing.assert_allclose(np.linalg.svd(theirs.rotated.T @ mine.rotated, compute_uv=False),
                                                       np.ones(mine.values.size), atol=5e-9)

    def test_later_passes_with_one_tall_block_match_two(self):
        # Blocks with the capacity of a trial basis through a cycle of a first solve and later passes, which
        # have fewer columns than the blocks were made for: every pass filters the blocks and re-lays them as
        # rows inside themselves.
        for devices, counts in (((0, 1), (23, 17, 17)), ((0, 1, 2, 3), (53, 47, 31, 31))):
            for eager in (False, True):
                # Two columns per slab and two or three per exchange chunk, in both layouts.
                sizes = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 400),
                             PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2))
                for named in ({}, dict(PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT='contiguous')):
                    with self.subTest(devices=devices, eager=eager, layout=named):
                        # One tall block is the default and is left to it; two are named.  On a clock that
                        # every reading moves on by one, the one call of a pass that forms the overlap takes 1.
                        ticks = itertools.count()
                        with patch.object(distributed_state, 'perf_counter', lambda: float(next(ticks))):
                            one = self._passes(devices, counts, eager=eager, roomy=True, **sizes, **named)
                            two = self._passes(devices, counts, eager=eager, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='2',
                                               **sizes, **named)
                        self.assertEqual((one.group.tall_blocks, two.group.tall_blocks), (1, 2))
                        # The seconds of the overlap add up over the passes of a group.
                        self.assertEqual((one.group.overlap_seconds, two.group.overlap_seconds), (len(counts) + 1.,) * 2)
                        self.assertEqual((one.group.passes, one.group.seconds['repartition']), (len(counts) + 1, 0.))
                        # No pass took a row block from the pool, with fewer columns than the blocks were made for too.
                        self.assertEqual(one.group.separate_row_blocks, 0)
                        self.assertEqual([step.layout for step in one.history], [step.layout for step in two.history])
                        for mine, theirs in zip(one.history, two.history, strict=True):
                            np.testing.assert_allclose(mine.values, theirs.values, rtol=0, atol=1e-10)
                            np.testing.assert_allclose(np.linalg.svd(theirs.rotated.T @ mine.rotated, compute_uv=False),
                                                       np.ones(mine.values.size), atol=5e-9)
                        # Blocks that are only as large as their columns get their row blocks from the pool
                        # where the rows do not happen to fit, and give the same Ritz pairs.
                        tight = self._passes(devices, counts, eager=eager, PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='1',
                                             **sizes, **named)
                        self.assertGreater(tight.group.separate_row_blocks, 0)
                        for mine, theirs in zip(tight.history, two.history, strict=True):
                            np.testing.assert_allclose(mine.values, theirs.values, rtol=0, atol=1e-10)

    def test_a_trial_basis_passes_a_first_solve_and_later_passes_in_its_own_blocks_by_default(self):
        # What a calculation does with nothing named: a random trial basis, a first solve and later passes
        # with fewer columns.  The step is the one with one tall block and no pass takes a row block from
        # the pool; two tall blocks, named, give the same Ritz values to round-off.
        sizes = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 400), PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2))
        for devices, counts in (((0, 1), (23, 17, 17)), ((0, 1, 2, 3), (53, 47, 31, 31))):
            for eager in (False, True):
                with self.subTest(devices=devices, eager=eager):
                    default = self._passes(devices, counts, eager=eager, random=True, **sizes)
                    two = self._passes(devices, counts, eager=eager, random=True,
                                       PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS='2', **sizes)
                    self.assertEqual((default.group.tall_blocks, default.group.passes, default.group.separate_row_blocks,
                                      default.group.seconds['repartition']), (1, len(counts) + 1, 0, 0.))
                    self.assertEqual((two.group.tall_blocks, two.group.separate_row_blocks), (2, 0))
                    for mine, theirs in zip(default.history, two.history, strict=True):
                        self.assertEqual(mine.layout, theirs.layout)
                        np.testing.assert_allclose(mine.values, theirs.values, rtol=0, atol=1e-10)

    def test_balanced_ranges_with_one_tall_block_match_two_and_three(self):
        # Balanced ranges move columns between the devices from pass to pass and lay every block out anew
        # with the capacity of a trial basis, so the default step finds its room after every move.
        sizes = dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(8 * 400), PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * self.rows * 2),
                     PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='balanced')
        for devices, counts in (((0, 1), (23, 17, 17)), ((0, 1, 2, 3), (53, 47, 31, 31)), ((0, 1, 2), (59, 54, 54))):
            for eager in (False, True):
                with self.subTest(devices=devices, eager=eager):
                    runs = {blocks: self._passes(
                        devices, counts, eager=eager, random=True, in_place=False, **sizes,
                        **({} if blocks is None else dict(PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=blocks)))
                        for blocks in (None, '2', '3')}
                    one = runs[None]
                    self.assertEqual((one.group.tall_blocks, one.group.passes, one.group.separate_row_blocks),
                                     (1, len(counts) + 1, 0))
                    for blocks in ('2', '3'):
                        for mine, theirs in zip(one.history, runs[blocks].history, strict=True):
                            self.assertEqual(mine.layout, theirs.layout)
                            np.testing.assert_allclose(mine.values, theirs.values, rtol=0, atol=1e-10)
                    # Columns did move in every one of them.
                    self.assertTrue(all(run.group.seconds['repartition'] > 0. for run in runs.values()))


if __name__ == '__main__':
    unittest.main()
