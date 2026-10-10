"""Actual-device parity and lifetime tests for captured FP64 block filters.

What surrounds a capture, the pause of the garbage collector, the end of a
capture that fails and the repetition of one that was invalidated, is also run
on a stand-in for the few CuPy calls of the capture, without a device.  So are
the graphs that the blocks of one width and degree share: what is recorded for
a plan and for the plans after it there, and a filter with them on the NumPy
device of test_distributed_exchange, whose streams hold their work back, which
is to give the columns of a filter with one graph per block to the last bit.
"""

from concurrent.futures import ThreadPoolExecutor
import gc
import os
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
from weakref import finalize, ref

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy import (
    CuPyHamiltonian,
    cupy_available,
    require_cupy,
)
from parsec_python.acceleration.backends import cupy_capture
from parsec_python.acceleration.backends.cupy_capture import (
    capture_graph,
    capture_invalidated,
    capture_statistics,
    collector_paused,
    end_failed_capture,
)
from parsec_python.acceleration.Eigensolvers import filter_graph
from parsec_python.acceleration.Eigensolvers.chebyshev import (
    FilterBlock,
    chebff_filter,
    subspace_filter,
    subspace_filter_blocks,
    uniform_filter_blocks,
)
from parsec_python.acceleration.Eigensolvers.distributed_filter import starting_sigmas
from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetrySCFEigensolver
from parsec_python.acceleration.SCF.single_point import _finalize_result
from parsec_python.acceleration.models import BackendInfo, BackendStatistics
from parsec_python.acceleration.tests.test_distributed_exchange import (
    _Array,
    _Stream,
    _stand_in,
    _table,
)


class _CaptureStream:
    """Stands for a CUDA stream: whether it captures, how often a capture ended."""

    def __init__(self, end_error=None):
        self.capturing, self.ended, self.end_error = False, 0, end_error

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def synchronize(self):
        if self.capturing:
            raise RuntimeError("a capturing stream cannot be synchronized")

    def begin_capture(self, mode=None):
        if self.capturing:
            raise RuntimeError("cannot capture twice")
        self.capturing = True

    def is_capturing(self):
        return self.capturing

    def end_capture(self):
        if not self.capturing:
            raise RuntimeError("the stream is not capturing")
        self.capturing = False
        self.ended += 1
        # An invalidated capture is over once it has been ended, and says so
        # with an error.
        if self.end_error is not None:
            raise self.end_error
        return object()


class _Cyclic:
    """Garbage for the cyclic collector; notes whether a stream captured then."""

    def __init__(self, destroyed, streams):
        self.myself, self.destroyed, self.streams = self, destroyed, streams

    def __del__(self):
        self.destroyed.append(any(stream.capturing for stream in self.streams))


class _CapturedOperator:
    """Stands for a Hamiltonian; ``launch`` is every step of its recurrence."""

    projector_count = 2
    projector_csr_data = None

    def __init__(self, launch):
        self.effective_potential = SimpleNamespace(device=SimpleNamespace(id=0))
        self.timing_stats = SimpleNamespace(
            filter_graph_buffer_bytes=0, filter_graph_builds=0
        )
        self.compact_finite_difference = SimpleNamespace(chebyshev_recurrence=launch)

    @staticmethod
    def custom_projector_projection(block, output=None):
        return output


class _DriverMemory(bytearray):
    """Stands for memory from the driver; its release can be watched, that of a bytearray cannot."""


class _StandInCase(unittest.TestCase):
    """Workspaces on stand-in streams; ``reuse`` is their PARSEC_CUPY_FILTER_GRAPH_REUSE."""

    reuse = "1"

    def setUp(self):
        self.addCleanup(gc.enable if gc.isenabled() else gc.disable)
        self.addCleanup(gc.set_threshold, *gc.get_threshold())
        gc.enable()
        # Every stream that a workspace created, and the error with which the
        # next ones end a capture.
        self.streams, self.end_error = [], None
        # The bytes of every allocation from the driver, and how many of them
        # there were whenever an event was waited for.
        self.allocated, self.waited = [], []
        # The bytes that the driver has given and not got back, now and at most.
        self.live = self.peak = 0

        def stream(non_blocking=False):
            self.streams.append(_CaptureStream(self.end_error))
            return self.streams[-1]

        def release(size):
            self.live -= size

        def memory(size):
            self.allocated.append(size)
            self.live += size
            self.peak = max(self.peak, self.live)
            taken = _DriverMemory(size)
            # Arrays keep the memory they lie in: this runs when the last of
            # them is gone.
            finalize(taken, release, size).atexit = False
            return taken

        cuda = SimpleNamespace(
            Memory=memory,
            MemoryPointer=lambda memory, offset: memory,
            alloc_pinned_memory=bytearray,
            Event=lambda disable_timing=False: SimpleNamespace(
                synchronize=lambda: self.waited.append(len(self.allocated))
            ),
            Stream=stream,
            runtime=SimpleNamespace(streamCaptureModeThreadLocal=1),
        )

        def ndarray(shape, dtype, memptr, order):
            return np.frombuffer(memptr, dtype=dtype).reshape(shape, order=order)

        cp = SimpleNamespace(cuda=cuda, float64=np.float64, ndarray=ndarray)
        for patcher in (
            patch.object(filter_graph, "require_cupy", lambda: (cp, None)),
            patch.dict(os.environ, PARSEC_CUPY_FILTER_GRAPH_REUSE=self.reuse),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.matrix = np.empty((17, 13))
        # Blocks of 6, 6 and 1 columns with three recurrence steps each.
        self.blocks = uniform_filter_blocks(13, 6, 3)

    def workspace(self, launch):
        """Filter graphs of a stand-in operator whose every step is ``launch``."""

        # Held here: the workspace refers to its operator weakly.
        self.operator = _CapturedOperator(launch)
        return filter_graph.BlockFilterGraphs(self.operator)

    def failing(self, call):
        """A launch that fails at its ``call``-th call, and the calls made."""

        calls = []

        def launch(*args, **kwargs):
            calls.append(1)
            if len(calls) == call:
                raise RuntimeError("injected launch failure")

        return launch, calls

    def in_use(self):
        """The bytes of the driver in use, and the most there were since this was last asked."""

        live, peak, self.peak = self.live, self.peak, self.live
        return live, peak


class CaptureStandInTests(_StandInCase):
    """What surrounds the captures of a workspace, on stand-in streams.

    With one graph per block: three captures of three launches for the plan.
    """

    reuse = "0"

    def test_every_block_is_captured_once_and_the_collector_is_as_it_was(self):
        for enabled in (True, False):
            with self.subTest(collector_enabled=enabled):
                (gc.enable if enabled else gc.disable)()
                capturing = []
                workspace = self.workspace(
                    lambda *args, **kwargs: capturing.append(
                        self.streams[-1].capturing
                    )
                )
                workspace._prepare(self.matrix, self.blocks)
                stream = self.streams[-1]
                # Three captures of three launches each, begun and ended on the
                # stream of the workspace.
                self.assertEqual(capturing, [True] * 9)
                self.assertIs(workspace.capture_stream, stream)
                self.assertEqual((stream.capturing, stream.ended), (False, 3))
                self.assertEqual(len(workspace.graphs), 3)
                self.assertEqual(
                    workspace.key, (17, ((0, 6, 3), (6, 12, 3), (12, 13, 3)))
                )
                self.assertEqual(self.operator.timing_stats.filter_graph_builds, 3)
                self.assertEqual(gc.isenabled(), enabled)
                # The same plan is not captured again.
                workspace._prepare(self.matrix, self.blocks)
                self.assertEqual(len(capturing), 9)
                self.assertIs(self.streams[-1], stream)

    def test_another_plan_takes_all_of_the_workspace_anew(self):
        # What the switch keeps for comparison: the buffers, the table, the
        # stream and the graphs of a plan are made again for the next one.
        workspace = self.workspace(lambda *args, **kwargs: None)
        workspace._prepare(self.matrix, self.blocks)
        first = [17 * 6 * 8] * 3 + [2 * 6 * 8, 9 * 32]
        self.assertEqual((self.allocated, self.waited), (first, []))
        self.assertEqual(self.in_use(), (sum(first),) * 2)
        # The same columns with two steps a block, once the device is done
        # with the plan before.
        workspace._prepare(self.matrix, uniform_filter_blocks(13, 6, 2))
        second = [17 * 6 * 8] * 3 + [2 * 6 * 8, 6 * 32]
        self.assertEqual((self.allocated, self.waited), (first + second, [5]))
        # The three new buffers were there before the old ones went.
        self.assertEqual(self.in_use(), (sum(second), sum(first) + 3 * 17 * 6 * 8))
        self.assertEqual([stream.ended for stream in self.streams], [3, 3])
        self.assertEqual(self.operator.timing_stats.filter_graph_builds, 6)

    def test_the_guard_collects_nothing_itself(self):
        destroyed, seen = [], []
        workspace = self.workspace(lambda *args, **kwargs: seen.append(list(destroyed)))
        # With the collector off, only a collection of the guard could destroy this.
        gc.disable()
        _Cyclic(destroyed, self.streams)
        workspace._prepare(self.matrix, self.blocks)
        self.assertEqual((seen, destroyed), ([[]] * 9, []))
        self.assertFalse(gc.isenabled())
        gc.collect()
        self.assertEqual(destroyed, [False])

    def test_no_collection_runs_while_a_stream_is_capturing(self):
        destroyed = []
        workspace = self.workspace(
            lambda *args, **kwargs: _Cyclic(destroyed, self.streams)
        )
        # A collection would follow nearly every allocation if the collector stayed on.
        gc.enable()
        gc.set_threshold(1, 1, 1)
        workspace._prepare(self.matrix, self.blocks)
        gc.collect()
        self.assertEqual(destroyed, [False] * 9)
        self.assertTrue(gc.isenabled())

    def test_a_failed_capture_is_ended_and_its_error_is_raised(self):
        for enabled in (True, False):
            with self.subTest(collector_enabled=enabled):
                (gc.enable if enabled else gc.disable)()
                # The second step of the second block fails.
                launch, calls = self.failing(5)
                workspace = self.workspace(launch)
                with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
                    workspace._prepare(self.matrix, self.blocks)
                stream = self.streams[-1]
                # The first block was captured; the capture of the second was
                # ended although it failed.
                self.assertEqual((stream.capturing, stream.ended), (False, 2))
                self.assertEqual((workspace.key, len(workspace.graphs)), (None, 1))
                self.assertEqual(gc.isenabled(), enabled)
                # The next call captures every block on a stream of its own.
                workspace._prepare(self.matrix, self.blocks)
                self.assertIsNot(self.streams[-1], stream)
                self.assertEqual(
                    (len(workspace.graphs), self.streams[-1].ended, len(calls)),
                    (3, 3, 14),
                )
                self.assertEqual(gc.isenabled(), enabled)

    def test_the_error_of_a_launch_is_raised_when_ending_its_capture_fails_too(self):
        # A launch into an invalidated capture fails, and so does the end of
        # that capture.
        self.end_error = RuntimeError("the capture was invalidated")
        launch, _calls = self.failing(2)
        workspace = self.workspace(launch)
        with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
            workspace._prepare(self.matrix, self.blocks)
        stream = self.streams[-1]
        self.assertEqual((stream.capturing, stream.ended), (False, 1))
        self.assertEqual((workspace.key, workspace.graphs), (None, []))
        self.assertTrue(gc.isenabled())

    def test_an_error_of_ending_a_capture_is_raised_and_the_capture_is_ended_once(self):
        self.end_error = RuntimeError("the capture was invalidated")
        workspace = self.workspace(lambda *args, **kwargs: None)
        with self.assertRaisesRegex(RuntimeError, "the capture was invalidated"):
            workspace._prepare(self.matrix, self.blocks)
        stream = self.streams[-1]
        self.assertEqual((stream.capturing, stream.ended), (False, 1))
        self.assertEqual((workspace.key, workspace.graphs), (None, []))
        self.assertTrue(gc.isenabled())

    def test_a_failed_capture_does_not_leave_the_key_of_the_graphs_it_replaced(self):
        # Nine launches capture the first plan; the second launch of the next
        # plan fails.
        launch, calls = self.failing(11)
        workspace = self.workspace(launch)
        workspace._prepare(self.matrix, self.blocks)
        former = workspace.key
        with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
            workspace._prepare(self.matrix, uniform_filter_blocks(13, 6, 4))
        self.assertIsNone(workspace.key)
        # The former plan finds no graphs of its own and is captured anew.
        workspace._prepare(self.matrix, self.blocks)
        self.assertEqual(
            (workspace.key, len(workspace.graphs), len(calls)), (former, 3, 20)
        )

    def test_an_invalidated_capture_is_recorded_again(self):
        calls = []

        def launch(*args, **kwargs):
            calls.append(1)
            # The second launch of the second block finds its capture
            # invalidated, once, as after the end of another thread.
            if len(calls) == 5:
                raise RuntimeError(
                    "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: operation failed due "
                    "to a previous error during capture"
                )

        workspace = self.workspace(launch)
        before = capture_statistics()
        with self.assertWarnsRegex(RuntimeWarning, r"is repeated \(attempt 2 of 4\)"):
            workspace._prepare(self.matrix, self.blocks)
        stream = self.streams[-1]
        # Three blocks and one repetition: four captures were ended, and the
        # two launches of the invalidated one were recorded for nothing.
        self.assertEqual((stream.capturing, stream.ended), (False, 4))
        # The process counts three captures that gave a graph and the
        # repetition, and has seen a capture take two attempts.
        after = capture_statistics()
        self.assertEqual(
            {name: after[name] - before[name] for name in ("captures", "repetitions")},
            {"captures": 3, "repetitions": 1},
        )
        self.assertGreaterEqual(after["most_attempts"], 2)
        self.assertEqual((len(workspace.graphs), len(calls)), (3, 11))
        self.assertEqual(workspace.key, (17, ((0, 6, 3), (6, 12, 3), (12, 13, 3))))
        self.assertTrue(gc.isenabled())

    def test_a_capture_that_stays_invalidated_is_given_up_after_four_attempts(self):
        calls = []

        def launch(*args, **kwargs):
            calls.append(1)
            raise RuntimeError(
                "cudaErrorStreamCaptureInvalidated: operation failed due to a "
                "previous error during capture"
            )

        workspace = self.workspace(launch)
        pauses, before = [], capture_statistics()
        with patch.object(cupy_capture.time, "sleep", pauses.append):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertRaisesRegex(RuntimeError, "CaptureInvalidated"):
                    workspace._prepare(self.matrix, self.blocks)
        stream = self.streams[-1]
        self.assertEqual((len(calls), len(caught)), (4, 3))
        # The waits before the second, third and fourth attempt.
        self.assertEqual(pauses, [0.0, 0.01, 0.1])
        # Three repetitions were counted, no capture gave a graph, and four
        # attempts are the most a capture can take.
        after = capture_statistics()
        self.assertEqual(
            {name: after[name] - before[name] for name in ("captures", "repetitions")},
            {"captures": 0, "repetitions": 3},
        )
        self.assertEqual(after["most_attempts"], 4)
        self.assertEqual((stream.capturing, stream.ended), (False, 4))
        self.assertEqual((workspace.key, workspace.graphs), (None, []))
        self.assertTrue(gc.isenabled())

    def test_capture_graph_repeats_only_what_was_invalidated(self):
        for text in (
            "cudaErrorStreamCaptureInvalidated: operation failed",
            "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: operation failed",
        ):
            self.assertTrue(capture_invalidated(RuntimeError(text)))
        # The error of a call that a capture refuses to its own thread is
        # not an invalidation and is not repeated.
        refused = RuntimeError(
            "cudaErrorStreamCaptureUnsupported: operation not permitted when "
            "stream is capturing"
        )
        self.assertFalse(capture_invalidated(refused))
        stream, modes = _CaptureStream(), []
        begin = stream.begin_capture
        stream.begin_capture = lambda mode=None: (modes.append(mode), begin(mode))[1]

        def refusing():
            raise refused

        before = capture_statistics()
        with self.assertRaises(RuntimeError) as raised:
            capture_graph(stream, refusing, mode=7)
        self.assertIs(raised.exception, refused)
        self.assertEqual((stream.capturing, stream.ended, modes), (False, 1, [7]))
        # Neither a capture nor a repetition was counted.
        self.assertEqual(capture_statistics(), before)
        # An invalidation that ending the capture reports is repeated too,
        # and what the recording returned comes back with the graph.
        stream = _CaptureStream(RuntimeError("cudaErrorStreamCaptureInvalidated"))
        recorded = []

        def record():
            recorded.append(1)
            if len(recorded) == 2:
                stream.end_error = None
            return "last"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph, value = capture_graph(stream, record)
        self.assertIsNotNone(graph)
        self.assertEqual((value, len(recorded), len(caught)), ("last", 2, 1))
        self.assertEqual((stream.capturing, stream.ended), (False, 2))
        # A filter that turns warnings into errors does not turn the
        # repetition into a failure.
        stream = _CaptureStream(RuntimeError("cudaErrorStreamCaptureInvalidated"))
        del recorded[:]
        before = capture_statistics()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            graph, value = capture_graph(stream, record)
        self.assertIsNotNone(graph)
        self.assertEqual((value, len(recorded), stream.ended), ("last", 2, 2))
        # The repetition is counted although its warning was dropped.
        after = capture_statistics()
        self.assertEqual(
            {name: after[name] - before[name] for name in ("captures", "repetitions")},
            {"captures": 1, "repetitions": 1},
        )

    def test_end_failed_capture_ends_what_is_open_and_raises_nothing(self):
        stream = _CaptureStream()
        # Not capturing: there is nothing to end.
        end_failed_capture(stream)
        self.assertEqual(stream.ended, 0)
        stream.begin_capture()
        end_failed_capture(stream)
        self.assertEqual((stream.capturing, stream.ended), (False, 1))
        # The error with which an invalidated capture ends is dropped.
        stream.end_error = RuntimeError("the capture was invalidated")
        stream.begin_capture()
        end_failed_capture(stream)
        self.assertEqual((stream.capturing, stream.ended), (False, 2))
        # A stream that cannot tell whether it captures is asked to end its capture.
        asked = []
        end_failed_capture(
            SimpleNamespace(
                is_capturing=lambda: 1 / 0, end_capture=lambda: asked.append(1)
            )
        )
        self.assertEqual(asked, [1])

    def test_collector_stays_off_until_the_last_of_several_threads_has_left(self):
        gc.enable()
        inside, leave, states = [Event(), Event()], [Event(), Event()], []

        def capture(index):
            with collector_paused():
                inside[index].set()
                leave[index].wait(timeout=10)
            states.append(gc.isenabled())

        threads = [Thread(target=capture, args=(index,)) for index in (0, 1)]
        for index, thread in enumerate(threads):
            thread.start()
            self.assertTrue(inside[index].wait(timeout=10))
            self.assertFalse(gc.isenabled())
        # The first thread leaves while the second is still inside.
        for index, thread in enumerate(threads):
            leave[index].set()
            thread.join(timeout=10)
        self.assertEqual(states, [False, True])
        self.assertTrue(gc.isenabled())

    def test_a_block_entered_while_another_is_open_collects_nothing(self):
        destroyed, stream = [], _CaptureStream()

        def enter():
            with collector_paused():
                pass

        with collector_paused():
            stream.begin_capture()
            _Cyclic(destroyed, [stream])
            # Another device thread prepares its graphs while this one captures.
            other = Thread(target=enter)
            other.start()
            other.join(timeout=10)
            self.assertEqual(destroyed, [])
            stream.end_capture()
        gc.collect()
        self.assertEqual(destroyed, [False])

    def test_collector_comes_back_on_if_any_entry_found_it_on(self):
        # Another party had the collector off for a moment of its own when
        # the first block was entered, and on again before the second was.
        gc.disable()
        with collector_paused():
            gc.enable()
            with collector_paused():
                self.assertFalse(gc.isenabled())
            self.assertFalse(gc.isenabled())
        self.assertTrue(gc.isenabled())

    def test_collector_comes_back_after_an_error_and_stays_off_where_it_was_off(self):
        gc.enable()
        with self.assertRaisesRegex(RuntimeError, "inside"):
            with collector_paused():
                self.assertFalse(gc.isenabled())
                raise RuntimeError("inside")
        self.assertTrue(gc.isenabled())
        gc.disable()
        with collector_paused():
            with collector_paused():
                self.assertFalse(gc.isenabled())
            self.assertFalse(gc.isenabled())
        self.assertFalse(gc.isenabled())


class SharedGraphStandInTests(_StandInCase):
    """The graphs that the blocks of one width and degree share, on stand-in streams.

    Two captures of three launches for the plan: one for its two blocks of
    six columns, one for the block of one column.
    """

    def captured(self, workspace, matrix, blocks):
        """Prepare ``blocks``; the captures that gave a graph meanwhile."""

        before = capture_statistics()["captures"]
        workspace._prepare(matrix, blocks)
        return capture_statistics()["captures"] - before

    def test_one_graph_is_captured_per_width_and_degree_and_the_collector_is_as_it_was(self):
        for enabled in (True, False):
            with self.subTest(collector_enabled=enabled):
                (gc.enable if enabled else gc.disable)()
                capturing = []
                workspace = self.workspace(
                    lambda *args, **kwargs: capturing.append(
                        self.streams[-1].capturing
                    )
                )
                self.assertEqual(self.captured(workspace, self.matrix, self.blocks), 2)
                stream = self.streams[-1]
                self.assertEqual(capturing, [True] * 6)
                self.assertIs(workspace.capture_stream, stream)
                self.assertEqual((stream.capturing, stream.ended), (False, 2))
                self.assertEqual(sorted(workspace.shared), [(1, 3), (6, 3)])
                # A graph for every block of the plan: the same one for the
                # two blocks of six columns, with a table of three rows.
                first, second, last = workspace.graphs
                self.assertIs(first, second)
                self.assertIsNot(first, last)
                self.assertEqual((first[2].shape, last[2].shape), ((3, 4), (3, 4)))
                self.assertEqual(
                    workspace.key, (17, ((0, 6, 3), (6, 12, 3), (12, 13, 3)))
                )
                self.assertEqual(self.operator.timing_stats.filter_graph_builds, 2)
                self.assertEqual(gc.isenabled(), enabled)
                # The same plan is not captured again.
                self.assertEqual(self.captured(workspace, self.matrix, self.blocks), 0)
                self.assertEqual(len(capturing), 6)
                self.assertIs(self.streams[-1], stream)

    def test_a_degree_change_records_what_is_new_and_allocates_its_tables_only(self):
        matrix, steps = np.empty((17, 37)), []
        workspace = self.workspace(
            lambda *args, **kwargs: steps.append(
                (id(kwargs["recurrence_parameters"]), kwargs["parameter_step"])
            )
        )
        # Six blocks of six columns, the first three one degree below and the
        # others one above, and a block of one column: the later passes of a
        # run whose degree falls by one, the first plan once more, and the
        # plan of a basis that has lost a block.
        plans = [subspace_filter_blocks(37, 6, degree, 1) for degree in (5, 4, 3, 5)]
        plans.append(subspace_filter_blocks(31, 6, 5, 1))
        counts, graphs = [], []
        for plan in plans:
            counts.append(self.captured(workspace, matrix, plan))
            graphs.append(list(workspace.graphs))
            if len(counts) == 1:
                buffers = workspace.buffers
                # A filter was submitted with the workspace since.
                workspace.has_upload = workspace.has_finished = True
        # Degrees 4 and 6, then 3 and 5, then 2 and the 4 that is there: one
        # graph per block would have taken 7, 7, 7, 7 and 6 captures.
        self.assertEqual(counts, [3, 3, 2, 0, 0])
        self.assertEqual(self.operator.timing_stats.filter_graph_builds, 8)
        degrees = (4, 6, 6, 3, 5, 5, 2, 4)
        # Every graph reads the rows of a table of its own from the first on.
        self.assertEqual(
            [step for _read, step in steps],
            [step for degree in degrees for step in range(degree)],
        )
        self.assertEqual(len({read for read, _step in steps}), 8)
        # The buffers, the projector buffer and the table of the 36 steps of
        # the first plan are allocated once, then a table per graph.
        self.assertEqual(
            self.allocated,
            [17 * 6 * 8] * 3 + [2 * 6 * 8, 36 * 32] + [32 * degree for degree in degrees],
        )
        self.assertIs(workspace.buffers, buffers)
        self.assertEqual(workspace.buffer_bytes, sum(self.allocated))
        # None of it went back to the driver.
        self.assertEqual(self.in_use(), (sum(self.allocated),) * 2)
        # The device was waited for before the tables of the second and of
        # the third plan were allocated, and not where nothing was recorded.
        self.assertEqual(self.waited, [8, 8, 11, 11])
        # One stream recorded every graph.
        self.assertEqual((len(self.streams), self.streams[0].ended), (1, 8))
        # The first plan finds its graphs again, and the shorter basis those
        # of its blocks.
        self.assertEqual(
            [now is before for now, before in zip(graphs[3], graphs[0], strict=True)],
            [True] * 7,
        )
        self.assertEqual(
            [entry is graphs[0][index] for entry, index in zip(graphs[4], (0, 1, 2, 3, 4, 6), strict=True)],
            [True] * 6,
        )
        self.assertEqual((workspace.parameters.shape, workspace.host_parameters.shape), ((30, 4),) * 2)

    def test_the_plans_of_a_run_record_a_graph_per_degree_and_not_per_block(self):
        # A sector of C2455H636 in a measured series: 1,314 states in 219
        # blocks, a first solve at degree 30 and six later plans whose degree
        # falls from 15 to 10, three less for the first half of the blocks
        # and three more for the others.  The run of four such sectors
        # recorded 6,133 graphs, one of them that of the Hartree solver.
        # A sector of C3469H804 there was trimmed from 1,842 to 1,841 states
        # after the first step: the last of its 307 blocks has five columns
        # in the later plans and takes a graph of its own in each of them.
        # That run recorded 8,597 graphs.
        for first, later, blocks, shared, recorded in (
            (1314, 1314, 219, 1 + 6 * 2, 6133),
            (1842, 1841, 307, 1 + 6 * 3, 8597),
        ):
            with self.subTest(states=(first, later)):
                plans = [uniform_filter_blocks(first, 6, 30)]
                plans += [subspace_filter_blocks(later, 6, degree, 3) for degree in range(15, 9, -1)]
                counts = {}
                for reuse in ("0", "1"):
                    with patch.dict(os.environ, PARSEC_CUPY_FILTER_GRAPH_REUSE=reuse):
                        workspace = self.workspace(lambda *args, **kwargs: None)
                    counts[reuse] = sum(self.captured(workspace, self.matrix, plan) for plan in plans)
                self.assertEqual(counts, {"0": 7 * blocks, "1": shared})
                self.assertEqual(4 * counts["0"] + 1, recorded)

    def test_wider_blocks_take_new_buffers_and_graphs_and_more_steps_a_longer_table(self):
        workspace = self.workspace(lambda *args, **kwargs: None)
        narrow = uniform_filter_blocks(5, 6, 3)
        self.assertEqual(self.captured(workspace, self.matrix, narrow), 1)
        self.assertEqual(self.allocated, [17 * 5 * 8] * 3 + [2 * 5 * 8, 3 * 32, 3 * 32])
        self.assertEqual(self.in_use(), (workspace.buffer_bytes,) * 2)
        workspace.has_upload = workspace.has_finished = True
        del self.allocated[:]
        # Six columns do not fit buffers of five: new buffers, a table for
        # nine steps and both graphs, once the device is done with the old.
        self.assertEqual(self.captured(workspace, self.matrix, self.blocks), 2)
        self.assertEqual(
            self.allocated, [17 * 6 * 8] * 3 + [2 * 6 * 8, 9 * 32, 3 * 32, 3 * 32]
        )
        self.assertEqual(sorted(workspace.shared), [(1, 3), (6, 3)])
        self.assertEqual(self.waited, [0, 0])
        # The old buffers went back to the driver before the new ones were
        # taken: it never held more than the workspace has now.
        self.assertEqual(workspace.buffer_bytes, sum(self.allocated))
        self.assertEqual(self.in_use(), (workspace.buffer_bytes,) * 2)
        # The narrow plan fits the wider buffers and the longer table; its
        # graph went with the buffers it was recorded for.
        del self.allocated[:]
        self.assertEqual(self.captured(workspace, self.matrix, narrow), 1)
        self.assertEqual(self.allocated, [3 * 32])
        self.assertEqual(workspace.buffers[0].shape, (17, 6))
        self.assertEqual(
            (workspace.parameters.shape, workspace.host_parameters.shape, workspace.table.shape),
            ((3, 4), (3, 4), (9, 4)),
        )
        self.assertEqual(self.in_use(), (workspace.buffer_bytes,) * 2)
        # A matrix of other rows takes new buffers and graphs as well, and
        # the old ones go first here too; the table stays.
        del self.allocated[:]
        self.assertEqual(self.captured(workspace, np.empty((19, 13)), self.blocks), 2)
        self.assertEqual(self.allocated, [19 * 6 * 8] * 3 + [2 * 6 * 8, 3 * 32, 3 * 32])
        self.assertEqual(sorted(workspace.shared), [(1, 3), (6, 3)])
        self.assertEqual(workspace.buffer_bytes, sum(self.allocated) + 9 * 32)
        self.assertEqual(self.in_use(), (workspace.buffer_bytes,) * 2)

    def test_a_failed_capture_leaves_no_graph_for_its_width_and_degree(self):
        for enabled in (True, False):
            with self.subTest(collector_enabled=enabled):
                (gc.enable if enabled else gc.disable)()
                # The second step of the second graph, that of the block of
                # one column, fails.
                launch, calls = self.failing(5)
                workspace = self.workspace(launch)
                with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
                    workspace._prepare(self.matrix, self.blocks)
                stream = self.streams[-1]
                self.assertEqual((stream.capturing, stream.ended), (False, 2))
                # The graph of the blocks of six columns stays; nothing names
                # the plan or a graph of the block of one column.
                self.assertEqual(
                    (workspace.key, workspace.graphs, list(workspace.shared)),
                    (None, [], [(6, 3)]),
                )
                self.assertEqual(gc.isenabled(), enabled)
                # The next call records the missing graph alone, on the
                # stream of the workspace.
                self.assertEqual(self.captured(workspace, self.matrix, self.blocks), 1)
                self.assertIs(self.streams[-1], stream)
                self.assertEqual(
                    (len(workspace.graphs), stream.ended, len(calls)), (3, 3, 8)
                )
                self.assertEqual(sorted(workspace.shared), [(1, 3), (6, 3)])
                self.assertEqual(gc.isenabled(), enabled)

    def test_a_failed_capture_for_another_plan_keeps_the_graphs_of_the_plan_before(self):
        # Six launches capture the first plan; the second launch for the next
        # plan fails.
        launch, calls = self.failing(8)
        workspace = self.workspace(launch)
        workspace._prepare(self.matrix, self.blocks)
        former = workspace.key
        with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
            workspace._prepare(self.matrix, uniform_filter_blocks(13, 6, 4))
        self.assertEqual(
            (workspace.key, workspace.graphs, sorted(workspace.shared)),
            (None, [], [(1, 3), (6, 3)]),
        )
        # The former plan has its graphs and records nothing.
        workspace._prepare(self.matrix, self.blocks)
        self.assertEqual(
            (workspace.key, len(workspace.graphs), len(calls)), (former, 3, 8)
        )

    def test_an_invalidated_capture_of_a_shared_graph_is_recorded_again(self):
        calls = []

        def launch(*args, **kwargs):
            calls.append(1)
            # The second launch of the second graph finds its capture
            # invalidated, once.
            if len(calls) == 5:
                raise RuntimeError(
                    "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: operation failed due "
                    "to a previous error during capture"
                )

        workspace = self.workspace(launch)
        before = capture_statistics()
        with self.assertWarnsRegex(RuntimeWarning, r"is repeated \(attempt 2 of 4\)"):
            workspace._prepare(self.matrix, self.blocks)
        stream = self.streams[-1]
        # Two graphs and one repetition: three captures were ended.
        self.assertEqual((stream.capturing, stream.ended), (False, 3))
        after = capture_statistics()
        self.assertEqual(
            {name: after[name] - before[name] for name in ("captures", "repetitions")},
            {"captures": 2, "repetitions": 1},
        )
        self.assertEqual((len(workspace.graphs), len(calls)), (3, 8))
        self.assertEqual(sorted(workspace.shared), [(1, 3), (6, 3)])
        self.assertTrue(gc.isenabled())

    def test_no_collection_runs_while_a_shared_graph_is_captured(self):
        destroyed = []
        workspace = self.workspace(
            lambda *args, **kwargs: _Cyclic(destroyed, self.streams)
        )
        gc.enable()
        gc.set_threshold(1, 1, 1)
        workspace._prepare(self.matrix, self.blocks)
        gc.collect()
        self.assertEqual(destroyed, [False] * 6)
        self.assertTrue(gc.isenabled())

    def test_the_switch_is_read_when_a_workspace_is_made(self):
        for value, reuse in (("1", True), ("on", True), ("0", False), (" Off ", False)):
            with patch.dict(os.environ, PARSEC_CUPY_FILTER_GRAPH_REUSE=value):
                workspace = self.workspace(lambda *args, **kwargs: None)
            # The route is that of the workspace, whatever is asked later.
            self.assertEqual(
                (workspace.reuse, filter_graph.graph_reuse_requested()), (reuse, True)
            )
            self.assertEqual(self.captured(workspace, self.matrix, self.blocks), 2 if reuse else 3)
            self.assertEqual(
                [staged is None for _graph, _output, staged in workspace.graphs],
                [not reuse] * 3,
            )
        with patch.dict(os.environ):
            del os.environ["PARSEC_CUPY_FILTER_GRAPH_REUSE"]
            self.assertTrue(filter_graph.graph_reuse_requested())
        with patch.dict(os.environ, PARSEC_CUPY_FILTER_GRAPH_REUSE="block"):
            with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_FILTER_GRAPH_REUSE"):
                self.workspace(lambda *args, **kwargs: None)


class GraphReportStandInTests(_StandInCase):
    """What a calculation reports of its filter graphs: those that were recorded, not the switch."""

    shared, per_block = "one per block width and degree", "one per block of a plan"

    def setUp(self):
        super().setUp()
        # A workspace refers to its operator weakly.
        self.operators = []

    def recorded(self, reuse, prepared=True):
        """A workspace made under this setting of the switch, with the graphs of a plan unless not ``prepared``."""

        self.operators.append(_CapturedOperator(lambda *args, **kwargs: None))
        with patch.dict(os.environ, PARSEC_CUPY_FILTER_GRAPH_REUSE=reuse):
            workspace = filter_graph.BlockFilterGraphs(self.operators[-1])
        if prepared:
            workspace._prepare(self.matrix, self.blocks)
        return workspace

    def solver(self, *operators):
        solver = object.__new__(CuPySymmetrySCFEigensolver)
        solver._operators = list(operators)
        return solver

    def test_a_sector_solver_reports_the_graphs_that_its_filters_recorded(self):
        def own(workspace):
            return SimpleNamespace(_filter_graph_workspace=workspace)

        def lent(*workspaces):
            # The devices that filter for a sector, each with a workspace.
            return SimpleNamespace(_distributed_filter=SimpleNamespace(graphs=dict(enumerate(workspaces))))

        self.assertEqual((filter_graph.graph_route_name(True), filter_graph.graph_route_name(False)),
                         (self.shared, self.per_block))
        for operators, expected in (
            # No filter of the process has recorded a graph: no workspace, or one that holds none.
            ((None, SimpleNamespace(), None, SimpleNamespace()), "none recorded"),
            ((own(self.recorded("1", prepared=False)), lent(self.recorded("0", prepared=False))), "none recorded"),
            ((own(self.recorded("1")), None, own(self.recorded("1"))), self.shared),
            ((None, own(self.recorded("0"))), self.per_block),
            # The workspaces of the devices of a sector count, also where its owner has recorded nothing.
            ((lent(self.recorded("1", prepared=False), self.recorded("1")), None), self.shared),
            ((lent(self.recorded("0"), self.recorded("0")),), self.per_block),
            # Workspaces made under both settings, which no run has.
            ((own(self.recorded("0")), own(self.recorded("1"))), f"{self.shared}; {self.per_block}"),
        ):
            with self.subTest(expected=expected, sectors=len(operators)):
                self.assertEqual(self.solver(*operators).recorded_filter_graphs, expected)
        # A capture that failed after a graph of the plan was recorded has left that graph.
        launch, _calls = self.failing(5)
        self.operators.append(_CapturedOperator(launch))
        workspace = filter_graph.BlockFilterGraphs(self.operators[-1])
        with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
            workspace._prepare(self.matrix, self.blocks)
        self.assertEqual((workspace.key, workspace.recorded), (None, True))
        self.assertEqual(self.solver(own(workspace)).recorded_filter_graphs, self.shared)

    def test_a_result_reports_the_recorded_graphs_in_the_place_of_the_switch(self):
        details = (
            ("orbital_sector_later_filter_precision", "float64"),
            ("orbital_sector_filter_graphs", self.shared),
            ("orbital_sector_local_potential_storage", "shared"),
        )

        def system():
            return SimpleNamespace(
                backend_info=BackendInfo(requested="auto", selected="cupy", details=details),
                materialize_final_wavefunctions=False,
                backend=SimpleNamespace(statistics=BackendStatistics()),
            )

        for workspace, expected in (
            (None, "none recorded"),
            (self.recorded("1"), self.shared),
            (self.recorded("0"), self.per_block),
        ):
            with self.subTest(expected=expected):
                solver = self.solver(SimpleNamespace(
                    _filter_graph_workspace=workspace, mixed_precision_recurrence=None))
                solver._state = SimpleNamespace(sector_state_counts=(3,))
                solver.memory_allocator_policy, solver.sector_state_storage = "pool", "device"
                final = _finalize_result(system(), object(), solver)
                # In its place among the details, whatever the switch said when they were written.
                self.assertEqual(
                    final.backend.details[:3],
                    (details[0], (details[1][0], expected), details[2]),
                )
        # A solver that says nothing of its graphs leaves the entry as the preparation wrote it.
        silent = SimpleNamespace(state=SimpleNamespace(sector_state_counts=(3,)),
                                 memory_allocator_policy="pool", sector_state_storage="device")
        self.assertEqual(_finalize_result(system(), object(), silent).backend.details[:3], details)


class _ReplayStream(_Stream):
    """A stream of the NumPy device that also captures, and counts its work.

    What is queued while it captures goes to a graph and does not run.  The
    count lets an event wait for the work queued before it and no more.
    """

    def __init__(self, runtime, device):
        super().__init__(runtime, device)
        self.recorded, self.queued, self.done = None, 0, 0

    def begin_capture(self, mode=None):
        if self.recorded is not None:
            raise RuntimeError("cannot capture twice")
        self.recorded = []

    def is_capturing(self):
        return self.recorded is not None

    def end_capture(self):
        recorded, self.recorded = self.recorded, None
        return _ReplayGraph(recorded)

    def queue(self, work):
        if self.runtime.local.device != self.device_id:
            raise AssertionError("a stream was used while another device was current")
        if self.recorded is not None:
            self.recorded.append(work)
            return
        self.pending.append(work)
        self.queued += 1
        if self.runtime.eager:
            self.run()

    def run(self, until=None):
        while self.pending and (until is None or self.done < until):
            self.done += 1
            self.pending.pop(0)()

    def synchronize(self):
        if self.recorded is not None:
            raise RuntimeError("a capturing stream cannot be synchronized")
        self.run()

    def wait_event(self, event):
        # For the work the event stood behind when this was asked, not for
        # what a later record of it stands behind.
        waited, mark = event.stream, event.mark
        self.queue(lambda: waited.run(mark))


class _ReplayGraph:
    """The launches that a capture recorded; a launch queues them again."""

    def __init__(self, recorded):
        self.recorded = recorded

    def launch(self, stream=None):
        for work in self.recorded:
            stream.queue(work)


class _ReplayEvent:
    """Complete once the work queued on its stream before it was recorded has run."""

    def __init__(self):
        self.stream = self.mark = None

    def record(self, stream):
        self.stream, self.mark = stream, stream.queued

    def synchronize(self):
        if self.stream is not None:
            self.stream.run(self.mark)


class _WorkspaceArray(_Array):
    """An array that a workspace laid into memory of the NumPy device.

    An upload on a stream reads its host values when the stream gets to it,
    like one from page-locked memory, which is where a workspace keeps the
    table it uploads: what is written there before then is what arrives.
    """

    def set(self, values, stream=None):
        if stream is None:
            return super().set(values)
        if not (self.flags.c_contiguous or self.flags.f_contiguous) or values.shape != self.shape:
            raise AssertionError("an upload fills a contiguous array of its own shape")
        stream.queue(lambda: np.copyto(np.asarray(self), values))


def _replay_device(eager):
    """The NumPy device of test_distributed_exchange with what a filter workspace asks besides.

    It gives no memory while one of its streams captures, and an upload from
    the host table of a workspace reads the table when its stream gets there.
    """

    cp, runtime = _stand_in(eager=eager)
    streams, empty, ndarray = [], cp.empty, cp.ndarray

    def stream(non_blocking=False, device=None):
        streams.append(
            _ReplayStream(runtime, runtime.local.device if device is None else device)
        )
        return streams[-1]

    def memory(size):
        if any(made.is_capturing() for made in streams):
            raise AssertionError("memory was taken from the driver inside a capture")
        return empty(size // 8)

    def copyto(target, source):
        runtime.local.streams[-1].queue(
            lambda: np.copyto(np.asarray(target), np.asarray(source))
        )

    # Work that names no stream goes to this one.
    runtime.local.defaults = {0: stream(device=0)}
    cp.copyto = copyto
    cp.ndarray = lambda *args, **kwargs: ndarray(*args, **kwargs).view(_WorkspaceArray)
    cp.cuda.Stream = stream
    cp.cuda.Event = lambda disable_timing=False: _ReplayEvent()
    cp.cuda.Memory = memory
    cp.cuda.MemoryPointer = lambda memory, offset: memory.data + offset
    cp.cuda.alloc_pinned_memory = bytearray
    cp.cuda.runtime.streamCaptureModeThreadLocal = 1
    return cp, runtime


class _ReplayOperator:
    """Stands for a Hamiltonian on the NumPy device: a host stencil, a potential and two projectors.

    Its two kernels, the projection and the fused recurrence step, run when
    their stream gets to them or when a graph that recorded them is launched,
    and read their arrays and the row of the coefficient table only then.
    """

    projector_count, projector_csr_data = 2, None

    def __init__(self, cp, runtime, rows):
        self.runtime = runtime
        self.stencil = sp.diags(
            [-0.2 * np.ones(rows - 1), 2.0 * np.ones(rows), -0.2 * np.ones(rows - 1)],
            [-1, 0, 1],
            format="csr",
        )
        self.projectors = sp.csr_matrix(
            np.random.default_rng(18).normal(size=(rows, 2)) * 0.02
        )
        self.signs = np.array([1.0, -1.0])
        with cp.cuda.Device(0):
            self.effective_potential = cp.asarray(np.linspace(-0.4, 0.3, rows))
        self.timing_stats = SimpleNamespace(
            filter_graph_buffer_bytes=0,
            filter_graph_builds=0,
            filter_graph_launches=0,
            hamiltonian_applications=0,
            orbital_vectors_applied=0,
        )
        self.compact_finite_difference = SimpleNamespace(
            chebyshev_recurrence=self.chebyshev_recurrence
        )

    def projection(self, block):
        return self.signs[:, None] * (self.projectors.T @ block)

    def step(self, row, current, previous, signed):
        """One recurrence step in the order of the fused kernel, with the coefficients ``row``."""

        center, scale, sigma, following = row
        value = self.stencil @ current
        value += np.asarray(self.effective_potential)[:, None] * current
        value += self.projectors @ signed
        value = (value - center * current) * scale
        if previous is not None:
            value -= sigma * previous
        return value * following

    def custom_projector_projection(self, block, output=None):
        self.runtime.local.streams[-1].queue(
            lambda: np.copyto(np.asarray(output), self.projection(np.asarray(block)))
        )
        return output

    def chebyshev_recurrence(
        self, current, potential, *, previous, projector_coefficients, output,
        recurrence_parameters, parameter_step, **_scalars,
    ):
        def kernel():
            value = self.step(
                np.array(np.asarray(recurrence_parameters)[parameter_step]),
                np.asarray(current),
                None if previous is None else np.asarray(previous),
                np.asarray(projector_coefficients),
            )
            np.copyto(np.asarray(output), value)

        self.runtime.local.streams[-1].queue(kernel)
        return output

    def filtered(self, columns, blocks, lower, upper, reference, reset, starts=None):
        """What the plan makes of ``columns`` on the host, block by block and step by step."""

        table, result, offset = _table(blocks, lower, upper, reference, reset, starts), np.empty_like(columns), 0
        for block in blocks:
            previous, current = None, columns[:, block.start : block.stop]
            for row in table[offset : offset + block.degree]:
                previous, current = current, self.step(row, current, previous, self.projection(current))
            result[:, block.start : block.stop] = current
            offset += block.degree
        return result


def _run_plans(columns, lower, upper, below):
    """The plans of a run of ``columns`` states and what they record.

    Per pass the blocks, the interval, the reference, whether sigma starts
    anew in every block and the sigma carried into each block where they are
    not the consecutive blocks of a plan; then the captures of the pass with
    one graph per block and with shared graphs.
    """

    first = uniform_filter_blocks(columns, 6, 7)
    later = subspace_filter_blocks(columns - 6, 6, 4, 2)
    # A device of a shared basis: the second and the last two blocks of a
    # plan side by side, each with the sigma that the plan carries into it.
    plan = subspace_filter_blocks(columns - 6, 6, 5, 2)
    picked, sigmas = (1, len(plan) - 2, len(plan) - 1), starting_sigmas(plan, lower, upper, False, below)
    held, stop = [], 0
    for index in picked:
        width = plan[index].stop - plan[index].start
        held.append(FilterBlock(stop, stop + width, plan[index].degree))
        stop += width
    return (
        # Two cycles of a first solve, sigma carried from block to block.
        (first, lower, upper, below, False, None, (7, 2)),
        (first, lower, upper, below, False, None, (0, 0)),
        # Later passes whose degree falls, with other bounds.
        (subspace_filter_blocks(columns, 6, 5, 2), lower, upper, lower, False, None, (7, 1)),
        (subspace_filter_blocks(columns, 6, 4, 2), lower, upper + 0.1, lower, False, None, (7, 3)),
        # A block less; then a lower bound of zero, where sigma stays at -1.
        (later, lower, upper + 0.1, lower, False, None, (6, 0)),
        (later, 0.0, upper, 0.0, False, None, (0, 0)),
        (uniform_filter_blocks(columns - 6, 6, 7), lower, upper, below, True, None, (6, 0)),
        (tuple(held), lower, upper, below, False, [sigmas[index] for index in picked], (3, 0)),
    )


class SharedGraphReplayTests(unittest.TestCase):
    """Filters with shared graphs on the NumPy device.

    Against one graph per block and against the recurrence taken step by
    step on the host, to the last bit.
    """

    rows, columns = 23, 37
    lower, upper, below = 1.9, 2.9, 1.3

    def run_passes(self, reuse, eager, launch=1):
        """Filter a basis in place with every plan; what the device and the host made of it.

        The passes go in turn to ``launch`` streams, and none is waited for
        before the next is submitted.
        """

        cp, runtime = _replay_device(eager)
        operator = _ReplayOperator(cp, runtime, self.rows)
        with patch.object(filter_graph, "require_cupy", lambda: (cp, None)), patch.dict(
            os.environ, PARSEC_CUPY_FILTER_GRAPH_REUSE=reuse
        ):
            workspace = filter_graph.BlockFilterGraphs(operator)
        expected = np.random.default_rng(139).normal(size=(self.rows, self.columns))
        with cp.cuda.Device(0):
            basis = cp.asarray(np.asfortranarray(expected))
        streams = [runtime.local.defaults[0]]
        streams += [cp.cuda.Stream(device=0) for _ in range(1, launch)]
        captures, passes, held = [], [], []
        for index, (blocks, lower, upper, reference, reset, starts, _captures) in enumerate(
            _run_plans(self.columns, self.lower, self.upper, self.below)
        ):
            # The stream of this pass: work that names no stream goes to it.
            stream = runtime.local.defaults[0] = streams[index % launch]
            count = blocks[-1].stop
            before, queued = capture_statistics()["captures"], stream.queued
            columns = basis[:, :count]
            result = workspace.apply(
                columns, blocks, lower, upper, reference, reset, initial_sigma=starts, out=columns
            )
            self.assertIs(result, columns)
            captures.append(capture_statistics()["captures"] - before)
            # The work that the pass queued and what the stream holds back after it.
            held.append((stream.queued - queued, len(stream.pending)))
            expected = expected.copy()
            expected[:, :count] = operator.filtered(
                expected[:, :count], blocks, lower, upper, reference, reset, starts
            )
            if eager:
                passes.append((np.array(np.asarray(basis)), expected))
        if not eager:
            # The stream still holds work back: nothing has waited for the
            # last pass.  Waiting for its stream, as a caller does, brings
            # the work of every other stream along.
            self.assertGreater(len(stream.pending), 0)
            stream.synchronize()
            self.assertEqual([len(made.pending) for made in streams], [0] * launch)
            passes.append((np.array(np.asarray(basis)), expected))
        return SimpleNamespace(
            passes=passes, captures=captures, held=held, operator=operator, runtime=runtime
        )

    def test_shared_graphs_filter_like_one_graph_per_block_and_like_the_host(self):
        plans = _run_plans(self.columns, self.lower, self.upper, self.below)
        launches = sum(len(plan[0]) for plan in plans)
        # What the shared graphs have to tell apart in these plans: sigma is
        # carried from block to block in a first solve, so two blocks of one
        # width and degree read different rows.
        blocks, lower, upper, reference, reset, starts, _captures = plans[0]
        table = _table(blocks, lower, upper, reference, reset, starts)
        self.assertEqual([(b.stop - b.start, b.degree) for b in blocks[:2]], [(6, 7)] * 2)
        self.assertFalse(np.array_equal(table[:7], table[7:14]))
        # Work done at once; work held back on one stream, where a pass is
        # behind the one before by its place; and held back on two streams
        # in turn, where only what the workspace waits for puts it there.
        for eager, launch in ((True, 1), (False, 1), (False, 2)):
            with self.subTest(eager=eager, launch=launch):
                single, shared = (self.run_passes(reuse, eager, launch) for reuse in ("0", "1"))
                self.assertEqual(len(shared.passes), len(plans) if eager else 1)
                for (one, _host), (other, host) in zip(single.passes, shared.passes, strict=True):
                    self.assertTrue(np.isfinite(other).all())
                    np.testing.assert_array_equal(other, one)
                    np.testing.assert_array_equal(other, host)
                self.assertEqual(
                    list(zip(single.captures, shared.captures)), [plan[6] for plan in plans]
                )
                if not eager and launch == 1:
                    # The plan with a block less records nothing with shared
                    # graphs and is queued behind the pass before it, which
                    # nothing has waited for; one graph per block waits for
                    # that pass and records.
                    (queued, pending), (shared_queued, shared_pending) = single.held[4], shared.held[4]
                    self.assertEqual(pending, queued)
                    self.assertGreater(shared_pending, shared_queued)
                for made in (single, shared):
                    self.assertEqual(made.operator.timing_stats.filter_graph_launches, launches)
                # A copy into the table of its graph before every launch of
                # a shared graph, and none where every block has its own.
                self.assertEqual(
                    (len(single.runtime.copies), len(shared.runtime.copies)), (0, launches)
                )
                self.assertEqual(
                    (single.operator.timing_stats.filter_graph_builds,
                     shared.operator.timing_stats.filter_graph_builds),
                    (36, 6),
                )


@unittest.skipUnless(cupy_available(), "CUDA unavailable")
class FilterGraphTests(unittest.TestCase):
    def tearDown(self):
        # Operators and filter graphs that a test left in a reference cycle
        # are destroyed here, between the tests, and not by a collection
        # inside the next one.
        gc.collect()

    def test_column_major_output_preserves_values(self):
        cp, operator = self.operator()
        matrix = cp.asarray(np.random.default_rng(125).normal(size=(257, 13)))
        settings = dict(degree=7, degree_delta=1, lower_bound=1.7, upper_bound=3.2, block_size=6)
        with patch.dict(os.environ, PARSEC_CUPY_FILTER_GRAPHS="1", PARSEC_CUPY_FILTER_COLUMN_MAJOR="0"):
            expected = subspace_filter(operator, matrix, **settings)
        with patch.dict(os.environ, PARSEC_CUPY_FILTER_GRAPHS="1", PARSEC_CUPY_FILTER_COLUMN_MAJOR="1"):
            actual = subspace_filter(operator, matrix, **settings)
        self.assertTrue(actual.flags.f_contiguous)
        np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected))

    def test_persistent_workspace_does_not_fragment_the_orbital_pool(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import (
            uniform_filter_blocks,
        )
        from parsec_python.acceleration.Eigensolvers.filter_graph import (
            BlockFilterGraphs,
        )

        cp, operator = self.operator()
        matrix = cp.empty((257, 7), dtype=cp.float64)
        # With shared graphs, which read two tables of their own here, and
        # with one graph per block, whatever the environment asks for.
        for reuse, count in (("1", 2), ("0", 0)):
            with self.subTest(reuse=reuse):
                with patch.dict(os.environ, PARSEC_CUPY_FILTER_GRAPH_REUSE=reuse):
                    cache = BlockFilterGraphs(operator)
                pool = cp.cuda.MemoryPool()
                try:
                    with cp.cuda.using_allocator(pool.malloc):
                        large = cp.empty(2**20, dtype=cp.float64)
                        del large
                        cached = pool.total_bytes()
                        cache._prepare(matrix, uniform_filter_blocks(7, 6, 5))
                        self.assertEqual(pool.used_bytes(), 0)
                        self.assertEqual(pool.total_bytes(), cached)
                        pool.free_all_blocks()
                        self.assertEqual(pool.total_bytes(), 0)
                finally:
                    # Also where a check above failed: the pool keeps no block
                    # that would go back to the driver only when the pool
                    # itself does.
                    pool.free_all_blocks()
                # The tables that shared graphs read are direct allocations too.
                tables = [table for _graph, _output, table in cache.shared.values()]
                self.assertEqual(len(tables), count)
                for array in (*cache.buffers, cache.coefficients, cache.parameters, *tables):
                    self.assertIsInstance(array.data.mem, cp.cuda.Memory)

    def test_shared_graphs_filter_like_one_graph_per_block_over_the_plans_of_a_run(self):
        cp, operator = self.operator()
        workspaces = []
        for reuse in ("0", "1"):
            with patch.dict(os.environ, PARSEC_CUPY_FILTER_GRAPH_REUSE=reuse):
                workspaces.append(filter_graph.BlockFilterGraphs(operator))
        single, shared = workspaces
        self.assertEqual((single.reuse, shared.reuse), (False, True))
        host = np.random.default_rng(139).normal(size=(257, 37))
        pointers = None
        # The plans and the captures that the stand-in device takes for them
        # in SharedGraphReplayTests.
        for blocks, lower, upper, reference, reset, starts, captures in _run_plans(
            37, 1.9, 2.9, 1.3
        ):
            matrix = cp.asarray(host[:, : blocks[-1].stop], order="F")
            results, counts = [], []
            for workspace in workspaces:
                before = capture_statistics()["captures"]
                result = workspace.apply(
                    matrix, blocks, lower, upper, reference, reset, initial_sigma=starts
                )
                results.append(cp.asnumpy(result))
                counts.append(capture_statistics()["captures"] - before)
            self.assertTrue(np.isfinite(results[1]).all())
            np.testing.assert_array_equal(results[1], results[0])
            self.assertEqual(tuple(counts), captures)
            # In place, as a caller that owns its columns filters them.
            for workspace in workspaces:
                owned = matrix.copy(order="F")
                self.assertIs(
                    workspace.apply(
                        owned, blocks, lower, upper, reference, reset, initial_sigma=starts, out=owned
                    ),
                    owned,
                )
                np.testing.assert_array_equal(cp.asnumpy(owned), results[0])
            # The buffers of the shared graphs stay where they are from plan
            # to plan.
            now = [buffer.data.ptr for buffer in shared.buffers]
            pointers = now if pointers is None else pointers
            self.assertEqual(now, pointers)
        self.assertEqual(
            sorted(shared.shared), [(1, 6), (1, 7), (6, 2), (6, 3), (6, 6), (6, 7)]
        )
        self.assertEqual(operator.timing_stats.filter_graph_builds, 36 + 6)

    def test_passes_that_nothing_waits_for_filter_like_passes_waited_for(self):
        cp, operator = self.operator()
        host = np.random.default_rng(139).normal(size=(257, 37))
        plans = _run_plans(37, 1.9, 2.9, 1.3)
        results = []
        # One graph per block with every pass waited for; then both routes
        # with the passes submitted one behind the other on two streams in
        # turn, where a plan that records nothing waits for nothing either.
        for reuse, waited in (("0", True), ("1", False), ("0", False)):
            with patch.dict(os.environ, PARSEC_CUPY_FILTER_GRAPH_REUSE=reuse):
                workspace = filter_graph.BlockFilterGraphs(operator)
            basis = cp.asarray(host, order="F")
            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
            for index, (blocks, lower, upper, reference, reset, starts, _captures) in enumerate(plans):
                columns = basis[:, : blocks[-1].stop]
                stream = streams[0 if waited else index % 2]
                with stream:
                    workspace.apply(
                        columns, blocks, lower, upper, reference, reset, initial_sigma=starts, out=columns
                    )
                if waited:
                    stream.synchronize()
            # The stream of the last pass alone is waited for.
            stream.synchronize()
            results.append(cp.asnumpy(basis))
        self.assertTrue(np.isfinite(results[0]).all())
        np.testing.assert_array_equal(results[1], results[0])
        np.testing.assert_array_equal(results[2], results[0])

    def test_a_failed_capture_is_ended_and_the_workspace_captures_again(self):
        cp, operator = self.operator()
        host = np.random.default_rng(127).normal(size=(257, 13))
        matrix = cp.asarray(host, order="F")
        blocks = uniform_filter_blocks(13, 6, 5)
        stencil = operator.compact_finite_difference
        recurrence = stencil.chebyshev_recurrence
        # With shared graphs and with one graph per block.
        for reuse in ("1", "0"):
            calls = []

            def failing(*args, **kwargs):
                calls.append(1)
                # The second step of the second graph, inside its capture.
                if len(calls) == 7:
                    raise RuntimeError("injected launch failure")
                return recurrence(*args, **kwargs)

            with self.subTest(reuse=reuse), patch.dict(
                os.environ, PARSEC_CUPY_FILTER_GRAPH_REUSE=reuse
            ):
                cache = filter_graph.BlockFilterGraphs(operator)
                enabled = gc.isenabled()
                with patch.object(stencil, "chebyshev_recurrence", failing):
                    with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
                        cache._prepare(matrix, blocks)
                self.assertFalse(cache.capture_stream.is_capturing())
                self.assertIsNone(cache.key)
                self.assertEqual(gc.isenabled(), enabled)
                # The workspace then captures what is missing and filters
                # like one that never failed.
                reference = filter_graph.BlockFilterGraphs(operator)
                expected = cp.asnumpy(reference.apply(matrix, blocks, 0.0, 4.0, -0.5, False))
                actual = cp.asnumpy(cache.apply(matrix, blocks, 0.0, 4.0, -0.5, False))
                self.assertEqual(len(cache.graphs), len(blocks))
                np.testing.assert_array_equal(actual, expected)

    def test_a_capture_invalidated_by_another_thread_is_recorded_again(self):
        cp, operator = self.operator()
        host = np.random.default_rng(131).normal(size=(257, 13))
        matrix = cp.asarray(host, order="F")
        blocks = uniform_filter_blocks(13, 6, 5)
        stencil = operator.compact_finite_difference
        recurrence, calls = stencil.chebyshev_recurrence, []
        cache = filter_graph.BlockFilterGraphs(operator)

        def eigensolve():
            # Measured on an A100 with CuPy 14.2: the end of a thread that
            # holds a cuSOLVER or a cuBLAS handle invalidates an open capture
            # on its device.  The eigensolve gives this thread a handle; by
            # itself, in a thread that lives on, it leaves the capture valid.
            with cp.cuda.Device(cache.device):
                cp.linalg.eigh(cp.eye(8))

        def disturbed(*args, **kwargs):
            calls.append(1)
            # Before the second step of the first block, inside its capture.
            if len(calls) == 2:
                thread = Thread(target=eigensolve)
                thread.start()
                thread.join()
            return recurrence(*args, **kwargs)

        reference = filter_graph.BlockFilterGraphs(operator)
        expected = cp.asnumpy(reference.apply(matrix, blocks, 0.0, 4.0, -0.5, False))
        with patch.object(stencil, "chebyshev_recurrence", disturbed):
            with self.assertWarnsRegex(RuntimeWarning, "is repeated"):
                actual = cp.asnumpy(cache.apply(matrix, blocks, 0.0, 4.0, -0.5, False))
        self.assertFalse(cache.capture_stream.is_capturing())
        self.assertEqual(len(cache.graphs), len(blocks))
        np.testing.assert_array_equal(actual, expected)

    def test_cache_does_not_retain_its_owner(self):
        cp, operator = self.operator(False)
        matrix = cp.asarray(np.random.default_rng(8).normal(size=(257, 7)))
        self.compare(operator, matrix)
        cp.cuda.get_current_stream().synchronize()
        owner = ref(operator)
        cache = operator._filter_graph_workspace
        del operator
        self.assertIsNone(owner())
        with self.assertRaises(ReferenceError):
            _ = cache.operator

    def operator(self, nonlocal_term=True):
        cp, _ = require_cupy()
        n = 257
        kinetic = sp.diags(
            [-0.2 * np.ones(n - 1), 2.0 * np.ones(n), -0.2 * np.ones(n - 1)],
            [-1, 0, 1],
            format="csr",
        )
        potential = np.linspace(-0.4, 0.3, n)
        projectors = (
            sp.csr_matrix(np.random.default_rng(18).normal(size=(n, 3)) * 0.02)
            if nonlocal_term
            else sp.csr_matrix((n, 0))
        )
        signs = np.array([1.0, -1.0, 1.0]) if nonlocal_term else np.zeros(0)
        with patch.dict(
            os.environ,
            {
                "PARSEC_CUPY_COMPACT_FD": "1",
                "PARSEC_CUPY_STENCIL_MAJOR": "1",
                "PARSEC_CUPY_FUSED_PROJECTOR_SCATTER": "1",
                "PARSEC_CUPY_PROJECTOR_PROJECTION": "1",
            },
        ):
            result = CuPyHamiltonian(
                kinetic, potential, projectors=projectors, projector_signs=signs
            )
        self.assertIsNotNone(result.compact_finite_difference)
        return cp, result

    def compare(
        self, operator, matrix, *, chebff=False, reset=False, lower=0.0, upper=4.0
    ):
        function = chebff_filter if chebff else subspace_filter
        args = (
            (operator, matrix, 7, lower, upper, -0.5)
            if chebff
            else (operator, matrix, 7, 2, lower, upper)
        )
        kwargs = {"block_size": 6, "reset_recurrence_per_block": reset}
        if not chebff:
            kwargs["mixed_precision"] = False
        with patch.dict(
            os.environ,
            {"PARSEC_CUPY_FILTER_GRAPHS": "0", "PARSEC_CUPY_BATCH_FILTERS": "0"},
        ):
            expected = function(*args, **kwargs)
        with patch.dict(
            os.environ,
            {"PARSEC_CUPY_FILTER_GRAPHS": "1", "PARSEC_CUPY_BATCH_FILTERS": "0"},
        ):
            actual = function(*args, **kwargs)
        self.assertGreater(operator.timing_stats.filter_graph_launches, 0)
        cp, _ = require_cupy()
        np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected))
        return actual

    def test_remainders_sigma_carry_reset_and_changed_bounds(self):
        cp, operator = self.operator()
        base = cp.asarray(np.random.default_rng(5).normal(size=(257, 38)))
        matrix = base[:, ::2]
        for chebff in (False, True):
            for reset in (False, True):
                self.compare(operator, matrix, chebff=chebff, reset=reset)
                builds = operator.timing_stats.filter_graph_builds
                self.compare(
                    operator,
                    matrix * 0.7,
                    chebff=chebff,
                    reset=reset,
                    lower=-0.2,
                    upper=4.2,
                )
                self.assertEqual(builds, operator.timing_stats.filter_graph_builds)

    def test_potential_update_resize_and_prior_result_lifetime(self):
        cp, operator = self.operator(False)
        matrix = cp.asarray(np.random.default_rng(8).normal(size=(257, 17)), order="F")
        with patch.dict(os.environ, {"PARSEC_CUPY_FILTER_GRAPHS": "1"}):
            first = subspace_filter(
                operator, matrix, 8, 2, 0.0, 4.0, mixed_precision=False
            )
            second = subspace_filter(
                operator, matrix * 0.5, 8, 2, 0.0, 4.0, mixed_precision=False
            )
        np.testing.assert_allclose(
            cp.asnumpy(second), 0.5 * cp.asnumpy(first), rtol=0, atol=1e-13
        )
        operator.update_local_potential(np.linspace(-0.1, 0.4, 257))
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            self.compare(operator, matrix[:, :13], lower=-0.2, upper=4.4)
        stream.synchronize()
        workspace = operator._filter_graph_workspace
        self.assertLess(workspace.buffer_bytes, 3 * 257 * 6 * 8 + 10000)

    def test_two_devices_capture_and_replay_independently(self):
        cp, _ = require_cupy()
        if cp.cuda.runtime.getDeviceCount() < 2:
            self.skipTest("two GPUs required")
        # Avoid concurrent mutation of process-wide environment during workers.
        with patch.dict(os.environ, {"PARSEC_CUPY_FILTER_GRAPHS": "1"}):
            operators = []
            for device in (0, 1):
                with cp.cuda.Device(device):
                    operators.append(self.operator()[1])

            def task(device):
                with cp.cuda.Device(device):
                    operator = operators[device]
                    matrix = cp.asarray(np.random.default_rng(3).normal(size=(257, 13)))
                    result = subspace_filter(
                        operator, matrix, 7, 2, 0.0, 4.0, mixed_precision=False
                    )
                    return cp.asnumpy(result), operator._filter_graph_workspace.device

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(task, (0, 1)))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        self.assertEqual([r[1] for r in results], [0, 1])


if __name__ == "__main__":
    unittest.main()
