"""Keep CUDA stream captures valid and repeat one that was invalidated.

A capture records the launches made on a stream between ``begin_capture`` and
``end_capture`` as a graph.  Some events in between invalidate it: the
launches that follow and ``end_capture`` then fail with
``cudaErrorStreamCaptureInvalidated``.  Which do was measured on an A100 with
CuPy 14.2, with one open capture in the thread-local or the relaxed mode and
one event in its own or in another thread of the same device:

* In the thread-local mode, calls of the capturing thread that take memory
  from the driver or return it (device memory outside a pool, a pool that
  grows or is emptied, pinned memory taken), that destroy a graph, begin
  another capture, synchronize a stream or an event, or copy to the host;
  a new array, a sum, an inner product and a cuSOLVER eigensolve of the
  capturing thread were refused as well.  CuPy returns memory and destroys
  graphs in destructors, and Python's cyclic garbage collector runs the
  destructors of every unreachable object of the process in whichever thread
  happens to reach a threshold.  Made by another thread, a collection
  included, these calls leave the capture valid; in the relaxed mode they do
  so from the capturing thread as well.
* In both modes, a device synchronization by any thread; the call is refused
  to its caller.
* In both modes, the end of another thread that holds a cuBLAS or a cuSOLVER
  handle of CuPy: a thread that has made a matrix product or solved a dense
  eigenproblem, or only asked for one of the handles.  An eigensolve, a
  Cholesky or a QR factorization of another thread that lives on leaves the
  capture valid, and so does the end of a thread that holds a cuSPARSE
  handle, that only worked with arrays, or that did nothing on the device.

The global capture mode and calls on other devices were not measured.  A
capture begun without a mode behaved as one in the relaxed mode.

:func:`collector_paused` keeps the collector out of a capture.
:func:`capture_graph` ends a capture that failed and repeats one that was
invalidated.  Every thread of a solver holds a cuBLAS handle.  The threads of
an executor end when the executor is collected, which can be while the
threads of a later solver capture; that is the likely, not the established,
source of the invalidations seen in the hybrid-driver tests.  The MPI runs
keep the threads of their solver for the whole calculation.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
from threading import Lock
import time
from typing import Any, Callable
import warnings


_PAUSE_LOCK = Lock()
_PAUSE_DEPTH = 0
_PAUSE_RESTORES = False
# Seconds to wait before the second, third and fourth attempt at a capture.
_RETRY_PAUSES = (0.0, 0.01, 0.1)
_COUNT_LOCK = Lock()
_CAPTURES = 0
_REPETITIONS = 0
_MOST_ATTEMPTS = 0


@contextmanager
def collector_paused():
    """Keep the cyclic garbage collector off inside the block.

    What becomes unreachable in a reference cycle stays until the collector
    is on again; an object that loses its last reference is destroyed at
    once, as always.  Nothing is collected on entry: what has to stay out of
    a capture is a collection in its own thread, and the collector being off
    does that.

    The switch of the collector belongs to the process, and the devices of a
    group capture at the same time on threads of their own, so the blocks are
    counted: the collector comes back on when the last of them is left, if
    one of them found it on at its entry.

    Only the collector is held back, and only by this switch: an object that
    a thread releases itself, a collection that a thread asks for and a
    collector that another party switches back on meanwhile are not.
    """

    global _PAUSE_DEPTH, _PAUSE_RESTORES
    with _PAUSE_LOCK:
        if not _PAUSE_DEPTH:
            _PAUSE_RESTORES = False
        # Another party may have the collector off for a moment of its own,
        # and switch it back on afterwards, just when a block is entered:
        # what the first entry finds is not always the state to return to.
        if gc.isenabled():
            _PAUSE_RESTORES = True
        _PAUSE_DEPTH += 1
        gc.disable()
    try:
        yield
    finally:
        with _PAUSE_LOCK:
            _PAUSE_DEPTH -= 1
            if not _PAUSE_DEPTH and _PAUSE_RESTORES:
                gc.enable()


def end_failed_capture(stream: Any) -> None:
    """End on ``stream`` a capture that did not give a graph.

    A stream stays in capture mode, and its thread under the restrictions of
    a capture, until the capture is ended on it, also once the capture has
    been invalidated.  Whatever was recorded is discarded.  Ending an
    invalidated capture reports the invalidation once more: errors of this
    cleanup are dropped, so that the error that stopped the capture is the
    one the caller propagates.
    """

    try:
        capturing = stream.is_capturing()
    except Exception:
        # The state is unknown: ending the capture is attempted.
        capturing = True
    if capturing:
        try:
            stream.end_capture()
        except Exception:
            pass


def capture_invalidated(error: BaseException) -> bool:
    """Whether ``error`` reports a capture that an earlier event invalidated.

    The runtime and the driver interface of CuPy name the same condition
    differently.  The error of a call that a capture refuses to its own
    thread is another one and is not meant here.
    """

    text = str(error)
    return "CaptureInvalidated" in text or "CAPTURE_INVALIDATED" in text


def capture_statistics() -> dict[str, int]:
    """Captures that gave a graph, repetitions, and the most attempts one capture took.

    Of this process, since it started.  A warning is shown once per call
    site by default, so it cannot say how often captures were recorded
    again; the counts can, and the largest number of attempts says whether a
    first repetition ever failed.  They are never reset: a caller that
    reports one calculation takes the difference of the two counts.  The MPI
    runner writes all three for every rank.  The text report of a process
    names the repetitions of its calculation where there were any; in an MPI
    run that is the root rank only.
    """

    with _COUNT_LOCK:
        return {
            "captures": _CAPTURES,
            "repetitions": _REPETITIONS,
            "most_attempts": _MOST_ATTEMPTS,
        }


def _count(captures: int = 0, repetitions: int = 0, attempts: int = 0) -> None:
    global _CAPTURES, _REPETITIONS, _MOST_ATTEMPTS
    with _COUNT_LOCK:
        _CAPTURES += captures
        _REPETITIONS += repetitions
        _MOST_ATTEMPTS = max(_MOST_ATTEMPTS, attempts)


def _report_repetition(attempt: int, error: BaseException) -> None:
    # The attempt that follows is the one counted: two after one invalidation.
    _count(repetitions=1, attempts=attempt + 2)
    try:
        warnings.warn(
            "a CUDA graph capture was invalidated and is repeated "
            f"(attempt {attempt + 2} of {len(_RETRY_PAUSES) + 1}): {error}",
            RuntimeWarning,
            stacklevel=3,
        )
    except Warning:
        # A filter that turns warnings into errors is not to turn a capture
        # that can be recorded again into a failure.
        pass


def capture_graph(stream: Any, record: Callable[[], Any], *, mode: Any = None):
    """Capture what ``record()`` launches on ``stream``.

    Returns the graph and what ``record`` returned.  ``record`` only launches
    work on the capturing stream: nothing runs before the graph is launched,
    so recording again is all that a repetition takes.  ``mode`` is passed to
    ``begin_capture`` where it is given.

    A capture that fails is ended on its stream whatever failed, and the
    first error is the one raised.  A capture that was invalidated is
    repeated up to three times, after 0, 10 and 100 ms: the end of another
    thread can invalidate it (see the module documentation), and nothing
    here decides when the threads of a released solver end.  Whether a
    second or third repetition has ever been needed was not recorded before
    the counts existed; their pauses are a margin, not a measured need.  A
    repetition is counted (:func:`capture_statistics`) and reported with a
    warning, which Python shows once per call site unless asked otherwise.
    An error of another kind, or a fourth invalidation, is raised.  An invalidation that the capturing thread causes itself without
    being refused, by a destructor for instance, is repeated like any other.
    """

    for attempt in range(len(_RETRY_PAUSES) + 1):
        if attempt:
            time.sleep(_RETRY_PAUSES[attempt - 1])
        graph = None
        if mode is None:
            stream.begin_capture()
        else:
            stream.begin_capture(mode=mode)
        try:
            recorded = record()
            graph = stream.end_capture()
            _count(captures=1, attempts=attempt + 1)
            return graph, recorded
        except Exception as error:
            if attempt == len(_RETRY_PAUSES) or not capture_invalidated(error):
                raise
            invalidation = error
        finally:
            if graph is None:
                end_failed_capture(stream)
        _report_repetition(attempt, invalidation)
    raise AssertionError("unreachable: the last attempt returns or raises")


__all__ = [
    "capture_graph",
    "capture_invalidated",
    "capture_statistics",
    "collector_paused",
    "end_failed_capture",
]
