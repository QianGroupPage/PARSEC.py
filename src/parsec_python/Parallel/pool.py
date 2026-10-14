"""Persistent worker pools for k-point and band-path parallelism.

The eigenproblems of different k-points (and of different points of a band
path) are independent; only the Fermi level and the density couple them.  A
pool therefore keeps, in every worker, the *state* of the tasks it owns -- the
Bloch operators, the eigenvectors that the next SCF step starts from -- and
exchanges only small messages with the driver: the potentials going out, the
eigenvalues and partial densities coming back.

A pool is built from a ``factory(tasks, worker_index)`` that returns the
worker object for one subset of the task ids.  The pool calls methods of that
object by name; a method is a generator (or returns an iterable) of
``(task_id, result)`` pairs, which are streamed back as soon as they exist so a
driver can report progress per task.

Backends (same interface):

``inline``
    one worker object in the calling process (the serial path).
``thread``
    one thread of this process per worker object, as in the multi-GPU solver:
    the NumPy/SciPy/BLAS kernels and the CuPy calls release the GIL, so
    several threads keep several cores or devices busy.  There is no
    ``fork``: forking after the OpenMP kernels of the native extension or a
    CUDA context exist is unsafe.  The workers share the BLAS threads of the
    process for the lifetime of the pool.

Task ids are dealt round-robin to the workers so each gets a similar share.
"""

from __future__ import annotations

import os
import queue
import threading
import traceback
from typing import Any, Callable, Iterable, Iterator, Sequence

from .blas import set_blas_threads

WORKERS_ENV = "PARSEC_KPOINT_WORKERS"


class WorkerError(RuntimeError):
    """An exception raised inside a pool worker (with its traceback text)."""


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not Linux
        return os.cpu_count() or 1


def resolve_workers(
    requested: int | str | None,
    n_tasks: int,
    *,
    problem_size: int | None = None,
    minimum_problem_size: int = 4000,
) -> int:
    """Number of worker threads for ``n_tasks`` independent tasks.

    ``requested`` is an integer, ``"auto"`` or ``None`` (then the environment
    variable ``PARSEC_KPOINT_WORKERS``, default ``auto``).  ``auto`` runs
    serially for a single task and for problems too small to repay the
    start-up (``problem_size`` = grid points, below ``minimum_problem_size``);
    otherwise it uses half the available CPUs (the sparse operators are
    memory-bandwidth bound, so more workers than that gain little), at most
    one per task.  An explicit number is capped only by the number of tasks.
    """

    if requested is None:
        requested = os.environ.get(WORKERS_ENV, "auto")
    if isinstance(requested, str):
        text = requested.strip().lower()
        requested = "auto" if text in ("", "auto") else int(text)
    if n_tasks < 2:
        return 1
    if requested == "auto":
        if problem_size is not None and problem_size < minimum_problem_size:
            return 1
        return max(1, min(n_tasks, available_cpus() // 2))
    return max(1, min(int(requested), n_tasks))


def _partition(task_ids: Sequence[int], workers: int) -> list[list[int]]:
    return [list(task_ids[index::workers]) for index in range(workers)]


class _InlinePool:
    kind = "inline"

    def __init__(self, task_ids: Sequence[int], factory: Callable[[list[int], int], Any]):
        self.n_workers = 1
        self._worker = factory(list(task_ids), 0)

    def stream(self, method: str, *args, **kwargs) -> Iterator[tuple[Any, Any]]:
        yield from getattr(self._worker, method)(*args, **kwargs)

    def close(self) -> None:
        self._worker = None


class _ThreadPool:
    kind = "thread"

    def __init__(self, task_ids, factory, workers: int):
        self.n_workers = workers
        # The workers' host-side dense algebra shares one BLAS: split its threads
        # between them for the lifetime of the pool.
        self._blas_restore = available_cpus() if set_blas_threads(max(1, available_cpus() // workers)) else None
        self._commands: list[queue.Queue] = []
        self._results: queue.Queue = queue.Queue()
        self._threads = []
        ready = [threading.Event() for _ in range(workers)]
        self._failures: dict[int, str] = {}
        for index, chunk in enumerate(_partition(task_ids, workers)):
            commands: queue.Queue = queue.Queue()
            thread = threading.Thread(
                target=self._serve,
                args=(commands, factory, chunk, index, ready[index]),
                daemon=True,
            )
            thread.start()
            self._commands.append(commands)
            self._threads.append(thread)
        for event in ready:
            event.wait()

    def _serve(self, commands, factory, tasks, index, ready) -> None:
        worker = None
        try:
            worker = factory(tasks, index)
        except BaseException:
            self._failures[index] = traceback.format_exc()
        ready.set()
        while True:
            message = commands.get()
            if message is None:
                return
            method, args, kwargs = message
            if index in self._failures:
                self._results.put((index, "error", self._failures[index]))
                continue
            try:
                for item in getattr(worker, method)(*args, **kwargs):
                    self._results.put((index, "item", item))
                self._results.put((index, "done", None))
            except BaseException:
                self._results.put((index, "error", traceback.format_exc()))

    def stream(self, method: str, *args, **kwargs) -> Iterator[tuple[Any, Any]]:
        for commands in self._commands:
            commands.put((method, args, kwargs))
        remaining = len(self._commands)
        while remaining:
            _, kind, payload = self._results.get()
            if kind == "item":
                yield payload
            elif kind == "done":
                remaining -= 1
            else:
                self.close()
                raise WorkerError(f"pool worker failed:\n{payload}")

    def close(self) -> None:
        for commands in self._commands:
            commands.put(None)
        for thread in self._threads:
            thread.join(timeout=5)
        self._commands = []
        self._threads = []
        if self._blas_restore is not None:
            set_blas_threads(self._blas_restore)
            self._blas_restore = None


class WorkerPool:
    """A pool of workers owning disjoint subsets of ``task_ids`` (see module doc)."""

    def __init__(self, backend) -> None:
        self._backend = backend

    @property
    def kind(self) -> str:
        return self._backend.kind

    @property
    def n_workers(self) -> int:
        return self._backend.n_workers

    def stream(self, method: str, *args, **kwargs) -> Iterator[tuple[Any, Any]]:
        """``(task_id, result)`` pairs of ``method`` as the workers produce them."""
        return self._backend.stream(method, *args, **kwargs)

    def results(self, method: str, *args, **kwargs) -> list[tuple[Any, Any]]:
        """All ``(task_id, result)`` pairs of ``method``."""
        return list(self.stream(method, *args, **kwargs))

    def by_task(self, method: str, *args, **kwargs) -> dict[Any, Any]:
        """``{task_id: result}`` of ``method``."""
        return dict(self.stream(method, *args, **kwargs))

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def make_pool(
    task_ids: Iterable[int],
    factory: Callable[[list[int], int], Any],
    *,
    workers: int | str | None = None,
    problem_size: int | None = None,
) -> WorkerPool:
    """Build a pool over ``task_ids``.

    The worker count comes from :func:`resolve_workers`; one worker means the
    inline backend (no thread at all).  For devices pass the number of devices
    as an explicit ``workers``.
    """

    task_ids = list(task_ids)
    count = resolve_workers(workers, len(task_ids), problem_size=problem_size)
    if count == 1:
        return WorkerPool(_InlinePool(task_ids, factory))
    return WorkerPool(_ThreadPool(task_ids, factory, count))


__all__ = [
    "WORKERS_ENV",
    "WorkerError",
    "WorkerPool",
    "available_cpus",
    "make_pool",
    "resolve_workers",
]
