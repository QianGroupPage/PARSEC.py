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
``process``
    forked processes, one worker object each: scales on CPU nodes and keeps a
    worker's memory private.  Fork inherits the prepared system, so nothing
    large is pickled.  Not usable after a CUDA context exists in the parent.
``thread``
    threads of this process, one worker object each: the layout for GPUs
    (one thread per device; kernels and cuBLAS calls release the GIL), as in
    the multi-GPU solver of the reference branch.

Task ids are dealt round-robin to the workers so each gets a similar share.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import traceback
from multiprocessing.connection import wait
from typing import Any, Callable, Iterable, Iterator, Sequence

from .blas import set_blas_threads

WORKERS_ENV = "PARSEC_KPOINT_WORKERS"

_IN_WORKER = False  # set in forked children: pools never nest


class WorkerError(RuntimeError):
    """An exception raised inside a pool worker (with its traceback text)."""


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not Linux
        return os.cpu_count() or 1


def fork_available() -> bool:
    return "fork" in mp.get_all_start_methods()


def resolve_workers(
    requested: int | str | None,
    n_tasks: int,
    *,
    problem_size: int | None = None,
    minimum_problem_size: int = 4000,
) -> int:
    """Number of CPU worker processes for ``n_tasks`` independent tasks.

    ``requested`` is an integer, ``"auto"`` or ``None`` (then the environment
    variable ``PARSEC_KPOINT_WORKERS``, default ``auto``).  ``auto`` runs
    serially for a single task, for problems too small to repay the start-up
    (``problem_size`` = grid points, below ``minimum_problem_size``), without
    ``fork``, and inside a worker; otherwise it uses half the available CPUs
    (the sparse operators are memory-bandwidth bound, so more processes than
    that gain little), at most one per task.
    """

    if requested is None:
        requested = os.environ.get(WORKERS_ENV, "auto")
    if isinstance(requested, str):
        text = requested.strip().lower()
        requested = "auto" if text in ("", "auto") else int(text)
    if _IN_WORKER or not fork_available() or n_tasks < 2:
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


def _serve_process(conn, factory, tasks, index, blas_threads) -> None:
    """Worker-process loop: build the worker object, then answer commands."""
    global _IN_WORKER
    _IN_WORKER = True
    if blas_threads:
        set_blas_threads(blas_threads)
    worker = None
    failure = None
    try:
        worker = factory(tasks, index)
    except BaseException:  # reported on the first command
        failure = traceback.format_exc()
    while True:
        try:
            message = conn.recv()
        except EOFError:
            break
        if message is None:
            break
        method, args, kwargs = message
        if failure is not None:
            conn.send(("error", failure))
            continue
        try:
            for item in getattr(worker, method)(*args, **kwargs):
                conn.send(("item", item))
            conn.send(("done", None))
        except BaseException:
            conn.send(("error", traceback.format_exc()))
    conn.close()


class _ProcessPool:
    kind = "process"

    def __init__(self, task_ids, factory, workers: int, blas_threads: int | None):
        context = mp.get_context("fork")
        self.n_workers = workers
        self._connections = []
        self._processes = []
        for index, chunk in enumerate(_partition(task_ids, workers)):
            parent, child = context.Pipe()
            process = context.Process(
                target=_serve_process,
                args=(child, factory, chunk, index, blas_threads),
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

    def stream(self, method: str, *args, **kwargs) -> Iterator[tuple[Any, Any]]:
        for connection in self._connections:
            connection.send((method, args, kwargs))
        active = list(self._connections)
        while active:
            for connection in wait(active):
                try:
                    kind, payload = connection.recv()
                except EOFError:
                    self.close()
                    raise WorkerError("a pool worker exited unexpectedly") from None
                if kind == "item":
                    yield payload
                elif kind == "done":
                    active.remove(connection)
                else:
                    self.close()
                    raise WorkerError(f"pool worker failed:\n{payload}")

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.send(None)
            except (OSError, ValueError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass
        self._connections = []
        self._processes = []


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
    kind: str = "process",
    problem_size: int | None = None,
    blas_threads: int | str | None = "auto",
) -> WorkerPool:
    """Build a pool over ``task_ids``.

    ``kind="process"`` resolves the worker count with :func:`resolve_workers`
    (one worker means the inline backend); ``kind="thread"`` takes an explicit
    ``workers`` count (e.g. the number of GPUs).  ``blas_threads="auto"`` gives
    every worker process ``cpus / workers`` BLAS threads.
    """

    task_ids = list(task_ids)
    if kind == "thread":
        count = max(1, min(int(workers or 1), len(task_ids)))
        if count == 1:
            return WorkerPool(_InlinePool(task_ids, factory))
        return WorkerPool(_ThreadPool(task_ids, factory, count))
    if kind != "process":
        raise ValueError(f"unknown pool kind {kind!r}")
    count = resolve_workers(workers, len(task_ids), problem_size=problem_size)
    if count == 1:
        return WorkerPool(_InlinePool(task_ids, factory))
    if blas_threads == "auto":
        blas_threads = max(1, available_cpus() // count)
    return WorkerPool(_ProcessPool(task_ids, factory, count, blas_threads))


__all__ = [
    "WORKERS_ENV",
    "WorkerError",
    "WorkerPool",
    "available_cpus",
    "fork_available",
    "make_pool",
    "resolve_workers",
]
