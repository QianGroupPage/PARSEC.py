"""A communicator of threads with the part of the mpi4py interface the k-point pools use (tests only).

``run_ranks(size, function)`` calls ``function(context)`` in ``size`` threads, each with a
:class:`ThreadContext` (``comm``, ``rank``, ``size``, ``allsum_host``, ``bcast``, ``check_equal``, as
:class:`parsec_python.acceleration.Parallel.mpi.MPIContext`), and returns the results in rank order.  The
collectives are those of MPI on host arrays: ``Allreduce`` (sums in rank order, the same on every rank),
``Bcast``, ``Alltoallv`` (``[buffer, (counts, displacements)]``), ``allgather``, ``bcast``, ``Split``.
A failing rank breaks the barriers of the others, so a test fails instead of hanging.  MPI ranks are
processes; threads share one BLAS, which must then be single-threaded (see :func:`run_ranks`).
"""

from __future__ import annotations

import copy
import threading

import numpy as np

_TIMEOUT = 600.0


class _Board:
    def __init__(self, size: int) -> None:
        self.size = size
        self.barrier = threading.Barrier(size, timeout=_TIMEOUT)
        self.slots: list = [None] * size
        self.lock = threading.Lock()
        self.children: dict = {}


class ThreadComm:
    def __init__(self, board: _Board, rank: int) -> None:
        self.board = board
        self.rank = rank
        self.size = board.size
        self._splits = 0

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size

    def _post(self, value) -> list:
        board = self.board
        board.slots[self.rank] = value
        board.barrier.wait()
        values = list(board.slots)
        board.barrier.wait()
        return values

    # ----- objects
    def allgather(self, value):
        return [copy.deepcopy(item) for item in self._post(value)]

    def bcast(self, value, root: int = 0):
        return copy.deepcopy(self._post(value if self.rank == root else None)[root])

    def Barrier(self) -> None:
        self.board.barrier.wait()

    # ----- buffers
    def Allreduce(self, send, receive, op=None) -> None:
        values = self._post(np.array(send, copy=True))
        total = np.array(values[0], copy=True)
        for value in values[1:]:
            total = total + value
        receive[...] = total

    def Reduce(self, send, receive, op=None, root: int = 0) -> None:
        values = self._post(np.array(send, copy=True))
        if self.rank == root:
            total = np.array(values[0], copy=True)
            for value in values[1:]:
                total = total + value
            receive[...] = total

    def Bcast(self, buffer, root: int = 0) -> None:
        values = self._post(np.array(buffer, copy=True) if self.rank == root else None)
        if self.rank != root:
            buffer[...] = values[root]

    def Alltoallv(self, send_spec, receive_spec) -> None:
        send, (send_counts, send_offsets) = send_spec[0], send_spec[1]
        receive, (receive_counts, receive_offsets) = receive_spec[0], receive_spec[1]
        values = self._post((np.array(send, copy=True), np.asarray(send_counts), np.asarray(send_offsets)))
        for source, (data, counts, offsets) in enumerate(values):
            count = int(receive_counts[source])
            if int(counts[self.rank]) != count:
                raise RuntimeError(f"Alltoallv: rank {source} sends {counts[self.rank]} to {self.rank}, which expects {count}")
            start = int(offsets[self.rank])
            target = int(receive_offsets[source])
            receive[target : target + count] = data[start : start + count]

    def Split(self, color: int, key: int = 0) -> "ThreadComm":
        values = self._post((int(color), int(key)))
        self._splits += 1
        members = sorted((r for r in range(self.size) if values[r][0] == color), key=lambda r: (values[r][1], r))
        with self.board.lock:
            board = self.board.children.setdefault((self._splits, color), _Board(len(members)))
        self.board.barrier.wait()
        return ThreadComm(board, members.index(self.rank))

    def Free(self) -> None:
        pass


class ThreadContext:
    """What the pools read of :class:`parsec_python.acceleration.Parallel.mpi.MPIContext`."""

    def __init__(self, comm: ThreadComm) -> None:
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.local_rank = comm.rank
        self.ranks_per_node = comm.size
        self.cuda_aware = False
        self.cuda_aware_reduce = False

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def allsum_host(self, array: np.ndarray) -> np.ndarray:
        total = np.empty_like(array)
        self.comm.Allreduce(array, total)
        array[...] = total
        return array

    def bcast(self, value, root: int = 0):
        return self.comm.bcast(value, root=root)

    def check_equal(self, value: float, label: str, tolerance: float = 0.0) -> None:
        values = self.comm.allgather(float(value))
        scale = max(abs(values[0]), 1.0)
        if any(abs(v - values[0]) > tolerance * scale for v in values):
            raise RuntimeError(f"the ranks disagree about {label}: {values}")

    def barrier(self) -> None:
        self.comm.Barrier()


def run_ranks(size: int, function) -> list:
    """``[function(context of rank r) for r in range(size)]``, each rank in its own thread.

    The BLAS of the process must be single-threaded (``OPENBLAS_NUM_THREADS=1`` before NumPy is
    imported, as :mod:`.pool_check` sets it): the OpenBLAS of the Windows wheels hangs or crashes when
    threads of its own pool serve several calling threads at once (QR and ``eigh`` from three threads).
    """
    board = _Board(size)
    results: list = [None] * size
    errors: list = [None] * size

    def run(rank: int) -> None:
        try:
            results[rank] = function(ThreadContext(ThreadComm(board, rank)))
        except BaseException as error:  # noqa: BLE001 - reported below
            errors[rank] = error
            board.barrier.abort()
            for child in list(board.children.values()):
                child.barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    first = next((error for error in errors if error is not None and not isinstance(error, threading.BrokenBarrierError)), None)
    if first is None:
        first = next((error for error in errors if error is not None), None)
    if first is not None:
        raise first
    return results


__all__ = ["ThreadComm", "ThreadContext", "run_ranks"]
