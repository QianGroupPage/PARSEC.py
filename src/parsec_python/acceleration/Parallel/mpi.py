"""MPI context for eigenproblems whose basis is distributed over ranks by blocks of grid rows.

A basis too large for one GPU (a two-component basis of 3,000 states on a few million grid points
is hundreds of GB) is split over the GPUs of several nodes: rank ``r`` holds rows
``[start_r, stop_r)`` of every orbital and the matching rows of the Hamiltonian.  The pieces that
need all rows are small: overlap matrices of a few thousand squared, the coefficients of the
projectors, and a few rows of halo for the finite-difference stencil.  They move with MPI.

Every rank runs the same program (SPMD): the SCF driver is replicated and only the eigensolves
and the formation of the density are collective.  ``PARSEC_MPI=1`` (or ``--mpi``) turns this on;
``mpirun``/``srun`` start the ranks, one per GPU.

Arrays move through the host unless the MPI library is CUDA aware and
``PARSEC_MPI_CUDA_AWARE=1`` says to hand it device buffers.  That is for the halo rows, which are large
point-to-point messages.  The sums over the ranks (a few MB, one per operator application) stay on the host
unless ``PARSEC_MPI_CUDA_AWARE_REDUCE=1``: on Cray MPICH a GPU-buffer Allreduce of that size cost three times
what the host-staged one did (131 s against 44 s over the same 6,791 applications on 8 GPUs).
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

MPI_ENV = "PARSEC_MPI"
CUDA_AWARE_ENV = "PARSEC_MPI_CUDA_AWARE"
CUDA_AWARE_REDUCE_ENV = "PARSEC_MPI_CUDA_AWARE_REDUCE"
_MESSAGE_ELEMENTS = 1 << 26  # elements per MPI call: stays far below the 2**31 count limit


def requested() -> bool:
    return os.environ.get(MPI_ENV, "").strip().lower() in ("1", "true", "on", "yes")


class RowPartition:
    """``count`` rows split into ``size`` contiguous blocks of nearly equal length."""

    def __init__(self, count: int, size: int) -> None:
        base, extra = divmod(int(count), int(size))
        lengths = np.full(size, base, dtype=np.int64)
        lengths[:extra] += 1
        self.offsets = np.concatenate(([0], np.cumsum(lengths)))
        self.count = int(count)
        self.size = int(size)

    def bounds(self, rank: int) -> tuple[int, int]:
        return int(self.offsets[rank]), int(self.offsets[rank + 1])

    def owner(self, rows: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.offsets, rows, side="right") - 1


class MPIContext:
    """The communicator of a run and the collective operations the distributed solver needs."""

    def __init__(self, comm=None) -> None:
        from mpi4py import MPI

        self.MPI = MPI
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.rank = int(self.comm.Get_rank())
        self.size = int(self.comm.Get_size())
        node = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
        self.local_rank = int(node.Get_rank())
        self.ranks_per_node = int(node.Get_size())
        node.Free()
        self.cuda_aware = os.environ.get(CUDA_AWARE_ENV, "").strip().lower() in ("1", "true", "on", "yes")
        self.cuda_aware_reduce = os.environ.get(CUDA_AWARE_REDUCE_ENV, "").strip().lower() in ("1", "true", "on", "yes")
        self.allsum_seconds = 0.0  # time of the sums over the ranks (``allsum``), for the report

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    # ------------------------------------------------------------- reductions
    def allsum_host(self, array: np.ndarray) -> np.ndarray:
        """Sum of a host array over the ranks, in place (the array is returned)."""
        if self.size == 1:
            return array
        flat = array.reshape(-1)
        if not flat.flags.c_contiguous or not array.flags.c_contiguous:
            raise ValueError("allsum_host needs a C-contiguous array")
        for start in range(0, flat.size, _MESSAGE_ELEMENTS):
            self.comm.Allreduce(self.MPI.IN_PLACE, flat[start : start + _MESSAGE_ELEMENTS], op=self.MPI.SUM)
        return array

    def allsum(self, array: Any, backend=None) -> Any:
        """Sum over the ranks of a host or device array; a device array comes back as a device array."""
        if self.size == 1:
            return array
        started = time.perf_counter()
        try:
            return self._allsum(array, backend)
        finally:
            self.allsum_seconds += time.perf_counter() - started

    def _allsum(self, array: Any, backend) -> Any:
        if isinstance(array, np.ndarray):
            return self.allsum_host(np.ascontiguousarray(array) if not array.flags.c_contiguous else array)
        if backend is None:
            raise ValueError("a device array needs the backend that owns it")
        backend.synchronize()
        if self.cuda_aware and self.cuda_aware_reduce:
            flat = array.reshape(-1)
            for start in range(0, flat.size, _MESSAGE_ELEMENTS):
                self.comm.Allreduce(self.MPI.IN_PLACE, flat[start : start + _MESSAGE_ELEMENTS], op=self.MPI.SUM)
            backend.synchronize()
            return array
        host = np.ascontiguousarray(backend.asnumpy(array))
        self.allsum_host(host)
        return backend.asarray(host)

    def bcast(self, value: Any, root: int = 0) -> Any:
        """A small Python object or a host array from ``root`` to every rank."""
        if self.size == 1:
            return value
        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)
            for start in range(0, flat.size, _MESSAGE_ELEMENTS):
                self.comm.Bcast(flat[start : start + _MESSAGE_ELEMENTS], root=root)
            return value
        return self.comm.bcast(value, root=root)

    def allgather_last_axis(self, local: np.ndarray, partition: RowPartition) -> np.ndarray:
        """Join the pieces ``local[..., rows_of_rank]`` into the whole array on every rank."""
        local = np.ascontiguousarray(local)
        if self.size == 1:
            return local
        lead = local.shape[:-1]
        width = int(np.prod(lead, dtype=np.int64)) if lead else 1
        # gather the leading axes as "rows" of length 1 so Allgatherv concatenates the grid axis
        moved = np.ascontiguousarray(local.reshape(width, -1))
        counts = [int(c) for c in np.diff(partition.offsets)]
        displacements = [int(d) for d in partition.offsets[:-1]]
        result = np.empty((width, partition.count), dtype=local.dtype)
        for line in range(width):
            self.comm.Allgatherv(moved[line], [result[line], (counts, displacements)])
        return result.reshape(lead + (partition.count,))

    # ------------------------------------------------------------------ checks
    def check_equal(self, value: float, label: str, tolerance: float = 0.0) -> None:
        """Stop with a message instead of a deadlock when the replicated drivers have diverged."""
        if self.size == 1:
            return
        values = self.comm.allgather(float(value))
        reference = values[0]
        scale = max(abs(reference), 1.0)
        if any(abs(v - reference) > tolerance * scale for v in values):
            raise RuntimeError(
                f"the MPI ranks disagree about {label}: {values}; the replicated SCF drivers diverged"
            )

    def barrier(self) -> None:
        if self.size > 1:
            self.comm.Barrier()


_context: MPIContext | None = None


def context() -> MPIContext | None:
    """The MPI context if the run asked for MPI (``PARSEC_MPI=1``), else ``None``."""
    global _context
    if not requested():
        return None
    if _context is None:
        _context = MPIContext()
    return _context


def reset() -> None:
    """Forget the cached context (tests)."""
    global _context
    _context = None


__all__ = ["MPIContext", "MPI_ENV", "RowPartition", "context", "requested", "reset"]
