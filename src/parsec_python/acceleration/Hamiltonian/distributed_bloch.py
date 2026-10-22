"""Hamiltonians whose rows are distributed over MPI ranks (:mod:`..Parallel.mpi`).

Rank ``r`` holds the rows ``[start_r, stop_r)`` of the grid: of every orbital block, of the
finite-difference matrices, of the projector matrices and of the local fields.  Applying the
Hamiltonian to a block needs

* the rows of the neighbouring ranks that the stencil reaches (the halo): a point-to-point exchange
  with the few ranks that own them, planned once per Laplacian.  Each stencil matrix is split into the
  part on the local rows and the part on the halo rows: the exchange is started, the local part is
  applied while it runs, and the halo part is added when the rows have arrived.  The rows travel through
  the host unless the MPI library is CUDA aware and ``PARSEC_MPI_CUDA_AWARE=1`` (then straight between
  the GPUs, overlapping the local product);
* the overlaps of the block with every projector, summed over the ranks (``allsum``): the projectors
  of an atom reach rows of more than one rank.

Everything else is the single-device operator of :mod:`.device_bloch` on the local rows, so the
kernels, the real-view products and the spin-orbit algebra are the ones already tested.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from ..Parallel.mpi import MPIContext, RowPartition
from .device_bloch import DeviceBlochConverter


class HaloPlan:
    """Which rows of a block this rank sends to and receives from the others for a stencil."""

    _TAG = 4711

    def __init__(self, context: MPIContext, partition: RowPartition, needed_columns: np.ndarray) -> None:
        self.context = context
        rank = context.rank
        start, stop = partition.bounds(rank)
        self.start, self.stop = start, stop
        self.local_rows = stop - start
        outside = np.unique(needed_columns[(needed_columns < start) | (needed_columns >= stop)])
        self.halo_rows = outside.astype(np.int64)  # sorted global rows, owners ascend with them
        owners = partition.owner(self.halo_rows) if outside.size else np.empty(0, dtype=np.int64)
        self.recv_peers = [int(p) for p in np.unique(owners)]
        self.recv_slices: dict[int, slice] = {}
        for peer in self.recv_peers:
            where = np.flatnonzero(owners == peer)
            self.recv_slices[peer] = slice(int(where[0]), int(where[-1]) + 1)
        # tell every owner which of its rows we need
        requests = context.comm.alltoall(
            [
                (self.halo_rows[self.recv_slices[peer]] if peer in self.recv_slices else np.empty(0, dtype=np.int64))
                for peer in range(context.size)
            ]
        )
        self.send_peers = [peer for peer in range(context.size) if len(requests[peer])]
        self.send_slices: dict[int, slice] = {}
        indices = []
        offset = 0
        for peer in self.send_peers:
            local = np.asarray(requests[peer], dtype=np.int64) - start
            if local.min() < 0 or local.max() >= self.local_rows:
                raise RuntimeError("a rank asked for rows it does not own")
            indices.append(local)
            self.send_slices[peer] = slice(offset, offset + local.size)
            offset += local.size
        self.send_index_host = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
        self._send_index = None  # on the device, built on first use
        # what the exchanges cost this rank: seconds in begin (pack, copy, post) and in finish (wait,
        # copy back), and the bytes received
        self.seconds = {"halo_begin": 0.0, "halo_wait": 0.0}
        self.bytes_received = 0

    @property
    def halo_size(self) -> int:
        return int(self.halo_rows.size)

    def split(self, matrix: sp.csr_matrix) -> tuple[sp.csr_matrix, sp.csr_matrix | None]:
        """The rows of this rank of a stencil matrix as ``(local, halo)``: the part on the columns of the
        local rows (renumbered from 0) and the part on the halo rows (numbered as ``halo_rows``)."""
        matrix = matrix.tocsr()
        rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
        columns = matrix.indices
        inside = (columns >= self.start) & (columns < self.stop)
        local = sp.csr_matrix(
            (matrix.data[inside], (rows[inside], columns[inside] - self.start)),
            shape=(matrix.shape[0], self.local_rows),
        )
        if inside.all():
            return local, None
        outside_columns = columns[~inside]
        position = np.searchsorted(self.halo_rows, outside_columns)
        if not np.array_equal(self.halo_rows[np.minimum(position, self.halo_size - 1)], outside_columns):
            raise RuntimeError("a stencil reaches rows that the halo plan does not exchange")
        halo = sp.csr_matrix(
            (matrix.data[~inside], (rows[~inside], position)), shape=(matrix.shape[0], self.halo_size)
        )
        return local, halo

    def begin(self, backend, block):
        """Start the exchange of the halo rows of ``block`` (local rows, vectors); ``None`` if there are none."""
        if self.halo_size == 0 and not self.send_peers:
            return None
        started = time.perf_counter()
        comm = self.context.comm
        vectors = int(block.shape[1])
        gpu_direct = self.context.cuda_aware  # device buffers to MPI (host arrays for a host backend)
        if self._send_index is None:
            self._send_index = backend.asarray(self.send_index_host)
        if gpu_direct:
            # device buffers straight to MPI: the packed rows must be complete before it reads them
            xp = backend.xp
            send = block[self._send_index] if self.send_peers else xp.empty((0, vectors), dtype=xp.complex128)
            received = xp.empty((self.halo_size, vectors), dtype=xp.complex128)
            backend.synchronize()
        else:
            send = np.ascontiguousarray(backend.asnumpy(block[self._send_index])) if self.send_peers else (
                np.empty((0, vectors), dtype=np.complex128)
            )
            received = np.empty((self.halo_size, vectors), dtype=np.complex128)
        requests = []
        for peer in self.recv_peers:
            requests.append(comm.Irecv(received[self.recv_slices[peer]], source=peer, tag=self._TAG))
        for peer in self.send_peers:
            requests.append(comm.Isend(send[self.send_slices[peer]], dest=peer, tag=self._TAG))
        self.seconds["halo_begin"] += time.perf_counter() - started
        self.bytes_received += int(self.halo_size) * vectors * 16
        return (requests, received, send, gpu_direct)

    def finish(self, backend, pending):
        """The halo rows (a device array of ``halo_size`` rows) once they have arrived."""
        requests, received, _send, gpu_direct = pending
        backend.synchronize()  # the local product, which ran meanwhile, is not part of the wait
        started = time.perf_counter()
        self.context.MPI.Request.Waitall(requests)
        if gpu_direct:
            backend.synchronize()  # what MPI wrote is visible to the kernels that follow
            halo = received
        else:
            halo = backend.asarray(received)
            backend.synchronize()
        self.seconds["halo_wait"] += time.perf_counter() - started
        return halo


class DistributedBlochConverter(DeviceBlochConverter):
    """Device operators for the rows of this rank (see the module docstring)."""

    def __init__(self, backend, context: MPIContext) -> None:
        super().__init__(backend)
        self.context = context
        self._partitions: dict[int, RowPartition] = {}
        self._plans: dict[int, tuple[object, HaloPlan]] = {}
        self._active_rows = 0

    # --------------------------------------------------------------- geometry
    def partition(self, count: int) -> RowPartition:
        part = self._partitions.get(count)
        if part is None:
            part = self._partitions[count] = RowPartition(count, self.context.size)
        return part

    def _bounds(self, count: int) -> tuple[int, int]:
        return self.partition(count).bounds(self.context.rank)

    def _rows(self, host_matrix):
        start, stop = self._bounds(host_matrix.shape[0])
        return host_matrix.tocsr()[start:stop]

    def _local(self, host_array):
        start, stop = self._bounds(host_array.shape[-1])
        return np.ascontiguousarray(host_array[..., start:stop])

    def _halo_plan(self, hamiltonian) -> HaloPlan:
        laplacian = hamiltonian.negative_laplacian
        key = id(laplacian)
        entry = self._plans.get(key)
        if entry is None or entry[0] is not laplacian:
            count = laplacian.shape[0]
            start, stop = self._bounds(count)
            columns = [laplacian.tocsr()[start:stop].indices]
            gradient = getattr(hamiltonian, "_k_gradient", None)
            if gradient is not None:
                columns.append(gradient.tocsr()[start:stop].indices)
            plan = HaloPlan(self.context, self.partition(count), np.concatenate(columns))
            entry = self._plans[key] = (laplacian, plan)
        return entry[1]

    def _derivative_parts(self, hamiltonian):
        backend = self.backend
        plan = self._halo_plan(hamiltonian)
        self._plan = plan

        def stencil(host_matrix):
            """Device ``(local part, halo part)`` of this rank's rows of a stencil matrix (cached)."""

            def build():
                local, halo = plan.split(host_matrix.tocsr()[plan.start : plan.stop])
                return (
                    backend.matrix(local, dtype=np.float64),
                    None if halo is None else backend.matrix(halo, dtype=np.float64),
                )

            return self._cached(host_matrix, build)

        laplacian, laplacian_halo = stencil(hamiltonian.negative_laplacian)
        host_gradient = getattr(hamiltonian, "_k_gradient", None)
        k_gradient = k_gradient_halo = None
        if host_gradient is not None:
            k_gradient, k_gradient_halo = stencil(host_gradient)
        self._halo_matrices = (laplacian_halo, k_gradient_halo)
        return laplacian, k_gradient

    def communication(self) -> dict[str, float]:
        """Seconds this rank spent in the halo exchanges (beginning, waiting), the bytes it received and
        the seconds in the sums over the ranks."""
        totals = {"halo_begin": 0.0, "halo_wait": 0.0, "halo_bytes": 0.0}
        for _, plan in self._plans.values():
            totals["halo_begin"] += plan.seconds["halo_begin"]
            totals["halo_wait"] += plan.seconds["halo_wait"]
            totals["halo_bytes"] += float(plan.bytes_received)
        totals["allsum"] = self.context.allsum_seconds
        return totals

    def _operator_extras(self) -> dict:
        plan = self._plan
        backend = self.backend
        laplacian_halo, k_gradient_halo = self._halo_matrices
        return {
            "halo_begin": lambda block: plan.begin(backend, block),
            "halo_finish": lambda pending: plan.finish(backend, pending),
            "laplacian_halo": laplacian_halo,
            "k_gradient_halo": k_gradient_halo,
            "allsum": lambda array: self.context.allsum(array, backend),
            "mpi": self.context,
            "row_bounds": (plan.start, plan.stop),
        }


__all__ = ["DistributedBlochConverter", "HaloPlan"]
