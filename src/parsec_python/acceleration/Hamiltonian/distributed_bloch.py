"""Hamiltonians whose rows are distributed over MPI ranks (:mod:`..Parallel.mpi`).

Rank ``r`` holds the rows ``[start_r, stop_r)`` of the grid: of every orbital block, of the
finite-difference matrices, of the projector matrices and of the local fields.  Applying the
Hamiltonian to a block needs

* the rows of the neighbouring ranks that the stencil reaches (``halo``): a point-to-point exchange
  with the few ranks that own them, planned once per Laplacian;
* the overlaps of the block with every projector, summed over the ranks (``allsum``): the projectors
  of an atom reach rows of more than one rank.

Everything else is the single-device operator of :mod:`.device_bloch` on the local rows, so the
kernels, the real-view products and the spin-orbit algebra are the ones already tested.
"""

from __future__ import annotations

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

    @property
    def halo_size(self) -> int:
        return int(self.halo_rows.size)

    def remap(self, matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Rows of this rank with columns renumbered: local rows first, then the halo rows."""
        columns = matrix.indices
        inside = (columns >= self.start) & (columns < self.stop)
        position = np.searchsorted(self.halo_rows, columns)
        outside_columns = columns[~inside]
        if outside_columns.size and not np.array_equal(
            self.halo_rows[position[~inside]], outside_columns
        ):
            raise RuntimeError("a stencil reaches rows that the halo plan does not exchange")
        renumbered = np.where(inside, columns - self.start, self.local_rows + position)
        return sp.csr_matrix(
            (matrix.data, renumbered, matrix.indptr),
            shape=(matrix.shape[0], self.local_rows + self.halo_size),
        )

    def extend(self, backend, block):
        """``block`` (local rows, vectors) with the halo rows appended below."""
        if self.halo_size == 0 and not self.send_peers:
            return block
        xp = backend.xp
        comm = self.context.comm
        vectors = int(block.shape[1])
        if self._send_index is None:
            self._send_index = backend.asarray(self.send_index_host)
        if self.send_peers:
            send = backend.asnumpy(block[self._send_index])
            send = np.ascontiguousarray(send)
        else:
            send = np.empty((0, vectors), dtype=np.complex128)
        received = np.empty((self.halo_size, vectors), dtype=np.complex128)
        requests = []
        for peer in self.recv_peers:
            requests.append(comm.Irecv(received[self.recv_slices[peer]], source=peer, tag=self._TAG))
        for peer in self.send_peers:
            requests.append(comm.Isend(send[self.send_slices[peer]], dest=peer, tag=self._TAG))
        self.context.MPI.Request.Waitall(requests)
        extended = xp.empty((self.local_rows + self.halo_size, vectors), dtype=xp.complex128)
        extended[: self.local_rows] = block
        if self.halo_size:
            extended[self.local_rows :] = backend.asarray(received)
        return extended


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
        laplacian = self._cached(
            hamiltonian.negative_laplacian,
            lambda: backend.matrix(
                plan.remap(hamiltonian.negative_laplacian.tocsr()[plan.start : plan.stop]),
                dtype=np.float64,
            ),
        )
        host_gradient = getattr(hamiltonian, "_k_gradient", None)
        k_gradient = None
        if host_gradient is not None:
            k_gradient = self._cached(
                host_gradient,
                lambda: backend.matrix(
                    plan.remap(host_gradient.tocsr()[plan.start : plan.stop]), dtype=np.float64
                ),
            )
        return laplacian, k_gradient

    def _operator_extras(self) -> dict:
        plan = self._plan
        backend = self.backend
        return {
            "halo": lambda block: plan.extend(backend, block),
            "allsum": lambda array: self.context.allsum(array, backend),
            "mpi": self.context,
            "row_bounds": (plan.start, plan.stop),
        }


__all__ = ["DistributedBlochConverter", "HaloPlan"]
