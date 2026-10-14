"""Optional halo overlap and validated fast path for experimental MPI replay.

No production backend uses this module.  The default retains collective
checks.  ``collective_checks=False`` is an explicit caller contract: all
ranks invoke the same operations/column widths in the same order, use only
prevalidated widths, and an outer runner calls ``MPI.Abort`` on ANY error.
It does not relax the Hamiltonian or Ritz accuracy checks.

Fast mode follows normal CuPy asynchronous return semantics.  Synchronize
the current stream before stopping a benchmark timer or consuming results
outside that stream.  Every MPI send/reduction explicitly synchronizes its
GPU producers before passing device buffers to MPI.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .mpi_domain import CollectiveDomainError, DistributedHamiltonian, _CUDA


def _subset_cuda():
    """Use exactly the frozen base arithmetic with a row indirection."""
    declaration = 'extern "C" __global__ void domain_apply(\n'
    row = ' int row=blockIdx.x*blockDim.x+threadIdx.x;'
    if _CUDA.count(declaration) != 1 or _CUDA.count(row) != 1:
        raise RuntimeError('base CUDA kernel ABI changed; subset kernel requires review')
    return _CUDA.replace(declaration, 'extern "C" __global__ void domain_subset(\n'
                         ' const int* selected_rows, int selected_count,\n').replace(
        row, ' int selected=blockIdx.x*blockDim.x+threadIdx.x;\n'
             ' if(selected>=selected_count) return;\n'
             ' int row=selected_rows[selected];')


@dataclass
class _PendingHalo:
    packed: dict[int, Any]
    receive: Any
    requests: list[Any]
    started: float


class OverlappedDistributedHamiltonian(DistributedHamiltonian):
    """Optional interior/boundary overlap without changing any row sum.

    KB coefficients are fully reduced before starting halo exchange.  Each
    interior row then evaluates its entire kinetic/local/nonlocal action
    while halo requests are live.  Boundary rows execute after receive
    completion.  This deliberately avoids splitting one row's summation or
    adding nonblocking collective ordering requirements.

    ``overlap=False`` isolates the benefit/cost of the checks policy while
    retaining the original base apply kernel.  ``validated_widths`` can
    list all replay widths at construction, e.g. ``(6, 48, 96)``.  Otherwise
    fast mode collectively validates its first width.  New later widths
    require an explicit collective ``validate_width(width)`` call outside
    the timed region.  A vector and an N-by-1 matrix share width one; the
    caller must nevertheless preserve the same dimensionality on all ranks.
    """

    def __init__(self, *args, overlap=True, collective_checks=True,
                 validated_widths=None, **kwargs):
        self._constructing = True
        self.collective_checks = bool(collective_checks)
        self.overlap = bool(overlap)
        self._validated_widths = set()
        super().__init__(*args, **kwargs)

        def prepare():
            interior = np.all(self.domain.neighbors < len(self.rows), axis=0)
            self.interior_rows = np.flatnonzero(interior).astype(np.int32)
            self.boundary_rows = np.flatnonzero(~interior).astype(np.int32)
            self.device_interior = self.xp.asarray(self.interior_rows)
            self.device_boundary = self.xp.asarray(self.boundary_rows)
            if self.gpu:
                self.subset_kernel = self.xp.RawKernel(_subset_cuda(), 'domain_subset')
                self.subset_kernel.compile()
            values = () if validated_widths is None else tuple(validated_widths)
            if any(int(w) != w or w < 1 for w in values):
                raise ValueError('validated_widths must contain positive integers')
            return tuple(sorted(set(map(int, values))))

        widths = self._guard(prepare, 'overlap setup')
        settings = self.comm.allgather((self.overlap, self.collective_checks, widths))
        if len(set(settings)) != 1:
            raise CollectiveDomainError('ranks disagree on overlap/check/width configuration')
        self._validated_widths.update(widths)
        self._constructing = False
        self.stats.update(overlap_applications=0, halo_wait_seconds=0.0,
                          interior_rows=len(self.interior_rows),
                          boundary_rows=len(self.boundary_rows))

    def _guard(self, function, stage):
        if self._constructing or self.collective_checks:
            return super()._guard(function, stage)
        # Deliberately no implicit synchronization/collective in this policy.
        # Communication boundaries below provide the required producer sync.
        return function()

    def validate_width(self, width):
        """Collectively register a width; call this on EVERY rank."""
        def check():
            if int(width) != width or width < 1:
                raise ValueError('validated width must be a positive integer')
            return int(width)

        value = DistributedHamiltonian._guard(self, check, 'width validation')
        widths = self.comm.allgather(value)
        if len(set(widths)) != 1:
            raise CollectiveDomainError('ranks disagree on validated width')
        self._validated_widths.add(value)

    def _columns(self, vectors):
        if self._constructing or self.collective_checks:
            return super()._columns(vectors)
        if np.iscomplexobj(vectors):
            raise ValueError('only real FP64 vectors are supported')
        x = self.xp.asarray(vectors, dtype=self.xp.float64)
        was_vector = x.ndim == 1
        if was_vector:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != len(self.rows) or x.shape[1] < 1:
            raise ValueError('vectors must have shape (local_rows, columns)')
        width = int(x.shape[1])
        if not self._validated_widths:
            self.validate_width(width)
        elif width not in self._validated_widths:
            raise ValueError(f'width {width} was not validated; call validate_width collectively')
        return x, was_vector

    def _sum(self, values):
        if self._constructing or self.collective_checks:
            return super()._sum(values)
        packed = self.xp.ascontiguousarray(values)
        self._sync()
        if self.transport == 'cuda':
            total = self.xp.empty_like(packed)
            self.comm.Allreduce(packed, total)
            return total
        host = self._host(packed)
        total = np.empty_like(host)
        self.comm.Allreduce(host, total)
        return self.xp.asarray(total)

    def _begin_exchange(self, x):
        width = int(x.shape[1])

        def pack():
            if self._width != width:
                self._ghosts = self.xp.empty((len(self.domain.ghost_rows), width),
                                            dtype=self.xp.float64, order='C')
                self._width = width
            packed = {peer: self.xp.ascontiguousarray(x[index, :])
                      for peer, index in self.send_indices.items()}
            if self.transport == 'host':
                packed = {peer: self._host(value) for peer, value in packed.items()}
                receive = np.empty(self._ghosts.shape, dtype=np.float64)
            else:
                receive = self._ghosts
            return packed, receive

        packed, receive = self._guard(pack, 'halo packing')
        # Includes prior operations producing x and all asynchronous pack kernels.
        self._sync()
        started = perf_counter()
        requests = []
        for peer, part in self.domain.receive_slices.items():
            requests.append(self.comm.Irecv(receive[part, :], source=peer, tag=41))
        for peer, value in packed.items():
            requests.append(self.comm.Isend(value, dest=peer, tag=41))
        return _PendingHalo(packed, receive, requests, started)

    def _finish_exchange(self, pending):
        started = perf_counter()
        for request in pending.requests:
            request.Wait()
        self.stats['halo_wait_seconds'] += perf_counter() - started

        def unpack():
            if self.transport == 'host':
                self._ghosts[...] = self.xp.asarray(pending.receive)
            return self._ghosts

        ghosts = self._guard(unpack, 'halo unpacking')
        self.stats['halo_seconds'] += perf_counter() - pending.started
        self.stats['halo_send_bytes'] += sum(v.nbytes for v in pending.packed.values())
        self.stats['halo_receive_bytes'] += int(pending.receive.nbytes)
        return ghosts

    def _exchange(self, x):
        # Used by the unchanged base apply when overlap=False.  It also
        # makes fast-mode send readiness explicit instead of relying on _guard.
        return self._finish_exchange(self._begin_exchange(x))

    def _apply_rows(self, x, coefficients, out, rows, device_rows):
        if len(rows) == 0:
            return
        if not self.gpu:
            block = np.zeros((len(rows), x.shape[1]), dtype=np.float64)
            n = len(self.rows)
            for slot, code in zip(self.domain.neighbors[:, rows], self.domain.codes[:, rows]):
                local, remote = (slot >= 0) & (slot < n), slot >= n
                block[local] += self.domain.palette[code[local], None] * x[slot[local]]
                block[remote] += self.domain.palette[code[remote], None] * self._ghosts[slot[remote]-n]
            block += self.potential[rows, None] * x[rows]
            if self.projector_count:
                block += self.host_projectors[rows, :] @ coefficients
            out[rows] = block
            return
        self.subset_kernel(((len(rows)+255)//256, (x.shape[1]+5)//6), (256,),
            (device_rows, np.int32(len(rows)), np.int32(len(self.rows)),
             np.int32(self.neighbors.shape[0]), np.int32(x.shape[1]),
             self.neighbors, self.codes, self.palette, self.potential, x,
             np.int64(x.strides[0]//8), np.int64(x.strides[1]//8), self._ghosts,
             self.bp, self.bj, self.bv, coefficients, np.int32(self.projector_count), out))

    def _projector_coefficients(self, x):
        """Return signed KB coefficients; default keeps the full MPI sum."""
        if self.projector_count:
            partial = self._guard(lambda: self.projector_transpose @ x,
                                  'local projector projection')
            coefficients = self._sum(partial) * self.signs[:, None]
            self.stats['projector_reduce_input_bytes'] += int(partial.nbytes)
            return coefficients
        return self.xp.empty((0, x.shape[1]), dtype=self.xp.float64)

    def apply(self, vectors):
        if not self.overlap:
            return super().apply(vectors)
        started = perf_counter()
        x, was_vector = self._columns(vectors)
        coefficients = self._projector_coefficients(x)

        def buffers():
            return (self.xp.ascontiguousarray(coefficients),
                    self.xp.empty(x.shape, dtype=self.xp.float64, order='F'))

        coefficients, out = self._guard(buffers, 'overlap output allocation')
        pending = self._begin_exchange(x)
        # Keep send/receive buffers alive while the independent row kernel runs.
        # MPI.Wait then drives MPI progress while CUDA executes asynchronously.
        try:
            self._guard(lambda: self._apply_rows(x, coefficients, out,
                        self.interior_rows, self.device_interior), 'interior Hamiltonian')
        finally:
            self._finish_exchange(pending)
        self._guard(lambda: self._apply_rows(x, coefficients, out,
                    self.boundary_rows, self.device_boundary), 'boundary Hamiltonian')
        self.stats['applications'] += 1
        self.stats['overlap_applications'] += 1
        self.stats['apply_seconds'] += perf_counter() - started
        return out[:, 0] if was_vector else out


__all__ = ['OverlappedDistributedHamiltonian']
