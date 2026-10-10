"""Experimental orbital-column distribution and row/column redistribution.

Each rank holds all grid rows for only its contiguous orbital columns.
Hamiltonian applications therefore communicate nothing between ranks, at
the explicit cost of replicated Hamiltonian metadata and potentials.
One packed FP64 Alltoallv changes to contiguous row ownership for Gram/Ritz
work; another changes back.  No rank gathers the complete orbital matrix.

GPU results follow CuPy stream semantics.  Every MPI boundary synchronizes
its device producers, and callers synchronize before timing GPU completion.
This module is a replay primitive, not a production SCF/eigensolver backend.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.sparse as sp

from .mpi_domain import CollectiveDomainError, SerialComm


class _SerialLayoutComm(SerialComm):
    def Alltoallv(self, send, receive):
        receive[0][...] = send[0]


def balanced_ranges(length, parts):
    """Return deterministic contiguous counts/offsets, allowing empty tails."""
    if int(length) != length or length < 0 or int(parts) != parts or parts < 1:
        raise ValueError('length must be nonnegative and parts positive integers')
    quotient, remainder = divmod(int(length), int(parts))
    counts = np.full(parts, quotient, dtype=np.int64)
    counts[:remainder] += 1
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    return counts, offsets


def _counts_displacements(counts):
    counts = np.asarray(counts, dtype=np.int64)
    displacements = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts[:-1])))
    # Portable MPI_Alltoallv uses C-int counts and displacements.  Do not
    # silently overflow when a future larger replay needs chunked exchange.
    if np.any(counts < 0) or np.any(counts > np.iinfo(np.int32).max) or np.any(
            displacements > np.iinfo(np.int32).max):
        raise ValueError('Alltoallv count/displacement exceeds int32; use smaller orbital batches')
    return counts.astype(np.int32), displacements.astype(np.int32)


class ColumnLayout:
    """Persistent local-column Hamiltonian plus collective layout exchanges.

    ``global_states`` is the total column count, not a per-rank count.
    Public array shapes are ``column_shape=(N, local_columns)`` and
    ``row_shape=(local_rows, M)``.  Orders within both global axes are kept.

    Setup and redistribution failures are propagated collectively where
    possible.  Local ``apply``/recurrence failures are intentionally local:
    a real MPI runner must abort the communicator on any unexpected error.
    All ranks must call redistributions in the same order.  Instances are
    not reentrant.  The original communicator is duplicated when supported.
    """

    def __init__(self, metadata, local_potential, projectors, signs,
                 global_states, *, comm=None, xp=np, transport='host'):
        incoming = _SerialLayoutComm() if comm is None else comm
        self._owns_comm = hasattr(incoming, 'Dup')
        self.comm = incoming.Dup() if self._owns_comm else incoming
        self.rank, self.size = self.comm.rank, self.comm.size
        self.xp, self.gpu, self.transport = xp, xp is not np, transport
        self.stats = dict(local_h_applications=0, column_to_row_calls=0,
                          row_to_column_calls=0, redistribution_send_bytes=0,
                          redistribution_receive_bytes=0,
                          redistribution_off_rank_send_bytes=0,
                          redistribution_off_rank_receive_bytes=0,
                          redistribution_seconds=0.0)
        if type(self.comm).__module__.startswith('mpi4py'):
            from mpi4py import MPI
            self._double = MPI.DOUBLE
        else:
            self._double = None  # Serial/simulated communicators ignore datatype.

        def setup():
            if transport not in ('host', 'cuda') or transport == 'cuda' and not self.gpu:
                raise ValueError('transport must be host, or cuda with CuPy')
            self.shape = tuple(map(int, metadata.shape))
            if self.shape[0] < 1 or self.shape[0] != self.shape[1]:
                raise ValueError('Hamiltonian metadata must be positive and square')
            if int(global_states) != global_states or global_states < 1:
                raise ValueError('global_states must be a positive integer')
            self.global_states = int(global_states)
            self.global_rows = self.shape[0]
            self.column_counts, self.column_offsets = balanced_ranges(self.global_states, self.size)
            self.row_counts, self.row_offsets = balanced_ranges(self.global_rows, self.size)
            self.column_start, self.column_stop = map(int, self.column_offsets[self.rank:self.rank+2])
            self.row_start, self.row_stop = map(int, self.row_offsets[self.rank:self.rank+2])
            self.local_column_count = self.column_stop-self.column_start
            self.local_row_count = self.row_stop-self.row_start
            self.column_shape = (self.global_rows, self.local_column_count)
            self.row_shape = (self.local_row_count, self.global_states)
            self._to_row_send = _counts_displacements(self.row_counts*self.local_column_count)
            self._to_row_receive = _counts_displacements(self.local_row_count*self.column_counts)
            self._to_column_send = _counts_displacements(self.local_row_count*self.column_counts)
            self._to_column_receive = _counts_displacements(self.row_counts*self.local_column_count)
            v = np.asarray(local_potential, dtype=np.float64)
            b = sp.csr_matrix(projectors, dtype=np.float64, copy=True)
            b.sum_duplicates()
            b.sort_indices()
            sg = np.asarray(signs, dtype=np.float64)
            if (v.shape != (self.global_rows,) or b.shape[0] != self.global_rows
                    or sg.shape != (b.shape[1],)):
                raise ValueError('potential/projector dimensions do not match Hamiltonian')
            if not all(np.all(np.isfinite(x)) for x in (v, b.data, sg)):
                raise ValueError('Hamiltonian data must be finite')
            self.projector_count = b.shape[1]
            if self.gpu:
                from ..backends.cupy import CuPyHamiltonian
                self.local_operator = CuPyHamiltonian(
                    None, v, (b, sg), retain_generic_laplacian=False,
                    finite_difference_metadata=metadata)
            else:
                self.kinetic = metadata.to_csr()
                self.potential, self.projectors, self.signs = v, b, sg
                self.projector_transpose = b.T.tocsr()

        self._guard(setup, 'column-layout setup')
        signatures = self.comm.allgather((self.shape, self.global_states,
                                          self.projector_count, transport))
        if len(set(signatures)) != 1:
            raise CollectiveDomainError('ranks disagree on orbital layout dimensions/transport')

    def _sync(self):
        if self.gpu:
            self.xp.cuda.get_current_stream().synchronize()

    def _host(self, values):
        return self.xp.asnumpy(values) if self.gpu else np.asarray(values)

    def _guard(self, function, stage):
        error, result = None, None
        try:
            result = function()
            self._sync()
        except Exception as exc:
            error = f'rank {self.rank}: {type(exc).__name__}: {exc}'
        errors = self.comm.allgather(error)
        if any(x is not None for x in errors):
            raise CollectiveDomainError(stage+': '+'; '.join(x for x in errors if x is not None))
        return result

    def _matrix(self, values, shape, *, output=False):
        if np.iscomplexobj(values):
            raise ValueError('orbital layouts require real FP64 values')
        matrix = self.xp.asarray(values, dtype=self.xp.float64)
        if matrix.shape != shape:
            raise ValueError(f'orbital matrix shape {matrix.shape} does not match {shape}')
        if output:
            if matrix is not values or not matrix.flags.f_contiguous:
                raise ValueError('output must be an existing Fortran-contiguous FP64 array')
            return matrix
        return self.xp.asfortranarray(matrix)

    def _column_block(self, values):
        raw = self.xp.asarray(values)
        was_vector = raw.ndim == 1
        if was_vector:
            if self.local_column_count != 1:
                raise ValueError('vector form requires exactly one local orbital')
            raw = raw[:, None]
        return self._matrix(raw, self.column_shape), was_vector

    def apply(self, local_columns):
        """Apply local full-row H, with zero world/sector MPI operations."""
        x, was_vector = self._column_block(local_columns)
        if self.local_column_count == 0:
            result = self.xp.empty(self.column_shape, dtype=self.xp.float64, order='F')
        elif self.gpu:
            result = self.local_operator.apply(x)
        else:
            result = self.kinetic @ x + self.potential[:, None]*x
            if self.projector_count:
                coefficients = self.signs[:, None]*(self.projector_transpose @ x)
                result += self.projectors @ coefficients
        self.stats['local_h_applications'] += 1
        return result[:, 0] if was_vector else result

    def __matmul__(self, local_columns):
        return self.apply(local_columns)

    def chebyshev_recurrence(self, local_columns, *, center, scale,
                             sigma_next=1.0, previous=None, sigma=0.0):
        """Use the production fused GPU recurrence, without MPI."""
        x, was_vector = self._column_block(local_columns)
        old = None if previous is None else self._column_block(previous)[0]
        if self.local_column_count and self.gpu:
            result = self.local_operator.chebyshev_recurrence(
                x, center=center, scale=scale, sigma_next=sigma_next,
                previous=old, sigma=sigma)
            self.stats['local_h_applications'] += 1
        else:
            result = (self.apply(x)-float(center)*x)*float(scale)
            if old is not None:
                result -= float(sigma)*old
            result *= float(sigma_next)
        return result[:, 0] if was_vector else result

    def _exchange(self, send, receive, send_layout, receive_layout):
        send_counts, send_offsets = send_layout
        receive_counts, receive_offsets = receive_layout

        def buffers():
            if self.transport == 'cuda':
                return send, receive
            return self._host(send), np.empty(receive.size, dtype=np.float64)

        outgoing, incoming = self._guard(buffers, 'redistribution buffer preparation')
        # _guard synchronizes device producers, including the packing kernels.
        self.comm.Alltoallv([outgoing, send_counts, send_offsets, self._double],
                            [incoming, receive_counts, receive_offsets, self._double])
        if self.transport == 'host':
            self._guard(lambda: self.xp.copyto(receive, self.xp.asarray(incoming)),
                        'redistribution host upload')
        self.stats['redistribution_send_bytes'] += int(send.size)*8
        self.stats['redistribution_receive_bytes'] += int(receive.size)*8
        self.stats['redistribution_off_rank_send_bytes'] += (int(send.size)-int(send_counts[self.rank]))*8
        self.stats['redistribution_off_rank_receive_bytes'] += (int(receive.size)-int(receive_counts[self.rank]))*8

    def columns_to_rows(self, local_columns, *, out=None):
        """One Alltoallv: ``(N, local M)`` to ``(local N, M)``.

        Destination-major packing uses Fortran order within each block, so
        received source-column blocks land directly in the final row-layout
        matrix without another complete unpack buffer.

        With one rank the layouts are identical.  A validated FP64
        Fortran-contiguous input is returned by reference when out is None;
        mutating that result also mutates the input.  An explicit out receives
        a copy.  Other inputs retain the usual conversion to FP64 Fortran
        storage.  This identity path performs no MPI.
        """
        started = perf_counter()
        if self.size == 1:
            result = self._identity_layout(local_columns, out)
            self.stats['column_to_row_calls'] += 1
            self.stats['redistribution_seconds'] += perf_counter()-started
            return result

        def pack():
            x = self._matrix(local_columns, self.column_shape)
            result = (self.xp.empty(self.row_shape, dtype=self.xp.float64, order='F')
                      if out is None else self._matrix(out, self.row_shape, output=True))
            send = self.xp.empty(x.size, dtype=self.xp.float64)
            counts, offsets = self._to_row_send
            for destination in range(self.size):
                first, last = map(int, self.row_offsets[destination:destination+2])
                offset, count = int(offsets[destination]), int(counts[destination])
                send[offset:offset+count] = x[first:last, :].ravel(order='F')
            return send, result

        send, result = self._guard(pack, 'column-to-row packing')
        self._exchange(send, result.ravel(order='F'), self._to_row_send, self._to_row_receive)
        self.stats['column_to_row_calls'] += 1
        self.stats['redistribution_seconds'] += perf_counter()-started
        return result

    def rows_to_columns(self, local_rows, *, out=None):
        """One Alltoallv: ``(local N, M)`` to ``(N, local M)``.

        Fortran-contiguous local row matrices are already packed by target
        column rank.  Incoming source-row blocks are unpacked into their
        original global row ranges after all MPI requests complete.

        The one-rank identity path has the same explicit alias/copy contract
        as columns_to_rows; other input layouts/dtypes are converted as usual.
        """
        started = perf_counter()
        if self.size == 1:
            result = self._identity_layout(local_rows, out)
            self.stats['row_to_column_calls'] += 1
            self.stats['redistribution_seconds'] += perf_counter()-started
            return result

        def prepare():
            x = self._matrix(local_rows, self.row_shape)
            result = (self.xp.empty(self.column_shape, dtype=self.xp.float64, order='F')
                      if out is None else self._matrix(out, self.column_shape, output=True))
            receive = self.xp.empty(result.size, dtype=self.xp.float64)
            return x, receive, result

        x, receive, result = self._guard(prepare, 'row-to-column preparation')
        self._exchange(x.ravel(order='F'), receive, self._to_column_send, self._to_column_receive)

        def unpack():
            counts, offsets = self._to_column_receive
            for source in range(self.size):
                first, last = map(int, self.row_offsets[source:source+2])
                offset, count = int(offsets[source]), int(counts[source])
                result[first:last, :] = receive[offset:offset+count].reshape(
                    (last-first, self.local_column_count), order='F')

        self._guard(unpack, 'row-to-column unpacking')
        self.stats['row_to_column_calls'] += 1
        self.stats['redistribution_seconds'] += perf_counter()-started
        return result

    def _identity_layout(self, values, out):
        """Reuse valid one-rank inputs, retaining normal input conversion."""
        source = self._matrix(values, self.column_shape)
        if out is None:
            return source
        target = self._matrix(out, self.column_shape, output=True)
        self.xp.copyto(target, source)
        return target

    def close(self):
        if self._owns_comm:
            self.comm.Free()
            self._owns_comm = False


def column_chebyshev_filter(layout, local_columns, *, degree, lower_bound,
                            upper_bound, reference_eigenvalue, initial_sigma=None,
                            return_final_sigma=False):
    """Uniform normalized filter with no communication inside its recurrence.

    Parameters are collectively checked once before the filter.  Per-block
    adaptive degrees and carry-sigma scheduling remain the replay caller's
    responsibility; this function does not change SCF filter policy.
    """
    def validate():
        if int(degree) != degree or degree < 1:
            raise ValueError('degree must be a positive integer')
        half = .5*(float(upper_bound)-float(lower_bound))
        center = .5*(float(upper_bound)+float(lower_bound))
        denominator = float(reference_eigenvalue)-center
        if not np.all(np.isfinite([half, center, denominator])) or half <= 0 or denominator == 0:
            raise ValueError('invalid Chebyshev bounds/reference')
        sigma_one = half/denominator
        sigma = sigma_one if initial_sigma is None else float(initial_sigma)
        if not np.isfinite(sigma):
            raise ValueError('initial sigma must be finite')
        block = layout._matrix(local_columns, layout.column_shape)
        return half, center, sigma_one, sigma, block

    half, center, sigma_one, sigma, block = layout._guard(validate, 'column filter validation')
    parameters = layout.comm.allgather((int(degree), half, center, sigma_one, sigma))
    if len(set(parameters)) != 1:
        raise CollectiveDomainError('ranks disagree on column filter parameters')
    previous = block.copy()
    current = layout.chebyshev_recurrence(block, center=center, scale=sigma_one/half)
    for _ in range(2, int(degree)+1):
        next_sigma = 1.0/(2.0/sigma_one-sigma)
        following = layout.chebyshev_recurrence(
            current, center=center, scale=2.0/half, sigma_next=next_sigma,
            previous=previous, sigma=sigma)
        previous, current, sigma = current, following, next_sigma
    return (current, sigma) if return_final_sigma else current


__all__ = ['ColumnLayout', 'balanced_ranges', 'column_chebyshev_filter']
