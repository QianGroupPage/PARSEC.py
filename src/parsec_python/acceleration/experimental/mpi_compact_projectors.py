"""Experimental communication reduction for spatially local KB projectors.

Only coefficients whose nonzero projector support crosses rank boundaries
need an MPI sum.  Wholly local coefficients remain on their owning rank;
other ranks have zero support for those columns and never use the omitted
remote coefficient.  The Hamiltonian and FP64 arithmetic are unchanged.
"""
from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp

from .mpi_domain import CollectiveDomainError, DistributedHamiltonian
from .mpi_domain_overlap import OverlappedDistributedHamiltonian


class CompactProjectorHamiltonian(OverlappedDistributedHamiltonian):
    """Reduce only shared-projector coefficients, retaining the overlap path.

    Constructor arguments match ``OverlappedDistributedHamiltonian``.
    ``overlap=False`` is rejected because the inherited original base apply
    intentionally has no experimental projector hook.

    ``projector_partition`` reports static column counts separately from
    cumulative byte counters in ``stats``.  Byte counts are per-rank MPI
    input payloads, not a claim about actual fabric wire traffic.
    """

    def __init__(self, metadata, local_potential, projectors, signs, owner,
                 *, comm=None, xp=np, transport='host', local_rows=None,
                 overlap=True, collective_checks=True, validated_widths=None):
        super().__init__(metadata, local_potential, projectors, signs, owner,
                         comm=comm, xp=xp, transport=transport, local_rows=local_rows,
                         overlap=overlap, collective_checks=collective_checks,
                         validated_widths=validated_widths)

        def classify():
            if not self.overlap:
                raise ValueError('compact projectors require overlap=True')
            b = sp.csr_matrix(projectors, dtype=np.float64, copy=True)
            b.sum_duplicates()
            b.sort_indices()
            support_owner = np.repeat(np.asarray(owner, dtype=np.int64), np.diff(b.indptr))
            nonzero = b.data != 0.0
            minimum = np.full(self.projector_count, self.size, dtype=np.int64)
            maximum = np.full(self.projector_count, -1, dtype=np.int64)
            np.minimum.at(minimum, b.indices[nonzero], support_owner[nonzero])
            np.maximum.at(maximum, b.indices[nonzero], support_owner[nonzero])
            shared = (maximum >= 0) & (minimum != maximum)
            self.shared_projector_columns = np.flatnonzero(shared).astype(np.int64)
            self.local_projector_columns = np.flatnonzero(
                (minimum == self.rank) & (maximum == self.rank)).astype(np.int64)
            self.shared_projector_count = len(self.shared_projector_columns)
            self.local_projector_count = len(self.local_projector_columns)
            self.zero_projector_count = int(np.count_nonzero(maximum < 0))
            self.device_shared_projector_columns = xp.asarray(self.shared_projector_columns)
            self.projector_partition = dict(
                total_columns=int(self.projector_count),
                shared_columns=self.shared_projector_count,
                wholly_local_columns_global=int(np.count_nonzero((maximum >= 0) & ~shared)),
                wholly_local_columns_this_rank=self.local_projector_count,
                zero_columns=self.zero_projector_count)

        # Setup validation is collective even in the explicitly fast run mode.
        DistributedHamiltonian._guard(self, classify, 'projector support classification')
        digest = hashlib.sha256(self.shared_projector_columns.tobytes()).hexdigest()
        signatures = self.comm.allgather((self.shared_projector_count, digest))
        if len(set(signatures)) != 1:
            raise CollectiveDomainError('ranks disagree on shared projector support')
        self.stats.update(projector_full_reduce_equivalent_bytes=0,
                          projector_reduction_saved_input_bytes=0)

    def _projector_coefficients(self, x):
        if not self.projector_count:
            return self.xp.empty((0, x.shape[1]), dtype=self.xp.float64)
        partial = self._guard(lambda: self.projector_transpose @ x,
                              'local projector projection')
        full_bytes = int(partial.nbytes)
        reduced_bytes = 0
        if self.shared_projector_count:
            shared = self._guard(lambda: self.xp.ascontiguousarray(
                partial[self.device_shared_projector_columns, :]),
                'shared projector coefficient packing')
            total = self._sum(shared)

            def scatter():
                partial[self.device_shared_projector_columns, :] = total

            self._guard(scatter, 'shared projector coefficient scatter')
            reduced_bytes = int(shared.nbytes)
        self.stats['projector_reduce_input_bytes'] += reduced_bytes
        self.stats['projector_full_reduce_equivalent_bytes'] += full_bytes
        self.stats['projector_reduction_saved_input_bytes'] += full_bytes-reduced_bytes
        return partial * self.signs[:, None]


__all__ = ['CompactProjectorHamiltonian']
