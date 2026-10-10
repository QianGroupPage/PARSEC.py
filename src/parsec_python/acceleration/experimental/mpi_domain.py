"""Experimental row-distributed FP64 Hamiltonian and generalized Ritz replay.

This module deliberately does not register an SCF backend.  Setup accepts a
replicated host operator, but orbital matrices remain row distributed.  The
steady-state stencil communicates only unique remote neighbors; the nonlocal
operator communicates projector coefficients, never the complete orbitals.

``transport='host'`` stages MPI messages through NumPy; ``'cuda'`` requires a
CUDA-aware MPI implementation and CuPy.  Callers select one GPU per MPI rank
before construction.  No mpi4py import is needed for serial/simulated tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from time import perf_counter
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular


class CollectiveDomainError(RuntimeError):
    """All surviving ranks report a local setup/compute failure together."""


class SerialComm:
    """Small MPI-compatible serial communicator for CPU reference tests."""

    rank = 0
    size = 1

    def allgather(self, value):
        return [value]

    def alltoall(self, values):
        return values

    def bcast(self, value, root=0):
        return value

    def Allreduce(self, send, recv):
        recv[...] = send


@dataclass(frozen=True)
class RowDomain:
    rows: np.ndarray
    ghost_rows: np.ndarray
    neighbors: np.ndarray
    codes: np.ndarray
    palette: np.ndarray
    receive_slices: dict[int, slice]


def build_row_domain(metadata, owner, rank, size, local_rows=None) -> RowDomain:
    """Remap slots without changing their order or the physical operator."""
    n = int(metadata.shape[0])
    ownership = np.asarray(owner)
    if ownership.shape != (n,) or not np.issubdtype(ownership.dtype, np.integer):
        raise ValueError('owner must contain one integer rank per global row')
    if np.any(ownership < 0) or np.any(ownership >= size):
        raise ValueError('owner contains an unavailable rank')
    expected = np.flatnonzero(ownership == rank)
    rows = expected if local_rows is None else np.asarray(local_rows)
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
            or not np.array_equal(np.sort(rows), expected)):
        raise ValueError('local_rows must be a permutation of the owned rows')
    if not len(rows):
        raise ValueError('empty partitions are not supported by this prototype')
    rows = np.asarray(rows, dtype=np.int64)
    slots = np.asarray(metadata.neighbors, dtype=np.int64)[:, rows].copy()
    active = slots >= 0
    if np.any(slots[active] >= n):
        raise ValueError('stencil neighbor is outside the global operator')
    candidates = np.unique(slots[active])
    remote = candidates[ownership[candidates] != rank]
    grouped = []
    receive_slices = {}
    offset = 0
    for peer in range(size):
        requested = remote[ownership[remote] == peer]
        if len(requested):
            grouped.append(requested)
            receive_slices[peer] = slice(offset, offset + len(requested))
            offset += len(requested)
    ghosts = np.concatenate(grouped) if grouped else np.empty(0, dtype=np.int64)
    lookup = np.full(n, -1, dtype=np.int64)
    lookup[rows] = np.arange(len(rows))
    lookup[ghosts] = len(rows) + np.arange(len(ghosts))
    slots[active] = lookup[slots[active]]
    if np.any(slots[active] < 0):
        raise RuntimeError('incomplete halo mapping')
    if len(rows) + len(ghosts) > np.iinfo(np.int32).max:
        raise ValueError('local index storage exceeds int32')
    return RowDomain(rows, ghosts, np.ascontiguousarray(slots, dtype=np.int32),
                     np.ascontiguousarray(metadata.coefficient_codes[:, rows]),
                     np.asarray(metadata.coefficient_palette, dtype=np.float64),
                     receive_slices)


_CUDA = r'''
extern "C" __global__ void domain_apply(
 int n, int slots, int width, const int* neighbors,
 const unsigned char* codes, const double* palette, const double* v,
 const double* x, long long xs0, long long xs1,
 const double* ghost, const long long* bp, const int* bj,
 const double* bv, const double* coeff, int nproj, double* y) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 int first_col=blockIdx.y*6;
 if(row>=n || first_col>=width) return;
 double value[6]={0,0,0,0,0,0};
 for(int s=0;s<slots;++s) {
   long long position=(long long)s*n+row;
   int k=neighbors[position];
   if(k<0) continue;
   double coefficient=palette[codes[position]];
   #pragma unroll
   for(int j=0;j<6;++j) {
     int col=first_col+j;
     if(col>=width) continue;
     double z=k<n ? x[(long long)k*xs0+(long long)col*xs1]
                  : ghost[(long long)(k-n)*width+col];
     value[j] += coefficient*z;
   }
 }
 #pragma unroll
 for(int j=0;j<6;++j) if(first_col+j<width)
   value[j] += v[row]*x[(long long)row*xs0+(long long)(first_col+j)*xs1];
 double nonlocal[6]={0,0,0,0,0,0};
 for(long long p=bp[row];p<bp[row+1];++p) {
   #pragma unroll
   for(int j=0;j<6;++j) if(first_col+j<width)
     nonlocal[j] += bv[p]*coeff[(long long)bj[p]*width+first_col+j];
 }
 #pragma unroll
 for(int j=0;j<6;++j) if(first_col+j<width)
   y[(long long)(first_col+j)*n+row]=value[j]+nonlocal[j];
}
'''


class DistributedHamiltonian:
    """One rank's persistent domain; methods are collective on ``comm``.

    Local arrays use the order in ``rows``.  ``shape`` is the global physical
    operator shape and is deliberately not accepted by existing local SCF
    solvers.  Use this module's explicit distributed routines instead.

    Collective guards add a small MPI reduction around local operations.
    They cannot recover a failed process or an MPI/network failure itself.
    Calls must occur in the same order on all ranks; this object is not
    reentrant and uses a private duplicated communicator when available.
    """

    def __init__(self, metadata, local_potential, projectors, signs, owner,
                 *, comm=None, xp=np, transport='host', local_rows=None):
        incoming = SerialComm() if comm is None else comm
        self._owns_comm = hasattr(incoming, 'Dup')
        self.comm = incoming.Dup() if self._owns_comm else incoming
        self.rank, self.size = self.comm.rank, self.comm.size
        self.xp = xp
        self.gpu = xp is not np
        self.transport = transport
        self.stats = dict(applications=0, halo_send_bytes=0, halo_receive_bytes=0,
                          projector_reduce_input_bytes=0, gram_reduce_input_bytes=0,
                          halo_seconds=0.0, apply_seconds=0.0)

        def setup():
            if transport not in ('host', 'cuda'):
                raise ValueError('transport must be host or cuda')
            if transport == 'cuda' and not self.gpu:
                raise ValueError('cuda transport requires CuPy')
            self.shape = tuple(map(int, metadata.shape))
            self.domain = build_row_domain(metadata, owner, self.rank, self.size,
                                           local_rows)
            self.rows = self.domain.rows
            self.local_shape = (len(self.rows), len(self.rows))
            v = np.asarray(local_potential, dtype=np.float64)
            if v.shape != (self.shape[0],) or not np.all(np.isfinite(v)):
                raise ValueError('local potential must be a finite global host vector')
            b = sp.csr_matrix(projectors, dtype=np.float64, copy=True)
            if b.shape[0] != self.shape[0]:
                raise ValueError('projector rows do not match the operator')
            b.sum_duplicates()
            b.sort_indices()
            sg = np.asarray(signs, dtype=np.float64)
            if sg.shape != (b.shape[1],) or not np.all(np.isfinite(sg)):
                raise ValueError('projector signs do not match projector columns')
            if not np.all(np.isfinite(b.data)):
                raise ValueError('projector values must be finite')
            self.projector_count = b.shape[1]
            self.host_projectors = b[self.rows, :].tocsr()
            self.potential = xp.asarray(v[self.rows])
            self.signs = xp.asarray(sg)
            self.neighbors = xp.asarray(self.domain.neighbors)
            self.codes = xp.asarray(self.domain.codes)
            self.palette = xp.asarray(self.domain.palette)
            if self.gpu:
                import cupyx.scipy.sparse as csp
                self.projector_transpose = csp.csr_matrix(
                    self.host_projectors.T.tocsr())
                self.bp = xp.asarray(self.host_projectors.indptr, dtype=xp.int64)
                self.bj = xp.asarray(self.host_projectors.indices, dtype=xp.int32)
                self.bv = xp.asarray(self.host_projectors.data)
                self.kernel = xp.RawKernel(_CUDA, 'domain_apply')
                self.kernel.compile()
            else:
                self.projector_transpose = self.host_projectors.T.tocsr()
            self._sync()

        self._guard(setup, 'domain setup')
        owner_hash = hashlib.sha256(np.asarray(owner, dtype=np.int64).tobytes()).hexdigest()
        signatures = self.comm.allgather((self.shape, self.projector_count, owner_hash))
        if len(set(signatures)) != 1:
            raise CollectiveDomainError('ranks disagree on global operator shape/ownership')
        requests = [np.empty(0, dtype=np.int64) for _ in range(self.size)]
        for peer, part in self.domain.receive_slices.items():
            requests[peer] = self.domain.ghost_rows[part]
        requested_from_me = self.comm.alltoall(requests)

        def map_sends():
            lookup = np.full(self.shape[0], -1, dtype=np.int64)
            lookup[self.rows] = np.arange(len(self.rows))
            self.send_indices = {}
            for peer, values in enumerate(requested_from_me):
                if len(values):
                    indices = lookup[np.asarray(values, dtype=np.int64)]
                    if np.any(indices < 0):
                        raise ValueError('peer requested a row not owned by this rank')
                    self.send_indices[peer] = xp.asarray(
                        indices, dtype=xp.int64)
            self._sync()

        self._guard(map_sends, 'halo request mapping')
        self._ghosts = None
        self._width = None

    def _sync(self):
        if self.gpu:
            self.xp.cuda.get_current_stream().synchronize()

    def _host(self, values):
        return self.xp.asnumpy(values) if self.gpu else np.asarray(values)

    def _guard(self, function, stage):
        """Finish local work, then agree whether the next collective is safe."""
        error, result = None, None
        try:
            result = function()
            self._sync()
        except Exception as exc:
            error = f'rank {self.rank}: {type(exc).__name__}: {exc}'
        good = np.array([error is None], dtype=np.int32)
        total = np.empty_like(good)
        self.comm.Allreduce(good, total)
        if int(total[0]) != self.size:
            messages = [x for x in self.comm.allgather(error) if x is not None]
            raise CollectiveDomainError(stage + ': ' + '; '.join(messages))
        return result

    def _sum(self, values):
        """MPI sum with contiguous buffers and explicit producer completion."""
        packed = self._guard(lambda: self.xp.ascontiguousarray(values),
                             'reduction packing')
        if self.transport == 'cuda':
            result = self.xp.empty_like(packed)
            self.comm.Allreduce(packed, result)
            self._sync()
            return result
        host = self._host(packed)
        total = np.empty_like(host)
        self.comm.Allreduce(host, total)
        return self.xp.asarray(total)

    def _columns(self, vectors):
        def convert():
            if np.iscomplexobj(vectors):
                raise ValueError('only real FP64 vectors are supported')
            x = self.xp.asarray(vectors, dtype=self.xp.float64)
            was_vector = x.ndim == 1
            if was_vector:
                x = x[:, None]
            if x.ndim != 2 or x.shape[0] != len(self.rows) or x.shape[1] < 1:
                raise ValueError('vectors must have shape (local_rows, columns)')
            return x, was_vector

        x, was_vector = self._guard(convert, 'vector validation')
        shapes = self.comm.allgather((int(x.shape[1]), was_vector))
        if len(set(shapes)) != 1:
            raise CollectiveDomainError('ranks disagree on vector column count')
        return x, was_vector

    def _exchange(self, x):
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
        started = perf_counter()
        requests = []
        for peer, part in self.domain.receive_slices.items():
            requests.append(self.comm.Irecv(receive[part, :], source=peer, tag=41))
        for peer, value in packed.items():
            requests.append(self.comm.Isend(value, dest=peer, tag=41))
        # Buffers remain live until every nonblocking operation completes.
        for request in requests:
            request.Wait()
        if self.transport == 'host':
            self._ghosts[...] = self.xp.asarray(receive)
        self._sync()
        self.stats['halo_seconds'] += perf_counter() - started
        self.stats['halo_send_bytes'] += sum(x.nbytes for x in packed.values())
        self.stats['halo_receive_bytes'] += int(receive.nbytes)
        return self._ghosts

    def apply(self, vectors):
        """Collective ``H @ vectors`` with no global orbital gather."""
        start = perf_counter()
        x, was_vector = self._columns(vectors)
        ghosts = self._exchange(x)
        if self.projector_count:
            partial = self._guard(lambda: self.projector_transpose @ x,
                                  'local projector projection')
            coefficients = self._sum(partial) * self.signs[:, None]
            self.stats['projector_reduce_input_bytes'] += int(partial.nbytes)
        else:
            coefficients = self.xp.empty((0, x.shape[1]), dtype=self.xp.float64)

        def action():
            if not self.gpu:
                out = np.zeros(x.shape, dtype=np.float64, order='F')
                n = len(self.rows)
                for slot, code in zip(self.domain.neighbors, self.domain.codes):
                    local = (slot >= 0) & (slot < n)
                    remote = slot >= n
                    out[local] += self.domain.palette[code[local], None] * x[slot[local]]
                    out[remote] += self.domain.palette[code[remote], None] * ghosts[slot[remote] - n]
                out += self.potential[:, None] * x
                if self.projector_count:
                    out += self.host_projectors @ coefficients
                return out
            out = self.xp.empty(x.shape, dtype=self.xp.float64, order='F')
            coeff = self.xp.ascontiguousarray(coefficients)
            self.kernel(((len(self.rows) + 255) // 256, (x.shape[1]+5)//6), (256,),
                        (np.int32(len(self.rows)), np.int32(self.neighbors.shape[0]),
                         np.int32(x.shape[1]), self.neighbors, self.codes, self.palette,
                         self.potential, x, np.int64(x.strides[0] // 8),
                         np.int64(x.strides[1] // 8), ghosts, self.bp, self.bj,
                         self.bv, coeff, np.int32(self.projector_count), out))
            return out

        result = self._guard(action, 'local Hamiltonian action')
        self.stats['applications'] += 1
        self.stats['apply_seconds'] += perf_counter() - start
        return result[:, 0] if was_vector else result

    def __matmul__(self, vectors):
        return self.apply(vectors)

    def close(self):
        """Collectively release this object's private MPI communicator."""
        if self._owns_comm:
            self.comm.Free()
            self._owns_comm = False


@dataclass(frozen=True)
class DistributedRitzResult:
    eigenvalues: np.ndarray
    local_vectors: Any
    residual_norms: np.ndarray | None
    overlap: np.ndarray
    projected_hamiltonian: np.ndarray
    condition_number: float
    orthogonality_error: float


def generalized_ritz(operator, local_basis, *, compute_residuals=True,
                     condition_max=1.0e8) -> DistributedRitzResult:
    """Distributed projection/rotation, replicated small dense root solve.

    The input columns need not be orthonormal.  Unsafe overlap matrices fail
    collectively: this prototype does not hide a full-orbital gather as a QR
    fallback.  A production distributed solver needs a distributed fallback.
    """
    xp = operator.xp
    x, _ = operator._columns(local_basis)
    hx = operator.apply(x)
    partial = operator._guard(lambda: xp.stack((x.T @ x, x.T @ hx)),
                              'local Ritz projection')
    matrices = operator._sum(partial)
    operator.stats['gram_reduce_input_bytes'] += int(partial.nbytes)
    packet, error = None, None
    if operator.rank == 0:
        try:
            host = operator._host(matrices)
            s = np.tril(host[0]) + np.tril(host[0], -1).T
            a = np.tril(host[1]) + np.tril(host[1], -1).T
            condition = float(np.linalg.cond(s))
            if not np.isfinite(condition) or condition > condition_max:
                raise np.linalg.LinAlgError(f'unsafe overlap condition {condition:.6g}')
            chol = np.linalg.cholesky(s)
            left = solve_triangular(chol, a, lower=True, check_finite=False)
            whitened = solve_triangular(chol, left.T, lower=True,
                                       check_finite=False).T
            whitened = np.tril(whitened) + np.tril(whitened, -1).T
            values, vectors = np.linalg.eigh(whitened)
            coeff = solve_triangular(chol.T, vectors, lower=False,
                                     check_finite=False)
            ortho = float(np.max(np.abs(coeff.T @ s @ coeff - np.eye(len(values)))))
            if not np.isfinite(ortho) or ortho > 5.0e-10:
                raise np.linalg.LinAlgError(f'Ritz orthogonality audit failed: {ortho}')
            packet = (values, coeff, s, a, condition, ortho)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    error = operator.comm.bcast(error, root=0)
    if error is not None:
        raise CollectiveDomainError('root generalized Ritz solve: ' + error)
    values, coeff, s, a, condition, ortho = operator.comm.bcast(packet, root=0)
    c = xp.asarray(coeff)
    rotated = operator._guard(lambda: x @ c, 'local Ritz rotation')
    residuals = None
    if compute_residuals:
        def residual_squares():
            r = hx @ c - rotated * xp.asarray(values)[None, :]
            return xp.sum(r * r, axis=0)
        squares = operator._guard(residual_squares, 'local Ritz residual')
        residuals = np.sqrt(np.maximum(operator._host(operator._sum(squares)), 0))
    return DistributedRitzResult(values, rotated, residuals, s, a, condition, ortho)


def chebyshev_filter(operator, local_vectors, *, degree, lower_bound,
                     upper_bound, reference_eigenvalue, initial_sigma=None,
                     return_final_sigma=False):
    """Collective normalized recurrence matching the existing FP64 equations.

    Caller-selected column packets amortize halo latency.  This is one
    uniform-degree packet; the production per-block degree/carry schedule is
    deliberately left to the replay caller.
    """
    def validate():
        if int(degree) != degree or degree < 1:
            raise ValueError('degree must be a positive integer')
        half = .5 * (float(upper_bound) - float(lower_bound))
        center = .5 * (float(upper_bound) + float(lower_bound))
        denominator = float(reference_eigenvalue) - center
        if (not np.all(np.isfinite([half, center, denominator]))
                or half <= 0 or denominator == 0):
            raise ValueError('invalid Chebyshev interval/reference')
        sigma_one = half / denominator
        sigma = sigma_one if initial_sigma is None else float(initial_sigma)
        if not np.isfinite(sigma):
            raise ValueError('invalid initial_sigma')
        return half, center, sigma_one, sigma

    half, center, sigma_one, sigma = operator._guard(validate, 'filter validation')
    settings = operator.comm.allgather((degree, half, center, sigma_one, sigma))
    if len(set(settings)) != 1:
        raise CollectiveDomainError('ranks disagree on Chebyshev parameters')
    block, was_vector = operator._columns(local_vectors)
    previous = block.copy()
    current = (operator.apply(block) - center * block) * (sigma_one / half)
    for _ in range(2, int(degree) + 1):
        sigma_next = 1.0 / (2.0 / sigma_one - sigma)
        following = sigma_next * ((2.0 / half) *
                    (operator.apply(current) - center * current) - sigma * previous)
        previous, current = current, following
        sigma = sigma_next
    result = current[:, 0] if was_vector else current
    return (result, sigma) if return_final_sigma else result


__all__ = ['CollectiveDomainError', 'SerialComm', 'RowDomain',
           'build_row_domain', 'DistributedHamiltonian', 'DistributedRitzResult',
           'generalized_ritz', 'chebyshev_filter']
