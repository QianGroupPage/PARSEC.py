"""The complex eigenproblem of one k-point with its basis shared by the ranks of a pool, column by column.

:func:`.chebff_spinor.solve_spinor_eigval` solves a Bloch Hamiltonian on one device: the first SCF
iteration runs CHEBFF (``filter_cycles`` cycles of filter / orthonormalization / Rayleigh-Ritz from a
random start), every later one a single SUBSPACE step on the saved Ritz vectors.  Here the ``N x m``
basis (``m = requested + buffer``) is shared by the ``p`` ranks of a communicator (a *pool* of
:mod:`..SCF.kpoint_pools`), one GPU each, with the same policy and the same numbers up to round-off:

* rank ``r`` holds **all N grid rows of its columns**: its share of the low-degree and its share of
  the high-degree filter blocks (:class:`ColumnLayout`), so that the filter work of the ranks is
  balanced in the SUBSPACE steps, whose two halves are filtered at degrees ``d -/+ delta``;
* every rank holds a full replica of the device Bloch operator
  (:class:`..Hamiltonian.device_bloch.DeviceBlochOperator`) and filters its own columns **without
  communication**; the start basis is the random stream of the one-device solve, which every rank
  replays, keeping its columns; the Lanczos bound of the filter interval is the pool root's;
* one orthonormalization and Rayleigh-Ritz pass moves the basis between columns and blocks of rows
  four times (:meth:`ColumnGroup.to_rows`, :meth:`ColumnGroup.to_columns`: ``Alltoallv`` in rounds of
  at most ``PARSEC_MPI_SHARED_EXCHANGE_BYTES`` per rank)::

      X (columns, filtered) -> rows;  X <- X R^-1  (shifted Cholesky QR, three passes: R^H R = X^H X
      summed over the ranks, the factor on the pool root)  ->  columns;  H X on the own columns  ->
      rows;  A = X^H (H X) summed over the ranks, A = V diag(e) V^H on the pool root, V broadcast;
      X <- X V and the residuals ||H x_j - e_j x_j|| from (H X) V on the rows  ->  columns.

  The moves of ``X`` before and after ``H`` replace the Householder QR of the one-device basis: a
  Rayleigh-Ritz of the filtered basis itself (``H X`` taken before the orthonormalization, ``X^H X``
  and ``X^H H X`` as a generalized problem, three moves) loses about ``eps * cond(X)`` of ``||H||``
  in the projected matrix, and a filtered basis has ``cond(X)`` up to ``1e10``.  The residuals cost no
  further application of ``H`` (``(H X) V = H (X V)``).

The small dense steps (Cholesky factor and its inverse, ``eigh``) run on the pool root and are
broadcast, so the ranks of a pool agree bit for bit; the sums over the ranks are ``Allreduce``.  The
exchanges and broadcasts hand device arrays to MPI where ``MPICH_GPU_SUPPORT_ENABLED=1``
(:func:`.mpi_shared_basis.shared_transport`), else they go through the host; the sums of the
``m x m`` matrices go through the host unless ``PARSEC_MPI_CUDA_AWARE_REDUCE=1``, as those of the
row-distributed solver (:mod:`..Parallel.mpi`: on Cray MPICH a device-buffer ``Allreduce`` of a few MB
cost three times the host-staged one).  The seconds of every collective start after the device has
finished the work queued before it, so that they count the communication only.  Everything uses only
operations NumPy and CuPy share, so the tests run this code on the host with a communicator of threads.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from parsec_python.Eigensolvers.chebyshev import subspace_filter_blocks
from parsec_python.Eigensolvers.spectral_bounds import bound_request
from parsec_python.Eigensolvers.subspace import _adapt_polynomial_degree, subspace_filter_bounds

from .chebff_spinor import (
    _Bounds,
    SpinorEigvalResult,
    SpinorEigvalState,
    _chain,
    _CountingOperator,
    _filter_columns,
    _filter_group,
    _lanczos_upper_bound,
    _residual_period,
    _row_chunk,
    _solve_triangular_inverse,
    _Stats,
)
from .mpi_shared_basis import exchange_bytes, row_edges, shared_transport


# --------------------------------------------------------------------------- layout
class ColumnLayout:
    """Which columns of an ``m``-column basis each of ``size`` ranks holds.

    The SUBSPACE filter of :func:`parsec_python.Eigensolvers.chebyshev.subspace_filter_blocks` splits
    the columns into the first ``ceil(full blocks / 2)`` memory blocks (degree ``d - delta``) and the
    rest (degree ``d + delta``); every rank takes a contiguous share of each part, so that every rank
    filters as many columns of each degree as the others.  ``spans(rank)`` lists
    ``(first global column, first local column, count)``; the local columns are in global order.
    """

    def __init__(self, columns: int, block_size: int, size: int) -> None:
        columns, block_size, size = int(columns), max(1, int(block_size)), int(size)
        if columns < 1 or size < 1:
            raise ValueError("a column layout needs columns and ranks")
        full = columns // block_size
        split = ((full + 1) // 2) * block_size
        parts = [(0, split), (split, columns)]
        self.columns = columns
        self.block_size = block_size
        self.size = size
        self._spans: list[list[tuple[int, int, int]]] = []
        for rank in range(size):
            spans: list[tuple[int, int, int]] = []
            offset = 0
            for start, stop in parts:
                first = start + (stop - start) * rank // size
                last = start + (stop - start) * (rank + 1) // size
                if last <= first:
                    continue
                if spans and spans[-1][0] + spans[-1][2] == first:
                    g0, l0, count = spans[-1]
                    spans[-1] = (g0, l0, count + last - first)
                else:
                    spans.append((first, offset, last - first))
                offset += last - first
            self._spans.append(spans)
        self.widths = tuple(sum(count for _, _, count in spans) for spans in self._spans)

    def spans(self, rank: int) -> list[tuple[int, int, int]]:
        return list(self._spans[rank])

    def global_columns(self, rank: int) -> np.ndarray:
        """The global indices of the local columns of ``rank``, in local order."""
        pieces = [np.arange(g0, g0 + count) for g0, _, count in self._spans[rank]]
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)

    def key(self) -> tuple[int, int, int]:
        return (self.columns, self.block_size, self.size)

    def __eq__(self, other) -> bool:
        return isinstance(other, ColumnLayout) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())


@dataclass
class ColumnEigvalState(SpinorEigvalState):
    """:class:`.chebff_spinor.SpinorEigvalState` of a shared basis: ``vectors`` is this rank's block
    ``(N, layout.widths[rank])`` of the columns of ``layout``."""

    layout: ColumnLayout | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------- group
class ColumnGroup:
    """The ranks of ``comm`` sharing the basis of one eigenproblem by columns, one device each.

    ``xp`` is the array namespace of the device (CuPy, or NumPy for the host double), ``sync`` waits
    for its stream.  ``mpi`` is the ``mpi4py.MPI`` module (its ``DOUBLE`` and ``SUM``), ``None`` for a
    communicator of the tests.  ``transport`` is ``cuda`` (device arrays to MPI) or ``host``.
    """

    root = 0

    def __init__(
        self,
        comm,
        xp,
        *,
        sync: Callable[[], None] | None = None,
        mpi=None,
        transport: str | None = None,
        reduce_transport: str | None = None,
    ) -> None:
        self.comm = comm
        self.xp = xp
        self.rank = int(comm.rank) if hasattr(comm, "rank") else int(comm.Get_rank())
        self.size = int(comm.size) if hasattr(comm, "size") else int(comm.Get_size())
        self.mpi = mpi
        self._sync = sync if sync is not None else (lambda: None)
        if xp is np:
            transport = "host"
        self.transport = shared_transport() if transport is None else transport
        if self.transport not in ("cuda", "host"):
            raise ValueError("transport must be cuda or host")
        if reduce_transport is None:
            from ..Parallel.mpi import CUDA_AWARE_REDUCE_ENV

            asked = os.environ.get(CUDA_AWARE_REDUCE_ENV, "").strip().lower() in ("1", "true", "on", "yes")
            reduce_transport = "cuda" if asked and self.transport == "cuda" else "host"
        self.reduce_transport = "host" if xp is np else reduce_transport
        # what the collectives cost this rank: seconds by kind and the bytes this rank sent
        self.seconds = {"to_rows": 0.0, "to_columns": 0.0, "reduce": 0.0, "broadcast": 0.0}
        self.sent_bytes = 0

    @property
    def is_root(self) -> bool:
        return self.rank == self.root

    # ----- host / device ------------------------------------------------------
    def _host(self, array) -> np.ndarray:
        if isinstance(array, np.ndarray):
            return array
        return self.xp.asnumpy(array)

    def _device(self, array):
        return array if self.xp is np else self.xp.asarray(array)

    # ----- collectives ----------------------------------------------------------
    def bcast(self, value: Any) -> Any:
        """A small Python object from the root to every rank."""
        if self.size == 1:
            return value
        return self.comm.bcast(value, root=self.root)

    def sum(self, array):
        """Sum of an array over the ranks (every rank gets the sum, as an array of its kind)."""
        if self.size == 1:
            return array
        xp = self.xp
        source = xp.ascontiguousarray(array)
        self._sync()
        started = time.perf_counter()
        if self.reduce_transport == "cuda":
            total = xp.empty_like(source)
            self._allreduce(source, total)
        else:
            host = np.ascontiguousarray(self._host(source))
            out = np.empty_like(host)
            self._allreduce(host, out)
            total = self._device(out)
        self.seconds["reduce"] += time.perf_counter() - started
        return total

    def _allreduce(self, send, receive) -> None:
        if self.mpi is None:
            self.comm.Allreduce(send, receive)
        else:
            self.comm.Allreduce(send, receive, op=self.mpi.SUM)

    def lower_sum_to_root(self, matrix):
        """The lower triangle of ``matrix`` (``m x m``) summed over the ranks, on the root only.

        The root gets a matrix with that triangle and zeros above it; the other ranks ``None``.  Only
        the root factors the Gram matrix and diagonalizes the projected Hamiltonian, both of which read
        the lower triangle, so the ranks send ``m (m + 1) / 2`` entries each to a ``Reduce``.
        """
        xp = self.xp
        count = int(matrix.shape[0])
        if self.size == 1:
            return xp.tril(matrix)
        rows, columns = self._lower_indices(count)
        packed = xp.ascontiguousarray(matrix[rows, columns])
        self._sync()
        started = time.perf_counter()
        if self.reduce_transport == "cuda":
            total = xp.empty_like(packed) if self.is_root else None
            self._reduce(packed, total)
        else:
            host = np.ascontiguousarray(self._host(packed))
            out = np.empty_like(host) if self.is_root else None
            self._reduce(host, out)
            total = None if out is None else self._device(out)
        self.seconds["reduce"] += time.perf_counter() - started
        if not self.is_root:
            return None
        result = xp.zeros((count, count), dtype=matrix.dtype)
        result[rows, columns] = total
        return result

    def _lower_indices(self, count: int):
        cached = getattr(self, "_lower", None)
        if cached is None or cached[0] != count:
            rows, columns = np.tril_indices(count)
            cached = (count, self._device(rows), self._device(columns))
            self._lower = cached
        return cached[1], cached[2]

    def _reduce(self, send, receive) -> None:
        if self.mpi is None:
            self.comm.Reduce(send, receive, root=self.root)
        else:
            self.comm.Reduce(send, receive, op=self.mpi.SUM, root=self.root)

    def bcast_array(self, array, shape: tuple[int, ...], dtype):
        """The root's array (device or host) on every rank, as an array of the device."""
        if self.size == 1:
            return array
        self._sync()
        started = time.perf_counter()
        xp = self.xp
        if self.transport == "cuda":
            buffer = (
                xp.ascontiguousarray(xp.asarray(array, dtype=dtype))
                if self.is_root
                else xp.empty(shape, dtype=dtype)
            )
            self._sync()
            self.comm.Bcast(buffer, root=self.root)
        else:
            host = (
                np.ascontiguousarray(self._host(array), dtype=dtype)
                if self.is_root
                else np.empty(shape, dtype=dtype)
            )
            self.comm.Bcast(host, root=self.root)
            buffer = self._device(host)
        self.seconds["broadcast"] += time.perf_counter() - started
        return buffer

    def _exchange(self, send, send_counts, receive, receive_counts) -> None:
        """One ``Alltoallv`` of complex buffers (``counts`` in complex elements), as float64 views."""
        xp = self.xp
        sends = np.asarray(send_counts, dtype=np.int64) * 2
        receives = np.asarray(receive_counts, dtype=np.int64) * 2

        def offsets(counts):
            return np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)

        self.sent_bytes += 8 * int(sends.sum())
        self._sync()
        if self.transport == "cuda":
            send_buffer, receive_buffer = send.view(xp.float64), receive.view(xp.float64)
        else:
            send_buffer = np.ascontiguousarray(self._host(send)).view(np.float64)
            receive_buffer = np.empty(2 * int(receive.size), dtype=np.float64)
        send_spec = [send_buffer, (sends, offsets(sends))]
        receive_spec = [receive_buffer, (receives, offsets(receives))]
        if self.mpi is not None:
            send_spec.append(self.mpi.DOUBLE)
            receive_spec.append(self.mpi.DOUBLE)
        self.comm.Alltoallv(send_spec, receive_spec)
        if self.transport != "cuda":
            host = receive_buffer.view(np.complex128)
            if xp is np:
                receive[...] = host
            else:
                receive.set(host)

    # ----- moves between columns and rows --------------------------------------
    def _round_rows(self, columns: int) -> int:
        """Rows per rank and round, so that a round's buffers hold at most :func:`exchange_bytes`."""
        return max(1, exchange_bytes() // (16 * max(int(columns), 1)))

    def to_rows(self, local, layout: ColumnLayout, dimension: int):
        """This rank's block of rows ``(N_r, m)`` (C order) of the basis whose columns the ranks hold.

        With one rank the rows are the columns: ``local`` itself comes back.
        """
        if self.size == 1:
            return local
        self._sync()
        started = time.perf_counter()
        xp = self.xp
        edges = row_edges(dimension, self.size)
        heights = [edges[q + 1] - edges[q] for q in range(self.size)]
        own = heights[self.rank]
        width = layout.widths[self.rank]
        result = xp.empty((own, layout.columns), dtype=xp.complex128)
        step = self._round_rows(layout.columns)
        rounds = -(-max(heights) // step) if max(heights) else 0
        for index in range(rounds):
            first = index * step
            counts = [max(0, min(step, height - first)) for height in heights]
            send = xp.empty(max(1, sum(counts) * width), dtype=xp.complex128)
            offset = 0
            for q, count in enumerate(counts):
                if count and width:
                    send[offset : offset + count * width].reshape(count, width)[...] = local[
                        edges[q] + first : edges[q] + first + count
                    ]
                offset += count * width
            mine = counts[self.rank]
            receive_counts = [mine * w for w in layout.widths]
            receive = xp.empty(max(1, sum(receive_counts)), dtype=xp.complex128)
            self._exchange(send, [count * width for count in counts], receive, receive_counts)
            del send
            offset = 0
            for source, w in enumerate(layout.widths):
                if mine and w:
                    piece = receive[offset : offset + mine * w].reshape(mine, w)
                    for g0, l0, count in layout.spans(source):
                        result[first : first + mine, g0 : g0 + count] = piece[:, l0 : l0 + count]
                offset += mine * w
            del receive
        self._sync()
        self.seconds["to_rows"] += time.perf_counter() - started
        return result

    def to_columns(self, rows, layout: ColumnLayout, dimension: int, out=None, *, copy: bool = True):
        """This rank's columns ``(N, w_r)`` (C order) of the basis whose blocks of rows the ranks hold.

        With one rank the columns are the rows: a copy of ``rows``, or ``rows`` itself with ``copy=False``.
        """
        if self.size == 1:
            return rows.copy() if copy else rows
        self._sync()
        started = time.perf_counter()
        xp = self.xp
        edges = row_edges(dimension, self.size)
        heights = [edges[q + 1] - edges[q] for q in range(self.size)]
        width = layout.widths[self.rank]
        target = out
        if target is None or tuple(target.shape) != (dimension, width):
            target = xp.empty((dimension, width), dtype=xp.complex128)
        step = self._round_rows(layout.columns)
        rounds = -(-max(heights) // step) if max(heights) else 0
        for index in range(rounds):
            first = index * step
            counts = [max(0, min(step, height - first)) for height in heights]
            mine = counts[self.rank]
            send_counts = [mine * w for w in layout.widths]
            send = xp.empty(max(1, sum(send_counts)), dtype=xp.complex128)
            offset = 0
            for destination, w in enumerate(layout.widths):
                if mine and w:
                    piece = send[offset : offset + mine * w].reshape(mine, w)
                    for g0, l0, count in layout.spans(destination):
                        piece[:, l0 : l0 + count] = rows[first : first + mine, g0 : g0 + count]
                offset += mine * w
            receive_counts = [count * width for count in counts]
            receive = xp.empty(max(1, sum(receive_counts)), dtype=xp.complex128)
            self._exchange(send, send_counts, receive, receive_counts)
            del send
            offset = 0
            for source, count in enumerate(counts):
                if count and width:
                    target[edges[source] + first : edges[source] + first + count] = receive[
                        offset : offset + count * width
                    ].reshape(count, width)
                offset += count * width
            del receive
        self._sync()
        self.seconds["to_columns"] += time.perf_counter() - started
        return target

    def statistics(self) -> dict[str, float]:
        values = {f"pool_{name}_seconds": value for name, value in self.seconds.items()}
        values["pool_sent_bytes"] = float(self.sent_bytes)
        return values


# ------------------------------------------------------------------ dense steps
def _check_finite_columns(xp, block, group: ColumnGroup) -> None:
    """Every rank of the pool stops together when the recurrence of any rank is not finite."""
    bad = 0.0 if block.shape[1] == 0 or bool(xp.all(xp.isfinite(block[:, :1]))) else 1.0
    if group.size > 1:
        bad = float(np.asarray(group._host(group.sum(xp.asarray(np.array([bad])))))[0])
    if bad > 0.0:
        raise FloatingPointError("Chebyshev recurrence produced nonfinite values")


def _gram(xp, rows, chunk: int):
    """``X^H X`` of the rows held here (``count x count``, Hermitian).

    On the device the real view ``Y = [Re X | Im X]`` (columns interleaved) gives it with one symmetric
    rank-k update, half the products of a complex matrix product:
    ``X^H X = (Y^T Y)[re, re] + (Y^T Y)[im, im] + i ((Y^T Y)[re, im] - (Y^T Y)[im, re])``.
    """
    own, count = rows.shape
    if xp is np:
        gram = np.zeros((count, count), dtype=np.complex128)
        for r0 in range(0, own, chunk):
            block = rows[r0 : min(r0 + chunk, own)]
            gram += block.conj().T @ block
        return gram
    from cupy import cublas

    real = xp.zeros((2 * count, 2 * count), dtype=xp.float64)
    for r0 in range(0, own, chunk):
        block = xp.ascontiguousarray(rows[r0 : min(r0 + chunk, own)]).view(xp.float64)
        cublas.syrk("T", block, out=real, alpha=1.0, beta=1.0, lower=True)
    real = xp.tril(real) + xp.tril(real, -1).T
    gram = (real[0::2, 0::2] + real[1::2, 1::2]) + 1j * (real[0::2, 1::2] - real[1::2, 0::2])
    return xp.ascontiguousarray(gram)


def _right_solve(xp, rows, lower, chunk: int) -> None:
    """``X <- X (L^H)^-1`` in place, by chunks of rows: ``conj(L) X^T = Y^T``, a triangular solve on the
    device (half the products of the multiplication by the inverse)."""
    own = rows.shape[0]
    if xp is np:
        inverse = _solve_triangular_inverse(xp, lower)
        for r0 in range(0, own, chunk):
            r1 = min(r0 + chunk, own)
            rows[r0:r1] = rows[r0:r1] @ inverse
        return
    from cupyx.scipy.linalg import solve_triangular

    conjugate = xp.ascontiguousarray(xp.conj(lower))
    for r0 in range(0, own, chunk):
        r1 = min(r0 + chunk, own)
        rows[r0:r1] = solve_triangular(conjugate, rows[r0:r1].T, lower=True, overwrite_b=True).T


def _cholesky_qr_rows(xp, rows, group: ColumnGroup, dimension: int) -> None:
    """Orthonormalize the columns of the row-distributed block ``rows`` in place.

    Shifted Cholesky QR, three passes (the first shifted so that it survives the conditioning of a
    filtered block): ``G = X^H X`` summed over the ranks (:func:`_gram`), ``G + s I = L L^H`` on the
    pool root, ``L`` broadcast, ``X <- X (L^H)^-1`` by chunks of rows (:func:`_right_solve`).  The
    arithmetic of :func:`.chebff_spinor._distributed_orthonormalize`, with the factor taken on one rank.
    """
    own, count = rows.shape
    chunk = _row_chunk(count)
    eps = float(np.finfo(np.float64).eps)
    for pass_index in range(3):
        gram = group.lower_sum_to_root(_gram(xp, rows, chunk))  # the root factors it (lower triangle)
        status, factor = None, None
        if group.is_root:
            gram = gram + xp.tril(gram, -1).conj().T
            trace = float(xp.real(xp.trace(gram)))
            shift = 11.0 * (dimension * count + count * (count + 1)) * eps * trace if pass_index == 0 else 0.0
            for _attempt in range(4):
                try:
                    lower = xp.linalg.cholesky(gram + shift * xp.eye(count, dtype=xp.complex128))
                    if bool(xp.all(xp.isfinite(lower))):
                        factor = lower
                        break
                except Exception:  # LinAlgError (NumPy) or the backend's equivalent: not positive definite
                    pass
                shift = max(shift * 100.0, 1.0e-14 * trace)
            if factor is None:
                status = "the block is numerically rank deficient: Cholesky QR failed"
        del gram
        status = group.bcast(status)
        if status is not None:
            raise FloatingPointError(status)
        factor = group.bcast_array(factor, (count, count), np.complex128)
        _right_solve(xp, rows, factor, chunk)
        del factor


def _orthonormal_ritz(
    xp,
    operator,
    holder: list,
    group: ColumnGroup,
    layout: ColumnLayout,
    dimension: int,
    requested: int,
    residuals: bool,
    chunk: int,
    stats: _Stats,
):
    """Orthonormalization and Rayleigh-Ritz of the filtered columns ``holder[0]``; see the module text.

    The filtered block is taken out of ``holder`` and released once it has moved to rows.  Returns
    this rank's rotated columns, the Ritz values (host, ascending) and the residual norms of the
    ``requested`` lowest pairs (host) or ``None``.
    """
    with stats.timed("orthonormalize_seconds"):
        filtered = holder.pop()
        rows = group.to_rows(filtered, layout, dimension)
        del filtered
        _cholesky_qr_rows(xp, rows, group, dimension)
    with stats.timed("ritz_seconds"):
        columns = group.to_columns(rows, layout, dimension)
        width = int(columns.shape[1])
        for c0 in range(0, width, chunk):  # H X written over X, chunk by chunk
            c1 = min(c0 + chunk, width)
            columns[:, c0:c1] = operator.apply(xp.ascontiguousarray(columns[:, c0:c1]))
        applied = group.to_rows(columns, layout, dimension)
        del columns
        own, count = rows.shape
        row_chunk = _row_chunk(count)
        projection = xp.zeros((count, count), dtype=xp.complex128)
        for r0 in range(0, own, row_chunk):
            r1 = min(r0 + row_chunk, own)
            projection += rows[r0:r1].conj().T @ applied[r0:r1]
        lower = group.lower_sum_to_root(projection)  # the root diagonalizes it
        del projection
        values = rotations = None
        if group.is_root:
            hermitian = lower + xp.tril(lower, -1).conj().T
            values, rotations = xp.linalg.eigh(hermitian)
            del hermitian
            values = np.ascontiguousarray(group._host(values), dtype=np.float64)
        del lower
        values = group.bcast(values)
        rotations = group.bcast_array(rotations, (count, count), np.complex128)
        for r0 in range(0, own, row_chunk):
            r1 = min(r0 + row_chunk, own)
            rows[r0:r1] = rows[r0:r1] @ rotations
    norms = None
    if residuals:
        with stats.timed("residual_seconds"):
            wanted = rotations[:, :requested]
            levels = xp.asarray(np.asarray(values[:requested], dtype=np.float64))
            squares = xp.zeros(requested, dtype=xp.float64)
            for r0 in range(0, own, row_chunk):
                r1 = min(r0 + row_chunk, own)
                difference = applied[r0:r1] @ wanted
                difference -= rows[r0:r1, :requested] * levels[None, :]
                squares += (difference.real * difference.real + difference.imag * difference.imag).sum(axis=0)
                del difference
            norms = np.sqrt(np.asarray(group._host(group.sum(squares)), dtype=np.float64))
    del applied, rotations
    with stats.timed("ritz_seconds"):
        local = group.to_columns(rows, layout, dimension, copy=False)
    del rows
    return local, np.asarray(values, dtype=np.float64), norms


# ------------------------------------------------------------------- the solves
def _bounds(
    xp, operator, dimension: int, steps: int, seed: int, group: ColumnGroup, stats: _Stats, *, middle=False,
    rule: str = "parsec",
):
    """The Lanczos interval of the pool root (every rank runs the same short Lanczos on its replica):
    ``(upper, theta_min)``, with ``middle`` also PARSEC's middle of the spectrum; ``rule`` and ``steps`` as
    :func:`parsec_python.Eigensolvers.spectral_bounds.bound_request` gives them for the filter's settings."""
    with stats.timed("lanczos_seconds"):
        found = _lanczos_upper_bound(
            xp, operator, dimension, steps, np.random.default_rng(seed), rule=rule,
            **({"with_middle": True} if middle else {}),
        )
    return group.bcast(tuple(float(value) for value in found))


def _random_columns(xp, seed: int, dimension: int, working: int, layout: ColumnLayout, rank: int, chunk: int):
    """This rank's columns of the start basis of :func:`.chebff_spinor._first_solve`: the same stream
    (``default_rng(seed)``, real then imaginary parts, ``chunk`` columns at a time), replayed in full."""
    rng = xp.random.default_rng(seed)
    local = xp.empty((dimension, layout.widths[rank]), dtype=xp.complex128)
    spans = layout.spans(rank)
    for c0 in range(0, working, chunk):
        c1 = min(c0 + chunk, working)
        real = rng.uniform(-1.0, 1.0, (dimension, c1 - c0))
        imaginary = rng.uniform(-1.0, 1.0, (dimension, c1 - c0))
        for g0, l0, count in spans:
            first, last = max(g0, c0), min(g0 + count, c1)
            if last > first:
                local[:, l0 + first - g0 : l0 + last - g0] = real[:, first - c0 : last - c0] + 1j * imaginary[
                    :, first - c0 : last - c0
                ]
        del real, imaginary
    return local


def _filter_group_fused(
    xp, operator, source, out, columns: slice, degree: int, center, half_span, sigma_one, entering, produced, chunk, stats
):
    """:func:`.chebff_spinor._filter_group` with every step of the recurrence in the stencil kernel of the
    operator (:meth:`..Hamiltonian.device_bloch.DeviceBlochOperator.chebyshev_step`): the same steps and
    coefficients, the elementwise passes inside the kernel."""
    start, stop = columns.start, columns.stop
    for c0 in range(start, stop, chunk):
        c1 = min(c0 + chunk, stop)
        local = slice(c0 - start, c1 - start)
        width = c1 - c0
        previous = xp.ascontiguousarray(source[:, c0:c1])
        current = operator.chebyshev_step(previous, None, center, sigma_one / half_span)
        stats.add("applications", 1)
        stats.add("application_columns", width)
        if degree >= 2:
            sigma_before = xp.asarray(np.ascontiguousarray(entering[:, local]))
            sigma_after = xp.asarray(np.ascontiguousarray(produced[:, local]))
            for step in range(degree - 1):
                following = operator.chebyshev_step(
                    current, previous, center, 2.0 / half_span, sigma_before[step], sigma_after[step]
                )
                stats.add("applications", 1)
                stats.add("application_columns", width)
                previous, current = current, following
        del previous
        out[:, c0:c1] = current
        del current


def _filter_local(
    xp, operator, source, layout: ColumnLayout, rank: int, groups, center, half_span, sigma_one, chunk, stats=None
):
    """Filter this rank's columns of every degree group ``(start, stop, degree, entering, produced)``.

    An operator with :attr:`fused_recurrence` takes every step of the recurrence in its stencil kernel.
    """
    out = xp.empty_like(source)
    fused = bool(getattr(operator, "fused_recurrence", False)) and stats is not None
    for start, stop, degree, entering, produced in groups:
        for g0, l0, count in layout.spans(rank):
            first, last = max(g0, start), min(g0 + count, stop)
            if last <= first:
                continue
            (_filter_group_fused if fused else _filter_group)(
                xp,
                operator,
                source,
                out,
                slice(l0 + first - g0, l0 + last - g0),
                int(degree),
                center,
                half_span,
                sigma_one,
                entering[:, first - start : last - start],
                produced[:, first - start : last - start],
                chunk,
                *((stats,) if fused else ()),
            )
    return out


def solve_column_eigval(
    operator: Any,
    requested_states: int,
    *,
    settings,
    group: ColumnGroup,
    state: ColumnEigvalState | None = None,
    xp=None,
    sync: Callable[[], None] | None = None,
    converge_first: bool = False,
    tolerance: float | None = None,
    max_extra_cycles: int | None = None,
    residual_period: int | None = None,
) -> SpinorEigvalResult:
    """One eigensolve of the SCF loop with the basis shared by the ranks of ``group``.

    The policy, arguments and result of :func:`.chebff_spinor.solve_spinor_eigval`; ``operator`` is
    this rank's replica of the whole operator; ``result.vectors`` and ``result.state.vectors`` are this
    rank's columns (``result.state.layout``); the eigenvalues and residuals are the same on every rank.
    """
    if xp is None:
        from ..backends.cupy import require_cupy

        xp, _ = require_cupy()
    stats = _Stats(sync)
    counted = _CountingOperator(operator, stats)
    dimension = int(operator.shape[0])
    working = min(int(requested_states) + int(settings.safety_buffer), dimension)
    if not 1 <= requested_states <= dimension:
        raise ValueError(f"requested_states must be between 1 and {dimension}, got {requested_states}")
    chunk = _filter_columns()
    period = _residual_period() if residual_period is None else int(residual_period)
    layout = ColumnLayout(working, settings.subspace.block_size, group.size)
    before = group.statistics()
    reusable = (
        state is not None
        and state.vectors is not None
        and state.operator_dimension == dimension
        and state.working_states == working
        and state.layout == layout
    )
    if not reusable:
        chebff = settings.chebff
        relative = None
        extra = 0
        if converge_first:
            relative = float(settings.chebdav.convergence_tolerance if tolerance is None else tolerance)
            if max_extra_cycles is None:
                env = os.environ.get("PARSEC_KPOINT_EXTRA_CYCLES", "").strip()
                max_extra_cycles = int(env) if env.isdigit() else 24
            extra = int(max_extra_cycles)
        vectors = _random_columns(xp, chebff.random_seed, dimension, working, layout, group.rank, chunk)
        bound_rule, bound_steps = bound_request(chebff)
        upper_bound, smallest_ritz = _bounds(
            xp, counted, dimension, bound_steps, chebff.random_seed + 1, group, stats, rule=bound_rule
        )
        lower_bound = (2.0 * smallest_ritz + upper_bound) / 3.0  # chebff.f90z
        total_cycles = chebff.filter_cycles if not converge_first else max(chebff.filter_cycles, chebff.filter_cycles + extra)
        residuals = None
        eigenvalues = np.empty(working)
        for cycle in range(total_cycles):
            half_span = 0.5 * (upper_bound - lower_bound)
            if half_span <= 0.0:
                raise ValueError("upper_bound must be greater than lower_bound")
            center = 0.5 * (upper_bound + lower_bound)
            denominator = smallest_ritz - center
            if denominator == 0.0:
                raise ValueError("reference_eigenvalue cannot equal the interval center")
            sigma_one = half_span / denominator
            entering, produced, _ = _chain(sigma_one, sigma_one, chebff.polynomial_degree)
            shape = (len(entering), working)
            one_degree = [
                (
                    0,
                    working,
                    chebff.polynomial_degree,
                    np.broadcast_to(np.asarray(entering)[:, None], shape),
                    np.broadcast_to(np.asarray(produced)[:, None], shape),
                )
            ]
            with stats.timed("filter_seconds"):
                # the filtered block lives only in ``holder``: the Rayleigh-Ritz pass releases it
                holder = [
                    _filter_local(
                        xp, counted, vectors, layout, group.rank, one_degree, center, half_span, sigma_one, chunk, stats
                    )
                ]
                del vectors
            _check_finite_columns(xp, holder[0], group)
            last = cycle + 1 == total_cycles
            measure = (converge_first and cycle + 1 >= chebff.filter_cycles) or (last and period > 0)
            vectors, eigenvalues, residuals = _orthonormal_ritz(
                xp, counted, holder, group, layout, dimension, int(requested_states), measure, chunk, stats
            )
            smallest_ritz = float(eigenvalues[0])
            largest_ritz = float(eigenvalues[-1])
            if largest_ritz >= upper_bound:
                new_upper = largest_ritz + 0.5 * (largest_ritz - upper_bound) + 1.0
                new_lower = min(lower_bound, (3.0 * lower_bound + new_upper) / 4.0)
                lower_bound, upper_bound = new_lower, new_upper
            else:
                lower_bound = min(
                    largest_ritz + 0.001 * (upper_bound - smallest_ritz),
                    largest_ritz + 0.05 * abs(largest_ritz),
                )
            if converge_first and cycle + 1 >= chebff.filter_cycles:
                scale = max(float(np.max(np.abs(eigenvalues[:requested_states]))), 1.0e-8)
                if bool(residuals.max() <= relative * scale) or last:
                    stats.add("extra_cycles", cycle + 1 - chebff.filter_cycles)
                    break
        new_state = ColumnEigvalState(
            vectors=vectors,
            eigenvalues=eigenvalues,
            working_states=working,
            operator_dimension=dimension,
            first_filter=True,
            filter_lower_bound=None,
            filters_completed=0,
            layout=layout,
        )
        stats.add("first_solves", 1)
    else:
        subspace = settings.subspace
        bound_rule, bound_steps = bound_request(subspace)
        upper, smallest, middle = _bounds(
            xp,
            counted,
            dimension,
            bound_steps,
            subspace.random_seed + state.filters_completed,
            group,
            stats,
            middle=True,
            rule=bound_rule,
        )
        # the lower-bound rule of every SUBSPACE filter (Smart_Subspace), as the one-device solver takes it
        lower_bound, upper_bound = subspace_filter_bounds(
            subspace.lower_bound_rule,
            float(np.max(state.eigenvalues)),
            state.filter_lower_bound,
            state.first_filter,
            _Bounds(upper, smallest, middle),
        )
        degree = _adapt_polynomial_degree(subspace.polynomial_degree, lower_bound, upper_bound)
        blocks = subspace_filter_blocks(working, subspace.block_size, degree, subspace.degree_delta)
        half_span = 0.5 * (upper_bound - lower_bound)
        center = 0.5 * (upper_bound + lower_bound)
        sigma_one = half_span / (lower_bound - center)  # the reference is the lower edge: -1
        # the sigma sequences of the one-device step (carried from one block to the next)
        entering_by_degree: dict[int, list[np.ndarray]] = {}
        produced_by_degree: dict[int, list[np.ndarray]] = {}
        carried: float | None = None
        for block in blocks:
            initial = sigma_one if (subspace.reset_recurrence_per_block or carried is None) else carried
            entering, produced, final = _chain(sigma_one, initial, block.degree)
            carried = final
            width = block.stop - block.start
            entering_by_degree.setdefault(block.degree, []).append(np.repeat(np.asarray(entering)[:, None], width, axis=1))
            produced_by_degree.setdefault(block.degree, []).append(np.repeat(np.asarray(produced)[:, None], width, axis=1))
        groups = []
        for group_degree in sorted({block.degree for block in blocks}):
            members = [block for block in blocks if block.degree == group_degree]
            start, stop = members[0].start, members[-1].stop
            if sum(block.stop - block.start for block in members) != stop - start:
                raise RuntimeError("blocks of equal degree are not contiguous")
            groups.append(
                (
                    start,
                    stop,
                    group_degree,
                    np.concatenate(entering_by_degree[group_degree], axis=1),
                    np.concatenate(produced_by_degree[group_degree], axis=1),
                )
            )
        with stats.timed("filter_seconds"):
            holder = [
                _filter_local(
                    xp, counted, state.vectors, layout, group.rank, groups, center, half_span, sigma_one, chunk, stats
                )
            ]
            state.vectors = None  # consumed
        _check_finite_columns(xp, holder[0], group)
        completed = state.filters_completed + 1
        measure = period > 0 and completed % period == 0
        vectors, eigenvalues, residuals = _orthonormal_ritz(
            xp, counted, holder, group, layout, dimension, int(requested_states), measure, chunk, stats
        )
        if eigenvalues.shape != (working,):
            raise RuntimeError("Rayleigh-Ritz changed the filtered subspace's eigenvalue count")
        new_state = ColumnEigvalState(
            vectors=vectors,
            eigenvalues=eigenvalues,
            working_states=working,
            operator_dimension=dimension,
            first_filter=False,
            filter_lower_bound=lower_bound,
            filters_completed=completed,
            layout=layout,
        )
        stats.add("subspace_steps", 1)
        stats.add("polynomial_degree", degree)
    after = group.statistics()
    for name, value in after.items():
        stats.add(name, value - before.get(name, 0.0))
    return SpinorEigvalResult(
        eigenvalues=np.asarray(new_state.eigenvalues[:requested_states], dtype=np.float64).copy(),
        vectors=new_state.vectors,
        state=new_state,
        stats=dict(stats.values),
        residual_norms=residuals,
    )


def requested_spans(layout: ColumnLayout, rank: int, requested: int) -> list[tuple[int, int, int]]:
    """``(first global column, first local column, count)`` of this rank's columns below ``requested``."""
    spans = []
    for g0, l0, count in layout.spans(rank):
        take = min(count, int(requested) - g0)
        if take > 0:
            spans.append((g0, l0, take))
    return spans


__all__ = [
    "ColumnEigvalState",
    "ColumnGroup",
    "ColumnLayout",
    "requested_spans",
    "solve_column_eigval",
]
