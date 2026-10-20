"""A sector basis shared by MPI ranks: molecular dynamics on several nodes.

A structure without point-group symmetry is solved as the single sector of
the identity.  On one node the GPUs share its ``N x m`` basis column by
column (:mod:`distributed_state`).  Here the ranks of a communicator share
it, one GPU each, so that the GPUs of several nodes work on one sector:

* every rank holds all grid rows of its own columns, with a full replica of
  the sector operator, and filters them without communication;
* the Rayleigh--Ritz step moves the filtered columns and ``H`` times them to
  row blocks (``Alltoallv``), forms its part of ``X.T X`` and ``X.T H X``,
  sums the parts (``Allreduce``), lets the root solve the small generalized
  problem and broadcast the coefficients, rotates its rows and moves them
  back to columns;
* the density and the nonlocal force terms of the sector are summed over
  the ranks by the commands of :class:`experimental.mpi_scf.MPISectorContext`.

:class:`MPIShardGroup` has the methods of :class:`distributed_state.SectorDeviceGroup`
that :func:`distributed_state.run_distributed_chebff` and
:func:`distributed_state.run_distributed_subspace` call, with the rank in
place of the device index, so that every rank runs those solves at the same
time on its own columns.  The columns of a rank are its shares of the lower
and of the upper half of the filter blocks (:func:`distributed_state.interleaved_ranges`),
which keeps the filter work of the ranks balanced in the first solve and in
later passes, which filter the two halves at different degrees.  The trial
basis is PARSEC's DLARNV stream in column order, each rank drawing its own
columns after skipping ahead, so that it equals the single-device basis.

The exchanges go in rounds of columns, each send and receive buffer at most
``PARSEC_MPI_SHARED_EXCHANGE_BYTES`` (default 1 GiB) on every rank, and pass
device arrays to MPI where ``MPICH_GPU_SUPPORT_ENABLED=1``
(``PARSEC_MPI_SHARED_TRANSPORT=auto``, the default, else ``cuda`` or
``host``, which stages them through the host).  On four nodes of A100 (16
ranks) they reached about 17 GB/s per rank, near what one network interface
carries, and the host-staged ones 3 GB/s.  A filter pipelined with the move to
rows (a round of columns in flight while the next one filters) was measured
and left out: the nonblocking ``Ialltoallv`` of device arrays did not move
beside the filter kernels, and the filter in rounds took twice as long
(C795H300 MD on 16 ranks: 8.1 against 6.8 s per step).

The small solve runs on the root's device where
:func:`rayleigh_ritz.dense_solve_on_device` says so, else on its host.
A failed stability audit raises :class:`rayleigh_ritz.GeneralizedRitzStabilityError`
on every rank; :meth:`MPIShardGroup.orthonormal_ritz` then orthonormalizes
the rows by shifted Cholesky QR (three passes) instead of gathering the
basis on one device.
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any
from weakref import ref

import numpy as np

from parsec_python.Eigensolvers.lapack_random import (
    _MODULUS,
    _MULTIPLIER,
    _integer_to_seed,
    _seed_to_integer,
)

from ..backends.cupy import device_stage, require_cupy
from .chebyshev import FilterBlock
from .distributed_filter import block_partitions, starting_sigmas
from .distributed_state import (
    DistributedBasis,
    SectorDeviceGroup,
    _NoStage,
    _as_layout,
    ColumnLayout,
    interleaved_ranges,
)
from .filter_graph import BlockFilterGraphs
from .lapack_random import LapackRandom
from .rayleigh_ritz import (
    GeneralizedRitzStabilityError,
    _lower_triangle_product,
    dense_solve_on_device,
    solve_whitened_ritz,
    solve_whitened_ritz_on_device,
)


def shared_transport() -> str:
    """``PARSEC_MPI_SHARED_TRANSPORT``: ``cuda``, ``host`` or ``auto`` (the default).

    ``auto`` passes device arrays to MPI where ``MPICH_GPU_SUPPORT_ENABLED``
    is ``1`` and stages them through the host otherwise.
    """

    value = os.environ.get("PARSEC_MPI_SHARED_TRANSPORT", "auto").strip().lower() or "auto"
    if value not in {"auto", "cuda", "host"}:
        raise ValueError("PARSEC_MPI_SHARED_TRANSPORT must be auto, cuda or host")
    if value == "auto":
        return "cuda" if os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "0").strip() == "1" else "host"
    return value


def exchange_bytes() -> int:
    """``PARSEC_MPI_SHARED_EXCHANGE_BYTES``: the send (and receive) buffer of one round."""

    raw = os.environ.get("PARSEC_MPI_SHARED_EXCHANGE_BYTES", str(1 << 30)).strip()
    value = int(raw)
    if value < 8:
        raise ValueError("PARSEC_MPI_SHARED_EXCHANGE_BYTES must be at least 8")
    return value


def row_edges(rows: int, size: int) -> list[int]:
    """Balanced contiguous row blocks, one per rank."""

    quotient, remainder = divmod(int(rows), int(size))
    edges = [0]
    for rank in range(int(size)):
        edges.append(edges[-1] + quotient + (1 if rank < remainder else 0))
    return edges


def _skip(seed, draws: int):
    """The DLARNV seed after ``draws`` values of the stream."""

    state = _seed_to_integer(seed)
    return _integer_to_seed((state * pow(_MULTIPLIER, int(draws), _MODULUS)) % _MODULUS)


class SharedBasis(DistributedBasis):
    """An ``N x m`` basis of which this rank holds the columns of ``layout.parts[rank]``.

    ``blocks`` has one entry per rank, the local block at the index of this
    rank and ``None`` elsewhere, so that the code written for the blocks of
    the devices of a group (:class:`distributed_state.DistributedBasis`)
    reaches the local one through :meth:`MPIShardGroup.map`.  ``shape`` is
    the shape of the whole basis, ``nbytes`` the bytes of this rank.
    """

    def __init__(self, group: "MPIShardGroup", block, layout) -> None:
        layout = _as_layout(layout)
        if len(layout.parts) != group.size:
            raise ValueError("a shared basis needs the column ranges of every rank")
        width = layout.widths[group.rank]
        if block.ndim != 2 or int(block.shape[1]) != width or not block.flags.f_contiguous:
            raise ValueError("the local block must be Fortran-ordered with the columns of this rank")
        blocks = [None] * group.size
        blocks[group.rank] = block
        self.group = group
        self.blocks = tuple(blocks)
        self.layout = layout
        self.shape = (int(block.shape[0]), layout.columns)

    @property
    def local(self):
        return self.blocks[self.group.rank]

    @property
    def nbytes(self) -> int:
        return int(self.local.nbytes)

    def __getitem__(self, key) -> "SharedBasis":
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] != slice(None)
            or not isinstance(key[1], slice)
            or key[1].start not in (None, 0)
            or key[1].step not in (None, 1)
        ):
            raise IndexError("a shared basis supports only basis[:, :count]")
        layout = self.layout.leading(len(range(self.shape[1])[key[1]]))
        width = layout.widths[self.group.rank]
        return SharedBasis(self.group, self.local[:, :width], layout)

    def take(self) -> list:
        blocks, self.blocks = list(self.blocks), ()
        return blocks

    def density(self, builder, occupations: np.ndarray, volume_element: float) -> np.ndarray:
        """This rank's part of the density of the leading ``len(occupations)`` columns.

        The ranks' parts are summed by the density command of the MPI
        context.  A rank that holds none of those columns returns zeros.
        """

        cp, _ = require_cupy()
        weights = np.asarray(occupations, dtype=np.float64)
        if weights.ndim != 1 or weights.size > self.shape[1]:
            raise ValueError("occupations do not select leading columns of this basis")
        selected = self[:, : weights.size]
        block = selected.local
        if not block.shape[1]:
            return np.zeros(int(block.shape[0]), dtype=np.float64)
        held = [weights[column : column + count] for column, _offset, count in selected.layout.spans(self.group.rank)]
        with cp.cuda.Device(self.group.device):
            return builder(block, held[0] if len(held) == 1 else np.concatenate(held), volume_element)


class _RowExchange:
    """The rounds of one move of column blocks to row blocks (or back), for one layout.

    Round ``j`` moves the local columns ``j * width`` onwards of every rank;
    ``width`` is the same on every rank, so that all of them take part in
    every round, an empty one included.
    """

    def __init__(self, group: "MPIShardGroup", layout, rows: int, width: int) -> None:
        self.group = group
        self.layout = layout
        self.rows = int(rows)
        self.edges = row_edges(rows, group.size)
        self.own = self.edges[group.rank + 1] - self.edges[group.rank]
        self.width = max(1, int(width))
        widest = max(layout.widths) if layout.widths else 0
        self.rounds = -(-widest // self.width) if widest else 0

    def counts(self, step: int) -> list[int]:
        first = step * self.width
        return [max(0, min(self.width, w - first)) for w in self.layout.widths]

    def heights(self) -> list[int]:
        return [self.edges[q + 1] - self.edges[q] for q in range(self.group.size)]

    # Columns to rows.

    def pack_columns(self, cp, local, step: int):
        """The send buffer and counts of round ``step`` from this rank's columns."""

        counts = self.counts(step)
        mine = counts[self.group.rank]
        first = step * self.width
        send_counts = [height * mine for height in self.heights()]
        send = cp.empty(max(1, sum(send_counts)), dtype=cp.float64)
        offset = 0
        for q, height in enumerate(self.heights()):
            if height and mine:
                piece = send[offset : offset + height * mine].reshape((mine, height)).T
                piece[...] = local[self.edges[q] : self.edges[q + 1], first : first + mine]
            offset += height * mine
        receive_counts = [self.own * count for count in counts]
        receive = cp.empty(max(1, sum(receive_counts)), dtype=cp.float64)
        return send, send_counts, receive, receive_counts

    def unpack_rows(self, receive, step: int, result) -> None:
        """Lay the received pieces of round ``step`` into the row block ``result``."""

        counts = self.counts(step)
        first = step * self.width
        offset = 0
        for r, count in enumerate(counts):
            if self.own and count:
                piece = receive[offset : offset + self.own * count].reshape((count, self.own)).T
                for column, local_offset, span in self.layout.spans(r, first, first + count):
                    start = local_offset - first
                    result[:, column : column + span] = piece[:, start : start + span]
            offset += self.own * count

    # Rows to columns.

    def pack_rows(self, cp, row_block, step: int):
        counts = self.counts(step)
        mine = counts[self.group.rank]
        first = step * self.width
        send_counts = [self.own * count for count in counts]
        send = cp.empty(max(1, sum(send_counts)), dtype=cp.float64)
        offset = 0
        for r, count in enumerate(counts):
            if self.own and count:
                piece = send[offset : offset + self.own * count].reshape((count, self.own)).T
                for column, local_offset, span in self.layout.spans(r, first, first + count):
                    start = local_offset - first
                    piece[:, start : start + span] = row_block[:, column : column + span]
            offset += self.own * count
        receive_counts = [height * mine for height in self.heights()]
        receive = cp.empty(max(1, sum(receive_counts)), dtype=cp.float64)
        return send, send_counts, receive, receive_counts

    def unpack_columns(self, receive, step: int, target) -> None:
        counts = self.counts(step)
        mine = counts[self.group.rank]
        first = step * self.width
        offset = 0
        for q, height in enumerate(self.heights()):
            if height and mine:
                piece = receive[offset : offset + height * mine].reshape((mine, height)).T
                target[self.edges[q] : self.edges[q + 1], first : first + mine] = piece
            offset += height * mine


class MPIShardGroup:
    """The ranks of ``comm`` sharing the basis of one sector, one device each.

    ``operator`` is this rank's replica of the sector operator, on its
    device.  ``devices`` has one entry per rank: index ``i`` stands for rank
    ``i``, as the device index does in :class:`distributed_state.SectorDeviceGroup`.
    """

    owner = 0

    def __init__(self, comm, operator, *, transport: str | None = None) -> None:
        self.comm = comm
        self.rank, self.size = int(comm.rank), int(comm.size)
        self.devices = tuple(range(self.size))
        self.device = int(operator.effective_potential.device.id)
        # The devices of this process, where code reads real device ids.
        self.local_devices = (self.device,)
        self._operator_ref = ref(operator)
        self.transport = shared_transport() if transport is None else transport
        if self.transport not in {"cuda", "host"}:
            raise ValueError("transport must be cuda or host")
        try:
            from mpi4py import MPI

            self._mpi = MPI if type(comm).__module__.startswith("mpi4py") else None
        except ImportError:
            self._mpi = None
        self.seconds = {
            name: 0.0
            for name in ("filter", "apply", "to_rows", "gram", "reduce", "dense", "rotate", "to_columns")
        }
        self.passes = 0
        self.exchanged_bytes = 0

    # ----- the interface of SectorDeviceGroup ---------------------------

    def map(self, function):
        """Run ``function(rank, device)`` for this rank only; a one-item list."""

        cp, _ = require_cupy()
        with cp.cuda.Device(self.device):
            value = function(self.rank, self.device)
            cp.cuda.get_current_stream().synchronize()
        return [value]

    def _operator(self, device=None):
        operator = self._operator_ref()
        if operator is None:
            raise ReferenceError("the operator of a shared basis was released")
        return operator

    def column_layout(self, blocks) -> ColumnLayout:
        """Each rank's shares of the two halves of ``blocks``, or contiguous shares."""

        blocks = tuple(blocks)
        ranges = interleaved_ranges(blocks, self.size)
        if ranges is None:
            partitions = block_partitions(blocks, self.size)
            ranges = [(blocks[first].start, blocks[last - 1].stop) for first, last in partitions]
            end = blocks[-1].stop
            ranges.extend([(end, end)] * (self.size - len(ranges)))
        return ColumnLayout(tuple(ranges))

    def random_basis(self, generator: LapackRandom, rows: int, layout) -> SharedBasis:
        """This rank's columns of PARSEC's column-major DLARNV stream.

        Each range starts ``column * rows`` values into the stream, the
        generator is left where the whole basis leaves it, and the local
        block equals those columns of the single-device trial basis.
        """

        cp, _ = require_cupy()
        layout = _as_layout(layout)
        rows = int(rows)
        on_device = os.environ.get("PARSEC_CUPY_DEVICE_RANDOM", "0").lower() in {"1", "true", "on"}
        start = generator.seed
        with cp.cuda.Device(self.device):
            block = cp.empty((rows, layout.widths[self.rank]), dtype=cp.float64, order="F")
            for column, offset, count in layout.spans(self.rank):
                generator.seed = _skip(start, column * rows)
                target = block[:, offset : offset + count]
                if on_device:
                    generator.device_uniform_minus_1_1(target.shape, column_major=True, out=target)
                else:
                    target.set(generator.uniform_minus_1_1((rows, count), column_major=True))
            cp.cuda.get_current_stream().synchronize()
        generator.seed = _skip(start, rows * layout.columns)
        return SharedBasis(self, block, layout)

    def filter(self, basis: SharedBasis, blocks, lower, upper, reference, reset) -> SharedBasis:
        """Chebyshev-filter this rank's columns in place; no communication but one broadcast.

        The plan and the interval are the root's, so that every rank filters
        with the same degrees whatever its own spectral bound gave.
        """

        cp, _ = require_cupy()
        started = perf_counter()
        blocks, lower, upper, reference = self.comm.bcast(
            (tuple(blocks), float(lower), float(upper), float(reference)), root=self.owner
        )
        partitions = SectorDeviceGroup._whole_block_partitions(basis.layout, blocks)
        if partitions is None:
            raise RuntimeError("the columns of a shared basis are no whole filter blocks of this plan")
        sigmas = starting_sigmas(blocks, lower, upper, reset, reference)
        layout = basis.layout
        block = basis.local
        basis.take()
        local, starts, offset = [], [], 0
        for first, last in partitions[self.rank]:
            shift = offset - blocks[first].start
            local.extend(FilterBlock(b.start + shift, b.stop + shift, b.degree) for b in blocks[first:last])
            starts.extend(sigmas[first:last])
            offset = local[-1].stop
        result = block
        if local:
            operator = self._operator()
            with cp.cuda.Device(self.device):
                workspace = getattr(operator, "_filter_graph_workspace", None)
                if workspace is None:
                    workspace = BlockFilterGraphs(operator)
                    operator._filter_graph_workspace = workspace
                result = workspace.apply(
                    block, tuple(local), lower, upper, reference, reset, initial_sigma=starts, out=block
                )
                cp.cuda.get_current_stream().synchronize()
        self.seconds["filter"] += perf_counter() - started
        return SharedBasis(self, result, layout)

    def ritz(self, operator: Any, filtered: SharedBasis, *, stages: bool):
        """Generalized Rayleigh--Ritz of the filtered basis over all ranks; consumes ``filtered``.

        Returns the Ritz values (on every rank), the rotated basis, written
        back into the local block, and the whitened projected Hamiltonian
        (on the root; an empty array on the other ranks).  On a failed
        stability audit every rank raises
        :class:`rayleigh_ritz.GeneralizedRitzStabilityError` with the intact
        filtered basis as ``error.filtered``.
        """

        cp, _ = require_cupy()
        layout = filtered.layout
        rows = int(filtered.shape[0])
        columns = filtered.local
        filtered.take()

        def stage(name):
            return device_stage(operator, name) if stages else _NoStage()

        with cp.cuda.Device(self.device):
            with stage("subspace_ritz_hamiltonian_seconds"):
                started = perf_counter()
                applied = cp.empty_like(columns, order="F")
                if columns.shape[1]:
                    operator.apply_into(columns, applied)
                cp.cuda.get_current_stream().synchronize()
                self.seconds["apply"] += perf_counter() - started
            with stage("subspace_ritz_projection_seconds"):
                started = perf_counter()
                row_applied = self.to_rows(applied, layout, rows)
                del applied
                row_basis = self.to_rows(columns, layout, rows)
                self.seconds["to_rows"] += perf_counter() - started
            with stage("subspace_ritz_projection_seconds"):
                started = perf_counter()
                packed = cp.ascontiguousarray(
                    cp.stack((row_basis.T @ row_basis, _lower_triangle_product(row_basis, row_applied)))
                )
                cp.cuda.get_current_stream().synchronize()
                self.seconds["gram"] += perf_counter() - started
                started = perf_counter()
                packed = self._sum(packed)
                self.seconds["reduce"] += perf_counter() - started
            eigenvalues, coefficients, whitened = self._small_solve(packed)
            del packed
            if eigenvalues is None:
                error = GeneralizedRitzStabilityError(coefficients)
                error.filtered = SharedBasis(self, columns, layout)
                raise error
            with stage("subspace_ritz_rotation_seconds"):
                started = perf_counter()
                # (X C).T = C.T X.T into the row block of H X, which is no longer needed.
                cp.matmul(coefficients.T, row_basis.T, out=row_applied.T)
                del row_basis
                cp.cuda.get_current_stream().synchronize()
                self.seconds["rotate"] += perf_counter() - started
                started = perf_counter()
                columns = self.to_columns(row_applied, layout, rows, target=columns)
                del row_applied
                self.seconds["to_columns"] += perf_counter() - started
        self.passes += 1
        return eigenvalues, SharedBasis(self, columns, layout), whitened

    def orthonormal_ritz(self, operator: Any, filtered: SharedBasis, generator) -> tuple:
        """Robust fallback: shifted Cholesky QR of the rows (three passes), then :meth:`ritz`."""

        cp, _ = require_cupy()
        from cupyx.scipy.linalg import solve_triangular

        layout = filtered.layout
        rows = int(filtered.shape[0])
        columns = filtered.local
        filtered.take()
        with cp.cuda.Device(self.device):
            row_basis = self.to_rows(columns, layout, rows)
            states = int(layout.columns)
            for step in range(3):
                overlap = self._sum(cp.ascontiguousarray(row_basis.T @ row_basis)[None])[0]
                status, factor = None, None
                if self.rank == self.owner:
                    try:
                        matrix = cp.asnumpy(overlap)
                        matrix = 0.5 * (matrix + matrix.T)
                        if step == 0:
                            # Fukaya et al.: a shift that makes the factorization succeed for
                            # any basis of condition number below about 1e15.
                            shift = 11.0 * (rows * states + states * (states + 1)) * np.finfo(float).eps
                            matrix = matrix + shift * np.linalg.norm(matrix, 2) * np.eye(states)
                        factor = np.linalg.cholesky(matrix)
                    except np.linalg.LinAlgError as exc:
                        status = f"shifted Cholesky QR failed: {exc}"
                status = self.comm.bcast(status, root=self.owner)
                if status is not None:
                    raise RuntimeError(
                        "the filtered basis failed the generalized-Ritz stability audit and could not "
                        "be orthonormalized across the ranks: " + status
                    )
                factor = self._broadcast_matrix(factor, states)
                # X <- X L^-T, row by row.
                row_basis = cp.asfortranarray(solve_triangular(factor, row_basis.T, lower=True).T)
            columns = self.to_columns(row_basis, layout, rows, target=columns)
            del row_basis
        return self.ritz(operator, SharedBasis(self, columns, layout), stages=False)

    # ----- collectives ----------------------------------------------------

    def _sync(self) -> None:
        cp, _ = require_cupy()
        cp.cuda.get_current_stream().synchronize()

    def _sum(self, packed):
        """Sum a device array over the ranks; every rank gets the sum."""

        cp, _ = require_cupy()
        self._sync()
        if self.transport == "cuda":
            total = cp.empty_like(packed)
            self.comm.Allreduce(packed, total)
            return total
        host = np.ascontiguousarray(cp.asnumpy(packed))
        total = np.empty_like(host)
        self.comm.Allreduce(host, total)
        return cp.asarray(total)

    def _broadcast_matrix(self, matrix, size: int):
        """The root's ``size x size`` matrix (device or host) as a device array on every rank."""

        cp, _ = require_cupy()
        if self.transport == "cuda":
            buffer = (
                cp.ascontiguousarray(cp.asarray(matrix, dtype=cp.float64))
                if self.rank == self.owner
                else cp.empty((size, size), dtype=cp.float64)
            )
            self._sync()
            self.comm.Bcast(buffer, root=self.owner)
            return cp.asfortranarray(buffer)
        host = (
            np.ascontiguousarray(cp.asnumpy(matrix) if isinstance(matrix, cp.ndarray) else matrix, dtype=np.float64)
            if self.rank == self.owner
            else np.empty((size, size), dtype=np.float64)
        )
        self.comm.Bcast(host, root=self.owner)
        return cp.asfortranarray(cp.asarray(host))

    def _small_solve(self, packed):
        """The root solves; every rank gets the Ritz values and the coefficients.

        Returns ``(None, message, None)`` on every rank when the root's
        solve failed its stability audit.
        """

        cp, _ = require_cupy()
        started = perf_counter()
        states = int(packed.shape[-1])
        status = None
        eigenvalues = coefficients = whitened = None
        if self.rank == self.owner:
            try:
                if dense_solve_on_device():
                    eigenvalues, coefficients, whitened = solve_whitened_ritz_on_device(packed[0], packed[1])
                    eigenvalues = cp.asnumpy(eigenvalues)
                else:
                    eigenvalues, coefficients, whitened = solve_whitened_ritz(
                        cp.asnumpy(packed[0]), cp.asnumpy(packed[1])
                    )
            except GeneralizedRitzStabilityError as error:
                status = str(error) or "generalized Ritz stability audit failed"
        status = self.comm.bcast(status, root=self.owner)
        if status is not None:
            self.seconds["dense"] += perf_counter() - started
            return None, status, None
        eigenvalues = self.comm.bcast(
            None if eigenvalues is None else np.asarray(eigenvalues, dtype=np.float64), root=self.owner
        )
        coefficients = self._broadcast_matrix(coefficients, states)
        if self.rank != self.owner:
            whitened = np.zeros((0, 0), dtype=np.float64)
        self.seconds["dense"] += perf_counter() - started
        return eigenvalues, coefficients, whitened

    def _round_width(self, rows: int) -> int:
        """Columns per rank and round for buffers of at most :func:`exchange_bytes`."""

        return max(1, exchange_bytes() // (8 * max(int(rows), 1)))

    def _specs(self, send, send_counts, receive, receive_counts):
        def offsets(counts):
            return np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)

        send_counts = np.asarray(send_counts, dtype=np.int64)
        receive_counts = np.asarray(receive_counts, dtype=np.int64)
        send_spec = [send, (send_counts, offsets(send_counts))]
        receive_spec = [receive, (receive_counts, offsets(receive_counts))]
        if self._mpi is not None:
            send_spec.append(self._mpi.DOUBLE)
            receive_spec.append(self._mpi.DOUBLE)
        self.exchanged_bytes += 8 * int(send_counts.sum())
        return send_spec, receive_spec

    def _exchange(self, send, send_counts, receive, receive_counts) -> None:
        """One ``Alltoallv`` of float64 buffers (device arrays, or staged through the host)."""

        cp, _ = require_cupy()
        self._sync()
        if self.transport == "cuda":
            send_spec, receive_spec = self._specs(send, send_counts, receive, receive_counts)
            self.comm.Alltoallv(send_spec, receive_spec)
            return
        host_send = np.ascontiguousarray(cp.asnumpy(send))
        host_receive = np.empty(int(receive.size), dtype=np.float64)
        send_spec, receive_spec = self._specs(host_send, send_counts, host_receive, receive_counts)
        self.comm.Alltoallv(send_spec, receive_spec)
        receive.set(host_receive)

    def to_rows(self, local, layout, rows: int):
        """This rank's row block (``rows_rank x m``, Fortran order) of the basis whose columns ranks hold."""

        cp, _ = require_cupy()
        plan = _RowExchange(self, layout, rows, self._round_width(rows))
        result = cp.empty((plan.own, layout.columns), dtype=cp.float64, order="F")
        for step in range(plan.rounds):
            send, send_counts, receive, receive_counts = plan.pack_columns(cp, local, step)
            self._exchange(send, send_counts, receive, receive_counts)
            plan.unpack_rows(receive, step, result)
            del send, receive
        self._sync()
        return result

    def to_columns(self, row_block, layout, rows: int, target=None):
        """The local columns (``rows x w_rank``, Fortran order) of a basis whose row blocks ranks hold."""

        cp, _ = require_cupy()
        plan = _RowExchange(self, layout, rows, self._round_width(rows))
        mine = layout.widths[self.rank]
        if target is None or tuple(target.shape) != (rows, mine) or not target.flags.f_contiguous:
            target = cp.empty((rows, mine), dtype=cp.float64, order="F")
        for step in range(plan.rounds):
            send, send_counts, receive, receive_counts = plan.pack_rows(cp, row_block, step)
            self._exchange(send, send_counts, receive, receive_counts)
            plan.unpack_columns(receive, step, target)
            del send, receive
        self._sync()
        return target


def attach_shared_group(operator, comm) -> MPIShardGroup:
    """Give ``operator`` (this rank's replica of the shared sector) its group of ranks."""

    group = MPIShardGroup(comm, operator)
    operator._mpi_shared_group = group
    return group


__all__ = [
    "MPIShardGroup",
    "SharedBasis",
    "attach_shared_group",
    "exchange_bytes",
    "row_edges",
    "shared_transport",
]
