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

The Rayleigh--Ritz step holds three shares of the basis on every rank at its
peak (the columns and the two row blocks of ``X`` and ``H X``); with
``PARSEC_MPI_SHARED_RITZ=rounds`` (:func:`shared_ritz_route`,
:meth:`MPIShardGroup._ritz_in_rounds`) it holds two: ``H X`` is formed, moved
and projected a round of columns at a time, and the rotated rows go back in
the same rounds.

The small solve runs on the root's device where
:func:`rayleigh_ritz.dense_solve_on_device` says so, else on its host; the
condition number of its overlap is estimated on the next rank meanwhile
(:meth:`MPIShardGroup._condition_helper`).
A failed stability audit raises :class:`rayleigh_ritz.GeneralizedRitzStabilityError`
on every rank; :meth:`MPIShardGroup.orthonormal_ritz` then orthonormalizes
the rows by shifted Cholesky QR (three passes) instead of gathering the
basis on one device, and the later passes of the sector keep that route.
Where the small solve is on the device (the default,
:func:`rayleigh_ritz.dense_solve_on_device`) the root factors the summed
overlaps on its device, with the Frobenius norm in the shift, and every rank
solves its rows in place; with ``PARSEC_CUPY_RITZ_DENSE_BACKEND=host`` the
root factors on the host with the 2-norm of an SVD, as before. Measured on
16 A100 (2026-11-08, bulk diamond of 1,728 atoms, 3,468 states, every pass
after the second on this route): 10.0 s per SCF step before, 2.4 s on the
device, the same energies within 3e-8 Ry after 12 steps.
"""

from __future__ import annotations

import functools
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
    condition_helper_policy,
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
    _condition_policy,
    _lower_triangle_product,
    dense_solve_on_device,
    solve_whitened_ritz,
    solve_whitened_ritz_on_device,
)
from .tall_product import transposed_product


# The tag of the messages between the root and the rank that estimates the condition number of its small solve.
_CONDITION_TAG = 7301

# What a rank does with an error that leaves the other ranks inside a collective (set_failure_handler).
_failure_handler = None


def set_failure_handler(handler=None) -> None:
    """Install (or, with ``None``, remove) what a rank does with an error inside a step of a shared basis.

    ``handler(group, error)`` is called on the rank where :meth:`MPIShardGroup.random_basis`,
    :meth:`~MPIShardGroup.filter`, :meth:`~MPIShardGroup.ritz` or :meth:`~MPIShardGroup.orthonormal_ritz`
    raised anything but :class:`rayleigh_ritz.GeneralizedRitzStabilityError` (which every rank raises
    together, after the root's verdict).  Such an error may stop a rank between two collectives while the
    others wait in the next one: the commands of the MPI context gather the errors of the ranks, which the
    ranks still inside an ``Alltoallv`` never reach.  The full-SCF runner installs a handler that reports
    the error and aborts the job (:mod:`benchmarks.mpi_full_scf`); without one the error is raised as it
    is.  Seen on 16 A100 ranks: the first eigensolve of 5,832 atoms waited 74 minutes on an error of ten of
    the ranks.
    """
    global _failure_handler
    _failure_handler = handler


def _ends_the_job_on_failure(method):
    """Hand an error of ``method`` that only some ranks may meet to :func:`set_failure_handler`'s handler."""

    @functools.wraps(method)
    def run(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except GeneralizedRitzStabilityError:
            raise
        except BaseException as error:
            if _failure_handler is not None and self.size > 1:
                _failure_handler(self, error)
            raise

    return run


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


def shared_ritz_route(default: str = "rows") -> str:
    """``PARSEC_MPI_SHARED_RITZ``: ``rows``, ``rounds`` or ``auto`` (the default), the Ritz step of a shared basis.

    ``rows`` moves the filtered columns and ``H`` times them to two row blocks
    of ``m`` columns each and keeps both, beside the columns, until the
    rotation: three shares of the basis at the peak of every rank.
    ``rounds`` keeps one row block, that of the basis, and forms ``H X`` a
    round of columns at a time, each projected as it arrives; the rotated
    rows go back in the same rounds (:meth:`MPIShardGroup._ritz_in_rounds`):
    two shares.  ``auto`` is ``default``, the route of the group
    (:class:`MPIShardGroup`): ``rounds`` for a periodic cell, whose runner asks
    for it, ``rows`` otherwise (clusters keep their bits).

    Measured on 16 A100 (2026-11-08, bulk diamond of 4,096 atoms, 8,204
    states, 12.3 GB of orbitals per rank): ``rounds`` held 27.1 GiB of arrays
    per rank (root 30.6) where ``rows`` held 36.6 (root 42.0), and its pass
    took 5.5 s in the exchanges, Gram products and rotation where ``rows``
    took 6.7 (fewer Gram products: the lower triangles of blocks of both
    matrices, no full overlap).  The energies of its first six SCF steps
    (-46,710 Ry, far from converged) were those of ``rows`` within 8e-7 Ry
    (6e-8 at the first); converged SCFs of 216 and 512 atoms on 4 ranks ended
    within 2e-12 and 2e-11 Ry of ``rows``, in the same number of steps.
    """

    value = os.environ.get("PARSEC_MPI_SHARED_RITZ", "auto").strip().lower() or "auto"
    if value not in {"auto", "rows", "rounds"}:
        raise ValueError("PARSEC_MPI_SHARED_RITZ must be auto, rows or rounds")
    if default not in {"rows", "rounds"}:
        raise ValueError("the default Ritz route of a shared basis must be rows or rounds")
    return default if value == "auto" else value


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


def _host_shifted_cholesky(matrix: np.ndarray, rows: int, *, shifted: bool) -> np.ndarray:
    """Lower Cholesky factor of the symmetrized overlap ``X.T X`` of ``rows`` rows, on the host.

    With ``shifted`` (the first of the three passes) the factor of
    ``A + s I`` with Fukaya et al.'s shift ``s = 11 (N m + m (m + 1)) u ||A||_2``,
    which makes the factorization succeed for any basis of condition number
    below about 1e15.  ``||A||_2`` comes from an SVD of the ``m x m`` matrix,
    ``O(m^3)`` on the host.
    """

    states = int(matrix.shape[0])
    matrix = 0.5 * (matrix + matrix.T)
    if shifted:
        shift = 11.0 * (rows * states + states * (states + 1)) * np.finfo(float).eps
        matrix = matrix + shift * np.linalg.norm(matrix, 2) * np.eye(states)
    return np.linalg.cholesky(matrix)


def _device_shifted_cholesky(overlap, rows: int, *, shifted: bool):
    """:func:`_host_shifted_cholesky` on the device that holds ``overlap`` (cuSOLVER).

    The shift takes the Frobenius norm of ``A = X.T X`` in place of its
    2-norm: an upper bound of it (at most ``sqrt(m)`` times larger) that costs
    ``O(m^2)``, the replacement Fukaya et al. name for the 2-norm, where the
    host takes an SVD.  The larger shift leaves the factor of the first pass
    less exact, which the two unshifted passes repair: the basis that the
    three passes return is orthonormal either way.  Raises
    ``numpy.linalg.LinAlgError`` where cuSOLVER reports a failed
    factorization or the factor is not finite.
    """

    import cupyx

    cp, _ = require_cupy()
    states = int(overlap.shape[0])
    matrix = 0.5 * (overlap + overlap.T)
    if shifted:
        shift = 11.0 * (rows * states + states * (states + 1)) * np.finfo(float).eps
        norm = float(cp.linalg.norm(matrix).item())
        matrix[cp.diag_indices(states)] += shift * norm
    with cupyx.errstate(linalg="raise"):
        factor = cp.linalg.cholesky(matrix)
    if not bool(cp.isfinite(factor).all().item()):
        raise np.linalg.LinAlgError("the Cholesky factor of the overlap is not finite")
    return factor


def _solve_lower_transpose_in_place(factor, block) -> None:
    """``block <- block L^-T`` in place for a lower triangular ``L`` (``factor``): one cuBLAS ``dtrsm``.

    ``block`` is a Fortran-ordered ``r x m`` float64 device array (the rows of
    this rank), ``factor`` an ``m x m`` one; the solve from the right needs
    neither a transposed copy of the rows nor a result array.
    """

    from cupy_backends.cuda.libs import cublas

    cp, _ = require_cupy()
    rows, states = (int(value) for value in block.shape)
    if not rows or not states:
        return
    if not block.flags.f_contiguous or block.dtype != cp.float64:
        raise ValueError("the rows must be a Fortran-ordered float64 block")
    factor = cp.asfortranarray(factor, dtype=cp.float64)
    one = np.array(1.0, dtype=np.float64)
    cublas.dtrsm(
        cp.cuda.device.get_cublas_handle(),
        cublas.CUBLAS_SIDE_RIGHT,
        cublas.CUBLAS_FILL_MODE_LOWER,
        cublas.CUBLAS_OP_T,
        cublas.CUBLAS_DIAG_NON_UNIT,
        rows,
        states,
        one.ctypes.data,
        factor.data.ptr,
        states,
        block.data.ptr,
        rows,
    )


def _device_condition_number(overlap):
    """The condition number of the summed overlap from its symmetric spectrum on the root's device, also near the guard.

    The ``condition`` of :func:`rayleigh_ritz.solve_whitened_ritz_on_device`, computed here before that returns:
    ``lambda_max / lambda_min`` from cuSOLVER's ``eigvalsh``, infinite for a spectrum that is not positive or not
    finite.  The solve of one device downloads the overlap for an SVD on the host where that number is above a
    hundredth of the limit (``_device_overlap_condition``), ``O(m^3)`` host work that a filtered bulk basis
    reaches in its later passes (the SCF step of 3,468 states where it happened took 7 s longer than the others,
    on 16 A100; the SVD grows as ``m^3``).  The number only decides whether the generalized solve is refused; the
    coefficients do not depend on it.
    """

    import cupyx

    cp, _ = require_cupy()
    try:
        with cupyx.errstate(linalg="raise"):
            values = cp.asnumpy(cp.linalg.eigvalsh(overlap))
    except np.linalg.LinAlgError as error:
        failure = error

        def failed():
            raise failure

        return failed
    number = (
        float(values[-1] / values[0])
        if values.size and np.all(np.isfinite(values)) and values[0] > 0
        else float("inf")
    )
    return lambda: number


def _stability_error(message: str, filtered) -> GeneralizedRitzStabilityError:
    """A :class:`rayleigh_ritz.GeneralizedRitzStabilityError` that carries the intact ``filtered`` basis."""

    error = GeneralizedRitzStabilityError(message)
    error.filtered = filtered
    return error


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

    def round_order(self) -> tuple[np.ndarray, list[int]]:
        """The basis column of every column of a row block in round order, and where each round starts in it.

        The receive buffer of round ``j`` of :meth:`pack_columns` is an
        ``own x n_j`` Fortran-ordered block of rows: the local columns
        ``j * width`` onwards of rank 0, then those of rank 1, and so on.
        Laid side by side, round after round, these blocks are the row block
        of the basis with its columns permuted: column ``p`` of it is column
        ``order[p]`` of the basis, and round ``j`` fills ``starts[j]:starts[j + 1]``.
        """

        columns, starts = [], [0]
        for step in range(self.rounds):
            first = step * self.width
            counts = self.counts(step)
            for r, count in enumerate(counts):
                for column, _offset, span in self.layout.spans(r, first, first + count):
                    columns.append(np.arange(column, column + span, dtype=np.int64))
            starts.append(starts[-1] + sum(counts))
        order = np.concatenate(columns) if columns else np.zeros(0, dtype=np.int64)
        return order, starts

    # Columns to rows.

    def pack_columns(self, cp, local, step: int, *, start: int | None = None, receive=None):
        """The send buffer and counts of round ``step`` from this rank's columns.

        ``local`` holds the local columns of the round from its column
        ``start`` on (``step * width`` by default, the whole local block).
        ``receive``, where given, is the buffer to receive into, of at least
        the received count; a new one otherwise.
        """

        counts = self.counts(step)
        mine = counts[self.group.rank]
        first = step * self.width if start is None else int(start)
        send_counts = [height * mine for height in self.heights()]
        send = cp.empty(max(1, sum(send_counts)), dtype=cp.float64)
        offset = 0
        for q, height in enumerate(self.heights()):
            if height and mine:
                piece = send[offset : offset + height * mine].reshape((mine, height)).T
                piece[...] = local[self.edges[q] : self.edges[q + 1], first : first + mine]
            offset += height * mine
        receive_counts = [self.own * count for count in counts]
        if receive is None:
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

    def __init__(self, comm, operator, *, transport: str | None = None, ritz_route: str = "rows") -> None:
        self.comm = comm
        # The Ritz step where PARSEC_MPI_SHARED_RITZ is auto (shared_ritz_route).
        shared_ritz_route(ritz_route)
        self.ritz_route = ritz_route
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

    @_ends_the_job_on_failure
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

    @_ends_the_job_on_failure
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

    @_ends_the_job_on_failure
    def ritz(self, operator: Any, filtered: SharedBasis, *, stages: bool):
        """Generalized Rayleigh--Ritz of the filtered basis over all ranks; consumes ``filtered``.

        Returns the Ritz values (on every rank), the rotated basis, written
        back into the local block, and the whitened projected Hamiltonian
        (on the root; an empty array on the other ranks).  On a failed
        stability audit every rank raises
        :class:`rayleigh_ritz.GeneralizedRitzStabilityError` with the intact
        filtered basis as ``error.filtered``.

        :func:`shared_ritz_route` chooses between the two row blocks of
        ``rows`` and the rounds of :meth:`_ritz_in_rounds`.
        """

        if shared_ritz_route(self.ritz_route) == "rounds":
            return self._ritz_in_rounds(operator, filtered, stages=stages)
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
                # The rows of the refused step go first, and no name of this frame holds the error: one that
                # did would close a cycle (error, traceback, this frame) that kept both row blocks until the
                # collector ran, through the fallback that follows (bulk diamond, 2.2 GiB of columns per rank on
                # 16 ranks: 13.3 GiB live at the peak of the step that fell back).
                del row_applied, row_basis
                raise _stability_error(coefficients, SharedBasis(self, columns, layout))
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

    def _ritz_in_rounds(self, operator: Any, filtered: SharedBasis, *, stages: bool):
        """:meth:`ritz` with one row block: ``H X`` formed, moved and projected a round of columns at a time.

        The rounds are those of the exchanges (:class:`_RowExchange`, at most
        :func:`exchange_bytes` per buffer).  Their receive buffers, side by
        side, are the row block ``R`` of the basis with its columns in round
        order (:meth:`_RowExchange.round_order`, the permutation ``P``), so
        that the basis moves to rows without a copy, and the small problem is
        that of ``X P``:

        * ``S = (X P).T (X P)`` and ``A = (X P).T H (X P)`` are formed as
          their lower triangles of blocks: round ``j`` of ``n_j`` columns
          from ``s_j`` on gives ``S[s_j:, s_j:s_j + n_j] = R[:, s_j:].T R_j``
          and, as its ``H X`` arrives, ``A[s_j:, s_j:s_j + n_j] = R[:, s_j:].T (H X)_j``
          (about half of the products of a full ``S`` and of ``A`` in eight slabs);
        * the root solves ``A C = S C Lambda`` (the Ritz values of ``X``, the
          coefficients ``C`` of ``X P``) and broadcasts ``C`` with its columns
          in round order;
        * round ``j`` of the move back sends the rows ``R C[:, P_j]`` of the
          rotated columns that go out in it, formed into its send buffer, and
          writes the received rows into the local columns of the filtered
          basis, whose rows ``R`` holds by then.

        Beside this rank's columns and ``R``, two shares of the basis, a
        round holds one block of ``H X`` columns, its send and receive buffers
        and the ``m x m`` matrices.  The Ritz values are those of :meth:`ritz`
        to round-off; within a degenerate eigenspace the Ritz vectors may
        differ by a rotation.  On a failed stability audit the columns are
        intact and every rank raises as :meth:`ritz` does.
        """

        cp, _ = require_cupy()
        layout = filtered.layout
        rows = int(filtered.shape[0])
        states = int(layout.columns)
        columns = filtered.local
        filtered.take()
        plan = _RowExchange(self, layout, rows, self._round_width(rows))
        order, starts = plan.round_order()

        def stage(name):
            return device_stage(operator, name) if stages else _NoStage()

        def round_columns(block, step):
            # Columns starts[step]:starts[step + 1] of a Fortran-ordered row block as the flat buffer of a round.
            piece = block.T[starts[step] : starts[step + 1]]
            return piece.reshape(-1) if piece.size else cp.empty(1, dtype=cp.float64)

        with cp.cuda.Device(self.device):
            row_basis = cp.empty((plan.own, states), dtype=cp.float64, order="F")
            packed = cp.zeros((2, states, states), dtype=cp.float64)
            with stage("subspace_ritz_projection_seconds"):
                started = perf_counter()
                for step in range(plan.rounds):
                    send, send_counts, receive, receive_counts = plan.pack_columns(
                        cp, columns, step, receive=round_columns(row_basis, step)
                    )
                    self._exchange(send, send_counts, receive, receive_counts)
                    del send, receive
                self._sync()
                self.seconds["to_rows"] += perf_counter() - started
                started = perf_counter()
                for step in range(plan.rounds):
                    first, last = starts[step], starts[step + 1]
                    if last > first:
                        packed[0, first:, first:last] = transposed_product(
                            cp, row_basis[:, first:], row_basis[:, first:last]
                        )
                self._sync()
                self.seconds["gram"] += perf_counter() - started
                for step in range(plan.rounds):
                    started = perf_counter()
                    mine = plan.counts(step)[self.rank]
                    local_first = step * plan.width
                    applied = cp.empty((rows, mine), dtype=cp.float64, order="F")
                    if mine:
                        operator.apply_into(columns[:, local_first : local_first + mine], applied)
                    self._sync()
                    self.seconds["apply"] += perf_counter() - started
                    started = perf_counter()
                    send, send_counts, receive, receive_counts = plan.pack_columns(cp, applied, step, start=0)
                    del applied
                    self._exchange(send, send_counts, receive, receive_counts)
                    del send
                    self._sync()
                    self.seconds["to_rows"] += perf_counter() - started
                    started = perf_counter()
                    first, last = starts[step], starts[step + 1]
                    if last > first:
                        received = receive[: plan.own * (last - first)].reshape((last - first, plan.own)).T
                        packed[1, first:, first:last] = transposed_product(cp, row_basis[:, first:], received)
                        del received
                    del receive
                    self._sync()
                    self.seconds["gram"] += perf_counter() - started
                started = perf_counter()
                packed = self._sum(packed)
                self.seconds["reduce"] += perf_counter() - started
            eigenvalues, coefficients, whitened = self._small_solve(packed, column_order=order)
            del packed
            if eigenvalues is None:
                # As in ritz(): the rows go first, and no name of this frame holds the error.
                del row_basis
                raise _stability_error(coefficients, SharedBasis(self, columns, layout))
            with stage("subspace_ritz_rotation_seconds"):
                for step in range(plan.rounds):
                    started = perf_counter()
                    first, last = starts[step], starts[step + 1]
                    counts = plan.counts(step)
                    send_counts = [plan.own * count for count in counts]
                    send = cp.empty(max(1, sum(send_counts)), dtype=cp.float64)
                    if plan.own and last > first:
                        # (R C_j).T = C_j.T R.T: the rows of the rotated columns of the round, rank after rank.
                        cp.matmul(
                            coefficients[:, first:last].T,
                            row_basis.T,
                            out=send[: plan.own * (last - first)].reshape((last - first, plan.own)),
                        )
                    self._sync()
                    self.seconds["rotate"] += perf_counter() - started
                    started = perf_counter()
                    receive_counts = [height * counts[self.rank] for height in plan.heights()]
                    receive = cp.empty(max(1, sum(receive_counts)), dtype=cp.float64)
                    self._exchange(send, send_counts, receive, receive_counts)
                    del send
                    plan.unpack_columns(receive, step, columns)
                    del receive
                    self._sync()
                    self.seconds["to_columns"] += perf_counter() - started
                del row_basis, coefficients
        self.passes += 1
        return eigenvalues, SharedBasis(self, columns, layout), whitened

    @_ends_the_job_on_failure
    def orthonormal_ritz(self, operator: Any, filtered: SharedBasis, generator) -> tuple:
        """Robust fallback: shifted Cholesky QR of the rows (three passes), then :meth:`ritz`."""

        cp, _ = require_cupy()
        from cupyx.scipy.linalg import solve_triangular

        layout = filtered.layout
        rows = int(filtered.shape[0])
        columns = filtered.local
        filtered.take()
        on_device = dense_solve_on_device()
        with cp.cuda.Device(self.device):
            row_basis = self.to_rows(columns, layout, rows)
            states = int(layout.columns)
            for step in range(3):
                overlap = self._sum(cp.ascontiguousarray(row_basis.T @ row_basis)[None])[0]
                status, factor = None, None
                if self.rank == self.owner:
                    try:
                        factor = (
                            _device_shifted_cholesky(overlap, rows, shifted=step == 0)
                            if on_device
                            else _host_shifted_cholesky(cp.asnumpy(overlap), rows, shifted=step == 0)
                        )
                    except np.linalg.LinAlgError as exc:
                        status = f"shifted Cholesky QR failed: {exc}"
                status = self.comm.bcast(status, root=self.owner)
                if status is not None:
                    raise RuntimeError(
                        "the filtered basis failed the generalized-Ritz stability audit and could not "
                        "be orthonormalized across the ranks: " + status
                    )
                factor = self._broadcast_matrix(factor, states)
                # X <- X L^-T, row by row: in place on the device route (no copy of the rows),
                # through the transposed rows as before on the host route.
                if on_device:
                    _solve_lower_transpose_in_place(factor, row_basis)
                else:
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

    def _condition_helper(self) -> int | None:
        """The rank that estimates the condition number of the root's small solve meanwhile, or ``None``.

        The device solve starts with the condition number of the summed
        overlap, a symmetric spectrum on the root's device
        (:func:`_device_condition_number`) that is only compared with its
        limit and takes about as long as the eigensolve after it (cuSOLVER on
        an A100: 0.95 and 1.03 s at 8,204 states, 2.31 and 2.58 s at 11,670),
        while the other ranks wait.  The next rank estimates it meanwhile, on
        a copy of the overlap that the root sends it
        (:meth:`_condition_beside`): the same function on the same matrix,
        the same number, the audits in their order, the coefficients the
        same bit for bit.  ``PARSEC_CUPY_RITZ_CONDITION_HELPER=off``
        (:func:`distributed_state.condition_helper_policy`) keeps the
        estimate on the root, as before; a group of one rank has no helper.
        The helper holds the copy and the work of the estimate, about three
        ``m x m`` arrays, during the small solve.
        """

        if self.size < 2 or condition_helper_policy() == "off":
            return None
        return (self.owner + 1) % self.size

    def _condition_beside(self, helper: int, asked: list):
        """The ``condition`` of :func:`rayleigh_ritz.solve_whitened_ritz_on_device` for an estimate on ``helper``.

        Given the symmetric overlap, complete on the root's device, it sends
        its order and the matrix to the helper and returns what waits for the
        number; a failed estimate comes back as its message and is raised
        here as the ``numpy.linalg.LinAlgError`` that the root's own estimate
        would have raised.  ``asked`` records that the helper was reached.
        """

        def start(overlap):
            asked.append(True)
            self.comm.send(int(overlap.shape[0]), dest=helper, tag=_CONDITION_TAG)
            self._send_matrix(overlap, helper)

            def wait():
                value = self.comm.recv(source=helper, tag=_CONDITION_TAG)
                if isinstance(value, str):
                    raise np.linalg.LinAlgError(value)
                return float(value)

            return wait

        return start

    def _estimate_condition_for_owner(self) -> None:
        """The helper's part of :meth:`_condition_beside`: receive the overlap, estimate, send the number back."""

        states = int(self.comm.recv(source=self.owner, tag=_CONDITION_TAG))
        if not states:
            return
        overlap = self._receive_matrix(states, self.owner)
        try:
            value = _device_condition_number(overlap)()
        except np.linalg.LinAlgError as error:
            value = str(error) or "the condition estimate of the overlap failed"
        del overlap
        self.comm.send(value, dest=self.owner, tag=_CONDITION_TAG)

    def _send_matrix(self, matrix, destination: int) -> None:
        """Send a contiguous device matrix to one rank (a device buffer, or staged through the host)."""

        cp, _ = require_cupy()
        matrix = cp.ascontiguousarray(matrix)
        self._sync()
        if self.transport == "cuda":
            self.comm.Send(matrix, dest=destination, tag=_CONDITION_TAG)
        else:
            self.comm.Send(np.ascontiguousarray(cp.asnumpy(matrix)), dest=destination, tag=_CONDITION_TAG)

    def _receive_matrix(self, size: int, source: int):
        """The ``size x size`` matrix that :meth:`_send_matrix` of ``source`` sends, on this rank's device."""

        cp, _ = require_cupy()
        if self.transport == "cuda":
            buffer = cp.empty((size, size), dtype=cp.float64)
            self._sync()
            self.comm.Recv(buffer, source=source, tag=_CONDITION_TAG)
            return buffer
        host = np.empty((size, size), dtype=np.float64)
        self.comm.Recv(host, source=source, tag=_CONDITION_TAG)
        return cp.asarray(host)

    def _small_solve(self, packed, column_order=None):
        """The root solves; every rank gets the Ritz values and the coefficients.

        Returns ``(None, message, None)`` on every rank when the root's
        solve failed its stability audit.  With ``column_order`` the root
        broadcasts the columns of the coefficients in that order.
        """

        cp, _ = require_cupy()
        started = perf_counter()
        states = int(packed.shape[-1])
        status = None
        eigenvalues = coefficients = whitened = None
        on_device = dense_solve_on_device()
        symmetric = _condition_policy(on_device=True) != "svd"
        helper = self._condition_helper() if on_device and symmetric else None
        if self.rank == self.owner:
            asked = []
            try:
                if on_device:
                    eigenvalues, coefficients, whitened = solve_whitened_ritz_on_device(
                        packed[0],
                        packed[1],
                        condition=(
                            None
                            if not symmetric
                            else _device_condition_number
                            if helper is None
                            else self._condition_beside(helper, asked)
                        ),
                    )
                    eigenvalues = cp.asnumpy(eigenvalues)
                else:
                    eigenvalues, coefficients, whitened = solve_whitened_ritz(
                        cp.asnumpy(packed[0]), cp.asnumpy(packed[1])
                    )
            except GeneralizedRitzStabilityError as error:
                status = str(error) or "generalized Ritz stability audit failed"
            finally:
                if helper is not None and not asked:
                    # The helper waits for an overlap that this solve never reached.
                    self.comm.send(0, dest=helper, tag=_CONDITION_TAG)
            if status is None and column_order is not None:
                order = np.asarray(column_order, dtype=np.int64)
                coefficients = coefficients[:, cp.asarray(order) if isinstance(coefficients, cp.ndarray) else order]
        elif self.rank == helper:
            self._estimate_condition_for_owner()
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


def attach_shared_group(operator, comm, *, ritz_route: str = "rows") -> MPIShardGroup:
    """Give ``operator`` (this rank's replica of the shared sector) its group of ranks.

    ``ritz_route`` is the Ritz step of the group where ``PARSEC_MPI_SHARED_RITZ``
    names none (:func:`shared_ritz_route`).
    """

    group = MPIShardGroup(comm, operator, ritz_route=ritz_route)
    operator._mpi_shared_group = group
    return group


__all__ = [
    "MPIShardGroup",
    "SharedBasis",
    "attach_shared_group",
    "exchange_bytes",
    "row_edges",
    "set_failure_handler",
    "shared_ritz_route",
    "shared_transport",
]
