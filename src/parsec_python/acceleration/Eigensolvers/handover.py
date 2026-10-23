"""The triangular hand-over of the Rayleigh--Ritz step (``PARSEC_CUPY_RITZ_HANDOVER``).

The Ritz step of a filtered basis ``X`` (``N x m``) factorizes the overlap
``X.T X = L L.T``, whitens the projection, ``W = L^-1 X.T H X L^-T = Z
diag(eps) Z.T``, and rotates the basis to its Ritz vectors ``X C`` with ``C =
L^-T Z``: a full ``m x m`` product, ``2 N m^2`` flops in every pass.

With ``triangular`` a later pass hands the next filter ``Q = X U`` instead,
``U = L^-T`` upper triangular: an orthonormal basis of the same span whose
leading columns span the leading columns of ``X`` (the factor is triangular),
formed by a triangular product of about half the flops.  The halves of the
degree split of the next filter (:func:`chebyshev.subspace_filter_blocks`)
are therefore kept apart: the lower half of ``Q`` spans the lower half of the
filtered columns.  The small eigenvectors ``Z`` are kept with the basis, so
that Ritz vector ``n`` is ``Q z_n`` (:class:`RitzHandover`).  The density is
formed from all columns minus the few Ritz vectors whose occupation is not
one (:func:`correction_set`), the forces likewise, and a full rotation is
taken wherever Ritz vectors themselves are read.

This module holds the switch and its knobs, the record that travels with a
basis, and the column orders and slabs of the product.  It imports nothing of
the Ritz step, which imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


_POLICIES = ("rotation", "triangular")
#: What a single Ritz step can be asked to hand over.
HANDOVER_MODES = ("rotation", "triangular", "factor", "values")
_PRODUCTS = ("columns", "tiles")


def ritz_handover_policy() -> str:
    """``PARSEC_CUPY_RITZ_HANDOVER``: ``rotation`` (the default) or ``triangular``.

    ``rotation`` hands the next filter the Ritz vectors of every pass, as
    before, bit for bit.  ``triangular`` hands later passes of a sector the
    orthonormal basis ``X L^-T`` and changes the first CHEBFF solve: its
    first cycle hands on the filtered columns as they are and the cycles up
    to the last one ``X L^-T`` (the last one rotates).  Unset or empty is
    ``rotation``; any other value is refused.
    """

    raw = os.environ.get("PARSEC_CUPY_RITZ_HANDOVER", "rotation").strip().lower()
    if not raw:
        return "rotation"
    if raw not in _POLICIES:
        raise ValueError("PARSEC_CUPY_RITZ_HANDOVER must be rotation or triangular")
    return raw


def handover_slab_columns() -> int:
    """``PARSEC_CUPY_RITZ_HANDOVER_COLUMNS``: columns of an output slab of the triangular product (128)."""

    name = "PARSEC_CUPY_RITZ_HANDOVER_COLUMNS"
    raw = os.environ.get(name, "128").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def first_cycle_condition_limit() -> float:
    """``PARSEC_CUPY_RITZ_HANDOVER_FIRST_CONDITION``: the guard of the first CHEBFF cycle (1e4).

    The first cycle hands on its filtered columns unchanged only where the
    condition number of their overlap is at most this; above it, and always
    with ``0``, it hands on ``X L^-T`` as the cycles after it do.
    """

    name = "PARSEC_CUPY_RITZ_HANDOVER_FIRST_CONDITION"
    raw = os.environ.get(name, "1e4").strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a nonnegative number") from error
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a nonnegative number")
    return value


def correction_share() -> int:
    """``PARSEC_CUPY_RITZ_HANDOVER_CORRECTION_SHARE`` (8): see :func:`correction_column_limit`."""

    name = "PARSEC_CUPY_RITZ_HANDOVER_CORRECTION_SHARE"
    raw = os.environ.get(name, "8").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def correction_column_limit(columns: int) -> int:
    """The most Ritz vectors that the density of a hand-over basis of ``columns`` corrects.

    ``max(64, columns // share)``.  A density that needs more takes the
    rotation for its sector instead, and the later passes of that sector
    rotate as well (:attr:`RitzHandover.disable`).
    """

    return max(64, int(columns) // correction_share())


def handover_product() -> str:
    """``PARSEC_CUPY_RITZ_HANDOVER_PRODUCT``: how the triangular product is cut, ``columns`` or ``tiles``.

    ``columns``, the default: one DGEMM per output slab over all rows,
    right to left, into the buffer of the Ritz step, which is then copied
    into the slab.  Where the buffer cannot hold the widest slab in all
    rows, the product takes ``tiles``: per tile of rows one DGEMM per output
    slab into the tile, which is copied back as the rotation copies its
    tiles.  On A100 the two ran at 0.531 to 0.557 and 0.554 to 0.616 of the
    rotation with the tiles of production (k11 to k14).
    """

    raw = os.environ.get("PARSEC_CUPY_RITZ_HANDOVER_PRODUCT", "columns").strip().lower()
    if raw not in _PRODUCTS:
        raise ValueError("PARSEC_CUPY_RITZ_HANDOVER_PRODUCT must be columns or tiles")
    return raw


def validate_handover_settings() -> str:
    """Read the switch and every knob once, so that a value they do not know stops a run early; returns the policy."""

    policy = ritz_handover_policy()
    handover_slab_columns()
    first_cycle_condition_limit()
    correction_share()
    handover_product()
    return policy


@dataclass(frozen=True)
class HandoverStep:
    """What one Ritz step handed over.

    ``mode`` is what ran: ``rotation`` (Ritz vectors), ``triangular``
    (``X L^-T``, with the eigenvectors ``Z`` of the whitened matrix as
    ``coefficients``, in slot order: Ritz vector ``n`` is ``sum_j basis[:, j]
    * coefficients[j, n]``), ``factor`` (``X L^-T`` without ``Z``) or
    ``values`` (the filtered columns as they are).  ``condition`` is the
    condition estimate of the overlap that the step judged, ``note`` why the
    step took another mode than it was asked for, ``band`` the row positions
    where the halves met on a shared basis.  ``rows``, where it is not None,
    says that slot ``j`` takes row ``rows[j]`` of ``coefficients`` (the
    eigenvectors of a shared basis stay in the order of its factorization,
    which spares a permuted copy of them).
    """

    mode: str
    coefficients: Any | None = None
    condition: float | None = None
    note: str | None = None
    band: tuple[int, int] | None = None
    product: str | None = None
    rows: Any | None = None

    def summary(self) -> dict:
        """Plain data for records: mode, condition, note, band and product."""

        return dict(
            mode=self.mode,
            condition=None if self.condition is None else float(self.condition),
            note=self.note,
            band=None if self.band is None else [int(item) for item in self.band],
            product=self.product,
        )


@dataclass(eq=False)
class RitzHandover:
    """The basis that a later pass handed over is ``Q = X L^-T``.

    Ritz vector ``n`` is ``sum_j basis[:, j] * coefficients[j, n]``, with
    ``coefficients`` the ``m x m`` eigenvectors of the whitened matrix in slot
    order on ``device`` (the owner of a shared basis).  One object is shared
    by the saved state and by every result that views the same memory.

    ``pending_count`` is a trim that the next pass carries out: it rotates
    and keeps that many Ritz pairs.  ``rotated`` says that the memory has
    been rotated in place since, so that its columns are the Ritz vectors
    now.  ``disable`` says that a density needed more corrections than
    :func:`correction_column_limit` allows: the later passes of the sector
    rotate.  ``rows``, where it is not None, is the row of ``coefficients``
    that slot ``j`` takes (see :class:`HandoverStep`).  The next pass filters
    the basis that this record describes in place, so it drops the
    coefficients when it starts (:func:`release_coefficients`).
    """

    coefficients: Any
    device: int
    columns: int
    pending_count: int | None = None
    rotated: bool = False
    disable: bool = False
    rows: Any | None = None


def release_coefficients(record: RitzHandover | None) -> None:
    """Drop the eigenvectors of a hand-over whose basis a pass is about to filter.

    Nothing reads them once the filter has overwritten the columns they
    describe; released here, their ``m x m`` array is not held beside those
    of the pass.  The pending trim and the marks stay.
    """

    if record is not None:
        record.coefficients = None
        record.rows = None


#: The most bytes of the Ritz vectors that one piece of a correction forms
#: on a device (:func:`correction_density`, the forces): 128 MiB.
CORRECTION_BYTES = 128 << 20


def slot_coefficients(cp, record: RitzHandover, columns) -> Any:
    """``coefficients[:, columns]`` with its rows in slot order: an ``m x k`` Fortran-ordered array of the record's device."""

    chosen = record.coefficients[:, cp.asarray(np.asarray(columns, dtype=np.int64))]
    if record.rows is not None:
        chosen = chosen[cp.asarray(np.asarray(record.rows, dtype=np.int64))]
    return cp.asfortranarray(chosen)


def uniform_slabs(columns: int, width: int) -> tuple[tuple[int, int], ...]:
    """Consecutive output slabs of ``width`` columns over ``[0, columns)``; the last may be narrower."""

    columns, width = int(columns), max(1, int(width))
    return tuple((start, min(columns, start + width)) for start in range(0, columns, width))


def handover_order(order: np.ndarray, halves: int | None):
    """Row positions in the order of the factorization, and the band ``(first, stop)`` or ``None``.

    ``order[p]`` is the slot (column of the basis) at row position ``p`` of
    a shared basis.  The order is the row order, except that between the
    first row position of the upper half (slots ``halves`` and up) and the
    last one of the lower half the lower-half columns go first, each half in
    row order.  Positions before that band are all lower half and those
    after it all upper half, so the first ``halves`` positions of the result
    are the lower-half slots.  ``halves`` of ``None`` is the row order.
    """

    order = np.asarray(order)
    positions = np.arange(order.size)
    if halves is None:
        return positions, None
    lower = order < int(halves)
    upper_at, lower_at = np.flatnonzero(~lower), np.flatnonzero(lower)
    if not upper_at.size or not lower_at.size or lower_at[-1] < upper_at[0]:
        return positions, None
    first, stop = int(upper_at[0]), int(lower_at[-1]) + 1
    band = positions[first:stop]
    return (
        np.concatenate((positions[:first], band[lower[band]], band[~lower[band]], positions[stop:])),
        (first, stop),
    )


def handover_slabs(spots, band, width: int) -> tuple[tuple[int, int], ...]:
    """Consecutive output slabs of the product of a row block.

    Unions of the rounds ``spots`` (``(begin, end)`` row positions), none cut
    inside ``band``, each at least ``width`` columns wide where the rounds
    allow (the last may be narrower).  A factor that is upper triangular in
    the order of :func:`handover_order` is then zero below every slab in row
    order: a column outside the band combines the positions before it, and
    one inside it positions before the end of the band, which ends its slab
    or lies before.
    """

    cuts = sorted({int(b) for b, _e in spots} | {int(e) for _b, e in spots})
    if band is not None:
        cuts = [cut for cut in cuts if not band[0] < cut < band[1]]
    slabs, start = [], cuts[0]
    for cut in cuts[1:]:
        if cut - start >= int(width) or cut == cuts[-1]:
            slabs.append((start, cut))
            start = cut
    return tuple(slabs)


def ritz_columns(vectors: Any, record: RitzHandover, columns) -> Any:
    """``Y = basis @ coefficients[:, columns]``: the Ritz vectors ``columns`` of a hand-over basis.

    A Fortran-ordered ``N x k`` device array on the device of the record.
    A basis on that device is multiplied in one product, formed transposed so
    that ``Y`` comes out Fortran-ordered; one spilled to the host goes there
    a tile of rows at a time; a basis that several devices share forms one
    partial product per device and sums them on its owner
    (:meth:`distributed_state.DistributedBasis.ritz_columns`).
    """

    from ..backends.cupy import require_cupy

    columns = np.asarray(columns, dtype=np.int64)
    shared = getattr(vectors, "ritz_columns", None)
    if callable(shared):
        return shared(record.coefficients, columns, rows=record.rows)
    cp, _ = require_cupy()
    with cp.cuda.Device(record.device):
        chosen = slot_coefficients(cp, record, columns)
        if isinstance(vectors, cp.ndarray):
            # (k x m)(m x N) is C-ordered k x N: its transpose is Y in F order.
            return cp.matmul(chosen.T, vectors.T).T
        rows, width = map(int, vectors.shape)
        result = cp.empty((rows, int(columns.size)), dtype=cp.float64, order="F")
        tile = max(1, min(rows, (1 << 30) // max(1, 8 * width)))
        for start in range(0, rows, tile):
            stop = min(rows, start + tile)
            block = cp.asarray(vectors[start:stop, :], dtype=cp.float64)
            result[start:stop, :] = block @ chosen
        return result


def correction_density(builder, vectors: Any, record: RitzHandover, columns, weights, volume_element: float):
    """The density of the Ritz vectors ``columns`` of a hand-over basis with ``weights``, on the host.

    ``builder`` is the device density builder, which is given the Ritz
    vectors a piece of rows at a time: at most :data:`CORRECTION_BYTES` of
    them exist at once, so the correction holds no array of all rows.  A
    basis that several devices share forms each piece from partial products
    of all devices summed on its owner
    (:meth:`distributed_state.DistributedBasis.correction_density`); a basis
    on one device multiplies its rows where they lie, one spilled to the
    host uploads them a piece at a time.
    """

    from ..backends.cupy import require_cupy

    columns = np.asarray(columns, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    shared = getattr(vectors, "correction_density", None)
    if callable(shared):
        return shared(builder, record.coefficients, columns, weights, volume_element, rows=record.rows)
    cp, _ = require_cupy()
    total_rows, width = map(int, vectors.shape)
    count = int(columns.size)
    density = np.zeros(total_rows, dtype=np.float64)
    if not count:
        return density
    with cp.cuda.Device(record.device):
        chosen = slot_coefficients(cp, record, columns)
        on_device = isinstance(vectors, cp.ndarray)
        per_row = 8 * (count if on_device else count + width)
        tile = max(1, min(total_rows, CORRECTION_BYTES // per_row))
        piece = cp.empty(tile * count, dtype=cp.float64)
        for start in range(0, total_rows, tile):
            stop = min(total_rows, start + tile)
            # A Fortran-ordered array of the rows of this piece, contiguous.
            out = piece[: (stop - start) * count].reshape((stop - start, count), order="F")
            if on_device:
                rows_times(cp, vectors, start, stop, chosen, out)
            else:
                out[...] = cp.asarray(vectors[start:stop, :], dtype=cp.float64) @ chosen
            density[start:stop] = builder(out, weights, volume_element)
    return density


def _leading(array) -> int | None:
    """The leading dimension of a column-major float64 matrix (unit row stride), or None."""

    if array.dtype != np.float64 or array.ndim != 2:
        return None
    rows, columns = map(int, array.shape)
    row_stride, column_stride = (int(value) for value in array.strides)
    if rows > 1 and row_stride != 8:
        return None
    if columns <= 1:
        return max(1, rows)
    if column_stride % 8 or column_stride // 8 < rows:
        return None
    return column_stride // 8


def rows_times(cp, matrix, start: int, stop: int, factor, out) -> None:
    """``out = matrix[start:stop, :] @ factor`` without copying the rows of a column-major ``matrix``.

    One DGEMM that reads the rows where they lie, with the leading
    dimensions of the three arrays (each must have a unit row stride, as a
    column-major matrix or a slice of rows of one has).  NumPy arrays, a
    test's stand-in for device arrays, take NumPy's product; a device array
    of another layout is refused, since a product through a copy of its rows
    would take that copy beside the basis.
    """

    rows, inner = map(int, matrix.shape)
    count = int(factor.shape[1])
    if not all(hasattr(item.data, "ptr") for item in (matrix, factor, out)):
        out[...] = matrix[start:stop, :] @ factor
        return
    leading = tuple(_leading(item) for item in (matrix, factor, out))
    if None in leading:
        raise ValueError("a product of rows needs column-major float64 arrays")
    if stop <= start or not count:
        return
    if not inner:
        out[...] = 0.0
        return
    from cupy.cuda import cublas

    one = np.asarray(1.0, dtype=np.float64)
    zero = np.asarray(0.0, dtype=np.float64)
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
    cublas.dgemm(
        handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N, stop - start, count, inner,
        one.ctypes.data, matrix.data.ptr + 8 * start, leading[0], factor.data.ptr, leading[1],
        zero.ctypes.data, out.data.ptr, leading[2],
    )


def correction_set(occupations: np.ndarray, columns: np.ndarray, width: int):
    """The Ritz vectors whose occupation is not one, and the weights ``1 - f`` of their correction.

    ``occupations`` are those of the selected states of a sector, which are
    its Ritz vectors ``columns``; the other of its ``width`` Ritz vectors are
    empty.  The set is exact: an occupation is full where it is the float
    ``1.0`` and nowhere else.  Returns the full occupations, the set and its
    weights.
    """

    full = np.zeros(int(width), dtype=np.float64)
    full[np.asarray(columns, dtype=np.int64)] = np.asarray(occupations, dtype=np.float64)
    correction = np.flatnonzero(full != 1.0)
    return full, correction, 1.0 - full[correction]


__all__ = [
    "HANDOVER_MODES",
    "HandoverStep",
    "RitzHandover",
    "CORRECTION_BYTES",
    "correction_column_limit",
    "correction_density",
    "correction_set",
    "correction_share",
    "first_cycle_condition_limit",
    "handover_order",
    "handover_product",
    "handover_slab_columns",
    "handover_slabs",
    "release_coefficients",
    "ritz_columns",
    "ritz_handover_policy",
    "rows_times",
    "slot_coefficients",
    "uniform_slabs",
    "validate_handover_settings",
]
