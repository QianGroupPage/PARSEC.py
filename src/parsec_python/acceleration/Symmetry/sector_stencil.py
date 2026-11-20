"""Packed finite-difference stencil of a symmetry sector, straight from the grid.

The sector route used to reach its GPU stencil in four steps: build the
full-grid ``-nabla_FD^2`` as CSR, cut the representative rows of a sector out
of it (``U_Gamma.T A U_Gamma``), audit that CSR, and transpose it into the
slot-major ``neighbor[slot, row]`` / ``coefficient_code[slot, row]`` layout.
Only the representative rows of that matrix were ever read, and it was the
largest host allocation of a calculation.

Row ``r`` of a sector is the representative grid point of one orbit.  Its
entries are the at most ``1 + 6*M`` stencil points of that one grid point,
each mapped to the sector column of its orbit and multiplied by its character
phase and ``sqrt(|O_row| / |O_column|)``.  This module writes the packed
arrays from exactly that description and visits no other row.

The result is that of the former route with its native reduction, element
for element.  That fixes three things:

* the stencil values are those of ``build_negative_laplacian_buffers`` (a
  periodic cell whose grid steps differ between the axes: those of its
  full-grid matrix, one set per axis, :func:`axis_stencil_values`);
* entries of a row with the same sector column are added in ascending order of
  their full-grid column, as ``reduce_sector_csr`` adds them, and a sum that is
  exactly zero is dropped;
* slots follow ascending sector column and the palette ascending IEEE bit
  pattern, as :func:`build_stencil_major_metadata` produces them.

Two implementations exist.  The C++/OpenMP kernel ``build_sector_stencil``
is used when the installed extension has it.  The NumPy one needs no
extension and gives the same arrays; it serves an extension built before the
kernel existed, and every periodic cell (the kernel builds no wrapped stencil
and takes one grid spacing).  Both compare every entry with its transpose, the
audit the former route made on the reduced CSR.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from math import factorial
import os
from threading import Lock

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import RealSpaceGrid

from ..backends.cupy_stencil_major import PALETTE_CAPACITY, PaletteOverflow, StencilMajorHostMetadata
from .representations import (
    ReflectionRepresentationDecomposition,
    operator_build_workers,
)


_ROUTES = ("direct", "numpy", "csr")
# Rows per NumPy block: about 3 MB for each (rows, 25) int64 temporary, so
# that four sectors built at once hold little besides their own arrays.
_BLOCK_ROWS = 1 << 14
# The transposed entries are the largest arrays of a NumPy build.  One sector
# is audited at a time; the other sectors go on building meanwhile.
_AUDIT_LOCK = Lock()


class SectorStencilUnavailable(RuntimeError):
    """The direct builder does not cover this input; use the CSR route."""


def sector_stencil_route() -> str:
    """Return the route selected by ``PARSEC_SECTOR_STENCIL``.

    ``direct`` (the default) builds every packed sector stencil from the
    grid, with the native kernel when the extension has it and with NumPy
    otherwise.  ``numpy`` does the same with NumPy alone.  ``csr`` keeps the
    former route through the full-grid matrix.
    """

    route = os.environ.get("PARSEC_SECTOR_STENCIL", "direct").strip().lower()
    if route not in _ROUTES:
        raise ValueError("PARSEC_SECTOR_STENCIL must be direct, numpy, or csr")
    return route


def native_kernel_available() -> bool:
    """Whether the installed extension has ``build_sector_stencil``."""

    from ..backends.native import _load_native
    from ..models import BackendUnavailableError

    try:
        return hasattr(_load_native(), "build_sector_stencil")
    except BackendUnavailableError:
        return False


def stencil_values(expansion_order: int, spacing: float) -> tuple[float, ...]:
    """Entries of the native full-grid ``-nabla_FD^2`` by shell, bit for bit.

    Element ``0`` is the diagonal and element ``j`` the entry of a neighbor
    ``j`` grid points away.  The operations and their order are those of
    ``centered_second_derivative_coefficients`` and
    ``build_negative_laplacian_buffers`` in ``native/finite_difference.cpp``,
    so every float64 is the same.
    """

    order = int(expansion_order)
    if order < 2 or order > 20 or order % 2:
        raise ValueError("expansion_order must be an even integer from 2 to 20")
    width = order // 2
    width_factorial = float(factorial(width))
    shells = [0.0] * (width + 1)
    positive_sum = 0.0
    for shell in range(1, width + 1):
        sign = 1.0 if shell % 2 == 1 else -1.0
        denominator = (
            float(shell * shell)
            * float(factorial(width - shell))
            * float(factorial(width + shell))
        )
        value = 2.0 * sign * width_factorial * width_factorial / denominator
        shells[shell] = value
        positive_sum += value
    center = -2.0 * positive_sum
    inverse_spacing_squared = 1.0 / (float(spacing) * float(spacing))
    return (
        -3.0 * center * inverse_spacing_squared,
        *(-value * inverse_spacing_squared for value in shells[1:]),
    )


def _unequal_steps(grid) -> bool:
    from .axis_reflection import unequal_step_grid

    return unequal_step_grid(grid)


def axis_stencil_values(grid) -> tuple[float, tuple[tuple[float, ...], ...]]:
    """``(diagonal, shells)`` of the full-grid ``-nabla_FD^2`` of ``grid``: the diagonal entry and, per axis, the
    entry of a neighbour ``1 .. M`` points away along it.

    A grid of one spacing: :func:`stencil_values`, the same shells on every axis.  A bulk cell with its sides along
    the axes whose steps differ (:func:`parsec_python.acceleration.Symmetry.axis_reflection.unequal_step_grid`):
    the values of its full-grid matrix bit for bit, ``-sum_a p_a c_0`` and ``-p_a c_j`` with ``p_a`` the axial
    coefficients of :func:`parsec_python.Laplacian.lattice_metric.second_derivative_plan` (``1/h_a^2``; the builder
    of that matrix is ``_build_oblique_negative_laplacian`` of :mod:`parsec_python.Laplacian.finite_difference`).
    """

    if not _unequal_steps(grid):
        values = stencil_values(grid.settings.expansion_order, grid.spacing)
        return values[0], (tuple(values[1:]),) * 3
    from ..Laplacian.deferred import unequal_step_values

    try:
        return unequal_step_values(grid)
    except ValueError as error:
        raise SectorStencilUnavailable(str(error)) from None


def _sector_rows(
    decomposition: ReflectionRepresentationDecomposition,
    representation: int,
) -> np.ndarray:
    """Full-grid row of every sector row, in sector order."""

    return np.ascontiguousarray(
        decomposition.reduction.representative_rows[
            decomposition.sector_orbit_indices(representation)
        ],
        dtype=np.int64,
    )


def _check_symmetric(maximum: float, scale: float) -> None:
    if maximum > 5.0e-13 * scale:
        raise ValueError(
            "representation operator is not symmetric: "
            f"maximum asymmetry {maximum:.3e}"
        )


def _asymmetry(metadata: StencilMajorHostMetadata) -> tuple[float, float]:
    """``max |A - A.T|`` and ``max |A|`` of a packed stencil.

    The entries are transposed as coefficient codes: five bytes each, where a
    float64 CSR, its transpose and their difference take twelve and more.
    With a symmetric structure the column lists of the rows and the row lists
    of the columns are the same arrays, and only entries whose codes differ
    can differ in value; ``max |A|`` is then taken over the palette, every
    value of which the builder below has in use.  Any other structure goes
    through ``A - A.T`` itself, in which an entry without a transpose counts
    with its full magnitude.
    """

    palette = metadata.coefficient_palette
    active = metadata.neighbors >= 0
    indptr = np.zeros(metadata.shape[0] + 1, dtype=np.int64)
    np.cumsum(np.count_nonzero(active, axis=0), out=indptr[1:])
    by_row = sp.csr_matrix(
        (
            metadata.coefficient_codes.T[active.T],
            metadata.neighbors.T[active.T],
            indptr,
        ),
        shape=metadata.shape,
    )
    del active
    by_column = by_row.tocsc()
    if not (
        np.array_equal(by_column.indptr, by_row.indptr)
        and np.array_equal(by_column.indices, by_row.indices)
    ):
        del by_row, by_column
        reduced = metadata.to_csr()
        difference = reduced - reduced.T
        return (
            float(np.max(np.abs(difference.data), initial=0.0)),
            float(np.max(np.abs(reduced.data), initial=1.0)),
        )
    different = np.flatnonzero(by_row.data != by_column.data)
    return (
        float(
            np.max(
                np.abs(
                    palette[by_row.data[different]]
                    - palette[by_column.data[different]]
                ),
                initial=0.0,
            )
        ),
        float(np.max(np.abs(palette), initial=1.0)),
    )


def _native_sector_stencil(
    native,
    grid: RealSpaceGrid,
    decomposition: ReflectionRepresentationDecomposition,
    representation: int,
) -> StencilMajorHostMetadata:
    reduction = decomposition.reduction
    size = decomposition.sector_size(representation)
    payload = native.build_sector_stencil(
        grid.integer_coordinates,
        grid.index_min,
        grid.lookup,
        int(grid.settings.expansion_order),
        float(grid.spacing),
        _sector_rows(decomposition, representation),
        reduction.full_to_wedge,
        decomposition.orbit_to_sector[representation],
        reduction.multiplicities,
        decomposition.phases[representation],
    )
    _check_symmetric(
        float(payload["maximum_asymmetry"]), float(payload["scale"])
    )
    return StencilMajorHostMetadata(
        shape=(size, size),
        neighbors=payload["neighbors"],
        coefficient_codes=payload["codes"],
        coefficient_palette=payload["palette"],
    )


def _padded_lookup(grid: RealSpaceGrid, width: int, wrap: bool = False) -> np.ndarray:
    """Row lookup with ``width`` inactive layers on every side.

    A stencil point of an active grid point then always has an entry, so the
    neighbor gather needs no bounds test.  ``wrap`` (a periodic box whose
    stencil wraps on every axis): the layers repeat the rows of the opposite
    faces instead, the columns of the wrapped matrix.
    """

    lookup = np.asarray(grid.lookup)
    if lookup.ndim != 3:
        raise ValueError("lookup must be a three-dimensional array")
    if wrap:
        if np.any(lookup < 0) or min(lookup.shape) <= 2 * width:
            raise SectorStencilUnavailable("the box does not wrap on every axis")
        return np.pad(lookup.astype(np.int64, copy=False), width, mode="wrap")
    padded = np.full(
        tuple(extent + 2 * width for extent in lookup.shape), -1, dtype=np.int64
    )
    padded[width:-width, width:-width, width:-width] = lookup
    return padded


def _merge_rows(
    targets: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Add equal sector columns of sorted rows and drop exact zeros.

    Each row holds its entries in ascending (sector column, full column)
    order, the valid ones first.  A run of one sector column is added from
    left to right, which is the order of ``reduce_sector_csr``.
    """

    count, slots = targets.shape
    starts = np.ones((count, slots), dtype=bool)
    starts[:, 1:] = targets[:, 1:] != targets[:, :-1]
    sums = values.copy()
    for slot in range(1, slots):
        continued = valid[:, slot] & ~starts[:, slot]
        sums[continued, slot] = sums[continued, slot - 1] + values[continued, slot]
    ends = np.ones((count, slots), dtype=bool)
    ends[:, :-1] = starts[:, 1:] | ~valid[:, 1:]
    keep = valid & ends & (sums != 0.0)
    position = np.cumsum(keep, axis=1) - 1
    row = np.broadcast_to(np.arange(count)[:, None], keep.shape)
    merged_targets = np.full((count, slots), -1, dtype=np.int64)
    merged_values = np.zeros((count, slots), dtype=np.float64)
    merged_targets[row[keep], position[keep]] = targets[keep]
    merged_values[row[keep], position[keep]] = sums[keep]
    return merged_targets, merged_values


def _stencil_columns(grid: RealSpaceGrid, width: int, padded: np.ndarray | None):
    """``columns(rows)``: the full-grid column of every stencil slot of ``rows``, ``(rows, 1 + 6 width)`` in the
    enumeration of the native builder (the center, then every signed shell of each axis); ``-1`` where the neighbour
    of a cluster's point is not in its grid.

    ``padded`` (:func:`_padded_lookup`): gathered from that array.  ``None``: a periodic cell whose rows are its
    points in row-major order (:func:`.axis_reflection._periodic_row_major`), its stencil wrapped on every axis: the
    columns of its wrapped lookup by index arithmetic, without that array (the size of the grid in int64, 0.5 GB for
    the 62 M points of a 22^3 diamond cell); a row that is not its own row-major index is refused.
    """

    coordinates = np.asarray(grid.integer_coordinates)
    index_min = np.asarray(getattr(grid, "index_min", np.zeros(3)), dtype=np.int64)
    shells = [shell for shell in range(-width, width + 1) if shell]
    if padded is not None:
        # (a periodic cell numbers its points from 0 on every axis)
        origin = index_min - width
        strides = (padded.shape[1] * padded.shape[2], padded.shape[2], 1)
        offsets = np.asarray(
            [0] + [shell * strides[axis] for axis in range(3) for shell in shells], dtype=np.int64
        )
        flat_lookup = padded.reshape(-1)

        def gathered(rows: np.ndarray) -> np.ndarray:
            local = coordinates[rows] - origin
            base = (local[:, 0] * padded.shape[1] + local[:, 1]) * padded.shape[2] + local[:, 2]
            return flat_lookup[base[:, None] + offsets[None, :]]

        return gathered
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    if np.any(counts <= 2 * width):
        raise SectorStencilUnavailable("the box does not wrap on every axis")
    strides = (int(counts[1] * counts[2]), int(counts[2]), 1)
    # the change of the row from a point to its neighbour, by axis, shell and coordinate along the axis
    moves = [
        np.stack(
            [((np.arange(count) + shell) % count - np.arange(count)) * stride for shell in shells]
        ).astype(np.int64)
        for count, stride in zip(counts, strides)
    ]

    def wrapped(rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        local = coordinates[rows].astype(np.int64, copy=False) - index_min
        if not np.array_equal((local[:, 0] * counts[1] + local[:, 1]) * counts[2] + local[:, 2], rows):
            raise SectorStencilUnavailable("the rows of this grid are not its points in row-major order")
        columns = np.empty((rows.size, 1 + 6 * width), dtype=np.int64)
        columns[:, 0] = rows
        for axis in range(3):
            first = 1 + axis * 2 * width
            np.add(rows[:, None], moves[axis][:, local[:, axis]].T, out=columns[:, first:first + 2 * width])
        return columns

    return wrapped


def _wrapped_columns_without_lookup(grid: RealSpaceGrid, wrap: bool) -> bool:
    """Whether the counts of :func:`sector_palette_sizes` and :func:`sector_palette_bounds` take the columns of
    ``grid`` by index arithmetic (:func:`_stencil_columns` without a padded lookup)."""

    from .axis_reflection import _periodic_row_major

    return bool(wrap) and _periodic_row_major(grid)


def _multiplicity_classes(reduction) -> tuple[np.ndarray, np.ndarray | None]:
    """The distinct orbit sizes of ``reduction`` and the index of every orbit's among them (``None`` where they
    are all one size)."""

    multiplicities = np.asarray(reduction.multiplicities, dtype=np.int64)
    distinct = np.unique(multiplicities)
    return distinct, None if distinct.size == 1 else np.searchsorted(distinct, multiplicities)


def _numpy_sector_stencil(
    grid: RealSpaceGrid,
    decomposition: ReflectionRepresentationDecomposition,
    representation: int,
    padded: np.ndarray | None,
    *,
    palette_only: bool = False,
    orbit_mask: np.ndarray | None = None,
    classes: tuple[np.ndarray, np.ndarray | None] | None = None,
) -> StencilMajorHostMetadata | np.ndarray:
    """The packed stencil of one sector; ``palette_only``: the distinct coefficients of it instead, as ascending
    IEEE bit patterns (uint64; the stencil before its codes and its audit, :func:`sector_palette_sizes`).

    ``orbit_mask`` (with ``palette_only``; a flag per orbit of the reduction): only the rows of the flagged orbits,
    and every value of the table of the sector besides (each stencil value times ``+-1`` times ``sqrt(m_row /
    m_column)`` of every pair of multiplicities of the reduction): a superset of the palette where the flagged rows
    hold every row that adds entries of one sector column (:func:`sector_palette_bounds`).  ``padded`` ``None``: the
    columns by index arithmetic (:func:`_stencil_columns`).  ``classes``: :func:`_multiplicity_classes` of the
    reduction, where the caller counts several sectors.
    """

    reduction = decomposition.reduction
    width = int(grid.settings.expansion_order) // 2
    # one value per axis and shell (the same on every axis of a grid of one spacing)
    diagonal, axis_values = axis_stencil_values(grid)
    sector_of_orbit = np.ascontiguousarray(
        decomposition.orbit_to_sector[representation], dtype=np.int64
    )
    if orbit_mask is None:
        rows = _sector_rows(decomposition, representation)
        size = int(rows.size)
    else:
        if not palette_only:
            raise ValueError("orbit_mask selects rows of a palette count only")
        # the flagged orbits this sector admits, in any order (a palette is a set)
        flagged = np.flatnonzero(np.asarray(orbit_mask, dtype=bool))
        flagged = flagged[sector_of_orbit[flagged] >= 0]
        rows = np.ascontiguousarray(reduction.representative_rows[flagged], dtype=np.int64)
        size = int(np.count_nonzero(sector_of_orbit >= 0))
    row_count = int(rows.size)
    full_size = int(reduction.full_size)
    if size > np.iinfo(np.int32).max:
        raise ValueError("stencil-major CUDA storage requires int32 grid rows")
    full_to_wedge = np.asarray(reduction.full_to_wedge)
    phases = np.asarray(decomposition.phases[representation])
    columns_of = _stencil_columns(grid, width, padded)

    # Stencil slots in the enumeration of the native builder: the center,
    # then every signed shell of each axis.
    slot_values = [diagonal]
    for axis in range(3):
        for signed_shell in range(-width, width + 1):
            if signed_shell:
                slot_values.append(axis_values[axis][abs(signed_shell) - 1])
    slots = len(slot_values)
    slot_values = np.asarray(slot_values, dtype=np.float64)

    # Every entry is one sortable integer: sector column, then full column,
    # then what its value depends on (slot, phase sign, column multiplicity).
    distinct, multiplicity_code = _multiplicity_classes(reduction) if classes is None else classes
    uniform = distinct.size == 1
    slot_bits = (slots - 1).bit_length()
    code_bits = 0 if uniform else int(distinct.size - 1).bit_length()
    detail_bits = slot_bits + 1 + code_bits
    column_bits = max(1, (full_size - 1).bit_length())
    if size.bit_length() + column_bits + detail_bits > 62:
        raise SectorStencilUnavailable("sector entries do not fit one sort key")
    detail_count = 1 << detail_bits
    detail_mask = detail_count - 1
    invalid = np.iinfo(np.int64).max

    # value[row multiplicity, detail] = (stencil value * phase) * sqrt ratio,
    # the product and its order in ``reduce_sector_csr``.
    details = np.arange(detail_count)
    detail_slot = details & ((1 << slot_bits) - 1)
    detail_sign = np.where((details >> slot_bits) & 1, -1.0, 1.0)
    signed = np.where(detail_slot < slots, slot_values[detail_slot % slots], 1.0)
    signed = signed * detail_sign
    if uniform:
        table = signed[None, :].copy()
    else:
        column_multiplicity = distinct[
            np.minimum(details >> (slot_bits + 1), distinct.size - 1)
        ].astype(np.float64)
        table = signed[None, :] * np.sqrt(
            distinct.astype(np.float64)[:, None] / column_multiplicity[None, :]
        )
    if not np.all(table != 0.0):
        raise SectorStencilUnavailable("a stencil value is zero")
    table_ids = table.shape[0] * detail_count
    if table_ids > np.iinfo(np.uint16).max:
        raise SectorStencilUnavailable("too many distinct entry values")

    if not palette_only:
        neighbors = np.full((slots, size), -1, dtype=np.int32)
        value_ids = np.full((slots, size), table_ids, dtype=np.uint16)
        row_widths = np.zeros(size, dtype=np.int64)
    used = np.zeros(table_ids + 1, dtype=np.int64)
    merged_rows: list[np.ndarray] = []
    merged_targets: list[np.ndarray] = []
    merged_values: list[np.ndarray] = []
    slot_ids = np.arange(slots, dtype=np.int64)

    for start in range(0, row_count, _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, row_count)
        block_rows = rows[start:stop]
        columns = columns_of(block_rows)
        if not np.array_equal(columns[:, 0], block_rows):
            raise ValueError(
                "lookup does not map integer_coordinates back to their rows"
            )
        if int(columns.max()) >= full_size:
            raise ValueError("lookup contains an active row outside the grid")
        # A column of -1 reads the last element; ``valid`` discards it.
        column_orbits = full_to_wedge[columns]
        targets = sector_of_orbit[column_orbits]
        valid = (columns >= 0) & (targets >= 0)
        column_phases = phases[columns]
        if np.any(valid & (column_phases == 0)):
            # Such an entry is zero and the CSR route drops it.
            raise SectorStencilUnavailable("an admitted grid point has phase zero")
        detail = slot_ids[None, :] | (
            (column_phases < 0).astype(np.int64) << slot_bits
        )
        if not uniform:
            detail = detail | (
                multiplicity_code[column_orbits] << (slot_bits + 1)
            )
        keys = (((targets << column_bits) | columns) << detail_bits) | detail
        keys[~valid] = invalid
        keys.sort(axis=1)

        valid = keys != invalid
        targets = keys >> (column_bits + detail_bits)
        ids = keys & detail_mask
        if not uniform:
            row_code = multiplicity_code[full_to_wedge[block_rows]]
            ids += (row_code * detail_count)[:, None]
        widths = np.count_nonzero(valid, axis=1)
        repeated = np.flatnonzero(
            np.any(valid[:, 1:] & (targets[:, 1:] == targets[:, :-1]), axis=1)
        )
        ids[~valid] = table_ids
        targets[~valid] = -1
        if repeated.size:
            merged = _merge_rows(
                targets[repeated],
                table.reshape(-1)[np.minimum(ids[repeated], table_ids - 1)],
                valid[repeated],
            )
            merged_rows.append(start + repeated)
            merged_targets.append(merged[0])
            merged_values.append(merged[1])
            widths[repeated] = np.count_nonzero(merged[0] >= 0, axis=1)
            targets[repeated] = merged[0]
            ids[repeated] = table_ids
        used += np.bincount(ids.reshape(-1), minlength=table_ids + 1)
        if not palette_only:
            neighbors[:, start:stop] = targets.T
            value_ids[:, start:stop] = ids.T
            row_widths[start:stop] = widths

    table_bits = table.reshape(-1).view(np.uint64)
    pieces = [table_bits[np.flatnonzero(used[:table_ids])]]
    if merged_rows:
        all_rows = np.concatenate(merged_rows)
        all_targets = np.concatenate(merged_targets)
        all_values = np.concatenate(merged_values)
        pieces.append(all_values[all_targets >= 0].view(np.uint64))
    if orbit_mask is not None:
        # every value an entry of a row left out may take (not a merged sum: those rows add none)
        pieces.append(np.ascontiguousarray(table[:, detail_slot < slots]).reshape(-1).view(np.uint64))
    palette_bits = np.unique(np.concatenate(pieces))
    if palette_only:
        return palette_bits
    slot_count = int(row_widths.max(initial=0))
    if slot_count < 1:
        raise ValueError("finite-difference operator contains no entries")
    if palette_bits.size > PALETTE_CAPACITY:
        raise PaletteOverflow(
            f"finite-difference operator has more than {PALETTE_CAPACITY} coefficients "
            f"({palette_bits.size} in sector {representation})"
        )
    code_of_id = np.zeros(table_ids + 1, dtype=np.uint8)
    code_of_id[:table_ids] = np.minimum(
        np.searchsorted(palette_bits, table_bits), palette_bits.size - 1
    )
    codes = code_of_id[value_ids[:slot_count]]
    if merged_rows:
        merged_codes = np.searchsorted(
            palette_bits, all_values.view(np.uint64)
        ).astype(np.uint8)
        merged_codes[all_targets < 0] = 0
        codes[:, all_rows] = merged_codes[:, :slot_count].T
    if slot_count < slots:
        neighbors = neighbors[:slot_count].copy()
    metadata = StencilMajorHostMetadata(
        shape=(size, size),
        neighbors=neighbors,
        coefficient_codes=codes,
        coefficient_palette=palette_bits.view(np.float64),
    )
    # The former route audits every reduced matrix.
    with _AUDIT_LOCK:
        _check_symmetric(*_asymmetry(metadata))
    return metadata


def build_sector_stencils(
    grid: RealSpaceGrid,
    decomposition: ReflectionRepresentationDecomposition,
    representations: Sequence[int] | None = None,
    *,
    implementation: str = "auto",
    wrap: bool = False,
) -> tuple[StencilMajorHostMetadata | None, ...]:
    """Build the packed stencil of every requested sector from the grid.

    The result keeps one slot per representation; sectors that were not
    requested are ``None``.  ``implementation`` is ``native``, ``numpy`` or
    ``auto``, which takes the native kernel when the extension has it.
    ``wrap``: the stencil of a periodic box, whose neighbours across a face
    are the rows of the opposite one (NumPy: the native kernel leaves them
    out).  :class:`SectorStencilUnavailable` asks the caller for the CSR route.
    """

    if implementation not in {"auto", "native", "numpy"}:
        raise ValueError("implementation must be auto, native, or numpy")
    if decomposition.full_size != grid.size:
        raise ValueError("grid does not match the representation decomposition")
    wanted = decomposition._selected_representations(representations)
    if wrap and implementation == "native":
        raise ValueError("the native kernel builds no wrapped stencil")
    # The native kernel takes one grid spacing: a cell whose steps differ between the axes takes NumPy's builder
    # (axis_stencil_values; such a cell is periodic and wraps, which the kernel does not build either).
    unequal = _unequal_steps(grid)
    if unequal and implementation == "native":
        raise SectorStencilUnavailable("the native kernel takes one grid step")
    native = None
    if implementation != "numpy" and not wrap and not unequal and native_kernel_available():
        from ..backends.native import _load_native

        native = _load_native()
    if native is None and implementation == "native":
        raise RuntimeError(
            "the native extension has no build_sector_stencil; rebuild it"
        )
    built: list[StencilMajorHostMetadata | None] = [
        None
    ] * decomposition.representation_count
    if native is not None:
        # The kernel is an OpenMP loop over the rows of one sector.
        for index in wanted:
            built[index] = _native_sector_stencil(
                native, grid, decomposition, index
            )
        return tuple(built)

    width = int(grid.settings.expansion_order) // 2
    padded = _padded_lookup(grid, width, wrap)

    def assemble(index: int) -> StencilMajorHostMetadata:
        return _numpy_sector_stencil(grid, decomposition, index, padded)

    workers = operator_build_workers(len(wanted))
    if workers == 1:
        for index in wanted:
            built[index] = assemble(index)
        return tuple(built)
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="parsec-sector-stencil",
    ) as executor:
        for index, item in zip(wanted, executor.map(assemble, wanted)):
            built[index] = item
    return tuple(built)


def _per_sector(wanted, work, prefix: str, count: int) -> tuple:
    """``work(index)`` of every sector in ``wanted`` (threads as the builder's), ``None`` for the others."""

    results: list = [None] * count
    workers = operator_build_workers(len(wanted))
    if workers == 1:
        for index in wanted:
            results[index] = work(index)
        return tuple(results)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=prefix) as executor:
        for index, item in zip(wanted, executor.map(work, wanted)):
            results[index] = item
    return tuple(results)


def sector_palette_sizes(
    grid: RealSpaceGrid,
    decomposition: ReflectionRepresentationDecomposition,
    representations: Sequence[int] | None = None,
    *,
    wrap: bool = False,
) -> tuple[int | None, ...]:
    """The number of distinct coefficients of the stencil of every requested sector (NumPy's builder, as
    :func:`build_sector_stencils` packs them, without the codes and the audit); ``None`` for a sector not requested.

    The palette of a packed stencil holds :data:`PALETTE_CAPACITY` values.  The table of a sector's entries has
    every stencil value (``1 + M`` on a cube, up to ``1 + 3M`` where the steps differ, :func:`axis_stencil_values`)
    times the phase ``+-1`` times the ratio ``sqrt(m_row / m_column)`` of the multiplicities of two orbits; the rows
    whose stencil holds two points of one orbit add them into sums besides.  A cube whose orbits differ in size (an
    even repeat of a smaller cell on ``n = 2 mod 4`` points per edge) needs 260 to 300 at the default order 12.
    :class:`SectorStencilUnavailable`: an input the NumPy builder does not cover.  :func:`sector_palette_bounds`
    bounds them for a fraction of this cost.
    """

    if decomposition.full_size != grid.size:
        raise ValueError("grid does not match the representation decomposition")
    wanted = decomposition._selected_representations(representations)
    width = int(grid.settings.expansion_order) // 2
    padded = None if _wrapped_columns_without_lookup(grid, wrap) else _padded_lookup(grid, width, wrap)

    classes = _multiplicity_classes(decomposition.reduction)

    def count(index: int) -> int:
        return int(
            _numpy_sector_stencil(grid, decomposition, index, padded, palette_only=True, classes=classes).size
        )

    return _per_sector(wanted, count, "parsec-sector-palette", decomposition.representation_count)


def merging_orbits(
    grid: RealSpaceGrid,
    decomposition: ReflectionRepresentationDecomposition,
    *,
    wrap: bool = False,
    padded: np.ndarray | None = None,
) -> np.ndarray:
    """A flag per orbit of the decomposition's reduction: whether the stencil of its representative point holds two
    points of one orbit (the point itself among them).

    Only such a row adds entries of one sector column into a sum (``_merge_rows``), in any sector: the sectors differ
    in the orbits they admit and the phases of the points, not in which points share an orbit.  The rows are those
    within reach of the stencil of a symmetry element (a plane, an axis or a point that some operation keeps):
    2 % of the orbits of a 10^3 diamond cell at order 8, 0.5 % of a 22^3 one.  One pass over the representative
    points of the orbits, ``1/|H|`` of the grid; ``padded``: the lookup of :func:`_padded_lookup` where the caller has
    it (built here where the columns need it).
    """

    if decomposition.full_size != grid.size:
        raise ValueError("grid does not match the representation decomposition")
    width = int(grid.settings.expansion_order) // 2
    if padded is None and not _wrapped_columns_without_lookup(grid, wrap):
        padded = _padded_lookup(grid, width, wrap)
    columns_of = _stencil_columns(grid, width, padded)
    reduction = decomposition.reduction
    rows = np.asarray(reduction.representative_rows, dtype=np.int64)
    full_to_wedge = np.asarray(reduction.full_to_wedge)
    # a point outside a cluster's grid: an orbit of its own for every slot
    outside = -1 - np.arange(1 + 6 * width, dtype=np.int64)[None, :]
    merging = np.zeros(rows.size, dtype=bool)

    def flag(start: int) -> None:
        stop = min(start + _BLOCK_ROWS, rows.size)
        columns = columns_of(rows[start:stop])
        orbits = np.where(columns >= 0, full_to_wedge[columns], outside)
        orbits.sort(axis=1)
        merging[start:stop] = np.any(orbits[:, 1:] == orbits[:, :-1], axis=1)

    starts = range(0, rows.size, _BLOCK_ROWS)
    workers = min(len(starts), operator_build_workers(max(1, len(starts))))
    if workers <= 1:
        for start in starts:
            flag(start)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="parsec-sector-merges") as executor:
            list(executor.map(flag, starts))
    return merging


def sector_palette_bounds(
    grid: RealSpaceGrid,
    decomposition: ReflectionRepresentationDecomposition,
    representations: Sequence[int] | None = None,
    *,
    wrap: bool = False,
) -> tuple[int | None, ...]:
    """An upper bound of :func:`sector_palette_sizes` for every requested sector, ``None`` for the others.

    The palette of a sector is the values its entries take from the table of the sector (each stencil value times
    ``+-1`` times ``sqrt(m_row / m_column)``) and the sums of the rows that add entries of one sector column.  The
    bound is the whole table, every product of the distinct stencil values, the two signs and every pair of the
    distinct multiplicities of the reduction, joined with the exact palette of the rows of :func:`merging_orbits`,
    which hold every such sum: no row is left out that adds one.  It is the count where every product occurs, and a
    few values more where some do not.  Its cost is that pass over the representative points and the count of the
    rows it flags: a 22^3 diamond cell at the benchmark's order 8 (62 M points, 10 sectors of 1.9 M rows each) needs
    no count of its 19 M sector rows.
    """

    if decomposition.full_size != grid.size:
        raise ValueError("grid does not match the representation decomposition")
    wanted = decomposition._selected_representations(representations)
    width = int(grid.settings.expansion_order) // 2
    padded = None if _wrapped_columns_without_lookup(grid, wrap) else _padded_lookup(grid, width, wrap)
    merging = merging_orbits(grid, decomposition, wrap=wrap, padded=padded)

    classes = _multiplicity_classes(decomposition.reduction)

    def bound(index: int) -> int:
        return int(
            _numpy_sector_stencil(
                grid, decomposition, index, padded, palette_only=True, orbit_mask=merging, classes=classes
            ).size
        )

    return _per_sector(wanted, bound, "parsec-sector-palette-bound", decomposition.representation_count)


__all__ = [
    "PALETTE_CAPACITY",
    "PaletteOverflow",
    "SectorStencilUnavailable",
    "axis_stencil_values",
    "build_sector_stencils",
    "merging_orbits",
    "native_kernel_available",
    "sector_palette_bounds",
    "sector_palette_sizes",
    "sector_stencil_route",
    "stencil_values",
]
