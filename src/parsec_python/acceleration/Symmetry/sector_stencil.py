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

* the stencil values are those of ``build_negative_laplacian_buffers``;
* entries of a row with the same sector column are added in ascending order of
  their full-grid column, as ``reduce_sector_csr`` adds them, and a sum that is
  exactly zero is dropped;
* slots follow ascending sector column and the palette ascending IEEE bit
  pattern, as :func:`build_stencil_major_metadata` produces them.

Two implementations exist.  The C++/OpenMP kernel ``build_sector_stencil``
is used when the installed extension has it.  The NumPy one needs no
extension and gives the same arrays; it serves an extension built before the
kernel existed.  Both compare every entry with its transpose, the audit the
former route made on the reduced CSR.
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

from ..backends.cupy_stencil_major import StencilMajorHostMetadata
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


def _padded_lookup(grid: RealSpaceGrid, width: int) -> np.ndarray:
    """Row lookup with ``width`` inactive layers on every side.

    A stencil point of an active grid point then always has an entry, so the
    neighbor gather needs no bounds test.
    """

    lookup = np.asarray(grid.lookup)
    if lookup.ndim != 3:
        raise ValueError("lookup must be a three-dimensional array")
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


def _numpy_sector_stencil(
    grid: RealSpaceGrid,
    decomposition: ReflectionRepresentationDecomposition,
    representation: int,
    padded: np.ndarray,
) -> StencilMajorHostMetadata:
    reduction = decomposition.reduction
    width = int(grid.settings.expansion_order) // 2
    values = stencil_values(grid.settings.expansion_order, grid.spacing)
    rows = _sector_rows(decomposition, representation)
    size = int(rows.size)
    full_size = int(reduction.full_size)
    if size > np.iinfo(np.int32).max:
        raise ValueError("stencil-major CUDA storage requires int32 grid rows")
    sector_of_orbit = np.ascontiguousarray(
        decomposition.orbit_to_sector[representation], dtype=np.int64
    )
    full_to_wedge = np.asarray(reduction.full_to_wedge)
    phases = np.asarray(decomposition.phases[representation])
    multiplicities = np.asarray(reduction.multiplicities, dtype=np.int64)
    coordinates = np.asarray(grid.integer_coordinates)
    origin = np.asarray(grid.index_min, dtype=np.int64) - width

    # Stencil slots in the enumeration of the native builder: the center,
    # then every signed shell of each axis.
    strides = (padded.shape[1] * padded.shape[2], padded.shape[2], 1)
    offsets = [0]
    shells = [0]
    for axis in range(3):
        for signed_shell in range(-width, width + 1):
            if signed_shell:
                offsets.append(signed_shell * strides[axis])
                shells.append(abs(signed_shell))
    slots = len(offsets)
    offsets = np.asarray(offsets, dtype=np.int64)
    slot_values = np.asarray(
        [values[shell] for shell in shells], dtype=np.float64
    )
    flat_lookup = padded.reshape(-1)

    # Every entry is one sortable integer: sector column, then full column,
    # then what its value depends on (slot, phase sign, column multiplicity).
    distinct = np.unique(multiplicities)
    uniform = distinct.size == 1
    multiplicity_code = (
        None if uniform else np.searchsorted(distinct, multiplicities)
    )
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

    neighbors = np.full((slots, size), -1, dtype=np.int32)
    value_ids = np.full((slots, size), table_ids, dtype=np.uint16)
    used = np.zeros(table_ids + 1, dtype=np.int64)
    row_widths = np.zeros(size, dtype=np.int64)
    merged_rows: list[np.ndarray] = []
    merged_targets: list[np.ndarray] = []
    merged_values: list[np.ndarray] = []
    slot_ids = np.arange(slots, dtype=np.int64)

    for start in range(0, size, _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, size)
        block_rows = rows[start:stop]
        local = coordinates[block_rows] - origin
        base = (
            local[:, 0] * padded.shape[1] + local[:, 1]
        ) * padded.shape[2] + local[:, 2]
        columns = flat_lookup[base[:, None] + offsets[None, :]]
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
        neighbors[:, start:stop] = targets.T
        value_ids[:, start:stop] = ids.T
        row_widths[start:stop] = widths

    slot_count = int(row_widths.max(initial=0))
    if slot_count < 1:
        raise ValueError("finite-difference operator contains no entries")
    table_bits = table.reshape(-1).view(np.uint64)
    pieces = [table_bits[np.flatnonzero(used[:table_ids])]]
    if merged_rows:
        all_rows = np.concatenate(merged_rows)
        all_targets = np.concatenate(merged_targets)
        all_values = np.concatenate(merged_values)
        pieces.append(all_values[all_targets >= 0].view(np.uint64))
    palette_bits = np.unique(np.concatenate(pieces))
    if palette_bits.size > 256:
        raise ValueError("finite-difference operator has more than 256 coefficients")
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
) -> tuple[StencilMajorHostMetadata | None, ...]:
    """Build the packed stencil of every requested sector from the grid.

    The result keeps one slot per representation; sectors that were not
    requested are ``None``.  ``implementation`` is ``native``, ``numpy`` or
    ``auto``, which takes the native kernel when the extension has it.
    :class:`SectorStencilUnavailable` asks the caller for the CSR route.
    """

    if implementation not in {"auto", "native", "numpy"}:
        raise ValueError("implementation must be auto, native, or numpy")
    if decomposition.full_size != grid.size:
        raise ValueError("grid does not match the representation decomposition")
    wanted = decomposition._selected_representations(representations)
    native = None
    if implementation != "numpy" and native_kernel_available():
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
    padded = _padded_lookup(grid, width)

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


__all__ = [
    "SectorStencilUnavailable",
    "build_sector_stencils",
    "native_kernel_available",
    "sector_stencil_route",
    "stencil_values",
]
