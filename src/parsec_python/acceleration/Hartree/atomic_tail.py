"""Static atomic tail of the Hartree boundary for the accelerated builders.

:func:`parsec_python.Hartree.build_atomic_tail` walks every stencil shell of
the whole grid and evaluates SciPy harmonics.  The functions here build the
same rows from the surface shell of the grid alone, evaluate the tail once
per unique exterior point with the recurrences of the fast multipoles, and
take the values either from host threads or from a device kernel
(:mod:`.cupy_atomic_tail`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from time import perf_counter
from typing import Callable

import numpy as np

from parsec_python.Grid import RealSpaceGrid
from parsec_python.Hartree import AtomicTail, point_charge_potential
from parsec_python.Hartree.harmonics import _positive_m_harmonic_rows
from parsec_python.Laplacian import second_derivative_coefficients

from .fast_multipole import FastMultipoleExpansion


_HOST_CHUNK_POINTS = 4096


def missing_stencil_entries(
    grid: RealSpaceGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate the missing stencil entries of a spherical grid.

    Returns the row of every entry, its operator coefficient ``-c_j/h**2``,
    the index of its exterior point among the unique ones, and the integer
    coordinates of those.  Only rows within a stencil half-width of the
    sphere can miss a neighbour, so only they are displaced.  The entries of
    one row follow the axis and shell order of the reference stencil walk.
    """

    if grid.settings.domain_shape != "sphere":
        raise ValueError("the atomic tail belongs to a spherical multipole boundary")
    width = grid.settings.stencil_half_width
    inverse_spacing_squared = 1.0 / grid.spacing**2
    coefficients = second_derivative_coefficients(grid.settings.expansion_order)
    squared = np.einsum("ij,ij->i", grid.coordinates, grid.coordinates)
    inner = max(grid.settings.radius - (width + 1) * grid.spacing, 0.0)
    shell = np.flatnonzero(squared >= inner * inner)
    base = grid.integer_coordinates[shell]
    rows, points, values = [], [], []
    for axis in range(3):
        for signed_shell in range(-width, width + 1):
            if signed_shell == 0:
                continue
            moved = base.copy()
            moved[:, axis] += signed_shell
            missing = grid.rows_for_integer_coordinates(moved) < 0
            if not missing.any():
                continue
            rows.append(shell[missing])
            points.append(moved[missing])
            values.append(
                np.full(
                    int(missing.sum()),
                    -coefficients[width + abs(signed_shell)]
                    * inverse_spacing_squared,
                )
            )
    if not rows:
        empty = np.empty(0, dtype=np.int64)
        return empty, np.empty(0), empty, np.empty((0, 3), dtype=np.int64)
    rows = np.concatenate(rows)
    points = np.concatenate(points).astype(np.int64, copy=False)
    values = np.concatenate(values)
    low = points.min(axis=0)
    span = points.max(axis=0) - low + 1
    key = (
        (points[:, 0] - low[0]) * span[1] + (points[:, 1] - low[1])
    ) * span[2] + (points[:, 2] - low[2])
    unique, inverse = np.unique(key, return_inverse=True)
    integer = np.stack(
        [
            unique // (span[1] * span[2]) + low[0],
            (unique // span[2]) % span[1] + low[1],
            unique % span[2] + low[2],
        ],
        axis=1,
    )
    return rows, values, inverse.reshape(-1), integer


def point_charge_moments(
    positions: np.ndarray,
    charges: np.ndarray,
    order: int,
) -> np.ndarray:
    """Return ``Q_lm`` of the point charges for ``0 <= m <= l <= order``.

    The array has shape ``(order+1, order+1)`` and is indexed ``[l, m]``,
    the layout of the positive-``m`` moments of the boundary builders.
    """

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(charges, dtype=np.float64)
    moments = np.zeros((order + 1, order + 1), dtype=np.complex128)
    for angular_momentum, magnetic, harmonic, radius in _positive_m_harmonic_rows(
        positions, order
    ):
        moments[angular_momentum, magnetic] = complex(
            np.sum(charges * radius**angular_momentum * np.conjugate(harmonic))
        )
    return moments


def _expansion(moments: np.ndarray, order: int) -> FastMultipoleExpansion:
    table: dict[tuple[int, int], complex] = {}
    for angular_momentum in range(order + 1):
        for magnetic in range(angular_momentum + 1):
            moment = complex(moments[angular_momentum, magnetic])
            table[(angular_momentum, magnetic)] = moment
            if magnetic:
                table[(angular_momentum, -magnetic)] = (
                    ((-1) ** magnetic) * np.conjugate(moment)
                )
    return FastMultipoleExpansion(order=order, moments=table)


def host_tail_values(
    points: np.ndarray,
    positions: np.ndarray,
    charges: np.ndarray,
    order: int,
    *,
    workers: int | None = None,
) -> np.ndarray:
    """Evaluate ``C_L`` at Cartesian points (bohr) on host threads.

    Each chunk of points is the Coulomb sum of the charges minus their
    order-``L`` series.  The chunks are fixed, so the values do not depend
    on the number of threads.
    """

    points = np.asarray(points, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(charges, dtype=np.float64)
    expansion = _expansion(point_charge_moments(positions, charges, order), order)
    result = np.empty(points.shape[0], dtype=np.float64)
    starts = range(0, points.shape[0], _HOST_CHUNK_POINTS)

    def evaluate(start: int) -> None:
        block = points[start : start + _HOST_CHUNK_POINTS]
        result[start : start + _HOST_CHUNK_POINTS] = point_charge_potential(
            block, positions, charges
        ) - expansion.potential(block)

    if workers is None:
        workers = min(16, os.cpu_count() or 1)
    if workers <= 1 or len(starts) <= 1:
        for start in starts:
            evaluate(start)
    else:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="parsec-atomic-tail"
        ) as executor:
            list(executor.map(evaluate, starts))
    return result


def build_atomic_tail_fast(
    grid: RealSpaceGrid,
    positions: np.ndarray,
    charges: np.ndarray,
    order: int,
    *,
    tail_values: Callable[..., np.ndarray] | None = None,
) -> AtomicTail:
    """Build the rows of the reference atomic tail from the surface shell.

    ``tail_values(points, positions, charges, order)`` returns ``C_L`` at
    Cartesian points; the default evaluates it on host threads.
    """

    started = perf_counter()
    rows, coefficients, point_index, integer_points = missing_stencil_entries(grid)
    evaluate = host_tail_values if tail_values is None else tail_values
    tail = np.asarray(
        evaluate(
            grid.physical_coordinates(integer_points), positions, charges, int(order)
        ),
        dtype=np.float64,
    )
    touched, entry_row = np.unique(rows, return_inverse=True)
    values = np.bincount(
        entry_row.reshape(-1),
        weights=-coefficients * tail[point_index],
        minlength=touched.size,
    )
    return AtomicTail(
        order=int(order),
        rows=touched.astype(np.int64, copy=False),
        values=values,
        maximum=float(np.abs(tail).max()) if tail.size else 0.0,
        seconds=perf_counter() - started,
    )


__all__ = [
    "build_atomic_tail_fast",
    "host_tail_values",
    "missing_stencil_entries",
    "point_charge_moments",
]
