"""First-shell neighbour rows of a real-space grid.

``setup.F90`` (lines 194-221) stores for every grid point the rows of its
neighbours along the three axes, shell by shell, in the order ``-x, +x, -y,
+y, -z, +z`` (``parallel%neibs``).  A neighbour outside a cluster domain is
the extra row ``nwedge + 1``, which always holds zero.  The six first-shell
entries are what the DIAGLA preconditioner ``fdo5scl.f90z`` averages over.

This module returns that first shell for the full (not symmetry-reduced)
grid of the Python port: an ``(n_grid, 6)`` integer array of row numbers,
``-1`` for a neighbour outside the cluster domain.  A periodic grid wraps
around, so every neighbour exists.
"""

from __future__ import annotations

import numpy as np

# setup.F90: indx(1..6) = (kx-1), (kx+1), (ky-1), (ky+1), (kz-1), (kz+1).
FIRST_SHELL_OFFSETS = np.array(
    [
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ],
    dtype=np.int64,
)


def nearest_neighbor_rows(grid) -> np.ndarray:
    """Rows of the six first-shell neighbours of every grid point.

    ``grid`` is a :class:`~parsec_python.Grid.RealSpaceGrid` or a
    :class:`~parsec_python.Grid.PeriodicRealSpaceGrid` (anything with
    ``integer_coordinates`` and ``rows_for_integer_coordinates``).  Column
    ``k`` holds the row of the neighbour at offset
    ``FIRST_SHELL_OFFSETS[k]``, or ``-1`` outside the domain.
    """

    points = np.asarray(grid.integer_coordinates, dtype=np.int64)
    rows = np.empty((points.shape[0], FIRST_SHELL_OFFSETS.shape[0]), dtype=np.int64)
    for column, offset in enumerate(FIRST_SHELL_OFFSETS):
        rows[:, column] = grid.rows_for_integer_coordinates(points + offset)
    return rows


__all__ = ["FIRST_SHELL_OFFSETS", "nearest_neighbor_rows"]
