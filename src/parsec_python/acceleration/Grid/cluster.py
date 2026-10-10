"""The cluster grid of the reference package, built slab by slab.

``parsec_python.Grid.build_cluster_grid`` writes the integer and the physical
coordinates of every point of the bounding cube before it keeps the active
ones.  A point of that cube is a triple of positions along three axes, so
the same arrays follow from three axis vectors: the active points of one
slab of constant x are found in a table of its y and z positions, and their
rows, coordinates and lookup entries are written while the slab is in the
cache.  Nothing of the size of the cube is formed but one Boolean per point.
"""

from __future__ import annotations

import os

import numpy as np

from parsec_python.Grid import RealSpaceGrid
from parsec_python.Grid.cluster import _inside_domain
from parsec_python.models import GridSettings

# Relative half-width of the shell around the sphere in which the reference
# test itself decides.  A sum of three squares formed in another order, or
# with a fused product, differs from the reference sum by a few units in the
# last place, which is six orders of magnitude less.
_SHELL = 1.0e-9


def fast_grid_requested() -> bool:
    """``PARSEC_FAST_GRID``: on unless 0, false, no or off.

    On, the accelerated preparation builds the cluster grid with
    :func:`build_cluster_grid_by_slabs`.  Off, it calls the builder of the
    reference package, as before.  Every array of the grid is the same either
    way: shape, type, layout and bytes.
    """

    return os.environ.get("PARSEC_FAST_GRID", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _axis_decisions(settings: GridSettings, positions: list[np.ndarray]) -> list[np.ndarray]:
    """For a box, whether each position of each axis lies inside.

    The reference test of a box is a conjunction over the three components,
    and zero satisfies the test of any component, so the reference function
    decides one axis from points that are zero in the other two.
    """

    decisions = []
    for axis, values in enumerate(positions):
        points = np.zeros((values.size, 3))
        points[:, axis] = values
        decisions.append(_inside_domain(settings, points))
    return decisions


def _active_points(settings: GridSettings, positions: list[np.ndarray]) -> np.ndarray:
    """Return which points of the bounding cube are active, as its Boolean cube.

    ``positions`` holds the physical coordinate of every position along each
    axis.  Entry ``[i, j, k]`` is the decision of the reference test for the
    point at those positions.
    """

    n = positions[0].size
    active = np.empty((n, n, n), dtype=bool)
    if settings.domain_shape != "sphere":
        inside = _axis_decisions(settings, positions)
        np.logical_and(
            inside[0][:, None, None],
            (inside[1][:, None] & inside[2][None, :])[None, :, :],
            out=active,
        )
        return active

    limit = settings.radius**2
    low, high = limit * (1.0 - _SHELL), limit * (1.0 + _SHELL)
    squares = [values * values for values in positions]
    distances = np.empty((n, n))
    shell = np.empty((n, n), dtype=bool)
    for slab in range(n):
        inside = active[slab]
        np.add(
            squares[0][slab] + squares[1][:, None],
            squares[2][None, :],
            out=distances,
        )
        np.less(distances, low, out=inside)
        np.less_equal(distances, high, out=shell)
        np.logical_xor(shell, inside, out=shell)
        if shell.any():
            y, z = np.nonzero(shell)
            points = np.column_stack(
                [
                    np.full(y.size, positions[0][slab]),
                    positions[1][y],
                    positions[2][z],
                ]
            )
            inside[y, z] = _inside_domain(settings, points)
    return active


def build_cluster_grid_by_slabs(settings: GridSettings) -> RealSpaceGrid:
    """Build the grid of ``build_cluster_grid`` without the cube of triples.

    Rows are numbered as there: x, then y, then z, each from its largest
    index to its smallest, which is the C order of the Boolean cube.
    """

    n = int(np.floor(2.0 * settings.enclosing_radius / settings.spacing)) + 2
    index_min = np.full(3, -(n // 2), dtype=int)
    index_max = np.full(3, n + index_min[0] - 1, dtype=int)

    axes = [np.arange(index_max[d], index_min[d] - 1, -1, dtype=int) for d in range(3)]
    shift = np.asarray(settings.shift)
    # The reference expression for one coordinate, on each axis value once.
    positions = [(axes[d] + shift[d]) * settings.spacing for d in range(3)]
    active = _active_points(settings, positions)
    counts = np.count_nonzero(active, axis=(1, 2))

    size = int(counts.sum())
    integer_coordinates = np.empty((size, 3), dtype=axes[0].dtype)
    coordinates = np.empty((size, 3), dtype=np.float64)
    lookup = np.empty((n, n, n), dtype=np.int64)
    start = 0
    for slab in range(n):
        # The lookup table runs from the smallest index to the largest.
        table = lookup[n - 1 - slab]
        table.fill(-1)
        stop = start + int(counts[slab])
        if stop == start:
            continue
        inside = active[slab]
        y, z = np.divmod(np.flatnonzero(inside), n)
        rows = integer_coordinates[start:stop]
        rows[:, 0] = axes[0][slab]
        rows[:, 1] = axes[1][y]
        rows[:, 2] = axes[2][z]
        rows = coordinates[start:stop]
        rows[:, 0] = positions[0][slab]
        rows[:, 1] = positions[1][y]
        rows[:, 2] = positions[2][z]
        table[::-1, ::-1][inside] = np.arange(start, stop)
        start = stop

    return RealSpaceGrid(
        settings=settings,
        integer_coordinates=integer_coordinates,
        coordinates=coordinates,
        index_min=index_min,
        index_max=index_max,
        lookup=lookup,
    )


__all__ = ["build_cluster_grid_by_slabs", "fast_grid_requested"]
