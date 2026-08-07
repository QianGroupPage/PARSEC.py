"""PARSEC-compatible uniform cluster grids.

This module contains the native Python implementation of PARSEC's isolated
real-space domain construction.  It can be imported independently of the SCF
driver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import GridSettings


@dataclass(frozen=True)
class RealSpaceGrid:
    """Active points of a uniform, nonperiodic PARSEC cluster grid.

    Integer points are ordered like ``grid_partition.f90``: x, then y, then z,
    all traversed from their maximum index to their minimum index.
    """

    settings: GridSettings
    integer_coordinates: np.ndarray
    coordinates: np.ndarray
    index_min: np.ndarray
    index_max: np.ndarray
    lookup: np.ndarray

    @property
    def spacing(self) -> float:
        return self.settings.spacing

    @property
    def volume_element(self) -> float:
        return self.spacing**3

    @property
    def size(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in (self.index_max - self.index_min + 1))

    def rows_for_integer_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Return active row numbers, or ``-1`` for points outside the domain."""
        points = np.asarray(points, dtype=int)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("integer grid points must have shape (n, 3)")
        local = points - self.index_min
        valid = np.all((local >= 0) & (local < np.asarray(self.lookup.shape)), axis=1)
        rows = np.full(points.shape[0], -1, dtype=np.int64)
        if np.any(valid):
            loc = local[valid]
            rows[valid] = self.lookup[loc[:, 0], loc[:, 1], loc[:, 2]]
        return rows

    def physical_coordinates(self, integer_points: np.ndarray) -> np.ndarray:
        integer_points = np.asarray(integer_points, dtype=float)
        return (integer_points + np.asarray(self.settings.shift)) * self.spacing

    def integrate(self, values: np.ndarray) -> float:
        values = np.asarray(values)
        if values.shape[0] != self.size:
            raise ValueError("values do not match the active grid")
        return float(np.sum(values) * self.volume_element)


def _inside_domain(settings: GridSettings, coordinates: np.ndarray) -> np.ndarray:
    if settings.domain_shape == "sphere":
        return np.einsum("ij,ij->i", coordinates, coordinates) <= settings.radius**2
    lengths = np.asarray(settings.box_lengths, dtype=float)
    return np.all(np.abs(coordinates) <= 0.5 * lengths + 1.0e-14, axis=1)


def build_cluster_grid(settings: GridSettings) -> RealSpaceGrid:
    """Build the full active cluster domain without symmetry reduction.

    PARSEC uses ``n = floor(2*radius/h) + 2`` for a confined system, integer
    bounds ``[-n/2, n-n/2-1]``, and positions ``h*(integer + shift)``.
    """
    n = int(np.floor(2.0 * settings.enclosing_radius / settings.spacing)) + 2
    index_min = np.full(3, -(n // 2), dtype=int)
    index_max = np.full(3, n + index_min[0] - 1, dtype=int)

    axes = [np.arange(index_max[d], index_min[d] - 1, -1, dtype=int) for d in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    integer_coordinates = np.column_stack([component.reshape(-1) for component in mesh])
    coordinates = (integer_coordinates + np.asarray(settings.shift)) * settings.spacing
    active = _inside_domain(settings, coordinates)
    integer_coordinates = integer_coordinates[active]
    coordinates = coordinates[active]

    lookup = np.full((n, n, n), -1, dtype=np.int64)
    local = integer_coordinates - index_min
    lookup[local[:, 0], local[:, 1], local[:, 2]] = np.arange(integer_coordinates.shape[0])

    return RealSpaceGrid(
        settings=settings,
        integer_coordinates=integer_coordinates,
        coordinates=coordinates,
        index_min=index_min,
        index_max=index_max,
        lookup=lookup,
    )


__all__ = ["RealSpaceGrid", "build_cluster_grid"]
