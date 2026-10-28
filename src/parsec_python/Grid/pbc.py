"""
Periodic real-space PARSEC grids over a general (orthorhombic or oblique) unit cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import PeriodicCell, PeriodicGridSettings


@dataclass(frozen=True)
class PeriodicRealSpaceGrid:
    """
    Made to mirror RealSpaceGrid in cluster.py (Potentially make abstract class for Grids)
    Every point of one periodic unit cell.

    Integer points are ordered row-major (axis 0 slowest, axis 2 fastest)
    over ``[0, n0) x [0, n1) x [0, n2)``.  Point ``(i, j, k)`` sits at
    ``i*s1 + j*s2 + k*s3`` with the step vectors ``s_m = a_m/n_m``, so an
    orthorhombic cell has the Cartesian axes as grid axes and an oblique one
    has grid axes along its lattice vectors.
    """

    cell: PeriodicCell
    settings: PeriodicGridSettings
    points_per_axis: np.ndarray
    integer_coordinates: np.ndarray
    coordinates: np.ndarray
    lookup: np.ndarray

    @property
    def is_orthorhombic(self) -> bool:
        """Whether the lattice vectors lie along the Cartesian axes (no mixed derivatives)."""
        return _is_diagonal(self.cell.lattice_vectors)

    @property
    def step_vectors(self) -> np.ndarray:
        """Rows ``s_m = a_m/n_m``: the displacement of one grid step along each grid axis."""
        return self.cell.lattice_vectors / self.points_per_axis[:, None]

    @property
    def spacing(self) -> float:
        """The grid spacing: shared by all three axes of an orthorhombic cell, the shortest step of an oblique one."""
        if not self.is_orthorhombic:
            return float(np.min(np.linalg.norm(self.step_vectors, axis=1)))
        side_lengths = np.diag(self.cell.lattice_vectors)
        return float(side_lengths[0] / self.points_per_axis[0])

    @property
    def volume_element(self) -> float:
        return self.cell.volume / float(np.prod(self.points_per_axis))

    @property
    def size(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.points_per_axis)

    def rows_for_integer_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Return active row numbers; every point maps in via wraparound."""
        points = np.asarray(points, dtype=int)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("integer grid points must have shape (n, 3)")
        wrapped = points % self.points_per_axis
        return self.lookup[wrapped[:, 0], wrapped[:, 1], wrapped[:, 2]]

    def physical_coordinates(self, integer_points: np.ndarray) -> np.ndarray:
        integer_points = np.asarray(integer_points, dtype=float)
        wrapped = integer_points % self.points_per_axis
        if not self.is_orthorhombic:
            return wrapped @ self.step_vectors
        return wrapped * self.spacing

    def integrate(self, values: np.ndarray) -> float:
        values = np.asarray(values)
        if values.shape[0] != self.size:
            raise ValueError("values do not match the active grid")
        return float(np.sum(values) * self.volume_element)


def _is_diagonal(lattice_vectors: np.ndarray) -> bool:
    off_diagonal = lattice_vectors - np.diag(np.diag(lattice_vectors))
    return bool(np.allclose(off_diagonal, 0.0, atol=1.0e-10))


def build_periodic_grid(
    cell: PeriodicCell,
    settings: PeriodicGridSettings,
    *,
    tolerance: float = 1.0e-6,
) -> PeriodicRealSpaceGrid:
    """
    Made to mirror build_cluster_grid in cluster.py (Potentially make abstract class for Grids)
    Build a periodic grid over one unit cell.

    ``spacing`` is a target.  In an orthorhombic cell the achieved spacing is
    ``side_length[i] / round(side_length[i] / spacing)`` for each axis, and
    must agree between axes to within ``tolerance``.  In an oblique cell the
    grid axes follow the lattice vectors, each with the even number of points
    ``2*round(|a_i| / (2*spacing))`` of the Fortran code; the steps along the three axes may differ (the Laplacian then
    carries the metric of the cell, see ``Laplacian.lattice_metric``).
    """

    lattice_vectors = cell.lattice_vectors
    if not _is_diagonal(lattice_vectors):
        return _build_oblique_grid(cell, settings)
    side_lengths = np.diag(lattice_vectors)
    if np.any(side_lengths <= 0.0):
        raise ValueError("lattice_vectors must have positive diagonal side lengths")
    spacing = settings.spacing

    points_per_axis = np.round(side_lengths / spacing).astype(np.int64)
    half_width = settings.stencil_half_width
    if np.any(points_per_axis <= 2 * half_width):
        raise ValueError(
            "the cell is too small for this stencil: each axis needs more "
            f"than {2 * half_width} grid points (expansion_order="
            f"{settings.expansion_order}), got {points_per_axis.tolist()}"
        )
    achieved_spacings = side_lengths / points_per_axis
    if not np.allclose(achieved_spacings, achieved_spacings[0], rtol=tolerance):
        raise ValueError(
            "cell side lengths are not commensurate with a single common "
            f"grid spacing near {spacing}; achieved per-axis spacings would "
            f"be {achieved_spacings.tolist()}, which disagree by more than "
            f"the relative tolerance {tolerance}"
        )

    n0, n1, n2 = (int(value) for value in points_per_axis)
    axes = [np.arange(n) for n in (n0, n1, n2)]
    mesh = np.meshgrid(*axes, indexing="ij")
    integer_coordinates = np.column_stack([component.reshape(-1) for component in mesh])
    achieved_spacing = float(achieved_spacings[0])
    coordinates = integer_coordinates.astype(np.float64) * achieved_spacing
    lookup = np.arange(n0 * n1 * n2, dtype=np.int64).reshape(n0, n1, n2)

    return PeriodicRealSpaceGrid(
        cell=cell,
        settings=settings,
        points_per_axis=points_per_axis,
        integer_coordinates=integer_coordinates,
        coordinates=coordinates,
        lookup=lookup,
    )


def _build_oblique_grid(cell: PeriodicCell, settings: PeriodicGridSettings) -> PeriodicRealSpaceGrid:
    lattice_vectors = cell.lattice_vectors
    lengths = np.linalg.norm(lattice_vectors, axis=1)
    # an even number of points per axis, the nearest to |a_i|/h (the Fortran code's 2*nint(L/(2h)))
    points_per_axis = (2 * np.round(lengths / (2.0 * settings.spacing))).astype(np.int64)
    half_width = settings.stencil_half_width
    if np.any(points_per_axis <= 2 * half_width):
        raise ValueError(
            "the cell is too small for this stencil: each axis needs more "
            f"than {2 * half_width} grid points (expansion_order="
            f"{settings.expansion_order}), got {points_per_axis.tolist()}"
        )
    n0, n1, n2 = (int(value) for value in points_per_axis)
    mesh = np.meshgrid(*[np.arange(n) for n in (n0, n1, n2)], indexing="ij")
    integer_coordinates = np.column_stack([component.reshape(-1) for component in mesh])
    coordinates = integer_coordinates.astype(np.float64) @ (lattice_vectors / points_per_axis[:, None])
    lookup = np.arange(n0 * n1 * n2, dtype=np.int64).reshape(n0, n1, n2)
    return PeriodicRealSpaceGrid(
        cell=cell,
        settings=settings,
        points_per_axis=points_per_axis,
        integer_coordinates=integer_coordinates,
        coordinates=coordinates,
        lookup=lookup,
    )


__all__ = ["PeriodicRealSpaceGrid", "build_periodic_grid"]
