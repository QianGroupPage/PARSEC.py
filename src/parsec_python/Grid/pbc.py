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
        """Whether the lattice vectors lie along the Cartesian axes with one common step (no mixed derivatives,
        one grid spacing).  A rectangular cell whose steps differ goes through the general (metric) code."""
        if not _is_diagonal(self.cell.lattice_vectors):
            return False
        steps = np.diag(self.cell.lattice_vectors) / self.points_per_axis
        return bool(np.allclose(steps, steps[0], rtol=1.0e-9, atol=0.0))

    @property
    def shift(self) -> np.ndarray:
        """Offset of the first point along each lattice axis, in grid steps."""
        return np.asarray(self.settings.shift, dtype=np.float64)

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
            return (wrapped + self.shift) @ self.step_vectors
        return (wrapped + self.shift) * self.spacing

    def integrate(self, values: np.ndarray) -> float:
        values = np.asarray(values)
        if values.shape[0] != self.size:
            raise ValueError("values do not match the active grid")
        return float(np.sum(values) * self.volume_element)


def _is_diagonal(lattice_vectors: np.ndarray) -> bool:
    off_diagonal = lattice_vectors - np.diag(np.diag(lattice_vectors))
    return bool(np.allclose(off_diagonal, 0.0, atol=1.0e-10))


def fortran_points_per_axis(lengths: np.ndarray, spacing: float) -> np.ndarray:
    """Points along each lattice vector as Fortran's ``adjust_step`` (``inipbc.f90``) chooses them.

    The loop counts ``n = 1, 2, ...`` while ``1/n`` is not below ``spacing/length`` and keeps the last ``n`` for which
    ``0.5*n`` is an integer: the largest even ``n`` whose step ``length/n`` is not below ``spacing``.  (Symmetry
    operations with fractional translations would add commensurability conditions; PARSEC.py finds none.)  The
    comparison is made on ``1/n`` and ``spacing/length`` in double precision, as Fortran does, so a spacing that is
    within round-off of ``length/n`` falls on the same side: ``9.6/0.4`` gives 22 points, ``9.6/(9.6/24)`` gives 24.
    """

    counts = []
    for length in np.asarray(lengths, dtype=float):
        chosen = 0
        ratio = spacing / float(length)
        for n in range(1, 1001):
            if 1.0 / n < ratio:
                break
            ytest = 0.5 * n - round(0.5 * n)  # the even-n condition: (1/2)/(1/n) is an integer
            if abs(ytest) <= 1.0e-8:
                chosen = n
        else:
            raise ValueError("no commensurate grid spacing found (Fortran's adjust_step error 266)")
        if chosen == 0:
            raise ValueError(f"the cell side {length} is shorter than two grid steps of {spacing}")
        counts.append(chosen)
    return np.array(counts, dtype=np.int64)


def fortran_grid_shift(lattice_vectors: np.ndarray, ignore_symmetry: bool) -> tuple[float, float, float]:
    """The grid shift of ``usrinputfile.F90``: half a step along every lattice vector, except zero when
    ``Ignore_Symmetry`` is set and along vectors that are not orthogonal to another one."""

    lattice = np.asarray(lattice_vectors, dtype=float)
    shift = np.zeros(3) if ignore_symmetry else np.full(3, 0.5)
    lengths = np.linalg.norm(lattice, axis=1)
    for i in range(3):
        for j in range(3):
            if i != j and abs(float(lattice[i] @ lattice[j])) > 1.0e-9 * lengths[i] * lengths[j]:
                shift[i] = 0.0
    return tuple(float(value) for value in shift)


def build_periodic_grid(
    cell: PeriodicCell,
    settings: PeriodicGridSettings,
    *,
    tolerance: float = 1.0e-6,
) -> PeriodicRealSpaceGrid:
    """
    Made to mirror build_cluster_grid in cluster.py (Potentially make abstract class for Grids)
    Build a periodic grid over one unit cell.

    ``spacing`` is a target, turned into points per lattice vector by ``settings.points_rule``.  ``"round"``: in an
    orthorhombic cell the achieved spacing is ``side_length[i] / round(side_length[i] / spacing)`` for each axis, and
    must agree between axes to within ``tolerance``; in an oblique cell the grid axes follow the lattice vectors, each
    with the nearest even number of points.  ``"fortran"``: Fortran's ``adjust_step`` (:func:`fortran_points_per_axis`)
    for any cell, with steps that may differ between axes (the Laplacian then carries the metric of the cell, see
    ``Laplacian.lattice_metric``).  Point ``(i, j, k)`` is at ``(i + shift_0, j + shift_1, k + shift_2)`` steps along
    the lattice vectors (``settings.shift``).
    """

    lattice_vectors = cell.lattice_vectors
    if settings.points_rule == "fortran":
        lengths = np.linalg.norm(lattice_vectors, axis=1)
        return _build_general_grid(cell, settings, fortran_points_per_axis(lengths, settings.spacing))
    if not _is_diagonal(lattice_vectors):
        lengths = np.linalg.norm(lattice_vectors, axis=1)
        # an even number of points per axis, the nearest to |a_i|/h
        return _build_general_grid(cell, settings, (2 * np.round(lengths / (2.0 * settings.spacing))).astype(np.int64))
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
    coordinates = (integer_coordinates.astype(np.float64) + np.asarray(settings.shift)) * achieved_spacing
    lookup = np.arange(n0 * n1 * n2, dtype=np.int64).reshape(n0, n1, n2)

    return PeriodicRealSpaceGrid(
        cell=cell,
        settings=settings,
        points_per_axis=points_per_axis,
        integer_coordinates=integer_coordinates,
        coordinates=coordinates,
        lookup=lookup,
    )


def _build_general_grid(
    cell: PeriodicCell, settings: PeriodicGridSettings, points_per_axis: np.ndarray
) -> PeriodicRealSpaceGrid:
    """Grid along the lattice vectors with ``points_per_axis`` points (any cell, steps that may differ)."""

    lattice_vectors = cell.lattice_vectors
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
    steps = lattice_vectors / points_per_axis[:, None]
    coordinates = (integer_coordinates.astype(np.float64) + np.asarray(settings.shift)) @ steps
    lookup = np.arange(n0 * n1 * n2, dtype=np.int64).reshape(n0, n1, n2)
    return PeriodicRealSpaceGrid(
        cell=cell,
        settings=settings,
        points_per_axis=np.asarray(points_per_axis, dtype=np.int64),
        integer_coordinates=integer_coordinates,
        coordinates=coordinates,
        lookup=lookup,
    )


__all__ = ["PeriodicRealSpaceGrid", "build_periodic_grid", "fortran_grid_shift", "fortran_points_per_axis"]
