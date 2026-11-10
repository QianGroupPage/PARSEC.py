"""Grids of a wire (1-D periodic) or a slab (2-D periodic) system.

Fortran PARSEC's grid for ``Boundary_Conditions: wire`` (``pbc%per = 1``)
and ``slab`` (``per = 2``):

* the lattice ``latt_vec`` holds the periodic vectors and unit vectors along
  the confined Cartesian directions (``usrinputfile.F90:777-779``): a wire is
  periodic along ``a1 = (a, 0, 0)``, a slab along ``a1``, ``a2`` in the
  ``xy`` plane with ``a3 = z``;
* periodic directions get the steps of ``adjust_step`` and the loop around it
  (an even number of points, ``h_i >= h``, wherever a step is commensurate
  with the translations; :func:`parsec_python.Grid.pbc.adjusted_steps`,
  ``inipbc.f90:100-121``), confined directions keep the
  input spacing and ``n = int(2*rmax/h) + 2`` points (``inipbc.f90:90-92``);
* integer points ``i = -n/2 .. n - n/2 - 1`` along every direction, lattice
  coordinates ``u_i = (shift_i + i)*h_i`` and Cartesian ``r = A_n u``
  (``grid_partition.f90:101-160``), with the shift rule of
  ``usrinputfile.F90:786-794``;
* the domain keeps the points whose confined part of ``u`` has length at
  most ``Boundary_Sphere_Radius``: the cylinder ``y**2 + z**2 <= R**2`` of a
  wire, the slab ``|z| <= R`` of a slab (``grid_partition.f90:146-160``);
* points are numbered with ``i``, then ``j``, then ``k`` running from their
  largest to their smallest value (``grid_partition.f90:146-160``).

The finite-difference stencil wraps around the cell along the periodic
directions and sees zero (orbitals) or the Hartree boundary values (Poisson
equation) beyond the domain along the confined ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ..models import PeriodicCell, PeriodicGridSettings
from .pbc import GeneralPeriodicGrid, adjusted_steps


@dataclass(frozen=True)
class ConfinedPeriodicGrid(GeneralPeriodicGrid):
    """Fortran PARSEC's grid of a wire (``periodicity = 1``) or slab (``2``).

    ``cell.lattice_vectors`` is Fortran's ``latt_vec`` (rows): the periodic
    vectors first, unit vectors along the confined directions.
    ``integer_coordinates`` are Fortran's indices (``index_min = -n/2``);
    ``points_per_axis`` the size of the index box, which along a confined
    direction holds more points than the domain.  ``rows_for_integer_coordinates``
    wraps the periodic directions and returns ``-1`` for a point outside the
    domain, as on a cluster grid.

    Like :class:`GeneralPeriodicGrid` it has no single spacing
    (:attr:`spacing` raises); :attr:`spacings` are the steps along the three
    lattice directions.
    """

    periodicity: int
    confinement_radius: float
    index_min: np.ndarray
    index_max: np.ndarray

    is_confined_periodic = True

    @cached_property
    def volume_element(self) -> float:
        """``hcub = |det(A_n diag(h))|`` (``inipbc.f90:147-161``), bohr**3."""

        return float(abs(np.linalg.det(self.normalized_lattice * self.spacings[None, :])))

    @property
    def periodic_lattice_vectors(self) -> np.ndarray:
        """The periodic lattice vectors (rows, bohr)."""

        return self.cell.lattice_vectors[: self.periodicity]

    @property
    def period(self) -> float:
        """Length of the periodic cell of a wire (``latt_vec(1,1)``), bohr."""

        return float(np.linalg.norm(self.cell.lattice_vectors[0]))

    @property
    def surface_area(self) -> float:
        """Area of the surface cell of a slab, ``|a1 x a2|`` (``a_surface_cell``)."""

        vectors = self.cell.lattice_vectors
        return float(abs(vectors[0, 0] * vectors[1, 1] - vectors[0, 1] * vectors[1, 0]))

    @property
    def box_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.points_per_axis)

    def wrap_periodic(self, points: np.ndarray) -> np.ndarray:
        """Integer points with the periodic indices brought into the box."""

        wrapped = np.array(points, dtype=np.int64, copy=True)
        for axis in range(self.periodicity):
            count = int(self.points_per_axis[axis])
            first = int(self.index_min[axis])
            wrapped[:, axis] = (wrapped[:, axis] - first) % count + first
        return wrapped

    def rows_for_integer_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Active rows; periodic directions wrap, ``-1`` outside the domain."""

        points = np.asarray(points, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("integer grid points must have shape (n, 3)")
        local = self.wrap_periodic(points) - self.index_min
        valid = np.all((local >= 0) & (local < self.points_per_axis), axis=1)
        rows = np.full(points.shape[0], -1, dtype=np.int64)
        if np.any(valid):
            inside = local[valid]
            rows[valid] = self.lookup[inside[:, 0], inside[:, 1], inside[:, 2]]
        return rows

    def lattice_coordinates(self, integer_points: np.ndarray) -> np.ndarray:
        """``u_i = (shift_i + i)*h_i`` of (periodically wrapped) integer points."""

        wrapped = self.wrap_periodic(np.asarray(integer_points)).astype(np.float64)
        return (wrapped + self.shift) * self.spacings

    def physical_coordinates(self, integer_points: np.ndarray) -> np.ndarray:
        """Cartesian ``A_n u`` of integer points; exterior points are not wrapped
        along the confined directions."""

        return self.lattice_coordinates(integer_points) @ self.normalized_lattice.T

    def fractional_coordinates(self, positions: np.ndarray) -> np.ndarray:
        """Coordinates of Cartesian positions along the periodic vectors."""

        positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        inverse = np.linalg.inv(self.cell.lattice_vectors)
        return (positions @ inverse)[:, : self.periodicity]

    def to_box(self, values: np.ndarray) -> np.ndarray:
        """Embed one grid vector in the zero-valued index box."""

        values = np.asarray(values)
        if values.shape != (self.size,):
            raise ValueError("field does not match the wire/slab grid")
        box = np.zeros(self.box_shape, dtype=values.dtype)
        local = self.integer_coordinates - self.index_min
        box[local[:, 0], local[:, 1], local[:, 2]] = values
        return box

    def from_box(self, box: np.ndarray) -> np.ndarray:
        """The grid vector of the active points of an index box."""

        local = self.integer_coordinates - self.index_min
        return np.ascontiguousarray(box[local[:, 0], local[:, 1], local[:, 2]])

    @property
    def box_boundary_modes(self) -> tuple[str, str, str]:
        """``scipy.ndimage`` modes of the box: wrap periodic, zero confined."""

        return tuple("wrap" if axis < self.periodicity else "constant" for axis in range(3))

    def image_translations(self, position: np.ndarray, radius: float) -> np.ndarray:
        """Lattice translations ``T`` for which ``position + T`` can lie within
        ``radius`` of a grid point (an exact, conservative finite set).

        Grid points have coordinates ``f_i = (shift_i + i)/n_i`` along every
        periodic vector, inside ``[-1/2, 1/2)``; an image ``m_i`` cells away
        along ``a_i`` is at least ``|f - f_a - m_i|`` times the distance
        between the lattice planes ``2*pi/|b_i|`` from every grid point.
        """

        fraction = self.fractional_coordinates(position)[0]
        reciprocal = 2.0 * np.pi * np.linalg.inv(self.cell.lattice_vectors).T
        ranges = []
        for axis in range(self.periodicity):
            plane = 2.0 * np.pi / float(np.linalg.norm(reciprocal[axis]))
            count = int(self.points_per_axis[axis])
            first = (int(self.index_min[axis]) + float(self.shift[axis])) / count
            last = (int(self.index_max[axis]) + float(self.shift[axis])) / count
            reach = float(radius) / plane
            lower = int(np.floor(first - fraction[axis] - reach)) - 1
            upper = int(np.ceil(last - fraction[axis] + reach)) + 1
            ranges.append(np.arange(lower, upper + 1))
        mesh = np.meshgrid(*ranges, indexing="ij")
        integers = np.column_stack([axis.reshape(-1) for axis in mesh])
        return integers.astype(np.float64) @ self.periodic_lattice_vectors

    def compensation_image_translations(
        self, position: np.ndarray, width: float
    ) -> np.ndarray:
        """Fortran's images of an atom for the Gaussian-compensated local terms.

        ``ionpot_wire.f90:149`` / ``ionpot_slab.f90:182-183`` (and the local
        forces) sum ``|m_i| <= nint(6*sigma/|a_i|) + 1`` cells around the atom.
        The atom is first brought into the cell (``-1/2 <= f_i < 1/2``) by
        whole lattice vectors, which leaves Fortran's set unchanged for an
        atom already inside it.
        """

        fraction = self.fractional_coordinates(position)[0]
        ranges = []
        for axis in range(self.periodicity):
            length = float(np.linalg.norm(self.cell.lattice_vectors[axis]))
            count = int(np.floor(6.0 * width / length + 0.5)) + 1
            centre = -int(np.floor(fraction[axis] + 0.5))
            ranges.append(np.arange(centre - count, centre + count + 1))
        mesh = np.meshgrid(*ranges, indexing="ij")
        integers = np.column_stack([axis.reshape(-1) for axis in mesh])
        return integers.astype(np.float64) @ self.periodic_lattice_vectors


def confined_lattice_vectors(periodic_vectors: np.ndarray) -> np.ndarray:
    """Fortran's ``latt_vec`` (rows) of a wire or slab.

    ``periodic_vectors`` holds one (wire) or two (slab) rows; the confined
    directions get unit vectors (``usrinputfile.F90:777-779``).
    """

    vectors = np.atleast_2d(np.asarray(periodic_vectors, dtype=np.float64))
    periodicity = vectors.shape[0]
    if periodicity not in (1, 2) or vectors.shape[1] != 3:
        raise ValueError("a wire has one periodic vector, a slab two (rows of 3)")
    lattice = np.eye(3)
    lattice[:periodicity] = vectors
    return lattice


def build_confined_periodic_grid(
    cell: PeriodicCell,
    settings: PeriodicGridSettings,
) -> ConfinedPeriodicGrid:
    """Build Fortran PARSEC's grid of a wire or slab cell.

    ``settings.periodicity`` (1 or 2) and ``settings.confinement_radius``
    (``Boundary_Sphere_Radius``) select the system; ``settings.shift`` is the
    grid shift in units of the steps.  A wire needs ``a1`` along ``x``, a
    slab ``a1`` and ``a2`` in the ``xy`` plane.
    """

    periodicity = int(settings.periodicity)
    if periodicity not in (1, 2):
        raise ValueError("a wire or slab grid needs periodicity 1 or 2")
    radius = float(settings.confinement_radius)
    lattice = np.asarray(cell.lattice_vectors, dtype=np.float64)
    if not np.allclose(lattice[periodicity:], np.eye(3)[periodicity:], atol=0.0, rtol=0.0):
        raise ValueError("the confined lattice directions must be the unit vectors")
    if periodicity == 1 and (lattice[0, 0] <= 0.0 or np.any(lattice[0, 1:] != 0.0)):
        raise ValueError("a wire is periodic along +x: a1 = (a, 0, 0) with a > 0")
    if periodicity == 2 and np.any(lattice[:2, 2] != 0.0):
        raise ValueError("the lattice vectors of a slab lie in the xy plane")
    lengths = np.sqrt(np.sum(lattice**2, axis=1))
    spacing = float(settings.spacing)
    confined_points = int(np.floor(2.0 * radius / spacing)) + 2
    points = [confined_points] * 3
    spacings = [spacing] * 3
    # adjust_step and the loop around it (PARSEC 1.5 inipbc.f90:131-252) run
    # with nper = per: only the periodic directions see the fractional
    # translations of the space group, and their smallest step is the input
    # of the next pass.
    translations = getattr(settings, "fractional_translations", ()) or ((), (), ())
    periodic_points, periodic_steps = adjusted_steps(
        lengths[:periodicity], spacing, translations[:periodicity]
    )
    points[:periodicity] = periodic_points
    spacings[:periodicity] = periodic_steps
    points_per_axis = np.asarray(points, dtype=np.int64)
    spacings_array = np.asarray(spacings, dtype=np.float64)
    half_width = settings.stencil_half_width
    if np.any(points_per_axis[:periodicity] < half_width):
        raise ValueError(
            "the cell is too small for this stencil: each periodic direction "
            f"needs at least {half_width} grid points (expansion_order="
            f"{settings.expansion_order}), got {points_per_axis[:periodicity].tolist()}"
        )
    shift = np.asarray(settings.shift, dtype=np.float64)
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        raise ValueError("the grid shift must have three finite entries")

    normalized = (lattice / lengths[:, None]).T
    gradient_transform = np.linalg.inv(normalized).T
    index_min = -(points_per_axis // 2)
    index_max = points_per_axis + index_min - 1

    axes = [np.arange(index_max[d], index_min[d] - 1, -1, dtype=np.int64) for d in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    integer_all = np.column_stack([component.reshape(-1) for component in mesh])
    lattice_all = (integer_all.astype(np.float64) + shift) * spacings_array
    confined = lattice_all.copy()
    confined[:, :periodicity] = 0.0
    distance = (
        confined[:, 0] * confined[:, 0]
        + confined[:, 1] * confined[:, 1]
        + confined[:, 2] * confined[:, 2]
    )
    keep = ~(distance > radius * radius)
    integer_coordinates = integer_all[keep]
    coordinates = lattice_all[keep] @ normalized.T

    lookup = np.full(tuple(int(value) for value in points_per_axis), -1, dtype=np.int64)
    local = integer_coordinates - index_min
    lookup[local[:, 0], local[:, 1], local[:, 2]] = np.arange(
        integer_coordinates.shape[0], dtype=np.int64
    )
    return ConfinedPeriodicGrid(
        cell=cell,
        settings=settings,
        points_per_axis=points_per_axis,
        integer_coordinates=integer_coordinates,
        coordinates=coordinates,
        lookup=lookup,
        spacings=spacings_array,
        shift=shift,
        normalized_lattice=normalized,
        gradient_transform=gradient_transform,
        periodicity=periodicity,
        confinement_radius=radius,
        index_min=index_min,
        index_max=index_max,
    )


__all__ = [
    "ConfinedPeriodicGrid",
    "build_confined_periodic_grid",
    "confined_lattice_vectors",
]
