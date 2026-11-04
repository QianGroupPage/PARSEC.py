"""
Periodic real-space PARSEC grids over a general (orthorhombic or oblique) unit cell.

Two sets of routines build them (``PeriodicGridSettings.fortran_routines``, the input's
``Use_Fortran_Routines``; :mod:`parsec_python.Input.routines`):

* the default (Andrew Grieder's): :class:`PeriodicRealSpaceGrid` for every cell, with grid axes along the
  lattice vectors and the step vectors ``s_m = a_m/n_m``; an oblique cell carries the metric of the cell in
  its Laplacian (:mod:`parsec_python.Laplacian.lattice_metric`);
* the Fortran-derived routines of the parity round: :class:`PeriodicRealSpaceGrid` for an orthorhombic cell
  whose three sides share one grid spacing (``n_i = round(L_i/h)``, points ``i*h``, the port's original
  convention), and :class:`GeneralPeriodicGrid` for every other cell (non-orthogonal lattice vectors, or
  sides without a common spacing): Fortran PARSEC's grid along the unit lattice directions (F ``inipbc.f90``,
  ``grid_partition.f90``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ..models import PeriodicCell, PeriodicGridSettings
from .cell import is_diagonal_lattice


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

    With the Fortran-derived routines (``settings.fortran_routines``) this class is the orthorhombic grid of
    one common spacing only, whose points start at the origin (every other cell gets :class:`GeneralPeriodicGrid`).
    """

    cell: PeriodicCell
    settings: PeriodicGridSettings
    points_per_axis: np.ndarray
    integer_coordinates: np.ndarray
    coordinates: np.ndarray
    lookup: np.ndarray

    @property
    def boundary_modes(self) -> tuple[str, str, str]:
        """How a finite-difference stencil treats each grid axis beyond the grid: ``wrap`` (periodic) or ``constant`` (zero)."""
        return ("wrap", "wrap", "wrap")

    @property
    def is_orthorhombic(self) -> bool:
        """Whether the lattice vectors lie along the Cartesian axes with one common step (no mixed derivatives,
        one grid spacing).  A rectangular cell whose steps differ goes through the general (metric) code."""
        if getattr(self.settings, "fortran_routines", False):
            # the Fortran-derived routines build this class for an orthorhombic cell of one spacing only
            return True
        if not _is_diagonal(self.cell.lattice_vectors):
            return False
        steps = np.diag(self.cell.lattice_vectors) / self.points_per_axis
        return bool(np.allclose(steps, steps[0], rtol=1.0e-9, atol=0.0))

    @cached_property
    def shift(self) -> np.ndarray:
        """Offset of the first point along each lattice axis, in grid steps.

        Zero on the orthorhombic grid of the Fortran-derived routines, which does not read ``settings.shift``.
        (A cached property: :class:`GeneralPeriodicGrid` stores its own as a field.)
        """
        if getattr(self.settings, "fortran_routines", False):
            return np.zeros(3, dtype=np.float64)
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


def uses_fortran_routines(grid) -> bool:
    """Whether ``grid`` belongs to the Fortran-derived routines (``Use_Fortran_Routines``).

    Every :class:`GeneralPeriodicGrid` (and wire/slab grid) does, and so does a grid whose settings say
    ``fortran_routines``; a cluster grid and the grids of the default routines do not.  Code with one routine
    of each set dispatches on it (``acceleration/V_ion/periodic_setup.py`` and its ``_fortran`` twin, ...).
    """
    if getattr(grid, "is_general_periodic", False):
        return True
    return bool(getattr(getattr(grid, "settings", None), "fortran_routines", False))


@dataclass(frozen=True)
class GeneralPeriodicGrid(PeriodicRealSpaceGrid):
    """Fortran PARSEC's periodic grid for a general (e.g. non-orthogonal) cell (Fortran-derived routines).

    Points are ``r = A_n u`` with lattice-direction coordinates
    ``u_i = (i + shift_i) * h_i``, ``i = 0 .. n_i-1``, where the columns of
    ``A_n`` (``normalized_lattice``, Fortran ``pbc%avec_norm``) are the unit
    lattice vectors (F ``grid_partition.f90:146-151``, ``inipbc.f90:127-134``).
    Fortran numbers ``i`` from ``-n_i/2``; both sets are the same modulo the
    lattice, and this one keeps every point inside the cell
    (fractional coordinates ``(i + shift_i)/n_i`` in ``[0, 1)``).  Integer
    points are ordered row-major as in :class:`PeriodicRealSpaceGrid`, and
    every neighbour lookup wraps around.

    ``spacings`` are the steps ``h_i`` along the three lattice directions
    (Fortran ``grid%step``), chosen by :func:`adjust_step`.
    ``gradient_transform`` is ``inv(A_n)^T`` (Fortran ``grid%grad_bvec_norm``):
    the Cartesian gradient is ``gradient_transform @ (d/du_1, d/du_2, d/du_3)``.

    There is no single grid spacing: :attr:`spacing` raises, so that code
    written for the orthorhombic grid fails loudly instead of using a wrong
    step.  Use :attr:`spacings`.  Code of the default routines that branches on
    :attr:`is_orthorhombic` must test ``is_general_periodic`` first: this grid
    has its own operators (:mod:`parsec_python.Laplacian.nonorthogonal`).
    """

    spacings: np.ndarray
    # keyword-only: the base class names a cached property ``shift`` (Andrew Grieder's grids)
    shift: np.ndarray = field(kw_only=True)
    normalized_lattice: np.ndarray
    gradient_transform: np.ndarray

    is_general_periodic = True

    @property
    def is_orthorhombic(self) -> bool:
        return False

    @property
    def step_vectors(self) -> np.ndarray:
        """Rows ``h_i * a_i/|a_i|``: one step along each lattice direction (Fortran's ``grid%step``)."""
        return self.spacings[:, None] * self.normalized_lattice.T

    @property
    def spacing(self) -> float:
        raise AttributeError(
            "a general periodic grid has one spacing per lattice direction "
            f"({self.spacings.tolist()} bohr); use GeneralPeriodicGrid.spacings"
        )

    def physical_coordinates(self, integer_points: np.ndarray) -> np.ndarray:
        integer_points = np.asarray(integer_points, dtype=float)
        lattice_coordinates = (
            (integer_points % self.points_per_axis) + self.shift
        ) * self.spacings
        return lattice_coordinates @ self.normalized_lattice.T


def adjust_step(
    length: float,
    spacing: float,
    *,
    maximum: int = 1000,
    translations: tuple[float, ...] = (),
) -> tuple[int, float]:
    """Number of points and step along one lattice direction (F ``adjust_step``).

    Translation of ``inipbc.f90:327-429``.  ``translations`` are the
    fractional translations ``|M t|`` of the space group along this
    direction (:meth:`parsec_python.Grid.space_group.SpaceGroup.commensurate_translations`;
    none for a symmorphic group or ``Ignore_Symmetry``).  With ``tau_min``
    the smallest of them (1 without any), ``x = tau_min/nn`` for
    ``nn = 1, 2, ...`` until ``x < h/L``; a trial is kept when every
    ``tau/x`` and ``0.5/x`` are integers (within ``1e-8``), so the last one
    kept is the smallest commensurate step ``x*L >= h`` with an even number
    of points.  The step is ``x*L`` and ``n = 2*nint(L/(2*step))``
    (``inipbc.f90:107-121``).  Example without translations: 10 bohr at
    ``h = 0.6`` gives ``n = 16`` and ``h = 0.625``.

    No trial commensurate with the translations (a translation such as
    0.27 of an atom moved along a three-fold axis, which no even ``n`` near
    ``L/h`` divides): Fortran's loop then ends without setting the step,
    and ``grid%step`` keeps the input spacing that ``inipbc.f90:84`` put
    there, so ``n = 2*nint(L/(2h))``.  This port takes that ``n`` with the
    step ``L/n``.  Fortran keeps the step ``h`` itself, a grid of ``n h``
    in a cell of length ``L`` (fcc Si2, a 0.27 translation, h 0.6: twelve
    points 0.600 bohr apart along 7.2549 bohr lattice vectors, 0.0204 Ry
    above its own run of twelve points with ``Ignore_Symmetry``, whose step
    is 0.605).  The operations then do not map the grid onto itself: what
    needs that (the density symmetrization) refuses them by name
    (:func:`parsec_python.Grid.space_group.grid_commensurability_problem`).
    """

    length = float(length)
    spacing = float(spacing)
    taus = tuple(float(value) for value in translations)
    tau_min = min(taus) if taus else 1.0
    step = None
    for trial in range(1, maximum + 1):
        if tau_min / float(trial) < spacing / length:
            break
        x = tau_min / float(trial)
        commensurate = True
        for tau in taus:
            ratio = tau / x
            if abs(ratio - np.floor(ratio + 0.5)) > 1.0e-8:
                commensurate = False
        ratio = 0.5 / x
        if abs(ratio - np.floor(ratio + 0.5)) > 1.0e-8:
            continue
        if not commensurate:
            continue
        step = x * length
    if step is None and taus:
        points = 2 * int(np.floor(length / (2.0 * spacing) + 0.5))
        if points >= 2:
            return points, length / float(points)
    if step is None:
        raise ValueError(
            f"a lattice vector of {length:.6g} bohr is shorter than two grid "
            f"spacings of {spacing:.6g} bohr; PARSEC needs an even number of "
            "points along every lattice direction"
        )
    points = 2 * int(np.floor(length / (2.0 * step) + 0.5))
    return points, step


def uses_general_periodic_grid(
    cell: PeriodicCell,
    settings: PeriodicGridSettings,
    *,
    tolerance: float = 1.0e-6,
) -> bool:
    """Whether ``cell`` gets the :class:`GeneralPeriodicGrid` (Fortran-derived routines).

    The orthorhombic grid takes every cell it always accepted: diagonal
    lattice vectors with positive sides, more than ``2*M`` points per axis
    (``M`` the stencil half width) and one common spacing within
    ``tolerance``.  Every other cell - which that grid refuses - gets
    Fortran's general grid instead.
    """

    lattice_vectors = cell.lattice_vectors
    if not is_diagonal_lattice(lattice_vectors):
        return True
    side_lengths = np.diag(lattice_vectors)
    if np.any(side_lengths <= 0.0):
        return True
    points_per_axis = np.round(side_lengths / settings.spacing).astype(np.int64)
    if np.any(points_per_axis <= 2 * settings.stencil_half_width):
        return True
    achieved_spacings = side_lengths / points_per_axis
    return not np.allclose(achieved_spacings, achieved_spacings[0], rtol=tolerance)


def periodic_grid_shift(
    lattice_vectors: np.ndarray, *, ignore_symmetry: bool
) -> tuple[float, float, float]:
    """Fortran's grid shift of a periodic cell, in units of the grid steps (Fortran-derived routines).

    ``shift = 0.5`` along every direction, ``0`` everywhere with
    ``Ignore_Symmetry``, and ``0`` along a periodic direction that is not
    orthogonal to another one (F ``usrinputfile.F90:786-794``).  Only the
    general grid applies it; the orthorhombic grid of this port has always
    used ``shift = 0``.
    """

    vectors = np.asarray(lattice_vectors, dtype=float)
    shift = [0.0, 0.0, 0.0] if ignore_symmetry else [0.5, 0.5, 0.5]
    metric = vectors @ vectors.T
    for first in range(3):
        for second in range(3):
            if first != second and abs(metric[first, second]) > 0.0:
                shift[first] = 0.0
    return tuple(shift)


def build_general_periodic_grid(
    cell: PeriodicCell,
    settings: PeriodicGridSettings,
) -> GeneralPeriodicGrid:
    """Build Fortran PARSEC's grid over one general periodic cell (Fortran-derived routines).

    ``|a_i|`` (``pbc%box_size``) and the target ``settings.spacing`` give
    ``n_i`` and ``h_i`` by :func:`adjust_step`; the lattice-direction
    coordinates use ``settings.shift``.  The finite-difference stencil wraps
    around the cell, which Fortran's wrap tables (``grid_partition.f90:543-697``)
    do correctly only when ``n_i >= M``: that is required here as well.
    """

    lattice_vectors = np.asarray(cell.lattice_vectors, dtype=float)
    lengths = np.sqrt(np.sum(lattice_vectors**2, axis=1))
    translations = getattr(settings, "fractional_translations", ()) or ((), (), ())
    points = []
    spacings = []
    for length, taus in zip(lengths, translations):
        count, step = adjust_step(length, settings.spacing, translations=taus)
        points.append(count)
        spacings.append(step)
    points_per_axis = np.asarray(points, dtype=np.int64)
    spacings_array = np.asarray(spacings, dtype=float)
    half_width = settings.stencil_half_width
    if np.any(points_per_axis < half_width):
        raise ValueError(
            "the cell is too small for this stencil: each lattice direction "
            f"needs at least {half_width} grid points (expansion_order="
            f"{settings.expansion_order}), got {points_per_axis.tolist()}"
        )
    shift = np.asarray(settings.shift, dtype=float)
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        raise ValueError("the grid shift must have three finite entries")

    normalized = (lattice_vectors / lengths[:, None]).T
    gradient_transform = np.linalg.inv(normalized).T

    n0, n1, n2 = (int(value) for value in points_per_axis)
    axes = [np.arange(n) for n in (n0, n1, n2)]
    mesh = np.meshgrid(*axes, indexing="ij")
    integer_coordinates = np.column_stack([component.reshape(-1) for component in mesh])
    lattice_coordinates = (integer_coordinates.astype(np.float64) + shift) * spacings_array
    coordinates = lattice_coordinates @ normalized.T
    lookup = np.arange(n0 * n1 * n2, dtype=np.int64).reshape(n0, n1, n2)

    return GeneralPeriodicGrid(
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
    )


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

    ``settings.fortran_routines`` (``Use_Fortran_Routines``) selects :func:`build_fortran_periodic_grid`, the
    grids of the Fortran-derived routines.
    """

    if getattr(settings, "fortran_routines", False):
        return build_fortran_periodic_grid(cell, settings, tolerance=tolerance)
    lattice_vectors = cell.lattice_vectors
    if cell.dimensions == 2:
        # a slab (Andrew Grieder's slab_boundary): its own grid, whatever the points rule
        return build_slab_grid(cell, settings)
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


def build_fortran_periodic_grid(
    cell: PeriodicCell,
    settings: PeriodicGridSettings,
    *,
    tolerance: float = 1.0e-6,
) -> PeriodicRealSpaceGrid:
    """The grid of the Fortran-derived routines (``Use_Fortran_Routines``).

    An orthorhombic unit cell whose sides share one spacing gets :class:`PeriodicRealSpaceGrid`: the achieved
    spacing is ``side_length[i] / round(side_length[i] / spacing)`` for each axis, and must agree between axes to
    within ``tolerance``; its points start at the origin (``settings.shift`` is not read).

    A cell this orthorhombic grid does not accept (see
    :func:`uses_general_periodic_grid`) gets Fortran's
    :class:`GeneralPeriodicGrid` from :func:`build_general_periodic_grid`.
    """

    if uses_general_periodic_grid(cell, settings, tolerance=tolerance):
        return build_general_periodic_grid(cell, settings)

    lattice_vectors = cell.lattice_vectors
    off_diagonal = lattice_vectors - np.diag(np.diag(lattice_vectors))
    if not np.allclose(off_diagonal, 0.0, atol=1.0e-10):
        raise ValueError(
            "build_periodic_grid only supports an orthorhombic cell "
            "(diagonal lattice_vectors); a general oblique cell needs a "
            "metric-tensor-weighted stencil that has not been implemented"
        )
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


@dataclass(frozen=True)
class SlabRealSpaceGrid(PeriodicRealSpaceGrid):
    """A slab: periodic in the two lattice vectors of the xy plane, a finite box along z.

    Points are ``i*s1 + j*s2 + (z0 + k*hz) e_z`` with ``i, j`` periodic and ``k = 0 ... n3-1`` the planes of the box,
    which are symmetric about ``z = 0`` (``z0 = -(n3 - 1) hz / 2``, the Fortran code's half-step shift).  A neighbor
    beyond the first or last plane is outside the box (row ``-1``): wavefunctions vanish there, and the Hartree
    potential takes the values of the sheet expansion (:mod:`parsec_python.Hartree.slab`).
    """

    z_step: float = 0.0
    z_origin: float = 0.0

    @property
    def boundary_modes(self) -> tuple[str, str, str]:
        return ("wrap", "wrap", "constant")

    @property
    def is_orthorhombic(self) -> bool:
        return False  # the in-plane axes and the metric always go through the general (oblique) stencil

    @property
    def step_vectors(self) -> np.ndarray:
        vectors = np.zeros((3, 3))
        vectors[:2] = self.cell.lattice_vectors[:2] / self.points_per_axis[:2, None]
        vectors[2, 2] = self.z_step
        return vectors

    @property
    def spacing(self) -> float:
        return float(np.min(np.linalg.norm(self.step_vectors, axis=1)))

    @property
    def cell_area(self) -> float:
        in_plane = self.cell.lattice_vectors[:2, :2]
        return float(abs(in_plane[0, 0] * in_plane[1, 1] - in_plane[0, 1] * in_plane[1, 0]))

    @property
    def volume_element(self) -> float:
        return self.cell_area * self.z_step / float(self.points_per_axis[0] * self.points_per_axis[1])

    def rows_for_integer_coordinates(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=int)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("integer grid points must have shape (n, 3)")
        n = self.points_per_axis
        rows = np.full(points.shape[0], -1, dtype=np.int64)
        inside = (points[:, 2] >= 0) & (points[:, 2] < n[2])
        wrapped = points[inside] % n
        rows[inside] = self.lookup[wrapped[:, 0], wrapped[:, 1], points[inside, 2]]
        return rows

    def physical_coordinates(self, integer_points: np.ndarray) -> np.ndarray:
        integer_points = np.asarray(integer_points, dtype=float)
        coordinates = np.empty_like(integer_points)
        coordinates[:, :2] = (integer_points[:, :2] % self.points_per_axis[:2]) @ self.step_vectors[:2, :2]
        coordinates[:, 2] = self.z_origin + integer_points[:, 2] * self.z_step
        return coordinates

    def integrate(self, values: np.ndarray) -> float:
        values = np.asarray(values)
        if values.shape[0] != self.size:
            raise ValueError("values do not match the active grid")
        return float(np.sum(values) * self.volume_element)


def build_slab_grid(cell: PeriodicCell, settings: PeriodicGridSettings) -> SlabRealSpaceGrid:
    """Grid of a slab cell: Fortran's ``2*nint(|a_i|/(2h))`` points along the lattice vectors of the plane and
    ``int(2R/h) + 2`` planes of spacing ``h`` along z, with ``2R`` the third lattice vector (``Boundary_Sphere_Radius``)."""

    lattice = cell.lattice_vectors
    height = float(lattice[2, 2])
    in_plane = (2 * np.round(np.linalg.norm(lattice[:2], axis=1) / (2.0 * settings.spacing))).astype(np.int64)
    planes = int(height / settings.spacing) + 2
    points_per_axis = np.array([in_plane[0], in_plane[1], planes], dtype=np.int64)
    half_width = settings.stencil_half_width
    # With exactly 2*half_width points the neighbors +-half_width are the same point; their weights simply add (the
    # periodic operator is still the exact one for the sampled field), as in the Fortran code.
    if np.any(points_per_axis[:2] < 2 * half_width):
        raise ValueError(
            "the slab cell is too small for this stencil: each in-plane axis needs at least "
            f"{2 * half_width} grid points (expansion_order={settings.expansion_order}), got {points_per_axis[:2].tolist()}"
        )
    n0, n1, n2 = (int(value) for value in points_per_axis)
    mesh = np.meshgrid(*[np.arange(n) for n in (n0, n1, n2)], indexing="ij")
    integer_coordinates = np.column_stack([component.reshape(-1) for component in mesh])
    z_step = float(settings.spacing)
    z_origin = -0.5 * (n2 - 1) * z_step
    steps = np.zeros((3, 3))
    steps[:2] = lattice[:2] / points_per_axis[:2, None]
    steps[2, 2] = z_step
    coordinates = integer_coordinates.astype(np.float64) @ steps
    coordinates[:, 2] += z_origin
    lookup = np.arange(n0 * n1 * n2, dtype=np.int64).reshape(n0, n1, n2)
    return SlabRealSpaceGrid(
        cell=cell,
        settings=settings,
        points_per_axis=points_per_axis,
        integer_coordinates=integer_coordinates,
        coordinates=coordinates,
        lookup=lookup,
        z_step=z_step,
        z_origin=z_origin,
    )



__all__ = [
    "GeneralPeriodicGrid",
    "PeriodicRealSpaceGrid",
    "SlabRealSpaceGrid",
    "adjust_step",
    "build_fortran_periodic_grid",
    "build_general_periodic_grid",
    "build_periodic_grid",
    "build_slab_grid",
    "fortran_grid_shift",
    "fortran_points_per_axis",
    "periodic_grid_shift",
    "uses_fortran_routines",
    "uses_general_periodic_grid",
]
