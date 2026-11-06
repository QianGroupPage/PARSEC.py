"""Fast preparation of a periodic cell (Fortran-derived routines): ionic potential, projectors, atomic densities.

This is the module ``periodic_setup`` of the parity round, kept for the grids of the
Fortran-derived routines (``Use_Fortran_Routines``, :func:`parsec_python.Grid.pbc.uses_fortran_routines`).
:mod:`.periodic_setup` (Andrew Grieder's) is the default: its replacements send a grid of these routines here,
and :func:`grid_ball` and :func:`fft_local_ionic_potential` below send every other grid there, so either
installed function gives each grid its own routine.  :func:`install` installs the dispatching set of
:mod:`.periodic_setup`.

The readable definitions (:func:`parsec_python.V_ion.ewald.ewald_local_ionic_potential`,
:func:`~parsec_python.V_ion.ionic_potential.build_nonlocal_projectors`,
:func:`~parsec_python.V_ion.ionic_potential.superpose_atomic_density`) measure the distance of
*every* grid point to every atom image, so their cost is ``atoms x grid points x images`` and a
864-atom cell (630k points) takes hours.  Three replacements, installed when the acceleration package
is imported, give the same numbers:

``grid_ball``
    the grid points within a radius of a centre, found from integer index ranges instead of a scan of
    the grid.  The projectors and the atomic densities vanish beyond a support radius, so each atom
    touches only the few thousand points around it.  The rows, displacements and distances are the
    ones of the dense scan (the same expressions on the same coordinates).

``fft_local_ionic_potential``
    the Ewald split of the local potential with the reciprocal part done as one FFT.  The potential of
    an atom is ``V_a + 2 Z erf(eta r)/r`` (short ranged, summed over the few images that reach the
    cell) plus ``-2 Z erf(eta r)/r`` summed over the lattice, whose Fourier series is evaluated for
    all atoms at once from the structure factor.  The reference takes ``eta = sqrt(pi)/V^(1/3)``, which
    makes the real-space sum reach about three cell lengths; any ``eta`` gives the same potential up to
    the constant ``-(2 pi/V) Z_total/eta^2`` that the G = 0 term would carry, which is restored below.
    Here ``eta`` is the largest one whose reciprocal series ends below the Nyquist frequency of the
    grid, so the series is represented exactly on the grid and the short-ranged sum needs only a ball
    around every atom, as wide as the pseudopotential table or the Gaussian tail, whichever is larger.

``periodic_atomic_density``
    the image sum of :func:`~parsec_python.V_ion.ionic_potential.superpose_atomic_density`
    (installed with :func:`~parsec_python.V_ion.ionic_potential.set_atomic_density_provider`).

``_translation_map``
    the real-space terms of :func:`~parsec_python.V_ion.ewald.ewald_ion_ion_energy` (one array of
    atom pairs per lattice translation) on worker threads, handed back in the order of the
    translations, in which the energy adds them (:func:`~parsec_python.V_ion.ewald.set_translation_map`).

Image blocks.  Both radial sums reach the end of the radial table (``rs(ns-1)``, near 118 bohr for
the POTRE files), so a small cell has thousands of images of every atom inside the ball and nearly
every image box is the whole grid.  Visiting the images one by one (``_atom_balls``, ``grid_ball``)
then costs more in Python than in arithmetic.  The image boxes of an atom are products of index ranges
along the three axes, so the squared displacements are one-dimensional tables per axis and image;
the distances of a block of images (runs of z images of a few y images of one x image) come from
one broadcast sum, ``(dx^2 + dy^2) + dz^2``, the order in which ``np.linalg.norm`` adds the
components.  Rounding is monotone, so the smallest and the largest squares of an image's box bound
its distances exactly (``_bounds``): images whose box lies outside the ball are left out, an image
inside the ball has its whole box in it, and the masks below run only for blocks that need them.
Worker threads (``PARSEC_PERIODIC_SETUP_THREADS``; NumPy releases the interpreter lock) evaluate the
blocks ahead; the calling thread adds them into the field in the image order of the definitions, so
the result does not depend on the threads:

* the local potential repeats the flushes of the ball loop: the terms of the images between two
  flushes are added point by point in image order (``np.bincount``'s order) and their sum is added to
  the potential, the flushes falling after the same images as before (the counts of the points in
  every ball decide them); a term the block does not add is exactly 0.0 (a point outside the ball,
  or beyond the table where ``erf(eta r) = 1``: ``-2Z/r + 2Z/r``), and adding 0.0 changes no partial
  sum of ``np.bincount``, which starts at +0.0 and so never holds -0.0;
* the densities add every image's interpolated table into the points of its box in translation
  order, as ``density[rows] += values`` does, with 0.0 outside the ball.

Where ``erf(eta r)`` is 1.0 exactly (``eta r >= 6``, checked once per process on a dense sample),
``2 Z erf(eta r)/r`` is ``2 Z/r`` and the error function is not evaluated; every other term is
evaluated by the readable radial functions themselves (``ParsecPseudopotential.local_potential``,
``V_ion.ewald._erf_over_r``).  The potential and both densities are therefore bit-identical to the
ball loop and to the readable sums (``acceleration/tests/test_periodic_setup.py``).  Species with
``Cubic_Spline`` and grids whose coordinates are not separable keep the ball loop or the readable sum.

``PARSEC_PERIODIC_SETUP_THREADS`` sets the worker threads (default ``OMP_NUM_THREADS``, else up to 8).
``PARSEC_PERIODIC_SETUP_REACH`` chooses how far the radial sums reach.  ``physical`` (the default since
2026-10-30) sums each radial function only out to the radius where its terms stop being nonzero or
above round-off: the reaches of the device builder :mod:`.periodic_device` (``local_reach``: the erfc
tail or the end of the non-Coulomb part of ``r V``; ``density_reach``: 1e-16 of the density peak).
That drops the round-off terms that ``table`` keeps (about 6000 images per atom for an 8-atom silicon
cube; 90-99 % of the preparation time of an orthorhombic cell) and changes the last bits of the fields
(1e-13 Ry, 1e-16 electrons/bohr^3) and of the energies (1e-11 Ry).  ``table`` (the default up to
2026-10-30) sums to the end of the radial tables, bit for bit the readable definitions; the accelerated
driver's cache of preparations keys the two apart.  The local potential of a ``Cubic_Spline`` species
takes the physical reach under both rules (``_local_reach``: the spline's own tail near the end of a
table is no part of the potential).
"""

from __future__ import annotations

import math
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
from scipy.special import erf, erfcinv

from parsec_python.Grid.cell import is_diagonal_lattice
from parsec_python.Grid.pbc import PeriodicRealSpaceGrid, uses_fortran_routines
from parsec_python.Pseudopotential.radial_derivatives import valence_density_table
from parsec_python.V_ion import ewald as _ewald
from parsec_python.V_ion import ionic_potential as _ionic

_NYQUIST_FRACTION = 0.85  # the reciprocal series ends at this fraction of the grid's Nyquist frequency
_POINTS_PER_CALL = 1_000_000  # grid points evaluated together (the radial spline is built once per call)
_STRUCTURE_FACTOR_ELEMENTS = 4_000_000  # complex numbers of one structure-factor block
_BLOCK_ELEMENTS = 1 << 19  # grid points x images of one block of images
_ERF_ONE_FROM = 6.0  # erf(x) is 1.0 exactly from here on (scipy: from 5.92; checked by _erf_saturates)
_ERF_CHECKED_UP_TO = 1000.0

REACH_RULES = ("table", "physical")
#: The rule where ``PARSEC_PERIODIC_SETUP_REACH`` is unset or empty (``table`` up to 2026-10-30).
DEFAULT_REACH_RULE = "physical"


def _axis_spacing(grid: PeriodicRealSpaceGrid) -> tuple[np.ndarray, np.ndarray, float]:
    n = np.asarray(grid.points_per_axis, dtype=np.int64)
    sides = np.diag(np.asarray(grid.cell.lattice_vectors, dtype=float))
    # ``build_periodic_grid`` places point ``i`` at ``(i + shift) * (side / n)`` of the first axis
    return n, sides, float(sides[0] / n[0])


def _plain_shift(grid) -> np.ndarray:
    """The shift of the orthorhombic grid of these routines (``grid.shift``; zero on Andrew Grieder's grids, whose
    index boxes here have always taken the margin of one index for it)."""
    if uses_fortran_routines(grid):
        return np.asarray(grid.shift, dtype=float)
    return np.zeros(3)


def _index_range(centre: float, radius: float, spacing: float, count: int, shift: float = 0.0):
    """Inclusive grid index range around ``centre`` (one index of margin for round-off), or ``None``; the point
    ``i`` lies at ``(i + shift) * spacing`` (``x - 0.0`` is ``x``: the ranges of a zero shift are the earlier ones)."""
    low = max(math.ceil((centre - radius) / spacing - shift) - 1, 0)
    high = min(math.floor((centre + radius) / spacing - shift) + 1, count - 1)
    return (low, high) if low <= high else None


def _box_rows(n: np.ndarray, ranges) -> np.ndarray:
    (i0, i1), (j0, j1), (k0, k1) = ranges
    i = np.arange(i0, i1 + 1, dtype=np.int64)
    j = np.arange(j0, j1 + 1, dtype=np.int64)
    k = np.arange(k0, k1 + 1, dtype=np.int64)
    return ((i[:, None, None] * n[1] + j[None, :, None]) * n[2] + k[None, None, :]).reshape(-1)


def _default_routines_grid(grid) -> bool:
    """A periodic grid of the default routines (Andrew Grieder's), which :mod:`.periodic_setup` handles."""
    return isinstance(grid, PeriodicRealSpaceGrid) and not uses_fortran_routines(grid)


def _default_bulk_grid(grid) -> bool:
    """Andrew Grieder's grid of a bulk cell (not his slab): point ``(i, j, k)`` at ``(i + s) a_0/n_0 + ...``."""
    from parsec_python.Grid.pbc import SlabRealSpaceGrid

    return (
        isinstance(grid, PeriodicRealSpaceGrid)
        and not isinstance(grid, SlabRealSpaceGrid)
        and not getattr(grid, "is_general_periodic", False)
        and not getattr(grid, "is_confined_periodic", False)
        and not uses_fortran_routines(grid)
    )


def _plain_orthorhombic(grid) -> bool:
    """The orthorhombic grid of :func:`parsec_python.Grid.pbc.build_fortran_periodic_grid`: one spacing ``h``, the
    points ``(i + shift) h`` (the shift of PARSEC 1.5's parity round; zero before).

    Fortran's general grid (:class:`parsec_python.Grid.pbc.GeneralPeriodicGrid`: oblique lattice
    vectors, a step per lattice direction) and the wire/slab grid
    (:class:`parsec_python.Grid.confined.ConfinedPeriodicGrid`) are subclasses with other point
    positions, which the index boxes below do not describe: they keep the readable definitions.
    """
    return (
        isinstance(grid, PeriodicRealSpaceGrid)
        and not getattr(grid, "is_general_periodic", False)
        # the orthorhombic grid of the default routines is Andrew Grieder's (periodic_setup.py)
        and uses_fortran_routines(grid)
    )


def grid_ball(grid, centre: np.ndarray, radius: float):
    """``(rows, relative, distances)`` of the grid points within ``radius`` of ``centre``.

    The result of :func:`parsec_python.V_ion.ionic_potential.dense_support` for a periodic grid,
    from the index box around the centre.  Other grids use the dense scan.
    """
    if _default_routines_grid(grid):
        from . import periodic_setup as _default

        return _default.grid_ball(grid, centre, radius)
    if getattr(grid, "is_general_periodic", False) and not getattr(grid, "is_confined_periodic", False):
        # Fortran's general (oblique) grid: Andrew Grieder's index box, the rows of the dense scan in its order
        # (the readable sums of an oblique cell spent 34-300 s scanning the grid)
        from . import periodic_setup as _default

        return _default.lattice_ball(grid, centre, radius)
    if not _plain_orthorhombic(grid):
        return _ionic.dense_support(grid, centre, radius)
    n, _, spacing = _axis_spacing(grid)
    shift = _plain_shift(grid)
    ranges = []
    for axis in range(3):
        found = _index_range(float(centre[axis]), radius, spacing, int(n[axis]), float(shift[axis]))
        if found is None:
            empty = np.empty(0, dtype=np.int64)
            return empty, np.empty((0, 3)), np.empty(0)
        ranges.append(found)
    rows = _box_rows(n, ranges)
    relative = grid.coordinates[rows] - centre
    distances = np.linalg.norm(relative, axis=1)
    keep = distances <= radius
    return rows[keep], relative[keep], distances[keep]


def _atom_balls(grid, position: np.ndarray, radius: float):
    """Yield ``(rows, distances)`` of the grid points within ``radius`` of each periodic image of an atom.

    One item per image box, so a cell much smaller than ``radius`` (thousands of overlapping images) is
    never held in memory at once; rows repeat across images and the caller sums over them.
    """
    n, sides, spacing = _axis_spacing(grid)
    shift = _plain_shift(grid)
    axis_images = []
    for axis in range(3):
        reach = int(math.ceil(radius / sides[axis])) + 1
        found = []
        for image in range(-reach, reach + 1):
            centre = float(position[axis]) + image * sides[axis]
            index_range = _index_range(centre, radius, spacing, int(n[axis]), float(shift[axis]))
            if index_range is not None:
                found.append((centre, index_range))
        axis_images.append(found)
    for cx, rx in axis_images[0]:
        for cy, ry in axis_images[1]:
            for cz, rz in axis_images[2]:
                box = _box_rows(n, (rx, ry, rz))
                relative = grid.coordinates[box] - np.array([cx, cy, cz])
                distance = np.linalg.norm(relative, axis=1)
                keep = distance <= radius
                if np.any(keep):
                    yield box[keep], distance[keep]


# ---------------------------------------------------------------------------
# Image blocks (see the module docstring).


def setup_threads() -> int:
    """Worker threads of the image blocks: ``PARSEC_PERIODIC_SETUP_THREADS``, else ``OMP_NUM_THREADS``,
    else the CPUs up to 8.  The results do not depend on it."""
    for name in ("PARSEC_PERIODIC_SETUP_THREADS", "OMP_NUM_THREADS"):
        raw = os.environ.get(name, "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be a positive integer, got {raw!r}") from error
            if value < 1:
                raise ValueError(f"{name} must be a positive integer, got {raw!r}")
            return value
    return max(1, min(8, os.cpu_count() or 1))


def reach_rule() -> str:
    """``PARSEC_PERIODIC_SETUP_REACH``: ``physical`` (default) or ``table`` (the readable sums, bit for bit)."""
    value = (
        os.environ.get("PARSEC_PERIODIC_SETUP_REACH", DEFAULT_REACH_RULE).strip().lower()
        or DEFAULT_REACH_RULE
    )
    if value not in REACH_RULES:
        raise ValueError(
            f"PARSEC_PERIODIC_SETUP_REACH must be one of {', '.join(REACH_RULES)}, got {value!r}"
        )
    return value


_ERF_SATURATION: dict[str, bool] = {}


def _erf_saturates() -> bool:
    """Whether ``erf(x)`` is 1.0 exactly on ``[_ERF_ONE_FROM, _ERF_CHECKED_UP_TO]`` (dense sample)."""
    if "value" not in _ERF_SATURATION:
        samples = np.concatenate(
            (
                np.linspace(_ERF_ONE_FROM, _ERF_ONE_FROM + 2.0, 200_001),
                np.linspace(_ERF_ONE_FROM + 2.0, _ERF_CHECKED_UP_TO, 200_001),
            )
        )
        _ERF_SATURATION["value"] = bool(np.all(erf(samples) == 1.0))
    return _ERF_SATURATION["value"]


def _smallest_with_product_at_least(scale: float, threshold: float) -> float:
    """The smallest double ``r`` with ``scale * r >= threshold`` in floating point (``scale > 0``)."""
    r = threshold / scale
    while scale * r < threshold:
        r = float(np.nextafter(r, math.inf))
    while scale * float(np.nextafter(r, -math.inf)) >= threshold:
        r = float(np.nextafter(r, -math.inf))
    return r


def _axis_coordinates(grid):
    """The coordinate of every index along each axis, or ``None`` if the points are no product grid.

    ``grid.coordinates[(i n1 + j) n2 + k] == (x[i], y[j], z[k])`` exactly, which the blocks need.
    """
    n = tuple(int(value) for value in grid.points_per_axis)
    coordinates = np.asarray(grid.coordinates)
    if coordinates.shape != (n[0] * n[1] * n[2], 3):
        return None
    cube = coordinates.reshape(n + (3,))
    axes = (cube[:, 0, 0, 0].copy(), cube[0, :, 0, 1].copy(), cube[0, 0, :, 2].copy())
    for axis, line in enumerate(axes):
        shape = [1, 1, 1]
        shape[axis] = n[axis]
        if not np.array_equal(cube[..., axis], np.broadcast_to(line.reshape(shape), n)):
            return None
    return axes


@dataclass(frozen=True)
class _AxisImages:
    """The images of one atom along one axis whose index range is not empty, in image order.

    ``squares[m]`` is ``(x[i] - c_m)^2`` for ``i`` in ``low[m]..high[m]``: the first operations of
    ``np.linalg.norm(grid.coordinates[box] - centre, axis=1)``.
    """

    low: tuple[int, ...]
    high: tuple[int, ...]
    squares: tuple[np.ndarray, ...]
    lowest: np.ndarray  # the smallest and the largest square of every image
    highest: np.ndarray

    def __len__(self) -> int:
        return len(self.low)

    def span(self, members: Sequence[int]) -> int:
        return max(self.high[m] for m in members) - min(self.low[m] for m in members) + 1


def _axis_images(
    line: np.ndarray, position: float, side, radius: float, spacing: float, count: int, shift: float = 0.0
) -> _AxisImages:
    """The images of :func:`_atom_balls` (and of the translations of ``superpose_atomic_density``,
    ``position + m * side``) along one axis: ``m`` up to ``ceil(radius / side) + 1``, the index ranges
    of ``_index_range`` (``shift``: that of the grid's points)."""
    reach = int(math.ceil(radius / side)) + 1
    low, high, squares = [], [], []
    for image in range(-reach, reach + 1):
        centre = position + image * side
        index_range = _index_range(float(centre), radius, spacing, count, shift)
        if index_range is None:
            continue
        displacement = line[index_range[0] : index_range[1] + 1] - centre
        low.append(index_range[0])
        high.append(index_range[1])
        squares.append(displacement * displacement)
    return _AxisImages(
        tuple(low),
        tuple(high),
        tuple(squares),
        np.array([float(np.min(square)) for square in squares]),
        np.array([float(np.max(square)) for square in squares]),
    )


def _bounds(images: Sequence[_AxisImages], block) -> tuple[np.ndarray, np.ndarray]:
    """Bounds of the distances of every image of the block, ``sqrt((dx^2 + dy^2) + dz^2)`` of the
    smallest and of the largest squares of its box: rounding is monotone, so every distance the
    block computes for the image lies between them."""
    x_images, y_images, z_images = images
    ys, zs = np.asarray(block.ys), np.asarray(block.zs)
    lower = np.sqrt(
        (x_images.lowest[block.x] + y_images.lowest[ys])[:, None] + z_images.lowest[zs][None, :]
    ).reshape(-1)
    upper = np.sqrt(
        (x_images.highest[block.x] + y_images.highest[ys])[:, None] + z_images.highest[zs][None, :]
    ).reshape(-1)
    return lower, upper


def _padded(images: _AxisImages, members: Sequence[int]) -> tuple[int, np.ndarray]:
    """``(first index, table)`` of the union range of ``members``, ``inf`` outside each image's range."""
    first = min(images.low[m] for m in members)
    last = max(images.high[m] for m in members)
    table = np.full((len(members), last - first + 1), np.inf)
    for row, m in enumerate(members):
        table[row, images.low[m] - first : images.high[m] - first + 1] = images.squares[m]
    return first, table


@dataclass(frozen=True)
class _Block:
    """Images ``(x, y, z)`` of one atom for ``y`` in ``ys``, ``z`` in ``zs``: ``y`` outer, ``z`` inner,
    the image order of the definitions.  Both are runs of consecutive images."""

    atom: int
    x: int
    ys: tuple[int, ...]
    zs: tuple[int, ...]

    @property
    def images(self) -> int:
        return len(self.ys) * len(self.zs)


def _atom_blocks(
    atom: int, images: Sequence[_AxisImages], radius: float, limit: int
) -> Iterator[_Block]:
    """The blocks of an atom in image order (about ``limit`` elements each, or one image), without
    the images whose box lies outside the ball.

    An image whose lower distance bound (:func:`_bounds`) exceeds ``radius`` has no point in the
    ball (no rows in the ball loop).  The images a block keeps beyond that bound (to stay a product
    of runs) have no point in it either.
    """
    x_images, y_images, z_images = images
    if not (len(x_images) and len(y_images) and len(z_images)):
        return
    z_span = z_images.span(range(len(z_images)))
    for x in range(len(x_images)):
        x_span = x_images.high[x] - x_images.low[x] + 1
        reaching = []  # (y, first z, last z) of the y images whose ball meets some z image
        for y in range(len(y_images)):
            planar = x_images.lowest[x] + y_images.lowest[y]
            zs = np.flatnonzero(np.sqrt(planar + z_images.lowest) <= radius)
            if zs.size:
                reaching.append((y, int(zs[0]), int(zs[-1])))

        def size(members) -> int:
            ys = range(members[0][0], members[-1][0] + 1)
            first = min(member[1] for member in members)
            last = max(member[2] for member in members)
            return len(ys) * (last - first + 1) * x_span * y_images.span(ys) * z_span

        def emit(members):
            ys = tuple(range(members[0][0], members[-1][0] + 1))
            first = min(member[1] for member in members)
            last = max(member[2] for member in members)
            if len(ys) > 1 or size(members) <= limit:
                yield _Block(atom, x, ys, tuple(range(first, last + 1)))
                return
            step = max(1, limit // (x_span * y_images.span(ys) * z_span))
            for start in range(first, last + 1, step):
                yield _Block(atom, x, ys, tuple(range(start, min(start + step, last + 1))))

        group: list[tuple[int, int, int]] = []
        for member in reaching:
            if group and (member[0] != group[-1][0] + 1 or size([*group, member]) > limit):
                yield from emit(group)
                group = []
            group.append(member)
        if group:
            yield from emit(group)


def _block_distances(
    images: Sequence[_AxisImages], block: _Block, window=None, *, squared: bool = False
):
    """Distances of the block's images over their union box (``inf`` outside each image's box), as
    ``(images, nx, ny, nz)``, with the box and whether any entry is such padding; ``window = (first,
    last)`` keeps the x indices in it (``None`` if the box has none there).  ``squared`` returns the
    squared distances."""
    x_images, y_images, z_images = images
    x_low, x_high = x_images.low[block.x], x_images.high[block.x]
    if window is not None:
        x_low, x_high = max(x_low, window[0]), min(x_high, window[1])
        if x_low > x_high:
            return None
    start = x_low - x_images.low[block.x]
    x_square = x_images.squares[block.x][start : start + x_high - x_low + 1]
    y_first, y_table = _padded(y_images, block.ys)
    z_first, z_table = _padded(z_images, block.zs)
    # (dx^2 + dy^2) + dz^2: np.linalg.norm's order
    planar = x_square[None, :, None] + y_table[:, None, :]
    distance = planar[:, None, :, :, None] + z_table[None, :, None, None, :]
    if not squared:
        np.sqrt(distance, out=distance)
    shape = (block.images,) + distance.shape[2:]
    y_range = (y_first, y_first + y_table.shape[1] - 1)
    z_range = (z_first, z_first + z_table.shape[1] - 1)
    padded = not (np.all(np.isfinite(y_table)) and np.all(np.isfinite(z_table)))
    return distance.reshape(shape), (x_low, x_high), y_range, z_range, padded


def _ordered_results(work: Callable, items: Iterable, threads: int) -> Iterator:
    """``work(item)`` for every item, in order, by ``threads`` workers at most ``threads + 2`` items ahead
    (the results in flight bound the memory)."""
    if threads <= 1:
        for item in items:
            yield work(item)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="parsec-periodic-setup") as pool:
        window = deque(pool.submit(work, item) for item in islice(iterator, threads + 2))
        while window:
            result = window.popleft().result()
            for item in islice(iterator, 1):
                window.append(pool.submit(work, item))
            yield result


@dataclass
class _BlockResult:
    """Terms of a block on the union box ``x_range x y_range x z_range``, one slab per image."""

    block: _Block
    counts: np.ndarray  # points in the ball of every image (ys x zs, flattened)
    values: np.ndarray  # (images, nx, ny, nz); 0.0 where an image adds nothing
    x_range: tuple[int, int]
    y_range: tuple[int, int]
    z_range: tuple[int, int]


def _view(field: np.ndarray, n: tuple[int, int, int], result: _BlockResult) -> np.ndarray:
    (x0, x1), (y0, y1), (z0, z1) = result.x_range, result.y_range, result.z_range
    return field.reshape(n)[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1]


def _image_counts(
    images: Sequence[_AxisImages],
    block: _Block,
    distance: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    radius: float,
    inside,
) -> np.ndarray:
    """Points of every image of the block in its ball (``inside(distances)``, e.g. ``r <= radius``):
    the whole box where the upper bound is inside, none where the lower bound is outside (both
    decided by ``inside`` itself), else counted on the distances."""
    x_images, y_images, z_images = images
    x_span = x_images.high[block.x] - x_images.low[block.x] + 1
    boxes = (
        x_span
        * np.array([y_images.high[y] - y_images.low[y] + 1 for y in block.ys])[:, None]
        * np.array([z_images.high[z] - z_images.low[z] + 1 for z in block.zs])[None, :]
    ).reshape(-1)
    all_inside = inside(upper)
    counts = np.where(all_inside, boxes, 0).astype(np.int64)
    for image in np.flatnonzero(~all_inside & inside(lower)):
        counts[image] = np.count_nonzero(inside(distance[image]))
    return counts


# ---------------------------------------------------------------------------
# The short-ranged sum of the local potential.


@dataclass(frozen=True)
class _ShortRangeSpecies:
    """What the terms ``V_a(r) + 2 Z erf(eta r)/r`` of one species need (linear ``r V``)."""

    potential: object
    local_l: int
    padding: int
    eta: float
    reach: float
    radii: np.ndarray
    weighted: np.ndarray  # r V_a, the table ``local_potential`` interpolates
    two_z: float
    generic_from: float  # from here to ``cutoff``: interp(r V)/r + 2Z/r (erf = 1, past the first sample)
    zero_from: float  # from here on the term is 0.0 exactly (beyond the table and erf = 1)
    cutoff: float

    def values(self, distance: np.ndarray, lower: float, upper: float) -> np.ndarray:
        """The terms on ``distance`` (modified), 0.0 beyond ``reach`` (outside the ball);
        ``lower``/``upper`` bound the finite distances (``inf`` pads the box)."""
        if self.zero_from <= self.reach:
            if upper >= self.zero_from:
                np.copyto(distance, np.inf, where=distance >= self.zero_from)
        elif upper > self.reach:
            np.copyto(distance, np.inf, where=distance > self.reach)
        # at r = inf the expression below is interp/inf + 2Z/inf = -0.0 + 0.0 = +0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.interp(distance, self.radii, self.weighted)
            np.divide(values, distance, out=values)
            values += self.two_z / distance
        if lower >= self.generic_from:
            return values
        special = np.flatnonzero(distance < self.generic_from)
        if special.size:
            radius = distance.reshape(-1)[special]
            values.reshape(-1)[special] = self.potential.local_potential(
                radius,
                self.local_l,
                use_spline=False,
                spline_padding_width=self.padding,
            ) + _ewald._erf_over_r(self.potential.ionic_charge, radius, self.eta)
        return values


def _short_range_species(potential, specification, padding, eta, reach) -> _ShortRangeSpecies:
    radii = np.asarray(potential.radii)
    cutoff = float(potential.interpolation_cutoff)
    erf_one = (
        _smallest_with_product_at_least(eta, _ERF_ONE_FROM)
        if _erf_saturates() and eta * reach <= _ERF_CHECKED_UP_TO
        else math.inf
    )
    # local_potential's interpolation branch: inside the table and past its first sample
    generic_from = max(erf_one, float(np.nextafter(float(radii[0]), math.inf)))
    return _ShortRangeSpecies(
        potential=potential,
        local_l=specification.local_angular_momentum,
        padding=padding,
        eta=eta,
        reach=reach,
        radii=radii,
        weighted=potential.radii * potential.channel_potentials[specification.local_angular_momentum],
        two_z=2.0 * potential.ionic_charge,
        generic_from=generic_from,
        zero_from=max(cutoff, generic_from),
        cutoff=cutoff,
    )


def _short_range_by_balls(
    total, grid, atoms, positions, potentials, specifications, eta, reach_of
) -> None:
    """``total += sum_a sum_images V_a + 2 Z erf(eta r)/r`` over the balls of :func:`_atom_balls`."""
    pending_rows: list[np.ndarray] = []
    pending_distances: list[np.ndarray] = []
    pending_points = 0
    batch_symbol = None

    def flush() -> None:
        nonlocal pending_rows, pending_distances, pending_points
        if not pending_rows:
            return
        rows = np.concatenate(pending_rows)
        distances = np.concatenate(pending_distances)
        potential = potentials[batch_symbol]
        specification = specifications[batch_symbol]
        values = potential.local_potential(
            distances,
            specification.local_angular_momentum,
            use_spline=specification.use_spline,
            spline_padding_width=grid.settings.stencil_half_width,
        ) + _ewald._erf_over_r(potential.ionic_charge, distances, eta)
        total_view[:] += np.bincount(rows, weights=values, minlength=total_view.size)
        pending_rows, pending_distances, pending_points = [], [], 0

    total_view = total
    for atom, position in zip(atoms, positions):
        if atom.symbol != batch_symbol:
            flush()
            batch_symbol = atom.symbol
        for rows, distances in _atom_balls(grid, position, reach_of(atom.symbol)):
            pending_rows.append(rows)
            pending_distances.append(distances)
            pending_points += rows.size
            if pending_points >= _POINTS_PER_CALL:
                flush()
    flush()


def _short_range_by_blocks(
    total, grid, atoms, positions, potentials, specifications, eta, reach_of, axes
) -> None:
    """The sum of :func:`_short_range_by_balls`, bit for bit, from image blocks (module docstring).

    Worker threads compute the terms of the blocks; the calling thread adds them, image after image,
    into the pending sum of the ball loop and flushes it where the ball loop does: when an atom of
    another species begins, after the image whose points bring the pending ones to
    ``_POINTS_PER_CALL`` (an image without points in its ball adds no rows), and at the end.
    """
    n_array, sides, spacing = _axis_spacing(grid)
    n = tuple(int(value) for value in n_array)
    shift = _plain_shift(grid)
    padding = grid.settings.stencil_half_width
    species = {
        symbol: _short_range_species(
            potentials[symbol], specifications[symbol], padding, eta, reach_of(symbol)
        )
        for symbol in dict.fromkeys(atom.symbol for atom in atoms)
    }
    atom_images = [
        [
            _axis_images(
                axes[axis],
                float(position[axis]),
                sides[axis],
                species[atom.symbol].reach,
                spacing,
                n[axis],
                float(shift[axis]),
            )
            for axis in range(3)
        ]
        for atom, position in zip(atoms, positions)
    ]

    def work(block: _Block) -> _BlockResult:
        terms = species[atoms[block.atom].symbol]
        images = atom_images[block.atom]
        distance, x_range, y_range, z_range, _ = _block_distances(images, block)
        lower, upper = _bounds(images, block)
        counts = _image_counts(
            images, block, distance, lower, upper, terms.reach, lambda r: r <= terms.reach
        )
        values = terms.values(distance, float(lower.min()), float(upper.max()))
        return _BlockResult(block, counts, values, x_range, y_range, z_range)

    def blocks() -> Iterator[_Block]:
        for index in range(len(atoms)):
            yield from _atom_blocks(
                index, atom_images[index], species[atoms[index].symbol].reach, _BLOCK_ELEMENTS
            )

    pending = np.zeros_like(total)  # the np.bincount of the ball loop between two flushes
    pending_points = 0
    batch_symbol = None
    next_atom = 0

    def flush() -> None:
        nonlocal pending_points
        if pending_points == 0:
            return
        total[:] += pending
        pending.fill(0.0)
        pending_points = 0

    def start_atoms(up_to: int) -> None:
        # the ball loop flushes when an atom of another species begins
        nonlocal next_atom, batch_symbol
        while next_atom <= up_to:
            if atoms[next_atom].symbol != batch_symbol:
                flush()
                batch_symbol = atoms[next_atom].symbol
            next_atom += 1

    for result in _ordered_results(work, blocks(), setup_threads()):
        start_atoms(result.block.atom)
        target = _view(pending, n, result)
        for image, count in enumerate(result.counts.tolist()):
            if count == 0:
                continue  # the ball loop yields no rows for this image
            target += result.values[image]
            pending_points += count
            if pending_points >= _POINTS_PER_CALL:
                flush()
    start_atoms(len(atoms) - 1)
    flush()


def _local_reach(potential, specification, eta: float, erfc_cutoff: float, rule: str) -> float:
    """Radius of the ball of ``V_a + 2 Z erf(eta r)/r`` around every image of one species (bohr).

    ``physical``: the erfc tail or the Coulomb onset of the table (``coulomb_radius``), whichever is
    larger.  ``table``: the end of the table for a species interpolated linearly (``r V`` is ``-2 Z``
    there to the last bit, so the terms past the onset are round-off).  A ``Cubic_Spline`` species
    takes the physical reach under both rules: the spline of ``V`` (not ``r V``) leaves ``-2 Z/r``
    between the samples, by 1e-9 Ry up to ~80 bohr and by up to 3e-5 Ry over the last ~40 bohr of a
    POTRE table, and a ball out to the table end adds that tail of every image (-3.0 Ry bohr^3 per Si
    or Ge atom, -0.47 per Na): -0.71 Ry for an 8-atom silicon cube, where the direct sum
    (:func:`parsec_python.V_ion.ewald.ewald_local_ionic_potential`, which stops at its erfc reach)
    and the physical reach agree to 6e-6 Ry.
    """
    if rule == "physical" or specification.use_spline:
        from .periodic_device import coulomb_radius

        return max(erfc_cutoff, coulomb_radius(potential, specification.local_angular_momentum))
    # beyond the table of the pseudopotential V_a is exactly -2 Z/r, but inside it V_a differs from
    # the Coulomb tail, so the ball reaches the larger of the two radii
    return max(erfc_cutoff, float(potential.interpolation_cutoff))


def _structure_factor_series(
    grid, positions: np.ndarray, charges: np.ndarray, eta: float, tolerance: float
) -> np.ndarray:
    """``sum_G c_G exp(iG.r)`` on the grid for ``c_G = -(8 pi/V) w(G) sum_a Z_a exp(-iG.r_a)``,
    ``w = exp(-G^2/4 eta^2)/G^2``, G != 0, evaluated by one FFT."""
    n, sides, _ = _axis_spacing(grid)
    volume = float(np.prod(sides))
    cutoff = 2.0 * eta * math.sqrt(-math.log(tolerance))
    steps = 2.0 * np.pi / sides
    half = np.minimum(np.floor(cutoff / steps).astype(np.int64), (n - 1) // 2)
    frequencies = [np.arange(-m, m + 1, dtype=np.int64) for m in half]
    g_axes = [steps[axis] * frequencies[axis] for axis in range(3)]
    g_squared = (
        g_axes[0][:, None, None] ** 2 + g_axes[1][None, :, None] ** 2 + g_axes[2][None, None, :] ** 2
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.exp(-g_squared / (4.0 * eta**2)) / g_squared
    weight[g_squared == 0.0] = 0.0
    weight[g_squared > cutoff**2] = 0.0

    # S(G) = sum_a Z_a exp(-i G.r_a) = sum_a Z_a px[a,x] py[a,y] pz[a,z]: one matrix product per block of atoms
    shape = tuple(int(m) for m in 2 * half + 1)
    structure = np.zeros(shape, dtype=np.complex128)
    block = max(1, _STRUCTURE_FACTOR_ELEMENTS // (shape[1] * shape[2]))
    for start in range(0, len(positions), block):
        stop = min(start + block, len(positions))
        phases = [
            np.exp(-1j * np.outer(positions[start:stop, axis], g_axes[axis])) for axis in range(3)
        ]
        yz = (phases[1][:, :, None] * phases[2][:, None, :]).reshape(stop - start, -1)
        structure += ((charges[start:stop, None] * phases[0]).T @ yz).reshape(shape)

    coefficients = (-8.0 * np.pi / volume) * weight * structure
    # the points are at (j + s) h: exp(i G.(j + s) h) = exp(2 pi i m j/n) exp(2 pi i m s/n) (no factor for s = 0)
    shifts = _plain_shift(grid)
    if np.any(shifts != 0.0):
        coefficients = coefficients * np.exp(
            2.0j
            * np.pi
            * (
                frequencies[0][:, None, None] * (shifts[0] / n[0])
                + frequencies[1][None, :, None] * (shifts[1] / n[1])
                + frequencies[2][None, None, :] * (shifts[2] / n[2])
            )
        )
    spectrum = np.zeros(tuple(int(v) for v in n), dtype=np.complex128)
    index = [np.mod(frequencies[axis], n[axis]) for axis in range(3)]
    spectrum[np.ix_(index[0], index[1], index[2])] = coefficients
    # ifftn divides by the number of points: the series is a plain sum over G
    return (np.fft.ifftn(spectrum) * spectrum.size).real.reshape(-1)


def _fft_handles(grid, lattice_vectors) -> bool:
    """Whether :func:`fft_local_ionic_potential` returns the potential (the converged Ewald sum)."""
    if not _plain_orthorhombic(grid):
        return False
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    off_diagonal = lattice_vectors - np.diag(np.diag(lattice_vectors))
    return bool(np.allclose(off_diagonal, 0.0, atol=1.0e-10))


def fft_local_ionic_potential(
    grid, atoms: Sequence, potentials, specifications, lattice_vectors, eta, tolerance
):
    """The local ionic potential of :func:`~parsec_python.V_ion.ewald.ewald_local_ionic_potential`."""
    if _default_routines_grid(grid):
        from . import periodic_setup as _default

        return _default.fft_local_ionic_potential(
            grid, atoms, potentials, specifications, lattice_vectors, eta, tolerance
        )
    if not _fft_handles(grid, lattice_vectors):
        return None
    n, sides, spacing = _axis_spacing(grid)
    volume = float(np.prod(sides))
    reference_eta, _, _ = _ewald._ewald_cutoffs(volume, eta, tolerance)

    nyquist = math.pi / float(np.max(sides / n))
    fast_eta = _NYQUIST_FRACTION * nyquist / (2.0 * math.sqrt(-math.log(tolerance)))
    erfc_cutoff = float(erfcinv(tolerance) / fast_eta)

    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)

    total = _structure_factor_series(grid, positions, charges, fast_eta, tolerance)

    # Short-ranged part: V_a + 2 Z erf(eta r)/r over the ball of every image
    rule = reach_rule()
    reaches = {
        symbol: _local_reach(potentials[symbol], specifications[symbol], fast_eta, erfc_cutoff, rule)
        for symbol in dict.fromkeys(atom.symbol for atom in atoms)
    }
    axes = _axis_coordinates(grid)
    if axes is None or any(specifications[symbol].use_spline for symbol in reaches):
        _short_range_by_balls(
            total, grid, atoms, positions, potentials, specifications, fast_eta, reaches.__getitem__
        )
    else:
        _short_range_by_blocks(
            total, grid, atoms, positions, potentials, specifications, fast_eta, reaches.__getitem__, axes
        )

    # The reference carries the G = 0 term of its own eta: -(2 pi/V) Z_total/eta^2 per unit of charge
    total += -(2.0 * np.pi * float(np.sum(charges)) / volume) * (
        1.0 / reference_eta**2 - 1.0 / fast_eta**2
    )
    return total


# ---------------------------------------------------------------------------
# Superposed atomic densities.


def _density_reach(potential, table: np.ndarray, rule: str) -> float:
    if rule == "physical":
        from .periodic_device import density_reach

        return density_reach(potential, table) if np.any(table) else 0.0
    return float(potential.interpolation_cutoff)


def periodic_atomic_density(
    grid, atoms: Sequence, potentials, specifications, lattice_vectors, core: bool, *, reaches=None
):
    """The density of :func:`~parsec_python.V_ion.ionic_potential.superpose_atomic_density`, or ``None``.

    For a cell on the orthorhombic grid of :func:`parsec_python.Grid.pbc.build_fortran_periodic_grid` (with its
    shift): every translation ``m * side`` with ``|m| <= ceil(cutoff / side) + 1``, the points
    of its ball (``grid_ball``, distance ``<= rs(ns-1)``) inside the table (``< rs(ns-1)``), the table
    interpolated linearly with its first value continued inwards, added in translation order.  A
    ``Cubic_Spline`` core density, another grid or lattice return ``None`` (the readable sum runs).

    ``reaches`` (the radius of each species): Andrew Grieder's bulk grid of a cell with diagonal lattice
    vectors (shifted by half a step, a step per axis) and the support radii of his readable sum
    (``atomic_density_support_radius``) - the same images, distances and order, so the same density.
    """
    if lattice_vectors is None:
        return None
    if reaches is None and not _plain_orthorhombic(grid):
        return None
    if reaches is not None and not _default_bulk_grid(grid):
        return None
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    side_lengths = np.diag(lattice_vectors)
    if not is_diagonal_lattice(lattice_vectors) or np.any(side_lengths <= 0.0):
        return None  # superpose_atomic_density then takes the translations of a general lattice
    contributing = [
        atom
        for atom in atoms
        if not core or potentials[atom.symbol].has_nonlinear_core_correction
    ]
    if core and any(specifications[atom.symbol].use_spline for atom in contributing):
        return None
    axes = _axis_coordinates(grid)
    if axes is None:
        return None
    n_array, _, spacing = _axis_spacing(grid)
    n = tuple(int(value) for value in n_array)
    # one step for every axis on the grid of these routines; Andrew Grieder's takes a step per axis
    steps = [spacing] * 3 if reaches is None else [float(side_lengths[axis] / n[axis]) for axis in range(3)]
    rule = reach_rule()

    tables: dict[str, tuple[np.ndarray, float]] = {}
    for atom in contributing:
        if atom.symbol in tables:
            continue
        potential = potentials[atom.symbol]
        if core:
            table = np.asarray(potential.core_density, dtype=float)
        else:
            table = valence_density_table(
                potential, read_from_file=specifications[atom.symbol].read_valence_density
            )
        reach = _density_reach(potential, table, rule) if reaches is None else float(reaches[atom.symbol])
        tables[atom.symbol] = (table, reach)

    # the shift of this module's grid (Andrew Grieder's keeps the margin of one index for his)
    shift = _plain_shift(grid)
    atom_images = []
    for atom in contributing:
        potential = potentials[atom.symbol]
        position = np.asarray(atom.position, dtype=np.float64)
        radius = tables[atom.symbol][1]
        atom_images.append(
            [
                _axis_images(
                    axes[axis], position[axis], side_lengths[axis], radius, steps[axis], n[axis], float(shift[axis])
                )
                for axis in range(3)
            ]
        )

    def work(block: _Block) -> _BlockResult:
        symbol = contributing[block.atom].symbol
        table, radius = tables[symbol]
        images = atom_images[block.atom]
        distance, x_range, y_range, z_range, padded = _block_distances(images, block)
        lower, upper = _bounds(images, block)
        # grid_ball keeps r <= radius, superpose_atomic_density then r < rs(ns-1); inf pads the box
        counts = _image_counts(images, block, distance, lower, upper, radius, lambda r: r < radius)
        values = np.interp(distance, potentials[symbol].radii, table, left=table[0])
        if padded or float(upper.max()) >= radius:
            np.copyto(values, 0.0, where=distance >= radius)
        return _BlockResult(block, counts, values, x_range, y_range, z_range)

    def blocks() -> Iterator[_Block]:
        for index in range(len(contributing)):
            yield from _atom_blocks(
                index, atom_images[index], tables[contributing[index].symbol][1], _BLOCK_ELEMENTS
            )

    density = np.zeros(grid.size, dtype=float)
    for result in _ordered_results(work, blocks(), setup_threads()):
        target = _view(density, n, result)
        for image, count in enumerate(result.counts.tolist()):
            if count:
                target += result.values[image]
    return density

def _density_provider(grid, atoms, potentials, specifications, lattice_vectors, core):
    if not uses_fortran_routines(grid):
        # Andrew Grieder's bulk grid of a cell with diagonal lattice vectors: his sum, from the index boxes
        # (the support radius of each species); other grids of his keep his readable sum (None)
        if not _default_bulk_grid(grid):
            return None
        reaches = {
            atom.symbol: _ionic.atomic_density_support_radius(
                potentials[atom.symbol], specifications[atom.symbol], core=core
            )
            for atom in atoms
            if not core or potentials[atom.symbol].has_nonlinear_core_correction
        }
        return periodic_atomic_density(
            grid, atoms, potentials, specifications, lattice_vectors, core, reaches=reaches
        )
    return periodic_atomic_density(grid, atoms, potentials, specifications, lattice_vectors, core)


# ---------------------------------------------------------------------------
# The real-space terms of the Ewald ion-ion energy.

_ION_ION_MEMORY = 1 << 31  # bytes the terms in flight may hold (about 48 bytes per atom pair each)


def _translation_map(function, translations, pairs: int):
    """The real-space terms of :func:`parsec_python.V_ion.ewald.ewald_ion_ion_energy` on worker
    threads, in the order of the translations (the energy adds them in that order: same bits)."""
    threads = min(setup_threads(), max(1, _ION_ION_MEMORY // (48 * max(1, pairs))))
    return _ordered_results(function, translations, threads)


def install() -> None:
    """Install the replacements of both sets of routines (:func:`.periodic_setup.install`).

    The set of this module alone (with ``converged_for=_fft_handles``, the atomic density and the threads of
    the Ewald terms) is what :func:`.periodic_setup.install` installs for the grids of these routines.
    """
    from . import periodic_setup as _default

    _default.install()


def uninstall() -> None:
    from . import periodic_setup as _default

    _default.uninstall()


__all__ = [
    "DEFAULT_REACH_RULE",
    "REACH_RULES",
    "fft_local_ionic_potential",
    "grid_ball",
    "install",
    "periodic_atomic_density",
    "reach_rule",
    "setup_threads",
    "uninstall",
]
