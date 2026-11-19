"""The nonlocal projectors of a periodic cell, all atoms at once.

:func:`parsec_python.V_ion.ionic_potential.build_nonlocal_projectors` visits the atoms and their images one by one:
for every image it finds the grid points within the support radius (:func:`.periodic_setup.grid_ball`), evaluates
the radial table and the real harmonics on them and appends one piece per harmonic, so a cell of many atoms spends
its time in the calls of Python (diamond of 32,768 atoms: 23 s on every rank).  :func:`batched_nonlocal_projectors`
does the same per block of atoms: the reaching images of all of them, the index boxes of all images at once, the
radial tables and harmonics of all their points in one evaluation each, and the columns written directly in CSC.

Every value is the same expression on the same numbers (the displacement ``grid.coordinates[row] - (position +
translation)``, its norm, the radial table at ``max(r, r_1)``, the harmonics of the displacement, ``sqrt(dV)``
times both), evaluated element by element, so the matrix is the reference's array for array: the same columns and
labels in the same order, the rows of a column ascending, the same values.  The reference sums an entry that two
images of one atom put on the same row (COO to CSC); a cell less than twice the support radius wide, where that can
happen, keeps the reference, as do the double grid, wires and slabs, oblique cells, the Fortran-derived
routines and potentials other than :class:`~parsec_python.Pseudopotential.ParsecPseudopotential`.  A k-point takes
the reference's Bloch phase ``exp(-ik.(r - T))`` of every entry (the same expression on the same numbers; the
matrix product of the phase may round differently with another BLAS: the tests allow 1e-15 of the largest).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid.cell import is_diagonal_lattice
from parsec_python.Grid.pbc import PeriodicRealSpaceGrid, uses_fortran_routines
from parsec_python.Pseudopotential import ParsecPseudopotential, ParsecRadialSpline
from parsec_python.V_ion import ionic_potential as _ionic

from .periodic_setup_fortran import setup_threads

_ATOMS_PER_BLOCK = 1024
_CANDIDATES_PER_PASS = 2_000_000  # index-box points tested together


def _row_major_box(grid) -> tuple[int, int, int] | None:
    """The points per axis of a bulk orthorhombic cell of the default routines whose rows are its box in row-major
    order (row ``(i n_1 + j) n_2 + k``, the order of :func:`.periodic_setup.lattice_ball`), else ``None``.

    A cell with its sides along the axes whose steps differ (the grid of the parser for most orthorhombic and
    tetragonal cells) has the same separable points ``(i + s) h_a`` along every axis, with its own ``h_a``: it is taken
    under ``PARSEC_PERIODIC_UNEQUAL_STEPS=auto`` (``off``: the reference loop, as before)."""

    if type(grid) is not PeriodicRealSpaceGrid or uses_fortran_routines(grid):
        return None
    if not grid.is_orthorhombic:
        from ..Symmetry.axis_reflection import periodic_unequal_steps_setting, unequal_step_grid

        if not (unequal_step_grid(grid) and periodic_unequal_steps_setting() == "auto"):
            return None
    if int(getattr(grid.settings, "double_grid_order", 1)) > 1 or getattr(grid, "is_confined_periodic", False):
        return None
    shape = tuple(int(value) for value in grid.points_per_axis)
    lookup = np.asarray(grid.lookup)
    if lookup.shape != shape or int(grid.size) != int(np.prod(shape)):
        return None
    flat = lookup.reshape(-1)
    if int(flat[0]) != 0 or int(flat[-1]) != int(grid.size) - 1:
        return None
    return shape


def _channels(potential, specification, width: int):
    """``[(l, table, sign, spline or None)]`` of the separable channels, in the reference's order."""

    local_l = specification.local_angular_momentum
    channels = []
    for angular_momentum in sorted(potential.radial_wavefunctions):
        if angular_momentum == local_l:
            continue
        radial_grid, denominator_sign = potential.radial_projector(angular_momentum, local_l)
        spline = (
            ParsecRadialSpline.from_positive_grid(potential.radii, radial_grid, width)
            if specification.use_spline
            else None
        )
        channels.append((angular_momentum, radial_grid, denominator_sign, spline))
    return channels


def _image_points(axes, shape, coordinates, centres, radius):
    """``(pair, rows, relative, distances)`` of the grid points within ``radius`` of each centre.

    The candidates are the index box of each centre (the interval of :func:`.periodic_setup._index_range` with its
    margin of one index), first tested with the separable squared distances and a margin, then with the
    reference's test ``|coordinates[row] - centre| <= radius``.  Per centre the rows ascend.
    """

    pairs = centres.shape[0]
    low = np.empty((pairs, 3), dtype=np.int64)
    high = np.empty((pairs, 3), dtype=np.int64)
    for axis in range(3):
        values = axes[axis]
        low[:, axis] = np.searchsorted(values, centres[:, axis] - radius, side="left") - 1
        high[:, axis] = np.searchsorted(values, centres[:, axis] + radius, side="right")
    low = np.maximum(low, 0)
    high = np.minimum(high, np.asarray(shape, dtype=np.int64) - 1)
    present = np.all(low <= high, axis=1)
    widths = np.where(present[:, None], high - low + 1, 0)
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty((0, 3)), np.empty(0))
    if not np.any(present):
        return empty
    width = widths.max(axis=0)
    per_pair = int(np.prod(width))
    loose = (radius * (1.0 + 1.0e-9) + 1.0e-9) ** 2
    found_pair, found_rows = [], []
    step = max(1, _CANDIDATES_PER_PASS // per_pair)
    for first in range(0, pairs, step):
        last = min(first + step, pairs)
        squares = []
        inside = []
        indices = []
        for axis in range(3):
            index = low[first:last, axis, None] + np.arange(width[axis], dtype=np.int64)[None, :]
            valid = index <= high[first:last, axis, None]
            index = np.minimum(index, shape[axis] - 1)
            offset = axes[axis][index] - centres[first:last, axis, None]
            squares.append(offset * offset)
            inside.append(valid)
            indices.append(index)
        near = (
            squares[0][:, :, None, None] + squares[1][:, None, :, None] + squares[2][:, None, None, :] <= loose
        )
        near &= inside[0][:, :, None, None] & inside[1][:, None, :, None] & inside[2][:, None, None, :]
        pair, a, b, c = np.nonzero(near)
        pair_global = pair + first
        rows = (indices[0][pair, a] * shape[1] + indices[1][pair, b]) * shape[2] + indices[2][pair, c]
        found_pair.append(pair_global)
        found_rows.append(rows)
    pair = np.concatenate(found_pair)
    rows = np.concatenate(found_rows)
    relative = coordinates[rows] - centres[pair]
    distances = np.linalg.norm(relative, axis=1)
    keep = distances <= radius
    return pair[keep], rows[keep], relative[keep], distances[keep]


def batched_nonlocal_projectors(
    grid,
    atoms,
    potentials,
    specifications,
    lattice_vectors=None,
    k_point=None,
    periodic_dimensions: int = 3,
):
    """The :class:`~parsec_python.V_ion.ionic_potential.NonlocalProjectorOperator` of
    :func:`~parsec_python.V_ion.ionic_potential.build_nonlocal_projectors`, or ``None`` for an input it does not
    take (the reference then runs)."""

    if lattice_vectors is None or periodic_dimensions != 3:
        return None
    # a k-point: the Bloch projectors of the reference, each image times exp(-ik.(r - T)) (complex)
    k = None if k_point is None else np.asarray(k_point, dtype=float)
    shape = _row_major_box(grid)
    if shape is None:
        return None
    lattice = np.asarray(lattice_vectors, dtype=float)
    if lattice.shape != (3, 3) or not is_diagonal_lattice(lattice) or np.any(np.diag(lattice) <= 0.0):
        return None
    if not np.array_equal(lattice, np.asarray(grid.cell.lattice_vectors, dtype=float)):
        return None
    sides = np.diag(lattice)
    width = grid.settings.stencil_half_width
    species = {}
    for atom in atoms:
        symbol = atom.symbol
        if symbol in species:
            continue
        potential = potentials[symbol]
        if type(potential) is not ParsecPseudopotential:
            return None
        radius = float(_ionic._projector_support_radius(potential))
        # two images of one atom on one grid point: the reference sums them (keep it)
        if not 2.0 * radius < float(np.min(sides)) * (1.0 - 1.0e-9):
            return None
        channels = _channels(potential, specifications[symbol], width)
        species[symbol] = (
            potential,
            radius,
            np.asarray(_ionic._periodic_image_translations(lattice, radius), dtype=float),
            channels,
            sum(2 * channel[0] + 1 for channel in channels),
        )

    coordinates = np.asarray(grid.coordinates, dtype=float)
    box = coordinates.reshape(shape[0], shape[1], shape[2], 3)
    axes = (
        np.ascontiguousarray(box[:, 0, 0, 0]),
        np.ascontiguousarray(box[0, :, 0, 1]),
        np.ascontiguousarray(box[0, 0, :, 2]),
    )
    lowest = np.array([values.min() for values in axes])
    highest = np.array([values.max() for values in axes])
    sqrt_dv = np.sqrt(grid.volume_element)
    centre_radius = {
        symbol: min(float(entry[0].radii[0]), _ionic.PROJECTOR_CENTER_ROUND_OFF) for symbol, entry in species.items()
    }
    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms]).reshape(-1, 3)
    symbols = [atom.symbol for atom in atoms]
    size = int(grid.size)
    threads = setup_threads()
    # two blocks per thread or more, of 128 to 1024 atoms
    per_block = min(_ATOMS_PER_BLOCK, max(128, -(-len(atoms) // (2 * threads))))
    blocks = [(first, min(first + per_block, len(atoms))) for first in range(0, len(atoms), per_block)]

    def points_of(block):
        """The points of every atom of the block, species by species, and the point count of each atom."""
        block_first, block_last = block
        found = {}
        counts = np.zeros(block_last - block_first, dtype=np.int64)
        for symbol, (_, radius, translations, _, _) in species.items():
            members = np.array(
                [index for index in range(block_first, block_last) if symbols[index] == symbol], dtype=np.int64
            )
            if members.size == 0:
                continue
            centres = positions[members][:, None, :] + translations[None, :, :]
            # images whose ball can meet the box of the grid (a margin; an image without points adds nothing)
            reach = np.all((centres + radius + 1.0e-9 >= lowest) & (centres - radius - 1.0e-9 <= highest), axis=2)
            member_of_pair, translation_of_pair = np.nonzero(reach)
            pair_centres = positions[members[member_of_pair]] + translations[translation_of_pair]
            pair, rows, relative, distances = _image_points(axes, shape, coordinates, pair_centres, radius)
            atom_of_entry = members[member_of_pair[pair]]
            # the rows of an atom ascending (its images in turn give each its own ascending run)
            order = np.argsort(atom_of_entry * size + rows, kind="stable")
            image = None if k is None else translations[translation_of_pair[pair]][order]
            found[symbol] = (atom_of_entry[order], rows[order], relative[order], distances[order], image)
            counts += np.bincount(atom_of_entry - block_first, minlength=block_last - block_first)
        return found, counts

    def values_of(block, found, first_column):
        """``(columns, rows, values)`` of the block, by column and in each column by row."""
        block_first, _ = block
        block_rows, block_columns, block_values = [], [], []
        for symbol, (atom_of_entry, rows, relative, distances, image) in found.items():
            # the reference's phase of each entry: exp(-1j * ((r - T) @ k)) for the translation T of its image
            phase = 1.0 if k is None else np.exp(-1j * ((coordinates[rows] - image) @ k))
            potential, _, _, channels, _ = species[symbol]
            base = first_column[atom_of_entry - block_first]
            channel_offset = 0
            for angular_momentum, radial_grid, _, spline in channels:
                if spline is not None:
                    radial = spline(np.maximum(distances, potential.radii[0]))
                else:
                    radial = np.interp(distances, potential.radii, radial_grid, left=radial_grid[0], right=0.0)
                harmonics = _ionic.real_spherical_harmonics(
                    angular_momentum, relative, center_radius=centre_radius[symbol]
                )
                for harmonic_index in range(2 * angular_momentum + 1):
                    # the reference's phase * (sqrt(dV) * radial * Y), its phase 1.0 at Gamma
                    projector = phase * (sqrt_dv * radial * harmonics[:, harmonic_index])
                    keep = np.abs(projector) > 1.0e-16
                    block_rows.append(rows[keep])
                    block_columns.append(base[keep] + channel_offset + harmonic_index)
                    block_values.append(projector[keep])
                channel_offset += 2 * angular_momentum + 1
        if not block_rows:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=float if k is None else complex)
        columns = np.concatenate(block_columns)
        order = np.argsort(columns, kind="stable")
        return columns[order], np.concatenate(block_rows)[order], np.concatenate(block_values)[order]

    threads = max(1, min(threads, len(blocks)))
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="parsec-periodic-projectors") as pool:
        points = list(pool.map(points_of, blocks))
        # columns: every atom with points, in atom order, its channels in turn, 2l+1 each
        column = 0
        signs: list[float] = []
        labels: list[tuple[int, int, int]] = []
        first_columns = []
        for (block_first, block_last), (_, counts) in zip(blocks, points):
            first_column = np.zeros(block_last - block_first, dtype=np.int64)
            for local in range(block_last - block_first):
                index = block_first + local
                first_column[local] = column
                if counts[local] == 0:
                    continue
                for angular_momentum, _, denominator_sign, _ in species[symbols[index]][3]:
                    for harmonic_index in range(2 * angular_momentum + 1):
                        signs.append(denominator_sign)
                        labels.append((index, angular_momentum, harmonic_index))
                    column += 2 * angular_momentum + 1
            first_columns.append(first_column)
        pieces = list(
            pool.map(
                lambda item: values_of(item[0], item[1][0], item[2]), zip(blocks, points, first_columns)
            )
        )
    del points
    pieces_columns = [piece[0] for piece in pieces]
    pieces_rows = [piece[1] for piece in pieces]
    pieces_values = [piece[2] for piece in pieces]

    if column == 0:
        matrix = sp.csc_matrix((size, 0), dtype=float if k is None else complex)
    else:
        columns = np.concatenate(pieces_columns) if pieces_columns else np.empty(0, dtype=np.int64)
        index_type = np.int32 if max(size, columns.size, column) < 2**31 else np.int64
        indptr = np.zeros(column + 1, dtype=index_type)
        np.cumsum(np.bincount(columns, minlength=column), out=indptr[1:])
        rows = np.concatenate(pieces_rows).astype(index_type, copy=False) if pieces_rows else np.empty(0, index_type)
        values = np.concatenate(pieces_values) if pieces_values else np.empty(0, dtype=float if k is None else complex)
        matrix = sp.csc_matrix((values, rows, indptr), shape=(size, column))
        matrix.has_canonical_format = True
    return _ionic.NonlocalProjectorOperator(
        projectors=matrix, signs=np.asarray(signs, dtype=float), labels=tuple(labels)
    )


__all__ = ["batched_nonlocal_projectors"]
