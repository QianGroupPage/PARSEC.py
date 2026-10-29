"""Fast preparation of a periodic cell: ionic potential, projectors and atomic densities.

The readable definitions (:func:`parsec_python.V_ion.ewald.ewald_local_ionic_potential`,
:func:`~parsec_python.V_ion.ionic_potential.build_nonlocal_projectors`,
:func:`~parsec_python.V_ion.ionic_potential.superpose_atomic_density`) measure the distance of
*every* grid point to every atom image, so their cost is ``atoms x grid points x images`` and a
864-atom cell (630k points) takes hours.  Two replacements, installed when the acceleration package is
imported, give the same numbers:

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
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.special import erfcinv

from parsec_python.Grid.pbc import PeriodicRealSpaceGrid
from parsec_python.V_ion import ewald as _ewald
from parsec_python.V_ion import ionic_potential as _ionic

_NYQUIST_FRACTION = 0.85  # the reciprocal series ends at this fraction of the grid's Nyquist frequency
_POINTS_PER_CALL = 1_000_000  # grid points evaluated together (the radial spline is built once per call)
_STRUCTURE_FACTOR_ELEMENTS = 4_000_000  # complex numbers of one structure-factor block


def _cell_geometry(grid: PeriodicRealSpaceGrid, lattice_vectors=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(n, lattice, inverse)``: points per axis, the rows ``a_m`` and their inverse.

    ``build_periodic_grid`` places point ``(i, j, k)`` at ``(i + s_0) a_0/n_0 + (j + s_1) a_1/n_1 + (k + s_2) a_2/n_2``
    (``s = grid.shift``), so the fractional coordinate along axis ``m`` of a point ``x`` is ``x . inverse[:, m]`` and its
    grid index ``n_m`` times that, minus ``s_m``.
    """
    n = np.asarray(grid.points_per_axis, dtype=np.int64)
    lattice = np.asarray(grid.cell.lattice_vectors if lattice_vectors is None else lattice_vectors, dtype=float)
    return n, lattice, np.linalg.inv(lattice)


def _index_range(centre: float, half_width: float, count: int, shift: float = 0.0):
    """Inclusive grid index range of the fractional interval ``centre +- half_width`` (one index of margin for
    round-off), or ``None``; the grid point ``i`` is at fractional position ``(i + shift)/count``."""
    low = max(math.ceil(count * (centre - half_width) - shift) - 1, 0)
    high = min(math.floor(count * (centre + half_width) - shift) + 1, count - 1)
    return (low, high) if low <= high else None


def _box_rows(n: np.ndarray, ranges) -> np.ndarray:
    (i0, i1), (j0, j1), (k0, k1) = ranges
    i = np.arange(i0, i1 + 1, dtype=np.int64)
    j = np.arange(j0, j1 + 1, dtype=np.int64)
    k = np.arange(k0, k1 + 1, dtype=np.int64)
    return ((i[:, None, None] * n[1] + j[None, :, None]) * n[2] + k[None, None, :]).reshape(-1)


def grid_ball(grid, centre: np.ndarray, radius: float):
    """``(rows, relative, distances)`` of the grid points within ``radius`` of ``centre``.

    The result of :func:`parsec_python.V_ion.ionic_potential.dense_support` for a periodic grid,
    from the index box around the centre.  A ball of radius ``r`` lies within ``r |inverse[:, m]|`` of its
    centre along fractional axis ``m``, in an oblique cell as well as a rectangular one.  Other grids use
    the dense scan.
    """
    if not isinstance(grid, PeriodicRealSpaceGrid):
        return _ionic.dense_support(grid, centre, radius)
    n, _, inverse = _cell_geometry(grid)
    fractional = np.asarray(centre, dtype=float) @ inverse
    half_width = radius * np.linalg.norm(inverse, axis=0)
    ranges = []
    for axis in range(3):
        found = _index_range(float(fractional[axis]), float(half_width[axis]), int(n[axis]), float(grid.shift[axis]))
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
    n, lattice, inverse = _cell_geometry(grid)
    fractional = np.asarray(position, dtype=float) @ inverse
    half_width = radius * np.linalg.norm(inverse, axis=0)
    axis_images = []
    for axis in range(3):
        reach = int(math.ceil(half_width[axis])) + 1
        found = []
        for image in range(-reach, reach + 1):
            index_range = _index_range(
                float(fractional[axis]) + image, float(half_width[axis]), int(n[axis]), float(grid.shift[axis])
            )
            if index_range is not None:
                found.append((image, index_range))
        axis_images.append(found)
    for ix, rx in axis_images[0]:
        for iy, ry in axis_images[1]:
            for iz, rz in axis_images[2]:
                box = _box_rows(n, (rx, ry, rz))
                image_centre = position + ix * lattice[0] + iy * lattice[1] + iz * lattice[2]
                relative = grid.coordinates[box] - image_centre
                distance = np.linalg.norm(relative, axis=1)
                keep = distance <= radius
                if np.any(keep):
                    yield box[keep], distance[keep]


def _structure_factor_series(
    grid, positions: np.ndarray, charges: np.ndarray, eta: float, tolerance: float, lattice_vectors=None
) -> np.ndarray:
    """``sum_G c_G exp(iG.r)`` on the grid for ``c_G = -(8 pi/V) w(G) sum_a Z_a exp(-iG.r_a)``,
    ``w = exp(-G^2/4 eta^2)/G^2``, G != 0, evaluated by one FFT.

    ``G = m_0 b_0 + m_1 b_1 + m_2 b_2`` with the reciprocal vectors ``b = 2 pi inverse^T``, so ``G.r`` of grid
    point ``(i, j, k)`` is ``2 pi (m_0 i/n_0 + m_1 j/n_1 + m_2 k/n_2)``: the integer triples ``m`` are the
    frequencies of the FFT, in an oblique cell as well as a rectangular one.
    """
    n, lattice, inverse = _cell_geometry(grid, lattice_vectors)
    volume = abs(float(np.linalg.det(lattice)))
    cutoff = 2.0 * eta * math.sqrt(-math.log(tolerance))
    # |m_i| = |G.a_i|/(2 pi) <= cutoff |a_i|/(2 pi)
    half = np.minimum(np.floor(cutoff * np.linalg.norm(lattice, axis=1) / (2.0 * np.pi)).astype(np.int64), (n - 1) // 2)
    frequencies = [np.arange(-m, m + 1, dtype=np.int64) for m in half]
    metric = (2.0 * np.pi) ** 2 * (inverse.T @ inverse)  # g_squared = m . metric . m
    m0, m1, m2 = (frequencies[0][:, None, None], frequencies[1][None, :, None], frequencies[2][None, None, :])
    g_squared = (
        metric[0, 0] * m0**2 + metric[1, 1] * m1**2 + metric[2, 2] * m2**2
        + 2.0 * (metric[0, 1] * m0 * m1 + metric[0, 2] * m0 * m2 + metric[1, 2] * m1 * m2)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.exp(-g_squared / (4.0 * eta**2)) / g_squared
    weight[g_squared == 0.0] = 0.0
    weight[g_squared > cutoff**2] = 0.0

    # S(G) = sum_a Z_a exp(-i G.r_a) = sum_a Z_a px[a,x] py[a,y] pz[a,z]: one matrix product per block of atoms
    shape = tuple(int(m) for m in 2 * half + 1)
    structure = np.zeros(shape, dtype=np.complex128)
    block = max(1, _STRUCTURE_FACTOR_ELEMENTS // (shape[1] * shape[2]))
    fractional = positions @ inverse
    for start in range(0, len(positions), block):
        stop = min(start + block, len(positions))
        phases = [
            np.exp(-2.0j * np.pi * np.outer(fractional[start:stop, axis], frequencies[axis])) for axis in range(3)
        ]
        yz = (phases[1][:, :, None] * phases[2][:, None, :]).reshape(stop - start, -1)
        structure += ((charges[start:stop, None] * phases[0]).T @ yz).reshape(shape)

    coefficients = (-8.0 * np.pi / volume) * weight * structure
    # the points are at (i + s)/n of the lattice vectors: exp(2 pi i m.(i + s)/n) = exp(2 pi i m.i/n) exp(2 pi i m.s/n)
    shifts = np.asarray(grid.shift, dtype=float)
    if np.any(shifts != 0.0):
        coefficients = coefficients * np.exp(
            2.0j * np.pi * (m0 * shifts[0] / n[0] + m1 * shifts[1] / n[1] + m2 * shifts[2] / n[2])
        )
    spectrum = np.zeros(tuple(int(v) for v in n), dtype=np.complex128)
    index = [np.mod(frequencies[axis], n[axis]) for axis in range(3)]
    spectrum[np.ix_(index[0], index[1], index[2])] = coefficients
    # ifftn divides by the number of points: the series is a plain sum over G
    return (np.fft.ifftn(spectrum) * spectrum.size).real.reshape(-1)


def fft_local_ionic_potential(
    grid, atoms: Sequence, potentials, specifications, lattice_vectors, eta, tolerance
):
    """The local ionic potential of :func:`~parsec_python.V_ion.ewald.ewald_local_ionic_potential`."""
    if not isinstance(grid, PeriodicRealSpaceGrid):
        return None
    n, lattice, _ = _cell_geometry(grid, lattice_vectors)
    volume = abs(float(np.linalg.det(lattice)))
    reference_eta, _, _ = _ewald._ewald_cutoffs(volume, eta, tolerance)

    # the series has |m_i| <= cutoff |a_i|/(2 pi), below n_i/2 when cutoff < pi/step_i for every axis
    nyquist = math.pi / float(np.max(np.linalg.norm(lattice, axis=1) / n))
    fast_eta = _NYQUIST_FRACTION * nyquist / (2.0 * math.sqrt(-math.log(tolerance)))
    erfc_cutoff = float(erfcinv(tolerance) / fast_eta)

    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)

    total = _structure_factor_series(grid, positions, charges, fast_eta, tolerance, lattice_vectors)

    # Short-ranged part: V_a + 2 Z erf(eta r)/r over the ball of every image, one radial call per batch
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
            use_spline=_ewald.local_potential_uses_spline(specification),
            spline_padding_width=grid.settings.stencil_half_width,
        ) + _ewald._erf_over_r(potential.ionic_charge, distances, fast_eta)
        total_view[:] += np.bincount(rows, weights=values, minlength=total_view.size)
        pending_rows, pending_distances, pending_points = [], [], 0

    total_view = total
    for atom, position in zip(atoms, positions):
        if atom.symbol != batch_symbol:
            flush()
            batch_symbol = atom.symbol
        # beyond the table of the pseudopotential V_a is exactly -2 Z/r, but inside it V_a differs from
        # the Coulomb tail, so the ball reaches the larger of the two radii
        reach = max(erfc_cutoff, float(potentials[atom.symbol].interpolation_cutoff))
        for rows, distances in _atom_balls(grid, position, reach):
            pending_rows.append(rows)
            pending_distances.append(distances)
            pending_points += rows.size
            if pending_points >= _POINTS_PER_CALL:
                flush()
    flush()

    # The reference carries the G = 0 term of its own eta: -(2 pi/V) Z_total/eta^2 per unit of charge
    total += -(2.0 * np.pi * float(np.sum(charges)) / volume) * (
        1.0 / reference_eta**2 - 1.0 / fast_eta**2
    )
    return total


def install() -> None:
    """Make the fast replacements the ones the reference drivers use."""
    _ionic.set_support_finder(grid_ball)
    _ewald.set_local_potential_provider(fft_local_ionic_potential)


def uninstall() -> None:
    _ionic.set_support_finder(None)
    _ewald.set_local_potential_provider(None)


__all__ = ["fft_local_ionic_potential", "grid_ball", "install", "uninstall"]
