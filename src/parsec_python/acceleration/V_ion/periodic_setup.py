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


def _axis_spacing(grid: PeriodicRealSpaceGrid) -> tuple[np.ndarray, np.ndarray, float]:
    n = np.asarray(grid.points_per_axis, dtype=np.int64)
    sides = np.diag(np.asarray(grid.cell.lattice_vectors, dtype=float))
    # ``build_periodic_grid`` places point ``i`` at ``i * (side / n)`` of the first axis
    return n, sides, float(sides[0] / n[0])


def _index_range(centre: float, radius: float, spacing: float, count: int):
    """Inclusive grid index range around ``centre`` (one index of margin for round-off), or ``None``."""
    low = max(math.ceil((centre - radius) / spacing) - 1, 0)
    high = min(math.floor((centre + radius) / spacing) + 1, count - 1)
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
    from the index box around the centre.  Other grids (and oblique cells) use the dense scan.
    """
    if not isinstance(grid, PeriodicRealSpaceGrid) or not grid.is_orthorhombic:
        # an oblique cell has no index box around a centre: the dense scan
        return _ionic.dense_support(grid, centre, radius)
    n, _, spacing = _axis_spacing(grid)
    ranges = []
    for axis in range(3):
        found = _index_range(float(centre[axis]), radius, spacing, int(n[axis]))
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
    axis_images = []
    for axis in range(3):
        reach = int(math.ceil(radius / sides[axis])) + 1
        found = []
        for image in range(-reach, reach + 1):
            centre = float(position[axis]) + image * sides[axis]
            index_range = _index_range(centre, radius, spacing, int(n[axis]))
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
    off_diagonal = lattice_vectors - np.diag(np.diag(lattice_vectors))
    if not np.allclose(off_diagonal, 0.0, atol=1.0e-10):
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
            use_spline=specification.use_spline,
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
