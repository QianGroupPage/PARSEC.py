"""The local ionic potential of a periodic cell on a GPU: :func:`.periodic_setup.fft_local_ionic_potential` there.

:func:`.periodic_setup.fft_local_ionic_potential` splits the potential of every atom as Ewald does: the series of the
Gaussian charges, ``sum_G c_G exp(iG.r)`` with ``c_G`` from the structure factor ``S(G) = sum_a Z_a exp(-iG.r_a)``
(one FFT), and the short-ranged ``V_a(r) + 2 Z erf(eta r)/r`` over the grid points within the reach of every image of
every atom.  On the host both grow with the atoms times the points they touch: bulk diamond with 32,768 atoms on
23.9 M points took 153 s on one node (about half each), the largest part of the preparation of the root.  Here:

* the structure factor is the same blocked product of the axis phases (``cupy.matmul``, complex128) and the series
  the same inverse FFT, on the device;
* the short-ranged terms are gathered point by point: the images of the atoms that reach the cell are sorted into
  cubic bins of the largest reach, and every grid point adds the terms of the images in the 27 bins around it, in
  the order of the bins and of the atoms in them (deterministic), with the radial rules of the host: the cubic
  spline of the table on PARSEC's origin-padded knots (``splint``; the periodic local potential takes the spline,
  :data:`parsec_python.V_ion.ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE`) or ``r V`` interpolated linearly and divided
  by ``r``, the Coulomb tail ``-2 Z/r`` from ``rs(ns-1)`` on, and ``2 Z erf(eta r)/r`` (``4 Z eta/sqrt(pi)`` at
  ``r = 0``), at the points with ``r <= reach``.

The terms are those of the host; the order of the sums differs, so the potential agrees to round-off (tests:
relative 1e-13).  ``PARSEC_PERIODIC_SETUP_DEVICE``: ``auto`` (the default) takes the device where CuPy has one and
``PARSEC_DEVICE`` is not ``cpu``, for the bulk grids of the default routines with diagonal lattice vectors; ``off``
keeps the host sums, bit for bit.  ``None`` sends the caller to the host.
"""

from __future__ import annotations

import math
import os

import numpy as np

from parsec_python.V_ion import ewald as _ewald
from parsec_python.V_ion import ionic_potential as _ionic

SWITCH = "PARSEC_PERIODIC_SETUP_DEVICE"
_THREADS = 128
# Complex numbers of one block of the yz phases of the structure factor on the device.
_STRUCTURE_FACTOR_ELEMENTS = 1 << 25


def setup_device_setting() -> str:
    """``auto`` or ``off`` from ``PARSEC_PERIODIC_SETUP_DEVICE``."""

    value = os.environ.get(SWITCH, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{SWITCH} must be auto or off, not {value!r}")
    return value


_KERNEL = r"""
__device__ __forceinline__ long long radial_segment(
    const double r, const double* radii, const int samples, const double scale_a, const double scale_b)
{
    // searchsorted(radii, r, side="right") - 1, clipped to [0, samples - 2]
    long long index = (long long)(log1p(r / scale_a) / scale_b) - 1;
    if (index < 0) index = 0;
    if (index > samples - 2) index = samples - 2;
    while (index > 0 && r < radii[index]) --index;
    while (index < samples - 2 && r >= radii[index + 1]) ++index;
    return index;
}

// One short-ranged term of a species: V_a(r) + 2 Z erf(eta r)/r.
// parameters (10 per species): reach, two_z, cutoff, spline (1) or linear (0), scale_a, scale_b,
// first radius, first value, offset of the first positive knot in the padded table, eta
__device__ __forceinline__ double short_range_term(
    const double r, const double* p, const double* knots, const double* values, const double* second,
    const double* radial_table, const int knot_offset, const int knot_count, const int table_offset)
{
    const double two_z = p[1], cutoff = p[2], scale_a = p[4], scale_b = p[5];
    const double first_radius = p[6], first_value = p[7], eta = p[9];
    const int positive = (int)p[8];  // index of radii[0] in the padded knots
    double value;
    if (!(r < cutoff)) {
        value = -two_z / r;
    } else if (p[3] > 0.5) {
        // ParsecRadialSpline.__call__: lower = searchsorted(knots, r, "right") - 1, clipped to [0, count - 2]
        const double* k = knots + knot_offset;
        const double* y = values + knot_offset;
        const double* y2 = second + knot_offset;
        long long lower;
        if (r < k[positive]) {
            lower = positive - 1;  // the knot 0.0 of the padding
        } else {
            lower = positive + radial_segment(r, k + positive, knot_count - positive, scale_a, scale_b);
        }
        if (lower > knot_count - 2) lower = knot_count - 2;
        const double step = k[lower + 1] - k[lower];
        const double left = (k[lower + 1] - r) / step;
        const double right = (r - k[lower]) / step;
        value = left * y[lower] + right * y[lower + 1]
            + ((left * left * left - left) * y2[lower] + (right * right * right - right) * y2[lower + 1])
              * (step * step) / 6.0;
    } else if (r <= first_radius) {
        value = first_value;
    } else {
        const double* radii = knots + knot_offset + positive;
        const double* rv = radial_table + table_offset;
        const long long j = radial_segment(r, radii, knot_count - positive, scale_a, scale_b);
        const double slope = (rv[j + 1] - rv[j]) / (radii[j + 1] - radii[j]);
        value = (slope * (r - radii[j]) + rv[j]) / r;
    }
    if (r == 0.0) {
        value += 2.0 * two_z * eta * 0.56418958354775628;  // 4 Z eta / sqrt(pi)
    } else {
        value += two_z * erf(eta * r) / r;
    }
    return value;
}

extern "C" __global__
void periodic_short_range_binned(
    const long long n0, const long long n1, const long long n2,
    const double* xs, const double* ys, const double* zs,
    const double ox, const double oy, const double oz, const double bin,
    const int nbx, const int nby, const int nbz, const int* bin_start,
    const double* ex, const double* ey, const double* ez, const int* kinds,
    const double* parameters, const double* knots, const double* values, const double* second,
    const double* radial_table, const int* knot_offsets, const int* knot_counts, const int* table_offsets,
    double* field)
{
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= n0 * n1 * n2) return;
    const double x = xs[point / (n1 * n2)];
    const double y = ys[(point / n2) % n1];
    const double z = zs[point % n2];
    int bx = (int)floor((x - ox) / bin), by = (int)floor((y - oy) / bin), bz = (int)floor((z - oz) / bin);
    bx = bx < 0 ? 0 : (bx >= nbx ? nbx - 1 : bx);
    by = by < 0 ? 0 : (by >= nby ? nby - 1 : by);
    bz = bz < 0 ? 0 : (bz >= nbz ? nbz - 1 : bz);
    double total = 0.0;
    for (int cx = (bx > 0 ? bx - 1 : 0); cx <= (bx + 1 < nbx ? bx + 1 : nbx - 1); ++cx) {
        for (int cy = (by > 0 ? by - 1 : 0); cy <= (by + 1 < nby ? by + 1 : nby - 1); ++cy) {
            for (int cz = (bz > 0 ? bz - 1 : 0); cz <= (bz + 1 < nbz ? bz + 1 : nbz - 1); ++cz) {
                const int b = (cx * nby + cy) * nbz + cz;
                for (int e = bin_start[b]; e < bin_start[b + 1]; ++e) {
                    const double dx = x - ex[e];
                    const double dy = y - ey[e];
                    const double dz = z - ez[e];
                    const int kind = kinds[e];
                    const double* p = parameters + 10 * kind;
                    const double reach = p[0];
                    const double square = dx * dx + dy * dy + dz * dz;
                    if (square > reach * reach * (1.0 + 1.0e-12) + 1.0e-300) continue;
                    const double r = sqrt(square);
                    if (!(r <= reach)) continue;
                    total += short_range_term(r, p, knots, values, second, radial_table,
                                              knot_offsets[kind], knot_counts[kind], table_offsets[kind]);
                }
            }
        }
    }
    field[point] += total;
}
"""

_COMPILED: dict[int, object] = {}


def _kernel(cp):
    device = int(cp.cuda.Device().id)
    if device not in _COMPILED:
        kernel = cp.RawKernel(_KERNEL, "periodic_short_range_binned", options=("--std=c++11",))
        kernel.compile()
        _COMPILED[device] = kernel
    return _COMPILED[device]


def _cupy():
    from .periodic_device import _cupy as device_cupy

    return device_cupy()


def _diagonal_cell(grid, lattice_vectors) -> bool:
    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    return bool(
        lattice.shape == (3, 3)
        and np.all(np.abs(lattice - np.diag(np.diag(lattice))) <= 1.0e-12 * np.max(np.abs(lattice)))
        and np.all(np.diag(lattice) > 0.0)
        and tuple(int(v) for v in grid.points_per_axis) == tuple(int(v) for v in np.asarray(grid.lookup).shape)
    )


def _structure_factor_series_device(cp, grid, positions, charges, eta, tolerance, lattice_vectors):
    """:func:`.periodic_setup._structure_factor_series` on the device (the same blocks of the same product)."""

    from . import periodic_setup as _setup

    n, lattice, inverse = _setup._cell_geometry(grid, lattice_vectors)
    volume = abs(float(np.linalg.det(lattice)))
    cutoff = 2.0 * eta * math.sqrt(-math.log(tolerance))
    half = np.minimum(np.floor(cutoff * np.linalg.norm(lattice, axis=1) / (2.0 * np.pi)).astype(np.int64), (n - 1) // 2)
    frequencies = [np.arange(-m, m + 1, dtype=np.int64) for m in half]
    metric = (2.0 * np.pi) ** 2 * (inverse.T @ inverse)
    m0, m1, m2 = (
        cp.asarray(frequencies[0])[:, None, None],
        cp.asarray(frequencies[1])[None, :, None],
        cp.asarray(frequencies[2])[None, None, :],
    )
    g_squared = (
        metric[0, 0] * m0**2 + metric[1, 1] * m1**2 + metric[2, 2] * m2**2
        + 2.0 * (metric[0, 1] * m0 * m1 + metric[0, 2] * m0 * m2 + metric[1, 2] * m1 * m2)
    ).astype(cp.float64)
    zero = g_squared == 0.0
    weight = cp.exp(-g_squared / (4.0 * eta**2)) / cp.where(zero, 1.0, g_squared)
    weight[zero] = 0.0
    weight[g_squared > cutoff**2] = 0.0
    del g_squared, zero

    shape = tuple(int(m) for m in 2 * half + 1)
    structure = cp.zeros(shape, dtype=cp.complex128)
    block = max(1, _STRUCTURE_FACTOR_ELEMENTS // (shape[1] * shape[2]))
    fractional = cp.asarray(positions @ inverse)
    device_charges = cp.asarray(charges)
    device_frequencies = [cp.asarray(axis, dtype=cp.float64) for axis in frequencies]
    for start in range(0, len(positions), block):
        stop = min(start + block, len(positions))
        phases = [
            cp.exp(-2.0j * np.pi * cp.outer(fractional[start:stop, axis], device_frequencies[axis]))
            for axis in range(3)
        ]
        yz = (phases[1][:, :, None] * phases[2][:, None, :]).reshape(stop - start, -1)
        structure += ((device_charges[start:stop, None] * phases[0]).T @ yz).reshape(shape)
        del phases, yz
    coefficients = (-8.0 * np.pi / volume) * weight * structure
    del structure, weight
    shifts = np.asarray(grid.shift, dtype=float)
    if np.any(shifts != 0.0):
        coefficients *= cp.exp(
            2.0j * np.pi * (m0 * shifts[0] / n[0] + m1 * shifts[1] / n[1] + m2 * shifts[2] / n[2])
        )
    spectrum = cp.zeros(tuple(int(v) for v in n), dtype=cp.complex128)
    index = [cp.asarray(np.mod(frequencies[axis], n[axis])) for axis in range(3)]
    spectrum[cp.ix_(index[0], index[1], index[2])] = coefficients
    del coefficients
    series = (cp.fft.ifftn(spectrum) * spectrum.size).real.reshape(-1)
    del spectrum
    return cp.ascontiguousarray(series)


def _image_entries(positions, kinds, reaches, axes, sides):
    """The images of the atoms that reach a grid point: centres ``(m, 3)``, species, in atom then image order."""

    centres, entry_kinds = [], []
    low = np.array([float(axis.min()) for axis in axes])
    high = np.array([float(axis.max()) for axis in axes])
    for index in range(positions.shape[0]):
        reach = float(reaches[kinds[index]])
        ranges = []
        for axis in range(3):
            first = math.ceil((low[axis] - reach - positions[index, axis]) / sides[axis]) - 1
            last = math.floor((high[axis] + reach - positions[index, axis]) / sides[axis]) + 1
            ranges.append(range(first, last + 1))
        for ix in ranges[0]:
            x = positions[index, 0] + ix * sides[0]
            if x < low[0] - reach or x > high[0] + reach:
                continue
            for iy in ranges[1]:
                y = positions[index, 1] + iy * sides[1]
                if y < low[1] - reach or y > high[1] + reach:
                    continue
                for iz in ranges[2]:
                    z = positions[index, 2] + iz * sides[2]
                    if z < low[2] - reach or z > high[2] + reach:
                        continue
                    centres.append((x, y, z))
                    entry_kinds.append(kinds[index])
    return np.asarray(centres, dtype=np.float64).reshape(-1, 3), np.asarray(entry_kinds, dtype=np.int32)


def device_fft_local_ionic_potential(grid, atoms, potentials, specifications, lattice_vectors, eta, tolerance):
    """:func:`.periodic_setup.fft_local_ionic_potential` on the device, or ``None`` (the host sums then run)."""

    from parsec_python.Grid.pbc import uses_fortran_routines

    from . import periodic_setup as _setup
    from .periodic_device import device_id

    if setup_device_setting() == "off" or uses_fortran_routines(grid) or not _setup._is_full_periodic(grid):
        return None
    if not atoms or not _diagonal_cell(grid, lattice_vectors):
        return None
    cp = _cupy()
    if cp is None:
        return None
    n, lattice, _ = _setup._cell_geometry(grid, lattice_vectors)
    sides = np.diag(np.asarray(lattice, dtype=np.float64)).copy()
    volume = abs(float(np.linalg.det(lattice)))
    reference_eta, _, _ = _ewald._ewald_cutoffs(volume, eta, tolerance)
    fast_eta, erfc_cutoff, _ = _setup.fast_splitting(grid, lattice_vectors, tolerance)

    symbols = list(dict.fromkeys(atom.symbol for atom in atoms))
    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)
    kinds = np.array([symbols.index(atom.symbol) for atom in atoms], dtype=np.int32)

    # the radial data of every species, as the host evaluates it
    padding = int(grid.settings.stencil_half_width)
    parameters, knots, values, second, tables = [], [], [], [], []
    knot_offsets, knot_counts, table_offsets = [], [], []
    reaches = []
    knot_cursor = table_cursor = 0
    for symbol in symbols:
        potential = potentials[symbol]
        specification = specifications[symbol]
        local_l = specification.local_angular_momentum
        reach = max(erfc_cutoff, _ionic.coulomb_onset_radius(potential, local_l))
        uses_spline = bool(_ewald.local_potential_uses_spline(specification))
        if uses_spline:
            _ionic.warn_spline_without_coulomb_tail(potential, local_l, reach)
        spline = potential.local_potential_spline(local_l, padding)
        radii = np.asarray(potential.radii, dtype=np.float64)
        channel = np.asarray(potential.channel_potentials[local_l], dtype=np.float64)
        positive = int(spline.knots.size - radii.size)
        if not np.array_equal(spline.knots[positive:], radii) or spline.knots[positive - 1] != 0.0:
            return None
        knots.append(np.asarray(spline.knots, dtype=np.float64))
        values.append(np.asarray(spline.values, dtype=np.float64))
        second.append(np.asarray(spline.second_derivatives, dtype=np.float64))
        tables.append(radii * channel)
        knot_offsets.append(knot_cursor)
        knot_counts.append(int(spline.knots.size))
        table_offsets.append(table_cursor)
        knot_cursor += int(spline.knots.size)
        table_cursor += int(radii.size)
        reaches.append(float(reach))
        parameters.append([
            float(reach),
            2.0 * float(potential.ionic_charge),
            float(potential.interpolation_cutoff),
            1.0 if uses_spline else 0.0,
            float(potential.radial_parameter_a),
            float(potential.radial_parameter_b),
            float(radii[0]),
            float(channel[0]),
            float(positive),
            float(fast_eta),
        ])

    coordinates = np.asarray(grid.coordinates, dtype=np.float64).reshape(int(n[0]), int(n[1]), int(n[2]), 3)
    axes = (coordinates[:, 0, 0, 0].copy(), coordinates[0, :, 0, 1].copy(), coordinates[0, 0, :, 2].copy())
    centres, entry_kinds = _image_entries(positions, kinds, reaches, axes, sides)
    bin_size = max(reaches) * (1.0 + 1.0e-9)
    origin = np.array([float(axis.min()) for axis in axes]) - max(reaches)
    extent = np.array([float(axis.max()) for axis in axes]) + max(reaches) - origin
    bins = np.maximum(np.ceil(extent / bin_size).astype(np.int64), 1)
    cell = np.clip(np.floor((centres - origin) / bin_size).astype(np.int64), 0, bins - 1)
    linear = (cell[:, 0] * bins[1] + cell[:, 1]) * bins[2] + cell[:, 2]
    order = np.argsort(linear, kind="stable")
    starts = np.zeros(int(np.prod(bins)) + 1, dtype=np.int64)
    np.add.at(starts, linear + 1, 1)
    starts = np.cumsum(starts)
    if starts[-1] >= 2**31:
        return None

    with cp.cuda.Device(device_id(cp)):
        field = _structure_factor_series_device(cp, grid, positions, charges, fast_eta, tolerance, lattice_vectors)
        sorted_centres = centres[order]
        points = int(np.prod(n))
        _kernel(cp)(
            ((points + _THREADS - 1) // _THREADS,),
            (_THREADS,),
            (
                np.int64(n[0]), np.int64(n[1]), np.int64(n[2]),
                cp.asarray(axes[0]), cp.asarray(axes[1]), cp.asarray(axes[2]),
                np.float64(origin[0]), np.float64(origin[1]), np.float64(origin[2]), np.float64(bin_size),
                np.int32(bins[0]), np.int32(bins[1]), np.int32(bins[2]),
                cp.asarray(starts.astype(np.int32)),
                cp.asarray(np.ascontiguousarray(sorted_centres[:, 0])),
                cp.asarray(np.ascontiguousarray(sorted_centres[:, 1])),
                cp.asarray(np.ascontiguousarray(sorted_centres[:, 2])),
                cp.asarray(entry_kinds[order]),
                cp.asarray(np.asarray(parameters, dtype=np.float64).reshape(-1)),
                cp.asarray(np.concatenate(knots)),
                cp.asarray(np.concatenate(values)),
                cp.asarray(np.concatenate(second)),
                cp.asarray(np.concatenate(tables)),
                cp.asarray(np.asarray(knot_offsets, dtype=np.int32)),
                cp.asarray(np.asarray(knot_counts, dtype=np.int32)),
                cp.asarray(np.asarray(table_offsets, dtype=np.int32)),
                field,
            ),
        )
        field += -(2.0 * np.pi * float(np.sum(charges)) / volume) * (1.0 / reference_eta**2 - 1.0 / fast_eta**2)
        result = cp.asnumpy(field)
        del field
        cp.get_default_memory_pool().free_all_blocks()
    return np.asarray(result, dtype=np.float64)


__all__ = ["SWITCH", "device_fft_local_ionic_potential", "setup_device_setting"]
