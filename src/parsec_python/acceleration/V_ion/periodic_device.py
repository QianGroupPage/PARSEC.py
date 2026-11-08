"""Ionic fields of a periodic cell summed on a GPU: local potential and atomic densities.

The readable builders (:func:`parsec_python.V_ion.ewald.ewald_local_ionic_potential`,
:func:`parsec_python.V_ion.ionic_potential.superpose_atomic_density`) and the fast preparation of
:mod:`.periodic_setup` sum every radial function over all lattice images out to the end of its
radial table, ``rs(ns-1)`` (near 118 bohr for the POTRE files).  An 8-atom silicon cube then sums
about 6000 images of every atom over every grid point; on the host that takes minutes for each
field, and a relaxation or a molecular-dynamics run rebuilds the fields after every move.  Most of
those terms are zero:

* the short-ranged piece of the Ewald split, ``V_a(r) + 2 Z erf(eta r)/r``, equals
  ``-2 Z erfc(eta r)/r`` wherever the table of ``r V_a`` is the Coulomb tail ``-2 Z``; beyond
  ``erfcinv(tolerance)/eta`` it is below the tolerance of the Ewald sum;
* an atomic density falls below ``1e-16`` of its peak long before the table ends (silicon:
  30.7 bohr).

Here each field is summed over the images within the radius where its terms are not zero (the
*reach*), on the device.  The long-ranged Fourier series of the potential is that of
:func:`.periodic_setup.fft_local_ionic_potential` with its ``eta`` (the largest one whose series
ends below the Nyquist frequency of the grid).  The radial rules are those of the readable code:
``r V`` interpolated linearly and divided by ``r`` (the first value below the first sample, the
Coulomb tail from ``rs(ns-1)`` on), the densities interpolated linearly (the first value continued
inwards).  The numbers differ from the full sums by the dropped terms and the order of the additions
(round-off, 1e-13 Ry for the potential of an 8-atom silicon cell).

Only the grid of :func:`parsec_python.Grid.pbc.build_periodic_grid` with diagonal lattice vectors is
handled (the grids :mod:`.periodic_setup` handles), and radial functions interpolated linearly:
``Cubic_Spline`` species keep the readable builders (``None`` is returned).  The builders are
installed for one calculation by :func:`device_periodic_preparation` (the accelerated command line
does so for the forces, relaxation and dynamics of a periodic cell); the forces of the same fields
are :mod:`parsec_python.acceleration.Forces.periodic_device`.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.special import erfcinv

from parsec_python.V_ion import ewald as _ewald
from parsec_python.V_ion import ionic_potential as _ionic

from . import periodic_setup_fortran as _setup

#: Relative deviation of ``r V`` from ``-2 Z`` below which the table is the Coulomb tail (to round-off).
COULOMB_TAIL_TOLERANCE = _ionic.COULOMB_ROUND_OFF_TOLERANCE

#: An atomic density is summed out to where it falls below this fraction of its peak (the reach of
#: the density terms of :mod:`parsec_python.Forces.periodic`).
DENSITY_TAIL_FRACTION = 1.0e-16

#: Ewald tolerance of the local potential of the periodic preparation
#: (:func:`parsec_python.V_ion.ewald.ewald_local_ionic_potential`).
EWALD_TOLERANCE = 1.0e-14

_THREADS = 128

# Device helpers shared with the force kernel (acceleration/Forces/periodic_device.py).
RADIAL_HELPERS = r"""
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

__device__ __forceinline__ double linear_value(
    const double r, const double* radii, const double* table, const long long j)
{
    // np.interp inside [radii[0], radii[-1]): slope * (r - x_j) + y_j
    const double slope = (table[j + 1] - table[j]) / (radii[j + 1] - radii[j]);
    return slope * (r - radii[j]) + table[j];
}

__device__ __forceinline__ void image_range(
    const double point, const double centre, const double reach, const double side,
    long long* low, long long* high)
{
    // the translations t with |point - (centre + t side)| <= reach, one more on each side
    *low = (long long)ceil((point - centre - reach) / side) - 1;
    *high = (long long)floor((point - centre + reach) / side) + 1;
}
"""

_FIELD_KERNEL = RADIAL_HELPERS + r"""
extern "C" __global__
void periodic_radial_field(
    const long long point_count,
    const double* coordinates,
    const int atom_count,
    const double* positions,
    const int* species,
    const double* sides,
    const double* radii,
    const double* tables,
    const int* offsets,
    const int* counts,
    const double* parameters,
    const int mode,
    double* field)
{
    // mode 0: V_a(r) + 2 Z erf(eta r)/r with r V interpolated linearly (r <= reach)
    // mode 1: a density interpolated linearly, the first value continued inwards (r < reach)
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count) return;
    const double px = coordinates[3 * point + 0];
    const double py = coordinates[3 * point + 1];
    const double pz = coordinates[3 * point + 2];
    const double lx = sides[0], ly = sides[1], lz = sides[2];
    double total = 0.0;
    for (int atom = 0; atom < atom_count; ++atom) {
        const int kind = species[atom];
        const double* r_table = radii + offsets[kind];
        const double* values = tables + offsets[kind];
        const int samples = counts[kind];
        const double* p = parameters + 8 * kind;
        const double scale_a = p[0], scale_b = p[1], first_radius = p[2], cutoff = p[3];
        const double reach = p[4], two_z = p[5], eta = p[6], first_value = p[7];
        const double reach2 = reach * reach;
        const double ax = positions[3 * atom + 0];
        const double ay = positions[3 * atom + 1];
        const double az = positions[3 * atom + 2];
        long long x0, x1, y0, y1, z0, z1;
        image_range(px, ax, reach, lx, &x0, &x1);
        image_range(py, ay, reach, ly, &y0, &y1);
        image_range(pz, az, reach, lz, &z0, &z1);
        for (long long tx = x0; tx <= x1; ++tx) {
            const double dx = px - (ax + (double)tx * lx);
            if (dx * dx > reach2) continue;
            for (long long ty = y0; ty <= y1; ++ty) {
                const double dy = py - (ay + (double)ty * ly);
                if (dx * dx + dy * dy > reach2) continue;
                for (long long tz = z0; tz <= z1; ++tz) {
                    const double dz = pz - (az + (double)tz * lz);
                    const double r = sqrt(dx * dx + dy * dy + dz * dz);
                    if (mode == 0) {
                        if (!(r <= reach)) continue;
                        double value;
                        if (r >= cutoff) {
                            value = -two_z / r;
                        } else if (r <= first_radius) {
                            value = first_value;
                        } else {
                            const long long j = radial_segment(r, r_table, samples, scale_a, scale_b);
                            value = linear_value(r, r_table, values, j) / r;
                        }
                        if (r == 0.0) {
                            value += 2.0 * two_z * eta * 0.56418958354775628;  // 1/sqrt(pi)
                        } else {
                            value += two_z * erf(eta * r) / r;
                        }
                        total += value;
                    } else {
                        if (!(r < reach) || !(r < cutoff)) continue;
                        if (r < first_radius) {
                            total += values[0];
                        } else {
                            const long long j = radial_segment(r, r_table, samples, scale_a, scale_b);
                            total += linear_value(r, r_table, values, j);
                        }
                    }
                }
            }
        }
    }
    field[point] = total;
}
"""

# The density of a general (oblique) lattice: the images of an atom whose fractional offset from a point lies
# within reach |inverse[:, m]| along every lattice direction m (the index box of periodic_setup.lattice_ball).
_LATTICE_DENSITY_KERNEL = RADIAL_HELPERS + r"""
extern "C" __global__
void periodic_lattice_density(
    const long long point_count,
    const double* coordinates,
    const int atom_count,
    const double* positions,
    const int* species,
    const double* lattice,
    const double* inverse,
    const double* widths,
    const double* radii,
    const double* tables,
    const int* offsets,
    const int* counts,
    const double* parameters,
    double* field)
{
    // a density interpolated linearly, the first value continued inwards (r < reach, r < cutoff)
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count) return;
    const double px = coordinates[3 * point + 0];
    const double py = coordinates[3 * point + 1];
    const double pz = coordinates[3 * point + 2];
    double total = 0.0;
    for (int atom = 0; atom < atom_count; ++atom) {
        const int kind = species[atom];
        const double* r_table = radii + offsets[kind];
        const double* values = tables + offsets[kind];
        const int samples = counts[kind];
        const double* p = parameters + 8 * kind;
        const double scale_a = p[0], scale_b = p[1], first_radius = p[2], cutoff = p[3];
        const double reach = p[4];
        if (!(reach > 0.0)) continue;
        const double dx = px - positions[3 * atom + 0];
        const double dy = py - positions[3 * atom + 1];
        const double dz = pz - positions[3 * atom + 2];
        long long low[3], high[3];
        for (int m = 0; m < 3; ++m) {
            // fractional coordinate of the offset along lattice vector m: d . inverse[:, m]
            const double f = dx * inverse[0 * 3 + m] + dy * inverse[1 * 3 + m] + dz * inverse[2 * 3 + m];
            low[m] = (long long)ceil(f - reach * widths[m]) - 1;
            high[m] = (long long)floor(f + reach * widths[m]) + 1;
        }
        for (long long t0 = low[0]; t0 <= high[0]; ++t0) {
            for (long long t1 = low[1]; t1 <= high[1]; ++t1) {
                for (long long t2 = low[2]; t2 <= high[2]; ++t2) {
                    const double ex = dx - ((double)t0 * lattice[0] + (double)t1 * lattice[3] + (double)t2 * lattice[6]);
                    const double ey = dy - ((double)t0 * lattice[1] + (double)t1 * lattice[4] + (double)t2 * lattice[7]);
                    const double ez = dz - ((double)t0 * lattice[2] + (double)t1 * lattice[5] + (double)t2 * lattice[8]);
                    const double r = sqrt(ex * ex + ey * ey + ez * ez);
                    if (!(r < reach) || !(r < cutoff)) continue;
                    if (r < first_radius) {
                        total += values[0];
                    } else {
                        const long long j = radial_segment(r, r_table, samples, scale_a, scale_b);
                        total += linear_value(r, r_table, values, j);
                    }
                }
            }
        }
    }
    field[point] = total;
}
"""

_COMPILED: dict[int, object] = {}
_COMPILED_LATTICE: dict[int, object] = {}


def _lattice_density_kernel(cp):
    device = int(cp.cuda.Device().id)
    if device not in _COMPILED_LATTICE:
        kernel = cp.RawKernel(_LATTICE_DENSITY_KERNEL, "periodic_lattice_density", options=("--std=c++11",))
        kernel.compile()
        _COMPILED_LATTICE[device] = kernel
    return _COMPILED_LATTICE[device]


def _field_kernel(cp):
    device = int(cp.cuda.Device().id)
    if device not in _COMPILED:
        kernel = cp.RawKernel(_FIELD_KERNEL, "periodic_radial_field", options=("--std=c++11",))
        kernel.compile()
        _COMPILED[device] = kernel
    return _COMPILED[device]


def _cupy():
    """CuPy where a device works and the run may use one (``PARSEC_DEVICE`` not ``cpu``), else ``None``."""

    import os

    if (os.environ.get("PARSEC_DEVICE", "auto").strip().lower() or "auto") == "cpu":
        return None
    try:
        from ..backends.cupy import cupy_available, require_cupy

        if not cupy_available():
            return None
        cp, _ = require_cupy()
        return cp
    except Exception:
        return None


def device_id(cp) -> int:
    """The device of these sums: the first of ``PARSEC_CUPY_DEVICES`` (that of the eigensolver)."""

    from ..Eigensolvers.symmetry import selected_device_ids

    return int(selected_device_ids(int(cp.cuda.Device().id))[0])


def handles(grid, lattice_vectors) -> bool:
    """Whether these builders take the cell: the orthorhombic grid of :mod:`.periodic_setup`."""

    return _setup._fft_handles(grid, lattice_vectors)


# ---------------------------------------------------------------------------
# Reaches and split parameters.


def coulomb_radius(potential, local_l: int) -> float:
    """Radius from which the table of ``r V_local`` is the Coulomb tail ``-2 Z``.

    The first radial sample from which every later sample of ``r V`` lies within
    :data:`COULOMB_TAIL_TOLERANCE` of ``-2 Z`` (relative).  Beyond it ``V_a(r) + 2 Z/r`` is
    round-off.  A table that never settles to round-off (FHIPP, MARTINS_OLD and MARTINS_WANG files
    carry ``r V + 2 Z`` at the precision of their digits out to their end) takes the radius from
    which it is ``-2 Z`` to that precision
    (:func:`parsec_python.V_ion.ionic_potential.coulomb_tail_radius`), else ``rs(ns-1)``.  Up to
    2026-11-07 every FHIPP table reached to ``rs(ns-1)`` (75 to 78 bohr): with ``Cubic_Spline`` the
    images then added the error of the spline of ``V`` near the end of the table (-6.9 Ry bohr^3
    per Si atom; the local potential of an orthorhombic silicon cell 0.05 Ry low) and the
    preparation of a 64-atom cell took 630 s.
    """

    radii = np.asarray(potential.radii, dtype=float)
    weighted = radii * np.asarray(potential.channel_potentials[local_l], dtype=float)
    two_z = 2.0 * float(potential.ionic_charge)
    deviating = np.flatnonzero(np.abs(weighted + two_z) > COULOMB_TAIL_TOLERANCE * max(two_z, 1.0))
    cutoff = float(potential.interpolation_cutoff)
    if deviating.size == 0:
        return float(radii[0])
    last = int(deviating[-1]) + 1
    onset = cutoff if last >= radii.size else min(cutoff, float(radii[last]))
    if onset >= cutoff:
        tail = _ionic.coulomb_tail_radius(potential, local_l)
        if tail is not None:
            return min(onset, tail)
    return onset


def density_reach(potential, table: np.ndarray) -> float:
    """Radius beyond which ``table`` stays below :data:`DENSITY_TAIL_FRACTION` of its peak.

    The rule of :func:`parsec_python.Forces.periodic._density_reach`, capped at ``rs(ns-1)``.
    """

    values = np.abs(np.asarray(table, dtype=float))
    peak = float(np.max(values, initial=0.0))
    cutoff = float(potential.interpolation_cutoff)
    if peak == 0.0:
        return 0.0
    above = np.flatnonzero(values > DENSITY_TAIL_FRACTION * peak)
    last = min(int(above[-1]) + 1, values.size - 1)
    return min(cutoff, float(potential.radii[last]))


@dataclass(frozen=True)
class SplitParameters:
    """The Ewald split of the device potential: its ``eta``, the reach of the short-ranged terms."""

    eta: float
    reference_eta: float
    erfc_cutoff: float
    tolerance: float


def split_parameters(grid, lattice_vectors, tolerance: float = EWALD_TOLERANCE, eta=None) -> SplitParameters:
    """``eta`` of :func:`.periodic_setup.fft_local_ionic_potential` and the reach of its erfc tail."""

    n, sides, _ = _setup._axis_spacing(grid)
    volume = float(np.prod(sides))
    reference_eta, _, _ = _ewald._ewald_cutoffs(volume, eta, tolerance)
    nyquist = math.pi / float(np.max(sides / n))
    fast_eta = _setup._NYQUIST_FRACTION * nyquist / (2.0 * math.sqrt(-math.log(tolerance)))
    return SplitParameters(
        eta=float(fast_eta),
        reference_eta=float(reference_eta),
        erfc_cutoff=float(erfcinv(tolerance) / fast_eta),
        tolerance=float(tolerance),
    )


def local_reach(potential, local_l: int, split: SplitParameters) -> float:
    """Reach of the short-ranged term ``V_a + 2 Z erf(eta r)/r`` of one species."""

    return max(split.erfc_cutoff, coulomb_radius(potential, local_l))


def linear_species(specifications: Mapping, symbols) -> bool:
    """Whether every species interpolates its radial functions linearly (no ``Cubic_Spline``)."""

    return not any(
        getattr(specifications[symbol], "use_spline", False)
        or getattr(specifications[symbol], "old_interpolation", False)
        for symbol in symbols
    )


def valence_density_table(potential, specification) -> np.ndarray:
    """The valence table :func:`parsec_python.V_ion.superpose_atomic_density` interpolates."""

    from parsec_python.Forces.periodic import _valence_density_table

    return _valence_density_table(potential, specification)


# ---------------------------------------------------------------------------
# Device sums.


def _upload_tables(cp, symbols, tables, potentials, parameters):
    radii, values, offsets, counts = [], [], [], []
    cursor = 0
    for symbol in symbols:
        samples = np.asarray(potentials[symbol].radii, dtype=np.float64)
        radii.append(samples)
        values.append(np.asarray(tables[symbol], dtype=np.float64))
        offsets.append(cursor)
        counts.append(samples.size)
        cursor += samples.size
    return {
        "radii": cp.asarray(np.concatenate(radii)),
        "tables": cp.asarray(np.concatenate(values)),
        "offsets": cp.asarray(np.asarray(offsets, dtype=np.int32)),
        "counts": cp.asarray(np.asarray(counts, dtype=np.int32)),
        "parameters": cp.asarray(np.asarray(parameters, dtype=np.float64).reshape(-1)),
    }


def _radial_field(cp, grid, atoms, symbols, potentials, tables, parameters, mode: int) -> np.ndarray:
    with cp.cuda.Device(device_id(cp)):
        return _radial_field_here(cp, grid, atoms, symbols, potentials, tables, parameters, mode)


def _radial_field_here(cp, grid, atoms, symbols, potentials, tables, parameters, mode: int) -> np.ndarray:
    coordinates = np.ascontiguousarray(grid.coordinates, dtype=np.float64)
    _, sides, _ = _setup._axis_spacing(grid)
    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    kinds = np.array([symbols.index(atom.symbol) for atom in atoms], dtype=np.int32)
    uploaded = _upload_tables(cp, symbols, tables, potentials, parameters)
    points = int(coordinates.shape[0])
    field = cp.zeros(points, dtype=cp.float64)
    if len(atoms):
        _field_kernel(cp)(
            ((points + _THREADS - 1) // _THREADS,),
            (_THREADS,),
            (
                np.int64(points),
                cp.asarray(coordinates),
                np.int32(len(atoms)),
                cp.asarray(positions.reshape(-1)),
                cp.asarray(kinds),
                cp.asarray(np.asarray(sides, dtype=np.float64)),
                uploaded["radii"],
                uploaded["tables"],
                uploaded["offsets"],
                uploaded["counts"],
                uploaded["parameters"],
                np.int32(mode),
                field,
            ),
        )
    return np.asarray(cp.asnumpy(field), dtype=np.float64)


def device_local_ionic_potential(grid, atoms, potentials, specifications, lattice_vectors, eta, tolerance):
    """The local ionic potential of :func:`.periodic_setup.fft_local_ionic_potential`, short range on the device.

    A provider of :func:`parsec_python.V_ion.ewald.set_local_potential_provider`: ``None`` for a
    cell or a species it does not take (the next provider or the readable sum then runs).
    """

    cp = _cupy()
    symbols = list(dict.fromkeys(atom.symbol for atom in atoms))
    if cp is None or not handles(grid, lattice_vectors) or not linear_species(specifications, symbols):
        return None
    split = split_parameters(grid, lattice_vectors, tolerance, eta)
    n, sides, _ = _setup._axis_spacing(grid)
    volume = float(np.prod(sides))
    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)
    total = _setup._structure_factor_series(grid, positions, charges, split.eta, tolerance)

    tables, parameters = {}, []
    for symbol in symbols:
        potential = potentials[symbol]
        local_l = specifications[symbol].local_angular_momentum
        potential.validate_local_channel(local_l)
        values = np.asarray(potential.channel_potentials[local_l], dtype=np.float64)
        tables[symbol] = np.asarray(potential.radii, dtype=np.float64) * values
        parameters.append(
            [
                potential.radial_parameter_a,
                potential.radial_parameter_b,
                float(potential.radii[0]),
                float(potential.interpolation_cutoff),
                local_reach(potential, local_l, split),
                2.0 * float(potential.ionic_charge),
                split.eta,
                float(values[0]),
            ]
        )
    total += _radial_field(cp, grid, atoms, symbols, potentials, tables, parameters, mode=0)
    # The G = 0 term of the reference eta (periodic_setup.fft_local_ionic_potential).
    total += -(2.0 * np.pi * float(np.sum(charges)) / volume) * (
        1.0 / split.reference_eta**2 - 1.0 / split.eta**2
    )
    return total


def lattice_handles(grid) -> bool:
    """Whether :func:`device_lattice_atomic_density` takes the grid: the general (oblique) grid of the
    Fortran-derived routines, whose point ``(i, j, k)`` lies at ``sum_m (i_m + s_m) a_m/n_m``."""

    from parsec_python.Grid.pbc import uses_fortran_routines

    return bool(
        getattr(grid, "is_general_periodic", False)
        and not getattr(grid, "is_confined_periodic", False)
        and uses_fortran_routines(grid)
    )


def device_lattice_atomic_density(grid, atoms, potentials, specifications, lattice_vectors, *, core: bool = False):
    """The superposed atomic density of an oblique cell (:func:`lattice_handles`) on the device, or ``None``.

    The sum of :func:`parsec_python.V_ion.ionic_potential.superpose_atomic_density` for such a grid: every image
    within the support radius of the species (``atomic_density_support_radius``, ``r < support``), the table
    interpolated linearly with its first value continued inwards.  One thread per grid point adds the atoms and
    their images, so the field equals the host's to round-off (1e-16 relative) instead of bit for bit.  ``None``
    without a usable device, for another grid or a ``Cubic_Spline`` core density (the host sum then runs).
    """

    if lattice_vectors is None or not lattice_handles(grid):
        return None
    cp = _cupy()
    symbols = list(dict.fromkeys(atom.symbol for atom in atoms))
    if cp is None:
        return None
    if core and any(getattr(specifications[symbol], "use_spline", False) for symbol in symbols):
        return None
    radii, values, offsets, counts, parameters = [], [], [], [], []
    cursor = 0
    for symbol in symbols:
        potential = potentials[symbol]
        specification = specifications[symbol]
        samples = np.asarray(potential.radii, dtype=np.float64)
        if core:
            has_core = bool(potential.has_nonlinear_core_correction)
            table = np.asarray(potential.core_density, dtype=np.float64) if has_core else np.zeros_like(samples)
            reach = _ionic.atomic_density_support_radius(potential, specification, core=True) if has_core else 0.0
        else:
            table = np.asarray(valence_density_table(potential, specification), dtype=np.float64)
            reach = _ionic.atomic_density_support_radius(potential, specification, core=False)
        radii.append(samples)
        values.append(table)
        offsets.append(cursor)
        counts.append(samples.size)
        cursor += samples.size
        parameters.append(
            [
                potential.radial_parameter_a,
                potential.radial_parameter_b,
                float(samples[0]),
                float(potential.interpolation_cutoff),
                float(reach),
                0.0,
                0.0,
                float(table[0]),
            ]
        )
    lattice = np.ascontiguousarray(np.asarray(lattice_vectors, dtype=np.float64))
    inverse = np.ascontiguousarray(np.linalg.inv(lattice))
    coordinates = np.ascontiguousarray(grid.coordinates, dtype=np.float64)
    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms]).reshape(-1)
    kinds = np.array([symbols.index(atom.symbol) for atom in atoms], dtype=np.int32)
    points = int(coordinates.shape[0])
    with cp.cuda.Device(device_id(cp)):
        field = cp.zeros(points, dtype=cp.float64)
        if len(atoms):
            _lattice_density_kernel(cp)(
                ((points + _THREADS - 1) // _THREADS,),
                (_THREADS,),
                (
                    np.int64(points),
                    cp.asarray(coordinates),
                    np.int32(len(atoms)),
                    cp.asarray(positions),
                    cp.asarray(kinds),
                    cp.asarray(lattice.reshape(-1)),
                    cp.asarray(inverse.reshape(-1)),
                    cp.asarray(np.linalg.norm(inverse, axis=0)),
                    cp.asarray(np.concatenate(radii)),
                    cp.asarray(np.concatenate(values)),
                    cp.asarray(np.asarray(offsets, dtype=np.int32)),
                    cp.asarray(np.asarray(counts, dtype=np.int32)),
                    cp.asarray(np.asarray(parameters, dtype=np.float64).reshape(-1)),
                    field,
                ),
            )
        return np.asarray(cp.asnumpy(field), dtype=np.float64)


def device_atomic_density(grid, atoms, potentials, specifications, lattice_vectors, *, core: bool = False):
    """The superposed atomic (valence or core) density of a periodic cell, summed on the device.

    A provider of :func:`parsec_python.V_ion.ionic_potential.set_atomic_density_provider`:
    ``None`` for what it does not take (an isolated or another periodic grid, a ``Cubic_Spline``
    core density).  An oblique cell of the Fortran-derived routines takes
    :func:`device_lattice_atomic_density`.
    """

    if lattice_vectors is None:
        return None
    if lattice_handles(grid):
        return device_lattice_atomic_density(grid, atoms, potentials, specifications, lattice_vectors, core=core)
    cp = _cupy()
    symbols = list(dict.fromkeys(atom.symbol for atom in atoms))
    if cp is None or not handles(grid, lattice_vectors):
        return None
    if core and any(getattr(specifications[symbol], "use_spline", False) for symbol in symbols):
        return None
    tables, parameters = {}, []
    for symbol in symbols:
        potential = potentials[symbol]
        specification = specifications[symbol]
        if core:
            table = (
                np.asarray(potential.core_density, dtype=np.float64)
                if potential.has_nonlinear_core_correction
                else np.zeros_like(np.asarray(potential.radii, dtype=np.float64))
            )
        else:
            table = np.asarray(valence_density_table(potential, specification), dtype=np.float64)
        tables[symbol] = table
        reach = density_reach(potential, table) if np.any(table) else 0.0
        parameters.append(
            [
                potential.radial_parameter_a,
                potential.radial_parameter_b,
                float(potential.radii[0]),
                float(potential.interpolation_cutoff),
                reach,
                0.0,
                0.0,
                float(table[0]),
            ]
        )
    return _radial_field(cp, grid, atoms, symbols, potentials, tables, parameters, mode=1)


def _device_density_provider(grid, atoms, potentials, specifications, lattice_vectors, core):
    return device_atomic_density(grid, atoms, potentials, specifications, lattice_vectors, core=core)


def _device_or_fft(grid, atoms, potentials, specifications, lattice_vectors, eta, tolerance):
    found = device_local_ionic_potential(
        grid, atoms, potentials, specifications, lattice_vectors, eta, tolerance
    )
    if found is not None:
        return found
    return _setup.fft_local_ionic_potential(
        grid, atoms, potentials, specifications, lattice_vectors, eta, tolerance
    )


@contextmanager
def device_periodic_preparation():
    """Build the local potential and the atomic densities of periodic cells on the device.

    For the calculation inside the ``with`` block: :func:`device_local_ionic_potential` (else the
    FFT potential of :mod:`.periodic_setup`) and :func:`device_atomic_density` (else the readable
    sum) replace the installed builders, which are restored afterwards.  Without a usable device
    nothing changes.
    """

    previous_potential = (_ewald._local_potential_provider, _ewald._provider_converged_for)
    previous_density = _ionic._atomic_density_provider
    if _cupy() is None:
        yield False
        return
    _ewald.set_local_potential_provider(_device_or_fft, converged_for=_setup._fft_handles)
    _ionic.set_atomic_density_provider(_device_density_provider)
    try:
        yield True
    finally:
        _ewald.set_local_potential_provider(previous_potential[0], converged_for=previous_potential[1])
        _ionic.set_atomic_density_provider(previous_density)


def device_preparation_active() -> bool:
    """Whether :func:`device_periodic_preparation` is in force."""

    return _ewald._local_potential_provider is _device_or_fft


__all__ = [
    "COULOMB_TAIL_TOLERANCE",
    "DENSITY_TAIL_FRACTION",
    "SplitParameters",
    "coulomb_radius",
    "density_reach",
    "device_atomic_density",
    "device_lattice_atomic_density",
    "device_local_ionic_potential",
    "device_periodic_preparation",
    "device_preparation_active",
    "handles",
    "lattice_handles",
    "local_reach",
    "split_parameters",
]
