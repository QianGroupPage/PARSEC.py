"""Forces of a periodic cell whose ionic fields were built on a GPU.

:mod:`parsec_python.Forces.periodic` differentiates the local potential and the densities of the
readable preparation (the converged Ewald sum, every image to the end of the radial tables).  The
forces here are the exact derivatives of the fields of
:mod:`parsec_python.acceleration.V_ion.periodic_device`, with the same split and the same reach:

``F_a = dV sum_r rho(r) [sum_T f_a'(|d_T|) d_T/|d_T|] + dV (8 pi/V) Z_a Im sum_G w_G G e^(-i G.R_a) Q(G)``

with ``f_a = V_a + 2 Z_a erf(eta r)/r`` over the images within its reach, ``d_T = r - R_a - T``,
``w_G = exp(-G^2/4 eta^2)/G^2`` on the G set of the Fourier series of the potential and
``Q(G) = sum_r rho(r) exp(i G.r)`` (one FFT of the density).  The core-correction term
``dV sum_r V_xc rho_c'`` and the self-consistency term of ``forcloc.f90``
``dV sum_r (V_out - V_in) rho_atom'`` are summed over the images within the reach of the
densities.  The slopes are those of the linear interpolants the fields sample, as in
:mod:`parsec_python.Forces.periodic`.  The short-ranged sums run in one CUDA kernel (one thread per
atom and chunk of grid points, the chunk sums of an atom added in grid order), the long-ranged sum
as one FFT on the device.

The forces of the accelerated Gamma-point path (:func:`compute_device_periodic_forces`) add the
Ewald ion-ion forces (:func:`parsec_python.Forces.ion_ion.ewald_ion_ion_forces`) and the
nonlocal term contracted where the orbitals are
(:mod:`parsec_python.acceleration.Forces.nonlocal_device`, with the projector gradients of the
lattice images), and are symmetrized as :func:`parsec_python.Forces.periodic.compute_periodic_forces`
symmetrizes them.  :func:`device_periodic_forces` installs the local terms as the ones the
readable periodic forces use for the calculation in its ``with`` block (the k-point path, whose
eigenproblems run on the device workers of the reference drivers).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from parsec_python.Forces import LocalForceTerms

from ..V_ion import periodic_device as _fields
from ..V_ion import periodic_setup_fortran as _setup

#: Grid points one CUDA thread sums for one atom.
CHUNK_POINTS = 256

#: Device bytes of the chunk sums of one launch: bounds the atoms per launch.
_PARTIAL_BYTES = 256 * 1024 * 1024

_THREADS = 128

_FORCE_KERNEL = _fields.RADIAL_HELPERS + r"""
__device__ __forceinline__ double erf_slope_over_r(const double two_z, const double r, const double eta)
{
    // g'(r)/r of g(r) = 2 Z erf(eta r)/r (Forces/periodic.py: _erf_over_r_slope_over_r)
    const double x = eta * r;
    const double two_over_sqrt_pi = 1.1283791670955126;
    if (x < 1.0e-2) {
        return two_z * eta * eta * eta * two_over_sqrt_pi
            * (-2.0 / 3.0 + 0.4 * x * x - x * x * x * x / 7.0);
    }
    return two_z * (x * two_over_sqrt_pi * exp(-x * x) - erf(x)) / (r * r * r);
}

__device__ __forceinline__ double table_slope(
    const double r, const double* radii, const double* table, const int samples,
    const double scale_a, const double scale_b)
{
    // slope of np.interp(r, radii, table): zero below the first sample
    if (r < radii[0]) return 0.0;
    const long long j = radial_segment(r, radii, samples, scale_a, scale_b);
    return (table[j + 1] - table[j]) / (radii[j + 1] - radii[j]);
}

extern "C" __global__
void periodic_local_force_chunks(
    const long long point_count,
    const int chunk_points,
    const long long chunk_count,
    const int atom_count,
    const double* coordinates,
    const double* density,
    const double* residual,
    const double* xc,
    const double* positions,
    const int* species,
    const double* sides,
    const double* radii,
    const double* local_rv,
    const double* valence,
    const double* core,
    const int* offsets,
    const int* counts,
    const double* parameters,
    double* partial)
{
    const long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= (long long)atom_count * chunk_count) return;
    const int atom = (int)(task / chunk_count);
    const long long chunk = task % chunk_count;
    const int kind = species[atom];
    const double* r_table = radii + offsets[kind];
    const double* rv_table = local_rv + offsets[kind];
    const double* v_table = valence + offsets[kind];
    const double* c_table = core + offsets[kind];
    const int samples = counts[kind];
    const double* p = parameters + 10 * kind;
    const double scale_a = p[0], scale_b = p[1], first_radius = p[2], cutoff = p[3];
    const double ionic_reach = p[4], valence_reach = p[5], core_reach = p[6];
    const double two_z = p[7], eta = p[8];
    double reach = ionic_reach;
    if (valence_reach > reach) reach = valence_reach;
    if (core_reach > reach) reach = core_reach;
    const double reach2 = reach * reach;
    const double ax = positions[3 * atom + 0];
    const double ay = positions[3 * atom + 1];
    const double az = positions[3 * atom + 2];
    const double lx = sides[0], ly = sides[1], lz = sides[2];

    double sums[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const long long first = chunk * chunk_points;
    long long last = first + chunk_points;
    if (last > point_count) last = point_count;
    for (long long point = first; point < last; ++point) {
        const double px = coordinates[3 * point + 0];
        const double py = coordinates[3 * point + 1];
        const double pz = coordinates[3 * point + 2];
        const double rho = density[point];
        const double dv = residual[point];
        const double vxc = xc[point];
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
                    if (!(r > 0.0) || r > reach) continue;
                    const double inverse = 1.0 / r;
                    if (r <= ionic_reach) {
                        double slope;
                        if (r >= cutoff) {
                            slope = two_z / (r * r);
                        } else if (r <= first_radius) {
                            slope = 0.0;
                        } else {
                            // d/dr of p(r)/r, p the linear interpolant of r V
                            const long long j = radial_segment(r, r_table, samples, scale_a, scale_b);
                            const double value = linear_value(r, r_table, rv_table, j);
                            const double derivative =
                                (rv_table[j + 1] - rv_table[j]) / (r_table[j + 1] - r_table[j]);
                            slope = (derivative - value / r) / r;
                        }
                        const double weight = rho * (slope * inverse + erf_slope_over_r(two_z, r, eta));
                        sums[0] += weight * dx;
                        sums[1] += weight * dy;
                        sums[2] += weight * dz;
                    }
                    if (r < valence_reach && r < cutoff) {
                        const double weight =
                            dv * table_slope(r, r_table, v_table, samples, scale_a, scale_b) * inverse;
                        sums[3] += weight * dx;
                        sums[4] += weight * dy;
                        sums[5] += weight * dz;
                    }
                    if (r < core_reach && r < cutoff) {
                        const double weight =
                            vxc * table_slope(r, r_table, c_table, samples, scale_a, scale_b) * inverse;
                        sums[6] += weight * dx;
                        sums[7] += weight * dy;
                        sums[8] += weight * dz;
                    }
                }
            }
        }
    }
    double* target = partial + 9 * task;
    for (int term = 0; term < 9; ++term) target[term] = sums[term];
}
"""

_COMPILED: dict[int, object] = {}


def _force_kernel(cp):
    device = int(cp.cuda.Device().id)
    if device not in _COMPILED:
        kernel = cp.RawKernel(_FORCE_KERNEL, "periodic_local_force_chunks", options=("--std=c++11",))
        kernel.compile()
        _COMPILED[device] = kernel
    return _COMPILED[device]


def takes(grid, lattice_vectors, specifications, atoms) -> bool:
    """Whether the device fields built the potential and densities of these atoms.

    The cell of :func:`parsec_python.acceleration.V_ion.periodic_device.handles`, the real-space
    local potential, linearly interpolated species and a usable device: exactly where
    :func:`~parsec_python.acceleration.V_ion.periodic_device.device_local_ionic_potential` and
    :func:`~parsec_python.acceleration.V_ion.periodic_device.device_atomic_density` return a field
    (inside :func:`~parsec_python.acceleration.V_ion.periodic_device.device_periodic_preparation`).
    """

    symbols = list(dict.fromkeys(atom.symbol for atom in atoms))
    return bool(
        getattr(grid.settings, "local_potential", "real_space") == "real_space"
        and _fields.handles(grid, lattice_vectors)
        and _fields.linear_species(specifications, symbols)
        and _fields._cupy() is not None
    )


def _long_range_forces(cp, grid, positions, charges, eta, tolerance, density) -> np.ndarray:
    """The force of the Fourier series of :func:`.periodic_setup._structure_factor_series`, on the device."""

    n, sides, _ = _setup._axis_spacing(grid)
    volume = float(np.prod(sides))
    cutoff = 2.0 * eta * np.sqrt(-np.log(tolerance))
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

    spectrum = cp.fft.fftn(cp.asarray(np.asarray(density, dtype=np.float64).reshape(tuple(int(v) for v in n))))
    index = [cp.asarray(np.mod(frequencies[axis], n[axis])) for axis in range(3)]
    # Q(G) = sum_r rho(r) exp(i G.r) = conj(fftn(rho)) at the frequencies of G.
    q = cp.conj(spectrum[cp.ix_(index[0], index[1], index[2])])
    amplitude = cp.asarray(weight) * q
    gx, gy, gz = (cp.asarray(axis) for axis in g_axes)
    forces = np.zeros((len(positions), 3), dtype=np.float64)
    for atom, (position, charge) in enumerate(zip(positions, charges)):
        px = cp.exp(-1j * gx * float(position[0]))
        py = cp.exp(-1j * gy * float(position[1]))
        pz = cp.exp(-1j * gz * float(position[2]))
        along_z = amplitude @ pz  # (x, y)
        along_x = cp.tensordot(px, amplitude, axes=(0, 0))  # (y, z)
        sums = cp.stack(
            (
                cp.sum(gx * px * (along_z @ py)),
                cp.sum(gy * py * (px @ along_z)),
                cp.sum(gz * pz * (py @ along_x)),
            )
        )
        forces[atom] = (8.0 * np.pi / volume) * float(charge) * np.imag(cp.asnumpy(sums))
    return forces * grid.volume_element


def device_periodic_local_forces(
    grid,
    atoms: Sequence,
    potentials: Mapping,
    specifications: Mapping,
    lattice_vectors,
    density,
    potential_residual,
    xc_potential,
    *,
    tolerance: float = _fields.EWALD_TOLERANCE,
) -> LocalForceTerms | None:
    """The three local terms of :func:`parsec_python.Forces.periodic.periodic_local_forces`, for the device fields.

    The derivatives of the potential of
    :func:`~parsec_python.acceleration.V_ion.periodic_device.device_local_ionic_potential` and of
    the densities of :func:`~parsec_python.acceleration.V_ion.periodic_device.device_atomic_density`
    (see the module text); ``None`` where those builders do not take the cell (:func:`takes`).
    """

    if not takes(grid, lattice_vectors, specifications, atoms):
        return None
    cp = _fields._cupy()
    with cp.cuda.Device(_fields.device_id(cp)):
        return _device_local_forces(
            cp,
            grid,
            atoms,
            potentials,
            specifications,
            lattice_vectors,
            density,
            potential_residual,
            xc_potential,
            tolerance,
        )


def _device_local_forces(
    cp, grid, atoms, potentials, specifications, lattice_vectors, density, potential_residual, xc_potential, tolerance
) -> LocalForceTerms:
    count = len(atoms)
    zeros = np.zeros((count, 3))
    if count == 0:
        return LocalForceTerms(zeros, zeros.copy(), zeros.copy())
    fields = tuple(
        np.ascontiguousarray(values, dtype=np.float64)
        for values in (density, potential_residual, xc_potential)
    )
    if any(values.shape != (grid.size,) for values in fields):
        raise ValueError("the fields of the local forces need one value per grid point")
    split = _fields.split_parameters(grid, lattice_vectors, tolerance)
    symbols = list(dict.fromkeys(atom.symbol for atom in atoms))
    residual_present = bool(np.any(fields[1] != 0.0))
    radii, local_rv, valence, core, offsets, counts, parameters = [], [], [], [], [], [], []
    cursor = 0
    for symbol in symbols:
        potential = potentials[symbol]
        specification = specifications[symbol]
        local_l = specification.local_angular_momentum
        samples = np.asarray(potential.radii, dtype=np.float64)
        radii.append(samples)
        local_rv.append(samples * np.asarray(potential.channel_potentials[local_l], dtype=np.float64))
        valence_table = np.asarray(_fields.valence_density_table(potential, specification), dtype=np.float64)
        valence.append(valence_table)
        has_core = bool(potential.has_nonlinear_core_correction)
        core_table = (
            np.asarray(potential.core_density, dtype=np.float64) if has_core else np.zeros_like(samples)
        )
        core.append(core_table)
        offsets.append(cursor)
        counts.append(samples.size)
        cursor += samples.size
        parameters.append(
            [
                potential.radial_parameter_a,
                potential.radial_parameter_b,
                float(samples[0]),
                float(potential.interpolation_cutoff),
                _fields.local_reach(potential, local_l, split),
                _fields.density_reach(potential, valence_table) if residual_present else 0.0,
                _fields.density_reach(potential, core_table) if has_core else 0.0,
                2.0 * float(potential.ionic_charge),
                split.eta,
                0.0,
            ]
        )
    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    kinds = np.array([symbols.index(atom.symbol) for atom in atoms], dtype=np.int32)
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)
    coordinates = np.ascontiguousarray(grid.coordinates, dtype=np.float64)
    _, sides, _ = _setup._axis_spacing(grid)
    points = int(coordinates.shape[0])
    chunks = max(1, (points + CHUNK_POINTS - 1) // CHUNK_POINTS)
    batch = max(1, int(_PARTIAL_BYTES // (72 * chunks)))
    result = np.empty((count, 9), dtype=np.float64)
    device = {
        "coordinates": cp.asarray(coordinates),
        "density": cp.asarray(fields[0]),
        "residual": cp.asarray(fields[1]),
        "xc": cp.asarray(fields[2]),
        "sides": cp.asarray(np.asarray(sides, dtype=np.float64)),
        "radii": cp.asarray(np.concatenate(radii)),
        "local": cp.asarray(np.concatenate(local_rv)),
        "valence": cp.asarray(np.concatenate(valence)),
        "core": cp.asarray(np.concatenate(core)),
        "offsets": cp.asarray(np.asarray(offsets, dtype=np.int32)),
        "counts": cp.asarray(np.asarray(counts, dtype=np.int32)),
        "parameters": cp.asarray(np.asarray(parameters, dtype=np.float64).reshape(-1)),
    }
    kernel = _force_kernel(cp)
    for start in range(0, count, batch):
        stop = min(count, start + batch)
        here = stop - start
        partial = cp.empty((here, chunks, 9), dtype=cp.float64)
        tasks = here * chunks
        kernel(
            ((tasks + _THREADS - 1) // _THREADS,),
            (_THREADS,),
            (
                np.int64(points),
                np.int32(CHUNK_POINTS),
                np.int64(chunks),
                np.int32(here),
                device["coordinates"],
                device["density"],
                device["residual"],
                device["xc"],
                cp.asarray(positions[start:stop].reshape(-1)),
                cp.asarray(kinds[start:stop]),
                device["sides"],
                device["radii"],
                device["local"],
                device["valence"],
                device["core"],
                device["offsets"],
                device["counts"],
                device["parameters"],
                partial,
            ),
        )
        result[start:stop] = cp.asnumpy(partial.sum(axis=1))
        del partial
    device.clear()
    result *= grid.volume_element
    ionic = result[:, 0:3] + _long_range_forces(
        cp, grid, positions, charges, split.eta, tolerance, fields[0]
    )
    return LocalForceTerms(
        ionic=ionic,
        self_consistency=result[:, 3:6].copy(),
        core_correction=result[:, 6:9].copy(),
    )


def _local_force_provider(grid, atoms, potentials, specifications, lattice_vectors, density, residual, xc):
    return device_periodic_local_forces(
        grid, atoms, potentials, specifications, lattice_vectors, density, residual, xc
    )


@contextmanager
def device_periodic_forces():
    """Build the fields of periodic cells on the device and differentiate them there.

    For the calculation inside the ``with`` block:
    :func:`parsec_python.acceleration.V_ion.periodic_device.device_periodic_preparation` and
    :func:`device_periodic_local_forces` as the local terms of
    :func:`parsec_python.Forces.periodic.periodic_local_forces` (where it takes the cell).
    Without a usable device nothing changes.
    """

    from parsec_python.Forces import periodic as _periodic

    previous = _periodic._local_force_provider
    with _fields.device_periodic_preparation() as active:
        if not active:
            yield False
            return
        _periodic.set_local_force_provider(_local_force_provider)
        try:
            yield True
        finally:
            _periodic.set_local_force_provider(previous)


def periodic_ionic_scope(translation):
    """:func:`device_periodic_forces` for an input with forces or moving atoms of a periodic cell.

    What the accelerated command line enters for such an input, on its own Gamma-point path and on
    the reference solver it hands the rest to (k-points, spin, ...); a context that changes nothing
    for every other input, so that their fields stay those of the readable builders bit for bit.
    """

    from contextlib import nullcontext

    problem = translation.problem
    moving = bool(
        getattr(translation, "compute_forces", False)
        or getattr(translation, "relaxation", None) is not None
        or getattr(translation, "molecular_dynamics", None) is not None
    )
    if problem.periodic_cell is None or not moving:
        return nullcontext(False)
    return device_periodic_forces()


def with_periodic_operations(backend):
    """An accelerated :class:`parsec_python.Relaxation.IonicBackend` that keeps a periodic structure on its group.

    The operations of :func:`parsec_python.Forces.periodic.periodic_force_operations` (as
    :func:`parsec_python.Relaxation.periodic_backend` at Gamma) instead of the point group of
    the atoms of an isolated system.
    """

    from parsec_python.Forces.periodic import periodic_force_operations

    return replace(backend, symmetry_operations=lambda system: periodic_force_operations(system, None))


# ---------------------------------------------------------------------------
# The accelerated Gamma-point path.


def periodic_projector_gradients(system):
    """Gradients of the projectors (with their lattice images) of a prepared periodic cell at Gamma.

    :func:`parsec_python.Forces.periodic.build_periodic_projector_gradients` with the slopes of
    ``PARSEC_PERIODIC_FORCE_SLOPES``, as :func:`~parsec_python.Forces.periodic.compute_periodic_forces`
    takes them.
    """

    from parsec_python.Forces.periodic import build_periodic_projector_gradients, radial_slope_setting

    return build_periodic_projector_gradients(
        system.grid,
        system.atoms,
        system.pseudopotentials,
        system.input.pseudopotentials,
        system.input.periodic_cell.lattice_vectors,
        radial_slopes=radial_slope_setting(),
    )


def update_periodic_reference(system, atoms):
    """The prepared periodic cell of moved atoms (:func:`parsec_python.SCF.pbc.update_periodic_geometry`).

    Inside :func:`device_periodic_forces` its local potential and densities are the device fields.
    """

    from parsec_python.SCF.pbc import update_periodic_geometry

    return update_periodic_geometry(system, atoms)


def compute_device_periodic_forces(system, result, *, symmetrize: bool = True, operations=None):
    """Forces of a finished accelerated SCF loop of a periodic cell at Gamma.

    The terms and their order are those of
    :func:`parsec_python.Forces.periodic.compute_periodic_forces`: Ewald ion-ion and local
    (inside :func:`device_periodic_forces` :func:`device_periodic_local_forces` of the device
    fields, outside it the readable terms of the readable fields), the nonlocal term from the
    orbitals on the device
    (:func:`parsec_python.acceleration.Forces.compute._nonlocal_forces` with the projector
    gradients of the lattice images), LDA+U and TS terms
    (:func:`parsec_python.acceleration.Forces.compute.correction_forces`: the LDA+U term from
    the orbitals on the device with the projector gradients of the cell images,
    ``Hamiltonian/hubbard_device.py``, which an ionic loop keeps there), symmetrization over
    :func:`parsec_python.Forces.periodic.periodic_force_operations`, net force removed.
    """

    from parsec_python.Forces import assemble_forces
    from parsec_python.Forces.ion_ion import ewald_ion_ion_forces
    from parsec_python.Forces.periodic import (
        periodic_force_fields,
        periodic_force_operations,
        periodic_local_forces,
    )
    from .compute import _nonlocal_forces, correction_forces

    if getattr(system.grid, "is_confined_periodic", False) or getattr(
        system.grid, "is_general_periodic", False
    ):
        raise NotImplementedError(
            "the accelerated forces of a periodic cell take the orthorhombic grid of a bulk "
            "cell; wires, slabs and Fortran's general grid run on the reference solver"
        )
    started = perf_counter()
    clock: dict[str, float] = {}
    lattice = np.asarray(system.input.periodic_cell.lattice_vectors, dtype=np.float64)
    atoms = system.atoms
    specifications = system.input.pseudopotentials
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    repulsion = ewald_ion_ion_forces(atoms, system.pseudopotentials, lattice)
    clock["ion_ion"] = perf_counter() - started
    mark = perf_counter()
    density, residual, xc = periodic_force_fields(system, result)
    # The readable entry: inside device_periodic_forces it differentiates the
    # device fields on the device, outside the fields of the readable builders.
    local = periodic_local_forces(
        system.grid, atoms, system.pseudopotentials, specifications, lattice, density, residual, xc
    )
    clock["local"] = perf_counter() - mark
    mark = perf_counter()
    separable = _nonlocal_forces(system, result, clock)
    clock["nonlocal"] = perf_counter() - mark
    mark = perf_counter()
    if not symmetrize:
        operations = None
    elif operations is None:
        operations = periodic_force_operations(system, None)
    forces = assemble_forces(
        positions,
        repulsion,
        local,
        separable,
        operations,
        phase_seconds=clock,
        periodic=True,
        keep_torque=True,
        **correction_forces(system, result),
    )
    clock["symmetrization"] = perf_counter() - mark
    return replace(forces, seconds=perf_counter() - started)


__all__ = [
    "CHUNK_POINTS",
    "compute_device_periodic_forces",
    "device_periodic_forces",
    "device_periodic_local_forces",
    "takes",
]
