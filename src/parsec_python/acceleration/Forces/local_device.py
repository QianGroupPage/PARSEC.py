"""Local-pseudopotential forces summed on a GPU.

The three grid integrals of :func:`parsec_python.Forces.local_forces` couple
every atom with every grid point: the Coulomb tail of the local potential has
no cutoff.  That is the cost of the ionic potential itself, and like it the
sums are formed on the device for a large structure.

One CUDA thread handles one atom and one chunk of consecutive grid points and
adds the three force terms of that chunk; the chunk sums of an atom are then
added in grid order.  The radial rules are those of the readable routine:
points closer than the first radial sample contribute nothing, the potential
slope is the Coulomb one from ``rs(npoint-6)`` outwards where the density
slopes are zero, and in between the tables of ``pseudo.f90`` are interpolated
linearly (``r**2 * dV/dr`` for the potential).  For a species with
``Cubic_Spline`` the slopes of the potential and of the core density are the
derivatives of PARSEC's padded cubic splines (``splint``): a point at or
beyond the first radial sample lies between two samples, so the kernel needs
the values and the second derivatives of the spline at the samples only.

The numbers differ from the readable routine by the order of the sums only.

The atoms are independent of one another, so several devices share them:
each device sums a contiguous part of the atom list over the whole grid.  An
atom is summed by one device in the same order whichever devices take part,
so the forces do not depend on their number.

A calculation with symmetry sectors sums over the wedge of the grid only
(:func:`cupy_wedge_local_forces`).  The three fields are invariant under the
group ``G`` of the sectors, so with ``n_w`` the number of grid points in the
orbit of wedge point ``w`` and ``K`` the vector kernel of a term,

``F_a = sum_r h(r) K(r - R_a) = sum_(g in G) g W(g^-1 R_a)``,
``W(R) = sum_w (n_w/|G|) h(w) K(w - R)``,

and ``g^-1 R_a`` is the position of an atom of the same species.  The wedge
sums ``W`` of all atoms cost ``|G|`` times less than the sums over the grid
and give the same forces up to the order of the additions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Sequence

import numpy as np

from parsec_python.Forces import LocalForceTerms, species_force_tables
from parsec_python.Grid import RealSpaceGrid
from parsec_python.Pseudopotential import ParsecPseudopotential, ParsecRadialSpline
from parsec_python.Pseudopotential.radial_derivatives import tail_start_radius
from parsec_python.models import Atom, SpeciesPotential

from ..backends.cupy import require_cupy


#: Grid points one CUDA thread sums for one atom.
CHUNK_POINTS = 2048

#: Device bytes of the chunk sums of one launch: bounds the atoms per launch.
_PARTIAL_BYTES = 256 * 1024 * 1024

_KERNEL = r"""
extern "C" __global__
void local_force_chunks(
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
    const double* radii,
    const double* local_table,
    const double* local_second,
    const double* valence_table,
    const double* core_table,
    const double* core_second,
    const int* table_offsets,
    const int* table_counts,
    const double* parameters,
    double* partial)
{
    const long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= (long long)atom_count * chunk_count) return;
    const int atom = (int)(task / chunk_count);
    const long long chunk = task % chunk_count;
    const int kind = species[atom];
    const double* r_table = radii + table_offsets[kind];
    const double* l_table = local_table + table_offsets[kind];
    const double* l_second = local_second + table_offsets[kind];
    const double* v_table = valence_table + table_offsets[kind];
    const double* c_table = core_table + table_offsets[kind];
    const double* c_second = core_second + table_offsets[kind];
    const int samples = table_counts[kind];
    const double scale_a = parameters[7 * kind + 0];
    const double scale_b = parameters[7 * kind + 1];
    const double inner = parameters[7 * kind + 2];
    const double tail = parameters[7 * kind + 3];
    const double two_z = parameters[7 * kind + 4];
    const bool has_core = parameters[7 * kind + 5] != 0.0;
    const bool splined = parameters[7 * kind + 6] != 0.0;
    const double ax = positions[3 * atom + 0];
    const double ay = positions[3 * atom + 1];
    const double az = positions[3 * atom + 2];

    double sums[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const long long first = chunk * chunk_points;
    long long last = first + chunk_points;
    if (last > point_count) last = point_count;
    for (long long point = first; point < last; ++point) {
        const double dx = coordinates[3 * point + 0] - ax;
        const double dy = coordinates[3 * point + 1] - ay;
        const double dz = coordinates[3 * point + 2] - az;
        const double r = sqrt(dx * dx + dy * dy + dz * dz);
        if (!(r >= inner)) continue;
        double potential_slope;
        double valence_slope = 0.0;
        double core_slope = 0.0;
        if (r >= tail) {
            potential_slope = two_z / (r * r);
        } else {
            long long index = (long long)(log1p(r / scale_a) / scale_b) - 1;
            if (index < 0) index = 0;
            if (index > samples - 2) index = samples - 2;
            while (index > 0 && r < r_table[index]) --index;
            while (index < samples - 2 && r >= r_table[index + 1]) ++index;
            const double width = r_table[index + 1] - r_table[index];
            const double step = (r - r_table[index]) / width;
            valence_slope = v_table[index]
                + step * (v_table[index + 1] - v_table[index]);
            if (splined) {
                // splint: y1 = dy/h - (3a^2-1)/6*h*y2(lo) + (3b^2-1)/6*h*y2(hi)
                const double left = (r_table[index + 1] - r) / width;
                const double lower = (3.0 * left * left - 1.0) / 6.0 * width;
                const double upper = (3.0 * step * step - 1.0) / 6.0 * width;
                potential_slope = (l_table[index + 1] - l_table[index]) / width
                    - lower * l_second[index] + upper * l_second[index + 1];
                if (has_core) {
                    core_slope = (c_table[index + 1] - c_table[index]) / width
                        - lower * c_second[index] + upper * c_second[index + 1];
                }
            } else {
                potential_slope = (l_table[index]
                    + step * (l_table[index + 1] - l_table[index])) / (r * r);
                if (has_core) {
                    core_slope = c_table[index]
                        + step * (c_table[index + 1] - c_table[index]);
                }
            }
        }
        const double inverse = 1.0 / r;
        const double ionic = potential_slope * density[point] * inverse;
        const double correction = valence_slope * residual[point] * inverse;
        const double core = core_slope * xc[point] * inverse;
        sums[0] += ionic * dx;
        sums[1] += ionic * dy;
        sums[2] += ionic * dz;
        sums[3] += correction * dx;
        sums[4] += correction * dy;
        sums[5] += correction * dz;
        sums[6] += core * dx;
        sums[7] += core * dy;
        sums[8] += core * dz;
    }
    double* target = partial + 9 * task;
    for (int term = 0; term < 9; ++term) target[term] = sums[term];
}
"""

_COMPILED: dict[int, object] = {}


def _kernel(cp):
    device = int(cp.cuda.Device().id)
    if device not in _COMPILED:
        kernel = cp.RawKernel(_KERNEL, "local_force_chunks", options=("--std=c++11",))
        kernel.compile()
        _COMPILED[device] = kernel
    return _COMPILED[device]


def cupy_local_forces(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    density: np.ndarray,
    potential_residual: np.ndarray,
    xc_potential: np.ndarray,
    *,
    device_id: int | None = None,
    device_ids: Sequence[int] | None = None,
    coordinates: np.ndarray | None = None,
) -> LocalForceTerms:
    """The three ``forcloc.f90`` terms, summed on CUDA devices.

    Arguments and result are those of
    :func:`parsec_python.Forces.local_forces`.  ``device_ids`` are the
    devices that share the atoms; without them ``device_id`` is the one
    device that sums, and ``None`` takes the current one.  ``coordinates``
    replaces the points of the grid by a subset of them; the three fields
    then hold one value per given point, with the weight of the point in the
    sum already applied.
    """

    cp, _ = require_cupy()
    count = len(atoms)
    if device_ids is None:
        device_ids = (int(cp.cuda.Device().id) if device_id is None else int(device_id),)
    device_ids = tuple(int(device) for device in device_ids)[: max(count, 1)]
    if not device_ids:
        raise ValueError("the local forces need at least one device")
    on_device = list(range(count))
    half_width = grid.settings.stencil_half_width
    # The spline of a species has ``half_width + 2`` artificial knots below
    # its first radial sample; the kernel reads the samples only.
    padding = half_width + 2

    symbols = list(dict.fromkeys(atom.symbol for atom in atoms))
    offsets, counts, parameters = [], [], []
    radii, valence_table = [], []
    local_table, local_second, core_table, core_second = [], [], [], []
    cursor = 0
    for symbol in symbols:
        potential = potentials[symbol]
        specification = specifications[symbol]
        tables = species_force_tables(potential, specification)
        samples = potential.radii
        offsets.append(cursor)
        counts.append(samples.size)
        cursor += samples.size
        radii.append(samples)
        valence_table.append(tables.valence_density_derivative)
        has_core = tables.core_density_derivative is not None
        zeros = np.zeros_like(samples)
        if specification.use_spline:
            local_values = potential.channel_potentials[
                specification.local_angular_momentum
            ]
            local_table.append(local_values)
            local_second.append(
                ParsecRadialSpline.from_positive_grid(
                    samples, local_values, half_width
                ).second_derivatives[padding:]
            )
            if has_core:
                core_table.append(potential.core_density)
                core_second.append(
                    ParsecRadialSpline.from_positive_grid(
                        samples, potential.core_density, half_width
                    ).second_derivatives[padding:]
                )
            else:
                core_table.append(zeros)
                core_second.append(zeros)
        else:
            local_table.append(tables.local_potential_derivative * samples * samples)
            local_second.append(zeros)
            core_table.append(tables.core_density_derivative if has_core else zeros)
            core_second.append(zeros)
        parameters.append(
            [
                potential.radial_parameter_a,
                potential.radial_parameter_b,
                samples[0],
                tail_start_radius(potential),
                2.0 * potential.ionic_charge,
                1.0 if has_core else 0.0,
                1.0 if specification.use_spline else 0.0,
            ]
        )

    positions = np.array(
        [atoms[index].position for index in on_device], dtype=np.float64
    )
    kinds = np.array(
        [symbols.index(atoms[index].symbol) for index in on_device], dtype=np.int32
    )
    coordinates = np.ascontiguousarray(
        grid.coordinates if coordinates is None else coordinates, dtype=np.float64
    )
    points = int(coordinates.shape[0])
    chunks = max(1, (points + CHUNK_POINTS - 1) // CHUNK_POINTS)
    batch = max(1, int(_PARTIAL_BYTES // (72 * chunks)))
    result = np.empty((len(on_device), 9), dtype=np.float64)
    fields = tuple(
        np.ascontiguousarray(values, dtype=np.float64)
        for values in (density, potential_residual, xc_potential)
    )
    if any(values.shape != (points,) for values in fields):
        raise ValueError("the fields of the local forces need one value per point")

    def direct_allocate(size):
        # Temporary device memory that bypasses the pool of the orbitals.
        return cp.cuda.MemoryPointer(cp.cuda.Memory(size), 0)

    def part(item):
        slot, device_number = item
        first = len(on_device) * slot // len(device_ids)
        last = len(on_device) * (slot + 1) // len(device_ids)
        if first == last:
            return
        with cp.cuda.Device(device_number), cp.cuda.using_allocator(direct_allocate):
            sum_atoms(first, last)

    def sum_atoms(first, last):
        kernel = _kernel(cp)
        device = {
            "coordinates": cp.asarray(coordinates),
            "density": cp.asarray(fields[0]),
            "residual": cp.asarray(fields[1]),
            "xc": cp.asarray(fields[2]),
            "radii": cp.asarray(np.concatenate(radii)),
            "local": cp.asarray(np.concatenate(local_table)),
            "local_second": cp.asarray(np.concatenate(local_second)),
            "valence": cp.asarray(np.concatenate(valence_table)),
            "core": cp.asarray(np.concatenate(core_table)),
            "core_second": cp.asarray(np.concatenate(core_second)),
            "offsets": cp.asarray(np.asarray(offsets, dtype=np.int32)),
            "counts": cp.asarray(np.asarray(counts, dtype=np.int32)),
            "parameters": cp.asarray(np.asarray(parameters, dtype=np.float64).reshape(-1)),
        }
        threads = 128
        for start in range(first, last, batch):
            stop = min(last, start + batch)
            atoms_here = stop - start
            partial = cp.empty((atoms_here, chunks, 9), dtype=cp.float64)
            tasks = atoms_here * chunks
            kernel(
                ((tasks + threads - 1) // threads,),
                (threads,),
                (
                    np.int64(points),
                    np.int32(CHUNK_POINTS),
                    np.int64(chunks),
                    np.int32(atoms_here),
                    device["coordinates"],
                    device["density"],
                    device["residual"],
                    device["xc"],
                    cp.asarray(positions[start:stop].reshape(-1)),
                    cp.asarray(kinds[start:stop]),
                    device["radii"],
                    device["local"],
                    device["local_second"],
                    device["valence"],
                    device["core"],
                    device["core_second"],
                    device["offsets"],
                    device["counts"],
                    device["parameters"],
                    partial,
                ),
            )
            result[start:stop] = cp.asnumpy(partial.sum(axis=1))
            del partial
        # Free on the owning device, before leaving its context.
        device.clear()

    if len(device_ids) == 1:
        part((0, device_ids[0]))
    else:
        # Compile once per device here, one after another, so that the device
        # threads find their kernels instead of compiling side by side.
        for device_number in dict.fromkeys(device_ids):
            with cp.cuda.Device(device_number):
                _kernel(cp)
        with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
            list(executor.map(part, enumerate(device_ids)))
    result *= grid.volume_element
    return LocalForceTerms(
        ionic=result[:, 0:3].copy(),
        self_consistency=result[:, 3:6].copy(),
        core_correction=result[:, 6:9].copy(),
    )


def sector_atom_images(reduction, atoms: Sequence[Atom], *, tolerance: float = 1.0e-6):
    """For every operation of ``reduction``, its matrix and where it finds each atom.

    Returns ``[(matrix, image), ...]``: ``image[a]`` is the atom at
    ``g^-1 R_a``, of the species of atom ``a``.  Raises ``ValueError`` when an
    operation does not map the atoms onto themselves: a calculation prepared
    with symmetry sectors holds for structures of that point group only.
    """

    from scipy.spatial import cKDTree

    operations = getattr(reduction, "operations", None)
    if operations is None:
        operations = [np.diag(signs) for signs in np.asarray(reduction.signs)]
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    symbols = np.array([atom.symbol for atom in atoms])
    tree = cKDTree(positions)
    images = []
    for operation in operations:
        matrix = np.asarray(operation, dtype=np.float64)
        # Row a of ``positions @ matrix`` is g^-1 R_a.
        distance, image = tree.query(positions @ matrix)
        if np.max(distance, initial=0.0) > tolerance or np.any(symbols[image] != symbols):
            raise ValueError(
                "the atoms do not have the symmetry of the sectors (a group of "
                f"{len(operations)} operations found for the structure the "
                "calculation was prepared with): keep the atoms on that group, as a "
                "relaxation with symmetry does, or prepare the calculation without symmetry"
            )
        images.append((matrix, image))
    return images


def cupy_wedge_local_forces(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    density,
    potential_residual,
    xc_potential,
    reduction,
    *,
    device_ids: Sequence[int] | None = None,
) -> LocalForceTerms:
    """The three terms from the symmetry wedge of the grid (see the module text).

    ``reduction`` is the orbit map of the symmetry sectors
    (:class:`parsec_python.acceleration.Symmetry.AxisReflectionReduction` or
    its signed-permutation form).  A field is a full-grid array or a scalar
    field on the wedge of this reduction.  The atoms must have the symmetry
    of the reduction, which a relaxation with symmetry keeps.
    """

    images = sector_atom_images(reduction, atoms)
    rows = np.asarray(reduction.representative_rows)
    weights = np.asarray(reduction.multiplicities, dtype=np.float64) / float(
        reduction.group_order
    )

    def on_wedge(field) -> np.ndarray:
        if getattr(field, "reduction", None) is reduction:
            return np.asarray(field.values, dtype=np.float64) * weights
        return np.asarray(field, dtype=np.float64)[rows] * weights

    partial = cupy_local_forces(
        grid,
        atoms,
        potentials,
        specifications,
        on_wedge(density),
        on_wedge(potential_residual),
        on_wedge(xc_potential),
        device_ids=device_ids,
        coordinates=np.asarray(grid.coordinates)[rows],
    )
    terms = (partial.ionic, partial.self_consistency, partial.core_correction)
    totals = [np.zeros_like(term) for term in terms]
    for matrix, image in images:
        # The atom found at g^-1 R_a contributes g W, a row of ``W @ matrix.T``.
        for total, term in zip(totals, terms):
            total += term[image] @ matrix.T
    return LocalForceTerms(
        ionic=totals[0], self_consistency=totals[1], core_correction=totals[2]
    )


def local_force_sums_setting() -> str:
    """``PARSEC_LOCAL_FORCE_SUMS``: ``wedge`` (default) or ``full``.

    ``full`` sums the local forces over the whole grid also where symmetry
    sectors would allow the wedge.
    """

    import os

    value = os.environ.get("PARSEC_LOCAL_FORCE_SUMS", "wedge").strip().lower() or "wedge"
    if value not in {"wedge", "full"}:
        raise ValueError("PARSEC_LOCAL_FORCE_SUMS must be wedge or full")
    return value


def local_force_device_ids(system) -> tuple[int, ...]:
    """Devices of this process that share the local forces of ``system``.

    They are the devices that solve its symmetry sectors: each holds a CUDA
    context and orbitals already.  A calculation without sectors sums on the
    current device.  A number in ``PARSEC_IONIC_GPU_COUNT`` limits them as it
    limits the devices of the ionic set-up
    (:func:`parsec_python.acceleration.V_ion.cupy_ionic.ionic_device_ids`).
    """

    from ..V_ion.cupy_ionic import ionic_gpu_count_setting

    cp, _ = require_cupy()
    sectors = getattr(system.backend, "symmetry_eigensolver", None)
    devices = tuple(int(device) for device in getattr(sectors, "device_ids", ()) or ())
    devices = tuple(dict.fromkeys(devices)) or (int(cp.cuda.Device().id),)
    setting = ionic_gpu_count_setting()
    if setting != "auto":
        try:
            limit = int(setting)
        except ValueError as error:
            raise ValueError(
                "PARSEC_IONIC_GPU_COUNT must be auto or a positive integer"
            ) from error
        if limit < 1:
            raise ValueError("PARSEC_IONIC_GPU_COUNT must be positive")
        devices = devices[:limit]
    return devices


__all__ = [
    "CHUNK_POINTS",
    "cupy_local_forces",
    "cupy_wedge_local_forces",
    "local_force_device_ids",
    "sector_atom_images",
    "local_force_sums_setting",
]
