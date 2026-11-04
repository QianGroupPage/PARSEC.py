"""The Tkatchenko--Scheffler correction with its grid and pair sums on a GPU.

The physics is that of :mod:`parsec_python.Energy.dispersion`
(``initchrg_f.f90`` and ``efvdw.f90``).  What costs time there are the
Hirshfeld volumes: every atom's free density is sampled on every grid point,
twice (normalization and weight denominator, then the effective volumes),
which for a cluster of a few hundred atoms on a grid of a million points is
most of a minute of NumPy.  Here the same sums run on one device, a slab of
grid points at a time:

1. ``S_a = sum_i rho_a(|r_i - R_a|)`` gives the normalization
   ``s_a = Z_a / (S_a h^3)`` and ``V_free,a = s_a sum_i d^3 rho_a h^3``;
2. per point, the denominator ``sum_a s_a rho_a`` and the weight
   ``w_i = rho_i / sum_a s_a rho_a`` (zero where no free density reaches);
3. ``V_eff,a = sum_i d^3 s_a rho_a w_i h^3``.

``rho_a`` is the free-atom valence density of the initial guess, linear in
the radial table with the value of the first sample below it and zero from
``rs(ns-1)`` on (``initchrg.f90``), as the reference samples it.  The pair sum
of ``efvdw.f90`` runs on the device too, a block of atom rows at a time
(:func:`device_ts_pair_energy`): the host never holds the ``atoms x atoms x
3`` displacements of the reference sum.  The sums are those of the reference
in another order: volumes, energy and forces agree to round-off.

A periodic cell (``Boundary_Conditions: bulk``) takes the rules of the
reference: the grid points and the atoms are moved into the cell centred on
the origin, every atom has one free density per Hirshfeld cell ``L``
(``Hf_One`` ..., centred at ``R_a - L``), and the pair sum adds the lattice
shells of ``efvdw.f90`` until a shell adds less than 7.3e-8 Ry (that shell
included).

The TS correction is not self-consistent in PARSEC (``efvdw`` runs after a
converged loop and adds no potential), so it adds nothing to the SCF loop.
Units: bohr, e/bohr^3, Ry.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from parsec_python.Energy.dispersion import (
    DAMPING_RADIUS_SCALE,
    DAMPING_STEEPNESS,
    PERIODIC_SHELL_TOLERANCE,
    DispersionResult,
    HirshfeldVolumes,
    _centred,
    free_atom_parameters,
    hirshfeld_translations,
    lattice_shell_steps,
    species_elements,
)

#: Grid points of one slab of the device sums.
_SLAB_POINTS = 1 << 21
#: Device bytes of the arrays of one block of atom rows of the pair sum.
_PAIR_BLOCK_BYTES = 256 * 1024 * 1024


def _free_atom_table(potential, specification) -> tuple[np.ndarray, np.ndarray, float]:
    """``(radii, density, cutoff)`` of one species, as the reference interpolates it."""

    from ..V_ion.native_ionic import _density_table

    radii = np.ascontiguousarray(potential.radii, dtype=np.float64)
    values, _use_spline = _density_table(potential, specification, core=False)
    return radii, np.ascontiguousarray(values, dtype=np.float64), float(potential.interpolation_cutoff)


def device_hirshfeld_volumes(
    grid,
    atoms: Sequence,
    potentials: Mapping,
    specifications: Mapping,
    density: np.ndarray,
    lattice_vectors: np.ndarray | None = None,
    cells: Sequence[int] = (0, 0, 0),
    *,
    device_id: int | None = None,
) -> HirshfeldVolumes:
    """:func:`parsec_python.Energy.dispersion.hirshfeld_volumes` on one device.

    ``lattice_vectors`` (rows) and ``cells`` as in the reference: the points and
    the atoms in the cell centred on the origin and one free density per
    Hirshfeld cell.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    for atom in atoms:
        if getattr(potentials[atom.symbol], "filtered_valence_density", None) is not None:
            raise NotImplementedError(
                "the device Hirshfeld volumes sample the unfiltered free-atom density"
            )
    coordinates = np.asarray(grid.coordinates, dtype=np.float64)
    translations = hirshfeld_translations(lattice_vectors, cells)
    if lattice_vectors is not None:
        coordinates = _centred(coordinates, lattice_vectors)
    density = np.asarray(density, dtype=np.float64)
    if density.shape != (coordinates.shape[0],):
        raise ValueError("the density does not match the grid")
    volume = float(grid.volume_element)
    count = len(atoms)
    tables = {}
    for atom in atoms:
        if atom.symbol not in tables:
            tables[atom.symbol] = _free_atom_table(
                potentials[atom.symbol], specifications[atom.symbol]
            )
    positions = np.array([atom.position for atom in atoms], dtype=np.float64).reshape(-1, 3)
    if lattice_vectors is not None:
        positions = np.array(
            [_centred(position, lattice_vectors) for position in positions], dtype=np.float64
        ).reshape(-1, 3)
    # One free density per atom and Hirshfeld cell, centred at R_a - L.
    images = np.array(
        [[position - translation for translation in translations] for position in positions],
        dtype=np.float64,
    )
    size = coordinates.shape[0]
    slabs = [(start, min(size, start + _SLAB_POINTS)) for start in range(0, size, _SLAB_POINTS)]
    device = int(cp.cuda.Device().id) if device_id is None else int(device_id)
    with cp.cuda.Device(device):
        device_tables = {
            symbol: (cp.asarray(radii), cp.asarray(values), cutoff)
            for symbol, (radii, values, cutoff) in tables.items()
        }
        device_images = cp.asarray(images)

        def sampled(points, index, image):
            """``(rho_a, d)`` of image ``image`` of atom ``index`` at the slab ``points``."""

            radii, values, cutoff = device_tables[atoms[index].symbol]
            offset = points - device_images[index, image]
            distance = cp.sqrt(cp.sum(offset * offset, axis=1))
            inside = distance < cutoff
            free = cp.where(
                inside,
                cp.interp(distance, radii, values, left=values[0]),
                0.0,
            )
            return free, cp.where(inside, distance, 0.0)

        # 1. Normalization and free volumes.
        charge = cp.zeros(count, dtype=cp.float64)
        moment = cp.zeros(count, dtype=cp.float64)
        for start, stop in slabs:
            points = cp.asarray(coordinates[start:stop])
            for index in range(count):
                for image in range(len(translations)):
                    free, distance = sampled(points, index, image)
                    charge[index] += cp.sum(free)
                    moment[index] += cp.sum(distance**3 * free)
            del points
        charge_host = np.asarray(cp.asnumpy(charge), dtype=np.float64) * volume
        for index, value in enumerate(charge_host):
            if value <= 0.0:
                atom = atoms[index]
                raise ValueError(
                    f"atom {index + 1} ({atom.symbol}) has no free-atom density on the grid"
                )
        scales = np.array(
            [potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64
        ) / charge_host
        free_volumes = scales * np.asarray(cp.asnumpy(moment), dtype=np.float64) * volume
        # 2. and 3. Weights and effective volumes, a slab at a time.
        effective = cp.zeros(count, dtype=cp.float64)
        for start, stop in slabs:
            points = cp.asarray(coordinates[start:stop])
            denominator = cp.zeros(stop - start, dtype=cp.float64)
            for index in range(count):
                for image in range(len(translations)):
                    free, _distance = sampled(points, index, image)
                    denominator += float(scales[index]) * free
            values = cp.asarray(density[start:stop])
            positive = denominator > 0.0
            weight = cp.where(
                positive, values / cp.where(positive, denominator, 1.0), 0.0
            )
            for index in range(count):
                for image in range(len(translations)):
                    free, distance = sampled(points, index, image)
                    effective[index] += cp.sum(
                        distance**3 * float(scales[index]) * free * weight
                    )
            del points, denominator, values, weight
        effective_volumes = np.asarray(cp.asnumpy(effective), dtype=np.float64) * volume
        del device_tables, device_images
    return HirshfeldVolumes(free=free_volumes, effective=effective_volumes)


def device_ts_pair_energy(
    positions: np.ndarray,
    c6: np.ndarray,
    polarizability: np.ndarray,
    vdw_radius: np.ndarray,
    lattice_vectors: np.ndarray | None = None,
    *,
    device_id: int | None = None,
) -> tuple[float, np.ndarray, int]:
    """The pair sum of ``efvdw.f90`` on a device: ``(energy Ry, forces Ry/bohr, shells)``.

    :func:`parsec_python.Energy.dispersion.ts_pair_energy`: the pairs of the
    cell, a block of atom rows at a time against all atoms, and with
    ``lattice_vectors`` the lattice shells ``max |n_i| = 1, 2, ...`` (all pairs,
    an atom with its images included), a block of translations at a time,
    until a shell adds less than ``PERIODIC_SHELL_TOLERANCE``.  No ``atoms x
    atoms x 3`` array is formed on the host; the same pair terms, summed in
    another order.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    count = positions.shape[0]
    if count < 2 and lattice_vectors is None:
        return 0.0, np.zeros((count, 3), dtype=np.float64), 0
    device = int(cp.cuda.Device().id) if device_id is None else int(device_id)
    rows = max(1, int(_PAIR_BLOCK_BYTES // (8 * 16 * count)))
    with cp.cuda.Device(device):
        xyz = cp.asarray(positions)
        c6_all = cp.asarray(c6, dtype=cp.float64)
        alpha = cp.asarray(polarizability, dtype=cp.float64)
        radius = cp.asarray(vdw_radius, dtype=cp.float64)
        energy = cp.zeros((), dtype=cp.float64)
        forces = cp.empty((count, 3), dtype=cp.float64)
        columns = cp.arange(count)
        for start in range(0, count, rows):
            stop = min(count, start + rows)
            ci, cj = c6_all[start:stop, None], c6_all[None, :]
            ai, aj = alpha[start:stop, None], alpha[None, :]
            pair_c6 = (2.0 * ci * cj) / (aj / ai * ci + ai / aj * cj)
            scaled_radius = DAMPING_RADIUS_SCALE * (radius[start:stop, None] + radius[None, :])
            displacement = xyz[start:stop, None, :] - xyz[None, :, :]
            squared = cp.sum(displacement * displacement, axis=2)
            off = cp.arange(start, stop)[:, None] != columns[None, :]
            # The diagonal (an atom with itself) takes a unit distance and no term.
            squared = cp.where(off, squared, 1.0)
            distance = cp.sqrt(squared)
            exponential = cp.exp(-DAMPING_STEEPNESS * (distance / scaled_radius - 1.0))
            damping = 1.0 / (1.0 + exponential)
            pair_energy = cp.where(off, -pair_c6 * damping / squared**3, 0.0)
            rfac = -6.0 / distance + DAMPING_STEEPNESS * exponential / (
                (1.0 + exponential) * scaled_radius
            )
            factor = pair_energy * rfac / distance
            energy += cp.sum(pair_energy)
            forces[start:stop] = -cp.sum(displacement * factor[:, :, None], axis=1)
        energy_host = 2.0 * (0.5 * float(cp.asnumpy(energy))) if count > 1 else 0.0
        shells = 0
        if lattice_vectors is not None:
            lattice = np.asarray(lattice_vectors, dtype=np.float64)
            pair_c6 = (2.0 * c6_all[:, None] * c6_all[None, :]) / (
                alpha[None, :] / alpha[:, None] * c6_all[:, None]
                + alpha[:, None] / alpha[None, :] * c6_all[None, :]
            )
            scaled_radius = DAMPING_RADIUS_SCALE * (radius[:, None] + radius[None, :])
            difference = xyz[:, None, :] - xyz[None, :, :]
            per_translation = 8 * 16 * count * count
            block = max(1, int(_PAIR_BLOCK_BYTES // per_translation))
            while True:
                shells += 1
                translations = lattice_shell_steps(shells) @ lattice
                shell_energy = cp.zeros((), dtype=cp.float64)
                for start in range(0, translations.shape[0], block):
                    shift = cp.asarray(translations[start : start + block])
                    # d = R_a - (R_b + L) for every translation L, a and b.
                    displacement = difference[None, :, :, :] - shift[:, None, None, :]
                    squared = cp.sum(displacement * displacement, axis=3)
                    distance = cp.sqrt(squared)
                    exponential = cp.exp(-DAMPING_STEEPNESS * (distance / scaled_radius - 1.0))
                    damping = 1.0 / (1.0 + exponential)
                    pair_energy = -pair_c6 * damping / squared**3
                    rfac = -6.0 / distance + DAMPING_STEEPNESS * exponential / (
                        (1.0 + exponential) * scaled_radius
                    )
                    factor = pair_energy * rfac / distance
                    shell_energy += cp.sum(pair_energy)
                    forces -= cp.sum(displacement * factor[..., None], axis=(0, 2))
                    del displacement, squared, distance, exponential, damping, pair_energy, rfac, factor
                shell_host = 2.0 * (0.5 * float(cp.asnumpy(shell_energy)))
                energy_host += shell_host
                if abs(shell_host) < PERIODIC_SHELL_TOLERANCE:
                    break
        forces_host = np.asarray(cp.asnumpy(forces), dtype=np.float64)
    # Hartree to Rydberg, as the reference: the energy above carries 2 * (1/2).
    return energy_host, 2.0 * forces_host, shells


def device_ts_dispersion(
    atoms, elements, volumes: HirshfeldVolumes, lattice_vectors: np.ndarray | None = None
) -> DispersionResult:
    """:func:`parsec_python.Energy.dispersion.ts_dispersion` with the device pair sum."""

    data = np.array([free_atom_parameters(element) for element in elements])
    ratio = volumes.ratio
    c6 = ratio**2 * data[:, 1]
    alpha = ratio * data[:, 0]
    radius = ratio ** (1.0 / 3.0) * data[:, 2]
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    energy, forces, shells = device_ts_pair_energy(positions, c6, alpha, radius, lattice_vectors)
    return DispersionResult(
        energy=energy,
        forces=forces,
        volumes=volumes,
        c6=c6,
        polarizability=alpha,
        vdw_radius=radius,
        shells=shells,
    )


def device_dispersion_correction(system, density: np.ndarray) -> DispersionResult:
    """``efvdw.f90`` of a prepared system and its converged valence density.

    The counterpart of :func:`parsec_python.Energy.dispersion.dispersion_correction`:
    a cluster, or a bulk cell with its Hirshfeld cells and lattice shells.
    """

    problem = system.input
    cell = getattr(problem, "periodic_cell", None)
    lattice = None if cell is None else cell.lattice_vectors
    if lattice is not None and getattr(system.grid, "is_confined_periodic", False):
        raise NotImplementedError(
            "the Tkatchenko-Scheffler correction of a wire or slab is not implemented"
        )
    volumes = device_hirshfeld_volumes(
        system.grid,
        system.atoms,
        system.pseudopotentials,
        problem.pseudopotentials,
        np.asarray(density, dtype=np.float64),
        lattice,
        problem.scf.hirshfeld_cells if lattice is not None else (0, 0, 0),
    )
    return device_ts_dispersion(
        system.atoms,
        species_elements(system.atoms, problem.pseudopotentials),
        volumes,
        lattice,
    )


def device_dispersion_in_use() -> bool:
    """Whether the reference periodic drivers form the TS sums on a device.

    They do where their k-point eigenproblems run on a CUDA device (the k-point
    backend of :mod:`parsec_python.acceleration.SCF.kpoint_parallel`), the GPU
    path of the reference periodic solver.
    """

    from ..SCF.kpoint_parallel import resolve_backend

    try:
        plan = resolve_backend()
    except (ValueError, RuntimeError):
        return False
    return plan is not None and plan.name == "cupy"


def install() -> None:
    """Make the device TS correction the one of the reference periodic drivers on a GPU."""

    from parsec_python.Energy.dispersion import set_device_dispersion

    set_device_dispersion(device_dispersion_correction, device_dispersion_in_use)


__all__ = [
    "device_dispersion_correction",
    "device_dispersion_in_use",
    "install",
    "device_hirshfeld_volumes",
    "device_ts_dispersion",
    "device_ts_pair_energy",
]
