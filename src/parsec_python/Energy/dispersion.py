"""Tkatchenko-Scheffler dispersion correction (PARSEC ``efvdw.f90``).

``Van_Der_Waals: true`` adds to the total energy of a converged SCF loop the
pairwise energy of Tkatchenko and Scheffler, PRL 102, 073005 (2009), in
Hartree units

``E_vdW = -1/2 sum_(i != j) f(R_ij) C6_ij / R_ij**6``,
``f(R) = 1 / (1 + exp(-d (R/(s_R R0_ij) - 1)))``,  ``d = 20``, ``s_R = 0.94``,
``C6_ij = 2 C6_i C6_j / (alpha_j/alpha_i C6_i + alpha_i/alpha_j C6_j)``,
``R0_ij = R0_i + R0_j``,

with the free-atom ``alpha``, ``C6`` and ``R0`` of :data:`FREE_ATOM_DATA`
scaled by the Hirshfeld volume ratio ``nu_i = V_eff,i / V_free,i`` of each
atom: ``C6_i = nu_i**2 C6_free``, ``alpha_i = nu_i alpha_free``,
``R0_i = nu_i**(1/3) R0_free``.  The volumes are (``initchrg_f.f90``,
``efvdw.f90``)

``V_free,a = sum_L sum_r |r + L - R_a|**3 rho_a(|r + L - R_a|) h**3``,
``V_eff,a  = sum_L sum_r |r + L - R_a|**3 w_aL(r) rho(r) h**3``,
``w_aL(r)  = rho_a(|r + L - R_a|) / sum_(b,L') rho_b(|r + L' - R_b|)``,

where ``rho_a`` is the atomic valence pseudo-density of the starting guess
(the same table and interpolation as
:func:`parsec_python.V_ion.superpose_atomic_density`), normalized so that
its sum over the grid and the cells ``L`` is the ionic charge of the atom,
and ``rho`` is the converged valence density (no core density).  The cells
are the central one for a cluster and ``(2 Hf_One + 1) (2 Hf_Two + 1)
(2 Hf_Three + 1)`` cells of a periodic system.  The derivative of the
volumes with respect to the positions is not part of the forces, as in
``efvdw.f90`` (whose force expressions come from FHI-aims).

A periodic cell adds the shells ``n = 1, 2, ...`` of lattice translations
``L`` with ``max |L_k| = n`` (all pairs including an atom and its own
images), each with the factor 1/2; the sum stops after the first shell whose
energy is below 7.3e-8 Ry, which is still added.

Deviations from the Fortran, stated:

* ``parsec.F90`` builds the free-atom densities, distances and weights once,
  for the input structure (``initchrg_f`` before the first SCF loop), and
  ``efvdw`` keeps using them after atoms move.  Here they are built for the
  structure of each converged density, so a relaxation or a dynamics run
  sees the volumes of its own atoms.  A single point is the same.
* A grid point that no free-atom density reaches has the weight 0 here; in
  the Fortran it is ``0/0``.
* An element without free-atom data is refused (``efvdw.f90`` would go on
  with undefined values).

Energies are returned in Ry and forces in Ry/bohr (``forcnloc.f90`` adds
``2 * vdw_forces``, the Hartree forces of ``efvdw``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from ..models import Atom, SpeciesPotential


#: Free-atom polarizability (bohr^3), C6 (Ha bohr^6) and vdW radius (bohr)
#: of ``efvdw.f90``, extracted from its ``if (clust%name(itype)==...)``
#: chain; the literals there are double precision.
FREE_ATOM_DATA: dict[str, tuple[float, float, float]] = {
    "H": (4.5, 6.5, 3.1),
    "He": (1.38, 1.46, 2.65),
    "Li": (164.2, 1387.0, 4.16),
    "Be": (38.0, 214.0, 4.17),
    "B": (21.0, 99.5, 3.89),
    "C": (12.0, 46.6, 3.59),
    "N": (7.4, 24.2, 3.34),
    "O": (5.4, 15.6, 3.19),
    "F": (3.8, 9.52, 3.04),
    "Ne": (2.67, 6.38, 2.91),
    "Na": (162.7, 1556.0, 3.73),
    "Mg": (71.0, 627.0, 4.27),
    "Al": (60.0, 528.0, 4.33),
    "Si": (37.0, 305.0, 4.2),
    "P": (25.0, 185.0, 4.01),
    "S": (19.6, 134.0, 3.86),
    "Cl": (15.0, 94.6, 3.71),
    "Ar": (11.1, 64.3, 3.55),
    "K": (292.9, 3897.0, 3.71),
    "Ca": (160.0, 2221.0, 4.65),
    "Sc": (120.0, 1383.0, 4.59),
    "Ti": (98.0, 1044.0, 4.51),
    "V": (84.0, 832.0, 4.44),
    "Cr": (78.0, 602.0, 3.99),
    "Mn": (63.0, 552.0, 3.97),
    "Fe": (56.0, 482.0, 4.23),
    "Co": (50.0, 408.0, 4.18),
    "Ni": (48.0, 373.0, 3.82),
    "Cu": (42.0, 253.0, 3.76),
    "Zn": (40.0, 284.0, 4.02),
    "Ga": (60.0, 498.0, 4.19),
    "Ge": (41.0, 354.0, 4.2),
    "As": (29.0, 246.0, 4.11),
    "Se": (25.0, 210.0, 4.04),
    "Br": (20.0, 162.0, 3.93),
    "Kr": (16.8, 129.6, 3.82),
    "Rb": (319.2, 4691.0, 3.72),
    "Sr": (199.0, 3170.0, 4.54),
    "Pd": (23.68, 157.5, 3.66),
    "Ag": (50.6, 339.0, 3.82),
    "I": (35.0, 385.0, 4.17),
    "Xe": (27.3, 285.9, 4.08),
    "Ba": (275.0, 5727.0, 4.77),
    "Ir": (42.51, 359.1, 4.0),
    "Pt": (39.68, 347.1, 3.92),
    "Au": (36.5, 298.0, 3.86),
    "Hg": (33.9, 392.0, 3.98),
    "Pb": (61.8, 697.0, 4.31),
    "Bi": (49.02, 571.0, 4.32),
}
#: Steepness ``d`` and radius scale ``s_R`` of the damping function.
DAMPING_STEEPNESS = 20.0
DAMPING_RADIUS_SCALE = 0.94
#: Energy of a shell of lattice translations below which the periodic sum
#: stops (Ry; "10^-6 eV" in ``usrinputfile.F90``).
PERIODIC_SHELL_TOLERANCE = 7.3e-8


def lattice_shell_steps(n: int) -> np.ndarray:
    """The integer steps ``(i, j, k)`` with ``max(|i|, |j|, |k|) = n``, as float rows.

    In the order of the loops ``for i, for j, for k`` over ``-n..n`` (the
    order the shell sums add their terms in), built a plane of fixed ``i``
    at a time: the whole square where ``|i| = n``, its border otherwise.
    """

    n = int(n)
    if n < 1:
        raise ValueError("a lattice shell has n >= 1")
    values = np.arange(-n, n + 1)
    j, k = (axis.ravel() for axis in np.meshgrid(values, values, indexing="ij"))
    square = np.column_stack((j, k))
    border = square[(np.abs(j) == n) | (np.abs(k) == n)]
    planes = []
    for i in values:
        plane = square if abs(int(i)) == n else border
        planes.append(np.column_stack((np.full(plane.shape[0], i), plane)))
    return np.concatenate(planes).astype(np.float64)


def damping_function(distance: np.ndarray, radius_sum: np.ndarray) -> np.ndarray:
    """``1 / (1 + exp(-20 (R/(0.94 R0_ij) - 1)))``; 1/2 at ``R = 0.94 R0_ij``."""

    distance = np.asarray(distance, dtype=np.float64)
    radius_sum = np.asarray(radius_sum, dtype=np.float64)
    return 1.0 / (
        1.0
        + np.exp(-DAMPING_STEEPNESS * (distance / (DAMPING_RADIUS_SCALE * radius_sum) - 1.0))
    )


def free_atom_parameters(element: str) -> tuple[float, float, float]:
    """``(alpha, C6, R0)`` of ``element`` in bohr^3, Ha bohr^6 and bohr."""

    try:
        return FREE_ATOM_DATA[element]
    except KeyError:
        raise ValueError(
            f"no Tkatchenko-Scheffler free-atom data for element {element!r}; "
            "efvdw.f90 knows " + ", ".join(FREE_ATOM_DATA)
        ) from None


def hirshfeld_translations(
    lattice_vectors: np.ndarray | None, cells: Sequence[int] = (0, 0, 0)
) -> np.ndarray:
    """Lattice translations of the Hirshfeld cells (the zero vector alone
    for a cluster).  ``lattice_vectors`` has the lattice vectors as rows."""

    if lattice_vectors is None:
        return np.zeros((1, 3))
    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    ranges = [np.arange(-int(n), int(n) + 1) for n in cells]
    steps = np.array(
        [(i, j, k) for i in ranges[0] for j in ranges[1] for k in ranges[2]],
        dtype=np.float64,
    )
    return steps @ lattice


@dataclass(frozen=True)
class HirshfeldVolumes:
    """Free and effective volumes ``int r^3 rho`` of every atom (e bohr^3)."""

    free: np.ndarray
    effective: np.ndarray

    @property
    def ratio(self) -> np.ndarray:
        return self.effective / self.free


def _free_atom_density(
    grid,
    atom: Atom,
    translation: np.ndarray,
    potentials,
    specifications,
) -> tuple[np.ndarray, np.ndarray]:
    """Unnormalized density of one atom image and its distances to the grid.

    The image sits at ``R_a - L``: the distance of grid point ``r`` to it is
    ``|r + L - R_a|``, the one of ``initchrg_f.f90``.  Points the density does
    not reach (``dist >= rs(ns-1)``) have the distance zero there as well.
    """

    from ..V_ion import superpose_atomic_density

    image = Atom(atom.symbol, np.asarray(atom.position, dtype=np.float64) - translation)
    density = superpose_atomic_density(grid, (image,), potentials, specifications)
    distance = np.linalg.norm(grid.coordinates - image.position, axis=1)
    cutoff = potentials[atom.symbol].interpolation_cutoff
    distance = np.where(distance < cutoff, distance, 0.0)
    return density, distance


class _CentredGrid:
    """A grid seen with its points moved into the cell centred on the origin."""

    def __init__(self, grid, coordinates: np.ndarray) -> None:
        self._grid = grid
        self.coordinates = coordinates

    def __getattr__(self, name):
        return getattr(self._grid, name)


def _centred(points, lattice_vectors: np.ndarray) -> np.ndarray:
    """``points`` moved to their images of fractional coordinates in [-1/2, 1/2)."""

    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    fractional = points @ np.linalg.inv(lattice)
    # A small tolerance keeps a point at exactly +1/2 (rounded) on PARSEC's side.
    shift = np.floor(fractional + 0.5 + 1.0e-9)
    return points - shift @ lattice


def _grid_in_centred_cell(grid, lattice_vectors: np.ndarray):
    """``grid`` with every point at its image of fractional coordinates in [-1/2, 1/2)."""

    return _CentredGrid(grid, _centred(grid.coordinates, lattice_vectors))


def hirshfeld_volumes(
    grid,
    atoms: Sequence[Atom],
    potentials,
    specifications: Mapping[str, SpeciesPotential],
    density: np.ndarray,
    lattice_vectors: np.ndarray | None = None,
    cells: Sequence[int] = (0, 0, 0),
) -> HirshfeldVolumes:
    """Hirshfeld free and effective volumes of ``initchrg_f.f90``/``efvdw.f90``.

    ``density`` is the valence density on the grid (e/bohr^3).  Each atom's
    free density is normalized to its ionic charge over all cells.  The
    densities are formed twice, once for the normalization and the weight
    denominator and once for the effective volumes, instead of being stored.
    """

    density = np.asarray(density, dtype=np.float64)
    volume = grid.volume_element
    translations = hirshfeld_translations(lattice_vectors, cells)
    if lattice_vectors is not None:
        # initchrg_f.f90 adds the Hirshfeld cells to the points of PARSEC's
        # cell, whose integer coordinates run from -n/2 to n/2-1 (the cell
        # centred on the origin); the grids of this port number 0..n-1.  The
        # same points, each moved into PARSEC's cell, give PARSEC's
        # truncation of the image sums.  The atoms likewise (this port keeps
        # them in [0, L); PARSEC's input atoms lie in its centred cell).
        grid = _grid_in_centred_cell(grid, lattice_vectors)
        atoms = tuple(
            replace(atom, position=tuple(_centred(atom.position, lattice_vectors)))
            for atom in atoms
        )
    scales = np.empty(len(atoms))
    free = np.zeros(len(atoms))
    denominator = np.zeros(grid.size)
    for index, atom in enumerate(atoms):
        images = [
            _free_atom_density(grid, atom, translation, potentials, specifications)
            for translation in translations
        ]
        charge = sum(float(np.sum(values)) for values, _ in images) * volume
        if charge <= 0.0:
            raise ValueError(f"atom {index + 1} ({atom.symbol}) has no free-atom density on the grid")
        scales[index] = potentials[atom.symbol].ionic_charge / charge
        for values, distance in images:
            scaled = scales[index] * values
            denominator += scaled
            free[index] += float(np.sum(distance**3 * scaled)) * volume
    weight_density = np.divide(
        density, denominator, out=np.zeros_like(density), where=denominator > 0.0
    )
    effective = np.zeros(len(atoms))
    for index, atom in enumerate(atoms):
        for translation in translations:
            values, distance = _free_atom_density(
                grid, atom, translation, potentials, specifications
            )
            effective[index] += (
                float(np.sum(distance**3 * scales[index] * values * weight_density))
                * volume
            )
    return HirshfeldVolumes(free=free, effective=effective)


@dataclass(frozen=True)
class DispersionResult:
    """The TS correction of one structure.

    ``energy`` in Ry, ``forces`` (atoms, 3) in Ry/bohr; ``c6``,
    ``polarizability`` and ``vdw_radius`` are the effective atomic values in
    Hartree atomic units; ``shells`` is the number of lattice shells summed
    (zero for a cluster).
    """

    energy: float
    forces: np.ndarray
    volumes: HirshfeldVolumes
    c6: np.ndarray
    polarizability: np.ndarray
    vdw_radius: np.ndarray
    shells: int = 0

    @property
    def forces_hartree(self) -> np.ndarray:
        """``vdw_forces`` of ``efvdw.f90`` in Ha/bohr."""

        return 0.5 * self.forces


def _pair_terms(displacement: np.ndarray, c6: np.ndarray, radius_sum: np.ndarray):
    """Hartree pair energies ``-f C6/R^6`` and the forces they give.

    ``displacement`` holds ``R_a - R_b(-image)`` with the pair axes in front.
    Returns the energies and ``-d * E * rfac / R`` with
    ``rfac = -6/R + 20 e/((1 + e) 0.94 R0_ij)``, ``e = exp(-20(R/(0.94 R0_ij) - 1))``
    (``efvdw.f90``).
    """

    squared = np.einsum("...k,...k->...", displacement, displacement)
    distance = np.sqrt(squared)
    scaled_radius = DAMPING_RADIUS_SCALE * radius_sum
    exponential = np.exp(-DAMPING_STEEPNESS * (distance / scaled_radius - 1.0))
    damping = 1.0 / (1.0 + exponential)
    energy = -c6 * damping / squared**3
    rfac = -6.0 / distance + DAMPING_STEEPNESS * exponential / (
        (1.0 + exponential) * scaled_radius
    )
    force = -displacement * (energy * rfac / distance)[..., None]
    return energy, force


def ts_pair_energy(
    positions: np.ndarray,
    c6: np.ndarray,
    polarizability: np.ndarray,
    vdw_radius: np.ndarray,
    lattice_vectors: np.ndarray | None = None,
) -> tuple[float, np.ndarray, int]:
    """Pair sum of ``efvdw.f90``: ``(energy Ry, forces Ry/bohr, shells)``.

    ``positions`` (atoms, 3) in bohr; the effective atomic ``c6``
    (Ha bohr^6), ``polarizability`` (bohr^3) and ``vdw_radius`` (bohr).
    """

    positions = np.asarray(positions, dtype=np.float64)
    c6 = np.asarray(c6, dtype=np.float64)
    alpha = np.asarray(polarizability, dtype=np.float64)
    radius = np.asarray(vdw_radius, dtype=np.float64)
    count = positions.shape[0]
    pair_c6 = (2.0 * c6[:, None] * c6[None, :]) / (
        alpha[None, :] / alpha[:, None] * c6[:, None]
        + alpha[:, None] / alpha[None, :] * c6[None, :]
    )
    pair_radius = radius[:, None] + radius[None, :]
    forces = np.zeros((count, 3))
    energy = 0.0
    if count > 1:
        off = ~np.eye(count, dtype=bool)
        displacement = positions[:, None, :] - positions[None, :, :]
        pair_energy, pair_force = _pair_terms(
            displacement[off], pair_c6[off], pair_radius[off]
        )
        energy = 2.0 * (0.5 * float(np.sum(pair_energy)))
        owners = np.nonzero(off)[0]
        for component in range(3):
            forces[:, component] = np.bincount(
                owners, weights=pair_force[:, component], minlength=count
            )
    shells = 0
    if lattice_vectors is not None:
        lattice = np.asarray(lattice_vectors, dtype=np.float64)
        while True:
            shells += 1
            translations = lattice_shell_steps(shells) @ lattice
            # d = R_a - (R_b + L) for every translation, a and b.
            displacement = (
                positions[None, :, None, :]
                - positions[None, None, :, :]
                - translations[:, None, None, :]
            )
            shell_c6 = np.broadcast_to(pair_c6, displacement.shape[:-1])
            shell_radius = np.broadcast_to(pair_radius, displacement.shape[:-1])
            pair_energy, pair_force = _pair_terms(displacement, shell_c6, shell_radius)
            shell_energy = 2.0 * (0.5 * float(np.sum(pair_energy)))
            energy += shell_energy
            forces += np.sum(pair_force, axis=(0, 2))
            if abs(shell_energy) < PERIODIC_SHELL_TOLERANCE:
                break
    # Hartree to Rydberg: the energy above already carries the factor two.
    return energy, 2.0 * forces, shells


def ts_dispersion(
    atoms: Sequence[Atom],
    elements: Sequence[str],
    volumes: HirshfeldVolumes,
    lattice_vectors: np.ndarray | None = None,
) -> DispersionResult:
    """The TS energy and forces of ``atoms`` with the given volumes."""

    data = np.array([free_atom_parameters(element) for element in elements])
    ratio = volumes.ratio
    c6 = ratio**2 * data[:, 1]
    alpha = ratio * data[:, 0]
    radius = ratio ** (1.0 / 3.0) * data[:, 2]
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    energy, forces, shells = ts_pair_energy(positions, c6, alpha, radius, lattice_vectors)
    return DispersionResult(
        energy=energy,
        forces=forces,
        volumes=volumes,
        c6=c6,
        polarizability=alpha,
        vdw_radius=radius,
        shells=shells,
    )


def species_elements(atoms: Sequence[Atom], specifications) -> list[str]:
    """The chemical element of every atom (``Element_Symbol`` or the label)."""

    return [
        specifications[atom.symbol].element_symbol or atom.symbol for atom in atoms
    ]


# The TS correction of the reference periodic drivers where their k-point
# eigenproblems run on a device: ``(correction(system, density), in_use())``,
# installed by the acceleration package (``acceleration/Energy``); None: the
# host sums of :func:`dispersion_correction` everywhere.
_DEVICE_DISPERSION: tuple | None = None


def set_device_dispersion(correction, in_use) -> None:
    """Install (or, with ``None``, remove) the device TS correction of periodic cells.

    ``correction(system, density)`` returns a :class:`DispersionResult`;
    ``in_use()`` says whether the run puts its periodic work on a device.
    """

    global _DEVICE_DISPERSION
    _DEVICE_DISPERSION = None if correction is None else (correction, in_use)


def device_dispersion_for(system):
    """The installed device correction for a periodic ``system`` that runs on a device, else None.

    Clusters keep the host sums on the reference path (their GPU path is the
    accelerated driver, which forms them on its device).
    """

    if _DEVICE_DISPERSION is None:
        return None
    if getattr(getattr(system, "input", None), "periodic_cell", None) is None:
        return None
    correction, in_use = _DEVICE_DISPERSION
    return correction if in_use() else None


def dispersion_correction(system, density: np.ndarray) -> DispersionResult:
    """``efvdw.f90`` for a prepared system and its converged valence density."""

    problem = system.input
    cell = getattr(problem, "periodic_cell", None)
    lattice = None if cell is None else cell.lattice_vectors
    volumes = hirshfeld_volumes(
        system.grid,
        system.atoms,
        system.pseudopotentials,
        problem.pseudopotentials,
        density,
        lattice,
        problem.scf.hirshfeld_cells if lattice is not None else (0, 0, 0),
    )
    return ts_dispersion(
        system.atoms,
        species_elements(system.atoms, problem.pseudopotentials),
        volumes,
        lattice,
    )


__all__ = [
    "device_dispersion_for",
    "set_device_dispersion",
    "DAMPING_RADIUS_SCALE",
    "DAMPING_STEEPNESS",
    "FREE_ATOM_DATA",
    "PERIODIC_SHELL_TOLERANCE",
    "DispersionResult",
    "HirshfeldVolumes",
    "damping_function",
    "dispersion_correction",
    "free_atom_parameters",
    "hirshfeld_translations",
    "hirshfeld_volumes",
    "lattice_shell_steps",
    "species_elements",
    "ts_dispersion",
    "ts_pair_energy",
]
