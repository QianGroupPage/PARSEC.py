"""LDA+U on-site Coulomb term of the d channel (PARSEC ``upot.F90``).

PARSEC adds to the Kohn--Sham Hamiltonian of spin channel ``s`` the low-rank
term (``plusu.f90z``)

``H_U = sum_a sum_(m,m') |phi_a,m> D^(a,s)_(m,m') <phi_a,m'|``

for every atom ``a`` of a species with ``ldaplusu_u`` or ``ldaplusu_j``.
The five projectors of an atom are sampled on the grid as (``upot.F90``)

``phi_a,m(r_i) = sqrt(h^3) * R_d(|r_i - R_a|) * Y_2m(r_i - R_a)``,

with ``R_d = u_d/r`` the radial pseudo-wavefunction of the d channel of the
pseudopotential file (``pseudo.f90`` divides the reduced function ``u_d`` of
the file by ``r`` at its end) interpolated linearly on the logarithmic grid,
and ``Y_2m`` the real d harmonics of ``ylm_real`` in the order xy, yz, xz,
3z^2-1, x^2-y^2.  The projectors are not renormalized on the grid and they
end at the radius ``r_c`` of the 0.95 rule below, where ``R_d`` is not zero;
the cut and the missing normalization are PARSEC's and are kept.  A grid
point belongs to the support of an atom if it lies within ``r_c`` of the atom
or, in a periodic cell, of one of its images one cell away in each periodic
direction; every such (point, image) pair is one entry of the projector.

From the occupied orbitals of channel ``s`` (``calcden.f90z``)

``N^(a,s)_(m',m) = sum_n f_n <phi_a,m|psi_n> <psi_n|phi_a,m'>``,

with ``f_n`` in ``[0, 1]`` the occupation of the state in its channel (also
without spin polarization), and from it the potential matrix

``D^(a,s) = -U N^(a,s) + 2J [P3 : N^(a,s) + P4 : N^(a,-s)] + (U-J)/2 * 1``

where ``(P : N)_(m,m') = sum_(k,l) P_(m,m',k,l) N_(k,l)`` (without spin
polarization ``N^(a,-s) = N^(a,s)``) and only the real part is kept.  The
energy that ``totnrg.f90`` adds to the electronic energy is

``E_U = (2/nspin) [ sum_(s,a,m) (U-J)/4 N^(a,s)_mm
                   - 1/2 sum_(s,a,m,m') D^(a,s)_(m,m') N^(a,s)_(m,m') ]``

(the first sum alone is the "first contribution" that ``calcden.f90z``
prints).  With ``J = 0`` the band energy ``(2/nspin) sum_s Tr(D N)`` plus
``E_U`` is the Dudarev form ``(2/nspin) (U/2) sum_(s,a) Tr[N (1 - N)]``.

PARSEC builds ``N`` and ``D`` at the start of every iteration but the first
from the orbitals and occupations of the iteration before; in the first
iteration ``D`` is zero.  The SCF drivers of :mod:`parsec_python.SCF` do the
same.

Two properties of the Fortran are kept on purpose and stated here:

* ``P3`` and ``P4`` are the complex tensors of ``initialize_p_matrices``
  (Liechtenstein, Anisimov and Zaanen, PRB 52, R5467 (1995)) written as
  single-precision literals; :func:`coulomb_tensors` rounds them to
  ``float32``.  They are applied to the occupation matrix of the real
  harmonics, for which they lack the symmetries of the complex basis they
  were written for: with ``J != 0`` the matrix ``D`` is real but not
  symmetric, so ``H_U`` is not Hermitian and the ``J`` term is not the
  derivative of ``E_U``.  The eigensolvers reproduce PARSEC there because
  they hand the same triangle of the projected matrix to the dense
  eigensolver; forces are only defined for ``J = 0``.
* The k-point branch of ``plusu.f90z`` never advances its projector offset
  (every Hubbard atom after the first applies the columns of the first) and
  ``calcden.f90z`` builds ``N`` without the Bloch phases ``u_pot%right``
  (it projects ``u_k`` instead of ``e^(ik.r) u_k``).  This module has the
  correct Bloch form instead (a deviation from the Fortran, which is right at
  ``k = 0`` only): with ``k_point`` the projector of the periodic Bloch
  factor ``u_k`` carries the phase ``exp(-i k.(r + L))`` of every support
  entry (the point ``r`` seen from the image ``L``: ``u_pot%left`` of
  ``upot.F90``; its conjugate in the overlaps is ``u_pot%right``), the
  overlaps are ``<phi_k|u_k>``, ``N`` is the k-weighted sum
  ``sum_k w_k sum_n f_nk <phi|psi_nk><psi_nk|phi>`` (:func:`kpoint_occupation_matrix`)
  and the term of a k-point is ``P_k D P_k^H`` with the one ``D`` of the
  summed ``N``.  Collinear spin keeps time reversal, so the sum over the
  Brillouin zone is real: :func:`time_reversal_occupations` keeps its real
  part, and an irreducible wedge of the space group is completed by
  :func:`symmetrize_occupations` (the rotation of the d harmonics, which
  ``calcden.f90z`` lacks as well).

Units: Ry and bohr.  Orbitals are Euclidean-normalized grid vectors
(``sqrt(h^3) psi``), so overlaps are plain sums over the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential
from ..V_ion import real_spherical_harmonics
from .hubbard_tensors import P3_ENTRIES, P4_ENTRIES


#: Angular momentum of the LDA+U channel: PARSEC implements d only.
HUBBARD_ANGULAR_MOMENTUM = 2
#: Number of real harmonics of the d channel.
HUBBARD_ORBITALS = 2 * HUBBARD_ANGULAR_MOMENTUM + 1
#: Fraction of the norm of ``u_d`` inside the projector radius (``upot.F90``).
HUBBARD_NORM_FRACTION = 0.95


@lru_cache(maxsize=1)
def coulomb_tensors() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(P3, P4)``, each ``complex128`` of shape ``(5, 5, 5, 5)``.

    Index order ``(m, m', k, l)`` with ``m`` in xy, yz, xz, 3z^2-1, x^2-y^2.
    Each literal of ``upot.F90`` is rounded to ``float32`` as the Fortran
    default-real ``cmplx(re, im)`` does, then widened.  The arrays are
    read-only.
    """

    tensors = []
    for entries in (P3_ENTRIES, P4_ENTRIES):
        tensor = np.zeros((5, 5, 5, 5), dtype=np.complex128)
        for m, mp, k, l, real, imaginary in entries:
            tensor[m - 1, mp - 1, k - 1, l - 1] = complex(
                float(np.float32(real)), float(np.float32(imaginary))
            )
        tensor.setflags(write=False)
        tensors.append(tensor)
    return tensors[0], tensors[1]


def _extended_radial_tables(
    potential: ParsecPseudopotential,
) -> tuple[np.ndarray, np.ndarray]:
    """``rs`` and ``wfspd`` of the d channel as ``upot.F90`` reads them.

    Fortran index ``j`` is position ``j - 1``: ``rs(1) = 0`` with
    ``wfspd(1) = 0`` (``pseudo.f90`` fills ``j <= 1``), then the radii of the
    file with ``wfspd(j) = u_d(r_j)/r_j``.
    """

    if HUBBARD_ANGULAR_MOMENTUM not in potential.radial_wavefunctions:
        raise ValueError(
            f"{potential.source} has no d channel: PARSEC's LDA+U term needs "
            "the d pseudo-wavefunction"
        )
    radii = np.asarray(potential.radii, dtype=np.float64)
    reduced = np.asarray(
        potential.radial_wavefunctions[HUBBARD_ANGULAR_MOMENTUM], dtype=np.float64
    )
    return (
        np.concatenate(([0.0], radii)),
        np.concatenate(([0.0], reduced / radii)),
    )


def hubbard_cutoff_radius(potential: ParsecPseudopotential) -> float:
    """Projector radius ``r_c`` of the d channel (``upot.F90``), in bohr.

    The trapezoid weights ``(r_(i+1) - r_(i-1))/2`` sum ``u_d(r_i)**2`` from
    the first radius of the file outwards; ``r_c`` is the first radius at
    which the sum exceeds 0.95.
    """

    radii, radial = _extended_radial_tables(potential)
    # tmpnorm starts from (wfspd(1)*rs(1))**2 = 0 and adds i = 2, 3, ...
    reduced_squared = (radial[1:-1] * radii[1:-1]) ** 2
    weights = 0.5 * (radii[2:] - radii[:-2])
    norm = np.cumsum(weights * reduced_squared)
    beyond = np.flatnonzero(norm > HUBBARD_NORM_FRACTION)
    if beyond.size == 0:
        raise ValueError(
            f"the d pseudo-wavefunction of {potential.source} never holds "
            f"{HUBBARD_NORM_FRACTION} of its norm inside the radial table"
        )
    return float(radii[1 + beyond[0]])


def interpolate_hubbard_radial(
    potential: ParsecPseudopotential, distances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``R_d`` at ``distances`` as ``upot.F90`` interpolates it, and its slope.

    A distance below the first radius of the file is raised to it.  The
    interval is the one of the index ``int(ln(1 + r/a)/b) + 1`` of the
    logarithmic grid ``r_j = a(exp(b(j-1)) - 1)``, as there.  The slope is
    that of the linear interpolant (zero where the distance was raised): the
    exact derivative of the sampled projector with respect to the distance.
    """

    radii, radial = _extended_radial_tables(potential)
    distance = np.asarray(distances, dtype=np.float64)
    first = radii[1]
    clamped = np.maximum(distance, first)
    inverse_a = 1.0 / potential.radial_parameter_a
    inverse_b = 1.0 / potential.radial_parameter_b
    # npt = idint(binv*log(one + ainv*rnorm)) + 1 is Fortran index npt,
    # position npt - 1 here.
    lower = np.floor(inverse_b * np.log(1.0 + inverse_a * clamped)).astype(np.int64)
    lower = np.clip(lower, 0, radii.size - 2)
    width = radii[lower + 1] - radii[lower]
    slope = (radial[lower + 1] - radial[lower]) / width
    values = radial[lower] + (clamped - radii[lower]) / width * (
        radial[lower + 1] - radial[lower]
    )
    slope = np.where(distance < first, 0.0, slope)
    return values, slope


def periodic_replica_translations(lattice_vectors: np.ndarray | None) -> np.ndarray:
    """Translations of the images ``upot.F90`` visits: one cell each way.

    ``None`` (an isolated system) gives the zero translation alone; a
    periodic cell the 27 combinations of ``-1, 0, 1`` times its lattice
    vectors (the rows of ``lattice_vectors``, as in
    :class:`parsec_python.models.PeriodicCell`).  A wire or slab passes its
    periodic vectors only (one or two rows: ``nreplica(1:pbc%per) = 1``).
    """

    if lattice_vectors is None:
        return np.zeros((1, 3))
    lattice = np.asarray(lattice_vectors, dtype=np.float64).reshape(-1, 3)
    if lattice.shape[0] == 3:
        steps = np.array(
            [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            dtype=np.float64,
        )
    else:
        from itertools import product

        steps = np.array(list(product((-1, 0, 1), repeat=lattice.shape[0])), dtype=np.float64)
    return steps @ lattice


def hubbard_replica_lattice(system) -> np.ndarray | None:
    """The lattice rows whose replicas the LDA+U projectors of ``system`` visit.

    ``None`` for an isolated system; the three cell vectors of a bulk cell;
    the periodic vectors alone of a wire or slab (``nreplica(1:pbc%per) = 1``
    of ``upot.F90``: no replica along a confined direction, whose
    ``latt_vec`` row is a unit vector), as the Gamma-point projectors of
    :mod:`parsec_python.SCF.wire_slab` are built.
    """

    cell = getattr(system.input, "periodic_cell", None)
    if cell is None:
        return None
    lattice = np.asarray(cell.lattice_vectors, dtype=np.float64)
    if getattr(system.grid, "is_confined_periodic", False):
        return lattice[: int(system.grid.periodicity)]
    return lattice


def hubbard_symmetry_lattice(system) -> np.ndarray:
    """The lattice of the ``M`` and ``t`` of ``system.input.space_group``.

    The cell vectors of a bulk cell; ``alatt`` of a wire or slab
    (:func:`parsec_python.Grid.space_group.confined_symmetry_lattice`, with
    ``4 rmax`` along the confined directions), the lattice
    ``symmetries.f90`` searched the group with.
    """

    lattice = np.asarray(system.input.periodic_cell.lattice_vectors, dtype=np.float64)
    if getattr(system.grid, "is_confined_periodic", False):
        from ..Grid.space_group import confined_symmetry_lattice

        return confined_symmetry_lattice(
            lattice, int(system.grid.periodicity), system.grid.confinement_radius
        )
    return lattice


def hubbard_support(
    coordinates: np.ndarray,
    position: np.ndarray,
    radius: float,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid rows and displacements ``r_i + L - R_a`` of every support entry.

    One entry per pair of a grid point and a translation ``L`` with
    ``|r_i + L - R_a| <= radius``; a point reached by two images gives two
    entries, which add up in the projector.
    """

    rows: list[np.ndarray] = []
    relative: list[np.ndarray] = []
    position = np.asarray(position, dtype=np.float64)
    for translation in translations:
        shifted = coordinates + (translation - position)
        # A cheap box test first, the exact sphere after it.
        box = np.all(np.abs(shifted) <= radius, axis=1)
        candidates = np.flatnonzero(box)
        displacement = shifted[candidates]
        # rr_tmp = sqrt(x*x+y*y+z*z) <= rc(ity) of upot.F90.
        distance = np.sqrt(
            displacement[:, 0] * displacement[:, 0]
            + displacement[:, 1] * displacement[:, 1]
            + displacement[:, 2] * displacement[:, 2]
        )
        inside = distance <= radius
        rows.append(candidates[inside])
        relative.append(displacement[inside])
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty((0, 3))
    return np.concatenate(rows), np.concatenate(relative)


def _double_grid_hubbard_columns(
    grid,
    potential: ParsecPseudopotential,
    position: np.ndarray,
    radius: float,
    translations: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Double-grid support and values of one atom (``upot.F90``, ``ndouble > 1``).

    Returns the rows, the displacements ``r + L - R_a`` of the coarse points
    and the five values per entry without the ``sqrt(h^3)`` factor:
    ``sum_f w_f R_d(|s_f|) Y_2m(s_f)`` over the fine points ``s_f`` of the
    point within ``r_c`` (weights of :func:`parsec_python.V_ion.double_grid.double_grid_points`;
    ``R_d`` interpolated as for ``N = 1``, the distance raised to the first
    radius of the file; harmonics with the true direction, as the
    Kleinman-Bylander double grid of this port).  A coarse point is an entry
    when any of its fine points lies within ``r_c`` (unclamped distance, as
    the support loop of ``upot.F90``).
    """

    from ..V_ion.double_grid import double_grid_support

    coordinates = np.asarray(grid.coordinates, dtype=np.float64)
    spacing = float(grid.spacing)
    position = np.asarray(position, dtype=np.float64)
    rows: list[np.ndarray] = []
    relative: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for translation in translations:
        support = double_grid_support(
            coordinates, position - translation, radius, order, spacing
        )
        if support.rows.size == 0:
            continue
        points, fine = support.radius.shape
        radial, _ = interpolate_hubbard_radial(potential, support.radius.reshape(-1))
        harmonics = real_spherical_harmonics(
            HUBBARD_ANGULAR_MOMENTUM, support.relative.reshape(-1, 3)
        )
        scaled = (radial[:, None] * harmonics).reshape(points, fine, HUBBARD_ORBITALS)
        values.append(np.einsum("pf,pfm->pm", support.weights, scaled))
        rows.append(support.rows)
        relative.append(coordinates[support.rows] + (translation - position))
    if not rows:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 3)),
            np.empty((0, HUBBARD_ORBITALS)),
        )
    return np.concatenate(rows), np.concatenate(relative), np.concatenate(values)


@dataclass(frozen=True)
class HubbardProjectors:
    """The LDA+U projectors of every atom with ``U`` or ``J``.

    ``projectors`` has shape ``(grid points, 5 * atoms)``: the five columns
    ``phi_a,m`` of the ``k``-th Hubbard atom are ``5k`` to ``5k + 4``.
    ``atom_indices[k]`` is the position of that atom in the atoms of the
    system, ``hubbard_u[k]`` and ``hubbard_j[k]`` its parameters in Ry.
    ``cutoff_radii`` maps each species with LDA+U to ``r_c`` (bohr) and
    ``support_sizes[k]`` is the number of support entries of the atom
    (``nlatom`` of ``upot.F90``).
    """

    projectors: sp.csc_matrix
    atom_indices: tuple[int, ...]
    hubbard_u: np.ndarray
    hubbard_j: np.ndarray
    cutoff_radii: Mapping[str, float]
    support_sizes: tuple[int, ...]

    @property
    def atom_count(self) -> int:
        return len(self.atom_indices)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.projectors.shape[0], self.projectors.shape[0])

    def overlaps(self, vectors: np.ndarray) -> np.ndarray:
        """``<phi_a,m|psi_n>`` with shape ``(atoms, 5, states)``.

        Bloch projectors (complex, built with a ``k_point``) are conjugated:
        ``<phi_k|u> = sum_r conj(phi_k(r)) u(r)``.
        """

        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors[:, None]
        if np.iscomplexobj(self.projectors.data):
            values = np.asarray(self.projectors.conj().T @ vectors)
        else:
            values = np.asarray(self.projectors.T @ vectors)
        return values.reshape(self.atom_count, HUBBARD_ORBITALS, vectors.shape[1])


def build_hubbard_projectors(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray | None = None,
    k_point: np.ndarray | None = None,
) -> HubbardProjectors | None:
    """Sample the projectors of ``upot.F90`` on ``grid``; ``None`` without LDA+U.

    ``lattice_vectors`` (rows) makes the support periodic with the images
    one cell away; a cell side shorter than ``r_c`` would need more images
    than ``upot.F90`` visits and is refused.  ``k_point`` (Cartesian,
    bohr^-1; needs ``lattice_vectors``) gives the Bloch projectors of the
    periodic factor ``u_k``: every support entry of the point ``r`` seen
    from the image ``L`` is multiplied by ``exp(-i k.(r + L))``
    (``u_pot%left`` of ``upot.F90``), the convention of
    :func:`parsec_python.V_ion.build_nonlocal_projectors`.

    With ``grid.settings.double_grid_order = N > 1`` every column is the
    weighted average of the projector over the ``(2N-1)^3`` fine points of
    each coarse point (:mod:`parsec_python.V_ion.double_grid`), and a coarse
    point belongs to the support when any of its fine points lies within
    ``r_c`` (``upot.F90``'s ``ndouble`` loops).
    """

    hubbard_atoms = [
        index
        for index, atom in enumerate(atoms)
        if specifications[atom.symbol].has_hubbard
    ]
    if not hubbard_atoms:
        return None
    translations = periodic_replica_translations(lattice_vectors)
    radii: dict[str, float] = {}
    for index in hubbard_atoms:
        symbol = atoms[index].symbol
        if symbol not in radii:
            radii[symbol] = hubbard_cutoff_radius(potentials[symbol])
    if lattice_vectors is not None:
        sides = np.linalg.norm(
            np.asarray(lattice_vectors, dtype=np.float64).reshape(-1, 3), axis=1
        )
        longest = max(radii.values())
        if np.any(sides < longest):
            raise ValueError(
                f"an LDA+U projector radius of {longest:.4f} bohr exceeds a "
                f"cell side ({', '.join(f'{side:.4f}' for side in sides)} bohr): "
                "upot.F90 visits the images one cell away only"
            )
    sqrt_volume = np.sqrt(grid.volume_element)
    coordinates = np.asarray(grid.coordinates, dtype=np.float64)
    order = int(getattr(getattr(grid, "settings", None), "double_grid_order", 1) or 1)
    if k_point is not None:
        if lattice_vectors is None:
            raise ValueError("k_point requires lattice_vectors")
        k_point = np.asarray(k_point, dtype=np.float64)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    sizes: list[int] = []
    for column_block, index in enumerate(hubbard_atoms):
        atom = atoms[index]
        potential = potentials[atom.symbol]
        if order > 1:
            entry_rows, relative, block_values = _double_grid_hubbard_columns(
                grid, potential, atom.position, radii[atom.symbol], translations, order
            )
            block_values = sqrt_volume * block_values
        else:
            entry_rows, relative = hubbard_support(
                coordinates, atom.position, radii[atom.symbol], translations
            )
            distance = np.linalg.norm(relative, axis=1)
            radial, _ = interpolate_hubbard_radial(potential, distance)
            harmonics = real_spherical_harmonics(HUBBARD_ANGULAR_MOMENTUM, relative)
            block_values = None
        sizes.append(int(entry_rows.size))
        phase = None
        if k_point is not None:
            # The point r seen from the image L sits at relative + R_a = r + L.
            image_points = relative + np.asarray(atom.position, dtype=np.float64)
            phase = np.exp(-1j * (image_points @ k_point))
        for m in range(HUBBARD_ORBITALS):
            rows.append(entry_rows)
            columns.append(
                np.full(entry_rows.size, HUBBARD_ORBITALS * column_block + m, dtype=np.int64)
            )
            column_values = (
                sqrt_volume * radial * harmonics[:, m]
                if block_values is None
                else block_values[:, m]
            )
            values.append(column_values if phase is None else phase * column_values)
    projectors = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(grid.size, HUBBARD_ORBITALS * len(hubbard_atoms)),
    ).tocsc()
    projectors.sum_duplicates()
    return HubbardProjectors(
        projectors=projectors,
        atom_indices=tuple(hubbard_atoms),
        hubbard_u=np.array(
            [specifications[atoms[index].symbol].hubbard_u for index in hubbard_atoms],
            dtype=np.float64,
        ),
        hubbard_j=np.array(
            [specifications[atoms[index].symbol].hubbard_j for index in hubbard_atoms],
            dtype=np.float64,
        ),
        cutoff_radii=radii,
        support_sizes=tuple(sizes),
    )


def occupation_matrices(
    projectors: HubbardProjectors,
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    *,
    keep_complex: bool = False,
) -> np.ndarray:
    """``N^(a)`` of one spin channel, shape ``(atoms, 5, 5)``.

    ``N[a, m', m] = sum_n f_n <phi_a,m|psi_n> conj(<phi_a,m'|psi_n>)``, the
    index order of ``u_pot%denmat(imp, im)`` in ``calcden.f90z``; it is
    symmetric for real orbitals.  ``occupations`` are those of the states in
    their channel, between zero and one.  ``keep_complex`` returns the
    complex (Hermitian) matrices of Bloch orbitals, which
    :func:`kpoint_occupation_matrix` sums over the k-points; otherwise the
    real part is returned, as before.
    """

    occupations = np.asarray(occupations, dtype=np.float64)
    occupied = np.flatnonzero(occupations != 0.0)
    if occupied.size == 0:
        return np.zeros(
            (projectors.atom_count, HUBBARD_ORBITALS, HUBBARD_ORBITALS),
            dtype=np.complex128 if keep_complex else np.float64,
        )
    vectors = np.asarray(wavefunctions)[:, occupied]
    dots = projectors.overlaps(vectors)
    weighted = dots * occupations[occupied][None, None, :]
    matrices = np.einsum("amn,akn->akm", weighted, dots.conj())
    if keep_complex:
        return np.asarray(matrices, dtype=np.complex128)
    if np.iscomplexobj(matrices):
        matrices = matrices.real
    return np.asarray(matrices, dtype=np.float64)


def kpoint_occupation_matrix(
    projectors: HubbardProjectors,
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    weight: float,
) -> np.ndarray:
    """``w_k N_k`` of one k-point and channel, complex ``(atoms, 5, 5)``.

    ``projectors`` are the Bloch projectors of the k-point
    (:func:`build_hubbard_projectors` with ``k_point``) and
    ``wavefunctions`` its Bloch factors ``u_nk``; ``weight`` is the k-point
    weight ``w_k`` (``kpwt`` of ``calcden.f90z``).  The k-point drivers sum
    these over the k-points and channels' tasks and pass the sum to
    :func:`time_reversal_occupations`.
    """

    return float(weight) * occupation_matrices(
        projectors, wavefunctions, occupations, keep_complex=True
    )


def time_reversal_occupations(occupations: np.ndarray) -> np.ndarray:
    """The real part of k-summed occupation matrices (collinear spin).

    Time reversal maps ``psi_nk`` to ``conj(psi_nk)`` at ``-k`` and
    ``N_(-k) = conj(N_k)`` for the real projectors, so the sum over a
    Brillouin-zone sample closed under ``k -> -k`` (every unreduced
    Monkhorst-Pack grid, and an irreducible wedge whose weights count the
    pairs, as ``irrbz.f90`` builds it for collinear runs) is real; a k-point
    list that leaves out the partners is completed by them.
    """

    return np.ascontiguousarray(np.real(np.asarray(occupations)), dtype=np.float64)


def d_harmonic_rotation(rotation: np.ndarray) -> np.ndarray:
    """``D(U)`` with ``Y(U^-1 v) = Y(v) D`` for the five real d harmonics.

    ``Y(v)`` is the row of :func:`parsec_python.V_ion.real_spherical_harmonics`
    (order xy, yz, xz, 3z^2-1, x^2-y^2).  ``D`` is orthogonal for a proper or
    improper rotation ``U`` (d harmonics are even), found by least squares on
    a fixed set of directions.
    """

    rotation = np.asarray(rotation, dtype=np.float64)
    directions = np.array(
        [
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
            [1.0, -1.0, 0.3], [0.2, 1.0, -1.0], [-1.0, 0.4, 1.0],
            [0.7, 0.3, 0.5], [-0.3, 0.8, 0.1], [0.5, -0.6, 0.9],
        ]
    )
    plain = real_spherical_harmonics(HUBBARD_ANGULAR_MOMENTUM, directions)
    rotated = real_spherical_harmonics(
        HUBBARD_ANGULAR_MOMENTUM, directions @ np.linalg.inv(rotation).T
    )
    matrix, *_ = np.linalg.lstsq(plain, rotated, rcond=None)
    return matrix


def hubbard_atom_permutations(
    atoms: Sequence[Atom],
    atom_indices: Sequence[int],
    lattice_vectors: np.ndarray,
    space_group,
    *,
    tolerance: float = 1.0e-6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(D(U_g), image)`` for every operation ``g`` of ``space_group``.

    ``image[k]`` is the Hubbard-atom position (in ``atom_indices``) of the
    atom that ``g`` maps the ``k``-th Hubbard atom onto: ``f' = M (f - t)``
    in fractional coordinates (``symm_scalar``), the same atom modulo a
    lattice vector.  ``U = A^T M A^-T`` is the Cartesian rotation of ``M``
    for the rows ``A`` of ``lattice_vectors``.
    """

    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    inverse = np.linalg.inv(lattice)
    fractional = np.array(
        [np.asarray(atoms[index].position, dtype=np.float64) @ inverse for index in atom_indices]
    ).reshape(-1, 3)
    symbols = [atoms[index].symbol for index in atom_indices]
    result = []
    for matrix, translation in zip(
        space_group.rotation_matrices, space_group.translation_vectors
    ):
        matrix = np.asarray(matrix, dtype=np.float64)
        moved = (fractional - translation) @ matrix.T
        image = np.empty(len(atom_indices), dtype=np.int64)
        for k, point in enumerate(moved):
            difference = fractional - point
            difference -= np.round(difference)
            distance = np.linalg.norm(difference @ lattice, axis=1)
            candidates = [
                j for j in np.flatnonzero(distance < tolerance) if symbols[j] == symbols[k]
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "a space-group operation does not map the LDA+U atoms onto "
                    "themselves"
                )
            image[k] = candidates[0]
        cartesian = lattice.T @ matrix @ np.linalg.inv(lattice.T)
        result.append((d_harmonic_rotation(cartesian), image))
    return result


def symmetrize_occupations(
    occupations: np.ndarray,
    operations: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """``N^b = (1/|G|) sum_g D(U_g) N^(g^-1 b) D(U_g)^T`` for every channel.

    ``occupations`` has shape ``(spins, atoms, 5, 5)`` (or ``(atoms, 5, 5)``)
    and ``operations`` comes from :func:`hubbard_atom_permutations`.  The
    k-point sum over an irreducible wedge holds one member of every star; the
    orbitals at ``U k`` are the rotated ones, whose matrices on the image
    atom are ``D N D^T``, so the average over the group completes the stars
    (``calcden.f90z`` does not; Fortran's wedge sum is unsymmetrized).
    """

    matrices = np.asarray(occupations, dtype=np.float64)
    single = matrices.ndim == 3
    block = matrices[None] if single else matrices
    total = np.zeros_like(block)
    for rotation, image in operations:
        rotated = np.einsum("mk,sakl,nl->samn", rotation, block, rotation)
        total[:, image] += rotated
    total /= len(operations)
    return total[0] if single else total


@dataclass(frozen=True)
class HubbardTerms:
    """Occupation and potential matrices of every channel and the energies.

    ``occupations`` and ``potentials`` have shape ``(spins, atoms, 5, 5)``
    (one spin without spin polarization).  ``first_contribution`` and
    ``energy`` are the two values ``calcden.f90z`` prints, in Ry;
    ``energy`` is ``etot_plusu``.
    """

    occupations: np.ndarray
    potentials: np.ndarray
    first_contribution: float
    energy: float

    @property
    def spin_count(self) -> int:
        return int(self.occupations.shape[0])

    @property
    def band_energy(self) -> float:
        """``(2/nspin) sum_(s,a) Tr(D N^T)``: what ``H_U`` adds to the band energy."""

        return float(
            2.0
            / self.spin_count
            * np.einsum("samk,samk->", self.potentials, self.occupations)
        )


def hubbard_terms(
    projectors: HubbardProjectors, occupations: Sequence[np.ndarray]
) -> HubbardTerms:
    """``D`` and ``E_U`` of ``calcden.f90z`` from the occupation matrices.

    ``occupations`` holds one ``(atoms, 5, 5)`` array per spin channel, up
    first: one array without spin polarization, two with it.
    """

    matrices = np.asarray(occupations, dtype=np.float64)
    if matrices.ndim != 4 or matrices.shape[0] not in (1, 2) or matrices.shape[1:] != (
        projectors.atom_count,
        HUBBARD_ORBITALS,
        HUBBARD_ORBITALS,
    ):
        raise ValueError("occupations must have shape (spins, atoms, 5, 5)")
    spins = matrices.shape[0]
    p3, p4 = coulomb_tensors()
    u = projectors.hubbard_u[None, :, None, None]
    j = projectors.hubbard_j[None, :, None, None]
    opposite = matrices[::-1] if spins == 2 else matrices
    same_contraction = np.einsum("mnkl,sakl->samn", p3, matrices)
    opposite_contraction = np.einsum("mnkl,sakl->samn", p4, opposite)
    potentials = (
        -u * matrices + 2.0 * j * (same_contraction + opposite_contraction)
    ).real
    diagonal = 0.5 * (projectors.hubbard_u - projectors.hubbard_j)
    potentials = potentials + diagonal[None, :, None, None] * np.eye(HUBBARD_ORBITALS)
    factor = 2.0 / spins
    trace = np.einsum("samm->sa", matrices)
    first = float(
        np.sum(0.25 * (projectors.hubbard_u - projectors.hubbard_j)[None, :] * trace)
    )
    second = float(-0.5 * np.einsum("samk,samk->", potentials, matrices))
    return HubbardTerms(
        occupations=matrices,
        potentials=np.asarray(potentials, dtype=np.float64),
        first_contribution=factor * first,
        energy=factor * (first + second),
    )


def hubbard_terms_from_orbitals(
    projectors: HubbardProjectors,
    wavefunctions: Sequence[np.ndarray],
    occupations: Sequence[np.ndarray],
) -> HubbardTerms:
    """:func:`hubbard_terms` of the orbitals of each channel, up first."""

    return hubbard_terms(
        projectors,
        [
            occupation_matrices(projectors, vectors, weights)
            for vectors, weights in zip(wavefunctions, occupations)
        ],
    )


@dataclass(frozen=True)
class HubbardOperator:
    """``sum_a sum_(m,m') |phi_a,m> D^(a)_(m,m') <phi_a,m'|`` of one channel."""

    projectors: HubbardProjectors
    potentials: np.ndarray

    def __post_init__(self) -> None:
        potentials = np.asarray(self.potentials, dtype=np.float64)
        if potentials.shape != (
            self.projectors.atom_count,
            HUBBARD_ORBITALS,
            HUBBARD_ORBITALS,
        ):
            raise ValueError("potentials must have shape (atoms, 5, 5)")
        object.__setattr__(self, "potentials", potentials)

    @property
    def shape(self) -> tuple[int, int]:
        return self.projectors.shape

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """``q_i += sum_(m,m') phi_m(i) D_(m,m') <phi_m'|p>`` (``plusu.f90z``)."""

        vectors = np.asarray(vectors)
        single = vectors.ndim == 1
        block = vectors[:, None] if single else vectors
        dots = self.projectors.overlaps(block)
        coefficients = np.einsum("amk,akn->amn", self.potentials, dots)
        result = np.asarray(
            self.projectors.projectors
            @ coefficients.reshape(-1, block.shape[1])
        )
        return result[:, 0] if single else result

    def as_sparse(self) -> sp.csr_matrix:
        """Materialize ``P D P^T`` for tests and small systems."""

        blocks = sp.block_diag(list(self.potentials), format="csr")
        matrix = self.projectors.projectors
        return (matrix @ blocks @ matrix.T).tocsr()


def hubbard_operator(
    projectors: HubbardProjectors | None,
    terms: HubbardTerms | None,
    spin: int = 0,
) -> HubbardOperator | None:
    """The term of channel ``spin``; ``None`` where ``D`` is still zero."""

    if projectors is None or terms is None:
        return None
    return HubbardOperator(projectors, terms.potentials[spin])


__all__ = [
    "HUBBARD_ANGULAR_MOMENTUM",
    "HUBBARD_NORM_FRACTION",
    "HUBBARD_ORBITALS",
    "HubbardOperator",
    "HubbardProjectors",
    "HubbardTerms",
    "build_hubbard_projectors",
    "coulomb_tensors",
    "d_harmonic_rotation",
    "hubbard_atom_permutations",
    "hubbard_cutoff_radius",
    "hubbard_operator",
    "hubbard_replica_lattice",
    "hubbard_support",
    "hubbard_symmetry_lattice",
    "hubbard_terms",
    "hubbard_terms_from_orbitals",
    "interpolate_hubbard_radial",
    "kpoint_occupation_matrix",
    "occupation_matrices",
    "periodic_replica_translations",
    "symmetrize_occupations",
    "time_reversal_occupations",
]
