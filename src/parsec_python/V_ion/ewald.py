"""
Ewald summation for the periodic ion-ion energy and forces and the periodic local ionic potential.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.special import erf, erfc, erfcinv

from ..Grid.cell import lattice_plane_spacings
from ..Grid.pbc import uses_fortran_routines
from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential


def _reciprocal_lattice(lattice_vectors: np.ndarray) -> np.ndarray:
    """Return reciprocal vectors ``b_i`` satisfying ``a_i . b_j = 2*pi*delta_ij``."""

    return 2.0 * np.pi * np.linalg.inv(lattice_vectors).T


def _lattice_points(
    basis: np.ndarray,
    reciprocal_basis: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Enumerate integer combinations of ``basis`` whose norm is at most ``cutoff``.

    The maximum index along direction ``i`` is bounded using the perpendicular
    spacing between lattice planes, ``2*pi/|reciprocal_basis[i]|``, which is
    always at least as generous as the true bound so no in-cutoff point is
    missed.
    """

    if cutoff <= 0.0:
        return np.zeros((1, 3), dtype=float)
    plane_spacing = 2.0 * np.pi / np.linalg.norm(reciprocal_basis, axis=1)
    max_index = np.ceil(cutoff / plane_spacing).astype(int) + 1
    ranges = [np.arange(-m, m + 1) for m in max_index]
    grid = np.meshgrid(*ranges, indexing="ij")
    integer_combinations = np.column_stack([axis.reshape(-1) for axis in grid])
    points = integer_combinations @ basis
    within_cutoff = np.linalg.norm(points, axis=1) <= cutoff
    return points[within_cutoff]


def _cell_center_and_radius(lattice_vectors: np.ndarray) -> tuple[np.ndarray, float]:
    """Center of the unit cell and the largest distance from it to a corner (the circumscribed radius)."""

    signs = np.array(np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], indexing="ij")).reshape(3, -1).T
    corners = signs @ lattice_vectors
    center = 0.5 * np.sum(lattice_vectors, axis=0)
    return center, float(np.max(np.linalg.norm(corners - center, axis=1)))


def _ewald_cutoffs(
    volume: float, eta: float | None, tolerance: float
) -> tuple[float, float, float]:
    """Return ``(eta, real_cutoff, reciprocal_cutoff)`` shared by both Ewald sums.

    Both :func:`ewald_ion_ion_energy` and :func:`ewald_local_ionic_potential`
    use this so that, called with the same ``eta``/``tolerance`` on the same
    cell (the normal way to use them together for one calculation), they
    enumerate exactly the same real- and reciprocal-space lattice points.
    """

    if eta is None:
        eta = float(np.sqrt(np.pi) / volume ** (1.0 / 3.0))
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be between 0 and 1")
    real_cutoff = float(erfcinv(tolerance) / eta)
    reciprocal_cutoff = float(2.0 * eta * np.sqrt(-np.log(tolerance)))
    return eta, real_cutoff, reciprocal_cutoff


#: Truncation of both sums of the ion-ion energy and forces: the real-space terms end where ``erfc(eta r)`` and the
#: reciprocal ones where ``exp(-G^2/4 eta^2)`` fall below it.  The real-space terms left out add up to about
#: ``2 pi rho Z^2 tolerance / eta^2`` per atom (Hartree; ``rho`` atoms per bohr^3), 1e-14 Ry for 4,096 carbon atoms of
#: diamond at the ``eta`` of :func:`ion_ion_splitting`, far below the 1e-12 Ry asked of the energy (1e-14 left
#: 1e-10 Ry there); the reciprocal terms left out are smaller still.
ION_ION_TOLERANCE = 1.0e-18

# Costs of the parts of the two sums (seconds; one laptop core, numpy/scipy, 2026-11-08), which choose ``eta``: a
# real-space pair (tree query, erfc and sum), an image atom of the tree, an element of the phase product of one
# atom, a complex multiply-add of the structure-factor matrix product.
_PAIR_SECONDS = 3.0e-7
_IMAGE_SECONDS = 2.0e-7
_PHASE_SECONDS = 1.7e-8
_PRODUCT_SECONDS = 1.5e-10
_PAIR_BLOCK = 1 << 21  # atom pairs of one block of the real-space sum (24 bytes each from the tree)
_PHASE_ELEMENTS = 1 << 22  # complex numbers of the phase product of one block of atoms (64 MiB)


def _ion_ion_inputs(atoms, potentials, lattice_vectors) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """``(lattice, volume, fractional, positions, charges)`` with every atom translated into the cell.

    The energy does not depend on which image of an atom the coordinates name; the sums below need the
    fractional coordinates in ``[0, 1]``.  An atom already inside keeps its position bit for bit.
    """

    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    if lattice.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3)")
    volume = float(abs(np.linalg.det(lattice)))
    if volume <= 0.0:
        raise ValueError("lattice_vectors must span a nondegenerate cell")
    positions = np.stack([np.asarray(atom.position, dtype=np.float64) for atom in atoms]).reshape(-1, 3)
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)
    fractional = positions @ np.linalg.inv(lattice)
    shift = np.floor(fractional)
    moved = np.any(shift != 0.0, axis=1)
    if np.any(moved):
        positions = positions.copy()
        positions[moved] -= shift[moved] @ lattice
        fractional = fractional - shift
    return lattice, volume, np.clip(fractional, 0.0, 1.0), positions, charges


def _reciprocal_half_widths(lattice: np.ndarray, cutoff: float) -> np.ndarray:
    """Largest ``|m_i|`` of a reciprocal vector ``G = sum_i m_i b_i`` with ``|G| <= cutoff``: ``m_i = G.a_i/2 pi``."""

    return np.floor(cutoff * np.linalg.norm(lattice, axis=1) / (2.0 * np.pi)).astype(np.int64)


def ion_ion_splitting(lattice_vectors: np.ndarray, atom_count: int, tolerance: float = ION_ION_TOLERANCE) -> float:
    """The Ewald parameter ``eta`` (1/bohr) of the ion-ion sums of a cell of ``atom_count`` atoms.

    Any ``eta`` gives the same energy; it moves work between the real-space pairs (``~ N^2 r_c^3/V`` with
    ``r_c = erfcinv(tolerance)/eta``) and the reciprocal sum (``~ N`` times the box of ``|m_i| <= G_c |a_i|/2 pi``,
    ``G_c = 2 eta sqrt(-ln tolerance)``).  This is the one of least estimated cost on the grid
    ``sqrt(pi)/V^(1/3) 2^(k/4)`` (the first factor the parameter of the direct sums before 2026-11-08): about
    0.4/bohr for 4,096 atoms of diamond (``r_c`` 15 bohr, 400 neighbours per atom, a box of 90^3 vectors).
    """

    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    volume = float(abs(np.linalg.det(lattice)))
    count = max(int(atom_count), 1)
    real_scale = float(erfcinv(tolerance))
    reciprocal_scale = 2.0 * np.sqrt(-np.log(tolerance))
    spacings = lattice_plane_spacings(lattice)
    reference = float(np.sqrt(np.pi) / volume ** (1.0 / 3.0))
    best_cost, best_eta = np.inf, reference
    for step in range(-8, 61):
        eta = reference * 2.0 ** (step / 4.0)
        cutoff = real_scale / eta
        pairs = count * count / volume * (4.0 / 3.0) * np.pi * cutoff**3
        images = count * float(np.prod(1.0 + 2.0 * cutoff / spacings))
        half = np.sort(_reciprocal_half_widths(lattice, reciprocal_scale * eta))
        plane = float((2 * half[0] + 1) * (2 * half[1] + 1))
        cost = (
            _PAIR_SECONDS * pairs
            + _IMAGE_SECONDS * images
            + count * plane * (_PHASE_SECONDS + _PRODUCT_SECONDS * (half[2] + 1))
        )
        if cost < best_cost:
            best_cost, best_eta = cost, eta
    return best_eta


def _real_space_pairs(positions: np.ndarray, fractional: np.ndarray, lattice: np.ndarray, cutoff: float):
    """Yield ``(a, b, displacement, distance)`` of every pair term of the real-space sum, in blocks of atoms ``a``.

    The terms are the atoms ``a`` of the cell and the images ``r_b + n.A`` (``n`` integer) within ``cutoff`` of
    them, without ``a`` itself (``b = a``, ``n = 0``); ``displacement = r_a - r_b - n.A``.  The candidate images
    are those with every fractional coordinate in ``[-c_i, 1 + c_i]``, ``c_i = cutoff/d_i`` (``d_i``: the spacing
    of the lattice planes ``i``), which holds every image within ``cutoff`` of the cell; a k-d tree of them gives
    the pairs (any cell, oblique or small).  Each pair ``(a, b, n)`` comes once, its mirror ``(b, a, -n)`` once.
    """

    from scipy.spatial import cKDTree

    count = len(positions)
    reach = cutoff / lattice_plane_spacings(lattice)
    span = np.ceil(reach).astype(np.int64) + 1
    axes = [np.arange(-s, s + 1, dtype=np.int64) for s in span]
    translations = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    origin = np.flatnonzero(np.all(translations == 0, axis=1))
    translations = np.concatenate((translations[origin], np.delete(translations, origin, axis=0)))
    shifted = fractional[None, :, :] + translations[:, None, :]
    inside = np.all((shifted >= -reach) & (shifted <= 1.0 + reach), axis=2)
    which, atom = np.nonzero(inside)  # translation-major: the translation n = 0 first, its rows the atoms
    image_atoms = atom
    image_positions = positions[atom] + translations[which].astype(np.float64) @ lattice
    tree = cKDTree(image_positions)
    neighbours = len(image_positions) / max(float(np.prod(1.0 + 2.0 * reach)), 1.0) * (4.0 / 3.0) * np.pi * (
        reach.max() ** 3
    )
    block = max(1, int(_PAIR_BLOCK // max(neighbours, 1.0)))
    for start in range(0, count, block):
        stop = min(start + block, count)
        found = cKDTree(positions[start:stop]).sparse_distance_matrix(tree, cutoff, output_type="ndarray")
        rows = found["i"].astype(np.int64) + start
        columns = found["j"].astype(np.int64)
        keep = columns != rows  # the image rows 0..N-1 are the atoms themselves (n = 0)
        rows, columns = rows[keep], columns[keep]
        distance = found["v"][keep]
        if np.any(distance <= 0.0):
            raise ValueError("two nuclei occupy the same position")
        yield rows, image_atoms[columns], positions[rows] - image_positions[columns], distance


def _reciprocal_box(lattice: np.ndarray, eta: float, tolerance: float):
    """The half box of reciprocal vectors of the sums: ``(axes, frequencies, weight, vectors)``.

    ``G = sum_i m_i b_i`` with ``|m_i| <= G_c |a_i|/2 pi``, ``G_c = 2 eta sqrt(-ln tolerance)``.  ``axes`` orders the
    lattice directions with the longest frequency range first; that range is halved, ``m >= 0`` (``S(-G)`` is the
    conjugate of ``S(G)``).  ``weight[m]`` is ``exp(-G^2/4 eta^2)/G^2`` for ``0 < |G| <= G_c`` and 0 elsewhere,
    doubled for ``m_0 > 0`` (``G`` stands for ``-G`` too); ``vectors`` returns the Cartesian components of ``G``.
    """

    cutoff = 2.0 * eta * np.sqrt(-np.log(tolerance))
    half = _reciprocal_half_widths(lattice, cutoff)
    axes = tuple(int(axis) for axis in np.argsort(-half, kind="stable"))
    half = half[list(axes)]
    frequencies = (
        np.arange(0, half[0] + 1, dtype=np.int64),
        np.arange(-half[1], half[1] + 1, dtype=np.int64),
        np.arange(-half[2], half[2] + 1, dtype=np.int64),
    )
    reciprocal = _reciprocal_lattice(lattice)[list(axes)]  # rows b_i in the order of ``axes``
    m0, m1, m2 = frequencies[0][:, None, None], frequencies[1][None, :, None], frequencies[2][None, None, :]

    def vectors(component: int) -> np.ndarray:
        return m0 * reciprocal[0, component] + m1 * reciprocal[1, component] + m2 * reciprocal[2, component]

    g_squared = sum(vectors(component) ** 2 for component in range(3))
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.exp(-g_squared / (4.0 * eta**2)) / g_squared
    weight[(g_squared == 0.0) | (g_squared > cutoff**2)] = 0.0
    weight[1:] *= 2.0
    return axes, frequencies, weight, vectors


def _axis_phases(fractional: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """``exp(2 pi i f m)`` of the atoms (rows) and the frequencies ``m`` (columns) of one axis."""

    return np.exp(2.0j * np.pi * np.multiply.outer(fractional, frequencies.astype(np.float64)))


def _structure_factors(fractional: np.ndarray, charges: np.ndarray, frequencies) -> np.ndarray:
    """``S(G) = sum_a Z_a exp(i G.r_a)`` over the box, ``G.r_a = 2 pi sum_i m_i f_ai``.

    The phase is a product of one factor per axis, so ``S`` of a block of atoms is one matrix product of the
    first axis's phases with the products of the other two: ``N x M_0 x M_1 x M_2`` multiply-adds in all.
    """

    m0, m1, m2 = (len(values) for values in frequencies)
    structure = np.zeros((m0, m1 * m2), dtype=np.complex128)
    block = max(1, _PHASE_ELEMENTS // (m1 * m2))
    buffer = np.empty((min(block, len(charges)), m1, m2), dtype=np.complex128)
    for start in range(0, len(charges), block):
        stop = min(start + block, len(charges))
        first = charges[start:stop, None] * _axis_phases(fractional[start:stop, 0], frequencies[0])
        product = np.multiply(
            _axis_phases(fractional[start:stop, 1], frequencies[1])[:, :, None],
            _axis_phases(fractional[start:stop, 2], frequencies[2])[:, None, :],
            out=buffer[: stop - start],
        )
        structure += first.T @ product.reshape(stop - start, m1 * m2)
    return structure.reshape(m0, m1, m2)


def ewald_ion_ion_terms(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    lattice_vectors: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = ION_ION_TOLERANCE,
    forces: bool = False,
) -> tuple[float, np.ndarray | None]:
    """The Ewald ion-ion energy (Ry) of one cell and, with ``forces``, its ``(atoms, 3)`` forces (Ry/bohr).

    In Hartree units, with ``d = |r_a - r_b + T|``, ``w(G) = exp(-G^2/4 eta^2)/G^2`` and
    ``S(G) = sum_a Z_a exp(i G.r_a)``,

    ``E = 1/2 sum'_(a,b,T) Z_a Z_b erfc(eta d)/d + (2 pi/V) sum_(G != 0) w(G) |S(G)|^2
          - eta/sqrt(pi) sum_a Z_a^2 - pi/(2 eta^2 V) (sum_a Z_a)^2``,

    the prime leaving out ``a = b`` at ``T = 0``, and

    ``F_a = sum'_(b,T) Z_a Z_b [erfc(eta d)/d^2 + 2 eta exp(-eta^2 d^2)/(sqrt(pi) d)] (r_a - r_b + T)/d
            + (4 pi/V) Z_a sum_(G != 0) w(G) G Im[exp(i G.r_a) S(G)^*]``;

    both are doubled for Rydberg units.  The real-space terms run over the pairs within
    ``r_c = erfcinv(tolerance)/eta`` (:func:`_real_space_pairs`), the reciprocal ones over ``|G| <= 2 eta
    sqrt(-ln tolerance)`` (:func:`_reciprocal_box`, :func:`_structure_factors`), so the forces are the derivative of
    exactly the sum the energy evaluates.  ``eta`` defaults to :func:`ion_ion_splitting`.  The cost is
    ``O(N^2 r_c^3/V + N M^3)``: 1-2 s for 4,096 atoms, where the direct sum over every translation and atom pair
    of the code before 2026-11-08 took 200 s.
    """

    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be between 0 and 1")
    count = len(atoms)
    if count == 0:
        lattice = np.asarray(lattice_vectors, dtype=np.float64)
        if lattice.shape != (3, 3):
            raise ValueError("lattice_vectors must have shape (3, 3)")
        if float(abs(np.linalg.det(lattice))) <= 0.0:
            raise ValueError("lattice_vectors must span a nondegenerate cell")
        return 0.0, (np.zeros((0, 3)) if forces else None)
    lattice, volume, fractional, positions, charges = _ion_ion_inputs(atoms, potentials, lattice_vectors)
    if eta is None:
        eta = ion_ion_splitting(lattice, count, tolerance)
    eta = float(eta)
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    cutoff = float(erfcinv(tolerance) / eta)
    force = np.zeros((count, 3)) if forces else None

    real_space_energy = 0.0
    for rows, others, displacement, distance in _real_space_pairs(positions, fractional, lattice, cutoff):
        products = charges[rows] * charges[others]
        screened = erfc(eta * distance)
        real_space_energy += 0.5 * float(np.sum(products * screened / distance))
        if forces:
            # -d/dd [erfc(eta d)/d], over d for the unit vector
            radial = products * (
                screened / distance**2 + 2.0 * eta * np.exp(-((eta * distance) ** 2)) / (np.sqrt(np.pi) * distance)
            ) / distance
            for component in range(3):
                force[:, component] += np.bincount(rows, weights=radial * displacement[:, component], minlength=count)

    axes, frequencies, weight, vectors = _reciprocal_box(lattice, eta, tolerance)
    ordered = fractional[:, list(axes)]
    structure = _structure_factors(ordered, charges, frequencies)
    reciprocal_energy = (2.0 * np.pi / volume) * float(np.sum(weight * (structure.real**2 + structure.imag**2)))
    if forces:
        _reciprocal_forces(force, ordered, charges, frequencies, weight, vectors, structure, volume)

    self_energy = -(eta / np.sqrt(np.pi)) * float(np.sum(charges**2))
    neutralizing_energy = -(np.pi / (2.0 * eta**2 * volume)) * float(np.sum(charges)) ** 2
    hartree_energy = real_space_energy + reciprocal_energy + self_energy + neutralizing_energy
    # PARSEC's Rydberg convention, matching ion_ion_energy's factor of two.
    return 2.0 * hartree_energy, (None if force is None else 2.0 * force)


def _reciprocal_forces(force, fractional, charges, frequencies, weight, vectors, structure, volume) -> None:
    """``force += (4 pi/V) Z_a sum_G w(G) G Im[exp(i G.r_a) S(G)^*]`` (Hartree) over the half box of ``weight``.

    ``C_k(G) = w(G) G_k S(G)^*`` contracted with the phases of every atom one axis after the other: a matrix product
    over the last axis, then two sums per atom.
    """

    m0, m1, m2 = (len(values) for values in frequencies)
    coefficients = np.stack([weight * vectors(component) for component in range(3)]) * np.conj(structure)[None]
    coefficients = coefficients.reshape(3 * m0 * m1, m2)
    block = max(1, _PHASE_ELEMENTS // (3 * m0 * m1))
    for start in range(0, len(charges), block):
        stop = min(start + block, len(charges))
        last = coefficients @ _axis_phases(fractional[start:stop, 2], frequencies[2]).T
        middle = np.einsum(
            "kxyb,by->kxb", last.reshape(3, m0, m1, stop - start),
            _axis_phases(fractional[start:stop, 1], frequencies[1]),
        )
        total = np.einsum("kxb,bx->bk", middle, _axis_phases(fractional[start:stop, 0], frequencies[0]))
        force[start:stop] += (4.0 * np.pi / volume) * charges[start:stop, None] * total.imag


def ewald_ion_ion_energy(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    lattice_vectors: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = ION_ION_TOLERANCE,
) -> float:
    """Return the periodic pairwise ion-ion repulsion in Rydberg.

    The Ewald sum of :func:`ewald_ion_ion_terms` for one unit cell (each ion counted once, not once per image), the
    neutralizing background included.  Atoms may be given outside the cell.
    """

    energy, _ = ewald_ion_ion_terms(atoms, potentials, lattice_vectors, eta=eta, tolerance=tolerance)
    return energy


def _erf_over_r(charge: float, radius: np.ndarray, eta: float) -> np.ndarray:
    """Return ``2*charge*erf(eta*r)/r``, with the finite ``r=0`` limit."""

    with np.errstate(invalid="ignore", divide="ignore"):
        values = 2.0 * charge * erf(eta * radius) / radius
    at_origin = radius == 0.0
    if np.any(at_origin):
        values = np.where(at_origin, 4.0 * charge * eta / np.sqrt(np.pi), values)
    return values


# The Fortran periodic local potential (``ionpbc.f90``) is the reciprocal-space sum of the form factors ``vql(G)`` of
# ``fourier_f.f90``: it never reads ``Cubic_Spline`` (``uspline`` is used for the cluster potential, the projectors, the
# core density and the cluster forces), and the form factors integrate the table with an eight-point rule, so the
# potential is the accurate one whatever the flag says.  Linear interpolation of the table in real space is not: its
# error (Delta r / r = 1 % on the logarithmic grid) shifts the Ge3 total energy by +0.7 mRy against Fortran.  The periodic
# local potential and its force therefore use the cubic spline of the table whatever ``Cubic_Spline`` says.  Setting this
# to ``False`` gives the linear mode (PARSEC's derivative table, a few percent off the interpolant's derivative).
LOCAL_POTENTIAL_USES_CUBIC_SPLINE = True


def local_potential_uses_spline(specification: SpeciesPotential, grid=None) -> bool:
    """Whether the periodic local potential of a species is evaluated with the cubic spline of its table.

    The grid of a run of the Fortran-derived routines (``Use_Fortran_Routines``,
    :func:`parsec_python.Grid.pbc.uses_fortran_routines`) follows the species' ``Cubic_Spline``, as the parity round
    did; every other grid takes :data:`LOCAL_POTENTIAL_USES_CUBIC_SPLINE` (Andrew Grieder's routines).
    """

    if grid is not None and uses_fortran_routines(grid):
        return bool(specification.use_spline)
    return LOCAL_POTENTIAL_USES_CUBIC_SPLINE


_local_potential_provider = None
_provider_converged_for = None

CONVERGED_TOLERANCE = 1.0e-26
"""Ewald tolerance at which the direct sum is converged (the images it still leaves out are below
round-off); a provider that returns the converged sum is equal to the direct sum at this tolerance."""


def set_local_potential_provider(provider=None, *, converged_for=None) -> None:
    """Install (or, with ``None``, remove) a faster :func:`ewald_local_ionic_potential`.

    ``provider(grid, atoms, potentials, specifications, lattice_vectors, eta, tolerance)`` returns the
    potential, or ``None`` for an input it does not handle (the direct sum below then runs).  The
    performance counterpart lives in ``parsec_python.acceleration.V_ion.periodic_setup``.
    ``converged_for(grid, lattice_vectors)`` says for which inputs the provider returns the
    converged sum; :func:`local_potential_tolerance` then reports :data:`CONVERGED_TOLERANCE`, so
    that the periodic local forces differentiate the potential the SCF used.
    """
    global _local_potential_provider, _provider_converged_for
    _local_potential_provider = provider
    _provider_converged_for = None if provider is None else converged_for


def local_potential_tolerance(grid, lattice_vectors: np.ndarray, tolerance: float) -> float:
    """The Ewald tolerance of the sum :func:`ewald_local_ionic_potential` returns for these inputs.

    ``tolerance`` for the direct sum; :data:`CONVERGED_TOLERANCE` where an installed provider
    returns the converged sum (:func:`set_local_potential_provider`).  The derivative of the
    potential (:func:`parsec_python.Forces.periodic.periodic_local_forces`) uses it.
    """
    if (
        _local_potential_provider is not None
        and _provider_converged_for is not None
        and _provider_converged_for(grid, np.asarray(lattice_vectors, dtype=np.float64))
    ):
        return min(float(tolerance), CONVERGED_TOLERANCE)
    return float(tolerance)


def ewald_local_image_sets(
    lattice_vectors: np.ndarray, eta: float | None, tolerance: float
) -> tuple[float, float, np.ndarray, float, np.ndarray, np.ndarray]:
    """Return ``(eta, real_cutoff, cell_center, cell_radius, translations, reciprocal_points)`` of the local potential.

    The short-ranged piece of an atom image at ``position + T`` reaches the grid point ``r`` when
    ``|r - position - T| <= real_cutoff``.  Grid points lie anywhere in the cell, so an image counts
    when its center is within ``real_cutoff + radius`` of the cell center (``radius``: circumscribed
    radius of the cell); ``position`` itself is inside the cell, which bounds ``|T|``.  ``translations`` holds
    the candidates (the caller keeps those within ``real_cutoff + radius`` of ``cell_center`` for its atom) and
    ``reciprocal_points`` the vectors ``G != 0`` of the series.  The potential and its forces
    (:func:`parsec_python.Forces.periodic_local.periodic_local_forces`) both enumerate them here.
    """

    volume = float(abs(np.linalg.det(lattice_vectors)))
    eta, real_cutoff, reciprocal_cutoff = _ewald_cutoffs(volume, eta, tolerance)
    reciprocal_vectors = _reciprocal_lattice(lattice_vectors)
    cell_center, cell_radius = _cell_center_and_radius(lattice_vectors)
    translations = _lattice_points(lattice_vectors, reciprocal_vectors, real_cutoff + 2.0 * cell_radius)
    reciprocal_points = _lattice_points(reciprocal_vectors, lattice_vectors, reciprocal_cutoff)
    reciprocal_points = reciprocal_points[np.any(reciprocal_points != 0.0, axis=1)]
    return eta, real_cutoff, cell_center, cell_radius, translations, reciprocal_points


def ewald_local_ionic_potential(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = 1.0e-14,
) -> np.ndarray:
    """Return the periodic local ionic potential on ``grid``, in Rydberg.

    This applies the same real-space/reciprocal-space Ewald split as
    :func:`ewald_ion_ion_energy` to that tail.  For one atom of charge
    ``Z_a``, decompose

    ``V_a(r) = [V_a(r) + 2*Z_a*erf(eta*r)/r]  +  [-2*Z_a*erf(eta*r)/r]``.

    The first bracket equals ``V_a(r)`` near the atom (where ``V_a`` is the
    tabulated pseudopotential) and equals ``-2*Z_a*erfc(eta*r)/r`` beyond the
    table (where ``V_a(r) = -2*Z_a/r`` exactly) -- either way it decays like a
    Gaussian tail, so summing it over periodic images converges with just a
    handful of nearby image shells, the same real-space cutoff construction
    used by ``ewald_ion_ion_energy``.

    The second bracket, summed over every lattice translation, is the
    periodic Coulomb potential of a periodized Gaussian charge distribution.
    By the standard Ewald/Poisson-summation identity, its ``G != 0``
    reciprocal-space Fourier series is

    ``sum_R (-2*Z_a*erf(eta*|r-R|)/|r-R|)
        = -2*Z_a * (4*pi/V_cell) * sum_{G!=0} exp(-G**2/(4*eta**2))/G**2
              * cos(G.(r-r_a))  +  (a G=0 constant, per unit charge)``.

    """

    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float64)
    if lattice_vectors.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3)")
    volume = float(abs(np.linalg.det(lattice_vectors)))
    if volume <= 0.0:
        raise ValueError("lattice_vectors must span a nondegenerate cell")

    total = np.zeros(grid.size, dtype=np.float64)
    if not atoms:
        return total

    if _local_potential_provider is not None:
        provided = _local_potential_provider(
            grid, atoms, potentials, specifications, lattice_vectors, eta, tolerance
        )
        if provided is not None:
            return provided

    if uses_fortran_routines(grid):
        # The Fortran-derived routines (Use_Fortran_Routines): the images of the parity round, also for an
        # atom given outside the cell.
        eta, real_cutoff, reciprocal_cutoff = _ewald_cutoffs(volume, eta, tolerance)
        reciprocal_vectors = _reciprocal_lattice(lattice_vectors)
        # The short-ranged piece of an atom image at ``position + T`` reaches the grid point ``r`` when
        # ``|r - position - T| <= real_cutoff``.  Grid points lie anywhere in the cell, so an image counts
        # when its center is within ``real_cutoff + radius`` of the cell center (``radius``: circumscribed
        # radius of the cell); ``position`` itself is inside the cell, which bounds ``|T|``
        # (``atom_reach`` keeps the bound for an atom given outside the cell).
        cell_center, cell_radius = _cell_center_and_radius(lattice_vectors)
        atom_reach = max([cell_radius] + [float(np.linalg.norm(np.asarray(atom.position) - cell_center)) for atom in atoms])
        all_translations = _lattice_points(
            lattice_vectors, reciprocal_vectors, real_cutoff + cell_radius + atom_reach
        )
        reciprocal_points = _lattice_points(
            reciprocal_vectors, lattice_vectors, reciprocal_cutoff
        )
        reciprocal_points = reciprocal_points[np.any(reciprocal_points != 0.0, axis=1)]
    else:
        eta, real_cutoff, cell_center, cell_radius, all_translations, reciprocal_points = ewald_local_image_sets(
            lattice_vectors, eta, tolerance
        )
    g_squared = (
        np.sum(reciprocal_points**2, axis=1) if reciprocal_points.size else None
    )
    reciprocal_weight = (
        np.exp(-g_squared / (4.0 * eta**2)) / g_squared
        if g_squared is not None
        else None
    )

    for atom in atoms:
        specification = specifications[atom.symbol]
        potential = potentials[atom.symbol]
        charge = potential.ionic_charge
        position = np.asarray(atom.position, dtype=np.float64)

        # Short-range piece: sum over the few nearby periodic images.
        reaching = np.linalg.norm(position + all_translations - cell_center, axis=1) <= real_cutoff + cell_radius
        for translation in all_translations[reaching]:
            displaced = position + translation
            radius = np.linalg.norm(grid.coordinates - displaced, axis=1)
            total += potential.local_potential(
                radius,
                specification.local_angular_momentum,
                use_spline=local_potential_uses_spline(specification, grid),
                spline_padding_width=grid.settings.stencil_half_width,
            )
            total += _erf_over_r(charge, radius, eta)

        # Long-range piece: reciprocal-space sum, G=0 dropped (see docstring).
        if reciprocal_points.size:
            displacement = grid.coordinates - position
            phase = displacement @ reciprocal_points.T
            total += -2.0 * charge * (4.0 * np.pi / volume) * (
                np.cos(phase) @ reciprocal_weight
            )

    return total


def ewald_alpha_z_energy(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    lattice_vectors: np.ndarray,
    electron_count: float,
    *,
    eta: float | None = None,
    tolerance: float = 1.0e-14,
) -> float:
    """Return the "alpha_Z" total-energy correction, in Rydberg.
    """

    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float64)
    if lattice_vectors.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3)")
    volume = float(abs(np.linalg.det(lattice_vectors)))
    if volume <= 0.0:
        raise ValueError("lattice_vectors must span a nondegenerate cell")
    if not atoms:
        return 0.0

    eta, _, _ = _ewald_cutoffs(volume, eta, tolerance)
    total_ionic_charge = float(
        sum(potentials[atom.symbol].ionic_charge for atom in atoms)
    )
    return float(electron_count) * (2.0 * np.pi / (eta**2 * volume)) * total_ionic_charge


__all__ = [
    "ION_ION_TOLERANCE",
    "LOCAL_POTENTIAL_USES_CUBIC_SPLINE",
    "ewald_alpha_z_energy",
    "ewald_ion_ion_energy",
    "ewald_ion_ion_terms",
    "ewald_local_ionic_potential",
    "ion_ion_splitting",
    "local_potential_tolerance",
    "local_potential_uses_spline",
    "set_local_potential_provider",
]
