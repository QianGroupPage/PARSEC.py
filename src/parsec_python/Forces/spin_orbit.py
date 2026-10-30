"""Forces of spinor orbitals: the spin-orbit projectors and the Kleinman-Bylander projectors acting on both spin components.

A spinor state ``n`` holds ``f_n`` electrons (``0 <= f_n <= 1``) and has the components ``u_n,up`` and ``u_n,down``.

*Scalar projectors.*  The Kleinman-Bylander projectors act alike on both components, so the nonlocal energy is the
spin-polarized one: ``E_NL = sum_n f_n sum_sigma sum_q s_q |<beta_q|u_n,sigma>|**2`` and the force of
:mod:`parsec_python.Forces.nonlocal_force` applies to the ``2 N`` columns of the two components at the weight ``f_n/2``
(the weight of one electron per state, as for a spin channel).

*Spin-orbit projectors.*  :func:`parsec_python.Hamiltonian.spinor_operator.apply_spin_orbit` writes the spin-orbit term
as a bilinear form in the overlaps ``o_(sigma,j) = <P_j|u_sigma>`` of the spinor with the ``16`` projector functions
``P_j = [V_ion,lm | V_so,lm]`` of each spin-orbit atom (complex spherical harmonics, KB sign ``s_lm`` per harmonic),

``E_SO = sum_n f_n sum_(sigma,j) conj(o_(sigma,j)) c_(sigma,j)(o)``,

where ``c(o)`` is linear in ``o`` (the coefficients that ``apply_spin_orbit`` multiplies the projectors with: ``lzsz``
on the same spin, ``lsxy`` ladders on the other).  A projector depends on the atom position only through ``r - R_a``,
so ``d o_(sigma,j)/dR_a = -<G_j|u_sigma>`` with ``G_j = grad_r P_j``, and, the coefficient map being Hermitian,

``F_a = -dE_SO/dR_a = 2 sum_n f_n Re sum_(sigma,j) conj(<G_j|u_n,sigma>) c_(sigma,j)(o_n)``.

Fortran's ``forcso`` (``soforc.F90``) differentiates the same bilinear form, with a warning that its spin-orbit forces
"may be inaccurate"; its ``flsxy`` pairs the same harmonic index on both sides where the Hamiltonian's ladder pairs
``lm`` with ``lm +- 1``, so it is not a reference for these forces.  They are exact derivatives of the energy of this
package (checked against finite differences of that energy at fixed orbitals in the unit tests).

The radial functions of the spin-orbit projectors are interpolated linearly in the Hamiltonian
(:func:`parsec_python.Eigensolvers.perturbative_soc.build_spin_orbit_projectors`), so the gradient takes the slope of that
interpolant and not a separate derivative table: the force is the derivative of the energy the SCF loop minimized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from ..Eigensolvers.perturbative_soc import AtomSpinOrbitProjectors
from ..Hamiltonian.spinor_operator import (
    _LADDER_FROM_DOWN,
    _LADDER_FROM_UP,
    _LZ_HALF,
    _QUARTER_LL1,
)
from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential
from ..V_ion import ionic_potential as _ionic
from ..V_ion.ionic_potential import _projector_support_radius
from .nonlocal_force import (
    _PROJECTOR_GRADIENT_FLOOR,
    NonlocalProjectorGradients,
    nonlocal_forces,
)

#: Device bytes one block of spinors and its overlaps may take.
_BLOCK_BYTES = 512 * 1024 * 1024

#: Radial functions the spin-orbit projectors have (``l = 1`` columns 0-2, ``l = 2`` columns 3-7).
_COLUMN_OF_L = {1: 0, 2: 3}


def complex_harmonics_with_gradients(
    angular_momentum: int, directions: np.ndarray, inverse_radius: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Complex harmonics ``Y_l^ml`` (``ml = -l..l``, Condon-Shortley) and their Cartesian gradients, ``l = 1, 2``.

    ``directions`` holds ``d/|d|`` and ``inverse_radius`` holds ``1/|d|``; the values are those of
    :func:`parsec_python.Eigensolvers.perturbative_soc.complex_spherical_harmonics`.  Each harmonic is a polynomial
    ``P`` of the direction ``u``; the chain rule through ``u = d/|d|`` gives ``grad P = (dP/du - u (u . dP/du))/|d|``.
    The results have shapes ``(points, 2l+1)`` and ``(points, 2l+1, 3)``.
    """

    unit = np.asarray(directions, dtype=np.float64)
    if unit.ndim != 2 or unit.shape[1] != 3:
        raise ValueError("directions must have shape (n, 3)")
    x, y, z = unit.T
    zero = np.zeros(x.shape, dtype=complex)
    if angular_momentum == 1:
        a = np.sqrt(3.0 / (8.0 * np.pi))
        b = np.sqrt(3.0 / (4.0 * np.pi))
        values = [a * (x - 1j * y), b * z + zero, -a * (x + 1j * y)]
        slopes = [
            (a + zero, -1j * a + zero, zero),
            (zero, zero, b + zero),
            (-a + zero, -1j * a + zero, zero),
        ]
    elif angular_momentum == 2:
        a = np.sqrt(15.0 / (32.0 * np.pi))
        b = np.sqrt(15.0 / (8.0 * np.pi))
        c = np.sqrt(5.0 / (16.0 * np.pi))
        xpiy, xmiy = x + 1j * y, x - 1j * y
        values = [
            a * xmiy * xmiy,
            b * z * xmiy,
            c * (3.0 * z * z - 1.0) + zero,
            -b * z * xpiy,
            a * xpiy * xpiy,
        ]
        slopes = [
            (2.0 * a * xmiy, -2j * a * xmiy, zero),
            (b * z + zero, -1j * b * z, b * xmiy),
            (zero, zero, 6.0 * c * z + zero),
            (-b * z + zero, -1j * b * z, -b * xpiy),
            (2.0 * a * xpiy, 2j * a * xpiy, zero),
        ]
    else:
        raise ValueError("spin-orbit complex harmonics support only l=1 and l=2")
    harmonics = np.column_stack(values)
    inverse = np.asarray(inverse_radius, dtype=np.float64)
    gradients = np.empty((unit.shape[0], len(values), 3), dtype=complex)
    for column, slope in enumerate(slopes):
        polynomial_gradient = np.column_stack(slope)
        radial_part = np.einsum("ij,ij->i", unit, polynomial_gradient)
        gradients[:, column, :] = (polynomial_gradient - radial_part[:, None] * unit) * inverse[:, None]
    return harmonics, gradients


def _interpolant_slope(radii: np.ndarray, values: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Slope of ``np.interp(radius, radii, values, left=values[0], right=0)``: zero outside the table."""

    last = radii.size - 1
    index = np.searchsorted(radii, radius, side="right") - 1
    inside = (index >= 0) & (index < last)
    clipped = np.clip(index, 0, last - 1)
    slope = (values[clipped + 1] - values[clipped]) / (radii[clipped + 1] - radii[clipped])
    return np.where(inside, slope, 0.0)


@dataclass(frozen=True)
class AtomSpinOrbitGradients:
    """``grad_r [V_ion | V_so]`` of one spin-orbit atom, on the support rows of its projectors.

    ``components`` has shape ``(3, rows, 16)``: the Cartesian component, the grid row of ``support_rows`` and the same
    ``16`` columns as :attr:`AtomSpinOrbitProjectors.stacked`.  For a k-point the Bloch factor of the projector is in
    the values but not differentiated (it does not depend on the atom position).
    """

    atom_index: int
    support_rows: np.ndarray
    components: np.ndarray


def build_spin_orbit_projector_gradients(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    projectors: Sequence[AtomSpinOrbitProjectors],
    lattice_vectors: np.ndarray | None = None,
    k_point: np.ndarray | None = None,
) -> list[AtomSpinOrbitGradients]:
    """Gradients of the projectors that :func:`build_spin_orbit_projectors` made, atom by atom.

    ``projectors`` are those projectors (the images, support radius, Bloch phase and harmonic columns are rebuilt from
    the same arguments, and the rows must come out equal).  Within ``_PROJECTOR_GRADIENT_FLOOR`` of an atom the
    gradient is evaluated at that distance, as for the scalar projectors.
    """

    sqrt_dv = np.sqrt(grid.volume_element)
    result: list[AtomSpinOrbitGradients] = []
    for atom_projectors in projectors:
        atom = atoms[atom_projectors.atom_index]
        potential = potentials[atom.symbol]
        local_l = specifications[atom.symbol].local_angular_momentum
        support_radius = _projector_support_radius(potential)
        position = np.asarray(atom.position, dtype=np.float64)
        translations = (
            np.zeros((1, 3))
            if lattice_vectors is None
            else _ionic._periodic_image_translations(lattice_vectors, support_radius)
        )
        rows_union = atom_projectors.support_rows
        components = np.zeros((3, rows_union.size, 16), dtype=complex)
        seen: list[np.ndarray] = []
        radial = {
            l: potential.spin_orbit_projectors(l, local_l)[:2]
            for l in _COLUMN_OF_L
            if l in potential.spin_orbit_channel_potentials
        }
        for translation in translations:
            relative = grid.coordinates - (position + translation)
            distance = np.linalg.norm(relative, axis=1)
            support = distance <= support_radius
            support_rows = np.flatnonzero(support)
            if not support_rows.size:
                continue
            seen.append(support_rows)
            scale = sqrt_dv * (
                np.ones(support_rows.size)
                if k_point is None
                else np.exp(-1j * ((grid.coordinates[support] - translation) @ np.asarray(k_point, dtype=float)))
            )
            clamped = np.maximum(distance[support], max(potential.radii[0], _PROJECTOR_GRADIENT_FLOOR))
            inverse = 1.0 / clamped
            unit = relative[support] * inverse[:, None]
            local_rows = np.searchsorted(rows_union, support_rows)
            for angular_momentum, (radial_ion, radial_so) in radial.items():
                width = 2 * angular_momentum + 1
                column = _COLUMN_OF_L[angular_momentum]
                harmonics, harmonic_gradients = complex_harmonics_with_gradients(angular_momentum, unit, inverse)
                for offset, table in ((0, radial_ion), (8, radial_so)):
                    value = np.interp(clamped, potential.radii, table, left=table[0], right=0.0)
                    slope = _interpolant_slope(potential.radii, table, clamped)
                    gradient = scale[:, None, None] * (
                        value[:, None, None] * harmonic_gradients
                        + (slope[:, None] * harmonics)[:, :, None] * unit[:, None, :]
                    )
                    for axis in range(3):
                        components[axis, local_rows, offset + column : offset + column + width] += gradient[:, :, axis]
        union = np.unique(np.concatenate(seen)) if seen else np.empty(0, dtype=np.int64)
        if not np.array_equal(union, rows_union):
            raise ValueError("the spin-orbit projectors were not built from these atoms, grid and lattice")
        result.append(AtomSpinOrbitGradients(atom_projectors.atom_index, rows_union, components))
    return result


@dataclass(frozen=True)
class SpinOrbitForceOperator:
    """The spin-orbit projectors and their gradients of all spin-orbit atoms as sparse matrices.

    ``overlap`` has shape ``(16 A, grid)`` and holds ``P^H`` (row ``16 a + j`` is ``conj(P_j)`` of the ``a``-th spin-orbit
    atom, ``j < 8`` the ``V_ion`` harmonics and ``j >= 8`` the ``V_so`` ones); ``gradient[c]`` the same for the
    component ``c`` of ``grad_r P``.  ``signs`` is ``(A, 8)``, the KB sign of each harmonic, and ``owners`` the index
    (into the atoms of the calculation) of each spin-orbit atom.
    """

    overlap: sp.csr_matrix
    gradient: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]
    signs: np.ndarray
    owners: np.ndarray

    @property
    def atom_count(self) -> int:
        return int(self.owners.size)


def _stack(blocks: list[sp.csr_matrix], grid_size: int) -> sp.csr_matrix:
    if not blocks:
        return sp.csr_matrix((0, grid_size), dtype=complex)
    return sp.csr_matrix(sp.vstack(blocks, format="csr"), dtype=complex)


def build_spin_orbit_force_operator(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    projectors: Sequence[AtomSpinOrbitProjectors],
    lattice_vectors: np.ndarray | None = None,
    k_point: np.ndarray | None = None,
) -> SpinOrbitForceOperator:
    """Sparse projectors and gradients of ``projectors`` (see :func:`build_spin_orbit_projector_gradients`)."""

    gradients = build_spin_orbit_projector_gradients(
        grid, atoms, potentials, specifications, projectors, lattice_vectors, k_point
    )
    overlap_blocks: list[sp.csr_matrix] = []
    gradient_blocks: list[list[sp.csr_matrix]] = [[], [], []]
    signs = np.zeros((len(projectors), 8))
    for index, (atom, gradient) in enumerate(zip(projectors, gradients)):
        rows = atom.support_rows
        columns = np.tile(rows, 16)
        row_index = np.repeat(np.arange(16), rows.size)

        def block(values: np.ndarray) -> sp.csr_matrix:  # values: (16, rows)
            return sp.csr_matrix((values.ravel(), (row_index, columns)), shape=(16, grid.size), dtype=complex)

        overlap_blocks.append(block(atom.stacked_conj_transpose))
        for axis in range(3):
            gradient_blocks[axis].append(block(gradient.components[axis].conj().T))
        signs[index, :3] = atom.sign[0]
        signs[index, 3:] = atom.sign[1]
    owners = np.fromiter((atom.atom_index for atom in projectors), dtype=np.int64, count=len(projectors))
    return SpinOrbitForceOperator(
        overlap=_stack(overlap_blocks, grid.size),
        gradient=tuple(_stack(blocks, grid.size) for blocks in gradient_blocks),  # type: ignore[arg-type]
        signs=signs,
        owners=owners,
    )


def spin_orbit_coefficients(xp, overlaps, signs):
    """The coefficients ``c(o)`` that ``apply_spin_orbit`` multiplies the projectors with, for all atoms at once.

    ``overlaps`` has shape ``(2, A, 16, states)`` (spin component, atom, projector column, state) and ``signs`` is
    ``(A, 8)``.  The result has the same shape.
    """

    sign = xp.asarray(signs)[None, :, :, None]
    io, so = sign * overlaps[:, :, :8], sign * overlaps[:, :, 8:]
    quarter = xp.asarray(_QUARTER_LL1)[None, :, None]
    coefficients = xp.empty_like(overlaps)
    for spin, (direction, ladder) in enumerate(((+1.0, _LADDER_FROM_DOWN), (-1.0, _LADDER_FROM_UP))):
        lz = xp.asarray(direction * _LZ_HALF)[None, :, None]
        ladder = xp.asarray(ladder.astype(np.complex128))  # same dtype as the overlaps: a plain matmul on a device
        other = 1 - spin
        coefficients[spin, :, :8] = lz * so[spin] + xp.matmul(ladder, so[other])
        coefficients[spin, :, 8:] = (
            lz * io[spin]
            + (quarter - 0.5 * lz) * so[spin]
            + xp.matmul(ladder, io[other] - 0.5 * so[other])
        )
    return coefficients


def _sparse_module(xp):
    if xp is np:
        return sp
    import cupyx.scipy.sparse as device_sparse  # type: ignore[import-not-found]

    return device_sparse


def spin_orbit_forces(
    operator: SpinOrbitForceOperator,
    spinors,
    weights: np.ndarray,
    atom_count: int,
    *,
    xp=np,
    sparse=None,
    block_bytes: int = _BLOCK_BYTES,
) -> np.ndarray:
    """``(atoms, 3)`` forces in Ry/bohr of the spin-orbit projectors on the spinors ``(2 grid, states)``.

    ``weights`` holds the electrons of each state times the k-point weight.  ``xp`` is the array module the spinors
    live in (CuPy keeps the orbitals of a k-point on its device and only the ``(atoms, 3)`` result comes back);
    ``sparse`` is its sparse module.
    """

    weights = np.asarray(weights, dtype=np.float64)
    forces = np.zeros((atom_count, 3), dtype=np.float64)
    occupied = np.flatnonzero(weights != 0.0)
    if operator.atom_count == 0 or not occupied.size:
        return forces
    sparse = _sparse_module(xp) if sparse is None else sparse
    overlap = sparse.csr_matrix(operator.overlap)
    gradient = [sparse.csr_matrix(component) for component in operator.gradient]
    grid_size = int(spinors.shape[0]) // 2
    atoms = operator.atom_count
    width = max(1, int(block_bytes // (16 * (2 * grid_size + 2 * 16 * atoms * 5))))
    per_atom = np.zeros((atoms, 3))
    for start in range(0, int(occupied.size), width):
        chosen = occupied[start : start + width]
        block = xp.ascontiguousarray(spinors[:, xp.asarray(chosen)]).astype(xp.complex128, copy=False)
        up, down = block[:grid_size], block[grid_size:]
        count = int(chosen.size)
        state_weights = xp.asarray(weights[chosen])
        overlaps = xp.stack((overlap @ up, overlap @ down)).reshape(2, atoms, 16, count)
        coefficients = spin_orbit_coefficients(xp, overlaps, operator.signs)
        for axis in range(3):
            slopes = xp.stack((gradient[axis] @ up, gradient[axis] @ down)).reshape(2, atoms, 16, count)
            coupling = xp.einsum("sajn,sajn->an", xp.conj(slopes), coefficients).real
            contribution = 2.0 * (coupling @ state_weights)
            per_atom[:, axis] += np.asarray(contribution.get() if hasattr(contribution, "get") else contribution)
    for axis in range(3):
        forces[:, axis] = np.bincount(operator.owners, weights=per_atom[:, axis], minlength=atom_count)
    return forces


def spinor_nonlocal_forces(
    nonlocal_operator,
    scalar_gradients: NonlocalProjectorGradients,
    spin_orbit: SpinOrbitForceOperator | None,
    spinors,
    weights: np.ndarray,
    atom_count: int,
    *,
    xp=np,
) -> np.ndarray:
    """``(atoms, 3)`` nonlocal force of spinor orbitals: the KB projectors on both components plus the spin-orbit term.

    ``spinors`` has shape ``(2 grid, states)`` and ``weights`` carries ``f_n`` times the k-point weight (the electrons of
    each state).  The scalar projectors give the force of the up and the down component at the weight ``f_n/2``.
    """

    weights = np.asarray(weights, dtype=np.float64)
    grid_size = int(spinors.shape[0]) // 2
    halves = 0.5 * weights
    total = np.zeros((atom_count, 3))
    for component in (spinors[:grid_size], spinors[grid_size:]):
        if xp is not np and not isinstance(component, np.ndarray):
            # the orbitals of a device stay there: the acceleration package supplies the contraction
            from ..acceleration.Forces.kpoint_nonlocal_device import device_nonlocal_forces

            total += device_nonlocal_forces(xp, nonlocal_operator, scalar_gradients, component, halves, atom_count)
        else:
            total += nonlocal_forces(nonlocal_operator, scalar_gradients, np.asarray(component), halves, atom_count)
    if spin_orbit is not None:
        total += spin_orbit_forces(spin_orbit, spinors, weights, atom_count, xp=xp)
    return total


__all__ = [
    "AtomSpinOrbitGradients",
    "SpinOrbitForceOperator",
    "build_spin_orbit_force_operator",
    "build_spin_orbit_projector_gradients",
    "complex_harmonics_with_gradients",
    "spin_orbit_coefficients",
    "spin_orbit_forces",
    "spinor_nonlocal_forces",
]
