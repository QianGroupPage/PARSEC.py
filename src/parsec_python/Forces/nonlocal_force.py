"""Forces of the Kleinman-Bylander projectors (``forcnloc.f90``).

With normalized projectors ``beta_q`` of sign ``s_q`` centred on atom ``a``
and spin-degenerate orbitals of occupation ``f_n`` in ``[0, 1]``, the nonlocal
energy is ``E_NL = 2*sum_n f_n sum_q s_q <beta_q|psi_n>**2``.  A projector
depends on the atom position only through ``r - R_a``, so

``F_a = -dE_NL/dR_a
      = 4*sum_n f_n sum_(q on a) s_q <beta_q|psi_n> <grad beta_q|psi_n>``,

where ``grad`` is the gradient with respect to the electron coordinate.  For
``beta(r) = w(|d|)*Y(d/|d|)`` with ``d = r - R_a``, ``nonloc.F90`` stores

``grad beta = w(|d|)*grad Y + w'(|d|)*Y*d/|d|``.

The radial derivative ``w'`` is the table ``dvw`` of ``pseudo.f90``.  Both
overlaps are plain sums over the grid because orbitals and projectors carry
the factor ``sqrt(dV)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from ..Grid import RealSpaceGrid
from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential, ParsecRadialSpline
from ..Pseudopotential.radial_derivatives import (
    radial_projector_with_derivative,
    spline_derivative,
)
from ..V_ion import NonlocalProjectorOperator
from ..V_ion import ionic_potential as _ionic
from ..V_ion.ionic_potential import _projector_support_radius


def real_harmonics_with_gradients(
    angular_momentum: int,
    directions: np.ndarray,
    inverse_radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Real harmonics and their Cartesian gradients through ``l = 3``.

    ``directions`` holds ``d/|d|`` and ``inverse_radius`` holds ``1/|d|``.
    The first result has shape ``(points, 2l+1)`` and the column order of
    :func:`parsec_python.V_ion.real_spherical_harmonics`; the second has
    shape ``(points, 2l+1, 3)``.

    Each harmonic is a polynomial ``P`` of the direction ``u``.  The chain
    rule through ``u = d/|d|`` gives

    ``grad P = (dP/du - u*(u . dP/du))/|d|``,

    which is what ``ylm_real`` of ``nonloc.F90`` writes out term by term.
    """

    unit = np.asarray(directions, dtype=np.float64)
    if unit.ndim != 2 or unit.shape[1] != 3:
        raise ValueError("directions must have shape (n, 3)")
    x, y, z = unit.T
    zero = np.zeros_like(x)
    one = np.ones_like(x)

    if angular_momentum == 0:
        values = [0.28209479177387814 * one]
        slopes = [(zero, zero, zero)]
    elif angular_momentum == 1:
        c = 0.4886025119029199
        values = [c * x, c * y, c * z]
        slopes = [(c * one, zero, zero), (zero, c * one, zero), (zero, zero, c * one)]
    elif angular_momentum == 2:
        c = 1.0925484305920792
        k = 0.31539156525252005
        values = [
            c * x * y,
            c * y * z,
            c * x * z,
            k * (3.0 * z * z - 1.0),
            0.5 * c * (x * x - y * y),
        ]
        slopes = [
            (c * y, c * x, zero),
            (zero, c * z, c * y),
            (c * z, zero, c * x),
            (zero, zero, 6.0 * k * z),
            (c * x, -c * y, zero),
        ]
    elif angular_momentum == 3:
        a = 0.5900435899266435
        b = 2.890611442640554
        c = 0.4570457994644658
        k = 0.3731763325901154
        e = 1.445305721320277
        values = [
            a * y * (3.0 * x * x - y * y),
            b * x * y * z,
            c * y * (5.0 * z * z - 1.0),
            k * z * (5.0 * z * z - 3.0),
            c * x * (5.0 * z * z - 1.0),
            e * z * (x * x - y * y),
            a * x * (x * x - 3.0 * y * y),
        ]
        slopes = [
            (6.0 * a * x * y, 3.0 * a * (x * x - y * y), zero),
            (b * y * z, b * x * z, b * x * y),
            (zero, c * (5.0 * z * z - 1.0), 10.0 * c * y * z),
            (zero, zero, k * (15.0 * z * z - 3.0)),
            (c * (5.0 * z * z - 1.0), zero, 10.0 * c * x * z),
            (2.0 * e * x * z, -2.0 * e * y * z, e * (x * x - y * y)),
            (3.0 * a * (x * x - y * y), -6.0 * a * x * y, zero),
        ]
    else:
        raise ValueError(
            "PARSEC scalar projectors support angular momenta l=0 through l=3"
        )

    harmonics = np.column_stack(values)
    gradients = np.empty((unit.shape[0], len(values), 3), dtype=np.float64)
    inverse = np.asarray(inverse_radius, dtype=np.float64)
    for column, slope in enumerate(slopes):
        polynomial_gradient = np.column_stack(slope)
        radial_part = np.einsum("ij,ij->i", unit, polynomial_gradient)
        gradients[:, column, :] = (
            polynomial_gradient - radial_part[:, None] * unit
        ) * inverse[:, None]
    return harmonics, gradients


def _support_box_rows(grid: RealSpaceGrid, position: np.ndarray, radius: float) -> np.ndarray:
    """Rows of the active grid inside a cube that contains the support ball.

    The lookup table of the grid gives them without forming an array of all
    grid points for every atom.  One extra cell covers rounding at the faces;
    the caller applies the exact sphere.  Rows come out in grid order.
    """

    shift = np.asarray(grid.settings.shift)
    lower = np.floor((position - radius) / grid.spacing - shift).astype(np.int64) - 1
    upper = np.ceil((position + radius) / grid.spacing - shift).astype(np.int64) + 1
    lower = np.maximum(lower, grid.index_min) - grid.index_min
    upper = np.minimum(upper, grid.index_max) - grid.index_min
    if np.any(lower > upper):
        return np.empty(0, dtype=np.int64)
    rows = grid.lookup[
        tuple(slice(int(first), int(last) + 1) for first, last in zip(lower, upper))
    ]
    return np.sort(rows[rows >= 0])


@dataclass(frozen=True)
class NonlocalProjectorGradients:
    """``grad beta_q`` for every column of a nonlocal projector operator.

    ``components[c]`` is a CSC matrix of shape ``(grid points, projectors)``
    whose column ``q`` is the Cartesian component ``c`` of the gradient of
    projector ``q`` with respect to the electron coordinate.  There is one
    column for every ``(atom, l, m)`` with ``l`` not the local channel, in
    the order of the projector builders; ``labels[q]`` names it, so that the
    columns can be matched with those of any projector operator.
    """

    components: tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix]
    signs: np.ndarray
    labels: tuple[tuple[int, int, int], ...]


def build_nonlocal_projector_gradients(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
) -> NonlocalProjectorGradients:
    """Sample the gradients of all KB projectors on the active grid.

    The loops, column order, support radius and radial interpolation are
    those of :func:`parsec_python.V_ion.build_nonlocal_projectors` for an
    isolated system.  As in ``nonloc.F90``, a distance below the first radial
    sample is replaced by that sample, also in the direction ``d/|d|``.
    """

    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] = ([], [], [])
    signs: list[float] = []
    labels: list[tuple[int, int, int]] = []
    column = 0
    sqrt_dv = np.sqrt(grid.volume_element)
    tables: dict[tuple[str, int], tuple] = {}

    for atom_index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        local_l = specification.local_angular_momentum
        position = np.asarray(atom.position, dtype=np.float64)

        support_radius = _projector_support_radius(potential)
        box_rows = _support_box_rows(grid, position, support_radius)
        relative = grid.coordinates[box_rows] - position
        radius = np.linalg.norm(relative, axis=1)
        support = radius <= support_radius
        support_rows = box_rows[support]
        relative = relative[support]
        clamped = np.maximum(radius[support], potential.radii[0])
        inverse = 1.0 / clamped
        unit = relative * inverse[:, None]

        for angular_momentum in sorted(potential.radial_wavefunctions):
            if angular_momentum == local_l:
                continue
            # The radial tables belong to the species, not to the atom.
            key = (atom.symbol, angular_momentum)
            if key not in tables:
                radial_grid, radial_slope, sign = radial_projector_with_derivative(
                    potential, angular_momentum, local_l
                )
                spline = (
                    ParsecRadialSpline.from_positive_grid(
                        potential.radii, radial_grid, grid.settings.stencil_half_width
                    )
                    if specification.use_spline
                    else None
                )
                tables[key] = (radial_grid, radial_slope, sign, spline)
            radial_grid, radial_slope, sign, spline = tables[key]
            if spline is not None:
                radial = spline(clamped)
                slope = spline_derivative(spline, clamped)
            else:
                radial = np.interp(clamped, potential.radii, radial_grid, right=0.0)
                slope = np.interp(clamped, potential.radii, radial_slope, right=0.0)

            harmonics, harmonic_gradients = real_harmonics_with_gradients(
                angular_momentum, unit, inverse
            )
            for harmonic_index in range(2 * angular_momentum + 1):
                signs.append(sign)
                labels.append((atom_index, angular_momentum, harmonic_index))
                gradient = sqrt_dv * (
                    radial[:, None] * harmonic_gradients[:, harmonic_index, :]
                    + (slope * harmonics[:, harmonic_index])[:, None] * unit
                )
                rows.append(support_rows)
                columns.append(np.full(support_rows.size, column, dtype=np.int64))
                for component in range(3):
                    values[component].append(gradient[:, component])
                column += 1

    def assemble(component: int) -> sp.csc_matrix:
        if column == 0:
            return sp.csc_matrix((grid.size, 0), dtype=float)
        return sp.coo_matrix(
            (
                np.concatenate(values[component]),
                (np.concatenate(rows), np.concatenate(columns)),
            ),
            shape=(grid.size, column),
        ).tocsc()

    return NonlocalProjectorGradients(
        components=(assemble(0), assemble(1), assemble(2)),
        signs=np.asarray(signs, dtype=float),
        labels=tuple(labels),
    )


# Below this distance (bohr) from an atom the radial gradient of a projector is taken at this distance.  The first samples of
# a POTRE table (r from 1e-6 bohr on) are rounding noise: u_l(r)/r flattens to a constant there instead of growing as r**l,
# and ``nonloc.F90``'s replacement of a shorter distance by the first sample then multiplies noise by 1/r for the
# gradient of the grid point of an atom that sits on a grid point (a p projector's gradient there came out 3 times
# too large at 1e-4 bohr and 800 times at the first sample).  From 1e-3 bohr the tables are smooth.
_PROJECTOR_GRADIENT_FLOOR = 1.0e-3


def build_periodic_projector_gradients(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    k_point: np.ndarray | None = None,
) -> NonlocalProjectorGradients:
    """Gradients of the periodic projectors of :func:`parsec_python.V_ion.build_nonlocal_projectors`.

    Column ``q`` is ``G_q(r) = e^{-ik.r} sum_T e^{ik.T} grad beta_q(r - R_a - T)``: the gradient of the projector
    with respect to the electron coordinate, with the Bloch factor of the projector but not differentiated
    (``e^{-ik.r}`` does not depend on the atom position).  Then ``d beta_q/d R_a = -G_q``, and the nonlocal energy
    ``E = sum_k w_k 2 sum_n f_n sum_q s_q |<beta_q|u_n>|**2`` has the force

    ``F_a = 4 sum_k w_k sum_n f_n sum_q s_q Re[conj(<beta_q|u_n>) <G_q|u_n>]``.

    The images, support radius, radial interpolation (spline or PARSEC's table) and column order are those of the
    projector builder; with ``k_point=None`` the columns are real (the Gamma point).  Within
    ``_PROJECTOR_GRADIENT_FLOOR`` of an atom the gradient is evaluated at that distance (see there).

    Known limitation: the projector of an atom that sits exactly on a grid point has no defined value at that point
    (the direction of a zero displacement; the harmonics of ``l >= 2`` evaluate to nonzero constants there), so the
    nonlocal energy has a small jump as the atom crosses the grid point.  The force is the derivative of the energy at
    the point, not of its smooth neighbours: for a hydrogen pair on a 0.7 bohr grid they differ by about 3 %.
    """

    complex_valued = k_point is not None
    value_dtype = complex if complex_valued else float
    k_point = None if k_point is None else np.asarray(k_point, dtype=float)
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] = ([], [], [])
    signs: list[float] = []
    labels: list[tuple[int, int, int]] = []
    column = 0
    sqrt_dv = np.sqrt(grid.volume_element)
    tables: dict[tuple[str, int], tuple] = {}

    for atom_index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        local_l = specification.local_angular_momentum
        position = np.asarray(atom.position, dtype=np.float64)

        support_radius = _projector_support_radius(potential)
        images = []
        for translation in _ionic._periodic_image_translations(lattice_vectors, support_radius):
            support_rows, relative, radius = _ionic._support_finder(grid, position + translation, support_radius)
            if not support_rows.size:
                continue
            phase = (
                1.0
                if k_point is None
                else np.exp(-1j * ((grid.coordinates[support_rows] - translation) @ k_point))
            )
            clamped = np.maximum(radius, max(potential.radii[0], _PROJECTOR_GRADIENT_FLOOR))
            inverse = 1.0 / clamped
            images.append((support_rows, relative * inverse[:, None], clamped, inverse, phase))
        if not images:
            continue

        for angular_momentum in sorted(potential.radial_wavefunctions):
            if angular_momentum == local_l:
                continue
            key = (atom.symbol, angular_momentum)
            if key not in tables:
                radial_grid, radial_slope, sign = radial_projector_with_derivative(
                    potential, angular_momentum, local_l
                )
                spline = (
                    ParsecRadialSpline.from_positive_grid(
                        potential.radii, radial_grid, grid.settings.stencil_half_width
                    )
                    if specification.use_spline
                    else None
                )
                tables[key] = (radial_grid, radial_slope, sign, spline)
            radial_grid, radial_slope, sign, spline = tables[key]
            channel_columns = list(range(column, column + 2 * angular_momentum + 1))
            column += len(channel_columns)
            for harmonic_index in range(len(channel_columns)):
                signs.append(sign)
                labels.append((atom_index, angular_momentum, harmonic_index))

            for support_rows, unit, clamped, inverse, phase in images:
                if spline is not None:
                    radial = spline(clamped)
                    slope = spline_derivative(spline, clamped)
                else:
                    radial = np.interp(clamped, potential.radii, radial_grid, right=0.0)
                    slope = np.interp(clamped, potential.radii, radial_slope, right=0.0)
                harmonics, harmonic_gradients = real_harmonics_with_gradients(angular_momentum, unit, inverse)
                for harmonic_index, target in enumerate(channel_columns):
                    scale = sqrt_dv if k_point is None else (phase * sqrt_dv)[:, None]
                    gradient = scale * (
                        radial[:, None] * harmonic_gradients[:, harmonic_index, :]
                        + (slope * harmonics[:, harmonic_index])[:, None] * unit
                    )
                    rows.append(support_rows)
                    columns.append(np.full(support_rows.size, target, dtype=np.int64))
                    for component in range(3):
                        values[component].append(gradient[:, component].astype(value_dtype, copy=False))

    def assemble(component: int) -> sp.csc_matrix:
        if column == 0:
            return sp.csc_matrix((grid.size, 0), dtype=value_dtype)
        return sp.coo_matrix(
            (np.concatenate(values[component]), (np.concatenate(rows), np.concatenate(columns))),
            shape=(grid.size, column),
            dtype=value_dtype,
        ).tocsc()

    return NonlocalProjectorGradients(
        components=(assemble(0), assemble(1), assemble(2)),
        signs=np.asarray(signs, dtype=float),
        labels=tuple(labels),
    )


def contract_nonlocal_forces(
    projector_overlaps: np.ndarray,
    gradient_overlaps: Sequence[np.ndarray],
    signs: np.ndarray,
    labels: Sequence[tuple[int, int, int]],
    occupations: np.ndarray,
    atom_count: int,
) -> np.ndarray:
    """Sum ``4*f_n*s_q*<beta_q|psi_n>*<grad beta_q|psi_n>`` into atoms.

    ``projector_overlaps`` and each of the three ``gradient_overlaps`` have
    shape ``(projectors, states)``.  The result is ``(atoms, 3)`` in Ry/bohr.
    Complex overlaps (Bloch orbitals) contribute ``4*f_n*s_q*Re[conj(<beta_q|u_n>)*<grad beta_q|u_n>]``; for
    real overlaps this is the product above.
    """

    occupation = np.asarray(occupations)
    owners = np.fromiter((label[0] for label in labels), dtype=np.int64, count=len(labels))
    forces = np.zeros((atom_count, 3), dtype=np.float64)
    if np.iscomplexobj(projector_overlaps) or any(np.iscomplexobj(component) for component in gradient_overlaps):
        # Bloch orbitals: 4 f s Re[conj(<beta|u>) <G|u>]
        overlaps = np.asarray(projector_overlaps)
        for component in range(3):
            coupling = np.real(np.conj(overlaps) * np.asarray(gradient_overlaps[component]))
            per_projector = 4.0 * np.asarray(signs) * (coupling @ occupation)
            forces[:, component] = np.bincount(owners, weights=per_projector, minlength=atom_count)
        return forces
    overlaps = np.asarray(projector_overlaps, dtype=np.float64)
    weights = 4.0 * np.asarray(signs)[:, None] * overlaps * occupation[None, :]
    for component in range(3):
        per_projector = np.einsum(
            "qn,qn->q", weights, np.asarray(gradient_overlaps[component])
        )
        forces[:, component] = np.bincount(
            owners, weights=per_projector, minlength=atom_count
        )
    return forces


def gradient_columns_for(
    operator_labels: Sequence[tuple[int, int, int]],
    gradient_labels: Sequence[tuple[int, int, int]],
) -> np.ndarray | slice:
    """Gradient columns in the column order of a projector operator.

    The two normally have the same labels in the same order, and the result
    is then the full slice.  A projector builder may leave out an atom whose
    support holds no grid point; the gradient columns are then selected by
    label.
    """

    if tuple(operator_labels) == tuple(gradient_labels):
        return slice(None)
    position = {label: column for column, label in enumerate(gradient_labels)}
    try:
        return np.fromiter(
            (position[label] for label in operator_labels),
            dtype=np.int64,
            count=len(operator_labels),
        )
    except KeyError as error:
        raise ValueError(
            f"no projector gradient for projector {error.args[0]}"
        ) from error


def nonlocal_forces(
    operator: NonlocalProjectorOperator,
    gradients: NonlocalProjectorGradients,
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    atom_count: int,
) -> np.ndarray:
    """Return the ``(atoms, 3)`` nonlocal forces in Ry/bohr.

    ``wavefunctions`` holds one Euclidean-normalized grid vector per column
    and ``occupations`` the matching spatial-orbital occupations in
    ``[0, 1]`` (times the k-point weight for Bloch orbitals, which may be
    complex).  Empty states are skipped.
    """

    occupations = np.asarray(occupations, dtype=np.float64)
    if operator.projectors.shape[1] == 0 or occupations.size == 0:
        return np.zeros((atom_count, 3), dtype=np.float64)
    columns = gradient_columns_for(operator.labels, gradients.labels)
    occupied = np.flatnonzero(occupations != 0.0)
    orbitals = np.asarray(wavefunctions)[:, occupied]
    if not np.iscomplexobj(orbitals):
        orbitals = orbitals.astype(np.float64, copy=False)

    def overlap(matrix: sp.spmatrix) -> np.ndarray:
        # <beta|u> = beta^H u; for real projectors and orbitals the plain transpose
        return np.asarray((matrix.conj().T if np.iscomplexobj(matrix.data) else matrix.T) @ orbitals)

    overlaps = overlap(operator.projectors)
    gradient_overlaps = [overlap(component)[columns] for component in gradients.components]
    return contract_nonlocal_forces(
        overlaps,
        gradient_overlaps,
        operator.signs,
        operator.labels,
        occupations[occupied],
        atom_count,
    )


__all__ = [
    "NonlocalProjectorGradients",
    "build_nonlocal_projector_gradients",
    "build_periodic_projector_gradients",
    "contract_nonlocal_forces",
    "gradient_columns_for",
    "nonlocal_forces",
    "real_harmonics_with_gradients",
]
