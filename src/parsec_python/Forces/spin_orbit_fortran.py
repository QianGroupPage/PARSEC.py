"""Atomic forces of the spin-orbit and non-collinear drivers, in Ry/bohr (Fortran-derived routines).

This is the module ``Forces/spin_orbit.py`` of the parity round, used with
``Use_Fortran_Routines`` (:func:`parsec_python.Relaxation.backend.fortran_spinor_backend`); the default spinor
forces are Andrew Grieder's :mod:`parsec_python.Forces.spin_orbit` and
:func:`parsec_python.Forces.total.compute_spinor_forces`.

Fortran PARSEC adds, for ``elec_st%is_so``, the spin-orbit term of
``soforc.F90`` (``forcso``: ``flzsz`` and ``flsxy``, the derivatives of the
``lzsz``/``lsxy`` terms of ``zmatvec_so.f90`` with the projector gradients
``dv_ion``/``dv_SO`` of ``nonloc.F90``) to the Kleinman-Bylander forces of
``forcnloc.f90`` on the spinor components (``forcnloc.f90:233-337``).  The
local terms (``forcloc.f90``/``forpbc.f90``) take the total density and the
spin-independent part of the potentials, and the result is symmetrized and
its net force removed as for a scalar calculation.

This module computes the Hellmann-Feynman forces of the spinor drivers of
:mod:`parsec_python.SCF` (``run_self_consistent_soc``,
``run_self_consistent_soc_spin_polarized``, ``run_self_consistent_noncollinear``
and their k-point forms) as the derivative of the energy they compute:

* Kleinman-Bylander term: each spinor holds one electron and both of its
  components see the scalar projectors, ``F_a = 2 sum_n w f_n sum_sigma
  sum_q s_q Re(conj(<beta_q|psi_n,sigma>) <grad beta_q|psi_n,sigma>)``.
* Spin-orbit term: per atom ``H_SO = [v_ion | v_so] M [v_ion | v_so]^H``
  (the ``lzsz``/``lsxy`` structure, :func:`parsec_python.Hamiltonian.spinor_operator.spin_orbit_coefficients`),
  ``M`` Hermitian, so ``F_a = 2 sum_n w f_n Re sum_spin (grad P^H psi_n)^H
  (M P^H psi_n)`` with ``P = [v_ion | v_so]`` and its gradient sampled with
  the slope of the interpolant the projectors use (exact derivative).
* The Zeeman terms of collinear spin (``xc_delta``) and of non-collinear
  magnetism (``B_xc . sigma``) depend on the atoms only through the density
  and magnetization: no term of their own.

Deviations from the Fortran (``soforc.F90``, ``pseudo.f90:930-952``),
reported in the report of the parity round:

* ``dvr_ion`` is ``j0 vlj(i0) + i0 vlj(i0+2)``, a value and not a slope, and
  ``dvr_so = 2 (dvlj(i0) - dvlj(i0+2))`` lacks the factor
  ``sqrt(h^3)/(2l+1)`` of ``vsor``; ``forcso`` multiplies both overlaps of a
  product by the KB sign ``cc``, which then cancels.  Fortran's spin-orbit
  forces are therefore not the derivative of its energy; here they are.
* ``forcnloc.f90`` sums the KB term over the up components only, times
  ``4/nspin`` with ``nspin = 1``: equal to the sum over both components for
  Kramers-degenerate states (time reversal), not with a magnetization.  Here
  both components are summed.

Units: bohr, Ry.  This module does not print.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..Eigensolvers.perturbative_soc import (
    _SOC_ANGULAR_MOMENTA,
    AtomSpinOrbitProjectors,
    build_spin_orbit_projectors,
)
from ..Hamiltonian.spinor_operator import spin_orbit_coefficients
from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential
from ..V_ion.ionic_potential import _periodic_image_translations, _projector_support_radius


_C1 = np.sqrt(3.0 / (8.0 * np.pi))
_C10 = np.sqrt(3.0 / (4.0 * np.pi))
_C22 = np.sqrt(15.0 / (32.0 * np.pi))
_C21 = np.sqrt(15.0 / (8.0 * np.pi))
_C20 = np.sqrt(5.0 / (16.0 * np.pi))


def complex_harmonics_with_gradients(
    angular_momentum: int, unit: np.ndarray, inverse: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``Y_l^m(d/|d|)`` and ``grad_d Y_l^m``, ``m = -l..l``, for ``l = 1, 2``.

    The harmonics of :func:`parsec_python.Eigensolvers.perturbative_soc.complex_spherical_harmonics`
    as functions of the unit vector ``u = d/|d|`` (``unit``); their gradient
    in ``d`` is the tangential part of ``dY/du`` over ``|d|``:
    ``(dY/du - (u . dY/du) u) / |d|`` (``inverse = 1/|d|``, zero at the
    atom).  Returns ``(n, 2l+1)`` and ``(n, 2l+1, 3)`` complex arrays.
    """

    u = np.asarray(unit, dtype=float)
    x, y, z = u.T
    zero = np.zeros_like(x)
    one = np.ones_like(x)
    if angular_momentum == 1:
        values = np.column_stack((_C1 * (x - 1j * y), _C10 * z.astype(complex), -_C1 * (x + 1j * y)))
        derivatives = np.stack(
            (
                _C1 * np.stack((one, -1j * one, zero), axis=1),
                _C10 * np.stack((zero, zero, one), axis=1).astype(complex),
                -_C1 * np.stack((one, 1j * one, zero), axis=1),
            ),
            axis=1,
        )
    elif angular_momentum == 2:
        minus = x - 1j * y
        plus = x + 1j * y
        values = np.column_stack(
            (
                _C22 * minus * minus,
                _C21 * z * minus,
                _C20 * (3.0 * z * z - 1.0).astype(complex),
                -_C21 * z * plus,
                _C22 * plus * plus,
            )
        )
        derivatives = np.stack(
            (
                2.0 * _C22 * minus[:, None] * np.stack((one, -1j * one, zero), axis=1),
                _C21 * np.stack((z, -1j * z, minus), axis=1),
                _C20 * np.stack((zero, zero, 6.0 * z), axis=1).astype(complex),
                -_C21 * np.stack((z, 1j * z, plus), axis=1),
                2.0 * _C22 * plus[:, None] * np.stack((one, 1j * one, zero), axis=1),
            ),
            axis=1,
        )
    else:
        raise ValueError("spin-orbit complex harmonics support only l=1 and l=2")
    radial_part = np.einsum("nmc,nc->nm", derivatives, u)
    gradients = (derivatives - radial_part[:, :, None] * u[:, None, :]) * np.asarray(
        inverse, dtype=float
    )[:, None, None]
    return values, gradients


@dataclass(frozen=True)
class SpinOrbitProjectorGradients:
    """``grad_r`` of one atom's ``[v_ion | v_so]`` (``nonloc.F90``'s ``dv_ion``/``dv_SO``).

    ``gradients[c]`` has the shape ``(len(support_rows), 16)`` of
    :attr:`parsec_python.Eigensolvers.perturbative_soc.AtomSpinOrbitProjectors.stacked`,
    on the same rows.
    """

    atom_index: int
    support_rows: np.ndarray
    gradients: np.ndarray


def _round_off_distance() -> float:
    from .periodic import ROUND_OFF_DISTANCE

    return ROUND_OFF_DISTANCE


def _interpolant_slope(points, knots, values):
    from .periodic import interpolant_slope

    return interpolant_slope(points, knots, values)


def build_spin_orbit_projector_gradients(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray | None = None,
    k_point: np.ndarray | None = None,
) -> list[SpinOrbitProjectorGradients]:
    """Gradients of the projectors of :func:`build_spin_orbit_projectors`.

    The same atoms, images, union of support rows and Bloch phases
    ``exp(-i k.(r - T))``; each image contributes
    ``sqrt(dV) [R'(|d|) Y d/|d| + R grad Y]`` with ``R'`` the slope of the
    linear interpolant of the radial table (``np.interp`` with the first value
    continued inwards and zero beyond the table), the exact derivative of the
    sampled projector.  The phase does not depend on the atom position, so
    ``-grad_r`` of a projector is its derivative with respect to the atom.
    With ``grid.settings.double_grid_order > 1`` the gradients are those of
    the double-grid projectors (:func:`_double_grid_spin_orbit_projector_gradients`).
    """

    order = int(getattr(getattr(grid, "settings", None), "double_grid_order", 1) or 1)
    if order > 1:
        return _double_grid_spin_orbit_projector_gradients(
            grid, atoms, potentials, specifications, lattice_vectors, k_point, order
        )
    sqrt_dv = np.sqrt(grid.volume_element)
    coordinates = grid.coordinates
    result: list[SpinOrbitProjectorGradients] = []
    for atom_index, atom in enumerate(atoms):
        specification = specifications[atom.symbol]
        if not specification.spin_orbit:
            continue
        potential = potentials[atom.symbol]
        local_l = specification.local_angular_momentum
        support_radius = _projector_support_radius(potential)
        position = np.asarray(atom.position, dtype=np.float64)
        translations = (
            np.zeros((1, 3))
            if lattice_vectors is None
            else _periodic_image_translations(lattice_vectors, support_radius)
        )
        images = []
        for translation in translations:
            relative = coordinates - (position + translation)
            radius = np.linalg.norm(relative, axis=1)
            support = radius <= support_radius
            rows = np.flatnonzero(support)
            if rows.size:
                phase = (
                    1.0
                    if k_point is None
                    else np.exp(
                        -1j * ((coordinates[support] - translation) @ np.asarray(k_point, dtype=float))
                    )
                )
                images.append((rows, relative[support], radius[support], phase))
        if not images:
            continue
        union = np.unique(np.concatenate([image[0] for image in images]))
        gradients = np.zeros((3, union.size, 16), dtype=complex)
        column_by_l = {1: 0, 2: 3}
        for angular_momentum in _SOC_ANGULAR_MOMENTA:
            if angular_momentum not in potential.spin_orbit_channel_potentials:
                continue
            width = 2 * angular_momentum + 1
            column = column_by_l[angular_momentum]
            radial_ion, radial_so, _ = potential.spin_orbit_projectors(angular_momentum, local_l)
            for rows, relative, radius, phase in images:
                inverse = np.zeros_like(radius)
                nonzero = radius > 0.0
                # A round-off distance (an atom within round-off of a grid point) is clamped, as the tabulated
                # slopes and nonloc.F90 clamp to the first radial knot: its 1/|d| made w(r0) grad Y 1e6 (Si64,
                # h 0.513: nonlocal forces of 82 Ry/bohr).  ROUND_OFF_DISTANCE keeps the exact slope of a
                # table whose first knot is far out.  A grid point exactly on the atom keeps 1/|d| = 0.
                inverse[nonzero] = 1.0 / np.maximum(radius[nonzero], min(potential.radii[0], _round_off_distance()))
                unit = relative * inverse[:, None]
                harmonics, harmonic_gradients = complex_harmonics_with_gradients(
                    angular_momentum, unit, inverse
                )
                local_rows = np.searchsorted(union, rows)
                phase_column = phase if np.isscalar(phase) else phase[:, None]
                for offset, radial in ((0, radial_ion), (8, radial_so)):
                    value = np.interp(
                        radius, potential.radii, radial, left=radial[0], right=0.0
                    )
                    slope = _interpolant_slope(radius, potential.radii, radial)
                    for component in range(3):
                        gradients[
                            component, local_rows, offset + column : offset + column + width
                        ] += phase_column * sqrt_dv * (
                            (slope * unit[:, component])[:, None] * harmonics
                            + value[:, None] * harmonic_gradients[:, :, component]
                        )
        result.append(
            SpinOrbitProjectorGradients(
                atom_index=atom_index, support_rows=union, gradients=gradients
            )
        )
    return result


def _double_grid_spin_orbit_projector_gradients(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray | None,
    k_point: np.ndarray | None,
    order: int,
) -> list[SpinOrbitProjectorGradients]:
    """Gradients of the double-grid spin-orbit projectors (``nonloc.F90:575-612``).

    The projectors of
    :func:`parsec_python.Eigensolvers.perturbative_soc._double_grid_spin_orbit_projectors`:
    a coarse point of an image sums ``w_f R(|s_f|) Y(s_f)`` over its fine
    points ``s_f = r + d_f - R_a - T`` inside ``r_c`` (the weights of
    :func:`parsec_python.V_ion.double_grid.double_grid_support`) and carries
    the Bloch phase of the coarse point.  Each fine sample contributes the
    gradient of :func:`build_spin_orbit_projector_gradients` (slope of the
    linear interpolant, exact derivative of the sampled projector) times its
    weight; the same support rows (union over the images) as the projectors.
    As for the Kleinman-Bylander double grid, the support itself (which
    coarse points have a fine point inside ``r_c``) is held fixed: the
    radial tables vanish at ``r_c``.
    """

    from ..V_ion.double_grid import double_grid_support

    sqrt_dv = np.sqrt(grid.volume_element)
    spacing = float(grid.spacing)
    coordinates = grid.coordinates
    result: list[SpinOrbitProjectorGradients] = []
    for atom_index, atom in enumerate(atoms):
        specification = specifications[atom.symbol]
        if not specification.spin_orbit:
            continue
        potential = potentials[atom.symbol]
        local_l = specification.local_angular_momentum
        support_radius = _projector_support_radius(potential)
        position = np.asarray(atom.position, dtype=np.float64)
        translations = (
            np.zeros((1, 3))
            if lattice_vectors is None
            else _periodic_image_translations(lattice_vectors, support_radius)
        )
        images = []
        for translation in translations:
            support = double_grid_support(
                coordinates, position + translation, support_radius, order, spacing
            )
            if support.rows.size:
                phase = (
                    1.0
                    if k_point is None
                    else np.exp(
                        -1j
                        * (
                            (coordinates[support.rows] - translation)
                            @ np.asarray(k_point, dtype=float)
                        )
                    )
                )
                images.append((support, phase))
        if not images:
            continue
        union = np.unique(np.concatenate([support.rows for support, _ in images]))
        gradients = np.zeros((3, union.size, 16), dtype=complex)
        column_by_l = {1: 0, 2: 3}
        for angular_momentum in _SOC_ANGULAR_MOMENTA:
            if angular_momentum not in potential.spin_orbit_channel_potentials:
                continue
            width = 2 * angular_momentum + 1
            column = column_by_l[angular_momentum]
            radial_ion, radial_so, _ = potential.spin_orbit_projectors(angular_momentum, local_l)
            for support, phase in images:
                points, fine = support.radius.shape
                radius = support.radius.reshape(-1)
                relative = support.relative.reshape(-1, 3)
                inverse = np.zeros_like(radius)
                nonzero = radius > 0.0
                # A round-off distance (an atom within round-off of a grid point) is clamped, as the tabulated
                # slopes and nonloc.F90 clamp to the first radial knot: its 1/|d| made w(r0) grad Y 1e6 (Si64,
                # h 0.513: nonlocal forces of 82 Ry/bohr).  ROUND_OFF_DISTANCE keeps the exact slope of a
                # table whose first knot is far out.  A grid point exactly on the atom keeps 1/|d| = 0.
                inverse[nonzero] = 1.0 / np.maximum(radius[nonzero], min(potential.radii[0], _round_off_distance()))
                unit = relative * inverse[:, None]
                harmonics, harmonic_gradients = complex_harmonics_with_gradients(
                    angular_momentum, unit, inverse
                )
                local_rows = np.searchsorted(union, support.rows)
                phase_column = phase if np.isscalar(phase) else phase[:, None]
                for offset, radial in ((0, radial_ion), (8, radial_so)):
                    value = np.interp(
                        radius, potential.radii, radial, left=radial[0], right=0.0
                    )
                    slope = _interpolant_slope(radius, potential.radii, radial)
                    for component in range(3):
                        sample = (slope * unit[:, component])[:, None] * harmonics + value[
                            :, None
                        ] * harmonic_gradients[:, :, component]
                        summed = np.einsum(
                            "pf,pfm->pm", support.weights, sample.reshape(points, fine, width)
                        )
                        gradients[
                            component, local_rows, offset + column : offset + column + width
                        ] += phase_column * (sqrt_dv * summed)
        result.append(
            SpinOrbitProjectorGradients(
                atom_index=atom_index, support_rows=union, gradients=gradients
            )
        )
    return result


def spin_orbit_energy(
    projectors: Sequence[AtomSpinOrbitProjectors],
    spinors: np.ndarray,
    occupations: np.ndarray,
    *,
    weight: float = 1.0,
) -> float:
    """``w sum_n f_n <psi_n|H_SO|psi_n>`` (Ry), the energy the forces differentiate."""

    spinors = np.asarray(spinors)
    n_grid = spinors.shape[0] // 2
    occupations = np.asarray(occupations, dtype=float)
    energy = 0.0
    for atom in projectors:
        rows = atom.support_rows
        gathered = np.stack((spinors[rows], spinors[rows + n_grid]))
        overlaps = atom.stacked_conj_transpose @ gathered
        coefficients = spin_orbit_coefficients(atom, overlaps)
        energy += float(
            np.real(np.einsum("sqn,sqn,n->", np.conj(overlaps), coefficients, occupations))
        )
    return weight * energy


def spin_orbit_forces(
    projectors: Sequence[AtomSpinOrbitProjectors],
    gradients: Sequence[SpinOrbitProjectorGradients],
    spinors: np.ndarray,
    occupations: np.ndarray,
    atom_count: int,
    *,
    weight: float = 1.0,
) -> np.ndarray:
    """``-d/dR_a`` of :func:`spin_orbit_energy` at fixed spinors, ``(atoms, 3)``.

    ``F_a = 2 w sum_n f_n Re sum_spin (G_c^H psi_n)^H (M P^H psi_n)`` with
    ``P = [v_ion | v_so]`` of the atom and ``G_c`` its gradient along ``c``
    (``flzsz``/``flsxy`` of ``soforc.F90`` written with the exact slopes).
    ``spinors`` is ``(2 n_grid, n_states)``, occupations in ``[0, 1]``.
    """

    forces = np.zeros((atom_count, 3), dtype=np.float64)
    occupations = np.asarray(occupations, dtype=float)
    occupied = np.flatnonzero(occupations != 0.0)
    if occupied.size == 0:
        return forces
    spinors = np.asarray(spinors)[:, occupied]
    weights = occupations[occupied]
    n_grid = spinors.shape[0] // 2
    by_atom = {item.atom_index: item for item in gradients}
    for atom in projectors:
        gradient = by_atom[atom.atom_index]
        if not np.array_equal(gradient.support_rows, atom.support_rows):
            raise ValueError("spin-orbit projectors and gradients do not share their rows")
        rows = atom.support_rows
        gathered = np.stack((spinors[rows], spinors[rows + n_grid]))
        coefficients = spin_orbit_coefficients(atom, atom.stacked_conj_transpose @ gathered)
        for component in range(3):
            slopes = np.conj(gradient.gradients[component]).T @ gathered
            forces[atom.atom_index, component] += 2.0 * weight * float(
                np.real(np.einsum("sqn,sqn,n->", np.conj(slopes), coefficients, weights))
            )
    return forces


def spinor_nonlocal_forces(
    operator,
    gradients,
    spinors: np.ndarray,
    occupations: np.ndarray,
    atom_count: int,
    *,
    weight: float = 1.0,
) -> np.ndarray:
    """The Kleinman-Bylander forces of spinors: both components, one electron each."""

    from .periodic import periodic_nonlocal_forces

    spinors = np.asarray(spinors)
    n_grid = spinors.shape[0] // 2
    total = np.zeros((atom_count, 3), dtype=np.float64)
    for half in (spinors[:n_grid], spinors[n_grid:]):
        total += periodic_nonlocal_forces(
            operator, gradients, half, occupations, atom_count, weight=weight, degeneracy=1.0
        )
    return total


# ---------------------------------------------------------------------------
# Assembly.


def _spinor_blocks(result):
    """``(k_point or None, weight, spinors, occupations)`` of a spinor result."""

    occupations = np.asarray(result.occupations, dtype=float)
    k_points = getattr(result, "k_points", None)
    if k_points is None:
        yield None, 1.0, np.asarray(result.spinors), occupations
        return
    blocks = getattr(result, "k_spinors", None)
    if blocks is None:
        raise ValueError(
            "the k-point spin-orbit result holds the spinors of its first k-point "
            "only: run it with keep_wavefunctions=True to compute forces"
        )
    weights = np.asarray(result.k_weights, dtype=float)
    offset = 0
    for index, block in enumerate(blocks):
        count = np.asarray(block).shape[1]
        yield (
            np.asarray(k_points[index], dtype=float),
            float(weights[index]),
            np.asarray(block),
            occupations[offset : offset + count],
        )
        offset += count


def spinor_force_fields(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(density, V_out - V_in, V_xc)`` of a spinor result for the local terms.

    The spin-independent parts: the channel averages of collinear spin, the
    scalar part of a non-collinear potential.
    """

    density = np.asarray(result.density, dtype=np.float64)
    xc = getattr(result, "xc_potential", None)
    if xc is None:
        raise ValueError(
            "the spin-orbit result holds no exchange-correlation potential: it "
            "comes from a driver without forces"
        )
    inputs = getattr(result, "scf_input_potential", None)
    outputs = getattr(result, "scf_output_potential", None)
    residual = (
        np.zeros_like(density)
        if inputs is None or outputs is None
        else np.asarray(outputs, dtype=np.float64) - np.asarray(inputs, dtype=np.float64)
    )
    return density, residual, np.asarray(xc, dtype=np.float64)


def compute_spin_orbit_forces(system, result, *, symmetrize: bool = True, operations=None):
    """Forces of a finished spinor SCF loop (spin-orbit or non-collinear).

    ``system`` is the prepared isolated or periodic system and ``result``
    the :class:`parsec_python.models.SelfConsistentSOCResult` of one of the
    spinor drivers (k-point forms with ``keep_wavefunctions=True``).  The
    order of ``forcnloc.f90``: ion-ion + local, nonlocal (Kleinman-Bylander
    and spin-orbit), the TS term of a converged loop
    (:func:`parsec_python.Forces.total.correction_forces`, ``efvdw.f90``:
    it depends on the atoms and the total density only), symmetrization,
    net force removed.  With ``Double_Grid_Order > 1`` every projector
    gradient (Kleinman-Bylander and spin-orbit) is that of the double-grid
    projector the Hamiltonian applies.
    """

    from time import perf_counter

    from .ion_ion import ion_ion_forces, point_charge_forces
    from .local import local_forces
    from .nonlocal_force import build_nonlocal_projector_gradients
    from .ion_ion import ewald_ion_ion_forces
    from .periodic import (
        build_periodic_projector_gradients,
        periodic_force_operations,
        periodic_local_forces,
        radial_slope_setting,
        reduced_kpoint_operations,
    )
    from .symmetry import atom_symmetry_operations, symmetrize_forces
    from .total import assemble_forces, correction_forces

    started = perf_counter()
    atoms = system.atoms
    count = len(atoms)
    specifications = system.input.pseudopotentials
    cell = getattr(system.input, "periodic_cell", None)
    lattice = None if cell is None else np.asarray(cell.lattice_vectors, dtype=np.float64)
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    density, residual, xc = spinor_force_fields(result)

    charged = None
    if lattice is None:
        repulsion = ion_ion_forces(atoms, system.pseudopotentials)
        point_charges = getattr(system, "point_charges", ())
        if point_charges:
            # forcept.f90, as in Forces.total.compute_forces.
            charged = point_charge_forces(atoms, system.pseudopotentials, point_charges)
            repulsion = repulsion + charged
        local = local_forces(
            system.grid, atoms, system.pseudopotentials, specifications, density, residual, xc
        )
    else:
        repulsion = ewald_ion_ion_forces(atoms, system.pseudopotentials, lattice)
        local = periodic_local_forces(
            system.grid,
            atoms,
            system.pseudopotentials,
            specifications,
            lattice,
            density,
            residual,
            xc,
        )

    separable = np.zeros((count, 3))
    spin_orbit = np.zeros((count, 3))
    slopes = None if lattice is None else radial_slope_setting()
    for k_point, weight, spinors, occupations in _spinor_blocks(result):
        if lattice is None:
            operator = system.nonlocal_operator
            gradients = build_nonlocal_projector_gradients(
                system.grid, atoms, system.pseudopotentials, specifications
            )
        else:
            from ..V_ion import build_nonlocal_projectors

            operator = build_nonlocal_projectors(
                system.grid, atoms, system.pseudopotentials, specifications, lattice,
                k_point=k_point,
            )
            gradients = build_periodic_projector_gradients(
                system.grid, atoms, system.pseudopotentials, specifications, lattice,
                k_point, radial_slopes=slopes,
            )
        separable += spinor_nonlocal_forces(
            operator, gradients, spinors, occupations, count, weight=weight
        )
        soc_projectors = build_spin_orbit_projectors(
            system.grid, atoms, system.pseudopotentials, specifications, lattice, k_point
        )
        soc_gradients = build_spin_orbit_projector_gradients(
            system.grid, atoms, system.pseudopotentials, specifications, lattice, k_point
        )
        spin_orbit += spin_orbit_forces(
            soc_projectors, soc_gradients, spinors, occupations, count, weight=weight
        )
    if lattice is not None:
        completion = reduced_kpoint_operations(system, result)
        if completion is not None:
            separable = symmetrize_forces(separable, completion)
            spin_orbit = symmetrize_forces(spin_orbit, completion)
    if not symmetrize:
        operations = None
    elif operations is None:
        operations = (
            atom_symmetry_operations(atoms)
            if lattice is None
            else periodic_force_operations(system, getattr(result, "k_points", None))
        )
    return assemble_forces(
        positions,
        repulsion,
        local,
        separable,
        operations,
        seconds=perf_counter() - started,
        point_charge=charged,
        periodic=lattice is not None,
        keep_torque=True,
        spin_orbit=spin_orbit,
        # TS-vdW (the spinor drivers); LDA+U is refused with
        # spinors.  Empty without them: the call stays what it was.
        **correction_forces(system, result),
    )


__all__ = [
    "SpinOrbitProjectorGradients",
    "build_spin_orbit_projector_gradients",
    "complex_harmonics_with_gradients",
    "compute_spin_orbit_forces",
    "spin_orbit_energy",
    "spin_orbit_forces",
    "spinor_force_fields",
    "spinor_nonlocal_forces",
]
