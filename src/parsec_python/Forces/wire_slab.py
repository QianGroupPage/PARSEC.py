"""Forces of a wire or slab calculation (``forcloc_wire.f90``, ``forcloc_slab.f90``).

The local terms follow the split of the local potential into the short-range
part ``V_short = V_loc + 2 Z erf(d/sigma)/d`` and the potential of the
Gaussian charges (:mod:`parsec_python.V_ion.wire_slab`).  For atom ``a`` and
its images ``R_a + T`` (Fortran's ``|m_i| <= nint(6 sigma/|a_i|) + 1``), with
``d = r - R_a - T``:

``F_l = sum_r n(r) V_short'(|d|) d/|d|``,
``F_c = sum_r dV_hxc(r) rho_atom'(|d|) d/|d|`` (the non-self-consistency
correction, ``dV_hxc`` the last ``V_out - V_in``),
``F_core = sum_r V_xc(r) rho_core'(|d|) d/|d|``,
``F_g = sum_r grad V_H(r) n_g,a(d)``,

all times ``h^3``; grid points closer than the first radial sample are left
out of every term (``forcloc_wire.f90:186-187``).  ``F_g`` is the force on
the Gaussian charge of the atom in the Hartree field of the electrons
(``-V_G`` enters the local potential); the gradient is the finite-difference
gradient of the grid (zero outside the domain), as ``gradmvs``.

The ion-ion forces (``forceion_wire.f90``, ``ewald_slab.f90``, charged
sheets) are those of the prepared system.  The nonlocal forces are those of
:mod:`parsec_python.Forces.nonlocal_force` with the projector gradients
summed over the images along the periodic directions (``nonloc.F90``).  As
``forcnloc.f90`` the forces are averaged over the space group of the wire or
slab (``symm_forces.f90``; a slab has none unless Fortran turns its symmetry
on), the net force is removed and no torque is reported
(``forcnloc.f90:550``).  Collinear spin: the total density, the channel
averages of ``V_xc`` and ``V_out - V_in``, and the nonlocal term of every
channel (Gamma through :func:`parsec_python.SCF.run_spin_scf`, k-points with
the orbitals the spin k-point loop keeps).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.special import erf

from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential, ParsecRadialSpline
from ..Pseudopotential.radial_derivatives import (
    evaluate_density_derivative,
    evaluate_local_potential_derivative,
    radial_projector_with_derivative,
    spline_derivative,
)
from ..V_ion.ionic_potential import _projector_support_radius
from ..V_ion.wire_slab import gaussian_width
from .local import LocalForceTerms, species_force_tables
from .nonlocal_force import (
    NonlocalProjectorGradients,
    nonlocal_forces,
    real_harmonics_with_gradients,
)
from .total import ForceResult, assemble_forces

#: Residual of the k-point re-diagonalization of the nonlocal forces (Ry).
KPOINT_FORCE_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class WireSlabLocalForceTerms(LocalForceTerms):
    """The four local terms of ``forcloc_wire.f90``/``forcloc_slab.f90``.

    ``ionic`` is ``F_l`` (the short-range potential), ``gaussian`` ``F_g``.
    """

    gaussian: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.ionic + self.self_consistency + self.core_correction + self.gaussian


@dataclass(frozen=True)
class WireSlabForceResult(ForceResult):
    """A :class:`~parsec_python.Forces.ForceResult` of a wire or slab.

    PARSEC prints no torque for a periodic system.
    """

    omit_torque = True


def wire_slab_local_forces(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    density: np.ndarray,
    potential_residual: np.ndarray,
    xc_potential: np.ndarray,
    hartree_gradient: np.ndarray,
) -> WireSlabLocalForceTerms:
    """The local force terms of a wire or slab (Ry/bohr, ``(atoms, 3)`` each).

    ``hartree_gradient`` is the Cartesian gradient of the Hartree potential,
    ``(3, n)``; the other fields are those of
    :func:`parsec_python.Forces.local.local_forces`.
    """

    density = np.asarray(density, dtype=np.float64)
    residual = np.asarray(potential_residual, dtype=np.float64)
    xc = np.asarray(xc_potential, dtype=np.float64)
    hartree_gradient = np.asarray(hartree_gradient, dtype=np.float64)
    count = len(atoms)
    ionic = np.zeros((count, 3))
    correction = np.zeros((count, 3))
    core = np.zeros((count, 3))
    gaussian = np.zeros((count, 3))
    tables = {}
    half_width = grid.settings.stencil_half_width
    for index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        if atom.symbol not in tables:
            tables[atom.symbol] = species_force_tables(potential, specification)
        species = tables[atom.symbol]
        charge = float(potential.ionic_charge)
        sigma = gaussian_width(potential)
        density_factor = charge / sigma**3 / np.sqrt(np.pi) / np.pi
        position = np.asarray(atom.position, dtype=np.float64)
        for translation in grid.compensation_image_translations(position, sigma):
            relative = grid.coordinates - (position + translation)
            radius = np.sqrt(np.einsum("ij,ij->i", relative, relative))
            included = radius >= potential.radii[0]
            inverse = np.zeros_like(radius)
            inverse[included] = 1.0 / radius[included]
            slope = evaluate_local_potential_derivative(
                potential,
                radius,
                specification.local_angular_momentum,
                table=species.local_potential_derivative,
                use_spline=specification.use_spline,
                spline_padding_width=half_width,
            )
            # d/dr [2 Z erf(r/sigma)/r] (forcloc_wire.f90:252-257).
            ratio = radius[included] / sigma
            gaussian_slope = np.zeros_like(radius)
            gaussian_slope[included] = 2.0 * charge * (
                2.0 / (np.sqrt(np.pi) * sigma) * np.exp(-ratio**2) * inverse[included]
                - erf(ratio) * inverse[included] ** 2
            )
            ionic[index] += ((slope + gaussian_slope) * density * inverse) @ relative
            atomic_slope = evaluate_density_derivative(
                potential,
                radius,
                species.valence_density,
                table=species.valence_density_derivative,
            )
            correction[index] += (atomic_slope * residual * inverse) @ relative
            if species.core_density_derivative is not None:
                core_slope = evaluate_density_derivative(
                    potential,
                    radius,
                    potential.core_density,
                    table=species.core_density_derivative,
                    use_spline=specification.use_spline,
                    spline_padding_width=half_width,
                )
                core[index] += (core_slope * xc * inverse) @ relative
            charge_density = np.where(
                included, density_factor * np.exp(-(radius / sigma) ** 2), 0.0
            )
            gaussian[index] += hartree_gradient @ charge_density
    volume = grid.volume_element
    return WireSlabLocalForceTerms(
        ionic=ionic * volume,
        self_consistency=correction * volume,
        core_correction=core * volume,
        gaussian=gaussian * volume,
    )


def wire_slab_projector_gradients(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    k_point: np.ndarray | None = None,
) -> NonlocalProjectorGradients:
    """Gradients of the Kleinman-Bylander projectors of a wire or slab.

    Every ``(atom, l, m)`` column is summed over the images of the atom
    along the periodic directions, as the projectors of
    :func:`parsec_python.V_ion.build_nonlocal_projectors` are; ``k_point``
    (Cartesian 1/bohr) gives each image the Bloch phase ``exp(-i k.(r - T))``
    of those projectors.  As in ``nonloc.F90`` a distance below the first
    radial sample is replaced by that sample.
    """

    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: tuple[list, list, list] = ([], [], [])
    signs: list[float] = []
    labels: list[tuple[int, int, int]] = []
    column = 0
    sqrt_dv = np.sqrt(grid.volume_element)
    complex_valued = k_point is not None
    k_point = None if k_point is None else np.asarray(k_point, dtype=np.float64)
    for atom_index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        local_l = specification.local_angular_momentum
        position = np.asarray(atom.position, dtype=np.float64)
        support_radius = _projector_support_radius(potential)
        images = []
        for translation in grid.image_translations(position, support_radius):
            relative = grid.coordinates - (position + translation)
            radius = np.linalg.norm(relative, axis=1)
            support = radius <= support_radius
            if not np.any(support):
                continue
            rows_here = np.flatnonzero(support)
            clamped = np.maximum(radius[support], potential.radii[0])
            phase = (
                1.0
                if k_point is None
                else np.exp(-1j * ((grid.coordinates[support] - translation) @ k_point))
            )
            images.append((rows_here, relative[support], clamped, phase))
        if not images:
            continue
        for angular_momentum in sorted(potential.radial_wavefunctions):
            if angular_momentum == local_l:
                continue
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
            channel = list(range(column, column + 2 * angular_momentum + 1))
            column += len(channel)
            for harmonic_index in range(len(channel)):
                signs.append(sign)
                labels.append((atom_index, angular_momentum, harmonic_index))
            for rows_here, relative, clamped, phase in images:
                inverse = 1.0 / clamped
                unit = relative * inverse[:, None]
                if spline is not None:
                    radial = spline(clamped)
                    slope = spline_derivative(spline, clamped)
                else:
                    radial = np.interp(clamped, potential.radii, radial_grid, right=0.0)
                    slope = np.interp(clamped, potential.radii, radial_slope, right=0.0)
                harmonics, harmonic_gradients = real_harmonics_with_gradients(
                    angular_momentum, unit, inverse
                )
                for harmonic_index, target in enumerate(channel):
                    gradient = sqrt_dv * (
                        radial[:, None] * harmonic_gradients[:, harmonic_index, :]
                        + (slope * harmonics[:, harmonic_index])[:, None] * unit
                    )
                    if complex_valued:
                        gradient = gradient * phase[:, None]
                    rows.append(rows_here)
                    columns.append(np.full(rows_here.size, target, dtype=np.int64))
                    for component in range(3):
                        values[component].append(gradient[:, component])

    dtype = complex if complex_valued else float

    def assemble(component: int) -> sp.csc_matrix:
        if column == 0:
            return sp.csc_matrix((grid.size, 0), dtype=dtype)
        return sp.coo_matrix(
            (
                np.concatenate(values[component]),
                (np.concatenate(rows), np.concatenate(columns)),
            ),
            shape=(grid.size, column),
            dtype=dtype,
        ).tocsc()

    return NonlocalProjectorGradients(
        components=(assemble(0), assemble(1), assemble(2)),
        signs=np.asarray(signs, dtype=float),
        labels=tuple(labels),
    )


def wire_slab_kpoint_nonlocal_forces(system, result) -> np.ndarray:
    """Nonlocal forces of a k-point SCF of a wire or slab (Ry/bohr).

    ``forcnloc.f90`` with k-points:
    ``F_a = 4 sum_k w_k sum_n f_nk sum_(q on a) s_q Re[conj(<beta_qk|u_nk>) <grad beta_qk|u_nk>]``
    with the Bloch-phase projectors of
    :func:`parsec_python.V_ion.build_nonlocal_projectors` and their gradients
    (:func:`wire_slab_projector_gradients`).  A semilocal k-point SCF need not
    keep its orbitals, so each k-point is diagonalized once more with the potential the
    converged loop returns (``input_effective_potential``), by CHEBDAV to a
    residual of :data:`KPOINT_FORCE_TOLERANCE`; the occupations are those of
    the result, state by state.  At self-consistency this is the eigenproblem of the loop to the
    convergence of the potential.

    A hybrid functional (a result with ``exact_exchange``, from
    :func:`parsec_python.SCF.hybrid_periodic.run_periodic_hybrid_scf`) has an
    exchange operator that ``input_effective_potential`` does not hold: its
    forces read the Bloch factors of the last inner loop (``k_wavefunctions``,
    always kept there), as the spin-polarized forces do
    (:func:`wire_slab_spin_kpoint_nonlocal_forces`).
    """

    from ..Eigensolvers import solve_eigval
    from ..Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
    from ..SCF.kpoints_soc import _eigval_settings
    from ..V_ion import build_nonlocal_projectors
    from .nonlocal_force import gradient_columns_for

    atoms = system.atoms
    specifications = system.input.pseudopotentials
    lattice_vectors = system.input.periodic_cell.lattice_vectors
    k_points = np.asarray(result.k_points, dtype=np.float64)
    weights = np.asarray(result.k_weights, dtype=np.float64)
    count = k_points.shape[0]
    occupations = np.asarray(result.occupations, dtype=np.float64).reshape(count, -1)
    states = occupations.shape[1]
    kept = None
    if getattr(result, "exact_exchange", None) is not None:
        # A hybrid: the eigenproblem of the loop has the exchange operator, so its own orbitals.
        kept = getattr(result, "k_wavefunctions", None)
        if kept is None:
            raise ValueError(
                "the k-point result of a hybrid functional holds no orbitals (k_wavefunctions): "
                "its forces need the Bloch factors of the Fock loop"
            )
    else:
        potential = np.asarray(result.input_effective_potential, dtype=np.float64)
        # The nonlocal force is not variational in the orbitals: their error
        # enters to first order, so the re-solve is a tight CHEBDAV whatever the
        # solver of the loop.
        base = _eigval_settings(system, system.input.eigensolver.filter_degree)
        settings = replace(
            base,
            initial_method="chebdav",
            chebdav=replace(base.chebdav, convergence_tolerance=KPOINT_FORCE_TOLERANCE),
        )
    forces = np.zeros((len(atoms), 3), dtype=np.float64)
    for index, k_point in enumerate(k_points):
        occupied = np.flatnonzero(occupations[index] != 0.0)
        if occupied.size == 0:
            continue
        operator = build_nonlocal_projectors(
            system.grid,
            atoms,
            system.pseudopotentials,
            specifications,
            lattice_vectors,
            k_point=k_point,
        )
        if operator.projectors.shape[1] == 0:
            continue
        if kept is not None:
            vectors = np.asarray(kept[index])[:, occupied]
        else:
            hamiltonian = KPointKohnShamHamiltonian(
                system.negative_laplacian, system.gradient, potential, operator, k_point
            )
            solution = solve_eigval(
                hamiltonian.as_linear_operator(), states, settings=settings, state=None
            )
            vectors = np.asarray(solution.vectors)[:, occupied]
        gradients = wire_slab_projector_gradients(
            system.grid, atoms, system.pseudopotentials, specifications, k_point=k_point
        )
        columns = gradient_columns_for(operator.labels, gradients.labels)
        overlaps = np.asarray(operator.projectors.conj().T @ vectors)
        weight = (
            4.0
            * weights[index]
            * np.asarray(operator.signs)[:, None]
            * np.conj(overlaps)
            * occupations[index, occupied][None, :]
        )
        owners = np.fromiter((label[0] for label in operator.labels), dtype=np.int64)
        for component in range(3):
            gradient_overlaps = np.asarray(
                gradients.components[component].conj().T @ vectors
            )[columns]
            per_projector = np.einsum("qn,qn->q", weight, gradient_overlaps).real
            forces[:, component] += np.bincount(
                owners, weights=per_projector, minlength=len(atoms)
            )
    return forces


def wire_slab_spin_kpoint_nonlocal_forces(system, result) -> np.ndarray:
    """Nonlocal forces of a collinear spin-polarized k-point SCF (Ry/bohr).

    ``forcnloc.f90`` with ``nspin = 2``: every channel's states hold one
    electron, so the weight ``4/nspin`` of the unpolarized formula is 2:
    ``F_a = 2 sum_s sum_k w_k sum_n f_nks sum_(q on a) s_q
    Re[conj(<beta_qk|u_nks>) <grad beta_qk|u_nks>]``.  The orbitals are the
    final ones of the loop (``keep_wavefunctions``: the eigenvectors of the
    last input potentials, as Fortran's), the occupations those of the result.
    """

    from ..V_ion import build_nonlocal_projectors
    from .nonlocal_force import gradient_columns_for

    vectors_by_spin = getattr(result, "k_wavefunctions", None)
    if vectors_by_spin is None:
        raise ValueError(
            "the spin-polarized k-point result holds no orbitals: run the k-point "
            "SCF with keep_wavefunctions=True to compute forces"
        )
    atoms = system.atoms
    specifications = system.input.pseudopotentials
    lattice_vectors = system.input.periodic_cell.lattice_vectors
    k_points = np.asarray(result.k_points, dtype=np.float64)
    weights = np.asarray(result.k_weights, dtype=np.float64)
    count = k_points.shape[0]
    occupations = (
        np.asarray(result.occupations_up, dtype=np.float64).reshape(count, -1),
        np.asarray(result.occupations_down, dtype=np.float64).reshape(count, -1),
    )
    forces = np.zeros((len(atoms), 3), dtype=np.float64)
    for index, k_point in enumerate(k_points):
        operator = build_nonlocal_projectors(
            system.grid,
            atoms,
            system.pseudopotentials,
            specifications,
            lattice_vectors,
            k_point=k_point,
        )
        if operator.projectors.shape[1] == 0:
            continue
        gradients = wire_slab_projector_gradients(
            system.grid, atoms, system.pseudopotentials, specifications, k_point=k_point
        )
        columns = gradient_columns_for(operator.labels, gradients.labels)
        owners = np.fromiter((label[0] for label in operator.labels), dtype=np.int64)
        for spin in range(2):
            occupied = np.flatnonzero(occupations[spin][index] != 0.0)
            if occupied.size == 0:
                continue
            vectors = np.asarray(vectors_by_spin[spin][index])[:, occupied]
            overlaps = np.asarray(operator.projectors.conj().T @ vectors)
            weight = (
                2.0
                * weights[index]
                * np.asarray(operator.signs)[:, None]
                * np.conj(overlaps)
                * occupations[spin][index, occupied][None, :]
            )
            for component in range(3):
                gradient_overlaps = np.asarray(
                    gradients.components[component].conj().T @ vectors
                )[columns]
                per_projector = np.einsum("qn,qn->q", weight, gradient_overlaps).real
                forces[:, component] += np.bincount(
                    owners, weights=per_projector, minlength=len(atoms)
                )
    return forces


def wire_slab_symmetry_operations(system, atoms=None):
    """The operations ``symm_forces.f90`` averages the forces of a wire or slab over.

    Those of ``system.input.space_group`` (the group of
    :func:`parsec_python.Grid.space_group.find_wire_slab_space_group` that
    the input reader keeps for a wire or slab), as
    :class:`parsec_python.Forces.symmetry.SpaceGroupAtomOperation` on
    ``atoms`` (default ``system.atoms``): ``symop`` maps ``r`` to
    ``U (r - A t)`` with ``A`` = ``alatt``; the image is the atom of the same
    species at that point modulo the periodic lattice vectors (``tol`` 1e-6
    bohr).  ``None`` where there is no group or it has the identity alone
    (Fortran: no symmetrization); an operation that maps no atom onto an atom
    stops as ``symm_forces`` does (error 411).
    """

    from ..Grid.space_group import confined_symmetry_lattice
    from .symmetry import POSITION_TOLERANCE, SpaceGroupAtomOperation

    group = getattr(system.input, "space_group", None)
    if group is None or group.order <= 1:
        return None
    atoms = system.atoms if atoms is None else tuple(atoms)
    periodicity = int(system.grid.periodicity)
    lattice = np.asarray(system.input.periodic_cell.lattice_vectors, dtype=np.float64)
    alatt = confined_symmetry_lattice(lattice, periodicity, system.grid.confinement_radius)
    inverse = np.linalg.inv(lattice)
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    symbols = [atom.symbol for atom in atoms]
    operations = []
    for rotation, translation in zip(group.cartesian_rotations, group.translations):
        matrix = np.asarray(rotation, dtype=np.float64)
        # Cartesian A t (the representation of SpaceGroupAtomOperation).
        cartesian = np.asarray(translation, dtype=np.float64) @ alatt
        shift = -(matrix @ cartesian)
        mapped = positions @ matrix.T + shift
        image = np.full(len(atoms), -1, dtype=np.int64)
        for index, point in enumerate(mapped):
            difference = (point[None, :] - positions) @ inverse
            difference[:, :periodicity] -= np.round(difference[:, :periodicity])
            distance = np.linalg.norm(difference @ lattice, axis=1)
            candidates = [
                other
                for other in np.flatnonzero(distance <= POSITION_TOLERANCE)
                if symbols[other] == symbols[index]
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "symmetrizing forces: unable to find the equivalent of atom "
                    f"{index + 1} (PARSEC symm_forces.f90, error 411)"
                )
            image[index] = candidates[0]
        operations.append(
            SpaceGroupAtomOperation(
                matrix=matrix,
                image=image,
                translation=cartesian,
                lattice_vectors=lattice.copy(),
                periodicity=periodicity,
            )
        )
    return tuple(operations)


def _as_wire_slab_result(result: ForceResult) -> WireSlabForceResult:
    # ForceResult.periodic: forcnloc.f90 with is_pbc, which a wire or slab is.
    values = {item.name: getattr(result, item.name) for item in fields(result)}
    values["periodic"] = True
    return WireSlabForceResult(**values)


def _spin_kpoint_force_fields(result):
    """``(density, V_out - V_in, V_xc, V_H)`` of a spin-polarized k-point result.

    ``forcloc`` reads the total density and the averages of the two
    channels' potential difference and exchange-correlation potential.
    """

    inputs = getattr(result, "scf_input_potentials", None)
    xc_channels = getattr(result, "xc_potentials", None)
    hartree_potential = getattr(result, "scf_hartree_potential", None)
    if inputs is None or xc_channels is None or hartree_potential is None:
        raise ValueError(
            "the spin-polarized k-point result holds no force fields: run the "
            "k-point SCF with keep_wavefunctions=True to compute forces"
        )
    density = np.asarray(result.density_up) + np.asarray(result.density_down)
    potentials = result.scf_potentials
    residual = 0.5 * (
        (np.asarray(potentials["potential_up"]) - np.asarray(inputs[0]))
        + (np.asarray(potentials["potential_down"]) - np.asarray(inputs[1]))
    )
    xc = 0.5 * (np.asarray(xc_channels[0]) + np.asarray(xc_channels[1]))
    from .periodic import meta_gga_core_potential

    return density, residual, meta_gga_core_potential(result, xc), np.asarray(hartree_potential)


def compute_wire_slab_forces(system, result, *, symmetrize: bool = True, operations=None) -> WireSlabForceResult:
    """Forces of a finished SCF of a wire or slab.

    ``system`` is a :class:`parsec_python.SCF.wire_slab.WireSlabPreparedSystem`
    and ``result`` its SCF result: a Gamma-point result with full-grid
    orbitals (also collinear spin through
    :func:`parsec_python.SCF.run_spin_scf`), an unpolarized k-point result
    (:func:`wire_slab_kpoint_nonlocal_forces`; a hybrid's reads its kept
    orbitals) or a collinear spin k-point
    result with its orbitals (:func:`wire_slab_spin_kpoint_nonlocal_forces`).
    As ``forcnloc.f90``: with ``symmetrize`` the forces are averaged over
    ``operations`` (default :func:`wire_slab_symmetry_operations`: the space
    group the input reader keeps; none for a slab, whose symmetry Fortran
    switches off), which also completes the forces of a k-point set reduced
    to the irreducible wedge.
    """

    started = perf_counter()
    atoms = system.atoms
    specifications = system.input.pseudopotentials
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    kpoints = getattr(result, "k_points", None) is not None
    spin_kpoints = hasattr(result, "density_up")
    if spin_kpoints and not kpoints:
        raise ValueError(
            "the forces of a spin-polarized wire or slab at Gamma are computed from "
            "the result of run_spin_scf (run_scf of a spin-polarized system)"
        )
    if spin_kpoints:
        density, residual, xc, hartree_potential = _spin_kpoint_force_fields(result)
    else:
        # As compute_forces: the total density, and the channel averages of the
        # potential difference and of V_xc for a spin-polarized result.
        density = np.asarray(result.density)
        from .periodic import meta_gga_core_potential

        xc = meta_gga_core_potential(result, np.asarray(result.xc_potential))
        residual = np.asarray(result.output_effective_potential) - np.asarray(
            result.input_effective_potential
        )
        hartree_potential = np.asarray(result.hartree_potential)
    hartree_gradient = np.vstack([matrix @ hartree_potential for matrix in system.gradient])
    local = wire_slab_local_forces(
        system.grid,
        atoms,
        system.pseudopotentials,
        specifications,
        density,
        residual,
        xc,
        hartree_gradient,
    )
    if spin_kpoints:
        separable = wire_slab_spin_kpoint_nonlocal_forces(system, result)
    elif kpoints:
        separable = wire_slab_kpoint_nonlocal_forces(system, result)
    else:
        gradients = wire_slab_projector_gradients(
            system.grid, atoms, system.pseudopotentials, specifications
        )
        separable = nonlocal_forces(
            system.nonlocal_operator,
            gradients,
            result.wavefunctions,
            (
                result.occupations
                if getattr(result, "spin", None) is None
                else 0.5 * np.asarray(result.occupations)
            ),
            len(atoms),
        )
    if not symmetrize:
        operations = None
    elif operations is None:
        operations = wire_slab_symmetry_operations(system)
    if operations is not None and len(operations) <= 1:
        operations = None
    from .total import correction_forces

    forces = assemble_forces(
        positions,
        np.asarray(system.ion_ion_forces, dtype=np.float64),
        local,
        separable,
        operations,
        seconds=perf_counter() - started,
        # LDA+U (forcnloc_u; J = 0), empty without it.
        **correction_forces(system, result),
    )
    return _as_wire_slab_result(forces)


__all__ = [
    "WireSlabForceResult",
    "WireSlabLocalForceTerms",
    "compute_wire_slab_forces",
    "wire_slab_kpoint_nonlocal_forces",
    "wire_slab_spin_kpoint_nonlocal_forces",
    "wire_slab_symmetry_operations",
    "wire_slab_local_forces",
    "wire_slab_projector_gradients",
]
