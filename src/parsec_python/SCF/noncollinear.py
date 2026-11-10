"""Self-consistent non-collinear magnetism (optionally with spin-orbit coupling).

The spinor wavefunctions ``psi = (psi_up, psi_down)`` give a charge density
``n = |psi_up|^2 + |psi_down|^2`` and a magnetization vector

``m = psi^dagger sigma psi = (2 Re(psi_up^* psi_down),
                              2 Im(psi_up^* psi_down),
                              |psi_up|^2 - |psi_down|^2)``

(summed over occupied states).  The exchange-correlation potential is
``V_avg * 1 + B_xc . sigma`` with the local-rotation form of
:mod:`parsec_python.V_xc.noncollinear`, so the Hamiltonian acting on each
spinor is

``H = [T + V_ion + V_H + V_avg + V_NL] * 1 + B_xc . sigma (+ L.S terms)``.

``V_avg`` and the three components of ``B_xc`` are mixed together as one
vector.  The direction of ``m`` is free to vary in space and to rotate during
the SCF; the initial moments are seeded per atom from
``Atom.initial_moment`` (Fortran's ``Initial_NCL_Moment``) with magnitude set
by the species' ``initial_spin_polarization``.  Without spin-orbit coupling
the energy is invariant under a global rotation of all moments, which the
tests use as an independent check.

:func:`run_self_consistent_noncollinear` handles an isolated system;
:func:`run_self_consistent_noncollinear_kpoints` a periodic cell sampled on a
k-point set (magnetism breaks time-reversal symmetry, so ``k`` and ``-k``
must both appear: Fortran's reduced grid uses no time reversal here,
``irrbz.f90``); unequal weights weight every level by its k-point
(:mod:`~parsec_python.SCF.kpoint_weights`).  The magnetization is a vector
field, which ``Symmetrize_Charge_Density`` (a scalar average) does not
treat: the input reader refuses that combination with a non-trivial group.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np

from ..Eigensolvers import EigvalState, solve_eigval
from ..Energy import total_energy_noncollinear
from ..Hamiltonian.spinor_operator import SpinorKohnShamHamiltonian
from ..Mixer import make_mixer, potential_residual_metrics
from ..V_ion import superpose_atomic_density
from ..models import SelfConsistentSOCIteration, SelfConsistentSOCResult
from .kpoint_solver import KPointJob, kept_vectors, make_kpoint_solver, split_pooled
from .kpoint_weights import KPointWeights
from .orbital_density import spinor_sums
from .single_point import first_diagonalization_tolerance, next_diagonalization_tolerance
from .stage_timer import StageTimer, solver_residual_max, solver_timings
from .kpoints_soc import (
    SpinorForceExtras,
    _eigval_settings,
    _k_point_operators,
    _number_of_states,
    _result_weights,
    _spinor_hamiltonian,
    _state_sz,
    _validated_k_points,
    export_subspaces,
    import_subspaces,
)


def spinor_density_and_magnetization(
    spinors: np.ndarray,
    occupations: np.ndarray,
    volume_element: float,
    weight: float = 1.0,
    xp=np,
) -> tuple[np.ndarray, np.ndarray]:
    """``(n, m)`` from stacked spinors ``(2*n_grid, n_states)``.

    ``n`` has shape ``(n_grid,)`` and ``m`` has shape ``(3, n_grid)``;
    ``weight`` is the k-point weight of these states.  ``xp`` is the array
    namespace the spinors live in (a device block gives device arrays); a block that holds only some
    rows of the grid gives those rows.
    """

    occupations = np.asarray(occupations, dtype=float)
    if spinors.ndim != 2 or spinors.shape[0] % 2:
        raise ValueError("spinors must have shape (2*n_grid, n_states)")
    if occupations.shape != (spinors.shape[1],):
        raise ValueError("occupation count does not match the spinor columns")
    if volume_element <= 0:
        raise ValueError("volume_element must be positive")
    scale = weight / volume_element
    up_sum, down_sum, cross_sum = spinor_sums(spinors, occupations, xp)
    rows = up_sum.shape[0]
    density = scale * (up_sum + down_sum)
    magnetization = xp.empty((3, rows))
    magnetization[0] = scale * (2.0 * cross_sum.real)
    magnetization[1] = scale * (2.0 * cross_sum.imag)
    magnetization[2] = scale * (up_sum - down_sum)
    return density, magnetization


@dataclass(frozen=True)
class NoncollinearMoments:
    """The magnetization diagnostics of ``bxc.F90`` (``bxc`` and ``nclmom``).

    ``absolute`` is ``Integral{|m|} = h^3 sum_i |m_i|``, ``atomic`` the
    ``(atoms, 3)`` sums ``h^3 sum m_i`` over the grid points within the
    ``Atomic_Radius`` of each atom (the Wigner-Seitz lists ``wsindw`` of
    ``nonloc.F90``, with the periodic images of a point in the neighbouring
    cells) and ``total`` the ``h^3 sum_i m_i`` of all points, all in Bohr
    magnetons.
    """

    absolute: float
    atomic: np.ndarray
    total: np.ndarray


def atomic_sphere_masks(
    coordinates: np.ndarray,
    atoms,
    radii,
    lattice_vectors: np.ndarray | None = None,
) -> np.ndarray:
    """``(atoms, points)``: the grid points within ``radii[a]`` of atom ``a``.

    A point is inside where it or one of its images in the 26 neighbouring
    cells (rows of ``lattice_vectors``, for a periodic system) is at most
    the radius from the nucleus (``nonloc.F90``: ``rr <= p_pot%rws``).
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    shifts = [np.zeros(3)]
    if lattice_vectors is not None:
        vectors = np.asarray(lattice_vectors, dtype=np.float64)
        shifts = [
            np.asarray(cell, dtype=np.float64) @ vectors
            for cell in product((-1, 0, 1), repeat=3)
        ]
    masks = np.zeros((len(atoms), coordinates.shape[0]), dtype=bool)
    for index, (atom, radius) in enumerate(zip(atoms, radii)):
        for shift in shifts:
            distance = np.linalg.norm(
                coordinates + shift - np.asarray(atom.position), axis=1
            )
            masks[index] |= distance <= float(radius)
    return masks


def noncollinear_moments(
    magnetization: np.ndarray, masks: np.ndarray, volume_element: float
) -> NoncollinearMoments:
    """The moments of a magnetization density ``(3, points)`` in mu_B/bohr^3."""

    magnetization = np.asarray(magnetization, dtype=np.float64)
    weight = float(volume_element)
    return NoncollinearMoments(
        absolute=weight * float(np.sum(np.linalg.norm(magnetization, axis=0))),
        atomic=weight * (np.asarray(masks, dtype=np.float64) @ magnetization.T),
        total=weight * np.sum(magnetization, axis=1),
    )


def _initial_magnetization(system) -> np.ndarray:
    """Per-atom seed ``m(r) = sum_a p_a * d_a * rho_a(r)`` (shape ``(3, n_grid)``).

    ``rho_a`` is atom ``a``'s own superposed atomic density, ``d_a`` its unit
    initial-moment direction and ``p_a`` the species' ``initial_spin_polarization``;
    the sum is scaled by the same factor that normalized the total initial
    density, so ``|m| <= n`` holds wherever ``p_a <= 1``.
    """

    periodic_cell = getattr(system.input, "periodic_cell", None)
    lattice_vectors = None if periodic_cell is None else periodic_cell.lattice_vectors
    specifications = system.input.pseudopotentials
    per_atom = []
    for atom in system.atoms:
        density_a = superpose_atomic_density(
            system.grid,
            [atom],
            system.pseudopotentials,
            specifications,
            lattice_vectors,
        )
        per_atom.append(density_a)
    unnormalized_charge = float(sum(np.sum(rho) for rho in per_atom))
    scale = (
        float(np.sum(system.initial_density)) / unnormalized_charge
        if unnormalized_charge > 0.0
        else 0.0
    )
    magnetization = np.zeros((3, system.grid.size))
    for atom, density_a in zip(system.atoms, per_atom):
        if atom.initial_moment is None:
            continue
        polarization = specifications[atom.symbol].initial_spin_polarization
        magnetization += (
            scale * polarization * np.asarray(atom.initial_moment)[:, None] * density_a
        )
    return magnetization


def symmetrize_noncollinear(density, magnetization, group, volume_element):
    """``Symmetrize_Charge_Density`` of a non-collinear density (PARSEC's rule).

    ``newrho.F90`` hands ``symm_scalar`` the total density and the densities
    of the local frame ``(rho +- |m|)/2`` (``isp = 1, 2, 3``); the direction
    of the magnetization (``spin3d`` of ``bxc.F90``) is not averaged.  This is
    the same: ``rho`` and ``|m|`` are averaged as scalars and ``m`` keeps its
    direction, ``m <- <|m|> m/|m|`` (zero where ``|m| = 0``).  Without
    spin-orbit coupling a spatial operation does not act on the spin, so the
    scalar average is the symmetry of the problem; with it the operations act
    on ``m`` as on an axial vector and the symmetry of a magnetic state is a
    subgroup, which neither this nor a full-group axial average respects
    (PARSEC's choice is kept).  Returns the density, the magnetization and
    the three reports.
    """

    from ..Grid.point_group import symmetrize_density

    magnetization = np.asarray(magnetization, dtype=np.float64)
    length = np.sqrt(np.sum(magnetization * magnetization, axis=0))
    new_density, total_report = symmetrize_density(density, group, volume_element)
    _, up_report = symmetrize_density(0.5 * (density + length), group, volume_element)
    _, down_report = symmetrize_density(0.5 * (density - length), group, volume_element)
    new_length, _ = symmetrize_density(length, group, volume_element)
    scale = np.divide(new_length, length, out=np.zeros_like(length), where=length > 0.0)
    return new_density, magnetization * scale[None, :], (total_report, up_report, down_report)


def _run_noncollinear(
    system,
    hamiltonian_for_k: Callable[[int, np.ndarray, np.ndarray], object],
    n_kpoints: int,
    weighting: KPointWeights,
    callback: Callable[[SelfConsistentSOCIteration], None] | None,
    k_points: np.ndarray | None = None,
    keep_wavefunctions: bool = False,
    iteration_state_callback: Callable[[object], None] | None = None,
    initial_potential: tuple[np.ndarray, np.ndarray] | None = None,
    extras: SpinorForceExtras | None = None,
    warm_start=None,
) -> SelfConsistentSOCResult:
    settings = system.input.scf
    n_grid = system.grid.size
    volume_element = system.grid.volume_element
    number_of_states = _number_of_states(system)
    ionic_potential = system.ionic_potential
    density_symmetry = getattr(system, "density_symmetry", None)

    timer = StageTimer()
    density = system.initial_density
    if warm_start is None:
        with timer.stage("initial_magnetization"):
            magnetization = _initial_magnetization(system)
    if initial_potential is None and warm_start is None:
        with timer.stage("setup_potentials"):
            hartree = system.solve_hartree(density, initial_potential=-ionic_potential)
            hartree_potential = hartree.potential
            xc = system.evaluate_xc_noncollinear(density, magnetization)
        input_potential = ionic_potential + hartree_potential + xc.potential
        input_field = xc.field
    else:
        if warm_start is not None:
            # Andrew Grieder's ionic loops: the structure before
            # the exchange-correlation field of the previous structure carries the moments to the moved atoms
            hartree_potential = np.array(warm_start.hartree_potential, dtype=np.float64)
            input_potential, input_field = (np.array(part, dtype=np.float64) for part in warm_start.input_potential)
        else:
            # The warm start of an ionic step (parsec.F90 after domove: vnew with
            # the ionic part replaced, the magnetization direction kept): the
            # scalar potential and the exchange-correlation field of the first
            # eigenproblem; -V_ion starts the first Poisson solve, as a restart.
            hartree_potential = -np.asarray(ionic_potential, dtype=np.float64)
            input_potential = np.array(initial_potential[0], dtype=np.float64)
            input_field = np.array(initial_potential[1], dtype=np.float64).reshape(3, n_grid)
            if input_potential.shape != np.shape(ionic_potential):
                raise ValueError("initial_potential must hold one value per grid point")

    def accumulate(task, vectors, occupations, xp=np):
        density_k, magnetization_k = spinor_density_and_magnetization(
            vectors, occupations, volume_element, weighting.of_k(task), xp
        )
        return {"density": density_k, "magnetization": magnetization_k}

    solver = make_kpoint_solver(
        KPointJob(
            n_kpoints,
            number_of_states,
            lambda degree, tolerance=None: _eigval_settings(system, degree, tolerance),
            lambda task, potentials: hamiltonian_for_k(
                task, potentials["V"], potentials["field"]
            ),
            accumulate,
            extras=extras if extras is not None else (lambda task, vectors, xp=np: _state_sz([vectors], xp)),
        ),
        problem_size=system.grid.size,
    )
    if extras is not None and extras.compute_forces and type(solver).__name__ == "MPIKPointSolver":
        raise NotImplementedError(
            "the nonlocal force of the row-distributed MPI k-point solver is not implemented: its extras are "
            "sums over row blocks, which does not apply to a product of overlaps"
        )
    mixer = make_mixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)
    # Dynamic_Diag_Tol: 0.1 for the first iteration (parsec.F90:1106-1111).
    dynamic_tolerance = system.input.eigensolver.dynamic_tolerance
    diagonalization_tolerance = first_diagonalization_tolerance(system.input.eigensolver)

    pooled_eigenvalues = np.empty(0)
    pooled_occupations = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False
    specifications = system.input.pseudopotentials
    periodic_cell = getattr(system.input, "periodic_cell", None)
    sphere_masks = atomic_sphere_masks(
        system.grid.coordinates,
        system.atoms,
        [getattr(specifications[atom.symbol], "atomic_radius", 0.0) for atom in system.atoms],
        None if periodic_cell is None else periodic_cell.lattice_vectors,
    )
    with solver:
        import_subspaces(solver, warm_start)
        timer.lap()  # what came before the loop (potentials, solver start-up) is not part of iteration 1
        for iteration in range(1, settings.max_iterations + 1):
            used_potential = input_potential  # what enters this eigenproblem (the mixed one replaces it below)

            with timer.stage("diagonalization"):
                eigenvalues_by_k = solver.solve(
                    {"V": input_potential, "field": input_field},
                    filter_degree,
                    diagonalization_tolerance if dynamic_tolerance else None,
                )
            residual_max = solver_residual_max(solver)

            with timer.stage("density"):
                stacked = np.concatenate(eigenvalues_by_k)
                state_weights = (
                    None
                    if weighting.uniform
                    else weighting.state_weights(eigenvalues_by_k, range(n_kpoints))
                )
                stacked_occupations, fermi_level = weighting.occupy(
                    stacked,
                    system.electron_count,
                    settings.fermi_temperature_kelvin,
                    1.0,
                    state_weights,
                )
                pooled_eigenvalues = stacked
                pooled_occupations = stacked_occupations

                parts = solver.accumulate(split_pooled(stacked_occupations, eigenvalues_by_k))
                density = parts["density"]
                magnetization = parts["magnetization"]
                symmetrization = ()
                if density_symmetry is not None:
                    density, magnetization, symmetrization = symmetrize_noncollinear(
                        density, magnetization, density_symmetry, volume_element
                    )

            with timer.stage("hartree"):
                hartree = system.solve_hartree(density, initial_potential=hartree_potential)
                hartree_potential = hartree.potential
            with timer.stage("xc"):
                xc = system.evaluate_xc_noncollinear(density, magnetization)
                output_potential = ionic_potential + hartree_potential + xc.potential
                output_field = xc.field

            input_combined = np.concatenate([input_potential, input_field.reshape(-1)])
            output_combined = np.concatenate([output_potential, output_field.reshape(-1)])
            with timer.stage("mixing_energy"):
                metrics = potential_residual_metrics(
                    input_combined,
                    output_combined,
                    np.tile(density, 4),
                    volume_element,
                    system.electron_count,
                )
                energy_occupations, band_energy_weight = weighting.band_energy_terms(
                    pooled_occupations, state_weights
                )
                energies = total_energy_noncollinear(
                    pooled_eigenvalues,
                    energy_occupations,
                    density,
                    magnetization,
                    input_potential,
                    input_field,
                    ionic_potential,
                    hartree_potential,
                    xc.potential,
                    xc.field,
                    xc.total_energy,
                    system.ion_ion_energy,
                    volume_element,
                    alpha_z_energy=system.alpha_z_energy,
                    band_energy_weight=band_energy_weight,
                )

                mixed = mixer.mix(input_combined, output_combined, iteration=iteration)
            lap = timer.lap()
            selected_residual = (
                metrics.plain if settings.use_plain_residual else metrics.weighted
            )
            if dynamic_tolerance:
                # parsec.F90:1302-1311 with this loop's SRE.
                diagonalization_tolerance = next_diagonalization_tolerance(
                    system.input.eigensolver, diagonalization_tolerance, (metrics.weighted,)
                )
            if callback is not None:
                first_k = eigenvalues_by_k[0].size
                callback(
                    SelfConsistentSOCIteration(
                        iteration=iteration,
                        weighted_residual=metrics.weighted,
                        plain_residual=metrics.plain,
                        energies=energies,
                        eigenvalues=tuple(float(v) for v in eigenvalues_by_k[0]),
                        occupations=tuple(float(v) for v in stacked_occupations[:first_k]),
                        fermi_level=float(fermi_level),
                        eigen_residual_max=residual_max,
                        diagonalization_seconds=lap.get("diagonalization", 0.0),
                        total_seconds=sum(lap.values()),
                        magnetic_moments=noncollinear_moments(
                            magnetization, sphere_masks, volume_element
                        ),
                        density_symmetrization=symmetrization,
                        **(
                            {"next_eigensolver_tolerance": diagonalization_tolerance}
                            if dynamic_tolerance
                            else {}
                        ),
                    )
                )
            if iteration_state_callback is not None:
                # Save_Intermediate_*: the local-frame fields of the final
                # parsec.dat after mixing; the spinors of an isolated loop.
                from .single_point import SpinorIterationFields

                iteration_state_callback(
                    SpinorIterationFields(
                        iteration=iteration,
                        eigenvalues=pooled_eigenvalues,
                        occupations=pooled_occupations,
                        # (the MPI solver gathers no spinors: an empty block, which
                        # parsec.dat then writes without wavefunction records)
                        spinors=None if k_points is not None else solver.vectors(0),
                        parsec_dat_fields=_noncollinear_parsec_dat_fields(
                            system,
                            mixed[:n_grid],
                            mixed[n_grid:].reshape(3, n_grid),
                            density,
                            magnetization,
                        ),
                        k_points=k_points,
                        k_weights=(
                            None if k_points is None else _result_weights(weighting, n_kpoints)
                        ),
                    )
                )
            last_input_potential = input_potential
            input_potential = mixed[:n_grid]
            input_field = mixed[n_grid:].reshape(3, n_grid)
            if (
                iteration > settings.degree_decrease_after
                and metrics.weighted < settings.approach_criterion
                and filter_degree > minimum_filter_degree
            ):
                filter_degree -= 1
            if selected_residual < settings.convergence_criterion:
                converged = True
                break

        moment_vector = volume_element * np.sum(magnetization, axis=1)
        from .single_point import dispersion_after_convergence, dispersion_wanted
        dispersion = None
        if dispersion_wanted(settings, converged):
            # efvdw.f90 after a converged loop: the TS correction of the density.
            dispersion, energies = dispersion_after_convergence(system, density, energies)
        subspaces = export_subspaces(solver)
        nonlocal_forces = None
        state_moments = None
        if k_points is not None:
            if extras is not None and extras.compute_forces:
                # the eigenvectors and occupations of the last step belong together
                extras.set_occupations(split_pooled(pooled_occupations, eigenvalues_by_k))
            if extras is not None:
                state_moments, nonlocal_forces = extras.collect(solver)
            else:
                state_moments = np.concatenate(solver.extras())
        return SelfConsistentSOCResult(
            converged=converged,
            iterations=iteration,
            atoms=system.atoms,
            electron_count=system.electron_count,
            eigenvalues=pooled_eigenvalues,
            occupations=pooled_occupations,
            spinors=solver.vectors(0),
            fermi_level=fermi_level,
            density=density,
            energies=energies,
            magnetic_moment=float(np.linalg.norm(moment_vector)),
            magnetic_moment_vector=moment_vector,
            magnetization=magnetization,
            k_points=k_points,
            k_weights=None if k_points is None else _result_weights(weighting, n_kpoints),
            state_moments=state_moments,
            timings=solver_timings(timer, solver),
            scf_potentials={
                "effective_potential": output_potential,
                "xc_field": output_field,
            },
            # The scalar parts the forces read (parsec_python.Forces.spin_orbit).
            scf_input_potential=last_input_potential,
            scf_output_potential=output_potential,
            xc_potential=xc.potential,
            k_spinors=(
                tuple(
                    kept_vectors(solver, task, "the non-collinear forces")
                    for task in range(n_kpoints)
                )
                if keep_wavefunctions and k_points is not None
                else None
            ),
            dispersion=dispersion,
            parsec_dat_fields=_noncollinear_parsec_dat_fields(
                system, input_potential, input_field, density, magnetization
            ),
            # pot%vnew and the field: where the next structure of an ionic
            # loop starts.
            next_potential=(input_potential, input_field),
            input_effective_potential=used_potential,
            output_effective_potential=output_potential,
            nonlocal_forces=nonlocal_forces,
            next_effective_potential=(input_potential, input_field),
            hartree_potential=hartree_potential,
            eigensolver_state=subspaces,
        )


def _noncollinear_parsec_dat_fields(system, potential, field, density, magnetization) -> dict:
    """``vnew(:,1:2)`` and ``rho(:,2:3)`` of a non-collinear run (``wfnsave.F90``).

    PARSEC stores the local-frame quantities: the potentials of the spin
    along and against the local magnetization, ``V +- |B|``, and the
    densities ``(rho +- |m|)/2`` (``newrho.F90``); the direction of ``m``
    (``spin3d``) is not written.  ``stype(2)`` is 4 with spin-orbit species,
    3 without.
    """

    potential = np.asarray(potential, dtype=np.float64)
    strength = np.sqrt(np.sum(np.asarray(field, dtype=np.float64) ** 2, axis=0))
    length = np.sqrt(np.sum(np.asarray(magnetization, dtype=np.float64) ** 2, axis=0))
    spin_orbit = any(
        getattr(specification, "spin_orbit", False)
        for specification in system.input.pseudopotentials.values()
    )
    return {
        "kind": 4 if spin_orbit else 3,
        "potentials": (potential + strength, potential - strength),
        "densities": (0.5 * (density + length), 0.5 * (density - length)),
    }


def run_self_consistent_noncollinear(
    system,
    soc_projectors: tuple = (),
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
    iteration_state_callback: Callable[[object], None] | None = None,
    initial_potential: tuple[np.ndarray, np.ndarray] | None = None,
    warm_start=None,
) -> SelfConsistentSOCResult:
    """Non-collinear SCF for an isolated system.

    ``warm_start`` (:class:`~parsec_python.SCF.single_point.SCFWarmStart`, whose ``input_potential`` is the pair
    ``(V, B_xc)``) continues from the loop of the previous structure of a relaxation or dynamics run.

    ``soc_projectors`` is
    :func:`~parsec_python.Eigensolvers.perturbative_soc.build_spin_orbit_projectors`'s
    output; leave it empty for non-collinear magnetism without spin-orbit.
    ``initial_potential`` is ``(V, B)``, the scalar potential and the
    exchange-correlation field (shape ``(3, n_grid)``) of the first
    eigenproblem in place of those of the initial density and moments (the
    warm start of an ionic step, ``result.next_potential`` of the structure
    before with its ionic part replaced).
    ``iteration_state_callback`` receives a
    :class:`parsec_python.SCF.single_point.SpinorIterationFields` after every
    iteration (``Save_Intermediate_*``).
    """
    from .single_point import refuse_hubbard_and_dispersion

    refuse_hubbard_and_dispersion(
        system, "run_self_consistent_noncollinear", dispersion=False
    )

    if not system.input.scf.noncollinear:
        raise ValueError(
            "run_self_consistent_noncollinear requires SCFSettings.noncollinear=True"
        )
    soc_projectors = tuple(soc_projectors)

    def hamiltonian_for_k(_k_index, potential, field):
        scalar = system.hamiltonian(potential)
        return SpinorKohnShamHamiltonian(
            scalar.negative_laplacian,
            scalar.effective_potential,
            scalar.nonlocal_operator,
            soc_projectors,
            xc_field=field,
        )

    return _run_noncollinear(
        system,
        hamiltonian_for_k,
        1,
        KPointWeights(weights=np.ones(1), uniform=True, weight=1.0),
        callback,
        iteration_state_callback=iteration_state_callback,
        initial_potential=initial_potential,
        warm_start=warm_start,
    )


def run_self_consistent_noncollinear_kpoints(
    system,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
    keep_wavefunctions: bool = False,
    iteration_state_callback: Callable[[object], None] | None = None,
    initial_potential: tuple[np.ndarray, np.ndarray] | None = None,
    compute_forces: bool = False,
    warm_start=None,
) -> SelfConsistentSOCResult:
    """Non-collinear SCF for a periodic cell on an unreduced uniform k grid.

    ``warm_start`` continues from the loop of the previous structure, as for the isolated driver.

    Spin-orbit projectors are built for every k-point from the species with
    ``SO_PSP`` (none gives plain non-collinear magnetism).
    ``keep_wavefunctions`` keeps the spinors of every k-point
    (``k_spinors``, for the forces); ``initial_potential`` is ``(V, B)`` as
    in :func:`run_self_consistent_noncollinear`; ``iteration_state_callback``
    receives a :class:`parsec_python.SCF.single_point.SpinorIterationFields`
    after every iteration (``Save_Intermediate_*``).
    ``compute_forces``
    also returns the nonlocal force (``result.nonlocal_forces``), accumulated from the orbitals of every k-point by
    a last pass after the SCF loop (see :class:`~parsec_python.SCF.kpoints_soc.SpinorForceExtras`); ``warm_start``
    continues from the structure before (Andrew Grieder's ionic loops).
    """
    from .single_point import refuse_hubbard_and_dispersion

    refuse_hubbard_and_dispersion(
        system, "run_self_consistent_noncollinear_kpoints", dispersion=False
    )

    if not system.input.scf.noncollinear:
        raise ValueError(
            "run_self_consistent_noncollinear_kpoints requires "
            "SCFSettings.noncollinear=True"
        )
    k_points, weighting = _validated_k_points(
        k_points, weights, "run_self_consistent_noncollinear_kpoints"
    )
    operator_cache: dict[int, tuple] = {}

    def operators_for(k_index):
        if k_index not in operator_cache:
            operator_cache[k_index] = _k_point_operators(
                system, k_points[k_index : k_index + 1]
            )[0]
        return operator_cache[k_index]

    def hamiltonian_for_k(k_index, potential, field):
        nonlocal_operator_k, soc_projectors_k = operators_for(k_index)
        return _spinor_hamiltonian(
            system,
            k_points[k_index],
            nonlocal_operator_k,
            soc_projectors_k,
            potential,
            xc_field=field,
        )

    if compute_forces and not weighting.uniform:
        raise NotImplementedError(
            "the nonlocal force of a non-collinear k-point run accumulated in the SCF step (compute_forces, "
            "Andrew Grieder's ionic loops) needs equal k-point weights"
        )
    extras = SpinorForceExtras(
        system, k_points, weighting.weight if weighting.uniform else 1.0, operators_for, compute_forces
    )
    return _run_noncollinear(
        system,
        hamiltonian_for_k,
        k_points.shape[0],
        weighting,
        callback,
        k_points,
        keep_wavefunctions=keep_wavefunctions,
        iteration_state_callback=iteration_state_callback,
        initial_potential=initial_potential,
        extras=extras,
        warm_start=warm_start,
    )


__all__ = [
    "NoncollinearMoments",
    "atomic_sphere_masks",
    "noncollinear_moments",
    "run_self_consistent_noncollinear",
    "run_self_consistent_noncollinear_kpoints",
    "spinor_density_and_magnetization",
    "symmetrize_noncollinear",
]
