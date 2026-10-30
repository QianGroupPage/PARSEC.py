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
:func:`run_self_consistent_noncollinear_kpoints` a periodic cell sampled on the
full, unreduced Monkhorst-Pack grid (magnetism breaks time-reversal symmetry,
so ``k`` and ``-k`` must both appear).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..Eigensolvers import EigvalState, solve_eigval
from ..Energy import total_energy_noncollinear
from ..Hamiltonian.spinor_operator import SpinorKohnShamHamiltonian
from ..Mixer import AndersonMixer, potential_residual_metrics
from ..Occupations import fermi_occupations
from ..V_ion import superpose_atomic_density
from ..models import SelfConsistentSOCIteration, SelfConsistentSOCResult
from .kpoint_solver import KPointJob, make_kpoint_solver, split_pooled
from .orbital_density import spinor_sums
from .stage_timer import StageTimer, solver_residual_max, solver_timings
from .kpoints_soc import (
    SpinorForceExtras,
    _eigval_settings,
    _k_point_operators,
    _number_of_states,
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


def _run_noncollinear(
    system,
    hamiltonian_for_k: Callable[[int, np.ndarray, np.ndarray], object],
    n_kpoints: int,
    weight: float,
    callback: Callable[[SelfConsistentSOCIteration], None] | None,
    k_points: np.ndarray | None = None,
    extras: SpinorForceExtras | None = None,
    warm_start=None,
) -> SelfConsistentSOCResult:
    settings = system.input.scf
    n_grid = system.grid.size
    volume_element = system.grid.volume_element
    number_of_states = _number_of_states(system)
    ionic_potential = system.ionic_potential

    timer = StageTimer()
    density = system.initial_density
    if warm_start is None:
        with timer.stage("initial_magnetization"):
            magnetization = _initial_magnetization(system)
        with timer.stage("setup_potentials"):
            hartree = system.solve_hartree(density, initial_potential=-ionic_potential)
            hartree_potential = hartree.potential
            xc = system.evaluate_xc_noncollinear(density, magnetization)
        input_potential = ionic_potential + hartree_potential + xc.potential
        input_field = xc.field
    else:
        # the exchange-correlation field of the previous structure carries the moments to the moved atoms
        hartree_potential = np.array(warm_start.hartree_potential, dtype=np.float64)
        input_potential, input_field = (np.array(part, dtype=np.float64) for part in warm_start.input_potential)

    def accumulate(task, vectors, occupations, xp=np):
        density_k, magnetization_k = spinor_density_and_magnetization(
            vectors, occupations, volume_element, weight, xp
        )
        return {"density": density_k, "magnetization": magnetization_k}

    solver = make_kpoint_solver(
        KPointJob(
            n_kpoints,
            number_of_states,
            lambda degree: _eigval_settings(system, degree),
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
    mixer = AndersonMixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)

    pooled_eigenvalues = np.empty(0)
    pooled_occupations = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False
    with solver:
        import_subspaces(solver, warm_start)
        timer.lap()  # what came before the loop (potentials, solver start-up) is not part of iteration 1
        for iteration in range(1, settings.max_iterations + 1):
            used_potential = input_potential  # what enters this eigenproblem (the mixed one replaces it below)

            with timer.stage("diagonalization"):
                eigenvalues_by_k = solver.solve({"V": input_potential, "field": input_field}, filter_degree)
            residual_max = solver_residual_max(solver)

            with timer.stage("density"):
                stacked = np.concatenate(eigenvalues_by_k)
                order = np.argsort(stacked, kind="stable")
                occupation_result = fermi_occupations(
                    stacked[order],
                    system.electron_count,
                    settings.fermi_temperature_kelvin,
                    degeneracy=weight,
                )
                stacked_occupations = np.empty_like(stacked)
                stacked_occupations[order] = occupation_result.occupations
                fermi_level = occupation_result.fermi_level
                pooled_eigenvalues = stacked
                pooled_occupations = stacked_occupations

                parts = solver.accumulate(split_pooled(stacked_occupations, eigenvalues_by_k))
                density = parts["density"]
                magnetization = parts["magnetization"]

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
                energies = total_energy_noncollinear(
                    pooled_eigenvalues,
                    pooled_occupations,
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
                    band_energy_weight=weight,
                )

                mixed = mixer.mix(input_combined, output_combined, iteration=iteration)
            lap = timer.lap()
            selected_residual = (
                metrics.plain if settings.use_plain_residual else metrics.weighted
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
                    )
                )
            input_potential = mixed[:n_grid]
            input_field = mixed[n_grid:].reshape(3, n_grid)
            if (
                iteration > 5
                and metrics.weighted < 100.0 * settings.convergence_criterion
                and filter_degree > minimum_filter_degree
            ):
                filter_degree -= 1
            if selected_residual < settings.convergence_criterion:
                converged = True
                break

        moment_vector = volume_element * np.sum(magnetization, axis=1)
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
            k_weights=None if k_points is None else np.full(n_kpoints, weight),
            state_moments=state_moments,
            timings=solver_timings(timer, solver),
            scf_potentials={
                "effective_potential": output_potential,
                "xc_field": output_field,
            },
            xc_potential=xc.potential,
            input_effective_potential=used_potential,
            output_effective_potential=output_potential,
            nonlocal_forces=nonlocal_forces,
            next_effective_potential=(input_potential, input_field),
            hartree_potential=hartree_potential,
            eigensolver_state=subspaces,
        )


def run_self_consistent_noncollinear(
    system,
    soc_projectors: tuple = (),
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
    warm_start=None,
) -> SelfConsistentSOCResult:
    """Non-collinear SCF for an isolated system.

    ``warm_start`` (:class:`~parsec_python.SCF.single_point.SCFWarmStart`, whose ``input_potential`` is the pair
    ``(V, B_xc)``) continues from the loop of the previous structure of a relaxation or dynamics run.

    ``soc_projectors`` is
    :func:`~parsec_python.Eigensolvers.perturbative_soc.build_spin_orbit_projectors`'s
    output; leave it empty for non-collinear magnetism without spin-orbit.
    """

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

    return _run_noncollinear(system, hamiltonian_for_k, 1, 1.0, callback, warm_start=warm_start)


def run_self_consistent_noncollinear_kpoints(
    system,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
    compute_forces: bool = False,
    warm_start=None,
) -> SelfConsistentSOCResult:
    """Non-collinear SCF for a periodic cell on an unreduced uniform k grid.

    ``warm_start`` continues from the loop of the previous structure, as for the isolated driver.

    Spin-orbit projectors are built for every k-point from the species with
    ``SO_PSP`` (none gives plain non-collinear magnetism).  ``compute_forces``
    also returns the nonlocal force (``result.nonlocal_forces``), accumulated from the orbitals of every k-point by
    a last pass after the SCF loop (see :class:`~parsec_python.SCF.kpoints_soc.SpinorForceExtras`).
    """

    if not system.input.scf.noncollinear:
        raise ValueError(
            "run_self_consistent_noncollinear_kpoints requires "
            "SCFSettings.noncollinear=True"
        )
    k_points, weight = _validated_k_points(
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

    extras = SpinorForceExtras(system, k_points, weight, operators_for, compute_forces)
    return _run_noncollinear(
        system, hamiltonian_for_k, k_points.shape[0], weight, callback, k_points, extras, warm_start
    )


__all__ = [
    "run_self_consistent_noncollinear",
    "run_self_consistent_noncollinear_kpoints",
    "spinor_density_and_magnetization",
]
