"""Periodic self-consistent spin-orbit coupling with k-point sampling.

Combines :mod:`self_consistent_soc` (SOC in the Hamiltonian from the first
SCF iteration, one shared spin-unpolarized ``V_eff``, ``degeneracy=1``
spinor states) with :mod:`kpoints` (pooling every k-point's eigenvalues into
one Fermi-level bisection, k-weighted density): each SCF iteration
diagonalizes one
:class:`~parsec_python.Hamiltonian.kpoint_spinor_operator.KPointSpinorKohnShamHamiltonian`
per k-point, built from that k-point's Bloch-phase-weighted ordinary
nonlocal projectors
(:func:`~parsec_python.V_ion.ionic_potential.build_nonlocal_projectors`'s
``k_point`` argument) and spin-orbit projectors
(:func:`~parsec_python.Eigensolvers.perturbative_soc.build_spin_orbit_projectors`'s
``k_point`` argument).

As in :mod:`kpoints`, every k-point must carry the same weight (an
unreduced uniform Monkhorst-Pack grid): each pooled eigenvalue is one
already-SOC-split spinor state holding at most one electron, so
``degeneracy = weight`` (not ``2*weight`` as in the non-SOC :mod:`kpoints`,
since spin is already folded into each 2-component eigenvector).

:func:`run_self_consistent_soc_kpoints_spin_polarized` adds collinear spin
polarization (spin along z): a diagonal ``+/-xc_delta`` Zeeman-like term in
each k-point's spinor Hamiltonian and separately mixed up/down potentials, as
in the cluster :mod:`self_consistent_soc_spin_polarized`.  Spin polarization
breaks time-reversal symmetry, so the k-point set must be the full unreduced
Monkhorst-Pack grid (never folded by ``k ~ -k``), which is already required.

Not included: non-collinear magnetism (an off-diagonal ``B_xc.sigma`` term).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..Eigensolvers import (
    ChebDavSettings,
    ChebFFSettings,
    EigvalSettings,
    EigvalState,
    SubspaceSettings,
    solve_eigval,
)
from ..Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from ..Energy import total_energy_no_degeneracy, total_energy_no_degeneracy_spin_polarized
from ..Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from ..Hamiltonian.kpoint_spinor_operator import KPointSpinorKohnShamHamiltonian
from .kpoint_solver import KPointJob, make_kpoint_solver, split_pooled
from .stage_timer import StageTimer, solver_residual_max, solver_timings
from ..Mixer import AndersonMixer, potential_residual_metrics
from ..Occupations import fermi_occupations
from ..SCF.pbc import PeriodicPreparedSinglePointSystem
from ..SCF.spin_polarized import _weighted_initial_polarization
from ..V_ion import build_nonlocal_projectors
from ..models import SelfConsistentSOCIteration, SelfConsistentSOCResult


def _number_of_states(system: PeriodicPreparedSinglePointSystem) -> int:
    requested = system.input.scf.number_of_states
    if requested is None:
        requested = int(np.ceil(system.electron_count)) + 6
    requested = int(requested)
    if requested + system.input.eigensolver.subspace_buffer >= 2 * system.grid.size:
        raise ValueError("the grid is too small for the requested states and eigensolver buffer")
    return requested


def _eigval_settings(
    system: PeriodicPreparedSinglePointSystem, filter_degree: int
) -> EigvalSettings:
    """First-solver and later-filter policy for the complex (k-point) drivers.

    ``Eigensolver: chebff`` or ``chebdav`` (the Fortran default): both have
    complex-capable arithmetic.  CHEBDAV converges the first solve to
    ``Diag_Tolerance``; CHEBFF only applies ``FF_MaxIter`` filter cycles.
    """
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method not in ("chebff", "chebdav"):
        raise NotImplementedError(
            f"Eigensolver={eigensolver_settings.method!r} is not supported for "
            "k-point / spin-orbit calculations; use 'chebff' or 'chebdav'"
        )
    return EigvalSettings(
        safety_buffer=eigensolver_settings.subspace_buffer,
        initial_method=eigensolver_settings.method,
        chebff=ChebFFSettings(
            polynomial_degree=eigensolver_settings.first_filter_degree,
            filter_cycles=eigensolver_settings.first_filter_cycles,
            lanczos_steps=10,
            block_size=eigensolver_settings.matvec_block_size,
            reset_recurrence_per_block=False,
            random_seed=eigensolver_settings.random_seed,
        ),
        chebdav=(
            ChebDavSettings(
                polynomial_degree=eigensolver_settings.first_filter_degree,
                convergence_tolerance=eigensolver_settings.tolerance,
                block_size=eigensolver_settings.matvec_block_size,
                workspace_window=12,
                lanczos_steps=5,
                max_outer_restarts=2,
                random_seed=eigensolver_settings.random_seed,
            )
            if eigensolver_settings.method == "chebdav"
            else ChebDavSettings()
        ),
        subspace=SubspaceSettings(
            polynomial_degree=filter_degree,
            degree_delta=eigensolver_settings.filter_degree_delta,
            lanczos_steps=eigensolver_settings.lanczos_steps,
            block_size=eigensolver_settings.matvec_block_size,
            reset_recurrence_per_block=False,
            random_seed=eigensolver_settings.random_seed,
        ),
    )


def _state_sz(spinors_by_k: list[np.ndarray]) -> np.ndarray:
    """``<S_z>`` (up weight minus down weight) of every state at every k-point,
    k-major like the pooled eigenvalues."""
    moments = []
    for spinors in spinors_by_k:
        n_grid = spinors.shape[0] // 2
        moments.append(
            np.sum(np.abs(spinors[:n_grid, :]) ** 2, axis=0)
            - np.sum(np.abs(spinors[n_grid:, :]) ** 2, axis=0)
        )
    return np.concatenate(moments)


def _validated_k_points(
    k_points: np.ndarray, weights: np.ndarray, caller: str
) -> tuple[np.ndarray, float]:
    k_points = np.asarray(k_points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_kpoints = k_points.shape[0]
    if k_points.ndim != 2 or k_points.shape[1] != 3 or weights.shape != (n_kpoints,):
        raise ValueError("k_points must be (n_kpoints, 3) and weights (n_kpoints,)")
    if n_kpoints < 1:
        raise ValueError("at least one k-point is required")
    if not np.allclose(weights, weights[0]):
        raise ValueError(
            f"{caller} requires every k-point to carry "
            "the same weight (an unreduced uniform Monkhorst-Pack grid)"
        )
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("k-point weights must sum to 1")
    return k_points, float(weights[0])


def _k_point_operators(
    system: PeriodicPreparedSinglePointSystem, k_points: np.ndarray
) -> list[tuple]:
    """Per-k-point Bloch-phase nonlocal and spin-orbit projectors.

    These depend only on geometry, pseudopotentials and k, not on the SCF
    density, so they are built once rather than every iteration.
    """
    lattice_vectors = system.input.periodic_cell.lattice_vectors
    specifications = system.input.pseudopotentials
    operators = []
    for k_point in k_points:
        nonlocal_operator_k = build_nonlocal_projectors(
            system.grid,
            system.atoms,
            system.pseudopotentials,
            specifications,
            lattice_vectors,
            k_point=k_point,
        )
        soc_projectors_k = tuple(
            build_spin_orbit_projectors(
                system.grid,
                system.atoms,
                system.pseudopotentials,
                specifications,
                lattice_vectors,
                k_point=k_point,
            )
        )
        operators.append((nonlocal_operator_k, soc_projectors_k))
    return operators


def _spinor_hamiltonian(
    system: PeriodicPreparedSinglePointSystem,
    k_point: np.ndarray,
    nonlocal_operator_k,
    soc_projectors_k,
    effective_potential: np.ndarray,
    xc_delta: np.ndarray | None = None,
    xc_field: np.ndarray | None = None,
) -> KPointSpinorKohnShamHamiltonian:
    scalar_hamiltonian_k = KPointKohnShamHamiltonian(
        system.negative_laplacian,
        system.gradient,
        effective_potential,
        nonlocal_operator_k,
        k_point,
    )
    return KPointSpinorKohnShamHamiltonian(
        scalar_hamiltonian_k, soc_projectors_k, xc_delta=xc_delta, xc_field=xc_field
    )


def _spinor_solver(
    system: PeriodicPreparedSinglePointSystem,
    k_points: np.ndarray,
    number_of_states: int,
    accumulate,
    *,
    tasks: int | None = None,
) -> object:
    """Parallel solver of the spinor Bloch problems, one task per k-point.

    The Bloch-phase nonlocal and spin-orbit projectors of a k-point are built
    once, by the worker that owns it.  ``potentials`` carries ``V`` (the local
    potential) and optionally ``delta`` (collinear ``(V_up - V_down)/2``) or
    ``field`` (non-collinear ``B_xc``).
    """
    cache: dict[int, tuple] = {}

    def make_hamiltonian(task, potentials):
        if task not in cache:
            cache[task] = _k_point_operators(system, k_points[task : task + 1])[0]
        nonlocal_operator_k, soc_projectors_k = cache[task]
        return _spinor_hamiltonian(
            system,
            k_points[task],
            nonlocal_operator_k,
            soc_projectors_k,
            potentials["V"],
            xc_delta=potentials.get("delta"),
            xc_field=potentials.get("field"),
        )

    return make_kpoint_solver(
        KPointJob(
            k_points.shape[0] if tasks is None else tasks,
            number_of_states,
            lambda degree: _eigval_settings(system, degree),
            make_hamiltonian,
            accumulate,
            extras=lambda task, vectors: _state_sz([vectors]),
        ),
        problem_size=system.grid.size,
    )


def run_self_consistent_soc_kpoints(
    system: PeriodicPreparedSinglePointSystem,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
) -> SelfConsistentSOCResult:
    """Run periodic self-consistent-SOC SCF on an already-prepared system.

    ``weights`` must all be equal, exactly as in
    :func:`~parsec_python.SCF.kpoints.run_scf_kpoints`.  At least one
    species in ``system.input.pseudopotentials`` must have
    ``SpeciesPotential.spin_orbit=True`` with a matching relativistic POTRE
    file; an all-empty SOC projector set at every k-point would just be an
    expensive way to compute the ordinary (SOC-free) k-point result already
    available more cheaply from :mod:`kpoints`.
    """
    settings = system.input.scf
    k_points, weight = _validated_k_points(
        k_points, weights, "run_self_consistent_soc_kpoints"
    )
    n_kpoints = k_points.shape[0]

    number_of_states = _number_of_states(system)
    ionic_potential = system.ionic_potential

    timer = StageTimer()
    density = system.initial_density
    with timer.stage("setup_potentials"):
        initial_hartree = system.solve_hartree(density, initial_potential=-ionic_potential)
        hartree_potential = initial_hartree.potential
        xc = system.evaluate_xc(density)
    input_potential = ionic_potential + hartree_potential + xc.potential

    dV = system.grid.volume_element
    n_grid = system.grid.size

    def accumulate(task, vectors, occupations):
        # rho(r) = sum_k w_k*(1/dV)*sum_n f_nk*(|u_up,nk|**2+|u_down,nk|**2).
        weighted = (np.abs(vectors) ** 2) @ occupations
        return {"density": (weight / dV) * (weighted[:n_grid] + weighted[n_grid:])}

    solver = _spinor_solver(system, k_points, number_of_states, accumulate)
    mixer = AndersonMixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)

    pooled_eigenvalues = np.empty(0)
    pooled_occupations = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False
    # SelfConsistentSOCResult.spinors keeps only k_points[0]'s converged
    # spinors (storing all k-points would multiply the memory); the
    # per-state <S_z> of every k-point is stored separately (state_moments).
    with solver:
        for iteration in range(1, settings.max_iterations + 1):
            eigval_settings = _eigval_settings(system, filter_degree)

            with timer.stage("diagonalization"):
                eigenvalues_by_k = solver.solve({"V": input_potential}, filter_degree)
            residual_max = solver_residual_max(solver)

            # Pool every k-point's already-SOC-split spinor eigenvalues into one
            # shared Fermi-level bisection.  degeneracy=weight (not 2*weight,
            # unlike the non-SOC kpoints.py): spin is already folded into each
            # 2-component eigenvector, so each pooled entry holds at most one
            # electron per full Brillouin-zone sum, matching
            # self_consistent_soc.py's degeneracy=1 at a single k-point.
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

                density = solver.accumulate(split_pooled(stacked_occupations, eigenvalues_by_k))[
                    "density"
                ]

            with timer.stage("hartree"):
                hartree = system.solve_hartree(density, initial_potential=hartree_potential)
                hartree_potential = hartree.potential
            with timer.stage("xc"):
                xc = system.evaluate_xc(density)
                output_potential = ionic_potential + hartree_potential + xc.potential

            with timer.stage("mixing_energy"):
                metrics = potential_residual_metrics(
                    input_potential,
                    output_potential,
                    density,
                    system.grid.volume_element,
                    system.electron_count,
                )
                energies = total_energy_no_degeneracy(
                    pooled_eigenvalues,
                    pooled_occupations,
                    density,
                    input_potential,
                    ionic_potential,
                    hartree_potential,
                    xc.potential,
                    xc.total_energy,
                    system.ion_ion_energy,
                    system.grid.volume_element,
                    alpha_z_energy=system.alpha_z_energy,
                    band_energy_weight=weight,
                )

                mixed_potential = mixer.mix(input_potential, output_potential, iteration=iteration)
            selected_residual = (
                metrics.plain if settings.use_plain_residual else metrics.weighted
            )
            if callback is not None:
                callback(
                    SelfConsistentSOCIteration(
                        iteration=iteration,
                        weighted_residual=metrics.weighted,
                        plain_residual=metrics.plain,
                        energies=energies,
                        eigenvalues=tuple(float(value) for value in eigenvalues_by_k[0]),
                        occupations=tuple(
                            float(value) for value in stacked_occupations[: eigenvalues_by_k[0].size]
                        ),
                        fermi_level=float(fermi_level),
                        eigen_residual_max=residual_max,
                    )
                )
            input_potential = mixed_potential
            if (
                iteration > 5
                and metrics.weighted < 100.0 * settings.convergence_criterion
                and filter_degree > minimum_filter_degree
            ):
                filter_degree -= 1
            if selected_residual < settings.convergence_criterion:
                converged = True
                break

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
            k_points=k_points,
            k_weights=np.full(n_kpoints, weight),
            state_moments=np.concatenate(solver.extras()),
            timings=solver_timings(timer, solver),
            scf_potentials={"effective_potential": output_potential},
        )


def run_self_consistent_soc_kpoints_spin_polarized(
    system: PeriodicPreparedSinglePointSystem,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
) -> SelfConsistentSOCResult:
    """Periodic self-consistent SOC with collinear spin polarization.

    Needs ``SCFSettings.spin_polarized`` and
    ``SCFSettings.self_consistent_spin_orbit``.  The spin density is
    ``rho_a(r) = sum_k w_k/dV * sum_n f_nk |u_a,nk(r)|**2`` for spinor
    component ``a`` in {up, down}.  The XC potential comes from
    ``system.evaluate_xc_spin_polarized`` (LSDA or collinear PBE, with
    wraparound gradients); each k-point's Hamiltonian uses
    ``V_avg = (V_up + V_down)/2`` for both components plus
    ``+/- xc_delta`` with ``xc_delta = (V_up - V_down)/2``, and the two
    potentials are mixed together as one stacked vector, mirroring
    :func:`~parsec_python.SCF.self_consistent_soc_spin_polarized.run_self_consistent_soc_spin_polarized`.

    Every k-point must carry the same weight and the set must be the full,
    unreduced grid: spin polarization breaks time-reversal symmetry, so
    ``k`` and ``-k`` are no longer equivalent and must both be present.
    The returned ``magnetic_moment`` is ``N_up - N_down`` from the converged
    spin densities.
    """
    settings = system.input.scf
    if not settings.spin_polarized:
        raise ValueError(
            "run_self_consistent_soc_kpoints_spin_polarized requires "
            "SCFSettings.spin_polarized=True"
        )
    if not settings.self_consistent_spin_orbit:
        raise ValueError(
            "run_self_consistent_soc_kpoints_spin_polarized requires "
            "SCFSettings.self_consistent_spin_orbit=True"
        )
    k_points, weight = _validated_k_points(
        k_points, weights, "run_self_consistent_soc_kpoints_spin_polarized"
    )
    n_kpoints = k_points.shape[0]
    n_grid = system.grid.size

    number_of_states = _number_of_states(system)
    ionic_potential = system.ionic_potential

    timer = StageTimer()
    polarization = _weighted_initial_polarization(system)
    density_up = 0.5 * (1.0 + polarization) * system.initial_density
    density_down = 0.5 * (1.0 - polarization) * system.initial_density
    with timer.stage("setup_potentials"):
        initial_hartree = system.solve_hartree(
            density_up + density_down, initial_potential=-ionic_potential
        )
        hartree_potential = initial_hartree.potential
        xc = system.evaluate_xc_spin_polarized(density_up, density_down)
    input_potential_up = ionic_potential + hartree_potential + xc.potential_up
    input_potential_down = ionic_potential + hartree_potential + xc.potential_down

    dV = system.grid.volume_element

    def accumulate(task, vectors, occupations):
        weighted = (np.abs(vectors) ** 2) @ occupations
        return {
            "up": (weight / dV) * weighted[:n_grid],
            "down": (weight / dV) * weighted[n_grid:],
        }

    solver = _spinor_solver(system, k_points, number_of_states, accumulate)
    mixer = AndersonMixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)

    pooled_eigenvalues = np.empty(0)
    pooled_occupations = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False
    density_up_out = density_up
    density_down_out = density_down

    with solver:
        for iteration in range(1, settings.max_iterations + 1):
            eigval_settings = _eigval_settings(system, filter_degree)
            effective_potential_avg = 0.5 * (input_potential_up + input_potential_down)
            xc_delta = 0.5 * (input_potential_up - input_potential_down)

            with timer.stage("diagonalization"):
                eigenvalues_by_k = solver.solve(
                    {"V": effective_potential_avg, "delta": xc_delta}, filter_degree
                )
            residual_max = solver_residual_max(solver)

            # One Fermi level over every (k, spinor-state) eigenvalue; each holds
            # at most one electron per full-BZ sum, so degeneracy = weight.
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
                density_up_out = parts["up"]
                density_down_out = parts["down"]
                density = density_up_out + density_down_out

            with timer.stage("hartree"):
                hartree = system.solve_hartree(density, initial_potential=hartree_potential)
                hartree_potential = hartree.potential
            with timer.stage("xc"):
                xc = system.evaluate_xc_spin_polarized(density_up_out, density_down_out)
                output_potential_up = ionic_potential + hartree_potential + xc.potential_up
                output_potential_down = ionic_potential + hartree_potential + xc.potential_down

            input_combined = np.concatenate([input_potential_up, input_potential_down])
            output_combined = np.concatenate([output_potential_up, output_potential_down])
            density_combined = np.concatenate([density_up_out, density_down_out])
            with timer.stage("mixing_energy"):
                metrics = potential_residual_metrics(
                    input_combined,
                    output_combined,
                    density_combined,
                    system.grid.volume_element,
                    system.electron_count,
                )
                energies = total_energy_no_degeneracy_spin_polarized(
                    pooled_eigenvalues,
                    pooled_occupations,
                    density_up_out,
                    density_down_out,
                    input_potential_up,
                    input_potential_down,
                    ionic_potential,
                    hartree_potential,
                    xc.potential_up,
                    xc.potential_down,
                    xc.total_energy,
                    system.ion_ion_energy,
                    system.grid.volume_element,
                    alpha_z_energy=system.alpha_z_energy,
                    band_energy_weight=weight,
                )

                mixed_combined = mixer.mix(input_combined, output_combined, iteration=iteration)
            selected_residual = (
                metrics.plain if settings.use_plain_residual else metrics.weighted
            )
            if callback is not None:
                callback(
                    SelfConsistentSOCIteration(
                        iteration=iteration,
                        weighted_residual=metrics.weighted,
                        plain_residual=metrics.plain,
                        energies=energies,
                        eigenvalues=tuple(float(value) for value in eigenvalues_by_k[0]),
                        occupations=tuple(
                            float(value) for value in stacked_occupations[: eigenvalues_by_k[0].size]
                        ),
                        fermi_level=float(fermi_level),
                        eigen_residual_max=residual_max,
                    )
                )
            input_potential_up = mixed_combined[:n_grid]
            input_potential_down = mixed_combined[n_grid:]
            if (
                iteration > 5
                and metrics.weighted < 100.0 * settings.convergence_criterion
                and filter_degree > minimum_filter_degree
            ):
                filter_degree -= 1
            if selected_residual < settings.convergence_criterion:
                converged = True
                break

        return SelfConsistentSOCResult(
            converged=converged,
            iterations=iteration,
            atoms=system.atoms,
            electron_count=system.electron_count,
            eigenvalues=pooled_eigenvalues,
            occupations=pooled_occupations,
            spinors=solver.vectors(0),
            fermi_level=fermi_level,
            density=density_up_out + density_down_out,
            energies=energies,
            magnetic_moment=float(
                system.grid.volume_element * np.sum(density_up_out - density_down_out)
            ),
            k_points=k_points,
            k_weights=np.full(n_kpoints, weight),
            state_moments=np.concatenate(solver.extras()),
            timings=solver_timings(timer, solver),
            scf_potentials={
                "effective_potential": 0.5 * (output_potential_up + output_potential_down),
                "xc_delta": 0.5 * (output_potential_up - output_potential_down),
            },
        )


__all__ = [
    "run_self_consistent_soc_kpoints",
    "run_self_consistent_soc_kpoints_spin_polarized",
]
