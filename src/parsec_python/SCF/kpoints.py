"""K-point-sampled periodic PARSEC-style SCF (real, spin-unpolarized).

The spin-unpolarized, non-SOC counterpart to :mod:`pbc` (Gamma-point-only)
for a Brillouin-zone sample built by
:func:`~parsec_python.Grid.kpoints.monkhorst_pack_grid`.  Each SCF iteration
diagonalizes one complex
:class:`~parsec_python.Hamiltonian.kpoint_operator.KPointKohnShamHamiltonian`
per k-point (built with that k-point's Bloch-phase-weighted nonlocal
projectors,
:func:`~parsec_python.V_ion.ionic_potential.build_nonlocal_projectors`'s
``k_point`` argument), pools every k-point's eigenvalues into *one* shared
Fermi-level bisection, and builds the density as the k-weighted sum of each
k-point's own density,

``rho(r) = sum_k w_k * (2/dV) * sum_n f_nk * |u_nk(r)|**2``.

This module requires every k-point to carry the *same* weight (an unreduced
uniform Monkhorst-Pack grid, see
:func:`~parsec_python.Grid.kpoints.monkhorst_pack_grid`'s docstring): that
is what lets the existing scalar-``degeneracy``
:func:`~parsec_python.Occupations.fermi_dirac.fermi_occupations` pool
eigenvalues from every k-point in one call (``degeneracy = 2*weight``,
spin-unpolarized), rather than needing a genuinely per-state weight
generalization of that bisection.  ``Energy.total_energy``'s
``band_energy_weight`` gets the same ``weight`` for the same reason.

Spin-orbit coupling lives in :mod:`~parsec_python.SCF.kpoints_soc`.
:func:`run_scf_kpoints_spin_polarized` adds collinear spin polarization
without spin-orbit: one Hamiltonian per (k-point, spin channel) -- sharing
each k-point's Bloch-phase nonlocal projectors, differing only in the spin
potential -- with every (k, spin) eigenvalue pooled into a single Fermi level.
"""

from __future__ import annotations

import time
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
from ..Energy import total_energy, total_energy_no_degeneracy_spin_polarized
from ..Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from ..Mixer import AndersonMixer, potential_residual_metrics
from ..Occupations import fermi_occupations
from ..SCF.pbc import PeriodicPreparedSinglePointSystem
from ..SCF.spin_polarized import (
    _number_of_states_per_channel,
    _weighted_initial_polarization,
)
from ..Forces.nonlocal_force import build_periodic_projector_gradients, nonlocal_forces
from ..V_ion import build_nonlocal_projectors
from .kpoint_solver import KPointJob, make_kpoint_solver, split_pooled
from .kpoints_soc import _eigval_settings
from .orbital_density import weighted_square_sum
from .stage_timer import StageTimer, solver_residual_max
from ..models import (
    RunTimings,
    SCFIteration,
    SinglePointResult,
    SpinPolarizedSCFIteration,
    SpinPolarizedSinglePointResult,
)


def _number_of_states(system: PeriodicPreparedSinglePointSystem) -> int:
    requested = system.input.scf.number_of_states
    if requested is None:
        requested = int(np.ceil(0.5 * system.electron_count)) + 6
    requested = int(requested)
    if requested + system.input.eigensolver.subspace_buffer >= system.grid.size:
        raise ValueError("the grid is too small for the requested states and eigensolver buffer")
    return requested


def _run_timings(system, timer: StageTimer, solver, started: float) -> RunTimings:
    """Stage table of a k-point SCF: the driver's stages plus what the solver measured inside."""
    seconds = timer.seconds
    detail = {"solver_setup": seconds.get("solver_setup", 0.0)}
    workers = getattr(solver, "n_workers", None)
    if workers is not None:
        detail["workers"] = float(workers)
    stats = getattr(solver, "stats", None)
    if stats is not None:
        for name, value in stats().items():
            detail[name] = float(value)
    return RunTimings(
        preparation=system.timings,
        initial_hartree_seconds=seconds.get("initial_hartree", 0.0),
        initial_xc_seconds=seconds.get("initial_xc", 0.0),
        diagonalization_seconds=seconds.get("diagonalization", 0.0),
        occupations_density_seconds=seconds.get("density", 0.0),
        hartree_seconds=seconds.get("hartree", 0.0),
        xc_seconds=seconds.get("xc", 0.0),
        mixing_energy_seconds=seconds.get("mixing_energy", 0.0),
        total_seconds=time.perf_counter() - started,
        detail=detail,
    )


def run_scf_kpoints(
    system: PeriodicPreparedSinglePointSystem,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
    compute_forces: bool = False,
    warm_start=None,
    convergence_criterion: float | None = None,
) -> SinglePointResult:
    """Run k-point-sampled periodic SCF on an already-prepared system.

    ``weights`` must all be equal (see the module docstring); this is
    validated here rather than silently mixing unevenly-weighted k-points
    into one scalar-degeneracy Fermi bisection.

    ``warm_start`` (:class:`~parsec_python.SCF.single_point.SCFWarmStart`) continues from the loop of the previous
    structure of a relaxation or a dynamics run: its input potential, the starting vector of the first Poisson solve and,
    with the atomic extrapolation, the Hartree and exchange-correlation potentials of the guessed density.  The k-point
    eigenproblems start cold (the saved eigensubspaces of the previous structure are not carried over).
    ``convergence_criterion`` replaces the ``Convergence_Criterion`` of the input for this loop (Ry).

    ``compute_forces`` also returns the nonlocal force (``result.nonlocal_forces``): the eigenvectors of a k-point
    run are not kept, so after the last SCF step each k-point's contribution
    ``4 w_k sum_n f_nk sum_q s_q Re[conj(<beta_q|u_nk>) <grad beta_q|u_nk>]`` is computed from them and summed.
    """
    settings = system.input.scf
    k_points = np.asarray(k_points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_kpoints = k_points.shape[0]
    if k_points.ndim != 2 or k_points.shape[1] != 3 or weights.shape != (n_kpoints,):
        raise ValueError("k_points must be (n_kpoints, 3) and weights (n_kpoints,)")
    if n_kpoints < 1:
        raise ValueError("at least one k-point is required")
    if not np.allclose(weights, weights[0]):
        raise ValueError(
            "run_scf_kpoints requires every k-point to carry the same weight "
            "(an unreduced uniform Monkhorst-Pack grid); a symmetry-reduced "
            "or explicitly weighted k-point list is not supported"
        )
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("k-point weights must sum to 1")
    weight = float(weights[0])

    number_of_states = _number_of_states(system)
    ionic_potential = system.ionic_potential
    lattice_vectors = system.input.periodic_cell.lattice_vectors
    cell_atoms = system.cell_atoms  # the atoms folded into the cell: the image sums assume them inside
    criterion = settings.convergence_criterion if convergence_criterion is None else float(convergence_criterion)

    started = time.perf_counter()
    timer = StageTimer()
    density = system.initial_density
    if warm_start is None:
        with timer.stage("initial_hartree"):
            initial_hartree = system.solve_hartree(density, initial_potential=-ionic_potential)
        hartree_potential = initial_hartree.potential
        with timer.stage("initial_xc"):
            xc = system.evaluate_xc(density)
        input_potential = ionic_potential + hartree_potential + xc.potential
    else:
        # the structure before the move supplies the first input potential and the starting vector of the first
        # Poisson solve; no density is needed before the first eigensolve
        hartree_potential = np.array(warm_start.hartree_potential, dtype=np.float64)
        input_potential = np.array(warm_start.input_potential, dtype=np.float64)
        if warm_start.density_guess is not None:
            # atomic extrapolation: the Hartree and exchange-correlation potentials of the guessed density replace
            # those of the last output density
            guess = np.array(warm_start.density_guess, dtype=np.float64)
            with timer.stage("initial_hartree"):
                hartree_potential = system.solve_hartree(guess, initial_potential=hartree_potential).potential
            with timer.stage("initial_xc"):
                moved_xc = system.evaluate_xc(guess)
            input_potential = (
                input_potential
                + (hartree_potential + moved_xc.potential)
                - np.array(warm_start.hxc_reference, dtype=np.float64)
            )

    # Bloch-phase projectors depend only on geometry and k: built once per k-point
    # (in the worker that owns it) and reused by every SCF step.
    projector_cache: dict[int, object] = {}

    def make_hamiltonian(task, potentials):
        if task not in projector_cache:
            projector_cache[task] = build_nonlocal_projectors(
                system.grid,
                cell_atoms,
                system.pseudopotentials,
                system.input.pseudopotentials,
                lattice_vectors,
                k_point=k_points[task],
            )
        return KPointKohnShamHamiltonian(
            system.negative_laplacian,
            system.gradient,
            potentials["V"],
            projector_cache[task],
            k_points[task],
        )

    def accumulate(task, vectors, occupations, xp=np):
        # rho(r) = sum_k w_k*(2/dV)*sum_n f_nk*|u_nk(r)|**2.  u_nk is complex
        # even though the resulting density is real.
        return {
            "density": (weight * 2.0 / system.grid.volume_element)
            * weighted_square_sum(vectors, occupations, xp)
        }

    # Occupations of the last SCF step by k-point, set before the force pass (the job's ``extras`` sees only vectors).
    final_occupations: dict[int, np.ndarray] = {}

    def kpoint_nonlocal_force(task, vectors, xp=np):
        if task not in projector_cache:
            make_hamiltonian(task, {"V": ionic_potential})
        gradients = build_periodic_projector_gradients(
            system.grid,
            cell_atoms,
            system.pseudopotentials,
            system.input.pseudopotentials,
            lattice_vectors,
            k_points[task],
        )
        if xp is not np and not isinstance(vectors, np.ndarray):
            # the orbitals of a device stay there: the overlaps are formed on it (the acceleration package, which
            # supplied the device array, supplies the contraction)
            from ..acceleration.Forces.kpoint_nonlocal_device import device_nonlocal_forces

            return device_nonlocal_forces(
                xp, projector_cache[task], gradients, vectors, weight * final_occupations[task], len(cell_atoms)
            )
        return nonlocal_forces(
            projector_cache[task],
            gradients,
            np.asarray(vectors),
            weight * final_occupations[task],
            len(cell_atoms),
        )

    with timer.stage("solver_setup"):
        solver = make_kpoint_solver(
            KPointJob(
                n_kpoints,
                number_of_states,
                lambda degree: _eigval_settings(system, degree),
                make_hamiltonian,
                accumulate,
                extras=kpoint_nonlocal_force if compute_forces else None,
            ),
            problem_size=system.grid.size,
        )
    if compute_forces and type(solver).__name__ == "MPIKPointSolver":
        raise NotImplementedError(
            "the nonlocal force of the row-distributed MPI k-point solver is not implemented: its extras are "
            "sums over row blocks, which does not apply to a product of overlaps"
        )
    mixer = AndersonMixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)

    nonlocal_force = None
    pooled_eigenvalues = np.empty(0)
    pooled_occupations = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False
    history: list[SCFIteration] = []

    with solver:
        timer.lap()  # what came before the loop (potentials, solver start-up) is not part of iteration 1
        for iteration in range(1, settings.max_iterations + 1):
            eigval_settings = _eigval_settings(system, filter_degree)

            with timer.stage("diagonalization"):
                eigenvalues_by_k = solver.solve({"V": input_potential}, filter_degree)
            residual_max = solver_residual_max(solver)

            # Pool every k-point's eigenvalues into one shared Fermi-level
            # bisection (flevel.f90's convention, generalized from spin
            # channels to k-points): degeneracy=2*weight because every pooled
            # entry is one (k,n) level that, at equal k-point weight, holds up
            # to 2*weight electrons after summing over the full BZ.
            with timer.stage("density"):
                stacked = np.concatenate(eigenvalues_by_k)
                order = np.argsort(stacked, kind="stable")
                occupation_result = fermi_occupations(
                    stacked[order],
                    system.electron_count,
                    settings.fermi_temperature_kelvin,
                    degeneracy=2.0 * weight,
                )
                stacked_occupations = np.empty_like(stacked)
                stacked_occupations[order] = occupation_result.occupations
                fermi_level = occupation_result.fermi_level
                pooled_eigenvalues = stacked
                pooled_occupations = stacked_occupations

                density = solver.accumulate(
                    split_pooled(stacked_occupations, eigenvalues_by_k)
                )["density"]

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
                energies = total_energy(
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

                mixed_potential = mixer.mix(
                    input_potential, output_potential, iteration=iteration
                )
            lap = timer.lap()
            selected_residual = (
                metrics.plain if settings.use_plain_residual else metrics.weighted
            )
            # The per-iteration record carries only the first k-point's
            # eigenvalues; the final result pools every k-point (k-major) with
            # the k-point list, and the reporter prints them all at the end.
            first_k_count = eigenvalues_by_k[0].size
            iteration_record = SCFIteration(
                iteration=iteration,
                weighted_residual=metrics.weighted,
                plain_residual=metrics.plain,
                eigen_residual_max=residual_max,
                hartree_residual=hartree.residual_norm,
                energies=energies,
                eigenvalues=tuple(float(value) for value in eigenvalues_by_k[0]),
                occupations=tuple(
                    float(value) for value in pooled_occupations[:first_k_count]
                ),
                fermi_level=float(fermi_level),
                density_minimum=float(np.min(density)),
                density_maximum=float(np.max(density)),
                diagonalization_seconds=lap.get("diagonalization", 0.0),
                hartree_seconds=lap.get("hartree", 0.0),
                occupations_density_seconds=lap.get("density", 0.0),
                xc_seconds=lap.get("xc", 0.0),
                mixing_energy_seconds=lap.get("mixing_energy", 0.0),
                total_seconds=sum(lap.values()),
            )
            history.append(iteration_record)
            if callback is not None:
                callback(iteration_record)
            input_potential = mixed_potential
            if (
                iteration > 5
                and metrics.weighted < 100.0 * criterion
                and filter_degree > minimum_filter_degree
            ):
                filter_degree -= 1
            if selected_residual < criterion:
                converged = True
                break
        if compute_forces:
            # the eigenvectors and occupations of the last step belong together
            final_occupations.update(enumerate(split_pooled(pooled_occupations, eigenvalues_by_k)))
            nonlocal_force = np.sum(solver.extras(), axis=0)
        timings = _run_timings(system, timer, solver, started)

    return SinglePointResult(
        converged=converged,
        iterations=iteration,
        atoms=system.atoms,
        electron_count=system.electron_count,
        grid=system.grid,
        pseudopotentials=system.pseudopotentials,
        density=density,
        core_density=system.core_density,
        ionic_potential=ionic_potential,
        hartree_potential=hartree_potential,
        xc_potential=xc.potential,
        input_effective_potential=input_potential,
        output_effective_potential=output_potential,
        next_effective_potential=mixed_potential,
        nonlocal_operator=None,
        eigenvalues=pooled_eigenvalues,
        occupations=pooled_occupations,
        wavefunctions=np.empty((system.grid.size, 0)),
        fermi_level=fermi_level,
        energies=energies,
        history=history,
        timings=timings,
        atomic_reference_correction=system.atomic_reference_correction,
        k_points=k_points,
        k_weights=weights,
        nonlocal_forces=nonlocal_force,
    )


def run_scf_kpoints_spin_polarized(
    system: PeriodicPreparedSinglePointSystem,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    callback: Callable[[SpinPolarizedSCFIteration], None] | None = None,
) -> SpinPolarizedSinglePointResult:
    """K-point-sampled periodic collinear spin-polarized SCF (no spin-orbit).

    Needs ``SCFSettings.spin_polarized`` (LSDA or collinear PBE per
    ``xc_functional``).  Each iteration diagonalizes one real-potential
    Hamiltonian per (k-point, spin channel); a channel's orbitals hold at
    most one electron, so every pooled level carries ``degeneracy = weight``
    and one Fermi level is found over all ``2 * n_kpoints * n_states``
    eigenvalues.  The spin densities are
    ``rho_a(r) = sum_k w_k/dV * sum_n f_nk,a |u_nk,a(r)|**2`` and the two spin
    potentials are mixed as one stacked vector, as in
    :func:`~parsec_python.SCF.spin_polarized.run_scf_spin_polarized`.

    Every k-point must carry the same weight and the set must be the full
    unreduced grid: spin polarization breaks time-reversal symmetry, so
    ``k`` and ``-k`` are not equivalent.  ``States_Num`` is a per-channel,
    per-k-point count.  The returned eigenvalues and occupations are pooled
    over k-points (k-point-major within each spin); each carries
    ``result.occupation_weight`` (= the k-point weight) in any sum such as
    the electron count or ``magnetic_moment``.  Wavefunctions are not retained.
    """
    settings = system.input.scf
    if not settings.spin_polarized:
        raise ValueError(
            "run_scf_kpoints_spin_polarized requires SCFSettings.spin_polarized=True"
        )
    k_points = np.asarray(k_points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_kpoints = k_points.shape[0]
    if k_points.ndim != 2 or k_points.shape[1] != 3 or weights.shape != (n_kpoints,):
        raise ValueError("k_points must be (n_kpoints, 3) and weights (n_kpoints,)")
    if n_kpoints < 1:
        raise ValueError("at least one k-point is required")
    if not np.allclose(weights, weights[0]):
        raise ValueError(
            "run_scf_kpoints_spin_polarized requires every k-point to carry "
            "the same weight (an unreduced uniform Monkhorst-Pack grid)"
        )
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("k-point weights must sum to 1")
    weight = float(weights[0])
    n_grid = system.grid.size

    number_of_states = _number_of_states_per_channel(system)
    ionic_potential = system.ionic_potential
    lattice_vectors = system.input.periodic_cell.lattice_vectors
    started = time.perf_counter()
    timer = StageTimer()
    polarization = _weighted_initial_polarization(system)
    density_up = 0.5 * (1.0 + polarization) * system.initial_density
    density_down = 0.5 * (1.0 - polarization) * system.initial_density
    with timer.stage("initial_hartree"):
        initial_hartree = system.solve_hartree(
            density_up + density_down, initial_potential=-ionic_potential
        )
    hartree_potential = initial_hartree.potential
    with timer.stage("initial_xc"):
        xc = system.evaluate_xc_spin_polarized(density_up, density_down)
    input_potential_up = ionic_potential + hartree_potential + xc.potential_up
    input_potential_down = ionic_potential + hartree_potential + xc.potential_down

    # Tasks are (spin, k) pairs, task = spin*n_kpoints + k; each worker builds the
    # Bloch-phase projectors of the k-points it owns once.
    projector_cache: dict[int, object] = {}

    def make_hamiltonian(task, potentials):
        spin_index, k_index = divmod(task, n_kpoints)
        if k_index not in projector_cache:
            projector_cache[k_index] = build_nonlocal_projectors(
                system.grid,
                system.atoms,
                system.pseudopotentials,
                system.input.pseudopotentials,
                lattice_vectors,
                k_point=k_points[k_index],
            )
        return KPointKohnShamHamiltonian(
            system.negative_laplacian,
            system.gradient,
            potentials["up" if spin_index == 0 else "down"],
            projector_cache[k_index],
            k_points[k_index],
        )

    def accumulate(task, vectors, occupations, xp=np):
        spin_index = task // n_kpoints
        return {
            "up" if spin_index == 0 else "down": (weight / system.grid.volume_element)
            * weighted_square_sum(vectors, occupations, xp)
        }

    with timer.stage("solver_setup"):
        solver = make_kpoint_solver(
            KPointJob(
                2 * n_kpoints,
                number_of_states,
                lambda degree: _eigval_settings(system, degree),
                make_hamiltonian,
                accumulate,
            ),
            problem_size=system.grid.size,
        )
    mixer = AndersonMixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)

    eigenvalues_up = eigenvalues_down = np.empty(0)
    occupations_up = occupations_down = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False

    with solver:
        timer.lap()  # what came before the loop (potentials, solver start-up) is not part of iteration 1
        for iteration in range(1, settings.max_iterations + 1):
            eigval_settings = _eigval_settings(system, filter_degree)

            with timer.stage("diagonalization"):
                by_task = solver.solve(
                    {"up": input_potential_up, "down": input_potential_down}, filter_degree
                )
            residual_max = solver_residual_max(solver)
            eigenvalues_by_spin_k = [by_task[:n_kpoints], by_task[n_kpoints:]]

            with timer.stage("density"):
                eigenvalues_up = np.concatenate(eigenvalues_by_spin_k[0])
                eigenvalues_down = np.concatenate(eigenvalues_by_spin_k[1])
                pooled = np.concatenate([eigenvalues_up, eigenvalues_down])
                order = np.argsort(pooled, kind="stable")
                occupation_result = fermi_occupations(
                    pooled[order],
                    system.electron_count,
                    settings.fermi_temperature_kelvin,
                    degeneracy=weight,
                )
                pooled_occupations = np.empty_like(pooled)
                pooled_occupations[order] = occupation_result.occupations
                occupations_up = pooled_occupations[: eigenvalues_up.size]
                occupations_down = pooled_occupations[eigenvalues_up.size :]
                fermi_level = occupation_result.fermi_level

                occupations_by_task = split_pooled(
                    occupations_up, eigenvalues_by_spin_k[0]
                ) + split_pooled(occupations_down, eigenvalues_by_spin_k[1])
                parts = solver.accumulate(occupations_by_task)
                density_up = parts.get("up", np.zeros(n_grid))
                density_down = parts.get("down", np.zeros(n_grid))

            with timer.stage("hartree"):
                hartree = system.solve_hartree(
                    density_up + density_down, initial_potential=hartree_potential
                )
            hartree_potential = hartree.potential
            with timer.stage("xc"):
                xc = system.evaluate_xc_spin_polarized(density_up, density_down)
            output_potential_up = ionic_potential + hartree_potential + xc.potential_up
            output_potential_down = ionic_potential + hartree_potential + xc.potential_down

            with timer.stage("mixing_energy"):
                input_combined = np.concatenate([input_potential_up, input_potential_down])
                output_combined = np.concatenate([output_potential_up, output_potential_down])
                density_combined = np.concatenate([density_up, density_down])
                metrics = potential_residual_metrics(
                    input_combined,
                    output_combined,
                    density_combined,
                    system.grid.volume_element,
                    system.electron_count,
                )
                energies = total_energy_no_degeneracy_spin_polarized(
                    pooled,
                    pooled_occupations,
                    density_up,
                    density_down,
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

                mixed_combined = mixer.mix(
                    input_combined, output_combined, iteration=iteration
                )
            lap = timer.lap()
            selected_residual = (
                metrics.plain if settings.use_plain_residual else metrics.weighted
            )
            if callback is not None:
                first_k = eigenvalues_by_spin_k[0][0].size
                callback(
                    SpinPolarizedSCFIteration(
                        iteration=iteration,
                        weighted_residual=metrics.weighted,
                        plain_residual=metrics.plain,
                        energies=energies,
                        eigenvalues_up=tuple(
                            float(value) for value in eigenvalues_by_spin_k[0][0]
                        ),
                        eigenvalues_down=tuple(
                            float(value) for value in eigenvalues_by_spin_k[1][0]
                        ),
                        occupations_up=tuple(float(v) for v in occupations_up[:first_k]),
                        occupations_down=tuple(float(v) for v in occupations_down[:first_k]),
                        fermi_level=float(fermi_level),
                        diagonalization_seconds=lap.get("diagonalization", 0.0),
                        total_seconds=sum(lap.values()),
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
        timings = _run_timings(system, timer, solver, started)

    return SpinPolarizedSinglePointResult(
        converged=converged,
        iterations=iteration,
        atoms=system.atoms,
        electron_count=system.electron_count,
        eigenvalues_up=eigenvalues_up,
        eigenvalues_down=eigenvalues_down,
        occupations_up=occupations_up,
        occupations_down=occupations_down,
        wavefunctions_up=np.empty((n_grid, 0)),
        wavefunctions_down=np.empty((n_grid, 0)),
        fermi_level=fermi_level,
        density_up=density_up,
        density_down=density_down,
        energies=energies,
        occupation_weight=weight,
        k_points=k_points,
        k_weights=weights,
        timings=timings,
        scf_potentials={
            "potential_up": output_potential_up,
            "potential_down": output_potential_down,
        },
    )


__all__ = ["run_scf_kpoints", "run_scf_kpoints_spin_polarized"]
