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
from ..V_ion import build_nonlocal_projectors
from .kpoint_parallel import KPointJob, KPointSolver, split_pooled
from .kpoints_soc import _eigval_settings
from ..models import (
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



def run_scf_kpoints(
    system: PeriodicPreparedSinglePointSystem,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
) -> SinglePointResult:
    """Run k-point-sampled periodic SCF on an already-prepared system.

    ``weights`` must all be equal (see the module docstring); this is
    validated here rather than silently mixing unevenly-weighted k-points
    into one scalar-degeneracy Fermi bisection.
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

    density = system.initial_density
    initial_hartree = system.solve_hartree(density, initial_potential=-ionic_potential)
    hartree_potential = initial_hartree.potential
    xc = system.evaluate_xc(density)
    input_potential = ionic_potential + hartree_potential + xc.potential

    # Bloch-phase projectors depend only on geometry and k: built once per k-point
    # (in the worker that owns it) and reused by every SCF step.
    projector_cache: dict[int, object] = {}

    def make_hamiltonian(task, potentials):
        if task not in projector_cache:
            projector_cache[task] = build_nonlocal_projectors(
                system.grid,
                system.atoms,
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

    def accumulate(task, vectors, occupations):
        # rho(r) = sum_k w_k*(2/dV)*sum_n f_nk*|u_nk(r)|**2.  u_nk is complex
        # even though the resulting density is real.
        return {
            "density": (weight * 2.0 / system.grid.volume_element)
            * ((np.abs(vectors) ** 2) @ occupations)
        }

    solver = KPointSolver(
        KPointJob(
            n_kpoints,
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

    pooled_eigenvalues = np.empty(0)
    pooled_occupations = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False
    history: list[SCFIteration] = []

    with solver:
        for iteration in range(1, settings.max_iterations + 1):
            eigval_settings = _eigval_settings(system, filter_degree)

            eigenvalues_by_k = solver.solve({"V": input_potential}, filter_degree)

            # Pool every k-point's eigenvalues into one shared Fermi-level
            # bisection (flevel.f90's convention, generalized from spin
            # channels to k-points): degeneracy=2*weight because every pooled
            # entry is one (k,n) level that, at equal k-point weight, holds up
            # to 2*weight electrons after summing over the full BZ.
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

            density = solver.accumulate(split_pooled(stacked_occupations, eigenvalues_by_k))[
                "density"
            ]

            hartree = system.solve_hartree(density, initial_potential=hartree_potential)
            hartree_potential = hartree.potential
            xc = system.evaluate_xc(density)
            output_potential = ionic_potential + hartree_potential + xc.potential

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

            mixed_potential = mixer.mix(input_potential, output_potential, iteration=iteration)
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
                eigen_residual_max=float("nan"),
                hartree_residual=hartree.residual_norm,
                energies=energies,
                eigenvalues=tuple(float(value) for value in eigenvalues_by_k[0]),
                occupations=tuple(
                    float(value) for value in pooled_occupations[:first_k_count]
                ),
                fermi_level=float(fermi_level),
                density_minimum=float(np.min(density)),
                density_maximum=float(np.max(density)),
            )
            history.append(iteration_record)
            if callback is not None:
                callback(iteration_record)
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
        atomic_reference_correction=system.atomic_reference_correction,
        k_points=k_points,
        k_weights=weights,
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
    polarization = _weighted_initial_polarization(system)
    density_up = 0.5 * (1.0 + polarization) * system.initial_density
    density_down = 0.5 * (1.0 - polarization) * system.initial_density
    initial_hartree = system.solve_hartree(
        density_up + density_down, initial_potential=-ionic_potential
    )
    hartree_potential = initial_hartree.potential
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

    def accumulate(task, vectors, occupations):
        spin_index = task // n_kpoints
        return {
            "up" if spin_index == 0 else "down": (weight / system.grid.volume_element)
            * ((np.abs(vectors) ** 2) @ occupations)
        }

    solver = KPointSolver(
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
        for iteration in range(1, settings.max_iterations + 1):
            eigval_settings = _eigval_settings(system, filter_degree)

            by_task = solver.solve(
                {"up": input_potential_up, "down": input_potential_down}, filter_degree
            )
            eigenvalues_by_spin_k = [by_task[:n_kpoints], by_task[n_kpoints:]]

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

            occupations_by_task = split_pooled(occupations_up, eigenvalues_by_spin_k[0]) + split_pooled(
                occupations_down, eigenvalues_by_spin_k[1]
            )
            parts = solver.accumulate(occupations_by_task)
            density_up = parts.get("up", np.zeros(n_grid))
            density_down = parts.get("down", np.zeros(n_grid))

            hartree = system.solve_hartree(
                density_up + density_down, initial_potential=hartree_potential
            )
            hartree_potential = hartree.potential
            xc = system.evaluate_xc_spin_polarized(density_up, density_down)
            output_potential_up = ionic_potential + hartree_potential + xc.potential_up
            output_potential_down = ionic_potential + hartree_potential + xc.potential_down

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

            mixed_combined = mixer.mix(input_combined, output_combined, iteration=iteration)
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
        scf_potentials={
            "potential_up": output_potential_up,
            "potential_down": output_potential_down,
        },
    )


__all__ = ["run_scf_kpoints", "run_scf_kpoints_spin_polarized"]
