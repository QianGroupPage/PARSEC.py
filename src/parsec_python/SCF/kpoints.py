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

Not included: spin-orbit coupling (the k-point-phase-weighted spin-orbit
projectors this would need are not built anywhere -- see
:mod:`~parsec_python.Hamiltonian.kpoint_operator`'s module docstring) and
spin polarization (would need per-(k,spin) pooling, not attempted here).
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
from ..Energy import total_energy
from ..Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from ..Mixer import AndersonMixer, potential_residual_metrics
from ..Occupations import fermi_occupations
from ..SCF.pbc import PeriodicPreparedSinglePointSystem
from ..V_ion import build_nonlocal_projectors
from ..models import SCFIteration, SinglePointResult


def _number_of_states(system: PeriodicPreparedSinglePointSystem) -> int:
    requested = system.input.scf.number_of_states
    if requested is None:
        requested = int(np.ceil(0.5 * system.electron_count)) + 6
    requested = int(requested)
    if requested + system.input.eigensolver.subspace_buffer >= system.grid.size:
        raise ValueError("the grid is too small for the requested states and eigensolver buffer")
    return requested


def _eigval_settings(
    system: PeriodicPreparedSinglePointSystem, filter_degree: int
) -> EigvalSettings:
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method != "chebff":
        raise NotImplementedError(
            f"Eigensolver={eigensolver_settings.method!r} is not supported for "
            "k-point sampling; only 'chebff' has a complex-capable trial "
            "basis and internal working arrays"
        )
    return EigvalSettings(
        safety_buffer=eigensolver_settings.subspace_buffer,
        initial_method="chebff",
        chebff=ChebFFSettings(
            polynomial_degree=eigensolver_settings.first_filter_degree,
            filter_cycles=eigensolver_settings.first_filter_cycles,
            lanczos_steps=10,
            block_size=eigensolver_settings.matvec_block_size,
            reset_recurrence_per_block=False,
            random_seed=eigensolver_settings.random_seed,
        ),
        chebdav=ChebDavSettings(),
        subspace=SubspaceSettings(
            polynomial_degree=filter_degree,
            degree_delta=eigensolver_settings.filter_degree_delta,
            lanczos_steps=eigensolver_settings.lanczos_steps,
            block_size=eigensolver_settings.matvec_block_size,
            reset_recurrence_per_block=False,
            random_seed=eigensolver_settings.random_seed,
        ),
    )


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

    eigval_states: list[EigvalState | None] = [None] * n_kpoints
    mixer = AndersonMixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)

    pooled_eigenvalues = np.empty(0)
    pooled_occupations = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False
    history: list[SCFIteration] = []

    for iteration in range(1, settings.max_iterations + 1):
        eigval_settings = _eigval_settings(system, filter_degree)

        eigenvalues_by_k: list[np.ndarray] = []
        vectors_by_k: list[np.ndarray] = []
        for k_index, k_point in enumerate(k_points):
            nonlocal_operator_k = build_nonlocal_projectors(
                system.grid,
                system.atoms,
                system.pseudopotentials,
                system.input.pseudopotentials,
                lattice_vectors,
                k_point=k_point,
            )
            hamiltonian_k = KPointKohnShamHamiltonian(
                system.negative_laplacian,
                system.gradient,
                input_potential,
                nonlocal_operator_k,
                k_point,
            )
            solution = solve_eigval(
                hamiltonian_k.as_linear_operator(),
                number_of_states,
                settings=eigval_settings,
                state=eigval_states[k_index],
            )
            eigval_states[k_index] = solution.state
            eigenvalues_by_k.append(np.asarray(solution.eigenvalues, dtype=float))
            vectors_by_k.append(np.asarray(solution.vectors))

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

        # rho(r) = sum_k w_k*(2/dV)*sum_n f_nk*|u_nk(r)|**2.  u_nk is
        # complex even though the resulting density is real.
        density = np.zeros(system.grid.size)
        offset = 0
        for k_index in range(n_kpoints):
            count = eigenvalues_by_k[k_index].size
            occ_k = stacked_occupations[offset : offset + count]
            offset += count
            u_k = vectors_by_k[k_index]
            weighted_density = (np.abs(u_k) ** 2) @ occ_k
            density += (weight * 2.0 / system.grid.volume_element) * weighted_density

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
        # Report the Gamma-point-equivalent first k-point's own eigenvalues
        # (a representative slice, not the full k-resolved band structure --
        # there is no single "the" eigenvalue list across multiple
        # k-points).  The full per-k-point results are not retained here.
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
    )


__all__ = ["run_scf_kpoints"]
