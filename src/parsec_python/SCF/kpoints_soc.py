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

Not included: combining with spin polarization/non-collinear magnetism
(would need a ``B_xc.sigma`` term -- see
:mod:`~parsec_python.Hamiltonian.kpoint_spinor_operator`'s module
docstring).
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
from ..Energy import total_energy_no_degeneracy
from ..Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from ..Hamiltonian.kpoint_spinor_operator import KPointSpinorKohnShamHamiltonian
from ..Mixer import AndersonMixer, potential_residual_metrics
from ..Occupations import fermi_occupations
from ..SCF.pbc import PeriodicPreparedSinglePointSystem
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
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method != "chebff":
        raise NotImplementedError(
            f"Eigensolver={eigensolver_settings.method!r} is not supported for "
            "periodic self-consistent SOC; only 'chebff' has a complex-capable "
            "trial basis and internal working arrays"
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
    k_points = np.asarray(k_points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_kpoints = k_points.shape[0]
    if k_points.ndim != 2 or k_points.shape[1] != 3 or weights.shape != (n_kpoints,):
        raise ValueError("k_points must be (n_kpoints, 3) and weights (n_kpoints,)")
    if n_kpoints < 1:
        raise ValueError("at least one k-point is required")
    if not np.allclose(weights, weights[0]):
        raise ValueError(
            "run_self_consistent_soc_kpoints requires every k-point to carry "
            "the same weight (an unreduced uniform Monkhorst-Pack grid)"
        )
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("k-point weights must sum to 1")
    weight = float(weights[0])

    number_of_states = _number_of_states(system)
    ionic_potential = system.ionic_potential
    lattice_vectors = system.input.periodic_cell.lattice_vectors
    specifications = system.input.pseudopotentials

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
    # SelfConsistentSOCResult.spinors has no single canonical value across
    # multiple k-points; this keeps only k_points[0]'s own converged
    # spinors as a representative sample (so magnetic_moment_per_state
    # still means something for that one k-point), not a full k-resolved
    # band structure.
    spinors_first_k = np.empty((2 * system.grid.size, 0), dtype=complex)

    for iteration in range(1, settings.max_iterations + 1):
        eigval_settings = _eigval_settings(system, filter_degree)

        eigenvalues_by_k: list[np.ndarray] = []
        spinors_by_k: list[np.ndarray] = []
        for k_index, k_point in enumerate(k_points):
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
            scalar_hamiltonian_k = KPointKohnShamHamiltonian(
                system.negative_laplacian,
                system.gradient,
                input_potential,
                nonlocal_operator_k,
                k_point,
            )
            hamiltonian_k = KPointSpinorKohnShamHamiltonian(
                scalar_hamiltonian_k, soc_projectors_k
            )
            solution = solve_eigval(
                hamiltonian_k.as_linear_operator(),
                number_of_states,
                settings=eigval_settings,
                state=eigval_states[k_index],
            )
            eigval_states[k_index] = solution.state
            eigenvalues_by_k.append(np.asarray(solution.eigenvalues, dtype=float))
            spinors_by_k.append(np.asarray(solution.vectors))
            if k_index == 0:
                spinors_first_k = spinors_by_k[0]

        # Pool every k-point's already-SOC-split spinor eigenvalues into one
        # shared Fermi-level bisection.  degeneracy=weight (not 2*weight,
        # unlike the non-SOC kpoints.py): spin is already folded into each
        # 2-component eigenvector, so each pooled entry holds at most one
        # electron per full Brillouin-zone sum, matching
        # self_consistent_soc.py's degeneracy=1 at a single k-point.
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

        # rho(r) = sum_k w_k*(1/dV)*sum_n f_nk*(|u_up,nk|**2+|u_down,nk|**2).
        n_grid = system.grid.size
        density = np.zeros(n_grid)
        offset = 0
        for k_index in range(n_kpoints):
            count = eigenvalues_by_k[k_index].size
            occ_k = stacked_occupations[offset : offset + count]
            offset += count
            spinor_k = spinors_by_k[k_index]
            weighted_density = (np.abs(spinor_k) ** 2) @ occ_k
            density_k = weighted_density[:n_grid] + weighted_density[n_grid:]
            density += (weight / system.grid.volume_element) * density_k

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
        spinors=spinors_first_k,
        fermi_level=fermi_level,
        density=density,
        energies=energies,
    )


__all__ = ["run_self_consistent_soc_kpoints"]
