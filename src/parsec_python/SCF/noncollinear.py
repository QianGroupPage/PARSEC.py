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
from .kpoints_soc import (
    _eigval_settings,
    _k_point_operators,
    _number_of_states,
    _spinor_hamiltonian,
    _validated_k_points,
)


def spinor_density_and_magnetization(
    spinors: np.ndarray,
    occupations: np.ndarray,
    volume_element: float,
    weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``(n, m)`` from stacked spinors ``(2*n_grid, n_states)``.

    ``n`` has shape ``(n_grid,)`` and ``m`` has shape ``(3, n_grid)``;
    ``weight`` is the k-point weight of these states.
    """

    spinors = np.asarray(spinors)
    occupations = np.asarray(occupations, dtype=float)
    if spinors.ndim != 2 or spinors.shape[0] % 2:
        raise ValueError("spinors must have shape (2*n_grid, n_states)")
    if occupations.shape != (spinors.shape[1],):
        raise ValueError("occupation count does not match the spinor columns")
    if volume_element <= 0:
        raise ValueError("volume_element must be positive")
    n_grid = spinors.shape[0] // 2
    up = spinors[:n_grid, :]
    down = spinors[n_grid:, :]
    scale = weight / volume_element
    cross = np.conj(up) * down
    density = scale * ((np.abs(up) ** 2 + np.abs(down) ** 2) @ occupations)
    magnetization = np.empty((3, n_grid))
    magnetization[0] = scale * (2.0 * cross.real @ occupations)
    magnetization[1] = scale * (2.0 * cross.imag @ occupations)
    magnetization[2] = scale * ((np.abs(up) ** 2 - np.abs(down) ** 2) @ occupations)
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
) -> SelfConsistentSOCResult:
    settings = system.input.scf
    n_grid = system.grid.size
    volume_element = system.grid.volume_element
    number_of_states = _number_of_states(system)
    ionic_potential = system.ionic_potential

    density = system.initial_density
    magnetization = _initial_magnetization(system)
    hartree = system.solve_hartree(density, initial_potential=-ionic_potential)
    hartree_potential = hartree.potential
    xc = system.evaluate_xc_noncollinear(density, magnetization)
    input_potential = ionic_potential + hartree_potential + xc.potential
    input_field = xc.field

    eigval_states: list[EigvalState | None] = [None] * n_kpoints
    mixer = AndersonMixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)

    pooled_eigenvalues = np.empty(0)
    pooled_occupations = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False
    spinors_first_k = np.empty((2 * n_grid, 0), dtype=complex)

    for iteration in range(1, settings.max_iterations + 1):
        eigval_settings = _eigval_settings(system, filter_degree)

        eigenvalues_by_k: list[np.ndarray] = []
        spinors_by_k: list[np.ndarray] = []
        for k_index in range(n_kpoints):
            hamiltonian = hamiltonian_for_k(k_index, input_potential, input_field)
            solution = solve_eigval(
                hamiltonian.as_linear_operator(),
                number_of_states,
                settings=eigval_settings,
                state=eigval_states[k_index],
            )
            eigval_states[k_index] = solution.state
            eigenvalues_by_k.append(np.asarray(solution.eigenvalues, dtype=float))
            spinors_by_k.append(np.asarray(solution.vectors))
            if k_index == 0:
                spinors_first_k = spinors_by_k[0]

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

        density = np.zeros(n_grid)
        magnetization = np.zeros((3, n_grid))
        offset = 0
        for k_index in range(n_kpoints):
            count = eigenvalues_by_k[k_index].size
            n_k, m_k = spinor_density_and_magnetization(
                spinors_by_k[k_index],
                stacked_occupations[offset : offset + count],
                volume_element,
                weight,
            )
            offset += count
            density += n_k
            magnetization += m_k

        hartree = system.solve_hartree(density, initial_potential=hartree_potential)
        hartree_potential = hartree.potential
        xc = system.evaluate_xc_noncollinear(density, magnetization)
        output_potential = ionic_potential + hartree_potential + xc.potential
        output_field = xc.field

        input_combined = np.concatenate([input_potential, input_field.reshape(-1)])
        output_combined = np.concatenate([output_potential, output_field.reshape(-1)])
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
        magnetic_moment=float(np.linalg.norm(moment_vector)),
        magnetic_moment_vector=moment_vector,
        magnetization=magnetization,
    )


def run_self_consistent_noncollinear(
    system,
    soc_projectors: tuple = (),
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
) -> SelfConsistentSOCResult:
    """Non-collinear SCF for an isolated system.

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

    return _run_noncollinear(system, hamiltonian_for_k, 1, 1.0, callback)


def run_self_consistent_noncollinear_kpoints(
    system,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
) -> SelfConsistentSOCResult:
    """Non-collinear SCF for a periodic cell on an unreduced uniform k grid.

    Spin-orbit projectors are built for every k-point from the species with
    ``SO_PSP`` (none gives plain non-collinear magnetism).
    """

    if not system.input.scf.noncollinear:
        raise ValueError(
            "run_self_consistent_noncollinear_kpoints requires "
            "SCFSettings.noncollinear=True"
        )
    k_points, weight = _validated_k_points(
        k_points, weights, "run_self_consistent_noncollinear_kpoints"
    )
    k_operators = _k_point_operators(system, k_points)

    def hamiltonian_for_k(k_index, potential, field):
        nonlocal_operator_k, soc_projectors_k = k_operators[k_index]
        return _spinor_hamiltonian(
            system,
            k_points[k_index],
            nonlocal_operator_k,
            soc_projectors_k,
            potential,
            xc_field=field,
        )

    return _run_noncollinear(
        system, hamiltonian_for_k, k_points.shape[0], weight, callback
    )


__all__ = [
    "run_self_consistent_noncollinear",
    "run_self_consistent_noncollinear_kpoints",
    "spinor_density_and_magnetization",
]
