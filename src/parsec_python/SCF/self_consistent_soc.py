"""Self-consistent spin-orbit coupling: Fortran's ``SO_from_scratch``.

Unlike :mod:`~parsec_python.Eigensolvers.perturbative_soc` (SOC added as a
one-shot small-matrix diagonalization after an ordinary scalar SCF
converges -- PARSEC's default, cheaper ``SCF_SO=false`` path), this module
puts the spin-orbit term directly in the Hamiltonian from the first SCF
iteration, using :class:`~parsec_python.Hamiltonian.spinor_operator.SpinorKohnShamHamiltonian`
as a matrix-free operator inside the same Chebyshev-filtered-subspace
eigensolver the scalar path already uses.

Both spinor components share one spin-unpolarized effective potential
``V_eff = V_ion,local + V_H[rho] + V_xc[rho]`` (the ordinary
:meth:`~parsec_python.SCF.single_point.PreparedSinglePointSystem.evaluate_xc`,
unchanged) -- this is ``SO_from_scratch`` *without* ``Non_Collinear_magnetism``,
matching the module docstring of :mod:`~parsec_python.Hamiltonian.spinor_operator`.
Combining self-consistent SOC with *collinear* spin polarization is
:mod:`~parsec_python.SCF.self_consistent_soc_spin_polarized`, a separate
driver built on the same :class:`~parsec_python.Hamiltonian.spinor_operator.SpinorKohnShamHamiltonian`
with its ``xc_delta`` field populated; fully non-collinear magnetism (a
rotating local moment / off-diagonal spin-density-matrix term) is still not
implemented.

The nonlinear map is the same shape as the scalar path,
``V_in -> eigensolver -> occupations -> rho -> (V_H,V_xc) -> V_out -> mix``,
except the eigensolver diagonalizes one ``2*n_grid``-dimensional complex
Hermitian operator instead of a real ``n_grid``-dimensional one, and each of
its ``2*n_grid``-length eigenvectors is one (already SOC-split) spinor state
holding at most one electron (``degeneracy=1``, matching
:mod:`~parsec_python.SCF.spin_polarized`'s pooled-channel convention, not
the unpolarized path's implicit factor of two).
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
from ..Eigensolvers.perturbative_soc import AtomSpinOrbitProjectors
from ..Energy import total_energy_no_degeneracy
from ..Hamiltonian.spinor_operator import SpinorKohnShamHamiltonian
from ..Mixer import AndersonMixer, potential_residual_metrics
from ..Occupations import fermi_occupations
from ..SCF.single_point import PreparedSinglePointSystem
from ..models import SelfConsistentSOCIteration, SelfConsistentSOCResult


def spinor_density_from_orbitals(
    spinors: np.ndarray,
    occupations: np.ndarray,
    volume_element: float,
) -> np.ndarray:
    """``rho = 1/dV * sum_n f_n * (|psi_n,up|**2 + |psi_n,down|**2)``.

    ``spinors`` is ``(2*n_grid, n_states)`` complex; each column is
    Euclidean-normalized as a *whole* 2-component spinor (the up and down
    halves do not separately normalize to 1), matching the eigensolver's
    convention for :class:`SpinorKohnShamHamiltonian`.
    """
    spinors = np.asarray(spinors)
    if spinors.ndim != 2 or spinors.shape[0] % 2:
        raise ValueError("spinors must have shape (2*n_grid, n_states)")
    occupations = np.asarray(occupations, dtype=float)
    if occupations.shape != (spinors.shape[1],):
        raise ValueError("occupation count does not match the spinor columns")
    if volume_element <= 0:
        raise ValueError("volume_element must be positive")
    n_grid = spinors.shape[0] // 2
    psi_up = spinors[:n_grid, :]
    psi_down = spinors[n_grid:, :]
    weighted = (np.abs(psi_up) ** 2 + np.abs(psi_down) ** 2) * occupations[None, :]
    return np.sum(weighted, axis=1) / volume_element


def _number_of_states(system: PreparedSinglePointSystem) -> int:
    """States requested from the ``2*n_grid``-dimensional spinor eigensolver.

    Each spinor state holds at most one electron (no spin degeneracy: spin
    is already explicit in the 2-component eigenvector), so this sizes like
    :mod:`~parsec_python.SCF.spin_polarized`'s per-channel count, not the
    unpolarized path's ``N_e/2``-orbital sizing.
    """
    requested = system.input.scf.number_of_states
    if requested is None:
        requested = int(np.ceil(system.electron_count)) + 6
    requested = int(requested)
    if requested + system.input.eigensolver.subspace_buffer >= 2 * system.grid.size:
        raise ValueError("the grid is too small for the requested states and eigensolver buffer")
    return requested


def _eigval_settings(system: PreparedSinglePointSystem, filter_degree: int) -> EigvalSettings:
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method != "chebff":
        raise NotImplementedError(
            f"Eigensolver={eigensolver_settings.method!r} is not supported for "
            "self-consistent SOC; only 'chebff' has a complex-capable trial "
            "basis and internal working arrays (CHEBDAV's are real-only)"
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


def _default_spinor_hamiltonian_builder(scalar_hamiltonian, soc_projectors):
    return SpinorKohnShamHamiltonian(
        scalar_hamiltonian.negative_laplacian,
        scalar_hamiltonian.effective_potential,
        scalar_hamiltonian.nonlocal_operator,
        soc_projectors,
    )


def run_self_consistent_soc(
    system: PreparedSinglePointSystem,
    soc_projectors: tuple[AtomSpinOrbitProjectors, ...],
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
    eigenproblem_solver: Callable[..., object] | None = None,
    spinor_hamiltonian_builder: Callable[..., object] | None = None,
    orbital_density_builder: Callable[..., np.ndarray] | None = None,
    mixer_factory: Callable[..., object] | None = None,
    total_energy_evaluator: Callable[..., object] | None = None,
) -> SelfConsistentSOCResult:
    """Run self-consistent-SOC PARSEC-style potential mixing.

    ``soc_projectors`` is
    :func:`~parsec_python.Eigensolvers.perturbative_soc.build_spin_orbit_projectors`'s
    output for ``system``'s atoms/pseudopotentials/specifications; an empty
    tuple degenerates this to an (expensive, real-eigenvalue-pair-doubled)
    ordinary scalar calculation, which is not a useful way to invoke this
    function but is not rejected -- the physics is well-defined either way.

    Requires ``system.input.scf.xc_functional`` supported by
    :meth:`~parsec_python.SCF.single_point.PreparedSinglePointSystem.evaluate_xc`
    (``'ca'`` or ``'pbe'``) and ``system.input.eigensolver.method == 'chebff'``.

    The five keyword-only callables mirror
    :func:`~parsec_python.SCF.single_point.run_scf`'s accelerated-backend
    injection points (all default to this module's/the reference's own
    behavior, so CPU callers see no change): ``eigenproblem_solver`` replaces
    :func:`~parsec_python.Eigensolvers.eigval.solve_eigval` (same
    ``(operator, requested_states, *, settings, state=None)`` call
    signature); ``spinor_hamiltonian_builder`` replaces the construction of
    :class:`~parsec_python.Hamiltonian.spinor_operator.SpinorKohnShamHamiltonian`
    from ``(scalar_hamiltonian, soc_projectors)`` -- an accelerated backend
    whose bound Hamiltonian does not expose raw
    ``negative_laplacian``/``nonlocal_operator`` attributes supplies its own
    complex spinor operator here instead; ``orbital_density_builder``
    replaces :func:`spinor_density_from_orbitals`; ``mixer_factory`` replaces
    :class:`~parsec_python.Mixer.AndersonMixer` (called as
    ``mixer_factory(system.input.mixing)``); ``total_energy_evaluator``
    replaces :func:`~parsec_python.Energy.total_energy_no_degeneracy`.
    """
    settings = system.input.scf
    number_of_states = _number_of_states(system)
    ionic_potential = system.ionic_potential
    solve_eigenproblem = (
        solve_eigval if eigenproblem_solver is None else eigenproblem_solver
    )
    build_spinor_hamiltonian = (
        _default_spinor_hamiltonian_builder
        if spinor_hamiltonian_builder is None
        else spinor_hamiltonian_builder
    )
    build_orbital_density = (
        spinor_density_from_orbitals
        if orbital_density_builder is None
        else orbital_density_builder
    )
    build_mixer = AndersonMixer if mixer_factory is None else mixer_factory
    evaluate_total_energy = (
        total_energy_no_degeneracy
        if total_energy_evaluator is None
        else total_energy_evaluator
    )

    initial_hartree = system.solve_hartree(
        system.initial_density, initial_potential=-ionic_potential
    )
    hartree_potential = initial_hartree.potential
    xc = system.evaluate_xc(system.initial_density)
    input_potential = ionic_potential + hartree_potential + xc.potential

    eigval_state: EigvalState | None = None
    mixer = build_mixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)

    eigenvalues = np.empty(0)
    occupations = np.empty(0)
    spinors = np.empty((2 * system.grid.size, 0), dtype=complex)
    fermi_level = float("nan")
    density = system.initial_density
    energies = None
    converged = False

    for iteration in range(1, settings.max_iterations + 1):
        eigval_settings = _eigval_settings(system, filter_degree)

        scalar_hamiltonian = system.hamiltonian(input_potential)
        hamiltonian = build_spinor_hamiltonian(scalar_hamiltonian, soc_projectors)
        operator_factory = getattr(
            hamiltonian, "as_eigensolver_operator", hamiltonian.as_linear_operator
        )
        solution = solve_eigenproblem(
            operator_factory(),
            number_of_states,
            settings=eigval_settings,
            state=eigval_state,
        )
        eigval_state = solution.state
        eigenvalues = np.asarray(solution.eigenvalues, dtype=float)
        spinors = np.asarray(solution.vectors)

        occupation_result = fermi_occupations(
            eigenvalues,
            system.electron_count,
            settings.fermi_temperature_kelvin,
            degeneracy=1.0,
        )
        occupations = occupation_result.occupations
        fermi_level = occupation_result.fermi_level

        density = build_orbital_density(
            spinors, occupations, system.grid.volume_element
        )

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

        energies = evaluate_total_energy(
            eigenvalues,
            occupations,
            density,
            input_potential,
            ionic_potential,
            hartree_potential,
            xc.potential,
            xc.total_energy,
            system.ion_ion_energy,
            system.grid.volume_element,
            alpha_z_energy=system.alpha_z_energy,
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
                    eigenvalues=tuple(float(value) for value in eigenvalues),
                    occupations=tuple(float(value) for value in occupations),
                    fermi_level=float(fermi_level),
                )
            )
        if (
            iteration > 5
            and metrics.weighted < 100.0 * settings.convergence_criterion
            and filter_degree > minimum_filter_degree
        ):
            filter_degree -= 1
        input_potential = mixed_potential
        if selected_residual < settings.convergence_criterion:
            converged = True
            break

    return SelfConsistentSOCResult(
        converged=converged,
        iterations=iteration,
        atoms=system.atoms,
        electron_count=system.electron_count,
        eigenvalues=eigenvalues,
        occupations=occupations,
        spinors=spinors,
        fermi_level=fermi_level,
        density=density,
        energies=energies,
    )


__all__ = ["run_self_consistent_soc", "spinor_density_from_orbitals"]
