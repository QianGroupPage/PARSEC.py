"""Collinear spin-polarized PARSEC-style SCF for isolated single points.

This is the spin-polarized counterpart to :mod:`single_point`, following the
same overall potential-mixing algorithm but iterating two independent
Kohn--Sham eigenproblems (spin up, spin down) that share the grid, kinetic
operator, KB nonlocal projectors, local ionic potential, and Hartree
potential, and differ only in their exchange-correlation field:

``H_sigma[V_eff,sigma] = -nabla_FD^2 + diag(V_eff,sigma) + V_NL``,
``V_eff,sigma = V_ion,local + V_H[rho] + V_xc,sigma[rho_up,rho_down]``.

Matching ``flevel.f90``, both channels' eigenvalues are pooled into *one*
Fermi-level bisection with a shared chemical potential (electron count is
fixed; the magnetic moment ``N_up-N_down`` is free), not two independently
constrained channels.

Unlike :func:`~parsec_python.SCF.single_point.run_scf`, this function does
not yet carry PARSEC's full per-iteration timing/history instrumentation or
CLI/Output wiring -- see :class:`~parsec_python.models.SpinPolarizedSinglePointResult`.
It exists to produce a converged collinear ground state (eigenvalues,
occupations, and orbitals for both channels) that
:mod:`parsec_python.Eigensolvers.perturbative_soc`-style code can then use
for PARSEC's default perturbative spin-orbit correction.
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
from ..Energy import total_energy_spin_polarized
from ..Mixer import AndersonMixer, potential_residual_metrics
from ..Occupations import density_from_orbitals, fermi_occupations
from ..SCF.single_point import PreparedSinglePointSystem
from ..V_xc import ca_lda_spin_polarized
from ..models import SpinPolarizedSinglePointResult


def _weighted_initial_polarization(system: PreparedSinglePointSystem) -> float:
    """Ionic-charge-weighted average of each species' ``Initial_Spin_Polarization``.

    PARSEC seeds the spin-polarized initial density per atom, via
    ``initchrg.f90``'s ``p_fac = 1 +/- p_pot%spol(itype)`` applied to each
    atom's own contribution before summing.  Splitting the *already summed*
    total initial density (as :func:`prepare_single_point` builds it, shared
    with the unpolarized path) can only apply one shared fraction, not a
    per-atom one; this weighted average is a deliberate simplification for
    the initial guess only.  It has no effect on the converged
    self-consistent solution beyond which local minimum SCF may find, and
    any nonzero value serves its sole purpose: breaking the up/down symmetry
    so SCF does not stay trapped at zero moment.
    """
    specifications = system.input.pseudopotentials
    total_charge = 0.0
    weighted = 0.0
    for atom in system.atoms:
        charge = system.pseudopotentials[atom.symbol].ionic_charge
        weighted += charge * specifications[atom.symbol].initial_spin_polarization
        total_charge += charge
    if total_charge <= 0:
        return 0.0
    return weighted / total_charge


def _number_of_states_per_channel(system: PreparedSinglePointSystem) -> int:
    """States requested from *each* spin channel's eigensolver.

    Each spin channel's orbitals hold at most one electron (no spin
    degeneracy), so in the fully polarized limit one channel alone could
    need close to the full electron count.  Unlike the unpolarized
    ``_number_of_states`` (sized for ``N_e/2`` two-electron orbitals plus a
    buffer), an explicit ``States_Num`` here is taken as a per-channel count
    directly, matching Fortran's ``nstate = nstate*elec_st%mxwd`` scaling
    convention for extra spinor/spin components.
    """
    requested = system.input.scf.number_of_states
    if requested is None:
        requested = int(np.ceil(system.electron_count)) + 6
    requested = int(requested)
    if requested + system.input.eigensolver.subspace_buffer >= system.grid.size:
        raise ValueError("the grid is too small for the requested states and eigensolver buffer")
    return requested


def _eigval_settings(system: PreparedSinglePointSystem, filter_degree: int) -> EigvalSettings:
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method not in {"chebff", "chebdav"}:
        raise NotImplementedError(
            f"Eigensolver={eigensolver_settings.method!r} is not yet ported "
            "to the spin-polarized SCF path"
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


def run_scf_spin_polarized(
    system: PreparedSinglePointSystem,
    *,
    callback: Callable[[int, float], None] | None = None,
) -> SpinPolarizedSinglePointResult:
    """Run collinear spin-polarized PARSEC-style potential mixing.

    Requires ``system.input.scf.xc_functional == 'ca'``; spin-polarized PBE
    (``exc_spn.f90``'s gradient-corrected branch) is not yet implemented.
    """
    settings = system.input.scf
    if not settings.spin_polarized:
        raise ValueError(
            "run_scf_spin_polarized requires SCFSettings.spin_polarized=True"
        )
    if settings.xc_functional != "ca":
        raise NotImplementedError(
            "spin-polarized PBE is not yet implemented; only xc_functional='ca' "
            "(LSDA) is supported by run_scf_spin_polarized"
        )

    number_of_states = _number_of_states_per_channel(system)
    ionic_potential = system.ionic_potential

    polarization = _weighted_initial_polarization(system)
    density_up = 0.5 * (1.0 + polarization) * system.initial_density
    density_down = 0.5 * (1.0 - polarization) * system.initial_density

    initial_hartree = system.solve_hartree(
        density_up + density_down, initial_potential=-ionic_potential
    )
    hartree_potential = initial_hartree.potential
    xc = ca_lda_spin_polarized(
        density_up, density_down, system.grid.volume_element, system.core_density
    )
    input_potential_up = ionic_potential + hartree_potential + xc.potential_up
    input_potential_down = ionic_potential + hartree_potential + xc.potential_down

    eigval_state_up: EigvalState | None = None
    eigval_state_down: EigvalState | None = None
    # One Anderson history over the concatenated (V_up, V_down) vector, the
    # spin-polarized analog of mixing a single scalar field -- PARSEC treats
    # vnew(:,1:2) as one combined potential for mixing purposes.
    mixer = AndersonMixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)

    eigenvalues_up = eigenvalues_down = np.empty(0)
    occupations_up = occupations_down = np.empty(0)
    wavefunctions_up = wavefunctions_down = np.empty((system.grid.size, 0))
    fermi_level = float("nan")
    energies = None
    converged = False

    for iteration in range(1, settings.max_iterations + 1):
        eigval_settings = _eigval_settings(system, filter_degree)

        hamiltonian_up = system.hamiltonian(input_potential_up)
        solution_up = solve_eigval(
            hamiltonian_up.as_linear_operator(),
            number_of_states,
            settings=eigval_settings,
            state=eigval_state_up,
        )
        eigval_state_up = solution_up.state
        eigenvalues_up = solution_up.eigenvalues
        wavefunctions_up = solution_up.vectors

        hamiltonian_down = system.hamiltonian(input_potential_down)
        solution_down = solve_eigval(
            hamiltonian_down.as_linear_operator(),
            number_of_states,
            settings=eigval_settings,
            state=eigval_state_down,
        )
        eigval_state_down = solution_down.state
        eigenvalues_down = solution_down.eigenvalues
        wavefunctions_down = solution_down.vectors

        # Pool both channels for PARSEC's single shared Fermi level
        # (flevel.f90): each pooled level is one non-degenerate orbital, so
        # degeneracy=1 here (contrast the unpolarized path's degeneracy=2).
        pooled = np.concatenate([eigenvalues_up, eigenvalues_down])
        order = np.argsort(pooled, kind="stable")
        occupation_result = fermi_occupations(
            pooled[order],
            system.electron_count,
            settings.fermi_temperature_kelvin,
            degeneracy=1.0,
        )
        pooled_occupations = np.empty_like(pooled)
        pooled_occupations[order] = occupation_result.occupations
        occupations_up = pooled_occupations[: eigenvalues_up.size]
        occupations_down = pooled_occupations[eigenvalues_up.size :]
        fermi_level = occupation_result.fermi_level

        density_up = density_from_orbitals(
            wavefunctions_up, occupations_up, system.grid.volume_element, degeneracy=1.0
        )
        density_down = density_from_orbitals(
            wavefunctions_down,
            occupations_down,
            system.grid.volume_element,
            degeneracy=1.0,
        )

        hartree = system.solve_hartree(
            density_up + density_down, initial_potential=hartree_potential
        )
        hartree_potential = hartree.potential
        xc = ca_lda_spin_polarized(
            density_up, density_down, system.grid.volume_element, system.core_density
        )
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

        energies = total_energy_spin_polarized(
            eigenvalues_up,
            occupations_up,
            eigenvalues_down,
            occupations_down,
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
        )

        mixed_combined = mixer.mix(input_combined, output_combined, iteration=iteration)
        input_potential_up = mixed_combined[: system.grid.size]
        input_potential_down = mixed_combined[system.grid.size :]

        selected_residual = (
            metrics.plain if settings.use_plain_residual else metrics.weighted
        )
        if callback is not None:
            callback(iteration, selected_residual)
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
        wavefunctions_up=wavefunctions_up,
        wavefunctions_down=wavefunctions_down,
        fermi_level=fermi_level,
        density_up=density_up,
        density_down=density_down,
        energies=energies,
    )


__all__ = ["run_scf_spin_polarized"]
