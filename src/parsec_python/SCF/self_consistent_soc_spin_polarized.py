"""Self-consistent spin-orbit coupling combined with collinear spin
polarization (Fortran's ``SO_from_scratch``/``SCF_SO`` together with
``Spin_Polarization``).

Fortran's own input parser requires ``Spin_Polarization=true`` whenever any
species has ``SO_PSP=true``, even under ``SCF_SO`` (``usrinputfile.F90``:
``if (p_pot%is_so) ... if (nspin==1) ... ierr=151``) -- there is no Fortran
input that runs self-consistent SOC without collinear spin machinery
alongside it. :mod:`~parsec_python.SCF.self_consistent_soc` (this module's
prerequisite) implements the simpler spin-unpolarized case anyway, since a
system with zero net moment converges to the same physics either way (see
``examples/0d_AuH/reduced_self_consistent_soc``); this module adds the
piece needed for a system with nonzero net moment.

Physically, one extra term is needed relative to
:mod:`self_consistent_soc`: the exchange-correlation potential built from
the *spinor-projected* spin density (``rho_up(r) = sum_n f_n |psi_up,n(r)|^2``,
``rho_down(r)`` likewise) is no longer spin-independent.
the spin-polarized XC functional (LSDA, or collinear PBE via
:meth:`~parsec_python.SCF.single_point.PreparedSinglePointSystem.evaluate_xc_spin_polarized`)
splits it into
``V_xc,avg = (V_xc,up + V_xc,down)/2`` (added to the ordinary spin-unpolarized
effective potential shared by both spinor channels, exactly as in
:mod:`self_consistent_soc`) and ``V_xc,delta = (V_xc,up - V_xc,down)/2`` (a
new diagonal collinear Zeeman-like term, ``+V_xc,delta`` on the up channel
and ``-V_xc,delta`` on the down channel --
:class:`~parsec_python.Hamiltonian.spinor_operator.SpinorKohnShamHamiltonian`'s
``xc_delta`` field). The spin-orbit L.S term itself
(:mod:`~parsec_python.Eigensolvers.perturbative_soc`'s projectors, applied via
``apply_lzsz``/``apply_lsxy``) is unchanged.

This is still not full non-collinear magnetism (Fortran's
``Non_Collinear_magnetism`` flag): the local moment is always aligned along
z (the spinor's own up/down projection axis), never a rotating vector field,
and there is no off-diagonal spin-density-matrix term. Periodic (k-point)
self-consistent SOC combined with spin polarization is also not
implemented -- see :mod:`~parsec_python.SCF.kpoints_soc`'s module docstring.
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
from ..Energy import total_energy_no_degeneracy_spin_polarized
from ..Hamiltonian.spinor_operator import SpinorKohnShamHamiltonian
from ..Mixer import AndersonMixer, potential_residual_metrics
from ..Occupations import fermi_occupations
from ..SCF.single_point import PreparedSinglePointSystem
from ..SCF.spin_polarized import _weighted_initial_polarization
from ..models import SelfConsistentSOCIteration, SelfConsistentSOCResult


def spinor_spin_densities(
    spinors: np.ndarray,
    occupations: np.ndarray,
    volume_element: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(rho_up, rho_down)`` from stacked spinor eigenvectors.

    Unlike :func:`~parsec_python.SCF.self_consistent_soc.spinor_density_from_orbitals`
    (which returns only the summed total density), each spinor component's
    own projected density is kept separate -- the input to
    the spin-polarized XC functional.
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
    density_up = (np.abs(psi_up) ** 2) @ occupations / volume_element
    density_down = (np.abs(psi_down) ** 2) @ occupations / volume_element
    return density_up, density_down


def _number_of_states(system: PreparedSinglePointSystem) -> int:
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


def run_self_consistent_soc_spin_polarized(
    system: PreparedSinglePointSystem,
    soc_projectors: tuple[AtomSpinOrbitProjectors, ...],
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
) -> SelfConsistentSOCResult:
    """Run self-consistent SOC combined with collinear spin polarization.

    Requires ``system.input.scf.spin_polarized`` and
    ``system.input.scf.self_consistent_spin_orbit`` both set, and
    ``xc_functional`` either ``'ca'`` (LSDA) or ``'pbe'`` (collinear
    spin-polarized PBE with the spinor-projected spin densities).
    """
    settings = system.input.scf
    if not settings.spin_polarized:
        raise ValueError(
            "run_self_consistent_soc_spin_polarized requires "
            "SCFSettings.spin_polarized=True"
        )
    if not settings.self_consistent_spin_orbit:
        raise ValueError(
            "run_self_consistent_soc_spin_polarized requires "
            "SCFSettings.self_consistent_spin_orbit=True"
        )

    number_of_states = _number_of_states(system)
    ionic_potential = system.ionic_potential
    n_grid = system.grid.size

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

    eigval_state: EigvalState | None = None
    mixer = AndersonMixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)

    eigenvalues = np.empty(0)
    occupations = np.empty(0)
    spinors = np.empty((2 * n_grid, 0), dtype=complex)
    fermi_level = float("nan")
    density_up_out = density_up
    density_down_out = density_down
    energies = None
    converged = False

    for iteration in range(1, settings.max_iterations + 1):
        eigval_settings = _eigval_settings(system, filter_degree)

        effective_potential_avg = 0.5 * (input_potential_up + input_potential_down)
        xc_delta = 0.5 * (input_potential_up - input_potential_down)

        scalar_hamiltonian = system.hamiltonian(effective_potential_avg)
        hamiltonian = SpinorKohnShamHamiltonian(
            scalar_hamiltonian.negative_laplacian,
            scalar_hamiltonian.effective_potential,
            scalar_hamiltonian.nonlocal_operator,
            soc_projectors,
            xc_delta=xc_delta,
        )
        solution = solve_eigval(
            hamiltonian.as_linear_operator(),
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

        density_up_out, density_down_out = spinor_spin_densities(
            spinors, occupations, system.grid.volume_element
        )
        density = density_up_out + density_down_out

        hartree = system.solve_hartree(density, initial_potential=hartree_potential)
        hartree_potential = hartree.potential
        xc = system.evaluate_xc_spin_polarized(density_up_out, density_down_out)
        output_potential_up = ionic_potential + hartree_potential + xc.potential_up
        output_potential_down = ionic_potential + hartree_potential + xc.potential_down

        input_combined = np.concatenate([input_potential_up, input_potential_down])
        output_combined = np.concatenate([output_potential_up, output_potential_down])
        density_combined = np.concatenate([density_up_out, density_down_out])
        metrics = potential_residual_metrics(
            input_combined,
            output_combined,
            density_combined,
            system.grid.volume_element,
            system.electron_count,
        )

        energies = total_energy_no_degeneracy_spin_polarized(
            eigenvalues,
            occupations,
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
        )

        mixed_combined = mixer.mix(input_combined, output_combined, iteration=iteration)
        input_potential_up = mixed_combined[:n_grid]
        input_potential_down = mixed_combined[n_grid:]

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
        density=density_up_out + density_down_out,
        energies=energies,
        magnetic_moment=float(
            system.grid.volume_element * np.sum(density_up_out - density_down_out)
        ),
    )


__all__ = ["run_self_consistent_soc_spin_polarized", "spinor_spin_densities"]
