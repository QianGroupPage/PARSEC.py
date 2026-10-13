"""Spin-polarized (collinear) PARSEC-style SCF iteration of an isolated system.

``Spin_Polarization: true`` gives the calculation two spin channels.  The
loop of ``parsec.F90`` is that of the unpolarized case with these changes:

* each channel ``s`` has its own local potential
  ``V_s = V_ion + V_H[rho] + V_xc,s[rho_up, rho_down]`` and its own
  eigenproblem, solved with its own saved subspace (``eigval.F90``);
* one Fermi level fills both channels, a state holding one electron
  (``flevel.f90``);
* the density of a channel is ``rho_s = h**-3 sum_i f_s,i |psi_s,i|**2`` and
  the Hartree potential is that of ``rho = rho_up + rho_down``
  (``newrho.F90``, ``rho_hart.F90``);
* the exchange-correlation functional is the spin-polarized one
  (``exc_spn.f90``);
* the residual norms are evaluated per channel and the larger one is tested
  (``getsre.f90``);
* the mixer sees the two potentials as one vector, the up channel first
  (``spinmix.f90``, ``spinunmix.f90``);
* the band energy and the double-counting integrals are sums over the
  channels (``totnrg.f90``).

The first potentials come from the starting densities of
:func:`parsec_python.SCF.single_point.initial_spin_densities`.
"""

from __future__ import annotations

import sys
import time
from typing import Callable

import numpy as np

from ..Eigensolvers import solve_eigval
from ..Energy import spin_total_energy
from ..Mixer import AndersonMixer, spin_potential_residual_metrics
from ..Occupations import (
    spin_density_from_orbitals,
    spin_fermi_occupations,
    spin_fixed_occupations,
)
from ..models import (
    RunTimings,
    SCFIteration,
    SinglePointResult,
    SpinChannelResult,
)
from .single_point import (
    PreparedSinglePointSystem,
    SCFWarmStart,
    _number_of_states,
    eigval_settings_for,
)


SPIN_NAMES = ("up", "down")


def _channel_solvers(eigenproblem_solver) -> tuple[Callable[..., object], ...]:
    """Return one eigenproblem solver per spin channel.

    The readable :func:`solve_eigval` carries its saved subspace in the
    state it is handed, so one function serves both channels.  A substituted
    solver that keeps a subspace of its own must say how it serves a second
    channel, through ``for_spin_channel(index)``; one that does not is
    refused rather than run with the subspace of the wrong channel.
    """

    if eigenproblem_solver is None:
        return (solve_eigval, solve_eigval)
    for_channel = getattr(eigenproblem_solver, "for_spin_channel", None)
    if not callable(for_channel):
        raise NotImplementedError(
            "the eigensolver of this execution backend keeps one saved "
            "subspace and cannot serve the two channels of a spin-polarized "
            f"calculation ({type(eigenproblem_solver).__name__})"
        )
    return (for_channel(0), for_channel(1))


class SpinOrbitalBlocks:
    """Orbitals of two spin channels that an execution backend keeps apart.

    A backend may hand the SCF loop orbitals that live on a device.  Joining
    them there would hold every orbital twice, so the result keeps the two
    blocks as they are and joins them where they are wanted:
    :meth:`to_full_host` returns one host array with the columns of the up
    channel followed by those of the down channel.
    """

    def __init__(self, up, down) -> None:
        self.blocks = (up, down)

    @property
    def shape(self) -> tuple[int, int]:
        up, down = self.blocks
        return (int(up.shape[0]), int(up.shape[1]) + int(down.shape[1]))

    @property
    def ndim(self) -> int:
        return 2

    def to_full_host(self) -> np.ndarray:
        """Download both blocks into one host array, up first."""

        host = np.empty(self.shape, dtype=np.float64)
        column = 0
        for block in self.blocks:
            download = getattr(block, "get", None)
            values = download() if callable(download) else np.asarray(block)
            host[:, column : column + values.shape[1]] = values
            column += values.shape[1]
        return host

    def to_full_device(self):
        """Join the two blocks with the array library they belong to."""

        up, down = self.blocks
        library = sys.modules.get(type(up).__module__.split(".")[0])
        stack = getattr(library, "hstack", None)
        if stack is None:
            raise TypeError(f"cannot join orbital blocks of type {type(up).__name__}")
        return stack((up, down))


def _join_orbitals(up, down):
    """Return the orbitals of both channels and a view of each channel.

    Host arrays are joined column-wise, up first, and the channels are views
    of the joined array.  Other arrays stay apart in
    :class:`SpinOrbitalBlocks` and are the channels themselves.
    """

    if isinstance(up, np.ndarray) and isinstance(down, np.ndarray):
        joined = np.hstack((up, down))
        count = up.shape[1]
        return joined, (joined[:, :count], joined[:, count:])
    return SpinOrbitalBlocks(up, down), (up, down)


def run_spin_scf(
    system: PreparedSinglePointSystem,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
    eigenproblem_solver: Callable[..., object] | None = None,
    orbital_density_builder: Callable[..., np.ndarray] | None = None,
    mixer_factory: Callable[..., object] | None = None,
    residual_metrics_evaluator: Callable[..., object] | None = None,
    total_energy_evaluator: Callable[..., object] | None = None,
    scalar_field_adapter: object | None = None,
    warm_start: SCFWarmStart | None = None,
) -> SinglePointResult:
    """Run PARSEC-style potential mixing with two spin channels.

    The arguments are those of :func:`parsec_python.SCF.run_scf`, which
    dispatches here.  ``eigenproblem_solver`` and ``orbital_density_builder``
    are the execution hooks of the accelerated package.  The hooks of a
    symmetry-reduced backend, which keep scalar fields on one value per
    orbit, have no spin-polarized counterpart and are refused.

    The returned result holds the two channels in ``spin`` and describes
    them together in its other fields, as
    :class:`parsec_python.models.SinglePointResult` documents.
    """

    reduced_hooks = {
        "mixer_factory": mixer_factory,
        "residual_metrics_evaluator": residual_metrics_evaluator,
        "total_energy_evaluator": total_energy_evaluator,
        "scalar_field_adapter": scalar_field_adapter,
    }
    given = [name for name, hook in reduced_hooks.items() if hook is not None]
    if given:
        raise NotImplementedError(
            "a spin-polarized calculation keeps its scalar fields on the full "
            "grid; the symmetry-reduced SCF hooks are not available for it "
            f"({', '.join(given)})"
        )
    if system.initial_spin_densities is None and warm_start is None:
        raise ValueError(
            "the system was not prepared for a spin-polarized calculation: "
            "it has no starting spin densities"
        )

    scf_start = time.perf_counter()
    solvers = _channel_solvers(eigenproblem_solver)
    ionic_potential = system.ionic_potential
    volume_element = system.grid.volume_element
    number_of_states = _number_of_states(system)
    settings = system.input.scf
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method not in {"chebff", "chebdav"}:
        raise NotImplementedError(
            f"Eigensolver={eigensolver_settings.method!r} is not yet ported "
            "to the strict native-Python SCF path; no alternate solver will "
            "be substituted."
        )

    # 1. First potentials.  The Hartree potential is that of the normalized
    # total starting density; the exchange-correlation potentials are those
    # of the two starting channels, which initchrg.f90 builds so that they
    # add up to twice that density (see initial_spin_densities).
    eigval_states: list[object | None] = [None, None]
    if warm_start is None:
        stage_start = time.perf_counter()
        initial_hartree = system.solve_hartree(
            system.initial_density.copy(), initial_potential=-ionic_potential
        )
        initial_hartree_seconds = time.perf_counter() - stage_start
        hartree_potential = initial_hartree.potential
        stage_start = time.perf_counter()
        xc = system.evaluate_spin_xc(*system.initial_spin_densities)
        initial_xc_seconds = time.perf_counter() - stage_start
        input_potentials = [
            ionic_potential + hartree_potential + xc_potential
            for xc_potential in xc.potentials
        ]
    else:
        # After an ionic move: the mixed potentials of both channels with
        # the new local ionic potential, the last Hartree potential as the
        # start of the first Poisson solve, and the saved subspaces.
        initial_hartree_seconds = 0.0
        initial_xc_seconds = 0.0
        hartree_potential = np.array(warm_start.hartree_potential, dtype=np.float64)
        if len(warm_start.input_potential) != 2:
            raise ValueError(
                "a spin-polarized warm start needs the potentials of both channels"
            )
        input_potentials = [
            np.array(potential, dtype=np.float64)
            for potential in warm_start.input_potential
        ]
        states = warm_start.eigensolver_state
        eigval_states = [None, None] if states is None else list(states)

    mixer = AndersonMixer(system.input.mixing)
    history: list[SCFIteration] = []
    filter_degree = eigensolver_settings.filter_degree
    minimum_filter_degree = max(10, eigensolver_settings.filter_degree_delta + 1)
    converged = False
    energies = None
    grid_size = system.grid.size
    eigenvalues: list[np.ndarray] = [np.empty(0), np.empty(0)]
    occupations: list[np.ndarray] = [np.empty(0), np.empty(0)]
    representations: list[np.ndarray] = [
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
    ]
    wavefunctions: list[object] = [np.empty((grid_size, 0)), np.empty((grid_size, 0))]
    densities: list[np.ndarray] = [np.empty(0), np.empty(0)]
    electrons = (0.0, 0.0)
    fermi_level = np.nan
    last_input_potentials = [potential.copy() for potential in input_potentials]
    mixed_potentials = [potential.copy() for potential in input_potentials]
    xc_potentials = list(xc.potentials) if warm_start is None else [None, None]
    hamiltonian_binding_total = 0.0
    diagonalization_total = 0.0
    occupations_density_total = 0.0
    hartree_total = 0.0
    xc_total = 0.0
    mixing_energy_total = 0.0

    for iteration in range(1, settings.max_iterations + 1):
        iteration_start = time.perf_counter()
        last_input_potentials = [potential.copy() for potential in input_potentials]
        eigval_settings = eigval_settings_for(eigensolver_settings, filter_degree)

        # 2. One eigenproblem per channel, each with the potential and the
        # saved subspace of that channel.  A backend binds the potential to
        # one Hamiltonian allocation, so the channels are solved in turn.
        hamiltonian_binding_seconds = 0.0
        diagonalization_seconds = 0.0
        residual_maxima: list[float] = []
        for spin in range(2):
            binding_start = time.perf_counter()
            hamiltonian = system.hamiltonian(input_potentials[spin])
            operator_factory = getattr(
                hamiltonian,
                "as_eigensolver_operator",
                hamiltonian.as_linear_operator,
            )
            hamiltonian_operator = operator_factory()
            hamiltonian_binding_seconds += time.perf_counter() - binding_start
            diagonalization_start = time.perf_counter()
            eigensolution = solvers[spin](
                hamiltonian_operator,
                number_of_states,
                settings=eigval_settings,
                state=eigval_states[spin],
            )
            eigval_states[spin] = eigensolution.state
            diagonalization_seconds += time.perf_counter() - diagonalization_start
            eigenvalues[spin] = np.asarray(eigensolution.eigenvalues, dtype=np.float64)
            wavefunctions[spin] = eigensolution.vectors
            sector_labels = getattr(eigensolution, "representations", None)
            representations[spin] = (
                np.ones(eigenvalues[spin].size, dtype=np.int32)
                if sector_labels is None
                else np.asarray(sector_labels, dtype=np.int32) + 1
            )
            if eigensolution.residual_norms is not None:
                residual_maxima.append(
                    float(np.max(eigensolution.residual_norms, initial=0.0))
                )
        hamiltonian_binding_total += hamiltonian_binding_seconds
        diagonalization_total += diagonalization_seconds
        eigen_residual_max = (
            max(residual_maxima) if len(residual_maxima) == 2 else float("nan")
        )

        # 3. One Fermi level for both channels, then the density of each.
        occupations_density_start = time.perf_counter()
        occupation_result = (
            spin_fermi_occupations(
                eigenvalues[0],
                eigenvalues[1],
                system.electron_count,
                settings.fermi_temperature_kelvin,
            )
            if settings.fixed_occupations is None
            else spin_fixed_occupations(
                eigenvalues[0],
                eigenvalues[1],
                system.electron_count,
                settings.fixed_occupations,
                settings.fixed_occupations_down,
            )
        )
        occupations = list(occupation_result.occupations)
        electrons = occupation_result.electrons
        fermi_level = occupation_result.fermi_level
        for spin in range(2):
            if orbital_density_builder is None:
                densities[spin] = spin_density_from_orbitals(
                    wavefunctions[spin], occupations[spin], volume_element
                )
            else:
                # A backend builder forms 2/dV sum f |psi|^2, the density of
                # doubly occupied orbitals.  Half the occupation gives the
                # one electron per state of a spin channel, bit for bit.
                densities[spin] = np.asarray(
                    orbital_density_builder(
                        wavefunctions[spin], 0.5 * occupations[spin], volume_element
                    ),
                    dtype=np.float64,
                )
        density = densities[0] + densities[1]
        occupations_density_seconds = time.perf_counter() - occupations_density_start
        occupations_density_total += occupations_density_seconds

        # 4. Hartree potential of the total density and the two
        # exchange-correlation potentials.
        hartree_start = time.perf_counter()
        hartree = system.solve_hartree(density, initial_potential=hartree_potential)
        hartree_seconds = time.perf_counter() - hartree_start
        hartree_total += hartree_seconds
        hartree_potential = hartree.potential
        xc_start = time.perf_counter()
        xc = system.evaluate_spin_xc(densities[0], densities[1])
        xc_seconds = time.perf_counter() - xc_start
        xc_total += xc_seconds
        xc_potentials = list(xc.potentials)
        output_potentials = [
            ionic_potential + hartree_potential + xc_potential
            for xc_potential in xc_potentials
        ]

        # 5. Residuals and energies.  The eigenpairs belong to the input
        # potentials and the Hartree and XC terms to the new densities, as
        # in the unpolarized loop.
        mixing_energy_start = time.perf_counter()
        metrics = spin_potential_residual_metrics(
            input_potentials,
            output_potentials,
            densities,
            volume_element,
            electrons,
        )
        energies = spin_total_energy(
            eigenvalues,
            occupations,
            densities,
            input_potentials,
            ionic_potential,
            hartree_potential,
            xc_potentials,
            xc.total_energy,
            system.ion_ion_energy,
            volume_element,
        )

        # 6. Mix the two potentials as one vector, up first, and test the
        # larger residual of the two channels.
        mixed = mixer.mix(
            np.concatenate(input_potentials),
            np.concatenate(output_potentials),
            iteration=iteration,
        )
        mixed_potentials = [mixed[:grid_size].copy(), mixed[grid_size:].copy()]
        selected_residual = (
            metrics.plain if settings.use_plain_residual else metrics.weighted
        )
        if (
            iteration > 5
            and metrics.weighted < 100.0 * settings.convergence_criterion
            and filter_degree > minimum_filter_degree
        ):
            filter_degree -= 1
        iteration_converged = selected_residual < settings.convergence_criterion
        mixing_energy_seconds = time.perf_counter() - mixing_energy_start
        mixing_energy_total += mixing_energy_seconds

        diagnostic = SCFIteration(
            iteration=iteration,
            weighted_residual=metrics.weighted,
            plain_residual=metrics.plain,
            eigen_residual_max=eigen_residual_max,
            hartree_residual=hartree.residual_norm,
            energies=energies,
            eigenvalues=tuple(
                float(value) for values in eigenvalues for value in values
            ),
            occupations=tuple(
                float(value) for values in occupations for value in values
            ),
            representations=tuple(
                int(value) for values in representations for value in values
            ),
            fermi_level=float(fermi_level),
            density_minimum=float(np.min(density)),
            density_maximum=float(np.max(density)),
            diagonalization_seconds=diagonalization_seconds,
            hartree_seconds=hartree_seconds,
            hamiltonian_binding_seconds=hamiltonian_binding_seconds,
            occupations_density_seconds=occupations_density_seconds,
            xc_seconds=xc_seconds,
            mixing_energy_seconds=mixing_energy_seconds,
            total_seconds=time.perf_counter() - iteration_start,
            spin_plain_residuals=metrics.plain_by_spin,
            spin_weighted_residuals=metrics.weighted_by_spin,
            spin_electrons=(float(electrons[0]), float(electrons[1])),
            spin_density_minimum=tuple(float(np.min(values)) for values in densities),
            spin_density_maximum=tuple(float(np.max(values)) for values in densities),
        )
        history.append(diagnostic)
        if callback is not None:
            callback(diagnostic)

        if iteration_converged:
            converged = True
            break
        input_potentials = mixed_potentials

    if energies is None:
        raise RuntimeError("SCF loop did not execute")

    run_timings = RunTimings(
        preparation=system.timings,
        initial_hartree_seconds=initial_hartree_seconds,
        initial_xc_seconds=initial_xc_seconds,
        hamiltonian_binding_seconds=hamiltonian_binding_total,
        diagonalization_seconds=diagonalization_total,
        occupations_density_seconds=occupations_density_total,
        hartree_seconds=hartree_total,
        xc_seconds=xc_total,
        mixing_energy_seconds=mixing_energy_total,
        total_seconds=time.perf_counter() - scf_start,
    )
    output_potentials = [
        ionic_potential + hartree_potential + xc_potential
        for xc_potential in xc_potentials
    ]
    # The orbitals are stored once: host channels are views of the joined
    # array, device channels the blocks the backend returned.
    stacked, channel_orbitals = _join_orbitals(wavefunctions[0], wavefunctions[1])
    channels = tuple(
        SpinChannelResult(
            eigenvalues=eigenvalues[spin],
            occupations=occupations[spin],
            wavefunctions=channel_orbitals[spin],
            representations=representations[spin],
            density=densities[spin],
            xc_potential=xc_potentials[spin],
            input_effective_potential=last_input_potentials[spin],
            output_effective_potential=output_potentials[spin],
            next_effective_potential=mixed_potentials[spin],
            electron_count=float(electrons[spin]),
            eigensolver_state=eigval_states[spin],
        )
        for spin in range(2)
    )
    return SinglePointResult(
        converged=converged,
        iterations=len(history),
        atoms=system.atoms,
        electron_count=system.electron_count,
        grid=system.grid,
        pseudopotentials=system.pseudopotentials,
        density=densities[0] + densities[1],
        core_density=system.core_density,
        ionic_potential=system.ionic_potential,
        hartree_potential=hartree_potential,
        # Averages of the two channels, as forcloc.f90 forms them.
        xc_potential=0.5 * (xc_potentials[0] + xc_potentials[1]),
        input_effective_potential=0.5
        * (last_input_potentials[0] + last_input_potentials[1]),
        output_effective_potential=0.5 * (output_potentials[0] + output_potentials[1]),
        next_effective_potential=0.5 * (mixed_potentials[0] + mixed_potentials[1]),
        nonlocal_operator=system.nonlocal_operator,
        eigenvalues=np.concatenate(eigenvalues),
        occupations=np.concatenate(occupations),
        wavefunctions=stacked,
        fermi_level=fermi_level,
        energies=energies,
        history=history,
        timings=run_timings,
        representations=np.concatenate(representations),
        atomic_reference_correction=system.atomic_reference_correction,
        eigensolver_state=tuple(eigval_states),
        spin=channels,
    )


__all__ = ["SPIN_NAMES", "SpinOrbitalBlocks", "run_spin_scf"]
