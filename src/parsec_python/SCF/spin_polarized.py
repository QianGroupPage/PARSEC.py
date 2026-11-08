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

The module also holds :func:`run_scf_spin_polarized`, the collinear
spin-polarized loop for periodic Gamma-point cells and for the spin-orbit
drivers (see the section of that name below); an isolated calculation without
spin-orbit coupling runs :func:`run_spin_scf`.
"""

from __future__ import annotations

from dataclasses import replace
import sys
import time
from typing import Callable, Union

import numpy as np

from ..Eigensolvers import (
    EigvalSettings,
    EigvalState,
    dynamic_diagonalization_tolerance,
    solve_eigval,
)
from ..Energy import spin_total_energy, total_energy_spin_polarized
from ..Grid import symmetrize_density
from ..Mixer import (
    make_mixer,
    potential_residual_metrics,
    spin_potential_residual_metrics,
)
from ..Occupations import (
    density_from_orbitals,
    fermi_occupations,
    spin_density_from_orbitals,
    spin_fermi_occupations,
    spin_fixed_occupations,
)
from ..models import (
    RunTimings,
    SCFIteration,
    SinglePointResult,
    SpinChannelResult,
    SpinPolarizedSCFIteration,
    SpinPolarizedSinglePointResult,
)
from ..Hamiltonian.hubbard import hubbard_operator, hubbard_terms_from_orbitals
from .pbc import PeriodicPreparedSinglePointSystem
from .single_point import (
    PreparedSinglePointSystem,
    SCFWarmStart,
    _number_of_states,
    dispersion_after_convergence,
    dispersion_wanted,
    eigval_settings_for,
    first_diagonalization_tolerance,
    hubbard_iteration_record,
    hubbard_terms_of_iteration,
    iteration_eigensolver_fields,
    loop_criterion,
    meta_gga_loop,
    preconditioner_neighbors,
    refine_levels,
    refuse_exchange_hooks,
    require_reference_hooks,
    warm_start_mixer,
)


SPIN_NAMES = ("up", "down")
#: Levels up to this far (Ry) above the Fermi level enter the convergence of a
#: start spectrum that a reduced adapter asks to solve again (test 1 of
#: acceleration/SCF/spin_sectors.py: the spread bound of its multiplets).
START_LEVEL_WINDOW = 3.0e-2


def _spin_occupations(eigenvalues, system, settings):
    """One Fermi level for both channels (or the occupations of ``occup.in``)."""

    if settings.fixed_occupations is None:
        return spin_fermi_occupations(
            eigenvalues[0],
            eigenvalues[1],
            system.electron_count,
            settings.fermi_temperature_kelvin,
        )
    return spin_fixed_occupations(
        eigenvalues[0],
        eigenvalues[1],
        system.electron_count,
        settings.fixed_occupations,
        settings.fixed_occupations_down,
    )


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
        """Download both blocks into one host array, up first.

        A block that exports itself (``to_full_host``: the orbitals of the
        symmetry sectors of an accelerated loop) is asked to.
        """

        host = np.empty(self.shape, dtype=np.float64)
        column = 0
        for block in self.blocks:
            export = getattr(block, "to_full_host", None)
            download = getattr(block, "get", None)
            values = (
                export()
                if callable(export)
                else download()
                if callable(download)
                else np.asarray(block)
            )
            host[:, column : column + values.shape[1]] = values
            column += values.shape[1]
        return host

    def to_full_device(self):
        """Join the two blocks with the array library they belong to."""

        up, down = self.blocks
        if callable(getattr(up, "to_full_device", None)):
            # The orbitals of the symmetry sectors: each channel expands itself.
            up, down = up.to_full_device(), down.to_full_device()
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
    convergence_criterion: float | None = None,
    iteration_state_callback: Callable[[object], None] | None = None,
    exchange_term: object | None = None,
) -> SinglePointResult:
    """Run PARSEC-style potential mixing with two spin channels.

    The arguments are those of :func:`parsec_python.SCF.run_scf`, which
    dispatches here.  ``eigenproblem_solver`` and ``orbital_density_builder``
    are the execution hooks of the accelerated package.  The hooks of a
    symmetry-reduced backend, which keep scalar fields on one value per
    orbit, are accepted from an adapter made for two channels
    (``scalar_field_adapter.spin_hooks()``, the symmetry sectors of the
    accelerated driver with ``Spin_Symmetry_Sectors``): its mixer takes the
    two channels' orbit values joined, up first, and its residual norms and
    energy take the per-channel arguments of
    :func:`parsec_python.Mixer.spin_potential_residual_metrics` and
    :func:`parsec_python.Energy.spin_total_energy`.  Other reduced hooks are
    refused.  Such an adapter may also check the occupations of the result
    of a loop without an exchange operator (``check_spin_result``).

    The returned result holds the two channels in ``spin`` and describes
    them together in its other fields, as
    :class:`parsec_python.models.SinglePointResult` documents; its fields are
    full-grid arrays also where the loop kept them on the orbits.
    """

    reduced_hooks = {
        "mixer_factory": mixer_factory,
        "residual_metrics_evaluator": residual_metrics_evaluator,
        "total_energy_evaluator": total_energy_evaluator,
        "scalar_field_adapter": scalar_field_adapter,
    }
    given = [name for name, hook in reduced_hooks.items() if hook is not None]
    if given:
        spin_hooks = getattr(scalar_field_adapter, "spin_hooks", None)
        expected = spin_hooks() if callable(spin_hooks) else None
        if expected is None or any(reduced_hooks[name] != hook for name, hook in expected.items()):
            raise NotImplementedError(
                "a spin-polarized calculation keeps its scalar fields on the full "
                "grid; the symmetry-reduced SCF hooks are not available for it "
                "unless an adapter of two channels gives them "
                f"({', '.join(given)})"
            )
    reduced = scalar_field_adapter is not None
    to_internal = (lambda values: values) if not reduced else scalar_field_adapter.from_full
    to_public = (lambda values: values) if not reduced else scalar_field_adapter.to_full
    evaluate_residual_metrics = (
        spin_potential_residual_metrics if residual_metrics_evaluator is None else residual_metrics_evaluator
    )
    evaluate_total_energy = spin_total_energy if total_energy_evaluator is None else total_energy_evaluator
    build_mixer = make_mixer if mixer_factory is None else mixer_factory
    if system.initial_spin_densities is None and warm_start is None:
        raise ValueError(
            "the system was not prepared for a spin-polarized calculation: "
            "it has no starting spin densities"
        )
    hubbard_projectors = require_reference_hooks(
        system,
        eigenproblem_solver=eigenproblem_solver,
        orbital_density_builder=orbital_density_builder,
    )
    if exchange_term is not None and getattr(system, "exchange_device", None) is None:
        # A hybrid functional: the exchange operator of each channel joins
        # its Hamiltonian (parsec_python.SCF.hybrid); the reference path, or
        # an accelerated system whose device Hamiltonians hold it.
        refuse_exchange_hooks(
            eigenproblem_solver=eigenproblem_solver,
            orbital_density_builder=orbital_density_builder,
        )
    # LDA+U potential matrices of both channels, zero in the first iteration.
    hubbard_state = None
    # A meta-GGA: tau and vtau of each channel (SCF.meta_gga).
    meta = meta_gga_loop(
        system,
        channels=2,
        adapter=scalar_field_adapter,
        warm_start=warm_start,
        exchange_term=exchange_term,
    )

    scf_start = time.perf_counter()
    level_detail = None
    solvers = _channel_solvers(eigenproblem_solver)
    ionic_potential = to_internal(system.ionic_potential)
    volume_element = system.grid.volume_element
    number_of_states = _number_of_states(system)
    settings = system.input.scf
    if convergence_criterion is not None:
        settings = replace(settings, convergence_criterion=float(convergence_criterion))
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method == "none":
        raise ValueError(
            "Eigensolver: none is PARSEC's TEST mode: setup only, no SCF"
        )
    neighbors = preconditioner_neighbors(system, eigensolver_settings)
    diagonalization_tolerance = first_diagonalization_tolerance(eigensolver_settings)

    # 1. First potentials.  The Hartree potential is that of the normalized
    # total starting density; the exchange-correlation potentials are those
    # of the two starting channels, which initchrg.f90 builds so that they
    # add up to twice that density (see initial_spin_densities).
    eigval_states: list[object | None] = [None, None]
    if warm_start is None:
        stage_start = time.perf_counter()
        initial_hartree = system.solve_hartree(
            to_internal(system.initial_density.copy()), initial_potential=-ionic_potential
        )
        initial_hartree_seconds = time.perf_counter() - stage_start
        hartree_potential = initial_hartree.potential
        stage_start = time.perf_counter()
        start_densities = tuple(to_internal(density) for density in system.initial_spin_densities)
        xc = (
            system.evaluate_spin_xc(*start_densities)
            if meta is None
            else meta.start_spin_xc(*start_densities)
        )
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
        hartree_potential = to_internal(
            np.array(warm_start.hartree_potential, dtype=np.float64)
        )
        if len(warm_start.input_potential) != 2:
            raise ValueError(
                "a spin-polarized warm start needs the potentials of both channels"
            )
        input_potentials = [
            to_internal(np.array(potential, dtype=np.float64))
            for potential in warm_start.input_potential
        ]
        states = warm_start.eigensolver_state
        eigval_states = [None, None] if states is None else list(states)

    # The mixer a warm start carries (the Fock loop's Exx_Mixer_History), else a new one.
    mixer = warm_start_mixer(
        warm_start,
        (lambda: build_mixer(system.input.mixing))
        if meta is None
        else (lambda: meta.mixer(build_mixer, system.input.mixing)),
    )
    if reduced:
        # The orbit values of both channels, up first, and back.
        join_channels = scalar_field_adapter.join_channels
        split_channels = scalar_field_adapter.split_channels
    else:
        join_channels = np.concatenate

        def split_channels(mixed):
            return [mixed[:grid_size].copy(), mixed[grid_size:].copy()]

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

    # SCFWarmStart.level_tolerance (run_scf): one more iteration on the same V_in.
    level_tolerance = None if warm_start is None else getattr(warm_start, "level_tolerance", None)
    refining = False
    # Test 2 of a reduced loop (acceleration/SCF/spin_sectors.py): the
    # criterion is tightened while the asymmetry of a converged state looks
    # like SCF noise; the loop counts as converged once the given criterion
    # was met.
    criterion_scale = 1.0
    symmetry_tightenings = 0
    last_asymmetry = None
    met_criterion = False
    for iteration in range(1, settings.max_iterations + 1 + int(level_tolerance is not None)):
        if iteration > settings.max_iterations and not refining:
            break
        iteration_start = time.perf_counter()
        last_input_potentials = [potential.copy() for potential in input_potentials]
        eigval_settings = eigval_settings_for(
            eigensolver_settings,
            filter_degree,
            tolerance=diagonalization_tolerance,
            neighbors=neighbors,
        )

        # 2. One eigenproblem per channel, each with the potential and the
        # saved subspace of that channel.  A backend binds the potential to
        # one Hamiltonian allocation, so the channels are solved in turn.
        hamiltonian_binding_seconds = 0.0
        diagonalization_seconds = 0.0
        residual_maxima: list[float] = []
        if hubbard_projectors is not None and iteration > 1:
            # calcden.f90z: N and D of both channels from the orbitals and
            # occupations of the iteration before.
            hubbard_state = hubbard_terms_of_iteration(
                system, hubbard_projectors, wavefunctions, occupations
            )
        solver_counts: list[tuple[int, int] | None] = []
        for spin in range(2):
            binding_start = time.perf_counter()
            if meta is not None and meta.kinetic_potential(spin) is not None:
                # A meta-GGA hybrid: the orbital term and the fixed exchange operator.
                hamiltonian = system.hamiltonian(
                    input_potentials[spin],
                    None,
                    None if exchange_term is None else exchange_term.operator(spin),
                    kinetic_potential=meta.kinetic_potential(spin),
                    **meta.flux_keywords(spin),
                )
            elif hubbard_projectors is None and exchange_term is None:
                hamiltonian = system.hamiltonian(input_potentials[spin])
            elif hubbard_projectors is None:
                hamiltonian = system.hamiltonian(
                    input_potentials[spin], None, exchange_term.operator(spin)
                )
            else:
                hamiltonian = system.hamiltonian(
                    input_potentials[spin],
                    hubbard_operator(hubbard_projectors, hubbard_state, spin),
                )
            operator_factory = getattr(
                hamiltonian,
                "as_eigensolver_operator",
                hamiltonian.as_linear_operator,
            )
            hamiltonian_operator = operator_factory()
            hamiltonian_binding_seconds += time.perf_counter() - binding_start
            diagonalization_start = time.perf_counter()
            if reduced:
                # The sector views of the channel's last orbitals are no state
                # once its next solve begins (as in run_scf).
                release = getattr(wavefunctions[spin], "release_intermediate_storage", None)
                if callable(release):
                    release()
            if refining:
                # The loop converged at the iteration before (run_scf).
                eigensolution, passes, seconds, change = refine_levels(
                    solvers[spin],
                    hamiltonian_operator,
                    number_of_states,
                    eigval_settings,
                    eigval_states[spin],
                    eigenvalues[spin],
                    None,
                    float(level_tolerance),
                    release=reduced,
                )
                level_detail = {
                    "level_passes": float(passes) + (level_detail or {}).get("level_passes", 0.0),
                    "level_seconds": float(seconds) + (level_detail or {}).get("level_seconds", 0.0),
                    "level_change": max(float(change), (level_detail or {}).get("level_change", 0.0)),
                }
            else:
                eigensolution = solvers[spin](
                    hamiltonian_operator,
                    number_of_states,
                    settings=eigval_settings,
                    state=eigval_states[spin],
                )
            eigval_states[spin] = eigensolution.state
            solver_counts.append(getattr(eigensolution, "iterative_counts", None))
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
        occupation_result = _spin_occupations(eigenvalues, system, settings)
        occupations = list(occupation_result.occupations)
        electrons = occupation_result.electrons
        fermi_level = occupation_result.fermi_level
        start_report = getattr(scalar_field_adapter, "start_spectrum_report", None)
        if reduced and warm_start is None and iteration == 1 and callable(start_report):
            # The levels of the start Hamiltonian, which has the point group
            # of the atoms: a reduced adapter refuses a partly filled
            # multiplet here (acceleration/SCF/spin_sectors.py), once the
            # levels are converged; until then the start Hamiltonian is
            # solved again from its subspaces.
            passes = 0
            change = np.inf
            while True:
                report = start_report(eigenvalues, occupations, representations)
                if report is None:
                    break
                if passes and change <= scalar_field_adapter.start_level_change:
                    scalar_field_adapter.abandon_start(report, passes)
                if passes >= scalar_field_adapter.start_spectrum_passes or hubbard_projectors is not None:
                    # Still moving: tests 2 and 3 decide at the end.
                    break
                previous = [values.copy() for values in eigenvalues]
                window = fermi_level + START_LEVEL_WINDOW
                diagonalization_start = time.perf_counter()
                for spin in range(2):
                    if meta is not None and meta.kinetic_potential(spin) is not None:
                        hamiltonian = system.hamiltonian(
                            input_potentials[spin],
                            None,
                            None if exchange_term is None else exchange_term.operator(spin),
                            kinetic_potential=meta.kinetic_potential(spin),
                            **meta.flux_keywords(spin),
                        )
                    elif exchange_term is None:
                        hamiltonian = system.hamiltonian(input_potentials[spin])
                    else:
                        hamiltonian = system.hamiltonian(
                            input_potentials[spin], None, exchange_term.operator(spin)
                        )
                    operator_factory = getattr(
                        hamiltonian, "as_eigensolver_operator", hamiltonian.as_linear_operator
                    )
                    release = getattr(wavefunctions[spin], "release_intermediate_storage", None)
                    if callable(release):
                        release()
                    eigensolution = solvers[spin](
                        operator_factory(),
                        number_of_states,
                        settings=eigval_settings,
                        state=eigval_states[spin],
                    )
                    eigval_states[spin] = eigensolution.state
                    eigenvalues[spin] = np.asarray(eigensolution.eigenvalues, dtype=np.float64)
                    wavefunctions[spin] = eigensolution.vectors
                    sector_labels = getattr(eigensolution, "representations", None)
                    representations[spin] = (
                        np.ones(eigenvalues[spin].size, dtype=np.int32)
                        if sector_labels is None
                        else np.asarray(sector_labels, dtype=np.int32) + 1
                    )
                diagonalization_total += time.perf_counter() - diagonalization_start
                passes += 1
                change = max(
                    float(np.max(np.abs(new - old)[(old <= window) | (new <= window)], initial=0.0))
                    if new.shape == old.shape
                    else np.inf
                    for new, old in zip(eigenvalues, previous)
                )
                occupation_result = _spin_occupations(eigenvalues, system, settings)
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
                built = orbital_density_builder(
                    wavefunctions[spin], 0.5 * occupations[spin], volume_element
                )
                # (A reduced builder returns the field of the orbits.)
                densities[spin] = built if reduced else np.asarray(built, dtype=np.float64)
        # A meta-GGA: tau of each channel, one electron per state.
        taus = (
            None
            if meta is None
            else [meta.tau(wavefunctions[spin], occupations[spin], 1.0, spin) for spin in range(2)]
        )
        symmetrization = ()
        density_symmetry = getattr(system, "density_symmetry", None)
        if density_symmetry is not None and reduced:
            raise NotImplementedError(
                "Symmetrize_Charge_Density is not implemented for spin channels in the symmetry sectors"
            )
        if density_symmetry is not None:
            # Symmetrize_Charge_Density: symm_scalar of newrho.F90 for the
            # total density and for each channel (isp = 1, 2, 3).
            _, total_report = symmetrize_density(
                densities[0] + densities[1], density_symmetry, volume_element
            )
            reports = [total_report]
            for spin in range(2):
                densities[spin], report = symmetrize_density(
                    densities[spin], density_symmetry, volume_element
                )
                reports.append(report)
                if taus is not None:
                    taus[spin], _ = symmetrize_density(taus[spin], density_symmetry, volume_element)
                    meta.symmetrize_gradient(spin, density_symmetry)
            symmetrization = tuple(reports)
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
        if meta is None:
            xc = system.evaluate_spin_xc(densities[0], densities[1])
        else:
            xc = system.evaluate_spin_xc(
                densities[0], densities[1], tau_up=taus[0], tau_down=taus[1], **meta.spin_gradient_keywords()
            )
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
        metrics = evaluate_residual_metrics(
            input_potentials,
            output_potentials,
            densities,
            volume_element,
            electrons,
        )
        energies = evaluate_total_energy(
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
            **(
                {}
                if hubbard_projectors is None
                else {
                    "hubbard_energy": 0.0
                    if hubbard_state is None
                    else hubbard_state.energy
                }
            ),
            # The alpha term of a periodic cell (forces, relaxation and
            # molecular dynamics of a spin-polarized bulk cell run here).
            **(
                {"alpha_z_energy": system.alpha_z_energy}
                if getattr(system, "alpha_z_energy", 0.0)
                else {}
            ),
        )
        if meta is not None:
            energies = meta.energies(energies, taus, getattr(xc, "fermi_hole_clamped", 0), xc=xc)
        if exchange_term is not None:
            # (After the meta-GGA's double counting: a meta-GGA hybrid keeps the
            # hybrid breakdown of its inner loops.)
            energies = exchange_term.iteration_energies(
                energies, tuple(wavefunctions), tuple(occupations)
            )

        # 6. Mix the two potentials as one vector, up first, and test the
        # larger residual of the two channels.
        if meta is None:
            mixed = mixer.mix(
                join_channels(input_potentials),
                join_channels(output_potentials),
                iteration=iteration,
            )
            mixed_potentials = split_channels(mixed)
        else:
            # The vtau of both channels joins the vector (SCF.meta_gga).
            mixed_potentials = meta.mix(
                mixer,
                input_potentials,
                output_potentials,
                xc.kinetic_potentials,
                iteration=iteration,
                join=join_channels,
                split=split_channels,
                densities=densities,
            )
        selected_residual = (
            metrics.plain if settings.use_plain_residual else metrics.weighted
        )
        if (
            iteration > settings.degree_decrease_after
            and metrics.weighted < settings.approach_criterion
            and filter_degree > minimum_filter_degree
        ):
            filter_degree -= 1
        if iteration == 1:
            first_residual = selected_residual
        iteration_converged = selected_residual < criterion_scale * loop_criterion(
            warm_start, settings.convergence_criterion, first_residual
        )
        if eigensolver_settings.dynamic_tolerance:
            # The smaller weighted SRE of the two channels (minval(sre)).
            diagonalization_tolerance = dynamic_diagonalization_tolerance(
                diagonalization_tolerance,
                metrics.weighted_by_spin,
                eigensolver_settings.tolerance,
            )
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
            density_minimum=(
                scalar_field_adapter.minimum(density) if reduced else float(np.min(density))
            ),
            density_maximum=(
                scalar_field_adapter.maximum(density) if reduced else float(np.max(density))
            ),
            diagonalization_seconds=diagonalization_seconds,
            hartree_seconds=hartree_seconds,
            hamiltonian_binding_seconds=hamiltonian_binding_seconds,
            occupations_density_seconds=occupations_density_seconds,
            xc_seconds=xc_seconds,
            mixing_energy_seconds=mixing_energy_seconds,
            total_seconds=time.perf_counter() - iteration_start,
            **({"level_refinement": True} if refining else {}),
            spin_plain_residuals=metrics.plain_by_spin,
            spin_weighted_residuals=metrics.weighted_by_spin,
            spin_electrons=(float(electrons[0]), float(electrons[1])),
            spin_density_minimum=tuple(
                scalar_field_adapter.minimum(values) if reduced else float(np.min(values))
                for values in densities
            ),
            spin_density_maximum=tuple(
                scalar_field_adapter.maximum(values) if reduced else float(np.max(values))
                for values in densities
            ),
            density_symmetrization=symmetrization,
            **hubbard_iteration_record(hubbard_state),
            **iteration_eigensolver_fields(
                tuple(solver_counts),
                diagonalization_tolerance
                if eigensolver_settings.dynamic_tolerance
                else None,
            ),
        )
        history.append(diagnostic)
        if callback is not None:
            callback(diagnostic)
        if iteration_state_callback is not None:
            from .single_point import SCFIterationFields

            iteration_channels = tuple(
                SpinChannelResult(
                    eigenvalues=eigenvalues[spin],
                    occupations=occupations[spin],
                    wavefunctions=wavefunctions[spin],
                    representations=representations[spin],
                    density=to_public(densities[spin]),
                    xc_potential=to_public(xc_potentials[spin]),
                    input_effective_potential=to_public(last_input_potentials[spin]),
                    output_effective_potential=to_public(output_potentials[spin]),
                    next_effective_potential=to_public(mixed_potentials[spin]),
                    electron_count=float(electrons[spin]),
                )
                for spin in range(2)
            )
            iteration_state_callback(
                SCFIterationFields(
                    iteration=iteration,
                    eigenvalues=np.concatenate(eigenvalues),
                    occupations=np.concatenate(occupations),
                    representations=np.concatenate(representations),
                    wavefunctions=None,
                    density=to_public(density),
                    next_effective_potential=0.5
                    * (
                        to_public(mixed_potentials[0]) + to_public(mixed_potentials[1])
                    ),
                    spin=iteration_channels,
                )
            )

        measure_asymmetry = getattr(scalar_field_adapter, "density_symmetry_measure", None)
        if iteration_converged and not refining and reduced and exchange_term is None and callable(measure_asymmetry):
            met_criterion = True
            asymmetry = measure_asymmetry([to_public(values) for values in densities])
            log = getattr(scalar_field_adapter, "symmetry_measures", None)
            if isinstance(log, list):
                log.append((iteration, asymmetry, criterion_scale * settings.convergence_criterion))
            if (
                asymmetry is not None
                and scalar_field_adapter.density_symmetry_tolerance
                <= asymmetry
                < scalar_field_adapter.density_symmetry_noise_bound
                and symmetry_tightenings < scalar_field_adapter.density_symmetry_tightenings
                and (last_asymmetry is None or asymmetry <= 0.5 * last_asymmetry)
                and iteration < settings.max_iterations
            ):
                # Noise shrinks with the residual, a broken state keeps its
                # asymmetry: go on to a criterion ten times tighter.
                symmetry_tightenings += 1
                last_asymmetry = asymmetry
                criterion_scale *= 0.1
                input_potentials = mixed_potentials
                if meta is not None:
                    meta.advance()
                continue
        if iteration_converged or refining:
            if level_tolerance is not None and not refining:
                # V_in stays: the next iteration solves the same Hamiltonians.
                refining = True
                continue
            converged = True
            break
        input_potentials = mixed_potentials
        if meta is not None:
            meta.advance()

    if energies is None:
        raise RuntimeError("SCF loop did not execute")
    # A loop that met its criterion and then ran out of iterations while
    # tightening it (test 2 of a reduced loop) has converged.
    converged = converged or met_criterion
    output_potentials = [
        ionic_potential + hartree_potential + xc_potential
        for xc_potential in xc_potentials
    ]
    if reduced:
        # The result holds full-grid fields, as that of the full-grid loop.
        densities = [to_public(values) for values in densities]
        xc_potentials = [to_public(values) for values in xc_potentials]
        last_input_potentials = [to_public(values) for values in last_input_potentials]
        output_potentials = [to_public(values) for values in output_potentials]
        mixed_potentials = [to_public(values) for values in mixed_potentials]
        hartree_potential = to_public(hartree_potential)
    dispersion = None
    if dispersion_wanted(settings, converged):
        dispersion, energies = dispersion_after_convergence(
            system, densities[0] + densities[1], energies
        )

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
        **({} if level_detail is None else {"detail": level_detail}),
    )
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
    result = SinglePointResult(
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
        hubbard=hubbard_state,
        dispersion=dispersion,
        **({} if meta is None else {"meta_gga": meta.record()}),
    )
    check = getattr(scalar_field_adapter, "check_spin_result", None)
    if exchange_term is None and callable(check):
        # The symmetry guard of a reduced loop (the symmetry sectors of the
        # accelerated driver): a result that breaks the symmetry of the atoms
        # or has fractional occupations is not kept.
        check(result)
    return result


# ----------------------------------------------------------------------------
# run_scf_spin_polarized: collinear spin polarization for isolated and periodic
# (Gamma-point) single points, the base of the perturbative and self-consistent
# spin-orbit drivers.
#
# Collinear spin-polarized PARSEC-style SCF for isolated and periodic
# (Gamma-point) single points.
#
# This is the spin-polarized counterpart to :mod:`single_point` (and, for
# periodic inputs, :mod:`pbc`), following the same overall potential-mixing
# algorithm but iterating two independent Kohn--Sham eigenproblems (spin up,
# spin down) that share the grid, kinetic operator, KB nonlocal projectors,
# local ionic potential, and Hartree potential, and differ only in their
# exchange-correlation field:
#
# ``H_sigma[V_eff,sigma] = -nabla_FD^2 + diag(V_eff,sigma) + V_NL``,
# ``V_eff,sigma = V_ion,local + V_H[rho] + V_xc,sigma[rho_up,rho_down]``.
#
# Matching ``flevel.f90``, both channels' eigenvalues are pooled into *one*
# Fermi-level bisection with a shared chemical potential (electron count is
# fixed; the magnetic moment ``N_up-N_down`` is free), not two independently
# constrained channels.
#
# ``run_scf_spin_polarized`` accepts either
# :class:`~parsec_python.SCF.single_point.PreparedSinglePointSystem` (isolated)
# or :class:`~parsec_python.SCF.pbc.PeriodicPreparedSinglePointSystem`
# (periodic, Gamma-point only) unchanged: every attribute/method this module
# uses (``ionic_potential``, ``initial_density``, ``core_density``,
# ``solve_hartree``, ``hamiltonian``, ``grid.volume_element``, ``grid.size``,
# ``electron_count``, ``ion_ion_energy``, ``alpha_z_energy``, ``atoms``,
# ``input``) already has the same shape and meaning on both.  Neither
# :mod:`parsec_python.Eigensolvers.perturbative_soc` nor real periodic k-point
# sampling is included: only the Gamma-point real-arithmetic SCF loop is
# shared, matching :mod:`pbc`'s own scope.
#
# Unlike :func:`~parsec_python.SCF.single_point.run_scf`, this function does
# not yet carry PARSEC's full per-iteration timing/history instrumentation --
# see :class:`~parsec_python.models.SpinPolarizedSinglePointResult`.  It is
# wired into the CLI/Output path (:mod:`parsec_python.cli`,
# :class:`~parsec_python.Output.ParsecTextReporter`), and produces a converged
# collinear ground state (eigenvalues, occupations, and orbitals for both
# channels) that, for isolated inputs,
# :mod:`parsec_python.Eigensolvers.perturbative_soc` then uses for PARSEC's
# default perturbative spin-orbit correction.
# ----------------------------------------------------------------------------


_PreparedSystem = Union[PreparedSinglePointSystem, PeriodicPreparedSinglePointSystem]


def _weighted_initial_polarization(system: _PreparedSystem) -> float:
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


def _number_of_states_per_channel(system: _PreparedSystem) -> int:
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


def _eigval_settings(
    system: _PreparedSystem,
    filter_degree: int,
    *,
    tolerance: float | None = None,
    neighbors: np.ndarray | None = None,
) -> EigvalSettings:
    """The policy of :func:`eigval_settings_for` (the same for every solver)."""

    return eigval_settings_for(
        system.input.eigensolver,
        filter_degree,
        tolerance=tolerance,
        neighbors=neighbors,
    )


def run_scf_spin_polarized(
    system: _PreparedSystem,
    *,
    callback: Callable[[SpinPolarizedSCFIteration], None] | None = None,
    iteration_state_callback: Callable[[object], None] | None = None,
) -> SpinPolarizedSinglePointResult:
    """Run collinear spin-polarized PARSEC-style potential mixing.

    The XC functional is ``system.input.scf.xc_functional``: ``'ca'`` is
    LSDA (``exc_spn.f90``); ``'pbe'`` is collinear spin-polarized PBE
    (:func:`~parsec_python.V_xc.pbe_spin_polarized`).

    LDA+U (``system.hubbard``) enters as in :func:`run_spin_scf`: from the
    second iteration on, ``D`` of each channel from the orbitals of the
    iteration before (``calcden.f90z``) joins that channel's Hamiltonian and
    ``E_U`` the energy; the TS correction (``Van_Der_Waals``) is evaluated on
    the converged density.
    ``iteration_state_callback`` receives the fields of every finished
    iteration (``Save_Intermediate_Charge_Density`` /
    ``Save_Intermediate_Eigenvalues``, parsec.F90:1381-1402: PARSEC writes
    ``parsec.dat`` and ``eigen.dat`` after every SCF iteration); it observes
    only.
    """
    settings = system.input.scf
    if not settings.spin_polarized:
        raise ValueError(
            "run_scf_spin_polarized requires SCFSettings.spin_polarized=True"
        )
    if getattr(settings, "is_meta_gga", False):
        raise NotImplementedError(
            f"run_scf_spin_polarized has no meta-GGA term ({settings.xc_functional!r}): a meta-GGA runs on "
            "run_spin_scf, or for a periodic cell on the k-point spin loop at the Gamma point"
        )
    hubbard_projectors = getattr(system, "hubbard", None)
    hubbard_state = None

    number_of_states = _number_of_states_per_channel(system)
    ionic_potential = system.ionic_potential

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

    eigval_state_up: EigvalState | None = None
    eigval_state_down: EigvalState | None = None
    # One mixer history over the concatenated (V_up, V_down) vector, the
    # spin-polarized analog of mixing a single scalar field -- PARSEC treats
    # vnew(:,1:2) as one combined potential for mixing purposes.
    mixer = make_mixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method == "none":
        raise ValueError(
            "Eigensolver: none is PARSEC's TEST mode: setup only, no SCF"
        )
    neighbors = preconditioner_neighbors(system, eigensolver_settings)
    diagonalization_tolerance = first_diagonalization_tolerance(eigensolver_settings)

    eigenvalues_up = eigenvalues_down = np.empty(0)
    occupations_up = occupations_down = np.empty(0)
    wavefunctions_up = wavefunctions_down = np.empty((system.grid.size, 0))
    fermi_level = float("nan")
    energies = None
    converged = False

    for iteration in range(1, settings.max_iterations + 1):
        eigval_settings = _eigval_settings(
            system,
            filter_degree,
            tolerance=diagonalization_tolerance,
            neighbors=neighbors,
        )

        hubbard_terms_now = {}
        if hubbard_projectors is not None:
            if iteration > 1:
                # calcden.f90z: N and D of both channels from the orbitals and
                # occupations of the iteration before.
                hubbard_state = hubbard_terms_from_orbitals(
                    hubbard_projectors,
                    [wavefunctions_up, wavefunctions_down],
                    [occupations_up, occupations_down],
                )
            hubbard_terms_now = {
                "hubbard_energy": 0.0 if hubbard_state is None else hubbard_state.energy
            }
            hamiltonian_up = system.hamiltonian(
                input_potential_up, hubbard_operator(hubbard_projectors, hubbard_state, 0)
            )
        else:
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

        if hubbard_projectors is not None:
            hamiltonian_down = system.hamiltonian(
                input_potential_down,
                hubbard_operator(hubbard_projectors, hubbard_state, 1),
            )
        else:
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
        if settings.fixed_occupations is not None:
            # occup.in (a negative Fermi_Temp): each channel takes its own
            # section in order of energy, as run_spin_scf does; no Fermi level.
            fixed = _spin_occupations((eigenvalues_up, eigenvalues_down), system, settings)
            occupations_up, occupations_down = (
                np.asarray(values, dtype=np.float64) for values in fixed.occupations
            )
            fermi_level = fixed.fermi_level
        else:
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
        symmetrization = ()
        density_symmetry = getattr(system, "density_symmetry", None)
        if density_symmetry is not None:
            # Symmetrize_Charge_Density of a periodic cell: isp = 1 (total,
            # reported), 2 and 3 (parsec.F90 lines 1223-1232).
            volume_element = system.grid.volume_element
            _, total_report = symmetrize_density(
                density_up + density_down, density_symmetry, volume_element
            )
            density_up, up_report = symmetrize_density(
                density_up, density_symmetry, volume_element
            )
            density_down, down_report = symmetrize_density(
                density_down, density_symmetry, volume_element
            )
            symmetrization = (total_report, up_report, down_report)

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
            # totnrg.f90 adds etot_plusu, zero before the first calcden.
            **hubbard_terms_now,
        )

        mixed_combined = mixer.mix(input_combined, output_combined, iteration=iteration)
        input_potential_up = mixed_combined[: system.grid.size]
        input_potential_down = mixed_combined[system.grid.size :]

        selected_residual = (
            metrics.plain if settings.use_plain_residual else metrics.weighted
        )
        if eigensolver_settings.dynamic_tolerance:
            # This loop measures one weighted SRE over both channels.
            diagonalization_tolerance = dynamic_diagonalization_tolerance(
                diagonalization_tolerance,
                (metrics.weighted,),
                eigensolver_settings.tolerance,
            )
        if callback is not None:
            callback(
                SpinPolarizedSCFIteration(
                    iteration=iteration,
                    weighted_residual=metrics.weighted,
                    plain_residual=metrics.plain,
                    energies=energies,
                    eigenvalues_up=tuple(float(value) for value in eigenvalues_up),
                    eigenvalues_down=tuple(float(value) for value in eigenvalues_down),
                    occupations_up=tuple(float(value) for value in occupations_up),
                    occupations_down=tuple(float(value) for value in occupations_down),
                    fermi_level=float(fermi_level),
                    density_symmetrization=symmetrization,
                    **hubbard_iteration_record(hubbard_state),
                    # Dynamic_Diag_Tol: the tolerance of the next iteration.
                    **(
                        {"next_eigensolver_tolerance": diagonalization_tolerance}
                        if eigensolver_settings.dynamic_tolerance
                        else {}
                    ),
                )
            )
        if iteration_state_callback is not None:
            from .single_point import IterationChannelFields, SCFIterationFields

            iteration_state_callback(
                SCFIterationFields(
                    iteration=iteration,
                    eigenvalues=np.concatenate([eigenvalues_up, eigenvalues_down]),
                    occupations=np.concatenate([occupations_up, occupations_down]),
                    representations=np.empty(0, dtype=np.int32),
                    wavefunctions=None,
                    density=density_up + density_down,
                    next_effective_potential=0.5
                    * (input_potential_up + input_potential_down),
                    spin=(
                        IterationChannelFields(
                            eigenvalues_up,
                            occupations_up,
                            density_up,
                            input_potential_up,
                            wavefunctions=wavefunctions_up,
                        ),
                        IterationChannelFields(
                            eigenvalues_down,
                            occupations_down,
                            density_down,
                            input_potential_down,
                            wavefunctions=wavefunctions_down,
                        ),
                    ),
                )
            )
        if (
            iteration > settings.degree_decrease_after
            and metrics.weighted < settings.approach_criterion
            and filter_degree > minimum_filter_degree
        ):
            filter_degree -= 1
        if selected_residual < settings.convergence_criterion:
            converged = True
            break

    dispersion = None
    if dispersion_wanted(settings, converged):
        dispersion, energies = dispersion_after_convergence(
            system, density_up + density_down, energies
        )
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
        scf_potentials={
            "potential_up": output_potential_up,
            "potential_down": output_potential_down,
        },
        hubbard=hubbard_state,
        dispersion=dispersion,
    )


__all__ = [
    "SPIN_NAMES",
    "SpinOrbitalBlocks",
    "run_scf_spin_polarized",
    "run_spin_scf",
]
