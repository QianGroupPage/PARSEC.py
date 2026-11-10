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

Equal k-point weights (an unreduced uniform Monkhorst-Pack grid, see
:func:`~parsec_python.Grid.kpoints.monkhorst_pack_grid`) pool every level
into the scalar-``degeneracy``
:func:`~parsec_python.Occupations.fermi_dirac.fermi_occupations`
(``degeneracy = 2*weight``) and give ``Energy.total_energy`` the same
``band_energy_weight``, as this module always did.  Unequal weights
(Fortran's irreducible wedge,
:func:`~parsec_python.Grid.kpoints.irreducible_monkhorst_pack`, or a weighted
``Kpoint_List``) weight every level by its k-point (``flevel.f90``,
``newrho.F90``, ``totnrg.f90``; :mod:`~parsec_python.SCF.kpoint_weights`).

With ``Symmetrize_Charge_Density`` the prepared system carries the
space-group permutations of its grid (``density_symmetry``) and every new
density is averaged over them before the Hartree potential (``parsec.F90``
lines 1223-1232), which the density of an irreducible wedge needs.

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
from ..Occupations import fixed_occupations, spin_fixed_occupations
from ..Grid.point_group import symmetrize_density
from ..Hamiltonian.kpoint_operator import KPointExchangeHamiltonian, KPointKohnShamHamiltonian
from ..Mixer import make_mixer, potential_residual_metrics, preconditioned_mixer
from ..SCF.pbc import PeriodicPreparedSinglePointSystem
from ..SCF.spin_polarized import (
    _number_of_states_per_channel,
    _weighted_initial_polarization,
)
from ..Forces.nonlocal_force import bloch_projector_gradients, nonlocal_forces
from ..V_ion import build_nonlocal_projectors
from .kpoint_hubbard import KPointHubbard, kpoint_backend
from .single_point import (
    LEVEL_PASSES,
    IterationChannelFields,
    SCFIterationFields,
    dispersion_after_convergence,
    dispersion_wanted,
    first_diagonalization_tolerance,
    hubbard_iteration_record,
    loop_criterion,
    next_diagonalization_tolerance,
    warm_start_mixer,
)
from .kpoint_solver import (
    KPointJob,
    kept_vectors,
    make_kpoint_solver,
    split_pooled,
    whole_orbitals,
)
from .kpoint_weights import validated_kpoint_weights
from .meta_gga_periodic import bloch_orbital_fields, kpoint_meta_gga_loop, symmetrized_gradient, symmetrized_tau
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


def _run_timings(system, timer: StageTimer, solver, started: float, levels: dict | None = None) -> RunTimings:
    """Stage table of a k-point SCF: the driver's stages plus what the solver measured inside.

    ``levels``: the level passes of a loop with ``level_tolerance`` (``level_passes``, ``level_seconds``,
    ``level_change``), added to the detail.
    """
    seconds = timer.seconds
    detail = {"solver_setup": seconds.get("solver_setup", 0.0)}
    if levels:
        detail.update({name: float(value) for name, value in levels.items()})
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


def _check_fixed_occupations(settings, n_kpoints: int) -> None:
    """Refuse the occupations of ``occup.in`` (a negative ``Fermi_Temp``) with more than one k-point.

    ``occup.in`` lists one occupation per state in order of energy; with one
    k-point (the Gamma point as a 1x1x1 mesh) the loops give them as the
    cluster and Gamma-point loops do, with several there is no such order.
    """

    if settings.fixed_occupations is not None and n_kpoints != 1:
        raise NotImplementedError(
            "the occupations of occup.in (a negative Fermi_Temp) are defined for one k-point; "
            f"this k-point loop has {n_kpoints}"
        )


def _refined_levels(solver, potentials: dict, filter_degree: int, tolerance, eigenvalues_by_task, level_tolerance: float):
    """More solves of every task with the same potentials, until no level moves by more than ``level_tolerance`` (Ry).

    The k-point counterpart of :func:`parsec_python.SCF.single_point.refine_levels` (``LEVEL_PASSES`` at most).
    Returns the eigenvalues by task and ``{"level_passes", "level_seconds", "level_change"}``.
    """

    started = time.perf_counter()
    previous = [np.asarray(values, dtype=np.float64) for values in eigenvalues_by_task]
    passes = 0
    change = float("inf")
    while passes < LEVEL_PASSES:
        current = [np.asarray(values, dtype=np.float64) for values in solver.solve(potentials, filter_degree, tolerance)]
        passes += 1
        change = max(
            (
                float(np.max(np.abs(new - old), initial=0.0)) if new.shape == old.shape else float("inf")
                for new, old in zip(current, previous)
            ),
            default=0.0,
        )
        previous = current
        if change < level_tolerance:
            break
    return previous, {
        "level_passes": float(passes),
        "level_seconds": float(time.perf_counter() - started),
        "level_change": float(change),
    }


def run_scf_kpoints(
    system: PeriodicPreparedSinglePointSystem,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
    initial_potential: np.ndarray | None = None,
    keep_wavefunctions: bool = False,
    iteration_state_callback: Callable[[SCFIterationFields], None] | None = None,
    compute_forces: bool = False,
    warm_start=None,
    convergence_criterion: float | None = None,
    exchange_term=None,
    mixer_factory: Callable | None = None,
    keep_states: bool = False,
    keep_vectors: Callable | None = None,
    backend: str | None = None,
    projector_cache: dict | None = None,
    meta_gga_start=None,
) -> SinglePointResult:
    """Run k-point-sampled periodic SCF on an already-prepared system.

    ``weights`` are the normalized k-point weights, equal or not (see the
    module docstring).  ``initial_potential`` is
    the local potential of the first iteration (Ry; ``Restart_Run``, see
    :mod:`parsec_python.SCF.restart`, or the warm start of an ionic step)
    in place of the one of the initial density.  ``keep_wavefunctions``
    returns the final Bloch factors of every k-point and the input potential
    of the last iteration (``k_wavefunctions``, ``scf_input_potential``), which
    the Ylm-projected DOS and the forces of
    :mod:`parsec_python.Forces.periodic` need; nothing else changes.
    ``iteration_state_callback`` receives the fields of every finished
    iteration (``Save_Intermediate_Charge_Density`` /
    ``Save_Intermediate_Eigenvalues``, parsec.F90:1381-1402: PARSEC writes
    ``parsec.dat`` and ``eigen.dat`` after every SCF iteration); it observes
    only.

    ``warm_start`` (:class:`~parsec_python.SCF.single_point.SCFWarmStart`) continues from the loop of the previous
    structure of a relaxation or a dynamics run: its input potential, the starting vector of the first Poisson solve and,
    with the atomic extrapolation, the Hartree and exchange-correlation potentials of the guessed density.  The k-point
    eigenproblems start cold (the saved eigensubspaces of the previous structure are not carried over).
    ``convergence_criterion`` replaces the ``Convergence_Criterion`` of the input for this loop (Ry).

    ``compute_forces`` also returns the nonlocal force (``result.nonlocal_forces``): the eigenvectors of a k-point
    run are not kept, so after the last SCF step each k-point's contribution
    ``4 w_k sum_n f_nk sum_q s_q Re[conj(<beta_q|u_nk>) <grad beta_q|u_nk>]`` is computed from them and summed.
    (``warm_start``, ``convergence_criterion`` and ``compute_forces`` are Andrew Grieder's ionic loops;
    ``initial_potential`` and ``keep_wavefunctions`` those of the Fortran-derived routines.)

    The inner loops of the Fock loop of a hybrid functional
    (:mod:`parsec_python.SCF.hybrid_periodic`) add: ``exchange_term``, the fixed
    exact-exchange operator of the loop (``exchange_term.operator(0, k)`` joins the
    Hamiltonian of k-point ``k``, :class:`~parsec_python.Hamiltonian.kpoint_operator.KPointExchangeHamiltonian`;
    ``exchange_term.task_cross`` gives each k-point's share of ``E_x(P, P0)`` where its orbitals are, and
    ``exchange_term.energies_from_cross`` the energies of the iteration); the warm start of
    :class:`~parsec_python.SCF.single_point.SCFWarmStart` with ``eigensolver_state`` (the Ritz states by k-point,
    imported into the solver), ``mixer``, ``relative_criterion`` and ``level_tolerance`` (one more iteration on the
    same ``V_in`` after convergence, its eigenproblems solved until no level moves by more than that);
    ``mixer_factory`` (the mixer of a loop without a carried one); ``keep_states`` (the Ritz states of the last
    iteration in ``result.eigensolver_state``); ``keep_vectors(solver, task)`` (what ``k_wavefunctions`` holds,
    host copies by default); ``backend`` (the k-point backend, :func:`~parsec_python.SCF.kpoint_solver.make_kpoint_solver`,
    ``"cpu"`` for the host exchange operator; ``None`` the default); ``projector_cache`` (a dict by k-point
    index that keeps the Bloch-phase projectors for the next loop of the same structure).  Without them the loop is the one above.

    A meta-GGA (``SCFSettings.is_meta_gga``) runs the loop of :mod:`parsec_python.SCF.meta_gga`: the k-point sum
    ``tau = sum_k w_k (1/h^3) sum_n f_nk |(G + ik) u_nk|^2`` is formed where the orbitals are (one more field of the
    job's ``accumulate``), each Bloch Hamiltonian holds ``(G + ik)^H diag(vtau_in) (G + ik)``, and ``(V, vtau)`` are
    mixed together; ``meta_gga_start`` (an object with ``kinetic_potential`` and, in the orbital form of the density
    gradient, ``gradient_flux``, such as the :class:`~parsec_python.SCF.single_point.SCFWarmStart` of a move) holds the
    ``vtau`` and ``W`` that go with ``initial_potential`` (the ionic loop of the Fortran-derived routines;
    ``warm_start`` otherwise).
    """
    settings = system.input.scf
    weights = np.asarray(weights, dtype=float)
    k_points, weighting = validated_kpoint_weights(k_points, weights, "run_scf_kpoints")
    n_kpoints = k_points.shape[0]
    _check_fixed_occupations(settings, n_kpoints)
    density_symmetry = getattr(system, "density_symmetry", None)
    # LDA+U with Bloch projectors (SCF/kpoint_hubbard.py); None without it.
    hubbard = KPointHubbard.for_system(system, k_points)
    # A meta-GGA: tau of the Bloch factors, vtau in every Hamiltonian (SCF/meta_gga_periodic.py).
    meta = kpoint_meta_gga_loop(
        system, channels=1, warm_start=warm_start if meta_gga_start is None else meta_gga_start
    )

    number_of_states = _number_of_states(system)
    ionic_potential = system.ionic_potential
    lattice_vectors = system.input.periodic_cell.lattice_vectors
    cell_atoms = system.cell_atoms  # the atoms folded into the cell: the image sums assume them inside
    criterion = settings.convergence_criterion if convergence_criterion is None else float(convergence_criterion)

    started = time.perf_counter()
    timer = StageTimer()
    density = system.initial_density
    if initial_potential is None and warm_start is None:
        with timer.stage("initial_hartree"):
            initial_hartree = system.solve_hartree(density, initial_potential=-ionic_potential)
        hartree_potential = initial_hartree.potential
        with timer.stage("initial_xc"):
            xc = system.evaluate_xc(density) if meta is None else meta.start_xc(density)
        input_potential = ionic_potential + hartree_potential + xc.potential
    elif warm_start is None:
        # restart_run.F90: the stored potential, no Hartree or XC evaluation
        # before the first eigenproblem; -V_ion starts the first Poisson solve.
        hartree_potential = -np.asarray(ionic_potential, dtype=np.float64)
        input_potential = np.array(initial_potential, dtype=np.float64)
        if input_potential.shape != np.shape(ionic_potential):
            raise ValueError("initial_potential must hold one value per grid point")
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
    # (in the worker that owns it) and reused by every SCF step; a caller's
    # ``projector_cache`` (by k-point index) keeps them for its next loops of the
    # same structure (the loops of a hybrid's Fock loop).
    projector_cache: dict[int, object] = {} if projector_cache is None else projector_cache
    # sum_c k_c G_c of each k-point, built where the task is solved and only if read (not by a device kernel)
    k_gradients: dict[int, dict] = {}
    # the k-point solver, once made: the MPI k-point pools build a k-point's projectors with the atoms dealt
    # over the ranks of the pool (solver.projector_builder, the same matrix)
    owner: dict = {}

    def make_hamiltonian(task, potentials):
        if task not in projector_cache:
            projector_cache[task] = getattr(owner.get("solver"), "projector_builder", build_nonlocal_projectors)(
                system.grid,
                cell_atoms,
                system.pseudopotentials,
                system.input.pseudopotentials,
                lattice_vectors,
                k_point=k_points[task],
                periodic_dimensions=system.input.periodic_cell.dimensions,
            )
        base = KPointKohnShamHamiltonian(
            system.negative_laplacian,
            system.gradient,
            potentials["V"],
            projector_cache[task],
            k_points[task],
            kinetic_potential=potentials.get("vtau"),
            gradient_flux=potentials.get("gradient_flux"),
            k_gradient_cache=k_gradients.setdefault(task, {}),
        )
        if exchange_term is not None:
            # A hybrid functional: a_x V_ACE of this k-point (built where the worker solves it).
            operator = exchange_term.operator(0, task)
            return base if operator is None else KPointExchangeHamiltonian(base, operator)
        if hubbard is None:
            return base
        return hubbard.hamiltonian(base, task, potentials.get("hubbard"))

    def accumulate(task, vectors, occupations, xp=np):
        # rho(r) = sum_k w_k*(2/dV)*sum_n f_nk*|u_nk(r)|**2.  u_nk is complex
        # even though the resulting density is real.
        parts = {
            "density": (weighting.of_k(task) * 2.0 / system.grid.volume_element)
            * weighted_square_sum(vectors, occupations, xp)
        }
        if exchange_term is not None:
            # (s/2) w_k sum_n f_nk <u_nk|V_ACE|u_nk>: this k-point's share of E_x(P, P0).
            parts["exchange_cross"] = exchange_term.task_cross(
                0, task, vectors, occupations, weighting.of_k(task), xp
            )
        if meta is not None:
            # w_k (1/h^3) sum_n f_nk |(G + ik) u_nk|^2 (and g of the orbital form), where the orbitals are.
            fields = bloch_orbital_fields(
                system, k_points[task], vectors, occupations, weighting.of_k(task), 2.0, xp,
                with_gradient=meta.orbital_gradient,
            )
            parts["tau"] = fields["tau"]
            if meta.orbital_gradient:
                parts["tau_gradient"] = fields["gradient"]
        if hubbard is not None:
            # calcden.f90z with the Bloch phases: w_k sum_n f_nk <phi|u><u|phi>,
            # where the worker keeps the orbitals (xp).
            parts["hubbard"] = hubbard.occupation(
                task, vectors, occupations, weighting.of_k(task), xp=xp
            )
        return parts

    # Occupations of the last SCF step by k-point, set before the force pass (the job's ``extras`` sees only vectors).
    final_occupations: dict[int, np.ndarray] = {}

    def kpoint_nonlocal_force(task, vectors, xp=np):
        if task not in projector_cache:
            make_hamiltonian(task, {"V": ionic_potential})
        # (on the confined grid of a wire or slab: the images of its periodic directions, as the projectors)
        gradients = bloch_projector_gradients(
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
                xp,
                projector_cache[task],
                gradients,
                vectors,
                weighting.of_k(task) * final_occupations[task],
                len(cell_atoms),
            )
        return nonlocal_forces(
            projector_cache[task],
            gradients,
            np.asarray(vectors),
            weighting.of_k(task) * final_occupations[task],
            len(cell_atoms),
        )

    with timer.stage("solver_setup"):
        solver = make_kpoint_solver(
            KPointJob(
                n_kpoints,
                number_of_states,
                lambda degree, tolerance=None: _eigval_settings(system, degree, tolerance),
                make_hamiltonian,
                accumulate,
                extras=kpoint_nonlocal_force if compute_forces else None,
            ),
            problem_size=system.grid.size,
            backend=kpoint_backend(system) if backend is None else backend,
        )
    owner["solver"] = solver
    if compute_forces and type(solver).__name__ == "MPIKPointSolver":
        raise NotImplementedError(
            "the nonlocal force of the row-distributed MPI k-point solver is not implemented: its extras are "
            "sums over row blocks, which does not apply to a product of overlaps"
        )
    build_mixer = mixer_factory or make_mixer
    # Mixing_Preconditioner: kerker wraps the mixer (Mixer.preconditioner).
    mixer = warm_start_mixer(
        warm_start,
        (lambda: preconditioned_mixer(build_mixer(system.input.mixing), system))
        if meta is None
        else (lambda: meta.mixer(build_mixer, system.input.mixing)),
    )
    # A Fock loop's inner loop: the Ritz states of the loop before, by k-point.
    carried_states = getattr(warm_start, "eigensolver_state", None)
    level_tolerance = getattr(warm_start, "level_tolerance", None)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)
    # Dynamic_Diag_Tol: 0.1 for the first iteration (parsec.F90:1106-1111).
    dynamic_tolerance = system.input.eigensolver.dynamic_tolerance
    diagonalization_tolerance = first_diagonalization_tolerance(system.input.eigensolver)

    nonlocal_force = None
    pooled_eigenvalues = np.empty(0)
    pooled_occupations = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False
    history: list[SCFIteration] = []
    hubbard_state = None
    hubbard_occupations = None
    refining = False
    levels: dict = {}
    first_residual = None

    with solver:
        if isinstance(carried_states, dict) and carried_states:
            solver.import_states(carried_states)
        timer.lap()  # what came before the loop (potentials, solver start-up) is not part of iteration 1
        for iteration in range(1, settings.max_iterations + 1 + int(level_tolerance is not None)):
            if iteration > settings.max_iterations and not refining:
                break
            eigval_settings = _eigval_settings(system, filter_degree)

            solve_potentials = {"V": input_potential}
            if meta is not None and meta.kinetic_potential(0) is not None:
                solve_potentials["vtau"] = meta.kinetic_potential(0)
                solve_potentials.update(meta.flux_keywords(0))
            if hubbard is not None:
                if iteration > 1:
                    # calcden.f90z: D from the orbitals of the iteration before.
                    hubbard_state = hubbard.terms([hubbard_occupations])
                if hubbard_state is not None:
                    solve_potentials["hubbard"] = hubbard_state.potentials[0]
            with timer.stage("diagonalization"):
                if refining:
                    # The loop converged at the iteration before: the same H[V_in] until no level moves by more
                    # than level_tolerance (SCFWarmStart.level_tolerance).
                    eigenvalues_by_k, levels = _refined_levels(
                        solver,
                        solve_potentials,
                        filter_degree,
                        diagonalization_tolerance if dynamic_tolerance else None,
                        eigenvalues_by_k,
                        float(level_tolerance),
                    )
                else:
                    eigenvalues_by_k = solver.solve(
                        solve_potentials,
                        filter_degree,
                        diagonalization_tolerance if dynamic_tolerance else None,
                    )
            residual_max = solver_residual_max(solver)

            # Pool every k-point's eigenvalues into one shared Fermi-level
            # bisection (flevel.f90): every pooled entry is one (k,n) level
            # holding up to 2*w_k electrons after summing over the full BZ
            # (degeneracy=2*weight at equal weights).
            with timer.stage("density"):
                stacked = np.concatenate(eigenvalues_by_k)
                state_weights = (
                    None
                    if weighting.uniform
                    else weighting.state_weights(eigenvalues_by_k, range(n_kpoints))
                )
                if settings.fixed_occupations is not None:
                    # occup.in at the one k-point of the loop (_check_fixed_occupations)
                    fixed = fixed_occupations(stacked, system.electron_count, settings.fixed_occupations)
                    stacked_occupations, fermi_level = np.asarray(fixed.occupations, dtype=float), fixed.fermi_level
                else:
                    stacked_occupations, fermi_level = weighting.occupy(
                        stacked,
                        system.electron_count,
                        settings.fermi_temperature_kelvin,
                        2.0,
                        state_weights,
                    )
                pooled_eigenvalues = stacked
                pooled_occupations = stacked_occupations

                accumulated = solver.accumulate(
                    split_pooled(stacked_occupations, eigenvalues_by_k)
                )
                density = accumulated["density"]
                tau = None if meta is None else symmetrized_tau(
                    accumulated["tau"], density_symmetry, system.grid.volume_element
                )
                if meta is not None and meta.orbital_gradient:
                    meta.gradients[0] = tuple(
                        symmetrized_gradient(accumulated["tau_gradient"], density_symmetry, lattice_vectors)
                    )
                if hubbard is not None:
                    hubbard_occupations = accumulated["hubbard"]
                symmetrization = ()
                if density_symmetry is not None:
                    # Symmetrize_Charge_Density: symm_scalar after newrho (on the device of a solver that
                    # offers it, the same average bit for bit: the MPI k-point pools).
                    density, report = getattr(solver, "density_symmetrizer", symmetrize_density)(
                        density, density_symmetry, system.grid.volume_element
                    )
                    symmetrization = (report,)

            with timer.stage("hartree"):
                hartree = system.solve_hartree(density, initial_potential=hartree_potential)
            hartree_potential = hartree.potential
            with timer.stage("xc"):
                xc = (
                    system.evaluate_xc(density)
                    if meta is None
                    else system.evaluate_xc(density, tau=tau, **meta.gradient_keywords(0))
                )
            output_potential = ionic_potential + hartree_potential + xc.potential

            with timer.stage("mixing_energy"):
                metrics = potential_residual_metrics(
                    input_potential,
                    output_potential,
                    density,
                    system.grid.volume_element,
                    system.electron_count,
                )
                energy_occupations, band_energy_weight = weighting.band_energy_terms(
                    pooled_occupations, state_weights
                )
                energies = total_energy(
                    pooled_eigenvalues,
                    energy_occupations,
                    density,
                    input_potential,
                    ionic_potential,
                    hartree_potential,
                    xc.potential,
                    xc.total_energy,
                    system.ion_ion_energy,
                    system.grid.volume_element,
                    alpha_z_energy=system.alpha_z_energy,
                    band_energy_weight=band_energy_weight,
                    # totnrg.f90 adds etot_plusu, zero before the first calcden.
                    **(
                        {}
                        if hubbard is None
                        else {
                            "hubbard_energy": 0.0
                            if hubbard_state is None
                            else hubbard_state.energy
                        }
                    ),
                )
                if meta is not None:
                    # The band energy holds 2 h^3 sum vtau_in tau of the orbital term.
                    energies = meta.energies(energies, (tau,), getattr(xc, "fermi_hole_clamped", 0), xc=xc)
                if exchange_term is not None:
                    # The operator's part of the band energy removed, the linearized exact exchange added.
                    energies = exchange_term.energies_from_cross(
                        energies, float(np.sum(accumulated["exchange_cross"]))
                    )

                if meta is not None:
                    mixed_potential = meta.mix(
                        mixer,
                        [input_potential],
                        [output_potential],
                        [xc.kinetic_potential],
                        iteration=iteration,
                        densities=[density],
                    )[0]
                else:
                    mixed_potential = mixer.mix(
                        input_potential, output_potential, iteration=iteration
                    )
            lap = timer.lap()
            selected_residual = (
                metrics.plain if settings.use_plain_residual else metrics.weighted
            )
            if dynamic_tolerance:
                # parsec.F90:1302-1311 after getsre.
                diagonalization_tolerance = next_diagonalization_tolerance(
                    system.input.eigensolver, diagonalization_tolerance, (metrics.weighted,)
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
                density_symmetrization=symmetrization,
                **({"level_refinement": True} if refining else {}),
                **hubbard_iteration_record(hubbard_state),
                **(
                    {"next_eigensolver_tolerance": diagonalization_tolerance}
                    if dynamic_tolerance
                    else {}
                ),
            )
            history.append(iteration_record)
            if callback is not None:
                callback(iteration_record)
            if iteration_state_callback is not None:
                iteration_state_callback(
                    SCFIterationFields(
                        iteration=iteration,
                        eigenvalues=pooled_eigenvalues,
                        occupations=pooled_occupations,
                        representations=np.empty(0, dtype=np.int32),
                        wavefunctions=None,
                        density=density,
                        next_effective_potential=mixed_potential,
                        k_points=k_points,
                        k_weights=weights,
                    )
                )
            last_input_potential = input_potential
            if refining:
                converged = True
                break
            if iteration == 1:
                first_residual = selected_residual
            if level_tolerance is not None and selected_residual < loop_criterion(
                warm_start, criterion, first_residual
            ):
                # V_in stays: the next iteration solves the same Hamiltonian (level passes).
                refining = True
                continue
            input_potential = mixed_potential
            if (
                iteration > settings.degree_decrease_after
                and metrics.weighted
                < (
                    # A loop criterion of the ionic loop (Relaxation_SCF_Upscale) gives
                    # 100 times that criterion, unless the input writes
                    # Convergence_Criterion_Approach, which holds as in every other loop.
                    settings.approach_criterion
                    if convergence_criterion is None or settings.convergence_criterion_approach is not None
                    else 100.0 * criterion
                )
                and filter_degree > minimum_filter_degree
            ):
                filter_degree -= 1
            if selected_residual < (
                criterion if warm_start is None else loop_criterion(warm_start, criterion, first_residual)
            ):
                converged = True
                break
            if meta is not None:
                meta.advance()
        if compute_forces:
            # the eigenvectors and occupations of the last step belong together
            final_occupations.update(enumerate(split_pooled(pooled_occupations, eigenvalues_by_k)))
            nonlocal_force = np.sum(solver.extras(), axis=0)
        timings = _run_timings(system, timer, solver, started, levels)
        kept = (
            {
                "k_wavefunctions": tuple(
                    kept_vectors(solver, task, "the orbitals of every k-point (keep_wavefunctions)")
                    if keep_vectors is None
                    else keep_vectors(solver, task)
                    for task in range(n_kpoints)
                ),
                "scf_input_potential": last_input_potential,
            }
            if keep_wavefunctions
            else {}
        )
        if keep_states:
            kept["eigensolver_state"] = solver.export_states()

    dispersion = None
    if dispersion_wanted(settings, converged):
        # efvdw.f90 after a converged loop (the k-summed valence density).
        dispersion, energies = dispersion_after_convergence(system, density, energies)
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
        hubbard=hubbard_state,
        dispersion=dispersion,
        **kept,
        nonlocal_forces=nonlocal_force,
        **({} if meta is None else {"meta_gga": meta.record()}),
    )


def run_scf_kpoints_spin_polarized(
    system: PeriodicPreparedSinglePointSystem,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    callback: Callable[[SpinPolarizedSCFIteration], None] | None = None,
    keep_wavefunctions: bool = False,
    initial_potentials: tuple[np.ndarray, np.ndarray] | None = None,
    iteration_state_callback: Callable[[SCFIterationFields], None] | None = None,
    warm_start=None,
    convergence_criterion: float | None = None,
    exchange_term=None,
    mixer_factory: Callable | None = None,
    keep_states: bool = False,
    keep_vectors: Callable | None = None,
    backend: str | None = None,
    projector_cache: dict | None = None,
    initial_spin_densities: tuple | None = None,
    meta_gga_start=None,
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

    Collinear spin polarization keeps time reversal within each channel (a
    real potential), so Fortran reduces such a grid to the irreducible wedge
    as well (``irrbz.f90`` skips ``k ~ -k`` only for non-collinear
    magnetism); unequal weights weight every level by its k-point.
    ``States_Num`` is a per-channel, per-k-point count.  The returned
    eigenvalues and occupations are pooled over k-points (k-point-major
    within each spin); at equal weights each carries
    ``result.occupation_weight`` (= the k-point weight) in any sum such as
    the electron count, and ``magnetic_moment`` uses ``k_weights``.
    Wavefunctions are not retained, unless ``keep_wavefunctions``
    (``result.k_wavefunctions[spin][k]``, with the input, mixed,
    exchange-correlation and Hartree potentials of the last iteration in
    ``scf_input_potentials``, ``next_potentials``, ``xc_potentials`` and
    ``scf_hartree_potential``: what the forces of
    :mod:`parsec_python.Forces.periodic` and
    :mod:`parsec_python.Forces.wire_slab` and the next ionic step need).
    ``initial_potentials`` are the up and down potentials of the first
    iteration (the warm start of an ionic step), in place of those of the
    initial densities.  With ``Symmetrize_Charge_Density``
    both channel densities are averaged over the space group (``isp = 2, 3``
    of ``parsec.F90``; the report of the total density is ``isp = 1``).
    ``iteration_state_callback`` receives the fields of every finished
    iteration (``Save_Intermediate_Charge_Density`` /
    ``Save_Intermediate_Eigenvalues``, parsec.F90:1381-1402: PARSEC writes
    ``parsec.dat`` and ``eigen.dat`` after every SCF iteration); it observes
    only.

    The inner loops of the Fock loop of a hybrid functional
    (:mod:`parsec_python.SCF.hybrid_periodic`) add what :func:`run_scf_kpoints` takes for them: ``warm_start``
    (``input_potential`` the (up, down) pair, ``hartree_potential``, ``eigensolver_state`` by task, ``mixer``,
    ``relative_criterion``, ``level_tolerance``), ``convergence_criterion``, ``exchange_term`` (the operator of
    channel ``s`` at k-point ``k`` is ``exchange_term.operator(s, k)``), ``mixer_factory``, ``keep_states`` and
    ``keep_vectors``; the iteration records are kept in ``result.history``.  ``backend`` is that of
    :func:`run_scf_kpoints`.  ``initial_spin_densities`` (up, down) replaces the start of one polarization
    fraction of the whole cell (the Fock loop of a hybrid passes the per-species densities of the Gamma loop).

    A meta-GGA runs as in :func:`run_scf_kpoints`, with ``tau`` and ``vtau``
    per channel (one electron per state); ``meta_gga_start`` holds the two
    ``vtau`` (and ``W``) that go with ``initial_potentials`` (``warm_start``
    otherwise).
    """
    settings = system.input.scf
    if not settings.spin_polarized:
        raise ValueError(
            "run_scf_kpoints_spin_polarized requires SCFSettings.spin_polarized=True"
        )
    weights = np.asarray(weights, dtype=float)
    k_points, weighting = validated_kpoint_weights(
        k_points, weights, "run_scf_kpoints_spin_polarized"
    )
    n_kpoints = k_points.shape[0]
    _check_fixed_occupations(settings, n_kpoints)
    weight = weighting.weight
    density_symmetry = getattr(system, "density_symmetry", None)
    n_grid = system.grid.size
    # LDA+U with Bloch projectors (SCF/kpoint_hubbard.py); None without it.
    hubbard = KPointHubbard.for_system(system, k_points)
    # A meta-GGA: tau and vtau of each channel (SCF/meta_gga_periodic.py).
    meta = kpoint_meta_gga_loop(
        system, channels=2, warm_start=warm_start if meta_gga_start is None else meta_gga_start
    )

    number_of_states = _number_of_states_per_channel(system)
    ionic_potential = system.ionic_potential
    lattice_vectors = system.input.periodic_cell.lattice_vectors
    started = time.perf_counter()
    timer = StageTimer()
    if initial_spin_densities is not None:
        # A caller's start (the Fock loop of a hybrid: the per-species densities of initchrg.f90 that the
        # Gamma loop of run_spin_scf starts from, SCF.pbc.with_initial_spin_densities).
        density_up, density_down = (np.asarray(value, dtype=np.float64) for value in initial_spin_densities)
    else:
        polarization = _weighted_initial_polarization(system)
        density_up = 0.5 * (1.0 + polarization) * system.initial_density
        density_down = 0.5 * (1.0 - polarization) * system.initial_density
    criterion = settings.convergence_criterion if convergence_criterion is None else float(convergence_criterion)
    if warm_start is not None:
        if getattr(warm_start, "density_guess", None) is not None:
            raise NotImplementedError("the spin-polarized k-point loop takes no extrapolated density")
        # The loop before supplies the (up, down) input potentials and the start of the first Poisson solve.
        hartree_potential = np.array(warm_start.hartree_potential, dtype=np.float64)
        input_potential_up, input_potential_down = (
            np.array(potential, dtype=np.float64) for potential in warm_start.input_potential
        )
    elif initial_potentials is None:
        with timer.stage("initial_hartree"):
            initial_hartree = system.solve_hartree(
                density_up + density_down, initial_potential=-ionic_potential
            )
        hartree_potential = initial_hartree.potential
        with timer.stage("initial_xc"):
            xc = (
                system.evaluate_xc_spin_polarized(density_up, density_down)
                if meta is None
                else meta.start_spin_xc(density_up, density_down)
            )
        input_potential_up = ionic_potential + hartree_potential + xc.potential_up
        input_potential_down = ionic_potential + hartree_potential + xc.potential_down
    else:
        # As run_scf_kpoints with initial_potential: no Hartree or XC
        # evaluation before the first eigenproblem.
        hartree_potential = -np.asarray(ionic_potential, dtype=np.float64)
        input_potential_up, input_potential_down = (
            np.array(potential, dtype=np.float64) for potential in initial_potentials
        )
        if input_potential_up.shape != np.shape(ionic_potential) or (
            input_potential_down.shape != np.shape(ionic_potential)
        ):
            raise ValueError("initial_potentials must hold one value per grid point")

    # Tasks are (spin, k) pairs, task = spin*n_kpoints + k; each worker builds the
    # Bloch-phase projectors of the k-points it owns once (a caller's projector_cache, by
    # k-point index, keeps them for the next loops of the same structure).
    projector_cache: dict[int, object] = {} if projector_cache is None else projector_cache
    # sum_c k_c G_c of each k-point, shared by its two channels (built only if read)
    k_gradients: dict[int, dict] = {}
    owner: dict = {}  # the k-point solver, once made (see run_scf_kpoints)

    def make_hamiltonian(task, potentials):
        spin_index, k_index = divmod(task, n_kpoints)
        if k_index not in projector_cache:
            projector_cache[k_index] = getattr(owner.get("solver"), "projector_builder", build_nonlocal_projectors)(
                system.grid,
                system.atoms,
                system.pseudopotentials,
                system.input.pseudopotentials,
                lattice_vectors,
                k_point=k_points[k_index],
                periodic_dimensions=system.input.periodic_cell.dimensions,
            )
        base = KPointKohnShamHamiltonian(
            system.negative_laplacian,
            system.gradient,
            potentials["up" if spin_index == 0 else "down"],
            projector_cache[k_index],
            k_points[k_index],
            kinetic_potential=potentials.get("vtau_up" if spin_index == 0 else "vtau_down"),
            gradient_flux=potentials.get("gradient_flux_up" if spin_index == 0 else "gradient_flux_down"),
            k_gradient_cache=k_gradients.setdefault(k_index, {}),
        )
        if exchange_term is not None:
            # A hybrid functional: a_x V_ACE of this channel and k-point.
            operator = exchange_term.operator(spin_index, k_index)
            return base if operator is None else KPointExchangeHamiltonian(base, operator)
        if hubbard is None:
            return base
        matrices = potentials.get("hubbard")
        return hubbard.hamiltonian(
            base, k_index, None if matrices is None else matrices[spin_index]
        )

    def accumulate(task, vectors, occupations, xp=np):
        spin_index, k_index = divmod(task, n_kpoints)
        name = "up" if spin_index == 0 else "down"
        parts = {
            name: (
                weighting.of_k(k_index) / system.grid.volume_element
            )
            * weighted_square_sum(vectors, occupations, xp)
        }
        if meta is not None:
            fields = bloch_orbital_fields(
                system, k_points[k_index], vectors, occupations, weighting.of_k(k_index), 1.0, xp,
                with_gradient=meta.orbital_gradient,
            )
            parts["tau_" + name] = fields["tau"]
            if meta.orbital_gradient:
                parts["tau_gradient_" + name] = fields["gradient"]
        if hubbard is not None:
            parts["hubbard_" + name] = hubbard.occupation(
                k_index, vectors, occupations, weighting.of_k(k_index), xp=xp
            )
        if exchange_term is not None:
            # (1/2) w_k sum_n f_nk <u_nk|V_ACE|u_nk> of this channel: its share of E_x(P, P0).
            parts["exchange_cross"] = exchange_term.task_cross(
                spin_index, k_index, vectors, occupations, weighting.of_k(k_index), xp
            )
        return parts

    with timer.stage("solver_setup"):
        solver = make_kpoint_solver(
            KPointJob(
                2 * n_kpoints,
                number_of_states,
                lambda degree, tolerance=None: _eigval_settings(system, degree, tolerance),
                make_hamiltonian,
                accumulate,
            ),
            problem_size=system.grid.size,
            backend=kpoint_backend(system) if backend is None else backend,
        )
    owner["solver"] = solver
    build_mixer = mixer_factory or make_mixer
    # Mixing_Preconditioner: kerker filters the charge part of the joined
    # channels' residual (Mixer.preconditioner).
    mixer = warm_start_mixer(
        warm_start,
        (lambda: preconditioned_mixer(build_mixer(system.input.mixing), system, channels=2))
        if meta is None
        else (lambda: meta.mixer(build_mixer, system.input.mixing)),
    )
    carried_states = getattr(warm_start, "eigensolver_state", None)
    level_tolerance = getattr(warm_start, "level_tolerance", None)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)
    # Dynamic_Diag_Tol: 0.1 for the first iteration (parsec.F90:1106-1111).
    dynamic_tolerance = system.input.eigensolver.dynamic_tolerance
    diagonalization_tolerance = first_diagonalization_tolerance(system.input.eigensolver)

    eigenvalues_up = eigenvalues_down = np.empty(0)
    occupations_up = occupations_down = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False
    hubbard_state = None
    hubbard_occupations = None
    history: list = []
    refining = False
    levels: dict = {}
    first_residual = None
    by_task = None

    with solver:
        if isinstance(carried_states, dict) and carried_states:
            solver.import_states(carried_states)
        timer.lap()  # what came before the loop (potentials, solver start-up) is not part of iteration 1
        for iteration in range(1, settings.max_iterations + 1 + int(level_tolerance is not None)):
            if iteration > settings.max_iterations and not refining:
                break
            eigval_settings = _eigval_settings(system, filter_degree)

            solve_potentials = {"up": input_potential_up, "down": input_potential_down}
            if meta is not None and meta.kinetic_potential(0) is not None:
                solve_potentials["vtau_up"] = meta.kinetic_potential(0)
                solve_potentials["vtau_down"] = meta.kinetic_potential(1)
                for spin_index, name in enumerate(("up", "down")):
                    for key, value in meta.flux_keywords(spin_index).items():
                        solve_potentials[key + "_" + name] = value
            if hubbard is not None:
                if iteration > 1:
                    # calcden.f90z: D of both channels from the iteration before.
                    hubbard_state = hubbard.terms(hubbard_occupations)
                if hubbard_state is not None:
                    solve_potentials["hubbard"] = hubbard_state.potentials
            with timer.stage("diagonalization"):
                if refining:
                    # The loop converged at the iteration before: the same H[V_in] until the levels settle.
                    by_task, levels = _refined_levels(
                        solver,
                        solve_potentials,
                        filter_degree,
                        diagonalization_tolerance if dynamic_tolerance else None,
                        by_task,
                        float(level_tolerance),
                    )
                else:
                    by_task = solver.solve(
                        solve_potentials,
                        filter_degree,
                        diagonalization_tolerance if dynamic_tolerance else None,
                    )
            residual_max = solver_residual_max(solver)
            eigenvalues_by_spin_k = [by_task[:n_kpoints], by_task[n_kpoints:]]

            with timer.stage("density"):
                eigenvalues_up = np.concatenate(eigenvalues_by_spin_k[0])
                eigenvalues_down = np.concatenate(eigenvalues_by_spin_k[1])
                pooled = np.concatenate([eigenvalues_up, eigenvalues_down])
                state_weights = (
                    None
                    if weighting.uniform
                    else weighting.state_weights(
                        by_task, [task % n_kpoints for task in range(2 * n_kpoints)]
                    )
                )
                if settings.fixed_occupations is not None:
                    # occup.in at the one k-point of the loop: each channel its own section in order of energy
                    fixed = spin_fixed_occupations(
                        eigenvalues_up,
                        eigenvalues_down,
                        system.electron_count,
                        settings.fixed_occupations,
                        settings.fixed_occupations_down,
                    )
                    pooled_occupations = np.concatenate(
                        [np.asarray(values, dtype=float) for values in fixed.occupations]
                    )
                    fermi_level = fixed.fermi_level
                else:
                    pooled_occupations, fermi_level = weighting.occupy(
                        pooled,
                        system.electron_count,
                        settings.fermi_temperature_kelvin,
                        1.0,
                        state_weights,
                    )
                occupations_up = pooled_occupations[: eigenvalues_up.size]
                occupations_down = pooled_occupations[eigenvalues_up.size :]

                occupations_by_task = split_pooled(
                    occupations_up, eigenvalues_by_spin_k[0]
                ) + split_pooled(occupations_down, eigenvalues_by_spin_k[1])
                parts = solver.accumulate(occupations_by_task)
                density_up = parts.get("up", np.zeros(n_grid))
                density_down = parts.get("down", np.zeros(n_grid))
                taus = None
                if meta is not None:
                    taus = [
                        symmetrized_tau(
                            parts.get("tau_" + name, np.zeros(n_grid)), density_symmetry, system.grid.volume_element
                        )
                        for name in ("up", "down")
                    ]
                    if meta.orbital_gradient:
                        for spin_index, name in enumerate(("up", "down")):
                            meta.gradients[spin_index] = tuple(
                                symmetrized_gradient(
                                    parts.get("tau_gradient_" + name, np.zeros((3, n_grid))),
                                    density_symmetry,
                                    lattice_vectors,
                                )
                            )
                if hubbard is not None:
                    hubbard_occupations = [
                        parts.get(
                            "hubbard_" + name,
                            np.zeros((hubbard.template.atom_count, 5, 5), dtype=complex),
                        )
                        for name in ("up", "down")
                    ]
                symmetrization = ()
                if density_symmetry is not None:
                    # Symmetrize_Charge_Density: isp = 1 (total, reported), 2, 3.
                    volume_element = system.grid.volume_element
                    symmetrize = getattr(solver, "density_symmetrizer", symmetrize_density)
                    _, total_report = symmetrize(
                        density_up + density_down, density_symmetry, volume_element
                    )
                    density_up, up_report = symmetrize(
                        density_up, density_symmetry, volume_element
                    )
                    density_down, down_report = symmetrize(
                        density_down, density_symmetry, volume_element
                    )
                    symmetrization = (total_report, up_report, down_report)

            with timer.stage("hartree"):
                hartree = system.solve_hartree(
                    density_up + density_down, initial_potential=hartree_potential
                )
            hartree_potential = hartree.potential
            with timer.stage("xc"):
                xc = (
                    system.evaluate_xc_spin_polarized(density_up, density_down)
                    if meta is None
                    else system.evaluate_spin_xc(
                        density_up, density_down, tau_up=taus[0], tau_down=taus[1], **meta.spin_gradient_keywords()
                    )
                )
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
                energy_occupations, band_energy_weight = weighting.band_energy_terms(
                    pooled_occupations, state_weights
                )
                energies = total_energy_no_degeneracy_spin_polarized(
                    pooled,
                    energy_occupations,
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
                    band_energy_weight=band_energy_weight,
                    **(
                        {}
                        if hubbard is None
                        else {
                            "hubbard_energy": 0.0
                            if hubbard_state is None
                            else hubbard_state.energy
                        }
                    ),
                )
                if meta is not None:
                    energies = meta.energies(energies, taus, getattr(xc, "fermi_hole_clamped", 0), xc=xc)
                if exchange_term is not None:
                    energies = exchange_term.energies_from_cross(
                        energies, float(np.sum(parts["exchange_cross"]))
                    )

                if meta is not None:
                    mixed_combined = np.concatenate(
                        meta.mix(
                            mixer,
                            [input_potential_up, input_potential_down],
                            [output_potential_up, output_potential_down],
                            xc.kinetic_potentials,
                            iteration=iteration,
                            join=np.concatenate,
                            split=lambda values: [values[:n_grid], values[n_grid:]],
                            densities=[density_up, density_down],
                        )
                    )
                else:
                    mixed_combined = mixer.mix(
                        input_combined, output_combined, iteration=iteration
                    )
            lap = timer.lap()
            selected_residual = (
                metrics.plain if settings.use_plain_residual else metrics.weighted
            )
            if dynamic_tolerance:
                # parsec.F90:1302-1311 with this loop's one (combined) SRE.
                diagonalization_tolerance = next_diagonalization_tolerance(
                    system.input.eigensolver, diagonalization_tolerance, (metrics.weighted,)
                )
            if callback is not None or exchange_term is not None or warm_start is not None:
                first_k = eigenvalues_by_spin_k[0][0].size
                record = SpinPolarizedSCFIteration(
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
                    density_symmetrization=symmetrization,
                    **hubbard_iteration_record(hubbard_state),
                    **(
                        {"next_eigensolver_tolerance": diagonalization_tolerance}
                        if dynamic_tolerance
                        else {}
                    ),
                    **({"level_refinement": True} if refining else {}),
                )
                history.append(record)
                if callback is not None:
                    callback(record)
            if iteration_state_callback is not None:
                iteration_state_callback(
                    SCFIterationFields(
                        iteration=iteration,
                        eigenvalues=pooled,
                        occupations=pooled_occupations,
                        representations=np.empty(0, dtype=np.int32),
                        wavefunctions=None,
                        density=density_up + density_down,
                        next_effective_potential=0.5
                        * (mixed_combined[:n_grid] + mixed_combined[n_grid:]),
                        spin=(
                            IterationChannelFields(
                                eigenvalues_up,
                                occupations_up,
                                density_up,
                                mixed_combined[:n_grid],
                            ),
                            IterationChannelFields(
                                eigenvalues_down,
                                occupations_down,
                                density_down,
                                mixed_combined[n_grid:],
                            ),
                        ),
                        k_points=k_points,
                        k_weights=weights,
                    )
                )
            last_inputs = (input_potential_up, input_potential_down)
            if refining:
                # As run_scf: the mixed potentials of the level iteration seed what follows.
                input_potential_up = mixed_combined[:n_grid]
                input_potential_down = mixed_combined[n_grid:]
                converged = True
                break
            if iteration == 1:
                first_residual = selected_residual
            if level_tolerance is not None and selected_residual < loop_criterion(
                warm_start, criterion, first_residual
            ):
                # V_in stays: the next iteration solves the same Hamiltonians (level passes).
                refining = True
                continue
            input_potential_up = mixed_combined[:n_grid]
            input_potential_down = mixed_combined[n_grid:]
            if (
                iteration > settings.degree_decrease_after
                and metrics.weighted < settings.approach_criterion
                and filter_degree > minimum_filter_degree
            ):
                filter_degree -= 1
            if selected_residual < (
                settings.convergence_criterion
                if warm_start is None and convergence_criterion is None
                else loop_criterion(warm_start, criterion, first_residual)
            ):
                converged = True
                break
            if meta is not None:
                meta.advance()
        timings = _run_timings(system, timer, solver, started, levels)
        k_wavefunctions = (
            tuple(
                tuple(
                    kept_vectors(
                        solver,
                        spin * n_kpoints + k,
                        "the orbitals of every k-point (keep_wavefunctions)",
                    )
                    if keep_vectors is None
                    else keep_vectors(solver, spin * n_kpoints + k)
                    for k in range(n_kpoints)
                )
                for spin in range(2)
            )
            if keep_wavefunctions
            else None
        )
        states = solver.export_states() if keep_states else None
    # What the forces (Forces.periodic, Forces.wire_slab) and the next ionic
    # step read: the last iteration's input potentials (of the kept orbitals),
    # the mixed potentials, the exchange-correlation and Hartree potentials.
    kept = (
        {
            "scf_input_potentials": last_inputs,
            "next_potentials": (input_potential_up, input_potential_down),
            "xc_potentials": (xc.potential_up, xc.potential_down),
            "scf_hartree_potential": hartree_potential,
        }
        if keep_wavefunctions
        else {}
    )

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
        k_wavefunctions=k_wavefunctions,
        hubbard=hubbard_state,
        dispersion=dispersion,
        **kept,
        history=history,
        eigensolver_state=states,
        **({} if meta is None else {"meta_gga": meta.record()}),
    )


__all__ = ["run_scf_kpoints", "run_scf_kpoints_spin_polarized"]
