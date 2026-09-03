"""Isolated, spin-unpolarized PARSEC-style SCF preparation and iteration.

``prepare_single_point`` constructs all density-independent objects.  The
nonlinear work happens in ``run_scf``:

``V_in -> eigensolver -> occupations -> rho -> (V_H,V_xc) -> V_out``

``V_in,V_out -> residual/energy/mixer -> next V_in``.

The first eigensolver call constructs a buffered subspace with the selected
CHEBFF or CHEBDAV algorithm; later calls reuse it through one SUBSPACE filter.
SCF convergence is tested on the local potential residual, not on an
eigenpair or energy residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np
import scipy.sparse as sp

from ..Eigensolvers import (
    ChebDavSettings,
    ChebFFSettings,
    EigvalSettings,
    EigvalState,
    SubspaceSettings,
    solve_eigval,
)
from ..Energy import total_energy
from ..Grid import RealSpaceGrid, build_cluster_grid
from ..Hamiltonian import KohnShamHamiltonian
from ..Hartree import HartreeResult, solve_hartree
from ..Laplacian import build_negative_laplacian
from ..Mixer import AndersonMixer, potential_residual_metrics
from ..Occupations import density_from_orbitals, fermi_occupations
from ..Pseudopotential import ParsecPseudopotential
from ..V_ion import (
    NonlocalProjectorOperator,
    build_local_ionic_potential,
    build_nonlocal_projectors,
    center_cluster_geometry,
    ion_ion_energy,
    ionic_charge,
    load_pseudopotentials,
    normalize_density,
    superpose_atomic_density,
)
from ..models import (
    PreparationTimings,
    RunTimings,
    SCFIteration,
    SinglePointInput,
    SinglePointResult,
)
from ..V_xc import XCResult, ca_lda, pbe


@dataclass(frozen=True)
class PreparedSinglePointSystem:
    """Static terms that can be prepared and inspected without running SCF.

    The grid kinetic operator ``-nabla_FD^2``, local ionic potential,
    nonlocal KB projectors, frozen NLCC density, and ion--ion energy do not
    change when only the electronic density is iterated.  Hartree and XC
    potentials are therefore absent here: they are rebuilt from the current
    density inside :func:`run_scf`.
    """

    input: SinglePointInput
    atoms: tuple
    electron_count: float
    pseudopotentials: dict[str, ParsecPseudopotential]
    grid: RealSpaceGrid
    negative_laplacian: sp.csr_matrix
    ionic_potential: np.ndarray
    nonlocal_operator: NonlocalProjectorOperator
    initial_density: np.ndarray
    core_density: np.ndarray
    ion_ion_energy: float
    atomic_reference_correction: float = 0.0
    timings: PreparationTimings = field(default_factory=PreparationTimings)

    def hamiltonian(self, effective_potential: np.ndarray) -> KohnShamHamiltonian:
        """Compose the current SCF Hamiltonian without forming a dense matrix.

        The supplied array is only the diagonal local field

        ``V_eff = V_ion,local + V_H + V_xc``.

        :class:`KohnShamHamiltonian` binds it to the already prepared kinetic
        and nonlocal terms, giving

        ``H = -nabla_FD^2 + diag(V_eff) + V_NL``.
        """
        return KohnShamHamiltonian(
            self.negative_laplacian,
            effective_potential,
            self.nonlocal_operator,
        )

    def solve_hartree(
        self,
        density: np.ndarray,
        initial_potential: np.ndarray | None = None,
        *,
        raise_on_nonconvergence: bool = True,
    ) -> HartreeResult:
        """Solve ``(-nabla_FD^2)V_H=8*pi*rho`` with isolated boundaries.

        ``initial_potential`` is only the iterative CG starting vector.  The
        density source and multipole/direct Dirichlet boundary determine the
        physical solution independently of that warm start.
        """
        return solve_hartree(
            density,
            self.grid,
            self.negative_laplacian,
            self.input.hartree,
            initial_potential,
            raise_on_nonconvergence=raise_on_nonconvergence,
        )

    def evaluate_xc(self, density: np.ndarray) -> XCResult:
        """Evaluate the selected XC functional including frozen NLCC."""

        if self.input.scf.xc_functional == "ca":
            return ca_lda(density, self.grid.volume_element, self.core_density)
        if self.input.scf.xc_functional == "pbe":
            return pbe(density, self.grid, self.core_density)
        raise ValueError(
            f"unsupported XC functional {self.input.scf.xc_functional!r}"
        )


def prepare_single_point(
    problem: SinglePointInput,
    *,
    negative_laplacian_builder: Callable[[RealSpaceGrid], sp.csr_matrix] | None = None,
    local_ionic_builder: Callable[..., np.ndarray] | None = None,
    nonlocal_projector_builder: Callable[..., NonlocalProjectorOperator] | None = None,
    atomic_density_builder: Callable[..., np.ndarray] | None = None,
) -> PreparedSinglePointSystem:
    """Build every static component, but do not enter the SCF loop.

    The optional builders are execution-only extension points for the sibling
    accelerated package.  Omitting them preserves the validated NumPy/SciPy
    construction exactly.  Supplied builders must return the same
    compressed-grid Laplacian, local ionic field, KB projector factorization,
    and atomic-density superposition respectively.
    """
    preparation_start = time.perf_counter()
    # 1. Geometry, pseudopotentials, and electron count.
    atoms = (
        center_cluster_geometry(problem.atoms)
        if problem.recenter_geometry
        else tuple(problem.atoms)
    )
    stage_start = time.perf_counter()
    pseudopotentials = load_pseudopotentials(
        problem.pseudopotentials,
        xc_functional=problem.scf.xc_functional,
    )
    pseudopotential_loading_seconds = time.perf_counter() - stage_start
    electron_count = ionic_charge(atoms, pseudopotentials) - problem.scf.net_charge
    if electron_count <= 0:
        raise ValueError("the requested system has no valence electrons")

    # 2. Real-space domain and finite-difference kinetic operator.  In
    # Rydberg units this full sparse operator is T=-nabla_FD^2.  It is static
    # and reused in every eigensolver Hamiltonian application.
    stage_start = time.perf_counter()
    grid = build_cluster_grid(problem.grid)
    grid_seconds = time.perf_counter() - stage_start
    for atom in atoms:
        coordinate = np.asarray(atom.position)
        if problem.grid.domain_shape == "sphere":
            inside = np.dot(coordinate, coordinate) <= problem.grid.radius**2
        else:
            inside = np.all(
                np.abs(coordinate)
                <= 0.5 * np.asarray(problem.grid.box_lengths) + 1.0e-14
            )
        if not inside:
            raise ValueError(f"atom {atom.symbol} at {coordinate} lies outside the domain")

    stage_start = time.perf_counter()
    laplacian_builder = (
        build_negative_laplacian
        if negative_laplacian_builder is None
        else negative_laplacian_builder
    )
    negative_laplacian = laplacian_builder(grid)
    finite_difference_seconds = time.perf_counter() - stage_start

    # 3. Local and Kleinman--Bylander nonlocal ionic terms.  Both are static,
    # but they enter H differently: V_ion,local joins the diagonal effective
    # potential, whereas V_NL remains a separate low-rank operator.
    stage_start = time.perf_counter()
    build_local = (
        build_local_ionic_potential
        if local_ionic_builder is None
        else local_ionic_builder
    )
    ionic_potential = build_local(
        grid, atoms, pseudopotentials, problem.pseudopotentials
    )
    local_ionic_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    build_nonlocal = (
        build_nonlocal_projectors
        if nonlocal_projector_builder is None
        else nonlocal_projector_builder
    )
    nonlocal_operator = build_nonlocal(
        grid, atoms, pseudopotentials, problem.pseudopotentials
    )
    nonlocal_ionic_seconds = time.perf_counter() - stage_start

    # 4. Initial valence density and nonlinear core-correction density.  SAD
    # remains the PARSEC-compatible default.  A file/ML provider returns the
    # same one-dimensional volume-density representation on this exact DFT
    # grid, so no later SCF, Hartree, XC, or Hamiltonian code branches on how
    # the initial guess was produced.  Frozen rho_core always comes from the
    # PP atomic data and is never replaced by an ML valence prediction.
    stage_start = time.perf_counter()
    build_atomic_density = (
        superpose_atomic_density
        if atomic_density_builder is None
        else atomic_density_builder
    )
    if problem.initial_density_settings.method == "sad":
        initial_density = build_atomic_density(
            grid, atoms, pseudopotentials, problem.pseudopotentials
        )
    else:
        # Imported lazily: default SAD calculations do not import or initialize
        # an ML framework.  The model itself runs in an isolated subprocess.
        from ..MLDensity import build_initial_density

        initial_density = build_initial_density(
            problem.initial_density_settings,
            grid,
            atoms,
            problem.pseudopotentials,
        ).density
    # PARSEC normalizes its atomic initial guess.  Applying the same operation
    # to an ML prediction removes model/grid-integration charge drift and is
    # the scientifically safe default.  Disabling it is diagnostic only.
    if problem.scf.normalize_initial_density:
        initial_density = normalize_density(initial_density, grid, electron_count)
    initial_density_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    core_density = build_atomic_density(
        grid,
        atoms,
        pseudopotentials,
        problem.pseudopotentials,
        core=True,
    )
    core_density_seconds = time.perf_counter() - stage_start

    # 5. Geometry-only ion--ion contribution to the total energy.
    stage_start = time.perf_counter()
    repulsion = ion_ion_energy(atoms, pseudopotentials)
    atomic_reference_correction = float(
        sum(
            problem.pseudopotentials[atom.symbol].atomic_energy_correction
            for atom in atoms
        )
    )
    ion_ion_seconds = time.perf_counter() - stage_start
    preparation_timings = PreparationTimings(
        pseudopotential_loading_seconds=pseudopotential_loading_seconds,
        grid_seconds=grid_seconds,
        finite_difference_seconds=finite_difference_seconds,
        local_ionic_seconds=local_ionic_seconds,
        nonlocal_ionic_seconds=nonlocal_ionic_seconds,
        initial_density_seconds=initial_density_seconds,
        core_density_seconds=core_density_seconds,
        ion_ion_seconds=ion_ion_seconds,
        total_seconds=time.perf_counter() - preparation_start,
    )
    return PreparedSinglePointSystem(
        input=problem,
        atoms=atoms,
        electron_count=electron_count,
        pseudopotentials=pseudopotentials,
        grid=grid,
        negative_laplacian=negative_laplacian,
        ionic_potential=ionic_potential,
        nonlocal_operator=nonlocal_operator,
        initial_density=initial_density,
        core_density=core_density,
        ion_ion_energy=repulsion,
        atomic_reference_correction=atomic_reference_correction,
        timings=preparation_timings,
    )


def _number_of_states(system: PreparedSinglePointSystem) -> int:
    """Resolve the number of eigenpairs returned to occupations/density.

    A spin-unpolarized orbital can hold two electrons, so at least
    ``ceil(N_e/2)`` states are needed at zero temperature.  Finite-temperature
    Fermi filling also needs an unoccupied state above the frontier.  The six
    default extra states here are requested physical outputs; the eigensolver
    independently adds ``subspace_buffer`` working vectors for filtering.
    This fallback is a Python convenience: PARSEC inputs normally specify
    ``States_Num`` explicitly for reproducible calculations.
    """
    requested = system.input.scf.number_of_states
    if requested is None:
        requested = int(np.ceil(0.5 * system.electron_count)) + 6
    requested = int(requested)
    if requested <= 0.5 * system.electron_count and system.input.scf.fermi_temperature_kelvin > 0:
        raise ValueError(
            "number_of_states must leave at least one unoccupied state at finite temperature"
        )
    if requested + system.input.eigensolver.subspace_buffer >= system.grid.size:
        raise ValueError("the grid is too small for the requested states and eigensolver buffer")
    return requested


def run_scf(
    system: PreparedSinglePointSystem,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
    eigenproblem_solver: Callable[..., object] | None = None,
    orbital_density_builder: Callable[..., np.ndarray] | None = None,
    mixer_factory: Callable[..., object] | None = None,
    residual_metrics_evaluator: Callable[..., object] | None = None,
    total_energy_evaluator: Callable[..., object] | None = None,
    scalar_field_adapter: object | None = None,
) -> SinglePointResult:
    """Run PARSEC-style potential mixing on a prepared system.

    At iteration ``k`` the eigensolver sees

    ``H[V_in^k] = -nabla_FD^2 + diag(V_in^k) + V_NL``

    with ``V_in^k = V_ion,local + V_H^in + V_xc^in``.  Its orbitals produce a
    new density and therefore an output field

    ``V_out^k = V_ion,local + V_H[rho^k] + V_xc[rho^k + rho_core]``.

    The mixer constructs ``V_in^(k+1)`` from these two local fields.  Kinetic
    and nonlocal terms are never mixed because they are density independent.
    """
    scf_start = time.perf_counter()
    # The validated reference path always selects ``solve_eigval`` here.  The
    # optional callable is a narrow execution hook used by the sibling
    # accelerated package to retain the same SCF/mixing/energy control flow
    # while keeping a CuPy eigensubspace resident on the device.  It is not a
    # physics or solver-policy fallback: the callable receives the exact
    # PARSEC-derived EigvalSettings assembled below and must honor them.
    solve_current_eigenproblem = (
        solve_eigval if eigenproblem_solver is None else eigenproblem_solver
    )
    build_orbital_density = (
        density_from_orbitals
        if orbital_density_builder is None
        else orbital_density_builder
    )
    # These optional execution hooks let a rigorously symmetry-reduced
    # backend perform scalar-field dot products and Anderson history on one
    # value per orbit.  Their default functions are exactly the original
    # full-grid algorithms, so the reference architecture is unchanged.
    build_mixer = AndersonMixer if mixer_factory is None else mixer_factory
    evaluate_residual_metrics = (
        potential_residual_metrics
        if residual_metrics_evaluator is None
        else residual_metrics_evaluator
    )
    evaluate_total_energy = (
        total_energy
        if total_energy_evaluator is None
        else total_energy_evaluator
    )
    # An accelerated exact-symmetry backend may retain physical scalar fields
    # on one value per orbit.  The readable reference path leaves this adapter
    # unset and therefore executes the original full-grid NumPy operations.
    # Only the public result boundary expands compact fields again.
    to_internal_scalar = (
        (lambda values: values)
        if scalar_field_adapter is None
        else scalar_field_adapter.from_full
    )
    to_public_scalar = (
        (lambda values: values)
        if scalar_field_adapter is None
        else scalar_field_adapter.to_full
    )
    ionic_potential = to_internal_scalar(system.ionic_potential)
    # 1. Establish the initial Hartree, XC, and effective potentials.  The
    # number below is the state count returned to occupations; solve_eigval
    # internally works with number_of_states + subspace_buffer vectors.
    number_of_states = _number_of_states(system)
    settings = system.input.scf
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method not in {"chebff", "chebdav"}:
        raise NotImplementedError(
            f"Eigensolver={eigensolver_settings.method!r} is not yet ported "
            "to the strict native-Python SCF path; no alternate solver will "
            "be substituted."
        )
    # PARSEC fixes CHEBFF's cycle count and later SUBSPACE pass count.  CHEBDAV
    # instead uses Diag_Tolerance for residual-prefix locking during this first
    # solve.  Neither selection is silently replaced by another eigensolver.
    # rho_0 is the normalized superposition of atomic valence densities built
    # during preparation.  It is only an initial SCF guess, not an eigendensity.
    density = to_internal_scalar(system.initial_density.copy())
    # ``-V_ion`` below is only the numerical starting vector for the first
    # Poisson solve; it is not an identity between ionic and Hartree fields.
    stage_start = time.perf_counter()
    initial_hartree = system.solve_hartree(
        density, initial_potential=-ionic_potential
    )
    initial_hartree_seconds = time.perf_counter() - stage_start
    hartree_potential = initial_hartree.potential
    # XC sees rho_valence + frozen rho_core (CA-LDA or PBE, as selected).  The
    # Hartree solve immediately above sees rho_valence only.
    stage_start = time.perf_counter()
    xc = system.evaluate_xc(density)
    initial_xc_seconds = time.perf_counter() - stage_start
    xc_potential = xc.potential
    # This array is the diagonal, local part of H.  V_NL is intentionally not
    # folded into it; PreparedSinglePointSystem.hamiltonian adds V_NL as an
    # operator when the eigensolver requests H@vector.
    input_potential = (
        ionic_potential + hartree_potential + xc_potential
    )

    # 2. Initialize state carried between nonlinear iterations.  eigval_state
    # is None only for the first diagonalization: solve_eigval then runs the
    # selected CHEBFF/CHEBDAV routine and stores its complete buffered Ritz
    # subspace.  Later calls feed that state to one cheaper SUBSPACE filter.
    eigval_state: object | None = None
    mixer = build_mixer(system.input.mixing)
    history: list[SCFIteration] = []
    # This is the later-SCF SUBSPACE polynomial degree.  The first-solver
    # degree is a separate setting.  Near SCF convergence PARSEC lowers the
    # later degree gradually, but never below minimum_filter_degree.
    filter_degree = eigensolver_settings.filter_degree
    minimum_filter_degree = max(
        10, eigensolver_settings.filter_degree_delta + 1
    )
    # PARSEC's ordinary condition is polym>10.  The delta+1 term is a Python
    # validity guard for unusual inputs, ensuring p-delta remains positive;
    # it is identical to the Fortran threshold for normal delta values 0--3.
    converged = False
    eigenvalues = np.empty(0)
    occupations = np.empty(0)
    representations = np.empty(0, dtype=np.int32)
    wavefunctions = np.empty((system.grid.size, 0))
    fermi_level = np.nan
    energies = None
    last_input_potential = input_potential.copy()
    mixed_potential = input_potential.copy()
    hamiltonian_binding_total = 0.0
    diagonalization_total = 0.0
    occupations_density_total = 0.0
    hartree_total = 0.0
    xc_total = 0.0
    mixing_energy_total = 0.0

    for iteration in range(1, settings.max_iterations + 1):
        iteration_start = time.perf_counter()
        # 3. Solve the Kohn--Sham eigenproblem with PARSEC's fixed number of
        # first-filter cycles or one later-filter pass.
        # The eigenproblem in this iteration uses V_in exactly as it existed
        # before constructing rho_new; save it for energy bookkeeping/output.
        last_input_potential = input_potential.copy()
        # Rebuild the small policy object so a degree reduced by the SCF logic
        # below is used by the next SUBSPACE call.  solve_eigval dispatches on
        # eigval_state, not on the nonlinear iteration number itself.
        eigval_settings = EigvalSettings(
            safety_buffer=eigensolver_settings.subspace_buffer,
            initial_method=eigensolver_settings.method,
            chebff=ChebFFSettings(
                polynomial_degree=eigensolver_settings.first_filter_degree,
                filter_cycles=eigensolver_settings.first_filter_cycles,
                # chebff.f90z requests ten steps; the non-BETA bound routine
                # then applies its own PARSEC clamp to eight.
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
                    # create_eigen_solver fixes winsize=12 for CHEBDAV.
                    workspace_window=12,
                    lanczos_steps=5,
                    max_outer_restarts=2,
                    random_seed=eigensolver_settings.random_seed,
                )
                if eigensolver_settings.method == "chebdav"
                # A CHEBFF degree of 10 is valid.  Do not feed that unrelated
                # setting through CHEBDAV's stricter >=15 validation merely
                # because EigvalSettings stores both explicit choices.
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
        # This is the Hamiltonian assembly/handoff point.  The lightweight
        # object combines the current mixed V_in with the static kinetic and
        # KB terms; LinearOperator routes all Lanczos and Chebyshev H@Q calls
        # through that same three-term action.
        binding_start = time.perf_counter()
        hamiltonian = system.hamiltonian(input_potential)
        operator_factory = getattr(
            hamiltonian,
            "as_eigensolver_operator",
            hamiltonian.as_linear_operator,
        )
        hamiltonian_operator = operator_factory()
        hamiltonian_binding_seconds = time.perf_counter() - binding_start
        hamiltonian_binding_total += hamiltonian_binding_seconds
        diagonalization_start = time.perf_counter()
        eigensolution = solve_current_eigenproblem(
            hamiltonian_operator,
            number_of_states,
            settings=eigval_settings,
            state=eigval_state,
        )
        # Persist all buffered Ritz pairs and filter history, not only the
        # number_of_states columns returned below.  This is what makes later
        # SCF diagonalizations saved-subspace updates instead of fresh solves.
        eigval_state = eigensolution.state
        diagonalization_seconds = time.perf_counter() - diagonalization_start
        diagonalization_total += diagonalization_seconds
        eigenvalues = eigensolution.eigenvalues
        wavefunctions = eigensolution.vectors
        sector_labels = getattr(eigensolution, "representations", None)
        representations = (
            np.ones(eigenvalues.size, dtype=np.int32)
            if sector_labels is None
            else np.asarray(sector_labels, dtype=np.int32) + 1
        )
        # First-cycle CHEBFF deliberately supplies no Ritz-residual acceptance
        # test, hence NaN for that selection.  CHEBDAV returns the residuals it
        # used for locking.  Later SUBSPACE residuals remain diagnostics and do
        # not control either filter work or SCF convergence.
        eigen_residual_max = (
            float("nan")
            if eigensolution.residual_norms is None
            else float(np.max(eigensolution.residual_norms, initial=0.0))
        )

        # 4. Fill the states and construct the new electron density.  Each
        # occupation lies in [0,1] for one spatial orbital and Fermi filling
        # enforces 2*sum_n(f_n)=N_e.  density_from_orbitals
        # applies the spin degeneracy of two and converts Euclidean-normalized
        # grid vectors q=sqrt(h^3)*psi back to a volume density:
        # rho_i = (2/h^3)*sum_n f_n*|q_i,n|^2.
        occupations_density_start = time.perf_counter()
        occupation_result = fermi_occupations(
            eigenvalues,
            system.electron_count,
            settings.fermi_temperature_kelvin,
        )
        occupations = occupation_result.occupations
        fermi_level = occupation_result.fermi_level
        density = build_orbital_density(
            wavefunctions, occupations, system.grid.volume_element
        )
        occupations_density_seconds = (
            time.perf_counter() - occupations_density_start
        )
        occupations_density_total += occupations_density_seconds

        # 5. Rebuild the density-dependent Hartree and selected XC potentials.
        # previous V_H is a warm start for Poisson CG; changing that initial
        # vector does not change the converged Poisson equation or boundary.
        hartree_start = time.perf_counter()
        hartree = system.solve_hartree(
            density, initial_potential=hartree_potential
        )
        hartree_seconds = time.perf_counter() - hartree_start
        hartree_total += hartree_seconds
        hartree_potential = hartree.potential
        xc_start = time.perf_counter()
        xc = system.evaluate_xc(density)
        xc_seconds = time.perf_counter() - xc_start
        xc_total += xc_seconds
        xc_potential = xc.potential
        # V_out is a newly evaluated *local* field.  It is not used to
        # rediagonalize immediately: convergence is measured against V_in,
        # then the mixer supplies the next iteration's local input field.
        output_potential = (
            ionic_potential + hartree_potential + xc_potential
        )

        # 6. Evaluate convergence diagnostics and the total energy.  With
        # R_i=V_out,i-V_in,i, the default weighted SRE is
        # sqrt[h^3*sum_i rho_i*R_i^2/N_e]; the optional plain norm omits rho/N_e.
        mixing_energy_start = time.perf_counter()
        metrics = evaluate_residual_metrics(
            input_potential,
            output_potential,
            density,
            system.grid.volume_element,
            system.electron_count,
        )
        # The eigenpairs came from H[V_in], whereas Hartree/XC above came from
        # the new density.  total_energy deliberately receives both input and
        # output fields to apply PARSEC's corresponding double-counting
        # correction; this is not an accidental old/new-potential mixture.
        # In compact form the electronic expression is
        #   E = E_band - integral rho*(V_in-V_ion)
        #       + 1/2 integral rho*V_H,out + E_xc,out.
        energies = evaluate_total_energy(
            eigenvalues,
            occupations,
            density,
            input_potential,
            ionic_potential,
            hartree_potential,
            xc_potential,
            xc.total_energy,
            system.ion_ion_energy,
            system.grid.volume_element,
        )
        # 7. Mix the effective potential and apply PARSEC's SCF stop test.
        # Only V_eff is mixed.  T, V_NL, and the fixed local ionic component
        # remain unchanged; V_ion appears in both input and output and thus
        # passes through the mixing algebra unchanged up to roundoff.
        # The first Anderson call is simple linear mixing V_in+alpha*(V_out-
        # V_in).  Later calls use the retained residual/input history and its
        # configured restart schedule.  PARSEC also performs mixing before the
        # stop test, so a converged result still has a diagnostic V_next.
        mixed_potential = mixer.mix(
            input_potential, output_potential, iteration=iteration
        )
        # SCF convergence is a local-potential fixed-point test, not an energy
        # difference and not the Ritz residual reported above.
        selected_residual = (
            metrics.plain if settings.use_plain_residual else metrics.weighted
        )
        # PARSEC's approach-to-convergence test always uses weighted SRE even
        # when plain SRE is selected for the final stop.  The factor 100 is the
        # current Python/default-PARSEC approach threshold; a separately
        # configurable Convergence_Criterion_Approach is not yet exposed.
        approach_residual = metrics.weighted
        if (
            iteration > 5
            and approach_residual < 100.0 * settings.convergence_criterion
            and filter_degree > minimum_filter_degree
        ):
            # Once reasonably close (100*vconv), save Hamiltonian applications
            # by lowering only the later-SCF polynomial one degree per step.
            filter_degree -= 1
        # The strict inequality is PARSEC's fixed-point stop rule.  Energy,
        # Hartree, and eigenpair residuals remain diagnostics only.
        iteration_converged = selected_residual < settings.convergence_criterion
        mixing_energy_seconds = time.perf_counter() - mixing_energy_start
        mixing_energy_total += mixing_energy_seconds

        # Capture a self-contained record after mixing so its timing contains
        # both energy/convergence work and the PARSEC-mandated mixer call.  The
        # callback still receives exactly one completed record per iteration.
        eigenvalue_history = tuple(float(value) for value in eigenvalues)
        occupation_history = tuple(float(value) for value in occupations)
        representation_history = tuple(
            int(value) for value in representations
        )
        density_minimum = (
            float(np.min(density))
            if scalar_field_adapter is None
            else scalar_field_adapter.minimum(density)
        )
        density_maximum = (
            float(np.max(density))
            if scalar_field_adapter is None
            else scalar_field_adapter.maximum(density)
        )
        iteration_seconds = time.perf_counter() - iteration_start
        diagnostic = SCFIteration(
            iteration=iteration,
            weighted_residual=metrics.weighted,
            plain_residual=metrics.plain,
            eigen_residual_max=eigen_residual_max,
            hartree_residual=hartree.residual_norm,
            energies=energies,
            eigenvalues=eigenvalue_history,
            occupations=occupation_history,
            representations=representation_history,
            fermi_level=float(fermi_level),
            density_minimum=density_minimum,
            density_maximum=density_maximum,
            diagonalization_seconds=diagonalization_seconds,
            hartree_seconds=hartree_seconds,
            hamiltonian_binding_seconds=hamiltonian_binding_seconds,
            occupations_density_seconds=occupations_density_seconds,
            xc_seconds=xc_seconds,
            mixing_energy_seconds=mixing_energy_seconds,
            total_seconds=iteration_seconds,
        )
        history.append(diagnostic)
        if callback is not None:
            callback(diagnostic)

        if iteration_converged:
            converged = True
            break
        input_potential = mixed_potential

    if energies is None:
        raise RuntimeError("SCF loop did not execute")
    # The result intentionally retains three local fields: the V_in used for
    # the final eigensolve, the V_out generated from its density, and the mixed
    # candidate that would seed one more iteration.  They coincide only at an
    # exactly converged fixed point.  Exhausting max_iterations returns this
    # latest consistent state with converged=False rather than raising.
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
    return SinglePointResult(
        converged=converged,
        iterations=len(history),
        atoms=system.atoms,
        electron_count=system.electron_count,
        grid=system.grid,
        pseudopotentials=system.pseudopotentials,
        density=to_public_scalar(density),
        core_density=system.core_density,
        ionic_potential=system.ionic_potential,
        hartree_potential=to_public_scalar(hartree_potential),
        xc_potential=to_public_scalar(xc_potential),
        input_effective_potential=to_public_scalar(last_input_potential),
        output_effective_potential=to_public_scalar(
            ionic_potential + hartree_potential + xc_potential
        ),
        next_effective_potential=to_public_scalar(mixed_potential),
        nonlocal_operator=system.nonlocal_operator,
        eigenvalues=eigenvalues,
        occupations=occupations,
        wavefunctions=wavefunctions,
        fermi_level=fermi_level,
        energies=energies,
        history=history,
        timings=run_timings,
        representations=representations,
        atomic_reference_correction=system.atomic_reference_correction,
    )
