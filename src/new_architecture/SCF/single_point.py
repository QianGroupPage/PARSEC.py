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

from dataclasses import dataclass
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
    SCFIteration,
    SinglePointInput,
    SinglePointResult,
)
from ..V_xc import XCResult, ca_lda


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
        """Evaluate CA-LDA at ``density + frozen_core_density``."""
        return ca_lda(density, self.grid.volume_element, self.core_density)


def prepare_single_point(problem: SinglePointInput) -> PreparedSinglePointSystem:
    """Build every static component, but do not enter the SCF loop."""
    # 1. Geometry, pseudopotentials, and electron count.
    atoms = (
        center_cluster_geometry(problem.atoms)
        if problem.recenter_geometry
        else tuple(problem.atoms)
    )
    pseudopotentials = load_pseudopotentials(problem.pseudopotentials)
    electron_count = ionic_charge(atoms, pseudopotentials) - problem.scf.net_charge
    if electron_count <= 0:
        raise ValueError("the requested system has no valence electrons")

    # 2. Real-space domain and finite-difference kinetic operator.  In
    # Rydberg units this full sparse operator is T=-nabla_FD^2.  It is static
    # and reused in every eigensolver Hamiltonian application.
    grid = build_cluster_grid(problem.grid)
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

    negative_laplacian = build_negative_laplacian(grid)

    # 3. Local and Kleinman--Bylander nonlocal ionic terms.  Both are static,
    # but they enter H differently: V_ion,local joins the diagonal effective
    # potential, whereas V_NL remains a separate low-rank operator.
    ionic_potential = build_local_ionic_potential(
        grid, atoms, pseudopotentials, problem.pseudopotentials
    )
    nonlocal_operator = build_nonlocal_projectors(
        grid, atoms, pseudopotentials, problem.pseudopotentials
    )

    # 4. Initial valence density and nonlinear core-correction density.  The
    # valence SAD is normalized to N_e and later supplies occupations/Hartree;
    # frozen rho_core is never normalized into N_e and enters only CA-LDA XC.
    initial_density = superpose_atomic_density(
        grid, atoms, pseudopotentials, problem.pseudopotentials
    )
    # PARSEC always performs this SAD normalization.  Disabling it is an
    # explicit Python diagnostic option rather than a Fortran-compatible mode.
    if problem.scf.normalize_initial_density:
        initial_density = normalize_density(initial_density, grid, electron_count)
    core_density = superpose_atomic_density(
        grid,
        atoms,
        pseudopotentials,
        problem.pseudopotentials,
        core=True,
    )

    # 5. Geometry-only ion--ion contribution to the total energy.
    repulsion = ion_ion_energy(atoms, pseudopotentials)
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
    density = system.initial_density.copy()
    # ``-V_ion`` below is only the numerical starting vector for the first
    # Poisson solve; it is not an identity between ionic and Hartree fields.
    initial_hartree = system.solve_hartree(
        density, initial_potential=-system.ionic_potential
    )
    hartree_potential = initial_hartree.potential
    # CA-LDA sees rho_valence + frozen rho_core.  The Hartree solve immediately
    # above sees rho_valence only.
    xc = system.evaluate_xc(density)
    xc_potential = xc.potential
    # This array is the diagonal, local part of H.  V_NL is intentionally not
    # folded into it; PreparedSinglePointSystem.hamiltonian adds V_NL as an
    # operator when the eigensolver requests H@vector.
    input_potential = (
        system.ionic_potential + hartree_potential + xc_potential
    )

    # 2. Initialize state carried between nonlinear iterations.  eigval_state
    # is None only for the first diagonalization: solve_eigval then runs the
    # selected CHEBFF/CHEBDAV routine and stores its complete buffered Ritz
    # subspace.  Later calls feed that state to one cheaper SUBSPACE filter.
    eigval_state: EigvalState | None = None
    mixer = AndersonMixer(system.input.mixing)
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
    wavefunctions = np.empty((system.grid.size, 0))
    fermi_level = np.nan
    energies = None
    last_input_potential = input_potential.copy()
    mixed_potential = input_potential.copy()

    for iteration in range(1, settings.max_iterations + 1):
        # 3. Solve the Kohn--Sham eigenproblem with PARSEC's fixed number of
        # first-filter cycles or one later-filter pass.
        # The eigenproblem in this iteration uses V_in exactly as it existed
        # before constructing rho_new; save it for energy bookkeeping/output.
        last_input_potential = input_potential.copy()
        diagonalization_start = time.perf_counter()
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
        hamiltonian = system.hamiltonian(input_potential)
        eigensolution = solve_eigval(
            hamiltonian.as_linear_operator(),
            number_of_states,
            settings=eigval_settings,
            state=eigval_state,
        )
        # Persist all buffered Ritz pairs and filter history, not only the
        # number_of_states columns returned below.  This is what makes later
        # SCF diagonalizations saved-subspace updates instead of fresh solves.
        eigval_state = eigensolution.state
        diagonalization_seconds = time.perf_counter() - diagonalization_start
        eigenvalues = eigensolution.eigenvalues
        wavefunctions = eigensolution.vectors
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
        occupation_result = fermi_occupations(
            eigenvalues,
            system.electron_count,
            settings.fermi_temperature_kelvin,
        )
        occupations = occupation_result.occupations
        fermi_level = occupation_result.fermi_level
        density = density_from_orbitals(
            wavefunctions, occupations, system.grid.volume_element
        )

        # 5. Rebuild the density-dependent Hartree and CA-LDA potentials.  The
        # previous V_H is a warm start for Poisson CG; changing that initial
        # vector does not change the converged Poisson equation or boundary.
        hartree_start = time.perf_counter()
        hartree = system.solve_hartree(
            density, initial_potential=hartree_potential
        )
        hartree_seconds = time.perf_counter() - hartree_start
        hartree_potential = hartree.potential
        xc = system.evaluate_xc(density)
        xc_potential = xc.potential
        # V_out is a newly evaluated *local* field.  It is not used to
        # rediagonalize immediately: convergence is measured against V_in,
        # then the mixer supplies the next iteration's local input field.
        output_potential = (
            system.ionic_potential + hartree_potential + xc_potential
        )

        # 6. Evaluate convergence diagnostics and the total energy.  With
        # R_i=V_out,i-V_in,i, the default weighted SRE is
        # sqrt[h^3*sum_i rho_i*R_i^2/N_e]; the optional plain norm omits rho/N_e.
        metrics = potential_residual_metrics(
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
        energies = total_energy(
            eigenvalues,
            occupations,
            density,
            input_potential,
            system.ionic_potential,
            hartree_potential,
            xc_potential,
            xc.total_energy,
            system.ion_ion_energy,
            system.grid.volume_element,
        )
        # Capture a self-contained record before mixing mutates the next input.
        diagnostic = SCFIteration(
            iteration=iteration,
            weighted_residual=metrics.weighted,
            plain_residual=metrics.plain,
            eigen_residual_max=eigen_residual_max,
            hartree_residual=hartree.residual_norm,
            energies=energies,
            eigenvalues=tuple(float(value) for value in eigenvalues),
            occupations=tuple(float(value) for value in occupations),
            fermi_level=float(fermi_level),
            density_minimum=float(np.min(density)),
            density_maximum=float(np.max(density)),
            diagonalization_seconds=diagonalization_seconds,
            hartree_seconds=hartree_seconds,
        )
        history.append(diagnostic)
        if callback is not None:
            callback(diagnostic)

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
        if selected_residual < settings.convergence_criterion:
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
    return SinglePointResult(
        converged=converged,
        iterations=len(history),
        atoms=system.atoms,
        electron_count=system.electron_count,
        grid=system.grid,
        pseudopotentials=system.pseudopotentials,
        density=density,
        core_density=system.core_density,
        ionic_potential=system.ionic_potential,
        hartree_potential=hartree_potential,
        xc_potential=xc_potential,
        input_effective_potential=last_input_potential,
        output_effective_potential=(
            system.ionic_potential + hartree_potential + xc_potential
        ),
        next_effective_potential=mixed_potential,
        nonlocal_operator=system.nonlocal_operator,
        eigenvalues=eigenvalues,
        occupations=occupations,
        wavefunctions=wavefunctions,
        fermi_level=fermi_level,
        energies=energies,
        history=history,
    )
