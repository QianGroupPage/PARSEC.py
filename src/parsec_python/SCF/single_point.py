"""Isolated PARSEC-style SCF preparation and iteration.

``prepare_single_point`` constructs all density-independent objects.  The
nonlinear work happens in ``run_scf``, here for a calculation without spin
polarization and in :mod:`parsec_python.SCF.spin_polarized` for one with two
spin channels:

``V_in -> eigensolver -> occupations -> rho -> (V_H,V_xc) -> V_out``

``V_in,V_out -> residual/energy/mixer -> next V_in``.

The first eigensolver call constructs a buffered subspace with the selected
CHEBFF or CHEBDAV algorithm; later calls reuse it through one SUBSPACE filter.
SCF convergence is tested on the local potential residual, not on an
eigenpair or energy residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Callable, Mapping
import warnings

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
from ..Hartree import (
    AtomicTail,
    HartreeBoundaryPlan,
    HartreeResult,
    build_atomic_tail as build_reference_atomic_tail,
    plan_hartree_boundary,
    solve_hartree,
    valence_point_charges,
)
from ..Laplacian import build_negative_laplacian
from ..Mixer import AndersonMixer, potential_residual_metrics
from ..Occupations import (
    check_fixed_occupations,
    check_spin_fixed_occupations,
    density_from_orbitals,
    fermi_occupations,
    fixed_occupations,
)
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
    GridSettings,
    InitialDensitySettings,
    PreparationTimings,
    RunTimings,
    SCFIteration,
    SinglePointInput,
    SinglePointResult,
)
from ..V_xc import (
    NoncollinearXCResult,
    SpinPolarizedXCResult,
    SpinXCResult,
    XCResult,
    ca_lda,
    ca_lda_spin,
    ca_lda_spin_polarized,
    noncollinear_xc,
    pbe,
    pbe_spin,
    pbe_spin_polarized,
)


@dataclass(frozen=True)
class PreparedSinglePointSystem:
    """Static terms that can be prepared and inspected without running SCF.

    The grid kinetic operator ``-nabla_FD^2``, local ionic potential,
    nonlocal KB projectors, frozen NLCC density, and ion--ion energy do not
    change when only the electronic density is iterated.  Hartree and XC
    potentials are therefore absent here: they are rebuilt from the current
    density inside :func:`run_scf`.

    ``hartree_boundary`` is the plan of the Hartree boundary values made
    from the geometry (:mod:`parsec_python.Hartree.boundary`), and
    ``hartree_boundary_tail`` the static atomic tail where the plan has one.
    A system built without a plan has PARSEC's multipole boundary.

    ``initial_spin_densities`` are the starting densities of the up and the
    down channel of a spin-polarized calculation and ``None`` otherwise
    (:func:`initial_spin_densities`).
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
    alpha_z_energy: float = 0.0
    timings: PreparationTimings = field(default_factory=PreparationTimings)
    hartree_boundary: HartreeBoundaryPlan | None = None
    hartree_boundary_tail: AtomicTail | None = None
    initial_spin_densities: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def hartree_boundary_tail_maximum(self) -> float | None:
        """``max |C_L|`` of the atomic tail in Ry, ``None`` without one."""

        tail = self.hartree_boundary_tail
        return None if tail is None else tail.maximum

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
        plan = self.hartree_boundary
        if (
            plan is not None
            and plan.atomic_tail
            and self.hartree_boundary_tail is None
        ):
            raise RuntimeError(
                "the Hartree boundary of this system includes the atomic "
                "tail, which was not built with it"
            )
        return solve_hartree(
            density,
            self.grid,
            self.negative_laplacian,
            self.input.hartree,
            initial_potential,
            raise_on_nonconvergence=raise_on_nonconvergence,
            boundary_tail=self.hartree_boundary_tail,
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

    def evaluate_xc_spin_polarized(
        self, density_up: np.ndarray, density_down: np.ndarray
    ) -> SpinPolarizedXCResult:
        """Collinear spin-polarized XC (LSDA or PBE) including frozen NLCC."""

        functional = self.input.scf.xc_functional
        if functional == "ca":
            return ca_lda_spin_polarized(
                density_up,
                density_down,
                self.grid.volume_element,
                self.core_density,
            )
        if functional == "pbe":
            return pbe_spin_polarized(
                density_up, density_down, self.grid, self.core_density
            )
        raise ValueError(f"unsupported XC functional {functional!r}")

    def evaluate_xc_noncollinear(
        self, density: np.ndarray, magnetization: np.ndarray
    ) -> NoncollinearXCResult:
        """Non-collinear XC (``V_avg`` and ``B_xc``) from ``n`` and ``m(r)``."""

        return noncollinear_xc(density, magnetization, self.evaluate_xc_spin_polarized)
    def evaluate_spin_xc(
        self, density_up: np.ndarray, density_down: np.ndarray
    ) -> SpinXCResult:
        """Evaluate the selected spin-polarized XC functional.

        Each channel sees its valence density plus half of the frozen NLCC
        density (``rho_hart.F90``).
        """

        if self.input.scf.xc_functional == "ca":
            return ca_lda_spin(
                density_up, density_down, self.grid.volume_element, self.core_density
            )
        if self.input.scf.xc_functional == "pbe":
            return pbe_spin(density_up, density_down, self.grid, self.core_density)
        raise ValueError(
            f"unsupported XC functional {self.input.scf.xc_functional!r}"
        )


@dataclass(frozen=True)
class SCFWarmStart:
    """What an SCF loop takes over from the loop of the previous structure.

    After atoms move, ``parsec.F90`` re-enters its SCF loop without a new
    starting density: the eigenproblem of the first iteration is posed with
    the last mixed potential, in which the old local ionic potential has been
    replaced by the new one, the saved wavefunctions seed a SUBSPACE filter
    instead of a first CHEBFF/CHEBDAV solve, and the old Hartree potential is
    the starting vector of the first Poisson solve.  The mixer and the filter
    degree start afresh.

    ``input_potential`` and ``hartree_potential`` are full-grid fields in Ry.
    ``eigensolver_state`` is the state object of the eigensolver that will be
    used; ``None`` asks for a first solve with the given potential.

    A spin-polarized loop takes ``input_potential`` and
    ``eigensolver_state`` as pairs, the up channel first; the Hartree
    potential is common to both channels.
    """

    input_potential: np.ndarray
    hartree_potential: np.ndarray
    eigensolver_state: object | None = None


def state_for_new_potential(state: object | None) -> object | None:
    """Mark a saved eigensubspace as the start of a new SCF loop.

    ``parsec.F90`` sets ``solver%firstfilt`` again after every SCF loop, so
    the first SUBSPACE call for the next structure recomputes its lower
    filter bound from the saved Ritz values.  A state object of another
    eigensolver takes the same request through its ``for_new_scf_loop``
    method, if it has one.
    """

    if state is None:
        return None
    restart = getattr(state, "for_new_scf_loop", None)
    if callable(restart):
        return restart()
    if isinstance(state, EigvalState):
        return replace(state, subspace=replace(state.subspace, first_filter=True))
    return state


def warm_start_for_moved_atoms(
    previous_system: "PreparedSinglePointSystem",
    previous_result: SinglePointResult,
    system: "PreparedSinglePointSystem",
) -> SCFWarmStart:
    """Build the warm start of ``system`` from the run before the move.

    ``parsec.F90`` after ``domove``: ``vnew = vnew - vion_old`` and then
    ``vnew = vnew + vion_new``, with ``vnew`` the potential the mixer left,
    for each spin channel where there are two.
    """

    channels = getattr(previous_result, "spin", None)
    if channels is not None:
        return SCFWarmStart(
            input_potential=tuple(
                np.asarray(channel.next_effective_potential)
                - np.asarray(previous_system.ionic_potential)
                + np.asarray(system.ionic_potential)
                for channel in channels
            ),
            hartree_potential=np.asarray(previous_result.hartree_potential),
            eigensolver_state=tuple(
                state_for_new_potential(channel.eigensolver_state)
                for channel in channels
            ),
        )
    return SCFWarmStart(
        input_potential=(
            np.asarray(previous_result.next_effective_potential)
            - np.asarray(previous_system.ionic_potential)
            + np.asarray(system.ionic_potential)
        ),
        hartree_potential=np.asarray(previous_result.hartree_potential),
        eigensolver_state=state_for_new_potential(previous_result.eigensolver_state),
    )


def _require_atoms_inside_domain(settings: GridSettings, atoms) -> None:
    """Refuse a nucleus outside the sphere or box (``isolat.f90``)."""

    for atom in atoms:
        coordinate = np.asarray(atom.position)
        if settings.domain_shape == "sphere":
            inside = np.dot(coordinate, coordinate) <= settings.radius**2
        else:
            inside = np.all(
                np.abs(coordinate)
                <= 0.5 * np.asarray(settings.box_lengths) + 1.0e-14
            )
        if not inside:
            raise ValueError(f"atom {atom.symbol} at {coordinate} lies outside the domain")


def _require_spin_support(problem: SinglePointInput) -> None:
    """Refuse what a spin-polarized calculation cannot start from."""

    if problem.initial_density_settings.method != "sad":
        raise NotImplementedError(
            "a spin-polarized calculation starts from the superposition of "
            "atomic densities with the Initial_Spin_Polarization of each "
            "species; an imported or ML starting density "
            f"(Initial_Density: {problem.initial_density_settings.method}) "
            "holds no spin densities"
        )


def initial_spin_densities(
    total: np.ndarray,
    species_densities: Mapping[str, np.ndarray],
    polarizations: Mapping[str, float],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Starting densities of the up and the down channel (``initchrg.f90``).

    ``total`` is the superposition of the atomic valence densities before
    normalization, ``species_densities`` its part from the atoms of each
    species, ``polarizations`` the ``Initial_Spin_Polarization`` ``p`` of
    each species and ``scale`` the factor that normalizes ``total`` to the
    electron count.  ``initchrg.f90`` accumulates

    ``rho_up = sum_species (1 + p) rho_species``,
    ``rho_down = sum_species (1 - p) rho_species``

    and scales both by the factor of the total density.  The two channels
    therefore add up to twice the total density: the first exchange-
    correlation potential, and it alone, is evaluated at that doubled
    density, while the first Hartree potential is that of the total density
    itself.  The starting magnetic moment PARSEC prints, the integral of
    ``rho_up - rho_down``, is ``2 p N_e`` for one species.  This is kept as
    it is in the Fortran: it changes the path of the SCF loop, not the
    self-consistent state.
    """

    total = np.asarray(total, dtype=np.float64)
    magnetization = np.zeros_like(total)
    for symbol, density in species_densities.items():
        magnetization += float(polarizations[symbol]) * np.asarray(
            density, dtype=np.float64
        )
    return (scale * (total + magnetization), scale * (total - magnetization))


def _species_densities(
    total: np.ndarray,
    build_atomic_density: Callable[..., np.ndarray],
    grid: RealSpaceGrid,
    atoms: tuple,
    pseudopotentials: dict[str, ParsecPseudopotential],
    specifications,
) -> dict[str, np.ndarray]:
    """Split the superposed atomic density into the parts of the species.

    Only species with atoms are returned.  Where they all start with the
    same polarization the split is not needed and the total stands for one
    of them; otherwise the density of each species is superposed on its own.
    """

    symbols = list(dict.fromkeys(atom.symbol for atom in atoms))
    polarizations = {
        specifications[symbol].initial_spin_polarization for symbol in symbols
    }
    if len(polarizations) == 1:
        return {symbols[0]: total}
    return {
        symbol: build_atomic_density(
            grid,
            tuple(atom for atom in atoms if atom.symbol == symbol),
            pseudopotentials,
            specifications,
        )
        for symbol in symbols
    }


def _require_core_radius_support(
    operator: NonlocalProjectorOperator,
    grid: RealSpaceGrid,
    atoms,
    pseudopotentials,
) -> None:
    """Check a substituted projector builder against ``Core_Cutoff_Radius``.

    The projectors of an old Martins species end at the core radius of the
    input (``ParsecPseudopotential.projector_support_radius``), in general
    between two radial samples, and ``nonloc.F90`` keeps the grid points at
    distances up to it.  A builder that takes the next radial sample instead,
    the rule of a ``MARTINS_NEW`` file, adds grid points and reports nothing:
    for a radius inside the projector the total energy of a five-atom
    cluster moved by 2e-2 Ry.  Such an operator is refused here.
    """

    limited = {
        index: pseudopotentials[atom.symbol].projector_support_radius
        for index, atom in enumerate(atoms)
        if pseudopotentials[atom.symbol].core_radius is not None
    }
    if not limited:
        return
    matrix = operator.projectors
    for column, (atom_index, _, _) in enumerate(operator.labels):
        radius = limited.get(atom_index)
        if radius is None:
            continue
        rows = matrix.indices[matrix.indptr[column] : matrix.indptr[column + 1]]
        if rows.size == 0:
            continue
        distance = np.linalg.norm(
            grid.coordinates[rows] - np.asarray(atoms[atom_index].position),
            axis=1,
        )
        if float(np.max(distance)) > radius * (1.0 + 1.0e-12):
            raise RuntimeError(
                f"the projector builder placed {atoms[atom_index].symbol} "
                f"projectors up to {float(np.max(distance)):.6f} bohr, beyond the "
                f"core radius {radius:.6f} bohr of its old Martins "
                "pseudopotential; it must take the support radius from "
                "ParsecPseudopotential.projector_support_radius"
            )


def prepare_single_point(
    problem: SinglePointInput,
    *,
    grid_builder: Callable[[GridSettings], RealSpaceGrid] | None = None,
    negative_laplacian_builder: Callable[[RealSpaceGrid], sp.csr_matrix] | None = None,
    local_ionic_builder: Callable[..., np.ndarray] | None = None,
    nonlocal_projector_builder: Callable[..., NonlocalProjectorOperator] | None = None,
    atomic_density_builder: Callable[..., np.ndarray] | None = None,
    orbital_operators_only: bool = False,
    build_atomic_tail: bool = True,
) -> PreparedSinglePointSystem:
    """Build every static component, but do not enter the SCF loop.

    The optional builders are execution-only extension points for the sibling
    accelerated package.  Omitting them preserves the validated NumPy/SciPy
    construction exactly.  Supplied builders must return the same cluster
    grid, compressed-grid Laplacian, local ionic field, KB projector
    factorization, and atomic-density superposition respectively.

    ``orbital_operators_only`` serves a process that applies the Hamiltonian
    for an SCF loop running in another process.  It builds the grid, the
    kinetic operator and the KB projectors and stops there: the local ionic
    potential and both densities are ``None`` and the two ionic energies are
    NaN.  Such a system must not be passed to :func:`run_scf`.

    The plan of the Hartree boundary is made here from the geometry, also
    with ``orbital_operators_only``.  Where it raises the multipole order
    above ``Solver_Lpole``, ``input.hartree.multipole_order`` of the returned
    system is the order in use and ``minimum_multipole_order`` the
    ``Solver_Lpole`` that was asked.  ``build_atomic_tail=False`` leaves the
    static tail of a plan that has one to the caller, who builds it for the
    Hartree solver it substitutes; ``solve_hartree`` of the returned system
    then refuses to run.
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
    if problem.scf.spin_polarized:
        _require_spin_support(problem)
    if problem.scf.fixed_occupations is not None:
        # initial.f90 stops before any setup when the occupations of
        # occup.in do not hold the electrons of the system.
        if problem.scf.spin_polarized:
            check_spin_fixed_occupations(
                problem.scf.fixed_occupations,
                problem.scf.fixed_occupations_down,
                electron_count,
            )
        else:
            check_fixed_occupations(problem.scf.fixed_occupations, electron_count)
    boundary_plan = plan_hartree_boundary(
        problem.hartree,
        problem.grid,
        *valence_point_charges(atoms, pseudopotentials, electron_count),
    )
    if boundary_plan.order != problem.hartree.multipole_order:
        # Every Hartree builder reads the order from the input of the
        # prepared system.  The Solver_Lpole that was asked stays beside it:
        # a preparation of that input starts its plan there again.
        problem = replace(
            problem,
            hartree=replace(
                problem.hartree,
                multipole_order=boundary_plan.order,
                minimum_multipole_order=boundary_plan.minimum_order,
            ),
        )

    # 2. Real-space domain and finite-difference kinetic operator.  In
    # Rydberg units this full sparse operator is T=-nabla_FD^2.  It is static
    # and reused in every eigensolver Hamiltonian application.
    stage_start = time.perf_counter()
    grid = (build_cluster_grid if grid_builder is None else grid_builder)(
        problem.grid
    )
    grid_seconds = time.perf_counter() - stage_start
    _require_atoms_inside_domain(problem.grid, atoms)

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
    ionic_potential = (
        None
        if orbital_operators_only
        else build_local(
            grid, atoms, pseudopotentials, problem.pseudopotentials
        )
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
    if nonlocal_projector_builder is not None:
        _require_core_radius_support(
            nonlocal_operator, grid, atoms, pseudopotentials
        )
    nonlocal_ionic_seconds = time.perf_counter() - stage_start

    if orbital_operators_only:
        # Everything below is read only by the SCF loop: the starting
        # density and its normalization, the frozen core density of the XC
        # term, and the ionic energies.  None of it enters H.
        return PreparedSinglePointSystem(
            input=problem,
            atoms=atoms,
            electron_count=electron_count,
            pseudopotentials=pseudopotentials,
            grid=grid,
            negative_laplacian=negative_laplacian,
            ionic_potential=None,
            nonlocal_operator=nonlocal_operator,
            initial_density=None,
            core_density=None,
            ion_ion_energy=float("nan"),
            atomic_reference_correction=float("nan"),
            timings=PreparationTimings(
                pseudopotential_loading_seconds=pseudopotential_loading_seconds,
                grid_seconds=grid_seconds,
                finite_difference_seconds=finite_difference_seconds,
                nonlocal_ionic_seconds=nonlocal_ionic_seconds,
                hartree_boundary_seconds=boundary_plan.seconds,
                total_seconds=time.perf_counter() - preparation_start,
            ),
            hartree_boundary=boundary_plan,
        )

    fields, stage_seconds = _density_and_energy_stages(
        problem,
        grid,
        atoms,
        pseudopotentials,
        electron_count,
        atomic_density_builder,
        boundary_plan,
        build_atomic_tail,
    )
    preparation_timings = PreparationTimings(
        pseudopotential_loading_seconds=pseudopotential_loading_seconds,
        grid_seconds=grid_seconds,
        finite_difference_seconds=finite_difference_seconds,
        local_ionic_seconds=local_ionic_seconds,
        nonlocal_ionic_seconds=nonlocal_ionic_seconds,
        **stage_seconds,
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
        **fields,
        timings=preparation_timings,
        hartree_boundary=boundary_plan,
    )


def _density_and_energy_stages(
    problem: SinglePointInput,
    grid: RealSpaceGrid,
    atoms: tuple,
    pseudopotentials: dict[str, ParsecPseudopotential],
    electron_count: float,
    atomic_density_builder: Callable[..., np.ndarray] | None,
    boundary_plan: HartreeBoundaryPlan | None,
    build_atomic_tail: bool,
    ion_ion_energy_builder: Callable[..., float] | None = None,
) -> tuple[dict[str, object], dict[str, float]]:
    """Run stages 4 to 6 of the preparation.

    Returns the fields of :class:`PreparedSinglePointSystem` they build and
    the seconds of each stage under its :class:`PreparationTimings` name.
    ``ion_ion_energy_builder`` is called in place of
    :func:`ion_ion_energy`, with its arguments.
    """
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
    spin_densities = None
    if problem.initial_density_settings.method == "sad":
        initial_density = build_atomic_density(
            grid, atoms, pseudopotentials, problem.pseudopotentials
        )
        if problem.scf.spin_polarized:
            # initchrg.f90 scales the two channels with the factor of the
            # total density, here also where that factor is left at one.
            spin_densities = initial_spin_densities(
                initial_density,
                _species_densities(
                    initial_density,
                    build_atomic_density,
                    grid,
                    atoms,
                    pseudopotentials,
                    problem.pseudopotentials,
                ),
                {
                    symbol: specification.initial_spin_polarization
                    for symbol, specification in problem.pseudopotentials.items()
                },
                (
                    electron_count / grid.integrate(initial_density)
                    if problem.scf.normalize_initial_density
                    else 1.0
                ),
            )
    else:
        # Imported lazily: default SAD calculations do not import or initialize
        # an ML framework.  The model itself runs in an isolated subprocess.
        from ..MLDensity import build_initial_density
        from ..MLDensity.providers import electron_count_warning

        initial_density = build_initial_density(
            problem.initial_density_settings,
            grid,
            atoms,
            problem.pseudopotentials,
        ).density
        message = electron_count_warning(
            grid.integrate(initial_density),
            electron_count,
            f"Initial_Density: {problem.initial_density_settings.method}",
        )
        if message is not None:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
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
    repulsion = (
        ion_ion_energy if ion_ion_energy_builder is None else ion_ion_energy_builder
    )(atoms, pseudopotentials)
    atomic_reference_correction = float(
        sum(
            problem.pseudopotentials[atom.symbol].atomic_energy_correction
            for atom in atoms
        )
    )
    ion_ion_seconds = time.perf_counter() - stage_start

    # 6. Static atomic tail of the Hartree boundary values, where the plan
    # made from the geometry has one.  Only the Hartree solver reads it.
    boundary_tail = None
    boundary_seconds = 0.0 if boundary_plan is None else boundary_plan.seconds
    if boundary_plan is not None and boundary_plan.atomic_tail and build_atomic_tail:
        boundary_tail = build_reference_atomic_tail(
            grid,
            *valence_point_charges(atoms, pseudopotentials, electron_count),
            boundary_plan.order,
        )
        boundary_seconds += boundary_tail.seconds
    return (
        dict(
            initial_density=initial_density,
            core_density=core_density,
            ion_ion_energy=repulsion,
            atomic_reference_correction=atomic_reference_correction,
            hartree_boundary_tail=boundary_tail,
            initial_spin_densities=spin_densities,
        ),
        dict(
            initial_density_seconds=initial_density_seconds,
            core_density_seconds=core_density_seconds,
            ion_ion_seconds=ion_ion_seconds,
            hartree_boundary_seconds=boundary_seconds,
        ),
    )


def complete_single_point(
    system: PreparedSinglePointSystem,
    *,
    local_ionic_builder: Callable[..., np.ndarray] | None = None,
    atomic_density_builder: Callable[..., np.ndarray] | None = None,
    build_atomic_tail: bool = True,
    ion_ion_energy_builder: Callable[..., float] | None = None,
) -> PreparedSinglePointSystem:
    """Add what ``orbital_operators_only`` left out of ``system``.

    The local ionic potential, both densities and the two ionic energies are
    built by the calls :func:`prepare_single_point` makes for them, from the
    grid, atoms and pseudopotentials of ``system`` and with the same optional
    builders, so each equals the field of a complete preparation bit for bit.
    None of them reads the kinetic operator or the projectors, and nothing
    before :func:`run_scf` reads them: a caller may run this on another
    thread beside the set-up that follows the preparation.

    ``ion_ion_energy_builder`` stands for :func:`ion_ion_energy` where it is
    given: it is called with the atoms and the pseudopotentials of ``system``
    at the place of that stage and must return its value.  A caller that has
    the energy computed elsewhere meanwhile hands it over through this hook;
    ``ion_ion_seconds`` is then the time the hook took, not that of the sum.

    The static atomic tail of the Hartree boundary is built with them where
    the plan of ``system`` has one, unless ``build_atomic_tail`` is false.

    The returned system shares every other field with ``system``.  The times
    of its stages are measured here.  ``total_seconds`` stays the wall
    time of the call that prepared ``system``: this call may have run beside
    other work of the caller, who alone knows what it added to the wall time.
    """
    if system.ionic_potential is not None:
        raise ValueError("the system already has the fields of the SCF loop")
    started = time.perf_counter()
    problem = system.input
    build_local = (
        build_local_ionic_potential
        if local_ionic_builder is None
        else local_ionic_builder
    )
    ionic_potential = build_local(
        system.grid, system.atoms, system.pseudopotentials, problem.pseudopotentials
    )
    local_ionic_seconds = time.perf_counter() - started
    fields, stage_seconds = _density_and_energy_stages(
        problem,
        system.grid,
        system.atoms,
        system.pseudopotentials,
        system.electron_count,
        atomic_density_builder,
        system.hartree_boundary,
        build_atomic_tail,
        ion_ion_energy_builder,
    )
    return replace(
        system,
        ionic_potential=ionic_potential,
        **fields,
        timings=replace(
            system.timings,
            local_ionic_seconds=local_ionic_seconds,
            **stage_seconds,
        ),
    )


def update_geometry(
    system: PreparedSinglePointSystem,
    atoms,
    *,
    local_ionic_builder: Callable[..., np.ndarray] | None = None,
    nonlocal_projector_builder: Callable[..., NonlocalProjectorOperator] | None = None,
    atomic_density_builder: Callable[..., np.ndarray] | None = None,
    build_atomic_tail: bool = True,
) -> PreparedSinglePointSystem:
    """Return ``system`` with its atoms at new positions on the same grid.

    This is what ``parsec.F90`` rebuilds after ``domove`` or ``moldyn``:
    ``isolat`` (every nucleus must still be inside the domain), ``nonloc``
    (KB projectors), ``corecd`` (frozen core density), ``ionpot`` (local
    ionic potential) and ``forceion`` (ion-ion energy).  The grid, the
    kinetic operator, the pseudopotentials and the electron count are shared
    with ``system``, and the positions are taken as they are: there is no
    second recentering.

    Two things are specific to this code.  The plan of the Hartree boundary
    keeps the order it chose for the first structure, so that the energies of
    a relaxation do not jump when an estimate crosses a threshold; its atomic
    tail, where it has one, is rebuilt for the new positions.  And a
    superposed atomic starting density is built for the new positions, also
    when the first structure started from another source: a moved structure
    normally continues from :class:`SCFWarmStart` and reads it only when it
    is run cold.

    The optional builders are those of :func:`prepare_single_point`.
    """

    started = time.perf_counter()
    atoms = tuple(atoms)
    if len(atoms) != len(system.atoms) or any(
        new.symbol != old.symbol for new, old in zip(atoms, system.atoms)
    ):
        raise ValueError("moved atoms must keep their number, order and species")
    problem = replace(
        system.input,
        atoms=atoms,
        recenter_geometry=False,
        initial_density_settings=InitialDensitySettings(),
    )
    grid = system.grid
    _require_atoms_inside_domain(problem.grid, atoms)
    pseudopotentials = system.pseudopotentials

    stage_start = time.perf_counter()
    build_nonlocal = (
        build_nonlocal_projectors
        if nonlocal_projector_builder is None
        else nonlocal_projector_builder
    )
    nonlocal_operator = build_nonlocal(
        grid, atoms, pseudopotentials, problem.pseudopotentials
    )
    if nonlocal_projector_builder is not None:
        _require_core_radius_support(
            nonlocal_operator, grid, atoms, pseudopotentials
        )
    nonlocal_ionic_seconds = time.perf_counter() - stage_start
    if system.ionic_potential is None:
        # A system prepared with ``orbital_operators_only`` applies the
        # Hamiltonian for an SCF loop that runs elsewhere: it has projectors
        # to move and nothing else.
        return replace(
            system,
            input=problem,
            atoms=atoms,
            nonlocal_operator=nonlocal_operator,
            timings=PreparationTimings(
                nonlocal_ionic_seconds=nonlocal_ionic_seconds,
                total_seconds=time.perf_counter() - started,
            ),
        )

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

    boundary_plan = system.hartree_boundary
    if boundary_plan is not None:
        # The seconds of the plan belong to the preparation that made it.
        boundary_plan = replace(boundary_plan, seconds=0.0)
    fields, stage_seconds = _density_and_energy_stages(
        problem,
        grid,
        atoms,
        pseudopotentials,
        system.electron_count,
        atomic_density_builder,
        boundary_plan,
        build_atomic_tail,
    )
    return replace(
        system,
        input=problem,
        atoms=atoms,
        ionic_potential=ionic_potential,
        nonlocal_operator=nonlocal_operator,
        **fields,
        timings=PreparationTimings(
            local_ionic_seconds=local_ionic_seconds,
            nonlocal_ionic_seconds=nonlocal_ionic_seconds,
            **stage_seconds,
            total_seconds=time.perf_counter() - started,
        ),
        hartree_boundary=boundary_plan,
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
    if (
        requested <= 0.5 * system.electron_count
        and system.input.scf.fermi_temperature_kelvin > 0
        and system.input.scf.fixed_occupations is None
    ):
        raise ValueError(
            "number_of_states must leave at least one unoccupied state at finite temperature"
        )
    if requested + system.input.eigensolver.subspace_buffer >= system.grid.size:
        raise ValueError("the grid is too small for the requested states and eigensolver buffer")
    return requested


def eigval_settings_for(eigensolver_settings, filter_degree: int) -> EigvalSettings:
    """Assemble the eigensolver policy of one SCF iteration.

    ``filter_degree`` is the present later-SCF SUBSPACE degree, which the
    SCF loop lowers near convergence; everything else comes from the input.
    """

    return EigvalSettings(
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
    warm_start: SCFWarmStart | None = None,
) -> SinglePointResult:
    """Run PARSEC-style potential mixing on a prepared system.

    ``warm_start`` continues from the SCF loop of a previous structure (see
    :class:`SCFWarmStart`); without it the loop starts from the prepared
    initial density.

    A spin-polarized system (``Spin_Polarization``) is handed to
    :func:`parsec_python.SCF.spin_polarized.run_spin_scf`, which runs the
    same loop with one potential and one eigenproblem per spin channel.

    At iteration ``k`` the eigensolver sees

    ``H[V_in^k] = -nabla_FD^2 + diag(V_in^k) + V_NL``

    with ``V_in^k = V_ion,local + V_H^in + V_xc^in``.  Its orbitals produce a
    new density and therefore an output field

    ``V_out^k = V_ion,local + V_H[rho^k] + V_xc[rho^k + rho_core]``.

    The mixer constructs ``V_in^(k+1)`` from these two local fields.  Kinetic
    and nonlocal terms are never mixed because they are density independent.
    """
    if getattr(system.input.scf, "spin_polarized", False):
        from .spin_polarized import run_spin_scf

        return run_spin_scf(
            system,
            callback=callback,
            eigenproblem_solver=eigenproblem_solver,
            orbital_density_builder=orbital_density_builder,
            mixer_factory=mixer_factory,
            residual_metrics_evaluator=residual_metrics_evaluator,
            total_energy_evaluator=total_energy_evaluator,
            scalar_field_adapter=scalar_field_adapter,
            warm_start=warm_start,
        )
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
    # 2. (below) Initialize state carried between nonlinear iterations.
    # eigval_state is None only for the first diagonalization: solve_eigval
    # then runs the selected CHEBFF/CHEBDAV routine and stores its complete
    # buffered Ritz subspace.  Later calls feed that state to one cheaper
    # SUBSPACE filter.
    eigval_state: object | None = None
    if warm_start is None:
        density = to_internal_scalar(system.initial_density.copy())
        # ``-V_ion`` below is only the numerical starting vector for the
        # first Poisson solve; it is not an identity between ionic and
        # Hartree fields.
        stage_start = time.perf_counter()
        initial_hartree = system.solve_hartree(
            density, initial_potential=-ionic_potential
        )
        initial_hartree_seconds = time.perf_counter() - stage_start
        hartree_potential = initial_hartree.potential
        # XC sees rho_valence + frozen rho_core (CA-LDA or PBE, as selected).
        # The Hartree solve immediately above sees rho_valence only.
        stage_start = time.perf_counter()
        xc = system.evaluate_xc(density)
        initial_xc_seconds = time.perf_counter() - stage_start
        xc_potential = xc.potential
        # This array is the diagonal, local part of H.  V_NL is intentionally
        # not folded into it; PreparedSinglePointSystem.hamiltonian adds V_NL
        # as an operator when the eigensolver requests H@vector.
        input_potential = (
            ionic_potential + hartree_potential + xc_potential
        )
    else:
        # The structure before the move supplies the first input potential,
        # the starting vector of the first Poisson solve and the subspace of
        # the first eigenproblem: no density is needed before the first
        # eigensolve, hence no initial Hartree or XC evaluation.
        initial_hartree_seconds = 0.0
        initial_xc_seconds = 0.0
        hartree_potential = to_internal_scalar(
            np.array(warm_start.hartree_potential, dtype=np.float64)
        )
        input_potential = to_internal_scalar(
            np.array(warm_start.input_potential, dtype=np.float64)
        )
        eigval_state = warm_start.eigensolver_state
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
        eigval_settings = eigval_settings_for(eigensolver_settings, filter_degree)
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
        # A memory-adaptive accelerated symmetry result may keep the preceding
        # iteration's sector arrays only long enough to construct rho.  Once a
        # successor solve begins those wrapper references are no longer
        # physical state (the eigensolver owns its exact restart state), so
        # release them before allocating the new filtered subspace.
        release_previous_orbitals = getattr(
            wavefunctions, "release_intermediate_storage", None
        )
        if callable(release_previous_orbitals):
            release_previous_orbitals()
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
        # With the occupations of occup.in (a negative Fermi_Temp) the
        # n-th lowest state takes the n-th of them and no level is searched.
        occupation_result = (
            fermi_occupations(
                eigenvalues,
                system.electron_count,
                settings.fermi_temperature_kelvin,
            )
            if settings.fixed_occupations is None
            else fixed_occupations(
                eigenvalues,
                system.electron_count,
                settings.fixed_occupations,
            )
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
            alpha_z_energy=getattr(system, "alpha_z_energy", 0.0),
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
        eigensolver_state=eigval_state,
    )
