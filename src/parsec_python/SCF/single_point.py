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

from dataclasses import dataclass, field, is_dataclass, replace
import os
import time
from typing import Callable, Mapping
import warnings

import numpy as np
import scipy.sparse as sp

from ..Eigensolvers import (
    DYNAMIC_TOLERANCE_START,
    ArpackSettings,
    ChebDavSettings,
    ChebFFSettings,
    DiaglaSettings,
    EigvalSettings,
    EigvalState,
    SubspaceSettings,
    dynamic_diagonalization_tolerance,
    solve_eigval,
)
from ..Embedding.frozen_density import embedded_exchange_correlation, embedding_energies
from ..Energy import total_energy
from ..Energy.total_energy import with_dispersion
from ..Grid import (
    GridPointGroup,
    RealSpaceGrid,
    build_cluster_grid,
    grid_point_group,
    nearest_neighbor_rows,
    symmetrize_density,
)
from ..Hamiltonian import KohnShamHamiltonian
from ..Hamiltonian.hubbard import (
    HubbardProjectors,
    build_hubbard_projectors,
    hubbard_operator,
    hubbard_terms_from_orbitals,
)
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
from ..Mixer import make_mixer, potential_residual_metrics
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
    check_core_density,
    cluster_centering_shift,
    ion_ion_energy,
    ionic_charge,
    load_pseudopotentials,
    normalize_density,
    point_charge_ion_energy,
    point_charge_potential,
    shifted_point_charges,
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
from ..V_xc.selection import (
    as_spin_polarized_result,
    evaluate_spin_xc as evaluate_spin_functional,
    evaluate_unpolarized_xc,
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

    ``point_charges`` are the point charges of the input where the atoms
    are, moved with them when the cluster was recentred (``initial.f90``):
    their potential is part of ``ionic_potential`` and their energy with the
    ions part of ``ion_ion_energy`` (:mod:`parsec_python.V_ion.external`).
    ``density_symmetry`` is, for ``Symmetrize_Charge_Density``, the point
    group of the atoms as permutations of the grid, which every new density
    of the SCF loop is averaged over; ``None`` otherwise.

    ``hubbard`` holds the LDA+U projectors of the atoms with ``ldaplusu_u``
    or ``ldaplusu_j`` (:mod:`parsec_python.Hamiltonian.hubbard`) and is
    ``None`` without LDA+U.

    ``embedding`` is, for FDET (``Potential_Field``), the frozen sample
    field on the grid (:class:`parsec_python.Embedding.frozen_density.FrozenSampleField`):
    its potential is part of ``ionic_potential``, and the point charges then
    enter the ion-ion energy only (``parsec.F90:1022-1028``); ``None``
    without FDET.
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
    point_charges: tuple = ()
    density_symmetry: GridPointGroup | None = None
    hubbard: HubbardProjectors | None = None
    embedding: object | None = None

    @property
    def hartree_boundary_tail_maximum(self) -> float | None:
        """``max |C_L|`` of the atomic tail in Ry, ``None`` without one."""

        tail = self.hartree_boundary_tail
        return None if tail is None else tail.maximum

    def hamiltonian(
        self,
        effective_potential: np.ndarray,
        hubbard: object | None = None,
        exchange: object | None = None,
    ) -> KohnShamHamiltonian:
        """Compose the current SCF Hamiltonian without forming a dense matrix.

        The supplied array is only the diagonal local field

        ``V_eff = V_ion,local + V_H + V_xc``.

        :class:`KohnShamHamiltonian` binds it to the already prepared kinetic
        and nonlocal terms, giving

        ``H = -nabla_FD^2 + diag(V_eff) + V_NL``,

        plus the LDA+U term ``hubbard`` of the channel where it is given and
        the exact-exchange term ``a_x V_ACE`` of a hybrid functional
        (``exchange``, :mod:`parsec_python.SCF.hybrid`).
        """
        if exchange is not None:
            return KohnShamHamiltonian(
                self.negative_laplacian,
                effective_potential,
                self.nonlocal_operator,
                hubbard,
                exchange,
            )
        if hubbard is None:
            return KohnShamHamiltonian(
                self.negative_laplacian,
                effective_potential,
                self.nonlocal_operator,
            )
        return KohnShamHamiltonian(
            self.negative_laplacian,
            effective_potential,
            self.nonlocal_operator,
            hubbard,
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
        """Evaluate the selected XC functional including frozen NLCC.

        For a hybrid functional this is its semilocal part (the reduced
        semilocal exchange and the correlation); the exact exchange is the
        operator of :mod:`parsec_python.SCF.hybrid`.
        """

        if getattr(self.input.scf, "is_hybrid", False):
            from .hybrid import scf_semilocal_xc

            return scf_semilocal_xc(self.input.scf, density, self.grid, self.core_density)
        if self.input.scf.xc_functional == "ca":
            return ca_lda(density, self.grid.volume_element, self.core_density)
        if self.input.scf.xc_functional == "pbe":
            return pbe(density, self.grid, self.core_density)
        return evaluate_unpolarized_xc(
            self.input.scf.xc_functional,
            density,
            self.grid,
            self.core_density,
            ionization_energy_difference=self.input.scf.ionization_energy_difference,
        )

    def evaluate_xc_spin_polarized(
        self, density_up: np.ndarray, density_down: np.ndarray
    ) -> SpinPolarizedXCResult:
        """Collinear spin-polarized XC (LSDA or PBE) including frozen NLCC."""

        if getattr(self.input.scf, "is_hybrid", False):
            return as_spin_polarized_result(
                self.evaluate_spin_xc(density_up, density_down)
            )
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
        return as_spin_polarized_result(
            evaluate_spin_functional(
                functional,
                density_up,
                density_down,
                self.grid,
                self.core_density,
                ionization_energy_difference=self.input.scf.ionization_energy_difference,
            )
        )

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
        density (``rho_hart.F90``).  For a hybrid functional, its semilocal
        part.
        """

        if getattr(self.input.scf, "is_hybrid", False):
            from .hybrid import scf_semilocal_spin_xc

            return scf_semilocal_spin_xc(
                self.input.scf, density_up, density_down, self.grid, self.core_density
            )
        if self.input.scf.xc_functional == "ca":
            return ca_lda_spin(
                density_up, density_down, self.grid.volume_element, self.core_density
            )
        if self.input.scf.xc_functional == "pbe":
            return pbe_spin(density_up, density_down, self.grid, self.core_density)
        return evaluate_spin_functional(
            self.input.scf.xc_functional,
            density_up,
            density_down,
            self.grid,
            self.core_density,
            ionization_energy_difference=self.input.scf.ionization_energy_difference,
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
    potential is common to both channels.  The spinor loops take one
    eigensubspace (a ``{k-point: state}`` dict for a k-point run) and, for
    collinear spin, ``input_potential`` as the pair ``(V_up, V_down)``, for
    non-collinear magnetism as ``(V, B_xc)`` with ``B_xc`` of shape ``(3, n)``.

    ``density_guess`` and ``hxc_reference`` carry the atomic extrapolation
    (:func:`warm_start_for_moved_atoms`): the loop then adds to
    ``input_potential`` the Hartree and exchange-correlation potentials of
    ``density_guess`` on the moved structure and subtracts ``hxc_reference``,
    the Hartree plus exchange-correlation part of the last output potential
    before the move.  Without them the first input potential is PARSEC's.
    ``first_iteration_degree`` is the filter degree of the first iteration,
    ``None`` for the later degree of the input.  An extrapolated potential is
    close to the one of the moved atoms while the saved orbitals still belong
    to the old structure; one stronger filter lets them catch up before the
    mixer sees their density.

    ``exchange_orbitals`` are, for a hybrid functional, the orbitals of the
    structure before (one :class:`parsec_python.SCF.hybrid.OrbitalSet` per
    spin channel set): the first exchange operator of the Fock loop is built
    from them instead of a semilocal start.  ``None`` otherwise.

    ``mixer`` is a mixer whose history this loop continues instead of a fresh
    one (the inner loops of a hybrid's Fock loop,
    :class:`parsec_python.Mixer.carried.CarriedAndersonMixer`); ``None``
    starts afresh, as PARSEC does.  ``relative_criterion`` lets the loop also
    stop once its residual fell below that share of its first residual (an
    inexact inner loop of the Fock loop, :func:`loop_criterion`); ``None``
    keeps the stop test of the input.  ``level_tolerance`` (Ry): once the loop
    has converged, one more iteration solves the same ``H[V_in]`` until no
    level moves by more than that (:func:`refine_levels`) and takes its
    density and energies from those eigenpairs; ``None`` stops at the
    converged iteration, as PARSEC does.
    """

    input_potential: np.ndarray
    hartree_potential: np.ndarray
    eigensolver_state: object | None = None
    density_guess: np.ndarray | None = None
    hxc_reference: np.ndarray | None = None
    first_iteration_degree: int | None = None
    exchange_orbitals: tuple | None = None
    mixer: object | None = None
    relative_criterion: float | None = None
    level_tolerance: float | None = None


#: Eigensolver passes of a level refinement (``SCFWarmStart.level_tolerance``) at most.
LEVEL_PASSES = 40


def refine_levels(solve, operator, count: int, settings, state, eigenvalues, vectors, tolerance: float, *, release=True):
    """More eigensolver passes with the Hamiltonian of a converged loop's last iteration.

    ``operator`` is that Hamiltonian (``H[V_in]`` of the last iteration, with
    its fixed terms); each pass is one call of ``solve`` on the loop's
    subspace ``state``.  The passes stop once no one of the ``count`` levels
    moved by more than ``tolerance`` (Ry) in the last one, after
    ``LEVEL_PASSES`` at most: the eigenpairs that the stop test of the loop
    does not see (the empty levels).  ``vectors`` are the orbitals the first
    pass replaces (``None``: already released).  Returns ``(solution,
    passes, seconds, change)``, ``change`` the largest move of a level in
    the last pass (Ry).
    """

    started = time.perf_counter()
    previous = np.asarray(eigenvalues, dtype=np.float64)
    solution = None
    passes = 0
    change = float("inf")
    while passes < LEVEL_PASSES:
        drop = getattr(vectors, "release_intermediate_storage", None) if release else None
        if callable(drop):
            # The views of the last orbitals are no state once the next solve begins.
            drop()
        solution = solve(operator, count, settings=settings, state=state)
        passes += 1
        state, vectors = solution.state, solution.vectors
        values = np.asarray(solution.eigenvalues, dtype=np.float64)
        change = (
            float(np.max(np.abs(values - previous), initial=0.0))
            if values.shape == previous.shape
            else float("inf")
        )
        previous = values
        if change < tolerance:
            break
    return solution, passes, time.perf_counter() - started, change


def warm_start_mixer(warm_start: SCFWarmStart | None, build: Callable[[], object]) -> object:
    """The mixer of a loop: the one ``warm_start`` carries, or ``build()``."""

    carried = getattr(warm_start, "mixer", None)
    return build() if carried is None else carried


def loop_criterion(warm_start: SCFWarmStart | None, criterion: float, first_residual: float) -> float:
    """The stop threshold of an SCF loop (Ry): ``criterion``, or for a loop with
    ``warm_start.relative_criterion`` ``max(criterion, share * first_residual)``,
    ``first_residual`` the selected residual of the loop's first iteration."""

    share = getattr(warm_start, "relative_criterion", None)
    if share is None:
        return criterion
    return max(float(criterion), float(share) * float(first_residual))


@dataclass(frozen=True)
class SCFIterationFields:
    """The fields of one finished SCF iteration, after mixing.

    What ``parsec.F90`` l.1381-1393 writes to ``parsec.dat`` after every
    iteration with ``Save_Intermediate_Charge_Density``: the levels and
    orbitals of the iteration, its output density and the mixed potential
    of the next iteration (``vnew``), full-grid arrays in Ry and e/bohr^3.
    ``spin`` holds the two channels of a spin-polarized loop, up first; the
    other fields then join or average them as
    :class:`parsec_python.models.SinglePointResult` does.  A k-point loop
    sets ``k_points`` and ``k_weights``; its levels are then pooled over the
    k-points (k-major) and it keeps no orbitals.
    """

    iteration: int
    eigenvalues: np.ndarray
    occupations: np.ndarray
    representations: np.ndarray
    wavefunctions: object
    density: np.ndarray
    next_effective_potential: np.ndarray
    spin: tuple | None = None
    k_points: np.ndarray | None = None
    k_weights: np.ndarray | None = None


@dataclass(frozen=True)
class IterationChannelFields:
    """One spin channel of :class:`SCFIterationFields` (levels pooled over
    k-points where there are any; ``next_effective_potential`` is the mixed
    potential of the channel, Fortran's ``vnew``)."""

    eigenvalues: np.ndarray
    occupations: np.ndarray
    density: np.ndarray
    next_effective_potential: np.ndarray
    wavefunctions: object = None
    representations: tuple = ()


@dataclass(frozen=True)
class SpinorIterationFields:
    """The fields of one finished iteration of a spin-orbit or non-collinear loop.

    What ``parsec.F90`` l.1381-1402 writes after every iteration of such a
    loop (``wfnsave.F90`` with ``mxwd = 2``): one set of spinor levels
    (pooled over k-points, k-major, where ``k_points`` is set), the spinors
    of an isolated loop (``None`` for a k-point loop, which keeps none) and
    ``parsec_dat_fields``, the two spin-component potentials (after mixing)
    and densities in the layout of
    :class:`parsec_python.models.SelfConsistentSOCResult`.
    """

    iteration: int
    eigenvalues: np.ndarray
    occupations: np.ndarray
    spinors: object
    parsec_dat_fields: dict | None
    k_points: np.ndarray | None = None
    k_weights: np.ndarray | None = None


#: Ways to start the SCF loop of moved atoms (``Potential_Extrapolation``).
POTENTIAL_EXTRAPOLATIONS = ("none", "atomic", "second_order")


@dataclass
class DensityHistory:
    """What second-order extrapolation keeps of one trajectory.

    ``change`` is ``rho_out - rho_atoms`` of the structure before the last
    move and ``positions`` the atoms that move went to; the next move
    continues from it only when it starts from those atoms.
    """

    change: np.ndarray | None = None
    positions: np.ndarray | None = None


def _positions(system) -> np.ndarray:
    return np.array([atom.position for atom in system.atoms], dtype=np.float64)


def atomic_density_guess(previous_system, previous_result, system, history=None):
    """The density of the moved atoms by atomic extrapolation, or ``None``.

    ``rho_out - rho_atoms(R_old) + rho_atoms(R_new)``: the last output
    density before the move with the superposed atomic densities moved
    along with their nuclei.  The superposition of the old structure is
    scaled to the electron count of the new one, so that the correction
    carries no charge.  ``None`` when the structure before the move did not
    start from superposed atomic densities (an imported or predicted first
    density) or when there are spin channels.

    With a :class:`DensityHistory` the extrapolation is of second order in
    the change ``d = rho_out - rho_atoms``:
    ``rho_atoms(R_new) + 2 d(R_old) - d(R_older)`` (``pot_extrapolation =
    'second_order'`` of Quantum ESPRESSO), from the second move on.
    """

    if getattr(previous_result, "spin", None) is not None or isinstance(
        getattr(previous_result, "next_effective_potential", None), tuple
    ):  # spin channels or a magnetization: the guess is a total density only
        return None
    settings = getattr(getattr(previous_system, "input", None), "initial_density_settings", None)
    if settings is not None and getattr(settings, "method", "sad") != "sad":
        return None
    old_atoms = np.asarray(previous_system.initial_density, dtype=np.float64)
    new_atoms = np.asarray(system.initial_density, dtype=np.float64)
    old_charge = float(np.sum(old_atoms))
    scale = float(np.sum(new_atoms)) / old_charge if old_charge else 1.0
    change = np.asarray(previous_result.density, dtype=np.float64) - scale * old_atoms
    guess = new_atoms + change
    if history is not None:
        if (
            history.change is not None
            and history.change.shape == change.shape
            and np.array_equal(history.positions, _positions(previous_system))
        ):
            guess += change - history.change
        history.change, history.positions = change, _positions(system)
    # The output density can be smaller than the atomic density it loses
    # where an atom moved away: the guess is a density, so it is not negative.
    return np.maximum(guess, 0.0)


def extrapolated_warm_start(
    warm_start, previous_system, previous_result, system, extrapolation, history=None
):
    """``warm_start`` with the extrapolation ``extrapolation`` added.

    ``history`` is the :class:`DensityHistory` of a second-order
    extrapolation; without one it is of first order.
    """

    if extrapolation not in POTENTIAL_EXTRAPOLATIONS:
        raise ValueError(
            "Potential_Extrapolation must be one of " + ", ".join(POTENTIAL_EXTRAPOLATIONS)
        )
    if extrapolation == "none":
        return warm_start
    guess = atomic_density_guess(
        previous_system,
        previous_result,
        system,
        history if extrapolation == "second_order" else None,
    )
    if guess is None:
        return warm_start
    eigensolver = system.input.eigensolver
    return replace(
        warm_start,
        density_guess=guess,
        hxc_reference=(
            np.asarray(previous_result.output_effective_potential, dtype=np.float64)
            - np.asarray(previous_system.ionic_potential, dtype=np.float64)
        ),
        # The degree of the first solve of the input (Chebdav_Degree) for the
        # first iteration: measured on nanodiamonds C35H36 and C87H76, the
        # extrapolation alone saved nothing, together with it a quarter to a
        # third of the SCF iterations of a relaxation and of MD.
        first_iteration_degree=max(
            int(eigensolver.filter_degree), int(eigensolver.first_filter_degree)
        ),
    )


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
    if isinstance(state, dict):  # the states of the k-points of a periodic run, by k-point
        return {task: state_for_new_potential(value) for task, value in state.items()}
    restart = getattr(state, "for_new_scf_loop", None)
    if callable(restart):
        return restart()
    if isinstance(state, EigvalState):
        return replace(state, subspace=replace(state.subspace, first_filter=True))
    if is_dataclass(state) and hasattr(state, "first_filter"):  # the device subspace of a spinor eigensolver
        return replace(state, first_filter=True)
    return state


def warm_start_for_moved_atoms(
    previous_system: "PreparedSinglePointSystem",
    previous_result: SinglePointResult,
    system: "PreparedSinglePointSystem",
    *,
    extrapolation: str = "none",
    history: DensityHistory | None = None,
) -> SCFWarmStart:
    """Build the warm start of ``system`` from the run before the move.

    ``parsec.F90`` after ``domove``: ``vnew = vnew - vion_old`` and then
    ``vnew = vnew + vion_new``, with ``vnew`` the potential the mixer left,
    for each spin channel where there are two.  ``extrapolation="atomic"``
    adds the change of the Hartree and exchange-correlation potentials that
    the superposed atomic densities make when they move with their nuclei
    (:func:`atomic_density_guess`); ``"second_order"`` extrapolates the
    density further along the trajectory that ``history`` follows.
    """

    warm_start = _parsec_warm_start(previous_system, previous_result, system)
    if getattr(previous_result, "exact_exchange", None) is not None:
        # A hybrid functional: the orbitals before the move start the Fock loop.
        from .hybrid import exchange_start_orbitals

        warm_start = replace(
            warm_start, exchange_orbitals=exchange_start_orbitals(previous_result)
        )
    if getattr(system, "embedding", None) is not None:
        # FDET of PARSEC 1.5: translation by whole grid steps, or the tip sphere
        # (parsec_python.Embedding.warm_start); FDET takes no extrapolation.
        from ..Embedding.warm_start import embedding_warm_start

        return embedding_warm_start(previous_system, previous_result, system, warm_start)
    return extrapolated_warm_start(
        warm_start,
        previous_system,
        previous_result,
        system,
        extrapolation,
        history,
    )


def extrapolation_keywords(extrapolation: str) -> dict:
    """Keywords of :func:`warm_start_for_moved_atoms` for one ionic loop.

    A second-order extrapolation gets a fresh :class:`DensityHistory`.
    """

    if extrapolation not in POTENTIAL_EXTRAPOLATIONS:
        raise ValueError(
            "Potential_Extrapolation must be one of " + ", ".join(POTENTIAL_EXTRAPOLATIONS)
        )
    if extrapolation == "second_order":
        return {"extrapolation": extrapolation, "history": DensityHistory()}
    return {"extrapolation": extrapolation}


def _parsec_warm_start(previous_system, previous_result, system) -> SCFWarmStart:

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
    potential = previous_result.next_effective_potential
    if isinstance(potential, tuple):
        # spinor orbitals with spin: the pair (V_up, V_down) of a collinear loop, or (V, B_xc) of a non-collinear one,
        # whose field (a 2-D array) belongs to the moments and does not move with the ionic potential
        swap = np.asarray(system.ionic_potential) - np.asarray(previous_system.ionic_potential)
        return SCFWarmStart(
            input_potential=tuple(
                part if np.ndim(part) > 1 else np.asarray(part) + swap for part in potential
            ),
            hartree_potential=np.asarray(previous_result.hartree_potential),
            eigensolver_state=state_for_new_potential(previous_result.eigensolver_state),
        )
    return SCFWarmStart(
        input_potential=(
            np.asarray(potential)
            - np.asarray(previous_system.ionic_potential)
            + np.asarray(system.ionic_potential)
        ),
        hartree_potential=np.asarray(previous_result.hartree_potential),
        eigensolver_state=state_for_new_potential(previous_result.eigensolver_state),
    )


def placed_point_charges(problem: SinglePointInput) -> tuple:
    """The point charges of ``problem`` moved as its atoms are placed.

    ``initial.f90`` shifts the point charges by the translation that centres
    the bounding box of the atoms, where it moves the atoms.
    """

    charges = tuple(getattr(problem, "point_charges", ()))
    if not charges or not problem.recenter_geometry:
        return charges
    shift = cluster_centering_shift(problem.atoms)
    return charges if shift is None else shifted_point_charges(charges, shift)


def _with_point_charge_potential(ionic_potential, grid, point_charges):
    """``ionic_potential`` plus the potential of the point charges (``ionpot.f90``)."""

    if ionic_potential is None or not point_charges:
        return ionic_potential
    return ionic_potential + point_charge_potential(grid.coordinates, point_charges)


def _embedding_field(problem: SinglePointInput, grid: RealSpaceGrid, atoms=(), pseudopotentials=None, tip=None):
    """The frozen sample field of an FDET input on ``grid``, ``None`` without FDET.

    ``tip`` is ``(S, spot-frame point charges)`` of the tip sphere
    (:func:`_tip_sphere_frame`); ``atoms`` are then in the tip frame.
    """

    settings = getattr(problem, "embedding", None)
    if settings is None:
        return None
    if tip is not None:
        from ..Embedding.tip_sphere import tip_sphere_field

        offset, spot_charges = tip
        return tip_sphere_field(problem, grid, atoms, pseudopotentials, offset, spot_point_charges=spot_charges)
    from ..Embedding.frozen_density import frozen_sample_field

    return frozen_sample_field(settings, grid)


def _tip_sphere_frame(problem: SinglePointInput, atoms, point_charges):
    """Atoms and point charges of an FDET tip sphere in the tip frame (``tipsphere_recenter``).

    Returns ``(atoms, point_charges, tip)`` with ``tip = (S, spot-frame
    point charges)``; unchanged with ``tip = None`` without the tip sphere.
    """

    settings = getattr(problem, "embedding", None)
    if settings is None or not settings.tip_sphere:
        return atoms, point_charges, None
    from ..Embedding.tip_sphere import place_tip

    moved_atoms, moved_charges, offset, spot_charges = place_tip(atoms, point_charges, problem.grid.spacing)
    return moved_atoms, moved_charges, (offset, spot_charges)


def _with_external_fields(ionic_potential, grid, point_charges, embedding):
    """The local ionic potential with the point charges or the FDET sample field.

    Without FDET the potential of the point charges is added (``ionpot.f90``).
    With FDET the sample ions act through the frozen field, which is added
    instead (``parsec.F90:1022-1028, 1087-1089``); the point charges then
    enter the ion-ion energy and forces only.
    """

    if embedding is None:
        return _with_point_charge_potential(ionic_potential, grid, point_charges)
    if ionic_potential is None:
        return None
    from ..Embedding.frozen_density import with_sample_potential

    return with_sample_potential(ionic_potential, embedding)


def _with_point_charge_energy(fields: dict, atoms, pseudopotentials, point_charges) -> dict:
    """Add ``E_QN`` to the ion-ion energy of ``fields`` (``forceion.f90``)."""

    if point_charges:
        fields["ion_ion_energy"] = float(fields["ion_ion_energy"]) + point_charge_ion_energy(
            atoms, pseudopotentials, point_charges
        )
    return fields


def density_point_group(grid: RealSpaceGrid, atoms, candidates=None) -> GridPointGroup:
    """The point group of ``atoms`` as permutations of ``grid``.

    The operations are those of :func:`parsec_python.Forces.symmetry.atom_symmetry_operations`,
    among ``candidates`` (matrices) where they are given.
    """

    from ..Forces.symmetry import atom_symmetry_operations

    operations = atom_symmetry_operations(atoms, candidates=candidates)
    return grid_point_group(grid, np.array([operation.matrix for operation in operations]))


def _require_atoms_inside_domain(settings: GridSettings, atoms) -> None:
    """Refuse a nucleus outside the sphere or box (``isolat.f90``)."""

    for atom in atoms:
        coordinate = np.asarray(atom.position)
        if settings.domain_shape == "sphere":
            inside = np.dot(coordinate, coordinate) <= settings.radius**2
        elif settings.domain_shape in ("ellipsoid", "cylinder"):
            from ..Grid.cluster import _inside_domain

            inside = bool(_inside_domain(settings, coordinate[None, :])[0])
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
    point_charges = placed_point_charges(problem)
    # FDET tip sphere: the tip at the centre of its own sphere (tipsphere_recenter).
    atoms, point_charges, tip = _tip_sphere_frame(problem, atoms, point_charges)
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
    # FDET: the frozen sample field on this grid (potfield.F90, tipsphere_field), None without.
    embedding = _embedding_field(problem, grid, atoms, pseudopotentials, tip)

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
    ionic_potential = _with_external_fields(ionic_potential, grid, point_charges, embedding)
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
    # The LDA+U projectors of upot.F90 (None without ldaplusu_u/j).
    hubbard = build_hubbard_projectors(
        grid, atoms, pseudopotentials, problem.pseudopotentials
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
            point_charges=point_charges,
            hubbard=hubbard,
            embedding=embedding,
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
    _with_point_charge_energy(fields, atoms, pseudopotentials, point_charges)
    density_symmetry = (
        density_point_group(grid, atoms) if problem.scf.symmetrize_density else None
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
        point_charges=point_charges,
        density_symmetry=density_symmetry,
        hubbard=hubbard,
        embedding=embedding,
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
    # corecd.f90: small negative values are set to zero, large ones stop.
    core_density = check_core_density(core_density)
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
    ionic_potential = _with_external_fields(
        ionic_potential, system.grid, system.point_charges, getattr(system, "embedding", None)
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
    _with_point_charge_energy(
        fields, system.atoms, system.pseudopotentials, system.point_charges
    )
    return replace(
        system,
        ionic_potential=ionic_potential,
        density_symmetry=(
            density_point_group(system.grid, system.atoms)
            if problem.scf.symmetrize_density
            else None
        ),
        **fields,
        timings=replace(
            system.timings,
            local_ionic_seconds=local_ionic_seconds,
            **stage_seconds,
        ),
    )


def geometry_thread_setting() -> str:
    """``PARSEC_GEOMETRY_THREAD``: ``on`` (default) or ``off``.

    ``on`` builds the projectors of moved atoms on a host thread while the
    local potential and the densities are made; ``off`` one after another.
    Neither changes a number.
    """

    value = os.environ.get("PARSEC_GEOMETRY_THREAD", "on").strip().lower() or "on"
    if value in {"1", "true", "yes"}:
        value = "on"
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in {"on", "off"}:
        raise ValueError("PARSEC_GEOMETRY_THREAD must be on or off")
    return value


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
    embedding = getattr(system, "embedding", None)
    point_charges = system.point_charges
    if embedding is not None and embedding.tip_sphere is not None:
        # FDET tip sphere: ``atoms`` are in the spot frame (manual.dat); the
        # tip goes back to the centre of its sphere and the field moves
        # (parsec.F90:2614-2637, 2794-2800).
        from ..Embedding.tip_sphere import moved_tip

        atoms, point_charges, embedding = moved_tip(system, atoms)
    problem = replace(
        system.input,
        atoms=atoms,
        recenter_geometry=False,
        initial_density_settings=InitialDensitySettings(),
    )
    grid = system.grid
    _require_atoms_inside_domain(problem.grid, atoms)
    pseudopotentials = system.pseudopotentials

    build_nonlocal = (
        build_nonlocal_projectors
        if nonlocal_projector_builder is None
        else nonlocal_projector_builder
    )

    def projectors():
        stage_start = time.perf_counter()
        operator = build_nonlocal(grid, atoms, pseudopotentials, problem.pseudopotentials)
        if nonlocal_projector_builder is not None:
            _require_core_radius_support(operator, grid, atoms, pseudopotentials)
        return operator, time.perf_counter() - stage_start

    # upot.F90 is called again after every move: new LDA+U projectors.
    hubbard = build_hubbard_projectors(
        grid, atoms, pseudopotentials, problem.pseudopotentials
    )

    pending = None
    if system.ionic_potential is not None and geometry_thread_setting() == "on":
        # The projectors are built on the host beside the local potential and
        # the densities, which the accelerated builders make on a device.
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parsec-projectors")
        pending = executor.submit(projectors)
        executor.shutdown(wait=False)
    else:
        nonlocal_operator, nonlocal_ionic_seconds = projectors()
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
            hubbard=hubbard,
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
    # The point charges stay where they are (parsec.F90 calls ionpot and
    # forceion again after the move); so does a classical FDET field
    # (parsec.F90:2780-2806 reads the same pot.dat again).
    ionic_potential = _with_external_fields(ionic_potential, grid, point_charges, embedding)
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
    _with_point_charge_energy(fields, atoms, pseudopotentials, point_charges)
    if pending is not None:
        nonlocal_operator, nonlocal_ionic_seconds = pending.result()
    density_symmetry = system.density_symmetry
    if density_symmetry is not None:
        # PARSEC keeps the group of the first structure; here it is kept as
        # far as the moved atoms still have its operations.
        density_symmetry = density_point_group(
            grid, atoms, candidates=density_symmetry.matrices
        )
    return replace(
        system,
        input=problem,
        atoms=atoms,
        ionic_potential=ionic_potential,
        nonlocal_operator=nonlocal_operator,
        density_symmetry=density_symmetry,
        **fields,
        timings=PreparationTimings(
            local_ionic_seconds=local_ionic_seconds,
            nonlocal_ionic_seconds=nonlocal_ionic_seconds,
            **stage_seconds,
            total_seconds=time.perf_counter() - started,
        ),
        hartree_boundary=boundary_plan,
        hubbard=hubbard,
        **(
            {}
            if embedding is None
            else {"embedding": embedding, "point_charges": point_charges}
        ),
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


def iterative_solver_fields(
    eigensolver_settings,
    tolerance: float,
    neighbors: np.ndarray | None = None,
) -> dict:
    """The ARPACK/DIAGLA part of :class:`EigvalSettings` for one iteration.

    Empty for CHEBFF and CHEBDAV, so their settings are those of every
    earlier version.  ``tolerance`` is ``Diag_Tolerance`` or the dynamic
    tolerance of the iteration; ``neighbors`` is the grid neighbour array of
    DIAGLA's preconditioner (:func:`parsec_python.Grid.nearest_neighbor_rows`).
    """

    method = eigensolver_settings.method
    if method == "arpack":
        return {
            "arpack": ArpackSettings(
                tolerance=tolerance,
                maximum_matvec=eigensolver_settings.maximum_matvec,
            ),
            "chebyshev_filtering": eigensolver_settings.filters_later_iterations,
        }
    if method == "diagla":
        return {
            "diagla": DiaglaSettings(
                tolerance=tolerance,
                maximum_matvec=eigensolver_settings.maximum_matvec,
                neighbors=neighbors,
            ),
            "chebyshev_filtering": eigensolver_settings.filters_later_iterations,
        }
    if method == "none":
        raise ValueError(
            "Eigensolver: none is PARSEC's TEST mode: setup only, no SCF"
        )
    return {}


def preconditioner_neighbors(system, eigensolver_settings) -> np.ndarray | None:
    """The first-shell grid neighbours DIAGLA's ``fdo5scl`` needs, else None."""

    if eigensolver_settings.method != "diagla":
        return None
    return nearest_neighbor_rows(system.grid)


def first_diagonalization_tolerance(eigensolver_settings) -> float:
    """``Dynamic_Diag_Tol``: 0.1 on the first SCF iteration of a loop."""

    if eigensolver_settings.dynamic_tolerance:
        return DYNAMIC_TOLERANCE_START
    return eigensolver_settings.tolerance


def next_diagonalization_tolerance(eigensolver_settings, tolerance: float, residuals) -> float:
    """``Dynamic_Diag_Tol``: the tolerance of the next SCF iteration.

    parsec.F90:1302-1311 after ``getsre`` (one SCF loop for every kind of
    run in PARSEC): ``min(tol, minval(sre)/10)``, not below
    ``Diag_Tolerance``.  ``residuals`` are the weighted SRE of the loop's
    channels (one value for the loops that mix one combined vector).
    """

    return dynamic_diagonalization_tolerance(
        tolerance, residuals, eigensolver_settings.tolerance
    )


def iteration_eigensolver_fields(counts, next_tolerance: float | None) -> dict:
    """Eigensolver fields of an :class:`SCFIteration` record.

    ``counts`` holds ``(nec, maxmvs)`` (or None) per spin channel; the
    record gets them only where every channel ran ARPACK or DIAGLA, and the
    tolerance of the next iteration only with ``Dynamic_Diag_Tol``.  Empty
    otherwise, so earlier calculations keep their records.  ``nec`` and
    ``maxmvs`` are integers, or tuples of one integer per symmetry
    representation (the accelerated sectors, as eigval.F90 lists them).
    """

    def values(count):
        if isinstance(count, (tuple, list)):
            return tuple(int(item) for item in count)
        return int(count)

    fields_ = {}
    if counts and all(count is not None for count in counts):
        fields_["eigensolver_counts"] = tuple(
            (values(converged), values(matvecs)) for converged, matvecs in counts
        )
    if next_tolerance is not None:
        fields_["next_eigensolver_tolerance"] = float(next_tolerance)
    return fields_


def solver_option_fields(
    eigensolver_settings, tolerance: float, filter_degree: int | None = None
) -> tuple[dict, dict, dict]:
    """The options of the PARSEC 1.5 round for one SCF step's eigensolvers.

    Three dictionaries of fields, for :class:`ChebFFSettings`,
    :class:`ChebDavSettings` and :class:`SubspaceSettings`, all empty at the
    defaults (the settings are then those of every earlier version):
    ``Vector_Rng_Seed`` (the LAPACK seed of the first-solve start vectors),
    ``Reorthogonalization_Threshold`` (the first ratio of ``orth_normal``) and
    PARSEC 1.5's residual-driven CHEBFF, whose tolerance is ``tolerance``,
    the eigensolver tolerance of the step (``Diag_Tolerance``, or 0.1 in the
    first step of ``Dynamic_Diag_Tol`` as 1.5 has it), and PARSEC 1.5's
    SUBSPACE rules (``Smart_Subspace``, ``Chebyshev_Scaled``,
    ``Bounds_Method``/``Bounds_Iter``: ``max(filter_degree/2, Bounds_Iter)``
    Lanczos steps, ``filter_degree`` being the SUBSPACE degree of the step,
    as ``solver%polym`` in ``chebff.f90z``/``subspace.f90z`` of 1.5).
    """

    chebff: dict = {}
    chebdav: dict = {}
    subspace: dict = {}
    if getattr(eigensolver_settings, "vector_seed", None) is not None:
        chebff["basis_seed"] = chebdav["basis_seed"] = eigensolver_settings.basis_seed
    threshold = getattr(eigensolver_settings, "reorthogonalization_threshold", 0.1)
    if threshold != 0.1:
        for fields_ in (chebff, chebdav, subspace):
            fields_["reorthogonalization_threshold"] = threshold
    residual_test = getattr(eigensolver_settings, "first_filter_residual_test", "off")
    if residual_test != "off":
        chebff["residual_tolerance"] = float(tolerance)
        chebff["residual_failure"] = residual_test
    rule = getattr(eigensolver_settings, "lanczos_bounds", "parsec")
    if rule != "parsec":
        degree = eigensolver_settings.filter_degree if filter_degree is None else filter_degree
        steps = max(int(degree) // 2, int(eigensolver_settings.lanczos_bound_steps))
        chebff["bounds"] = rule
        chebff["bound_steps"] = steps
        # lancz_bound_full_orth serves the CHEBFF first solve only; 1.5's
        # CHEBDAV keeps its five steps (chebdav.f90z:336 of 1.5).
        subspace["bounds"] = chebdav["bounds"] = "lanczos"
        subspace["bound_steps"] = steps
    lower_rule = getattr(eigensolver_settings, "subspace_lower_bound", "monotone")
    if lower_rule != "monotone":
        subspace["lower_bound_rule"] = lower_rule
    if getattr(eigensolver_settings, "scaled_subspace_filter", False):
        subspace["scaled_filter"] = True
    return chebff, chebdav, subspace


def eigval_settings_for(
    eigensolver_settings,
    filter_degree: int,
    *,
    tolerance: float | None = None,
    neighbors: np.ndarray | None = None,
) -> EigvalSettings:
    """Assemble the eigensolver policy of one SCF iteration.

    ``filter_degree`` is the present later-SCF SUBSPACE degree, which the
    SCF loop lowers near convergence; ``tolerance`` replaces
    ``Diag_Tolerance`` for ``Dynamic_Diag_Tol``; ``neighbors`` serves
    DIAGLA's preconditioner.  Everything else comes from the input.
    """

    if tolerance is None:
        tolerance = eigensolver_settings.tolerance
    chebff_options, chebdav_options, subspace_options = solver_option_fields(
        eigensolver_settings, tolerance, filter_degree
    )
    return EigvalSettings(
        **iterative_solver_fields(eigensolver_settings, tolerance, neighbors),
        safety_buffer=eigensolver_settings.working_buffer,
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
            **chebff_options,
        ),
        chebdav=(
            ChebDavSettings(
                polynomial_degree=eigensolver_settings.first_filter_degree,
                convergence_tolerance=tolerance,
                block_size=eigensolver_settings.matvec_block_size,
                # create_eigen_solver fixes winsize=12 for CHEBDAV.
                workspace_window=12,
                lanczos_steps=5,
                max_outer_restarts=2,
                random_seed=eigensolver_settings.random_seed,
                **chebdav_options,
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
            **subspace_options,
        ),
    )


def refuse_exchange_hooks(**hooks) -> None:
    """Stop an SCF loop that would leave out the exchange operator of a hybrid.

    The operator joins the Hamiltonian of :meth:`PreparedSinglePointSystem.hamiltonian`
    and reads the full-grid host orbitals of every iteration; an execution
    backend that applies a Hamiltonian of its own (or keeps fields on a
    symmetry wedge) would drop it, unless the system holds the operator in
    its device Hamiltonians (``exchange_device`` of the accelerated systems,
    :mod:`parsec_python.acceleration.SCF.hybrid`).
    """

    given = sorted(name for name, hook in hooks.items() if hook is not None)
    if given:
        raise NotImplementedError(
            "self-consistent hybrid functionals run on the reference SCF path only; "
            f"the execution hooks {', '.join(given)} would drop the exact-exchange operator"
        )


def refuse_hubbard_and_dispersion(system, driver: str, *, dispersion: bool = True) -> None:
    """Stop a driver that has neither the LDA+U term nor the TS correction.

    Only :func:`run_scf` and the isolated spin loop
    :func:`parsec_python.SCF.spin_polarized.run_spin_scf` carry them; every
    other driver would leave them out without a word.  ``dispersion=False``
    lets the TS correction pass (a driver that changes no energy).
    """

    problem = getattr(system, "input", None)
    hubbard = getattr(system, "hubbard", None) is not None
    dispersion = dispersion and bool(
        getattr(getattr(problem, "scf", None), "van_der_waals", False)
    )
    if hubbard or dispersion:
        feature = "LDA+U (ldaplusu_u/ldaplusu_j)" if hubbard else "Van_Der_Waals"
        raise NotImplementedError(f"{feature} is not implemented in {driver}")


def require_reference_hooks(system, **hooks) -> object | None:
    """Return the LDA+U projectors of ``system``, refusing execution hooks.

    The LDA+U term reads the full-grid orbitals of every iteration and joins
    the Hamiltonian of :meth:`PreparedSinglePointSystem.hamiltonian`; an
    accelerated eigensolver or a symmetry-reduced field representation
    applies a Hamiltonian of its own and would leave the term out, unless
    the system holds the term in that Hamiltonian itself (``hubbard_device``
    of the accelerated systems).
    """

    projectors = getattr(system, "hubbard", None)
    if projectors is None:
        return None
    given = sorted(name for name, hook in hooks.items() if hook is not None)
    # An accelerated system whose device Hamiltonians hold the term and that
    # forms the occupation matrices where its orbitals are
    # (acceleration.Hamiltonian.hubbard_device) takes its hooks along.
    if given and getattr(system, "hubbard_device", None) is None:
        raise NotImplementedError(
            "LDA+U (ldaplusu_u/ldaplusu_j) runs on the reference SCF path only; "
            f"the execution hooks {', '.join(given)} would drop its term"
        )
    return projectors


def hubbard_terms_of_iteration(system, projectors, wavefunctions, occupations):
    """``calcden.f90z`` of the orbitals of each channel, where the system keeps them.

    :func:`parsec_python.Hamiltonian.hubbard.hubbard_terms_from_orbitals`,
    unless the system forms the occupation matrices itself (the devices of
    an accelerated system, ``hubbard_device``).
    """

    device = getattr(system, "hubbard_device", None)
    if device is not None:
        return device.terms(wavefunctions, occupations)
    return hubbard_terms_from_orbitals(projectors, wavefunctions, occupations)


def hubbard_iteration_record(terms) -> dict[str, float]:
    """The :class:`SCFIteration` fields of the LDA+U terms of one iteration."""

    if terms is None:
        return {}
    return {
        "hubbard_first_contribution": float(terms.first_contribution),
        "hubbard_energy": float(terms.energy),
    }


def dispersion_wanted(settings, converged: bool) -> bool:
    """Whether the loop ends with the TS correction (``Van_Der_Waals``).

    ``parsec.F90`` calls ``efvdw`` and the second ``totnrg`` only when the
    residual fell below the criterion: an unconverged loop has neither.
    """

    return bool(converged and getattr(settings, "van_der_waals", False))


def dispersion_after_convergence(system, density, energies):
    """The TS correction of the converged ``density`` and the energies with it."""

    from ..Energy.dispersion import dispersion_correction

    # An accelerated system forms the grid sums on its devices, and so do the
    # periodic drivers where their k-point eigenproblems run on one.
    correction = getattr(system, "dispersion_device", None)
    if correction is None:
        from ..Energy.dispersion import device_dispersion_for

        correction = device_dispersion_for(system)
    dispersion = (dispersion_correction if correction is None else correction)(
        system, density
    )
    return dispersion, with_dispersion(energies, dispersion.energy)


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
    convergence_criterion: float | None = None,
    iteration_state_callback: Callable[[SCFIterationFields], None] | None = None,
    exchange_term: object | None = None,
) -> SinglePointResult:
    """Run PARSEC-style potential mixing on a prepared system.

    ``warm_start`` continues from the SCF loop of a previous structure (see
    :class:`SCFWarmStart`); without it the loop starts from the prepared
    initial density.  ``convergence_criterion`` replaces the
    ``Convergence_Criterion`` of the input for this loop (Ry), as a
    relaxation with ``Relaxation_SCF_Upscale`` asks.
    ``iteration_state_callback`` receives the :class:`SCFIterationFields`
    of every iteration after its ``callback`` (``parsec.dat`` after every
    iteration); it observes and changes nothing.

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

    With FDET (``system.embedding``) ``V_ion,local`` holds the frozen sample
    field, the XC terms read ``rho + rho_core + rho_A`` and both fields hold
    the non-additive kinetic potential ``v_nad`` of the iteration
    (:mod:`parsec_python.Embedding.frozen_density`); the energies are the
    FDET energies of ``totnrg.f90``.

    A hybrid ``Correlation_Type`` (``system.input.scf.is_hybrid``) runs the
    Fock loop of :func:`parsec_python.SCF.hybrid.run_hybrid_scf`, whose inner
    loops come back here with ``exchange_term``: the fixed exact-exchange
    operator of the loop (:class:`parsec_python.SCF.hybrid.FockOperator`).
    It joins every Hamiltonian of the loop, ``H = H[V_in] + a_x V_ACE``, and
    the energies of every iteration (:meth:`FockOperator.iteration_energies`).
    """
    if exchange_term is None and getattr(system.input.scf, "is_hybrid", False):
        from .hybrid import run_hybrid_scf

        return run_hybrid_scf(
            system,
            callback=callback,
            eigenproblem_solver=eigenproblem_solver,
            orbital_density_builder=orbital_density_builder,
            mixer_factory=mixer_factory,
            residual_metrics_evaluator=residual_metrics_evaluator,
            total_energy_evaluator=total_energy_evaluator,
            scalar_field_adapter=scalar_field_adapter,
            warm_start=warm_start,
            convergence_criterion=convergence_criterion,
            iteration_state_callback=iteration_state_callback,
        )
    if exchange_term is not None and getattr(system, "exchange_device", None) is None:
        # An accelerated system whose device Hamiltonians hold the operator
        # (acceleration.SCF.hybrid) takes its hooks along.
        refuse_exchange_hooks(
            eigenproblem_solver=eigenproblem_solver,
            orbital_density_builder=orbital_density_builder,
            mixer_factory=mixer_factory,
            residual_metrics_evaluator=residual_metrics_evaluator,
            total_energy_evaluator=total_energy_evaluator,
            scalar_field_adapter=scalar_field_adapter,
        )
    embedding = getattr(system, "embedding", None)
    if embedding is not None and (
        getattr(system.input.scf, "spin_polarized", False) or scalar_field_adapter is not None
    ):
        raise NotImplementedError(
            "FDET (Potential_Field) runs spin-unpolarized on the full grid only (no symmetry sectors)"
        )
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
            convergence_criterion=convergence_criterion,
            **(
                {}
                if iteration_state_callback is None
                else {"iteration_state_callback": iteration_state_callback}
            ),
            **({} if exchange_term is None else {"exchange_term": exchange_term}),
        )
    scf_start = time.perf_counter()
    level_detail = None
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
    # backend perform scalar-field dot products and mixer history on one
    # value per orbit.  Their default functions are exactly the original
    # full-grid algorithms, so the reference architecture is unchanged.
    # make_mixer builds the mixer of Mixing_Method (Anderson by default).
    build_mixer = make_mixer if mixer_factory is None else mixer_factory
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
    hubbard_projectors = require_reference_hooks(
        system,
        eigenproblem_solver=eigenproblem_solver,
        orbital_density_builder=orbital_density_builder,
        mixer_factory=mixer_factory,
        residual_metrics_evaluator=residual_metrics_evaluator,
        total_energy_evaluator=total_energy_evaluator,
        scalar_field_adapter=scalar_field_adapter,
    )
    # LDA+U potential matrices of the present iteration: zero (None) in the
    # first iteration of every SCF loop, as upot.F90 leaves them.
    hubbard_state = None
    # 1. Establish the initial Hartree, XC, and effective potentials.  The
    # number below is the state count returned to occupations; solve_eigval
    # internally works with number_of_states + subspace_buffer vectors.
    number_of_states = _number_of_states(system)
    settings = system.input.scf
    if convergence_criterion is not None:
        settings = replace(settings, convergence_criterion=float(convergence_criterion))
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method == "none":
        raise ValueError(
            "Eigensolver: none is PARSEC's TEST mode: setup only, no SCF"
        )
    # DIAGLA smooths its residuals over the six grid neighbours (fdo5scl).
    neighbors = preconditioner_neighbors(system, eigensolver_settings)
    # Dynamic_Diag_Tol: 0.1 for the first iteration of every SCF loop
    # (parsec.F90:1106-1111), then the SRE rule after each iteration.
    diagonalization_tolerance = first_diagonalization_tolerance(eigensolver_settings)
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
        if embedding is None:
            xc = system.evaluate_xc(density)
        else:
            xc, nonadditive = embedded_exchange_correlation(system, density)
        initial_xc_seconds = time.perf_counter() - stage_start
        xc_potential = xc.potential
        # This array is the diagonal, local part of H.  V_NL is intentionally
        # not folded into it; PreparedSinglePointSystem.hamiltonian adds V_NL
        # as an operator when the eigensolver requests H@vector.
        input_potential = (
            ionic_potential + hartree_potential + xc_potential
        )
        if embedding is not None:
            # parsec.F90:1296: vold = vion + vxc + vhart + vnadd.
            input_potential = input_potential + nonadditive.potential
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
        density_guess = getattr(warm_start, "density_guess", None)
        if density_guess is not None:
            # Atomic extrapolation: the Hartree and XC potentials of the
            # guessed density replace those of the last output density.
            guess = to_internal_scalar(np.array(density_guess, dtype=np.float64))
            stage_start = time.perf_counter()
            moved_hartree = system.solve_hartree(
                guess, initial_potential=hartree_potential
            )
            initial_hartree_seconds = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            if embedding is None:
                moved_xc = system.evaluate_xc(guess)
                moved_xc_potential = moved_xc.potential
            else:
                # The reference holds v_nad of the last output potential.
                moved_xc, moved_nonadditive = embedded_exchange_correlation(system, guess)
                moved_xc_potential = moved_xc.potential + moved_nonadditive.potential
            initial_xc_seconds = time.perf_counter() - stage_start
            hartree_potential = moved_hartree.potential
            input_potential = (
                input_potential
                + (hartree_potential + moved_xc_potential)
                - to_internal_scalar(
                    np.array(warm_start.hxc_reference, dtype=np.float64)
                )
            )
    mixer = warm_start_mixer(warm_start, lambda: build_mixer(system.input.mixing))
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

    # SCFWarmStart.level_tolerance: after the converged iteration one more
    # (``refining``) on the same V_in, its eigenpairs to that tolerance.
    level_tolerance = None if warm_start is None else getattr(warm_start, "level_tolerance", None)
    refining = False
    for iteration in range(1, settings.max_iterations + 1 + int(level_tolerance is not None)):
        if iteration > settings.max_iterations and not refining:
            break
        iteration_start = time.perf_counter()
        # 3. Solve the Kohn--Sham eigenproblem with PARSEC's fixed number of
        # first-filter cycles or one later-filter pass.
        # The eigenproblem in this iteration uses V_in exactly as it existed
        # before constructing rho_new; save it for energy bookkeeping/output.
        last_input_potential = input_potential.copy()
        # Rebuild the small policy object so a degree reduced by the SCF logic
        # below is used by the next SUBSPACE call.  solve_eigval dispatches on
        # eigval_state, not on the nonlinear iteration number itself.
        first_degree = (
            getattr(warm_start, "first_iteration_degree", None)
            if warm_start is not None and iteration == 1
            else None
        )
        eigval_settings = eigval_settings_for(
            eigensolver_settings,
            filter_degree if first_degree is None else max(filter_degree, int(first_degree)),
            tolerance=diagonalization_tolerance,
            neighbors=neighbors,
        )
        # This is the Hamiltonian assembly/handoff point.  The lightweight
        # object combines the current mixed V_in with the static kinetic and
        # KB terms; LinearOperator routes all Lanczos and Chebyshev H@Q calls
        # through that same three-term action.
        binding_start = time.perf_counter()
        if hubbard_projectors is None and exchange_term is None:
            hamiltonian = system.hamiltonian(input_potential)
        elif hubbard_projectors is None:
            # A hybrid functional: the fixed exchange operator of the loop.
            hamiltonian = system.hamiltonian(
                input_potential, None, exchange_term.operator(0)
            )
        else:
            if iteration > 1:
                # calcden.f90z: N and D from the orbitals and occupations of
                # the iteration before, at the start of this one.
                hubbard_state = hubbard_terms_of_iteration(
                    system, hubbard_projectors, [wavefunctions], [occupations]
                )
            hamiltonian = system.hamiltonian(
                input_potential, hubbard_operator(hubbard_projectors, hubbard_state)
            )
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
        if refining:
            # The loop converged at the iteration before: the same H[V_in]
            # until no level moves by more than level_tolerance.
            eigensolution, level_passes, level_seconds, level_change = refine_levels(
                solve_current_eigenproblem,
                hamiltonian_operator,
                number_of_states,
                eigval_settings,
                eigval_state,
                eigenvalues,
                None,
                float(level_tolerance),
            )
            level_detail = {
                "level_passes": float(level_passes),
                "level_seconds": float(level_seconds),
                "level_change": float(level_change),
            }
        else:
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
        symmetrization = ()
        density_symmetry = getattr(system, "density_symmetry", None)
        if density_symmetry is not None:
            # Symmetrize_Charge_Density: symm_scalar of newrho.F90 after
            # every new density (parsec.F90 lines 1223-1232).  A group that
            # brings its own method averages the fields of an accelerated
            # loop that keeps one value per symmetry orbit.
            symmetrize = getattr(density_symmetry, "symmetrize_density", None)
            density, report = (
                symmetrize(density, system.grid.volume_element)
                if callable(symmetrize)
                else symmetrize_density(
                    density, density_symmetry, system.grid.volume_element
                )
            )
            symmetrization = (report,)
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
        if embedding is None:
            xc = system.evaluate_xc(density)
        else:
            # FDET: XC of rho + rho_core + rho_A and T_nad, v_nad (exc.f90:816-822).
            xc, nonadditive = embedded_exchange_correlation(system, density)
        xc_seconds = time.perf_counter() - xc_start
        xc_total += xc_seconds
        xc_potential = xc.potential
        # V_out is a newly evaluated *local* field.  It is not used to
        # rediagonalize immediately: convergence is measured against V_in,
        # then the mixer supplies the next iteration's local input field.
        output_potential = (
            ionic_potential + hartree_potential + xc_potential
        )
        if embedding is not None:
            # parsec.F90:1631-1633: vnew = vhart + vxc + vion + vnadd.
            output_potential = output_potential + nonadditive.potential

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
            # totnrg.f90 adds etot_plusu, zero before the first calcden.
            **(
                {}
                if hubbard_projectors is None
                else {
                    "hubbard_energy": 0.0
                    if hubbard_state is None
                    else hubbard_state.energy
                }
            ),
        )
        if embedding is not None:
            # totnrg.f90:159-197: the FDET energy of tip and sample.
            energies = embedding_energies(
                energies, embedding, density, ionic_potential, nonadditive, system.grid.volume_element
            )
        if exchange_term is not None:
            # The exchange operator's part of the band energy is removed and
            # the exact exchange of the functional this loop minimizes added.
            energies = exchange_term.iteration_energies(
                energies, (wavefunctions,), (occupations,)
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
        # when plain SRE is selected for the final stop.  The threshold is
        # Convergence_Criterion_Approach, by default 100*vconv.
        approach_residual = metrics.weighted
        if (
            iteration > settings.degree_decrease_after
            and approach_residual < settings.approach_criterion
            and filter_degree > minimum_filter_degree
        ):
            # Once reasonably close (100*vconv), save Hamiltonian applications
            # by lowering only the later-SCF polynomial one degree per step.
            filter_degree -= 1
        # The strict inequality is PARSEC's fixed-point stop rule.  Energy,
        # Hartree, and eigenpair residuals remain diagnostics only.
        if iteration == 1:
            first_residual = selected_residual
        iteration_converged = selected_residual < loop_criterion(
            warm_start, settings.convergence_criterion, first_residual
        )
        if eigensolver_settings.dynamic_tolerance:
            diagonalization_tolerance = dynamic_diagonalization_tolerance(
                diagonalization_tolerance,
                (metrics.weighted,),
                eigensolver_settings.tolerance,
            )
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
            density_symmetrization=symmetrization,
            **({"level_refinement": True} if refining else {}),
            **hubbard_iteration_record(hubbard_state),
            **iteration_eigensolver_fields(
                (getattr(eigensolution, "iterative_counts", None),),
                diagonalization_tolerance
                if eigensolver_settings.dynamic_tolerance
                else None,
            ),
        )
        history.append(diagnostic)
        if callback is not None:
            callback(diagnostic)
        if iteration_state_callback is not None:
            iteration_state_callback(
                SCFIterationFields(
                    iteration=iteration,
                    eigenvalues=eigenvalues,
                    occupations=occupations,
                    representations=representations,
                    wavefunctions=wavefunctions,
                    density=to_public_scalar(density),
                    next_effective_potential=to_public_scalar(mixed_potential),
                )
            )

        if iteration_converged or refining:
            if level_tolerance is not None and not refining:
                # V_in stays: the next iteration solves the same Hamiltonian.
                refining = True
                continue
            converged = True
            break
        input_potential = mixed_potential

    if energies is None:
        raise RuntimeError("SCF loop did not execute")
    dispersion = None
    if dispersion_wanted(settings, converged):
        dispersion, energies = dispersion_after_convergence(
            system, to_public_scalar(density), energies
        )
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
        **({} if level_detail is None else {"detail": level_detail}),
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
            if embedding is None
            else output_potential
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
        hubbard=hubbard_state,
        dispersion=dispersion,
        embedding=None if embedding is None else nonadditive,
    )
