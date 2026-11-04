"""Strict PARSEC first-solver-to-SUBSPACE eigensolver dispatch.

Provenance
----------
This module translates the relevant scalar, isolated-system policy in
PARSEC's ``src/eigval.F90``:

* the first compatible solve calls the selected ``chebff_diag`` or
  ``chebdav_diag`` implementation;
* saved Ritz vectors and values initialize later filtering;
* subsequent SCF solves call ``subspace`` exactly once; and
* an incompatible eigenstate allocation restarts the selected first solver.

``Eigensolver: arpack`` (``arpk.f90z``) and ``diagla`` (``diagla.f90z``)
follow the same rule when ``Chebyshev_Filtering`` is on (``solver%do_subsp``:
the selected solver once, SUBSPACE afterwards); without it eigval.F90 runs
the selected solver in every SCF iteration (lines 312-346): ARPACK from a
new random start vector, DIAGLA from the orbitals of the previous iteration.

Only Python/NumPy implementations are called (ARPACK through SciPy only
when it is the selected solver).  There is intentionally no solver
fallback, Fortran subprocess, shared-library interface, or residual
criterion hidden in this dispatcher.

``Dynamic_Diag_Tol`` (:func:`dynamic_diagonalization_tolerance`) sets the
tolerance the SCF drivers hand to CHEBDAV, ARPACK and DIAGLA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .arpack import ArpackResult, ArpackSettings, run_arpack
from .chebdav import ChebDavResult, ChebDavSettings, run_chebdav
from .chebff import ChebFFResult, ChebFFSettings, run_chebff
from .diagla import DiaglaResult, DiaglaSettings, run_diagla
from .iterative import matvec_budget
from .subspace import (
    SubspaceResult,
    SubspaceSettings,
    SubspaceState,
    run_subspace_filter,
)


InitialSolver = Literal["chebff", "chebdav", "arpack", "diagla"]
SolverPath = Literal["chebff", "chebdav", "subspace", "arpack", "diagla"]
# The solvers eigval.F90 may run in every SCF iteration.
ITERATIVE_SOLVERS = frozenset({"arpack", "diagla"})
# parsec.F90:1106-1111 (intended): the first SCF iteration of a loop with
# Dynamic_Diag_Tol uses this eigensolver tolerance.
DYNAMIC_TOLERANCE_START = 0.1


def dynamic_diagonalization_tolerance(
    tolerance: float,
    weighted_residuals,
    floor: float,
) -> float:
    """Eigensolver tolerance of the next SCF iteration (``Dynamic_Diag_Tol``).

    parsec.F90:1304-1311 after ``getsre``::

        toler = min(toler, minval(elec_st%sre)/10)
        toler = max(toler, dyntoler)

    with ``weighted_residuals`` the weighted SRE of each spin channel
    (``elec_st%sre``) and ``floor`` the ``Diag_Tolerance`` of the input.
    This is the rule the Fortran intends.  Its effective behaviour differs:
    usrinputfile.F90:1497 already sets ``toler = 0.1`` when it reads
    ``Dynamic_Diag_Tol``, so ``dyntoler = toler`` (parsec.F90:1109) stores
    0.1 as the floor and the tolerance stays 0.1 throughout, the input
    ``Diag_Tolerance`` being lost.
    """

    smallest = min(float(value) for value in weighted_residuals)
    return max(min(float(tolerance), smallest / 10.0), float(floor))


@dataclass(frozen=True)
class EigvalSettings:
    """Complete first/later eigensolver policy for one SCF sequence.

    ``safety_buffer`` augments the requested occupied/thermal states with
    extra working vectors.  ``initial_method`` selects CHEBFF or CHEBDAV,
    their corresponding settings control creation of the first working
    subspace, and ``subspace`` controls its reuse after the Hamiltonian
    changes in later SCF iterations.
    """

    safety_buffer: int = 6
    initial_method: InitialSolver = "chebff"
    chebff: ChebFFSettings = field(default_factory=ChebFFSettings)
    chebdav: ChebDavSettings = field(default_factory=ChebDavSettings)
    subspace: SubspaceSettings = field(default_factory=SubspaceSettings)
    arpack: ArpackSettings = field(default_factory=ArpackSettings)
    diagla: DiaglaSettings = field(default_factory=DiaglaSettings)
    # Chebyshev_Filtering for ARPACK and DIAGLA; None follows the rule of
    # usrinputfile.F90 (on for CHEBFF/CHEBDAV, off otherwise).
    chebyshev_filtering: bool | None = None

    def __post_init__(self) -> None:
        if self.safety_buffer < 0:
            raise ValueError("safety_buffer cannot be negative")
        if self.initial_method not in {"chebff", "chebdav", "arpack", "diagla"}:
            raise ValueError(
                "initial_method must be 'chebff', 'chebdav', 'arpack' or 'diagla'"
            )
        if (
            self.initial_method not in ITERATIVE_SOLVERS
            and self.chebyshev_filtering is False
        ):
            raise ValueError(
                "PARSEC always follows CHEBFF and CHEBDAV by the SUBSPACE filter"
            )

    @property
    def later_subspace_filter(self) -> bool:
        """``solver%do_subsp``: later SCF iterations run SUBSPACE."""

        if self.initial_method not in ITERATIVE_SOLVERS:
            return True
        return bool(self.chebyshev_filtering)


@dataclass(frozen=True)
class EigvalState:
    """Persistent buffered Ritz subspace carried between SCF iterations.

    The saved vectors contain ``working_states = requested_states + buffer``
    columns, not merely the eigenpairs returned to occupations.  Their values,
    vectors, filter cutoff, and filter counter live inside ``subspace``.
    ``initial_method`` records which first solver created it, while
    ``solves_completed`` distinguishes a valid saved state from an
    uninitialized object.
    """

    operator_dimension: int
    requested_states: int
    working_states: int
    initial_method: InitialSolver
    subspace: SubspaceState
    solves_completed: int
    # ARPACK: the LAPACK seed of dgetv0, kept between solves.
    start_seed: tuple[int, int, int, int] | None = None
    # DIAGLA: evec columns beyond the working states (the window), the
    # initial guesses of the next solve together with ``subspace.vectors``.
    window_vectors: np.ndarray | None = None


@dataclass(frozen=True)
class EigvalResult:
    """Requested eigensystem plus the full state needed by the next call."""

    eigenvalues: np.ndarray
    vectors: np.ndarray
    residual_norms: np.ndarray | None
    state: EigvalState
    solver_path: SolverPath
    restarted: bool
    restart_reason: str | None
    chebff_result: ChebFFResult | None = None
    chebdav_result: ChebDavResult | None = None
    subspace_result: SubspaceResult | None = None
    arpack_result: ArpackResult | None = None
    diagla_result: DiaglaResult | None = None

    @property
    def iterative_counts(self) -> tuple[int, int] | None:
        """``(nec, maxmvs)`` of an ARPACK or DIAGLA solve, as eigval.F90
        reports them; ``None`` for the other solvers."""

        solved = self.arpack_result or self.diagla_result
        if solved is None:
            return None
        return (int(solved.converged_states), int(solved.matvecs))

    @property
    def working_eigenvalues(self) -> np.ndarray:
        """Include PARSEC's safety-buffer states."""

        return self.state.subspace.eigenvalues

    @property
    def working_vectors(self) -> np.ndarray:
        """Include PARSEC's safety-buffer vectors."""

        return self.state.subspace.vectors


def _operator_dimension(operator: Any) -> int:
    shape = getattr(operator, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("operator must expose a square two-dimensional shape")
    return int(shape[0])


def _working_state_count(
    dimension: int,
    requested_states: int,
    safety_buffer: int,
) -> int:
    """Add the safety buffer without exceeding the operator dimension."""
    if not 1 <= requested_states <= dimension:
        raise ValueError(
            f"requested_states must be between 1 and {dimension}, "
            f"got {requested_states}"
        )
    return min(dimension, requested_states + safety_buffer)


def _state_incompatibility(
    state: EigvalState,
    *,
    operator_dimension: int,
    requested_states: int,
    working_states: int,
    initial_method: InitialSolver,
) -> str | None:
    """Return why PARSEC's saved eigensubspace cannot be reused."""

    if state.operator_dimension != operator_dimension:
        return "operator_dimension_changed"
    if state.requested_states != requested_states:
        return "requested_state_count_changed"
    if state.working_states != working_states:
        return "working_state_count_changed"
    if state.initial_method != initial_method:
        return "initial_solver_changed"
    if state.solves_completed < 1:
        return "invalid_saved_solve_count"

    saved = state.subspace
    if saved.operator_dimension != operator_dimension:
        return "saved_subspace_dimension_changed"
    if saved.working_states != working_states:
        return "saved_subspace_state_count_changed"
    if np.shape(saved.eigenvalues) != (working_states,):
        return "saved_eigenvalue_shape_changed"
    if np.shape(saved.vectors) != (operator_dimension, working_states):
        return "saved_vector_shape_changed"
    return None


def _state_from_first_solver(
    result: ChebFFResult | ChebDavResult,
    *,
    requested_states: int,
    initial_method: InitialSolver,
) -> EigvalState:
    """Convert CHEBFF/CHEBDAV output into PARSEC's ``eig_init`` state."""

    first = result.state
    subspace = SubspaceState(
        operator_dimension=first.operator_dimension,
        working_states=first.wanted_states,
        eigenvalues=first.eigenvalues,
        vectors=first.vectors,
        # subspace.f90z recomputes this from max(saved Ritz) on firstfilt.
        filter_lower_bound=None,
        first_filter=True,
        filters_completed=0,
    )
    return EigvalState(
        operator_dimension=first.operator_dimension,
        requested_states=requested_states,
        working_states=first.wanted_states,
        initial_method=initial_method,
        subspace=subspace,
        solves_completed=1,
    )


def _requested_view(
    values: np.ndarray,
    vectors: np.ndarray,
    requested_states: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hide buffer states from occupations while retaining them in state."""
    return (
        np.asarray(values[:requested_states]),
        np.asarray(vectors[:, :requested_states]),
    )


def solve_eigval(
    operator: Any,
    requested_states: int,
    *,
    settings: EigvalSettings = EigvalSettings(),
    state: EigvalState | None = None,
) -> EigvalResult:
    """Run the selected real non-``BETA`` PARSEC eigensolver policy once.

    The dispatcher is called once per nonlinear SCF iteration:

    * with ``state=None``, construct ``requested_states + safety_buffer``
      approximate eigenvectors using the explicitly selected CHEBFF or
      CHEBDAV first solver;
    * with a compatible state, reuse all buffered vectors in exactly one
      later-SCF SUBSPACE filter/Rayleigh--Ritz update; or
    * with an incompatible state, explicitly restart that selected solver.

    With ARPACK or DIAGLA selected the solver runs instead of CHEBFF/CHEBDAV
    (see :func:`_solve_iterative`), and in every call unless
    ``settings.later_subspace_filter``.

    A restart is never converted into a different eigensolver.  Returned
    ``eigenvalues``/``vectors`` are truncated to ``requested_states``, while
    ``result.state`` always retains the complete working subspace.
    """

    dimension = _operator_dimension(operator)
    working_states = _working_state_count(
        dimension,
        requested_states,
        settings.safety_buffer,
    )
    # Reuse is safe only when grid dimension and both requested/working state
    # allocations are unchanged.  SCF changes matrix values, which is expected
    # and does not invalidate a saved basis of the same vector space.
    reason = None
    if state is not None:
        reason = _state_incompatibility(
            state,
            operator_dimension=dimension,
            requested_states=requested_states,
            working_states=working_states,
            initial_method=settings.initial_method,
        )

    if settings.initial_method in ITERATIVE_SOLVERS and (
        state is None or reason is not None or not settings.later_subspace_filter
    ):
        return _solve_iterative(
            operator,
            requested_states,
            dimension=dimension,
            working_states=working_states,
            settings=settings,
            state=state,
            reason=reason,
        )

    if state is None or reason is not None:
        # First nonlinear iteration, or an explicit allocation restart.  The
        # selected first solver is never replaced by the other one.
        if settings.initial_method == "chebff":
            first = run_chebff(
                operator,
                working_states,
                settings=settings.chebff,
            )
            first_residuals = None
            chebff_result: ChebFFResult | None = first
            chebdav_result: ChebDavResult | None = None
        else:
            first = run_chebdav(
                operator,
                working_states,
                settings=settings.chebdav,
            )
            first_residuals = np.asarray(
                first.residual_norms[:requested_states]
            )
            chebff_result = None
            chebdav_result = first

        next_state = _state_from_first_solver(
            first,
            requested_states=requested_states,
            initial_method=settings.initial_method,
        )
        eigenvalues, vectors = _requested_view(
            first.eigenvalues,
            first.vectors,
            requested_states,
        )
        return EigvalResult(
            eigenvalues=eigenvalues,
            vectors=vectors,
            # CHEBFF has no residual acceptance test.  CHEBDAV explicitly
            # computes these residuals for contiguous-prefix locking.
            residual_norms=first_residuals,
            state=next_state,
            solver_path=settings.initial_method,
            restarted=state is not None,
            restart_reason=reason,
            chebff_result=chebff_result,
            chebdav_result=chebdav_result,
        )

    # Later nonlinear iterations start from the previous Hamiltonian's Ritz
    # vectors.  Because successive SCF Hamiltonians should be nearby, PARSEC
    # elects to perform exactly one filter/Rayleigh--Ritz update; it does not
    # verify that choice with an eigen-residual acceptance test.
    later = run_subspace_filter(
        operator,
        state.subspace,
        settings=settings.subspace,
    )
    next_state = EigvalState(
        operator_dimension=dimension,
        requested_states=requested_states,
        working_states=working_states,
        initial_method=state.initial_method,
        subspace=later.state,
        solves_completed=state.solves_completed + 1,
    )
    eigenvalues, vectors = _requested_view(
        later.eigenvalues,
        later.vectors,
        requested_states,
    )
    return EigvalResult(
        eigenvalues=eigenvalues,
        vectors=vectors,
        residual_norms=np.asarray(later.residual_norms[:requested_states]),
        state=next_state,
        solver_path="subspace",
        restarted=False,
        restart_reason=None,
        subspace_result=later,
    )


def _solve_iterative(
    operator: Any,
    requested_states: int,
    *,
    dimension: int,
    working_states: int,
    settings: EigvalSettings,
    state: EigvalState | None,
    reason: str | None,
) -> EigvalResult:
    """One ARPACK or DIAGLA solve of ``working_states`` eigenpairs.

    eigval.F90 calls the solver with ``nn`` wanted pairs; the budget is
    ``max(Maximum_matvec, n * n_states)`` (parsec.F90:687).  ARPACK
    continues the LAPACK stream of its start vectors from ``state``; DIAGLA
    starts from the saved vectors (``nn + winsize`` columns), or from zero
    vectors -- PARSEC's freshly allocated ``evec`` -- on the first solve.
    The pairs become the saved subspace exactly as after CHEBFF, so a run
    with ``Chebyshev_Filtering`` continues with SUBSPACE.
    """

    reusable = state is not None and reason is None
    method = settings.initial_method
    if method == "arpack":
        budget = matvec_budget(
            settings.arpack.maximum_matvec, dimension, requested_states
        )
        solved = run_arpack(
            operator,
            working_states,
            settings=settings.arpack,
            matvec_budget=budget,
            start_seed=None if state is None else state.start_seed,
        )
        residuals = None
        start_seed = solved.start_seed
        window_vectors = None
        arpack_result: ArpackResult | None = solved
        diagla_result: DiaglaResult | None = None
    else:
        budget = matvec_budget(
            settings.diagla.maximum_matvec, dimension, requested_states
        )
        guesses = None
        if reusable and state.window_vectors is not None:
            guesses = np.hstack([state.subspace.vectors, state.window_vectors])
        solved = run_diagla(
            operator,
            working_states,
            settings=settings.diagla,
            matvec_budget=budget,
            initial_vectors=guesses,
        )
        residuals = np.asarray(solved.residual_norms[:requested_states])
        start_seed = None
        window_vectors = solved.all_vectors[:, working_states:]
        arpack_result = None
        diagla_result = solved
    subspace = SubspaceState(
        operator_dimension=dimension,
        working_states=working_states,
        eigenvalues=np.asarray(solved.eigenvalues),
        vectors=np.asarray(solved.vectors),
        # subspace.f90z recomputes this from max(saved Ritz) on firstfilt.
        filter_lower_bound=None,
        first_filter=True,
        filters_completed=0,
    )
    next_state = EigvalState(
        operator_dimension=dimension,
        requested_states=requested_states,
        working_states=working_states,
        initial_method=method,
        subspace=subspace,
        solves_completed=state.solves_completed + 1 if reusable else 1,
        start_seed=start_seed,
        window_vectors=window_vectors,
    )
    eigenvalues, vectors = _requested_view(
        solved.eigenvalues, solved.vectors, requested_states
    )
    return EigvalResult(
        eigenvalues=eigenvalues,
        vectors=vectors,
        residual_norms=residuals,
        state=next_state,
        solver_path=method,
        restarted=state is not None and reason is not None,
        restart_reason=reason,
        arpack_result=arpack_result,
        diagla_result=diagla_result,
    )


class StrictEigvalSolver:
    """Optional stateful façade around :func:`solve_eigval`.

    The SCF driver stores ``EigvalState`` explicitly because it also records
    iteration diagnostics.  This convenience class provides the same policy
    for independent callers that prefer internal state management.
    """

    def __init__(self, settings: EigvalSettings = EigvalSettings()) -> None:
        self.settings = settings
        self._state: EigvalState | None = None

    @property
    def state(self) -> EigvalState | None:
        return self._state

    def reset(self) -> None:
        self._state = None

    def solve(self, operator: Any, requested_states: int) -> EigvalResult:
        result = solve_eigval(
            operator,
            requested_states,
            settings=self.settings,
            state=self._state,
        )
        self._state = result.state
        return result
