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

Only Python/NumPy implementations are called.  There is intentionally no
ARPACK fallback, Fortran subprocess, shared-library interface, or residual
criterion hidden in this dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .chebdav import ChebDavResult, ChebDavSettings, run_chebdav
from .chebff import ChebFFResult, ChebFFSettings, run_chebff
from .subspace import (
    SubspaceResult,
    SubspaceSettings,
    SubspaceState,
    run_subspace_filter,
)


InitialSolver = Literal["chebff", "chebdav"]
SolverPath = Literal["chebff", "chebdav", "subspace"]


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

    def __post_init__(self) -> None:
        if self.safety_buffer < 0:
            raise ValueError("safety_buffer cannot be negative")
        if self.initial_method not in {"chebff", "chebdav"}:
            raise ValueError("initial_method must be 'chebff' or 'chebdav'")


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
