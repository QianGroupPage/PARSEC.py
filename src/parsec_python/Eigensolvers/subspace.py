"""One later-SCF PARSEC Chebyshev subspace-filter step.

Provenance
----------
The control flow is translated from the non-``BETA`` branch of PARSEC's
``subspace`` and internal ``cheby_filter`` routines in
``src/subspace.f90z``.  Each call:

1. updates the unwanted-spectrum lower bound from saved Ritz values,
2. obtains a fresh short-Lanczos upper bound,
3. adapts the polynomial degree for a long unwanted interval,
4. filters the lower and upper vector groups at ``degree - delta`` and
   ``degree + delta``, respectively,
5. orthonormalizes, and
6. performs exactly one Rayleigh--Ritz rotation.

There is no residual acceptance test and no alternate eigensolver path.
The ``reset_recurrence_per_block=False`` default intentionally reproduces
the current Fortran routine's shared-``sigma`` block trajectory; callers may
select the mathematically independent-block form explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .chebyshev import FilterBlock, subspace_filter, subspace_filter_blocks
from .orthogonalize import orthonormalize
from .rayleigh_ritz import rayleigh_ritz
from .spectral_bounds import LanczosBoundResult, lanczos_upper_bound, spectral_bound


@dataclass(frozen=True)
class SubspaceSettings:
    """Controls corresponding to PARSEC's later-SCF subspace filter.

    One call performs one filter/orthonormalization/Rayleigh--Ritz update;
    there is no internal iteration to an eigenpair tolerance.  Vector blocks
    in the lower and upper halves of the working subspace use degrees near
    ``polynomial_degree - degree_delta`` and
    ``polynomial_degree + degree_delta``, respectively.

    ``random_seed``, arbitrary Lanczos-step overrides, and the recurrence reset
    switch are Python reproducibility/testing controls.  PARSEC's ordinary
    real path uses a five-step request and shared recurrence state.
    ``reorthogonalization_threshold`` is the first ratio test of
    ``orth_normal`` (0.1; PARSEC 1.5 uses 0.68).

    PARSEC 1.5's SUBSPACE rules (``subspace.f90z`` of 1.5), off by default:
    ``lower_bound_rule`` ``"monotone"`` (the earlier PARSEC, see
    :func:`_next_filter_lower_bound`), ``"smart"`` or ``"adaptive"``
    (``Smart_Subspace`` true or false, :func:`parsec15_lower_bound`);
    ``scaled_filter`` (``Chebyshev_Scaled``: the recurrence normalized at the
    lowest saved Ritz value ``aL`` instead of the lower bound, the same span
    in exact arithmetic); ``bounds`` ``"parsec"`` or ``"lanczos"`` (1.5's
    :func:`spectral_bounds.lanczos_bound_parsec15`, ``bound_steps`` steps
    without a clamp, ``lanczos_steps`` when it is ``None``).  1.5's single filter degree for all vectors is not
    taken over: on Si5H12 (20 states, degree 15) a later SUBSPACE call of 1.5 costs 330 Hamiltonian products
    against 313 with the split degrees, 5 % more, and saves no SCF iterations.
    """

    polynomial_degree: int = 15
    degree_delta: int = 3
    lanczos_steps: int = 5
    block_size: int = 6
    reset_recurrence_per_block: bool = False
    random_seed: int = 19
    reorthogonalization_threshold: float = 0.1
    lower_bound_rule: str = "monotone"
    scaled_filter: bool = False
    bounds: str = "parsec"
    bound_steps: int | None = None

    def __post_init__(self) -> None:
        if self.lower_bound_rule not in ("monotone", "smart", "adaptive"):
            raise ValueError("lower_bound_rule must be monotone, smart or adaptive")
        if self.bounds not in ("parsec", "lanczos"):
            raise ValueError("SUBSPACE bounds must be parsec or lanczos")
        if not 0.0 < self.reorthogonalization_threshold < 1.0:
            raise ValueError("reorthogonalization_threshold must be in (0, 1)")
        if self.polynomial_degree < 1:
            raise ValueError("polynomial_degree must be positive")
        if self.degree_delta < 0:
            raise ValueError("degree_delta cannot be negative")
        if self.degree_delta >= self.polynomial_degree:
            raise ValueError("degree_delta must be smaller than polynomial_degree")
        if self.lanczos_steps < 1:
            raise ValueError("lanczos_steps must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")


@dataclass(frozen=True)
class SubspaceState:
    """Reusable buffered eigensubspace and bound history for one SCF run.

    ``first_filter`` does not mean the first CHEBFF/CHEBDAV calculation.  It
    is true in the state converted from either first solver and selects the
    special lower-bound rule on the first later-SCF SUBSPACE call.
    ``filter_lower_bound`` then records the unwanted-spectrum boundary used
    by the most recent call.
    """

    operator_dimension: int
    working_states: int
    eigenvalues: np.ndarray
    vectors: np.ndarray
    filter_lower_bound: float | None = None
    first_filter: bool = True
    filters_completed: int = 0


@dataclass(frozen=True)
class SubspaceResult:
    """Output and complete state after one filtered-subspace update."""

    eigenvalues: np.ndarray
    vectors: np.ndarray
    residual_norms: np.ndarray
    state: SubspaceState
    lanczos_bound: LanczosBoundResult
    polynomial_degree_used: int
    filter_blocks: tuple[FilterBlock, ...]

    @property
    def approximate_converged_states(self) -> int:
        """Number PARSEC reports as converged, irrespective of residuals."""

        return self.state.working_states


def _operator_dimension(operator: Any) -> int:
    """Validate the square matrix-like interface and return its dimension."""

    shape = getattr(operator, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("operator must expose a square two-dimensional shape")
    return int(shape[0])


def _validate_state(operator: Any, state: SubspaceState) -> None:
    """Check that all saved buffered arrays can be reused with ``operator``.

    Only allocation compatibility is relevant here.  The numerical entries
    of the Hamiltonian are expected to change as SCF mixes its local
    potential; reusing the old Ritz vectors for that nearby Hamiltonian is
    the purpose of this routine.
    """

    dimension = _operator_dimension(operator)
    if state.operator_dimension != dimension:
        raise ValueError(
            "saved subspace belongs to an operator with a different dimension"
        )
    if not 1 <= state.working_states <= dimension:
        raise ValueError("working_states is inconsistent with operator dimension")
    if np.shape(state.eigenvalues) != (state.working_states,):
        raise ValueError("saved eigenvalues do not match working_states")
    if np.shape(state.vectors) != (dimension, state.working_states):
        raise ValueError("saved vectors do not match operator and state dimensions")
    if not state.first_filter and state.filter_lower_bound is None:
        raise ValueError("a continued subspace state needs filter_lower_bound")
    if state.filters_completed < 0:
        raise ValueError("filters_completed cannot be negative")


def _next_filter_lower_bound(state: SubspaceState) -> float:
    """Choose PARSEC's monotone unwanted-spectrum lower boundary.

    For the first SUBSPACE call following CHEBFF or CHEBDAV,

    ``l = max(max(saved_eigenvalues), 0)``.

    On later calls the boundary cannot move below its saved value:

    ``l = max(previous_l, max(saved_eigenvalues))``.

    Thus all buffered Ritz values lie at or below the start of the interval
    treated as unwanted by the next Chebyshev filter.
    """

    largest_saved_ritz = float(np.max(state.eigenvalues))
    if state.first_filter or state.filter_lower_bound is None:
        # subspace.f90z firstfilt branch (non-BETA build).
        return max(largest_saved_ritz, 0.0)
    return max(float(state.filter_lower_bound), largest_saved_ritz)


# eigen_solver_module.F90:322 of 1.5: solver%lowerb starts at mach_eps.
_MACHINE_EPSILON = float(np.finfo(float).eps)


def parsec15_lower_bound(
    state: SubspaceState,
    rule: str,
    bound: LanczosBoundResult,
) -> tuple[float, float]:
    """PARSEC 1.5's SUBSPACE lower bound and bound repair (``subspace.f90z:237-359`` of 1.5).

    With ``r_max`` the largest saved Ritz value and ``l_p`` the lower bound
    of the previous call (``mach_eps`` before the first):

    * first call after the first solver: ``g = 0.99 r_max`` if
      ``l_p < 2 eps``, else ``max(l_p, (r_max + l_p)/2)``; ``smart``
      (``Smart_Subspace: .true.``) takes ``max(min(g, middle), 0)`` when
      ``r_max > 0`` and ``r_max`` otherwise (``middle`` of the Lanczos bound),
      ``adaptive`` (``.false.``) takes ``max(g, 0)``;
    * later calls: ``smart`` takes ``r_max``; ``adaptive`` takes ``l_p`` if
      ``q = |r_max/l_p| > 1.3``, ``(l_p + r_max)/2`` if ``q > 1``, else
      ``r_max + 0.05``;
    * if then the upper bound ``u`` lies below the lower bound ``l``:
      ``l' = (u + theta_min)/2`` and ``u' = u + 0.1 (l - u)``.

    Returns ``(lower, upper)``.
    """

    largest = float(np.max(state.eigenvalues))
    previous = (
        _MACHINE_EPSILON if state.filter_lower_bound is None else float(state.filter_lower_bound)
    )
    if state.first_filter:
        if previous < 2.0 * _MACHINE_EPSILON:
            suggestion = 0.99 * largest
        else:
            suggestion = max(previous, 0.5 * (largest + previous))
        if rule == "smart":
            lower = max(min(suggestion, float(bound.middle)), 0.0) if largest > 0.0 else largest
        else:
            lower = max(suggestion, 0.0)
    elif rule == "smart":
        lower = largest
    else:
        ratio = abs(largest / previous) if previous != 0.0 else float("inf")
        if ratio > 1.30:
            lower = previous
        elif ratio > 1.0:
            lower = 0.5 * (previous + largest)
        else:
            lower = largest + 0.05
    upper = float(bound.upper_bound)
    if upper < lower:
        suggestion = lower
        lower = 0.5 * (upper + float(bound.lower_bound))
        upper = upper + 0.1 * (suggestion - upper)
    return float(lower), float(upper)


def _adapt_polynomial_degree(
    requested_degree: int,
    lower_bound: float,
    upper_bound: float,
) -> int:
    """Raise too-small degrees for exceptionally wide spectral intervals.

    PARSEC applies the ordered table to ``Delta E = upper_bound-lower_bound``:

    ``Delta E > 1400 -> p >= 15``
    ``Delta E > 1200 -> p >= 11``
    ``Delta E > 1000 -> p >=  9``
    ``Delta E >  850 -> p >=  8``.

    The widths are in Rydberg.  The first matching branch wins.  These are
    lower limits, so an already larger user/SCF degree is never reduced here.
    """

    interval = upper_bound - lower_bound
    if interval > 1400.0:
        return max(requested_degree, 15)
    if interval > 1200.0:
        return max(requested_degree, 11)
    if interval > 1000.0:
        return max(requested_degree, 9)
    if interval > 850.0:
        return max(requested_degree, 8)
    return requested_degree


def run_subspace_filter(
    operator: Any,
    state: SubspaceState,
    *,
    settings: SubspaceSettings = SubspaceSettings(),
) -> SubspaceResult:
    """Apply one PARSEC later-SCF filter and Rayleigh--Ritz rotation.

    The input vectors are the complete buffered Ritz subspace saved by the
    preceding SCF solve.  Given a new Hamiltonian ``H[V_in]``, this routine:

    1. defines the unwanted interval ``[l, u]`` from saved Ritz values and a
       fresh short-Lanczos upper estimate;
    2. applies one blockwise normalized Chebyshev filter;
    3. orthonormalizes the filtered columns; and
    4. solves ``(Q.T @ H @ Q) U = U diag(epsilon)`` and rotates ``Psi=Q@U``.

    Returned full-space residual norms are diagnostics only.  They neither
    repeat this filter nor stop the outer SCF calculation.
    """

    _validate_state(operator, state)
    # Stage 1a: the lower edge is derived from the high end of the saved
    # buffered eigenspectrum, not merely from the states occupied in SCF.
    lower_bound = _next_filter_lower_bound(state)

    # Stage 1b: obtain a fresh conservative upper edge.  PARSEC's global
    # random generator naturally advances between SCF calls.  Python uses the
    # explicit seed+saved-call-count convention to retain deterministic,
    # independently reproducible per-call starting vectors.  The same NumPy
    # generator is also used for rare orthogonalization replacements here;
    # PARSEC would use its separate saved DLARNV stream for those replacements.
    generator = np.random.default_rng(
        settings.random_seed + state.filters_completed
    )
    if settings.bounds == "parsec":
        bound = lanczos_upper_bound(
            operator,
            steps=settings.lanczos_steps,
            rng=generator,
        )
    else:
        bound = spectral_bound(
            operator,
            settings.bounds,
            settings.lanczos_steps if settings.bound_steps is None else settings.bound_steps,
            rng=generator,
        )
    upper_bound = float(bound.upper_bound)
    if settings.lower_bound_rule != "monotone":
        lower_bound, upper_bound = parsec15_lower_bound(
            state, settings.lower_bound_rule, bound
        )
    degree = _adapt_polynomial_degree(
        settings.polynomial_degree,
        lower_bound,
        upper_bound,
    )
    # PARSEC rounds the degree split to complete memory blocks.  The first
    # ceil(number_of_full_blocks/2) blocks get p-delta; all other complete
    # blocks and any final partial block get p+delta.
    blocks = subspace_filter_blocks(
        state.working_states,
        settings.block_size,
        degree,
        settings.degree_delta,
    )

    # Stage 2: in the non-BETA Fortran branch, lower_bound is both the start of
    # the unwanted interval and the recurrence's normalization reference.
    # The production false reset flag also retains PARSEC's scalar-sigma carry
    # from one memory block to the next.
    filtered = subspace_filter(
        operator,
        state.vectors,
        degree=degree,
        degree_delta=settings.degree_delta,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        block_size=settings.block_size,
        reset_recurrence_per_block=settings.reset_recurrence_per_block,
        # Chebyshev_Scaled (1.5: lowerb0 = aL, subspace.f90z:449-453).
        reference_eigenvalue=(
            float(np.min(state.eigenvalues)) if settings.scaled_filter else None
        ),
    )
    # Stages 3--4: restore an orthonormal basis and diagonalize only the small
    # projected Hamiltonian.  H itself remains matrix-free.
    basis = orthonormalize(
        filtered,
        rng=generator,
        reorthogonalization_threshold=settings.reorthogonalization_threshold,
    ).basis
    ritz = rayleigh_ritz(operator, basis)
    eigenvalues = np.asarray(ritz.eigenvalues, dtype=float)
    vectors = np.asarray(ritz.wavefunctions)
    residual_norms = np.asarray(ritz.residual_norms, dtype=float)

    expected_values_shape = (state.working_states,)
    expected_vectors_shape = (
        state.operator_dimension,
        state.working_states,
    )
    if eigenvalues.shape != expected_values_shape:
        raise RuntimeError(
            "Rayleigh--Ritz changed the filtered subspace's eigenvalue count"
        )
    if vectors.shape != expected_vectors_shape:
        raise RuntimeError(
            "Rayleigh--Ritz changed the filtered subspace's vector dimensions"
        )

    # Save all working (requested + safety-buffer) pairs.  The lower boundary
    # saved here is the one used for this filter; the next call combines it
    # with the then-current largest buffered Ritz value.
    next_state = SubspaceState(
        operator_dimension=state.operator_dimension,
        working_states=state.working_states,
        eigenvalues=eigenvalues,
        vectors=vectors,
        filter_lower_bound=lower_bound,
        first_filter=False,
        filters_completed=state.filters_completed + 1,
    )
    return SubspaceResult(
        eigenvalues=eigenvalues,
        vectors=vectors,
        residual_norms=residual_norms,
        state=next_state,
        lanczos_bound=bound,
        polynomial_degree_used=degree,
        filter_blocks=blocks,
    )
