"""One device-resident later-SCF PARSEC SUBSPACE update."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Any

import numpy as np

from parsec_python.Eigensolvers.subspace import SubspaceSettings

from ..backends.cupy import device_stage
from .chebyshev import FilterBlock, subspace_filter, subspace_filter_blocks
from .handover import RitzHandover, release_coefficients
from .orthogonalize import orthonormalize_complete_subspace
from .rayleigh_ritz import (
    DeviceRayleighRitzResult,
    GeneralizedRitzStabilityError,
    generalized_rayleigh_ritz,
    generalized_ritz_requested,
    rayleigh_ritz,
    streaming_ritz_requested,
)
from .spectral_bounds import LanczosBoundResult, lanczos_upper_bound


@dataclass(frozen=True)
class DeviceSubspaceState:
    """Buffered Ritz state kept on the GPU across nonlinear iterations."""

    operator_dimension: int
    working_states: int
    eigenvalues: Any
    vectors: Any
    filter_lower_bound: float | None = None
    first_filter: bool = True
    filters_completed: int = 0
    ritz_workspace: Any | None = None
    generalized_ritz_failed: bool = False
    # None: the columns are Ritz vectors.  Otherwise the basis that a later
    # pass handed over, X L^-T, with the eigenvectors that make its Ritz
    # vectors (:class:`handover.RitzHandover`, PARSEC_CUPY_RITZ_HANDOVER).
    handover: RitzHandover | None = None


@dataclass(frozen=True)
class DeviceSubspaceResult:
    eigenvalues: Any
    vectors: Any
    residual_norms: Any | None
    state: DeviceSubspaceState
    lanczos_bound: LanczosBoundResult
    polynomial_degree_used: int
    filter_blocks: tuple[FilterBlock, ...]
    rayleigh_ritz: DeviceRayleighRitzResult
    # What the Ritz step of the pass handed over where a hand-over was asked
    # for or a trim was pending (:meth:`handover.HandoverStep.summary`).
    handover_pass: dict | None = None


def as_ritz_state(state: DeviceSubspaceState) -> DeviceSubspaceState:
    """The state of a basis whose memory holds Ritz vectors again, a pending trim applied.

    For a hand-over record that was rotated in place since
    (:attr:`handover.RitzHandover.rotated`): the trim that was pending, or
    else the present width, is the view of the leading columns that a trim
    of Ritz vectors takes, and the state has no record.
    """

    record = state.handover
    if record is None:
        return state
    count = state.working_states if record.pending_count is None else int(record.pending_count)
    return replace(
        state,
        working_states=count,
        eigenvalues=state.eigenvalues[:count],
        vectors=state.vectors[:, :count],
        handover=None,
    )


def _validate_state(operator: Any, state: DeviceSubspaceState) -> None:
    dimension = int(operator.shape[0])
    if tuple(operator.shape) != (dimension, dimension):
        raise ValueError("operator must be square")
    if state.operator_dimension != dimension:
        raise ValueError("saved subspace belongs to another operator dimension")
    if state.eigenvalues.shape != (state.working_states,):
        raise ValueError("saved eigenvalues do not match working_states")
    if state.vectors.shape != (dimension, state.working_states):
        raise ValueError("saved vectors do not match the operator")
    if not state.first_filter and state.filter_lower_bound is None:
        raise ValueError("continued state requires a filter lower bound")


def _next_filter_lower_bound(state: DeviceSubspaceState) -> float:
    largest = float(state.eigenvalues.max().item())
    if state.first_filter or state.filter_lower_bound is None:
        return max(largest, 0.0)
    return max(float(state.filter_lower_bound), largest)


def adapt_polynomial_degree(
    requested_degree: int,
    lower_bound: float,
    upper_bound: float,
) -> int:
    """Apply PARSEC's spectral-width lower limits to the filter degree."""

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


def filter_degree(
    settings: SubspaceSettings,
    state: DeviceSubspaceState,
    lower_bound: float,
    upper_bound: float,
) -> int:
    """Polynomial degree of this pass, including PARSEC's spectral-width floor."""

    requested_degree=settings.polynomial_degree
    experimental_cap=int(os.environ.get("PARSEC_CUPY_FILTER_DEGREE_CAP", "0"))
    if experimental_cap:
        if experimental_cap<=settings.degree_delta:
            raise ValueError("experimental filter cap must exceed degree_delta")
        if state.filters_completed>=5:
            requested_degree=min(requested_degree,experimental_cap)
    return adapt_polynomial_degree(requested_degree, lower_bound, upper_bound)


def run_subspace_filter(
    operator: Any,
    state: DeviceSubspaceState,
    *,
    settings: SubspaceSettings = SubspaceSettings(),
    compute_residuals: bool = True,
    spectral_bound: LanczosBoundResult | None = None,
    consume_state: bool = False,
    handover: str = "rotation",
) -> DeviceSubspaceResult:
    """Perform one filter/orthogonalization/Ritz pass on the GPU.

    Residuals are optional because the later-SCF PARSEC policy does not use
    them to accept eigenpairs or to control SCF convergence.  Diagnostic
    callers retain the default; the production accelerated SCF disables the
    extra ``(H Q) C - (Q C) Lambda`` grid-by-state work.

    ``consume_state`` explicitly transfers ownership of the old vectors.
    After filtering they may be overwritten by HX and the rotated result.
    Callers must not retain snapshots/views of the input state when enabled.

    ``handover="triangular"`` hands the next pass ``X L^-T`` and keeps the
    eigenvectors that make its Ritz vectors in the state
    (``PARSEC_CUPY_RITZ_HANDOVER``, :mod:`handover`), where the generalized
    step streams one array.  A state that holds a hand-over with a pending
    trim filters all its columns, rotates, and keeps the Ritz pairs of the
    trim: the trim of a hand-over is carried out here.
    """

    record_in = state.handover
    if record_in is not None and record_in.rotated:
        # Rotated in place since the pass that handed it over: Ritz vectors.
        state = as_ritz_state(state)
    pending = None if state.handover is None else state.handover.pending_count
    # The filter overwrites the basis that they describe.
    release_coefficients(state.handover)
    mode = "triangular" if handover == "triangular" and pending is None else "rotation"
    note = None
    if handover != "rotation" and mode == "rotation":
        note = "deferred trim"
    _validate_state(operator, state)
    lower_bound = _next_filter_lower_bound(state)
    generator = np.random.default_rng(
        settings.random_seed + state.filters_completed
    )
    with device_stage(operator, "subspace_bound_seconds"):
        bound = (
            lanczos_upper_bound(
                operator, steps=settings.lanczos_steps, rng=generator
            )
            if spectral_bound is None
            else spectral_bound
        )
    upper_bound = float(bound.upper_bound)
    degree = filter_degree(settings, state, lower_bound, upper_bound)
    blocks = subspace_filter_blocks(
        state.working_states,
        settings.block_size,
        degree,
        settings.degree_delta,
    )
    # A consumed state may be filtered in place when the following Ritz solve
    # needs no second tall array either (see ``streaming_ritz_requested``).
    filter_in_place = bool(
        consume_state
        and not compute_residuals
        and streaming_ritz_requested(operator)
    )
    with device_stage(operator, "subspace_filter_seconds"):
        filtered = subspace_filter(
            operator,
            state.vectors,
            degree=degree,
            degree_delta=settings.degree_delta,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            block_size=settings.block_size,
            reset_recurrence_per_block=settings.reset_recurrence_per_block,
            out=state.vectors if filter_in_place else None,
        )
    use_generalized_ritz = bool(
        not state.generalized_ritz_failed
        and generalized_ritz_requested(*map(int, filtered.shape))
    )
    generalized_ritz_failed = state.generalized_ritz_failed
    if use_generalized_ritz:
        try:
            with device_stage(operator, "subspace_ritz_seconds"):
                ritz = generalized_rayleigh_ritz(
                    operator,
                    filtered,
                    workspace=(state.vectors if consume_state and not compute_residuals
                               else state.ritz_workspace),
                    compute_residuals=bool(compute_residuals),
                    consume_basis=True,
                    handover=mode,
                )
        except GeneralizedRitzStabilityError:
            # An ill-conditioned overlap is not an SCF failure.  Householder
            # QR is the robust complete-basis fallback and remains selected
            # for subsequent iterations of this eigensolver state.
            generalized_ritz_failed = True
            note = "QR fallback"
            with device_stage(operator, "subspace_orthogonalization_seconds"):
                basis = orthonormalize_complete_subspace(
                    filtered, rng=generator
                ).basis
            with device_stage(operator, "subspace_ritz_seconds"):
                ritz = rayleigh_ritz(
                    operator,
                    basis,
                    compute_residuals=bool(compute_residuals),
                )
    else:
        note = "QR route"
        with device_stage(operator, "subspace_orthogonalization_seconds"):
            basis = orthonormalize_complete_subspace(filtered, rng=generator).basis
        with device_stage(operator, "subspace_ritz_seconds"):
            ritz = rayleigh_ritz(
                operator,
                basis,
                compute_residuals=bool(compute_residuals),
            )
    if ritz.eigenvalues.shape != (state.working_states,):
        raise RuntimeError("Rayleigh--Ritz changed the working state count")
    if ritz.wavefunctions.shape != (
        state.operator_dimension,
        state.working_states,
    ):
        raise RuntimeError("Rayleigh--Ritz changed the saved vector shape")

    if consume_state and not compute_residuals:
        # The consumed old input supplies next iteration's scratch. Retaining
        # a second persistent tall workspace defeats the ownership transfer.
        ritz = replace(ritz, workspace=None)
    step = ritz.handover
    record = None
    if step is not None and step.mode == "triangular":
        record = RitzHandover(
            step.coefficients,
            int(step.coefficients.device.id),
            state.working_states,
        )
    eigenvalues, vectors, working = ritz.eigenvalues, ritz.wavefunctions, state.working_states
    residual_norms = ritz.residual_norms
    if pending is not None:
        # The trim of the pass before, on the Ritz vectors of this one: the
        # view of the leading columns, as a trim of Ritz vectors takes it.
        pending = int(pending)
        eigenvalues, vectors, working = eigenvalues[:pending], vectors[:, :pending], pending
        if residual_norms is not None:
            residual_norms = residual_norms[:pending]
    handover_pass = None
    if handover != "rotation" or pending is not None:
        handover_pass = (
            step.summary()
            if step is not None
            else dict(mode="rotation", condition=None, note=note, band=None, product=None)
        )
        if step is not None and step.mode == "rotation" and note is not None:
            handover_pass["note"] = note
        handover_pass["deferred_trim"] = pending
    next_state = DeviceSubspaceState(
        operator_dimension=state.operator_dimension,
        working_states=working,
        eigenvalues=eigenvalues,
        vectors=vectors,
        filter_lower_bound=lower_bound,
        first_filter=False,
        filters_completed=state.filters_completed + 1,
        ritz_workspace=ritz.workspace,
        generalized_ritz_failed=generalized_ritz_failed,
        handover=record,
    )
    return DeviceSubspaceResult(
        eigenvalues=eigenvalues,
        vectors=vectors,
        residual_norms=residual_norms,
        state=next_state,
        lanczos_bound=bound,
        polynomial_degree_used=degree,
        filter_blocks=blocks,
        rayleigh_ritz=ritz,
        handover_pass=handover_pass,
    )


__all__ = [
    "DeviceSubspaceResult",
    "DeviceSubspaceState",
    "SubspaceSettings",
    "adapt_polynomial_degree",
    "as_ritz_state",
    "filter_degree",
    "run_subspace_filter",
]
