"""Float64 CuPy translation of PARSEC's Chebyshev--Davidson solver.

This is the device counterpart of
``parsec_python.Eigensolvers.chebdav``.  It follows the same non-``BETA``
``flag='new'`` policy from ``chebdav.f90z``: a short Lanczos bound estimate,
Chebyshev-filtered Davidson expansion blocks, contiguous-prefix locking,
inner and outer restarts, and PARSEC's approximate filtered-subspace cleanup
when the restart/iteration limits are reached.

All grid-sized bases, Hamiltonian images, residuals, and Ritz vectors remain
CuPy float64 arrays.  Only sequential control scalars, compact iteration
diagnostics, deterministic random-vector generation, and the tiny Lanczos
tridiagonal solve cross to the host.  No CPU replacement for CHEBDAV or
alternate algorithm is used as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from parsec_python.Eigensolvers.chebdav import (
    ChebDavIteration,
    ChebDavSettings,
)

from ..backends.cupy import device_stage, require_cupy
from .small_dense import symmetric_eigh
from .chebyshev import chebff_filter, chebyshev_filter
from .lapack_random import LapackRandom
from .orthogonalize import (
    chebdav_block_orth_requested,
    orthonormalize,
    orthonormalize_appended_block,
    orthonormalize_complete_appended_subspace,
)
from .spectral_bounds import LanczosBoundResult, lanczos_upper_bound


@dataclass(frozen=True)
class DeviceChebDavState:
    """Complete CHEBDAV state with all grid-sized arrays on the GPU.

    The fields intentionally match the reference ``ChebDavState``.  As in
    PARSEC, ``matrix_vector_products`` omits the final cleanup projection's
    Hamiltonian application even though that operation is performed.
    """

    operator_dimension: int
    wanted_states: int
    maximum_subspace_dimension: int
    eigenvalues: Any
    vectors: Any
    residual_norms: Any
    truly_converged_states: int
    approximate_cleanup_used: bool
    matrix_vector_products: int
    iterations_completed: int
    inner_restarts: int
    outer_restarts: int
    filter_lower_bound: float
    spectral_upper_bound: float


@dataclass(frozen=True)
class DeviceChebDavResult:
    """Lowest device-resident Ritz pairs and CHEBDAV diagnostics."""

    eigenvalues: Any
    vectors: Any
    residual_norms: Any
    state: DeviceChebDavState
    lanczos_bound: LanczosBoundResult
    iterations: tuple[ChebDavIteration, ...]

    @property
    def approximate_converged_states(self) -> int:
        """Match PARSEC's post-cleanup reported convergence count."""

        return self.state.wanted_states


def _operator_dimension(operator: Any) -> int:
    shape = getattr(operator, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("operator must expose a square two-dimensional shape")
    dimension = int(shape[0])
    if dimension < 1:
        raise ValueError("operator dimension must be positive")
    dtype = getattr(operator, "dtype", np.float64)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        raise NotImplementedError(
            "CuPy CHEBDAV currently implements PARSEC's real scalar path"
        )
    return dimension


def _apply(operator: Any, vectors: Any):
    """Apply ``operator`` without moving the vector block off the device."""

    cp, _ = require_cupy()
    block = cp.asarray(vectors, dtype=cp.float64)
    result = cp.asarray(operator @ block, dtype=cp.float64)
    if result.shape != block.shape:
        raise ValueError("operator must preserve the vector or block shape")
    # Do not reduce ``isfinite(result)`` here.  This helper is called in the
    # Davidson loop, so ``.item()`` would insert an otherwise unnecessary
    # device synchronization after every Hamiltonian block.  The complete
    # returned eigensystem is checked once at the coarse solver boundary.
    return result


def _rayleigh_ritz_upper(operator: Any, basis: Any):
    """Device Rayleigh--Ritz with CHEBDAV's authoritative upper triangle.

    ``chebdav.f90z`` calls ``DSYEV('U')``.  Mirroring only the computed upper
    triangle before ``cupy.linalg.eigh`` preserves that floating-point policy;
    the general accelerated Rayleigh--Ritz helper deliberately uses the lower
    triangle for the other PARSEC eigensolvers.
    """

    cp, _ = require_cupy()
    basis = cp.asarray(basis, dtype=cp.float64)
    applied = _apply(operator, basis)
    raw = basis.T @ applied
    upper = cp.triu(raw)
    projected = upper + cp.triu(upper, 1).T
    eigenvalues, rotations = symmetric_eigh(projected)
    vectors = basis @ rotations
    applied_vectors = applied @ rotations
    residual_norms = cp.linalg.norm(
        applied_vectors - vectors * eigenvalues[None, :], axis=0
    )
    return eigenvalues, vectors, residual_norms


def _active_subspace_limit(
    wanted_states: int,
    block_size: int,
    maximum_subspace_dimension: int,
) -> int:
    """Translate the piecewise ``kactmax`` table in ``chebdav.f90z``."""

    if wanted_states >= 2000:
        limit = min(max(15 * block_size, wanted_states // 10), 400)
    elif wanted_states >= 1000:
        limit = max(15 * block_size, 150)
    elif wanted_states >= 500:
        limit = max(15 * block_size, 80)
    elif wanted_states >= 300:
        limit = max(14 * block_size, 70)
    elif wanted_states >= 80:
        limit = max(10 * block_size, 50)
    elif wanted_states >= 36:
        limit = 36
    else:
        limit = 30
    return max(block_size, min(limit, maximum_subspace_dimension))


def _source_max_iterations(
    wanted_states: int,
    block_size: int,
    active_limit: int,
) -> int:
    iteration_limit = max(min(wanted_states * 5, 400), wanted_states * 3)
    return max(active_limit + 20 * block_size, iteration_limit)


def _initial_lower_bound(lowest_estimate: float, upper_bound: float) -> float:
    return max(0.0, (2.2 * lowest_estimate + 0.8 * upper_bound) / 3.0)


def _inner_restart_size(active_limit: int, block_size: int) -> int:
    retained = min(
        max(active_limit // 2, active_limit - 2 * block_size) + 2,
        active_limit - block_size,
    )
    return max(block_size, retained)


def _orthonormalize_appended(
    operator: Any,
    basis: Any,
    existing_columns: int,
    appended_columns: int,
    generator: LapackRandom,
) -> None:
    """Run PARSEC ``orth_normal`` only on a newly appended device block."""

    stop = existing_columns + appended_columns
    use_block = chebdav_block_orth_requested(
        int(basis.shape[0]), appended_columns
    )
    if use_block:
        result = orthonormalize_appended_block(
            basis,
            existing_columns=existing_columns,
            active_columns=stop,
            rng=generator,
        )
        stats = getattr(operator, "timing_stats", None)
        if stats is not None:
            stats.initial_block_orth_calls += 1
            if result.algorithm != "block_cgs2_device_mgs2":
                stats.initial_block_orth_fallbacks += 1
        return
    orthonormalize(
        basis[:, :stop],
        existing_columns=existing_columns,
        existing_are_orthonormal=True,
        rng=generator,
        copy=False,
    )


def _insert_locked_pair(
    eigenvalues: Any,
    residual_norms: Any,
    basis: Any,
    locked_count: int,
    value: float,
    residual_norm: float,
    vector_column: int,
) -> tuple[int, bool]:
    """Insert one locked pair into the ascending device-resident prefix."""

    cp, _ = require_cupy()
    vector = basis[:, vector_column].copy()
    position = int(
        cp.searchsorted(eigenvalues[:locked_count], value, side="right").item()
    )
    reordered = position < locked_count
    if reordered:
        eigenvalues[position + 1 : locked_count + 1] = eigenvalues[
            position:locked_count
        ].copy()
        residual_norms[position + 1 : locked_count + 1] = residual_norms[
            position:locked_count
        ].copy()
        basis[:, position + 1 : locked_count + 1] = basis[
            :, position:locked_count
        ].copy()
    eigenvalues[position] = value
    residual_norms[position] = residual_norm
    basis[:, position] = vector
    return locked_count + 1, reordered


def _stable_lowest_order(values: Any, count: int):
    """Return a stable ascending device ordering for the lowest ``count``."""

    cp, _ = require_cupy()
    # CuPy's stable sort preserves the original column order for exact ties,
    # matching ``np.argsort(..., kind="stable")`` in the reference solver.
    return cp.argsort(values, kind="stable")[:count]


def _ritz_scalar(
    device_values: Any,
    host_values: np.ndarray | None,
    index: int,
) -> float:
    """Read a Ritz scalar, reusing the existing host LAPACK result when set."""

    if host_values is not None:
        return float(host_values[index])
    return float(device_values[index].item())


def _final_approximate_subspace(
    operator: Any,
    basis: Any,
    locked_values: Any,
    locked_residuals: Any,
    locked_count: int,
    source_subspace_dimension: int,
    wanted_states: int,
    polynomial_degree: int,
    block_size: int,
    lower_bound: float,
    upper_bound: float,
    generator: LapackRandom,
):
    """Apply CHEBDAV's restart-limited filtered-subspace cleanup on CUDA."""

    cp, _ = require_cupy()
    bounded_source_dimension = min(
        int(source_subspace_dimension),
        int(basis.shape[1]),
    )
    target_columns = (
        wanted_states + 1
        if bounded_source_dimension > wanted_states
        and basis.shape[1] > wanted_states
        else wanted_states
    )

    if bounded_source_dimension < wanted_states:
        missing = wanted_states - bounded_source_dimension
        replacement = generator.uniform_minus_1_1(
            (basis.shape[0], missing), column_major=True
        )
        basis[:, bounded_source_dimension:wanted_states] = cp.asarray(
            replacement, dtype=cp.float64
        )

    active_count = target_columns - locked_count
    if active_count <= 0:
        order = _stable_lowest_order(locked_values[:locked_count], wanted_states)
        return (
            locked_values[:locked_count][order][:wanted_states].copy(),
            basis[:, :locked_count][:, order][:, :wanted_states].copy(),
            locked_residuals[:locked_count][order][:wanted_states].copy(),
            0,
        )

    active = basis[:, locked_count:target_columns]
    active[:, :] = chebff_filter(
        operator,
        active,
        degree=polynomial_degree,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        reference_eigenvalue=lower_bound,
        block_size=block_size,
        reset_recurrence_per_block=False,
    )
    if active_count > max(4 * block_size, 32):
        # The source cleanup may add the complete missing eigenspace in one
        # operation.  Route that wide suffix through the large-subspace path;
        # the ordinary Davidson loop continues to use its exact six-column
        # PARSEC block orthogonalization above.
        orthonormalize_complete_appended_subspace(
            basis[:, :target_columns],
            existing_columns=locked_count,
            active_columns=target_columns,
            rng=generator,
        )
    else:
        _orthonormalize_appended(
            operator,
            basis,
            locked_count,
            active_count,
            generator,
        )
    active_values, active_vectors, active_residuals = _rayleigh_ritz_upper(
        operator,
        basis[:, locked_count:target_columns],
    )

    values = cp.concatenate((locked_values[:locked_count], active_values))
    vectors = cp.concatenate(
        (basis[:, :locked_count], active_vectors), axis=1
    )
    residuals = cp.concatenate(
        (locked_residuals[:locked_count], active_residuals)
    )
    order = _stable_lowest_order(values, wanted_states)
    return (
        values[order].copy(),
        vectors[:, order].copy(),
        residuals[order].copy(),
        # PARSEC counts the filter but omits the cleanup projection H action.
        polynomial_degree * active_count,
    )


def run_chebdav(
    operator: Any,
    wanted_states: int,
    *,
    settings: ChebDavSettings = ChebDavSettings(),
    spectral_bound: LanczosBoundResult | None = None,
) -> DeviceChebDavResult:
    """Compute the lowest states with PARSEC's first-SCF CHEBDAV policy.

    ``wanted_states`` is the complete physical-plus-safety working count.  The
    fixed Davidson workspace window is allocated separately.  Returned Ritz
    arrays and the state retained for later SUBSPACE filtering remain on the
    active CUDA device.
    """

    cp, _ = require_cupy()
    dimension = _operator_dimension(operator)
    wanted_states = int(wanted_states)
    if not 1 <= wanted_states <= dimension:
        raise ValueError(
            f"wanted_states must be between 1 and {dimension}, got {wanted_states}"
        )

    block_size = min(
        int(settings.block_size),
        wanted_states,
        max(1, dimension // 3),
    )
    maximum_subspace_dimension = min(
        dimension,
        wanted_states + int(settings.workspace_window),
    )
    active_limit = _active_subspace_limit(
        wanted_states,
        block_size,
        maximum_subspace_dimension,
    )
    iteration_limit = (
        _source_max_iterations(wanted_states, block_size, active_limit)
        if settings.max_iterations is None
        else int(settings.max_iterations)
    )

    basis_generator = LapackRandom()
    if spectral_bound is None:
        bound_generator = np.random.default_rng(settings.random_seed)
        with device_stage(operator, "initial_bound_seconds"):
            bound = lanczos_upper_bound(
                operator,
                steps=settings.lanczos_steps,
                rng=bound_generator,
            )
    else:
        bound = spectral_bound
    upper_bound = float(bound.upper_bound)
    lower_bound = _initial_lower_bound(bound.lower_bound, upper_bound)
    if not lower_bound < upper_bound:
        raise RuntimeError("CHEBDAV estimated an invalid filter interval")

    basis = cp.zeros(
        (dimension, maximum_subspace_dimension), dtype=cp.float64
    )
    applied_basis = cp.zeros((dimension, active_limit), dtype=cp.float64)
    projected = cp.zeros((active_limit, active_limit), dtype=cp.float64)
    locked_values = cp.empty(maximum_subspace_dimension, dtype=cp.float64)
    locked_residuals = cp.empty(maximum_subspace_dimension, dtype=cp.float64)
    full_workspace_ritz_projection = os.environ.get(
        "PARSEC_CUPY_CHEBDAV_FULL_WORKSPACE_RITZ", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    ritz_projection_workspace = (
        cp.empty(
            (maximum_subspace_dimension, block_size),
            dtype=cp.float64,
            order="F",
        )
        if full_workspace_ritz_projection and basis.flags.c_contiguous
        else None
    )

    initial_host = basis_generator.uniform_minus_1_1(
        (dimension, block_size), column_major=True
    )
    initial = cp.asarray(initial_host, dtype=cp.float64, order="F")
    with device_stage(operator, "initial_filter_seconds"):
        basis[:, :block_size] = chebyshev_filter(
            operator,
            initial,
            degree=settings.polynomial_degree,
            lower_bound=lower_bound,
        upper_bound=upper_bound,
        reference_eigenvalue=lower_bound,
        )
    del initial_host, initial
    with device_stage(operator, "initial_orthogonalization_seconds"):
        _orthonormalize_appended(
            operator, basis, 0, block_size, basis_generator
        )

    matrix_vector_products = bound.steps + settings.polynomial_degree * block_size
    locked_count = 0
    retained_active = 0
    cleanup_subspace_dimension = block_size
    running_ritz_max = 0.0
    inner_restarts = 0
    outer_restarts = 0
    records: list[ChebDavIteration] = []
    complete = False
    last_ritz_values = cp.empty(0, dtype=cp.float64)
    last_ritz_values_host: np.ndarray | None = None
    reuse_host_ritz_values = os.environ.get(
        "PARSEC_CUPY_REUSE_HOST_RITZ_VALUES", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    last_full_active_size = 0
    last_newly_locked = 0

    # Fortran uses ``num_iter <= max_iter`` and increments at loop entry, so
    # at most max_iter+1 projected Davidson solves are performed.
    for iteration in range(1, iteration_limit + 2):
        locked_before = locked_count
        new_start = locked_count + retained_active
        new_stop = new_start + block_size
        if new_stop > maximum_subspace_dimension:
            break

        new_block = basis[:, new_start:new_stop]
        with device_stage(operator, "initial_projection_seconds"):
            applied_new = _apply(operator, new_block)
            applied_basis[:, retained_active : retained_active + block_size] = (
                applied_new
            )
        matrix_vector_products += block_size

        full_active_size = retained_active + block_size
        source_subspace_dimension = locked_before + full_active_size
        cleanup_subspace_dimension = source_subspace_dimension
        active_basis = basis[
            :, locked_count : locked_count + full_active_size
        ]
        with device_stage(operator, "initial_projection_seconds"):
            if ritz_projection_workspace is None:
                last_columns = active_basis.T @ applied_new
            else:
                cp.matmul(
                    basis.T,
                    applied_new,
                    out=ritz_projection_workspace,
                )
                last_columns = ritz_projection_workspace[
                    locked_count : locked_count + full_active_size,
                    :,
                ]
            for local_column in range(block_size):
                projected_column = retained_active + local_column
                projected[
                    : projected_column + 1,
                    projected_column,
                ] = last_columns[
                    : projected_column + 1,
                    local_column,
                ]
            projected_view = projected[:full_active_size, :full_active_size]
            upper = cp.triu(projected_view)
            small = upper + cp.triu(upper, 1).T
            diagonalization = symmetric_eigh(
                small,
                return_host_values=reuse_host_ritz_values,
            )
        if reuse_host_ritz_values:
            ritz_values, rotations, ritz_values_host = diagonalization
        else:
            ritz_values, rotations = diagonalization
            ritz_values_host = None
        last_ritz_values = ritz_values
        last_ritz_values_host = ritz_values_host
        last_full_active_size = full_active_size

        inner_restart = full_active_size + block_size > active_limit
        if inner_restart:
            retained_after_rr = min(
                _inner_restart_size(active_limit, block_size),
                full_active_size,
            )
            inner_restarts += 1
        else:
            retained_after_rr = full_active_size

        with device_stage(operator, "initial_rotation_seconds"):
            basis[:, locked_count : locked_count + retained_after_rr] = (
                active_basis @ rotations[:, :retained_after_rr]
            )
            applied_basis[:, :retained_after_rr] = (
                applied_basis[:, :full_active_size]
                @ rotations[:, :retained_after_rr]
            )

        smallest_ritz = abs(
            _ritz_scalar(ritz_values, ritz_values_host, 0)
        )
        largest_ritz = abs(
            _ritz_scalar(ritz_values, ritz_values_host, -1)
        )
        running_ritz_max = max(
            running_ritz_max,
            smallest_ritz,
            largest_ritz,
        )
        residual_tolerance = settings.convergence_tolerance * running_ritz_max
        if locked_before > wanted_states // 2:
            residual_tolerance *= 2.0 * (locked_before + 1) / wanted_states

        tested: list[float] = []
        reordered = False
        with device_stage(operator, "initial_residual_seconds"):
            for active_index in range(retained_after_rr):
                vector_column = locked_before + active_index
                residual = (
                    applied_basis[:, active_index]
                    - basis[:, vector_column] * ritz_values[active_index]
                )
                residual_norm = float(cp.linalg.norm(residual).item())
                tested.append(residual_norm)
                if residual_norm >= residual_tolerance:
                    break
                locked_count, pair_reordered = _insert_locked_pair(
                    locked_values,
                    locked_residuals,
                    basis,
                    locked_count,
                    _ritz_scalar(
                        ritz_values, ritz_values_host, active_index
                    ),
                    residual_norm,
                    vector_column,
                )
                reordered = reordered or pair_reordered

        newly_locked = locked_count - locked_before
        last_newly_locked = newly_locked
        if (locked_count >= wanted_states and not reordered) or (
            locked_count >= wanted_states + 1
        ):
            records.append(
                ChebDavIteration(
                    number=iteration,
                    locked_before=locked_before,
                    locked_after=locked_count,
                    active_dimension_before_restart=full_active_size,
                    active_dimension_retained=retained_after_rr,
                    residual_tolerance=float(residual_tolerance),
                    tested_residual_norms=np.asarray(tested, dtype=np.float64),
                    inner_restart=inner_restart,
                    outer_restart=False,
                )
            )
            complete = True
            break

        active_after_lock = retained_after_rr - newly_locked
        outer_restart_test_dimension = locked_count + retained_after_rr
        outer_restart = (
            outer_restart_test_dimension
            > maximum_subspace_dimension - block_size
        )
        if outer_restart and (
            outer_restarts > settings.max_outer_restarts
            and source_subspace_dimension >= wanted_states
        ):
            cleanup_subspace_dimension = source_subspace_dimension
            records.append(
                ChebDavIteration(
                    number=iteration,
                    locked_before=locked_before,
                    locked_after=locked_count,
                    active_dimension_before_restart=full_active_size,
                    active_dimension_retained=active_after_lock,
                    residual_tolerance=float(residual_tolerance),
                    tested_residual_norms=np.asarray(tested, dtype=np.float64),
                    inner_restart=inner_restart,
                    outer_restart=True,
                )
            )
            break

        if outer_restart:
            restart_total = min(
                maximum_subspace_dimension - 2 * block_size,
                source_subspace_dimension,
            )
            restart_total = max(locked_count, restart_total)
            requested_active_after_restart = restart_total - locked_count
            active_after_lock = min(
                active_after_lock,
                requested_active_after_restart,
            )
            outer_restarts += 1

        if newly_locked:
            applied_basis[:, :active_after_lock] = applied_basis[
                :, newly_locked : newly_locked + active_after_lock
            ].copy()
        retained_active = active_after_lock
        cleanup_subspace_dimension = locked_count + retained_active

        projected[:retained_active, :retained_active] = 0.0
        if retained_active:
            diagonal_values = ritz_values[
                newly_locked : newly_locked + retained_active
            ]
            diagonal_indices = cp.arange(retained_active)
            projected[diagonal_indices, diagonal_indices] = diagonal_values

        if retained_active < block_size:
            candidates_host = basis_generator.uniform_minus_1_1(
                (dimension, block_size), column_major=True
            )
            candidates = cp.asarray(candidates_host, dtype=cp.float64, order="F")
            cutoff_base = (
                float(locked_values[locked_count - 1].item())
                if locked_count
                else lower_bound
            )
        else:
            candidates = basis[
                :, locked_count : locked_count + block_size
            ].copy()
            first_nonconverged = _ritz_scalar(
                ritz_values, ritz_values_host, newly_locked
            )
            cutoff_base = (
                max(
                    first_nonconverged,
                    float(locked_values[locked_count - 1].item()),
                )
                if locked_count
                else first_nonconverged
            )

        second_largest = _ritz_scalar(
            ritz_values,
            ritz_values_host,
            max(0, full_active_size - 2),
        )
        lower_bound = (cutoff_base + 6.0 * second_largest) / 7.0
        if lower_bound >= upper_bound:
            lower_bound = float(np.nextafter(upper_bound, -np.inf))

        append_start = locked_count + retained_active
        append_stop = append_start + block_size
        if append_stop > maximum_subspace_dimension:
            break
        with device_stage(operator, "initial_filter_seconds"):
            basis[:, append_start:append_stop] = chebyshev_filter(
                operator,
                candidates,
                degree=settings.polynomial_degree,
                lower_bound=lower_bound,
            upper_bound=upper_bound,
            reference_eigenvalue=lower_bound,
            )
        matrix_vector_products += settings.polynomial_degree * block_size
        with device_stage(operator, "initial_orthogonalization_seconds"):
            _orthonormalize_appended(
                operator,
                basis,
                append_start,
                block_size,
                basis_generator,
            )

        records.append(
            ChebDavIteration(
                number=iteration,
                locked_before=locked_before,
                locked_after=locked_count,
                active_dimension_before_restart=full_active_size,
                active_dimension_retained=retained_active,
                residual_tolerance=float(residual_tolerance),
                tested_residual_norms=np.asarray(tested, dtype=np.float64),
                inner_restart=inner_restart,
                outer_restart=outer_restart,
            )
        )

    true_converged = min(locked_count, wanted_states)
    approximate_cleanup_used = not complete

    if complete:
        order = _stable_lowest_order(
            locked_values[:locked_count], wanted_states
        )
        eigenvalues = locked_values[:locked_count][order].copy()
        vectors = basis[:, :locked_count][:, order].copy()
        residual_norms = locked_residuals[:locked_count][order].copy()
    else:
        if last_ritz_values.size:
            first_index = min(last_newly_locked, last_ritz_values.size - 1)
            cutoff_base = _ritz_scalar(
                last_ritz_values,
                last_ritz_values_host,
                first_index,
            )
            if locked_count:
                cutoff_base = max(
                    cutoff_base,
                    float(locked_values[locked_count - 1].item()),
                )
            second_largest = _ritz_scalar(
                last_ritz_values,
                last_ritz_values_host,
                max(0, last_full_active_size - 2),
            )
            lower_bound = (cutoff_base + 6.0 * second_largest) / 7.0
            if lower_bound >= upper_bound:
                lower_bound = float(np.nextafter(upper_bound, -np.inf))
        with device_stage(operator, "initial_cleanup_seconds"):
            (
                eigenvalues,
                vectors,
                residual_norms,
                cleanup_mv,
            ) = _final_approximate_subspace(
                operator,
                basis,
                locked_values,
                locked_residuals,
                locked_count,
                cleanup_subspace_dimension,
                wanted_states,
                settings.polynomial_degree,
                block_size,
                lower_bound,
            upper_bound,
            basis_generator,
        )
        matrix_vector_products += cleanup_mv

    # One coarse validation preserves an actionable nonfinite-result error
    # without synchronizing every Hamiltonian application in the Davidson
    # loop.  The reductions are queued together and only the final scalar is
    # transferred to the host.
    all_finite = (
        cp.all(cp.isfinite(eigenvalues))
        & cp.all(cp.isfinite(vectors))
        & cp.all(cp.isfinite(residual_norms))
    )
    if not bool(all_finite.item()):
        raise FloatingPointError("CHEBDAV produced a nonfinite eigensystem")

    state = DeviceChebDavState(
        operator_dimension=dimension,
        wanted_states=wanted_states,
        maximum_subspace_dimension=maximum_subspace_dimension,
        eigenvalues=eigenvalues,
        vectors=vectors,
        residual_norms=residual_norms,
        truly_converged_states=true_converged,
        approximate_cleanup_used=approximate_cleanup_used,
        matrix_vector_products=matrix_vector_products,
        iterations_completed=len(records),
        inner_restarts=inner_restarts,
        outer_restarts=outer_restarts,
        filter_lower_bound=float(lower_bound),
        spectral_upper_bound=float(upper_bound),
    )
    return DeviceChebDavResult(
        eigenvalues=eigenvalues,
        vectors=vectors,
        residual_norms=residual_norms,
        state=state,
        lanczos_bound=bound,
        iterations=tuple(records),
    )


__all__ = [
    "ChebDavIteration",
    "ChebDavSettings",
    "DeviceChebDavResult",
    "DeviceChebDavState",
    "run_chebdav",
]
