"""CuPy-resident normalized Chebyshev filters used by PARSEC."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from ..backends.cupy import require_cupy
from ..backends.cupy_mixed_precision import mixed_filter_policy


@dataclass(frozen=True)
class FilterBlock:
    """One contiguous vector block and its polynomial degree."""

    start: int
    stop: int
    degree: int


def _validate_degree(degree: int, name: str = "degree") -> int:
    integer = int(degree)
    if integer != degree or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _as_device_columns(vectors: Any):
    cp, _ = require_cupy()
    array = cp.asarray(vectors, dtype=cp.float64)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError("vectors must be a nonempty vector or column matrix")
    return array, was_vector


def _normalized_recurrence_step(
    operator: Any,
    current: Any,
    *,
    center: float,
    scale: float,
    sigma_next: float = 1.0,
    previous: Any | None = None,
    sigma: float = 0.0,
):
    """Use a fused device step when exposed, otherwise retain literal CuPy.

    ``CuPyHamiltonian.chebyshev_recurrence`` fuses the local stencil and the
    array-wide recurrence operations.  Keeping this feature-detected helper
    means all small/fake/reference operators continue to exercise the simple
    and independently readable expression below.
    """

    fused = getattr(operator, "chebyshev_recurrence", None)
    if callable(fused):
        return fused(
            current,
            center=center,
            scale=scale,
            sigma_next=sigma_next,
            previous=previous,
            sigma=sigma,
        )
    following = operator @ current
    following = (following - center * current) * scale
    if previous is not None:
        following -= sigma * previous
    following *= sigma_next
    return following


def _mixed_filter_requested(
    operator: Any,
    vector_count: int | None = None,
) -> bool:
    """Use FP32 only where asked, and under ``auto`` only for enough work.

    FP64 is the default (:func:`mixed_filter_policy`).
    """

    policy = mixed_filter_policy()
    if policy == "off":
        return False
    available = getattr(operator, "mixed_precision_recurrence", None) is not None
    if not available or policy == "on":
        return available
    if vector_count is None:
        # ``chebyshev_filter`` receives one six-vector memory block; its
        # enclosing SUBSPACE caller has already applied the complete-basis
        # work policy before opting this block in.
        return True
    raw_threshold = os.environ.get(
        "PARSEC_CUPY_MIXED_FILTER_MIN_WORK", "100000000"
    ).strip()
    try:
        threshold = int(raw_threshold)
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_MIXED_FILTER_MIN_WORK must be an integer"
        ) from error
    if threshold < 0:
        raise ValueError(
            "PARSEC_CUPY_MIXED_FILTER_MIN_WORK cannot be negative"
        )
    work = int(operator.shape[0]) * int(vector_count) * int(vector_count)
    return work >= threshold


def _mixed_recurrence_step(
    operator: Any,
    current: Any,
    **parameters: Any,
):
    return operator.chebyshev_recurrence_float32(current, **parameters)


def chebyshev_filter(
    operator: Any,
    vectors: Any,
    degree: int,
    lower_bound: float,
    upper_bound: float,
    reference_eigenvalue: float,
    *,
    initial_sigma: float | None = None,
    return_final_sigma: bool = False,
    mixed_precision: bool = False,
):
    """Apply PARSEC's normalized recurrence entirely on the GPU.

    Only scalar recurrence coefficients live on the host.  The three ``N x b``
    recurrence buffers and every Hamiltonian application stay device-resident.
    """

    cp, _ = require_cupy()
    degree = _validate_degree(degree)
    block, was_vector = _as_device_columns(vectors)
    if tuple(operator.shape) != (block.shape[0], block.shape[0]):
        raise ValueError("operator shape must match vector rows")

    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)
    reference_eigenvalue = float(reference_eigenvalue)
    if not np.isfinite([lower_bound, upper_bound, reference_eigenvalue]).all():
        raise ValueError("filter bounds and reference must be finite")
    half_span = 0.5 * (upper_bound - lower_bound)
    if half_span <= 0.0:
        raise ValueError("upper_bound must be greater than lower_bound")
    center = 0.5 * (upper_bound + lower_bound)
    denominator = reference_eigenvalue - center
    if denominator == 0.0:
        raise ValueError("reference_eigenvalue cannot equal the interval center")

    sigma_one = half_span / denominator
    sigma = sigma_one if initial_sigma is None else float(initial_sigma)
    if not np.isfinite(sigma):
        raise ValueError("initial_sigma must be finite")

    use_mixed_precision = bool(
        mixed_precision and _mixed_filter_requested(operator)
    )
    recurrence_step = (
        _mixed_recurrence_step
        if use_mixed_precision
        else _normalized_recurrence_step
    )
    recurrence_block = (
        block.astype(cp.float32, order="F") if use_mixed_precision else block
    )
    previous = recurrence_block.copy()
    current = recurrence_step(
        operator,
        recurrence_block,
        center=center,
        scale=sigma_one / half_span,
    )
    if current.shape != block.shape:
        raise ValueError("operator must preserve vector-block shape")

    for _ in range(2, degree + 1):
        sigma_next = 1.0 / (2.0 / sigma_one - sigma)
        following = recurrence_step(
            operator,
            current,
            center=center,
            scale=2.0 / half_span,
            sigma_next=sigma_next,
            previous=previous,
            sigma=sigma,
        )
        previous, current = current, following
        sigma = sigma_next

    if use_mixed_precision:
        current = current.astype(cp.float64, order="F")
    filtered = current[:, 0] if was_vector else current
    if return_final_sigma:
        return filtered, float(sigma)
    return filtered


def uniform_filter_blocks(
    vector_count: int,
    block_size: int,
    degree: int,
) -> tuple[FilterBlock, ...]:
    degree = _validate_degree(degree)
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    return tuple(
        FilterBlock(start, min(start + block_size, vector_count), degree)
        for start in range(0, vector_count, block_size)
    )


def _batching_requested() -> bool:
    """Return whether the allocation-guarded GPU block batching is enabled."""

    value = os.environ.get("PARSEC_CUPY_BATCH_FILTERS", "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _batch_workspace_fits(matrix: Any, block_count: int) -> bool:
    """Keep enough free device memory for three full recurrence work arrays."""

    if block_count <= 1:
        return False
    cp, _ = require_cupy()
    free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    # Besides the caller-owned input, batching needs previous, current, and
    # following N-by-state arrays.  Leave 40% of currently free memory for
    # cuSPARSE/cuBLAS workspaces and the persistent Hamiltonian/eigensubspace.
    required = 3 * int(matrix.nbytes)
    return required <= int(0.60 * free_bytes)


def _batched_sigma_schedule(
    blocks: tuple[FilterBlock, ...],
    sigma_one: float,
    reset_recurrence_per_block: bool,
) -> tuple[tuple[int, np.ndarray, np.ndarray], ...] | None:
    """Precompute exact per-column recurrence scalars for batched matvecs.

    PARSEC carries the final scalar ``sigma`` from one memory block to the
    next even though the vector columns themselves are independent.  This
    schedule retains that source-level behavior while allowing all blocks
    that are still active at polynomial step ``k`` to share one GPU SpMM.

    A schedule is returned only when block degrees are nondecreasing, as they
    are for CHEBFF and PARSEC's low/high-degree SUBSPACE split.  The caller
    falls back to the literal block loop for any unusual ordering.
    """

    if not blocks:
        return ()
    degrees = [block.degree for block in blocks]
    if any(right < left for left, right in zip(degrees, degrees[1:])):
        return None

    by_block: list[dict[int, tuple[float, float]]] = []
    carried_sigma: float | None = None
    for block in blocks:
        sigma = (
            sigma_one
            if reset_recurrence_per_block or carried_sigma is None
            else carried_sigma
        )
        block_steps: dict[int, tuple[float, float]] = {}
        for step in range(2, block.degree + 1):
            sigma_next = 1.0 / (2.0 / sigma_one - sigma)
            block_steps[step] = (float(sigma), float(sigma_next))
            sigma = sigma_next
        carried_sigma = sigma
        by_block.append(block_steps)

    schedule: list[tuple[int, np.ndarray, np.ndarray]] = []
    for step in range(2, max(degrees) + 1):
        active_block = next(
            (index for index, block in enumerate(blocks) if block.degree >= step),
            None,
        )
        if active_block is None:
            continue
        start = blocks[active_block].start
        stop = blocks[-1].stop
        sigma = np.empty(stop - start, dtype=np.float64)
        sigma_next = np.empty(stop - start, dtype=np.float64)
        for block, block_steps in zip(
            blocks[active_block:], by_block[active_block:]
        ):
            if block.degree < step:
                return None
            current_value, next_value = block_steps[step]
            relative_start = block.start - start
            relative_stop = block.stop - start
            sigma[relative_start:relative_stop] = current_value
            sigma_next[relative_start:relative_stop] = next_value
        schedule.append((start, sigma, sigma_next))
    return tuple(schedule)


def _batched_block_filter(
    operator: Any,
    matrix: Any,
    blocks: tuple[FilterBlock, ...],
    lower_bound: float,
    upper_bound: float,
    reference_eigenvalue: float,
    reset_recurrence_per_block: bool,
):
    """Apply PARSEC block recurrences with one SpMM per polynomial step."""

    cp, _ = require_cupy()
    half_span = 0.5 * (float(upper_bound) - float(lower_bound))
    if half_span <= 0.0:
        raise ValueError("upper_bound must be greater than lower_bound")
    center = 0.5 * (float(upper_bound) + float(lower_bound))
    denominator = float(reference_eigenvalue) - center
    if denominator == 0.0:
        raise ValueError("reference_eigenvalue cannot equal the interval center")
    sigma_one = half_span / denominator
    schedule = _batched_sigma_schedule(
        blocks,
        sigma_one,
        reset_recurrence_per_block,
    )
    if schedule is None:
        return None

    previous = matrix.copy()
    current = operator @ matrix
    current -= center * matrix
    current *= sigma_one / half_span
    for start, host_sigma, host_sigma_next in schedule:
        active = current[:, start:]
        following = operator @ active
        following -= center * active
        following *= 2.0 / half_span
        sigma = cp.asarray(host_sigma, dtype=cp.float64)
        sigma_next = cp.asarray(host_sigma_next, dtype=cp.float64)
        following -= previous[:, start:] * sigma[None, :]
        following *= sigma_next[None, :]
        previous[:, start:] = active
        current[:, start:] = following
    return current


def chebff_filter(
    operator: Any,
    vectors: Any,
    degree: int,
    lower_bound: float,
    upper_bound: float,
    reference_eigenvalue: float,
    *,
    block_size: int = 6,
    reset_recurrence_per_block: bool = False,
    batch_blocks: bool | None = None,
    out: Any | None = None,
):
    """Apply CHEBFF in memory blocks, preserving PARSEC's sigma carry.

    ``out`` (which may be ``vectors`` itself when the caller owns it) is used
    only by the captured-graph route; always use the returned array.
    """

    cp, _ = require_cupy()
    matrix, was_vector = _as_device_columns(vectors)
    blocks = uniform_filter_blocks(matrix.shape[1], block_size, degree)
    from .filter_graph import graph_filter

    captured = graph_filter(
        operator, matrix, blocks, lower_bound, upper_bound,
        reference_eigenvalue, reset_recurrence_per_block,
        out=out if out is matrix else None,
    )
    if captured is not None:
        return captured[:, 0] if was_vector else captured
    use_batching = _batching_requested() if batch_blocks is None else batch_blocks
    if use_batching and _batch_workspace_fits(matrix, len(blocks)):
        batched = _batched_block_filter(
            operator,
            matrix,
            blocks,
            lower_bound,
            upper_bound,
            reference_eigenvalue,
            reset_recurrence_per_block,
        )
        if batched is not None:
            return batched[:, 0] if was_vector else batched

    order = "F" if os.environ.get("PARSEC_CUPY_FILTER_COLUMN_MAJOR", "0") == "1" else "K"
    filtered = cp.empty_like(matrix, dtype=cp.float64, order=order)
    carried_sigma: float | None = None
    for block in blocks:
        result, final_sigma = chebyshev_filter(
            operator,
            matrix[:, block.start : block.stop],
            block.degree,
            lower_bound,
            upper_bound,
            reference_eigenvalue,
            initial_sigma=None if reset_recurrence_per_block else carried_sigma,
            return_final_sigma=True,
            mixed_precision=False,
        )
        filtered[:, block.start : block.stop] = result
        carried_sigma = final_sigma
    return filtered[:, 0] if was_vector else filtered


def subspace_filter_blocks(
    vector_count: int,
    block_size: int,
    degree: int,
    degree_delta: int,
) -> tuple[FilterBlock, ...]:
    """Return PARSEC's block-rounded low/high-degree SUBSPACE split."""

    vector_count = int(vector_count)
    block_size = int(block_size)
    degree = _validate_degree(degree)
    degree_delta = int(degree_delta)
    if vector_count < 1 or block_size < 1:
        raise ValueError("vector_count and block_size must be positive")
    if degree_delta < 0 or degree_delta >= degree:
        raise ValueError("degree_delta must be nonnegative and smaller than degree")

    full_block_count = vector_count // block_size
    low_block_count = (full_block_count + 1) // 2
    blocks: list[FilterBlock] = []
    for block_index in range(full_block_count):
        start = block_index * block_size
        blocks.append(
            FilterBlock(
                start,
                start + block_size,
                degree - degree_delta
                if block_index < low_block_count
                else degree + degree_delta,
            )
        )
    remainder_start = full_block_count * block_size
    if remainder_start < vector_count:
        blocks.append(
            FilterBlock(remainder_start, vector_count, degree + degree_delta)
        )
    return tuple(blocks)


def subspace_filter(
    operator: Any,
    vectors: Any,
    degree: int,
    degree_delta: int,
    lower_bound: float,
    upper_bound: float,
    *,
    block_size: int = 6,
    reset_recurrence_per_block: bool = False,
    batch_blocks: bool | None = None,
    mixed_precision: bool | None = None,
    out: Any | None = None,
):
    """Apply the one-pass later-SCF PARSEC filter on the GPU.

    ``out`` (which may be ``vectors`` itself when the caller owns it) is used
    only by the captured-graph and distributed routes; always use the
    returned array.
    """

    cp, _ = require_cupy()
    matrix, was_vector = _as_device_columns(vectors)
    use_mixed_precision = (
        _mixed_filter_requested(operator, int(matrix.shape[1]))
        if mixed_precision is None
        else bool(mixed_precision)
    )
    blocks = subspace_filter_blocks(
        matrix.shape[1], block_size, degree, degree_delta
    )
    if os.environ.get("PARSEC_CUPY_DISTRIBUTED_FILTER", "0") == "1":
        if use_mixed_precision:
            raise ValueError("distributed filter experiment requires FP64")
        from .distributed_filter import distributed_filter
        return distributed_filter(operator, matrix, blocks, lower_bound,
                                  upper_bound, reset_recurrence_per_block,
                                  out=out if out is matrix else None)
    if not use_mixed_precision:
        from .filter_graph import graph_filter

        captured = graph_filter(
            operator, matrix, blocks, lower_bound, upper_bound,
            lower_bound, reset_recurrence_per_block,
            out=out if out is matrix else None,
        )
        if captured is not None:
            return captured[:, 0] if was_vector else captured
    use_batching = _batching_requested() if batch_blocks is None else batch_blocks
    if use_batching and _batch_workspace_fits(matrix, len(blocks)):
        batched = _batched_block_filter(
            operator,
            matrix,
            blocks,
            lower_bound,
            upper_bound,
            lower_bound,
            reset_recurrence_per_block,
        )
        if batched is not None:
            return batched[:, 0] if was_vector else batched

    if use_mixed_precision and _mixed_filter_requested(operator):
        # The blocks below run the FP32 recurrence: ``chebyshev_filter`` asks
        # the same of the operator.  No other route of this pass does, and
        # the precision report is read from this count.
        stats = getattr(operator, "timing_stats", None)
        if stats is not None:
            stats.subspace_filter_float32_passes += 1
    order = "F" if os.environ.get("PARSEC_CUPY_FILTER_COLUMN_MAJOR", "0") == "1" else "K"
    filtered = cp.empty_like(matrix, dtype=cp.float64, order=order)
    carried_sigma: float | None = None
    for block in blocks:
        result, final_sigma = chebyshev_filter(
            operator,
            matrix[:, block.start : block.stop],
            block.degree,
            lower_bound,
            upper_bound,
            lower_bound,
            initial_sigma=None if reset_recurrence_per_block else carried_sigma,
            return_final_sigma=True,
            mixed_precision=use_mixed_precision,
        )
        filtered[:, block.start : block.stop] = result
        carried_sigma = final_sigma
    return filtered[:, 0] if was_vector else filtered


__all__ = [
    "FilterBlock",
    "chebff_filter",
    "chebyshev_filter",
    "subspace_filter",
    "subspace_filter_blocks",
    "uniform_filter_blocks",
]
