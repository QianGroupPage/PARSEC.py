"""Normalized Chebyshev filtering primitives translated from PARSEC.

Fortran provenance
------------------
PARSEC commit ``f432777750c2efc633adeb26faff641500b39b4a``:

* ``src/chebff.f90z`` routine ``cheby_filterscal1``;
* ``src/subspace.f90z`` routine ``cheby_filter``.

Both routines use the same normalized three-term recurrence but choose the
reference eigenvalue and vector-block degrees differently.  If ``[l, u]`` is
the unwanted spectral interval, define

``e = (u-l)/2``, ``c = (u+l)/2``, and ``X = (H-cI)/e``.

Thus the unwanted spectrum is mapped to ``[-1, 1]``, where Chebyshev
polynomials remain bounded, while wanted eigenvalues below ``l`` are mapped
below ``-1`` and grow rapidly with polynomial degree.  The recurrence is
normalized at a reference eigenvalue so that this growth does not overflow
the wanted part of the subspace unnecessarily.

The Fortran source leaves ``sigma`` mutated between memory blocks.  The
default preserves that source behavior; callers wanting mathematically
independent blocks must explicitly request
``reset_recurrence_per_block=True``.  This distinction matters most for
CHEBFF.  In the non-``BETA`` later-SCF filter the reference is ``l`` itself,
for which the ideal recurrence has the fixed value ``sigma=-1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FilterBlock:
    """Half-open vector-column interval and its polynomial degree."""

    start: int
    stop: int
    degree: int

    @property
    def size(self) -> int:
        return self.stop - self.start


def _validate_degree(degree: int, name: str = "degree") -> int:
    integer = int(degree)
    if integer != degree or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _as_column_matrix(vectors: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(vectors)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, np.newaxis]
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError("vectors must be a nonempty vector or column matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("vectors must contain only finite values")
    return array, was_vector


def _validate_operator(operator: Any, row_count: int) -> None:
    shape = getattr(operator, "shape", None)
    if shape is None or tuple(shape) != (row_count, row_count):
        raise ValueError("operator shape must match the vector row dimension")


def _filtered_dtype(operator: Any, vectors: np.ndarray) -> np.dtype:
    operator_dtype = getattr(operator, "dtype", np.float64)
    return np.result_type(vectors.dtype, operator_dtype, np.float64)


def chebyshev_filter(
    operator: Any,
    vectors: np.ndarray,
    degree: int,
    lower_bound: float,
    upper_bound: float,
    reference_eigenvalue: float,
    *,
    initial_sigma: float | None = None,
    return_final_sigma: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Apply PARSEC's normalized Chebyshev recurrence to one vector block.

    ``[lower_bound, upper_bound]`` is the unwanted interval.  The reference
    eigenvalue supplies the normalization point below that interval.  With

    ``X = (H-cI)/e`` and ``x0 = (reference_eigenvalue-c)/e``, PARSEC uses

    ``q0 = V``

    ``sigma1 = 1/x0``

    ``q1 = sigma1*X*q0``

    and, for ``j >= 2``,

    ``sigma_j = 1/(2/sigma1-sigma_(j-1))``

    ``q_j = sigma_j*(2*X*q_(j-1)-sigma_(j-1)*q_(j-2))``.

    When the recurrence starts with ``sigma1``, these equations produce

    ``q_m = T_m(X)V / T_m(x0)``.

    For CHEBFF, ``x0 < -1``: wanted components around the reference retain
    order-one scale while components in the mapped unwanted interval are
    suppressed by ``1/abs(T_m(x0))``.  For non-``BETA`` SUBSPACE, the
    reference equals the lower interval endpoint, so ``x0=-1``; unwanted
    components remain bounded while lower-energy components are amplified.

    Passing ``initial_sigma`` reproduces PARSEC's source-level carry of
    recurrence state from an earlier memory block.  The first-degree scaling
    still uses the block-independent ``sigma1`` exactly as in Fortran, but a
    carried value generally means the later block is no longer exactly the
    normalized polynomial written above.
    """

    degree = _validate_degree(degree)
    block, was_vector = _as_column_matrix(vectors)
    _validate_operator(operator, block.shape[0])

    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)
    reference_eigenvalue = float(reference_eigenvalue)
    if not np.isfinite([lower_bound, upper_bound, reference_eigenvalue]).all():
        raise ValueError("filter bounds and reference must be finite")
    # e and c affinely map the unwanted interval [l,u] to [-1,1].
    half_span = 0.5 * (upper_bound - lower_bound)
    if half_span <= 0.0:
        raise ValueError("upper_bound must be greater than lower_bound")
    center = 0.5 * (upper_bound + lower_bound)
    denominator = reference_eigenvalue - center
    if denominator == 0.0:
        raise ValueError("reference_eigenvalue cannot equal the interval center")

    # sigma_one = 1/x0, where x0 is the mapped reference eigenvalue.
    sigma_one = half_span / denominator
    sigma = sigma_one if initial_sigma is None else float(initial_sigma)
    if not np.isfinite(sigma):
        raise ValueError("initial_sigma must be finite")

    # q0 = V and q1 = sigma1 * ((H-cI)/e) * V.
    previous = block.copy()
    current = np.asarray(operator @ block)
    if current.shape != block.shape:
        raise ValueError("operator must preserve the vector-block shape")
    current = (current - center * block) * (sigma_one / half_span)

    # Build q_j without forming either H or T_j(H) as a dense matrix.
    for _ in range(2, degree + 1):
        sigma_next = 1.0 / (2.0 / sigma_one - sigma)
        following = np.asarray(operator @ current)
        if following.shape != block.shape:
            raise ValueError("operator must preserve the vector-block shape")
        following = (
            (following - center * current) * (2.0 / half_span)
            - sigma * previous
        )
        following *= sigma_next
        previous, current = current, following
        sigma = sigma_next

    if not np.all(np.isfinite(current)):
        raise FloatingPointError("Chebyshev recurrence produced nonfinite values")
    filtered = current[:, 0] if was_vector else current
    if return_final_sigma:
        return filtered, float(sigma)
    return filtered


def _uniform_blocks(
    vector_count: int,
    block_size: int,
    degree: int,
) -> tuple[FilterBlock, ...]:
    return tuple(
        FilterBlock(start, min(start + block_size, vector_count), degree)
        for start in range(0, vector_count, block_size)
    )


def chebff_filter(
    operator: Any,
    vectors: np.ndarray,
    degree: int,
    lower_bound: float,
    upper_bound: float,
    reference_eigenvalue: float,
    *,
    block_size: int = 7,
    reset_recurrence_per_block: bool = False,
) -> np.ndarray:
    """Apply CHEBFF's nominally uniform-degree filter in memory blocks.

    Every vector column is assigned the same polynomial degree.  Blocking
    limits temporary storage and changes only how many columns are passed to
    one matrix-free ``H @ block`` operation.

    There is one source-level subtlety: ``cheby_filterscal1`` initializes
    ``sigma`` once outside its block loop.  Consequently the first block
    starts the normalized recurrence at ``sigma1``, but a later block starts
    its second-degree update from the preceding block's final ``sigma`` even
    though its first-degree vector still uses ``sigma1``.  The default
    reproduces that trajectory exactly.  Setting
    ``reset_recurrence_per_block=True`` applies the conventional identical
    normalized polynomial independently to every block.
    """

    degree = _validate_degree(degree)
    matrix, was_vector = _as_column_matrix(vectors)
    _validate_operator(operator, matrix.shape[0])
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be positive")

    filtered = np.empty_like(
        matrix,
        dtype=_filtered_dtype(operator, matrix),
    )
    # Keep this scalar outside the loop to mirror its scope in chebff.f90z.
    carried_sigma: float | None = None
    for block in _uniform_blocks(matrix.shape[1], block_size, degree):
        result, final_sigma = chebyshev_filter(
            operator,
            matrix[:, block.start:block.stop],
            block.degree,
            lower_bound,
            upper_bound,
            reference_eigenvalue,
            initial_sigma=(
                None if reset_recurrence_per_block else carried_sigma
            ),
            return_final_sigma=True,
        )
        filtered[:, block.start:block.stop] = result
        carried_sigma = final_sigma
    return filtered[:, 0] if was_vector else filtered


def subspace_filter_blocks(
    vector_count: int,
    block_size: int,
    degree: int,
    degree_delta: int,
) -> tuple[FilterBlock, ...]:
    """Return PARSEC's block-rounded ``polym-dpm``/``polym+dpm`` split.

    Saved Ritz vectors are ordered from low to high energy.  Of the complete
    memory blocks, the first ``ceil(number_of_full_blocks/2)`` receive the
    cheaper degree ``polym-dpm``.  All remaining complete blocks and any
    final partial block receive ``polym+dpm``.  The higher Ritz/safety vectors
    therefore get the stronger filter.  This is a block-rounded split, not an
    exact half-column split; if fewer than one complete block exists, every
    vector belongs to the higher-degree remainder.
    """

    vector_count = int(vector_count)
    block_size = int(block_size)
    degree = _validate_degree(degree)
    degree_delta = int(degree_delta)
    if vector_count < 1:
        raise ValueError("vector_count must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if degree_delta < 0 or degree_delta >= degree:
        raise ValueError("degree_delta must be nonnegative and smaller than degree")

    full_block_count = vector_count // block_size
    low_block_count = (full_block_count + 1) // 2
    blocks: list[FilterBlock] = []
    for block_index in range(full_block_count):
        start = block_index * block_size
        blocks.append(
            FilterBlock(
                start=start,
                stop=start + block_size,
                degree=(
                    degree - degree_delta
                    if block_index < low_block_count
                    else degree + degree_delta
                ),
            )
        )
    remainder_start = full_block_count * block_size
    if remainder_start < vector_count:
        blocks.append(
            FilterBlock(
                start=remainder_start,
                stop=vector_count,
                degree=degree + degree_delta,
            )
        )
    return tuple(blocks)


def subspace_filter(
    operator: Any,
    vectors: np.ndarray,
    degree: int,
    degree_delta: int,
    lower_bound: float,
    upper_bound: float,
    *,
    block_size: int = 7,
    reset_recurrence_per_block: bool = False,
) -> np.ndarray:
    """Apply PARSEC's later-SCF filter with its exact degree split.

    The non-``BETA`` Fortran routine supplies ``lower_bound`` both as the
    unwanted interval edge and as the normalization reference.  Hence the
    ideal mapped reference is ``x0=-1`` and ``sigma1=-1``.  The filter is not
    an eigensolver by itself: its output still has to be orthonormalized and
    Rayleigh--Ritz rotated by :func:`subspace.run_subspace_filter`.
    """

    matrix, was_vector = _as_column_matrix(vectors)
    _validate_operator(operator, matrix.shape[0])
    blocks = subspace_filter_blocks(
        matrix.shape[1],
        block_size,
        degree,
        degree_delta,
    )
    filtered = np.empty_like(
        matrix,
        dtype=_filtered_dtype(operator, matrix),
    )
    carried_sigma: float | None = None
    for block in blocks:
        result, final_sigma = chebyshev_filter(
            operator,
            matrix[:, block.start:block.stop],
            block.degree,
            lower_bound,
            upper_bound,
            # In non-BETA subspace.f90z, lowb is both the unwanted
            # interval boundary and the recurrence normalization point.
            lower_bound,
            initial_sigma=(
                None if reset_recurrence_per_block else carried_sigma
            ),
            return_final_sigma=True,
        )
        filtered[:, block.start:block.stop] = result
        carried_sigma = final_sigma
    return filtered[:, 0] if was_vector else filtered
