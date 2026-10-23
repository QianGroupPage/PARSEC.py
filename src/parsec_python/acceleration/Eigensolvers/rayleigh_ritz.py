"""Device-resident Rayleigh--Ritz projection and rotation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

from ..backends.cupy import device_stage, require_cupy
from .handover import (
    HANDOVER_MODES,
    HandoverStep,
    handover_product,
    handover_slab_columns,
    uniform_slabs,
)
from .small_dense import symmetric_eigh
from .tall_product import transposed_product


_DEFAULT_GENERALIZED_RITZ_WORK_THRESHOLD = 100_000_000
# Rows mirrored at a time when a small host matrix is made symmetric.
_MIRROR_BLOCK = 64
# The columns in whose multiples the right-hand side of a tall product costs
# on an A100: the default of :func:`gram_multiple`.
_GRAM_MULTIPLE = 64
# The most bytes of the slab buffer of a sector on one device where neither a
# budget nor a limit is named (:func:`one_array_buffer_limit`), and the most
# that such a buffer took before, from which the steps of a shared basis
# still start (:func:`_streaming_bytes`).
_ONE_ARRAY_BUFFER_BYTES = 2 << 30
_FORMER_BUFFER_BYTES = 4 << 30


class GeneralizedRitzStabilityError(np.linalg.LinAlgError):
    """Raised when the non-orthogonal Ritz basis fails a safety audit."""


@dataclass(frozen=True)
class DeviceRayleighRitzResult:
    eigenvalues: Any
    wavefunctions: Any
    applied_wavefunctions: Any | None
    projected_hamiltonian: Any
    residual_norms: Any | None
    workspace: Any | None = None
    algorithm: str = "orthonormal_rayleigh_ritz"
    # What the step handed over where a hand-over was asked for
    # (:class:`handover.HandoverStep`); None for the rotation that was asked.
    handover: Any | None = None


def generalized_ritz_requested(row_count: int, column_count: int) -> bool:
    """Return whether the audited non-orthogonal Ritz route is selected.

    The route is profitable only for a large complete basis.  Explicitly
    selecting an orthogonalization algorithm remains a source-comparison
    request and therefore disables automatic generalized Ritz unless this
    policy is forced ``on`` separately.
    """

    policy = os.environ.get("PARSEC_CUPY_GENERALIZED_RITZ", "auto")
    policy = policy.strip().lower()
    if policy not in {"auto", "on", "off", "1", "0", "true", "false"}:
        raise ValueError(
            "PARSEC_CUPY_GENERALIZED_RITZ must be auto, on, or off"
        )
    if policy in {"on", "1", "true"}:
        return True
    if policy in {"off", "0", "false"}:
        return False
    explicit_orthogonalization = os.environ.get(
        "PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION"
    )
    if explicit_orthogonalization is not None and (
        explicit_orthogonalization.strip().lower() != "auto"
    ):
        return False
    raw_threshold = os.environ.get(
        "PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD",
        str(_DEFAULT_GENERALIZED_RITZ_WORK_THRESHOLD),
    ).strip()
    try:
        threshold = int(raw_threshold)
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD must be an integer"
        ) from error
    if threshold < 0:
        raise ValueError(
            "PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD cannot be negative"
        )
    work = int(row_count) * int(column_count) * int(column_count)
    return work >= threshold


def _generalized_condition_limit() -> float:
    raw = os.environ.get(
        "PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX", "1.0e8"
    ).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX must be numeric"
        ) from error
    if not np.isfinite(value) or value <= 1.0:
        raise ValueError(
            "PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX must exceed one"
        )
    return value


def _mirrored_lower(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric matrix that has the lower triangle of ``matrix``.

    The entries above the diagonal of ``matrix`` are ignored and the result
    is a new C-ordered array.  The lower triangle is mirrored a block of
    rows at a time: the transposed read of one block stays within the
    cache, which a transposed pass over a matrix of a few thousand states
    does not.
    """

    full = np.array(matrix, dtype=np.float64, order="C")
    count = int(full.shape[0])
    for start in range(0, count, _MIRROR_BLOCK):
        stop = min(count, start + _MIRROR_BLOCK)
        diagonal = full[start:stop, start:stop]
        diagonal[...] = np.tril(diagonal) + np.tril(diagonal, -1).T
        full[start:stop, stop:] = full[stop:, start:stop].T
    return full


def _orthogonality_error(
    coefficients: np.ndarray,
    overlap: np.ndarray,
) -> float:
    """Return the largest entry of ``|C.T G C - I|``.

    The identity is subtracted on the diagonal of the product itself and
    the largest magnitude is the larger of its two extreme entries, which
    needs no identity, difference or absolute-value array.
    """

    deviation = coefficients.T @ overlap @ coefficients
    np.fill_diagonal(deviation, deviation.diagonal() - 1.0)
    return float(
        np.maximum(
            np.max(deviation, initial=0.0),
            -np.min(deviation, initial=0.0),
        )
    )


def _condition_policy(on_device: bool) -> str:
    """Return ``svd`` or ``symmetric`` from ``PARSEC_CUPY_RITZ_CONDITION``.

    Left unset, the policy follows the place of the small solve.  The host
    solve keeps the SVD.  The device solve takes the symmetric spectrum: an
    SVD there downloads the overlap in every pass, which is most of what
    solving on the device saves.
    """

    policy = os.environ.get("PARSEC_CUPY_RITZ_CONDITION")
    if policy is None:
        return "symmetric" if on_device else "svd"
    if policy not in ("svd", "symmetric"):
        raise ValueError("Ritz condition policy must be svd or symmetric")
    return policy


def _overlap_condition(overlap: np.ndarray) -> float:
    """Optional symmetric spectrum screen, retaining SVD near the guard.

    For an SPD overlap its 2-norm condition number is lambda_max/lambda_min.
    Near the acceptance boundary (or for an indefinite/failed estimate), use
    the original SVD calculation. Cholesky and orthogonality audits remain.
    """
    policy=_condition_policy(on_device=False)
    if policy=="symmetric":
        try:
            values=np.linalg.eigvalsh(overlap)
        except np.linalg.LinAlgError:
            return float(np.linalg.cond(overlap))
        if np.all(np.isfinite(values)) and values[0]>0:
            estimate=float(values[-1]/values[0])
            if estimate < .01*_generalized_condition_limit():
                return estimate
    return float(np.linalg.cond(overlap))


def gram_multiple() -> int:
    """Columns of which the right-hand width of a tall Gram product is a multiple.

    ``PARSEC_CUPY_RITZ_GRAM_MULTIPLE`` sets it; 64 is the default.  For the
    product ``A.T @ B`` of two tall arrays cuBLAS takes on an A100 the time
    of a ``B`` whose columns are rounded up to the next multiple of 64.
    Timed there one product at a time, 1,032,628 rows by 3,704 columns of
    ``A``: 0.44 to 0.48 ms per column of the rounded width at each of 15
    widths of ``B`` from 124 to 640, which is 12.0, 13.9, 14.5 and 15.4
    Tflop/s at 136, 272, 464 and 616 columns against 16.8 to 17.3 at 192,
    256, 320, 384, 512 and 640.  Over 24 first runs (3,480 to 39,368
    electrons on 1 to 16 GPUs) the Gram stage ran at 10.5 to 15.6 Tflop/s
    as its products were cut, and at 15.6 to 17.4 once every width is
    counted as rounded up.  The columns of ``A`` cost what they are there:
    a step in them fits those runs worse at every size tried.

    That is the A100.  On a workstation GPU (RTX 5070, CUDA 12.9) the
    columns of ``B`` cost in the same steps of 64, but those of ``A`` in
    coarse steps as well, so that more slabs do not always gain: the Gram
    stage of one array, timed alone with slabs cut by 64 against 1, took
    21% less at 665 columns, 4% less at 1,842, 10% more at 897 (slabs of
    64 for 112 and 113) and the same at 442.  ``1`` is the setting for a
    device on which the stage does not gain.

    The Ritz step therefore cuts the slabs of its two Gram matrices, and
    the rounds of a shared basis, at whole multiples
    (:func:`_whole_multiple`, :func:`_gram_slab_width`,
    :func:`_streaming_shape`, :func:`distributed_state.block_rounds`,
    :func:`distributed_state.projection_slabs`).  A slab only narrows for
    it: the bytes of a budget stay the most that a slab takes, and what a
    fit rule counts for the workspace is what it counted.  The sums of the
    lower triangles are cut into other pieces, so the results differ by
    round-off from those of the former widths.  ``1`` selects those, bit
    for bit, on every route.  The overlap of a shared basis has been cut
    by a width of its own since, a whole multiple as well
    (:func:`distributed_state.overlap_slab_width`): its former widths take
    ``PARSEC_CUPY_RITZ_GRAM_SLABS=8`` beside the ``1``.
    """

    raw = os.environ.get("PARSEC_CUPY_RITZ_GRAM_MULTIPLE", str(_GRAM_MULTIPLE)).strip()
    try:
        multiple = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_RITZ_GRAM_MULTIPLE must be an integer"
        ) from error
    if multiple < 1:
        raise ValueError("PARSEC_CUPY_RITZ_GRAM_MULTIPLE must be positive")
    return multiple


def _whole_multiple(width: int, multiple: int | None = None) -> int:
    """``width`` columns cut down to a whole multiple of :func:`gram_multiple`.

    A width below one multiple stays what it is: nothing narrower comes
    closer to a multiple.  ``multiple`` is that of the caller where it has
    read one.
    """

    width = int(width)
    if multiple is None:
        multiple = gram_multiple()
    return width - width % int(multiple) if width >= multiple else width


def _gram_slab_width(columns: int) -> int:
    """Columns per slab of a lower-triangle Gram product.

    ``PARSEC_CUPY_RITZ_GRAM_SLABS`` sets the largest number of slabs, 8 by
    default.  The width is the column count divided by it, rounded up, so
    fewer slabs can result: 17 columns in at most 8 slabs are 3 columns wide,
    which makes 6 slabs.  Multiplying each slab by the columns from its own
    first one onwards costs ``(slabs + 1) / (2 * slabs)`` of the full
    product: 56% with 8 slabs.  More slabs gain little and make the tall
    products narrow; one slab is the full product.

    Slabs of 64 columns and more are cut down to a whole multiple of 64
    (:func:`gram_multiple`), which makes more of them: 1,842 columns in
    slabs of 231 become nine of 192 and one of 114.  The narrower last slab
    is multiplied by its own columns only, the smallest product of all.

    That is the cut of a sector on one device, of both Gram matrices where
    it keeps two arrays, and of the projection of a shared basis with three
    tall blocks.  The overlap of a shared basis is cut by a width of its
    own, 128 columns, unless the count is named here: a named count is the
    cut of every route (:func:`distributed_state.overlap_slab_width`).
    """

    raw = os.environ.get("PARSEC_CUPY_RITZ_GRAM_SLABS", "8").strip()
    try:
        slabs = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_RITZ_GRAM_SLABS must be an integer"
        ) from error
    if slabs < 1:
        raise ValueError("PARSEC_CUPY_RITZ_GRAM_SLABS must be positive")
    width = max(1, -(-int(columns) // slabs))
    return width if width >= int(columns) else _whole_multiple(width)


def _lower_triangle_product(left: Any, right: Any, width: int | None = None):
    """Return the lower triangle of ``left.T @ right`` from column slabs.

    The small solve, :func:`solve_whitened_ritz` on the host and
    :func:`solve_whitened_ritz_on_device` on the device, reads the lower
    triangles of ``X.T X`` and ``X.T (H X)`` only, so each slab of columns of
    ``right`` is multiplied by the columns of ``left`` from its own first one
    onwards.  Entries above the diagonal slabs stay zero.

    A slab of an even number of columns below 64 is multiplied in two
    pieces (:func:`tall_product.transposed_product`), as is every other
    product of two tall arrays in this module.

    ``width`` is the columns of a slab where the caller cuts by a rule of
    its own (:func:`distributed_state.overlap_slab_width`); without one the
    slabs are those of :func:`_gram_slab_width`.
    """

    cp, _ = require_cupy()
    columns = int(left.shape[1])
    width = _gram_slab_width(columns) if width is None else max(1, int(width))
    if width >= columns:
        return transposed_product(cp, left, right)
    product = cp.zeros((columns, columns), dtype=cp.float64, order="F")
    for start in range(0, columns, width):
        stop = min(columns, start + width)
        product[start:, start:stop] = transposed_product(
            cp, left[:, start:], right[:, start:stop]
        )
    return product


def _overlap_policy() -> str:
    """How ``PARSEC_CUPY_RITZ_SYRK`` has the overlap formed: ``slabs``, ``syrk`` or ``full``.

    ``auto``, the default, and ``slabs`` are the slabs of
    :func:`_lower_triangle_product`, ``on`` is FP64 DSYRK and ``off`` one
    full GEMM.
    """

    policy = os.environ.get(
        "PARSEC_CUPY_RITZ_SYRK", "auto"
    ).strip().lower()
    if policy not in {"auto", "on", "off", "slabs", "1", "0", "true", "false"}:
        raise ValueError(
            "PARSEC_CUPY_RITZ_SYRK must be auto, on, off, or slabs"
        )
    if policy in {"off", "0", "false"}:
        return "full"
    return "slabs" if policy in {"auto", "slabs"} else "syrk"


def _symmetric_overlap(matrix: Any, width: int | None = None):
    """Form the lower triangle of ``X.T X``.

    ``PARSEC_CUPY_RITZ_SYRK`` selects how (:func:`_overlap_policy`): from
    GEMMs on trailing columns (see :func:`_lower_triangle_product`), with
    FP64 DSYRK or from one full GEMM.  ``width`` is the columns of a slab
    where the caller has a rule for them; only the slabs read it.
    """

    cp, _ = require_cupy()
    policy = _overlap_policy()
    if policy == "full":
        return transposed_product(cp, matrix, matrix)
    if policy == "slabs":
        # DSYRK does half the operations of the full product but stays below
        # the GEMM rate (8 to 9 against 13 to 17 Tflop/s on an A100); the
        # slabs do little more than half at that rate.
        if width is None:
            return _lower_triangle_product(matrix, matrix)
        return _lower_triangle_product(matrix, matrix, width)
    # cuBLAS is column-major.  ``matrix`` is made Fortran-contiguous by the
    # caller, so op(A)=A.T with n=states and k=grid points computes exactly
    # the lower triangle consumed by the generalized Ritz solve.
    from cupy.cuda import cublas

    rows, columns = map(int, matrix.shape)
    output = cp.empty((columns, columns), dtype=cp.float64, order="F")
    alpha = np.asarray(1.0, dtype=np.float64)
    beta = np.asarray(0.0, dtype=np.float64)
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
    cublas.dsyrk(
        handle,
        cublas.CUBLAS_FILL_MODE_LOWER,
        cublas.CUBLAS_OP_T,
        columns,
        rows,
        alpha.ctypes.data,
        matrix.data.ptr,
        rows,
        beta.ctypes.data,
        output.data.ptr,
        columns,
    )
    return output


def streaming_ritz_requested(operator: Any | None = None) -> bool:
    """Whether Rayleigh--Ritz may trade a little time for bounded workspace.

    With ``PARSEC_CUPY_STREAMING_RITZ=1`` a routine that owns its basis keeps
    one ``N x m`` array instead of two or three: ``H X`` is formed and
    projected a slab of columns at a time, and the rotation overwrites the
    basis one tile of rows at a time.  ``0`` keeps the two arrays.  ``auto``,
    the default, leaves the choice to the owner of ``operator``.  The
    symmetry eigensolver marks the operator of every sector
    ``low_memory_ritz`` (see ``PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION``
    there for the exception), so a sector whose basis lies on one device
    keeps one array; an operator that nobody marked keeps two.
    """

    value = os.environ.get("PARSEC_CUPY_STREAMING_RITZ", "auto").strip().lower()
    if value not in {"0", "1", "on", "off", "true", "false", "auto"}:
        raise ValueError("PARSEC_CUPY_STREAMING_RITZ must be on, off, or auto")
    if value == "auto":
        return bool(getattr(operator, "low_memory_ritz", False))
    return value in {"1", "on", "true"}


def _column_copy_requested() -> bool:
    """Whether the in-place rotation writes its tiles back along columns.

    The default; ``PARSEC_CUPY_ROTATE_COLUMN_COPY=0`` writes them back along
    rows.  The values written are the same either way; only the order in
    which the copy visits them changes.
    """

    value = os.environ.get(
        "PARSEC_CUPY_ROTATE_COLUMN_COPY", "1"
    ).strip().lower()
    if value not in {"0", "1", "on", "off", "true", "false"}:
        raise ValueError("PARSEC_CUPY_ROTATE_COLUMN_COPY must be on or off")
    return value in {"1", "on", "true"}


def _streaming_bytes(basis_bytes: int, most: int = _FORMER_BUFFER_BYTES) -> int:
    """Bytes of one projection slab or rotation tile.

    ``PARSEC_CUPY_STREAMING_RITZ_BYTES`` sets it.  Otherwise it is an eighth
    of the basis, at least 1 GiB and at most ``most``: narrow slabs make
    the tall products inefficient and pass over the basis more often.
    ``most`` is 4 GiB where the caller names nothing, as the steps of a
    shared basis start from it (:func:`distributed_state.slab_columns`).
    A sector on one device names its own limit
    (:func:`one_array_buffer_limit`), and a caller that has made sure of
    the memory a higher one (see :func:`distributed_state.wider_slabs_fit`).
    """

    raw = os.environ.get("PARSEC_CUPY_STREAMING_RITZ_BYTES", "").strip()
    if not raw:
        return min(max(int(basis_bytes) // 8, 1 << 30), int(most))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_STREAMING_RITZ_BYTES must be an integer"
        ) from error
    if value < 1:
        raise ValueError("PARSEC_CUPY_STREAMING_RITZ_BYTES must be positive")
    return value


def one_array_buffer_limit(rows: int) -> int:
    """The most bytes of the slab buffer of a sector on one device where no budget is set.

    ``PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES`` names it.  Unset, it is 2
    GiB, or 64 columns of all ``rows`` where 2 GiB hold fewer, up to the 4
    GiB that it was.  The buffer is an eighth of the basis within it and
    at least 1 GiB (:func:`_streaming_bytes`), so a basis of up to 16 GiB
    keeps the buffer it had, one of 16 to 32 GiB took that eighth and
    takes 2 GiB, and a larger one took 4.  The limit acts where
    ``PARSEC_CUPY_STREAMING_RITZ_BYTES`` is not set and on nothing else:
    a budget that is set is the buffer as before, and no step of a shared
    basis reads the limit.

    First runs on A100 nodes with the budget set to 2 GiB against the
    default of 4 in one allocation: 14,680 electrons on 2 GPUs peaked at
    78,109 instead of 80,157 MiB on the fullest device in 231.4 against
    231.6 s, and on 4 GPUs at 41,049 instead of 43,141 MiB in 123.9
    against 124.5 s; 19,392 electrons on 4 GPUs at 58,155 instead of
    60,251 MiB in 180.7 against 180.5 s.  Their slabs are 64 columns wide
    where they were 192 and 128, and the total energies moved by 5e-9 to
    5e-8 Ry: the projection is summed from other slabs, and the rows are
    rotated in tiles of half the height.  With 3 GiB the slabs of 19,392
    electrons stay 128 columns and only the tile is lower: 59,227 MiB,
    180.4 s and the total energy to the last bit.

    The slabs of those runs are one whole multiple
    (:func:`gram_multiple`).  A narrower one pays for 64 columns in every
    product, and none has been timed alone.  A sector of more than
    4,194,304 rows, of which 2 GiB hold fewer than 64 columns, therefore
    keeps slabs of 64 columns, 2 to 4 GiB of buffer, and from 8,388,608
    rows the 4 GiB and the narrower slab that it had.  That guard is
    derived from the slabs of 64 columns and not measured.  It acts only
    for such a sector whose basis lies on one device and holds more than
    16 GiB; on A100 80 GB nodes no case of the measured series has one,
    since its only sectors of that many rows, those of 39,368 electrons
    (5,660,200 rows, 207.8 GiB), are shared by four devices.

    ``4294967296`` selects the former buffer, at most 4 GiB, for a basis
    of any size, bit for bit.  ``PARSEC_CUPY_STREAMING_RITZ_BYTES`` at
    that value does not: a basis below 32 GiB took an eighth of itself
    and takes 4 GiB with it.
    """

    name = "PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES"
    raw = os.environ.get(name)
    if raw is None:
        whole = 8 * int(rows) * _GRAM_MULTIPLE
        return max(_ONE_ARRAY_BUFFER_BYTES, min(whole, _FORMER_BUFFER_BYTES))
    try:
        value = int(raw.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _streaming_views(rows: int, columns: int, basis_bytes: int) -> tuple[int, int, int]:
    """Columns of the slab of the budget and of the slab that is cut from it, and rows of the tile.

    For a ``rows x columns`` basis of these bytes on one device.  The slab
    and the tile are as many columns and rows as the bytes of the budget
    hold (:func:`_streaming_cut`).  A slab of 64 columns and more that is
    not the whole basis is then cut down to a whole multiple of 64
    (:func:`gram_multiple`), since every slab is the right-hand side of a
    tall product: the 103 columns of 2 GiB at 2,604,846 rows become 64, as
    the 206 of 4 GiB became 192.  The tile is the left-hand side of its
    product and keeps its rows, so the rotation is cut as it was.
    """

    budget_width, tile = _streaming_cut(rows, columns, basis_bytes)
    width = budget_width if budget_width >= columns else _whole_multiple(budget_width)
    return budget_width, width, tile


def _streaming_shape(matrix: Any) -> tuple[int, int]:
    """Columns of one projection slab and rows of one rotation tile (:func:`_streaming_views`)."""

    rows, columns = map(int, matrix.shape)
    _budget_width, width, tile = _streaming_views(rows, columns, matrix.nbytes)
    return width, tile


def _streaming_cut(rows: int, columns: int, basis_bytes: int) -> tuple[int, int]:
    """Slab columns and tile rows that the budget of a sector on one device holds.

    For a ``rows x columns`` basis of these bytes, before
    :func:`_streaming_views` cuts the slab down to a whole multiple.  The
    budget is that of :func:`_streaming_bytes` within the limit of one
    array (:func:`one_array_buffer_limit`).
    """

    budget = _streaming_bytes(basis_bytes, one_array_buffer_limit(rows))
    return (
        max(1, min(columns, budget // (8 * rows))),
        max(1, min(rows, budget // (8 * columns))),
    )


def streaming_workspace_bytes(rows: int, columns: int) -> int:
    """Bytes that a rule counts for the buffer of :func:`_streaming_workspace`.

    For a float64 basis of ``rows x columns``, before it exists: the rule
    that decides where the states of the symmetry sectors are kept counts
    the buffer beside them
    (:meth:`symmetry.CuPySymmetrySCFEigensolver._state_fit_bytes`).

    The slab is counted with the columns of its budget, as the fit rules of
    a shared basis count theirs (:func:`gram_multiple`): the count does not
    move with that switch, and the same sectors stay on their devices at
    every multiple.  The buffer itself is the count with a multiple of 1
    and never more: a slab cut down to a whole multiple leaves the tile as
    the larger view, which is within ``8 * columns`` bytes of the budget.
    """

    rows, columns = int(rows), int(columns)
    budget_width, _width, tile = _streaming_views(rows, columns, 8 * rows * columns)
    return 8 * max(rows * budget_width, tile * columns)


def _streaming_workspace(matrix: Any):
    """Return the projection slab and the rotation tile as views of one buffer.

    The slab receives ``H`` times some columns (all rows), the tile the
    rotated basis for some rows (all columns).  Both get the bytes of the
    budget (:func:`_streaming_cut`), but whole columns and whole rows round them
    differently, so the two differ by a few megabytes, and by more where
    the slab was cut down to a multiple of 64 columns
    (:func:`_streaming_views`).  A memory-pool block
    released by the smaller cannot serve the larger: two separate buffers
    would both stay in the pool until it is emptied after the SCF step.  The
    slab and the tile are never in use together, so one buffer of the larger
    size backs both.

    The buffer stays allocated between the two, during the small solve.  A
    solve on the device (:func:`dense_solve_on_device`) therefore takes its
    work arrays, a few ``m x m`` ones, in addition to the buffer and not out
    of a released slab.
    """

    cp, _ = require_cupy()
    rows, columns = map(int, matrix.shape)
    width, tile = _streaming_shape(matrix)
    backing = cp.empty(max(rows * width, tile * columns), dtype=cp.float64)
    slab = cp.ndarray(
        (rows, width), dtype=cp.float64, memptr=backing.data, order="F"
    )
    rotated = cp.ndarray(
        (tile, columns), dtype=cp.float64, memptr=backing.data, order="F"
    )
    return slab, rotated


def _streamed_projection(operator: Any, matrix: Any, slab: Any | None = None):
    """Return the lower triangle of ``X.T (H X)`` without holding all of ``H X``.

    ``H X`` is formed a slab of columns at a time.  The projection is
    symmetric and the small solve (:func:`solve_whitened_ritz` on the host,
    :func:`solve_whitened_ritz_on_device` on the device) reads its lower
    triangle only, so each slab is projected on the columns from its own
    first one onwards, which halves the work of the tall products.  Entries
    above the diagonal slabs stay zero.

    ``slab`` is the slab of :func:`_streaming_workspace`; one is allocated
    here when the caller has none to share.
    """

    cp, _ = require_cupy()
    rows, columns = map(int, matrix.shape)
    if slab is None:
        width, _tile = _streaming_shape(matrix)
        slab = cp.empty((rows, width), dtype=cp.float64, order="F")
    width = int(slab.shape[1])
    projection = cp.zeros((columns, columns), dtype=cp.float64, order="F")
    apply_into = getattr(operator, "apply_into", None)
    for start in range(0, columns, width):
        stop = min(columns, start + width)
        block = matrix[:, start:stop]
        with device_stage(operator, "subspace_ritz_hamiltonian_seconds"):
            applied = slab[:, : stop - start]
            if callable(apply_into):
                applied = apply_into(block, applied)
            else:
                applied[...] = operator @ block
        with device_stage(operator, "subspace_ritz_projection_seconds"):
            projection[start:, start:stop] = transposed_product(
                cp, matrix[:, start:], applied
            )
    return projection


def _rotate_in_place(
    matrix: Any, coefficients: Any, rotated: Any | None = None
) -> None:
    """Overwrite ``matrix`` with ``matrix @ coefficients`` by row tiles.

    A tile of rows of the column-major basis is a BLAS matrix whose leading
    dimension is the full column length, so cuBLAS reads it where it lies;
    only the product takes a tile of workspace before it is written back.

    ``rotated`` is the tile of :func:`_streaming_workspace`; one is allocated
    here when the caller has none to share.
    """

    cp, _ = require_cupy()
    rows, columns = map(int, matrix.shape)
    if tuple(coefficients.shape) != (columns, columns):
        raise ValueError("an in-place rotation needs square coefficients")
    if rotated is None:
        _width, tile = _streaming_shape(matrix)
        rotated = cp.empty((tile, columns), dtype=cp.float64, order="F")
    if (
        rotated.dtype != cp.dtype(cp.float64)
        or rotated.ndim != 2
        or int(rotated.shape[1]) != columns
        or not rotated.flags.f_contiguous
    ):
        raise ValueError(
            "a rotation tile must be Fortran-ordered float64 rows of all columns"
        )
    tile = int(rotated.shape[0])
    factors = cp.asfortranarray(coefficients, dtype=cp.float64)
    direct = bool(matrix.flags.f_contiguous and matrix.dtype == cp.dtype(cp.float64))
    if direct:
        try:
            from cupy.cuda import cublas

            one = np.asarray(1.0, dtype=np.float64)
            zero = np.asarray(0.0, dtype=np.float64)
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
        except Exception:
            direct = False
    column_copy = _column_copy_requested()
    for start in range(0, rows, tile):
        stop = min(rows, start + tile)
        product = rotated[: stop - start]
        if direct:
            cublas.dgemm(
                handle,
                cublas.CUBLAS_OP_N,
                cublas.CUBLAS_OP_N,
                stop - start,
                columns,
                columns,
                one.ctypes.data,
                matrix.data.ptr + 8 * start,
                rows,
                factors.data.ptr,
                columns,
                zero.ctypes.data,
                rotated.data.ptr,
                tile,
            )
        else:
            product[...] = cp.ascontiguousarray(matrix[start:stop, :]) @ factors
        # Each tile is read completely before any of its rows is rewritten.
        if column_copy:
            # An elementwise copy advances the last axis fastest.  Through
            # the transposed views that axis runs down a column of both
            # arrays, so neighbouring threads touch neighbouring addresses
            # instead of addresses one whole column apart.
            matrix[start:stop, :].T[...] = product.T
        else:
            matrix[start:stop, :] = product


def _triangular_in_place(
    matrix: Any,
    factor: Any,
    buffer: Any,
    slabs,
    product: str | None = None,
) -> str:
    """Overwrite ``matrix`` with ``matrix @ factor`` where ``factor[b:, a:b]`` is zero for every slab ``(a, b)``.

    ``slabs`` are consecutive output slabs that cover the columns; the
    factor of a hand-over is upper triangular in the order of the columns,
    or upper triangular but for dense diagonal blocks that lie inside one
    slab each (:func:`handover.handover_slabs`).  Output slab ``(a, b)`` is
    then ``matrix[:, :b] @ factor[:b, a:b]``: every product runs over the
    leading ``b`` columns only, which is about half of the flops of a
    rotation by a full matrix.

    ``buffer`` is the workspace of the step, a contiguous float64 array that
    shares no memory with ``matrix``.  :func:`handover.handover_product`
    says how the product is cut.  ``columns``: the slabs from the last to
    the first, each one DGEMM over all rows into the buffer and copied into
    its columns; a slab reads only columns before its end, which no slab
    after it in this order has written.  ``tiles``, and wherever the buffer
    cannot hold the widest slab in all rows: per tile of rows of all
    columns, as many as the buffer holds, one DGEMM per slab into the tile,
    which goes back as in :func:`_rotate_in_place`.  Returns the product
    that ran.
    """

    cp, _ = require_cupy()
    rows, columns = map(int, matrix.shape)
    if tuple(factor.shape) != (columns, columns):
        raise ValueError("a triangular hand-over needs a square factor")
    slabs = tuple((int(first), int(last)) for first, last in slabs)
    if (
        not slabs
        or slabs[0][0] != 0
        or slabs[-1][1] != columns
        or any(last <= first for first, last in slabs)
        or any(earlier[1] != later[0] for earlier, later in zip(slabs[:-1], slabs[1:]))
    ):
        raise ValueError("the output slabs must cover the columns in order")
    if buffer.dtype != cp.dtype(cp.float64) or not (
        buffer.flags.c_contiguous or buffer.flags.f_contiguous
    ):
        raise ValueError("the buffer of a triangular product must be contiguous float64")
    elements = int(buffer.size)
    widest = max(last - first for first, last in slabs)
    kind = handover_product() if product is None else str(product)
    if kind not in ("columns", "tiles"):
        raise ValueError("a triangular product is cut by columns or by tiles")
    if kind == "columns" and rows * widest > elements:
        kind = "tiles"
    if kind == "tiles" and elements < columns:
        raise ValueError("the buffer of a triangular product holds no row of all columns")
    if not rows:
        return kind
    factors = cp.asfortranarray(factor, dtype=cp.float64)
    direct = bool(matrix.flags.f_contiguous and matrix.dtype == cp.dtype(cp.float64))
    if direct:
        try:
            from cupy.cuda import cublas

            one = np.asarray(1.0, dtype=np.float64)
            zero = np.asarray(0.0, dtype=np.float64)
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
        except Exception:
            direct = False
    stream = cp.cuda.get_current_stream()
    if kind == "columns":
        for first, last in reversed(slabs):
            width = last - first
            out = cp.ndarray(
                (rows, width), dtype=cp.float64, memptr=buffer.data, order="F"
            )
            if direct:
                cublas.dgemm(
                    handle,
                    cublas.CUBLAS_OP_N,
                    cublas.CUBLAS_OP_N,
                    rows,
                    width,
                    last,
                    one.ctypes.data,
                    matrix.data.ptr,
                    rows,
                    factors.data.ptr + 8 * first * columns,
                    columns,
                    zero.ctypes.data,
                    out.data.ptr,
                    rows,
                )
                # The columns of a slab are one piece of a column-major array.
                (matrix.data + 8 * first * rows).copy_from_device_async(
                    out.data, 8 * rows * width, stream
                )
            else:
                out[...] = matrix[:, :last] @ factors[:last, first:last]
                matrix[:, first:last] = out
        return kind
    tile = min(rows, elements // columns)
    rotated = cp.ndarray((tile, columns), dtype=cp.float64, memptr=buffer.data, order="F")
    column_copy = _column_copy_requested()
    for start in range(0, rows, tile):
        stop = min(rows, start + tile)
        product_rows = rotated[: stop - start]
        for first, last in slabs:
            if direct:
                cublas.dgemm(
                    handle,
                    cublas.CUBLAS_OP_N,
                    cublas.CUBLAS_OP_N,
                    stop - start,
                    last - first,
                    last,
                    one.ctypes.data,
                    matrix.data.ptr + 8 * start,
                    rows,
                    factors.data.ptr + 8 * first * columns,
                    columns,
                    zero.ctypes.data,
                    rotated.data.ptr + 8 * first * tile,
                    tile,
                )
            else:
                product_rows[:, first:last] = (
                    cp.ascontiguousarray(matrix[start:stop, :last]) @ factors[:last, first:last]
                )
        # Each tile is read completely before any of its rows is rewritten.
        if column_copy:
            matrix[start:stop, :].T[...] = product_rows.T
        else:
            matrix[start:stop, :] = product_rows
    return kind


def solve_whitened_ritz(
    raw_overlap: np.ndarray,
    raw_projection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the small generalized Ritz problem on the host with all audits.

    Only the lower triangles of ``X.T X`` and ``X.T H X`` are read.  Returns
    the Ritz values, the coefficient matrix ``C`` with ``C.T (X.T X) C = I``
    and the whitened projected Hamiltonian.  The eigensolver reads the lower
    triangle of the latter; its upper triangle is the round-off counterpart
    left by the two triangular solves, not a mirror image.  Raises
    :class:`GeneralizedRitzStabilityError` when the overlap is too
    ill-conditioned, its Cholesky factorization fails, or the coefficients
    miss the orthogonality audit.  Every route that forms the two Gram
    matrices (one device, row blocks on several devices) shares this code.
    """

    overlap = _mirrored_lower(raw_overlap)
    projected = _mirrored_lower(raw_projection)
    try:
        condition = _overlap_condition(overlap)
    except np.linalg.LinAlgError as error:
        raise GeneralizedRitzStabilityError(
            "filtered overlap condition estimate failed"
        ) from error
    if not np.isfinite(condition) or condition > _generalized_condition_limit():
        raise GeneralizedRitzStabilityError(
            f"filtered overlap condition number {condition:.3e} is unsafe"
        )
    try:
        cholesky = np.linalg.cholesky(overlap)
        left_projected = solve_triangular(
            cholesky,
            projected,
            lower=True,
            check_finite=False,
        )
        whitened = solve_triangular(
            cholesky,
            left_projected.T,
            lower=True,
            check_finite=False,
        ).T
        # LAPACK and cuSOLVER read the lower triangle of the whitened matrix
        # only, so it is not made symmetric first.
        dense_backend = os.environ.get("PARSEC_CUPY_RITZ_EIGH_BACKEND", "host").strip().lower()
        if dense_backend == "cupy":
            import cupyx

            cp, _ = require_cupy()
            # Retain the same audited host Cholesky whitening and coefficient
            # checks. Only the symmetric eigendecomposition uses cuSOLVER;
            # transfers are included in complete-SCF performance comparisons.
            with cupyx.errstate(linalg="raise"):
                device_values, device_vectors = cp.linalg.eigh(cp.asarray(whitened))
            host_eigenvalues = cp.asnumpy(device_values)
            whitened_vectors = cp.asnumpy(device_vectors)
        elif dense_backend == "host":
            host_eigenvalues, whitened_vectors = np.linalg.eigh(whitened)
        else:
            raise ValueError("PARSEC_CUPY_RITZ_EIGH_BACKEND must be host or cupy")
        coefficients = solve_triangular(
            cholesky.T,
            whitened_vectors,
            lower=False,
            check_finite=False,
        )
    except np.linalg.LinAlgError as error:
        raise GeneralizedRitzStabilityError(
            "filtered overlap Cholesky/Ritz solve failed"
        ) from error

    audit_error = _orthogonality_error(coefficients, overlap)
    if not np.isfinite(audit_error) or audit_error > 5.0e-10:
        raise GeneralizedRitzStabilityError(
            f"generalized Ritz orthogonality audit failed ({audit_error:.3e})"
        )

    return host_eigenvalues, coefficients, whitened


def dense_solve_on_device() -> bool:
    """Whether the small generalized Ritz problem is solved on the device.

    With ``PARSEC_CUPY_RITZ_DENSE_BACKEND=device``, the default, the two Gram
    matrices stay on the device that formed them and
    :func:`solve_whitened_ritz_on_device` replaces their download, the host
    solve and the upload of the coefficients.  ``host`` keeps
    :func:`solve_whitened_ritz`, whose eigensolver
    ``PARSEC_CUPY_RITZ_EIGH_BACKEND`` selects.

    The condition number of the overlap follows the same choice unless
    ``PARSEC_CUPY_RITZ_CONDITION`` names a policy (see
    :func:`_condition_policy`): with ``svd`` the device solve downloads the
    overlap every pass for the host SVD.
    """

    value = os.environ.get(
        "PARSEC_CUPY_RITZ_DENSE_BACKEND", "device"
    ).strip().lower()
    if value not in {"host", "device"}:
        raise ValueError("PARSEC_CUPY_RITZ_DENSE_BACKEND must be host or device")
    return value == "device"


def _device_overlap_condition(overlap: Any) -> float:
    """:func:`_overlap_condition` of a symmetric overlap held on the device.

    The symmetric spectrum, the policy of this solve unless
    ``PARSEC_CUPY_RITZ_CONDITION`` names another, comes from cuSOLVER, which
    is given a finite matrix only.  The SVD is left to the host: the ``svd``
    policy, an estimate near the guard and an indefinite, non-finite or
    failed estimate download the overlap and decide exactly as the host solve
    does.
    """

    import cupyx

    cp, _ = require_cupy()
    policy = _condition_policy(on_device=True)
    if policy == "symmetric" and bool(cp.isfinite(overlap).all().item()):
        try:
            with cupyx.errstate(linalg="raise"):
                values = cp.asnumpy(cp.linalg.eigvalsh(overlap))
        except np.linalg.LinAlgError:
            return float(np.linalg.cond(cp.asnumpy(overlap)))
        if np.all(np.isfinite(values)) and values[0] > 0:
            estimate = float(values[-1] / values[0])
            if estimate < 0.01 * _generalized_condition_limit():
                return estimate
    return float(np.linalg.cond(cp.asnumpy(overlap)))


def _judge_condition(estimate, *arguments) -> float:
    """Raise unless ``estimate(*arguments)`` is a safe condition number of the overlap.

    The first two audits of the device solve: an estimate that fails, then
    a number that is not finite or beyond the limit.  Returns the number,
    which the hand-over judges once more (:func:`solve_whitened_handover_on_device`).
    """

    try:
        condition = estimate(*arguments)
    except np.linalg.LinAlgError as error:
        raise GeneralizedRitzStabilityError(
            "filtered overlap condition estimate failed"
        ) from error
    finally:
        # An estimate that another thread made hands its error over through
        # ``estimate``, and the traceback of that error holds this frame: a
        # frame that still named ``estimate`` would close a cycle, which
        # keeps the arrays of both threads until the collector runs.
        del estimate
    if not np.isfinite(condition) or condition > _generalized_condition_limit():
        raise GeneralizedRitzStabilityError(
            f"filtered overlap condition number {condition:.3e} is unsafe"
        )
    return float(condition)


def solve_whitened_ritz_on_device(
    raw_overlap: Any,
    raw_projection: Any,
    condition: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Solve the small generalized Ritz problem on the current device.

    The device counterpart of :func:`solve_whitened_ritz` for Gram matrices
    that are device arrays: symmetrization, Cholesky factorization,
    triangular solves, eigendecomposition and orthogonality audit run in
    cuSOLVER and cuBLAS, and so does the condition number as far as
    :func:`_device_overlap_condition` allows.  Thresholds and
    :class:`GeneralizedRitzStabilityError` conditions are those of the host
    solve.  Returns the Ritz values, the Fortran-ordered coefficients and
    the whitened matrix as device arrays.  They agree with the host solve to
    round-off, not bit for bit.

    The condition number is only compared with its limit, and estimating it
    takes about as long as everything else here.  A caller with an idle
    device names a ``condition``: it is called with the symmetric overlap,
    a contiguous array of the current device that is complete by then, and
    starts the estimate elsewhere; what it returns is called once, waits
    for the number and returns it, or raises what the estimate raised.
    The factorization and the eigensolve run meanwhile.  The number is
    judged before anything of theirs is, whatever they raised, so the
    audits, their order and their errors are those of the solve that
    estimates first, and the coefficients are the same bit for bit.  What
    is new is work that an unsafe overlap would have spared: the
    factorization of an overlap that is not finite, which cuSOLVER
    completes, while its eigensolver is still given finite matrices only.
    """

    import cupyx
    from cupyx.scipy.linalg import solve_triangular as device_solve_triangular

    cp, _ = require_cupy()
    lower = cp.tril(raw_overlap)
    overlap = lower + cp.tril(lower, -1).T
    lower = cp.tril(raw_projection)
    projected = lower + cp.tril(lower, -1).T
    del lower
    # What waits for a number that is estimated elsewhere.  It is taken out
    # before it is called: an error of the estimate holds this frame in its
    # traceback, as it does that of :func:`_judge_condition`.
    waiting = []
    if condition is None:
        _judge_condition(_device_overlap_condition, overlap)
    else:
        # Another device reads the overlap from here until its number is judged.
        cp.cuda.get_current_stream().synchronize()
        waiting.append(condition(overlap))
    try:
        # cuSOLVER reports a failed factorization or eigensolve in a status
        # word that CuPy inspects only when asked to raise.
        with cupyx.errstate(linalg="raise"):
            # Column-major factors are passed to cuBLAS where they lie.
            cholesky = cp.asfortranarray(cp.linalg.cholesky(overlap))
            left_projected = device_solve_triangular(
                cholesky,
                projected,
                lower=True,
            )
            whitened = device_solve_triangular(
                cholesky,
                left_projected.T,
                lower=True,
            ).T
            # The triangular solves report nothing.  A non-finite whitened
            # matrix is rejected here, before cuSOLVER, as the host
            # eigensolver or the audit rejects it.
            if not bool(cp.isfinite(whitened).all().item()):
                raise np.linalg.LinAlgError("whitened projection is not finite")
            # cuSOLVER reads the lower triangle of the whitened matrix only.
            eigenvalues, whitened_vectors = cp.linalg.eigh(whitened)
            coefficients = device_solve_triangular(
                cp.asfortranarray(cholesky.T),
                whitened_vectors,
                lower=False,
            )
    except np.linalg.LinAlgError as error:
        if waiting:
            _judge_condition(waiting.pop())
        raise GeneralizedRitzStabilityError(
            "filtered overlap Cholesky/Ritz solve failed"
        ) from error
    except BaseException:
        # The estimate has ended before the overlap is released, and an
        # unsafe number is still the first thing to report.
        if waiting:
            _judge_condition(waiting.pop())
        raise
    if waiting:
        _judge_condition(waiting.pop())

    coefficient_overlap = coefficients.T @ overlap @ coefficients
    audit_error = float(
        cp.abs(
            coefficient_overlap
            - cp.eye(coefficient_overlap.shape[0], dtype=cp.float64)
        ).max().item()
    )
    if not np.isfinite(audit_error) or audit_error > 5.0e-10:
        raise GeneralizedRitzStabilityError(
            f"generalized Ritz orthogonality audit failed ({audit_error:.3e})"
        )

    return eigenvalues, coefficients, whitened


@dataclass(frozen=True)
class WhitenedHandover:
    """The small solve of a hand-over (:func:`solve_whitened_handover_on_device`).

    ``mode`` is what ran: ``triangular``, ``factor`` or ``values`` (which
    becomes ``factor`` above its condition limit, ``note`` says so).
    ``eigenvalues`` are ascending; ``factor`` is ``U = L^-T``, upper
    triangular and Fortran-ordered in the order of the given pair (None for
    ``values``); ``eigenvectors`` the ``Z`` of the whitened matrix
    (``triangular`` only); ``whitened`` that matrix; ``condition`` the
    estimate of the overlap that the solve judged.
    """

    mode: str
    eigenvalues: Any
    factor: Any | None
    eigenvectors: Any | None
    whitened: Any
    condition: float
    note: str | None = None


def _values_note(mode: str, condition: float, limit: float) -> tuple[str, str | None]:
    """``values`` within its limit, else ``factor`` and why."""

    if mode != "values":
        return mode, None
    limit = float(limit)
    if limit > 0.0 and condition <= limit:
        return mode, None
    if limit <= 0.0:
        return "factor", "values not allowed"
    # The number itself is the condition of the step.
    return "factor", "condition above the values limit"


def _handover_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in HANDOVER_MODES or mode == "rotation":
        raise ValueError("a hand-over solve is triangular, factor or values")
    return mode


def solve_whitened_handover_on_device(
    raw_overlap: Any,
    raw_projection: Any,
    condition: Any | None = None,
    *,
    mode: str,
    values_condition_limit: float = 0.0,
) -> WhitenedHandover:
    """The small solve of a hand-over on the current device.

    Mirror, condition number, Cholesky factorization, both triangular solves
    and the finiteness check are those of :func:`solve_whitened_ritz_on_device`,
    statement for statement, with its errors in their order, so that the Ritz
    values of ``triangular`` (``eigh`` of the same whitened matrix) are those
    of the rotation, bit for bit.  ``factor`` and ``values`` take the values
    alone (``eigvalsh``).  No ``C`` is formed: unless ``values`` ran, ``U =
    L^-T`` is, and audited as the coefficients are,
    ``max |U.T S U - I| <= 5e-10``.  ``values`` asked where the condition
    number exceeds ``values_condition_limit`` (always with 0) runs as
    ``factor``.  ``condition`` is that of the rotation's solve.
    """

    import cupyx
    from cupyx.scipy.linalg import solve_triangular as device_solve_triangular

    mode = _handover_mode(mode)
    cp, _ = require_cupy()
    lower = cp.tril(raw_overlap)
    overlap = lower + cp.tril(lower, -1).T
    lower = cp.tril(raw_projection)
    projected = lower + cp.tril(lower, -1).T
    del lower
    waiting = []
    judged = None
    if condition is None:
        judged = _judge_condition(_device_overlap_condition, overlap)
    else:
        cp.cuda.get_current_stream().synchronize()
        waiting.append(condition(overlap))
    try:
        with cupyx.errstate(linalg="raise"):
            cholesky = cp.asfortranarray(cp.linalg.cholesky(overlap))
            left_projected = device_solve_triangular(
                cholesky,
                projected,
                lower=True,
            )
            whitened = device_solve_triangular(
                cholesky,
                left_projected.T,
                lower=True,
            ).T
            if not bool(cp.isfinite(whitened).all().item()):
                raise np.linalg.LinAlgError("whitened projection is not finite")
            if mode == "triangular":
                eigenvalues, whitened_vectors = cp.linalg.eigh(whitened)
            else:
                eigenvalues = cp.linalg.eigvalsh(whitened)
                whitened_vectors = None
    except np.linalg.LinAlgError as error:
        if waiting:
            _judge_condition(waiting.pop())
        raise GeneralizedRitzStabilityError(
            "filtered overlap Cholesky/Ritz solve failed"
        ) from error
    except BaseException:
        if waiting:
            _judge_condition(waiting.pop())
        raise
    if waiting:
        judged = _judge_condition(waiting.pop())
    mode, note = _values_note(mode, judged, values_condition_limit)
    factor = None
    if mode != "values":
        columns = int(overlap.shape[0])
        factor = cp.asfortranarray(
            device_solve_triangular(
                cp.asfortranarray(cholesky.T),
                cp.eye(columns, dtype=cp.float64),
                lower=False,
            )
        )
        factor_overlap = factor.T @ overlap @ factor
        audit_error = float(
            cp.abs(factor_overlap - cp.eye(columns, dtype=cp.float64)).max().item()
        )
        if not np.isfinite(audit_error) or audit_error > 5.0e-10:
            raise GeneralizedRitzStabilityError(
                f"generalized Ritz orthogonality audit failed ({audit_error:.3e})"
            )
    return WhitenedHandover(mode, eigenvalues, factor, whitened_vectors, whitened, float(judged), note)


def solve_whitened_handover(
    raw_overlap: np.ndarray,
    raw_projection: np.ndarray,
    *,
    mode: str,
    values_condition_limit: float = 0.0,
) -> WhitenedHandover:
    """The host counterpart of :func:`solve_whitened_handover_on_device`.

    The algebra of :func:`solve_whitened_ritz` on the downloaded pair, with
    its condition number and its errors.  The eigensolve of ``triangular``
    follows ``PARSEC_CUPY_RITZ_EIGH_BACKEND`` as that of the rotation does;
    ``factor`` and ``values`` take the values alone from the same place.
    The arrays returned are host arrays.
    """

    mode = _handover_mode(mode)
    overlap = _mirrored_lower(raw_overlap)
    projected = _mirrored_lower(raw_projection)
    try:
        condition = _overlap_condition(overlap)
    except np.linalg.LinAlgError as error:
        raise GeneralizedRitzStabilityError(
            "filtered overlap condition estimate failed"
        ) from error
    if not np.isfinite(condition) or condition > _generalized_condition_limit():
        raise GeneralizedRitzStabilityError(
            f"filtered overlap condition number {condition:.3e} is unsafe"
        )
    try:
        cholesky = np.linalg.cholesky(overlap)
        left_projected = solve_triangular(
            cholesky,
            projected,
            lower=True,
            check_finite=False,
        )
        whitened = solve_triangular(
            cholesky,
            left_projected.T,
            lower=True,
            check_finite=False,
        ).T
        dense_backend = os.environ.get("PARSEC_CUPY_RITZ_EIGH_BACKEND", "host").strip().lower()
        whitened_vectors = None
        if dense_backend == "cupy":
            import cupyx

            cp, _ = require_cupy()
            with cupyx.errstate(linalg="raise"):
                if mode == "triangular":
                    device_values, device_vectors = cp.linalg.eigh(cp.asarray(whitened))
                    whitened_vectors = cp.asnumpy(device_vectors)
                else:
                    device_values = cp.linalg.eigvalsh(cp.asarray(whitened))
            host_eigenvalues = cp.asnumpy(device_values)
        elif dense_backend == "host":
            if mode == "triangular":
                host_eigenvalues, whitened_vectors = np.linalg.eigh(whitened)
            else:
                host_eigenvalues = np.linalg.eigvalsh(whitened)
        else:
            raise ValueError("PARSEC_CUPY_RITZ_EIGH_BACKEND must be host or cupy")
    except np.linalg.LinAlgError as error:
        raise GeneralizedRitzStabilityError(
            "filtered overlap Cholesky/Ritz solve failed"
        ) from error
    mode, note = _values_note(mode, float(condition), values_condition_limit)
    factor = None
    if mode != "values":
        factor = np.asfortranarray(
            solve_triangular(
                cholesky.T,
                np.eye(int(overlap.shape[0])),
                lower=False,
                check_finite=False,
            )
        )
        audit_error = _orthogonality_error(factor, overlap)
        if not np.isfinite(audit_error) or audit_error > 5.0e-10:
            raise GeneralizedRitzStabilityError(
                f"generalized Ritz orthogonality audit failed ({audit_error:.3e})"
            )
    return WhitenedHandover(
        mode, host_eigenvalues, factor, whitened_vectors, whitened, float(condition), note
    )


def generalized_rayleigh_ritz(
    operator: Any,
    basis: Any,
    *,
    workspace: Any | None = None,
    compute_residuals: bool = False,
    consume_basis: bool = False,
    handover: str = "rotation",
    values_condition_limit: float = 0.0,
) -> DeviceRayleighRitzResult:
    """Solve Rayleigh--Ritz directly in a non-orthogonal filtered basis.

    For filtered columns ``X``, this routine solves

    ``(X.T H X) C = (X.T X) C diag(epsilon)``.

    A Cholesky factor of the small overlap matrix whitens the generalized
    problem.  This is algebraically equivalent to orthonormalizing ``X`` and
    applying ordinary Rayleigh--Ritz, but avoids a tall Householder QR.  The
    overlap condition number, Cholesky factorization, and final coefficient
    orthogonality are audited.  Callers must catch
    :class:`GeneralizedRitzStabilityError` and use robust QR when an unusually
    ill-conditioned filtered basis fails any audit.  The small problem is
    solved without leaving the device, or by :func:`solve_whitened_ritz` on
    the host when :func:`dense_solve_on_device` says so.

    ``workspace`` is a persistent device ``N x m`` array receiving ``H X``.
    Reusing it avoids a costly large allocation/memory-pool eviction in every
    nonlinear iteration.

    With ``consume_basis=True`` the caller transfers its filtered basis to
    this routine. If residuals are disabled and workspace rotation is enabled,
    HX is overwritten only after all projection and stability checks finish;
    the consumed basis becomes the next HX workspace. The two buffers are
    distinct, so no GEMM has overlapping inputs and output. Callers must not
    retain the consumed basis for subsequent use.

    ``handover`` other than ``rotation`` asks for a hand-over
    (``PARSEC_CUPY_RITZ_HANDOVER``, :mod:`handover`): ``triangular`` hands
    back ``X L^-T`` and the eigenvectors of the whitened matrix,
    ``factor`` ``X L^-T`` alone, ``values`` the filtered columns as they are
    (``factor`` above ``values_condition_limit``), each in the memory of the
    basis.  Only the streaming route, one array that the step owns, hands
    over; elsewhere the step rotates.  ``result.handover`` says what ran
    (:class:`handover.HandoverStep`); it is None where nothing else was asked.
    """

    if handover not in HANDOVER_MODES:
        raise ValueError(f"unknown Ritz hand-over {handover!r}")
    cp, _ = require_cupy()
    matrix = cp.asarray(basis, dtype=cp.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("basis must be a nonempty two-dimensional matrix")
    if tuple(operator.shape) != (matrix.shape[0], matrix.shape[0]):
        raise ValueError("operator shape must match basis rows")
    # SUBSPACE assembles independently filtered blocks into a row-major
    # array.  The real-space CUDA kernel traverses one orbital down the grid,
    # and the tall Gram/rotation GEMMs likewise favor column-contiguous
    # storage.  A single device copy (about 0.03 s for Si28H36) avoids strided
    # reads in all three much larger operations; it does not change a value.
    if not matrix.flags.f_contiguous:
        matrix = cp.asfortranarray(matrix)
    # The streaming route needs sole ownership of the basis and no residuals:
    # it never forms the complete ``H X`` and it rotates the basis in place.
    streaming = (
        streaming_ritz_requested(operator)
        and consume_basis
        and not compute_residuals
    )
    dense_on_device = dense_solve_on_device()
    if streaming:
        workspace = None
        applied_basis = None
        with device_stage(operator, "subspace_ritz_projection_seconds"):
            raw_overlap = _symmetric_overlap(matrix)
        # One buffer is the projection slab now and the rotation tile later.
        slab, tile = _streaming_workspace(matrix)
        raw_projection = _streamed_projection(operator, matrix, slab)
    else:
        if (
            workspace is None
            or not isinstance(workspace, cp.ndarray)
            or workspace.dtype != cp.dtype(cp.float64)
            or workspace.shape != matrix.shape
            or not workspace.flags.f_contiguous
            or cp.may_share_memory(workspace, matrix)
        ):
            workspace = cp.empty(matrix.shape, dtype=cp.float64, order="F")

        with device_stage(operator, "subspace_ritz_hamiltonian_seconds"):
            apply_into = getattr(operator, "apply_into", None)
            if callable(apply_into):
                applied_basis = apply_into(matrix, workspace)
            else:
                workspace[...] = operator @ matrix
                applied_basis = workspace

        with device_stage(operator, "subspace_ritz_projection_seconds"):
            raw_overlap = _symmetric_overlap(matrix)
            raw_projection = _lower_triangle_product(matrix, applied_basis)
    with device_stage(operator, "subspace_ritz_projection_seconds"):
        packed = cp.stack((raw_overlap, raw_projection))
        if not dense_on_device:
            packed = np.asarray(cp.asnumpy(packed), dtype=np.float64)
    del raw_overlap, raw_projection
    if handover != "rotation" and streaming:
        # The hand-over: the same pair, a small solve without C, and a
        # triangular product (or none) in place of the rotation.
        handover_solve = (
            solve_whitened_handover_on_device if dense_on_device else solve_whitened_handover
        )
        try:
            solution = handover_solve(
                packed[0],
                packed[1],
                mode=handover,
                values_condition_limit=values_condition_limit,
            )
        except GeneralizedRitzStabilityError:
            del slab, tile
            raise
        del packed
        product = None
        if solution.mode == "values":
            del slab, tile
        else:
            # The buffer of the slab and the tile, whole: the product needs
            # all its bytes, columns of all rows or tiles of all columns.
            buffer = cp.ndarray(
                (max(int(slab.size), int(tile.size)),), dtype=cp.float64, memptr=slab.data
            )
            del slab, tile
            factor = cp.asarray(solution.factor, dtype=cp.float64)
            rows, columns = map(int, matrix.shape)
            width = handover_slab_columns()
            if handover_product() == "columns":
                width = max(1, min(width, int(buffer.size) // max(1, rows)))
            with device_stage(operator, "subspace_ritz_handover_seconds"):
                product = _triangular_in_place(
                    matrix, factor, buffer, uniform_slabs(columns, width)
                )
            del buffer, factor
        return DeviceRayleighRitzResult(
            eigenvalues=cp.asarray(solution.eigenvalues, dtype=cp.float64),
            wavefunctions=matrix,
            applied_wavefunctions=None,
            projected_hamiltonian=cp.asarray(solution.whitened, dtype=cp.float64),
            residual_norms=None,
            workspace=None,
            algorithm="streaming_generalized_cholesky_rayleigh_ritz",
            handover=HandoverStep(
                mode=solution.mode,
                coefficients=(
                    None
                    if solution.eigenvectors is None
                    else cp.asfortranarray(cp.asarray(solution.eigenvectors, dtype=cp.float64))
                ),
                condition=solution.condition,
                note=solution.note,
                product=product,
            ),
        )
    # The host solve returns host arrays, which are uploaded below; the
    # device solve returns device arrays, which are used as they are.
    solve = solve_whitened_ritz_on_device if dense_on_device else solve_whitened_ritz
    try:
        ritz_values, coefficients, whitened = solve(packed[0], packed[1])
    except GeneralizedRitzStabilityError:
        # The slab and tile buffer goes before the caller's QR fallback,
        # which may run while the traceback still refers to this frame.
        if streaming:
            del slab, tile
        raise
    del packed

    device_coefficients = cp.asarray(coefficients, dtype=cp.float64)
    eigenvalues = cp.asarray(ritz_values, dtype=cp.float64)
    with device_stage(operator, "subspace_ritz_rotation_seconds"):
        rotation = os.environ.get("PARSEC_CUPY_RITZ_ROTATION", "allocate").strip().lower()
        if rotation not in {"allocate", "reuse"}:
            raise ValueError("PARSEC_CUPY_RITZ_ROTATION must be allocate or reuse")
        if streaming:
            _rotate_in_place(matrix, device_coefficients, tile)
            wavefunctions = matrix
        elif rotation == "reuse" and consume_basis and not compute_residuals:
            # CuPy's matmul requires C-contiguous out to avoid an internal
            # temporary. Compute (XC).T = C.T X.T into the C-contiguous view
            # of our F-contiguous workspace; neither transpose copies data.
            # All uses of HX have completed on this same stream.
            cp.matmul(device_coefficients.T, matrix.T, out=workspace.T)
            wavefunctions, workspace = workspace, matrix
        else:
            wavefunctions = matrix @ device_coefficients
    if compute_residuals:
        applied_wavefunctions = applied_basis @ device_coefficients
        residuals = (
            applied_wavefunctions
            - wavefunctions * eigenvalues[None, :]
        )
        residual_norms = cp.linalg.norm(residuals, axis=0)
    else:
        applied_wavefunctions = None
        residual_norms = None
    return DeviceRayleighRitzResult(
        eigenvalues=eigenvalues,
        wavefunctions=wavefunctions,
        applied_wavefunctions=applied_wavefunctions,
        projected_hamiltonian=cp.asarray(whitened, dtype=cp.float64),
        residual_norms=residual_norms,
        workspace=workspace,
        algorithm=(
            "streaming_generalized_cholesky_rayleigh_ritz"
            if streaming
            else "generalized_cholesky_rayleigh_ritz"
        ),
        handover=(
            None
            if handover == "rotation"
            else HandoverStep(mode="rotation", note="two arrays: no hand-over off the streaming route")
        ),
    )


def rayleigh_ritz(
    operator: Any,
    basis: Any,
    *,
    compute_residuals: bool = True,
) -> DeviceRayleighRitzResult:
    """Compute a device Rayleigh--Ritz projection and rotation.

    ``CHEBFF`` only needs the Ritz values and rotated vectors to update its
    next filter interval.  PARSEC's CHEBFF path likewise does not form Ritz
    residuals, so ``compute_residuals=False`` skips the extra grid-by-state
    ``(H Q) C`` multiplication and residual reduction.  SUBSPACE keeps the
    default because its residual norms are exposed as iteration diagnostics.
    """

    cp, _ = require_cupy()
    basis = cp.asarray(basis, dtype=cp.float64)
    if basis.ndim != 2 or basis.shape[1] < 1:
        raise ValueError("basis must be a nonempty two-dimensional matrix")
    if tuple(operator.shape) != (basis.shape[0], basis.shape[0]):
        raise ValueError("operator shape must match basis rows")

    applied_basis = operator @ basis
    if applied_basis.shape != basis.shape:
        raise ValueError("operator must preserve the basis shape")
    raw_projection = transposed_product(cp, basis, applied_basis)
    lower = cp.tril(raw_projection)
    projected = lower + cp.tril(lower, -1).T
    eigenvalues, rotations = symmetric_eigh(projected)
    wavefunctions = basis @ rotations
    if compute_residuals:
        applied_wavefunctions = applied_basis @ rotations
        residuals = applied_wavefunctions - wavefunctions * eigenvalues[None, :]
        residual_norms = cp.linalg.norm(residuals, axis=0)
    else:
        applied_wavefunctions = None
        residual_norms = None
    return DeviceRayleighRitzResult(
        eigenvalues=eigenvalues,
        wavefunctions=wavefunctions,
        applied_wavefunctions=applied_wavefunctions,
        projected_hamiltonian=projected,
        residual_norms=residual_norms,
        algorithm="orthonormal_rayleigh_ritz",
    )


__all__ = [
    "DeviceRayleighRitzResult",
    "GeneralizedRitzStabilityError",
    "dense_solve_on_device",
    "generalized_rayleigh_ritz",
    "generalized_ritz_requested",
    "gram_multiple",
    "one_array_buffer_limit",
    "rayleigh_ritz",
    "solve_whitened_ritz",
    "solve_whitened_ritz_on_device",
    "streaming_ritz_requested",
    "streaming_workspace_bytes",
]
