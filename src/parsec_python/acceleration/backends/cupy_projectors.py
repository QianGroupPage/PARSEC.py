"""Low-launch-overhead CUDA factors for sparse KB projectors.

Representation sectors contain only a handful of projectors and at most a
few vector columns per Chebyshev block.  Dispatching a general cuSPARSE SpMM
for every ``B.T @ X`` therefore spends more time in library setup than in the
dot products.  Short projector rows use one canonical-order CUDA thread per
projector/vector pair; genuinely long rows use one 128-thread reduction
block per projector, which carries up to six vector columns
(:func:`projector_block_columns`).  The large ``B @ coeff`` scatter remains
fused into the stencil kernel.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

import numpy as np

from .cupy_compile import compile_cupy_raw


_SOURCE = r"""
extern "C" __global__
void sparse_projector_projection(
    const int projector_count,
    const int width,
    const int* __restrict__ row_offsets,
    const int* __restrict__ grid_rows,
    const double* __restrict__ projector_values,
    const double* __restrict__ signs,
    const double* __restrict__ vectors,
    const long long vector_row_stride,
    const long long vector_column_stride,
    double* __restrict__ output,
    const long long output_row_stride,
    const long long output_column_stride
) {
    const int pair = blockIdx.x;
    const int pair_count = projector_count * width;
    if (pair >= pair_count) {
        return;
    }
    const int projector = pair / width;
    const int column = pair - projector * width;
    double partial = 0.0;
    for (
        int position = row_offsets[projector] + threadIdx.x;
        position < row_offsets[projector + 1];
        position += blockDim.x
    ) {
        partial += projector_values[position] * vectors[
            static_cast<long long>(grid_rows[position]) * vector_row_stride +
            static_cast<long long>(column) * vector_column_stride
        ];
    }
    __shared__ double sums[128];
    sums[threadIdx.x] = partial;
    __syncthreads();
    for (int offset = 64; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sums[threadIdx.x] += sums[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[
            static_cast<long long>(projector) * output_row_stride +
            static_cast<long long>(column) * output_column_stride
        ] = signs[projector] * sums[0];
    }
}

extern "C" __global__
void sparse_projector_projection6(
    const int projector_count,
    const int width,
    const int* __restrict__ row_offsets,
    const int* __restrict__ grid_rows,
    const double* __restrict__ projector_values,
    const double* __restrict__ signs,
    const double* __restrict__ vectors,
    const long long vector_row_stride,
    const long long vector_column_stride,
    double* __restrict__ output,
    const long long output_row_stride,
    const long long output_column_stride
) {
    // One block per projector for up to six columns.  A thread keeps, for
    // every column, the strided partial sum that the kernel above forms in
    // the block of that column, and every column is reduced by the same
    // tree: the coefficients are those of that kernel to the last bit.  The
    // index and the value of an entry are read once for the six columns.
    const int projector = blockIdx.x;
    if (projector >= projector_count) {
        return;
    }
    double partial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (
        int position = row_offsets[projector] + threadIdx.x;
        position < row_offsets[projector + 1];
        position += blockDim.x
    ) {
        const double value = projector_values[position];
        const long long base =
            static_cast<long long>(grid_rows[position]) * vector_row_stride;
#pragma unroll
        for (int column = 0; column < 6; ++column) {
            if (column < width) {
                partial[column] += value * vectors[
                    base + static_cast<long long>(column) * vector_column_stride
                ];
            }
        }
    }
    __shared__ double sums[6][128];
#pragma unroll
    for (int column = 0; column < 6; ++column) {
        sums[column][threadIdx.x] = partial[column];
    }
    __syncthreads();
    for (int offset = 64; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
#pragma unroll
            for (int column = 0; column < 6; ++column) {
                sums[column][threadIdx.x] += sums[column][threadIdx.x + offset];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int column = 0; column < 6; ++column) {
            if (column < width) {
                output[
                    static_cast<long long>(projector) * output_row_stride +
                    static_cast<long long>(column) * output_column_stride
                ] = signs[projector] * sums[column][0];
            }
        }
    }
}

extern "C" __global__
void sparse_projector_projection_serial(
    const int projector_count,
    const int width,
    const int* __restrict__ row_offsets,
    const int* __restrict__ grid_rows,
    const double* __restrict__ projector_values,
    const double* __restrict__ signs,
    const double* __restrict__ vectors,
    const long long vector_row_stride,
    const long long vector_column_stride,
    double* __restrict__ output,
    const long long output_row_stride,
    const long long output_column_stride
) {
    const int pair = blockDim.x * blockIdx.x + threadIdx.x;
    const int pair_count = projector_count * width;
    if (pair >= pair_count) {
        return;
    }
    const int projector = pair / width;
    const int column = pair - projector * width;
    double value = 0.0;
    for (
        int position = row_offsets[projector];
        position < row_offsets[projector + 1];
        ++position
    ) {
        value += projector_values[position] * vectors[
            static_cast<long long>(grid_rows[position]) * vector_row_stride +
            static_cast<long long>(column) * vector_column_stride
        ];
    }
    output[
        static_cast<long long>(projector) * output_row_stride +
        static_cast<long long>(column) * output_column_stride
    ] = signs[projector] * value;
}

extern "C" __global__
void sparse_projector_scatter(
    const int grid_size,
    const int width,
    const int* __restrict__ row_offsets,
    const int* __restrict__ projector_columns,
    const double* __restrict__ projector_values,
    const double* __restrict__ coefficients,
    const long long coefficient_row_stride,
    const long long coefficient_column_stride,
    double* __restrict__ output,
    const long long output_row_stride,
    const long long output_column_stride
) {
    const int pair = blockDim.x * blockIdx.x + threadIdx.x;
    const int pair_count = grid_size * width;
    if (pair >= pair_count) {
        return;
    }
    const int row = pair / width;
    const int column = pair - row * width;
    double value = 0.0;
    for (
        int position = row_offsets[row];
        position < row_offsets[row + 1];
        ++position
    ) {
        value += projector_values[position] * coefficients[
            static_cast<long long>(projector_columns[position])
                * coefficient_row_stride
            + static_cast<long long>(column) * coefficient_column_stride
        ];
    }
    output[
        static_cast<long long>(row) * output_row_stride
        + static_cast<long long>(column) * output_column_stride
    ] = value;
}
"""


_KERNEL_CACHE: dict[int, tuple[Any, Any, Any, Any]] = {}
_KERNEL_LOCK = Lock()

# Threads of a CUDA block of the projection kernels, and the vector columns
# that one block of ``sparse_projector_projection6`` carries: both are sizes
# of that kernel's arrays.
_THREADS = 128
_BLOCK_COLUMNS = 6


def projector_block_columns() -> int:
    """Return ``PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS``: 6, the default, or 1.

    The vector columns that one CUDA block of the tree reduction carries, the
    reduction of projector rows of at least 256 entries
    (``PARSEC_CUPY_PROJECTOR_REDUCTION``).  With 6 a block serves one
    projector and up to six columns: it reads the index and the value of an
    entry once and reduces six partial sums behind the same barriers.  With
    1, the former kernel, a block serves one projector and one column.  Both
    form, for every column, the same strided partial sums and reduce them by
    the same tree, so the coefficients agree to the last bit.

    On an A100 one projection of six columns of a synthetic factor took
    0.066 against 0.097 ms with 4,601 projectors of 324 entries on 2,873,400
    rows and 0.133 against 0.193 ms with 9,449 of 311 on 5,660,200, 1.47 and
    1.45 times as fast, with identical coefficients for every width from one
    to six columns.

    The canonical-order kernel of short rows is one thread per projector and
    column under either value.  An operator reads the switch when it is
    built and keeps its kernel, also in the filter graphs it records.
    """

    raw = os.environ.get("PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS", "").strip()
    if raw in ("", str(_BLOCK_COLUMNS)):
        return _BLOCK_COLUMNS
    if raw == "1":
        return 1
    raise ValueError("PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS must be 6 or 1")


def projection_reduction(parallel: bool, block_columns: int) -> str:
    """What an operator reports for its projection kernel.

    ``serial`` is the canonical-order kernel, which ``block_columns`` does
    not reach, ``parallel`` the tree reduction with a block per projector
    and column and ``parallel6`` the one with a block per projector and six
    columns.
    """

    if int(block_columns) not in (1, _BLOCK_COLUMNS):
        raise ValueError("a block of the projection carries 6 columns or 1")
    if not parallel:
        return "serial"
    return "parallel" if int(block_columns) == 1 else "parallel6"


def projection_launches(
    reduction: str, projector_count: int, width: int
) -> tuple[tuple[int, int, int], ...]:
    """Return ``(blocks, first column, last column)`` of every projection launch.

    The launches that form the coefficients of ``width`` vector columns with
    the kernel of ``reduction`` (:func:`projection_reduction`), each with
    blocks of 128 threads.  ``serial`` and ``parallel`` take all columns in
    one launch, of a thread or of a block per projector and column.
    ``parallel6`` takes six columns per launch, the narrower rest in the
    last, with a block per projector: a block of more than six columns is
    launched as the stencil kernels take it.
    """

    projector_count, width = int(projector_count), int(width)
    if reduction == "parallel6":
        return tuple(
            (projector_count, start, min(start + _BLOCK_COLUMNS, width))
            for start in range(0, width, _BLOCK_COLUMNS)
        )
    pair_count = projector_count * width
    if reduction == "parallel":
        return ((pair_count, 0, width),)
    if reduction == "serial":
        return (((pair_count + _THREADS - 1) // _THREADS, 0, width),)
    raise ValueError(f"unknown projection reduction {reduction!r}")


def _kernels(cp: Any) -> tuple[Any, Any, Any, Any]:
    """Compile the factor kernels once per CUDA device.

    In this order: the tree projection of one column per block, the scatter,
    the canonical-order projection and the tree projection of six columns
    per block.
    """

    device_id = int(cp.cuda.Device().id)
    with _KERNEL_LOCK:
        selected = _KERNEL_CACHE.get(device_id)
        if selected is None:
            module = cp.RawModule(
                code=_SOURCE,
                options=("--std=c++11",),
                name_expressions=(
                    "sparse_projector_projection",
                    "sparse_projector_projection6",
                    "sparse_projector_projection_serial",
                    "sparse_projector_scatter",
                ),
            )
            compile_cupy_raw(module)
            selected = (
                module.get_function("sparse_projector_projection"),
                module.get_function("sparse_projector_scatter"),
                module.get_function("sparse_projector_projection_serial"),
                module.get_function("sparse_projector_projection6"),
            )
            _KERNEL_CACHE[device_id] = selected
    return selected


# Where :func:`_kernels` holds the projection kernel of a reduction.
_PROJECTION_KERNEL = {"parallel": 0, "serial": 2, "parallel6": 3}


class CuPySparseProjectorProjection:
    """Cache canonical ``B.T`` CSR buffers for repeated small-block dots."""

    def __init__(
        self,
        cp: Any,
        projector_transpose: Any,
        signs: Any,
        *,
        max_row_entries: int | None = None,
        block_columns: int | None = None,
    ) -> None:
        if isinstance(projector_transpose, tuple):
            row_offsets, grid_rows, values, shape = projector_transpose
        else:
            row_offsets = projector_transpose.indptr
            grid_rows = projector_transpose.indices
            values = projector_transpose.data
            shape = projector_transpose.shape
        if row_offsets.dtype != cp.dtype(cp.int32) or (
            grid_rows.dtype != cp.dtype(cp.int32)
        ):
            raise ValueError("custom projector projection requires int32 CSR")
        self.cp = cp
        self.row_offsets = row_offsets
        self.grid_rows = grid_rows
        self.values = values
        self.signs = signs
        self.projector_count = int(shape[0])
        policy = os.environ.get(
            "PARSEC_CUPY_PROJECTOR_REDUCTION", "auto"
        ).strip().lower()
        if policy not in {"auto", "parallel", "serial"}:
            raise ValueError(
                "PARSEC_CUPY_PROJECTOR_REDUCTION must be auto, parallel, "
                "or serial"
            )
        # A shared-memory tree pays for seven barriers.  Below two warps of
        # work per lane block, the one-thread canonical traversal is faster
        # and retains the exact source summation order.  Long rows expose
        # enough independent products to amortize the reduction barriers.
        self.max_row_entries = int(max_row_entries or 0)
        self.parallel_reduction = (
            policy == "parallel"
            or (policy == "auto" and self.max_row_entries >= 256)
        )
        # The columns that a block of the tree reduction carries: those the
        # caller names, else PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS.  The
        # canonical-order kernel has a thread per projector and column.
        if block_columns is None:
            block_columns = projector_block_columns()
        self.reduction_mode = projection_reduction(
            self.parallel_reduction, block_columns
        )
        self.block_columns = (
            _BLOCK_COLUMNS if self.reduction_mode == "parallel6" else 1
        )
        # The operator launches this kernel for as long as it lives, in the
        # filter graphs it records as well as outside them.
        self.kernel = _kernels(cp)[_PROJECTION_KERNEL[self.reduction_mode]]

    def __call__(self, vectors: Any, *, output: Any | None = None):
        cp = self.cp
        # Chebyshev recurrences pass an already validated float64 CuPy view.
        # Avoid routing that hot path through ``cp.asarray`` thousands of
        # times; modular host/other-dtype callers retain the conversion.
        block = (
            vectors
            if isinstance(vectors, cp.ndarray)
            and vectors.dtype == cp.dtype(cp.float64)
            else cp.asarray(vectors, dtype=cp.float64)
        )
        was_vector = block.ndim == 1
        if was_vector:
            block = block[:, None]
        if block.ndim != 2:
            raise ValueError("projector input must be a vector or column block")
        width = int(block.shape[1])
        if output is None:
            output = cp.empty(
                (self.projector_count, width), dtype=cp.float64, order="F"
            )
        elif output.shape != (self.projector_count, width) or output.dtype != cp.dtype(cp.float64):
            raise ValueError("projector output must match the float64 coefficients")
        itemsize = int(block.dtype.itemsize)
        for blocks, start, stop in projection_launches(
            self.reduction_mode, self.projector_count, width
        ):
            # One launch takes the arrays as they are; the launches of a
            # block of more than six columns take views of its columns.
            whole = start == 0 and stop == width
            source = block if whole else block[:, start:stop]
            target = output if whole else output[:, start:stop]
            self.kernel(
                (blocks,),
                (_THREADS,),
                (
                    np.int32(self.projector_count),
                    np.int32(stop - start),
                    self.row_offsets,
                    self.grid_rows,
                    self.values,
                    self.signs,
                    source,
                    np.int64(source.strides[0] // itemsize),
                    np.int64(source.strides[1] // itemsize),
                    target,
                    np.int64(target.strides[0] // itemsize),
                    np.int64(target.strides[1] // itemsize),
                ),
            )
        return output[:, 0] if was_vector else output


class CuPySparseProjectorFactors:
    """Own raw device CSR factors without constructing a cuSPARSE matrix.

    Production representation Hamiltonians always use a custom canonical
    ``B.T`` projection followed by a stencil-fused ``B`` scatter.  Uploading
    CuPy sparse-matrix wrappers in that path needlessly initializes cuSPARSE
    once per cold process and duplicates the same CSR buffers.  This class
    keeps precisely the arrays consumed by the two CUDA kernels instead.

    The standalone :meth:`apply` method retains the modular nonlocal API with
    the same factorization and canonical CSR row traversal.
    """

    def __init__(
        self,
        cp: Any,
        host_projectors: Any,
        signs: Any,
        *,
        block_columns: int | None = None,
    ) -> None:
        host = host_projectors.tocsr(copy=True)
        host.sum_duplicates()
        host.sort_indices()
        host_transpose = host.T.tocsr(copy=True)
        host_transpose.sum_duplicates()
        host_transpose.sort_indices()
        if host.indices.dtype != np.int32 or host.indptr.dtype != np.int32:
            raise ValueError("custom projector factors require int32 host CSR")
        if (
            host_transpose.indices.dtype != np.int32
            or host_transpose.indptr.dtype != np.int32
        ):
            raise ValueError("custom projector transpose requires int32 host CSR")

        self.cp = cp
        self.grid_size = int(host.shape[0])
        self.projector_count = int(host.shape[1])
        self.row_offsets = cp.asarray(host.indptr, dtype=cp.int32)
        self.projector_columns = cp.asarray(host.indices, dtype=cp.int32)
        self.values = cp.asarray(host.data, dtype=cp.float64)
        self.signs = cp.asarray(signs, dtype=cp.float64)
        transpose = (
            cp.asarray(host_transpose.indptr, dtype=cp.int32),
            cp.asarray(host_transpose.indices, dtype=cp.int32),
            cp.asarray(host_transpose.data, dtype=cp.float64),
            host_transpose.shape,
        )
        row_lengths = np.diff(host_transpose.indptr)
        self.projection = CuPySparseProjectorProjection(
            cp,
            transpose,
            self.signs,
            max_row_entries=int(row_lengths.max(initial=0)),
            block_columns=block_columns,
        )
        self.scatter_kernel = _kernels(cp)[1]

    @property
    def csr_data(self) -> tuple[Any, Any, Any]:
        """Device CSR buffers consumed by the fused stencil scatter."""

        return self.row_offsets, self.projector_columns, self.values

    def signed_coefficients(self, vectors: Any):
        return self.projection(vectors)

    def scatter(self, coefficients: Any):
        cp = self.cp
        coefficient_block = cp.asarray(coefficients, dtype=cp.float64)
        was_vector = coefficient_block.ndim == 1
        if was_vector:
            coefficient_block = coefficient_block[:, None]
        if coefficient_block.ndim != 2 or (
            coefficient_block.shape[0] != self.projector_count
        ):
            raise ValueError("projector coefficients have an incompatible shape")
        width = int(coefficient_block.shape[1])
        output = cp.empty((self.grid_size, width), dtype=cp.float64, order="F")
        threads = 128
        pair_count = self.grid_size * width
        itemsize = int(output.dtype.itemsize)
        self.scatter_kernel(
            ((pair_count + threads - 1) // threads,),
            (threads,),
            (
                np.int32(self.grid_size),
                np.int32(width),
                self.row_offsets,
                self.projector_columns,
                self.values,
                coefficient_block,
                np.int64(coefficient_block.strides[0] // itemsize),
                np.int64(coefficient_block.strides[1] // itemsize),
                output,
                np.int64(output.strides[0] // itemsize),
                np.int64(output.strides[1] // itemsize),
            ),
        )
        return output[:, 0] if was_vector else output

    def apply(self, vectors: Any):
        return self.scatter(self.signed_coefficients(vectors))


__all__ = [
    "CuPySparseProjectorFactors",
    "CuPySparseProjectorProjection",
    "projection_launches",
    "projection_reduction",
    "projector_block_columns",
]
