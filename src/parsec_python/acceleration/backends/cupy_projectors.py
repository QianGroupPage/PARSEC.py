"""Low-launch-overhead CUDA factors for sparse KB projectors.

Representation sectors contain only a handful of projectors and at most a
few vector columns per Chebyshev block.  Dispatching a general cuSPARSE SpMM
for every ``B.T @ X`` therefore spends more time in library setup than in the
dot products.  Short projector rows use one canonical-order CUDA thread per
projector/vector pair; genuinely long rows use one 128-thread reduction
block.  The large ``B @ coeff`` scatter remains fused into the stencil kernel.
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


_KERNEL_CACHE: dict[int, tuple[Any, Any, Any]] = {}
_KERNEL_LOCK = Lock()


def _kernels(cp: Any) -> tuple[Any, Any, Any]:
    """Compile both factor kernels once per CUDA device."""

    device_id = int(cp.cuda.Device().id)
    with _KERNEL_LOCK:
        selected = _KERNEL_CACHE.get(device_id)
        if selected is None:
            module = cp.RawModule(
                code=_SOURCE,
                options=("--std=c++11",),
                name_expressions=(
                    "sparse_projector_projection",
                    "sparse_projector_projection_serial",
                    "sparse_projector_scatter",
                ),
            )
            compile_cupy_raw(module)
            selected = (
                module.get_function("sparse_projector_projection"),
                module.get_function("sparse_projector_scatter"),
                module.get_function("sparse_projector_projection_serial"),
            )
            _KERNEL_CACHE[device_id] = selected
    return selected


class CuPySparseProjectorProjection:
    """Cache canonical ``B.T`` CSR buffers for repeated small-block dots."""

    def __init__(
        self,
        cp: Any,
        projector_transpose: Any,
        signs: Any,
        *,
        max_row_entries: int | None = None,
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
        self.reduction_mode = (
            "parallel" if self.parallel_reduction else "serial"
        )
        kernels = _kernels(cp)
        self.kernel = kernels[0] if self.parallel_reduction else kernels[2]

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
        threads = 128
        pair_count = self.projector_count * width
        itemsize = int(block.dtype.itemsize)
        grid = (pair_count,) if self.parallel_reduction else (
            (pair_count + threads - 1) // threads,
        )
        self.kernel(
            grid,
            (threads,),
            (
                np.int32(self.projector_count),
                np.int32(width),
                self.row_offsets,
                self.grid_rows,
                self.values,
                self.signs,
                block,
                np.int64(block.strides[0] // itemsize),
                np.int64(block.strides[1] // itemsize),
                output,
                np.int64(output.strides[0] // itemsize),
                np.int64(output.strides[1] // itemsize),
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

    def __init__(self, cp: Any, host_projectors: Any, signs: Any) -> None:
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
]
