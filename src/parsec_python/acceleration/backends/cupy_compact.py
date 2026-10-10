"""Lossless compact CUDA action for repeated finite-difference coefficients.

PARSEC's centered real-space Laplacian contains only the diagonal coefficient
and one coefficient for every signed shell.  A generic CSR stores one float64
value for every nonzero.  This module retains canonical CSR row/summation
order but stores each value as a uint8 code into an exact float64 palette and
each grid column as int32.  A CuPy RawKernel applies up to six orbital columns
per row, reusing the compact stencil metadata across those columns and, when
requested, adding the local potential before writing the result.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np
import scipy.sparse as sp

from .cupy_compile import compile_cupy_raw


_CUDA_SOURCE = r"""
extern "C" __global__
void compact_stencil_spmm6(
    const long long row_count,
    const long long* __restrict__ indptr,
    const int* __restrict__ indices,
    const unsigned char* __restrict__ coefficient_codes,
    const double* __restrict__ coefficient_palette,
    const double* __restrict__ local_potential,
    const int add_local,
    const double* __restrict__ vectors,
    const long long vector_row_stride,
    const long long vector_column_stride,
    const int width,
    double* __restrict__ output,
    const long long output_row_stride,
    const long long output_column_stride
) {
    const long long row =
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }

    double accumulator[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const long long start = indptr[row];
    const long long stop = indptr[row + 1];
    for (long long offset = start; offset < stop; ++offset) {
        const int source_row = indices[offset];
        const double coefficient =
            coefficient_palette[coefficient_codes[offset]];
#pragma unroll
        for (int column = 0; column < 6; ++column) {
            if (column < width) {
                accumulator[column] += coefficient * vectors[
                    static_cast<long long>(source_row) * vector_row_stride +
                    static_cast<long long>(column) * vector_column_stride
                ];
            }
        }
    }
    if (add_local) {
        const double potential = local_potential[row];
#pragma unroll
        for (int column = 0; column < 6; ++column) {
            if (column < width) {
                accumulator[column] += potential * vectors[
                    row * vector_row_stride +
                    static_cast<long long>(column) * vector_column_stride
                ];
            }
        }
    }
#pragma unroll
    for (int column = 0; column < 6; ++column) {
        if (column < width) {
            output[
                row * output_row_stride +
                static_cast<long long>(column) * output_column_stride
            ] = accumulator[column];
        }
    }
}
"""


_KERNEL_CACHE: dict[int, Any] = {}
_KERNEL_CACHE_LOCK = Lock()


def _compiled_kernel(cp: Any) -> Any:
    """Return one compiled compact-stencil kernel per CUDA device."""

    device_id = int(cp.cuda.Device().id)
    with _KERNEL_CACHE_LOCK:
        kernel = _KERNEL_CACHE.get(device_id)
        if kernel is None:
            kernel = cp.RawKernel(
                _CUDA_SOURCE,
                "compact_stencil_spmm6",
                options=("--std=c++11",),
            )
            compile_cupy_raw(kernel)
            _KERNEL_CACHE[device_id] = kernel
    return kernel


def _coefficient_palette(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return an exact float64 bit-pattern palette and uint8 codes."""

    data = np.ascontiguousarray(values, dtype=np.float64)
    # Operate on IEEE-754 bits so +0/-0 and any other bit-distinct float64
    # values remain distinct.
    bits = data.view(np.uint64).ravel()
    if bits.size == 0:
        return bits.view(np.float64), np.empty(data.shape, dtype=np.uint8)
    # A stencil has at most a few dozen distinct coefficients among tens of
    # millions of entries.  Sorting every entry (np.unique) took seconds per
    # symmetry sector.  Instead grow the sorted palette from the entries it
    # does not yet contain and encode by binary search; the result equals
    # np.unique(bits, return_inverse=True) exactly.
    stride = max(1, bits.size // 1_000_000)
    palette = np.unique(bits[::stride])
    while True:
        if palette.size > 256:
            return None
        codes = np.searchsorted(palette, bits)
        np.minimum(codes, palette.size - 1, out=codes)
        missing = np.flatnonzero(palette[codes] != bits)
        if missing.size == 0:
            break
        palette = np.union1d(palette, bits[missing[:1_000_000]])
    return palette.view(np.float64), codes.astype(np.uint8, copy=False).reshape(data.shape)


class CuPyCompactFiniteDifference:
    """Device-resident compact CSR-order stencil and fused local action."""

    chunk_width = 6

    def __init__(self, cp: Any, matrix: sp.spmatrix) -> None:
        host = sp.csr_matrix(matrix, dtype=np.float64)
        host.sum_duplicates()
        host.sort_indices()
        if host.shape[0] != host.shape[1]:
            raise ValueError("finite-difference operator must be square")
        if host.shape[0] > np.iinfo(np.int32).max:
            raise ValueError("compact CUDA stencil requires int32 grid rows")
        encoded = _coefficient_palette(host.data)
        if encoded is None:
            raise ValueError("finite-difference operator has more than 256 coefficients")
        palette, codes = encoded

        self.cp = cp
        self.shape = host.shape
        self.palette_size = int(palette.size)
        self.indptr = cp.asarray(host.indptr, dtype=cp.int64)
        self.indices = cp.asarray(host.indices, dtype=cp.int32)
        self.coefficient_codes = cp.asarray(codes, dtype=cp.uint8)
        self.coefficient_palette = cp.asarray(palette, dtype=cp.float64)
        # All representation operators use the same kernel ABI.  Compile it
        # once; sparse buffers and numerical data remain sector-private.
        self.kernel = _compiled_kernel(cp)

    @property
    def storage_mode(self) -> str:
        return "int32_columns_uint8_coefficient_palette"

    def apply(
        self,
        vectors: Any,
        local_potential: Any | None = None,
    ):
        """Apply the compact stencil, optionally followed by diagonal V_local."""

        cp = self.cp
        block = cp.asarray(vectors, dtype=cp.float64)
        was_vector = block.ndim == 1
        if was_vector:
            block = block[:, None]
        if block.ndim != 2 or block.shape[0] != self.shape[0]:
            raise ValueError("vectors do not match the compact stencil")
        if not block.flags.f_contiguous:
            block = cp.asfortranarray(block)

        if local_potential is None:
            add_local = 0
            # A valid device pointer is still required by the raw signature.
            potential = self.coefficient_palette
        else:
            potential = cp.asarray(local_potential, dtype=cp.float64)
            if potential.shape != (self.shape[0],):
                raise ValueError("local potential does not match the compact stencil")
            add_local = 1

        output = cp.empty(block.shape, dtype=cp.float64, order="F")
        threads = 256
        grid = ((self.shape[0] + threads - 1) // threads,)
        itemsize = int(block.dtype.itemsize)
        for start in range(0, int(block.shape[1]), self.chunk_width):
            stop = min(start + self.chunk_width, int(block.shape[1]))
            source = block[:, start:stop]
            target = output[:, start:stop]
            self.kernel(
                grid,
                (threads,),
                (
                    np.int64(self.shape[0]),
                    self.indptr,
                    self.indices,
                    self.coefficient_codes,
                    self.coefficient_palette,
                    potential,
                    np.int32(add_local),
                    source,
                    np.int64(source.strides[0] // itemsize),
                    np.int64(source.strides[1] // itemsize),
                    np.int32(stop - start),
                    target,
                    np.int64(target.strides[0] // itemsize),
                    np.int64(target.strides[1] // itemsize),
                ),
            )
        return output[:, 0] if was_vector else output


__all__ = ["CuPyCompactFiniteDifference"]
