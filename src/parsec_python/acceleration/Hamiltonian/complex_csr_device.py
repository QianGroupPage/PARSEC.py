"""A complex CSR matrix on the device with a block product on row-major complex blocks.

The operators of the sectors of a k-point (:mod:`..SCF.kpoint_sectors`) are complex sparse matrices of no
stencil form: the reduced kinetic term (``-nabla^2 - 2ik.grad + |k|^2`` or the Bloch Laplacian, about as many
entries per row as the stencil) and the reduced Bloch projectors.  The CuPy product of a complex CSR matrix and
a row-major block (cuSPARSE, with the conversions of the block and the result between the two layouts) ran about
forty times below the stencil kernel per row and column on A100 (diamond 7^3, 2 x 2 x 2 mesh).  Here one thread
forms one entry of the result, ``out[r, c] = sum_p M[r, j_p] x[j_p, c]`` (plus ``d[r] x[r, c]`` where a diagonal
is given), with the threads of a warp along the columns of one row: the row's entries are read once per warp and
the block rows it gathers are read along their columns.  Rows of very many entries (the adjoint of the
projectors, one row per projector over its whole support) split their sum over the threads of one block
(:func:`complex_csr_adjoint_kernel`), a fixed order of summation, so the product is the same in every run.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_SOURCE = r"""
#include <cuComplex.h>
extern "C" __global__ void complex_csr_block_product(
    const long long rows, const long long count,
    const int* __restrict__ indptr, const int* __restrict__ indices, const double2* __restrict__ values,
    const double2* __restrict__ block, const long long block_stride,
    double2* __restrict__ output, const long long output_stride,
    const double* __restrict__ diagonal, const int with_diagonal, const int accumulate)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * count) return;
    const long long r = t / count;
    const long long c = t - r * count;
    double re = 0.0, im = 0.0;
    const int end = indptr[r + 1];
    for (int p = indptr[r]; p < end; ++p) {
        const double2 a = values[p];
        const double2 x = block[(long long)indices[p] * block_stride + c];
        re += a.x * x.x - a.y * x.y;
        im += a.x * x.y + a.y * x.x;
    }
    if (with_diagonal) {
        const double d = diagonal[r];
        const double2 x = block[r * block_stride + c];
        re += d * x.x;
        im += d * x.y;
    }
    const long long o = r * output_stride + c;
    if (accumulate) {
        output[o].x += re;
        output[o].y += im;
    } else {
        output[o] = make_double2(re, im);
    }
}

// One block of 256 threads per (row, tile of 8 columns): thread (lane, column) = (threadIdx.x % 32,
// threadIdx.x / 32); the 32 lanes of a column stride the row's entries and their partial sums are added in a
// fixed tree (shared memory), the same order in every run.
extern "C" __global__ void complex_csr_long_rows(
    const long long rows, const long long count,
    const int* __restrict__ indptr, const int* __restrict__ indices, const double2* __restrict__ values,
    const double2* __restrict__ block, const long long block_stride,
    double2* __restrict__ output, const long long output_stride, const int accumulate)
{
    __shared__ double2 partial[256];
    const long long r = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int column_in_tile = threadIdx.x >> 5;
    const long long c = (long long)blockIdx.y * 8 + column_in_tile;
    double re = 0.0, im = 0.0;
    if (r < rows && c < count) {
        const int end = indptr[r + 1];
        for (int p = indptr[r] + lane; p < end; p += 32) {
            const double2 a = values[p];
            const double2 x = block[(long long)indices[p] * block_stride + c];
            re += a.x * x.x - a.y * x.y;
            im += a.x * x.y + a.y * x.x;
        }
    }
    partial[threadIdx.x] = make_double2(re, im);
    __syncthreads();
    for (int width = 16; width > 0; width >>= 1) {
        if (lane < width) {
            partial[threadIdx.x].x += partial[threadIdx.x + width].x;
            partial[threadIdx.x].y += partial[threadIdx.x + width].y;
        }
        __syncthreads();
    }
    if (lane == 0 && r < rows && c < count) {
        const long long o = r * output_stride + c;
        if (accumulate) {
            output[o].x += partial[threadIdx.x].x;
            output[o].y += partial[threadIdx.x].y;
        } else {
            output[o] = partial[threadIdx.x];
        }
    }
}
"""

_KERNELS: dict[tuple[int, str], object] = {}
# rows of more entries than this on average take the kernel that splits a row over 32 threads
_LONG_ROWS = 256


def _kernel(cp, name: str):
    device = int(cp.cuda.Device().id)
    kernel = _KERNELS.get((device, name))
    if kernel is None:
        kernel = cp.RawKernel(_SOURCE, name, options=("--std=c++11", "--fmad=false"))
        kernel.compile()
        _KERNELS[(device, name)] = kernel
    return kernel


class DeviceComplexCSR:
    """A complex128 CSR matrix on the current device (int32 indices) with :meth:`product` on row-major blocks.

    ``data``, ``indices`` and ``indptr`` are the device arrays (what the memory accounting of the k-point pools
    reads); ``dtype`` is complex128.
    """

    dtype = np.complex128

    def __init__(self, cp, matrix) -> None:
        matrix = sp.csr_matrix(matrix, dtype=np.complex128)
        matrix.sum_duplicates()
        matrix.sort_indices()
        if matrix.nnz >= 2**31 or max(matrix.shape) >= 2**31:
            raise ValueError("a complex device CSR matrix needs fewer than 2**31 entries and rows")
        self.cp = cp
        self.shape = tuple(int(value) for value in matrix.shape)
        self.indptr = cp.asarray(matrix.indptr.astype(np.int32))
        self.indices = cp.asarray(matrix.indices.astype(np.int32))
        self.data = cp.asarray(np.ascontiguousarray(matrix.data))
        self.long_rows = matrix.nnz > _LONG_ROWS * max(1, self.shape[0])
        self.nbytes = int(self.indptr.nbytes + self.indices.nbytes + self.data.nbytes)

    def product(self, block, output=None, *, diagonal=None, accumulate: bool = False):
        """``output (+)= M block (+ diag(diagonal) block)`` for a complex block of shape ``(columns of M, k)``."""

        cp = self.cp
        if block.ndim != 2 or block.shape[0] != self.shape[1]:
            raise ValueError("the block does not match the matrix")
        block = cp.ascontiguousarray(block, dtype=cp.complex128)
        rows, count = self.shape[0], int(block.shape[1])
        if output is None:
            output = cp.empty((rows, count), dtype=cp.complex128)
            accumulate = False
        elif output.shape != (rows, count) or not output.flags.c_contiguous or output.dtype != cp.complex128:
            raise ValueError("the output must be a row-major complex block of the product's shape")
        if rows * count == 0:
            return output
        if diagonal is not None and self.shape[0] != self.shape[1]:
            raise ValueError("a diagonal term needs a square matrix")
        if self.long_rows and diagonal is None:
            tiles = (count + 7) // 8
            _kernel(cp, "complex_csr_long_rows")(
                (rows, tiles),
                (256,),
                (
                    np.int64(rows), np.int64(count), self.indptr, self.indices, self.data,
                    block, np.int64(block.strides[0] // 16),
                    output, np.int64(output.strides[0] // 16), np.int32(bool(accumulate)),
                ),
            )
            return output
        threads = 256
        total = rows * count
        _kernel(cp, "complex_csr_block_product")(
            ((total + threads - 1) // threads,),
            (threads,),
            (
                np.int64(rows), np.int64(count), self.indptr, self.indices, self.data,
                block, np.int64(block.strides[0] // 16),
                output, np.int64(output.strides[0] // 16),
                self.data if diagonal is None else cp.ascontiguousarray(diagonal, dtype=cp.float64),
                np.int32(diagonal is not None), np.int32(bool(accumulate)),
            ),
        )
        return output


__all__ = ["DeviceComplexCSR"]
