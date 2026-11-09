"""The local part of a Bloch Hamiltonian on an orthorhombic periodic grid as one CUDA kernel.

:class:`.device_bloch.DeviceBlochOperator` applies ``-nabla^2 - 2i k.grad + V + |k|^2`` to a complex
block with two sparse products on its real view (the Laplacian and the pre-summed ``k.grad``, cuSPARSE)
and three elementwise passes.  On an orthorhombic periodic grid both matrices are the same axial stencil
at every point (the neighbours wrap around the cell), so the whole local term is

``(H u)_p = (c_0 + v_p) u_p + sum_a sum_{s=1..M} [ c_{a,s} (u_{p+s e_a} + u_{p-s e_a})
                                                 - 2i k_a g_{a,s} (u_{p+s e_a} - u_{p-s e_a}) ]``

with ``c`` the coefficients of ``-nabla^2``, ``g`` those of the first derivative (antisymmetric) and
``v = V + |k|^2``.  One kernel reads each neighbour once per output element and writes the result
directly: on an A100 several times the throughput of the sparse route.  The same kernel takes one step
of the Chebyshev recurrence of the filters, ``Y = (((H X - c X) a) - b_j X_previous) d_j`` with the
nonlocal part of ``H X`` computed before (:meth:`BlochStencil.apply` with ``added``, ``previous``,
``center``, ``scale``, ``before``, ``after``), which saves the five elementwise passes of the recurrence.  The coefficients are read
from the matrices themselves (row 0), and the matrices are rebuilt from them and compared entry by
entry before the kernel is used (:meth:`BlochStencil.from_matrices` returns ``None`` otherwise: a
general or oblique grid, a confined grid, any matrix that is not this stencil).  The sums run in another
order than the sparse products, so results agree to round-off.

The grid index is ``(x n_1 + y) n_2 + z`` (:class:`parsec_python.Grid.pbc.PeriodicRealSpaceGrid`); the
block is C-ordered ``(points, columns)`` complex128.  A thread per ``(y, z, column)``, consecutive threads on
consecutive columns, marches along x with the neighbours along x in registers.  On an A100 (64 columns,
Expansion_Order 8, ``acceleration/benchmarks/bloch_stencil_speed.py``): 11.6, 11.1 and 9.8 G grid-point
columns per second at 72^3, 108^3 and 144^3 points, against 1.0 for the sparse products and 9.7, 9.7, 9.5
for a kernel of a thread per element (2026-11-08); a recurrence step 10-15 % less.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
import scipy.sparse as sp

_SOURCE = r"""
#include <cuComplex.h>
// A thread per (y, z, column) marching along x: the neighbours along x pass through a window of
// registers, so that each element is read once from memory for them; the neighbours along y and z come
// from the caches.
extern "C" __global__ void bloch_stencil_march(
    const double2* __restrict__ u,
    double2* __restrict__ out,
    const double* __restrict__ diagonal,
    const double* __restrict__ laplacian,
    const double* __restrict__ gradient,
    const int n0, const int n1, const int n2,
    const long long columns,
    const long long lanes,
    const double2* __restrict__ nonlocal,
    const int has_nonlocal,
    const double2* __restrict__ previous,
    const int has_previous,
    const double center,
    const double scale,
    const double* __restrict__ before,
    const double* __restrict__ after,
    const int has_after)
{
    const long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    const long long column = lane % columns;
    const long long yz = lane / columns;
    const int z = (int)(yz % n2);
    const int y = (int)(yz / n2);
    const long long plane = (long long)n1 * n2;
    double c0[HALF_WIDTH], q0[HALF_WIDTH], c1[HALF_WIDTH], q1[HALF_WIDTH], c2[HALF_WIDTH], q2[HALF_WIDTH];
    long long dy_plus[HALF_WIDTH], dy_minus[HALF_WIDTH], dz_plus[HALF_WIDTH], dz_minus[HALF_WIDTH];
#pragma unroll
    for (int s = 1; s <= HALF_WIDTH; ++s) {
        c0[s - 1] = laplacian[s - 1];
        q0[s - 1] = gradient[s - 1];
        c1[s - 1] = laplacian[HALF_WIDTH + s - 1];
        q1[s - 1] = gradient[HALF_WIDTH + s - 1];
        c2[s - 1] = laplacian[2 * HALF_WIDTH + s - 1];
        q2[s - 1] = gradient[2 * HALF_WIDTH + s - 1];
        int yp = y + s; if (yp >= n1) yp -= n1;
        int ym = y - s; if (ym < 0) ym += n1;
        int zp = z + s; if (zp >= n2) zp -= n2;
        int zm = z - s; if (zm < 0) zm += n2;
        dy_plus[s - 1] = (long long)(yp - y) * n2 * columns;
        dy_minus[s - 1] = (long long)(ym - y) * n2 * columns;
        dz_plus[s - 1] = (long long)(zp - z) * columns;
        dz_minus[s - 1] = (long long)(zm - z) * columns;
    }
    double2 window[2 * HALF_WIDTH + 1];
#pragma unroll
    for (int s = -HALF_WIDTH; s <= HALF_WIDTH; ++s) {
        int xs = s < 0 ? s + n0 : s;
        window[s + HALF_WIDTH] = u[((long long)xs * plane + yz) * columns + column];
    }
    for (int x = 0; x < n0; ++x) {
        const long long point = (long long)x * plane + yz;
        const long long index = point * columns + column;
        const double2 centre = window[HALF_WIDTH];
        const double d = diagonal[point];
        double re = d * centre.x;
        double im = d * centre.y;
#pragma unroll
        for (int s = 1; s <= HALF_WIDTH; ++s) {
            {
                const double2 a = window[HALF_WIDTH + s];
                const double2 b = window[HALF_WIDTH - s];
                re += c0[s - 1] * (a.x + b.x) + q0[s - 1] * (a.y - b.y);
                im += c0[s - 1] * (a.y + b.y) - q0[s - 1] * (a.x - b.x);
            }
            {
                const double2 a = u[index + dy_plus[s - 1]];
                const double2 b = u[index + dy_minus[s - 1]];
                re += c1[s - 1] * (a.x + b.x) + q1[s - 1] * (a.y - b.y);
                im += c1[s - 1] * (a.y + b.y) - q1[s - 1] * (a.x - b.x);
            }
            {
                const double2 a = u[index + dz_plus[s - 1]];
                const double2 b = u[index + dz_minus[s - 1]];
                re += c2[s - 1] * (a.x + b.x) + q2[s - 1] * (a.y - b.y);
                im += c2[s - 1] * (a.y + b.y) - q2[s - 1] * (a.x - b.x);
            }
        }
        if (has_nonlocal) {
            const double2 extra = nonlocal[index];
            re += extra.x;
            im += extra.y;
        }
        re = (re - center * centre.x) * scale;
        im = (im - center * centre.y) * scale;
        if (has_previous) {
            const double2 p = previous[index];
            const double b = before[column];
            re -= b * p.x;
            im -= b * p.y;
        }
        if (has_after) {
            const double f = after[column];
            re *= f;
            im *= f;
        }
        out[index] = make_double2(re, im);
#pragma unroll
        for (int s = 0; s < 2 * HALF_WIDTH; ++s) {
            window[s] = window[s + 1];
        }
        if (x + 1 < n0) {
            int xn = x + 1 + HALF_WIDTH;
            if (xn >= n0) xn -= n0;
            window[2 * HALF_WIDTH] = u[((long long)xn * plane + yz) * columns + column];
        }
    }
}
"""

_THREADS = 256
_KERNELS: dict[tuple[int, int], Any] = {}
_LOCK = Lock()


def _kernel(cp, half_width: int):
    key = (int(cp.cuda.Device().id), int(half_width))
    with _LOCK:
        kernel = _KERNELS.get(key)
        if kernel is None:
            module = cp.RawModule(code=_SOURCE.replace("HALF_WIDTH", str(int(half_width))))
            kernel = module.get_function("bloch_stencil_march")
            _KERNELS[key] = kernel
        return kernel


def grid_shape_of(gradient, size: int) -> tuple[int, int, int] | None:
    """``(n0, n1, n2)`` of a periodic grid from its first derivatives, ``None`` where they do not tell.

    Row 0 of ``d/dx_2`` reaches back to ``z = n2 - 1`` (index ``n2 - 1``), of ``d/dx_1`` to
    ``y = n1 - 1`` (index ``(n1 - 1) n2``) and of ``d/dx_0`` to ``x = n0 - 1``; the stencils are checked
    against the matrices afterwards (:meth:`AxialStencils.from_matrices`).
    """
    reach = []
    for matrix in gradient:
        csr = sp.csr_matrix(matrix)
        columns = csr.indices[csr.indptr[0] : csr.indptr[1]]
        if columns.size == 0:
            return None
        reach.append(int(columns.max()))
    n2 = reach[2] + 1
    if reach[1] % n2:
        return None
    n1 = reach[1] // n2 + 1
    if reach[0] % (n1 * n2):
        return None
    n0 = reach[0] // (n1 * n2) + 1
    if n0 * n1 * n2 != int(size):
        return None
    return n0, n1, n2


def _offsets(columns: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Signed axial offsets ``(n, 3)`` of the columns of row 0 (index ``(x n1 + y) n2 + z``)."""
    coordinates = np.stack(np.unravel_index(columns, shape), axis=1)
    sizes = np.asarray(shape)
    return np.where(coordinates > sizes // 2, coordinates - sizes, coordinates)


def _axial_stencil(row_columns, row_values, shape, half_width: int, symmetric: bool):
    """``(centre, coefficients[3, M])`` of an axial stencil from row 0, or ``None`` if it is not one."""
    offsets = _offsets(np.asarray(row_columns), shape)
    centre = 0.0
    coefficients = np.zeros((3, half_width))
    seen = np.zeros((3, 2 * half_width + 1), dtype=bool)
    for offset, value in zip(offsets, np.asarray(row_values, dtype=np.float64)):
        moving = np.flatnonzero(offset)
        if moving.size == 0:
            centre = float(value)
            continue
        if moving.size != 1:
            return None
        axis = int(moving[0])
        shift = int(offset[axis])
        if abs(shift) > half_width or seen[axis, shift + half_width]:
            return None
        seen[axis, shift + half_width] = True
        if shift > 0:
            coefficients[axis, shift - 1] = value
    for offset, value in zip(offsets, np.asarray(row_values, dtype=np.float64)):
        moving = np.flatnonzero(offset)
        if moving.size != 1:
            continue
        axis = int(moving[0])
        shift = int(offset[axis])
        if shift < 0:
            expected = coefficients[axis, -shift - 1]
            if (value != expected) if symmetric else (value != -expected):
                return None
    return centre, coefficients


def _rebuilt(shape, half_width: int, centre: float, coefficients: np.ndarray, symmetric: bool) -> sp.csr_matrix:
    """The matrix of the axial stencil on the periodic grid of ``shape`` (to compare with the given one)."""
    n0, n1, n2 = shape
    size = n0 * n1 * n2
    points = np.arange(size, dtype=np.int64)
    x, rest = np.divmod(points, n1 * n2)
    y, z = np.divmod(rest, n2)
    rows, cols, values = [], [], []
    if centre != 0.0:
        rows.append(points)
        cols.append(points)
        values.append(np.full(size, centre))
    for axis, (coordinate, count, stride) in enumerate(((x, n0, n1 * n2), (y, n1, n2), (z, n2, 1))):
        for shift in range(1, half_width + 1):
            value = coefficients[axis, shift - 1]
            if value == 0.0:
                continue
            for sign in (1, -1):
                moved = (coordinate + sign * shift) % count
                rows.append(points)
                cols.append(points + (moved - coordinate) * stride)
                values.append(np.full(size, value if (symmetric or sign > 0) else -value))
    if not rows:
        return sp.csr_matrix((size, size))
    matrix = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(size, size)
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def _same(given: sp.spmatrix, rebuilt: sp.csr_matrix) -> bool:
    given = sp.csr_matrix(given, dtype=np.float64)
    given.sum_duplicates()
    given.eliminate_zeros()
    if given.shape != rebuilt.shape or given.nnz != rebuilt.nnz:
        return False
    difference = (given - rebuilt).tocsr()
    difference.eliminate_zeros()
    return difference.nnz == 0


@dataclass(frozen=True)
class AxialStencils:
    """The ``-nabla^2`` and the three first derivatives of an orthorhombic periodic grid as stencils."""

    shape: tuple[int, int, int]
    half_width: int
    centre: float  # the diagonal of -nabla^2
    laplacian: np.ndarray  # (3, M): c_{a,s}
    gradient: np.ndarray  # (3, M): g_{a,s} of d/dx_a at +s

    @classmethod
    def from_matrices(cls, shape, laplacian: sp.spmatrix, gradient) -> "AxialStencils | None":
        """The stencils of these matrices, or ``None`` where they are not axial stencils of ``shape``."""
        shape = tuple(int(value) for value in shape)
        size = int(np.prod(shape))
        if laplacian.shape != (size, size) or len(gradient) != 3 or min(shape) < 3:
            return None
        lap = sp.csr_matrix(laplacian, dtype=np.float64)
        row = slice(lap.indptr[0], lap.indptr[1])
        moving = _offsets(lap.indices[row], shape)
        half_width = int(np.max(np.abs(moving))) if moving.size else 0
        if half_width < 1 or 2 * half_width >= min(shape):
            return None  # a stencil that wraps onto itself: leave it to the matrices
        found = _axial_stencil(lap.indices[row], lap.data[row], shape, half_width, symmetric=True)
        if found is None:
            return None
        centre, laplacian_coefficients = found
        gradient_coefficients = np.zeros((3, half_width))
        for axis, matrix in enumerate(gradient):
            g = sp.csr_matrix(matrix, dtype=np.float64)
            grow = slice(g.indptr[0], g.indptr[1])
            stencil = _axial_stencil(g.indices[grow], g.data[grow], shape, half_width, symmetric=False)
            if stencil is None or stencil[0] != 0.0:
                return None
            other = np.delete(stencil[1], axis, axis=0)
            if np.any(other != 0.0):
                return None  # a derivative along axis a with entries along another axis
            gradient_coefficients[axis] = stencil[1][axis]
        if not _same(lap, _rebuilt(shape, half_width, centre, laplacian_coefficients, True)):
            return None
        for axis, matrix in enumerate(gradient):
            only = np.zeros((3, half_width))
            only[axis] = gradient_coefficients[axis]
            if not _same(matrix, _rebuilt(shape, half_width, 0.0, only, False)):
                return None
        return cls(shape, half_width, float(centre), laplacian_coefficients, gradient_coefficients)


class BlochStencil:
    """The local term of ``H_k`` on one device: the kernel and its coefficients for one k-point."""

    def __init__(self, cp, stencils: AxialStencils, k_point) -> None:
        self.cp = cp
        self.stencils = stencils
        k = np.asarray(k_point if k_point is not None else np.zeros(3), dtype=np.float64).reshape(3)
        # -2i k_a g_{a,s} (u+ - u-): q_{a,s} = 2 k_a g_{a,s} enters as re += q di, im -= q dr
        self.laplacian = cp.asarray(np.ascontiguousarray(stencils.laplacian.reshape(-1)))
        self.gradient = cp.asarray(np.ascontiguousarray((2.0 * k[:, None] * stencils.gradient).reshape(-1)))
        self.kernel = _kernel(cp, stencils.half_width)

    @property
    def device_bytes(self) -> int:
        return int(self.laplacian.nbytes + self.gradient.nbytes)

    def diagonal(self, diagonal):
        """``c_0 + v``: the diagonal the kernel adds (``v = V + |k|^2`` of the operator)."""
        return diagonal + self.stencils.centre

    def apply(
        self,
        block,
        diagonal_with_centre,
        *,
        added=None,
        previous=None,
        center: float = 0.0,
        scale: float = 1.0,
        before=None,
        after=None,
    ):
        """``(-nabla^2 - 2i k.grad + v) block`` for a C-ordered complex128 block ``(points, columns)``.

        With ``added`` (an array of the block's shape added to the local term), ``previous``,
        ``before`` and ``after`` (one value per column) the step of the Chebyshev recurrence
        ``(((L block + added) - center block) scale - before previous) after``; without them
        ``center = 0`` and ``scale = 1`` leave the local term exactly.
        """
        cp = self.cp
        if block.dtype != cp.complex128 or not block.flags.c_contiguous:
            block = cp.ascontiguousarray(block, dtype=cp.complex128)
        out = cp.empty_like(block)
        points, columns = int(block.shape[0]), int(block.shape[1])
        total = points * columns
        if total == 0:
            return out
        if previous is not None and before is None:
            raise ValueError("a recurrence step with a previous block needs its coefficients")

        def same_shape(array):
            if array is None:
                return block, np.int32(0)
            array = cp.ascontiguousarray(array, dtype=cp.complex128)
            if tuple(array.shape) != tuple(block.shape):
                raise ValueError("the arrays of a recurrence step must have the shape of the block")
            return array, np.int32(1)

        def per_column(values):
            if values is None:
                return self.laplacian, np.int32(0)
            values = cp.ascontiguousarray(cp.asarray(values, dtype=cp.float64).reshape(-1))
            if values.size != columns:
                raise ValueError("one coefficient per column is needed")
            return values, np.int32(1)

        nonlocal_part, has_nonlocal = same_shape(added)
        previous_part, has_previous = same_shape(previous)
        before_part, _ = per_column(before if previous is not None else None)
        after_part, has_after = per_column(after)
        n0, n1, n2 = self.stencils.shape
        work = n1 * n2 * columns  # a thread per (y, z, column), marching along x
        blocks = (work + _THREADS - 1) // _THREADS
        self.kernel(
            (blocks,),
            (_THREADS,),
            (
                block,
                out,
                diagonal_with_centre,
                self.laplacian,
                self.gradient,
                np.int32(n0),
                np.int32(n1),
                np.int32(n2),
                np.int64(columns),
                np.int64(work),
                nonlocal_part,
                has_nonlocal,
                previous_part,
                has_previous,
                np.float64(center),
                np.float64(scale),
                before_part,
                after_part,
                has_after,
            ),
        )
        return out


__all__ = ["AxialStencils", "BlochStencil", "grid_shape_of"]
