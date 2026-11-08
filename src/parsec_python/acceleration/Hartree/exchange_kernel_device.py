"""The periodic exchange kernel ``K(q + G)`` built on a CUDA device in one launch (CuPy, float64).

The performance counterpart of :meth:`parsec_python.Hartree.exchange_kernel.PeriodicExchangeKernel.values` for a
bulk cell: the same formula, element by element, in one CUDA kernel.  Each thread takes one point of the FFT grid,
its ``fftfreq`` index ``m``, the mean of the kernel over the Nyquist aliases ``m_a + n_a`` of the axes where it
sits on the index ``n_a/2`` of an even axis, the x_gamma weights of the components that carry them, and the element
``K_0`` where ``q + G = 0``.  The scalars (the reciprocal metric, ``K_0``, the weights' factor) come from the host
object, which computed them in its constructor.

Why: the array module path ``values(q, xp=cupy)`` launches some 250 small kernels (6-7 ms per ``q`` on an A100
for any grid size, plus CuPy's compilation of each of them in a fresh process), while a pair solve of a 48^3
cell takes 0.1 ms; with up to ``(2N - 1)^3`` distinct ``q = k - k'`` a Fock loop can neither rebuild per block
nor always cache them all.  One launch builds the array in about the time of a copy.

A padded slab (the default kernel of a slab) is a bulk kernel on the padded box and is built here as well; the
truncated slab kernel needs a transform along ``z`` of the sheets of the erf part and goes through
``values(q, xp=cupy)`` (:func:`device_values` falls back to it).  The results equal the host values to round-off
(different summation order of the alias means and of ``exp``/``expm1`` on the device).
"""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np

from ..backends.cupy_compile import compile_cupy_raw

_CUDA_SOURCE = r"""
__device__ __forceinline__ bool on_double_grid(const double coordinate, const int count) {
    const double scaled = 0.5 * coordinate * (double)count;
    return fabs(scaled - rint(scaled)) <= 1.0e-7;
}

extern "C" __global__ void periodic_exchange_kernel(
    double* __restrict__ out,
    const int n0, const int n1, const int n2,
    const double f0, const double f1, const double f2,
    const double g00, const double g11, const double g22,
    const double g01, const double g02, const double g12,
    const double alpha, const double beta, const double four_omega_squared,
    const int weighted_long, const int weighted_short, const double factor,
    const int k0, const int k1, const int k2,
    const double element, const double threshold)
{
    const long long total = (long long)n0 * n1 * n2;
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int i2 = (int)(index % n2);
    const long long rest = index / n2;
    const int i1 = (int)(rest % n1);
    const int i0 = (int)(rest / n1);
    const int m0 = i0 < (n0 + 1) / 2 ? i0 : i0 - n0;
    const int m1 = i1 < (n1 + 1) / 2 ? i1 : i1 - n1;
    const int m2 = i2 < (n2 + 1) / 2 ? i2 : i2 - n2;
    const bool nyquist0 = (n0 % 2 == 0) && (i0 == n0 / 2);
    const bool nyquist1 = (n1 % 2 == 0) && (i1 == n1 / 2);
    const bool nyquist2 = (n2 % 2 == 0) && (i2 == n2 / 2);
    const bool weighted = weighted_long || weighted_short;
    const double four_pi = 4.0 * 3.14159265358979323846;
    double sum = 0.0;
    int count = 0;
    for (int combo = 0; combo < 8; ++combo) {
        const int c0 = combo & 1, c1 = (combo >> 1) & 1, c2 = (combo >> 2) & 1;
        if ((c0 && !nyquist0) || (c1 && !nyquist1) || (c2 && !nyquist2)) continue;
        const double x0 = f0 + (double)(m0 + c0 * n0);
        const double x1 = f1 + (double)(m1 + c1 * n1);
        const double x2 = f2 + (double)(m2 + c2 * n2);
        const double squared = g00 * x0 * x0 + g11 * x1 * x1 + g22 * x2 * x2
            + 2.0 * (g01 * x0 * x1 + g02 * x0 * x2 + g12 * x1 * x2);
        double value;
        if (squared <= threshold) {
            value = element;
        } else {
            double weight = 1.0;
            if (weighted) {
                const bool on = on_double_grid(x0, k0) && on_double_grid(x1, k1) && on_double_grid(x2, k2);
                weight = on ? 0.0 : factor;
            }
            double kernel;
            if (beta == 0.0) {
                kernel = (four_pi * alpha) / squared;
                if (weighted_long) kernel *= weight;
            } else {
                const double x = squared / four_omega_squared;
                if (weighted_long == weighted_short) {
                    kernel = (four_pi / squared) * (alpha * exp(-x) - (alpha + beta) * expm1(-x));
                    if (weighted_long) kernel *= weight;
                } else {
                    kernel = (four_pi / squared) * (alpha * weight - beta * expm1(-x));
                }
            }
            value = 2.0 * kernel;
        }
        sum += value;
        ++count;
    }
    out[index] = sum / (double)count;
}
"""

_KERNELS: dict[int, Any] = {}
_LOCK = Lock()
_THREADS = 256


def _compiled(cp) -> Any:
    device_id = int(cp.cuda.Device().id)
    with _LOCK:
        kernel = _KERNELS.get(device_id)
        if kernel is None:
            kernel = cp.RawKernel(_CUDA_SOURCE, "periodic_exchange_kernel", options=("--std=c++11",))
            compile_cupy_raw(kernel)
            _KERNELS[device_id] = kernel
    return kernel


def device_values(kernel, q, *, out=None):
    """``kernel.values(q)`` (Rydberg) as a CuPy float64 array on the current device.

    ``kernel`` is a :class:`parsec_python.Hartree.exchange_kernel.PeriodicExchangeKernel`; ``out`` (optional) a
    C-contiguous float64 CuPy array of ``kernel.shape`` to fill.  A truncated slab kernel goes through
    ``kernel.values(q, xp=cupy)``.
    """

    import cupy as cp

    if kernel.truncated:
        values = kernel.values(q, xp=cp)
        if out is None:
            return values
        out[...] = values
        return out
    fraction = kernel.mesh_fraction(q)
    shape = tuple(int(n) for n in kernel.shape)
    if out is None:
        out = cp.empty(shape, dtype=cp.float64)
    elif out.shape != shape or out.dtype != cp.float64 or not out.flags.c_contiguous:
        raise ValueError("out must be a C-contiguous float64 array of the kernel's shape")
    g = kernel.metric
    omega = kernel.omega or 1.0
    total = int(np.prod(shape))
    blocks = (total + _THREADS - 1) // _THREADS
    _compiled(cp)(
        (blocks,),
        (_THREADS,),
        (
            out,
            np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
            np.float64(fraction[0]), np.float64(fraction[1]), np.float64(fraction[2]),
            np.float64(g[0, 0]), np.float64(g[1, 1]), np.float64(g[2, 2]),
            np.float64(g[0, 1]), np.float64(g[0, 2]), np.float64(g[1, 2]),
            np.float64(kernel.alpha), np.float64(kernel.beta), np.float64(4.0 * omega * omega),
            np.int32(bool(kernel.weighted[0])), np.int32(bool(kernel.weighted[1])),
            np.float64(kernel.extrapolation_factor),
            np.int32(kernel.k_mesh[0]), np.int32(kernel.k_mesh[1]), np.int32(kernel.k_mesh[2]),
            np.float64(kernel.divergence_element), np.float64(kernel.singular_threshold),
        ),
    )
    return out


__all__ = ["device_values"]
