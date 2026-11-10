"""The kinetic-energy density of Bloch factors on the device where they are.

The device counterpart of :func:`parsec_python.SCF.meta_gga_periodic.bloch_orbital_fields`
for the k-point workers of :mod:`parsec_python.acceleration.SCF.kpoint_parallel`
(the job's ``accumulate`` runs with ``xp`` = CuPy on the device of the
worker's orbitals):

``tau = scale * sum_n w_n sum_c |(G_c + i k_c) u_n|^2``   and, in the orbital
form of the density gradient,
``g_c = 4 scale * sum_n w_n Re[conj(u_n) (G_c u_n)]``,

with ``G_c`` the shared first-derivative matrices of the run (uploaded once
per device and kept while the host matrices exist) applied to the real view
of the complex block.  Hartree units as the host function; nothing here
prints or opens files.
"""

from __future__ import annotations

import threading

import numpy as np

_UPLOADS: list[tuple[int, object, object]] = []  # (device id, host matrix, device matrix)
_LOCK = threading.Lock()
_KEPT = 12  # four runs of three matrices on a device


def _device_matrix(xp, matrix):
    device_id = int(xp.cuda.Device().id)
    with _LOCK:
        for known_device, host, device in _UPLOADS:
            if known_device == device_id and host is matrix:
                return device
    import scipy.sparse as sp
    from cupyx.scipy import sparse as cpsparse

    device = cpsparse.csr_matrix(sp.csr_matrix(matrix, dtype=np.float64))
    with _LOCK:
        _UPLOADS.insert(0, (device_id, matrix, device))
        del _UPLOADS[_KEPT:]
    return device


def device_bloch_fields(xp, gradient, vectors, weights, k_point, scale: float, *, with_gradient: bool = False):
    """``scale * sum_n w_n |(G + ik) u_n|^2`` of a device block (grid x states), and with ``with_gradient``
    also ``g`` (3 x grid, ``4 scale sum_n w_n Re[conj(u_n) G u_n]``): device arrays."""

    host_weights = np.asarray(weights, dtype=np.float64)
    size = int(vectors.shape[0])
    tau = xp.zeros(size, dtype=xp.float64)
    density_gradient = xp.zeros((3, size), dtype=xp.float64) if with_gradient else None
    selected = np.flatnonzero(host_weights != 0.0)
    if selected.size == 0:
        return (tau, density_gradient) if with_gradient else tau
    if selected.size == host_weights.size:
        block = vectors
    elif np.array_equal(selected, np.arange(selected.size)):
        block = vectors[:, : selected.size]
    else:
        block = vectors[:, xp.asarray(selected)]
    block = xp.ascontiguousarray(block, dtype=xp.complex128)
    occupied = xp.asarray(host_weights[selected])
    k = np.zeros(3) if k_point is None else np.asarray(k_point, dtype=np.float64).reshape(3)
    for axis, matrix in enumerate(gradient):
        device = _device_matrix(xp, matrix)
        derivative = xp.ascontiguousarray(device @ block.view(xp.float64)).view(xp.complex128)
        if with_gradient:
            density_gradient[axis] = (
                block.real * derivative.real + block.imag * derivative.imag
            ) @ occupied
        if k[axis] != 0.0:
            derivative += (1j * float(k[axis])) * block
        tau += (derivative.real * derivative.real + derivative.imag * derivative.imag) @ occupied
    tau *= float(scale)
    if not with_gradient:
        return tau
    density_gradient *= 4.0 * float(scale)
    return tau, density_gradient


# ----- the pointwise part of the orbital terms of a Bloch Hamiltonian -----------------------

_POINTWISE_SOURCE = r"""
const double* m = &mix[0];
complex<double> d0 = m[0] * a0 + m[1] * a1 + m[2] * a2;
complex<double> d1 = m[3] * a0 + m[4] * a1 + m[5] * a2;
complex<double> d2 = m[6] * a0 + m[7] * a1 + m[8] * a2;
const complex<double> ik0(0.0, k0), ik1(0.0, k1), ik2(0.0, k2);
complex<double> f0 = v * (d0 + ik0 * u);
complex<double> f1 = v * (d1 + ik1 * u);
complex<double> f2 = v * (d2 + ik2 * u);
local = -(ik0 * f0 + ik1 * f1 + ik2 * f2) + 2.0 * (w0 * d0 + w1 * d1 + w2 * d2);
f0 += 2.0 * w0 * u;
f1 += 2.0 * w1 * u;
f2 += 2.0 * w2 * u;
b0 = m[0] * f0 + m[3] * f1 + m[6] * f2;
b1 = m[1] * f0 + m[4] * f1 + m[7] * f2;
b2 = m[2] * f0 + m[5] * f1 + m[8] * f2;
"""
_POINTWISE_KERNELS: dict = {}


def _pointwise_kernel(cp):
    kernel = _POINTWISE_KERNELS.get("kernel")
    if kernel is None:
        kernel = cp.ElementwiseKernel(
            "complex128 a0, complex128 a1, complex128 a2, complex128 u, float64 v, float64 w0, float64 w1, "
            "float64 w2, raw float64 mix, float64 k0, float64 k1, float64 k2",
            "complex128 b0, complex128 b1, complex128 b2, complex128 local",
            _POINTWISE_SOURCE,
            "meta_gga_bloch_pointwise",
            options=("--fmad=false",),
        )
        _POINTWISE_KERNELS["kernel"] = kernel
    return kernel


def pointwise_terms(xp, along, block, vtau, fluxes, mix, k_point, device_mix=None):
    """The pointwise part of the meta-GGA terms of a Bloch Hamiltonian: ``(b, local)``.

    ``along`` are ``L_i u`` (the derivatives along the grid directions, ``G_c
    = sum_i mix[c, i] L_i``; ``mix`` ``None`` is the identity), ``block`` is
    ``u``.  With ``d_c = G_c u`` and ``f_c = vtau (d_c + i k_c u)``:
    ``local = sum_c [-i k_c f_c + 2 W_c d_c]`` and
    ``b_i = sum_c mix[c, i] (f_c + 2 W_c u)``, so that the term is
    ``local + sum_i L_i^T b_i``.  One kernel on CuPy arrays; NumPy for the
    host double.  ``device_mix`` is ``mix`` (row-major, 9 values) already on the device.
    """

    k = np.zeros(3) if k_point is None else np.asarray(k_point, dtype=np.float64).reshape(3)
    matrix = np.eye(3) if mix is None else np.asarray(mix, dtype=np.float64)
    weights = vtau[:, None]
    if xp is np:
        derivatives = [sum(matrix[c, i] * along[i] for i in range(3)) for c in range(3)]
        local = np.zeros_like(block)
        outgoing = [np.zeros_like(block) for _ in range(3)]
        for c in range(3):
            flux = weights * (derivatives[c] + (1j * k[c]) * block)
            local -= (1j * k[c]) * flux
            if fluxes is not None:
                local += (2.0 * fluxes[c][:, None]) * derivatives[c]
                flux = flux + (2.0 * fluxes[c][:, None]) * block
            for i in range(3):
                outgoing[i] += matrix[c, i] * flux
        return outgoing, local
    shape = block.shape
    outgoing = [xp.empty(shape, dtype=xp.complex128) for _ in range(3)]
    local = xp.empty(shape, dtype=xp.complex128)
    if fluxes is None:
        w = (0.0, 0.0, 0.0)
    else:
        w = tuple(flux[:, None] for flux in fluxes)
    if device_mix is None:
        device_mix = xp.asarray(matrix.reshape(-1))
    _pointwise_kernel(xp)(
        along[0], along[1], along[2], block, weights, w[0], w[1], w[2], device_mix,
        float(k[0]), float(k[1]), float(k[2]),
        outgoing[0], outgoing[1], outgoing[2], local,
    )
    return outgoing, local


__all__ = ["device_bloch_fields", "pointwise_terms"]
