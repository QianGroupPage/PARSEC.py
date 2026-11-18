"""The Ewald ion-ion sums of :func:`parsec_python.V_ion.ewald.ewald_ion_ion_terms` on a device.

The host evaluates the real-space pairs from a k-d tree on one core and the structure factors with NumPy: 16.9 s
of the 70 s before the SCF of the 64,000 atoms of a 20^3 diamond cell on an A100 node (7.4 s each, 2026-11-17),
and again at every step of a relaxation (the forces).  Here:

* the real-space sum: the images of the host (:func:`parsec_python.V_ion.ewald.candidate_images`) binned into
  cells of at least the cutoff; one thread per atom adds the terms of the images within the cutoff in the 27 cells
  around its own (a fixed order: the same sum at every call);
* the reciprocal sum: the host's block products (phases of one axis times the products of the other two) with
  CuPy, the energy and the forces of the same half box.

The terms are the host's (the same ``eta``, cutoff, images and reciprocal vectors), the sums in another order:
the energy and the forces agree to round-off.  ``PARSEC_DEVICE_EWALD=off`` (or ``PARSEC_DEVICE=cpu``, no CuPy, a
cell of fewer than :data:`MINIMUM_ATOMS` atoms) keeps the host sums, bit for bit.
"""

from __future__ import annotations

import os

import numpy as np

from parsec_python.V_ion import ewald as _ewald

EWALD_ENV = "PARSEC_DEVICE_EWALD"
# below this many atoms the host sums take well under a second and a kernel's compilation would cost more
MINIMUM_ATOMS = 1024
# cells per axis of the real-space bins at most (larger cells than the cutoff are as correct)
_MAXIMUM_CELLS = 256
_THREADS = 128


def device_ewald_setting() -> str:
    """``PARSEC_DEVICE_EWALD``: ``auto`` (default: the device sums for cells of :data:`MINIMUM_ATOMS` atoms or
    more where a device is usable) or ``off`` (the host sums)."""

    value = os.environ.get(EWALD_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{EWALD_ENV} must be auto or off, got {value!r}")
    return value


_REAL_SPACE_KERNEL = r"""
extern "C" __global__ void ewald_real_space(
    const long long count, const double* __restrict__ positions, const double* __restrict__ charges,
    const double* __restrict__ images, const long long* __restrict__ image_atoms,
    const long long* __restrict__ image_rows, const long long* __restrict__ start,
    const int nx, const int ny, const int nz, const double ox, const double oy, const double oz,
    const double cell, const double cutoff, const double eta, const int forces,
    double* __restrict__ energy, double* __restrict__ force, int* __restrict__ error)
{
    const long long a = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= count) return;
    const double x = positions[3 * a], y = positions[3 * a + 1], z = positions[3 * a + 2];
    const int cx = min(max((int)floor((x - ox) / cell), 0), nx - 1);
    const int cy = min(max((int)floor((y - oy) / cell), 0), ny - 1);
    const int cz = min(max((int)floor((z - oz) / cell), 0), nz - 1);
    const double qa = charges[a];
    const double root_pi = 1.7724538509055160273;
    double e = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
    for (int i = max(cx - 1, 0); i <= min(cx + 1, nx - 1); ++i) {
        for (int j = max(cy - 1, 0); j <= min(cy + 1, ny - 1); ++j) {
            for (int k = max(cz - 1, 0); k <= min(cz + 1, nz - 1); ++k) {
                const long long c = ((long long)i * ny + j) * nz + k;
                for (long long p = start[c]; p < start[c + 1]; ++p) {
                    if (image_rows[p] == a) continue;  /* the atom itself (n = 0) */
                    const double rx = x - images[3 * p];
                    const double ry = y - images[3 * p + 1];
                    const double rz = z - images[3 * p + 2];
                    const double d = sqrt(rx * rx + ry * ry + rz * rz);
                    if (d > cutoff) continue;
                    if (d <= 0.0) { atomicExch(error, 1); continue; }
                    const double qq = qa * charges[image_atoms[p]];
                    const double s = erfc(eta * d);
                    e += qq * s / d;
                    if (forces) {
                        const double radial =
                            qq * (s / (d * d) + 2.0 * eta * exp(-(eta * d) * (eta * d)) / (root_pi * d)) / d;
                        fx += radial * rx;
                        fy += radial * ry;
                        fz += radial * rz;
                    }
                }
            }
        }
    }
    energy[a] = e;
    if (forces) {
        force[3 * a] = fx;
        force[3 * a + 1] = fy;
        force[3 * a + 2] = fz;
    }
}
"""

_COMPILED: dict[int, object] = {}


def _kernel(cp):
    device = int(cp.cuda.Device().id)
    if device not in _COMPILED:
        kernel = cp.RawKernel(_REAL_SPACE_KERNEL, "ewald_real_space", options=("--std=c++11",))
        kernel.compile()
        _COMPILED[device] = kernel
    return _COMPILED[device]


def _bins(image_positions: np.ndarray, cutoff: float):
    """``(origin, cell, shape, order, start)``: the images sorted by cells of side ``cell >= cutoff``."""

    lows = image_positions.min(axis=0)
    highs = image_positions.max(axis=0)
    extent = float(np.max(highs - lows))
    cell = max(float(cutoff) * (1.0 + 1.0e-9), extent / _MAXIMUM_CELLS, 1.0e-6)
    origin = lows - 1.0e-9 * max(cell, 1.0)
    shape = np.maximum(np.ceil((highs - origin) / cell).astype(np.int64), 1)
    cells = np.clip(np.floor((image_positions - origin) / cell).astype(np.int64), 0, shape - 1)
    flat = (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2]
    order = np.argsort(flat, kind="stable")
    start = np.zeros(int(np.prod(shape)) + 1, dtype=np.int64)
    np.cumsum(np.bincount(flat, minlength=int(np.prod(shape))), out=start[1:])
    return origin, cell, shape, order, start


class DeviceEwaldTerms:
    """The two sums of :func:`parsec_python.V_ion.ewald.set_ion_ion_device`."""

    def _cupy(self, count: int):
        if device_ewald_setting() == "off" or count < MINIMUM_ATOMS:
            return None
        from .periodic_device import _cupy

        return _cupy()

    def real_space(self, positions, fractional, lattice, charges, cutoff, eta, forces):
        count = int(len(positions))
        cp = self._cupy(count)
        if cp is None:
            return None
        from .periodic_device import device_id

        image_positions, image_atoms, _ = _ewald.candidate_images(positions, fractional, lattice, cutoff)
        origin, cell, shape, order, start = _bins(image_positions, cutoff)
        with cp.cuda.Device(device_id(cp)):
            energy = cp.zeros(count, dtype=cp.float64)
            force = cp.zeros((count, 3) if forces else 1, dtype=cp.float64)
            error = cp.zeros(1, dtype=cp.int32)
            _kernel(cp)(
                ((count + _THREADS - 1) // _THREADS,),
                (_THREADS,),
                (
                    np.int64(count),
                    cp.asarray(np.ascontiguousarray(positions, dtype=np.float64)),
                    cp.asarray(np.ascontiguousarray(charges, dtype=np.float64)),
                    cp.asarray(np.ascontiguousarray(image_positions[order], dtype=np.float64)),
                    cp.asarray(np.asarray(image_atoms, dtype=np.int64)[order]),
                    cp.asarray(order.astype(np.int64)),  # the row of every sorted image (rows < N: the atoms)
                    cp.asarray(start),
                    np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
                    np.float64(origin[0]), np.float64(origin[1]), np.float64(origin[2]),
                    np.float64(cell), np.float64(cutoff), np.float64(eta), np.int32(1 if forces else 0),
                    energy, force, error,
                ),
            )
            if int(error.get()[0]):
                raise ValueError("two nuclei occupy the same position")
            total = 0.5 * float(cp.sum(energy).get())
            found = cp.asnumpy(force) if forces else None
        return total, found

    def reciprocal(self, fractional, charges, frequencies, weight, vectors, volume, forces):
        count = int(len(charges))
        cp = self._cupy(count)
        if cp is None:
            return None
        from .periodic_device import device_id

        m0, m1, m2 = (len(values) for values in frequencies)
        with cp.cuda.Device(device_id(cp)):
            ordered = cp.asarray(np.ascontiguousarray(fractional, dtype=np.float64))
            charge = cp.asarray(np.asarray(charges, dtype=np.float64))
            frequency = [cp.asarray(np.asarray(values, dtype=np.float64)) for values in frequencies]

            def phases(start, stop, axis):
                # exp(2 pi i f m) of the atoms (rows) and the frequencies of one axis
                return cp.exp(2.0j * np.pi * (ordered[start:stop, axis][:, None] * frequency[axis][None, :]))

            block = max(1, _ewald._PHASE_ELEMENTS // (m1 * m2))
            structure = cp.zeros((m0, m1 * m2), dtype=cp.complex128)
            for start in range(0, count, block):
                stop = min(start + block, count)
                first = charge[start:stop, None] * phases(start, stop, 0)
                product = (phases(start, stop, 1)[:, :, None] * phases(start, stop, 2)[:, None, :]).reshape(
                    stop - start, m1 * m2
                )
                structure += first.T @ product
                del first, product
            structure = structure.reshape(m0, m1, m2)
            weights = cp.asarray(np.asarray(weight, dtype=np.float64))
            energy = (2.0 * np.pi / volume) * float(
                cp.sum(weights * (structure.real**2 + structure.imag**2)).get()
            )
            found = None
            if forces:
                # (4 pi/V) Z_a sum_G w(G) G Im[exp(i G.r_a) S(G)^*]: the contraction of the host, axis after axis
                coefficients = cp.stack(
                    [weights * cp.asarray(np.asarray(vectors(component), dtype=np.float64)) for component in range(3)]
                ) * cp.conj(structure)[None]
                coefficients = coefficients.reshape(3 * m0 * m1, m2)
                part = cp.zeros((count, 3), dtype=cp.float64)
                block = max(1, _ewald._PHASE_ELEMENTS // (3 * m0 * m1))
                for start in range(0, count, block):
                    stop = min(start + block, count)
                    last = coefficients @ phases(start, stop, 2).T
                    middle = cp.einsum("kxyb,by->kxb", last.reshape(3, m0, m1, stop - start), phases(start, stop, 1))
                    total = cp.einsum("kxb,bx->bk", middle, phases(start, stop, 0))
                    part[start:stop] = (4.0 * np.pi / volume) * charge[start:stop, None] * total.imag
                    del last, middle, total
                found = cp.asnumpy(part)
                del coefficients, part
            del structure, weights, ordered
        return energy, found


__all__ = ["DeviceEwaldTerms", "EWALD_ENV", "MINIMUM_ATOMS", "device_ewald_setting"]
