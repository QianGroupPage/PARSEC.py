"""FP64 CuPy Kronecker-product Poisson solver for batches of pair densities.

The device counterpart of :class:`parsec_python.Hartree.kronecker.KroneckerPoisson`.
The plan (one-dimensional eigenvectors, spectral factors, face correction
matrices, Gaussian factors and exterior tails of the screened kernels) is
built by the reference class on the host and uploaded once; every solve then
runs :func:`parsec_python.Hartree.kronecker.solve_block` with CuPy arrays, so
the device performs exactly the reference algorithm: batched GEMMs
(``cupy.matmul``, strided-batched cuBLAS) for the moments, the transforms
along z, y and x and the Gaussian smoothing, an elementwise spectral factor,
and gathers/scatters between the active points and the box.

Densities may be host or device arrays of shape ``(n_active, nb)``; the
potentials come back as the same kind of array.  A request is split into
batches whose work arrays fit ``memory_budget`` bytes (half the free device
memory by default).  CuPy is imported only when a solver is constructed.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from parsec_python.Hartree.kronecker import KroneckerPoisson, solve_block

from ..backends.cupy import require_cupy

# Arrays of the box size live at once in one batch (box, right-hand side,
# two transform temporaries, spectral coefficients, inverse transform): the
# measured peak of the erfc split path (memory pool, laptop) is 7.8.
WORK_ARRAYS_PER_DENSITY = 8
# Host arrays larger than this go to the device in slices: one transfer of
# the 2 GB exterior tails of a 170**3 box failed to get its pinned staging
# buffer on a Windows laptop (cudaErrorAlreadyMapped).
UPLOAD_CHUNK_BYTES = 1 << 28


def _upload(cp, array):
    """``cupy.asarray`` of a host array, in slices of the leading axis when it is large."""

    array = np.ascontiguousarray(array)
    if array.nbytes <= UPLOAD_CHUNK_BYTES or array.ndim == 0 or array.shape[0] < 2:
        return cp.asarray(array)
    result = cp.empty(array.shape, dtype=array.dtype)
    rows = max(1, int(UPLOAD_CHUNK_BYTES // max(1, array[0].nbytes)))
    for start in range(0, array.shape[0], rows):
        result[start : start + rows] = cp.asarray(array[start : start + rows])
    return result


class DeviceKroneckerPoisson:
    """Batched Kronecker-product Poisson solves on one CUDA device.

    ``source`` is a :class:`~parsec_python.Hartree.kronecker.KroneckerPoisson`
    to upload, or a cluster grid from which one is built with the remaining
    arguments (same meaning as there).  ``device_id`` selects the device
    (the current one by default).
    """

    def __init__(
        self,
        source,
        finite_difference_order: int | None = None,
        kernel: str = "coulomb",
        omega: float | None = None,
        multipole_order: int | None = None,
        *,
        padding: float = 0.0,
        screening: str = "split",
        rank_tolerance: float = 1.0e-14,
        device_id: int | None = None,
        memory_budget: int | None = None,
        weights: tuple[float, float] | None = None,
    ) -> None:
        cp, _ = require_cupy()
        self.cp = cp
        if isinstance(source, KroneckerPoisson):
            host = source
        else:
            keywords = {} if multipole_order is None else {"multipole_order": multipole_order}
            host = KroneckerPoisson(
                source,
                finite_difference_order,
                kernel=kernel,
                omega=omega,
                padding=padding,
                screening=screening,
                rank_tolerance=rank_tolerance,
                weights=weights,
                **keywords,
            )
        self.host = host
        self.size = host.size
        self.kernel = host.kernel
        self.omega = host.omega
        self.box_shape = host.box_shape
        self.box_size = host.box_size
        self.device_id = int(cp.cuda.Device().id if device_id is None else device_id)
        started = perf_counter()
        with cp.cuda.Device(self.device_id):
            self.plan = host.plan.mapped(lambda array: _upload(cp, array))
            cp.cuda.Device(self.device_id).synchronize()
            if memory_budget is None:
                free, _total = cp.cuda.runtime.memGetInfo()
                pool = cp.get_default_memory_pool()
                memory_budget = int(0.5 * (free + pool.free_bytes()))
        self.upload_seconds = perf_counter() - started
        self.memory_budget = int(memory_budget)
        self.plan_bytes = int(self.plan.nbytes())

    @property
    def batch_size(self) -> int:
        """Densities per batch within :attr:`memory_budget` (at least one)."""

        per_density = 8 * (WORK_ARRAYS_PER_DENSITY * self.box_size + 3 * self.size)
        return max(1, int(self.memory_budget // per_density))

    @property
    def description(self) -> str:
        return (
            self.host.description.replace("Kronecker-product", "FP64 CUDA Kronecker-product", 1)
            + f" on device {self.device_id}"
        )

    def solve_batch(self, densities, *, batch_size: int | None = None):
        """Potentials (Ry) of densities ``(n_active, nb)``, as host or device array like the input."""

        cp = self.cp
        on_device = isinstance(densities, cp.ndarray)
        step = int(self.batch_size if batch_size is None else batch_size)
        if step < 1:
            raise ValueError("batch_size must be positive")
        with cp.cuda.Device(self.device_id):
            data = cp.asarray(densities, dtype=cp.float64) if on_device else np.asarray(densities, dtype=np.float64)
            single = data.ndim == 1
            if single:
                data = data[:, None]
            if data.ndim != 2 or data.shape[0] != self.size:
                raise ValueError("densities must have shape (active grid points, densities)")
            count = int(data.shape[1])
            result = cp.empty((self.size, count), dtype=cp.float64)
            for start in range(0, count, step):
                stop = min(start + step, count)
                block = cp.asarray(data[:, start:stop])
                result[:, start:stop] = solve_block(self.plan, block, cp).T
            if single:
                result = result[:, 0]
            return result if on_device else cp.asnumpy(result)

    def __call__(self, density):
        if tuple(density.shape) != (self.size,):
            raise ValueError("density does not match the active grid")
        return self.solve_batch(density)


__all__ = ["DeviceKroneckerPoisson", "UPLOAD_CHUNK_BYTES", "WORK_ARRAYS_PER_DENSITY"]
