"""Fused CuPy density construction from device-resident orbitals."""

from __future__ import annotations

from contextlib import nullcontext
from time import perf_counter
from typing import Any

import numpy as np

from ..Eigensolvers.spill_buffers import copies_directly, spill_buffer_mode, upload


_DENSITY_KERNEL = r"""
extern "C" __global__
void orbital_density(
    const long long row_count,
    const int state_count,
    const double factor,
    const double* __restrict__ wavefunctions,
    const long long row_stride,
    const long long column_stride,
    const double* __restrict__ occupations,
    double* __restrict__ density
) {
    const long long row =
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    double value = 0.0;
    for (int state = 0; state < state_count; ++state) {
        const double orbital = wavefunctions[
            row * row_stride + static_cast<long long>(state) * column_stride
        ];
        value += occupations[state] * orbital * orbital;
    }
    density[row] = factor * value;
}
"""


class CuPyDeviceDensityBuilder:
    """Build host density while retaining all orbital columns on the GPU.

    Orbitals that are handed over on the host, the columns of a sector
    whose states are kept there, are uploaded first.  A Fortran-ordered
    matrix of doubles goes to the device by one copy from where it lies
    (:func:`Eigensolvers.spill_buffers.upload`) unless
    ``PARSEC_CUPY_SPILL_BUFFERS=fresh`` names the former upload,
    ``cupy.asarray``, which stages it in a block of CuPy's pinned pool; any
    other host array takes that call as before.  The device array is the
    same to the last bit.  ``host_uploads`` counts these uploads, their
    bytes and the wall seconds of the calls; those of the former upload
    end where ``cupy.asarray`` returns, which can be before its copy to
    the device has completed.
    """

    def __init__(self, cp: Any, timing_stats: Any | None = None) -> None:
        self.cp = cp
        self.timing_stats = timing_stats
        # Read here, so that a value it does not know stops a run when it
        # is prepared.
        self.direct_host_upload = spill_buffer_mode() == "persistent"
        self.host_uploads = dict(
            direct=self.direct_host_upload, calls=0, bytes=0, seconds=0.0
        )
        self.kernel = None
        if hasattr(cp, "RawKernel"):
            self.kernel = cp.RawKernel(
                _DENSITY_KERNEL,
                "orbital_density",
                options=("--std=c++11",),
            )
            self.kernel.compile()

    def __call__(
        self,
        wavefunctions: Any,
        occupations: np.ndarray,
        volume_element: float,
    ) -> np.ndarray:
        # CuPy.asarray copies arrays from another CUDA device to the current
        # device.  Reduce each sector where its orbitals already live instead
        # of gathering O(grid * states) data onto GPU 0 at every SCF step.
        device = getattr(wavefunctions, "device", None)
        context = device if hasattr(device, "__enter__") else nullcontext()
        with context:
            return self._on_current_device(wavefunctions, occupations, volume_element)

    def _on_current_device(
        self,
        wavefunctions: Any,
        occupations: np.ndarray,
        volume_element: float,
    ) -> np.ndarray:
        cp = self.cp
        started = perf_counter()
        # A device builder only: the shim of a test has no device to copy to.
        on_host = self.kernel is not None and isinstance(wavefunctions, np.ndarray)
        if on_host and self.direct_host_upload and copies_directly(wavefunctions):
            vectors = upload(cp, wavefunctions)
        else:
            vectors = cp.asarray(wavefunctions, dtype=cp.float64)
        if on_host:
            self.host_uploads["calls"] += 1
            self.host_uploads["bytes"] += int(wavefunctions.nbytes)
            self.host_uploads["seconds"] += perf_counter() - started
        occupation_values = np.asarray(occupations, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError("wavefunctions must have shape (grid_points, states)")
        if occupation_values.shape != (vectors.shape[1],):
            raise ValueError("occupation count does not match wavefunction columns")
        if volume_element <= 0.0:
            raise ValueError("volume_element must be positive")

        if self.kernel is None:
            # NumPy-backed test shims and unusual CuPy builds retain a clear,
            # allocation-based fallback without changing SCF behavior.
            density = (2.0 / volume_element) * np.sum(
                np.asarray(vectors)
                * np.asarray(vectors)
                * occupation_values[None, :],
                axis=1,
            )
        else:
            device_occupations = cp.asarray(occupation_values, dtype=cp.float64)
            density = cp.empty(vectors.shape[0], dtype=cp.float64)
            threads = 256
            blocks = (int(vectors.shape[0]) + threads - 1) // threads
            itemsize = int(vectors.dtype.itemsize)
            self.kernel(
                (blocks,),
                (threads,),
                (
                    np.int64(vectors.shape[0]),
                    np.int32(vectors.shape[1]),
                    np.float64(2.0 / volume_element),
                    vectors,
                    np.int64(vectors.strides[0] // itemsize),
                    np.int64(vectors.strides[1] // itemsize),
                    device_occupations,
                    density,
                ),
            )
        host_density = np.asarray(cp.asnumpy(density), dtype=np.float64)
        if self.timing_stats is not None:
            self.timing_stats.density_calls += 1
            self.timing_stats.density_seconds += perf_counter() - started
        return host_density


__all__ = ["CuPyDeviceDensityBuilder"]
