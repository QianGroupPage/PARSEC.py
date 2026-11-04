"""The tetrahedron sums of ``Create_DOS`` on a GPU, with the numbers of the host.

:func:`parsec_python.Output.dos.tetrahedron_spectrum` evaluates the linear-tetrahedron spectrum
(``spec_tet.F90``) of every (tetrahedron, band) row at the mesh points of its window and sums the
pairs onto the mesh with ``bincount``.  With the Ylm-projected DOS the work is
``pairs x (1 + atoms x harmonics)`` values: the Si8 cube on a 3x3x3 grid with ``YLM_DOS_L: 3`` and
``DOS_Pnum: 2000`` holds 447 thousand pairs x 129 values, 5.3 s on the host of an A100 node (18 % of
the eigensolves of its k-point SCF; the report of the parity round).

This module installs (:func:`install`, from :mod:`parsec_python.acceleration`) a provider that runs
the readable :func:`parsec_python.Output.dos.spectrum_sums` with CuPy arrays: every pair quantity is
the same sequence of element-wise IEEE double operations as on the host (one CuPy kernel per NumPy
operation, so nothing is contracted into a fused multiply-add).  The sums onto the mesh points keep
the order of ``bincount`` - for every mesh point (and column) one thread adds the pairs of the rows
in row order, starting from zero (:data:`_ORDERED_SUM`) - so the DOS file is bit for bit the one of
the host path, which the tests check.  The corner values are evaluated a block of columns at a time
so the pair arrays stay within a part of the free device memory.

The device is the first CUDA device of the run (``--device`` / ``PARSEC_DEVICE`` and
``PARSEC_CUPY_DEVICES``, as for the k-point workers; ``PARSEC_KPOINT_BACKEND`` is not read).
``PARSEC_DOS_DEVICE`` selects: ``auto`` (default) the device from :data:`AUTO_VALUES` pair values
on (5 million values take 0.05 s on the host of an A100 node, 57 million 5.3 s), ``on`` the device
whenever there is one, ``off`` the host.  The pair arrays of one column block take at most
:data:`BLOCK_BYTES` (or a quarter of the free device memory), so the DOS adds little to the device
memory peak of the run.
"""

from __future__ import annotations

import os

import numpy as np

from parsec_python.Output import dos as _dos

DEVICE_ENV = "PARSEC_DOS_DEVICE"  # auto (default) | on | off
#: Pair values (pairs x (1 + columns)) from which ``auto`` uses the device.
AUTO_VALUES = 20_000_000
#: Bytes and fraction of the free device memory the pair arrays of one column block may take.
BLOCK_BYTES = 1 << 30
_MEMORY_FRACTION = 0.25
#: Pair arrays of one column block alive at once in ``spectrum_sums`` (``f``, its four corners and
#: their differences, the terms and ``fms``), counted generously.
_LIVE_ARRAYS = 24

_ORDERED_SUM = r"""
extern "C" __global__ void ordered_bin_sum(
    const double* weights,      // (pairs, columns), row-major
    const long long* istart,    // (rows,) first mesh point of every row's window, 1-based
    const long long* counts,    // (rows,) window lengths
    const long long* first,     // (rows,) first pair of every row
    const long long rows,
    const long long points,
    const long long columns,
    double* out)                // (columns, points)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= points * columns) return;
    const long long bin = t / columns;
    const long long column = t - bin * columns;
    const long long index = bin + 1;
    double sum = 0.0;
    for (long long r = 0; r < rows; ++r) {
        const long long k = index - istart[r];
        if (k >= 0 && k < counts[r]) sum += weights[(first[r] + k) * columns + column];
    }
    out[column * points + bin] = sum;
}
"""

_kernels: dict[int, object] = {}


def _mode() -> str:
    value = os.environ.get(DEVICE_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "on", "off"):
        raise ValueError(f"{DEVICE_ENV} must be auto, on or off, not {value!r}")
    return value


def _device_id() -> int | None:
    """The first CUDA device of the run (``--device`` / ``PARSEC_DEVICE``), ``None`` without one.

    The devices are those of the k-point device plan (``PARSEC_CUPY_DEVICES``); its explicit override
    ``PARSEC_KPOINT_BACKEND`` concerns the k-point eigensolves and is not read here.
    """

    from ..SCF.kpoint_parallel import resolve_backend

    device = os.environ.get("PARSEC_DEVICE", "auto").strip().lower() or "auto"
    requested = {"cpu": "cpu", "gpu": "cupy", "auto": "auto"}.get(device)
    if requested is None:
        raise ValueError(f"PARSEC_DEVICE must be auto, gpu or cpu, not {device!r}")
    plan = resolve_backend(requested)
    if plan is None or plan.name != "cupy" or not plan.device_ids:
        return None
    return int(plan.device_ids[0])


def _kernel(cp, device_id: int):
    kernel = _kernels.get(device_id)
    if kernel is None:
        kernel = cp.RawKernel(_ORDERED_SUM, "ordered_bin_sum")
        _kernels[device_id] = kernel
    return kernel


def ordered_bin_sum(cp, kernel, rows, weights, points: int):
    """``bincount`` of the pair ``weights`` on the device, in pair order (one thread per bin)."""

    single = weights.ndim == 1
    block = cp.ascontiguousarray(weights[:, None] if single else weights, dtype=cp.float64)
    columns = int(block.shape[1])
    out = cp.empty((columns, points), dtype=cp.float64)
    istart, counts, first = rows
    threads = 128
    total = points * columns
    kernel(
        ((total + threads - 1) // threads,),
        (threads,),
        (
            block,
            istart,
            counts,
            first,
            np.int64(istart.size),
            np.int64(points),
            np.int64(columns),
            out,
        ),
    )
    return out[0] if single else out


def device_spectrum_sums(volume: float, rows, mesh):
    """:func:`parsec_python.Output.dos.spectrum_sums` on the GPU; ``None`` leaves them to the host."""

    mode = _mode()
    if mode == "off":
        return None
    columns = 0 if rows.fs is None else int(rows.fs.shape[2])
    if mode == "auto" and rows.row.size * (1 + columns) < AUTO_VALUES:
        return None
    device_id = _device_id()
    if device_id is None:
        return None
    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    with cp.cuda.Device(device_id):
        free, _total = cp.cuda.runtime.memGetInfo()
        per_column = rows.row.size * 8 * _LIVE_ARRAYS
        budget = min(BLOCK_BYTES, _MEMORY_FRACTION * free)
        block = max(1, min(max(columns, 1), int(budget // max(per_column, 1))))
        layout = tuple(cp.asarray(array, dtype=cp.int64) for array in (rows.istart, rows.counts, rows.first))
        kernel = _kernel(cp, device_id)
        edos, fgrid = _dos.spectrum_sums(
            volume,
            rows,
            mesh,
            xp=cp,
            bin_sum=lambda _index, weights, points: ordered_bin_sum(cp, kernel, layout, weights, points),
            column_block=block,
        )
        host = cp.asnumpy(edos), (None if fgrid is None else cp.asnumpy(fgrid))
        cp.get_default_memory_pool().free_all_blocks()
    return host


def install() -> None:
    """Make :func:`device_spectrum_sums` the provider of the tetrahedron sums."""

    _dos.set_spectrum_provider(device_spectrum_sums)


__all__ = ["AUTO_VALUES", "BLOCK_BYTES", "DEVICE_ENV", "device_spectrum_sums", "install", "ordered_bin_sum"]
