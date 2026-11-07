"""Device memory of the GPU Hartree objects, for the choices of ``auto``.

``PARSEC_HARTREE_LINEAR_BACKEND`` unset or ``auto`` puts the conjugate
gradients of the Hartree solve (and, on a full grid, the boundary values)
on a GPU.  Where nothing else plans the device around those arrays, the
choice asks this module what they take and what the device has free, and
gives way to the host CG where they would crowd the calculation out; it
gives way too where building them fails for want of memory.  No device is
touched beyond asking for its free memory.
"""

from __future__ import annotations

# Without the memory rules of symmetry sectors nothing plans the device around
# the GPU Hartree arrays.  The full-grid runs measured on A100 (2026-11-07: O2,
# CF4, H2O, the O atom, protonated water) took 10 to 17 times those arrays
# beside them, so they are kept on the GPU only where they take at most this
# share of what their device has free when they are planned.
FREE_SHARE_BESIDE_FULL_GRID = 1.0 / 20.0
# The pair solves of an energy refinement run after the SCF, with its arrays
# still held: a new full-grid CG is made only where it takes at most this
# share of the free memory, which leaves the rest to the pair densities.
FREE_SHARE_OF_PAIR_SOLVES = 1.0 / 2.0
# Bytes of the streaming full-grid GPU boundary per grid row (an estimate).
BOUNDARY_BYTES_PER_ROW = 60


def planned_cg_bytes(rows: int, stencil_half_width: int) -> int:
    """Device bytes of the GPU CG (``cupy_prepared.py``) of ``rows`` rows.

    An int32 neighbor and a one-byte coefficient code per stencil entry
    (``6 h + 1`` entries per row for the half width ``h``) and five FP64
    work vectors per row.
    """

    width = 6 * int(stencil_half_width) + 1
    return int(rows) * (5 * width + 5 * 8)


def planned_gpu_hartree_bytes(rows: int, stencil_half_width: int, *, gpu_boundary: bool) -> int:
    """:func:`planned_cg_bytes`, with the streaming GPU boundary if asked."""

    return planned_cg_bytes(rows, stencil_half_width) + (
        BOUNDARY_BYTES_PER_ROW * int(rows) if gpu_boundary else 0
    )


def device_free_bytes(device_id: int | None = None) -> int | None:
    """What CuPy may still allocate on a device, or ``None`` where no device answers.

    The driver's free memory plus the free blocks of CuPy's pool, within
    the limit of the pool (``CUPY_GPU_MEMORY_LIMIT``) where one is set.
    ``None`` is the current device.
    """

    try:
        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        device = cp.cuda.Device().id if device_id is None else int(device_id)
        with cp.cuda.Device(device):
            free, _total = cp.cuda.runtime.memGetInfo()
            pool = cp.get_default_memory_pool()
            available = int(free) + int(pool.free_bytes())
            limit = int(pool.get_limit())
            if limit > 0:
                available = min(available, limit - int(pool.used_bytes()))
            return max(0, available)
    except Exception:  # noqa: BLE001 - no device to ask: no plan
        return None


def cuda_allocation_errors() -> tuple[type[BaseException], ...]:
    """The errors of a GPU allocation that a choice of ``auto`` gives way to.

    ``MemoryError`` (CuPy's ``OutOfMemoryError``) and CuPy's CUDA runtime
    and driver errors, ``RuntimeError`` subclasses through which an
    exhausted device also answers (stream creation, module load, graph
    instantiation).
    """

    errors: list[type[BaseException]] = [MemoryError]
    try:
        import cupy

        errors.extend((cupy.cuda.runtime.CUDARuntimeError, cupy.cuda.driver.CUDADriverError))
    except Exception:  # noqa: BLE001 - CuPy absent: no CUDA errors to name
        pass
    return tuple(errors)
