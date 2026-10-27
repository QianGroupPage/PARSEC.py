"""Best-effort control of the BLAS thread count of the running process.

Worker processes of a k-point pool share the machine, so each should use
``cores / workers`` BLAS threads instead of all of them.  Thread pools of the
BLAS libraries are fixed when the library loads and are normally steered by
environment variables read at import time, which is too early for a forked
worker.  This does what ``threadpoolctl`` does, without the dependency: find
the BLAS libraries already mapped into the process and call their
``*_set_num_threads`` entry point.
"""

from __future__ import annotations

import ctypes
import os

_SETTERS = (
    "openblas_set_num_threads",
    "openblas_set_num_threads64_",
    "scipy_openblas_set_num_threads",
    "scipy_openblas_set_num_threads64_",
    "MKL_Set_Num_Threads",
    "bli_thread_set_num_threads",
)


def _loaded_blas_libraries() -> set[str]:
    try:
        with open("/proc/self/maps", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:  # not Linux
        return set()
    paths = set()
    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        name = os.path.basename(path).lower()
        if ".so" in name and any(tag in name for tag in ("blas", "mkl_rt", "libmkl", "blis")):
            paths.add(path)
    return paths


def set_blas_threads(count: int) -> bool:
    """Ask every loaded BLAS library to use ``count`` threads; ``True`` if any answered."""
    count = max(1, int(count))
    answered = False
    for path in _loaded_blas_libraries():
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        for name in _SETTERS:
            function = getattr(library, name, None)
            if function is None:
                continue
            try:
                function.argtypes = [ctypes.c_int]
                function.restype = None
                function(count)
                answered = True
            except Exception:  # pragma: no cover - a library that refuses is simply skipped
                continue
    return answered


__all__ = ["set_blas_threads"]
