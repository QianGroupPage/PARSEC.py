"""Best-effort control of the BLAS thread count of the running process.

Worker processes of a k-point pool share the machine, so each should use
``cores / workers`` BLAS threads instead of all of them.  Thread pools of the
BLAS libraries are fixed when the library loads and are normally steered by
environment variables read at import time, which is too early for a forked
worker.  This does what ``threadpoolctl`` does, without the dependency: find
the BLAS libraries already mapped into the process (``/proc/self/maps`` on
Linux, the module list of the process on Windows) and call their
``*_set_num_threads`` entry point.
"""

from __future__ import annotations

import ctypes
import os
from typing import Mapping

_SETTERS = (
    "openblas_set_num_threads",
    "openblas_set_num_threads64_",
    "scipy_openblas_set_num_threads",
    "scipy_openblas_set_num_threads64_",
    "MKL_Set_Num_Threads",
    "bli_thread_set_num_threads",
)
_GETTERS = (
    "openblas_get_num_threads",
    "openblas_get_num_threads64_",
    "scipy_openblas_get_num_threads",
    "scipy_openblas_get_num_threads64_",
    "MKL_Get_Max_Threads",
    "bli_thread_get_num_threads",
)
_TAGS = ("blas", "mkl_rt", "libmkl", "blis")


def _linux_libraries() -> set[str] | None:
    try:
        with open("/proc/self/maps", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:  # not Linux
        return None
    paths = set()
    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        name = os.path.basename(path).lower()
        if ".so" in name and any(tag in name for tag in _TAGS):
            paths.add(path)
    return paths


def _windows_libraries() -> set[str]:
    """The BLAS DLLs loaded into this process (NumPy's and SciPy's OpenBLAS, for instance)."""
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)
    ]
    psapi.EnumProcessModules.restype = wintypes.BOOL
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    process = kernel32.GetCurrentProcess()
    count = 512
    while True:
        modules = (wintypes.HMODULE * count)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
            return set()
        if needed.value <= ctypes.sizeof(modules):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 16
    paths = set()
    buffer = ctypes.create_unicode_buffer(32768)
    for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        if not kernel32.GetModuleFileNameW(module, buffer, len(buffer)):
            continue
        name = os.path.basename(buffer.value).lower()
        if name.endswith(".dll") and any(tag in name for tag in _TAGS):
            paths.add(buffer.value)
    return paths


def _loaded_blas_libraries() -> set[str]:
    paths = _linux_libraries()
    if paths is not None:
        return paths
    if os.name == "nt":
        try:
            return _windows_libraries()
        except Exception:  # pragma: no cover - an API that refuses leaves the BLAS as it is
            return set()
    return set()  # elsewhere (macOS): not found, nothing is changed


def _libraries():
    for path in _loaded_blas_libraries():
        try:
            yield path, ctypes.CDLL(path)
        except OSError:
            continue


def blas_threads() -> dict[str, int]:
    """The thread count of every loaded BLAS library that tells it, by the path of the library."""
    counts = {}
    for path, library in _libraries():
        for name in _GETTERS:
            function = getattr(library, name, None)
            if function is None:
                continue
            try:
                function.argtypes = []
                function.restype = ctypes.c_int
                counts[path] = int(function())
                break
            except Exception:  # pragma: no cover - a library that refuses is simply skipped
                continue
    return counts


def set_blas_threads(count: int, *, previous: Mapping[str, int] | None = None) -> bool:
    """Ask every loaded BLAS library to use ``count`` threads; ``True`` if any answered.

    A library named in ``previous`` (what :func:`blas_threads` returned) gets its own count back
    instead.
    """
    count = max(1, int(count))
    previous = previous or {}
    answered = False
    for path, library in _libraries():
        wanted = max(1, int(previous.get(path, count)))
        for name in _SETTERS:
            function = getattr(library, name, None)
            if function is None:
                continue
            try:
                function.argtypes = [ctypes.c_int]
                function.restype = None
                function(wanted)
                answered = True
            except Exception:  # pragma: no cover - a library that refuses is simply skipped
                continue
    return answered


__all__ = ["blas_threads", "set_blas_threads"]
