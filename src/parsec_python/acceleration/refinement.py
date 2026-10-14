"""Post-SCF energy refinement on the accelerated backends.

The physics is that of :mod:`parsec_python.Energy.refinement`.  This module
supplies what an accelerated calculation holds differently: the orbitals,
which a GPU run keeps on the device (or as lazy symmetry sectors) and which
are brought to the host as full-grid columns once, and the pair-density
Poisson solver, taken from the Hartree components of the prepared system
(:mod:`parsec_python.acceleration.Hartree.pair_poisson`).
"""

from __future__ import annotations

import sys

from parsec_python.Energy.refinement import (
    EnergyRefinement,
    process_memory_high_water as posix_memory_high_water,
    refine_energy as refine_reference_energy,
)

from .Hartree.pair_poisson import build_pair_poisson
from .SCF.single_point import materialize_wavefunctions


def process_memory_high_water() -> int | None:
    """Resident high-water mark of this process in bytes, ``None`` if unknown.

    ``ru_maxrss`` where the system has it, the peak working set on Windows.
    """

    value = posix_memory_high_water()
    if value is not None or sys.platform != "win32":
        return value
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.PeakWorkingSetSize)
    except (AttributeError, OSError):
        return None


def refine_energy(
    system,
    result,
    functional: str = "b3lyp",
    *,
    progress=None,
) -> EnergyRefinement:
    """Evaluate the refined energy of an accelerated SCF result.

    ``system`` is the prepared accelerated system the result came from (the
    one of the last structure after a relaxation).  The orbitals of
    ``result`` are materialized on the host if they are not there yet.
    """

    # Before the orbitals come to the host: they are part of what the
    # refinement needs.
    memory_before = process_memory_high_water()
    materialize_wavefunctions(system, result)
    pair_poisson = build_pair_poisson(system)
    return refine_reference_energy(
        system,
        result,
        functional,
        pair_poisson=pair_poisson,
        pair_solver=pair_poisson.description,
        progress=progress,
        memory_probe=process_memory_high_water,
        memory_before=memory_before,
    )


__all__ = ["process_memory_high_water", "refine_energy"]
