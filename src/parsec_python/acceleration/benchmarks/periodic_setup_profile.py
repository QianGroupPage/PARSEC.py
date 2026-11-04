"""Time the preparation of a periodic cell, stage by stage, and fingerprint its fields.

Diamond silicon cubes like ``examples/3d_Si`` (conventional 10.26 bohr cells, ``Si_POTRE.DAT`` with
local p, ``Expansion_Order`` 12): ``si8`` is the example itself (h 0.45, 12,167 points), ``si64``
2x2x2 cells (h 0.45, 97,336 points), ``si216`` 3x3x3 (h 0.45, 314,432 points), ``si864`` 6x6x3
cells (h 0.57, 629,856 points: the 864-atom cell of the fast periodic preparation's docstring).
Each case writes its ``parsec.in`` into the output folder, parses it, runs
:func:`parsec_python.SCF.pbc.prepare_periodic_single_point` once (a first run: nothing is cached),
and prints the stage seconds of ``PreparationTimings``, the peak resident memory of the process and
the SHA-256 of the local potential, the two densities and the projector values, so two source trees
can be compared from their logs.

    python -m parsec_python.acceleration.benchmarks.periodic_setup_profile OUT [case ...]
        [--reach table|physical] [--threads N] [--save]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SI_POTENTIAL = ROOT / "examples" / "3d_Si" / "Si_POTRE.DAT"
SIDE = 10.26

_DIAMOND = np.array(
    [
        [0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
    ]
)

CASES = {
    "si8": ((1, 1, 1), 0.45),
    "si64": ((2, 2, 2), 0.45),
    "si216": ((3, 3, 3), 0.45),
    "si864": ((6, 6, 3), 0.57),
}


def _input_text(cells, spacing: float) -> str:
    positions = [
        (site + np.array([i, j, k])) * SIDE
        for i in range(cells[0])
        for j in range(cells[1])
        for k in range(cells[2])
        for site in _DIAMOND
    ]
    coordinates = "\n".join(f"  {x:.6f} {y:.6f} {z:.6f}" for x, y, z in positions)
    return f""" Periodic_System: .true.
 Boundary_Conditions: bulk
 begin Cell_Shape
    {SIDE * cells[0]:.6f}  {SIDE * cells[1]:.6f}  {SIDE * cells[2]:.6f}
 end Cell_Shape
 Lattice_Vector_Scale: 1.0 bohr
 Coordinate_Unit: Cartesian_Bohr
 Grid_Spacing:  {spacing}
 Expansion_Order:  12
 States_Num:  {2 * len(positions) + 8}
 Fermi_Temp:  300.0
 Max_Iter:  1
 Correlation_Type: ca
 Atom_Types_Num:  1
 Atom_Type: Si
 Local_Component: p
 begin Atom_Coord
{coordinates}
 end Atom_Coord
"""


def _peak_resident_mib() -> float:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return float("nan")
        return counters.PeakWorkingSetSize / 2**20
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _fingerprint(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()[:16]


def run_case(name: str, out: Path, save: bool) -> None:
    from parsec_python.Input import parse_parsec_input
    from parsec_python.SCF.pbc import prepare_periodic_single_point

    cells, spacing = CASES[name]
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "parsec.in").write_text(_input_text(cells, spacing))
    shutil.copy(SI_POTENTIAL, folder / "Si_POTRE.DAT")
    problem = parse_parsec_input(folder / "parsec.in").problem
    before = _peak_resident_mib()
    start = time.perf_counter()
    system = prepare_periodic_single_point(problem)
    wall = time.perf_counter() - start
    timings = system.timings
    projectors = system.nonlocal_operator.projectors.tocsc()
    print(
        f"{name}: {len(system.atoms)} atoms, {system.grid.size} points, preparation {wall:.2f} s "
        f"(local potential {timings.local_ionic_seconds:.2f}, valence density "
        f"{timings.initial_density_seconds:.2f}, core density {timings.core_density_seconds:.2f}, "
        f"projectors {timings.nonlocal_ionic_seconds:.2f}, ion-ion {timings.ion_ion_seconds:.2f}, "
        f"finite differences {timings.finite_difference_seconds:.2f}); peak resident "
        f"{_peak_resident_mib():.0f} MiB (before {before:.0f})",
        flush=True,
    )
    print(
        f"{name}: sha256 potential {_fingerprint(system.ionic_potential)} density "
        f"{_fingerprint(system.initial_density)} core {_fingerprint(system.core_density)} "
        f"projectors {_fingerprint(projectors.data)} ion-ion {system.ion_ion_energy!r}",
        flush=True,
    )
    if save:
        np.savez(
            folder / "fields.npz",
            ionic_potential=system.ionic_potential,
            initial_density=system.initial_density,
            core_density=system.core_density,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("cases", nargs="*", metavar="case", help=f"of {', '.join(CASES)} (default si8)")
    parser.add_argument("--reach", choices=("table", "physical"), default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--save", action="store_true", help="write fields.npz into each case folder")
    arguments = parser.parse_args(argv)
    unknown = [name for name in arguments.cases if name not in CASES]
    if unknown:
        parser.error(f"unknown case(s) {', '.join(unknown)}; choose from {', '.join(CASES)}")
    if arguments.reach is not None:
        os.environ["PARSEC_PERIODIC_SETUP_REACH"] = arguments.reach
    if arguments.threads is not None:
        os.environ["PARSEC_PERIODIC_SETUP_THREADS"] = str(arguments.threads)
    import parsec_python  # noqa: F401  (installs the fast periodic preparation)

    print(
        f"parsec_python {Path(parsec_python.__file__).parent}; reach "
        f"{os.environ.get('PARSEC_PERIODIC_SETUP_REACH', 'default')}, threads "
        f"{os.environ.get('PARSEC_PERIODIC_SETUP_THREADS', 'default')}, OMP_NUM_THREADS "
        f"{os.environ.get('OMP_NUM_THREADS', 'unset')}, CPUs {os.cpu_count()}",
        flush=True,
    )
    for name in arguments.cases or ["si8"]:
        run_case(name, arguments.out, arguments.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
