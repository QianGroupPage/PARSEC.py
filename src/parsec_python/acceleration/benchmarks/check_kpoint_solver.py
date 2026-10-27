#!/usr/bin/env python3
"""Time the k-point SCF with the host eigensolver against the device-resident one.

``PARSEC_KPOINT_SOLVER`` selects the route of the k-point workers on a device:

``host``    the host eigensolver (QR, Rayleigh-Ritz, CHEBDAV/CHEBFF bookkeeping in NumPy) with only
            the Chebyshev filter on the GPU, one upload and one download per block of columns;
``device``  the Ritz subspaces of the k-points stay on the GPU and every SUBSPACE step (filter,
            QR, Rayleigh-Ritz) runs there; the first solve is CHEBFF on the device or, for
            ``Eigensolver: chebdav``, CHEBDAV on the host whose result moves to the device.

This script runs the same input through ``parsec_python.main`` once per route (and round), on a
copy of the input with ``Max_Iter`` lowered and the ``bandstruc`` block removed so that a run is
short, and prints the wall time, the SCF time, the iterations, the total energy and the ratio of
the two routes.  The first run of a process on a fresh CuPy kernel cache is much slower than the
rest (kernel compilation), so rounds are repeated: read the last round.

Usage (inside a GPU allocation, from the repository root):

    PYTHONPATH=src python -m parsec_python.acceleration.benchmarks.check_kpoint_solver \\
        --input /path/to/GaAs/parsec/soc/parsec.in --iterations 6 --rounds 2

``--backend numpy-device`` runs the device code on the host (a check of the script itself).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

NUMBER = r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"


def prepare_copy(source: Path, directory: Path, iterations: int) -> Path:
    """Copy the input and its pseudopotential files; shorten the run."""
    for item in source.parent.iterdir():
        if item.is_file() and item.suffix.upper() in (".DAT", ".IN", ".UPF", ".PSP", ".TXT") or item == source:
            shutil.copy(item, directory / item.name)
    target = directory / source.name
    text = target.read_text()
    text = re.sub(r"(?im)^\s*Max_Iter\w*\s*:.*$", f"Max_Iter: {iterations}", text)
    if not re.search(r"(?im)^\s*Max_Iter\s*:", text):
        text += f"\nMax_Iter: {iterations}\n"
    text = re.sub(r"(?is)begin\s+bandstruc.*?end\s+bandstruc\s*", "", text)
    text = re.sub(r"(?im)^\s*bandstruc_points\s.*$", "", text)
    target.write_text(text)
    return target


def run(route: str, source: Path, arguments, label: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="kpoint_solver_") as name:
        directory = Path(name)
        target = prepare_copy(source, directory, arguments.iterations)
        environment = dict(os.environ)
        environment["PARSEC_KPOINT_SOLVER"] = route
        environment.setdefault("PYTHONPATH", "")
        if arguments.backend:
            environment["PARSEC_KPOINT_BACKEND"] = arguments.backend
        if arguments.workers:
            environment["PARSEC_KPOINT_WORKERS"] = str(arguments.workers)
        command = [
            sys.executable, "-m", "parsec_python.main", str(target),
            "--device", arguments.device, "--no-archive", "--quiet",
            "--log", str(directory / "parsec.out"),
        ]
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=directory, env=environment, capture_output=True, text=True)
        wall = time.perf_counter() - start
        log = (directory / "parsec.out").read_text(errors="replace") if (directory / "parsec.out").exists() else ""
        if completed.returncode not in (0, 3) and not log:
            print(completed.stderr[-2000:])
        iterations = len(re.findall(r"(?:SOC iter|SCF iter) #", log))
        scf = re.findall(r"Time for self-consistent field \[sec\]\s*:\s*" + NUMBER, log)
        energy = re.findall(r"Total Energy\s*=\s*" + NUMBER + r"\s*\[Ry\]", log)
        return {
            "label": label,
            "route": route,
            "returncode": completed.returncode,
            "wall": wall,
            "scf": float(scf[-1]) if scf else None,
            "iterations": iterations,
            "energy": float(energy[-1]) if energy else None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="parsec.in of a periodic k-point calculation")
    parser.add_argument("--iterations", type=int, default=6, help="Max_Iter of the shortened runs")
    parser.add_argument("--rounds", type=int, default=2, help="repeat both routes (the first round warms the caches)")
    parser.add_argument("--routes", default="host,device")
    parser.add_argument("--device", default="gpu", choices=("gpu", "cpu", "auto"))
    parser.add_argument("--backend", default=None, help="PARSEC_KPOINT_BACKEND override (numpy-device on a host)")
    parser.add_argument("--workers", type=int, default=None, help="PARSEC_KPOINT_WORKERS")
    arguments = parser.parse_args()
    source = arguments.input.resolve()
    results = []
    for round_index in range(1, arguments.rounds + 1):
        for route in arguments.routes.split(","):
            label = f"round {round_index}"
            print(f"running {route} ({label}) ...", flush=True)
            results.append(run(route, source, arguments, label))
    print(f"\n{'route':8s} {'round':8s} {'wall s':>9s} {'SCF s':>9s} {'iter':>5s} {'s/iter':>8s} {'E (Ry)':>16s} rc")
    for r in results:
        per = r["scf"] / r["iterations"] if r["scf"] and r["iterations"] else None
        print(
            f"{r['route']:8s} {r['label']:8s} {r['wall']:9.1f} "
            f"{(r['scf'] if r['scf'] is not None else float('nan')):9.1f} {r['iterations']:5d} "
            f"{(per if per is not None else float('nan')):8.2f} "
            f"{(r['energy'] if r['energy'] is not None else float('nan')):16.8f} {r['returncode']}"
        )
    last = {r["route"]: r for r in results if r["label"] == f"round {arguments.rounds}"}
    if "host" in last and "device" in last and last["host"]["scf"] and last["device"]["scf"]:
        print(
            f"\nlast round: device route is x{last['host']['scf'] / last['device']['scf']:.2f} faster in SCF time "
            f"than the host route; dE = {last['device']['energy'] - last['host']['energy']:+.2e} Ry"
            " (the runs stop at Max_Iter, so they agree only as far as the SCF has converged)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
