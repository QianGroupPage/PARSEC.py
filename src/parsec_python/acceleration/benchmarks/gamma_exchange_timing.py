#!/usr/bin/env python3
"""Time the device exchange of a periodic cell at the Gamma point (real orbitals) on the GPUs of one node.

The loop of isolated systems (``apply_device_exchange``) with the cuFFT pair
solver (``DevicePeriodicPairSolvers``) on random orthonormal real orbitals of
a cubic box: one build of every column (first and warm), its pair solves and
the seconds the solves took, beside one solve of ``--batch`` densities timed
alone.

Usage::

    PYTHONPATH=src python -m parsec_python.acceleration.benchmarks.gamma_exchange_timing \
        --edge 68 --states 140 --occupied 128 --devices 4
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edge", type=int, default=68)
    parser.add_argument("--length", type=float, default=20.52)
    parser.add_argument("--states", type=int, default=140)
    parser.add_argument("--occupied", type=int, default=128)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--small-build-points", type=float, default=None,
        help="replace SMALL_BUILD_POINTS (0: every build on every device)",
    )
    args = parser.parse_args()
    if args.small_build_points is not None:
        from parsec_python.acceleration.Hamiltonian import bloch_exchange_device

        bloch_exchange_device.SMALL_BUILD_POINTS = float(args.small_build_points)

    import cupy as cp

    from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import DevicePeriodicPairSolvers
    from parsec_python.acceleration.Hamiltonian.exchange_device import apply_device_exchange, channel_orbitals

    from .bloch_exchange_timing import DeviceCoulombKernel

    n = args.edge
    points = n**3
    devices = tuple(range(min(args.devices, cp.cuda.runtime.getDeviceCount())))
    kernel = DeviceCoulombKernel(np.eye(3) * args.length, (n, n, n))
    solvers = DevicePeriodicPairSolvers(kernel, SimpleNamespace(shape=(n, n, n), flat=None), args.batch)
    vectors, _ = cp.linalg.qr(cp.random.random((points, args.states)))
    occupations = np.array([1.0] * args.occupied + [0.0] * (args.states - args.occupied))
    channel = channel_orbitals(vectors, occupations, 2.0, args.length**3 / points)
    result = {"edge": n, "states": args.states, "occupied": args.occupied, "devices": len(devices), "batch": args.batch}
    for name in ("first", "warm"):
        for device in devices:
            cp.cuda.Device(device).synchronize()
        started = time.perf_counter()
        application = apply_device_exchange(channel, solvers, devices)
        for device in devices:
            cp.cuda.Device(device).synchronize()
        result[name] = {
            "build_seconds": round(time.perf_counter() - started, 3),
            "solver_seconds_max_device": round(application.poisson_seconds, 3),
            "pair_solves": application.pair_solves,
        }
        del application
    densities = cp.asfortranarray(cp.random.random((points, args.batch)))
    solvers.solve(cp, 0, densities)
    cp.cuda.Device().synchronize()
    started = time.perf_counter()
    for _ in range(10):
        solvers.solve(cp, 0, densities)
    cp.cuda.Device().synchronize()
    result["one_solve_ms"] = round((time.perf_counter() - started) / 10 * 1.0e3, 3)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
