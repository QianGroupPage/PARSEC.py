#!/usr/bin/env python3
"""Time the device exchange of a periodic hybrid on the GPUs of one node (first and warm builds, peak memory).

The cost of ``W = V_x U``, ACE and the exchange term of the Hamiltonian
depends on the sizes only (grid, FFT box, k-mesh, states, occupied states,
devices), so random orthonormal Bloch factors of those sizes stand in for the
orbitals of an SCF.  Each representative k-point lies on the device of its
eigensolver worker (``rep % devices``, the order of the k-point pool).  The
kernel is ``8 pi/|q+G|^2`` built on the device per ``q`` (as the exchange
kernel of the code does).

Prints, per case: pair solves, seconds of the first build (cuFFT plans and
kernel arrays made then) and of a second build, the share of the transforms,
the ACE seconds, one application of the exchange term to the states of a
k-point, and the peak device memory of the pool on every device.

Usage::

    PYTHONPATH=src python -m parsec_python.acceleration.benchmarks.bloch_exchange_timing \
        --case si-prim --mesh 4 4 4 --devices 4

``--case`` presets: ``si-prim`` (2-atom fcc cell, 18^3 grid, 4 occupied of 8),
``si64`` (64-atom cubic cell, 56^3 grid, 128 of 140), ``hbn`` (h-BN slab,
12x12x54 grid with a 2x padded FFT box along z, 4 of 8), or ``--box``,
``--fft``, ``--occupied``, ``--states``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

PRESETS = {
    "si-prim": {
        "lattice": [[0.0, 5.13, 5.13], [5.13, 0.0, 5.13], [5.13, 5.13, 0.0]],
        "box": (18, 18, 18),
        "fft": None,
        "occupied": 4,
        "states": 8,
    },
    "si64": {
        "lattice": [[20.52, 0.0, 0.0], [0.0, 20.52, 0.0], [0.0, 0.0, 20.52]],
        "box": (56, 56, 56),
        "fft": None,
        "occupied": 128,
        "states": 140,
    },
    "hbn": {
        "lattice": [[4.74, 0.0, 0.0], [-2.37, 4.105, 0.0], [0.0, 0.0, 21.6]],
        "box": (12, 12, 54),
        "fft": (12, 12, 108),
        "occupied": 4,
        "states": 8,
    },
}


class DeviceCoulombKernel:
    """``8 pi/|q+G|^2`` (``q + G = 0`` -> 0) on an FFT box, built where ``xp`` lives."""

    def __init__(self, box_lattice, shape) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(box_lattice, dtype=float)).T

    def values(self, q, xp=np):
        q = np.asarray(q, dtype=float)
        m = [xp.asarray(np.fft.fftfreq(n, d=1.0 / n)) for n in self.shape]
        b = self.reciprocal
        gx = m[0][:, None, None] * b[0, 0] + m[1][None, :, None] * b[1, 0] + m[2][None, None, :] * b[2, 0] + q[0]
        gy = m[0][:, None, None] * b[0, 1] + m[1][None, :, None] * b[1, 1] + m[2][None, None, :] * b[2, 1] + q[1]
        gz = m[0][:, None, None] * b[0, 2] + m[1][None, :, None] * b[1, 2] + m[2][None, None, :] * b[2, 2] + q[2]
        squared = gx * gx + gy * gy + gz * gz
        return xp.where(squared > 1e-12, 8.0 * np.pi / xp.maximum(squared, 1e-300), 0.0)


def _mesh(lattice, mesh):
    """Representatives of a Gamma-centred mesh under time reversal: k (Cartesian), partner flags, mesh size."""

    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=float)).T
    fractions = [
        np.array([i / mesh[0], j / mesh[1], l / mesh[2]])
        for i in range(mesh[0])
        for j in range(mesh[1])
        for l in range(mesh[2])
    ]

    def same(a, b):
        d = a - b
        return bool(np.allclose(d - np.round(d), 0.0))

    representatives = []
    for f in fractions:
        if not any(same(f, o) or same(f, -o) for o in representatives):
            representatives.append(f - np.round(f))
    partners = np.array([not same(f, -f) for f in representatives])
    return np.array([f @ reciprocal for f in representatives]), partners, len(fractions)


class PeakTracker:
    """Peak bytes in use in the default memory pool of every device (an allocator that watches)."""

    def __init__(self, cp) -> None:
        self.cp = cp
        self.peak: dict[int, int] = {}
        pool = cp.get_default_memory_pool()

        def malloc(size):
            memory = pool.malloc(size)
            device = int(cp.cuda.Device().id)
            used = pool.used_bytes()
            if used > self.peak.get(device, 0):
                self.peak[device] = used
            return memory

        cp.cuda.set_allocator(malloc)

    def reset(self) -> None:
        self.peak = {}


def run_case(args) -> dict:
    import cupy as cp

    from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import (
        DevicePeriodicPairSolvers,
        apply_device_bloch_exchange,
        compress_device_bloch_exchange,
        device_bloch_channel,
    )
    from types import SimpleNamespace

    preset = dict(PRESETS[args.case]) if args.case else {}
    lattice = np.asarray(args.lattice if args.lattice else preset["lattice"], dtype=float).reshape(3, 3)
    box = tuple(args.box or preset["box"])
    fft = tuple(args.fft or preset.get("fft") or box)
    occupied = int(args.occupied or preset["occupied"])
    states = int(args.states or preset["states"])
    devices = tuple(range(min(args.devices, cp.cuda.runtime.getDeviceCount())))
    tracker = PeakTracker(cp)
    k_points, partners, mesh_size = _mesh(lattice, args.mesh)
    count = len(k_points)
    points = int(np.prod(box))
    volume_element = abs(np.linalg.det(lattice)) / points
    rng = np.random.default_rng(1)
    orbitals = []
    for rep in range(count):
        with cp.cuda.Device(devices[rep % len(devices)]):
            block = cp.asarray(rng.standard_normal((points, states)) + 1j * rng.standard_normal((points, states)))
            vectors, _ = cp.linalg.qr(block)
            orbitals.append(cp.ascontiguousarray(vectors))
    occupations = tuple(np.array([1.0] * occupied + [0.0] * (states - occupied)) for _ in range(count))
    stretch = np.array(fft, dtype=float) / np.array(box, dtype=float)
    kernel = DeviceCoulombKernel(lattice * stretch[:, None], fft)
    orbital_set = SimpleNamespace(
        k_points=k_points, weights=np.where(partners, 2.0, 1.0) / mesh_size, orbitals=tuple(orbitals),
        occupations=occupations, spin_weight=2.0, partners=partners, mesh_size=mesh_size,
    )
    channel = device_bloch_channel(orbital_set, volume_element)
    solvers = DevicePeriodicPairSolvers(kernel, SimpleNamespace(shape=box, flat=None), args.batch)
    for device in devices:
        cp.cuda.Device(device).synchronize()
    result = {
        "case": args.case or "custom", "box": box, "fft": fft, "mesh": list(args.mesh), "representatives": count,
        "mesh_size": mesh_size, "states": states, "occupied": occupied, "devices": len(devices),
        "batch": args.batch,
    }
    builds = []
    for attempt in range(2):
        tracker.reset()
        started = time.perf_counter()
        application = apply_device_bloch_exchange(channel, solvers, devices)
        for device in devices:
            cp.cuda.Device(device).synchronize()
        build = time.perf_counter() - started
        started = time.perf_counter()
        operators = compress_device_bloch_exchange(application, fraction=0.25)
        for device in devices:
            cp.cuda.Device(device).synchronize()
        ace = time.perf_counter() - started
        builds.append({
            "build_seconds": round(build, 4), "transform_seconds_max_device": round(application.poisson_seconds, 4),
            "ace_seconds": round(ace, 4), "pair_solves": application.pair_solves,
            "energy": application.energy,
            "peak_gib": {str(d): round(v / 2**30, 3) for d, v in sorted(tracker.peak.items())},
        })
        if attempt == 0:
            # one application of the term to the states of the first k-point (two ZGEMMs)
            with cp.cuda.Device(channel.devices[0]):
                vectors = channel.orbitals[0]
                operators[0].apply(vectors)
                cp.cuda.Device().synchronize()
                started = time.perf_counter()
                for _ in range(10):
                    operators[0].apply(vectors)
                cp.cuda.Device().synchronize()
                result["term_apply_seconds"] = round((time.perf_counter() - started) / 10, 6)
        for operator in operators:
            if operator is not None:
                operator.release()
    result["first"], result["warm"] = builds
    result["kernel_arrays_built"] = solvers.cache.built
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=sorted(PRESETS), default=None)
    parser.add_argument("--lattice", type=float, nargs=9, default=None)
    parser.add_argument("--box", type=int, nargs=3, default=None)
    parser.add_argument("--fft", type=int, nargs=3, default=None)
    parser.add_argument("--occupied", type=int, default=None)
    parser.add_argument("--states", type=int, default=None)
    parser.add_argument("--mesh", type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--small-build-points", type=float, default=None,
        help="replace SMALL_BUILD_POINTS (0: every build on every device)",
    )
    parser.add_argument("--chunk-bytes", type=float, default=None, help="replace CHUNK_BYTES")
    parser.add_argument(
        "--series", nargs="+", default=None,
        help="several preset cases in one process, each CASE:M0,M1,M2 (e.g. si-prim:4,4,4 hbn:6,6,1)",
    )
    args = parser.parse_args()
    if args.small_build_points is not None:
        from parsec_python.acceleration.Hamiltonian import bloch_exchange_device

        bloch_exchange_device.SMALL_BUILD_POINTS = float(args.small_build_points)
    if args.chunk_bytes is not None:
        from parsec_python.acceleration.Hamiltonian import bloch_exchange_device

        bloch_exchange_device.CHUNK_BYTES = int(args.chunk_bytes)
    if args.series:
        for item in args.series:
            name, mesh = item.split(":")
            case = argparse.Namespace(**vars(args))
            case.case, case.mesh = name, tuple(int(value) for value in mesh.split(","))
            print(json.dumps(run_case(case)), flush=True)
        return 0
    if args.case is None and args.box is None:
        parser.error("give --case or --box")
    print(json.dumps(run_case(args)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
