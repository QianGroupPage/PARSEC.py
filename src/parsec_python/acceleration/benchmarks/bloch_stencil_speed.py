"""Throughput of the local Bloch term on one GPU: cuSPARSE products against the kernels of bloch_stencil.

    python -m parsec_python.acceleration.benchmarks.bloch_stencil_speed [points per edge ...] [--columns 64]

For each cubic periodic grid (spacing 0.374 bohr, Expansion_Order 8, a general k-point, no atoms) the
apply of a block of ``columns`` complex columns and one step of the Chebyshev recurrence are timed:
the sparse route of DeviceBlochOperator and the kernel of :mod:`..Hamiltonian.bloch_stencil`.  Prints milliseconds and grid-point columns per second.
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("edges", type=int, nargs="*", default=[72, 108])
    parser.add_argument("--columns", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    import cupy as cp

    from parsec_python import PeriodicCell, PeriodicGridSettings
    from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter
    from parsec_python.Grid.pbc import build_periodic_grid
    from parsec_python.Laplacian import build_gradient, build_negative_laplacian

    spacing = 6.740652 / 18
    for edge in args.edges:
        grid = build_periodic_grid(
            PeriodicCell(lattice_vectors=edge * spacing * np.eye(3)),
            PeriodicGridSettings(spacing=spacing * (1 - 1e-9), expansion_order=8),
        )
        laplacian, gradient = build_negative_laplacian(grid), tuple(build_gradient(grid))
        k = np.array([0.23, 0.11, -0.07])
        host = SimpleNamespace(
            negative_laplacian=laplacian,
            gradient=gradient,
            k_point=k,
            _k_gradient=sum(value * matrix for value, matrix in zip(k, gradient)).tocsr(),
            _diagonal=np.random.default_rng(0).normal(size=grid.size) + float(k @ k),
            nonlocal_operator=SimpleNamespace(projectors=sp.csr_matrix((grid.size, 0)), signs=np.zeros(0)),
        )
        rng = np.random.default_rng(1)
        shape = (grid.size, args.columns)
        block = cp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
        previous = cp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
        sigma = cp.asarray(rng.normal(size=args.columns))
        results = {}
        for name in ("sparse", "kernel"):
            converter = DeviceBlochConverter(CupyArrayBackend(0))
            converter.fused_stencil = name != "sparse"
            operator = converter.operator(host)

            def step():
                if name == "sparse":
                    following = operator.apply(block)
                    following -= 1.3 * block
                    following *= 0.7
                    following -= sigma[None, :] * previous
                    following *= sigma[None, :]
                    return following
                return operator.chebyshev_step(block, previous, 1.3, 0.7, sigma, sigma)

            for label, call in (("apply", lambda: operator.apply(block)), ("step", step)):
                value = call()
                cp.cuda.Device().synchronize()
                started = time.perf_counter()
                for _ in range(args.repeats):
                    value = call()
                cp.cuda.Device().synchronize()
                seconds = (time.perf_counter() - started) / args.repeats
                results[(name, label)] = (seconds, value)
                print(
                    f"{edge:4d}^3 {name:7s} {label:6s} {seconds * 1e3:9.3f} ms "
                    f"{grid.size * args.columns / seconds / 1e9:7.2f} Gpt-col/s",
                    flush=True,
                )
        reference = results[("sparse", "step")][1]
        for name in ("kernel",):
            error = float(cp.max(cp.abs(results[(name, "step")][1] - reference)))
            print(f"{edge:4d}^3 {name} step max |difference| {error:.2e} (scale {float(cp.max(cp.abs(reference))):.2e})")


if __name__ == "__main__":
    main()
