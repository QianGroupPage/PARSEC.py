#!/usr/bin/env python3
"""Seconds of one batch of pair solves of the periodic exchange by the FFT box edge (cuFFT, FP64).

The FFT box of a periodic exchange is the grid of the cell (a slab's padded
along z), so its edges are those of the grid: cuFFT is fastest for edges
with small prime factors.  For every edge ``n`` given this prints the
milliseconds of one batch of ``--batch`` complex pair solves (``fftn``, the
kernel product, ``ifftn``: the k-point path) and of real ones (``rfftn``,
``irfftn``: the Gamma point), averaged over ``--repeats`` after a warm-up.

Usage::

    PYTHONPATH=src python -m parsec_python.acceleration.benchmarks.exchange_fft_sizes 56 64 68 72 --batch 32
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("edges", type=int, nargs="+")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    import cupy as cp
    import cupyx.scipy.fft as cufft

    for n in args.edges:
        shape = (args.batch, n, n, n)
        kernel = cp.random.random((n, n, n))
        half = cp.ascontiguousarray(kernel[..., : n // 2 + 1])
        complex_box = cp.random.random(shape) + 1j * cp.random.random(shape)
        real_box = cp.random.random(shape)

        def complex_solve():
            values = cufft.fftn(complex_box, axes=(1, 2, 3), overwrite_x=True)
            values *= kernel[None]
            return cufft.ifftn(values, axes=(1, 2, 3), overwrite_x=True)

        def real_solve():
            values = cufft.rfftn(real_box, axes=(1, 2, 3))
            values *= half[None]
            return cufft.irfftn(values, s=(n, n, n), axes=(1, 2, 3), overwrite_x=True)

        result = {"edge": n, "batch": args.batch}
        for name, solve in (("complex_ms", complex_solve), ("real_ms", real_solve)):
            solve()
            cp.cuda.Device().synchronize()
            started = time.perf_counter()
            for _ in range(args.repeats):
                solve()
            cp.cuda.Device().synchronize()
            result[name] = round((time.perf_counter() - started) / args.repeats * 1.0e3, 3)
        print(json.dumps(result), flush=True)
        del complex_box, real_box
        cp.get_default_memory_pool().free_all_blocks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
