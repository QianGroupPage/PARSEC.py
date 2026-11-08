#!/usr/bin/env python3
"""Run ``main.py`` on one input and report its wall time and the peak device memory of every GPU.

The accelerated command line (``parsec_python.acceleration.cli.main``) runs in
this process; a thread samples ``cudaMemGetInfo`` of every visible device
every 50 ms (memory in use by any context of the device, the CuPy pools'
cached blocks included) and the largest value is printed at the end, with the
program's wall time, as one JSON line.  Used for the timing runs of periodic
hybrids (``examples/periodic_hybrid_gpu``).

Usage::

    PYTHONPATH=src python -m parsec_python.acceleration.benchmarks.periodic_hybrid_run \
        examples/periodic_hybrid_gpu/si_k444/parsec.in --output out/si.out --log out/si.log --no-archive
"""

from __future__ import annotations

import json
import sys
import threading
import time


def _sampler(stop: threading.Event, peaks: dict, interval: float = 0.05) -> None:
    import cupy as cp

    count = cp.cuda.runtime.getDeviceCount()
    while True:
        for device in range(count):
            with cp.cuda.Device(device):
                free, total = cp.cuda.runtime.memGetInfo()
            used = int(total) - int(free)
            if used > peaks.get(device, 0):
                peaks[device] = used
        if stop.wait(interval):
            return


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from parsec_python.acceleration.cli import main as accelerated_main

    peaks: dict[int, int] = {}
    stop = threading.Event()
    thread = threading.Thread(target=_sampler, args=(stop, peaks), daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        code = accelerated_main(argv)
    finally:
        seconds = time.perf_counter() - started
        stop.set()
        thread.join(timeout=5)
    print(
        json.dumps(
            {
                "input": argv[0] if argv else None,
                "exit": code,
                "wall_seconds": round(seconds, 2),
                "peak_device_gib": {str(device): round(value / 2**30, 3) for device, value in sorted(peaks.items())},
            }
        ),
        flush=True,
    )
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
