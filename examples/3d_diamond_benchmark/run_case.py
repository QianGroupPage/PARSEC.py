"""Make one bulk-diamond case with make_diamond.py and run it with the accelerated CLI, sampling GPU memory.

Usage (from the repository root, src on the python path):
    python examples/3d_diamond_benchmark/run_case.py <n1> <n2> <n3> <work dir> [--kmesh K1 K2 K3] [--gpus G]
        [--extra-states N] [--set 'Keyword: value' ...] [-- <further arguments of parsec_python.main>]

Writes <work dir>/parsec.in and C_POTRE.DAT, runs ``python -m parsec_python.main parsec.in --device gpu --gpus G``
there, and writes <work dir>/memory.json: the largest used memory of every visible GPU (nvidia-smi, every 0.5 s) and
the wall time of the command.
"""

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sample(peaks, stop):
    while not stop.is_set():
        try:
            text = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10).stdout
            for line in text.strip().splitlines():
                index, used = (int(value) for value in line.split(","))
                peaks[index] = max(peaks.get(index, 0), used)
        except Exception:
            pass
        stop.wait(0.5)


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs=3)
    parser.add_argument("work")
    parser.add_argument("--kmesh", type=int, nargs=3)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--extra-states", type=int)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    make = [sys.executable, str(HERE / "make_diamond.py"), *map(str, args.n), str(work / "parsec.in")]
    if args.kmesh:
        make += ["--kmesh", *map(str, args.kmesh)]
    if args.extra_states is not None:
        make += ["--extra-states", str(args.extra_states)]
    for setting in args.set:
        make += ["--set", setting]
    subprocess.run(make, check=True)
    shutil.copy(HERE / "C_POTRE.DAT", work / "C_POTRE.DAT")
    peaks, stop = {}, threading.Event()
    thread = threading.Thread(target=sample, args=(peaks, stop), daemon=True)
    thread.start()
    start = time.perf_counter()
    command = [sys.executable, "-m", "parsec_python.main", "parsec.in", "--device", "gpu", "--gpus", str(args.gpus),
               *extra]
    code = subprocess.run(command, cwd=work).returncode
    seconds = time.perf_counter() - start
    stop.set()
    thread.join()
    (work / "memory.json").write_text(json.dumps(dict(exit=code, wall_seconds=seconds, peak_mib=peaks), indent=1))
    print(json.dumps(dict(exit=code, wall_seconds=round(seconds, 1), peak_mib=peaks)))
    sys.exit(code)


if __name__ == "__main__":
    main()
