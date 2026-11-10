"""One bulk-diamond case on MPI ranks (one GPU each): the k-point pools of ``parsec_python.main --mpi``.

Start it on every rank (``srun -n R ... python run_mpi_kpoints.py ...``, one rank per GPU).  Rank 0 writes
<work dir>/parsec.in with make_diamond.py and copies C_POTRE.DAT; every rank then runs
``parsec_python.main parsec.in --mpi --device gpu`` in this process and samples the memory of its GPU
(nvidia-smi, every 0.5 s) and its own resident host memory (``host_rss_mib``); each rank writes
<work dir>/memory_rank<r>.json and prints one line ``RANK r`` with its peaks and wall time.

Usage (from the repository root, src on the python path):
    python examples/3d_diamond_benchmark/run_mpi_kpoints.py <n1> <n2> <n3> <work dir> [--kmesh K1 K2 K3]
        [--kpoints 'f1 f2 f3 w; ...'] [--set 'Keyword: value' ...] [--serial] [--pools P1,P2,...] [--sizes n,n,...]
        [-- <further arguments of parsec_python.main>]

``--kpoints`` writes a Kpoint_List in reciprocal-lattice coordinates with weights (Kpoint_Method: manual),
e.g. the irreducible points of a Gamma-centred 2x2x2 mesh of a cubic cell: '0 0 0 1; 0.5 0 0 3; 0.5 0.5 0 3;
0.5 0.5 0.5 1'.  ``--serial`` runs the same input in one process without MPI on one GPU (``--gpus 1``):
the one-node k-point solver, the reference of the pools.  ``--pools 4,1`` runs the input once for each
number of pools (``--kpoint-pools P``), one after the other in the same MPI processes, in <work dir>/P<P>;
``--sizes 3,4,5`` does so for each n x n x n cell in place of n1 n2 n3, in <work dir>/n<n>.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _rank_of_launch() -> tuple[int, int]:
    for rank, size in (("SLURM_PROCID", "SLURM_NTASKS"), ("PMI_RANK", "PMI_SIZE"), ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE")):
        if rank in os.environ:
            return int(os.environ[rank]), int(os.environ.get(size, "1"))
    return 0, 1


def _host_rss_mib() -> int:
    """The resident host memory of this process (VmRSS of /proc/self/status, MiB; 0 elsewhere)."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _sample(peaks: dict, stop: threading.Event) -> None:
    while not stop.is_set():
        peaks["host_rss_mib"] = max(peaks.get("host_rss_mib", 0), _host_rss_mib())
        try:
            text = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            for line in text.strip().splitlines():
                index, used = (int(value) for value in line.split(","))
                peaks[index] = max(peaks.get(index, 0), used)
        except Exception:
            pass
        stop.wait(0.5)


def _write_input(args, work: Path) -> None:
    make = [sys.executable, str(HERE / "make_diamond.py"), *map(str, args.n), str(work / "parsec.in")]
    if args.kmesh:
        make += ["--kmesh", *map(str, args.kmesh)]
    for setting in args.set:
        make += ["--set", setting]
    subprocess.run(make, check=True)
    if args.kpoints:
        rows = [row.split() for row in args.kpoints.split(";") if row.strip()]
        block = "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n"
        block += "".join("  " + " ".join(row) + "\n" for row in rows) + "end Kpoint_List\n"
        text = (work / "parsec.in").read_text()
        text = text.replace("Atom_Types_Num: 1", block + "Atom_Types_Num: 1", 1)
        (work / "parsec.in").write_text(text)
    shutil.copy(HERE / "C_POTRE.DAT", work / "C_POTRE.DAT")


def main() -> int:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs=3)
    parser.add_argument("work")
    parser.add_argument("--kmesh", type=int, nargs=3)
    parser.add_argument("--kpoints")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--pools")
    parser.add_argument("--sizes")
    args = parser.parse_args(argv)
    rank, size = _rank_of_launch()
    root = Path(args.work).resolve()
    cells = [tuple(args.n)] if not args.sizes else [(n, n, n) for n in (int(v) for v in args.sizes.split(","))]
    from parsec_python.acceleration.cli import main as parsec_main

    launch = ["--gpus", "1"] if args.serial else ["--mpi"]
    sweep = [None] if not args.pools else [int(value) for value in args.pools.split(",")]
    status = 0
    for cell in cells:
        work = root if not args.sizes else root / f"n{cell[0]}"
        args.n = list(cell)
        _ready(rank, work / f"input.ready.{size}", lambda: (work.mkdir(parents=True, exist_ok=True), _write_input(args, work)))
        for pools in sweep:
            folder = work if pools is None else work / f"P{pools}"
            if pools is not None:
                def copy(folder=folder, work=work):
                    folder.mkdir(exist_ok=True)
                    for name in ("parsec.in", "C_POTRE.DAT"):
                        shutil.copy(work / name, folder / name)

                _ready(rank, folder / "ready", copy)
            peaks, stop = {}, threading.Event()
            thread = threading.Thread(target=_sample, args=(peaks, stop), daemon=True)
            thread.start()
            started = time.perf_counter()
            os.chdir(folder)
            options = [] if pools is None else ["--kpoint-pools", str(pools)]
            code = parsec_main(["parsec.in", *launch, "--device", "gpu", *options, *extra])
            seconds = time.perf_counter() - started
            stop.set()
            thread.join()
            record = dict(
                rank=rank, ranks=size, cell=list(cell), pools=pools, exit=code, wall_seconds=round(seconds, 2), peak_mib=peaks
            )
            (folder / f"memory_rank{rank}.json").write_text(json.dumps(record, indent=1))
            print("RANK " + json.dumps(record), flush=True)
            status = status or int(code)
            _release_device_memory()  # the next run's peak starts from an empty pool
    return status


def _ready(rank: int, marker: Path, make) -> None:
    """Rank 0 makes the files and writes ``marker``; the other ranks wait for it."""
    if rank == 0:
        make()
        marker.write_text("ready")
        return
    deadline = time.time() + 600
    while not marker.exists():
        if time.time() > deadline:
            raise SystemExit(f"rank {rank}: no {marker}")
        time.sleep(0.5)
    time.sleep(1.0)


def _release_device_memory() -> None:
    try:
        from parsec_python.acceleration.Hartree import periodic_device
        from parsec_python.acceleration.V_xc import periodic_xc
        import cupy

        periodic_device.clear_cache()
        periodic_xc.clear_cache()
        import gc

        gc.collect()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001 - no CuPy: nothing to release
        pass


if __name__ == "__main__":
    sys.exit(main())
