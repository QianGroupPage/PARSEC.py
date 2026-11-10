"""Make one bulk-diamond case and run it with the full-SCF runner on every MPI rank of a launch, Gamma point.

Usage (from the repository root, src on the python path, started by srun as one rank per GPU, each rank seeing
its own GPU, as ``scripts/run_multi_gpu_scf.sh --rank-per-gpu`` starts them; or as one process for one GPU):
    python examples/3d_diamond_benchmark/run_mpi_case.py <n1> <n2> <n3> <work dir> [--extra-states N]
        [--set 'Keyword: value' ...] [--devices 0[,1,...]] [--no-launch-settings] [-- <further arguments of the runner>]

Every rank writes its own copy of the input (make_diamond.py) and of C_POTRE.DAT under
<work dir>/inputs/rank<r>, so that no rank waits for a file another one writes: the runner compares their
contents across the ranks. Then it runs the runner in a child process (which takes the rank of the launch),
``python -m parsec_python.acceleration.benchmarks.mpi_full_scf --input ... --output-dir <work dir>/run --devices 0``
(``--devices 0,1,2,3`` for one rank on the four GPUs of its node),
whose root writes parsec.out, timing.json (per-rank GPU memory peaks and stage times) and the result archive there;
afterwards rank 0 writes <work dir>/summary.json (summarize_timing.py) and the other ranks remove their input copies.
Unless --no-launch-settings is given, the environment settings of scripts/run_multi_gpu_scf.sh are applied first
(each only where the environment names none), read from that script, so that a run here takes the routes of a
launch through it.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def rank_of_launch() -> int:
    for name in ("SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None and value.strip().isdigit():
            return int(value)
    return 0


def launch_settings() -> dict[str, str]:
    """``NAME=VALUE`` of every ``export NAME="${NAME:-VALUE}"`` line of scripts/run_multi_gpu_scf.sh, with its
    thread settings (OMP_PROC_BIND=false and no OMP_PLACES, as the script sets them)."""
    text = (ROOT / "scripts" / "run_multi_gpu_scf.sh").read_text()
    settings = dict(re.findall(r'^export (\w+)="\$\{\1:-([^}]*)\}"$', text, flags=re.MULTILINE))
    settings["OMP_PROC_BIND"] = "false"
    return settings


def main() -> int:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs=3)
    parser.add_argument("work")
    parser.add_argument("--extra-states", type=int)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--devices", default="0")
    parser.add_argument("--no-launch-settings", action="store_true")
    args = parser.parse_args(argv)
    work = Path(args.work)
    inputs = work / "inputs" / f"rank{rank_of_launch()}"
    inputs.mkdir(parents=True, exist_ok=True)
    make = [sys.executable, str(HERE / "make_diamond.py"), *map(str, args.n), str(inputs / "parsec.in")]
    if args.extra_states is not None:
        make += ["--extra-states", str(args.extra_states)]
    for setting in args.set:
        make += ["--set", setting]
    subprocess.run(make, check=True, stdout=subprocess.DEVNULL if rank_of_launch() else None)
    shutil.copy(HERE / "C_POTRE.DAT", inputs / "C_POTRE.DAT")
    environment = dict(os.environ)
    if not args.no_launch_settings:
        for name, value in launch_settings().items():
            if name == "OMP_PROC_BIND":
                environment[name] = value
            else:
                environment.setdefault(name, value)
        environment.pop("OMP_PLACES", None)
    command = [sys.executable, "-m", "parsec_python.acceleration.benchmarks.mpi_full_scf",
               "--input", str(inputs / "parsec.in"), "--output-dir", str(work / "run"), "--devices", args.devices, *extra]
    # A child process: it inherits the rank of the launch (the PMI variables of srun) and initializes MPI itself.
    code = subprocess.run(command, env=environment).returncode
    if rank_of_launch() == 0:
        timing = work / "run" / "timing.json"
        if timing.exists():
            # The numbers of the run in a small file beside it (summarize_timing.py).
            subprocess.run([sys.executable, str(HERE / "summarize_timing.py"), str(timing),
                            "--json", str(work / "summary.json")], stdout=subprocess.DEVNULL)
    else:
        # The copies of the other ranks have served their comparison.
        shutil.rmtree(inputs, ignore_errors=True)
        if code == 3:
            # 3 is an SCF that ran to Max_Iter (relaxation_exit_code): rank 0 reports it. Another rank that
            # left with it first would let srun stop rank 0 before its summary.
            return 0
    return code


if __name__ == "__main__":
    raise SystemExit(main())
