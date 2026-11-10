"""Run the inputs of this example with the accelerated command line, one process per input.

From the repository root::

    python examples/3d_hybrid_functionals/run_examples.py                  # every input
    python examples/3d_hybrid_functionals/run_examples.py Si_k222 hBN_k661/pbe0
    python examples/3d_hybrid_functionals/run_examples.py --list

Each input runs as ``main.py <input> --pp-dir examples/3d_hybrid_functionals --no-archive`` (through
``parsec_python.acceleration.benchmarks.periodic_hybrid_run``, which adds the wall time and the peak memory of
every GPU of the process) with ``--gpus`` of the table below: one GPU for every input but the 64-atom cell, which
takes four.  ``parsec.out`` is written beside the input (``--out DIR`` writes ``DIR/<case>/parsec.out``
instead), and one JSON line per run goes to ``runs.jsonl`` there.  The reference logs ``parsec_python.out`` came
from these commands on one A100 node; ``--install DIR`` copies the logs of a run directory into the example as
``parsec_python.out``, with the paths of the run shortened to repository paths.  ``collect_results.py`` makes
the tables of the README from the logs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
EXAMPLE = HERE.relative_to(REPOSITORY).as_posix()

# GPUs of the reference runs (an A100 node has four); every other input takes one.
GPUS = {"Si64_k222": 4}


def cases() -> list[str]:
    """Every input of the example as ``<system>/<variant>``, in the order of the README."""

    systems = ["Si_k222", "Si_k444", "Si_disp_k222", "C_k333", "MgO_k333", "hBN_k661", "O2_box", "Si64_gamma", "Si64_k222"]
    variants = ["pbe", "pbe0", "hse06", "hse06_default", "hf", "b3lyp", "pbe0_periodic", "hse06_periodic"]
    found = [path.parent.relative_to(HERE).as_posix() for path in HERE.glob("*/*/parsec.in")]

    def rank(case: str) -> tuple:
        system, variant = case.split("/")
        return (systems.index(system) if system in systems else len(systems), system,
                variants.index(variant) if variant in variants else len(variants), variant)

    return sorted(found, key=rank)


def select(names: list[str]) -> list[str]:
    every = cases()
    if not names:
        return every
    chosen: list[str] = []
    for name in names:
        name = name.strip("/").replace("\\", "/")
        matches = [case for case in every if case == name or case.startswith(name + "/")]
        if not matches:
            raise SystemExit(f"run_examples: no input {name!r} (see --list)")
        chosen.extend(case for case in matches if case not in chosen)
    return chosen


def run(case: str, out: Path | None, gpus: str | None, python: str) -> dict:
    folder = HERE / case
    log_dir = folder if out is None else out / case
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "parsec.out"
    count = gpus or str(GPUS.get(case.split("/")[0], 1))
    command = [
        python, "-m", "parsec_python.acceleration.benchmarks.periodic_hybrid_run",
        str(folder / "parsec.in"), "--pp-dir", str(HERE), "--log", str(log), "--no-archive", "--gpus", count,
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=folder, capture_output=True, text=True,
                               env=_environment())
    seconds = time.perf_counter() - started
    (log_dir / "stdout.txt").write_text(completed.stdout + completed.stderr)
    record: dict = {"case": case, "exit": completed.returncode, "gpus": int(count), "seconds": round(seconds, 1)}
    for line in completed.stdout.splitlines():
        if line.startswith("{") and '"wall_seconds"' in line:
            measured = json.loads(line)
            record["wall_seconds"] = measured["wall_seconds"]
            record["peak_device_gib"] = measured["peak_device_gib"]
    return record


def _environment() -> dict:
    import os

    environment = dict(os.environ)
    source = str(REPOSITORY / "src")
    environment["PYTHONPATH"] = source + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    return environment


def install(source: Path) -> None:
    """Copy ``source/<case>/parsec.out`` into the example as ``parsec_python.out``, paths shortened."""

    for case in cases():
        log = source / case / "parsec.out"
        if not log.is_file():
            continue
        text = log.read_text()
        # the files of the example (inputs, pseudopotentials) wherever the repository was, then the log folder of
        # the run: any other path that ends in <case>/  (POSIX paths: the reference logs come from Linux nodes)
        text = re.sub(r"[^\s'\"]*/" + re.escape(EXAMPLE) + "/", EXAMPLE + "/", text)
        text = re.sub(r"(?<![\w/])/[^\s'\"]*/" + re.escape(case) + "/", f"{EXAMPLE}/{case}/", text)
        left = sorted(set(re.findall(r"(?:^|\s)(/[^\s'\"]+)", text, re.M)))
        if left:
            print(f"{case}: absolute paths left in the log: {left[:5]}")
        (HERE / case / "parsec_python.out").write_text(text, newline="\n")
        print(f"installed {case}/parsec_python.out")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help="<system> or <system>/<variant> (default: every input)")
    parser.add_argument("--out", type=Path, default=None, help="write the logs to DIR/<case>/parsec.out")
    parser.add_argument("--gpus", default=None, help="GPUs of every run (default: the table of this script)")
    parser.add_argument("--python", default=sys.executable, help="the Python that runs main.py")
    parser.add_argument("--list", action="store_true", help="list the inputs and stop")
    parser.add_argument("--install", type=Path, default=None, metavar="DIR",
                        help="copy DIR/<case>/parsec.out into the example as parsec_python.out and stop")
    arguments = parser.parse_args(argv)
    if arguments.list:
        for case in cases():
            print(f"{case:28s} {GPUS.get(case.split('/')[0], 1)} GPU(s)")
        return 0
    if arguments.install is not None:
        install(arguments.install)
        return 0
    selected = select(arguments.cases)
    summary_dir = HERE if arguments.out is None else arguments.out
    summary_dir.mkdir(parents=True, exist_ok=True)
    status = 0
    with open(summary_dir / "runs.jsonl", "a") as summary:
        for case in selected:
            record = run(case, arguments.out, arguments.gpus, arguments.python)
            print(json.dumps(record), flush=True)
            summary.write(json.dumps(record) + "\n")
            summary.flush()
            status = max(status, int(record["exit"] != 0))
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
