"""Run the inputs of this example with the accelerated command line, one process per input.

From the repository root (libxc's functionals need the library, see "Functionals of libxc" in
``src/parsec_python/README.md``)::

    export PARSEC_LIBXC_LIBRARY=.../libxc-7.1.2-cuda-sm80/lib/libxc.so
    python examples/0d_meta_gga/run_examples.py --out RUNS                  # every input
    python examples/0d_meta_gga/run_examples.py --out RUNS H2O O2/r2scan    # some of them
    python examples/0d_meta_gga/run_examples.py --list

Every input is copied to ``RUNS/<molecule>/<variant>/`` (with its ``occup.in``) and runs there as
``main.py parsec.in --pp-dir examples/0d_meta_gga --no-archive --gpus 1``; ``parsec.out`` (and ``parsec.dat``)
stay in that folder.  One JSON line per run goes to ``RUNS/runs.jsonl``: exit code, wall time of the process, the
peak device memory of the process and its children (``nvidia-smi``, where it exists) and the program's own wall
time from its log.  The reference logs ``parsec_python.out`` came from these commands on one A100; ``--install
RUNS`` copies the logs of a run folder into the example as ``parsec_python.out``, with the paths of the run
shortened to repository paths.  ``collect_results.py`` makes the tables of the README from the logs.

``examples/3d_meta_gga/run_examples.py`` runs this script on the inputs of that example.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]

# README order of the systems and variants of this example
SYSTEMS = ["H2O", "CH4", "NH3", "N2", "CO", "O2", "H2O_dist"]
VARIANTS = ["pbe", "scan", "r2scan", "tpss", "tpssh", "r2scan0", "pbe_h013", "r2scan_h013"]
_WALL = re.compile(r"Total (?:accelerated )?Python wall time \[sec\]\s*:\s*([0-9.]+)")


def cases(example: Path, systems: list[str], variants: list[str]) -> list[str]:
    """Every input of ``example`` as ``<system>/<variant>``, in the order of the README."""

    found = [path.parent.relative_to(example).as_posix() for path in example.glob("*/*/parsec.in")]

    def rank(case: str) -> tuple:
        system, variant = case.split("/")
        return (systems.index(system) if system in systems else len(systems), system,
                variants.index(variant) if variant in variants else len(variants), variant)

    return sorted(found, key=rank)


def select(every: list[str], names: list[str]) -> list[str]:
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


def _family(root: int) -> set[int]:
    """The process ``root`` and its descendants (``ps``; the root alone where ``ps`` fails)."""

    try:
        table = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {root}
    children: dict[int, list[int]] = {}
    for line in table.split("\n"):
        fields = line.split()
        if len(fields) == 2:
            children.setdefault(int(fields[1]), []).append(int(fields[0]))
    family, stack = set(), [root]
    while stack:
        pid = stack.pop()
        if pid not in family:
            family.add(pid)
            stack.extend(children.get(pid, []))
    return family


def _sample_memory(process: subprocess.Popen, peak: dict, stop: threading.Event) -> None:
    """Largest device memory (MiB) of the process and its children, from ``nvidia-smi`` every 0.5 s."""

    query = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    while not stop.is_set():
        try:
            table = subprocess.run(query, capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return
        family = _family(process.pid)
        used = 0
        for line in table.split("\n"):
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit() and int(fields[0]) in family:
                used += int(fields[1])
        peak["mib"] = max(peak.get("mib", 0), used)
        stop.wait(0.5)


def run(example: Path, case: str, out: Path, gpus: int, python: str) -> dict:
    source = example / case
    folder = out / case
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for name in ("parsec.in", "occup.in"):
        if (source / name).is_file():
            shutil.copy(source / name, folder / name)
    command = [python, str(REPOSITORY / "src" / "parsec_python" / "main.py"), str(folder / "parsec.in"),
               "--pp-dir", str(example), "--no-archive", "--gpus", str(gpus)]
    started = time.perf_counter()
    with open(folder / "stdout.txt", "w") as stdout:
        process = subprocess.Popen(command, cwd=folder, stdout=stdout, stderr=subprocess.STDOUT, text=True)
        peak: dict = {}
        stop = threading.Event()
        sampler = threading.Thread(target=_sample_memory, args=(process, peak, stop), daemon=True)
        sampler.start()
        code = process.wait()
        stop.set()
        sampler.join(timeout=15)
    seconds = time.perf_counter() - started
    log = folder / "parsec.out"
    walls = _WALL.findall(log.read_text()) if log.is_file() else []
    return {"case": case, "exit": code, "gpus": gpus, "seconds": round(seconds, 1),
            "program_seconds": float(walls[-1]) if walls else None,
            "peak_device_mib": peak.get("mib")}


def install(example: Path, every: list[str], source: Path) -> None:
    """Copy ``source/<case>/parsec.out`` into the example as ``parsec_python.out``, paths shortened."""

    name = example.relative_to(REPOSITORY).as_posix()
    for case in every:
        log = source / case / "parsec.out"
        if not log.is_file():
            continue
        text = log.read_text()
        # the run folder of the case and the example folder of the run (the pseudopotentials)
        text = re.sub(r"[^\s'\"(]*/" + re.escape(case) + "/", f"{name}/{case}/", text)
        text = re.sub(r"[^\s'\"(]*/" + re.escape(name) + "/", f"{name}/", text)
        # the libxc library of the machine: the variable that names it
        text = re.sub(r"[^\s'\"(]*/(libxc\.so|xc\.dll|libxc\.dylib)\b", r"$PARSEC_LIBXC_LIBRARY", text)
        (example / case / "parsec_python.out").write_text(text, newline="\n")
        print(f"installed {case}/parsec_python.out")


def main(argv: list[str], example: Path = HERE, systems: list[str] = SYSTEMS, variants: list[str] = VARIANTS) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help="<system> or <system>/<variant> (default: every input)")
    parser.add_argument("--out", type=Path, default=None, help="run folder: RUNS/<system>/<variant>/parsec.out")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs of every run (default 1)")
    parser.add_argument("--python", default=sys.executable, help="the Python that runs main.py")
    parser.add_argument("--list", action="store_true", help="list the inputs and stop")
    parser.add_argument("--install", type=Path, default=None, metavar="RUNS",
                        help="copy RUNS/<case>/parsec.out into the example as parsec_python.out and stop")
    arguments = parser.parse_args(argv)
    every = cases(example, systems, variants)
    if arguments.list:
        print("\n".join(every))
        return 0
    if arguments.install is not None:
        install(example, every, arguments.install)
        return 0
    if arguments.out is None:
        parser.error("--out RUNS is required to run")
    arguments.out.mkdir(parents=True, exist_ok=True)
    status = 0
    with open(arguments.out / "runs.jsonl", "a") as summary:
        for case in select(every, arguments.cases):
            record = run(example, case, arguments.out, arguments.gpus, arguments.python)
            print(json.dumps(record), flush=True)
            summary.write(json.dumps(record) + "\n")
            summary.flush()
            status = max(status, int(record["exit"] != 0))
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
