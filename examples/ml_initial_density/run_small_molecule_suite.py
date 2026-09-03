"""Run and summarize the matched small-molecule initial-density cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys
import time


METHODS = ("sad", "scdp", "charge3net")


def _last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text, flags=re.IGNORECASE)
    return None if not values else float(values[-1])


def _summarize(case: Path, returncode: int, wall: float) -> dict[str, object]:
    log = case / "parsec.out"
    text = log.read_text(encoding="utf-8") if log.is_file() else ""
    iterations = re.findall(r"SCF iter #\s*([0-9]+)", text)
    return {
        "molecule": case.parent.name,
        "method": case.name,
        "status": (
            "converged"
            if "Self-consistency convergence achieved." in text
            else "failed" if returncode else "not_converged"
        ),
        "iterations": int(iterations[-1]) if iterations else "",
        "total_energy_ry": _last_float(r"Total Energy\s*=\s*([-+0-9.Ee]+)", text),
        "initial_density_seconds": _last_float(
            r"Initial valence-density setup\s*:\s*([-+0-9.Ee]+)", text
        ),
        "scf_seconds": _last_float(
            r"Time for self-consistent field \[sec\]\s*:\s*([-+0-9.Ee]+)", text
        ),
        "solver_wall_seconds": _last_float(
            r"Total accelerated Python wall time \[sec\]\s*:\s*([-+0-9.Ee]+)", text
        ),
        "process_wall_seconds": f"{wall:.6f}",
        "returncode": returncode,
    }


def main() -> int:
    example_root = Path(__file__).resolve().parent
    repository = example_root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", action="append", default=[])
    parser.add_argument("--method", action="append", choices=METHODS, default=[])
    parser.add_argument(
        "--backend", choices=("auto", "scipy", "native", "cupy"), default="auto"
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    molecules = arguments.molecule or sorted(
        item.name
        for item in (example_root / "small_molecules").iterdir()
        if item.is_dir()
    )
    methods = arguments.method or list(METHODS)
    launcher = repository / "src" / "parsec_python" / "main.py"
    pp_root = example_root / "pseudopotentials"
    rows: list[dict[str, object]] = []
    failed = False
    for molecule in molecules:
        for method in methods:
            case = example_root / "small_molecules" / molecule / method
            command = [
                sys.executable,
                str(launcher),
                str(case / "parsec.in"),
                "--pp-dir",
                str(pp_root),
                "--backend",
                arguments.backend,
                "--no-archive",
                "--quiet",
            ]
            if arguments.dry_run:
                command.append("--dry-run")
            print(f"[{molecule:14s} {method:10s}]", flush=True)
            started = time.perf_counter()
            completed = subprocess.run(command, cwd=repository, check=False)
            wall = time.perf_counter() - started
            if arguments.dry_run:
                failed |= completed.returncode != 0
                print(
                    f"  {'validated' if completed.returncode == 0 else 'failed'}; "
                    f"wall={wall:.2f} s",
                    flush=True,
                )
                continue
            row = _summarize(case, completed.returncode, wall)
            rows.append(row)
            failed |= completed.returncode != 0
            print(
                f"  {row['status']}; iterations={row['iterations']}; "
                f"wall={wall:.2f} s",
                flush=True,
            )

    if not arguments.dry_run and rows:
        output = example_root / "results_summary.csv"
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
