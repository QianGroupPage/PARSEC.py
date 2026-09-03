"""Compare CHEBDAV first solves with the preserved CHEBFF suite outputs.

Only the first nonlinear eigensolve differs: both paths reuse the same GPU
SUBSPACE filtering implementation on later SCF iterations.  Trial inputs and
logs live in a temporary directory, so the canonical ``parsec.in`` and
``parsec.out`` files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


METHODS = ("sad", "scdp", "charge3net")
DEFAULT_MOLECULES = ("CH4", "CO2", "C2H6", "C3H8O", "C10H8")


def _last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text, flags=re.IGNORECASE)
    return None if not values else float(values[-1])


def _metrics(text: str) -> dict[str, object]:
    iterations = re.findall(r"SCF iter #\s*([0-9]+)", text)
    return {
        "status": (
            "converged"
            if "Self-consistency convergence achieved." in text
            else "not_converged"
        ),
        "iterations": int(iterations[-1]) if iterations else "",
        "total_energy_ry": _last_float(r"Total Energy\s*=\s*([-+0-9.Ee]+)", text),
        "initial_eigensolver_seconds": _last_float(
            r"GPU initial-eigensolver synchronized time \[sec\]\s*=\s*([-+0-9.Ee]+)",
            text,
        ),
        "diagonalization_seconds": _last_float(
            r"Diagonalization subtotal\s*:\s*([-+0-9.Ee]+)", text
        ),
        "scf_seconds": _last_float(
            r"Time for self-consistent field \[sec\]\s*:\s*([-+0-9.Ee]+)", text
        ),
        "solver_wall_seconds": _last_float(
            r"Total accelerated Python wall time \[sec\]\s*:\s*([-+0-9.Ee]+)",
            text,
        ),
    }


def _trial_input(text: str, case: Path, degree: int) -> str:
    updated, count = re.subn(
        r"(?im)^Eigensolver:\s*\S+\s*$", "Eigensolver: chebdav", text
    )
    if count != 1:
        raise ValueError(f"expected one Eigensolver label in {case / 'parsec.in'}")
    updated, count = re.subn(
        r"(?im)^Chebdav_Degree:\s*\d+\s*$",
        f"Chebdav_Degree: {degree}",
        updated,
    )
    if count != 1:
        raise ValueError(f"expected one Chebdav_Degree label in {case / 'parsec.in'}")
    density = case.parent / "densities" / f"{case.name}.npz"
    if case.name in {"scdp", "charge3net"}:
        updated, count = re.subn(
            r"(?im)^ML_Density_File:\s*.*$",
            lambda _match: f"ML_Density_File: {density.resolve()}",
            updated,
        )
        if count != 1:
            raise ValueError(f"expected one ML_Density_File label in {case / 'parsec.in'}")
    return updated


def main() -> int:
    example_root = Path(__file__).resolve().parent
    repository = example_root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", action="append", default=[])
    parser.add_argument("--method", action="append", choices=METHODS, default=[])
    parser.add_argument("--degree", type=int, default=15)
    parser.add_argument(
        "--output", type=Path, default=example_root / "chebdav_trial.csv"
    )
    arguments = parser.parse_args()
    if arguments.degree < 15:
        parser.error("PARSEC CHEBDAV requires --degree >= 15")

    molecules = arguments.molecule or list(DEFAULT_MOLECULES)
    methods = arguments.method or list(METHODS)
    launcher = repository / "src" / "parsec_python" / "main.py"
    pp_root = example_root / "pseudopotentials"
    work_parent = repository / ".tmp"
    work_parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failed = False
    with tempfile.TemporaryDirectory(
        prefix="ml-eigensolver-", dir=work_parent
    ) as directory:
        work = Path(directory)
        for molecule in molecules:
            for method in methods:
                case = example_root / "small_molecules" / molecule / method
                baseline_text = (case / "parsec.out").read_text(encoding="utf-8")
                baseline = _metrics(baseline_text)
                trial_input = work / "parsec.in"
                trial_input.write_text(
                    _trial_input(
                        (case / "parsec.in").read_text(encoding="utf-8"),
                        case,
                        arguments.degree,
                    ),
                    encoding="utf-8",
                )
                print(f"[{molecule:8s} {method:10s} CHEBDAV-{arguments.degree}]", flush=True)
                started = time.perf_counter()
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(launcher),
                        str(trial_input),
                        "--pp-dir",
                        str(pp_root),
                        "--backend",
                        "auto",
                        "--no-archive",
                        "--quiet",
                    ],
                    cwd=repository,
                    check=False,
                )
                process_wall = time.perf_counter() - started
                trial_log = work / "parsec.out"
                trial = _metrics(
                    trial_log.read_text(encoding="utf-8") if trial_log.is_file() else ""
                )
                row = {
                    "molecule": molecule,
                    "method": method,
                    "candidate": f"chebdav-{arguments.degree}",
                    "baseline_status": baseline["status"],
                    "candidate_status": trial["status"],
                    "baseline_iterations": baseline["iterations"],
                    "candidate_iterations": trial["iterations"],
                    "baseline_energy_ry": baseline["total_energy_ry"],
                    "candidate_energy_ry": trial["total_energy_ry"],
                    "energy_difference_ry": (
                        None
                        if baseline["total_energy_ry"] is None
                        or trial["total_energy_ry"] is None
                        else float(trial["total_energy_ry"])
                        - float(baseline["total_energy_ry"])
                    ),
                    "baseline_initial_eigensolver_seconds": baseline[
                        "initial_eigensolver_seconds"
                    ],
                    "candidate_initial_eigensolver_seconds": trial[
                        "initial_eigensolver_seconds"
                    ],
                    "baseline_diagonalization_seconds": baseline[
                        "diagonalization_seconds"
                    ],
                    "candidate_diagonalization_seconds": trial[
                        "diagonalization_seconds"
                    ],
                    "baseline_solver_wall_seconds": baseline["solver_wall_seconds"],
                    "candidate_solver_wall_seconds": trial["solver_wall_seconds"],
                    "candidate_process_wall_seconds": f"{process_wall:.6f}",
                    "returncode": completed.returncode,
                }
                rows.append(row)
                failed |= completed.returncode != 0 or trial["status"] != "converged"
                print(
                    f"  iter {baseline['iterations']} -> {trial['iterations']}; "
                    f"solver wall {baseline['solver_wall_seconds']} -> "
                    f"{trial['solver_wall_seconds']} s; {trial['status']}",
                    flush=True,
                )
                if trial_log.is_file():
                    trial_log.unlink()

    output = arguments.output
    if not output.is_absolute():
        output = (repository / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
