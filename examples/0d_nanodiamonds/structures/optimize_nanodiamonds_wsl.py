"""Run the nanodiamond xTB optimization workflow with a Linux xTB in WSL.

This companion to :mod:`optimize_nanodiamonds` exists because precompiled
Windows/ifort xTB builds can exhaust their fixed process stack for large
systems.  It preserves the same GFN-FF -> GFN2-xTB methods, convergence levels,
geometry audits, and output layout while launching xTB under WSL with an
unlimited process stack.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess
import time

from ase.io import read, write

import optimize_nanodiamonds as common


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if len(drive) != 1 or not drive.isalpha():
        raise ValueError(f"cannot map non-drive Windows path into WSL: {resolved}")
    tail = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def _version(executable: str) -> str:
    completed = subprocess.run(
        ["wsl", "bash", "-lc", f"exec {shlex.quote(executable)} --version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return next(
        line.strip().lstrip("*").strip()
        for line in completed.stdout.splitlines()
        if "xtb version" in line
    )


def _run_stage(
    *,
    stage_root: Path,
    source_geometry: Path,
    executable: str,
    threads: int,
    method: str,
    opt_level: str,
) -> common.StageResult:
    stage_root.mkdir(parents=True, exist_ok=False)
    input_atoms = read(source_geometry, format="xyz")
    input_path = stage_root / "input.xyz"
    write(input_path, input_atoms, format="xyz", comment=f"input for {method}")
    log_path = stage_root / "xtb.out"
    if method == "GFN-FF":
        method_arguments = ["--gfnff"]
    elif method == "GFN2-xTB":
        method_arguments = ["--gfn", "2"]
    else:
        raise ValueError(f"unsupported method: {method}")
    scientific_arguments = [
        "input.xyz",
        *method_arguments,
        "--chrg",
        "0",
        "--uhf",
        "0",
        "--opt",
        opt_level,
    ]
    # OMP/MKL variables are set inside WSL because ordinary Windows variables
    # are not automatically exported through WSL interop.  shlex.quote makes
    # every path and scientific argument a single inert shell word.
    shell = "; ".join(
        [
            "ulimit -s unlimited",
            f"export OMP_NUM_THREADS={threads}",
            f"export MKL_NUM_THREADS={threads}",
            "export OMP_MAX_ACTIVE_LEVELS=1",
            f"cd {shlex.quote(_wsl_path(stage_root))}",
            "exec "
            + " ".join(
                shlex.quote(item) for item in [executable, *scientific_arguments]
            ),
        ]
    )
    command = [
        "wsl",
        "bash",
        "-lc",
        shell,
    ]
    print(f"    {method} --opt {opt_level} ({len(input_atoms)} atoms) ...", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    log_path.write_text(completed.stdout, encoding="utf-8")
    output_path = stage_root / "xtbopt.xyz"
    converged = "GEOMETRY OPTIMIZATION CONVERGED" in completed.stdout
    normal = "normal termination of xtb" in completed.stdout
    if completed.returncode != 0 or not converged or not normal or not output_path.is_file():
        raise RuntimeError(
            f"{method} failed in {stage_root}; return code={completed.returncode}, "
            f"converged={converged}, normal_termination={normal}. See {log_path}."
        )
    energy = common._last_number(completed.stdout, "TOTAL ENERGY")
    gradient = common._last_number(completed.stdout, "GRADIENT NORM")
    print(
        f"      converged in {wall_seconds:.2f} s; "
        f"E={energy if energy is not None else float('nan'):.12f} Eh",
        flush=True,
    )
    return common.StageResult(
        name=stage_root.name,
        method=method,
        opt_level=opt_level,
        input_path=input_path,
        output_path=output_path,
        log_path=log_path,
        command=tuple([executable, *scientific_arguments]),
        wall_seconds=wall_seconds,
        energy_hartree=energy,
        gradient_norm_hartree_per_bohr=gradient,
    )


def optimize_case(
    *,
    case_root: Path,
    executable: str,
    version: str,
    threads: int,
    preopt_min_atoms: int,
    gfnff_opt_level: str,
    gfn2_opt_level: str,
    vacuum_padding: float,
) -> dict[str, object]:
    optimized_root = case_root / "optimized"
    if optimized_root.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {optimized_root}")
    initial_path = case_root / "structure.xyz"
    initial = read(initial_path, format="xyz")
    print(f"  {case_root.name}: {len(initial)} atoms", flush=True)
    stages: list[common.StageResult] = []
    source = initial_path
    if len(initial) >= preopt_min_atoms:
        stage = _run_stage(
            stage_root=optimized_root / "01_gfnff",
            source_geometry=source,
            executable=executable,
            threads=threads,
            method="GFN-FF",
            opt_level=gfnff_opt_level,
        )
        stages.append(stage)
        source = stage.output_path
    stage = _run_stage(
        stage_root=optimized_root / ("02_gfn2" if stages else "01_gfn2"),
        source_geometry=source,
        executable=executable,
        threads=threads,
        method="GFN2-xTB",
        opt_level=gfn2_opt_level,
    )
    stages.append(stage)
    final = read(stage.output_path, format="xyz")
    return common._write_final_files(
        case_root=case_root,
        initial=initial,
        optimized=final,
        stage_results=stages,
        xtb_version=version,
        threads=threads,
        vacuum_padding_angstrom=vacuum_padding,
    )


def main() -> int:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xtb", required=True, help="absolute Linux path to xtb in WSL")
    parser.add_argument(
        "--root",
        type=Path,
        default=script_root,
        help="folder holding the case folders and the manifest (default: this script's folder)",
    )
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1) - 4))
    parser.add_argument("--preopt-min-atoms", type=int, default=50)
    parser.add_argument("--gfnff-opt-level", default="normal")
    parser.add_argument("--gfn2-opt-level", default="tight")
    parser.add_argument("--vacuum-padding", type=float, default=5.0)
    args = parser.parse_args()
    root = args.root.resolve()

    version = str(_version(args.xtb))
    print(f"Linux xTB {version}; OpenMP threads={args.threads}", flush=True)
    rows = []
    for case_name in args.case:
        rows.append(
            optimize_case(
                case_root=root / case_name,
                executable=args.xtb,
                version=version,
                threads=args.threads,
                preopt_min_atoms=args.preopt_min_atoms,
                gfnff_opt_level=args.gfnff_opt_level,
                gfn2_opt_level=args.gfn2_opt_level,
                vacuum_padding=args.vacuum_padding,
            )
        )

    manifest = root / "optimization_manifest.csv"
    merged: dict[str, dict[str, object]] = {}
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                merged[str(row["case"])] = dict(row)
    for row in rows:
        merged[str(row["case"])] = row
    ordered = [merged[key] for key in sorted(merged)]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(ordered)
    print(f"Optimization manifest: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
