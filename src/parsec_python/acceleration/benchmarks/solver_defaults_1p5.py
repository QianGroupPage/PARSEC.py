"""PARSEC 1.5's eigensolver defaults, one at a time and together, on the command lines of this port.

PARSEC 1.5 changed five defaults of the eigensolvers and the SCF control (``usrinputfile.F90``,
``parsec.F90`` and ``orth_normal.f90z`` of 1.5): Eigensolver chebdav -> chebff, FF_MaxIter 2 -> 3,
Convergence_Criterion_Approach 100 -> 2 times the criterion, the later filter degree lowered from
iteration 5 (was 6) on, and the Gram-Schmidt reorthogonalization threshold 0.1 -> 0.68.  This script
runs each change alone and all together, for several start vectors (``Vector_Rng_Seed``), against a
tightly converged reference of the same input, and writes one JSON record per run and a summary
table: iterations, matrix-vector products (the "Orbital vectors applied" of the accelerated driver,
or the columns the readable Hamiltonian applied), total energy and eigenvalues against the reference.
Times are recorded but not meant to be compared when the node is shared.

    python -m parsec_python.acceleration.benchmarks.solver_defaults_1p5 OUT CASE [--cli accelerated|reference]
        [--variants base,chebff_ff2,...] [--seeds 0,1,2] [--source parsec.in] [--gpus N] [--keep]

CASE is h2o, h2o_small, si5h12, si28h36, si8 (bulk Si8 at Gamma), si8_k222 (Si8 on a 2x2x2 Monkhorst-Pack grid)
or ``input`` with ``--source`` (a parsec.in whose pseudopotentials sit beside it).  Seed 0 is the
canonical start basis (no Vector_Rng_Seed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

_H2O = """Boundary_Sphere_Radius: 12
Grid_Spacing: 0.30
Eigensolver: chebdav
States_Num: 8
Convergence_Criterion: 1e-5
Atom_Types_Num: 2
Atom_Type: O
Local_Component: p
begin Atom_Coord
  0.000000000 0.000000000 0.000000000
end Atom_Coord
Atom_Type: H
Local_Component: s
begin Atom_Coord
  1.600000000 0.400000000 0.000000000
 -1.600000000 0.400000000 0.000000000
end Atom_Coord
"""

# Si5H12 (5 Si + 12 H, 32 electrons, the case compared against the PARSEC 1.5 binary), with this
# port's defaults (symmetry on, CHEBDAV first solve).
_SI5H12 = """Boundary_Sphere_Radius: 12.0
Grid_Spacing: 0.50
Correlation_Type: ca
States_Num: 20
Fermi_Temp: 80.0
Max_Iter: 100
Convergence_Criterion: 1e-6
Chebdav_Degree: 20
Chebyshev_Degree: 15
Mixing_Method: anderson
Atom_Types_Num: 2
Atom_Type: Si
Local_Component: p
begin Atom_Coord
 -2.565303 -2.565303  2.565303
 -2.565303  2.565303 -2.565303
  2.565303 -2.565303 -2.565303
  0.000000  0.000000  0.000000
  2.565303  2.565303  2.565303
end Atom_Coord
Atom_Type: H
Local_Component: s
begin Atom_Coord
 -3.096273 -4.935608  1.256938
 -4.935614 -1.256935  3.096270
 -1.256932 -3.096276  4.935611
 -3.096273  4.935608 -1.256938
 -4.935614  1.256935 -3.096270
 -1.256932  3.096276 -4.935611
  3.096273 -1.256938 -4.935608
  1.256932 -4.935611 -3.096276
  4.935614 -3.096270 -1.256935
  3.096273  1.256938  4.935608
  1.256932  4.935611  3.096276
  4.935614  3.096270  1.256935
end Atom_Coord
"""

CASES = {
    "h2o": dict(text=_H2O, pseudopotentials=("examples/0d_water_geoopt/O_POTRE.DAT", "examples/0d_water_geoopt/H_POTRE.DAT")),
    # A coarser water for the readable command line (about 1/4 of the points of h2o).
    "h2o_small": dict(
        text=_H2O.replace("Boundary_Sphere_Radius: 12", "Boundary_Sphere_Radius: 10").replace(
            "Grid_Spacing: 0.30", "Grid_Spacing: 0.40"
        ),
        pseudopotentials=("examples/0d_water_geoopt/O_POTRE.DAT", "examples/0d_water_geoopt/H_POTRE.DAT"),
    ),
    "si5h12": dict(text=_SI5H12, pseudopotentials=("examples/0d_Si28H36/Si_POTRE.DAT", "examples/0d_Si28H36/H_POTRE.DAT")),
    "si28h36": dict(source="examples/0d_Si28H36/parsec.in"),
    "si8": dict(source="examples/3d_Si/parsec.in"),
    "si8_k222": dict(
        source="examples/3d_Si/parsec.in",
        append="begin Monkhorst_Pack_Grid\n  2 2 2\nend Monkhorst_Pack_Grid\nKpoint_Method: mp\n",
    ),
}

#: The 1.5 changes: label -> value (None removes the label); "{approach}" is 2 x the criterion.
VARIANTS = {
    "base": {},
    "chebdav": {"eigensolver": "chebdav", "ff_maxiter": None},
    "chebff_ff2": {"eigensolver": "chebff", "ff_maxiter": "2"},
    "chebff_ff3": {"eigensolver": "chebff", "ff_maxiter": "3"},
    "ff3": {"ff_maxiter": "3"},
    "approach2x": {"convergence_criterion_approach": "{approach}"},
    "decrease4": {"chebyshev_degree_decrease_after": "4"},
    "ff3_decrease4": {"ff_maxiter": "3", "chebyshev_degree_decrease_after": "4"},
    "all15": {
        "eigensolver": "chebff",
        "ff_maxiter": "3",
        "convergence_criterion_approach": "{approach}",
        "chebyshev_degree_decrease_after": "4",
    },
    "reorth068": {"reorthogonalization_threshold": "0.68"},
    "all15_reorth": {
        "eigensolver": "chebff",
        "ff_maxiter": "3",
        "convergence_criterion_approach": "{approach}",
        "chebyshev_degree_decrease_after": "4",
        "reorthogonalization_threshold": "0.68",
    },
}

_LABEL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*[:=]?")


def _label(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.lower().startswith(("begin", "end")):
        return None
    match = _LABEL.match(line)
    return match.group(1).lower() if match else None


def _criterion(text: str) -> float:
    """``Convergence_Criterion`` in Ry (default 2e-4 Ry; an eV or Ha unit is converted)."""

    for line in text.splitlines():
        if _label(line) == "convergence_criterion":
            value = line.split(":", 1)[1] if ":" in line else line.split(None, 1)[1]
            tokens = value.split()
            number = float(tokens[0].lower().replace("d", "e"))
            unit = tokens[1].lower() if len(tokens) > 1 else "ry"
            return number * {"ry": 1.0, "ev": 1.0 / 13.605693122994, "ha": 2.0, "hartree": 2.0}.get(unit, 1.0)
    return 2.0e-4


def edit_input(text: str, changes: dict[str, str | None]) -> str:
    """``text`` with the labels of ``changes`` removed and the given values appended."""

    lines = [line for line in text.splitlines() if _label(line) not in changes]
    for key, value in changes.items():
        if value is not None:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def _variant_changes(name: str, criterion: float, seed: int, tight: bool) -> dict[str, str | None]:
    changes = {key: (value.format(approach=f"{2.0 * criterion:.6e} Ry") if value else value)
               for key, value in VARIANTS[name].items()}
    if seed:
        changes["vector_rng_seed"] = str(seed)
    if tight:
        changes["convergence_criterion"] = f"{min(criterion * 1.0e-3, 1.0e-7):.3e}"
        changes["max_iter"] = "300"
    return changes


def _case_text(case: str, source: Path | None) -> tuple[str, list[Path]]:
    if case == "input":
        if source is None:
            raise SystemExit("CASE input needs --source")
        directory = source.parent
        return source.read_text(), sorted(directory.glob("*_POT*.DAT")) + sorted(directory.glob("*.psp8"))
    spec = CASES[case]
    if "text" in spec:
        return spec["text"], [ROOT / item for item in spec["pseudopotentials"]]
    path = ROOT / spec["source"]
    text = path.read_text() + spec.get("append", "")
    return text, sorted(path.parent.glob("*_POT*.DAT"))


def _child(mode: str, input_path: Path, extra: list[str]) -> int:
    """Run one calculation in this process (``--child``); count the columns the readable H applied."""

    counts = {"columns": 0, "calls": 0}
    if mode == "reference":
        from parsec_python.Hamiltonian.operator import KohnShamHamiltonian

        original = KohnShamHamiltonian.apply

        def counted(self, vectors):
            array = np.asarray(vectors)
            counts["columns"] += 1 if array.ndim == 1 else int(array.shape[1])
            counts["calls"] += 1
            return original(self, vectors)

        KohnShamHamiltonian.apply = counted
        from parsec_python.cli import main
    else:
        from parsec_python.acceleration.cli import main
    started = time.perf_counter()
    code = main([str(input_path), *extra])
    seconds = time.perf_counter() - started
    (input_path.parent / "child_counts.json").write_text(
        json.dumps({"code": code, "seconds": seconds, **counts})
    )
    return int(code or 0)


def _summarize(run_dir: Path) -> dict:
    record: dict = {}
    counts_path = run_dir / "child_counts.json"
    if counts_path.exists():
        record.update(json.loads(counts_path.read_text()))
    archive = run_dir / "parsec_python_results.npz"
    if not archive.exists():
        record["error"] = "no result archive"
        log = run_dir / "child.log"
        if log.exists():
            record["log_tail"] = log.read_text(errors="replace").splitlines()[-15:]
        return record
    with np.load(archive, allow_pickle=False) as data:
        record["converged"] = bool(data["converged"])
        record["iterations"] = int(data["iterations"])
        record["total_energy_ry"] = float(data["energy_total_ry"])
        record["eigenvalues_ry"] = [float(x) for x in np.ravel(data["eigenvalues_ry"])]
        record["occupations"] = [float(x) for x in np.ravel(data["occupations"])]
        for key in (
            "vectors_applied",
            "applications",
            "eigensolver_first_calls",
            "eigensolver_subspace_calls",
            "eigensolver_first_seconds",
            "eigensolver_subspace_seconds",
        ):
            name = f"backend_statistic_{key}"
            if name in data.files:
                record[key] = float(data[name])
        history = np.asarray(data["scf_history"])
        if history.size:
            record["sre_history"] = [float(x) for x in history[:, 1]]
        timing = [name for name in data.files if name.startswith("timing_scf_")]
        record["scf_timings"] = {name[len("timing_scf_"):]: float(data[name]) for name in timing if data[name].ndim == 0}
    if record.get("columns"):
        record["products"] = record["columns"]
    elif "vectors_applied" in record:
        record["products"] = record["vectors_applied"]
    return record


def _compare(record: dict, reference: dict) -> None:
    if "total_energy_ry" not in record or "total_energy_ry" not in reference:
        return
    record["delta_energy_ry"] = record["total_energy_ry"] - reference["total_energy_ry"]
    values = np.asarray(record["eigenvalues_ry"])
    reference_values = np.asarray(reference["eigenvalues_ry"])
    occupations = np.asarray(reference["occupations"])
    count = min(values.size, reference_values.size)
    difference = np.abs(values[:count] - reference_values[:count])
    occupied = occupations[:count] > 1.0e-6
    record["max_delta_eigenvalue_ry"] = float(difference.max(initial=0.0))
    record["max_delta_occupied_eigenvalue_ry"] = float(difference[occupied].max(initial=0.0))


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--child"]:
        return _child(arguments[1], Path(arguments[2]), arguments[3:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("case", choices=sorted([*CASES, "input"]))
    parser.add_argument("--cli", choices=("accelerated", "reference"), default="accelerated")
    parser.add_argument("--variants", default="base,chebdav,chebff_ff2,chebff_ff3,approach2x,decrease4,all15")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--gpus", default=None, help="--gpus of the accelerated command line")
    parser.add_argument("--label", default=None, help="name of the case in the records (default CASE)")
    parser.add_argument("--no-reference", action="store_true", help="skip the tight reference run")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="LABEL=VALUE",
        help="a label of every run (e.g. Use_Fortran_Routines=.true. for a periodic case: an option the "
        "default routines do not read selects the Fortran-derived ones, which would mix two sets of routines)",
    )
    parser.add_argument("--keep", action="store_true", help="keep the run folders (parsec.out, archives)")
    parser.add_argument("--work", type=Path, default=None, help="folder of the runs (default a temporary one)")
    options = parser.parse_args(arguments)

    text, pseudopotentials = _case_text(options.case, options.source)
    if options.set:
        # Labels are matched in lower case (edit_input compares the lower-case label of each line).
        text = edit_input(text, {key.lower(): value for key, value in (item.split("=", 1) for item in options.set)})
    criterion = _criterion(text)
    label = options.label or options.case
    out = options.out / label / options.cli
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"solver15_{label}_")) if options.work is None else options.work
    work.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in options.seeds.split(",") if item != ""]
    variants = [item for item in options.variants.split(",") if item]
    for name in variants:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name}")
    plan = [] if options.no_reference else [("reference", "base", 0, True)]
    plan += [(f"{name}_s{seed}", name, seed, False) for name in variants for seed in seeds]
    records = {}
    extra = ["--quiet"] if options.cli == "accelerated" else []
    if options.cli == "accelerated" and options.gpus:
        extra += ["--gpus", str(options.gpus)]
    for run_name, variant, seed, tight in plan:
        run_dir = work / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        for item in pseudopotentials:
            shutil.copy2(item, run_dir / item.name)
        changes = _variant_changes(variant, criterion, seed, tight)
        (run_dir / "parsec.in").write_text(edit_input(text, changes))
        started = time.perf_counter()
        with open(run_dir / "child.log", "w") as log:
            completed = subprocess.run(
                [sys.executable, "-m", "parsec_python.acceleration.benchmarks.solver_defaults_1p5",
                 "--child", options.cli, str(run_dir / "parsec.in"), *extra],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")},
            )
        record = _summarize(run_dir)
        record.update(
            run=run_name,
            variant=variant,
            seed=seed,
            changes=changes,
            exit_code=completed.returncode,
            wall_seconds=time.perf_counter() - started,
        )
        if "reference" in records and run_name != "reference":
            _compare(record, records["reference"])
        records[run_name] = record
        (out / f"{run_name}.json").write_text(json.dumps(record, indent=1))
        shutil.copy2(run_dir / "parsec.in", out / f"{run_name}.parsec.in")
        print(
            f"{label:10s} {run_name:18s} exit={completed.returncode} it={record.get('iterations')} "
            f"conv={record.get('converged')} products={record.get('products')} "
            f"E={record.get('total_energy_ry')} dE={record.get('delta_energy_ry')} "
            f"deig_occ={record.get('max_delta_occupied_eigenvalue_ry')} wall={record['wall_seconds']:.1f}",
            flush=True,
        )
        if not options.keep:
            for name in ("parsec_python_results.npz",):
                (run_dir / name).unlink(missing_ok=True)
    (out / "summary.json").write_text(json.dumps(records, indent=1))
    _write_table(out / "summary.txt", records)
    if options.work is None and not options.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0


def _write_table(path: Path, records: dict) -> None:
    rows = ["run                 conv  it  products      E_tot[Ry]          dE[Ry]      max|deig_occ|[Ry]  wall[s]"]
    for name, record in records.items():
        rows.append(
            f"{name:18s} {str(record.get('converged')):5s} {record.get('iterations', '-')!s:>3} "
            f"{record.get('products', float('nan')):10.0f} {record.get('total_energy_ry', float('nan')):18.10f} "
            f"{record.get('delta_energy_ry', float('nan')):12.3e} {record.get('max_delta_occupied_eigenvalue_ry', float('nan')):12.3e} "
            f"{record.get('wall_seconds', float('nan')):8.1f}"
        )
    path.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
