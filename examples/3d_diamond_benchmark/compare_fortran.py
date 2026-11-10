"""Eigenvalues, Fermi level and total energy of bulk cells: Fortran PARSEC 1.5 against this code (CPU).

Usage (from the repository root, src on the python path, on a CPU node; ``--launch`` is the MPI prefix of the
Fortran binary, e.g. ``srun -n 128 -c 2 --cpu-bind=cores``):
    python examples/3d_diamond_benchmark/compare_fortran.py --binary <parsec.mpi> --launch "<srun ...>" --work DIR
        [--modules 'PrgEnv-gnu cray-fftw'] [--other-src SRC] [--cases diamond1,si8] [--tight]

Cases: ``diamond1`` (one conventional cell of make_diamond.py, its keywords that PARSEC 1.5 does not read left out)
and ``si8`` (examples/3d_Si).  Each case runs the Fortran binary and ``python -m parsec_python.main --device cpu``
with the default routines and with ``Use_Fortran_Routines: .true.``, for this tree and (``--other-src``) another
source tree, e.g. a copy of an earlier commit's src.  ``--tight`` asks every run for Convergence_Criterion and
Diag_Tolerance 1e-6.  The table lists the last total energy and Fermi level of every parsec.out and the occupied
eigenvalues against Fortran's: the largest difference and the mean shift.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON_ONLY = (
    "mixing_safeguard", "mixing_step_limit", "mixing_growth_trigger", "mixing_backoff", "initial_density",
    "normalize_initial_density",
)


def _keyword(line: str) -> str:
    return line.split(":", 1)[0].strip().lower() if ":" in line else ""


def _with_settings(text: str, settings: dict[str, str]) -> str:
    lines = text.splitlines()
    for keyword, value in settings.items():
        hits = [i for i, line in enumerate(lines) if _keyword(line) == keyword.lower()]
        if hits:
            lines[hits[0]] = f"{keyword}: {value}"
        else:
            lines.insert(0, f"{keyword}: {value}")
    return "\n".join(lines) + "\n"


def _case_inputs(name: str, work: Path) -> tuple[str, list[Path]]:
    if name == "diamond1":
        target = work / "make" / "parsec.in"
        subprocess.run([sys.executable, str(HERE / "make_diamond.py"), "1", "1", "1", str(target)], check=True,
                       stdout=subprocess.DEVNULL)
        text = "\n".join(line for line in target.read_text().splitlines() if _keyword(line) not in PYTHON_ONLY)
        return text + "\n", [HERE / "C_POTRE.DAT"]
    if name == "si8":
        folder = ROOT / "examples" / "3d_Si"
        return (folder / "parsec.in").read_text(), [folder / "Si_POTRE.DAT"]
    raise ValueError(f"unknown case {name}")


def _summary(path: Path) -> dict:
    text = path.read_text(errors="replace") if path.exists() else ""
    energies = re.findall(r"Total Energy =\s+(-?[0-9.]+)", text)
    fermi = re.findall(r"Fermi level at\s+(-?[0-9.]+)", text)
    tables = re.split(r"State\s+Eigenvalue \[Ry\]", text)
    eigenvalues, occupations = [], []
    if len(tables) > 1:
        for line in tables[-1].splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 4 and fields[0].isdigit():
                eigenvalues.append(float(fields[1]))
                occupations.append(float(fields[3]))
            elif eigenvalues and not fields:
                break
    return {
        "energy": float(energies[-1]) if energies else float("nan"),
        "fermi": float(fermi[-1]) if fermi else float("nan"),
        "eigenvalues": eigenvalues,
        "occupations": occupations,
        "iterations": len(energies),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", required=True)
    parser.add_argument("--launch", default="")
    parser.add_argument("--modules", default="", help="modules to load for the binary (a login shell runs it)")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--other-src", type=Path)
    parser.add_argument("--cases", default="diamond1,si8")
    parser.add_argument("--tight", action="store_true")
    arguments = parser.parse_args(argv)
    this_src = ROOT / "src"
    trees = [("this", this_src)] + ([("other", arguments.other_src)] if arguments.other_src else [])
    settings = {"Convergence_Criterion": "1e-6 Ry", "Diag_Tolerance": "1e-6"} if arguments.tight else {}
    for case in arguments.cases.split(","):
        text, files = _case_inputs(case, arguments.work / case)
        text = _with_settings(text, settings)
        runs = {}
        folder = arguments.work / case / "fortran"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "parsec.in").write_text(text)
        for item in files:
            shutil.copy(item, folder / item.name)
        command = shlex.split(arguments.launch) + [arguments.binary]
        if arguments.modules:
            command = ["bash", "-lc", f"module load {arguments.modules} >/dev/null 2>&1; exec {shlex.join(command)}"]
        with open(folder / "fortran.log", "w") as log:
            subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT)
        runs["fortran"] = _summary(folder / "parsec.out")
        for label, source in trees:
            for routines in ("default", "fortran"):
                folder = arguments.work / case / f"{label}_{routines}"
                folder.mkdir(parents=True, exist_ok=True)
                body = text if routines == "default" else _with_settings(text, {"Use_Fortran_Routines": ".true."})
                (folder / "parsec.in").write_text(body)
                for item in files:
                    shutil.copy(item, folder / item.name)
                environment = dict(os.environ, PYTHONPATH=str(source))
                subprocess.run([sys.executable, "-m", "parsec_python.main", "parsec.in", "--device", "cpu"],
                               cwd=folder, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                runs[f"{label}_{routines}"] = _summary(folder / "parsec.out")
        reference = runs["fortran"]
        print(f"== {case}: E total, Fermi level (Ry), occupied eigenvalues against Fortran PARSEC 1.5", flush=True)
        for label, run in runs.items():
            occupied = [i for i, f in enumerate(reference["occupations"]) if f > 0.5]
            count = min(len(run["eigenvalues"]), len(reference["eigenvalues"]))
            rows = [i for i in occupied if i < count]
            differences = [run["eigenvalues"][i] - reference["eigenvalues"][i] for i in rows]
            shift = sum(differences) / len(differences) if differences else float("nan")
            spread = max((abs(d - shift) for d in differences), default=float("nan"))
            print(
                f"  {label:<16s} E {run['energy']:16.8f}  dE {run['energy'] - reference['energy']:+.2e}  "
                f"Fermi {run['fermi']:10.4f}  shift {shift:+.5f}  max |d - shift| {spread:.2e}  "
                f"({run['iterations']} it, {len(run['eigenvalues'])} states)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
