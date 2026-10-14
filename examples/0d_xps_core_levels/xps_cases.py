"""Shared steps of the two core-level examples in this folder.

An example takes the structures published with a paper and the
pseudopotentials in ``pseudopotentials/``, writes one ``parsec.in`` for the
neutral molecule (``IS``) and one for every core hole (``FS...``: the atom with
the hole carries the potential made with one core electron removed, and the
molecule the charge +1), runs them, and reads the two reference-corrected
totals that the solver prints.  ``reference_energies.csv`` in each example
holds those totals from the runs that the README reports.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
MAIN = REPOSITORY / "src" / "parsec_python" / "main.py"
PSEUDOPOTENTIALS = HERE / "pseudopotentials"

RYDBERG_EV = 13.605693123
HARTREE_EV = 2.0 * RYDBERG_EV
#: distance from the outermost nucleus to the wall of the spherical domain
MARGIN_ANGSTROM = 5.0
SPACING_ANGSTROM = 0.10
#: Matvec_Blocksize of CONTROLS
BLOCK = 16
DONE = "B3LYP-refined all-electron total"

CONTROLS = """\
Correlation_Type: pbe
Energy_Refinement: b3lyp
Spin_Polarization: false
States_Num: {states}
Net_Charges: {charge} e
Fermi_Temp: 1160.451812 K
Max_Iter: 100
Convergence_Criterion: 5e-5 Ry

Eigensolver: chebff
Chebdav_Degree: 24
FF_MaxIter: 2
Matvec_Blocksize: 16
Chebyshev_Degree: 12
Chebyshev_Degree_Delta: 3
Subspace_Buffer_Size: 6
Diag_Tolerance: 1e-5

Mixing_Method: Anderson
Mixing_Param: 0.30
Memory_Param: 4
Restart_Mixing: 20
Solver_Lpole: 9
Full_Hartree: false

Minimization: manual
Movement_Num: 0
Skip_Force: true
Output_Level: 1
Output_All_States: false
"""

#: what is read from a log: the two totals, the parts of the B3LYP energy, the times
ENERGY = {
    "pbe": re.compile(r"PBE all-electron total \(as above\)\s*:\s*(-?\d+\.\d+) \[Ry\]"),
    "b3lyp": re.compile(r"B3LYP-refined all-electron total\s*:\s*(-?\d+\.\d+) \[Ry\]"),
}
PARTS = {
    "hf": re.compile(r"Exact \(Hartree-Fock\) exchange\s*=\s*(-?\d+\.\d+) \[Ry\]"),
    "slater": re.compile(r"Slater exchange\s*=\s*(-?\d+\.\d+) \[Ry\]"),
    "b88": re.compile(r"Becke 88 exchange\s*=\s*(-?\d+\.\d+) \[Ry\]"),
    "vwn": re.compile(r"VWN \(RPA\) correlation\s*=\s*(-?\d+\.\d+) \[Ry\]"),
    "lyp": re.compile(r"Lee-Yang-Parr correlation\s*=\s*(-?\d+\.\d+) \[Ry\]"),
    "exc_pbe": re.compile(r"Exc \(PBE, self-consistent\)\s*=\s*(-?\d+\.\d+) \[Ry\]"),
}
TIMES = {
    "scf_s": re.compile(r"Time for self-consistent field \[sec\] :\s*([\d.]+)"),
    "refinement_s": re.compile(r"Time for energy refinement \[sec\] :\s*([\d.]+)"),
    "wall_s": re.compile(r"Total accelerated Python wall time \[sec\] :\s*([\d.]+)"),
}
REFERENCE_FIELDS = ["folder", *ENERGY, *PARTS, *TIMES]


# ---------------------------------------------------------------- inputs

def species_table() -> dict[str, dict]:
    """Local component, valence electrons and energy correction of every potential."""

    return json.loads((PSEUDOPOTENTIALS / "species.json").read_text(encoding="utf-8"))


def read_xyz(path: Path) -> tuple[str, list[tuple[str, tuple[float, float, float]]]]:
    """Comment line and atoms; every coordinate line counts, whatever the header says."""

    lines = path.read_text().splitlines()
    atoms = []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) >= 4:
            atoms.append((fields[0], (float(fields[1]), float(fields[2]), float(fields[3]))))
    return lines[1].strip(), atoms


def enclosing_sphere(points) -> tuple[tuple[float, ...], float]:
    """Centre and radius of the smallest sphere around the nuclei (Badoiu-Clarkson)."""

    centre = list(points[0])
    for step in range(1, 20001):
        far = max(points, key=lambda p: math.dist(p, centre))
        centre = [c + (f - c) / (step + 1) for c, f in zip(centre, far)]
    return tuple(centre), max(math.dist(p, centre) for p in points)


def write_case(folder: Path, atoms, species: dict, *, hole: tuple[int, str] | None = None, charge: int = 0,
               spacing: float = SPACING_ANGSTROM, margin: float = MARGIN_ANGSTROM) -> dict:
    """Write ``folder/parsec.in``.

    ``hole`` is (atom number from 0, species label such as ``C-1s``); ``charge``
    is the net charge of the molecule before the core electron is removed.
    """

    centre, extent = enclosing_sphere([xyz for _, xyz in atoms])
    radius = extent + margin
    groups: dict[str, list] = {}
    for number, (symbol, xyz) in enumerate(atoms):
        label = hole[1] if hole and number == hole[0] else symbol
        if species[label]["element"] != symbol:
            raise ValueError(f"{folder}: atom {number} is {symbol}, not {species[label]['element']}")
        groups.setdefault(label, []).append(tuple(a - c for a, c in zip(xyz, centre)))
    # a core-hole potential has one more unit of ionic charge, not one more electron
    electrons = sum(species[label]["valence_electrons"] * len(rows) for label, rows in groups.items()) - charge
    if abs(electrons - 2 * round(electrons / 2)) > 1e-9:
        raise ValueError(f"{folder}: odd number of valence electrons")
    occupied = round(electrons / 2)
    # at least 8 empty states, rounded up to whole blocks of the eigensolver
    states = BLOCK * math.ceil((occupied + 8) / BLOCK)
    text = [
        "Boundary_Conditions: cluster",
        "Cluster_Domain_Shape: sphere",
        f"Boundary_Sphere_Radius: {radius:.6f} ang",
        f"Grid_Spacing: {spacing:g} ang",
        "Expansion_Order: 8",
        "Coordinate_Unit: Cartesian_Ang",
        "Ignore_Symmetry: false",
        "",
        f"Atom_Types_Num: {len(groups)}",
        f"Total_Atom_Num: {len(atoms)}",
        "",
    ]
    for label in sorted(groups, key=lambda name: ("-" not in name, name)):
        entry = species[label]
        text += [
            f"Atom_Type: {label}",
            f"Element_Symbol: {entry['element']}",
            "Pseudopotential_Format: martins_new",
            f"Local_Component: {entry['local']}",
            "Read_VCD: true",
            "Cubic_Spline: true",
            f"Atomic_Energy_Correction: {entry['correction_rydberg']:.6f} Ry",
            "begin Atom_Coord",
            *(f" {x:15.10f} {y:15.10f} {z:15.10f}" for x, y, z in groups[label]),
            "end Atom_Coord",
            "",
        ]
    text.append(CONTROLS.format(states=states, charge=charge + (1 if hole else 0)))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "parsec.in").write_text("\n".join(text), encoding="utf-8", newline="\n")
    points = 4.0 / 3.0 * math.pi * (radius / spacing) ** 3
    return {"atoms": len(atoms), "occupied": occupied, "states": states,
            "radius_angstrom": round(radius, 4), "grid_points_estimate": round(points)}


def write_index(cases: Path, index: list[dict]) -> None:
    (cases / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8", newline="\n")


def read_index(cases: Path) -> list[dict]:
    return json.loads((cases / "index.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- runs

def finished(log: Path, marker: str = DONE) -> bool:
    return log.is_file() and marker in log.read_text(encoding="utf-8", errors="replace")


def run_queue(cases: Path, *, select=None, largest: float | None = None, cpu_poisson: bool = False,
              marker: str = DONE) -> None:
    """Run the cases of ``cases/index.json``, cheapest first; finished ones are skipped.

    ``marker`` is the text of a log that tells a finished run.
    """

    queue = []
    for entry in read_index(cases):
        if select and not select(entry):
            continue
        if largest and entry["grid_points_estimate"] > largest:
            continue
        pairs = entry["occupied"] * (entry["occupied"] + 1) // 2
        queue.append((entry["grid_points_estimate"] * (entry["states"] * 20 + pairs), entry))
    queue.sort(key=lambda item: item[0])

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY / "src")
    if not cpu_poisson:
        environment["PARSEC_HARTREE_LINEAR_BACKEND"] = "cupy"
    for number, (_, entry) in enumerate(queue, 1):
        folder = cases / entry["folder"]
        log = folder / "parsec.out"
        label = f"[{number}/{len(queue)}] {entry['folder']} {entry['molecule']}"
        if finished(log, marker):
            print(f"{label}: done before", flush=True)
            continue
        started = time.time()
        with open(folder / "console.txt", "w", encoding="utf-8") as console:
            code = subprocess.run(
                [sys.executable, str(MAIN), "parsec.in", "--pp-dir", str(PSEUDOPOTENTIALS),
                 "--backend", "auto", "--symmetry", "auto", "--log", "parsec.out", "--no-archive"],
                cwd=folder, env=environment, stdout=console, stderr=subprocess.STDOUT,
            ).returncode
        status = "ok" if code == 0 and finished(log, marker) else f"FAILED (exit {code})"
        line = f"{label}: {status}, {time.time() - started:.0f} s"
        print(line, flush=True)
        with open(cases / "progress.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ---------------------------------------------------------------- results

def read_log(path: Path) -> dict | None:
    """Energies (Ry) and times of a finished log, ``None`` if it is missing or unfinished."""

    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    found = {}
    for key, pattern in {**ENERGY, **PARTS, **TIMES}.items():
        matches = pattern.findall(text)
        if not matches and key in ENERGY:
            return None
        found[key] = float(matches[-1]) if matches else float("nan")
    return found


def read_reference(path: Path) -> dict[str, dict]:
    """``reference_energies.csv``: the values of ``read_log`` for every case of the reported runs."""

    if not path.is_file():
        return {}
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["folder"]: {key: float(value) for key, value in row.items() if key != "folder"}
                for row in csv.DictReader(handle)}


def write_reference(path: Path, values: dict[str, dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(REFERENCE_FIELDS)
        for folder in sorted(values):
            row = values[folder]
            writer.writerow([folder, *(f"{row[key]:.8f}" for key in (*ENERGY, *PARTS)),
                             *(f"{row[key]:.1f}" for key in TIMES)])


def collect(cases: Path, reference: Path, source: str) -> dict[str, dict]:
    """Values per case folder from the logs (``source='logs'``) or from the reference file."""

    if source == "reference":
        return read_reference(reference)
    values = {}
    for entry in read_index(cases):
        log = read_log(cases / entry["folder"] / "parsec.out")
        if log:
            values[entry["folder"]] = log
    return values


def scaled_exchange_total(log: dict, alpha: float) -> float:
    """All-electron total (Ry) refined with ``alpha`` times the exact exchange of B3LYP.

    ``0.2 alpha E_x(HF) + (1 - 0.2 alpha) / 0.8 * (0.08 E_x(Slater) + 0.72 E_x(B88))
    + 0.19 E_c(VWN) + 0.81 E_c(LYP)`` in place of the PBE exchange-correlation
    energy; ``alpha = 1`` is B3LYP, 0 has no exact exchange and 5 only exact exchange.
    """

    exchange = (0.2 * alpha * log["hf"]
                + (1.0 - 0.2 * alpha) / 0.8 * (0.08 * log["slater"] + 0.72 * log["b88"]))
    return log["pbe"] - log["exc_pbe"] + exchange + 0.19 * log["vwn"] + 0.81 * log["lyp"]


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")
