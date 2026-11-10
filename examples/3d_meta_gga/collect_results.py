"""The tables of the README from the logs of this example and the Quantum ESPRESSO numbers below.

    python examples/3d_meta_gga/collect_results.py                  # the logs parsec_python.out of the example
    python examples/3d_meta_gga/collect_results.py --runs RUNS      # RUNS/<case>/parsec.out and RUNS/runs.jsonl

E - E_PBE (mRy per cell, each code against its own PBE on the same mesh), the band gap on the mesh (lowest empty
minus highest occupied level over the k-points of the run, eV), iterations and costs.

QE: Quantum ESPRESSO 7.5 + libxc 7.0.0 with the same FHI98PP potential (UPF), ``ecutwfc`` 200 Ry, ``ecutrho``
1600 Ry, ``K_POINTS automatic n n n 0 0 0`` (``qe/``); QE's levels have its own zero of the potential, so only the
gap is compared.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("meta_gga_example_logs", HERE.parent / "0d_meta_gga" / "collect_results.py")
logs_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(logs_module)

MESHES = ["Si_k222", "Si_k444"]
LABELS = {"Si_k222": "2x2x2", "Si_k444": "4x4x4"}
FUNCTIONALS = ["pbe", "scan", "r2scan", "tpss"]
NAMES = {"pbe": "PBE", "scan": "SCAN", "r2scan": "r2SCAN", "tpss": "TPSS"}

# QE at 200 Ry: E (Ry per cell), highest occupied and lowest empty level (eV) on the mesh
QE = {
    "Si_k222/pbe": {"E": -15.57491928, "vbm": 6.5275, "cbm": 7.1377},
    "Si_k444/pbe": {"E": -15.74871754, "vbm": 6.2738, "cbm": 6.9740},
    "Si_k222/scan": {"E": -15.58260674, "vbm": 6.3800, "cbm": 7.1732},
    "Si_k444/scan": {"E": -15.75816089, "vbm": 6.1433, "cbm": 7.1038},
    "Si_k222/r2scan": {"E": -15.57440302, "vbm": 6.4259, "cbm": 7.1907},
    "Si_k444/r2scan": {"E": -15.75460582, "vbm": 6.2111, "cbm": 7.0671},
    "Si_k222/tpss": {"E": -15.56026914, "vbm": 6.4930, "cbm": 7.4558},
    "Si_k444/tpss": {"E": -15.72842970, "vbm": 6.2591, "cbm": 7.1300},
}
_KPOINTS = re.compile(r"\((\d+) irreducible k-points\)")
_POINTS = re.compile(r"Grid points\s*[:=]\s*(\d+)|Full active grid points\s*=\s*(\d+)")


def band_edges(text: str) -> tuple[float, float] | None:
    """Highest occupied and lowest empty level (eV) of the converged levels of every k-point."""

    start = text.rfind("Converged eigenvalues by k-point")
    if start < 0:
        return None
    occupied, empty = [], []
    for line in text[start:].split("\n")[1:]:
        fields = line.split()
        if len(fields) == 4 and fields[0].isdigit():
            (occupied if float(fields[3]) > 0.5 else empty).append(float(fields[2]))
        elif fields and not (fields[0] in ("k-point", "State")):
            break
    return (max(occupied), min(empty)) if occupied and empty else None


def load(runs: Path | None) -> tuple[dict, dict]:
    logs, costs = {}, {}
    for mesh in MESHES:
        for functional in FUNCTIONALS:
            case = f"{mesh}/{functional}"
            path = (runs / case / "parsec.out") if runs is not None else (HERE / case / "parsec_python.out")
            record = logs_module.read(path)
            if record is None:
                continue
            text = path.read_text(errors="replace")
            record["edges"] = band_edges(text)
            kpoints = _KPOINTS.search(text)
            record["kpoints"] = int(kpoints.group(1)) if kpoints else None
            logs[case] = record
    if runs is not None and (runs / "runs.jsonl").is_file():
        for line in (runs / "runs.jsonl").read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                costs[record["case"]] = record
    return logs, costs


def tables(logs: dict, costs: dict) -> list[str]:
    lines = ["### PBE\n", "| mesh | E this code (Ry/cell) | E QE (Ry/cell) | difference (Ry) | gap this code (eV) | gap QE (eV) | dgap (meV) |",
             "|---|---|---|---|---|---|---|"]
    for mesh in MESHES:
        ours, theirs = logs.get(f"{mesh}/pbe"), QE[f"{mesh}/pbe"]
        if ours is None:
            continue
        gap, reference = ours["edges"][1] - ours["edges"][0], theirs["cbm"] - theirs["vbm"]
        lines.append(f"| {LABELS[mesh]} | {ours['E']:.8f} | {theirs['E']:.8f} | {ours['E'] - theirs['E']:+.2e} | "
                     f"{gap:.4f} | {reference:.4f} | {1000 * (gap - reference):+.1f} |")
    lines += ["", "### Meta-GGAs\n",
              "| functional | mesh | E - E_PBE this code (mRy) | E - E_PBE QE (mRy) | this code - QE (mRy) | gap this code (eV) | "
              "gap QE (eV) | dgap (meV) | SCF iterations | gradient form |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for functional in FUNCTIONALS[1:]:
        for mesh in MESHES:
            case = f"{mesh}/{functional}"
            ours, base = logs.get(case), logs.get(f"{mesh}/pbe")
            if ours is None or base is None:
                continue
            theirs = QE[case]
            difference = ours["E"] - base["E"]
            reference = theirs["E"] - QE[f"{mesh}/pbe"]["E"]
            gap, gap_qe = ours["edges"][1] - ours["edges"][0], theirs["cbm"] - theirs["vbm"]
            note = "" if ours["converged"] else " (Max_Iter)"
            lines.append(f"| {NAMES[functional]}{note} | {LABELS[mesh]} | {1000 * difference:+.3f} | {1000 * reference:+.3f} | "
                         f"{1000 * (difference - reference):+.3f} | {gap:.4f} | {gap_qe:.4f} | {1000 * (gap - gap_qe):+.1f} | "
                         f"{ours['scf']} | {ours['form'] or '-'} |")
    lines += ["", "### Costs (one A100)\n",
              "| mesh | irreducible k-points | functional | SCF iterations | program wall (s) | peak device memory (GiB) |",
              "|---|---|---|---|---|---|"]
    for mesh in MESHES:
        for functional in FUNCTIONALS:
            case = f"{mesh}/{functional}"
            if case not in logs:
                continue
            record, memory = logs[case], costs.get(case, {}).get("peak_device_mib")
            lines.append(f"| {LABELS[mesh]} | {record['kpoints']} | {NAMES[functional]} | {record['scf']} | "
                         f"{'-' if record['wall'] is None else format(record['wall'], '.1f')} | "
                         f"{f'{memory / 1024:.1f}' if memory else '-'} |")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=None, help="a run folder of run_examples.py")
    arguments = parser.parse_args(argv)
    print("\n".join(tables(*load(arguments.runs))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
