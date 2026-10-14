#!/usr/bin/env python3
"""Total energies of the pseudo-atoms against Table 1 of the paper.

    python analyze.py                # table on the screen
    python analyze.py --write        # also RESULTS.md
    python analyze.py --check        # compare a new set of runs with the stored
                                     # reference and with the paper; exit code 1
                                     # if a tolerance is exceeded

The energy of a case is the last ``Total Energy`` of its log
``cases/.../parsec.out``; without a finished log it is taken from
``reference_energies.csv``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import xps_cases as xps  # noqa: E402

CASES = HERE / "cases"
REFERENCE = HERE / "reference_energies.csv"
FINISHED = "Total accelerated Python wall time"
TOTAL = re.compile(r"Total Energy =\s*(-?\d+\.\d+) \[Ry\]")
WALL = re.compile(r"Total accelerated Python wall time \[sec\] :\s*([\d.]+)")
MOMENT = re.compile(r"Net magnetic moment \(in Bohr magnetons\):\s*(-?\d+\.\d+)")
# Table 1 of the paper, eV: pseudo-atom and pseudo-ion, spin-unpolarized and spin-polarized.
TABLE_1 = {
    ("B", "atom"): (-70.483, -70.796), ("B", "ion"): (-128.688, -129.236),
    ("C", "atom"): (-145.943, -147.314), ("C", "ion"): (-240.665, -242.723),
    ("N", "atom"): (-262.446, -265.824), ("N", "ion"): (-401.742, -406.408),
    ("O", "atom"): (-428.168, -429.765), ("O", "ion"): (-619.405, -621.455),
}
SPINS = ("unpolarized", "polarized")
#: tolerances of --check, eV
TOLERANCE = {"reference": 0.0005, "paper": 0.02, "paper_polarization": 0.005}


def read_logs() -> dict[str, dict]:
    values = {}
    for entry in xps.read_index(CASES):
        log = CASES / entry["folder"] / "parsec.out"
        if not xps.finished(log, FINISHED):
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        moment = MOMENT.findall(text)
        values[entry["folder"]] = {"total": float(TOTAL.findall(text)[-1]), "wall_s": float(WALL.findall(text)[-1]),
                                   "moment": float(moment[-1]) if moment else 0.0}
    return values


def read_reference() -> dict[str, dict]:
    if not REFERENCE.is_file():
        return {}
    with open(REFERENCE, encoding="utf-8", newline="") as handle:
        return {row["folder"]: {key: float(row[key]) for key in ("total", "moment", "wall_s")}
                for row in csv.DictReader(handle)}


def write_reference(values: dict[str, dict]) -> None:
    with open(REFERENCE, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["folder", "total", "moment", "wall_s"])
        for folder in sorted(values):
            row = values[folder]
            writer.writerow([folder, f"{row['total']:.8f}", f"{row['moment']:.4f}", f"{row['wall_s']:.1f}"])


def signed(value: float) -> str:
    return f"{round(value, 3) + 0.0:+.3f}"


def table(values: dict[str, dict]) -> list[dict]:
    rows = []
    for (element, kind), printed in TABLE_1.items():
        row = {"element": element, "kind": kind, "paper": printed}
        found = [values.get(f"{element}/{kind}/{spin}") for spin in SPINS]
        if all(found):
            row["here"] = tuple(value["total"] * xps.RYDBERG_EV for value in found)
            row["moment"] = found[1]["moment"]
        rows.append(row)
    return rows


def report(rows: list[dict]) -> str:
    out = ["## Pseudo-atoms and pseudo-ions (Table 1), total energy in eV\n",
           "\"Polarization\" is the energy with spin polarization minus the energy without.\n",
           "| element | | unpolarized here | paper | difference | polarized here | paper | difference | "
           "polarization here | paper | difference | moment |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        paper = row["paper"]
        if "here" not in row:
            out.append(f"| {row['element']} | {row['kind']} | not run | {paper[0]:.3f} | | not run | {paper[1]:.3f} | | | "
                       f"{paper[1] - paper[0]:.3f} | | |")
            continue
        here = row["here"]
        out.append(f"| {row['element']} | {row['kind']} | {here[0]:.3f} | {paper[0]:.3f} | {signed(here[0] - paper[0])} | "
                   f"{here[1]:.3f} | {paper[1]:.3f} | {signed(here[1] - paper[1])} | {here[1] - here[0]:.3f} | "
                   f"{paper[1] - paper[0]:.3f} | {signed((here[1] - here[0]) - (paper[1] - paper[0]))} | "
                   f"{abs(row['moment']):.2f} |")
    done = [row for row in rows if "here" in row]
    if done:
        totals = [abs(row["here"][k] - row["paper"][k]) for row in done for k in (0, 1)]
        gains = [abs((row["here"][1] - row["here"][0]) - (row["paper"][1] - row["paper"][0])) for row in done]
        out += ["", f"Largest difference from the paper: {max(totals):.3f} in a total energy, "
                    f"{max(gains):.3f} in a polarization energy."]
    out.append("")
    return "\n".join(out)


def check(rows: list[dict], stored: list[dict]) -> list[str]:
    problems = []
    for row, old in zip(rows, stored):
        label = f"{row['element']} {row['kind']}"
        if "here" not in row:
            problems.append(f"{label}: not run")
            continue
        for k, spin in enumerate(SPINS):
            if "here" in old and abs(row["here"][k] - old["here"][k]) > TOLERANCE["reference"]:
                problems.append(f"{label} {spin}: {row['here'][k]:.4f}, reference {old['here'][k]:.4f}")
            if abs(row["here"][k] - row["paper"][k]) > TOLERANCE["paper"]:
                problems.append(f"{label} {spin}: {row['here'][k]:.3f}, paper {row['paper'][k]}")
        gain = (row["here"][1] - row["here"][0]) - (row["paper"][1] - row["paper"][0])
        if abs(gain) > TOLERANCE["paper_polarization"]:
            problems.append(f"{label}: polarization energy differs from the paper by {gain:+.4f}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write RESULTS.md")
    parser.add_argument("--check", action="store_true", help="test the logs against the reference and the paper")
    parser.add_argument("--update-reference", action="store_true", help="rewrite reference_energies.csv from the logs")
    arguments = parser.parse_args()

    logs, stored = read_logs(), read_reference()
    if arguments.update_reference:
        write_reference({**stored, **logs})
        stored = read_reference()
        print(f"{REFERENCE.name}: {len(stored)} cases")
    if arguments.check:
        problems = check(table(logs), table(stored))
        print("\n".join(problems) if problems else
              f"all {len(logs)} cases within {TOLERANCE['reference']} eV of the reference, {TOLERANCE['paper']} eV of "
              f"the paper, and the polarization energies within {TOLERANCE['paper_polarization']} eV")
        raise SystemExit(1 if problems else 0)
    text = report(table({**stored, **logs}))
    print(text)
    if arguments.write:
        (HERE / "RESULTS.md").write_text(
            "# Pseudo-atoms of Table 1, without and with spin polarization\n\nWritten by `analyze.py --write`.\n\n" + text,
            encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
