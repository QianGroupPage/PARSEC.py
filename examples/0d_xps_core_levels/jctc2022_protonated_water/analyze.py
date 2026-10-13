#!/usr/bin/env python3
"""O 1s binding energies of the protonated-water clusters against Figure 5(a) of the paper.

    python analyze.py                # table on the screen
    python analyze.py --write        # also RESULTS.md
    python analyze.py --check        # compare a new set of runs with the stored
                                     # reference and with the figure; exit code 1
                                     # if a tolerance is exceeded

The binding energy of the hydronium oxygen, ``E_AE(FS) - E_AE(IS)``, is given
relative to that of the water molecule (case ``O/09`` of ``../jctc2022_1s``).
The energies come from the logs ``cases/.../parsec.out`` where a cluster has
both, otherwise from ``reference_energies.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import xps_cases as xps  # noqa: E402

CASES = HERE / "cases"
REFERENCE = HERE / "reference_energies.csv"
WATER = HERE.parent / "jctc2022_1s"
METHODS = (("pbe", "pp_pbe", "PP-PBE"), ("b3lyp", "pp_pbe_b3lyp", "PP-PBE(B3LYP)"))
#: tolerances of --check, eV; the figure is read with an uncertainty of about 0.03
TOLERANCE = {"reference": 0.003, "figure": 0.08}


def water() -> dict:
    """Binding energy of the water molecule, from its logs or its stored totals."""

    values = xps.collect(WATER / "cases", WATER / "reference_energies.csv", "logs")
    if not ("O/09/IS" in values and "O/09/FS" in values):
        values = xps.read_reference(WATER / "reference_energies.csv")
    return {key: (values["O/09/FS"][key] - values["O/09/IS"][key]) * xps.RYDBERG_EV for key, _, _ in METHODS}


def shifts(values: dict[str, dict]) -> list[dict]:
    with open(HERE / "paper" / "figure_5a.csv", encoding="utf-8") as handle:
        figure = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    reference = water()
    results = []
    for row in figure:
        n = int(row["n"])
        initial, final = values.get(f"{n:02d}/IS"), values.get(f"{n:02d}/FS")
        entry = {"n": n, "figure": {key: float(row[key]) for _, key, _ in METHODS}}
        if initial and final:
            for key, _, _ in METHODS:
                entry[key] = (final[key] - initial[key]) * xps.RYDBERG_EV - reference[key]
            entry["time_s"] = initial["wall_s"] + final["wall_s"]
        results.append(entry)
    return results


def report(results: list[dict]) -> str:
    out = ["## O 1s of the hydronium oxygen of H3O+(H2O)n relative to the water molecule, eV\n",
           "The values of the paper are read from the image of its Figure 5(a) and are uncertain by about 0.03.\n",
           "| n | PP-PBE here | figure | difference | PP-PBE(B3LYP) here | figure | difference |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in results:
        figure = entry["figure"]
        if "pbe" not in entry:
            out.append(f"| {entry['n']} | not run | {figure['pp_pbe']:.2f} | | not run | {figure['pp_pbe_b3lyp']:.2f} | |")
            continue
        out.append(f"| {entry['n']} | {entry['pbe']:.3f} | {figure['pp_pbe']:.2f} | {entry['pbe'] - figure['pp_pbe']:+.3f} | "
                   f"{entry['b3lyp']:.3f} | {figure['pp_pbe_b3lyp']:.2f} | {entry['b3lyp'] - figure['pp_pbe_b3lyp']:+.3f} |")
    done = [e for e in results if "pbe" in e]
    if done:
        out += ["", f"Over the {len(done)} clusters done: mean absolute difference from the figure " +
                " / ".join(f"{xps.mean(abs(e[key] - e['figure'][column]) for e in done):.3f}" for key, column, _ in METHODS) +
                ", largest " +
                " / ".join(f"{max(abs(e[key] - e['figure'][column]) for e in done):.3f}" for key, column, _ in METHODS) +
                " (PP-PBE / PP-PBE(B3LYP))."]
    out.append("")
    return "\n".join(out)


def check(results: list[dict], stored: list[dict]) -> list[str]:
    problems = []
    for entry, old in zip(results, stored):
        if "pbe" not in entry:
            problems.append(f"n = {entry['n']}: not run")
            continue
        for key, column, name in METHODS:
            if "pbe" in old and abs(entry[key] - old[key]) > TOLERANCE["reference"]:
                problems.append(f"n = {entry['n']}: {name} {entry[key]:.4f}, reference {old[key]:.4f}")
            if abs(entry[key] - entry["figure"][column]) > TOLERANCE["figure"]:
                problems.append(f"n = {entry['n']}: {name} {entry[key]:.3f}, figure {entry['figure'][column]}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write RESULTS.md")
    parser.add_argument("--check", action="store_true", help="test the logs against the reference and the figure")
    parser.add_argument("--update-reference", action="store_true", help="rewrite reference_energies.csv from the logs")
    arguments = parser.parse_args()

    logs = xps.collect(CASES, REFERENCE, "logs")
    stored = xps.read_reference(REFERENCE)
    if arguments.update_reference:
        xps.write_reference(REFERENCE, {**stored, **logs})
        print(f"{REFERENCE.name}: {len({**stored, **logs})} cases")
        stored = xps.read_reference(REFERENCE)
    if arguments.check:
        problems = check(shifts(logs), shifts(stored))
        print("\n".join(problems) if problems else
              f"all {len(logs)} cases within {TOLERANCE['reference']} eV of the reference and "
              f"{TOLERANCE['figure']} eV of the figure")
        raise SystemExit(1 if problems else 0)
    text = report(shifts({**stored, **logs}))
    print(text)
    if arguments.write:
        (HERE / "RESULTS.md").write_text(
            "# O 1s binding energies of protonated water clusters\n\nWritten by `analyze.py --write`.\n\n" + text,
            encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
