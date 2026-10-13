#!/usr/bin/env python3
"""Binding energies of the cases against the tables of the paper.

    python analyze.py                # tables on the screen
    python analyze.py --write        # also RESULTS.md
    python analyze.py --check        # compare a new set of runs with the stored
                                     # reference and with the paper; exit code 1
                                     # if a tolerance is exceeded

The energies come from the logs ``cases/.../parsec.out`` when every case has a
finished one, otherwise from ``reference_energies.csv`` (the runs reported in
the README); ``--source`` forces one of the two.  A binding energy is
``E_AE(FS) - E_AE(IS)`` of the two reference-corrected totals that the solver
prints, once for PBE and once for the B3LYP-refined energy.  The potential
with a 2p hole is spherical, so a 2p binding energy is the average over the
multiplet; the paper lists 2p3/2 values, and only shifts are compared for 2p.
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
METHODS = (("pbe", "pp_pbe", "PP-PBE"), ("b3lyp", "pp_pbe_b3lyp", "PP-PBE(B3LYP)"))
BLOCKS = (("S", "1s"), ("P", "1s"), ("S", "2s"), ("P", "2s"), ("S", "2p"), ("P", "2p"))
# Absolute values of the reference molecules in Tables S1 and S4 (PP-PBE, PP-PBE(B3LYP)) and
# the precision they are printed with.
REFERENCE_ABSOLUTE = {("S", "1s"): (2492.11, 2492.11), ("P", "1s"): (2162.71, 2162.77),
                      ("S", "2s"): (224.3, 224.44), ("P", "2s"): (185.72, 185.71)}
#: tolerances of --check, eV
TOLERANCE = {"reference": 0.003, "paper_absolute": 0.07, "paper_shift": 0.06}


def binding_energies(values: dict[str, dict]) -> list[dict]:
    with open(HERE / "paper" / "tables.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    results = []
    for row in rows:
        initial = values.get(f"{row['structure']}/IS")
        final = values.get(f"{row['structure']}/FS_{row['element']}-{row['shell']}")
        entry = {"block": (row["element"], row["shell"]), "molecule": row["molecule"],
                 "paper": {key: float(row[key]) for key in ("exp_shift", "exp_abs", "pp_pbe", "pp_pbe_b3lyp")}}
        if initial and final:
            for key, _, _ in METHODS:
                entry[key] = (final[key] - initial[key]) * xps.RYDBERG_EV
        results.append(entry)
    for block in BLOCKS:
        members = [e for e in results if e["block"] == block]
        for entry in members:
            if "pbe" in entry and "pbe" in members[0]:
                for key, _, _ in METHODS:
                    entry[f"{key}_shift"] = entry[key] - members[0][key]
    return results


def report(results: list[dict]) -> str:
    out = ["## Reference molecules: absolute 1s and 2s binding energies, eV\n",
           "| core level | molecule | method | here | paper | difference |", "|---|---|---|---:|---:|---:|"]
    for block, printed in REFERENCE_ABSOLUTE.items():
        reference = next(e for e in results if e["block"] == block)
        if "pbe" not in reference:
            continue
        for column, (key, _, label) in enumerate(METHODS):
            out.append(f"| {block[0]} {block[1]} | {reference['molecule']} | {label} | {reference[key]:.3f} | "
                       f"{printed[column]} | {reference[key] - printed[column]:+.3f} |")
    out.append("")
    for block in BLOCKS:
        rows = [e for e in results if e["block"] == block]
        out.append(f"## {block[0]} {block[1]}: shifts relative to {rows[0]['molecule']}, eV\n")
        out.append("| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for entry in rows:
            paper = entry["paper"]
            if "pbe_shift" not in entry:
                out.append(f"| {entry['molecule']} | {paper['exp_shift']:.2f} | not run | {paper['pp_pbe']:.2f} | | "
                           f"not run | {paper['pp_pbe_b3lyp']:.2f} | |")
                continue
            out.append(
                f"| {entry['molecule']} | {paper['exp_shift']:.2f} | {entry['pbe_shift']:.3f} | {paper['pp_pbe']:.2f} | "
                f"{entry['pbe_shift'] - paper['pp_pbe']:+.3f} | {entry['b3lyp_shift']:.3f} | {paper['pp_pbe_b3lyp']:.2f} | "
                f"{entry['b3lyp_shift'] - paper['pp_pbe_b3lyp']:+.3f} |")
        out.append("")
    out += ["## Summary, eV\n",
            "Mean absolute error (MAE) of the shifts against experiment and differences from the paper, over the "
            "molecules of a block other than its reference.\n",
            "| core level | molecules | method | MAE here | MAE of the paper's values | "
            "mean abs. difference from the paper | largest |",
            "|---|---:|---|---:|---:|---:|---:|"]
    for block in BLOCKS:
        rows = [e for e in results if e["block"] == block and "pbe_shift" in e][1:]
        for key, paper_key, label in METHODS:
            if not rows:
                continue
            here = xps.mean(abs(e[f"{key}_shift"] - e["paper"]["exp_shift"]) for e in rows)
            theirs = xps.mean(abs(e["paper"][paper_key] - e["paper"]["exp_shift"]) for e in rows)
            differences = [abs(e[f"{key}_shift"] - e["paper"][paper_key]) for e in rows]
            out.append(f"| {block[0]} {block[1]} | {len(rows)} | {label} | {here:.3f} | {theirs:.3f} | "
                       f"{xps.mean(differences):.3f} | {max(differences):.3f} |")
    out.append("")
    return "\n".join(out)


def check(results: list[dict], stored: list[dict]) -> list[str]:
    """Deviations beyond the tolerances: new runs against the stored reference and against the paper."""

    problems = []
    first = {block: next(e for e in results if e["block"] == block) for block in BLOCKS}
    for entry, old in zip(results, stored):
        label = f"{entry['block'][0]} {entry['block'][1]} {entry['molecule']}"
        if "pbe" not in entry:
            problems.append(f"{label}: not run")
            continue
        for column, (key, paper_key, name) in enumerate(METHODS):
            if "pbe" in old and abs(entry[key] - old[key]) > TOLERANCE["reference"]:
                problems.append(f"{label}: {name} {entry[key]:.4f}, reference {old[key]:.4f}")
            if entry is first[entry["block"]]:
                printed = REFERENCE_ABSOLUTE.get(entry["block"])
                if printed and abs(entry[key] - printed[column]) > TOLERANCE["paper_absolute"]:
                    problems.append(f"{label}: {name} {entry[key]:.3f}, paper {printed[column]}")
            elif abs(entry[f"{key}_shift"] - entry["paper"][paper_key]) > TOLERANCE["paper_shift"]:
                problems.append(f"{label}: {name} shift {entry[f'{key}_shift']:.3f}, paper {entry['paper'][paper_key]}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("logs", "reference"), default=None)
    parser.add_argument("--write", action="store_true", help="write RESULTS.md")
    parser.add_argument("--check", action="store_true", help="test the logs against the reference and the paper")
    parser.add_argument("--update-reference", action="store_true", help="rewrite reference_energies.csv from the logs")
    arguments = parser.parse_args()

    logs = xps.collect(CASES, REFERENCE, "logs")
    complete = len(logs) == len(xps.read_index(CASES))
    source = arguments.source or ("logs" if complete else "reference")
    if arguments.update_reference:
        xps.write_reference(REFERENCE, logs)
        print(f"{REFERENCE.name}: {len(logs)} cases")
    if arguments.check:
        problems = check(binding_energies(logs), binding_energies(xps.read_reference(REFERENCE)))
        print("\n".join(problems) if problems else
              f"all {len(logs)} cases within {TOLERANCE['reference']} eV of the reference; absolute 1s and 2s values "
              f"within {TOLERANCE['paper_absolute']} eV and shifts within {TOLERANCE['paper_shift']} eV of the paper")
        raise SystemExit(1 if problems else 0)
    text = report(binding_energies(logs if source == "logs" else xps.read_reference(REFERENCE)))
    print(text)
    if arguments.write:
        origin = "the logs in `cases/`" if source == "logs" else "`reference_energies.csv`"
        (HERE / "RESULTS.md").write_text(
            f"# S and P 1s, 2s and 2p binding energies of the 2024 set\n\nWritten by `analyze.py --write` from {origin}.\n\n"
            + text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
