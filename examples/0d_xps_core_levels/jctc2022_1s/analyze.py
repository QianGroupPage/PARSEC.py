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
prints, once for PBE and once for the B3LYP-refined energy.
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
SETS = ("B", "C", "N", "O")
#: key of the energy, column of the paper's value, label
METHODS = (("pbe", "pp_pbe", "PP-PBE"), ("b3lyp", "pp_pbe_b3lyp", "PP-PBE(B3LYP)"))
# Table 1 of the paper, eV: (AE atom, AE ion, PP atom, PP ion), spin-unpolarized then spin-polarized.
TABLE_1 = {
    "B": ((-669.486, -462.410, -70.483, -128.688), (-669.753, -469.348, -70.796, -129.236)),
    "C": ((-1027.632, -723.254, -145.943, -240.665), (-1028.869, -733.736, -147.314, -242.723)),
    "N": ((-1481.741, -1061.468, -262.446, -401.742), (-1484.864, -1075.962, -265.824, -406.408)),
    "O": ((-2040.902, -1486.074, -428.168, -619.405), (-2042.420, -1498.513, -429.765, -621.455)),
}
# Table 2: PP-PBE of the reference molecules with the spin-unpolarized atomic energies.
TABLE_2_UNPOLARIZED = {"B": 198.04, "C": 296.71, "N": 411.94, "O": 544.93}
# Table S1: absolute values of the reference molecules (PP-PBE, PP-PBE(B3LYP)), which use the
# spin-polarized atomic energies of Table 1, and the mean absolute errors printed there.
REFERENCE_ABSOLUTE = {"B": (191.60, 191.77), "C": (288.16, 288.15), "N": (401.86, 401.64), "O": (534.46, 534.09)}
PRINTED_MAE = {"B": (0.23, 0.11), "C": (0.20, 0.18), "N": (0.15, 0.14), "O": (0.11, 0.12), "all": (0.16, 0.14)}
# Table S2: errors of the N shifts with 0 ... 5 times the exact exchange of B3LYP.
TABLE_S2_MAE = (0.15, 0.14, 0.16, 0.21, 0.26, 0.31)
TABLE_S2_ME = (0.02, 0.08, 0.13, 0.19, 0.25, 0.31)
# Table S1 prints the computed values of Cl3CCN (N 12) and HNCO (N 13) in each other's rows;
# they are compared exchanged.
INTERCHANGED = {("N", 12): ("N", 13), ("N", 13): ("N", 12)}
#: tolerances of --check, eV
TOLERANCE = {"reference": 0.003, "paper_absolute": 0.04, "paper_shift": 0.05}


def correction(row) -> float:
    ae_atom, ae_ion, pp_atom, pp_ion = row
    return (ae_ion - pp_ion) - (ae_atom - pp_atom)


#: what the spin-polarized atomic energies of Table 1 take off a binding energy
POLARIZED_SHIFT = {name: correction(rows[0]) - correction(rows[1]) for name, rows in TABLE_1.items()}


def paper_table() -> dict[tuple[str, int], dict]:
    rows = {}
    with open(HERE / "paper" / "table_s1.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(line for line in handle if not line.startswith("#")):
            rows[(row["set"], int(row["index"]))] = {
                key: float(value) if key not in ("set", "molecule") else value for key, value in row.items()}
    return {key: {**row, **{column: rows[INTERCHANGED.get(key, key)][column] for _, column, _ in METHODS}}
            for key, row in rows.items()}


def binding_energies(values: dict[str, dict]) -> list[dict]:
    results = []
    for (name, number), row in sorted(paper_table().items(), key=lambda item: (SETS.index(item[0][0]), item[0][1])):
        initial, final = (values.get(f"{name}/{number:02d}/{state}") for state in ("IS", "FS"))
        entry = {"set": name, "index": number, "molecule": row["molecule"], "paper": row}
        if initial and final:
            for key, _, _ in METHODS:
                entry[key] = (final[key] - initial[key]) * xps.RYDBERG_EV
            entry["logs"] = (initial, final)
        results.append(entry)
    for name in SETS:
        reference = next(e for e in results if e["set"] == name and e["index"] == 1)
        for entry in results:
            if entry["set"] == name and "pbe" in entry and "pbe" in reference:
                for key, _, _ in METHODS:
                    entry[f"{key}_shift"] = entry[key] - reference[key]
    return results


def report(results: list[dict]) -> str:
    species = xps.species_table()
    out = ["## Atomic energies of the pseudopotential generation (Table 1, spin-unpolarized), eV\n",
           "| element | all-electron atom | paper | all-electron ion | paper | pseudo-atom | paper | pseudo-ion | paper |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in SETS:
        ours = (species[name]["all_electron_hartree"], species[f"{name}-1s"]["all_electron_hartree"],
                species[name]["pseudo_hartree"], species[f"{name}-1s"]["pseudo_hartree"])
        out.append(f"| {name} | " + " | ".join(f"{value * xps.HARTREE_EV:.3f} | {printed:.3f}"
                                               for value, printed in zip(ours, TABLE_1[name][0])) + " |")
    out += ["", "## Reference molecules: absolute binding energies, eV\n",
            "The paper's Table S1 uses atomic energies of spin-polarized atoms. The constant per element that "
            "this takes off a binding energy follows from its Table 1 (" +
            ", ".join(f"{name} {POLARIZED_SHIFT[name]:.3f}" for name in SETS) + ") and is applied in the fifth column.\n",
            "| set | molecule | method | here | paper Table 2 | here, constant applied | paper Table S1 | difference |",
            "|---|---|---|---:|---:|---:|---:|---:|"]
    for name in SETS:
        reference = next(e for e in results if e["set"] == name and e["index"] == 1)
        if "pbe" not in reference:
            continue
        for column, (key, _, label) in enumerate(METHODS):
            shifted = reference[key] - POLARIZED_SHIFT[name]
            unpolarized = f"{TABLE_2_UNPOLARIZED[name]:.2f}" if key == "pbe" else ""
            out.append(f"| {name} | {reference['molecule']} | {label} | {reference[key]:.3f} | {unpolarized} | {shifted:.3f} | "
                       f"{REFERENCE_ABSOLUTE[name][column]:.2f} | {shifted - REFERENCE_ABSOLUTE[name][column]:+.3f} |")
    out.append("")
    for name in SETS:
        rows = [e for e in results if e["set"] == name]
        out.append(f"## {name} 1s: shifts relative to {rows[0]['molecule']}, eV\n")
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
                f"{entry['pbe_shift'] - paper['pp_pbe']:+.3f} | {entry['b3lyp_shift']:.3f} | "
                f"{paper['pp_pbe_b3lyp']:.2f} | {entry['b3lyp_shift'] - paper['pp_pbe_b3lyp']:+.3f} |")
        out.append("")
        if name == "N":
            out.append("Table S1 prints the values of Cl3CCN and HNCO in each other's rows; the \"paper\" columns "
                       "above have them exchanged.\n")
    out += ["## Summary, eV\n",
            "Mean absolute error (MAE) of the shifts against experiment over the molecules of a set, the reference "
            "molecule counted as in Table S1; differences from the paper over the other molecules.\n",
            "| set | molecules | method | MAE here | MAE in the paper | mean abs. difference from the paper | largest |",
            "|---|---:|---|---:|---:|---:|---:|"]
    for name in (*SETS, "all"):
        members = [e for e in results if name in ("all", e["set"]) and "pbe_shift" in e]
        rows = [e for e in members if e["index"] != 1]
        for column, (key, paper_key, label) in enumerate(METHODS):
            if not rows:
                continue
            here = xps.mean(abs(e[f"{key}_shift"] - e["paper"]["exp_shift"]) for e in members)
            differences = [abs(e[f"{key}_shift"] - e["paper"][paper_key]) for e in rows]
            out.append(f"| {name} | {len(members)} | {label} | {here:.3f} | {PRINTED_MAE[name][column]:.2f} | "
                       f"{xps.mean(differences):.3f} | {max(differences):.3f} |")
    out.append("")
    rows = [e for e in results if e["set"] == "N" and "logs" in e]
    reference = next((e for e in rows if e["index"] == 1), None)
    if reference and len(rows) > 1:
        out += ["## N 1s shifts refined with alpha times the exact exchange of B3LYP (Table S2), eV\n",
                "From the parts of the B3LYP energy of each case. Errors against experiment, the reference molecule "
                "counted.\n",
                "| alpha | MAE here | paper | mean error here | paper |", "|---:|---:|---:|---:|---:|"]
        for alpha, printed_mae, printed_me in zip(range(6), TABLE_S2_MAE, TABLE_S2_ME):
            def binding(entry):
                initial, final = entry["logs"]
                return (xps.scaled_exchange_total(final, alpha)
                        - xps.scaled_exchange_total(initial, alpha)) * xps.RYDBERG_EV
            errors = [binding(e) - binding(reference) - e["paper"]["exp_shift"] for e in rows]
            out.append(f"| {alpha} | {xps.mean(abs(x) for x in errors):.3f} | {printed_mae:.2f} | "
                       f"{xps.mean(errors):+.3f} | {printed_me:+.2f} |")
        out.append("")
    return "\n".join(out)


def check(results: list[dict], stored: list[dict]) -> list[str]:
    """Deviations beyond the tolerances: new runs against the stored reference and against the paper."""

    problems = []
    for entry, old in zip(results, stored):
        label = f"{entry['set']}/{entry['index']:02d} {entry['molecule']}"
        if "pbe" not in entry:
            problems.append(f"{label}: not run")
            continue
        for column, (key, paper_key, name) in enumerate(METHODS):
            if "pbe" in old and abs(entry[key] - old[key]) > TOLERANCE["reference"]:
                problems.append(f"{label}: {name} {entry[key]:.4f}, reference {old[key]:.4f}")
            if entry["index"] == 1:
                shifted = entry[key] - POLARIZED_SHIFT[entry["set"]]
                if abs(shifted - REFERENCE_ABSOLUTE[entry["set"]][column]) > TOLERANCE["paper_absolute"]:
                    problems.append(f"{label}: {name} {shifted:.3f}, paper {REFERENCE_ABSOLUTE[entry['set']][column]}")
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
              f"all {len(logs)} cases within {TOLERANCE['reference']} eV of the reference; reference molecules within "
              f"{TOLERANCE['paper_absolute']} eV and shifts within {TOLERANCE['paper_shift']} eV of the paper")
        raise SystemExit(1 if problems else 0)
    text = report(binding_energies(logs if source == "logs" else xps.read_reference(REFERENCE)))
    print(text)
    if arguments.write:
        origin = "the logs in `cases/`" if source == "logs" else "`reference_energies.csv`"
        (HERE / "RESULTS.md").write_text(
            f"# 1s binding energies of the 2022 set\n\nWritten by `analyze.py --write` from {origin}.\n\n" + text,
            encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
