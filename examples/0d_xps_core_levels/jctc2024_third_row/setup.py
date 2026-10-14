#!/usr/bin/env python3
"""Write the S and P core-level cases of Liu et al., JCTC 20, 6134 (2024).

For every molecule that ``paper/tables.csv`` names (structures in
``paper/structures/<formula>.xyz``):

    cases/<formula>/IS/parsec.in               neutral molecule
    cases/<formula>/FS_<S|P>-<1s|2s|2p>/parsec.in
                                    the first S (P) atom carries the potential
                                    made with one electron removed from that
                                    shell, and the molecule the charge +1

    python setup.py                  # the cases as stored
    python setup.py --spacing 0.08   # another grid spacing (angstrom)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import xps_cases as xps  # noqa: E402


def table_rows() -> list[dict]:
    with open(HERE / "paper" / "tables.csv", encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spacing", type=float, default=xps.SPACING_ANGSTROM, help="grid spacing in angstrom")
    parser.add_argument("--cases", type=Path, default=HERE / "cases", help="folder to write the cases to")
    arguments = parser.parse_args()

    species = xps.species_table()
    index, written = [], set()
    for row in table_rows():
        formula, label = row["structure"], f"{row['element']}-{row['shell']}"
        _, atoms = xps.read_xyz(HERE / "paper" / "structures" / f"{formula}.xyz")
        holder = next(number for number, (symbol, _) in enumerate(atoms) if symbol == row["element"])
        for state, hole in (("IS", None), (f"FS_{label}", (holder, label))):
            folder = f"{formula}/{state}"
            if folder in written:
                continue
            written.add(folder)
            facts = xps.write_case(arguments.cases / folder, atoms, species, hole=hole, spacing=arguments.spacing)
            index.append({"molecule": row["molecule"], "structure": formula, "state": state, "folder": folder, **facts})
    xps.write_index(arguments.cases, index)
    print(f"{len(index)} cases written under {arguments.cases} (spacing {arguments.spacing:g} angstrom)")


if __name__ == "__main__":
    main()
