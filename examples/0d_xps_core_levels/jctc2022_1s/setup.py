#!/usr/bin/env python3
"""Write the 1s core-level cases of Xu, Prendergast, Qian, JCTC 18, 5471 (2022).

For every molecule of ``paper/structures/<B|C|N|O>/<nn>.xyz`` (Table S1 of the
paper, in its order):

    cases/<set>/<nn>/IS/parsec.in    neutral molecule
    cases/<set>/<nn>/FS/parsec.in    the atom of the set carries the potential
                                     made with one 1s electron removed, and the
                                     molecule the charge +1

    python setup.py                  # the cases as stored
    python setup.py --spacing 0.08   # another grid spacing (angstrom)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import xps_cases as xps  # noqa: E402

SETS = ("B", "C", "N", "O")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spacing", type=float, default=xps.SPACING_ANGSTROM, help="grid spacing in angstrom")
    parser.add_argument("--cases", type=Path, default=HERE / "cases", help="folder to write the cases to")
    arguments = parser.parse_args()

    species = xps.species_table()
    index = []
    for name in SETS:
        for path in sorted((HERE / "paper" / "structures" / name).glob("*.xyz")):
            molecule, atoms = xps.read_xyz(path)
            holders = [number for number, (symbol, _) in enumerate(atoms) if symbol == name]
            if len(holders) != 1:
                raise ValueError(f"{path}: expected exactly one {name} atom")
            for state, hole in (("IS", None), ("FS", (holders[0], f"{name}-1s"))):
                folder = f"{name}/{path.stem}/{state}"
                facts = xps.write_case(arguments.cases / folder, atoms, species, hole=hole, spacing=arguments.spacing)
                index.append({"set": name, "index": int(path.stem), "molecule": molecule, "state": state,
                              "folder": folder, **facts})
    xps.write_index(arguments.cases, index)
    print(f"{len(index)} cases written under {arguments.cases} (spacing {arguments.spacing:g} angstrom)")


if __name__ == "__main__":
    main()
