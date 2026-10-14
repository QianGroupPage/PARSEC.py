#!/usr/bin/env python3
"""Write the protonated-water cases of Figure 5 of Xu, Prendergast, Qian, JCTC 18, 5471 (2022).

For every cluster H3O+(H2O)n of ``paper/structures/<nn>.xyz``, n = 1 ... 20:

    cases/<nn>/IS/parsec.in    the cluster, charge +1
    cases/<nn>/FS/parsec.in    the hydronium oxygen (the one with three hydrogens
                               bound) carries the potential made with one 1s
                               electron removed; charge +2

The water molecule that the binding energies are referred to is case ``O/09``
of ``../jctc2022_1s``.

    python setup.py                  # the cases as stored
    python setup.py --spacing 0.08   # another grid spacing (angstrom)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import xps_cases as xps  # noqa: E402

#: an O-H distance below this (angstrom) counts as a bond
BOND = 1.25


def hydronium_oxygen(atoms) -> int:
    """Number of the oxygen with three hydrogens bound."""

    hydrogens = [xyz for symbol, xyz in atoms if symbol == "H"]
    found = [number for number, (symbol, xyz) in enumerate(atoms)
             if symbol == "O" and sum(math.dist(xyz, h) < BOND for h in hydrogens) == 3]
    if len(found) != 1:
        raise ValueError(f"expected one hydronium oxygen, found {len(found)}")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spacing", type=float, default=xps.SPACING_ANGSTROM, help="grid spacing in angstrom")
    parser.add_argument("--cases", type=Path, default=HERE / "cases", help="folder to write the cases to")
    arguments = parser.parse_args()

    species = xps.species_table()
    index = []
    for path in sorted((HERE / "paper" / "structures").glob("*.xyz")):
        _, atoms = xps.read_xyz(path)
        holder = hydronium_oxygen(atoms)
        for state, hole in (("IS", None), ("FS", (holder, "O-1s"))):
            folder = f"{path.stem}/{state}"
            facts = xps.write_case(arguments.cases / folder, atoms, species, hole=hole, charge=1,
                                   spacing=arguments.spacing)
            index.append({"n": int(path.stem), "molecule": f"H3O+(H2O){int(path.stem)}", "state": state,
                          "folder": folder, **facts})
    xps.write_index(arguments.cases, index)
    print(f"{len(index)} cases written under {arguments.cases} (spacing {arguments.spacing:g} angstrom)")


if __name__ == "__main__":
    main()
