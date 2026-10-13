#!/usr/bin/env python3
"""Build the CH4/CF4 C 1s cases with the pseudopotentials of the shipped libraries.

The geometry, grid and solver controls are those of ``../python_pbe`` (the
reproduction of the ARES benchmark).  Only the pseudopotentials change: the
ground-state potentials come from ``src/pp_generation/NCPPs`` and the carbon
potential with a 1s hole from ``src/pp_generation/CoreHole_NCPPs``.  Local
component and atomic energy correction of every species are read from the
``parsec_atom_type.in`` file stored beside each potential.

    python setup.py            # PBE and CA
    python setup.py --xc pbe
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PACKAGE = REPOSITORY / "src" / "pp_generation"
SOURCE = HERE.parent / "python_pbe"
XC = {"pbe": ("PBE", "pbe"), "ca": ("CA", "ca")}
BLOCK = re.compile(r"^Atom_Type:.*?(?=^begin Atom_Coord)", re.M | re.S)


def species_directory(label: str, xc_dir: str) -> Path:
    """Library folder of an input species such as ``C`` or ``C-1s``."""

    element = label.split("-")[0]
    if "-" in label:
        return PACKAGE / "CoreHole_NCPPs" / xc_dir / element / label / "PARSEC"
    return PACKAGE / "NCPPs" / xc_dir / element / "PARSEC"


def build(xc: str) -> None:
    xc_dir, correlation = XC[xc]
    potentials = HERE / xc / "pseudopotentials"
    potentials.mkdir(parents=True, exist_ok=True)
    for molecule in ("CH4", "CF4"):
        for state in ("IS", "FS_1s"):
            text = (SOURCE / molecule / state / "parsec.in").read_text(encoding="utf-8")

            def replace(match: re.Match[str]) -> str:
                label = match.group(0).splitlines()[0].split(":", 1)[1].strip()
                folder = species_directory(label, xc_dir)
                shutil.copy2(folder / f"{label}_POTRE.DAT", potentials / f"{label}_POTRE.DAT")
                snippet = (folder / "parsec_atom_type.in").read_text(encoding="utf-8")
                lines = [x for x in snippet.splitlines() if x.strip() and not x.startswith("#")]
                # keep the reader options of the benchmark input
                lines[-1:-1] = ["Read_VCD: true", "Cubic_Spline: true"]
                return "\n".join(lines) + "\n"

            text = BLOCK.sub(replace, text)
            text = re.sub(r"^Correlation_Type:.*$", f"Correlation_Type: {correlation}", text, flags=re.M)
            target = HERE / xc / molecule / state
            target.mkdir(parents=True, exist_ok=True)
            (target / "parsec.in").write_text(text, encoding="utf-8", newline="\n")
    print(f"{xc}: four cases written under {HERE / xc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xc", choices=tuple(XC), nargs="+", default=list(XC))
    for name in parser.parse_args().xc:
        build(name)
