#!/usr/bin/env python3
"""Write the pseudo-atoms of Table 1 of Xu, Prendergast, Qian, JCTC 18, 5471 (2022).

For B, C, N and O, the atom in its ground-state potential and the ion in the
potential made with one 1s electron removed (charge +1), each without and
with spin polarization:

    cases/<element>/<atom|ion>/<unpolarized|polarized>/parsec.in, occup.in

The occupations are fixed (``Fermi_Temp`` negative, ``occup.in``): 2s filled,
the 2p electrons spread evenly over the three p orbitals, and with spin
polarization the 2p shell filled by Hund's rule, so the density is spherical.

    python setup.py                  # the cases as stored
    python setup.py --spacing 0.07   # another grid spacing (angstrom)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import xps_cases as xps  # noqa: E402

#: 2p electrons of the neutral atom
P_ELECTRONS = {"B": 1, "C": 2, "N": 3, "O": 4}
RADIUS_ANGSTROM = 7.0
SPACING_ANGSTROM = 0.10
STATES = 16

INPUT = """\
Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere
Boundary_Sphere_Radius: {radius:g} ang
Grid_Spacing: {spacing:g} ang
Expansion_Order: 8
Coordinate_Unit: Cartesian_Ang
Ignore_Symmetry: false

Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: {label}
Element_Symbol: {element}
Pseudopotential_Format: martins_new
Local_Component: {local}
Read_VCD: true
Cubic_Spline: true
{initial}begin Atom_Coord
 0.0 0.0 0.0
end Atom_Coord

Correlation_Type: pbe
Spin_Polarization: {spin}
States_Num: {states}
Net_Charges: {charge} e
Fermi_Temp: -1 K
Max_Iter: 150
Convergence_Criterion: 1e-6 Ry

Eigensolver: chebff
Chebdav_Degree: 40
FF_MaxIter: 4
Matvec_Blocksize: 16
Chebyshev_Degree: 20
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


def section(title: str, s_occupation: float, p_occupation: float) -> list[str]:
    """One section of ``occup.in``: the 2s state, then the three 2p states."""

    lines = [title, "0", "4", f"1 {s_occupation:.12f}"]
    return lines + [f"{state} {p_occupation:.12f}" for state in (2, 3, 4)]


def occupations(element: str, polarized: bool) -> str:
    electrons = P_ELECTRONS[element]
    if not polarized:
        # one section; an occupation of one is a doubly occupied orbital
        return "\n".join(section("2s2 2p%d, spherical" % electrons, 1.0, electrons / 6.0)) + "\n"
    up, down = min(electrons, 3), max(electrons - 3, 0)
    return "\n".join(section(f"spin up: 2s1 2p{up}", 1.0, up / 3.0)
                     + section(f"spin down: 2s1 2p{down}", 1.0, down / 3.0)) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spacing", type=float, default=SPACING_ANGSTROM, help="grid spacing in angstrom")
    parser.add_argument("--radius", type=float, default=RADIUS_ANGSTROM, help="radius of the domain in angstrom")
    parser.add_argument("--cases", type=Path, default=HERE / "cases", help="folder to write the cases to")
    arguments = parser.parse_args()

    species = xps.species_table()
    index = []
    for element in P_ELECTRONS:
        for kind, label, charge in (("atom", element, 0), ("ion", f"{element}-1s", 1)):
            entry = species[label]
            for spin in ("unpolarized", "polarized"):
                polarized = spin == "polarized"
                folder = arguments.cases / element / kind / spin
                folder.mkdir(parents=True, exist_ok=True)
                moment = min(P_ELECTRONS[element], 6 - P_ELECTRONS[element])
                initial = (f"Initial_Spin_Polarization: {moment / entry['valence_electrons']:.4f}\n"
                           if polarized else "")
                (folder / "parsec.in").write_text(INPUT.format(
                    radius=arguments.radius, spacing=arguments.spacing, label=label, element=element,
                    local=entry["local"], initial=initial,
                    spin="true" if polarized else "false", states=STATES, charge=charge),
                    encoding="utf-8", newline="\n")
                (folder / "occup.in").write_text(occupations(element, polarized), encoding="utf-8", newline="\n")
                points = 4.0 / 3.0 * math.pi * (arguments.radius / arguments.spacing) ** 3
                index.append({"element": element, "molecule": f"{label} {spin}", "kind": kind, "spin": spin,
                              "folder": f"{element}/{kind}/{spin}", "occupied": 4, "states": STATES,
                              "radius_angstrom": arguments.radius, "grid_points_estimate": round(points)})
    xps.write_index(arguments.cases, index)
    print(f"{len(index)} cases written under {arguments.cases} (spacing {arguments.spacing:g} angstrom)")


if __name__ == "__main__":
    main()
