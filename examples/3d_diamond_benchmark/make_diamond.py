"""PARSEC input of a bulk-diamond supercell: n1 x n2 x n3 conventional cubic cells (8 atoms each), PBE.

Settings as the nanodiamond series (examples/0d_nanodiamonds/gpu_benchmark): grid 0.198 A (18 points per cubic edge
a = 3.567 A), Expansion_Order 8, the same eigensolver (chebff, first solve 3 x degree 30, then degree 15 + 3, buffer
6, Diag_Tolerance 1e-4) and mixing (Anderson 0.3, memory 4, safeguard) blocks, Convergence_Criterion 2e-4 Ry,
Fermi_Temp 500 K, occupied + 6 states; the PBE carbon pseudopotential of src/pp_generation/NCPPs/PBE/C/PARSEC
(martins_new, local d, cubic spline).

Usage: python make_diamond.py <n1> <n2> <n3> <output parsec.in> [--kmesh K1 K2 K3] [--extra-states N]
       [--set 'Keyword: value' ...]   (replaces the line of that keyword, or adds it before Atom_Types_Num)
The pseudopotential file C_POTRE.DAT must sit next to the input (copy it from the NCPPs library).
--kmesh writes a Gamma-centred mesh (a zero Monkhorst_Pack_Shift) with Use_Fortran_Routines: .false., so that the
grid stays at 18 points per edge and the mesh is reduced to the irreducible wedge on the default routines.
"""

import argparse
import math
from pathlib import Path

A_ANG = 3.567
BOHR_ANG = 0.529177210903
A_BOHR = A_ANG / BOHR_ANG
POINTS_PER_EDGE = 18
BASIS = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]


def atoms(n1, n2, n3):
    out = []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                for shift in (0.0, 0.25):
                    for b in BASIS:
                        out.append(((i + b[0] + shift) * A_BOHR, (j + b[1] + shift) * A_BOHR, (k + b[2] + shift) * A_BOHR))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs=3)
    parser.add_argument("output")
    parser.add_argument("--kmesh", type=int, nargs=3)
    parser.add_argument("--extra-states", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="'KEYWORD: VALUE'")
    args = parser.parse_args()
    n1, n2, n3 = args.n
    coords = atoms(n1, n2, n3)
    occupied = 2 * len(coords)  # 4 valence electrons per carbon
    extra = args.extra_states if args.extra_states is not None else 6  # as the nanodiamond inputs
    lengths = [n * A_BOHR for n in (n1, n2, n3)]
    kpoints = ""
    label = "Gamma point"
    if args.kmesh:
        k1, k2, k3 = args.kmesh
        label = f"Gamma-centred {k1}x{k2}x{k3} Monkhorst-Pack mesh, irreducible wedge"
        # A Monkhorst_Pack_Shift block alone selects the Fortran-derived routines, whose grid follows the space group
        # (16 points per cubic edge for diamond's quarter translations); the default routines keep the grid of
        # Grid_Spacing (18 points per edge) and reduce the mesh by the operations that map it onto itself (Td about
        # an atom) and time reversal, which gives the wedge of Oh.
        kpoints = ("# the default routines: the grid of Grid_Spacing (18 points per cubic edge), the mesh reduced by\n"
                   "# the space-group operations that map it onto itself and time reversal\n"
                   "Use_Fortran_Routines: .false.\n"
                   f"Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n{k1} {k2} {k3}\nend Monkhorst_Pack_Grid\n"
                   "begin Monkhorst_Pack_Shift\n0.0 0.0 0.0\nend Monkhorst_Pack_Shift\n")
    text = f"""# Bulk diamond, {n1}x{n2}x{n3} conventional cubic cells ({len(coords)} atoms, a = {A_ANG} A), {label}, PBE
# settings of the nanodiamond series: grid a/18 = 0.198 A, Expansion_Order 8, chebff, 2e-4 Ry

Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  {lengths[0]:.10f} 0.0 0.0
  0.0 {lengths[1]:.10f} 0.0
  0.0 0.0 {lengths[2]:.10f}
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: {A_BOHR / POINTS_PER_EDGE * (1.0 - 1.0e-9):.14f}
Expansion_Order: 8
Correlation_Type: pbe
Spin_Polarization: false
States_Num: {occupied + extra}
Fermi_Temp: 500 K
Max_Iter: 70
Convergence_Criterion: 2e-4 Ry

Eigensolver: chebff
Chebdav_Degree: 30
FF_MaxIter: 3
Matvec_Blocksize: 6
Chebyshev_Degree: 15
Chebyshev_Degree_Delta: 3
Subspace_Buffer_Size: 6
Diag_Tolerance: 1e-4

Mixing_Method: Anderson
Mixing_Param: 0.30
Memory_Param: 4
Restart_Mixing: 20
Mixing_Safeguard: true
Mixing_Step_Limit: 2.0
Mixing_Growth_Trigger: 2.0
Mixing_Backoff: 0.5

Initial_Density: sad
Normalize_Initial_Density: true
Skip_Force: true
Output_Level: 1
Output_All_States: false
{kpoints}Atom_Types_Num: 1

Atom_Type: C
Pseudopotential_Format: martins_new
Local_Component: d
Cubic_Spline: true
begin Atom_Coord
""" + "".join(f"  {x:.12f} {y:.12f} {z:.12f}\n" for x, y, z in coords) + "end Atom_Coord\n"
    for setting in args.set:
        keyword = setting.split(":", 1)[0].strip()
        lines = text.split("\n")
        hits = [i for i, line in enumerate(lines) if line.split(":", 1)[0].strip().lower() == keyword.lower()]
        if hits:
            lines[hits[0]] = setting
        else:
            lines.insert(lines.index("Atom_Types_Num: 1"), setting)
        text = "\n".join(lines)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)
    print(f"{args.output}: {len(coords)} atoms, {occupied} occupied + {extra} empty states, grid "
          f"{n1 * POINTS_PER_EDGE} x {n2 * POINTS_PER_EDGE} x {n3 * POINTS_PER_EDGE}")


if __name__ == "__main__":
    main()
