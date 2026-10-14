# Ground-state norm-conserving pseudopotentials, H to No

Troullier–Martins pseudopotentials for the ground state of each element, for
two functionals, in three formats. Generated in the scheme of FHI98PP and
regenerated with that program for comparison (`../README.md`, which also
lists what to cite); this README describes the build of 12 October 2026.

| folder | functional | solver setting | elements |
|---|---|---|---|
| `PBE/` | Perdew–Burke–Ernzerhof GGA | `Correlation_Type: pbe` | 102: H–No |
| `CA/` | LDA, Ceperley–Alder / Perdew–Zunger | `Correlation_Type: ca` | 101: H–No without Ho |

Not provided: Lr (no ghost-free form with either functional) and Ho with CA
(the atom does not converge). The matching potentials with a core hole are in
`../CoreHole_NCPPs`.

## Layout

```text
<XC>/<Element>/
    PARSEC/<Element>_POTRE.DAT     read by the Python solver (--pp-dir)
    PARSEC/parsec_atom_type.in     the Atom_Type lines for parsec.in
    UPF/<prefix>.UPF               UPF v2 (Quantum ESPRESSO, ARES)
    FHI/<Element>_FHIPP.DAT        FHI98PP table on the generation mesh
    diagnostics/                   input (.ini), log (.dat), test reports, report.json
manifest.csv                       one row per potential
```

`parsec_atom_type.in` holds the two values a calculation needs besides the
file: `Local_Component`, the channel that passed the ghost test, and
`Atomic_Energy_Correction`, the all-electron minus pseudo energy of the atom.
Do not use another local component than the listed one without testing it; for
many elements the other channels have ghost states.

The `.ini` file in a `diagnostics` folder is the FHI98PP input of that
potential; FHI98PP regenerates the potential from it.

## What was chosen

The rules are in `../README.md`. In this build (the same for PBE and CA unless
noted):

| `lmax` | local | elements |
|---|---|---|
| 0 | s (purely local) | H, He, Fr |
| 1 | p | K, Ga, Ge, Rb, In, Sn, Sb, I, Xe, Tl, Pb, Bi, Po |
| 2 | d | Li–Ne, Na–Ar, As, Se, Br, Kr, Te, At, Rn |
| 2 | p | Sc–Cr, Sr–Cd, Cs, Ba, La, Hf–Hg, Ra, Ac |
| 2 | s | Ca, Mn–Zn |
| 3 | p | Ce–Lu, Th–Bk (CA: also Cf) |
| 3 | s | Cf (PBE), Es, Fm, Md, No |

- Elements with `lmax` below that of the FHI98PP element database (K, Ga, Ge, Rb,
  In, Sn, Sb, I, Xe, Tl, Pb, Bi, Po from 2 to 1; La, Ac from 3 to 2): with that `lmax`
  no local channel is ghost-free. The dropped channel holds no valence
  electrons.
- H is purely local by review. Fr is purely local because neither its p nor its
  d channel gives a ghost-free form: a weak potential for an element with
  low-lying p and d states.
- Nonlinear core correction, as in the element database: Li, B, Na, Mg, K, Ca, Rb, Sr,
  In, Cs, Ba, Tl–Rn, Fr, Ra, Ac–No.
- A radius differs from the element database by one or two mesh points (2.5 %
  each), because a member of the family (ground state or a core hole) cannot be
  constructed at the database value: K and Y (s radius);
  La, Gd, Cm, Es, Fm with both functionals; Md, No with PBE; Ac, Th with CA.
  The `note` column of the manifest gives the channel and the step.
- Smallest ghost-free margins: Li 0.0031 Ha (PBE), 0.0033 (CA); Cu 0.0045,
  0.0050; K 0.0060, 0.0063. All others are above 0.006 Ha.

The PBE potentials of H–Kr replace the earlier library of this repository
(made with the FHI98PP program). They have the same `lmax` and local channel
for all 36 elements, and for 35 of them the same tables (FHI files to 4e-10,
PARSEC files to 1e-12); K differs in the s radius as stated above.

## Checks

- Every potential was regenerated with the FHI98PP program from the input in
  its `diagnostics` folder: 203 of 203 reproduced; tables agree to 4e-9, total
  energies to the five decimals the FHI98PP log prints, and the ghost verdicts
  of the selected local channel agree in all cases. (For Z > 80 a copy of the
  program with one array bound corrected was used; the original writes
  outside its table of default radii there.)
- `ecut_1meV_Ry` in the manifest is the plane-wave cutoff at which the kinetic
  energy of the pseudo wave functions is converged to 1 meV per state, from the
  momentum-space test of the generation. It ranges from 7 Ry (Cs) to 169 Ry (Ne) and
  is a guide for the grid spacing (pi/sqrt(Ecut) bohr), not a convergence test
  of a molecule.

What has not been checked: the potentials in molecules or solids, apart from
H, C and F in a CH4/CF4 test, where C 1s binding energies agree with
all-electron calculations to 0.08 eV (`examples/0_CH4_CF4/python_library_pps`).
The potentials
of Z = 57–71 and 81–102 come from inputs written for this library
(`../README.md`); those with an f shell in the valence (Ce–Lu, Pa–No)
are hard, and none of them has been used in a calculation here.
