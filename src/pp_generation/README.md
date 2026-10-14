# Pseudopotentials

This folder holds two libraries of norm-conserving pseudopotentials for the
solver, and `pp_generation`, a command-line and Python interface that drives an
installed FHI98PP or ATOM to generate and ghost-check further ones.

## The libraries

```text
NCPPs/<XC>/<Element>/                         ground state
CoreHole_NCPPs/<XC>/<Element>/<Element>-<shell>/    one electron removed from <shell>
    PARSEC/   <label>_POTRE.DAT, parsec_atom_type.in
    UPF/      *.UPF
    FHI/      <label>_FHIPP.DAT
    diagnostics/   input, log, test reports, report.json
NCPPs/manifest.csv, CoreHole_NCPPs/manifest.csv
```

`<XC>` is `PBE` or `CA`:

| folder | functional | FHI98PP `iexc` | UPF label | solver setting |
|---|---|---|---|---|
| `PBE` | Perdew–Burke–Ernzerhof GGA | 6 | `PBE` | `Correlation_Type: pbe` |
| `CA` | LDA, Ceperley–Alder in the Perdew–Zunger 1981 parametrization | 3 | `PZ` | `Correlation_Type: ca` |

Each library has its own README with the numbers of the current build; each
`manifest.csv` has one row per potential with status, `lmax`, local channel,
smallest ghost margin, ionic charge, both total energies, the atomic energy
correction, cutoff radii, core radius, the shift of the reference energies,
the plane-wave cutoff at which the kinetic energy is converged to 1 meV, and
whether a core hole is matched to its ground state.

A `_POTRE.DAT` file is the UPF file of the same potential, resampled onto the
radial mesh of the solver by `src/tools/upf_to_parsec.py`; `report.json` holds
the SHA-256 of both. Since 12 October 2026 that resampling is cubic. The files
of the repository before that date were resampled linearly and differ from the
present ones at the radii between those of the UPF mesh, by up to 5e-3 Ry in
the potentials of the light elements (the heaviest change most, up to
0.03 Ry bohr in `r V`). With the former files the total energy of the atom
was off by -6.8e-4 Ry for O, -0.8e-4 Ry for C, +3.3e-4 Ry for Si and
+8.7e-4 Ry for Cl (PBE, pseudo-atom solved on a radial mesh); with the
present ones it agrees with the generation within 1e-6 Ry. Energy differences
between structures with the same potentials change much less.
`upf_to_parsec.py --interpolation linear` writes the former files.

Both libraries are complete in this branch; each manifest lists every
potential.

### Using a potential

Point the solver at the `PARSEC` folder of the element (`--pp-dir`) or copy
the `_POTRE.DAT` file beside the input, and paste `parsec_atom_type.in` into
`parsec.in`:

```text
Atom_Type: C-1s
Element_Symbol: C
Pseudopotential_Format: martins_new
Local_Component: d
Atomic_Energy_Correction: -35.465193 Ry
```

`Local_Component` is the channel that passed the ghost test.
`Atomic_Energy_Correction` is the all-electron minus the pseudo total energy of
the generating atom; with it the solver reports a reference-corrected
all-electron total, which is what binding energies are built from (potentials
with different ionic charge do not share an energy zero).

### Core-hole potentials

A core-hole potential is generated from the atom with the occupation of one
core shell reduced by one electron. Its ionic charge is one higher than that
of the ground-state potential, and the generating atom is a +1 ion.

*Matched to the ground state.* A binding energy is the difference of two
calculations, one with the ground-state potential and one with the core-hole
potential on the excited atom. So that the two differ only by the hole, a
core-hole potential has the `lmax`, the cutoff radii and the local channel of
the ground-state potential of the same element and functional. Where a hole is
not ghost-free in that form it has its own choice and the manifest says so.

*Using one.* Give the excited atom its own `Atom_Type` (`C-1s`) with
`Element_Symbol: C`. With the photoelectron removed (full core hole) the system
has one electron less: add `1 e` to `Net_Charges`. With `Net_Charges` unchanged
the extra ionic charge is screened by one more valence electron (neutralized
core hole). A 2p hole is spherically averaged; it stands for the average of the
2p1/2 and 2p3/2 holes.

*What is not in it.* The atoms are spin-unpolarized and scalar-relativistic.
An atom with a core hole is strongly spin-polarized, which shifts absolute
binding energies by a constant of several eV per element and shell (Xu,
Prendergast, Qian, J. Chem. Theory Comput. **18**, 5471 (2022)); binding-energy
*shifts* between chemical environments are not affected.

**PBE(B3LYP)** of the binding-energy papers needs no potential of its own. It
is a PBE calculation with PBE potentials followed by one evaluation of the
B3LYP exchange-correlation energy on the PBE orbitals; the atomic energy
correction stays the PBE one. In the solver it is the input line
`Energy_Refinement: b3lyp`; see `examples/0_CH4_CF4/python_pbe_b3lyp`.

### How the libraries were made

Troullier–Martins potentials in the scheme, the input format and the tests of
FHI98PP (references below). They were generated with a Python port of that
program, which is not part of this repository, and then every one of the 1907
potentials was regenerated with the FHI98PP program itself from the input in
its `diagnostics` folder: all are reproduced, tables to 4e-9, total energies to
the five decimals of the FHI98PP log, and the ghost-state verdicts of the
selected local channel are equal. (For Z > 80 a copy of FHI98PP with one array
bound corrected was used; the original writes outside its table of default
radii there.)

The inputs of Z = 1–56 and 72–80 are those of the element database of FHI98PP,
with the departures listed in the READMEs of the libraries. Those of
Z = 57–71 and 81–102 were written for these libraries: where abinit's table of
FHI pseudopotentials exists for the element, the radii were chosen to
reproduce that table (to 1e-10 for the occupied channels); the others follow
their neighbours. Eu, Tb, Dy, Ho use 4f^(n-1) 5d^1 6s^2 because the atom with
4f^n 6s^2 does not converge.

How the form of each potential was chosen:

1. *Ground state.* The highest `lmax` not above the input's and not below an
   occupied valence state that has a ghost-free local channel, then the
   channel with the largest worst-case margin. A margin below 0.002 Ha counts
   only if nothing else passes.
2. *Core holes.* The same input records as the ground state plus the hole.
   Reference energies that an input fixes for an unoccupied channel are
   shifted with the highest valence level, which moves down by 5 to 10 eV.
3. *A member cannot be constructed.* The Troullier–Martins equations have no
   solution at isolated mesh points. Then one channel's radius is moved by one
   or two mesh points (2.5 % each) for the whole family, so radii stay equal.
4. *A hole is not ghost-free in the ground-state form.* It takes its own best
   form and is marked `unmatched`. A hole without any ghost-free form is marked
   `failed`.

What a passed ghost test does not tell: transferability to other chemical
environments, convergence with grid spacing, and whether a semicore shell
should be in the valence. Check a potential on a small system you know before
relying on it. Lanthanide and actinide potentials with the f shell in the
valence are hard and have not been tested in molecules or solids; treat them
as starting points.

### Citing

For work that uses these potentials, cite the program and the methods behind
them:

- M. Fuchs and M. Scheffler, Comput. Phys. Commun. **119**, 67 (1999)
  (FHI98PP);
- N. Troullier and J. L. Martins, Phys. Rev. B **43**, 1993 (1991)
  (construction);
- L. Kleinman and D. M. Bylander, Phys. Rev. Lett. **48**, 1425 (1982), and
  X. Gonze, R. Stumpf, M. Scheffler, Phys. Rev. B **44**, 8503 (1991)
  (separable form and ghost-state analysis);
- S. G. Louie, S. Froyen, M. L. Cohen, Phys. Rev. B **26**, 1738 (1982), for
  the potentials with a nonlinear core correction.

## Generating further potentials

`pp_generation` is a general command-line and Python interface for generating
and ghost-checking norm-conserving pseudopotentials. It is independent of the
DFT solver and delegates atomic generation to an installed FHI98PP or ATOM
toolchain; neither program is distributed here.

The interface supports ordinary reference configurations and explicit
core-hole configurations such as `1s` or `2p`. It can request FHI, PARSEC,
UPF, PSP8, SIESTA, or CPW2000-family output when the selected backend and
installed converters support that format.

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m pp_generation --help
```

Examples:

```powershell
# Ordinary PBE Troullier--Martins Ni potential
python -m pp_generation Ni --backend fhi98pp --xc pbe --scheme tm `
  --format fhi --format parsec --format upf -o generated\Ni

# Remove one electron from the Ni 2p shell
python -m pp_generation Ni --backend fhi98pp --xc pbe --scheme tm `
  --core-hole 2p --hole-charge 1.0 `
  --format fhi --format parsec --format upf -o generated\Ni-2p
```

`--xc ca` is the functional behind `Correlation_Type ca` of the solver
(`iexc = 3`). Until October 2026 this package generated "ca" potentials with
`iexc = 8`, which is the Perdew–Wang 1992 parametrization.

Use `--input-file` for a backend-native expert input when the generic atomic
configuration is insufficient; the `.ini` file in the `diagnostics` folder of
a library potential is such an input. `--local-channel` forces a particular
Kleinman--Bylander local channel; otherwise the package can scan available
choices and reject ghosted candidates. `--allow-ghosts` is intended only for
diagnostics.

Generated files are not scientifically accepted merely because the generator
finishes. Inspect the report and validate logarithmic derivatives,
eigenvalues, cutoff convergence, transferability, ghost states, XC
consistency, core-hole charge, and relativistic requirements before using a
potential in production. In particular, scalar FHI98PP output does not become
spin--orbit-resolved through file conversion.
