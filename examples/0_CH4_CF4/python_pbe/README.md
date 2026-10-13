# Python reproduction of the ARES CH4/CF4 C 1s benchmark

This directory reproduces the PBE part of the supplied ARES benchmark with
the native Python real-space solver.  It does not call ARES or PARSEC.

## Directory correspondence

The supplied `0_ARES_CH4_CF4` tree contains three distinct pieces:

- `CH4/` is an earlier standalone/manual ARES trial (`ares.in`, `POSCAR`,
  ordinary C/H UPFs, Slurm script and output).  Its zero-length `ares.log` is
  not the production reference.
- `CF4/` contains only the source `CF4.xyz` geometry; it is not a calculation.
- `test_CH4_CF4/` is the complete, generated eight-case validation suite and
  is the reference used here.  Its `setup.py` builds cases, `run*.qsub`/`.sh`
  launch them, and `analyze.py` forms binding energies and shifts.  Salvaged
  and Slurm files are run history, not additional physical cases.

The production calculation tree is

```text
0_ARES_CH4_CF4/test_CH4_CF4/
  pbe/{CH4,CF4}/{IS,FS_1s}/
  b3lyp/{CH4,CF4}/{IS,FS_1s}/
```

`IS` is the neutral ground-state calculation with the ordinary carbon
pseudopotential.  `FS_1s` is the fully screened final state with a carbon
pseudopotential generated after removing one 1s electron.  The ARES `b3lyp`
tree first performs the same PBE SCF and then evaluates one non-self-consistent
B3LYP energy; it is not a B3LYP SCF calculation.

Inside each terminal ARES case, `ares.in` holds numerical controls, `POSCAR`
the coordinates, `ISOCAR` the species-to-PP assignment, and the copied UPFs
the actual pseudopotentials.  `ares.log` is the scalar/energy report and
`ares.band` the eigenvalue output; `srun.err` is launcher stderr.

The Python PBE tree mirrors the physical cases:

```text
python_pbe/
  CH4/{IS,FS_1s}/
  CF4/{IS,FS_1s}/
  pseudopotentials/
  analyze.py
```

The input label `C-1s` means “carbon with one full 1s hole.”  It remains tied
to physical element `C` through `Element_Symbol: C`, so the label changes PP
selection without inventing a chemical element.  The final-state inputs use
`Net_Charges: 1 e`; the C-1s PP has five valence electrons, leaving the same
explicit electron count as the ARES `DCHARGE=-1` convention.

## Numerical correspondence

The molecular coordinates, grid spacings, finite-difference order, number of
states, electron temperature, PBE pseudopotentials, and spherical domains are
taken from the ARES runs.  The radii are written with enough precision to give
the same Cartesian box and active point counts:

| molecule | spacing | Cartesian box | active points |
|---|---:|---:|---:|
| CH4 | 0.12 angstrom | 102 x 102 x 102 | 539,152 |
| CF4 | 0.07 angstrom | 182 x 182 x 182 | 3,103,688 |

The Python SCF uses its audited Chebyshev-filtered subspace solver and Anderson
potential mixing.  ARES reports a density residual, whereas the Python solver
tests a charge-weighted potential residual; the numerical tolerances therefore
do not have identical meanings.  The CF4 threshold is `5e-5 Ry`, at which its
total energy is already stable much more tightly than the core-level chemical
shift being tested.

ARES inputs request spin polarization, but their final spin-up and spin-down
bands are paired and equal for these closed-shell, fully screened cases.  The
present Python calculations are therefore unpolarized.  A genuinely magnetic
or fractional-hole calculation will require a spin-PBE implementation before
it is a valid reproduction.

## Pseudopotential and all-electron reference

The POTRE files were converted from the exact UPF files in the supplied ARES
case, which were themselves produced by FHI98PP.  For example, from the
repository root:

```powershell
.venv312\Scripts\python.exe src\tools\upf_to_parsec.py `
  examples\0_CH4_CF4\0_ARES_CH4_CF4\test_CH4_CF4\pbe\CH4\IS\C.pbe-mt-cpi.UPF `
  examples\0_CH4_CF4\python_pbe\pseudopotentials\C_POTRE.DAT

.venv312\Scripts\python.exe src\tools\upf_to_parsec.py `
  --allow-ionized-reference `
  examples\0_CH4_CF4\0_ARES_CH4_CF4\test_CH4_CF4\pbe\CH4\FS_1s\C.pbe-mt-cpi-1s.UPF `
  examples\0_CH4_CF4\python_pbe\pseudopotentials\C-1s_POTRE.DAT
```

The initial and final carbon pseudopotentials have different ionic charges,
so their raw pseudopotential total energies do not share an energy zero.  Each
species input therefore includes the FHI98PP all-electron-minus-pseudo atomic
reference correction.  Binding energies must be formed from the reported
`Reference-corrected all-electron total`,

```text
BE(C 1s) = E_AE(FS_1s) - E_AE(IS).
```

The chemically transferable validation quantity is the shift

```text
Delta BE = BE(CF4) - BE(CH4).
```

## Running

From any one of the four case directories:

```powershell
..\..\..\..\..\..\..\.venv312\Scripts\python.exe `
  ..\..\..\..\..\main.py parsec.in `
  --backend auto --symmetry auto --log parsec.out `
  --no-archive
```

Remove `--no-archive` and add `--output parsec_result.npz` when the converged
grid fields are needed.  CF4 has more than three million active points, so its
archive is intentionally omitted for an energy-only reproduction.

After all four jobs finish, compare directly with the supplied ARES logs:

```powershell
.venv312\Scripts\python.exe `
  examples\0_CH4_CF4\python_pbe\analyze.py
```

## PBE reproduction results

All four Python calculations converged.  Energies below are the
reference-corrected all-electron values, in eV:

| system | Python | ARES | Python - ARES |
|---|---:|---:|---:|
| CH4 IS | -1101.55954131 | -1101.48773900 | -0.07180231 |
| CH4 FS_1s | -804.40558245 | -804.34015900 | -0.06542345 |
| CF4 IS | -11909.77825593 | -11909.43066320 | -0.34759273 |
| CF4 FS_1s | -11602.06985734 | -11601.71550480 | -0.35435254 |

The physically relevant energy differences reproduce much more closely:

| quantity | Python | ARES | Python - ARES |
|---|---:|---:|---:|
| BE(CH4) | 297.15395886 | 297.14758000 | +0.00637886 |
| BE(CF4) | 307.70839859 | 307.71515840 | -0.00675981 |
| CF4-CH4 shift | 10.55443973 | 10.56757840 | -0.01313867 |

The larger offsets in individual absolute totals are systematic discretization
and implementation differences.  They cancel within each matched ordinary/
core-hole PP pair; the final chemical-shift disagreement is about 13 meV.

For this cold run, CH4 IS/FS used 28/21 SCF iterations and CF4 IS/FS used
40/37.  CF4 wall times were 793/627 seconds on the tested hybrid machine.  Its
dominant cost was the native Hartree linear solve (446/407 seconds), followed
by GPU diagonalization (225/111 seconds); PBE evaluation itself was not the
dominant kernel.

## B3LYP status

The ARES one-shot B3LYP refinement cannot be represented by another local
`V_xc(r)` switch.  It contains nonlocal Hartree-Fock exchange and therefore
requires occupied-orbital pair densities and Poisson solves (or an equivalent
exact-exchange algorithm), in addition to B88 exchange and LYP/VWN
correlation.  It is deliberately not approximated in this PBE milestone.
Implementing it as a separate post-SCF energy module preserves both the code
architecture and the physical meaning of the ARES result.

That module now exists (`Energy_Refinement: b3lyp`).  The same four cases
with the refinement, the comparison with the ARES `b3lyp` tree and the reason
why the energy parts of the two codes differ are in
[`../python_pbe_b3lyp`](../python_pbe_b3lyp/README.md).  Its tables convert
Rydberg to eV with the CODATA constant; the tables above use the `[eV]` lines
of `parsec.out` (13.6058 eV per Rydberg), which puts their binding energies
2.3 to 2.4 meV higher.
