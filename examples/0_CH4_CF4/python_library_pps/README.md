# C 1s binding energies of CH4 and CF4 with the library pseudopotentials

The benchmark of `../python_pbe`, repeated with the pseudopotentials of
`src/pp_generation/NCPPs` and `src/pp_generation/CoreHole_NCPPs` instead of the
potentials of the paper, for PBE and for LDA (CA), and compared with
all-electron calculations. Geometries, grids and solver controls are those of
`../python_pbe`; only the pseudopotentials (and for CA the functional) differ.

## Result

Binding energy = reference-corrected all-electron total of the final state
(carbon potential with a 1s hole, one electron removed) minus that of the
initial state. In eV:

| | CH4 | CF4 | shift |
|---|---:|---:|---:|
| **PBE** | | | |
| library potentials | 297.102 | 307.349 | 10.246 |
| all-electron, spin-unpolarized hole | 297.019 | 307.290 | 10.272 |
| all-electron, spin-polarized hole | 289.999 | 300.207 | 10.208 |
| **LDA (CA)** | | | |
| library potentials | 293.905 | 304.442 | 10.537 |
| all-electron, spin-unpolarized hole | 293.832 | 304.382 | 10.550 |
| all-electron, spin-polarized hole | 287.074 | 297.547 | 10.473 |
| experiment | 290.80 | 301.85 | 11.05 |

- The pseudopotential calculation works with spin-unpolarized atoms and
  molecules, so its all-electron counterpart is the calculation with occupation
  1 in the C 1s orbital of a spin-restricted molecule. The library potentials
  agree with it to 0.06–0.08 eV in the binding energies and to 0.03 eV (PBE)
  and 0.01 eV (CA) in the shift.
- Spin polarization of the core hole lowers the binding energies by 7.02 eV
  (CH4) and 7.08 eV (CF4) with PBE: the offset the papers describe, constant to
  0.06 eV between these two molecules.

## Reproducing

```powershell
python examples\0_CH4_CF4\python_library_pps\setup.py        # writes the eight cases
```

then in each of `pbe\{CH4,CF4}\{IS,FS_1s}` and `ca\...`:

```powershell
..\..\..\..\..\..\.venv312\Scripts\python.exe `
  ..\..\..\..\..\..\src\parsec_python\main.py parsec.in `
  --pp-dir ..\..\pseudopotentials `
  --backend auto --symmetry auto --log parsec.out --no-archive
```

and finally `python examples\0_CH4_CF4\python_library_pps\analyze.py`, which
prints the first table. The kept logs are `parsec_python.out` (their file paths
are shortened to the repository root). `setup.py` copies
the potentials from the two libraries and takes `Local_Component` and
`Atomic_Energy_Correction` of each species from the `parsec_atom_type.in` file
beside the potential.

The kept logs are first runs without cache, double precision, with the Poisson
solves on the GPU (`PARSEC_HARTREE_LINEAR_BACKEND=cupy`), on A100 GPUs with
80 GB, one run per GPU: 10–20 s per CH4 case, 59–120 s per CF4 case
(3.1 million grid points). On a laptop with an RTX 5070 Laptop GPU (8 GB) the
same cases took 20–33 s and 336–402 s with the library files as they were
before 12 October 2026 (see `src/pp_generation/README.md`).

## All-electron reference

`all_electron_reference.py` (PySCF 2.14): spin-free X2C, aug-cc-pwCVQZ on
carbon and aug-cc-pVQZ on hydrogen and fluorine, grid level 5, the geometries
of the `IS` inputs for both states. The unpolarized final state keeps
occupation 1 in the C 1s orbital, followed by overlap with the ground-state
orbital; the polarized one is a spin-unrestricted calculation with the maximum
overlap method. All calculations converged. With the triple-zeta sets the PBE
values are 297.046 / 307.315 (unpolarized) and 290.026 / 300.229 (polarized):
the binding energies change by 0.03 eV and the shift by 0.003 eV from triple to
quadruple zeta. For CA the functional is Slater exchange with Perdew–Zunger
correlation (`LDA_X,LDA_C_PZ`), the functional of `Correlation_Type: ca`.
