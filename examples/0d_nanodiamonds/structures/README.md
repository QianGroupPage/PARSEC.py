# How the nanodiamond structures were made

This folder holds the geometries behind the inputs of `../fortran_comparison/`
and `../gpu_benchmark/`, the scripts that made them, and their summaries.
Coordinates are in Angstrom, centred at the origin, carbons first.

| Case | Formula | Geometry | File in this folder | Used by |
|---|---|---|---|---|
| k00 | C1H4 | GFN2-xTB optimized | `k00_C1H4/optimized/structure_gfn2.xyz` | `fortran_comparison/` |
| k01 | C5H12 | GFN2-xTB optimized | `k01_C5H12/optimized/structure_gfn2.xyz` | `fortran_comparison/` |
| k02 | C35H36 | GFN2-xTB optimized | `k02_C35H36/optimized/structure_gfn2.xyz` | `fortran_comparison/` |
| k03 | C87H76 | GFN2-xTB optimized | `k03_C87H76/optimized/structure_gfn2.xyz` | `fortran_comparison/` |
| k04 | C185H124 | GFN2-xTB optimized | `k04_C185H124/optimized/structure_gfn2.xyz` | `fortran_comparison/` |
| k05 | C459H204 | GFN2-xTB optimized, symmetrized | `k05_C459H204/optimized_symmetrized/structure_gfn2.xyz` | `fortran_comparison/` |
| k06 | C795H300 | GFN2-xTB optimized, symmetrized | `k06_C795H300/optimized_symmetrized/structure_gfn2.xyz` | `fortran_comparison/`, `gpu_benchmark/` |
| k07 | C1213H412 | GFN2-xTB optimized, symmetrized | `k07_C1213H412/optimized_symmetrized/structure_gfn2.xyz` | `gpu_benchmark/` |
| k08 to k14 | C1653H508 to C9449H1572 | ideal, not relaxed | none; regenerate as below | `gpu_benchmark/` |

The coordinates of every published input (`parsec.in`, and `parsec_fortran.in`
of the Fortran comparison) equal these geometries, for k08 to k14 the
regenerated ones, to the 10 decimals they print, atom by atom in the same
order.

## Construction rule

The clusters are cut with ASE from a cubic diamond lattice with
`a = 3.567` Angstrom, so `d_CC = sqrt(3) a / 4 = 1.5445563076` Angstrom.

1. Seed: every carbon site within `|r| <= k d_CC` of a central carbon.
2. Surface completion: a removed lattice site bonded to two or more retained
   carbons is added as carbon; this repeats until no such site is left.
   Capping both bonds to such a site with H would put two H atoms 0.74 Angstrom
   apart.
3. Every remaining dangling bond is capped by H along its tetrahedral
   direction, `d_CH = 1.09` Angstrom.
4. Carbons come first, ordered by distance from the centre, then by x, y and
   z. The hydrogens follow in the order of the carbons they cap.

| k | Seed cutoff (Angstrom) | Seed C | Added surface C | Cluster |
|---:|---:|---:|---:|---|
| 0 | 0.000000 | 1 | 0 | C1H4 |
| 1 | 1.544556 | 5 | 0 | C5H12 |
| 2 | 3.089113 | 29 | 6 | C35H36 |
| 3 | 4.633669 | 87 | 0 | C87H76 |
| 4 | 6.178225 | 167 | 18 | C185H124 |
| 5 | 7.722782 | 357 | 102 | C459H204 |
| 6 | 9.267338 | 633 | 162 | C795H300 |
| 7 | 10.811894 | 943 | 270 | C1213H412 |
| 8 | 12.356450 | 1371 | 282 | C1653H508 |
| 9 | 13.901007 | 1963 | 492 | C2455H636 |
| 10 | 15.445563 | 2809 | 660 | C3469H804 |
| 11 | 16.990119 | 3689 | 912 | C4601H988 |
| 12 | 18.534676 | 4687 | 972 | C5659H1132 |
| 13 | 20.079232 | 6047 | 1020 | C7067H1308 |
| 14 | 21.623788 | 7565 | 1884 | C9449H1572 |

## Optimization (k00 to k07)

The ideal clusters k00 to k07 were optimized with
[xTB](https://github.com/grimme-lab/xtb): neutral, closed shell, 28 OpenMP
threads. Clusters with fewer than 50 atoms (k00, k01) were optimized directly
with GFN2-xTB. Larger ones were first preoptimized with GFN-FF, and the
GFN-FF geometry was then optimized with GFN2-xTB:

```text
xtb input.xyz --gfnff  --chrg 0 --uhf 0 --opt normal   # energy 5e-6 Eh, gradient 1e-3 Eh/bohr
xtb input.xyz --gfn 2  --chrg 0 --uhf 0 --opt tight    # energy 1e-6 Eh, gradient 8e-4 Eh/bohr
```

k00 to k04 used xTB 6.7.1pre (commit 5071a88) on Windows through
`optimize_nanodiamonds.py`. k05 to k07 used a Linux build of xTB 6.7.1
(commit 5fb5a1c) under WSL through `optimize_nanodiamonds_wsl.py`, which runs
xTB with an unlimited stack; the Windows build can run out of stack for large
systems. Both scripts then rotate the xTB geometry onto the ideal one
(mass-weighted fit), centre it, and check that every C-C and C-H bond of the
ideal cluster is kept and no bond is formed (`optimized/geometry_audit.json`).
All eight optimizations converged with the topology unchanged.

| Case | Path | GFN2-xTB energy (Eh) | Final gradient (Eh/bohr) | RMSD from ideal (Angstrom) | Largest displacement (Angstrom) |
|---|---|---:|---:|---:|---:|
| k00 | GFN2-xTB | -4.175218520024 | 3.03e-6 | 0.00393 | 0.0078 |
| k01 | GFN2-xTB | -16.835616103877 | 3.11e-5 | 0.01255 | 0.0145 |
| k02 | GFN-FF, GFN2-xTB | -93.732333667486 | 1.52e-4 | 0.02413 | 0.0264 |
| k03 | GFN-FF, GFN2-xTB | -226.140604772819 | 1.54e-4 | 0.02180 | 0.0587 |
| k04 | GFN-FF, GFN2-xTB | -462.182317836196 | 3.36e-4 | 0.02920 | 0.0694 |
| k05 | GFN-FF, GFN2-xTB | -1095.379883636084 | 1.37e-4 | 0.05137 | 0.0679 |
| k06 | GFN-FF, GFN2-xTB | -1870.291790304032 | 9.60e-5 | 0.05341 | 0.0946 |
| k07 | GFN-FF, GFN2-xTB | -2830.667612423038 | 2.66e-5 | 0.05414 | 0.0940 |

## Symmetrization (k05 to k07)

The ideal clusters have the point group Td: 24 signed permutations of the x, y
and z axes map each of them onto itself. The optimized k05 to k07 deviate from
this symmetry by up to 5e-5 Angstrom, a residual of the optimization.
`symmetrize_optimized_nanodiamonds.py` averages the optimized coordinates over
these 24 operations, with the atom permutations of the ideal cluster, and
writes `optimized_symmetrized/`. The raw xTB geometries in `optimized/` stay
as they were.

| Case | Symmetry violation before (Angstrom) | Largest shift (Angstrom) | Violation after (Angstrom) |
|---|---:|---:|---:|
| k05 | 4.9e-5 | 2.5e-5 | 7e-15 |
| k06 | 3.8e-5 | 1.9e-5 | 7e-15 |
| k07 | 2.3e-5 | 1.2e-5 | 7e-15 |

## Files

- `generate_nanodiamonds.py`: the ideal clusters (needs ASE).
- `optimize_nanodiamonds.py`, `optimize_nanodiamonds_wsl.py`: the xTB
  optimization, with an xTB executable or with a Linux xTB under WSL.
- `symmetrize_optimized_nanodiamonds.py`: the symmetrization (needs SciPy).
- `cluster_manifest.csv`, `generation_summary.json`: the ideal series k00 to
  k07 and the construction settings.
- `optimization_manifest.csv`: the xTB results of k00 to k07.
- Per case: `structure.xyz` (ideal) with `geometry_audit.json` (bond lengths,
  valences, shortest H-H distance); `optimized/structure_gfn2.xyz` with
  `optimization_summary.json` (xTB version, commands, energies, gradients, wall
  times) and `geometry_audit.json`; for k05 to k07
  `optimized_symmetrized/structure_gfn2.xyz` with
  `symmetrization_summary.json`.

The scripts also write extended XYZ files, POSCAR files and a PBE input
template `parsec.in` for each case. These are not kept here; the inputs of this
example are those of `../fortran_comparison/` and `../gpu_benchmark/`.

## Reproduce

Python with NumPy, SciPy and ASE (checked with ASE 3.29.0, NumPy 2.4.4,
SciPy 1.17.1). Run from this folder and write into an empty folder `DIR`
outside the repository:

```bash
python generate_nanodiamonds.py --output-root DIR [options]
```

| Options | Reproduces |
|---|---|
| none (defaults `--maximum-carbon-atoms 200 --maximum-multiplier 8`) | k00 to k04: `structure.xyz` and `geometry_audit.json`, byte for byte |
| `--maximum-carbon-atoms 1001 --maximum-multiplier 12 --include-first-at-or-above-limit` | k05 to k07: the same files, and `cluster_manifest.csv` and `generation_summary.json`, byte for byte |
| `--maximum-carbon-atoms 5659 --maximum-multiplier 12 --include-first-at-or-above-limit` | k08 to k12: the coordinates of `../gpu_benchmark/kNN_<formula>/parsec.in` |
| `--maximum-carbon-atoms 9449 --maximum-multiplier 40 --include-first-at-or-above-limit` | k13 and k14: the coordinates of `../gpu_benchmark/kNN_<formula>/parsec.in` |

`--maximum-multiplier` sets the size of the reference lattice. Another value
gives the same atoms at the same positions, but the round-off of the
coordinates (about 1e-15 Angstrom) can change the order of atoms at the same
distance from the centre. The values above reproduce the order of the
published files.

The optimization needs xTB. It reads the generator's files in `DIR`, writes
`kNN_<formula>/optimized/` and `optimization_manifest.csv` there, and refuses
to overwrite an existing `optimized/`. Without `--case`,
`optimize_nanodiamonds.py` takes every case in `DIR`:

```bash
python optimize_nanodiamonds.py --root DIR --xtb /path/to/xtb --threads 28 [--case kNN_<formula> ...]
python optimize_nanodiamonds_wsl.py --root DIR --xtb /linux/path/to/xtb --threads 28 --case kNN_<formula> [...]
```

The symmetrization of the optimized geometries of this folder, in a copy:

```bash
for k in k05_C459H204 k06_C795H300 k07_C1213H412; do
    mkdir -p DIR/$k/optimized
    cp $k/structure.xyz $k/geometry_audit.json DIR/$k/
    cp $k/optimized/structure_gfn2.xyz DIR/$k/optimized/
done
python symmetrize_optimized_nanodiamonds.py --root DIR \
    --case k05_C459H204 --case k06_C795H300 --case k07_C1213H412
```

This gives `optimized_symmetrized/structure_gfn2.xyz` and
`symmetrization_summary.json` identical to the files here.
