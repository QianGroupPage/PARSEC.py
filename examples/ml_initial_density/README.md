# ML initial-density small-molecule regression suite

This directory compares three ways to initialize the same PBE real-space DFT
calculation:

- `sad`: the current PARSEC-compatible superposition of pseudopotential atomic
  valence densities;
- `scdp`: the SCDP prediction saved by the old code framework;
- `charge3net`: the ChargE3Net prediction saved by the old code framework.

C60 is intentionally excluded. The suite contains the other 14 directories
from `tests/tests/small_molecules`, giving 42 calculations in total.

## What is held fixed

Within each molecule, all three inputs use the same atoms, spherical domain,
0.2 Angstrom grid spacing, eighth-order finite differences, PBE functional,
PBE H/C/N/O norm-conserving pseudopotentials, eigensolver, mixing, temperature,
and convergence criterion. Only `Initial_Density` changes.

The canonical inputs use CHEBDAV degree 15 for the first nonlinear
diagonalization and the existing GPU PARSEC `SUBSPACE` filter on every later
SCF iteration. This was selected only after a 42-case comparison against
CHEBFF; the former CHEBFF output is retained beside every current output.

The legacy model arrays are volume densities in `e/angstrom^3` on 70x70x70
(7 Angstrom radius) or 90x90x90 (9 Angstrom radius) Cartesian lattices. Their
portable files retain explicit origin and voxel-vector metadata. The old
positive-box coordinates are translated by the same bounding-box midpoint
used to recenter the nuclei, then linearly sampled at the active PARSEC grid
points. Each imported field has the correct valence-electron integral before
normalization; normalization remains enabled as the safe general default.

These are offline regression runs: the SCDP and ChargE3Net repositories are
not invoked. A named provider plus `ML_Density_File` preserves the density's
provenance while loading a previously generated prediction. Remove the
`ML_Density_File` line from an input to exercise direct model inference
instead (after configuring its model environment and checkpoint).

## Layout

```text
ml_initial_density/
  pseudopotentials/            shared PBE H/C/N/O POTRE files
  small_molecules/
    H2O/                        one directory per molecule
      densities/
        scdp.npz
        charge3net.npz
      sad/parsec.in
      sad/parsec.out
      sad/parsec_chebff.out
      scdp/parsec.in
      scdp/parsec.out
      charge3net/parsec.in
      charge3net/parsec.out
  case_manifest.csv             source-grid and legacy-file provenance
  results_summary.csv           current CHEBDAV convergence/energy/timings
  results_summary_chebff.csv    preserved CHEBFF baseline summary
  eigensolver_comparison.csv    controlled CHEBFF/CHEBDAV comparison
  chebdav_degree_ch4.csv        degree-15/degree-20 CH4 check
  prepare_small_molecule_cases.py
  run_small_molecule_suite.py
  compare_eigensolvers.py
  plot_chebff_chebdav.py       Figure-5-style convergence/timing plot
  figures/
    chebff_chebdav_iterations_timing.png
    chebff_chebdav_iterations_timing.pdf
    chebff_chebdav_figure_data.csv
```

## Rebuild and run

From the repository root, rebuild the cases from the local legacy result tree:

```powershell
.\.venv312\Scripts\python.exe `
  examples\ml_initial_density\prepare_small_molecule_cases.py
```

Validate every generated input without starting an SCF calculation:

```powershell
.\.venv312\Scripts\python.exe `
  examples\ml_initial_density\run_small_molecule_suite.py --dry-run
```

Run all 42 calculations with the fastest automatically selected backend:

```powershell
.\.venv312\Scripts\python.exe `
  examples\ml_initial_density\run_small_molecule_suite.py
```

Run only one molecule or one method:

```powershell
.\.venv312\Scripts\python.exe `
  examples\ml_initial_density\run_small_molecule_suite.py `
  --molecule H2O --method scdp
```

The runner writes `parsec.out` in each method directory and regenerates
`results_summary.csv`. To run one input without the suite helper:

```powershell
.\.venv312\Scripts\python.exe src\parsec_python\main.py `
  examples\ml_initial_density\small_molecules\H2O\scdp\parsec.in `
  --pp-dir examples\ml_initial_density\pseudopotentials --no-archive
```

Regenerate the CHEBFF/CHEBDAV comparison figure and its paired audit table:

```powershell
.\.venv312\Scripts\python.exe `
  examples\ml_initial_density\plot_chebff_chebdav.py
```

The timing panels report `solver_wall_seconds`: setup plus SCF time measured
inside the calculation driver. They do not include Python interpreter startup
in the outer suite launcher. Dashed horizontal lines show the mean for each
initial-density method, matching the convention used in Figure 5 of the paper.

## Results from the current code

All 42 calculations converged. Iteration counts are:

| Molecule | SAD | SCDP | ChargE3Net | Final-energy spread (meV) |
|---|---:|---:|---:|---:|
| C10H8 | 13 | 10 | 8 | 0.044 |
| C2H2 | 13 | 8 | 10 | 0.002 |
| C2H4 | 13 | 9 | 8 | 0.003 |
| C2H6 | 16 | 15 | 13 | 0.001 |
| C3H8O | 13 | 14 | 18 | 0.007 |
| C6H6 | 14 | 10 | 11 | 0.002 |
| CH3CH2CH2OH | 19 | 14 | 18 | <0.001 |
| CH3CH2OCH3 | 12 | 13 | 19 | 0.012 |
| CH3CH2OH | 15 | 12 | 17 | <0.001 |
| CH3CHOHCH3 | 19 | 19 | 20 | <0.001 |
| CH3CN | 16 | 12 | 12 | 0.001 |
| CH4 | 18 | 12 | 9 | 0.002 |
| CO2 | 10 | 12 | 11 | <0.001 |
| H2O | 13 | 12 | 13 | <0.001 |

The mean CHEBDAV iteration counts are 14.57 (SAD), 12.29 (SCDP), and
13.36 (ChargE3Net), compared with CHEBFF's 21.14, 22.50, and 22.21. The
controlled audit found CHEBDAV faster in 40/42 calculations and reduced its
summed solver time from 439.85 to 328.56 seconds (1.34x speedup). The two
exceptions were SAD cases whose reduced iteration count did not repay the
more expensive first eigensolve. Degree 20 did not improve the CH4 result, so
the source-valid minimum degree 15 is used.

The maximum final-energy spread among the three starts is now
`3.23e-6 Ry = 0.044 meV`, well below the `2e-4 Ry` SCF criterion. The largest
CHEBDAV-versus-CHEBFF final-energy difference was `4.423e-5 Ry`, also below
that criterion. Detailed results are in `results_summary.csv` and
`eigensolver_comparison.csv`; each `parsec_chebff.out` is the preserved
baseline corresponding to the current `parsec.out`.

## Why CHEBDAV reveals the ML benefit

Old-architecture `diagmeth=3` used `first_filt` for the first SCF iteration
and `chefsi1` (Chebyshev-filtered subspace iteration) thereafter. Current
CHEBFF is the closest PARSEC implementation of that policy, and both current
first-solver choices already use the same `SUBSPACE` implementation after the
first iteration. There is therefore no missing “chebsup” later-SCF path.

CHEBFF intentionally performs a fixed amount of approximate first-filter
work without accepting individual Ritz pairs by residual. CHEBDAV spends more
time building and residual-testing a better first eigensubspace. For these
ML-density tests, that additional first-step work prevents eigensolver error
from obscuring differences in the starting Kohn--Sham potential; fewer later
SCF/subspace iterations more than recover its cost in nearly every case.

See [the complete interface guide](../../src/parsec_python/MLDensity/README.md)
for direct-model environments, schemas, units, cache behavior, normalization,
and model-domain limitations.
