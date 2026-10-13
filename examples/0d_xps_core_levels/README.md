# Core-level binding energies of molecules: published benchmark sets

The cases of two papers on the pseudopotential ΔSCF method for core-level
binding energies and its refinement step PP-PBE(B3LYP), run with the solver
of this repository:

- `jctc2022_1s`: Q. Xu, D. Prendergast, J. Qian, *Real-Space Pseudopotential
  Method for the Calculation of 1s Core-Level Binding Energies*, J. Chem.
  Theory Comput. **18**, 5471 (2022). B, C, N and O 1s of 54 molecules
  (Tables 1, 2, 3, S1 and S2).
- `jctc2022_atoms`: the pseudo-atoms of Table 1 of the same paper. B, C, N and
  O and their ions with a 1s hole, without and with spin polarization.
- `jctc2022_protonated_water`: Figure 5(a) of the same paper. O 1s of the
  hydronium oxygen in the clusters H3O+(H2O)n, n = 1 to 20, relative to the
  water molecule.
- `jctc2024_third_row`: L. Liu, Q. Xu, L. dos Anjos Cunha, H. Xin,
  M. Head-Gordon, J. Qian, *Real-Space Pseudopotential Method for the
  Calculation of Third-Row Elements X-ray Photoelectron Spectroscopic
  Signatures*, J. Chem. Theory Comput. **20**, 6134 (2024). S and P 1s, 2s and
  2p of 21 molecules (Tables S1, S4 and S5, structures optimized with PBE).

The structures and the FHI98PP inputs of the pseudopotentials are those of the
Supporting Information of the papers; the values of the papers' tables are the
reference.

## Layout

```text
0d_xps_core_levels/
  xps_cases.py              shared steps: inputs, queue, log reader
  pseudopotentials/         the potentials of both sets and species.json
  jctc2022_1s/
    paper/structures/<B|C|N|O>/<nn>.xyz   molecules of Table S1, in its order
    paper/fhi98pp_inputs/*.ini            inputs of the potentials
    paper/table_s1.csv                    Table S1
    cases/<set>/<nn>/{IS,FS}/parsec.in    neutral molecule and 1s core hole
    reference_energies.csv                totals of every case
    setup.py  run.py  analyze.py
    RESULTS.md
  jctc2022_atoms/
    cases/<B|C|N|O>/<atom|ion>/<unpolarized|polarized>/parsec.in, occup.in
    reference_energies.csv
    setup.py  run.py  analyze.py
    RESULTS.md
  jctc2022_protonated_water/
    paper/structures/<nn>.xyz             H3O+(H2O)n, global minimum of the published file
    paper/figure_5a.csv                   the points of Figure 5(a), read from its image
    cases/<nn>/{IS,FS}/parsec.in          cluster (charge +1) and 1s hole on the hydronium oxygen
    reference_energies.csv
    setup.py  run.py  analyze.py
    RESULTS.md
  jctc2024_third_row/
    paper/structures/<formula>.xyz
    paper/fhi98pp_inputs/*.ini
    paper/tables.csv                      Tables S1, S4, S5
    cases/<formula>/{IS,FS_<S|P>-<1s|2s|2p>}/parsec.in
    reference_energies.csv
    setup.py  run.py  analyze.py
    RESULTS.md
```

`paper/` holds copies of files of the Supporting Information (on figshare
under CC BY-NC 4.0:
[10.1021/acs.jctc.2c00474.s002](https://doi.org/10.1021/acs.jctc.2c00474.s002),
[10.1021/acs.jctc.4c00535.s002](https://doi.org/10.1021/acs.jctc.4c00535.s002))
and the tables of its PDF files typed into CSV.

`pseudopotentials/` holds the potentials generated from those inputs
(Troullier–Martins, PBE, no core correction; FHI98PP, M. Fuchs and
M. Scheffler, Comput. Phys. Commun. **119**, 67 (1999)) in the format the
solver reads. `species.json` gives for each the local component, the number
of valence electrons and the all-electron and pseudo energies of the
generating atom, from which the `Atomic_Energy_Correction` of the inputs
follows. The local component is the highest channel of the input: p for B, C,
N, O and F, d for Si, P, S, Cl and Br, s for H. A potential with a core hole
(`C-1s`, `S-2p`, ...) is generated with one electron removed from that shell.

## Running

From `jctc2022_1s`, `jctc2022_atoms`, `jctc2022_protonated_water` or
`jctc2024_third_row`, with the Python environment of the repository:

```powershell
python run.py              # all cases; --sets C, --only C/02 C/15, ... select some
python analyze.py          # tables against the paper
python analyze.py --check  # exit code 1 if a run differs from the reference
```

`run.py` starts one ordinary run of the solver per case, with the Poisson
solves on the GPU (`--cpu-poisson` for the CPU), and skips cases whose log
`parsec.out` is finished. A case has `Energy_Refinement: b3lyp`, and its log
ends with the two totals that are read:

```text
PBE all-electron total (as above)           :         -80.96260980 [Ry]
B3LYP-refined all-electron total            :         -81.00444122 [Ry]
```

A binding energy is the difference of these totals between the final state
(`FS`, the atom with the hole carries the core-hole potential and the molecule
the charge +1) and the initial state (`IS`). `reference_energies.csv` holds
the totals, the parts of the B3LYP energy and the times of every case of the
runs reported below; without logs `analyze.py` works from it. `--check`
compares new logs with it (0.003 eV) and with the paper (tolerances in the
script).

`setup.py` rewrites the inputs, for example with another `--spacing`.

The cases of `jctc2022_atoms` have no refinement. Their occupations are fixed
(`Fermi_Temp` negative, `occup.in` beside each input, one section per spin
channel) and the energy read is the last `Total Energy` of the log.

The stored runs were made on GPUs with 80 GB. The molecules of the two sets and
the atoms also fit the 8 GB GPU of the tested laptop; of the protonated-water
clusters only the two smallest were run there. `run.py --largest 3e6` selects
the cases up to that many grid points.

## Settings

| | papers (ARES) | here |
|---|---|---|
| domain | sphere, Dirichlet boundary, multipole boundary values to l = 9 | sphere 5.0 Å beyond the outermost nucleus, same boundary treatment |
| grid | 0.07 to 0.12 Å (2022 runs kept in `../0_CH4_CF4`), 0.10 Å (2024 input) | 0.10 Å, finite differences of order 8 |
| spin | spin-polarized runs that end unpolarized | unpolarized |
| atomic reference energies | from the potential generation | the same (spin-unpolarized atoms) |
| refinement | `Lrefine=T` | `Energy_Refinement: b3lyp` |

With 0.10 Å the binding energies of CH4 and CF4 are 0.008 and 0.003 eV above
those of runs with 0.07 Å (`setup.py --spacing 0.07`). The totals of CH4
change by 0.0003 eV at most when the wall is moved from 4 Å to 8 Å beyond the
outermost nucleus. `States_Num` is the number of occupied states plus at least 8,
rounded up to a multiple of 16.

With `Spin_Polarization: true` the runs of CO and of CO with a C 1s hole end
without a magnetic moment and with the totals of the unpolarized runs, PBE and
refined, within 1e-8 Ry (both converged to 1e-7 Ry).

An atom of `jctc2022_atoms` is alone in a sphere of 7 Å, on the same grid,
with 16 states per spin channel and `Spin_Polarization: true` in the polarized
cases.

## Results

All 224 runs converged. Energies in eV; "PBE / refined" stands for PP-PBE /
PP-PBE(B3LYP). The complete tables are `RESULTS.md` in each folder.

2022 set:

| quantity | paper | here |
|---|---|---|
| Table 1, atomic energies of the generation, 16 spin-unpolarized values | | within 0.001 |
| Table 2, PP-PBE of the four reference molecules | 198.04, 296.71, 411.94, 544.93 | 198.042, 296.718, 411.945, 544.931 |
| Table S1, the same with its spin-polarized atomic energies, PBE | 191.60, 288.16, 401.86, 534.46 | 191.606, 288.160, 401.862, 534.463 |
| Table S1, the same, refined | 191.77, 288.15, 401.64, 534.09 | 191.800, 288.165, 401.642, 534.095 |
| Table S1, shifts of the other 50 molecules, PBE / refined | | mean absolute difference 0.007 / 0.010, largest 0.029 / 0.037 |
| Table 3, error against experiment, PBE (B, C, N, O, all) | 0.23, 0.20, 0.15, 0.11, 0.16 | 0.228, 0.205, 0.151, 0.108, 0.167 |
| Table 3, the same, refined | 0.11, 0.18, 0.14, 0.12, 0.14 | 0.109, 0.175, 0.142, 0.115, 0.138 |
| Table S2, N shifts with 0 to 5 times the exact exchange | 0.15, 0.14, 0.16, 0.21, 0.26, 0.31 | 0.155, 0.142, 0.164, 0.208, 0.252, 0.302 |

- The absolute values of Table S1 use atomic energies of spin-polarized atoms.
  The constant per element that this takes off a binding energy follows from
  Table 1 of the paper (B 6.436, C 8.558, N 10.083, O 10.468) and is applied
  for the third and fourth row; shifts do not depend on it.
- Table 1 prints -733.736 eV for the all-electron carbon ion with spin
  polarization. A spin-polarized all-electron calculation of the four ions with
  the 1s hole in the minority spin (radial, scalar-relativistic, PBE; the
  program is not part of this repository) gives the printed values of B, N and
  O within 0.001 eV and -733.376 eV for C, which would make the constant of C
  8.199. The comparison above uses the printed 8.558, with which the carbon
  values of Table S1 are reproduced.
- Table S1 prints the values of Cl3CCN and HNCO (N set) in each other's rows:
  here Cl3CCN 1.499 / 1.818 and HNCO 1.942 / 2.098, in the table 1.95 / 2.10
  and 1.49 / 1.82. They are compared exchanged.
- The errors of Table 3 count the reference molecule of each set.
- Table S2 is evaluated from the parts of the B3LYP energy that every case
  lists; it needs no further runs.

Atoms of Table 1 of the 2022 paper, total energy of the pseudo-atom without
and with spin polarization and their difference:

| | unpolarized, here | paper | polarized, here | paper | spin polarization, here | paper |
|---|---:|---:|---:|---:|---:|---:|
| B atom | -70.485 | -70.483 | -70.798 | -70.796 | -0.313 | -0.313 |
| B ion, 1s hole | -128.686 | -128.688 | -129.234 | -129.236 | -0.548 | -0.548 |
| C atom | -145.946 | -145.943 | -147.317 | -147.314 | -1.371 | -1.371 |
| C ion, 1s hole | -240.672 | -240.665 | -242.727 | -242.723 | -2.056 | -2.058 |
| N atom | -262.454 | -262.446 | -265.829 | -265.824 | -3.376 | -3.378 |
| N ion, 1s hole | -401.751 | -401.742 | -406.413 | -406.408 | -4.662 | -4.666 |
| O atom | -428.175 | -428.168 | -429.770 | -429.765 | -1.596 | -1.597 |
| O ion, 1s hole | -619.402 | -619.405 | -621.450 | -621.455 | -2.048 | -2.050 |

- An atom has its 2s orbital filled and its 2p electrons spread evenly over
  the three 2p orbitals; with spin polarization the 2p shell is filled by
  Hund's rule (1, 2, 3 and 2 Bohr magnetons for B, C, N and O). An ion is the
  atom in the potential with a 1s hole, with the charge +1.
- Largest difference: 0.009 in a total energy (N ion), 0.004 in an energy of
  spin polarization (N ion). The table of the paper comes from the radial
  program that generates the potentials. Here the atom is solved on the grid
  of 0.10 Å in a sphere of 7 Å, with the potentials as the solver reads them:
  resampled onto its radial mesh, in their separable form.

Protonated water (Figure 5(a) of the 2022 paper), binding energy of the 1s
level of the hydronium oxygen relative to the water molecule of the 2022 set:

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PBE, here | 9.929 | 7.890 | 6.187 | 5.630 | 5.546 | 5.025 | 5.278 | 4.803 | 4.565 | 4.211 |
| PBE, figure | 9.94 | 7.86 | 6.17 | 5.59 | 5.50 | 4.98 | 5.29 | 4.80 | 4.56 | 4.18 |
| refined, here | 9.983 | 7.975 | 6.295 | 5.743 | 5.663 | 5.142 | 5.414 | 4.935 | 4.694 | 4.342 |
| refined, figure | 9.99 | 7.97 | 6.31 | 5.71 | 5.62 | 5.10 | 5.42 | 4.93 | 4.69 | 4.32 |

| n | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PBE, here | 3.930 | 3.692 | 3.713 | 3.515 | 3.371 | 3.516 | 3.436 | 3.343 | 3.233 | 3.169 |
| PBE, figure | 3.92 | 3.62 | 3.68 | 3.49 | 3.35 | 3.51 | 3.40 | 3.32 | 3.18 | 3.15 |
| refined, here | 4.069 | 3.821 | 3.853 | 3.657 | 3.510 | 3.657 | 3.580 | 3.484 | 3.376 | 3.314 |
| refined, figure | 4.05 | 3.75 | 3.82 | 3.64 | 3.49 | 3.66 | 3.56 | 3.45 | 3.32 | 3.30 |

- The paper gives the figure only. Its points were read from the image
  (0.013 eV per pixel; the squares of the PBE curve lie partly under the
  circles of the refined one), which leaves an uncertainty of about 0.03.
  Mean absolute difference 0.027 / 0.024, largest 0.072 / 0.071 (n = 12).
- The structures are the first entry of each published file, the global
  minimum. The hole is on the oxygen with three hydrogens bound (Figure S2 of
  the paper); the initial state has the charge +1, the final state +2.

2024 set:

| quantity | paper | here |
|---|---|---|
| Table S1, S 1s of H2S and P 1s of P(CH3)3, PBE | 2492.11, 2162.71 | 2492.114, 2162.701 |
| Table S1, the same, refined | 2492.11, 2162.77 | 2492.126, 2162.775 |
| Table S4, S 2s of S8 and P 2s of P4, PBE | 224.3, 185.72 | 224.367, 185.717 |
| Table S4, the same, refined | 224.44, 185.71 | 224.435, 185.701 |
| Tables S1, S4, S5, shifts of 33 further entries, PBE / refined | | mean absolute difference 0.018 / 0.022, largest 0.053 / 0.059 |

- A potential with a 2p hole is spherical, so a 2p binding energy here is the
  average over the multiplet; the paper lists 2p3/2 values. Only shifts are
  compared for 2p.
- Two entries of the paper's tables are not in this set: S 2s of PSF3 and
  P 2p of PF3, which differ from the tables by 0.10 and 0.17.

## Time and memory

The stored energies are from first calculations (no cache, no restart) in
double precision on A100 GPUs with 80 GB (Perlmutter, one run per GPU, solver
defaults). One job of 85 minutes on 16 GPUs ran them all:

| set | runs | all runs | median run | shortest | longest |
|---|---:|---:|---:|---:|---:|
| 2022, 54 molecules | 108 | 2.6 GPU hours | 60 s | 19 s (CO) | 509 s (B(OC2H5)3) |
| 2022, atoms of Table 1 | 16 | 0.2 GPU hours | 47 s | 22 s (O ion) | 108 s (B atom, polarized) |
| 2022, protonated water | 40 | 14.1 GPU hours | 871 s | 59 s (H3O+(H2O)1) | 3426 s (H3O+(H2O)20) |
| 2024, 21 molecules | 60 | 1.7 GPU hours | 83 s | 28 s (H2S) | 464 s (P(CF3)3) |

| run | grid points | occupied / computed states | SCF steps | SCF [s] | pair solves | refinement [s] | whole run [s] | host memory [GiB] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CO, neutral | 723,160 | 5 / 16 | 24 | 12 | 15 | 2 | 19 | 2.1 |
| P(CF3)3, neutral | 1,854,640 | 40 / 48 | 41 | 65 | 820 | 288 | 363 | 5.4 |
| B(OC2H5)3, neutral | 3,246,592 | 30 / 48 | 61 | 145 | 496 | 336 | 509 | 9.5 |
| H3O+(H2O)1 | 1,360,280 | 8 / 16 | 38 | 41 | 36 | 9 | 59 | 3.3 |
| H3O+(H2O)5 | 2,278,264 | 24 / 32 | 41 | 79 | 300 | 140 | 231 | 5.6 |
| H3O+(H2O)10 | 3,260,032 | 44 / 64 | 40 | 100 | 990 | 640 | 765 | 10.2 |
| H3O+(H2O)15 | 3,939,528 | 64 / 80 | 44 | 172 | 2080 | 1751 | 1990 | 14.3 |
| H3O+(H2O)20 | 4,140,056 | 84 / 96 | 43 | 176 | 3655 | 3209 | 3426 | 16.0 |

The refinement is one Poisson solve per pair of orbitals that hold electrons:
0.13 s on 0.7 million points, 0.26 s on 1.4 million, 0.88 s on 4.1 million. It
takes 37 % of the time of the 2022 molecules, 44 % of the 2024 set and 88 % of
the water clusters. Where a run counts one orbital more than the occupied ones
(B(OC2H5)3: 496 pairs, H3O+(H2O)20: 3655), the first empty level has an
occupation just above 1e-12, the threshold of the exact exchange, at the Fermi
temperature of the inputs.

On a laptop with 32 logical cores, an RTX 5070 Laptop GPU with 8 GB and 32 GB
of memory, the runs of the next table, the second state of the two water
clusters and the 16 atoms give the totals of the A100 runs in all eight
printed decimals:

| run | grid points | occupied / computed states | SCF steps | SCF [s] | pair solves | refinement [s] | whole run [s] | host memory [GiB] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CO, neutral | 723,160 | 5 / 16 | 24 | 24 | 15 | 4 | 32 | 2.0 |
| CH3CN, neutral | 1,246,488 | 8 / 16 | 44 | 93 | 36 | 19 | 118 | 2.9 |
| P(CF3)3, neutral | 1,854,640 | 40 / 48 | 41 | 142 | 820 | 709 | 862 | 5.4 |
| B(OC2H5)3, neutral | 3,246,592 | 30 / 48 | 61 | 540 | 496 | 888 | 1439 | 12.3 |
| H3O+(H2O)1 | 1,360,280 | 8 / 16 | 38 | 59 | 36 | 20 | 82 | 3.1 |
| H3O+(H2O)2 | 1,990,968 | 12 / 32 | 44 | 119 | 78 | 74 | 198 | 5.1 |

There a pair solve takes 0.3 s on 0.7 million points and 1.8 s on 3.2 million,
and the GPU held 7.7 of its 8.0 GiB during B(OC2H5)3. An atom has 1,436,528
grid points and no refinement: the 16 atoms take 17 minutes, the oxygen atom
28 SCF steps and 53 s without spin polarization (3.6 GiB of host memory,
2.9 GiB on the GPU) and 25 steps and 86 s with it (3.8 and 3.3 GiB). The
complete 2022 and 2024 sets took 5.2 and 2.8 hours on this laptop (median run
95 s and 142 s), measured before the potentials were resampled again on
12 October 2026 (`src/pp_generation/README.md`).
