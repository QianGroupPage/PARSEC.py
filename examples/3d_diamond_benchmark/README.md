# Bulk diamond on many GPUs

Supercells of bulk diamond from 8 to 64,000 atoms, at the Gamma point and with Gamma-centred Monkhorst-Pack meshes,
computed on 16 NVIDIA A100 80 GB GPUs (four Perlmutter nodes). The series measures how large a periodic cell this
code can treat: the orbitals of one cell are shared by all ranks (one rank per GPU, any number of nodes), or, where
the cell has a point group about its origin, divided into its symmetry sectors; with k-points the ranks form pools
of k-points.

## Settings

`make_diamond.py n1 n2 n3 out/parsec.in [--kmesh K1 K2 K3] [--set "Keyword: value"]` writes one input:

- **Cell:** n1 x n2 x n3 conventional cubic cells of diamond (a = 3.567 Angstrom, 8 atoms each).
- **Functional and potential:** PBE with the PBE carbon pseudopotential of the library
  (`src/pp_generation/NCPPs/PBE/C/PARSEC`, copied here as `C_POTRE.DAT`; Troullier-Martins, local d, cubic
  spline).
- **Grid:** 18 points per cubic edge (0.198 Angstrom), finite differences of order 8; the grid of a supercell
  has 18 n points per edge.
- **Solver and mixing:** the blocks of the nanodiamond series (`examples/0d_nanodiamonds/gpu_benchmark`): CHEBFF,
  first solve 3 x degree 30, then degree 15 (+3), buffer 6, Anderson mixing 0.3 with memory 4 and the safeguard;
  occupied + 6 states; Fermi-Dirac at 500 K; SCF converged to 2e-4 Ry.
- **k-points (`--kmesh`):** a Gamma-centred mesh reduced to its irreducible wedge on the default routines
  (`Use_Fortran_Routines: .false.` with a zero `Monkhorst_Pack_Shift`). At 18 points per edge the 24 symmorphic
  operations (Td about an atom) map the grid onto itself; with time reversal they give the same wedge as the full
  group: 4, 4 and 10 points for 2x2x2, 3x3x3 and 4x4x4 on a cubic supercell. The density is symmetrized with them.

The defaults that matter for bulk cells (since November 2026): the later-SCF filter takes PARSEC 1.5's smart lower
bound, and the local potential has PARSEC 1.5's zero average (with the energy's `E_alpha`) and the
exchange-correlation potential its zero average (`Zero_Average_Vxc`). Total energies are those of the earlier
convention; eigenvalues and the Fermi level are now those of Fortran PARSEC 1.5.

## Running

```bash
# one GPU, the accelerated CLI
python examples/3d_diamond_benchmark/run_case.py 4 4 4 out/d4 --gpus 1
# Gamma on four nodes of four GPUs, one rank per GPU (inside a Slurm allocation)
python examples/3d_diamond_benchmark/make_diamond.py 8 8 8 d8/parsec.in && cp examples/3d_diamond_benchmark/C_POTRE.DAT d8/
scripts/run_multi_gpu_scf.sh --nodes 4 --rank-per-gpu d8/parsec.in d8_out
# k-points on 16 ranks (k-point pools; --kpoint-pools P sets their number)
srun -N 4 -n 16 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none bash -c \
  'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID exec python examples/3d_diamond_benchmark/run_mpi_kpoints.py 6 6 6 k6 --kmesh 2 2 2'
```

`run_mpi_case.py` runs the Gamma case on every rank of an `srun` launch with the settings of
`scripts/run_multi_gpu_scf.sh` and writes `summary.json` (`summarize_timing.py`); `run_mpi_kpoints.py` does the same
for `parsec_python.main --mpi` and samples the GPU and host memory of every rank. `compare_fortran.py` compares
eigenvalues, Fermi level and energy with the Fortran PARSEC 1.5 binary on small cells; `profile_setup.py` times the
stages of the periodic preparation.

## Results on 16 A100 (4 nodes), November 2026

Every run is a first calculation in a new process (no SCF or symmetry cache), with the software on node-local
storage and one warm-up run per allocation; program time is the code's own wall time from start to the end of the
SCF. GPU memory is the largest nvidia-smi reading of any GPU (it includes CuPy's pool cache), host memory the largest
resident memory of any rank. Code: the round's integration head (244c4ccd for most rows, d1391c63 for n = 9 and
k3 n = 7; the two differ in no code path these runs take).

### Gamma point

| n | atoms | electrons | grid points | states | SCF iterations | program (s) | SCF (s) | setup (s) | later step (s) | GPU (GB) | host (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 64 | 256 | 46,656 | 134 | 8 | 1.7 | 0.9 | 0.6 | 0.02 | 0.6 | 1.0 |
| 3 | 216 | 864 | 157,464 | 438 | 10 | 4.3 | 1.6 | 2.6 | 0.06 | 0.9 | 1.2 |
| 4 | 512 | 2,048 | 373,248 | 1,030 | 10 | 6.0 | 3.0 | 3.0 | 0.18 | 1.9 | 1.5 |
| 5 | 1,000 | 4,000 | 729,000 | 2,006 | 10 | 12.3 | 7.0 | 5.1 | 0.52 | 5.2 | 2.0 |
| 6 | 1,728 | 6,912 | 1,259,712 | 3,462 | 11 | 29.3 | 20.8 | 8.5 | 1.44 | 9.7 | 2.6 |
| 7 | 2,744 | 10,976 | 2,000,376 | 5,494 | 11 | 63.6 | 49.8 | 13.6 | 3.68 | 16.6 | 3.7 |
| 8 | 4,096 | 16,384 | 2,985,984 | 8,198 | 11 | 140.4 | 118.9 | 21.3 | 8.89 | 35.3 | 5.4 |
| 8x8x9 | 4,608 | 18,432 | 3,359,232 | 9,222 | 11 | 181.7 | 158.0 | 23.5 | 11.82 | 43.7 | 5.7 |
| 9 | 5,832 | 23,328 | 4,251,528 | 11,670 | 11 | 308.0 | 273.6 | 34.4 | 20.44 | 69.8 | 6.5 |

9 x 9 x 10 (6,480 atoms, 12,966 states) does not fit: the root GPU, which also holds the m x m matrices of the
Rayleigh-Ritz step, runs out at 75.6 GB plus a 5.4 GB request; the run stops with `FULL_SCF_FAILED` after 69 s.
The other GPUs of n = 9 held 51-52 GiB of arrays (the root 61 GiB).
(These rows are the one sector of the identity; the memory round of 2026-11-15 brought n = 10, 8,000 atoms, to
651.5 s and 68.1 GB.)

### Gamma point in the sectors of the point group (2026-11-16)

With an atom at the origin and 18 points per cubic edge, the cell keeps Td about that atom, and the orbitals take
the weighted sectors of D2d (`PARSEC_PERIODIC_SYMMETRY=auto`, the default; `src/parsec_python/acceleration/README.md`,
"Point groups of periodic cells"): 1/16 of the grid times the states of the one sector of the identity. Four nodes,
one process per node on its four GPUs (`scripts/run_multi_gpu_scf.sh --nodes 4 parsec.in out`, the launch of the
nanodiamond series), first runs; GPU memory as reported by the runner for the fullest GPU. The times are those of
the code with the periodic Hartree solve by the FFT of the stencil symbol and the local ionic potential on the GPU
(below); "before" is the program time of the same runs on code 51b2e6b0, which had neither:

| n | atoms | electrons | grid points | states | SCF iterations | program (s) | SCF (s) | setup (s) | later step (s) | GPU (GiB) | before (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1,728 | 6,912 | 1,259,712 | 3,462 | 9 | 5.7 | 1.9 | 3.7 | 0.12 | 1.0 | 18.4 |
| 8 | 4,096 | 16,384 | 2,985,984 | 8,198 | 9 | 12.4 | 5.4 | 6.8 | 0.37 | 2.5 | 29.7 |
| 10 | 8,000 | 32,000 | 5,832,000 | 16,006 | 9 | 36.2 | 19.5 | 16.6 | 1.25 | 6.5 | 71.2 |
| 12 | 13,824 | 55,296 | 10,077,696 | 27,654 | 10 | 106.8 | 78.1 | 28.5 | 5.40 | 14.8 | 173.8 |
| 14 | 21,952 | 87,808 | 16,003,008 | 43,910 | 10 | 193.0 | 152.9 | 39.9 | 10.85 | 32.1 | 347.8 |
| 15 | 27,000 | 108,000 | 19,683,000 | 54,006 | 9 | 280.5 | 229.9 | 50.5 | 17.87 | 46.4 | 479.4 |
| 16 | 32,768 | 131,072 | 23,887,872 | 65,542 | 9 | 424.8 | 361.7 | 62.9 | 28.35 | 66.3 | 690.7 |

n = 16 was the largest cube that fits with the point group alone (n = 17 would need about 1.4 times the memory;
with the translations below n = 20 fits). The energy per atom is
-11.4039349 Ry from n = 10 on; the energies of the two codes agree within 3e-10 Ry. SCF to 1e-7 Ry, the sectors
give the energy of the one sector of the identity within 4e-14 Ry (n = 1), 5e-13 Ry (n = 2) and 9e-13 Ry (n = 3).

Of the 690.7 s of n = 16 on code 51b2e6b0, the setup took 239.6 s (153 s the local ionic potential of the root,
on the host; 30 s the finite differences) and the periodic Hartree solves 84 s (CG on the GPU, about 9 s a step).
Three changes, each exact or with a switch that restores the former bits (`src/parsec_python/acceleration/README.md`):

- The Hartree solve divides by the eigenvalues of the wrapped finite-difference Laplacian, the FFT of its stencil
  (`PARSEC_PERIODIC_HARTREE_METHOD=fft`, the default; `cg` the former solve): 0.26 s a step at n = 16, 2.5 s for
  the whole SCF, and the GPU of the solve no longer holds the matrix.
- The local ionic potential (the structure-factor series and the short-ranged terms of every image) is summed on
  the GPU (`PARSEC_PERIODIC_SETUP_DEVICE=auto`, the default; `off` the host): 1.0 s at n = 16.
- The finite-difference matrix of the wrapped box is written directly in its final order (the same arrays): 5.3 s.

The setup of n = 16 is now 62.9 s, of which 23 s are the projectors and 11 s the initial density; of its SCF,
355 s are eigensolves (133 s the first). The host memory of the root fell from 34.1 to 14.6 GiB. With one rank
per GPU (`--rank-per-gpu`, sector groups of four ranks) n = 10 took 33.8 s and 10.2 GiB.

### Gamma point with the translations by half the cell (2026-11-16)

A supercell of n cubes repeats a smaller cell, and its grid of 18 n points per edge maps onto itself under the
translations by half the cell: for an even n all seven (the cell is 2 x 2 x 2 cells of n/2 cubes), for an odd n the
three face-centring ones `(0, a/2, a/2) n` of the fcc lattice. With Td about the origin they give a group of 192 or 96
operations, whose real characters are the time-reversal invariant k-points of the smaller cell, and the orbitals take
10 or 5 weighted sectors of a subgroup of 32 or 16 (`PARSEC_PERIODIC_TRANSLATIONS=auto`, the default;
`src/parsec_python/acceleration/README.md`, "Translations by half the cell"): about 1/102 of the rows times states
of the one sector of the identity, against 1/16 with Td alone. The projectors of a cell are also built for all atoms
at once (`acceleration/V_ion/periodic_projectors.py`; the same arrays, 23.0 s -> 1.5 s at n = 16). The same launch
and settings as above, first runs; "before" is the program time of the table above (code d50bf3d0):

| n | atoms | electrons | grid points | states | SCF iterations | program (s) | SCF (s) | setup (s) | later step (s) | GPU (GiB) | before (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1,728 | 6,912 | 1,259,712 | 3,462 | 7 | 4.4 | 1.3 | 3.0 | 0.08 | 0.7 | 5.7 |
| 8 | 4,096 | 16,384 | 2,985,984 | 8,198 | 8 | 9.2 | 4.0 | 5.0 | 0.30 | 1.2 | 12.4 |
| 10 | 8,000 | 32,000 | 5,832,000 | 16,006 | 7 | 19.8 | 10.6 | 9.2 | 0.90 | 2.4 | 36.2 |
| 12 | 13,824 | 55,296 | 10,077,696 | 27,654 | 7 | 43.7 | 27.2 | 16.4 | 2.29 | 4.4 | 106.8 |
| 14 | 21,952 | 87,808 | 16,003,008 | 43,910 | 7 | 92.5 | 64.6 | 27.7 | 5.33 | 8.0 | 193.0 |
| 15 | 27,000 | 108,000 | 19,683,000 | 54,006 | 8 | 170.7 | 135.6 | 35.0 | 10.73 | 23.9 | 280.5 |
| 16 | 32,768 | 131,072 | 23,887,872 | 65,542 | 7 | 187.8 | 142.9 | 44.7 | 11.81 | 16.1 | 424.8 |
| 18 | 46,656 | 186,624 | 34,012,224 | 93,318 | 7 | 362.1 | 292.3 | 69.7 | 24.25 | 30.1 | - |
| 20 | 64,000 | 256,000 | 46,656,000 | 128,006 | 7 | 668.1 | 561.7 | 106.3 | 46.66 | 53.8 | - |

n = 20 is the largest cube that fits (n = 22 would need about 1.8 times the memory; an odd n has the three
translations only, so n = 15 takes more time and memory than n = 16). The energy per atom is -11.4039349 Ry from
n = 10 on, as above. SCF to 1e-7 Ry, the translations give the energy of the point group alone within 1e-14 Ry
(n = 1, 96 operations), 2e-13 Ry (n = 2), 2e-12 Ry (n = 3) and 2e-12 Ry (n = 4); at the 2e-4 Ry of the series the
two SCFs take different paths (7 to 8 steps instead of 9 to 10) and their energies differ by up to 5e-4 Ry for
32,768 atoms (1.5e-8 Ry per atom). The ten sectors of n = 16 hold 2,054 states each, on 16 GPUs six of them on two
GPUs and four on one; its first eigensolve takes 71 s and a later SCF step 12 s.

### Gamma point: projectors of one atom per orbit, binned densities (2026-11-16)

The sectors of a group take the projectors of one atom per orbit of the group, scaled by the square root of the
orbit size (`PARSEC_SECTOR_PROJECTOR_ORBITS=auto`: the same sector operators to round-off, a 32nd of the columns
for the subgroup of 32 of these cells), and the superposed atomic density of the first SCF step is summed on the GPU
from bins of the atom images (`PARSEC_PERIODIC_DENSITY_BINS=auto`: the same terms in another order). The same launch
and settings as above, first runs on one allocation from small to large; "before" is the program time of the table
above (code 6275754b):

| n | atoms | SCF iterations | program (s) | SCF (s) | setup (s) | later step (s) | GPU (GiB) | before (s) |
|---|---|---|---|---|---|---|---|---|
| 6 | 1,728 | 7 | 22.6 (*) | 14.8 | 7.8 | 0.06 | 0.8 | 4.4 |
| 8 | 4,096 | 8 | 7.0 | 2.1 | 4.8 | 0.16 | 1.1 | 9.2 |
| 10 | 8,000 | 7 | 12.2 | 3.7 | 8.4 | 0.34 | 2.1 | 19.8 |
| 12 | 13,824 | 7 | 22.5 | 7.5 | 14.8 | 0.69 | 3.9 | 43.7 |
| 14 | 21,952 | 7 | 37.0 | 14.5 | 22.4 | 1.32 | 7.5 | 92.5 |
| 15 | 27,000 | 8 | 77.8 | 50.2 | 27.4 | 4.22 | 23.2 | 170.7 |
| 16 | 32,768 | 7 | 64.3 | 30.2 | 33.9 | 2.84 | 15.2 | 187.8 |
| 18 | 46,656 | 7 | 112.9 | 63.4 | 49.3 | 6.03 | 29.0 | 362.1 |
| 20 | 64,000 | 7 | 195.5 | 125.2 | 70.2 | 11.90 | 52.2 | 668.1 |

(*) the first item of the allocation: its first eigensolve (13.4 s) compiles the GPU kernels of the code; the same
cell took 4.1 s after another one. The energies are those of the table above within 2e-9 Ry; SCF to 1e-7 Ry the
projectors of one atom per orbit give those of every atom within 1e-14 (8 and 64 atoms), 5e-11 (216) and 1.2e-10 Ry
(512). At n = 16 the SCF is 30.2 s instead of 142.9 s (the first eigensolve 11.7 s instead of 70.7, a later step
2.8 s instead of 11.8): the projectors of every atom in each of the ten sectors had cost more than the stencil. The
setup is now the larger part (n = 20: 70.2 s of 195.5, of it 8.2 s the density instead of 36.4).

### k-points (irreducible wedge of a Gamma-centred mesh)

Program time in seconds, SCF iterations in brackets; P is the number of k-point pools the code chose.

| n | atoms | 2x2x2 (4 k) | 3x3x3 (4 k) | 4x4x4 (10 k) |
|---|---|---|---|---|
| 2 | 64 | 2.3 (8), P 4 | 2.7 (8), P 4 | 4.4 (8), P 2 |
| 3 | 216 | 8.3 (7), P 4 | 4.5 (7), P 4 | 8.6 (7), P 2 |
| 4 | 512 | 15.7 (8), P 4 | 15.8 (8), P 4 | 38.4 (8), P 2 |
| 5 | 1,000 | 59.4 (8), P 4 | 52.5 (7), P 4 | 136.4 (7), P 2 |
| 6 | 1,728 | 199.0 (8), P 4 | 199.1 (8), P 4 | 617.9 (8), P 1 |
| 7 | 2,744 | 856.9 (8), P 1 | 889.7 (8), P 1 | not run |

At n = 7 the four bases do not fit the GPUs together: every rank keeps two of its four subspaces on its GPU and the
others in host memory (51 GB per rank at the peak), and all 16 ranks share every k-point (P = 1). GPU memory at
n = 6 (P = 4): 74.6 GB by nvidia-smi, the estimate of the code 69 GB per GPU. The first k-point run of an allocation
compiles the device kernels of the Bloch operator (2x2x2 n = 2: 38 s the first time, 2.3 s warm).

### k-points in the sectors of their little groups (2026-11-17)

Each k-point of the wedge is split into the sectors of the commuting involutions of its little group: the
translations by half the cell (all seven for an even n, the three face-centring ones for an odd n), the twofold
axes and mirrors of Td that fix k (`PARSEC_KPOINT_SECTORS=auto`, the default; `src/parsec_python/acceleration/README.md`,
"The sectors of a k-point"). The sectors of all k-points are the tasks of the k-point pools, which are planned by
the size of each task (`PARSEC_KPOINT_POOL_ASSIGN=auto`); their complex matrices run in a CSR kernel of their own on
the device. The same inputs and launch as the table above (the default routines, 18 points per cubic edge, the mesh
reduced by Td and time reversal, the density symmetrized), first runs; "before" is the code before the sectors (the
path of `PARSEC_KPOINT_SECTORS=off`):

| n | atoms | mesh (k) | sectors | pools | SCF iterations | program (s) | first step (s) | later step (s) | GPU (GiB) | before (s) | before, GPU (GiB) | E/atom (Ry) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1,728 | 3x3x3 (4) | 64 | 16 x 1 | 8 | 18.3 | 7.9 | 1.1 | 6.0 | 195.7 | 26.5 | -11.4039207 |
| 7 | 2,744 | 2x2x2 (4) | 32 | 8 x 2 | 8 | 87.7 | 37.1 | 6.8 | 33.6 | 586.1 | 54.1 | -11.4039172 |
| 8 | 4,096 | 2x2x2 (4) | 64 | 16 x 1 | 8 | 85.9 | 37.6 | 6.2 | 25.9 | - | - | -11.4039214 |
| 9 | 5,832 | 2x2x2 (4) | 32 | 8 x 2 | 8 | 477.1 | 172.8 | 42.8 | 58.1 | - | - | -11.4039242 |
| 10 | 8,000 | 2x2x2 (4) | 64 | 16 x 1 | 8 | 374.1 | 143.3 | 31.0 | 49.5 | - | - | -11.4039262 |

"pools" is the number of k-point pools times their ranks; "later step" the median of the steps after the first.
The energies of the first two rows are those without sectors within 8e-6 Ry in total (the convergence criterion of
these inputs is 2e-4 Ry). An even n is faster than the odd n below it: its seven translations give each k-point twice
the sectors. An odd n has the three face-centring translations only, so two of the four k-points of n = 9 have four
sectors each, of 1.06 million rows and 3,805 states (65 GB a basis), besides 8 of half and 16 of a quarter of that
size. Planned by their sizes, each of the eight largest sectors gets a pool of two ranks with one medium and two small
ones (every subspace on the GPUs, about 60 GB of estimate per GPU). Planned as before (`PARSEC_KPOINT_POOL_ASSIGN=cyclic`:
every task as large as the largest, dealt round-robin), n = 9 did not run: with one rank per pool the large bases went to
host memory beside small tasks, four of them on one node, more than its 256 GB; with four ranks per pool
(`--kpoint-pools 4`) a pool stopped after five to seven steps. The plan by sizes changed n = 7 from 119.6 s
(20.3 GiB) to 87.7 s, n = 8 from 105.6 s (14.8 GiB) to 85.9 s and n = 10 from 404.7 s (51.0 GiB, two subspaces of each
rank in host memory, 37 GiB per rank) to 374.1 s with every subspace on the GPU; it keeps the same total energies to
the last digit. n = 6 has the same plan either way.

The other drivers of k-points take the sectors too (2026-11-17; the same inputs with one keyword changed, first runs):

| run | GPUs | with sectors | without | GPU, with/without (GiB) | total energy apart (Ry) |
|---|---|---|---|---|---|
| n = 6, 3x3x3, `Spin_Polarization: true` | 16 (MPI pools) | 50.9 s (89.4 s with the spin PBE on the host) | 392.2 s | 8.9 / 41.8 | 2.9e-6 |
| n = 3, 2x2x2, one relaxation step (`Minimization: bfgs`, `Movement_Num: 1`) | 4 (one process) | 83.1 s | 97.9 s | 1.7 / 6.2 | 2.3e-7 |
| n = 3, 1x1x2, `Correlation_Type: pbe0` | 4 (one process) | 53.5 s | 69.4 s | 14.6 / 15.7 | 1.3e-6 |

The two channels of a spin-polarized k-point share its sectors; the forces of a relaxation come from the orbitals of
the sectors expanded to the whole grid (in one process, or on the ranks of the MPI pools, each over its columns, since
the evening of the same day); the exchange of PBE0 is
reduced to the sectors, which keep their plans and Ritz states from one loop of the Fock loop to the next (the inner
SCF steps 0.7 instead of 2.1 s). The spin-polarized PBE of these drivers ran on the host of every rank, 37 of the 81 s
of SCF; on the GPU (the formula of the host with forward derivatives, `src/parsec_python/acceleration/V_xc/dual.py`)
it takes 0.5 s and the run 50.9 s (later steps 2.1 instead of 6.8 s), the same total energy to the digits printed.

Relaxations with k-points on the MPI pools (`--mpi`, 16 A100, `Minimization: bfgs`, `Movement_Num: 1`; no space
group: the mesh of a relaxation is reduced by time reversal alone, `PARSEC_KPOINT_TIME_REVERSAL`, since 2026-11-18):
n = 3 with the 2x2x2 mesh (8 points, each its own partner) 15.9 s, against 83.1 s in one process on four GPUs, the same
total energy (-2463.22850177 Ry) and forces to the digits printed; n = 6 with the 3x3x3 mesh (14 of its 27 points)
135.6 s, of which 118.2 s SCF, within 9e-8 Ry and 6e-6 Ry/bohr of the whole mesh (largest force 4.4e-5 Ry/bohr).
Before the same day these took 31.1 s and 381.0 s: the whole mesh (205.4 s of SCF at n = 6) and a starting density
from the readable sum (20.9 s and 159.3 s: the scope of the forces on the GPU had replaced the builder of the bulk
grid, which the single points kept, 0.24 s at n = 6).

The setup of large Gamma cells (2026-11-17): the Ewald ion-ion sums on the GPU (`acceleration/V_ion/ewald_device.py`:
the images of the host in cells of the cutoff for the real-space pairs, CuPy for the structure factors;
`PARSEC_DEVICE_EWALD=off` keeps the host sums bit for bit), the finite-difference matrix filled by slabs on threads
and the image centres of the atomic density for all atoms of a species at once (both the same bits). n = 20 on 16
A100, each run second in its allocation: 151.2 s instead of 176.2 s, the setup before the SCF 44.9 s instead of 70.0
(ion-ion 0.7 instead of 16.8 s, finite differences 5.8 instead of 10.1, atomic density 3.5 instead of 8.1), the SCF
106.3 s either way, the same total and ion-ion energies to the digits printed. With the sector stencils of the cell
built from its grid (2026-11-18: the reduction of its full matrix checked every row and column once per sector) and the
atom-matching trees of the symmetry detection kept: 141.9 s, the setup 35.4 s, the same energies.

### Scaling on one size

| case | 1 GPU | 4 GPUs (1 node) | 8 GPUs (2 nodes) | 16 GPUs (4 nodes) |
|---|---|---|---|---|
| Gamma n = 5, program (SCF) s | 33.6 (28.1) | 18.0 (12.2) | 15.1 (9.6) | 12.3 (7.0) |
| 2x2x2 n = 5, program (SCF) s | - | 180.7 (175.8) | 99.8 (94.7) | 59.4 (54.6) |

### Energy per atom

| case | E/atom (Ry) |
|---|---|
| Gamma n = 6, 7, 8, 9 | -11.4039330, -11.4039345, -11.4039348, -11.4039348 |
| 2x2x2 n = 6, 7 | -11.4039108, -11.4039172 |
| 3x3x3 n = 6, 7 | -11.4039207, -11.4039244 |
| 4x4x4 n = 6 | -11.4039169 |

A Gamma-centred m x m x m mesh on an n-cell supercell samples the same k-points as the Gamma point of the
(m n)-cell supercell, but the k-point runs discretize the Bloch operator `-(grad + ik)^2` on u, which differs from
the supercell's finite differences on `exp(ikr) u` by a term of the order of the stencil (h^7): +0.30 mRy/atom at
k = (1/2, 0, 0) 2 pi/a with this grid, +0.04 mRy/atom at 24 points per edge. That is why the k-point energies per
atom lie slightly above the converged Gamma value of the large supercells.
