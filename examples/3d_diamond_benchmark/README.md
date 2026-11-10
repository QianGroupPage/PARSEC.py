# Bulk diamond on many GPUs

Supercells of bulk diamond from 8 to 5,832 atoms, at the Gamma point and with Gamma-centred Monkhorst-Pack meshes,
computed on 16 NVIDIA A100 80 GB GPUs (four Perlmutter nodes). The series measures how large a periodic cell this
code can treat: the orbitals of one cell are shared by all ranks (one rank per GPU, any number of nodes), and with
k-points the ranks form pools of k-points.

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
