# Time and memory of the GPU benchmark

Figures and data of the `gpu_benchmark/` series (k06 to k14, C795H300 to
C9449H1572, 3,480 to 39,368 valence electrons) on 1 to 16 NVIDIA A100 80 GB
GPUs. They are the measurements behind the two GPU tables of `../README.md`.

## How they were measured

- Code: `master` 0f47a3f of 23 October 2026, default settings, double
  precision, first calculations (no symmetry cache), the inputs of
  `gpu_benchmark/` (sphere 5 Angstrom beyond the outermost atom). Besides the
  settings of the launch script, 30 of the 32 runs set
  `PARSEC_CUPY_SECTOR_STATE_STORAGE=device` and
  `PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR=pool` (the basis on the GPUs, the CuPy
  memory pool), as the earlier series did; k09 on 1 GPU and k10 on 2 GPUs leave
  both to the code, which chose the same for k10.
- Hardware: Perlmutter at NERSC, nodes of four A100-SXM4-80GB and one AMD EPYC
  7763; one MPI rank per node with 32 CPUs and 16 OpenMP threads, launched with
  `scripts/run_multi_gpu_scf.sh`. 1, 2 and 4 GPUs use one node; 8 and 16 GPUs
  use two and four nodes, the four nodes of the 16-GPU runs under one network
  switch. Before the runs the Python environment, the sources and the CuPy
  kernel cache were copied to the node-local `/tmp` of every node; inputs and
  outputs stayed on the project file system. On Perlmutter, since its
  maintenance of 21 October 2026, reading them from the project file system
  (CFS) added 1 to 3 s to the first SCF step of runs on four nodes, at times.
- One measured run per configuration after a warm-up run (k09 on 1 GPU and k10
  on 2 GPUs, with the storage of the basis left to the code: the second of two
  runs on a node; the first run on a freshly prepared node took 22 to 31 s
  longer, in its first Hartree solve and its first eigensolve). Every run
  converged in the same number of SCF steps as the outputs in
  `gpu_benchmark/`, with total energies within 2e-7 Ry of them.
- Program time runs from the end of input parsing to the final report. Peak
  GPU memory is the largest value `nvidia-smi` showed on the fullest GPU,
  sampled every 200 ms, including the CUDA context and the cache of the CuPy
  memory pool.
- Power laws `y = a N^p` are least-squares fits of log y against log N; the
  intervals in `data/fits.csv` are 95 % confidence intervals of the exponent.

## Figures

Each figure is in `figures/` as PNG and PDF.

| File | What it shows |
|---|---|
| `time_and_memory_vs_electrons` | Program time and peak GPU memory against the number of electrons, one line per GPU count, with fitted power laws |
| `time_and_memory_vs_gpus` | Program time, peak GPU memory and host memory against the number of GPUs, one line per cluster |
| `time_by_stage` | Program time of every run split into first eigensolve, later eigensolves, density, Hartree, other SCF work and preparation |
| `gpu_memory_by_gpu_count` | Peak GPU memory of every run, by GPU count |
| `stage_time_vs_electrons_16_gpus` | Seconds of each stage of the eigensolver of one symmetry sector on 16 GPUs, with fitted power laws |
| `stage_share_16_gpus` | Share of the program time by stage on 16 GPUs |
| `peak_memory_against_basis` | Measured peak of the fullest GPU against the part of the basis that GPU holds |
| `basis_against_gpu_memory` | The basis of all four sectors against the memory of 1 to 16 GPUs, and the largest system the present layout admits |

## What they show

- **Time.** With the basis in GPU memory a complete calculation takes 5.7 s
  (k06 on 16 GPUs) to 225 s (k14 on 16 GPUs). At a fixed number of GPUs the
  program time grows as N^1.52 (16 GPUs) to N^1.77 (2 GPUs) in the number of
  electrons N, and as N^1.76 above 10,000 electrons on 16 GPUs.
- **Where the time goes.** The Chebyshev filter (sparse) grows as N^1.80. The
  dense products of the Rayleigh-Ritz step grow faster (Gram matrices N^2.38,
  rotation N^2.66, on 16 GPUs), so their share of the program time rises from
  5 % for k06 to 47 % for k14. Copies of the basis between the GPUs of a sector
  take 11 to 15 % of the program time on 16 GPUs. Part of these copies runs
  beside the Gram products; the stage seconds count both in full, so for the
  largest clusters the shares add up to about the whole program time.
- **Memory.** The dense basis (grid points by states, per symmetry sector) must
  stay in GPU memory and grows as N^1.86. The measured peak of a GPU is about
  2.1 GiB plus 1.06 times the part of the basis it holds. The peak grows as
  N^1.43 (16 GPUs) to N^1.72 (2 GPUs).
- **The limit is GPU memory.** At every GPU count the largest cluster that fits
  finishes in under four minutes. A sector is shared by at most the four GPUs
  of one node, so a
  system with four sectors stops at 16 GPUs: on 80 GB cards at about 43,400
  electrons, 10 % above k14. By extrapolation a cluster of 100,000 electrons
  would take about 36 minutes on 16 GPUs, but needs 293 GiB of basis per GPU,
  3.7 times what an 80 GB card has.
- **Basis in host memory.** k09 does not fit on one card with this input: the
  code keeps its basis in page-locked host memory and moves one sector at a
  time to the GPU (348 s, against 91 s on two GPUs).

## Data

- `data/runs.csv`: one row per measured run (32 rows). Main columns: `case`,
  `electrons`, `gpus`, `route` (`single`: every sector on one GPU; `shared`:
  the GPUs of a sector share its basis; `spill`: the basis in host memory),
  `program_s`, `scf_s`, `scf_steps`, `total_energy_ry`, `peak_gpu_gib` (fullest
  GPU), `host_rss_gib` (fullest MPI rank), `sector_rows` and `sector_states`
  (size of the basis of the largest sector), `basis_gib_fullest_gpu`, and the
  seconds of the stages (`first_eig_s`, `later_eig_s`, `density_s`,
  `hartree_s`; on the shared route also `filter_s`, `gram_s`, `rotate_s`,
  `to_rows_s`, `to_columns_s` of one sector).
- `data/fits.csv`: every fitted power law: quantity, GPU count, range of
  sizes, exponent with its 95 % interval, prefactor and R^2.
