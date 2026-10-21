# Hydrogen-terminated nanodiamonds

Inputs and reference outputs for nanodiamond clusters from C1H4 to
C9449H1572: 5 to 11,021 atoms, 8 to 39,368 valence electrons. They are meant
both as correctness checks and as GPU benchmarks.

- `fortran_comparison/` holds k00 to k06 (up to C795H300). Each system was
  computed by Fortran PARSEC and by this code from the same physical input.
- `gpu_benchmark/` holds k06 to k14 (C795H300 to C9449H1572). This is the
  series on which the multi-GPU solver of this code was measured on
  NVIDIA A100 80 GB GPUs. The tables below give the reference times and GPU
  memory on 1 to 16 GPUs.
- `performance/` has figures and data of those measurements: time and GPU
  memory against the number of electrons and of GPUs, fitted power laws, where
  the time goes, and what limits the size of a system.

All inputs share the following settings:

- **Functional and potentials:** local-density approximation (Ceperley-Alder)
  with the norm-conserving potentials `C_POTRE.DAT` and `H_POTRE.DAT` of this
  folder.
- **Grid:** spacing 0.2 Angstrom, finite differences of order 8.
- **Domain:** a sphere that ends 5 Angstrom beyond the outermost atom
  (`Boundary_Sphere_Radius` in each input).
- **Occupations:** Fermi-Dirac at 500 K.
- **SCF:** converged to a residual of 2e-4 Ry.

Every cluster has the twofold rotations about the x, y and z axes (the group
D2), so the code splits the eigenproblem into four independent symmetry
sectors. Geometries:

- k00 to k07 are GFN2-xTB optimized;
- k08 to k14 are cut from the ideal diamond lattice, terminated with hydrogen
  and not relaxed.

k06 appears in both folders with the same atoms. The two inputs differ in
solver controls and in the Hartree boundary (see the notes below), so their
total energies differ by 3.2e-2 Ry. The benchmark inputs use first-step filter
settings that keep the large clusters at 10 or 11 SCF steps.

```text
0d_nanodiamonds/
  C_POTRE.DAT, H_POTRE.DAT      potentials read by every input below
  fortran_comparison/kNN_<formula>/
    parsec.in                   input of this code
    parsec_reference.out        its output: master 529bab9e, one A100 80 GB
    parsec_fortran.in           Fortran PARSEC input of the same system
    fortran_reference.out       Fortran PARSEC output (commit f432777, 28 MPI ranks)
  gpu_benchmark/kNN_<formula>/
    parsec.in                   input of this code
    parsec_reference.out        its output on 16 A100 80 GB (four nodes)
```

The output headers are unchanged except that machine paths were shortened to
file names.

## Quick start: `run_benchmark.sh`

`run_benchmark.sh` runs the cases you name with the launch settings of the
reference times. It then compares every run with its reference: total energy,
SCF steps, program time and peak GPU memory. It writes a table to the screen
and to `summary.csv`.

```bash
# the smallest benchmark case on one GPU
examples/0d_nanodiamonds/run_benchmark.sh --gpus-per-node 1 k06

# three cases on the four GPUs of one node
examples/0d_nanodiamonds/run_benchmark.sh k06 k07 k08

# inside a Slurm allocation of four nodes (salloc or sbatch): the largest case on 16 GPUs
examples/0d_nanodiamonds/run_benchmark.sh --nodes 4 k14

# every Fortran comparison case on one GPU, one plain main.py process each
examples/0d_nanodiamonds/run_benchmark.sh --fortran-set --launcher main --gpus-per-node 1 all
```

The script needs the Python environment of the repository README: NumPy,
SciPy, CuPy and the native extension. The default launcher also needs mpi4py
built with the MPI of the site; `--launcher main` does without it, on one node.
It checks these at the start, and `--check` prints the commands without
running them; `--help` lists every option. Each line of the table looks like:

```text
  k06 C795H300: ok, 10 steps (reference 10), E = -9437.81995096 Ry (difference 0.0e+00), 9.9 s (A100 9.9), peak 4.3 GiB (A100 4.3)
```

A run whose total energy is within 1e-5 Ry of the reference is reported as
`ok`. The A100 columns hold the reference for the same total number of GPUs,
blank where the case did not fit on that many 80 GB GPUs.

Before the table, the smallest case of the set runs once on the same GPUs.
That run compiles the CUDA kernels into CuPy's cache, as the warm-up runs
before the reference times did; `--no-warmup` skips it. Use the default
launcher for timing comparisons. `--launcher main` runs `main.py` with the
code's defaults, which keep the Hartree solver and parts of the set-up on the
CPU; on four A100 it took about three times as long for k06, with the same
energy.

## Running this code by hand

From the repository root, one process on one node uses the GPUs it is given:

```bash
python src/parsec_python/main.py \
    examples/0d_nanodiamonds/gpu_benchmark/k06_C795H300/parsec.in \
    --pp-dir examples/0d_nanodiamonds --device gpu --gpus 1 --no-archive
```

One process places the four symmetry sectors on up to four GPUs (`--gpus 2`
or `--gpus 4`) with the code's default settings. The times in the tables
below were measured with `scripts/run_multi_gpu_scf.sh`, which applies the
measured launch settings and runs one MPI rank per node. On 8 and 16 GPUs the
GPUs of a node share the basis of a sector. Runner options such as `--pp-dir`
follow the output directory:

```bash
# one node, two of its GPUs
scripts/run_multi_gpu_scf.sh --nodes 1 --gpus-per-node 2 \
    examples/0d_nanodiamonds/gpu_benchmark/k09_C2455H636/parsec.in out_k09_2gpu \
    --pp-dir examples/0d_nanodiamonds

# four nodes of four GPUs (16 GPUs)
scripts/run_multi_gpu_scf.sh --nodes 4 \
    examples/0d_nanodiamonds/gpu_benchmark/k14_C9449H1572/parsec.in out_k14_16gpu \
    --pp-dir examples/0d_nanodiamonds
```

Read the head of that script for the modules and Python environment it
expects. The code decides at the first eigensolve whether the basis of a
sector fits on its GPUs. With less memory per GPU than 80 GB, a case needs
more GPUs than the tables show; with more memory, fewer.

## Running Fortran PARSEC

Copy `parsec_fortran.in` as `parsec.in` into an empty directory together with
the two potential files. Run the PARSEC executable there. The references used
`mpirun -np 28` on a 16-core AMD Ryzen 9 8945HX workstation under WSL.

## Reference results

### Fortran comparison (k00-k06)

| Case | Formula | Atoms | Valence electrons | Fortran: SCF steps | Fortran: total energy (Ry) | This code: SCF steps | This code: total energy (Ry) | Difference (Ry) | Fortran wall time, 28 MPI ranks (s) | This code, one A100 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k00 | C1H4 | 5 | 8 | 10 | -16.05943063 | 10 | -16.05943063 | +0.0e+00 | 2.6 | 35.5 |
| k01 | C5H12 | 17 | 32 | 16 | -71.08399739 | 16 | -71.08399749 | -1.0e-07 | 5.7 | 7.5 |
| k02 | C35H36 | 71 | 176 | 15 | -441.84362635 | 13 | -441.84362610 | +2.5e-07 | 32.6 | 12.5 |
| k03 | C87H76 | 163 | 424 | 14 | -1082.61328284 | 16 | -1082.61328393 | -1.1e-06 | 121.5 | 18.4 |
| k04 | C185H124 | 309 | 864 | 11 | -2258.73457597 | 11 | -2258.73457616 | -1.9e-07 | 463.7 | 17.4 |
| k05 | C459H204 | 663 | 2,040 | 34 | -5484.94139248 | 44 | -5484.94139062 | +1.9e-06 | 3,945.0 | 94.7 |
| k06 | C795H300 | 1,095 | 3,480 | 38 | -9437.85232766 | 35 | -9437.85232426 | +3.4e-06 | 11,213.8 | 129.2 |

The Fortran times are for 28 MPI ranks on a 16-core AMD Ryzen 9 8945HX
workstation; this code ran on one A100 80 GB of a Perlmutter node at NERSC.
The two columns therefore compare complete runs on different hardware, not
kernels. k00 was the first run of this code version on that node: its time
includes compiling the CUDA kernels into CuPy's cache (the next case, k01,
took 7.5 s).

### GPU benchmark (k06-k14): program wall time in seconds

| Case | Formula | Valence electrons | Grid points | States computed | SCF steps | Total energy (Ry) | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs | 16 GPUs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k06 | C795H300 | 3,480 | 2,710,168 | 1,767 | 10 | -9437.81995096 | 22.0 | 14.5 | 9.9 | 6.9 | 6.0 |
| k07 | C1213H412 | 5,264 | 4,125,288 | 2,659 | 10 | -14346.75097475 | 43.6 | 25.5 | 16.8 | 11.2 | 8.5 |
| k08 | C1653H508 | 7,120 | 4,778,880 | 3,588 | 10 | -19488.31721372 | 68.6 | 38.7 | 23.5 | 15.5 | 11.4 |
| k09 | C2455H636 | 10,456 | 7,699,168 | 5,256 | 11 | -28808.04029769 | 355.4 (a) | 91.0 | 51.8 | 32.8 | 22.0 |
| k10 | C3469H804 | 14,680 | 10,419,384 | 7,367 | 11 | -40596.27462396 |  | 182.4 | 98.5 | 61.2 | 38.5 |
| k11 | C4601H988 | 19,392 | 11,493,600 | 9,724 | 10 | -53752.09127639 |  |  | 145.3 | 87.3 | 52.7 |
| k12 | C5659H1132 | 23,768 | 15,147,328 | 11,911 | 10 | -66016.14132692 |  |  |  | 145.9 | 84.8 |
| k13 | C7067H1308 | 29,576 | 16,522,040 | 14,815 | 10 | -82319.08242217 |  |  |  | 211.4 | 119.5 |
| k14 | C9449H1572 | 39,368 | 22,640,800 | 19,711 | 10 | -109862.37143299 |  |  |  |  | 239.4 |

Measured on Perlmutter at NERSC: nodes of four A100-SXM4-80GB, one MPI rank
per node, launched with `scripts/run_multi_gpu_scf.sh`. Runs on 1, 2 and
4 GPUs use one node; runs on 8 and 16 GPUs use two and four nodes. One run per
cell after a warm-up run (k09 on 1 GPU and k10 on 2 GPUs: one run). Measured on
21 October 2026 with `master` 224a876. Blank cells: the basis does not fit in
the memory of that many 80 GB GPUs. (a) With this input the basis of k09 does
not fit on one card, so the code keeps it in page-locked host buffers (78 GiB
of host memory in all) and moves one sector at a time to the GPU. Without the
`Boundary_Sphere_Radius` line, the code chooses a sphere 0.7 Angstrom smaller,
which fits on one card (205 s with the code of 10 October 2026).
`performance/` has figures of these times and memory peaks and how they grow
with the size of the system.

### GPU benchmark (k06-k14): peak memory of the fullest GPU in GiB

| Case | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs | 16 GPUs |
|---|---:|---:|---:|---:|---:|
| k06 | 11.2 | 6.5 | 4.3 | 3.2 | 2.2 |
| k07 | 23.1 | 12.5 | 7.3 | 6.0 | 4.0 |
| k08 | 34.8 | 18.5 | 10.3 | 7.3 | 5.3 |
| k09 | 23.6 (a) | 41.7 | 22.6 | 13.8 | 9.1 |
| k10 |  | 76.3 | 40.1 | 22.0 | 13.1 |
| k11 |  |  | 56.8 | 30.6 | 17.7 |
| k12 |  |  |  | 47.6 | 26.7 |
| k13 |  |  |  | 63.5 | 35.2 |
| k14 |  |  |  |  | 61.2 |

The peak memory is the largest value that `nvidia-smi`, sampled every 200 ms,
showed on the fullest GPU. It includes the CUDA context and the cache of the
CuPy memory pool. A card has 79.25 GiB.

Notes:

- **Hartree boundary.** The inputs of `fortran_comparison/` contain
  `Hartree_Boundary_Tolerance: off`, which selects PARSEC's Hartree boundary
  (multipoles up to `Solver_Lpole` 9). Without that line this code raises the
  multipole order until the boundary error is below 1e-3 Ry, and the total
  energy of the larger clusters moves by design (C795H300: 3.2e-2 Ry). The
  inputs of `gpu_benchmark/` use that default boundary.
- **Solver controls.** Both codes use the same solver controls, except the
  Fortran runs of k05 and k06. With the mixing of these inputs, the Fortran run
  of k05 did not converge within 70 steps. The Fortran references of k05 and
  k06 therefore use Anderson mixing 0.20, a restart every 5 steps and up to 100
  steps.
- **What to expect on other hardware.** A correct run should reproduce the
  total energy to about 1e-6 Ry. The number of SCF steps can differ by one or
  two, and by more for the slowly converging k05 and k06 of the Fortran
  comparison. Times are of a first calculation of each structure (no symmetry
  cache) in double precision. Program time runs from the end of input parsing
  to the final report and includes CUDA initialization.
- **Version of the benchmark references.** The times and memory peaks of the
  two GPU tables (and the references of `run_benchmark.sh`) were measured with
  `master` 224a876 of 21 October 2026. The `gpu_benchmark/` outputs and the
  total energies in the table come from the code published on 10 October 2026;
  the code of 21 October gives the same SCF step counts in all 32 cells and
  total energies within 2e-7 Ry of them. Its stencil kernel keeps more loads in
  flight than that of the code of 17 October, with the same total energies to
  the last bit: the SCF time is 3 to 24 % lower (10 % for k14 on 16 GPUs) and
  the memory peaks are the same within 0.05 GiB. Against the code of 10
  October the code of 17 October had changed the program time by -7 to +1 % on
  1 to 16 GPUs (-46 % for k09 on one GPU, whose basis lives in host memory) and
  lowered the peak of the fullest GPU by up to 2.5 GiB.
