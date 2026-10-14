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

## Running this code

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
| k06 | C795H300 | 3,480 | 2,710,168 | 1,767 | 10 | -9437.81995096 | 27.8 | 17.0 | 11.1 | 7.8 | 6.0 |
| k07 | C1213H412 | 5,264 | 4,125,288 | 2,659 | 10 | -14346.75097475 | 58.1 | 32.7 | 20.4 | 13.3 | 9.5 |
| k08 | C1653H508 | 7,120 | 4,778,880 | 3,588 | 10 | -19488.31721372 | 90.1 | 49.3 | 28.7 | 18.6 | 12.6 |
| k09 | C2455H636 | 10,456 | 7,699,168 | 5,256 | 11 | -28808.04029769 | 745.1 (a) | 117.3 | 65.0 | 39.7 | 25.3 |
| k10 | C3469H804 | 14,680 | 10,419,384 | 7,367 | 11 | -40596.27462396 |  | 231.6 | 123.8 | 73.8 | 44.4 |
| k11 | C4601H988 | 19,392 | 11,493,600 | 9,724 | 10 | -53752.09127639 |  |  | 180.1 | 103.8 | 61.4 |
| k12 | C5659H1132 | 23,768 | 15,147,328 | 11,911 | 10 | -66016.14132692 |  |  |  | 172.5 | 98.9 |
| k13 | C7067H1308 | 29,576 | 16,522,040 | 14,815 | 10 | -82319.08242217 |  |  |  | 249.2 | 140.2 |
| k14 | C9449H1572 | 39,368 | 22,640,800 | 19,711 | 10 | -109862.37143299 |  |  |  |  | 279.9 |

Measured on Perlmutter at NERSC: nodes of four A100-SXM4-80GB, one MPI rank
per node, launched with `scripts/run_multi_gpu_scf.sh`. Runs on 1, 2 and
4 GPUs use one node; runs on 8 and 16 GPUs use two and four nodes. One run per
cell after a warm-up run. Blank cells: the basis does not fit in the memory
of that many 80 GB GPUs. (a) With this input the basis of k09 does not fit on
one card, so the code keeps it in host memory (110 GiB) and moves one sector at
a time to the GPU. Without the `Boundary_Sphere_Radius` line, the code chooses
a sphere 0.7 Angstrom smaller, which fits on one card: 205 s.

### GPU benchmark (k06-k14): peak memory of the fullest GPU in GiB

| Case | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs | 16 GPUs |
|---|---:|---:|---:|---:|---:|
| k06 | 11.2 | 6.5 | 4.2 | 3.2 | 2.2 |
| k07 | 23.1 | 12.5 | 7.3 | 6.1 | 4.1 |
| k08 | 34.8 | 18.5 | 10.3 | 7.3 | 5.3 |
| k09 | 23.9 (a) | 42.1 | 23.0 | 15.6 | 10.0 |
| k10 |  | 78.3 | 42.1 | 24.5 | 14.9 |
| k11 |  |  | 58.8 | 32.0 | 19.6 |
| k12 |  |  |  | 49.5 | 28.5 |
| k13 |  |  |  | 65.5 | 36.1 |
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
- **Version of the benchmark references.** The `gpu_benchmark/` outputs and
  times come from the code published on 10 October 2026. The present `master`
  gives the same SCF step counts and total energies within 2e-7 Ry. In the
  five cells measured side by side (k06 on 1 and 16 GPUs, k10 on 4, k12 and
  k14 on 16), its program time is 3.6 to 5.4 % lower.
