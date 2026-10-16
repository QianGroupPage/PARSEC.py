# PARSEC.py

PARSEC.py is a native Python implementation of isolated, real-space
Kohn--Sham DFT based on the algorithms in the PARSEC Fortran code. It reads
PARSEC-style `parsec.in` files and PARSEC `*_POTRE.DAT` norm-conserving
pseudopotentials directly. It does not call PARSEC, WSL, MPI, or Fortran at
runtime.

The maintained solver now has one canonical package: `src/parsec_python`.
Readable scientific modules and optimized execution backends live in that
package together. Normal calculations automatically select the fastest
accuracy-preserving combination available on the machine.

The former implementation is preserved in the Git branch
`old_architecture` and is intentionally absent from the default `main`
branch.

## Contents

**Getting started**

- [Quick start](#quick-start)
  - [CPU or GPU, and how much of each](#cpu-or-gpu-and-how-much-of-each)
- [Linux installation](#linux-installation)
  - [1. Create a Python environment (CPU-only is sufficient)](#1-create-a-python-environment-cpu-only-is-sufficient)
  - [2. Add the C++/OpenMP CPU kernels (recommended)](#2-add-the-copenmp-cpu-kernels-recommended)
  - [3. Add NVIDIA GPU acceleration (optional)](#3-add-nvidia-gpu-acceleration-optional)
  - [4. Threads, new terminals and clusters](#4-threads-new-terminals-and-clusters)
  - [5. Several GPUs and several nodes](#5-several-gpus-and-several-nodes)
  - [Troubleshooting](#troubleshooting)
- [Optional native extension](#optional-native-extension)

**Pseudopotentials**

- [Pseudopotentials](#pseudopotentials)
  - [Libraries in this repository](#libraries-in-this-repository)
  - [Using a library potential](#using-a-library-potential)
  - [Generating further potentials](#generating-further-potentials)
  - [Core-hole potentials](#core-hole-potentials)
  - [Converting a UPF file](#converting-a-upf-file)
  - [How the libraries were made, and what to cite](#how-the-libraries-were-made-and-what-to-cite)

**How the solver works**

- [Scientific flow](#scientific-flow)
- [Acceleration policy](#acceleration-policy)
- [Optional ML initial densities](#optional-ml-initial-densities)
  - [Run directly with ChargE3Net](#run-directly-with-charge3net)
  - [Run directly with SCDP](#run-directly-with-scdp)
  - [Run from an existing `.npz` or `.npy`](#run-from-an-existing-npz-or-npy)

**The code**

- [Source layout](#source-layout)
- [Modular Python API](#modular-python-api)
- [Tests](#tests)

**Reference**

- [Documentation](#documentation)
- [License](#license)

## Quick start

Linux users: see [Linux installation](#linux-installation) below for Bash commands,
CPU-only setup, and optional C++/OpenMP + NVIDIA GPU acceleration.

Python 3.12 and the repository's `.venv312` environment are recommended on
Windows:

```powershell
git clone https://github.com/QianGroupPage/PARSEC.py.git
Set-Location PARSEC.py
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r src\parsec_python\requirements.txt
```

Place `parsec.in` and its pseudopotentials in one calculation directory:

```text
calculation/
├── parsec.in
├── C_POTRE.DAT
└── H_POTRE.DAT
```

From the repository root, validate an input without building the grid:

```powershell
python src\parsec_python\main.py path\to\calculation\parsec.in --dry-run
```

Run the calculation:

```powershell
python src\parsec_python\main.py path\to\calculation\parsec.in --no-archive
```

The default text result is `parsec.out` beside the input. Unless
`--no-archive` is used, arrays and metadata are also stored in
`parsec_python_results.npz`.

A sphere input may leave `Boundary_Sphere_Radius` out. The radius and the
tolerance of the Hartree boundary are then chosen from the geometry and the
free atoms of the pseudopotential files, for an estimated energy error of
`Domain_Energy_Tolerance` (default `1e-3 Ry`; this is not the SCF criterion
`Convergence_Criterion`). `parsec.out` and `--dry-run` print the choice and
the two input lines that reproduce it; after the SCF every sphere run
reports what its sphere adds to the energy, estimated from the converged
density. An input with a radius keeps its sphere. Its Hartree boundary
values differ from PARSEC's, and from those of earlier versions of this
code, where an estimate from the geometry says PARSEC's are inaccurate;
`Hartree_Boundary_Tolerance: off` in the input restores them. With
`PARSEC_DOMAIN_REPORT=0` in the environment its report leaves the estimates
of the sphere out.
See "Sphere radius left to the code" in
[the package README](src/parsec_python/README.md); the switches that came
with it for the GPU drivers, their defaults and the values that restore the
former behaviour are in one table in
[the acceleration README](src/parsec_python/acceleration/README.md).

When `src` is importable, the equivalent package command is:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m parsec_python path\to\calculation\parsec.in --no-archive
```

For example:

```powershell
python src\parsec_python\main.py `
  examples\0d_naphthalene\parsec.in `
  --no-archive
```

### CPU or GPU, and how much of each

The same command runs on a CPU and on a GPU. By default (`--device auto`) it
takes the GPU when CuPy is installed and finds a usable NVIDIA GPU, and the
CPU otherwise. Name the device to be sure of it:

```powershell
python src\parsec_python\main.py parsec.in                 # GPU if one is usable, otherwise CPU
python src\parsec_python\main.py parsec.in --device gpu    # GPU; stops with a message if none is usable
python src\parsec_python\main.py parsec.in --device cpu    # CPU only; no GPU is touched
```

How many GPUs and CPU threads it takes can be named as well. Each choice has
a default and a variable that a job script can set instead of the option; an
option outranks its variable:

| Choice | Option | Default | Variable |
|---|---|---|---|
| Device | `--device auto\|gpu\|cpu` | `auto` | `PARSEC_DEVICE` |
| GPUs | `--gpus N\|all`, or `--gpu-ids 0,2` for particular ones | `all`: every GPU the process can see | `PARSEC_CUPY_DEVICES` |
| CPU threads | `--threads N\|all\|auto` | `auto`: the CPUs the process may use, less four kept free for Python and the GPU driver | `OMP_NUM_THREADS` |

```powershell
python src\parsec_python\main.py parsec.in --device gpu --gpus 2 --threads 16
python src\parsec_python\main.py parsec.in --device cpu --threads all
```

`--gpus` or `--gpu-ids` alone asks for a GPU run. `--threads` sets the team
of the C++/OpenMP kernels (setup and Hartree on a GPU run, the whole solver
on a CPU run with the native extension); the BLAS of NumPy keeps its own
variables (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`).

What ran is at the top of `parsec.out`, and `--dry-run` prints the same plan
without starting a calculation:

```text
 Execution resources:
 --------------------
 Device            = GPU (auto, the default: a usable GPU was found)
 GPUs              = 2 of 2 visible, index 0,1; all visible, the default
 CPU threads       = 28 of 32 CPUs allowed (affinity of the process); the default: allowed CPUs less 4
```

A GPU run needs an NVIDIA GPU with its driver and CuPy
(`python -m pip install cupy-cuda12x`; see
[step 3 of the Linux installation](#3-add-nvidia-gpu-acceleration-optional),
the same command serves Windows), and is fastest with the
[native extension](#optional-native-extension) beside it. A CPU run needs
neither; with the extension it uses the C++/OpenMP solver, without it SciPy.
Results are in double precision on either device.

On a cluster, ask the scheduler for the CPUs and GPUs and start the same
command inside the job. The defaults count what the job was given, not what
the node has: the GPUs are those the scheduler made visible
(`CUDA_VISIBLE_DEVICES`, which the code never changes; `--gpu-ids` counts
inside it), and the CPUs are those the process is bound to, or fewer where
`SLURM_CPUS_PER_TASK` (or `PBS_NP`, `NCPUS`, `NSLOTS`, `LSB_DJOB_NUMPROC`)
says so. For example with Slurm:

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=2
source .venv/bin/activate
python src/parsec_python/main.py path/to/calculation/parsec.in --device gpu
```

One command is one process on one node: it divides the symmetry sectors of
the system among its GPUs. Do not start it with `mpirun` or several `srun`
tasks, which would run the same calculation several times. For several nodes,
and for the settings the multi-GPU solver was measured with, use
`scripts/run_multi_gpu_scf.sh`
([section 5](#5-several-gpus-and-several-nodes)).

### Other options

`--symmetry auto` is the default as well. On the GPU the Hamiltonian and
the eigensolver run in CuPy and, with the native extension, preparation and
Hartree in C++/OpenMP; every selection and every fallback is recorded in
`parsec.out` under "Acceleration backend".

| Option | Meaning |
|---|---|
| `--device auto\|gpu\|cpu`, `--gpus`, `--gpu-ids`, `--threads` | What the calculation runs on (above). |
| `--backend auto\|scipy\|native\|cupy` | Which implementation executes, for controlled comparisons: `scipy` and `native` are CPU backends, `cupy` is the pure GPU backend, and none of the three falls back. `auto` follows `--device`. |
| `--symmetry auto|on|off` | Detect exact supported symmetries, require them, or force the full grid. |
| `--symmetry-cache DIR` | Keep exact-key symmetry maps and operators in `DIR`. No cache is the default (`--no-symmetry-cache` says so explicitly): a first calculation is fastest without one. |
| `--pp-dir DIR` | Search an additional directory for `*_POTRE.DAT`. |
| `--dry-run` | Parse and validate without constructing the grid. |
| `--no-archive` | Write only the text output. |
| `--profile-operator` | Time individual Hamiltonian actions. |
| `--resident` | Submit to a warmed local worker while creating fresh SCF state. |
| `--debug` | Re-raise failures with a traceback. |

## Linux installation

These commands use Bash and a source checkout. Python 3.12 is recommended;
the native extension requires Python 3.10 or newer. A GPU is optional. Neither
the PARSEC Fortran executable nor MPI is needed for this Python solver.

### 1. Create a Python environment (CPU-only is sufficient)

On Ubuntu 24.04 / recent Debian-based systems, install the prerequisites:

```bash
sudo apt update
sudo apt install git python3 python3-venv python3-dev build-essential
```

On other distributions, install the equivalent packages. On a university
cluster without administrator access, use the site's Python/compiler modules
or a user-managed environment instead of `sudo`. Use matching Python development
headers when building the extension for a non-system Python.

```bash
git clone https://github.com/zeyizhang-berkeley/parsec_python.git PARSEC.py
cd PARSEC.py
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r src/parsec_python/requirements.txt
```

If you already have a checkout, skip cloning and enter its root directory.
This installs NumPy and SciPy; CPU calculations are now available:

```bash
python src/parsec_python/main.py examples/h2_canonical_nodg/parsec.in \
  --backend scipy --dry-run
python src/parsec_python/main.py examples/h2_canonical_nodg/parsec.in \
  --backend scipy --no-archive
```

`--dry-run` validates the input without running SCF. The second command writes
`parsec.out` beside the input and replaces an existing log of that name; copy a
case to a new calculation directory when preserving reference outputs.

### 2. Add the C++/OpenMP CPU kernels (recommended)

The extension needs a C++17 compiler with OpenMP and CMake 3.24 or newer.
GCC/G++ on Linux is a suitable toolchain. From the activated environment and
repository root:

```bash
python -m pip install --upgrade "cmake>=3.24" ninja
python -m pip install -v ./src/parsec_python/acceleration/native
python -c "import parsec_accelerated_native as n; print(n.build_info())"
```

Check that `openmp_enabled` is `True`. Rebuild the extension after updating its
C++ sources: this version is meant for extension 0.6.1 (`build_info()['version']`).
With an older build a cluster whose Hartree boundary is raised above order 9
stops with a message that asks for the rebuild, the stencils of the symmetry
sectors are built by a slower NumPy route, and the launch script of section 5
stops at its start, because the GPU Hartree boundary needs the newer
extension. Windows `.pyd` files and Windows wheels cannot be reused on Linux.

```bash
python src/parsec_python/main.py examples/h2_canonical_nodg/parsec.in \
  --backend native --no-archive
```

The repository-root `pyproject.toml` also builds this extension: `pip install .`
is **not** an installation of the Python solver package. Use the launcher above,
or make the source directory importable for module/API use:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m parsec_python path/to/calculation/parsec.in --no-archive
```

### 3. Add NVIDIA GPU acceleration (optional)

First ensure the NVIDIA driver works, then choose a compatible CuPy/CUDA build.
For CUDA 12.x, with a suitable installed CUDA Toolkit:

```bash
nvidia-smi
python -m pip install cupy-cuda12x
```

If you do not have a system CUDA Toolkit, current CuPy releases also offer
`python -m pip install "cupy-cuda12x[ctk]"` to install CUDA component dependencies
in the environment; a compatible NVIDIA driver is still required. Choose one
installation route and do not install conflicting CuPy distributions together.
The CUDA version displayed by `nvidia-smi` describes driver capability, not
proof that a CUDA Toolkit is installed. See the
[official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for driver/runtime compatibility and other CUDA versions.

Check device access and a small FP64 calculation:

```bash
python -c "import cupy as cp; print(cp.__version__); print('devices:', cp.cuda.runtime.getDeviceCount()); x = cp.arange(10, dtype=cp.float64); print('sum:', x.sum().item())"
```

With CuPy installed the default (`--device auto`) already takes the GPU.
Ask for it by name to have the run stop, instead of falling back to the CPU,
if the GPU cannot be used:

```bash
python src/parsec_python/main.py examples/h2_canonical_nodg/parsec.in \
  --device gpu --no-archive
```

`Execution resources:` in `parsec.out` names the device, the GPUs and the CPU
threads. With the native extension installed as well this is the **hybrid**
policy: `Acceleration backend:` reports `Selected backend = cupy` and
`hartree_backend = native`. Explicit `--backend cupy` is a different
comparison policy (everything in CuPy), not a synonym for a GPU run.
Set `Output_Level: 2` in `parsec.in` for full backend diagnostics; SCF output
and final timing tables remain available at the default level `1`.

AMD ROCm support in CuPy is experimental and does not establish support for
this solver's custom GPU kernels. These GPU instructions target NVIDIA CUDA;
an AMD CPU can still run the CPU/OpenMP path normally.

### 4. Threads, new terminals and clusters

In a new terminal, return to the checkout and activate the Linux environment:

```bash
cd /path/to/PARSEC.py
source .venv/bin/activate
```

Linux uses `.venv/bin/activate`, not Windows `Scripts/Activate.ps1`; the launcher
uses the active Linux interpreter. Do not share a Windows virtual environment
with Linux/WSL.

The native default is the number of CPUs the process may use minus four, with
a minimum of one thread; individual small kernels may use fewer workers. Name
the number when needed (`--threads all` takes every allowed CPU):

```bash
python src/parsec_python/main.py path/to/calculation/parsec.in --threads 8 --no-archive
export OMP_NUM_THREADS=8  # the same from the environment; --threads outranks it
```

On a cluster, request CPU/GPU resources with the site's scheduler before running
jobs; ["CPU or GPU, and how much of each"](#cpu-or-gpu-and-how-much-of-each)
has a job script. The defaults follow the allocation: the CPUs the process is
bound to (or `SLURM_CPUS_PER_TASK` and its like where a site does not bind
its jobs) and the GPUs the scheduler made visible; the "Execution resources"
lines of `parsec.out`, or of `--dry-run` inside the job, say what was found.
Do not use `mpirun` to launch this single-process command: that would start
independent calculations, potentially writing to the same files, rather than
distribute one SCF calculation.

For a new-system timing comparison, run without `--resident` and without
`--symmetry-cache`: no symmetry cache is read or written unless that option
names a directory. A cache pays only for repeated calculations of the same
structure and grid; `src/parsec_python/acceleration/README.md` gives the
measured figures. CuPy compilation/driver caches are separate, so this is
not a guarantee of a completely cold GPU runtime.

### 5. Several GPUs and several nodes

`scripts/run_multi_gpu_scf.sh` runs one SCF calculation of an isolated system
(the runner refuses periodic cells) on the GPUs of one node or of several
nodes, one MPI rank per node, with the launch settings the multi-GPU solver
was measured with (1 to 16 A100 80 GB GPUs). Most of these settings are not
defaults of the code: started without them, a calculation takes slower
routes. The script needs the native extension, CuPy and mpi4py built with the
MPI of the machine, and runs inside an existing Slurm allocation; call it by
its path in the checkout rather than handing it to `sbatch`:

```bash
scripts/run_multi_gpu_scf.sh parsec.in output_directory
scripts/run_multi_gpu_scf.sh --nodes 1 --gpus-per-node 2 parsec.in output_directory
```

The first form uses every node of the allocation with four GPUs each.
`src/parsec_python/acceleration/MULTI_GPU.md` ("Several nodes") lists the
settings and says which ones the solver chooses for itself.

### Troubleshooting

- **`No module named parsec_python`:** use `python src/parsec_python/main.py ...`
  from the repository root, or set `PYTHONPATH` as above. Installing the native
  extension alone does not install the Python package.
- **Missing Python headers, compiler or OpenMP:** install the matching development
  headers and a C++17/OpenMP toolchain; check the verbose build output. Avoid
  `-ffast-math` when modifying build flags.
- **CuPy import/NVRTC/CUDA library errors:** check the active environment, driver,
  CuPy distribution and CUDA library paths. On clusters, load the appropriate
  CUDA module; `nvidia-smi` succeeding is not a complete kernel test.
- **The run used the CPU although the machine has a GPU:** `Execution
  resources` in `parsec.out` gives the reason (CuPy missing, no visible
  device). `--device gpu` makes such a run stop with that reason instead.
- **Unexpected SciPy fallback:** examine the printed fallback reason and verify
  the native and CuPy imports separately. `auto` can fall back; `--device gpu`
  and an explicit `native` or `cupy` backend fail if what they name is
  unavailable.
- **Out of GPU memory:** try `--device cpu` or a machine with more memory.
  Do not silently coarsen the grid or loosen convergence to make a job fit.

## Optional native extension

The C++/OpenMP extension is optional but recommended for fastest automatic
execution. Its source is in `src/parsec_python/acceleration/native`. A built
wheel can be installed into `.venv312` with:

```powershell
python -m pip install --force-reinstall --no-deps `
  src\parsec_python\acceleration\native\dist\parsec_accelerated_native-*.whl
```

Verify the loaded binary:

```powershell
python -c "import parsec_accelerated_native as n; print(n.__file__); print(n.build_info())"
```

The OpenMP default uses the logical processors the process may use minus four
unless `--threads` or `OMP_NUM_THREADS` names the number.

## Pseudopotentials

PARSEC.py reads norm-conserving pseudopotentials in PARSEC's `*_POTRE.DAT`
format. The repository ships libraries of them for the whole periodic table
in [`src/pp_generation`](src/pp_generation/README.md).

### Libraries in this repository

| Library | Content | Location |
|---|---|---|
| Ground state | 102 elements (H–No) for PBE, 101 for LDA (CA) | `src/pp_generation/NCPPs/{PBE,CA}/<Element>/` |
| Core hole | One potential per core shell, generated with one electron removed from that shell. All elements, 1704 potentials. | `src/pp_generation/CoreHole_NCPPs/{PBE,CA}/<Element>/<Element>-<shell>/` |

Every potential comes in three formats: `PARSEC/` for this solver, `UPF/` for
Quantum ESPRESSO and ARES, and `FHI/` (the FHI98PP table). Its input, log and
test reports are in `diagnostics/`. Each library has a `manifest.csv` with one
row per potential and a README that says what was chosen and what was checked:
[ground state](src/pp_generation/NCPPs/README.md),
[core hole](src/pp_generation/CoreHole_NCPPs/README.md).

### Using a library potential

Point `--pp-dir` at the `PARSEC` folder of each species (or copy the
`*_POTRE.DAT` files beside `parsec.in`) and paste the lines of
`PARSEC/parsec_atom_type.in` into the species block of `parsec.in`:

```text
Atom_Type: Si
Element_Symbol: Si
Pseudopotential_Format: martins_new
Local_Component: d
Atomic_Energy_Correction: -572.190341 Ry
```

`Local_Component` is the channel that passed the ghost-state test; do not
replace it by another one without testing. Use `Correlation_Type: pbe` with
the potentials of `PBE/` and `Correlation_Type: ca` with those of `CA/`.

### Generating further potentials

`pp_generation` is an interface to an installed FHI98PP or ATOM for generating
and ghost-checking a potential and converting it; neither program is
distributed here. The input of every library potential is in its
`diagnostics` folder.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m pp_generation Ni --backend fhi98pp --xc pbe --format parsec -o generated/Ni
```

See [its README](src/pp_generation/README.md).

### Core-hole potentials

A core-hole potential describes the atom whose core electron has been
removed, for the calculation of core-level binding energies. It is generated
with the same `lmax`, cutoff radii and local channel as the ground-state
potential of the element, so the two differ only by the hole. In `parsec.in`
the excited atom gets its own `Atom_Type` (for example `Si-2p`) with
`Element_Symbol: Si`; with the photoelectron removed, add `1 e` to
`Net_Charges`. The binding energy is the difference of the two
reference-corrected all-electron totals that the solver prints.

### Converting a UPF file

```powershell
python src\tools\upf_to_parsec.py input.UPF output_POTRE.DAT
```

### How the libraries were made, and what to cite

The potentials are Troullier–Martins potentials in the scheme of **FHI98PP**.
They were generated with a Python port of that program, which is not part of
this repository, and every one (1907 in all) was regenerated with the FHI98PP
program itself from the same input: all are reproduced, with tables agreeing
to 4e-9 and identical ghost-state verdicts. For work that uses them, cite

> M. Fuchs and M. Scheffler, Comput. Phys. Commun. **119**, 67 (1999)

together with the references listed in
[`src/pp_generation/README.md`](src/pp_generation/README.md#citing).

The potentials have been used in a molecule only for H, C and F; those of the
lanthanides and actinides, which have the f shell in the valence, are untested
starting points. The library READMEs list what is missing (Lr; Ho for CA; some
core holes of Ce, Th, Es, Fm and No).

## Scientific flow

The isolated single-point workflow is:

1. Parse the atomic geometry and numerical settings from `parsec.in`.
2. Read each species' `*_POTRE.DAT` file.
3. Center the cluster and build its spherical or box-shaped Cartesian grid.
4. Construct the high-order finite-difference representation of
   `-nabla^2`.
5. Interpolate and superpose local ionic potentials; build separable
   Kleinman--Bylander nonlocal projector factors.
6. Build the selected SAD, file, SCDP, or ChargE3Net starting density on the
   exact DFT grid and normalize the electron count.
7. Detect exact supported spatial symmetry and, when useful, construct
   representation-sector operators.
8. For each SCF iteration, solve Poisson's equation for `V_H`, evaluate
   `V_xc`, apply

   `H = -nabla_FD^2 + V_ion,local + V_H + V_xc + V_NL`,

   filter/diagonalize the occupied subspace, rebuild occupations and density,
   mix the effective potential, and evaluate the energy.
9. Stop according to the translated PARSEC convergence criterion or
   `Max_Iter`, then write the PARSEC-shaped report.

The Hamiltonian and nonlocal potential remain matrix-free during iterative
eigensolution. Static terms are prepared once; only density-dependent local
fields change during SCF.

## Parallel k-points, band paths and GPUs (periodic systems)

The eigenproblems of different k-points (and spin channels) are independent: only the Fermi
level and the density couple them. A periodic SCF with k-points, and every point of a band path,
is therefore spread over workers (`src/parsec_python/acceleration/Parallel`, installed into the serial reference drivers of `SCF/kpoint_solver.py`) that each keep the Bloch projectors, eigenvectors and Ritz
subspaces of their own k-points; only potentials, eigenvalues and summed densities cross between
them. This covers the scalar, spin-polarized, spin-orbit, spin-orbit + collinear-spin and
non-collinear drivers and the `bandstruc` calculation. The results are the serial ones up to
round-off (the eigenvalues of a k-point are the same computation; densities differ only by the
order of the sum over k-points).

```bash
python -m parsec_python.main parsec.in --kpoint-workers 8          # CPU worker threads
python -m parsec_python.main parsec.in --kpoint-workers auto       # default: half the CPUs
python -m parsec_python.main parsec.in --device gpu                # GPUs, see below
python -m parsec_python.main parsec.in --device cpu                # never touch a GPU
```

or the environment variables `PARSEC_KPOINT_WORKERS` (`auto` or a number; `auto` stays serial for a
single k-point and for small grids) and `PARSEC_KPOINT_BACKEND` (`cpu`, `cupy`, `auto`, `numpy-device`; an explicit override of `--device`).
The workers are threads of one process, as in the multi-GPU solver (there is no `fork`, which is
unsafe after the native OpenMP kernels or a CUDA context exist); the NumPy/SciPy/BLAS kernels release
the GIL, and each worker runs a single-threaded BLAS (a multi-threaded BLAS called from several threads at once gave wrong eigenpairs; the BLAS thread count is restored when the pool closes). The sparse Bloch operators are
memory-bandwidth bound, so the gain depends on the machine: 2.9x for 8 workers and 1.9x for 4 on a
4-P-core + 8-E-core laptop (4-atom Co, 28^3 grid, 44 spinor states, 8 k-points, 3 SCF iterations),
more on a server with more memory channels. Memory grows with the number of workers only through
each worker's own k-points.

**GPUs.** `--device gpu` (the default `--device auto` uses CuPy when it works, else the CPU; `--device cpu`
never does; `--gpus`/`--gpu-ids` choose the devices) puts the Bloch Hamiltonian, including the spin-orbit and Zeeman terms, and the whole
Chebyshev filter on the GPU, which is where almost all the matrix-vector products are. One worker
thread drives each CUDA device (`PARSEC_CUPY_DEVICES=auto|current|0,1,...` as in the multi-GPU
solver), and the k-points are dealt out round-robin. In an SCF the Ritz subspace of every k-point
stays on its device and each later SCF step (the SUBSPACE step: filter, QR, Rayleigh-Ritz) runs
there; the first solve is CHEBFF on the device, and for `Eigensolver: chebdav` it is CHEBFF followed by SUBSPACE
steps on the first potential until CHEBDAV's residual test (`Diag_Tolerance`, relative to the largest Ritz
value) is met or `PARSEC_KPOINT_REFINE_STEPS` (default 8) steps were taken: the device counterpart of CHEBDAV
(`PARSEC_KPOINT_FIRST_SOLVE=chebdav` runs the host CHEBDAV with the device filter instead). The same
converged first solve serves every k-point of a band path, with the existing retry from another random
start when the residual stays above the band tolerance. The eigenvectors come to the host only to form
the density. If the subspaces of a worker's k-points would take more than 75% of the device memory they
stay on the host between solves (`PARSEC_KPOINT_STATE_STORAGE=auto|device|host`).
`PARSEC_KPOINT_SOLVER=host` restores the former route (host eigensolver, only the filter on the
device); `python -m parsec_python.acceleration.benchmarks.check_kpoint_solver` times the two routes on
an input. Static matrices (Laplacian, projectors) are uploaded
once per device; an SCF step uploads only the local potential and fields. The host BLAS runs one thread per device worker.
`python -m parsec_python.acceleration.benchmarks.check_gpu_kpoints` compares the device operator, the filter, a short non-collinear
spin-orbit SCF and a short band path against the CPU and times them; run it first on a GPU node.
It needs `cupy`; `--backend numpy-device` runs the same device code on the host. This is a first
version of the GPU path: `check_gpu_kpoints` passes on an NVIDIA A100-SXM4-80GB with CuPy 14.2.0
(operator, filter, a short non-collinear spin-orbit SCF and a band path against the CPU); other
devices and multi-GPU runs have not been checked yet.

## Band structures (periodic systems)

A `bandstruc` block adds a non-self-consistent band structure after the SCF, as in
Fortran PARSEC's `calc_bands`. Each line is a path segment in *relative*
(reciprocal-lattice) coordinates followed by a label, and `bandstruc_points` is the number of
k-points on the shortest segment (default 20); longer segments get proportionally more, so the
density along the path is constant:

```
begin bandstruc
1 0.0 0.0 0.0   0.5 0.0 0.0   gamma-X
2 0.5 0.0 0.0   0.5 0.5 0.0   X-M
3 0.5 0.5 0.0   0.0 0.0 0.0   M-gamma
end bandstruc
bandstruc_points 20
```

The Hamiltonian follows the SCF mode, so the same block gives scalar, collinear spin-polarized
(up and down channels), self-consistent spin-orbit (`SO_from_scratch` with `SO_PSP`),
spin-orbit plus collinear spin, and non-collinear bands. Running the same input with and
without the spin-orbit switches gives the effect of spin-orbit coupling. The perturbative
spin-orbit route (`Spin_Polarization` with `SO_PSP` but no `SO_from_scratch`) is rejected for
bands.

At each k-point the Bloch Hamiltonian is built from the converged SCF potential and solved from
scratch by the SCF's first eigensolver (`Eigensolver`, default `chebdav`, which converges to
`Diag_Tolerance`; `chebff` only applies `FF_MaxIter` filter cycles and can leave the upper bands
poorly converged). The log reports the largest residual `|Hv - ev|` and warns when the bands are
not converged.

CHEBDAV adds `Matvec_Blocksize` vectors per step and, at a highly symmetric k-point whose
degenerate level is larger than that block (an 8-fold level in a folded cubic cell, 16-fold with
spin-orbit), can leave reported bands unconverged. This is an extension beyond Fortran: such a
k-point is re-solved with a larger block and a new random start (up to five attempts, keeping the
smallest residual), and the log and archive record how many k-points needed it.

The converged SCF potentials are stored in the SCF archive (`parsec_python_results.npz`), which is
written before the band step. To recompute bands (another path, `States_Num`, solver settings)
without repeating the SCF, pass the archive back with `--bands-from` (the log goes to
`parsec_bands.out`, so the SCF log is kept):

```bash
python -m parsec_python.main parsec.in --bands-from parsec_python_results.npz
```

Outputs (written next to `parsec.in`):

* `bands.dat` and `bands_plot.dat` in Fortran's formats (eigenvalues in Ry / eV, not shifted by
  the Fermi level; `bands.dat` carries the SCF Fermi level),
* `parsec_python_bands.npz` with the k-points, segment names, eigenvalues, residuals and (for
  spinor Hamiltonians) `<sigma>` of every state.

Overlay several runs, e.g. without and with spin-orbit coupling:

```bash
python -m parsec_python.Output.bands noso/bands.dat soc/bands.dat \
    --labels "no SOC" "with SOC" --names "G-X,X-M,M-G" -o bands.png
```

PARSEC.py needs an orthorhombic cell, so cubic conventional cells are used for fcc/bcc
crystals and their bands are folded into the simple-cubic Brillouin zone. Fortran's
`nscf_kpoints` (an explicit k-point list read from a `parsec.dat` restart) is not implemented.

## Modular Python API

With `src` on `PYTHONPATH`, individual scientific pieces can be used without
running SCF:

```python
from parsec_python import (
    build_cluster_grid,
    build_negative_laplacian,
    ca_lda,
    pbe,
    read_parsec_pseudopotential,
    solve_hartree,
)
```

The public workflow functions use accelerated execution by default:

```python
from parsec_python import prepare_single_point, run_scf, run_single_point

system = prepare_single_point("calculation/parsec.in", backend="auto")
result = run_scf(system)
```

For audits and algorithm study, explicit aliases retain the readable SciPy
path:

```python
from parsec_python import (
    prepare_reference_single_point,
    run_reference_scf,
    run_reference_single_point,
)
```
With `Minimization: bfgs | simple | manual` or `Molecular_Dynamics: true` the
single point becomes one step of an ionic loop: the atomic forces are
evaluated, the atoms are moved as PARSEC moves them, the terms that depend on
their positions are rebuilt on the same grid, and the next SCF loop continues
from the previous wavefunctions. `Skip_Force: false` adds the forces to a
single point. See
[Forces, relaxation and molecular dynamics](src/parsec_python/FORCES_AND_RELAXATION.md).

## Acceleration policy

Acceleration changes execution, not the physical model:

- C++/OpenMP handles loop-heavy finite-difference setup, radial interpolation,
  multipole boundaries, CA-LDA, sparse Hamiltonian application, and conjugate
  gradient where profitable.
- CuPy keeps large orbital blocks, Chebyshev filtering, orthogonalization,
  Rayleigh--Ritz work, nonlocal projections, and suitable Poisson operations
  on the GPU.
- Exact symmetry-representation decomposition reduces independent sectors
  automatically; unsupported or absent symmetry safely uses the full domain.
- NumPy/SciPy remain the reference implementation and handle small dense
  operations where GPU launch/transfer costs would be slower.
- Backend decisions, fallbacks, timings, and symmetry choices are reported.

See [the acceleration guide](src/parsec_python/acceleration/README.md) and
[the optimization audit](src/parsec_python/acceleration/ACCELERATION_AUDIT.md)
for implementation and parity details.

## Optional ML initial densities

PARSEC.py can use a density predicted by **ChargE3Net** or **SCDP** as the
initial SCF guess. This changes only the starting density: the Hartree and XC
potentials, Hamiltonian, eigensolver, mixing, convergence test, and final
energy remain ordinary PARSEC.py calculations.

For direct inference, install ChargE3Net or SCDP in its own environment. Their
PyTorch/e3nn versions are independent of the PARSEC.py `.venv312` environment.
A separate XYZ or POSCAR is not required: the interface reads the species and
coordinates from `parsec.in` and evaluates the model at the exact active DFT
grid points.

The pretrained molecular checkpoints used by this interface were trained on
QM9. They should be used for H/C/N/O/F molecules only unless a different,
validated checkpoint is supplied.

### Run directly with ChargE3Net

After installing the official ChargE3Net repository and confirming that
`models/charge3net_qm9.pt` is present, add this block to `parsec.in`:

```text
Initial_Density: charge3net
ML_Density_Model: qm9
ML_Density_Repository: C:\path\to\charge3net
ML_Density_Python: C:\path\to\charge3net\.venv\Scripts\python.exe
ML_Density_Device: auto
Normalize_Initial_Density: true
```

### Run directly with SCDP

Install the official SCDP repository in a separate environment and download
one of its published QM9 checkpoints. For the fast checkpoint, use:

```text
Initial_Density: scdp
ML_Density_Model: fast
ML_Density_Repository: C:\path\to\scdp
ML_Density_Python: C:\path\to\scdp\.venv\Scripts\python.exe
ML_Density_Checkpoint: C:\path\to\scdp_fast.ckpt
ML_Density_Device: auto
Normalize_Initial_Density: true
```

Run either case with the normal command:

```powershell
python src\parsec_python\main.py path\to\calculation\parsec.in --no-archive
```

Before SCF, PARSEC.py launches the external environment, generates a portable
exact-grid `density.npz`, validates its values and units, normalizes it to the
pseudopotential valence-electron count, and caches it under
`.parsec_ml_density_cache` beside `parsec.in`. An unchanged model, geometry,
and grid reuse the exact cached prediction. Set `ML_Density_Regenerate: true`
to force a new prediction.

### Run from an existing `.npz` or `.npy`

The generated `.npz` can be copied with the calculation and reused without
installing ChargE3Net, SCDP, PyTorch, or the original checkpoint:

```text
Initial_Density: charge3net
ML_Density_File: charge3net_qm9_density.npz
ML_Density_Model: qm9
ML_Density_Units: auto
ML_Density_Interpolation: linear
ML_Density_Negative_Policy: clip
Normalize_Initial_Density: true
```

Use `Initial_Density: scdp` for a stored SCDP prediction, or
`Initial_Density: file` for a provider-neutral density. `.npz` is preferred
because it stores coordinates, units, and provenance. A legacy `.npy` must be
a three-dimensional density on exactly the same underlying Cartesian grid;
with automatic units it is interpreted as electrons per cubic angstrom.

For installation commands, checkpoint details, cache discovery, file schemas,
and a complete start-to-finish example, see the
[ML-density interface guide](src/parsec_python/MLDensity/README.md). The
[42-case regression suite](examples/ml_initial_density/README.md) contains
runnable SAD, ChargE3Net, and SCDP comparisons with stored predictions.

## Source layout

```text
examples/                          runnable calculations and benchmark data
├── README.md                      case guide and execution commands
├── h2_canonical_nodg/
├── h2_full_nonlocal/
├── 0d_benzene/
├── 0d_naphthalene/
├── 0d_Si28H36/
└── 0_CH4_CF4/
scripts/
└── run_multi_gpu_scf.sh           launch script for several GPUs and nodes
src/
├── parsec_python/                 canonical DFT package
│   ├── main.py                    default command-line launcher
│   ├── __main__.py                python -m parsec_python
│   ├── driver.py                  readable preparation/reference workflow
│   ├── reference_main.py          explicit readable CLI launcher
│   ├── models.py                  shared physical/settings/result data
│   ├── Input/                     parsec.in translation and validation
│   ├── MLDensity/                 optional SCDP/ChargE3Net initial guesses
│   ├── Grid/                      isolated real-space domains
│   ├── Laplacian/                 high-order finite differences
│   ├── Pseudopotential/           PARSEC POTRE reader and radial splines
│   ├── V_ion/                     local potential and nonlocal projectors
│   ├── Hartree/                   Poisson solve and open-boundary data
│   ├── V_xc/                      CA/PZ LDA and PBE, also spin-polarized
│   ├── Hamiltonian/               matrix-free Kohn--Sham operator
│   ├── Eigensolvers/              PARSEC-style ChebFF/CHEBDAV/subspace code
│   ├── Occupations/               Fermi level, occupations, and density
│   ├── Mixer/                     SCF potential mixing
│   ├── Energy/                    total-energy bookkeeping
│   ├── SCF/                       self-consistent single-point loop
│   ├── Forces/                    atomic forces of isolated systems
│   ├── Relaxation/                BFGS, steepest descent, manual movements
│   ├── MolecularDynamics/         PARSEC molecular dynamics
│   ├── Output/                    PARSEC-shaped text reporting
│   ├── acceleration/              internal optimized implementation
│   │   ├── backends/              SciPy, CuPy, and native adapters
│   │   ├── Eigensolvers/          device-resident GPU eigensolvers
│   │   ├── Symmetry/              automatic representation decomposition
│   │   ├── native/                optional C++/OpenMP extension
│   │   ├── resident.py            warmed-process execution
│   │   └── tests/                 backend, parity, and GPU tests
│   └── tests/                     scientific and architecture tests
├── pp_generation/                 pseudopotential libraries and generation interface
│   ├── NCPPs/                     ground-state library, PBE and CA
│   ├── CoreHole_NCPPs/            core-hole library, PBE and CA
│   ├── backends/                  drivers of an installed FHI98PP or ATOM
│   └── tests/
└── tools/
    └── upf_to_parsec.py           UPF-to-PARSEC converter
```

There are no longer separate `new_architecture` and
`accelerated_architecture` packages. `parsec_python.acceleration` is an
internal implementation layer, not a second public solver.
Runnable inputs and benchmark reference data are deliberately outside the
importable package under [`examples/`](examples/README.md).

## Modular Python API

With `src` on `PYTHONPATH`, individual scientific pieces can be used without
running SCF:

```python
from parsec_python import (
    build_cluster_grid,
    build_negative_laplacian,
    ca_lda,
    pbe,
    read_parsec_pseudopotential,
    solve_hartree,
)
```

The public workflow functions use accelerated execution by default:

```python
from parsec_python import prepare_single_point, run_scf, run_single_point

system = prepare_single_point("calculation/parsec.in", backend="auto")
result = run_scf(system)
```

For audits and algorithm study, explicit aliases retain the readable SciPy
path:

```python
from parsec_python import (
    prepare_reference_single_point,
    run_reference_scf,
    run_reference_single_point,
)
```

## Tests

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s src\parsec_python\tests -p "test_*.py" -v
python -m unittest discover -s src\parsec_python\acceleration\tests -p "test_*.py" -v
python -m unittest discover -s src\pp_generation\tests -p "test_*.py" -v
```

The acceleration suite includes real CUDA tests when CuPy and a CUDA device
are available; otherwise those runtime-specific tests skip cleanly.

## Documentation

- [PARSEC algorithm map](src/parsec_python/PARSEC_ALGORITHM.md)
- [Python implementation map](src/parsec_python/PYTHON_IMPLEMENTATION.md)
- [Package architecture](src/parsec_python/ARCHITECTURE.md)
- [Forces, relaxation and molecular dynamics](src/parsec_python/FORCES_AND_RELAXATION.md)
- [Acceleration implementation](src/parsec_python/acceleration/README.md)
- [ML initial-density interface](src/parsec_python/MLDensity/README.md)
- [Pseudopotentials: libraries, citation and generation interface](src/pp_generation/README.md)
- [Ground-state pseudopotential library](src/pp_generation/NCPPs/README.md)
- [Core-hole pseudopotential library](src/pp_generation/CoreHole_NCPPs/README.md)

## License

PARSEC.py is distributed under the GNU General Public License, version 3 or
(at your option) any later version. See [LICENSE](LICENSE) for the Berkeley Lab
license notice, [COPYING](COPYING) for the complete GNU GPL v3 text, and
[COPYRIGHT](COPYRIGHT) for the copyright notice and U.S. Government rights.

The pseudopotential libraries in `src/pp_generation` were generated in the
scheme of FHI98PP by Martin Fuchs and Matthias Scheffler
(Fritz-Haber-Institut der Max-Planck-Gesellschaft); the references to cite are
in [`src/pp_generation/README.md`](src/pp_generation/README.md#citing).
FHI98PP itself is not part of this repository.
