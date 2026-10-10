# Acceleration implementation

The [GPU setup and memory profile](GPU_SETUP_OPTIMIZATION.md) documents the
Perlmutter-tested ionic preparation, bounded native CSR construction, and
column-major filter output, including the launch settings of those measurements.
The [streaming Hartree and spatial-ordering study](GPU_SFC_OPTIMIZATION.md)
adds Ritz workspace reuse, GPU boundary construction, measured memory
limits, and the spatial-ordering variants that were not retained.

`parsec_python.acceleration` is the internal performance layer of the canonical
`parsec_python` package. It reuses the readable modules' input translation,
physical models, pseudopotentials, real-space conventions, Hartree and XC
treatment, SCF policy, convergence test, energy expression, and result
objects. The public workflow and command-line entry points select this layer
by default; explicit `*_reference` API aliases and `reference_main.py` retain
the readable SciPy path for audits.

Backend changes must preserve the selected physical model and float64 result
accuracy. Component arrays, SCF behavior, and energies are tested against the
readable implementation and PARSEC reference cases.

## Installation and command line

Install the reference NumPy/SciPy dependencies first:

```powershell
python -m pip install -r src\parsec_python\requirements.txt
```

SciPy then works immediately. To add the C++17/OpenMP backend, build its
separate extension from the repository root:

```powershell
python -m pip install -v src\parsec_python\acceleration\native
```

CuPy is optional; install the CuPy wheel matching the machine's CUDA runtime.
The package intentionally does not pin a CUDA-specific wheel.

Installing both optional runtimes enables the fastest default composition:
the native extension constructs the finite-difference operator and solves the
Hartree system, while CuPy keeps the Hamiltonian and eigensolver work on the
GPU. Exact commuting Cartesian signed permutations are detected automatically
for scalar fields and Hartree. The CuPy eigensolver uses all real
one-dimensional representations of the proved diagonal-reflection subgroup,
as PARSEC does. Fixed-point orbits are admitted by their exact stabilizer
characters, so representation dimensions may differ. All operations,
dimensions, device assignments, and fallback decisions are reported.

The canonical launcher accepts the same supported `parsec.in` and adjacent
`*_POTRE.DAT` files on every backend:

```powershell
python src\parsec_python\main.py INPUT
python src\parsec_python\main.py INPUT --backend auto
python src\parsec_python\main.py INPUT --backend scipy
python src\parsec_python\main.py INPUT --backend native
python src\parsec_python\main.py INPUT --backend cupy
python src\parsec_python\main.py INPUT --symmetry off
python src\parsec_python\main.py INPUT --resident
```

On Windows, this source-tree launcher automatically re-runs itself with the
validated project-local `.venv312` interpreter when that environment exists.
This prevents a bare `python main.py parsec.in` from loading the legacy
`.venv` and its obsolete native extension. Set
`PARSEC_ACCELERATED_USE_ACTIVE_PYTHON=1` only when intentionally testing a
different fully configured environment.

The complete selection/profile form is:

```text
python src\parsec_python\main.py INPUT \
  --backend {auto,scipy,native,cupy} \
  --symmetry {auto,on,off} \
  [--symmetry-cache DIRECTORY | --no-symmetry-cache] \
  [--profile-operator]
```

The symmetry mode defaults to `auto`: detect exact supported operations,
apply every reduction supported by the selected backend, and safely retain
the full grid if no nontrivial group or orbital representation is usable.
`on` requires nontrivial usable symmetry and reports an error instead of
falling back; `off` skips detection and forces the full-grid comparison path.
A PARSEC input with `Ignore_Symmetry=true` selects `off` unless an explicit
command-line mode overrides it.

No symmetry cache is read or written unless a directory is named. A first
calculation of a structure is fastest without one: the symmetry maps take a
fraction of a second, and a process builds the packed stencil of every sector
it owns from the grid (below). `--symmetry-cache DIRECTORY` names a
persistent cache; `--no-symmetry-cache` spells the default out, and the two
together are an error. The rule is the same for `main.py`,
`python -m parsec_python`, the resident worker and
`benchmarks/mpi_full_scf.py`; `prepare_single_point` takes the directory as
`symmetry_cache_directory` and has no cache without it. Earlier versions
used `.parsec_cache/symmetry` beside the input unless `--no-symmetry-cache`
was given. No run reads that directory any more unless it is named, and none
deletes it; `--symmetry-cache .parsec_cache/symmetry`, given from the input
directory, is the former behaviour with the same keys, files and loading.
The report says which it was: `symmetry_cache_directory` is `disabled` or
the directory (`not used` in a run with symmetry off that was given one),
also in the short text report, and `timing.json` of the MPI runner has the
key as `null` or the directory. The key speaks of the persistent cache only.
The resident worker keeps reference systems and operator bundles in memory
from one request to the next whatever these flags say, so its later requests
for a structure are no first calculations although they report `disabled`;
the detailed report of a GPU run (`Output_Level: 2`) then has
`reference_static_cache = hit` and `orbital_operator_cache = memory-hit`.

A cache pays only for repeated calculations of the same structure and grid,
and what a hit spares depends on the route of the sector stencils (below).
On the former full-grid route (`PARSEC_SECTOR_STENCIL=csr`) it spares the
full-grid matrix. On the default route it exchanges builds for reads: of the
symmetry maps, the character phases, the geometry of a native Hartree
boundary, and the stencils and reduced projector factors, which a process
without a cache builds for the sectors it owns and with one reads for all
sectors. An MPI rank that owns one sector of four gains nothing by it
(figures below). A single process that owns all sectors can gain at most a
part of its preparation: for 10,456 electrons (C2455H636) on 4 GPUs, with
the GPU Hartree boundary, these builds took 1.6 s of the 6.7 s before the
first SCF step of a 69 s calculation.
Complete SCF runs on A100 nodes with the version before the sector stencils
were built from the grid, 23,768 electrons (C5659H1132) on 16 GPUs: 136.0 s
with a host high-water mark of 13.1 GiB per rank without a cache, 148.0 s
and 17.6 GiB for the run that filled one, 124.5 s and 5.4 GiB for the run
that read it; 3,480 electrons (C795H300) on 16 GPUs took 10.9, 13.4 and
8.8 s. With the stencils from the grid and the fast symmetry maps the run
without a cache takes 114.1 s and 5.4 GiB: less time than the former run
that read its cache, in the same memory. With that version, in one
allocation, the same cluster on 16 GPUs took 114.7 s without a
cache, 119.4 s for the run that filled a named one (2.2 GB on disk) and
117.6 s for the run that read it, with 7.2, 12.0 and 10.3 s before the first
SCF step; C795H300 on 16 GPUs took 8.3, 9.2 and 8.4 s; C2455H636 on 4 GPUs,
one process that owns all four sectors, took 68.5, 69.9 and 67.7 s. A rank
with one sector of four now loses time to a cache even when it hits, and a
process with all sectors gained 0.8 s in that one pair of runs. A geometry
that changes from one
calculation to the next (a relaxation, dynamics) never hits, and every miss
writes a new bundle.

In a named directory, symmetry geometry, native multipole/boundary geometry,
character phases, and representation operators are content-addressed. The
SHA-256 key covers
the complete labeled atomic geometry, exact active grid, detector tolerances,
canonical finite-difference/projector buffers, projector signs,
grid-to-wedge map, and character phases. A grid, geometry, pseudopotential,
or symmetry change therefore builds a new entry. Loaded orbit metadata is
validated before use; a missing, old, or damaged entry is rebuilt by the
exact detector.
A key only addresses its cache entry: without a cache directory (and, for the
reduced operators, without the resident bundle cache) it is not hashed and the
report lists it as `not computed (cache disabled)`.
The cache stores GPU-ready stencil-major coefficient codes and palettes rather
than duplicate reduced CSR matrices. When representation sectors have the
same neighbor topology, one host archive array and one CUDA int32 neighbor
allocation are shared across all sectors; only representation-dependent
codes, palettes, and KB factors remain separate. Native multipole geometry is
exported to its own exact-key archive, so later Python processes reconstruct
the C++ boundary builder without repeating its angular and missing-neighbor
setup.

The default symmetry/GPU path does not build the full-grid finite-difference
matrix, with or without a cache. A process builds the packed stencil of every
sector it owns from the grid lookup table, the orbit and phase maps and the
stencil coefficients (`Symmetry/sector_stencil.py`), visiting only the
representative rows of that sector. With a cache directory it builds or loads
the stencils of all sectors instead, since a cache entry is the complete
bundle. For an MPI rank that owns one sector of four the cache then no longer
pays: on a workstation, with four sectors of 1.1 million rows at order 12,
loading the entry took 0.84 s and 0.54 GiB, and building and writing it
2.75 s and 1.66 GiB for a file of 0.81 GiB, against 0.72 s and 0.22 GiB to
build the one stencil without a cache. The measured series ran without a
cache, which is the default now. The arrays are those of the former route
(full-grid CSR, `reduce_sector_csr`, audit, packing) element for element:
same neighbor order, coefficient codes and palette, repeated columns added in
the same order. That order is the one of `PARSEC_NATIVE_SECTOR_ASSEMBLY=1`,
which `scripts/run_multi_gpu_scf.sh` exports and every measured run had.
The variable is 0 when unset, and the SciPy reduction it then selects adds
three or more repeated columns of a row in another order. Against a former
run started without the variable the results are therefore not bit for bit
the same: entries of a sector near the symmetry axes and planes can differ in
the last bits (up to 41 entries per sector and at most 7e-15 on test grids of
24,000 points of orders 6 to 12, half-shifted and with points on the planes;
none on the grid of the 3,480-electron cluster C795H300), and the total energies of
small test molecules moved by 1e-13 to 1e-11 Ry. `PARSEC_SECTOR_STENCIL=csr`
restores the former route with either setting of the variable. A symmetry
cache does not tell the two orders apart: its key covers the inputs of the
stencils, not the route that summed them, so a run that reads an entry
(`orbital_operator_stencil_builder` is then `cached`) takes the numbers of the
run that wrote it. Compare a cached run bit for bit only with the run that
wrote its entry, or write the entry with the variable set to 1. The
C++/OpenMP kernel `build_sector_stencil` builds the arrays when the installed
extension has it; an extension built before the kernel existed is served by
a NumPy builder with the same result, several times slower and with a higher
host peak (see `GPU_SETUP_OPTIMIZATION.md`). Three settings of
`PARSEC_SECTOR_STENCIL` select the route; any other value is an error:

| `PARSEC_SECTOR_STENCIL` | Sector stencils |
|---|---|
| `direct` (default) | from the grid; the native kernel if present, NumPy otherwise |
| `numpy` | from the grid, NumPy only |
| `csr` | the former route: the full-grid CSR, reduced and packed |

The descriptor that stands for the matrix carries a SHA-256 provenance key
over every discrete grid and stencil input. It is hashed only where a cache
looks it up, and the reduced bundle of either route is stored under the same
key in the same format. Only CuPy execution takes the descriptor; the native
and SciPy backends read the matrix and build it as before. A symmetry
fallback, an explicit full-grid run, the pure-CuPy Poisson solver, a Hartree
solve without the orbital reduction, or a request for the full operator
(`system.reference.hamiltonian`, the component profile) materializes the same
validated C++ CSR. The report records whether this happened
(`finite_difference_full_grid_materialization`) and what packed the stencils
(`orbital_operator_stencil_builder`). In a resident process the matrix is
built by the first calculation that reads it and shared by the later ones,
which report `reused_from_resident_reference`. The largest native Hartree
coefficient table is stored as a memory-mapped NPY sidecar, eliminating one
full archive copy while retaining the same native-owned complex128 buffers.

### Resident fast-start mode

An opt-in resident worker removes repeated Python imports, CUDA context
creation, kernel-module loading, and allocator initialization across separate
calculations:

```powershell
python src\parsec_python\main.py --resident-start
python src\parsec_python\main.py INPUT --resident --no-archive
python src\parsec_python\main.py --resident-status
python src\parsec_python\main.py --resident-stop
```

Starting explicitly is optional because the first `--resident` calculation
starts the worker automatically. Requests run sequentially. Every request
reparses `parsec.in`, validates pseudopotentials and caches, and constructs
fresh physical, eigensolver, mixer, and SCF state. Only process/runtime
artifacts are retained, so this changes startup cost rather than the DFT
trajectory. The ordinary command without `--resident` remains a standalone
process. Stop the worker when its GPU resources should be released.

Canonical runs use these default filenames beside the input:

- `parsec.out`
- `parsec_python_results.npz`

Explicit `--log` and `--output` controls select other paths. Use
`--profile-repeats N` to control the average reported by
`--profile-operator`.

## Backends

| Backend | Execution strategy | Optional dependency | Intended use |
|---|---|---|---|
| `scipy` | Canonical float64 CSR/CSC operators, allocation-reduced fused host action, and fast recurrence multipoles | NumPy and SciPy | Portable accelerated baseline and numerical reference for other backends |
| `native` | Cached C++17 finite-difference construction plus OpenMP-capable fused Hamiltonian action and Poisson CG | Compiled `parsec_accelerated_native` extension | Multicore CPU acceleration without changing SCF physics |
| `cupy` | Float64 device-resident Hamiltonian, CHEBFF/CHEBDAV/SUBSPACE state, an opt-in FP32 later filter, and shared-CSR Poisson CG | CuPy build matching the installed CUDA runtime and a usable GPU | GPU eigensolver and Hartree linear algebra |
| `auto` | Accuracy-preserving hybrid when possible: native finite-difference construction and Hartree, plus a device-resident CuPy Hamiltonian and eigensolver | Native extension and CuPy for the complete hybrid; missing components fall back with recorded provenance | Fastest default execution while retaining a reproducible component-by-component fallback trail |

`auto` selects components independently. When both optional runtimes are
available, finite-difference construction and Hartree use the native C++
kernels, while Hamiltonian applications and CHEBFF/CHEBDAV/SUBSPACE use
CuPy. If only one accelerator is available, `auto` uses the compatible
combination and records each choice. Explicit `scipy`, `native`, and `cupy`
remain clean end-to-end comparison modes rather than hybrid aliases.

### SciPy

The SciPy backend retains the same sparse finite-difference matrix and
Kleinman--Bylander projector factors as `parsec_python`. For each block
`Q`, it evaluates

```text
H Q = A Q + V_eff[:, None] Q + B diag(sign(D)) (B.T Q)
```

and accumulates the local and nonlocal contributions into the CSR product
instead of allocating three full grid-by-block result arrays and two addition
outputs. Only the length-`N_grid` local potential changes between SCF
iterations.

### Native C++/OpenMP

The optional native extension canonicalizes and copies the finite-difference
CSR buffers and sparse projector factors once. Its cached fused Hamiltonian
then updates only the local diagonal field between iterations. For spherical
Hartree problems it also caches active-point angular geometry and every
missing exterior stencil neighbor. OpenMP may parallelize the compiled block
action, multipole/RHS construction, and Hartree CG loops. When
`OMP_NUM_THREADS` is unset, extension import detects the logical processors
available to the process and reserves four: for example, 32 detected gives a
28-thread native default. Set `OMP_NUM_THREADS` before launching Python to
override that policy explicitly. Repeated grid-vector kernels choose a useful
team no larger than that maximum, currently one worker per 8,192 points. This
avoids waking more threads than a symmetry wedge can feed while allowing
larger domains to scale to the configured maximum. The resident GPU worker
also defaults tiny (order at most 64) host OpenBLAS eigensolves to one thread;
an explicit `OPENBLAS_NUM_THREADS` remains authoritative.

The same extension can construct the active-domain finite-difference CSR
matrix with native compressed-grid lookup loops. It must reproduce the
reference row ordering, centered coefficients, and zero exterior orbital
boundary exactly. A missing or unloadable extension is an availability issue,
not permission to change algorithms.

The extension also caches active-grid coordinates for one-time ionic setup.
Local `r*V(r)`/spline interpolation, valence and NLCC density sampling, and
KB support/radial/real-harmonic loops run with OpenMP. POTRE parsing, KB
denominator construction, projector labels/signs, and sparse CSC assembly
remain visible in Python. A cached native CA/PZ-LDA evaluator handles the
repeated exchange-correlation grid loop while preserving the reference
float64 branch formulas. Under symmetry it evaluates one physical value per
orbit and applies integer orbit multiplicities to the energy quadrature.

### CuPy

The CuPy path uses float64 for physical operators, initial eigensolver
acceptance, Rayleigh--Ritz, densities, potentials, convergence, and energies. The finite-difference operator,
projector factors and cached transpose, local potential, Chebyshev blocks, and
saved eigensolver state remain on the device across Hamiltonian applications
and SCF iterations. Synchronization is restricted to coarse transfer and
solver timing boundaries.

Every filter runs in FP64 by default. `PARSEC_CUPY_MIXED_FILTER` is an opt-in
to an FP32 later-SCF Chebyshev recurrence: `off`, the default, builds none;
`on` builds one for every operator; `auto` builds one for operators of at
least 100,000 real-space rows (`PARSEC_CUPY_MIXED_FILTER_MIN_ROWS`) and uses
it in passes with `N_grid * N_working_states^2 >= 100000000`
(`PARSEC_CUPY_MIXED_FILTER_MIN_WORK`). The recurrence uses FP32 stencil
coefficients, projector factors, vectors, and local-potential shadows. Its
output is converted back to FP64 before the generalized Ritz projection and
every SCF/energy operation, and initial CHEBDAV/CHEBFF remains entirely FP64.
Shared symmetry-sector potentials update both the FP64 field and every enabled
FP32 shadow before a solve.

The FP32 path has not been validated above 80 states: its complete-SCF check
is Si28H36, where every printed SCF energy matched the FP64 run (see
`ACCELERATION_AUDIT.md`). Compare a complete SCF with an FP64 run before
relying on it for anything larger. On an A100, one FP32 recurrence step of a
nanodiamond sector ran 1.29 to 1.31 times faster than the FP64 step and agreed
with it to 1e-7 relative. It does not reach every route: a basis shared among
devices is always filtered in FP64, `PARSEC_CUPY_DISTRIBUTED_FILTER=1` rejects
it, it refuses a stencil packed into implicit tiles, and it writes the
filtered basis into a second `N x m` array where the FP64 filter of a
single-array sector works in place.

`orbital_sector_later_filter_precision` in the backend report of a result
says what the later passes ran in: `float64` if none of them, in any sector of
any rank, took the FP32 recurrence, and otherwise how many of how many passes
did and in which sectors. Every sector solve carries that with its record, so
the root rank of an MPI run reports the sectors of the other ranks too. The
setup report, written before the SCF, can only say what was prepared: `float64`
if no sector of the process holds an FP32 recurrence, `... prepared for
sectors ...` otherwise. Where either of the two is not `float64`, the text
log adds a line `Later filter precision as run` at the end. The entry of a
calculation without symmetry sectors, `gpu_later_subspace_filter_precision`,
is still the statement of the preparation.

The stencil of a symmetry sector is stored as affine tiles of 16 rows
(`GPU_COMPACT_OPTIMIZATION.md`) where one device filters the basis of the
sector and the sector has at least 650,000 rows
(`PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS`), or that number over the square root
of the sectors that its device filters one after another: 459,620 rows with
two sectors on a device, 325,000 with four or more. That is `PARSEC_CUPY_IMPLICIT_TILE=auto`,
the default; the kernels do the arithmetic of the slot-major stencil in the
same order, and complete calculations on one device per sector gave the same
energies to the last bit. A sector whose basis or filter is given several
devices (8 and 16 GPUs) is packed from a row count of its own, 1,100,000
(`PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS`): its owner packs the tiles
and the other devices are given its packed arrays. That row count is
assumed, not measured against others; complete calculations with a tiled
group have run on 8 and 16 A100 GPUs since.
`0` keeps the slot-major stencil everywhere, as before, and a tile size packs
every stencil. The limits of one device were measured on A100; `0` is the
setting for a device whose tile kernels do not gain, where the host packing
is paid for nothing. `auto` keeps the slot-major stencil for a sector that
is to hold the FP32 recurrence, and outside symmetry sectors. Tiles that
`auto` chose and that cannot be built leave the sector on the slot-major
stencil; a tile size named in the environment is built or is an error, and
the other devices of a group build the tiles of their owner or the
calculation stops.
`orbital_sector_finite_difference_storage` in the backend report names the
layout that was built.

Large generalized Rayleigh--Ritz overlap matrices form only the lower triangle
of `X.T @ X`, from FP64 GEMMs on trailing columns (at most
`PARSEC_CUPY_RITZ_GRAM_SLABS` column slabs, 8 by default). cuBLAS DSYRK does
as few operations, but ran at 8 to 9 Tflop/s on an A100 where these products
reach 13 to 17. Set `PARSEC_CUPY_RITZ_SYRK=on` for DSYRK or `off` for one
full GEMM. Optional
asynchronous stage timing is enabled with `PARSEC_CUPY_STAGE_TIMING=1`; it
reports first-CHEBDAV and later-SUBSPACE spectral-bound, filtering,
orthogonalization, projection, residual/locking, cleanup, Ritz-Hamiltonian,
overlap, and rotation subtotals without synchronizing every Hamiltonian
application.

The projection `X.T @ (H X)` of that solve is read in its lower triangle only
as well. Where `H X` is held as a second array beside the basis, and on every
device of a sector whose basis is spread over several devices, it is
multiplied a slab of columns at a time by the basis columns from the slab's
first one onwards: at most `PARSEC_CUPY_RITZ_GRAM_SLABS` slabs, 8 by default,
which is 56% of the single full FP64 GEMM used before. Set
`PARSEC_CUPY_RITZ_GRAM_SLABS=1` to restore that full product.

The slabs of both Gram matrices, and the rounds of a shared basis below, are
cut at whole multiples of 64 columns (`PARSEC_CUPY_RITZ_GRAM_MULTIPLE`, 64 by
default). For a tall product `A.T @ B` cuBLAS takes on an A100 the time of a
`B` whose columns are rounded up to the next multiple of 64. Timed there one
product at a time, 1,032,628 rows by 3,704 columns of `A`, a product took
0.44 to 0.48 ms per column of the rounded width at each of 15 widths of `B`
from 124 to 640 columns: 12.0, 13.9, 14.5 and 15.4 Tflop/s at 136, 272, 464
and 616 columns against 16.8 to 17.3 at 192, 256, 320, 384, 512 and 640. Over
24 first runs (3,480 to 39,368 electrons on 1 to 16 GPUs) the Gram stage ran
at 10.5 to 15.6 Tflop/s as its products were cut, a spread of 10.7%, and at
15.6 to 17.4 with a spread of 3.4% once every width is counted as rounded up.
The columns of `A` cost what they are on an A100: a step in them fits those
runs worse at every size tried (4.9% at 64 columns, 20% at 512).
So a slab of 64 columns and more is cut down to the multiple below it, and a
narrower one stays what it is:

- the overlap, an eighth of the columns per slab: 192 instead of 231 columns
  for a sector of 14,680 electrons (ten slabs instead of eight), 64 instead
  of 84 for 5,264 electrons and 576 instead of 616 for 39,368;
- the projection of the one-array route, the columns of the streaming budget
  per slab: 192 instead of 206 for 14,680 electrons, 128 instead of 130, 164
  and 186 for 5,264, 10,456 and 19,392, and 64 instead of 112 for 7,120;
- the projection of a shared basis, whose product is as wide as a round: see
  the one-block step below.

The narrower last slab of a matrix is multiplied by its own columns only, the
smallest product. A slab only narrows: the bytes of a budget stay the most
that a slab takes, the rotation tile of the one-array route keeps its rows,
and the fit rule of a shared basis counts the slabs of the budget as before,
so the same bases are shared. Both matrices are summed from other pieces:
the results differ by round-off from those of the former widths.
`PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` selects those, bit for bit, on every
route, and the serial control of the runner names it.
The rule ran on A100 nodes, the default against
`PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` in one allocation: the last table of
switches, below, has those runs. Before them its widths ran there for two cases,
set by hand with the switches that existed (first runs, one allocation, the
same code on both sides): 14,680 electrons on 4 GPUs with slabs of 192
columns (`PARSEC_CUPY_STREAMING_RITZ_BYTES=4001043456`) and ten overlap slabs
of 185 (`PARSEC_CUPY_RITZ_GRAM_SLABS=10`) took 118.5 instead of 122.2 s of
SCF, the projection stage 66.7 instead of 81.7 thread-seconds, at 42,861
instead of 43,141 MiB on the fullest device and with the total energy 2e-8 Ry
apart; 5,264 electrons on 1 GPU with slabs of 128 columns and eleven of 61
took 54.5 instead of 56.4 s, the stage 3.3 instead of 5.2 s, 9e-10 Ry apart.
That budget also shortened the rotation tile, which the rule leaves alone:
the peak of the rule is that of a multiple of 1 (43,097 MiB on both sides for
14,680 electrons on 4 GPUs, where the rule itself took 118.4 instead of 121.6
s of SCF and 64.2 instead of 81.2 thread-seconds in the projection stage).
Another device can cost otherwise. On a workstation GPU (RTX 5070, CUDA
12.9) a product took the same time at every width of `B` between two
multiples of 64, to 0.1 ms in 104 to 520, but the columns of `A` cost in
coarse steps there as well, so that more slabs do not always gain: the two
Gram matrices of one array, timed alone with slabs cut by 64 against 1, took
1.87 instead of 2.36 s at 400,000 rows by 665 columns, 6.44 instead of 6.68
s at 250,000 by 1,842, 2.51 instead of 2.28 s at 300,000 by 897 (slabs of 64
for 112 and 113) and the same at 442 columns. `1` is the setting for a
device on which the stage does not gain. A complete SCF there (864
electrons, 280,790 rows per sector, slabs of 30 columns by a budget) gave
with `1` the density, eigenvalues, energies and potentials of the code
before the rule to the last bit, and so did the default, whose slabs no
sector that small reaches; a multiple of 8 named for it moved the total
energy by 1e-10 Ry in the same nine steps.

The steps of that solve, and the placement of the sector bases and of the
GPU Hartree objects, take by default the routes that measured fastest or
smallest on A100 80 GB nodes (nanodiamond sectors of 19 to 84 GiB). Each has
a setting that selects the former one:

| Environment variable | Default | Former route |
|---|---|---|
| `PARSEC_CUPY_RITZ_SYRK` | `auto`: overlap from GEMMs on trailing columns | `on`: DSYRK |
| `PARSEC_CUPY_RITZ_DENSE_BACKEND` | `device`: the small generalized problem is solved by cuSOLVER where the Gram matrices were formed | `host`: download, host LAPACK, upload |
| `PARSEC_CUPY_RITZ_CONDITION_HELPER` | `auto`: beside that solve on the owner of a shared basis, another device of the sector estimates the condition number of the overlap, unless it holds the Hartree objects | `off`: the owner estimates it first; `any`: the Hartree device helps too, where a sector has no other |
| `PARSEC_CUPY_STREAMING_RITZ` | `auto`: a symmetry sector whose basis lies on one device keeps one `N x m` array; `H X` is projected a slab of columns at a time and the basis is rotated in place | `0`: two arrays |
| `PARSEC_CUPY_ROTATE_COLUMN_COPY` | `1`: the tiles of that rotation are written back along columns | `0` |
| `PARSEC_CUPY_DISTRIBUTED_STATE` | `auto`: a sector that was given several devices (MPI sector runs) shares its basis among them if its blocks fit | `0`: the whole basis on the sector owner |
| `PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS` | `1`: a device of a shared basis holds one block of its share in the Ritz step; its rows are laid into the block of its columns and `H X` is formed a slab of columns at a time | `2`: the rows as a second block; `3`: `H X` as a third block as well |
| `PARSEC_CUPY_DISTRIBUTED_STATE_SLABS` | `equal`: in the step with two blocks the slabs of `H X` cut a column range into equal parts, up to 6 GiB each where the device has the room (one block always cuts equal slabs, of at most 2 GiB on two devices and on four and of at most 4 GiB on three) | `full`: with two blocks, slabs of the full budget, at most 4 GiB, and a narrower last one |
| `PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT` | `interleaved`: a device of a shared basis holds its share of the lower and of the upper half of the columns, which later passes filter at different degrees | `contiguous`: one range of neighbouring columns |
| `PARSEC_CUPY_EXCHANGE_CONCURRENT`, `PARSEC_CUPY_EXCHANGE_COLUMN_COPIES` | `1`: a device of a shared basis receives from all peers at once, and copies inside a device walk down columns. On the way to their rows four devices take a chunk that all of them send from one partner per turn instead (`PARSEC_CUPY_EXCHANGE_PAIRS`, in the last table of switches below) | `0` |
| `PARSEC_HARTREE_DEVICE` | `auto`: see `Hartree/GPU_HARTREE.md` | `off` |

The overlap condition number of the small problem follows its solve: the
device solve reads the symmetric spectrum and the host solve keeps the SVD,
unless `PARSEC_CUPY_RITZ_CONDITION` names `symmetric` or `svd` for both.
That spectrum is only compared with its limit and takes about as long as the
factorization and the eigensolve after it (0.09 of 0.20 s at 2,978 states on
an A100, timed alone). On the owner of a shared basis the other devices of
the sector wait meanwhile, so the first of them that does not hold the
Hartree objects of the process copies the overlap and estimates the number
while the owner factorizes and solves; the number is judged before anything
of the solve is, so the audits, their order and their errors are unchanged
and the coefficients are the same bit for bit. `condition_device` and
`condition_seconds` of a shared sector in `timing.json` name the device of
the last estimate and the seconds that helpers spent beside the `dense`
stage. A sector on one device has no helper, and the root's sector whose
only other device holds the Hartree objects (8 GPUs) keeps the estimate on
its owner unless the setting says `any`. The helper takes memory for it: its
two copies of the overlap fit what its Gram arrays have just freed, the work
array of the spectrum does not. With CUDA 12.9 on a laptop that array is
4.05 times the overlap (274 MiB at 2,978 states), taken again in every SCF
step, and the first estimate of a thread takes 64 MiB more for its cuSOLVER
handle: about 0.33 GiB on the helper for 23,768 electrons and 0.48 GiB for
29,576, where that device peaked 1.2 and 1.7 GiB below the owner of its
sector, so the largest device of a run should stay what it was. An error of
the helper other than a failed estimate, out of memory among them, ends the
run; nothing falls back to the owner. The solve with a helper first ran where
one device stood for both, with the same bits. Sectors on several devices
have run with it on 8 and 16 A100 GPUs since (`condition_device` of their
records names the device of the estimate); the memory of a helper on an A100
has not been measured apart. The
blocks of a shared basis fit if they take at most
`PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION` of a device, 0.85 by default.
Only a CHEBFF first solve creates a shared basis. It is filtered in FP64 and
takes the generalized solve of the filtered basis (described under
saved-SUBSPACE below) in every pass. That includes every cycle of the first
solve, where one device orthonormalizes the basis unless
`PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ=1`. `auto` therefore shares only a basis
for which `PARSEC_CUPY_GENERALIZED_RITZ` and its work threshold select that
solve; `PARSEC_CUPY_DISTRIBUTED_STATE=1` shares regardless of them. A sector
whose blocks do not fit stays on its owner with one array.
A device of a shared basis holds one block of its share of the basis in the
Ritz step (`PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=1`, the default): its row
block lies in the memory of its columns. The
columns leave their block in rounds, a slab of every device per round, the
last columns first; the slabs are equal parts of every block, of at most the
bytes of the slab of the one-array route (`PARSEC_CUPY_STREAMING_RITZ_BYTES`,
by default an eighth of the sector basis but at least 1 and at most 4 GiB)
whatever `PARSEC_CUPY_DISTRIBUTED_STATE_SLABS` says, and with four or more
devices per sector, where that budget is not set, of at most an even share
of 8 GiB as well: 2 GiB with four. With two devices per sector a slab takes
at most 2 GiB too where the budget is not set
(`PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES`). The workspace and
the rotation tile have the size of the slabs that were cut. A slab is moved
into the workspace and
sent to the row layout of all devices, and the rows arrive in front of those
of the rounds before at the end of the block, in the memory that the slabs
so far have left. `H` times the slab follows the same way and is projected on the rows
that the block holds by then, one product per round as wide as the slabs of
all devices together. That width is a multiple of 64 columns wherever the
blocks allow it (`PARSEC_CUPY_RITZ_GRAM_MULTIPLE`, above): in a whole round
every device gives the same slab, the widest that makes the round a multiple
(32 or 64 columns each of two devices; 16, 32, 48 or 64 each of four) and is
no wider than the equal slabs were, as many such rounds as the narrowest
block has slabs; what the blocks hold beyond those rounds goes first, cut
into equal slabs as before, where the product is shortest. Equal slabs of
all devices keep a block as full as it was, so the rows still lie in the
block of their columns. A sector whose blocks or equal slabs are narrower
than such a slab is cut as before, and `slab_cut` says which: `multiple` or
`equal`. The summed Gram matrices are put back into the order
of the basis for the small solve, the rows are rotated in place and return
in the rounds reversed. The basis crosses the links as often as with two
blocks (its columns, `H` times them and the rows back: 2.25 times the basis
with four devices) and is copied twice more inside each device. After a
failed stability audit the rounds are reversed without the rotation, which
puts the filtered columns back bit for bit in the memory the step holds
already, and the fallback runs as before. The blocks of a trial basis are
given about one column of room for the step; a block without it, such as the
fallback leaves, gets a row block from the pool. The runner's `timing.json`
counts those per shared sector (`separate_row_blocks` under
`distributed_state`, next to `tall_blocks`, the blocks of the last step, and
`slab_cut` and `slab_columns`, the cut of its slabs and the columns of the
widest, and `projection_columns`, the columns of the widest right-hand side
of a product of its projection): where the count is not 0, a device held two
tall blocks in those passes, more than the fit rule counted. Beside the
`seconds` of the stages of a shared sector, `overlap_seconds` is the part of
`gram` that the overlap took; what is left of `gram` is the projection of
the rounds. The fit rule counts
one block, the workspace and the chunk: 26.0 GiB instead of 51.0 for the
84 GiB sector basis of 23,768 electrons on 16 GPUs and 31.1 instead of 61.2
for the 52 GiB of 19,392 electrons on 8, so that 16 GPUs share a sector basis
of up to 249 GiB instead of 117 and 8 GPUs one of up to 125 GiB, the 84 GiB
one among them (47.0 GiB), which
two blocks cannot place there (93.0). A sector on one device (four sectors on
1, 2 or 4 GPUs) shares nothing and never reads the setting; a basis that only
the rule of two blocks refused is now shared instead of staying on its owner.
First runs on A100 nodes (no symmetry cache, strict parity passed), one block
against two: 23,768 electrons on 16 GPUs peaked at 33.3 GiB per device in
135.5 s, against 55.3 GiB and 137.1 s with two blocks and slabs of the full
budget and 57.9 GiB and 133.9 s with two blocks and equal slabs, all three in
one allocation; 19,392 electrons on 8 GPUs at 37.8 instead of 65.1 GiB in
140.9 against 140.2 s; 19,392 electrons on 16 GPUs at 23.8 GiB in 88.5 s and
14,680 on 8 at 29.0 GiB in 101.6 s, where two blocks had taken 38.3 GiB and
86.4 s and 47.7 GiB and 102.2 s in another allocation; and 23,768 electrons
ran on 8 GPUs, at 54.9 GiB in 223.9 s. No pass of these runs took a row block
from the pool. Over the ten SCF steps the Ritz stages of a sector took 0.5 s
(19,392 electrons on 8 GPUs) and 2 s (23,768 on 16) longer than with two
blocks and equal slabs, and the total energies differ from those of two
blocks by 3e-8 to 5e-8 Ry. That is why one block is the default. Those runs
used the step as it was first written. As it is now, named `1` beside the
other defaults of this version in one allocation, it ran 23,768 electrons on
16 GPUs at 33.2 GiB in 114.1 and 114.4 s, where two blocks took 57.9 GiB and
113.1 s with equal slabs and 55.2 GiB and 116.6 s with full ones; 10,456
electrons on 8 GPUs at 16.5 GiB in 45.9 s against 26.4 GiB and 46.4 s; 3,480
and 14,680 electrons on 16 GPUs at 2.5 GiB in 8.1 s and 18.8 GiB in 53.9 s;
and 19,392 and 23,768 on 8 at 37.7 GiB in 119.0 s and 54.8 GiB in 195.7 s.
None of the 48 shared sectors of the runs with one block took a row block
from the pool, and the energies again differ from those of two blocks by 3e-8
to 5e-8 Ry. All of these runs named the block count. The default itself, the
variable unset, has run since: in one series of first runs, 3,480 to
29,576 electrons on 8 GPUs and 3,480 to 39,368 on 16,
no rank named the block count or a slab budget, and each of the 68
shared sectors recorded one tall block and no row block from the pool.
The runs that named it all cut slabs of at most 4 GiB, with which the rule
counted 30.0 GiB for the 84 GiB basis on 16 GPUs and shared up to 233 GiB. A
round projects
the slabs of all devices of a sector in one product, so four devices reach
the same width with slabs half as large, and the workspace is two slabs. With
the budget set to 2 GiB, first runs on 16 GPUs peaked at 30.1 instead of 33.2
GiB per device for 23,768 electrons (rounds of about 271 instead of 496
columns) and at 37.9 instead of 41.3 GiB for 29,576 electrons (247 instead of
463). The time moved in both directions. The eigensolver of the first run
took 94.2 s where six runs with 4 GiB took 93.1 to 93.2 s, and its SCF 108.2
instead of 107.1 to 107.4 s: about 1% more, in the Gram stage (16.2 against
15.6 s per sector) and on the way back to the columns (4.0 against 3.7 s).
That of the second took 137.9 instead of 140.2 s, and its SCF 152.7 instead
of 155.0 to 155.5 s: about 1.5% less, the Gram stage 26.6 against 29.3 s. The
program totals, 115.1 against 114.7 s and 161.9 against 164.3 s, also hold
the setup, which varies by 0.4 s between such runs. The total energies differ
from those of the 4 GiB runs by 6e-8 and 2e-8 Ry; the first passed strict
parity, the second was compared with no reference. That is the default of a
sector on four devices now. The default itself, the budget unset, has run
since, 3,480 to 39,368 electrons on 16 GPUs in that series (above). It also
changed the cut of 14,680 and 19,392 electrons on 16 GPUs (rounds of at most
371 and 349 columns instead of 614 and 609, a workspace of 3.6 and 3.8
instead of 6.0 and 6.6 GiB) and of 39,368 electrons (184 instead of 354
columns, 3.9 instead of 7.5 GiB, for the basis that ran: 5,660,200 rows and
4,928 columns). Of these three, 19,392 electrons ran with it beside slabs of
4 GiB in one allocation, the budget the only difference: SCF 65.8 against
66.0 s and 21,515 against 24,371 MiB on the fullest device, before the
Hartree boundary was brought in. The other two ran with it and not beside
slabs of 4 GiB alone: which way the cut by itself moves their time is not
known. Sectors of 10,456 electrons and fewer are cut by the budget as
before. Those are the rounds of `PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1`: the
default cuts them further, at whole multiples of 64 columns (below), into
rounds of 320, 320 and 128 columns for the three.
A sector on two devices (8 GPUs) was left at 4 GiB at first: with 2 GiB its
rounds are about 135 columns wide for 23,768 electrons, which had not run,
and the products of the two-block step were slower at 90 to 141 columns (Gram
stage 17.5 s) than at 184 to 189 (14.5 s). First runs on 8 GPUs with the
budget set to 2 GiB, beside runs with 4 GiB in one allocation, peaked at 49.7
instead of 53.5 GiB per device for 23,768 electrons (22 rounds of at most 136
columns instead of 11 of 271) and at 65.5 instead of 69.2 GiB for 29,576
electrons (29 rounds of 128 instead of 15 of 248). The time moved in both
directions here too: the first took 181.8 instead of 179.7 s, its Gram stage
35.5 to 36.0 instead of 33.2 to 33.4 s per sector, and the second 256.2
instead of 258.7 s, 52.1 to 52.7 instead of 55.1 to 55.3 s. A budget of 3 GiB
gave 51.5 and 67.2 GiB in 181.4 and 258.3 s. All six runs converged in ten
steps, and the total energies of 2 and 4 GiB differ by 4e-8 and 3e-8 Ry. So
slabs of at most 2 GiB are the default of a sector on two devices as well
(`PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES`, 2147483648 unless it is
set): the fit rule counts 47.0 instead of 51.0 GiB for the 84 GiB basis and
62.0 instead of 66.0 for the 114 GiB one of 29,576 electrons, and two devices
share a basis of up to 125 GiB instead of 117. The default itself, the budget
unset, has run on 8 GPUs since, for these two and for 14,680 and 19,392
electrons, whose cut it changes as well (rounds of at most 205 and 174
columns instead of 369 and 348, a workspace of 4.0 and 3.7 instead of 7.2
and 7.5 GiB): the last table of switches, below, has the peaks.
The limit is the same for a basis of any size. Up to 16 GiB a
slab is the eighth of the basis that it was; from there to 32 GiB it took
that eighth and now takes 2 GiB, in five to nine rounds instead of four or
five, where no cluster has run. The sectors of 10,456 electrons on 8 GPUs
(18.8 GiB) keep their five rounds of 132 columns, and those of smaller
clusters their cut: their results stay what they were bit for bit. A limit
that began at 32 GiB, where a slab took the full 4 GiB, was tried first and
dropped: a basis just below held a workspace of nearly 8 GiB and one just
above 4, so the few columns that a trim takes from a sector after its first
solve could double the workspace that the fit rule, which is asked for the
first solve, had counted for it. With one limit a slab never narrows as
columns are added. A sector on three devices keeps slabs of 4 GiB at every
size: none has run.
`PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=4294967296` gives a sector on
two devices (8 GPUs) the slabs it had, whatever the size of its basis and in
every step of a run: the limit is then that of the budget itself.
`PARSEC_CUPY_STREAMING_RITZ_BYTES=4294967296` does so for a sector on four or
more devices (16 GPUs), and it is the way back for those runs only: a sector
on three devices or on one has its former slabs without either, and with that
budget a basis below 32 GiB gets slabs of 4 GiB where it took an eighth of
itself, on two or three devices and in the one-array route of a sector on
one. For a sector on two devices it gives back the slabs of the steps in
which the basis holds 32 GiB or more, as in the runs above, and no others: not
those of a smaller basis, and not those of a sector once a trim has taken it
below 32 GiB.
Those two settings give back the columns that a slab may have. The rounds are
cut from them at whole multiples of 64 columns since
(`PARSEC_CUPY_RITZ_GRAM_MULTIPLE`): the rounds that a run had, and with them
its bits, take `PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` beside either. With the
slabs of 2 GiB the rounds of the shared sectors of the measured series
become, on 8 GPUs, nine of 192 columns after one of 114 for nine of 205
(14,680 electrons), eighteen of 128 after one of 127 for fourteen of 174
(19,392), 23 of 128 after one of 34 for 22 of 136 (23,768) and 28 of 128
after one of 120 for 29 of 126 to 128 (29,576); on 16 GPUs five of 320 after
one of 242 for five of 371 (14,680), seven of 320 after one of 191 for seven
of 349 (19,392), eleven of 256 after one of 162 for eleven of 273 (23,768),
nineteen of 192 after one of 56 for fifteen of 248 (29,576) and 38 of 128
after one of 64 for 27 of 184 (39,368). The workspace of the slabs that are
cut falls with them, by 0.3 to 1.2 GiB per device on 16 GPUs and by 0.2 to
0.9 on 8, where 29,576 electrons keep theirs (derived from the cut). All of
these cuts but that of 19,392 electrons on 16 GPUs have run on A100 nodes
since, in first runs against `PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` (the last
table of switches, below): their records read `multiple`, with slabs of 96, 66,
64 and 64 columns on 8 GPUs and of 80, 64, 48 and 32 on 16, and the fullest
device held 0.2 to 1.2 GiB less, the same for 29,576 electrons on 8 GPUs.
More rounds take time in the stages around the product. In the runs that
halved the slabs (above) a round more took 1.2 and 1.3 ms per pass on two
devices and 6.1 to 8.3 ms on four, `apply`, `to_rows` and `to_columns`
together. Held against that, the rounds of 29,576 and 39,368 electrons on 16
GPUs, 20 for 15 and 39 for 27, were to keep 0.1 to 0.2 and 0.6 to 0.9 s per
sector of the 0.6 and 1.9 s that the rounded widths give them (derived),
beside 1.4 and 1.6 s from the overlap. Measured in those first runs, one
pair each: the overlap of the two sectors took 2.1 and 2.4 s less and the
products of their rounds 0.8 and 3.2 s less, while `apply`, `to_rows`,
`to_columns` and `rotate` together took 0.5 and 5.0 s more, 3.3 s of the
latter on the way back to the columns. So the rounds of 29,576 electrons
kept 0.3 s per sector and those of 39,368 lost 1.9 s, less than their
overlap gained (SCF 275.3 against 275.8 s). On 8 GPUs the rounds of 23,768
electrons kept 6.6 s per sector and those of 14,680 and 19,392 1.2 s each.
Those runs received every chunk from all its senders at once. With the turns
of four devices (`PARSEC_CUPY_EXCHANGE_PAIRS`, below) the whole rounds have
run on 16 GPUs since, with all these switches together, beside a run with
every switch of the last table at its former value in the same allocation: for 39,368
electrons the four stages together took 81.4 against 84.3 s per sector, the
products of the rounds 26.4 against 30.0 s and the overlap 31.4 against 33.8
s. Equal slabs in turns took 79.2 s in those stages in another allocation,
2.2 s less than the whole rounds: 0.9 s on the way to the rows, 0.5 s each on
the way back and in the rotation, 0.2 s in `apply`. So with the turns the
rounds of 39,368 electrons kept about 1.3 s per sector. The way back took
10.6 s there, where the run without turns had 13.3 s and seven runs with
equal slabs 10.0 to 10.1 s. For 23,768 electrons the stages took 24.6 s
(24.9 with equal slabs in turns) and the products of the rounds 6.9 against
8.5 s. And the
slab of a whole round can be little more than half of the widest equal one:
a sector of 3,714 columns on two devices, ten more than that of 29,576
electrons, is cut into 58 rounds of 64 columns after one of 2 where it had 30
of 122 to 124, for 3 to 5% of its projection (derived) and half the
workspace. That cut has not been measured, and no product narrower than 124
columns was timed alone. `overlap_seconds` separates the parts: in a run
against `PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` the overlap shows what its slabs
gave, what is left of `gram` what the rounds gave, and `apply`, `to_rows`,
`to_columns` and `rotate`, whose tile follows the slab, what they cost.
`PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=2` selects the former step, in which a
device holds two blocks of its share of the basis: its columns and its rows
of all columns. The results of the two steps differ by round-off. `H X` is
never stored as a whole: each device applies `H` to a slab of its columns, the
slabs go to the row layout through the exchange buffers and are projected
there, and the rows are rotated in place. Next to the two blocks a device
holds a workspace of two slabs and one exchange chunk. A slab takes at most
the bytes of the slab of the one-array route
(`PARSEC_CUPY_STREAMING_RITZ_BYTES`, by default an eighth of the sector basis
but at least 1 and at most 4 GiB) and at most half of a block. Every column
range of a device is cut into as few slabs of that width as it takes, all of
them equal, and the workspace has the size of the slabs that were cut. Where
the budget is not set and the blocks fit a device with a workspace of two
slabs of 6 GiB, by the rule above, that is the upper limit instead of 4 GiB.
Every such step asks that rule, also of a basis that
`PARSEC_CUPY_DISTRIBUTED_STATE=1` shares without it:
`PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION` then still decides the width
of the slabs, and a mistyped value stops the run where the trial basis is
sized. The rule counts the memory of the device and knows no limit of the
memory pool: a run under `CUPY_GPU_MEMORY_LIMIT` names the former cut or
sets the budget.
The former cut, `PARSEC_CUPY_DISTRIBUTED_STATE_SLABS=full`, takes slabs of
the full budget and leaves what remains of a range as a last one: the ranges
of about 372 columns of 23,768 electrons on 16 GPUs became 141, 141 and 90,
and a run with the budget set to slabs of 189 columns, which cut them into
two, took 133.1 instead of 135.8 s for 2.7 GiB more. The equal cut gives
those ranges two slabs of 184 to 189 columns (workspace 10.6 instead of 8.0
GiB) and the ranges of about 304 columns of 19,392 electrons on 16 GPUs two
of about 152 instead of 186 and 118 (6.5 instead of 8.0 GiB). The results of
the two cuts differ by round-off: the projection is summed over other slabs,
and the rows are rotated in tiles of another height, a tile being as tall as
a slab of the workspace lets it be. In first runs of one allocation the equal
cut took 133.9 instead of 137.1 s for 23,768 electrons on 16 GPUs, at 57.9
instead of 55.3 GiB, and beside the other defaults of this version 113.1
instead of 116.6 s at 57.9 instead of 55.2 GiB; apart from 19,392 electrons
on 8 GPUs (140.2 s, 65.1 GiB, no run of the former cut beside it) its effect
on the smaller sectors has not been measured.
`PARSEC_CUPY_DISTRIBUTED_STATE_SLABS` acts on this step only: named beside
the default of one block it changes nothing, and `slab_cut` of the run reads
`equal` or `multiple`. A comparison of the two cuts names
`PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=2` on both sides.
In this step every slab is the right-hand side of a product of its own, so a
slab of 64 columns and more is cut down to a whole multiple here too, the
full slab of the budget or the widest of the equal ones, and a range ends
with a narrower one: ranges of 372 columns in two slabs of 186 become 128,
128 and 116 (`PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` for the cuts as they were).
That has not run on a GPU.
`PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=3` selects
the step before that, which holds `H` times the columns as a third block; the
results of the two differ by round-off. For 23,768 electrons on 16 GPUs two
blocks peaked at 55.3 GiB instead of 68.3 in 140.7 against 138.9 s, and
19,392 electrons on 8 GPUs took 143.6 s with a shared basis that three
blocks cannot hold there (193.2 s on the sector owners). The fit rule
counts what the setting selects. It keeps no room for the fallback after a
failed stability audit, which gathers the whole basis on the sector owner
next to the owner's columns, `1 + 1/devices` times the basis. A basis can
therefore be shared although that fallback could not run, with two blocks
on two devices as well: the 52.2 GiB per sector of that run on 8 GPUs fit,
and the fallback would need 78.3 GiB of a device of 79.25. One block shares
bases that no device could gather at all.
The columns of a device of a shared basis are two ranges: its share of the
lower and of the upper half of the columns. Later passes filter the lower
half at a lower degree than the upper half. With one range of neighbouring
columns per device, the former layout, half of the devices then filter at
the higher degree only and the others wait for them; with a share of each
half all of them do the same work in every pass, and no column moves and no
block grows. For 19,392 electrons on 16 GPUs the later passes filtered a
sector in 16.4 instead of 19.5 s and the program took 86.3 instead of 89.3
s; 23,768 electrons took 135.8 instead of 140.0 s, and 19,392 electrons on
8 GPUs 138.2 instead of 144.0 s, each at the same peak memory.
`PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT=contiguous` selects the former
layout. The trial basis is created in either layout from the same random
stream, and the rows of the Ritz step hold the columns in their own order,
so the results of the two differ by round-off only. A sector with fewer
filter blocks in a half than it has devices keeps contiguous ranges. So does
`PARSEC_CUPY_DISTRIBUTED_STATE_RANGES=balanced`, which moves columns between
the devices from pass to pass instead and takes headroom in every block for
it; `interleaved` named together with it is an error.
On the way to the rows all devices of a sector begin to receive a chunk at
the same moment, each from every other one: twelve copies at once with four
devices per sector (16 GPUs). On the four A100 of one node, pieces of 0.49
GiB issued so arrived at 58.9 GiB/s per device (17.3, 25.1 and 25.1 ms per
chunk in three repeats), and in three turns of two pairs, both directions of
a pair at once, at 83.8 GiB/s (17.6 ms each time). The stage seconds of
first runs say the same of the program from 14,680 electrons on: 50 to 53
GiB/s per device on that way with four devices per sector, and 80 to 82
with two (8 GPUs), which are one pair. The four devices of a sector
therefore take a chunk that all of them send in those three turns by
default. In a turn a device receives from one partner, which receives from
it, on the stream it computes on, and all four are done with a turn before
the next begins: two more waits for all devices per chunk, which earlier
runs put at about 0.3 ms each. `PARSEC_CUPY_EXCHANGE_PAIRS=0` issues the
twelve together as before. The bytes that arrive are the same and so are
the results, bit for bit. Once a
device has sent all its columns, the chunks that the others still send are
received as before, from all their senders at once. They have not been
timed in turns: with one sender a turn would be a single copy where three
started together, and a receiver alone took three sources side by side at
three times the rate of one (262.5 against 87.5 GiB/s). The step with one
tall block, the default, gives all devices equally many slabs, and every
chunk of the bases of 3,480 to 39,368 electrons on 16 GPUs has four senders:
with the equal slabs that the turns were timed with, and with the whole
rounds of `PARSEC_CUPY_RITZ_GRAM_MULTIPLE`, in which every device gives the
same slab (derived from the column ranges of those bases). The whole rounds
are more and their pieces smaller, 8 to 154 chunk steps per pass where the
equal cut had 6 to 108. For three of the bases the slab of a whole round is
just wider than a chunk of 1 GiB (`PARSEC_CUPY_EXCHANGE_CHUNK_BYTES`), which
holds 31 columns of 39,368 electrons, 43 of 29,576 and 92 of 10,456 where
the slabs have 32, 48 and 96: every such slab travels as a chunk and a
second one of 1, 5 and 4 columns, which are 76 of the 154, 38 of the 78 and
6 of the 14 chunk steps of a pass, each with its three turns (derived). The
turns have run with the whole rounds on 16 GPUs for 3,480, 23,768 and 39,368
electrons, with all these switches together: the way to the rows of a sector took 0.36,
6.68 and 17.38 s, where equal slabs took 0.39, 6.78 and 16.50 s in turns
(another allocation) and 0.41, 9.05 and 21.62 s received at once (the same
one). So the turns keep their gain with the whole rounds, less about 0.9 s
per sector for 39,368 electrons, whose way has 46 chunk steps more per pass.
A chunk of 1,500,000,000 bytes holds a slab of every whole round of these
bases: 78, 40 and 8 chunk steps per pass and none of fewer than 18 columns,
for 0.4 GiB more in the exchange buffer of every device, which the rule that
shares a basis counts (0.724 instead of 0.719 of a device for 39,368
electrons). That chunk has run with the whole rounds on 16 GPUs beside the
default, in one allocation (23,768, 29,576 and 39,368 electrons,
totals equal to the last bit): the way to the rows of a sector took 6.36,
8.73 and 16.04 s against 6.64, 9.32 and 17.38 s, the way back 4.28, 5.84 and
10.65 s against 3.91, 5.56 and 10.56 s, the SCF 93.2, 132.3 and 266.1 s
against 92.9, 132.9 and 267.5 s, and the fullest GPU held 408 MiB more in
each. That is at most 0.5 % of the SCF for 0.4 GiB of every device, so the
default chunk stays at 1 GiB. With the equal slabs of
`PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` the larger chunk would be the one just
short of a slab (46 columns against 44 for 39,368 electrons).
A block that is a whole number of such slabs gives nothing to the round that
goes first, whose chunks then have fewer senders and go together. The slabs
of the step with two blocks end with
their column range: there 23 % of the bytes of 10,456 electrons on 16 GPUs
travel in chunks of one to three senders, as many for 14,680 where its slabs
are cut wider, 6 to 9 % for 3,480 to 7,120 electrons, and none or next to
none from 19,392 on. `to_rows_copies` of a shared sector in `timing.json`
names the order of its last way to the rows: `pairs` (the chunks that all
four send in turns), `together` or, under
`PARSEC_CUPY_EXCHANGE_CONCURRENT=0`, `queued`, which takes no turns. A
sector on two devices is unchanged, and so is one on three or on more than
four, which have not been timed. The way back is unchanged too: there every
device fetches chunk after chunk at its own pace, with no wait
for the others, and the stage seconds give its four devices 75 to 81 GiB/s
each once the cost of a round is taken out (61 if none is); turns would add
three waits per chunk to it, about what they could save. At 72 to 84 GiB/s
on the way to the rows, less the waits, the program would take 1.7 to 2.5 s
less of 106.9 for 23,768 electrons on 16 GPUs and 4.1 to 6.1 s less of 290.8
for 39,368. That was derived; in first runs since, the SCF took 2.0 s less
of 99.6 for 23,768 electrons and 5.6 s less of 279.6 for 39,368 (the last
table of switches, below). For the small bases the sign was
open, and the turns are taken there too. Their stage is not limited by the
copies: its seconds give 22 to 31 GiB/s per device for 3,480 electrons on 16
GPUs (pieces of 0.05 GiB, six chunks per pass, 0.38 to 0.52 s per sector of
a program of 8.2 s), 36 to 40 for 5,264, 39 to 44 for 7,120 and 46 to 49 for
10,456, and on 8 GPUs, where a sector is one pair, 51 to 55, 61 to 65, 59 to
66 and 75 to 79 against 80 to 82 from 14,680 on. At the two rates measured
for pieces of 0.49 GiB the turns can save 3,480 electrons 0.06 s per run.
Their 156 more waits cost 0.02 to 0.05 s at 0.12 to 0.35 ms each, and 0.17 s
at the 1.1 ms per wait that the stage seconds leave room for: between 0.04
s less and 0.11 s more per run, 1.4 % of the program. The same arithmetic
gives 5,264 electrons between 0.11 s less and 0.02 s more, 7,120 electrons
0.01 to 0.18 s less and 10,456 electrons 0.16 to 0.46 s less of 28.3. No
piece size but 0.49 GiB has been timed in turns. The turns have run on the
host stand-in of the exchange and, with the one GPU of a laptop named four
times as the devices of a sector, on device memory: the Ritz steps with
one, two and three blocks gave the same bits with them as without. Every
copy stays inside the device there. Sectors on four devices have run with
the turns since, 3,480 to 39,368 electrons on 16 GPUs, default against `0` in
one allocation: the same bits, and the times in the last table of switches.
`PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION`, 0 by default, returns to two
arrays on a device where they take no more than that fraction of its memory
(0.8 was the rule before one array became the default). The one-array route
measured the same time as two arrays with 24.2 instead of 40.7 GiB for 10,456
electrons on four GPUs. For 19,392 electrons on 16 GPUs the device solve cut
the small solve from 15.5 to 1.9 s, and the exchanges of the shared basis,
chunked and with both switches on, took 7.1 instead of 17.3 s. The
`--serial-control` mode of
`benchmarks/mpi_full_scf.py` keeps the former routes wherever its launcher
names none, so that it stays an independent reference.

Set-up and filter routes that became defaults after first runs (no symmetry
cache) on the same nodes, each with the setting that selects the former one:

| Environment variable | Default | Former route | Described in |
|---|---|---|---|
| `PARSEC_CUPY_MIXED_FILTER` | `off`: every filter in FP64 | `auto`: FP32 later filter for large sectors | above |
| `PARSEC_CUPY_IMPLICIT_TILE` | `auto`: affine tiles of 16 rows for a sector of at least `PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS` (650,000) rows that one device filters, of that over the square root of the sectors of the device where it filters several | `0`: the slot-major stencil; `PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS` at 1,100,000, 1,555,634 or 2,200,000 for one, two or four sectors on every device (4, 2 or 1 GPUs): the former limit of 1,100,000 rows | above, `GPU_COMPACT_OPTIMIZATION.md` |
| `PARSEC_CUPY_IMPLICIT_PACK_WORKERS` | `4`: threads that pack the chunks of a tiled stencil | `1` | `GPU_COMPACT_OPTIMIZATION.md` |
| `PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS` | `1100000`: under `PARSEC_CUPY_IMPLICIT_TILE=auto`, affine tiles also for a sector of at least that many rows whose basis or filter several devices share (8 and 16 GPUs); the owner packs them and the others read its arrays | `PARSEC_CUPY_IMPLICIT_TILE=0`: the slot-major stencil on every device | above, `GPU_COMPACT_OPTIMIZATION.md` |
| `PARSEC_IONIC_GPU_COUNT` | `auto`: the GPU ionic sums run on every device of an MPI rank that solves symmetry sectors, on one device otherwise | `1` | `GPU_SETUP_OPTIMIZATION.md` |
| `PARSEC_CUPY_FILTER_GRAPH_REUSE` | `1`: one recorded filter graph per block width and degree, kept from plan to plan | `0`: one per block of a plan | `MULTI_GPU.md` |
| `PARSEC_SECTOR_STENCIL` | `direct`: sector stencils from the grid | `csr`: reduced from the full-grid matrix | above |
| `PARSEC_OVERLAP_IONIC_SETUP` | `1`: the GPU ionic fields on a thread beside symmetry and sector set-up | `0`: in line | `GPU_SETUP_OPTIMIZATION.md` |
| `PARSEC_OVERLAP_CUDA_CONTEXTS` | `1`: `benchmarks/mpi_full_scf.py` creates the CUDA contexts of a rank on a thread beside its preparation | `0`: before it | `GPU_SETUP_OPTIMIZATION.md` |
| `PARSEC_SYMMETRY_FAST_MAPS` | `1`: symmetry detection and phases without passes over the grid coordinates | `0` | `GPU_SETUP_OPTIMIZATION.md` |

What was compared on A100 nodes, in first runs with these routes together
(strict parity passed, the filter in FP64 on both sides): all of them at
their defaults against the former route of each gave the same total energy
to the last bit, and the same differences of eigenvalues, density,
potentials and SCF history from the serial reference, for 3,480, 7,120,
10,456, 14,680 and 19,392 electrons on 4 GPUs, where no sector basis is
shared. So did each of these switched back alone, for 10,456 electrons on 4
GPUs and 23,768 on 16: `PARSEC_CUPY_FILTER_GRAPH_REUSE=0`,
`PARSEC_SECTOR_STENCIL=csr` (under `PARSEC_NATIVE_SECTOR_ASSEMBLY=1`),
`PARSEC_IONIC_GPU_COUNT=1`, and the three switches of the preparation
together; `PARSEC_CUPY_IMPLICIT_TILE=0` for 10,456 and 19,392 electrons on
4. With a shared basis the result follows its Ritz step (round-off between
the block counts, see above): 23,768 electrons on 16 GPUs with two blocks
and full slabs named beside all these defaults gave the energy of the
former routes to the last bit. Not run apart: the three switches of the
preparation from each other, and `PARSEC_CUPY_IMPLICIT_PACK_WORKERS` from
the tiles; the tests compare what they build. The FP32 filter
(`PARSEC_CUPY_MIXED_FILTER=auto`) is another precision, not round-off, and
was not run in these comparisons. `PARSEC_SECTOR_STENCIL=csr` without
`PARSEC_NATIVE_SECTOR_ASSEMBLY=1` differs by round-off, which was measured
on small test cases only (above).
The serial control of the runner keeps the former route of each except the
FP32 filter. A symmetry cache is outside that: an entry that another run
wrote gives the control the maps and stencils of that run, and its log says
so; a control that names no cache, the default, stays independent. The
backend report says
what ran: `orbital_sector_later_filter_precision`,
`orbital_sector_finite_difference_storage` with
`orbital_sector_tile_pack_workers`, `ionic_gpu_count_requested` with
`ionic_gpu_devices`, `orbital_sector_filter_graphs`,
`orbital_operator_stencil_builder`, `ionic_setup` and `symmetry_fast_maps`;
the runner records `cuda_context_initialization_overlapped` for every rank,
and `cuda_context_driver_devices` with `cuda_context_driver_seconds` for the
contexts that its thread created through the driver library. That creation,
the rule of the state storage and the download of a spill came after these
runs: they are in the last table of switches, below. The measured
runs named `device` and `pool`, the route that rule now takes by itself where
the sectors fit; `orbital_sector_state_storage` and `orbital_memory_allocator`
say what a run had.
`orbital_sector_filter_graphs` is the switch while a calculation is
prepared. In a result it is what the sector filters of the reporting
process recorded: `none recorded` unless `PARSEC_CUPY_FILTER_GRAPHS` or
`PARSEC_CUPY_DISTRIBUTED_FILTER` is set or a sector basis is shared.
`scripts/run_multi_gpu_scf.sh` and every measured run set the first, which
is off when unset. The text report adds a line where its detailed setup
named another route, and the runner counts the captures of every rank
(`graph_captures`).
`PARSEC_CUPY_FILTER_GRAPH_REUSE` and `PARSEC_SECTOR_STENCIL` are read when the
preparation starts, so a value they do not know stops the run there.

A further set of switches, in one table: each with its default, the
setting that selects the former route, what it does to the results and the
entry that says which one ran. The first eight keep the results, bit for bit
or at round-off. Each of them has run on A100 nodes in the code that
introduced it, switched on and off in one allocation, and all eight together
in the code as it was before the Hartree boundary was brought in:
first runs of 3,480 to 29,576 electrons on 1 to 16 GPUs, every item at its
default against every item on its former route, and each of the 59 runs that
have a reference passed strict parity. The two sides gave the same total
energy to the last bit on 1, 2, 4 and 8 GPUs and, up to 10,456 electrons, on
16; from 14,680 electrons on 16 GPUs the slabs of 2 GiB change the slab count
of the Ritz step and the totals differ by 1.3e-8 to 6.0e-8 Ry. (A sector on
two devices has cut slabs of at most 2 GiB since: on 8 GPUs the two sides
now differ at round-off as well, from 14,680 electrons.) For 23,768
electrons on 16 GPUs the defaults took 105.5 s and 30,421 MiB on the fullest
device where the former routes took 118.2 s and 34,013 MiB; for 29,576
electrons 147.3 s and 38,453 MiB against 162.7 s and 42,257 MiB. The switches
of the Hartree boundary change results by design where its plan engages. They
had not run on an A100 when these runs were made, nor had the code with
them; both have run there since (`result.hartree_boundary` in the
`timing.json` of a run says what it had).

| Environment variable | Default | Former route | Results | Says what ran | Described in |
|---|---|---|---|---|---|
| `PARSEC_CUPY_STREAMING_RITZ_BYTES` | unset: slabs of at most 2 GiB in the one-block Ritz step of a sector on four or more devices (16 GPUs) | `4294967296`, for 16 GPUs only; the rounds a run had take `PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` beside it | round-off | `slab_columns` of a shared sector in `timing.json` | above |
| `PARSEC_CUPY_RITZ_CONDITION_HELPER` | `auto`: another device of a shared sector estimates the condition number of the overlap beside the small solve of its owner, unless it holds the Hartree objects | `off` | same bits | `condition_device` and `condition_seconds` of a shared sector in `timing.json` | its row in the first table above |
| `PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS` | `650000`, over the square root of the sectors of the device: the rows from which a sector alone on its device packs affine tiles | `1100000`, `1555634` or `2200000` on 4, 2 or 1 GPUs | same bits | `orbital_sector_finite_difference_storage` | the row of `PARSEC_CUPY_IMPLICIT_TILE` above |
| `PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS` | `1100000`: the rows from which a sector on several devices (8 and 16 GPUs) packs them | `PARSEC_CUPY_IMPLICIT_TILE=0` | same bits | `stencil_storage` of a shared sector in `timing.json` | its row above |
| `PARSEC_CUPY_SECTOR_POOL_RELEASE` | `1`: a finished sector returns the unused pool blocks of its stream where a device solves several sectors in turn (2 GPUs) | `0` | same bits | `sector_pool_release` of a rank in `timing.json` | `MULTI_GPU.md` |
| `PARSEC_CUPY_DENSITY_COLLECTION` | `changed`: the young generations are collected before the pool release after a density while the bytes in use stay | `full` | same bits | `density_pool_release` of a rank in `timing.json` | `MULTI_GPU.md` |
| `PARSEC_SYMMETRY_SCF_BUFFERS` | `1`: the wedge sums and the Anderson step of the root in arrays it keeps | `0` | same bits | `scf_scalar_field_buffers` | below |
| `PARSEC_FAST_GRID` | `1`: the cluster grid slab by slab | `0` | same bits | `grid_builder` | `GPU_SETUP_OPTIMIZATION.md` |
| `PARSEC_HARTREE_BOUNDARY` | `auto`: the Hartree boundary values follow the plan made from the geometry (estimate of the omitted potential, order raised from `Solver_Lpole` until the estimate meets the tolerance, atomic tail; see `Hartree/GPU_HARTREE.md`). This changes results of clusters from about 70 atoms up | `legacy`: PARSEC's multipole expansion at `Solver_Lpole`, bitwise the former boundary | changed by design where the plan engages; `legacy` gives the former bits | `hartree_boundary`, `hartree_multipole_order`, `hartree_boundary_tolerance`, `hartree_boundary_estimate`, `hartree_atomic_tail`, `hartree_boundary_plan_seconds`; `result.hartree_boundary` in `timing.json` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_BOUNDARY_TOLERANCE`, `PARSEC_HARTREE_ATOMIC_TAIL` | unset: the input keywords `Hartree_Boundary_Tolerance` (`1e-3 Ry`) and `Hartree_Atomic_Tail` (`auto`) | `off` and `off` | changed by design | `hartree_boundary_tolerance`; `hartree_atomic_tail` with `hartree_atomic_tail_max` and `hartree_atomic_tail_seconds` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_LPOLE` | unset: the input keyword `Solver_Lpole` (0 to 60, default 9; orders above 9 need native extension 0.6.0 or newer) |  | changed by design | `hartree_multipole_order` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_BOUNDARY_KERNEL` | `auto`: the GPU boundary of an engaged plan is evaluated once per unique exterior point, on the wedge of an axis-reflection group or, without one, on the full grid (round-off change against the full-grid kernels; see `Hartree/GPU_HARTREE.md`). On the laptop GPU the raised order costs less than the former boundary on the D2 wedge and, without such a group, up to about twice as much at order 34 | `full`: the full-grid kernels (the serial control of the MPI runner takes them) | round-off | `hartree_boundary_kernel`, `hartree_boundary_device_bytes` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_BOUNDARY_BACKEND` | unset: the native builders; where the plan is engaged and they have no orbit table (no reduction, or the table of the raised order above 512 MiB) the GPU kernels if the orbital backend is CuPy. `cupy`: the GPU boundary. `auto`: see `GPU_SFC_OPTIMIZATION.md` | `native`: the native builders whatever they cost | round-off where the GPU kernels take the boundary | `hartree_boundary_kernel`, `hartree_backend` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_ATOMIC_TAIL_VALUES` | `auto`: the values of the atomic tail at the exterior points come from an FP64 device kernel where the orbital backend is CuPy and from host threads elsewhere | `host`: host threads on every backend (the serial control of the MPI runner takes them); there is no former route, the tail is new | round-off | `hartree_atomic_tail_values` | `Hartree/GPU_HARTREE.md` |
| `PARSEC_HARTREE_BOUNDARY_CHECK` | `0`: off. `n` compares, after the SCF, the boundary values of the final density at a sample of `n` unique exterior points (the innermost, those of the largest atomic tail, those nearest to an atom, random ones) with its direct Coulomb sum on the Hartree device and reports `hartree_boundary_check_*`; the maximum is that of the sample, and `n` at or above the number of points checks all (GPU boundary only; no effect on the results; its seconds are kept out of the SCF wall time) |  | none | `hartree_boundary_check_*` | `Hartree/GPU_HARTREE.md` |

The switches below were added last. Each ran on A100 nodes against its
own former value in one allocation (first runs, FP64; each item in the code
that introduced it, the last two rows together),
which is the last column. With every one at its former value, and an input
that gives its `Boundary_Sphere_Radius`, a calculation is that of the code
before these switches bit for bit. So it was on a laptop GPU through the MPI
runner, every array of the archive and `parsec.out` apart from its times:
for 32 and 176 electrons, as a run and as a serial control, with the states
on the device and spilled (for a serial control apart from the line that
lists its routes, which names the settings of the column "Serial control"
below beside the 19 it had); for 864 electrons with a budget of 70 columns
per slab, which the multiple cuts to 64 (the defaults then give another
total energy, by 9e-11 Ry); and for 176 and 864 electrons with every sector
basis shared, that one GPU named two and four times as the devices of a
sector, where every copy stays inside the device. On A100 nodes each switch
went back by itself, and all nine together in one code (first runs,
the defaults beside them in the same allocation): 3,480
electrons on 1 and 16 GPUs, 14,680 on 4, 23,768 and 29,576 on 8 and 23,768
and 39,368 on 16 gave, to the last bit, the total energies of the series of
first runs made before these switches. The
defaults gave, to the last bit, the totals of the runs of the last two rows
(3,480 on 1 GPU, 14,680 on 4 and 8, 19,392, 23,768 and 29,576 on 8, 23,768
and 39,368 on 16) and on 8 and 16 GPUs their peaks, and took 267.9 against
278.8 s of SCF for 39,368 electrons on 16 GPUs, 93.3 against 100.0 s for
23,768, 241.6 against 251.1 and 166.5 against 173.7 s for 29,576 and 23,768
on 8, and 117.6 against 123.5 s for 14,680 on 4.

| Switch | Default | Former value | Results | Says what ran | Serial control | A100 nodes, default against former |
|---|---|---|---|---|---|---|
| the input line `Boundary_Sphere_Radius` | absent or `auto`: the default rule of the parser gives the sphere its radius and, where the input has no such line, its `Hartree_Boundary_Tolerance`, from `Domain_Energy_Tolerance` (`1e-3 Ry`) | the line with a length; the two lines that `parsec.out` prints give the calculation of the rule again | an input with a radius: same bits; one without was refused | `result.domain` and `per_rank[].domain_rule_seconds` in `timing.json`, the `domain_*` details | no entry: the sphere is the input's | 3,480 to 39,368 electrons on 1, 4 and 16 GPUs got the radius, tolerance and order derived for them and, to the last bit, the energies of the runs that had named these by hand; ten systems without hydrogen at the surface got 4.6 to 8.2 ang; the rule took 0.03 to 1.27 s (15.6 s once, the first run on its node) |
| `PARSEC_DOMAIN_REPORT` | on: an input with a radius also gets the estimates of its sphere at set-up and the block "Density at the sphere" after the SCF | `0` | same bits | `result.domain` (null with `0` for an input with a radius) and its `after_scf.seconds` | not named: a report, no route of the solver | inputs with a radius kept the bits of the runs made before the rule (3,480 electrons on 4 GPUs with the report on and off, 23,768 and 39,368 on 16 with it on); the block took 0.04, 0.30 and 1.40 s with 0, 1 and 5 estimates of the omitted potential |
| `PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES` | `2147483648`: slabs of at most 2 GiB in the one-block Ritz step of a sector on two devices (8 GPUs), where `PARSEC_CUPY_STREAMING_RITZ_BYTES` is not set | `4294967296`: the slabs a run had; its rounds and bits take `PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` beside it | round-off where the rounds of a sector change: bases above 20 GiB and some of 16 to 20 GiB, on 8 GPUs from 14,680 electrons; same bits elsewhere | `slab_columns` of a shared sector in `timing.json` | not named: it shares no basis | the fullest device peaked at 25,373, 33,643, 50,859 and 67,031 MiB for 14,680, 19,392, 23,768 and 29,576 electrons on 8 GPUs, 3.2 to 3.8 GiB below the former slabs; SCF 175.2 against 172.9 s for 23,768 electrons; the former value gave the former bits and peak (54,795 MiB), in the code that introduced the item, where no round was cut at a multiple |
| `PARSEC_CUPY_DENSITY_RELEASE` | `threads`: after a density the pool of every device that a process empties is emptied by the thread of that device, and the pinned pool meanwhile; an MPI rank waits for them behind the collective calls of its density command. Every such pool is empty before the process goes on, as before. The devices of a shared basis are left out since (last row): on 8 and 16 GPUs nothing is emptied by default | `serial`: one device after another in the thread that built the density | same bits | `release`, `release_seconds` and `device_release_seconds` under `density_pool_release` of a rank | `serial` | density stage of a run 2.70 against 2.94 s (29,576 electrons, 8 GPUs), 2.12 against 2.37 (14,680, 4), 1.78 against 2.11 (23,768, 16), 1.43 against 1.78 (14,680, 16) and 0.58 against 0.62 (5,264, 8); the device threads of the root together took 1.2 to 3.7 times the seconds of its `serial` release, so the devices slow each other down and less is gained than the longest device in place of the sum; the four pairs on 8 and 16 GPUs emptied the pools of shared sectors, as the code did then |
| `PARSEC_CUPY_EXCHANGE_PAIRS` | `1`: the four devices of a shared basis (16 GPUs) copy a chunk that all of them send to their rows in three turns of two pairs | `0`: all twelve copies at once | same bits | `to_rows_copies` of a shared sector | not named: it shares no basis | the way to the rows of a sector took 6.83 against 8.88 s for 23,768 electrons and 16.56 against 21.72 s for 39,368, the SCF 97.6 against 99.6 and 274.0 against 279.6 s; 1.93 to 2.00 against 2.41 to 2.49 s for 10,456, and 0.52 to 0.54 against 0.48 to 0.54 s for 3,480; all with the rounds of equal slabs, before those of whole multiples |
| `PARSEC_CUDA_CONTEXT_CREATION` | `driver`: the context thread of `benchmarks/mpi_full_scf.py` creates the contexts through the CUDA driver library, outside Python's interpreter lock | `runtime`: inside CuPy's first calls on each device, which hold the lock | same bits | `cuda_context_driver_devices` and `cuda_context_driver_seconds` of a rank | `runtime` | preparation of the root for 3,480 electrons 1.62 to 1.77 against 2.57 to 2.70 s on 16 GPUs, 1.76 to 1.79 against 2.39 to 2.51 on 8, 2.83 to 3.03 against 3.70 to 4.44 on 4 and 2.65 to 2.74 against 3.22 to 3.26 on 1; 5.48 against 6.33 s for 23,768 on 16; every device of every rank listed, no capture repeated |
| `PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION` | unset: `auto` of `PARSEC_CUPY_SECTOR_STATE_STORAGE` and of `PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR` keeps the sector states on their devices and the pool unless the count of a device exceeds `PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION` of it (0.978; `1` for all of it) | `0.5`: the host spill and the direct allocator from half of a device's memory in sector vectors | same bits: it chooses the storage and the allocator | `orbital_sector_state_storage`, `orbital_memory_allocator` and `sector_state_fit_bytes` of a rank | `0.5`: its Ritz step takes two arrays per sector, and the count is that of one | 19,392 electrons with neither value named had the bits and the peak of the run with both named, on 4 GPUs in 183.2 against 183.7 s where the former rule had stopped, on 8 and 16 likewise; 14,680 on 2 GPUs took 242.5 s and 80,157 MiB; 10,456 on one GPU ended by the spill in 770.6 s with 24,465 MiB, and stopped for memory with the share at `1` |
| `PARSEC_CUPY_SPILL_DOWNLOAD_ORDER` | `A`: a sector basis that is spilled to the host is downloaded in the order it has, without a second array on its device | `C`: in C order, by a C-ordered copy on the device and a copy on the host | same bits | the variable under `environment` of a rank | `C` | 7,120 electrons spilled on one GPU: 254.1 s and a peak of 11,085 MiB against 458.9 s and 18,175 MiB, the same total energy |
| `PARSEC_CUPY_RITZ_GRAM_MULTIPLE` | `64`: the slabs of the two Gram matrices of the Ritz step, and the rounds of a shared basis, are cut at whole multiples of 64 columns, on every route (1 to 16 GPUs); a slab only narrows | `1` | round-off wherever a slab or a round changes: the overlap of every sector from 512 columns, the projection of one array from 64 columns of budget, and the rounds of a shared basis from 32 columns per slab on two devices and 16 on four, which is every shared sector of the measured series | `slab_cut` (`multiple`), `slab_columns` and `projection_columns` of a shared sector in `timing.json`; the stage `subspace_ritz_projection_seconds`, and `gram` with the `overlap_seconds` in it | `1`: its projection is formed in slabs; its overlap is DSYRK's (`PARSEC_CUPY_RITZ_SYRK=on`) and has none | SCF of a run 166.9 against 173.6 s for 23,768 electrons on 8 GPUs, 242.1 against 246.2 for 29,576, 98.6 against 101.3 for 19,392 and 69.5 against 70.5 for 14,680; on 16 GPUs 95.4 against 96.9 (23,768), 136.0 against 138.1 (29,576), 275.3 against 275.8 (39,368), 41.9 against 42.0 (14,680) and 22.7 against 23.0 (10,456); 118.4 against 121.6 for 14,680 on 4, 112.4 against 114.9 for 10,456 on 2 and 60.1 against 60.9 on 4; the fullest device of a shared sector 0.2 to 1.2 GiB lower (62,709 against 63,913 MiB for 39,368 electrons) and the same for 29,576 on 8 GPUs, whose slabs stay (67,083 against 67,079); total energies within 6e-8 Ry, 1.2e-7 for 39,368. Per sector the overlap took 0.1 to 4.2 s less and the products of the rounds 0.05 to 6.4 s less; the stages around the rounds took 5.0 s more for 39,368 electrons on 16 GPUs, whose rounds lost 1.9 s. Three totals were those of the first run on a node (64.3 against 56.5 s for 5,264 electrons and 95.7 against 86.9 for 7,120 on one GPU, 180.1 against 176.5 for 19,392 on 4) while their projection stage fell (3.35 against 5.17 s, 7.03 against 7.36, 112.4 against 127.4 thread-seconds) |
| `PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE` | `0`: after a density nothing is collected or returned for a sector whose basis several devices share (8 and 16 GPUs): its next step takes the same blocks again; its group returns a slab workspace that a pass has outgrown before it takes the larger one. The pinned pool then keeps its blocks too: 0.3 to 1.1 GiB of host memory more on the root rank, measured with the release off. A sector on one device empties its pool as before | `1`: the pools of the devices of a shared basis are emptied after every density | same bits | `shared`, `shared_kept` and `calls` under `density_pool_release` of a rank in `timing.json` | not named: it shares no basis | density stage of a run 0.88 against 1.56 s (14,680 electrons, 8 GPUs), 1.85 against 2.64 (23,768, 8), 1.68 against 2.70 (29,576, 8), 0.80 against 1.23 (14,680, 16), 1.01 against 1.67 (23,768, 16) and 0.99 against 1.89 (29,576, 16); SCF 0.6 to 2.1 s less; the same total energy to the last bit; the fullest device 2 MiB higher in four pairs and 44 and 48 MiB higher for 23,768 and 29,576 electrons on 8 GPUs (50,677 and 67,083 MiB); `calls` 0 and `shared_kept` as many as the densities on every rank; the root rank 0.7 to 1.1 GiB higher at its high-water mark of host memory, the other ranks the same to 0.01 GiB |

The first row is described in the reference `README.md` ("Sphere radius left
to the code") and below, the slabs, the turns and the multiple above, the
two releases, the rule of the state storage and the download in
`MULTI_GPU.md`, the creation of the contexts in `GPU_SETUP_OPTIMIZATION.md`.

Some rows meet. The rule of the state storage counts a shared basis by the
blocks of the rule that shares it, so a sector on two devices is counted
with its two slabs of 2 GiB: 62.0 instead of 66.0 GiB for 29,576 electrons
on 8 GPUs, and the former count with the former limit named. Neither
decision changes by it: both counts lie below the share of a device that the
sharing rule allows. The multiple does not move that count, nor the count of
a sector on one device: both hold the slabs of their budget, which the
multiple only narrows. The buffer of one array is then less than 8 bytes per
column below its count, and the workspace of a shared sector up to 1.2 GiB
below what the equal slabs took. So the same sectors stay on their devices
at every multiple.
The multiple cuts the rounds from the slabs that the limit of two devices
and the budget of four allow. The former value of either gives back the
slabs a run had; the rounds, and with them the bits, come back only with
`PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1` beside it.
The turns of four devices and the slab limit of two never act on one sector.
The turns are taken with the slabs of 2 GiB that a sector on four devices
cuts and, since, with the whole rounds of the multiple: every chunk of the
measured bases keeps its four senders (derived), in 8 to 154 chunk steps per
pass where the equal slabs made 6 to 108. Where the slab of a whole round is
just wider than a chunk of 1 GiB, for 10,456, 29,576 and 39,368 electrons,
every other chunk step carries 4, 5 or 1 columns in its three turns (above).
Each of the two ran on A100 nodes without the other, and both together in
one code for 3,480, 23,768 and 39,368 electrons: the way to the rows
of 39,368 electrons took 17.4 s per sector, where equal slabs in turns took
16.5 s and equal slabs received at once 21.6 s.
The two releases: `PARSEC_CUPY_DENSITY_RELEASE` says how the pools of
several devices are emptied, the last row whether those of a shared basis
are. On 8 and 16 GPUs every sector is shared, so by default no pool is
emptied there, no device thread is given a release and a rank has nothing to
wait for behind its density command (`release` `threads`, `shared` `keep`,
`calls` 0). The threads act on 2 and 4 GPUs, and on 8 and 16 with the last
row at `1`. A shared basis is never spilled to the host: a rank whose
sectors are all shared keeps its pools under either storage of the states,
and sector states that a rank keeps on the host beside a shared basis are
collected for and empty the pinned pool as they do alone. The block
"Density at the sphere" after the SCF reads the density of the result on the
host and takes nothing from a device, whatever the pools hold then.

Which limit a sector's tiles follow is a matter of its devices. On 1, 2 and 4
GPUs every sector is alone on a device that filters four, two or one of
them: 325,000, 459,620 and 650,000 rows. On 8 and 16 GPUs the devices of a
node share a sector: 1,100,000 rows, whatever the first limit says. `0` and a
tile size are taken as they are on every route, so the stencils a launch had
are those of `PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS` at 2,200,000, 1,555,634 or
1,100,000 on 1, 2 or 4 GPUs and of `PARSEC_CUPY_IMPLICIT_TILE=0` on 8 and 16.
The serial control keeps the former route of
`PARSEC_CUPY_SECTOR_POOL_RELEASE`, `PARSEC_CUPY_DENSITY_COLLECTION`,
`PARSEC_CUPY_DENSITY_RELEASE`, `PARSEC_CUPY_RITZ_GRAM_MULTIPLE`,
`PARSEC_SYMMETRY_SCF_BUFFERS` and `PARSEC_FAST_GRID` and the slot-major
stencil at every size; it shares no basis, so it reads neither the slab
budget of that step nor the slab limit of two devices nor the turns of four
nor the helper nor the row count of a group nor the release of a shared
basis, and names none of them. Its contexts are CuPy's also where a launcher
asks it for the overlap (`PARSEC_CUDA_CONTEXT_CREATION=runtime`). Where its
launcher leaves the storage of the sector states or their allocator to the
solver it decides by the former rule
(`PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION=0.5`) and downloads a spilled basis
by the former call (`PARSEC_CUPY_SPILL_DOWNLOAD_ORDER=C`): its Ritz step
takes two arrays per sector, which the count of the rule that came after
does not hold; with both values named, as in every measured control, neither
is read. The report of the sphere (`PARSEC_DOMAIN_REPORT`) is the
launcher's. Of the Hartree boundary it names the kernels and the evaluator
of the tail, `PARSEC_HARTREE_BOUNDARY_KERNEL=full` and
`PARSEC_HARTREE_ATOMIC_TAIL_VALUES=host`, and nothing else: the order, the
tolerance and the tail are the launcher's and the plan made from the
geometry, which the control must share with the run it is compared with.
`result.hartree_boundary` in the `timing.json` of both says what each had.
At a raised order the full-grid kernels hold more on the first device of a
control than PARSEC's boundary did. Where a control does not fit beside them
(expected for 14,680 electrons on four devices), its launcher names
`PARSEC_HARTREE_BOUNDARY_BACKEND=native`: the boundary on the host, see
`Hartree/GPU_HARTREE.md`.

The sphere radius has no environment switch. An input that leaves
`Boundary_Sphere_Radius` out gets the radius and the Hartree tolerance of
the default rule in the parser (reference `README.md`, "Sphere radius left
to the code"); the line `Boundary_Sphere_Radius` restores the former
calculation, and the report prints it with the tolerance, so that the two
lines give the same grid, plan and result bit for bit. The rows of the table
above that begin with `PARSEC_HARTREE_` act in the driver after the parsing:
they change the boundary of the run and never the sphere, and the set-up
block then says that `Domain_Energy_Tolerance` covers the wall only.

The report of the sphere has one: `PARSEC_DOMAIN_REPORT`, on by default and
read by the reporter of both drivers. `0` (or `off`) gives an input with a
radius the `parsec.out`, the backend details and the work after the SCF it
had before the rule: no estimate at set-up, no block "Density at the
sphere", no `domain_*` detail, `result.domain` null. The results are the
same bits with and without it. A radius the rule chose keeps its lines and
its record and loses the block after the SCF. What a run had is in three
places:

| Where | What |
|---|---|
| `parsec.out`, grid data | the radius, the outermost atom and vacuum, the wall estimate before the SCF, what set the radius, the two input lines (radius of the rule only) |
| `parsec.out`, real-space setup | the energy the boundary values of the plan in use are estimated to leave; for a radius of the input also the wall estimate of its sphere |
| `parsec.out`, after the SCF | "Density at the sphere": the estimate from the converged density and the radius and tolerance that would meet the wall share, with what set that radius where the density did not |
| backend details, result archive | `domain_radius`, `domain_vacuum`, `domain_energy_tolerance`, `domain_wall_estimate`, `domain_boundary_energy`, `domain_wall_after_scf`, `domain_radius_for_wall_share` |
| `timing.json` of the MPI runner | `result.domain` with the same, the shell sums of the fit and, under `after_scf`, `estimates`, `shell_seconds`, `radius_seconds` and `seconds`; `per_rank[].domain_rule_seconds` |

The MPI runner compares, before any rank prepares, the radius and the
tolerance every rank resolved, bit for bit, and stops where they differ. The
seconds of the rule are inside `reported_program_seconds`: 0.03 s for 3,480
electrons and 0.22 s for 39,368, where one estimate of the omitted potential
is needed (workstation host; pseudopotential loading included). For an input
with a radius the parser does what it did; the wall estimate of its sphere
(a few milliseconds), the shell sums after the SCF (0.156 s at 22.6 million
grid points on the same host) and the search for the radius under them are
made by the root while it reports. That search asks the largest multipole
order for the tolerance of the rule and makes no estimate of the omitted
potential for 3,480 electrons, none or one (0.12 s) for 23,768, and for
39,368 one (0.17 s) or four to seven where the wall would leave less than
4 ang of vacuum: 0.33 to 1.2 s for the block after the SCF of C9449H1572 on
that host, where the whole reporting after its SCF took 0.17 s on 16 A100
before. An input with a `Hartree_Boundary_Tolerance` line gets
no estimate. A failure of a report is printed in its place and does not
reach the result.

CuPy supports either translated first-SCF path: fixed-cycle CHEBFF or
locking/restart CHEBDAV. Both retain the complete buffered Ritz space on the
device and use the translated saved-SUBSPACE path on later SCF iterations.
The selected first solver is never replaced by CHEBFF, CHEBDAV, SciPy, native
code, ARPACK, or the Fortran executable. Large basis blocks, Hamiltonian
images, residuals, filters, orthogonalization, and Ritz rotations use float64
operations. By default projected symmetric eigensystems of order at most 64
use host LAPACK, which measured faster than repeated small CUDA solver
launches; larger projected problems remain in CuPy. Set
`PARSEC_CUPY_HOST_EIGH_MAX=0` to keep every projected solve on the GPU. Other
deliberate host work is limited to deterministic PARSEC-compatible random-
vector generation, scalar control/diagnostics, and the at-most 8-by-8 Lanczos
tridiagonal solve.

CHEBDAV reuses the float64 Ritz eigenvalues already returned by that host
LAPACK solve for scalar convergence and filter-window decisions. This removes
many one-value CUDA stream synchronizations without changing the projected
matrix, eigenvectors, locking policy, or tolerances. Set
`PARSEC_CUPY_REUSE_HOST_RITZ_VALUES=0` for the explicit device-scalar control
path.

For CHEBDAV operators with at least 100,000 real-space rows, the default
orthogonalizes each appended source-sized block with FP64 block CGS2 against
the existing basis followed by two device MGS passes within the normally
six-column block. This preserves the Davidson span and all residual-locking,
restart, filter, and cleanup rules while replacing synchronized
column-by-column projections against the growing basis with cuBLAS level-3
work. Every block receives a rank/orthogonality audit; an unsafe block falls
back first to Householder QR and then to the replacement-capable literal
PARSEC MGS routine. Smaller symmetry sectors retain literal PARSEC MGS because
GPU setup is not profitable there. Set `PARSEC_CUPY_CHEBDAV_BLOCK_ORTH=off`
for a source-arithmetic comparison, `on` to force it, or change the automatic
row crossover with `PARSEC_CUPY_CHEBDAV_BLOCK_ORTH_MIN_ROWS`.
The projection against the growing prefix uses the complete contiguous
C-order Davidson workspace for the coefficient GEMM. For the normal blocks
of at most six vectors, a row-major CUDA kernel then applies only the active
prefix and subtracts the update in one pass; larger/user-forced blocks retain
the full-workspace cuBLAS update. This is algebraically the same prefix
projection, but avoids repeated noncontiguous-prefix packing, zeroing inactive
coefficients, and a second full-workspace GEMM. Set
`PARSEC_CUPY_CHEBDAV_FUSED_PREFIX_UPDATE=0` for the former full-workspace
update or
`PARSEC_CUPY_CHEBDAV_FULL_WORKSPACE_CGS=0` for the direct-prefix control.
The incremental Ritz projection likewise multiplies the complete contiguous
Davidson workspace by the six new Hamiltonian images and consumes only the
active row interval, avoiding an implicit noncontiguous active-basis copy.
Set `PARSEC_CUPY_CHEBDAV_FULL_WORKSPACE_RITZ=0` for that control path.

The exact 48-bit DLARNV random sequence is generated with 2,048 skip-ahead
lanes on NumPy. This preserves every value and the final seed bit-for-bit but
avoids the scalar Python loop that previously dominated initial-basis setup.

For a proved commuting Cartesian-reflection action, the GPU path constructs
every real character expansion `U_Gamma`. For orbit `O_w` with stabilizer
`S_w`, representation `Gamma` contains that orbit exactly when
`chi_Gamma(s)=1` for every `s` in `S_w`; otherwise that orbit is zero in the
sector. On an admitted orbit,
`U_Gamma[i,w]=chi_Gamma(g_i)/sqrt(|O_w|)`. This includes the free-action case
and permits exact representation-dependent dimensions. Static terms are
projected as
`A_Gamma = U_Gamma.T @ A @ U_Gamma` and
`B_Gamma = U_Gamma.T @ B`. Each wedge Hamiltonian keeps an independent
CHEBFF/CHEBDAV-to-SUBSPACE state across SCF iterations. The initial allocation
matches PARSEC's integer policy,
`floor(N_states/N_rep) + Subspace_Buffer_Size`; sectors are grown only when
their last Ritz value does not bracket the globally requested cutoff. The
sector spectra are then stably sorted. During SCF, the selected vectors remain
on the normalized wedge: admitted phases are `+1` or `-1`, and rejected
stabilizer orbits are zero, so squared orbitals give the same scalar density
on every orbit image. The code evaluates that density once on the wedge and expands
only the length-grid scalar field. Signed full-grid orbitals are materialized
once for the final public result. PARSEC-compatible one-based representation
labels are printed and archived. The full-grid GPU Hamiltonian is not
allocated in this path. The kinetic stencil of a sector is built from the
grid (see the cache section above) and all reduced KB factors are constructed
from one canonical sparse gather. Where no cache entry is read, the default,
independent representation
assemblies run serially for at most two representations and on at most four
CPU workers for larger decompositions; the native stencil kernel is itself an
OpenMP loop and takes the sectors in turn. This cold-time policy was selected by
fresh-cache complete-run A/B measurements; set
`PARSEC_SYMMETRY_OPERATOR_WORKERS=N` to override it. Signed-permutation grid
maps use exact integer axis/sign/offset
arithmetic after validating the affine lattice offset once, instead of
rounding a full float64 coordinate transform for every operation. Identical
CUDA stencil kernels are compiled once and shared across sectors. A nonblocking-stream
scheduler is implemented for independent sectors, but measurements showed
bandwidth/compute contention on one GPU; the single-device default is
serialized.
Set `PARSEC_CUPY_SECTOR_SCHEDULER=streams` to profile overlap and optionally
limit it with `PARSEC_CUPY_SECTOR_STREAMS=N`.
Independent one-vector representation Lanczos bounds can also be profiled
with `PARSEC_CUPY_COLLECTIVE_LANCZOS=1`. It is off by default because it was
numerically identical but slower on the measured GPU; large filters remain
serialized regardless.

On a multi-GPU host, `PARSEC_CUPY_DEVICES=auto` (the default) assigns
representations round-robin to the visible devices and solves independent
sectors concurrently. Use `current` to restrict execution to the current
device, or a comma-separated list such as `0,1`. These are logical indices
inside the scheduler's allocation: never overwrite `CUDA_VISIBLE_DEVICES`.
With the default sequential scheduler, each GPU runs one sector at a time;
different GPUs run concurrently, including when sector sizes are unequal.
Static operators and saved subspaces stay on their assigned GPUs. Each sector
forms its density on that same GPU, including non-contiguous orbital views,
then downloads only a grid-length density vector. Global eigenvalue ordering,
occupation assignment, and the FP64 scalar SCF remain shared and unchanged.

Requested final host wavefunctions are expanded in bounded column blocks on
the CPU, without gathering all orbitals into one GPU's memory. The final host
array still needs `8 * grid_points * states` bytes. Ordinary CLI runs do not
request this large array unless wavefunction output is enabled.

Through `main.py` this is **symmetry-sector parallelism inside one process**,
not spatial-domain or general orbital MPI decomposition: every sector lies on
one device, so the available exact real one-dimensional representations limit
useful GPU concurrency (four for the nanodiamond cases). An asymmetric
calculation without this decomposition remains a single-GPU solve. Extra
devices do not automatically accelerate it. Native preparation and Hartree
work use the CPU/OpenMP backend under `--backend auto`; do not launch
multiple copies of the Python driver with `mpirun`. More devices than
sectors, and more than one node, are the MPI runner's
(`benchmarks/mpi_full_scf.py`, one MPI rank per node): a rank divides its
devices among the sectors it owns, and the devices of a sector share its
basis where it fits (`PARSEC_CUPY_DISTRIBUTED_STATE`, above). The 8- and
16-GPU figures of this document were run that way.

For a Slurm example, validation commands, memory rules and timing interpretation,
see [the multi-GPU guide](MULTI_GPU.md).

For the centered finite-difference operator, production CuPy transposes each
short canonical CSR row into a stencil-major layout
``neighbor[slot, grid_row]``. Adjacent CUDA threads therefore read contiguous
int32 neighbor rows and uint8 coefficient codes; the palette retains the
exact float64 coefficient bits. Each thread still visits its row in canonical
CSR order. A second kernel fuses kinetic, the current local diagonal,
optional nonlocal image, and the normalized Chebyshev recurrence into one
grid pass for blocks of up to six orbitals. For KB projectors a canonical-
order CUDA kernel computes the small `diag(signs) @ B.T @ X` contraction
without the launch overhead of many tiny cuSPARSE calls, while the large
`B @ coefficients` scatter is fused into that same row kernel. This avoids a
full-grid nonlocal temporary without ever forming a dense nonlocal matrix.
Already-resident float64 inputs retain their existing row/column strides;
the raw kernels consume those strides directly instead of forcing thousands
of temporary Fortran-order copies. Resident workers also give NVRTC a private
writable `.parsec_cache/cupy-temp` workspace, preventing a restricted system
temporary directory from silently disabling the custom projector kernel.
The default `PARSEC_CUPY_PROJECTOR_REDUCTION=auto` retains canonical serial
summation for short projector rows and selects a deterministic 128-thread
tree reduction only for rows with at least 256 entries. `serial` and
`parallel` force either policy for profiling.
Set `PARSEC_CUPY_CUSTOM_PROJECTOR_DOT=0` to restore cuSPARSE and
`PARSEC_CUPY_FUSED_PROJECTORS=0` for an unfused comparison. The earlier compact CSR-order
kernel and generic CuPy CSR remain automatic fallbacks. Set
`PARSEC_CUPY_STENCIL_MAJOR=0` before Python starts to exercise the compact
CSR-order fallback explicitly.

Orbitals also remain on the device through density construction. In symmetry
mode the fused row kernel works on the wedge and downloads a wedge-length
density. Density, local ionic/Hartree/XC potentials, Anderson history,
residual norms, and energy quadrature then remain one physical value per orbit
through the entire SCF loop; expansion occurs only at the final public result.
Without orbital symmetry the code downloads the ordinary full-grid density. The final requested
wavefunctions are expanded/downloaded once. Later-SUBSPACE Ritz residuals are disabled in
the production SCF adapter because they are reporting diagnostics and do not
control filtering, occupations, density, or SCF convergence. Direct modular
eigensolver calls retain residual construction by default.

Saved-SUBSPACE basis treatment uses a size-adaptive default. Small complete
bases retain the audited modified Gram--Schmidt path described below. When
the estimated complete-basis work `N * states^2` reaches 100,000,000, the
default solves Rayleigh--Ritz directly in the non-orthogonal filtered basis,

`(X.T H X) C = (X.T X) C diag(epsilon)`.

A Cholesky factor of the small overlap matrix whitens this generalized
eigenproblem. The code audits its condition number, factorization, and final
coefficient orthogonality; any unsafe basis falls back to stable blocked
Householder QR for that and subsequent SCF iterations. This constructs the
same Ritz subspace without a tall QR in the well-conditioned regime. A
symmetry sector keeps one column-major array and projects `H X` a slab of
columns at a time (`PARSEC_CUPY_STREAMING_RITZ` above); elsewhere a
persistent column-major `H X` workspace avoids repeated large CuPy allocations
and strided real-space-kernel reads. Set `PARSEC_CUPY_GENERALIZED_RITZ=off` to
disable it, or `on` to request an audited attempt for every complete basis.
Only a sector basis that `PARSEC_CUPY_DISTRIBUTED_STATE=1` forces onto
several devices takes this solve whatever the setting says.
`PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD` and
`PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX` control the automatic work and
stability limits.

This leaves the small representation sectors used by the symmetry benchmarks
on their measured-fast MGS route. Override that orthogonalization route with
`PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION=mgs`, `qr`, or `cholqr2`; the automatic
threshold can be changed with `PARSEC_CUPY_SUBSPACE_QR_WORK_THRESHOLD`.

In the complete-basis MGS route, filtered bases execute the common PARSEC
first-projection branch with device scalars and transfer all norm decisions
once; any failed 0.1 test restores the untouched input and reruns the literal
two-pass/replacement routine. Small CHEBDAV appended blocks use the literal
path directly. That literal path
queues the unchanged input and first-projection norms in their original order
but downloads each pair together, removing one synchronization per tested
column without changing the 0.1/0.68 decisions. Set
`PARSEC_CUPY_SPECULATIVE_MGS=off` for a literal trace or `all` to speculate
on appended blocks too. Exact QR and two-pass Cholesky-QR implementations remain
available explicitly for architecture-specific profiling.

CHEBFF does not form unused Ritz residuals. Optional cross-block Chebyshev
batching can reduce launch count, but the measured GPU was bandwidth limited
and ran faster with PARSEC's ordinary block traversal. It is therefore opt-in
with `PARSEC_CUPY_BATCH_FILTERS=1`, not the default.

### Hartree acceleration

All accelerated backends use an associated-Legendre recurrence for the same
normalized complex multipole moments and boundary potential as the reference
SciPy-special-function implementation. It keeps only a few grid-length work
arrays instead of storing a dense density-to-boundary map. In the native
spherical path, a reusable C++ object caches angular coordinates and groups
missing stencil entries by interior row. Each SCF call then forms all moments
and `8*pi*rho - A_IB*V_B` in one compiled float64/OpenMP operation. Box
calculations retain the exact Python direct discrete-Coulomb boundary.

In the default hybrid path, this reusable Hartree boundary geometry is built
on a CPU worker while the independent reduced orbital operators are loaded or
constructed on the GPU. The main thread joins the worker before the first
Hartree solve, so no SCF work races and the resulting arrays are identical to
inline construction. Set `PARSEC_OVERLAP_HARTREE_SETUP=0` for an inline A/B
control. The output records the worker time, join wait, and estimated hidden
setup time.

After the identical boundary-corrected right-hand side is formed, SciPy uses
the reference-equivalent host CG recurrence, native uses a cached
C++17/OpenMP CSR CG solver, and CuPy reuses the very same device CSR allocation
as the Kohn-Sham Hamiltonian. The native solver retains canonical row
summation order but losslessly compacts the repeated finite-difference matrix
to int32 column rows and one-byte codes into a float64 coefficient palette.
Fixed-size parallel dot-product blocks are merged deterministically, and the
residual norm is reused for `beta` exactly as PARSEC's bundled SPARSKIT CG
does. Warm starts, tolerances, matrix-vector budgets, breakdown rules, and
final true-residual diagnostics are preserved.
The native iteration fuses `A @ p` with the canonical-block
`p dot (A @ p)` reduction, eliminating a second traversal while retaining the
original CSR row order and deterministic 4,096-row reduction topology.
In the default hybrid, the native CG implementation is selected for Hartree
while the Kohn--Sham Hamiltonian and eigensolver remain on the GPU; this avoids
the reduction-heavy GPU CG path without moving eigensolver state off-device.
After two completed SCF solves, native CG forms a clipped two-step
chronological prediction from the two preceding right-hand sides and Hartree
solutions. CG still solves the unchanged linear system to the unchanged
tolerance and recomputes the final true residual. Set
`PARSEC_HARTREE_CHRONOLOGICAL_GUESS=0` to use only the immediately preceding
Hartree potential.

Before constructing native CG, `auto` tests all 48 Cartesian signed-
permutation operations against the labeled atoms and every active lattice
point, then selects the largest exact commuting involution subgroup. If a
nontrivial group is proved, the totally symmetric Poisson system is projected as
`A_w = U.T @ A @ U`, `b_w = U.T @ b`, solved on the wedge, and expanded by
`U`. Native extension 0.5 (0.6.0 stores its angular arrays for the order
in use, up to 60, with unchanged arithmetic; 0.6.1 hands the prefactors of
orders up to 9 to the GPU boundary in the layout of 0.5 again, the only one a
Python tree from before 0.6.0 reads, so a build of it can sit in such a
checkout) additionally precomputes orbit-summed multipole
coefficients (and stores them where a symmetry cache is named), constructs the boundary-corrected
normalized `b_w` directly, and fuses the CG matrix-vector/dot traversal, so
repeated SCF steps never form the full-grid Hartree RHS. Orbit normalization
also handles points lying on reflection planes.
`PARSEC_HARTREE_SYMMETRY=0` disables this optimization for a controlled
Hartree-only comparison. Prefer `--symmetry off` when both Hartree and the
orbital eigensolver must use the full grid; a failed proof in `auto` always
falls back safely.

When orbital sectors are active, the density and local potentials are exact
totally symmetric scalar fields. Residual norms, Anderson history/Gram
matrices, and density-potential energy integrals therefore retain one physical
value per orbit and use multiplicity-weighted quadrature. No repeated scalar
field expansion is needed by the Hamiltonian, Hartree, XC, mixer, or energy
path. The formulas, convergence criterion, and
energy terms are unchanged, including for unequal orbit sizes.

`PARSEC_SYMMETRY_SCF_BUFFERS` (default `1`) removes allocations from that
host work of the root: the weighted sums read the orbit multiplicities as
float64, an energy forms the weighted density once for its four integrals,
and the Anderson step works on blocks of 16,384 orbits in arrays the mixer
keeps, twelve wedge fields beside a history of four. `0` allocates an array
for every integral and every term of the step, as before. With the float64
multiplicities the root so keeps thirteen wedge fields through the SCF that
the former step allocated and returned in every mix: 0.37 GiB for the
3,786,832 orbits of the 23,768-electron cluster, 0.07 GiB for 3,480
electrons. Its resident memory between steps is higher by that, and its
high-water mark by up to that wherever the SCF sets the mark outside a mix;
the peak inside a mix is the same. `scf_scalar_field_buffers` in the
backend report says `on` or `off`, as the preparation found the switch. Each
floating-point operation keeps its operands and their order, and the dense
products of the step receive arrays of the former shape and layout, so the
norms, energies and mixed potentials are the same to the last bit wherever
the BLAS products do not depend on the address of an operand (true of the
NumPy tested). With the orbit map of the 23,768-electron cluster (3,786,832
orbits) and the mixing controls of its input, ten steps of norms, energy and
mixing took 1.19 instead of 2.98 s on a workstation, the Anderson step with
a full history 0.099 instead of 0.297 s, every mixed potential, residual and
energy term equal bit for bit; so were a complete SCF of 424 electrons on
its GPU. Those seconds are a workstation's. On A100 nodes the switch ran on
and off in one allocation together with `PARSEC_FAST_GRID` and
`PARSEC_CUPY_DENSITY_COLLECTION` (3,480 electrons on 4, 8 and 16 GPUs, 10,456
and 19,392 on 4, 23,768 on 16), with the same total energies to the last bit.
The serial control of the runner keeps
`0`, and `full` for `PARSEC_CUPY_DENSITY_COLLECTION` and `serial` for
`PARSEC_CUPY_DENSITY_RELEASE` (`MULTI_GPU.md`).

## Selection, fallback, and provenance

An explicitly requested `scipy`, `native`, or `cupy` backend is a strict,
clean comparison mode. If its runtime is unavailable or the selected
physical/eigensolver path is unsupported, the run reports an actionable error
instead of silently composing it with another backend or choosing another
solver.

`auto` is the hybrid and fallback-enabled mode. It probes capabilities,
selects valid implementations independently for finite-difference
construction, Hamiltonian/eigensolver execution, and Hartree, and records why
any preferred component could not be used. The dry run, text report, and
archive provenance identify at least:

- requested and selected backend;
- finite-difference builder and Hartree backend;
- float dtype and CPU/GPU device;
- implementation description and build/runtime details;
- symmetry mode, detected group order, wedge size, orbital-sector policy,
  and any representation fallback;
- every fallback reason;
- sparse Laplacian size and nonlocal projector count where available.

This makes a successful fallback visible and keeps performance results
auditable.

## Timing and profiling

The reference source timings remain available and are carried into accelerated
results. Static preparation reports pseudopotential loading, grid creation,
finite-difference construction, local and nonlocal ionic setup, valence/core
density setup, ion-ion energy, and total preparation wall time. SCF reports
initial Hartree and XC, Hamiltonian binding, diagonalization,
occupation/density construction, iterative Hartree and XC, mixing/energy, and
total SCF wall time.

Accelerated backends add coarse execution statistics:

- initialization and optional warmup;
- local-potential update count and time;
- complete Hamiltonian application count, orbital-vector count, total time,
  and average time;
- host-to-device, synchronized device, and device-to-host time when relevant;
- accelerated Hartree call, boundary/RHS, linear-solve, and transfer totals;
- backend selection and fallback provenance.

`--profile-operator` requests one representative synchronized breakdown of
finite-difference, local-potential, and nonlocal-projector application. It is
an opt-in diagnostic. Production eigensolvers do not place timers or device
synchronizations around every Hamiltonian term inside Chebyshev recurrences,
because that would materially distort the workload being measured.

## Package layout

```text
parsec_python/acceleration/
├── backends/       SciPy, optional native, and optional CuPy execution layers
├── benchmarks/     The MPI full-SCF runner (mpi_full_scf.py), captures and timing tools
├── experimental/   The MPI sector context of that runner (mpi_scf.py) and replay prototypes
├── Grid/           The reference cluster grid, built slab by slab
├── Laplacian/      Exact-key lazy full-grid finite-difference descriptor
├── Hamiltonian/    Backend-bound matrix-free Hamiltonian API
├── Eigensolvers/   CuPy full-grid and representation-sector eigensolvers
├── Hartree/        Fast multipoles plus SciPy/native/CuPy Poisson solvers
├── Symmetry/       Reflection orbits, characters, sector stencils, projected operators
├── Occupations/    Fused device-resident orbital-density construction
├── V_ion/          Native radial local/density/KB setup wrappers
├── V_xc/           Cached native CA/PZ-LDA evaluator
├── SCF/            Reference SCF composition with backend substitution
├── Output/         Reference report plus backend provenance/statistics
├── native/         Optional CMake/pybind11 C++17/OpenMP extension
├── tests/          Backend parity, selection, fallback, and timing tests
├── models.py       Backend identity, provenance, statistics, result wrapper
├── cli.py          Accelerated command-line orchestration
├── resident.py     Authenticated local warmed-worker runtime
└── driver.py       Optimized preparation and workflow orchestration
```

See [ACCELERATION_AUDIT.md](ACCELERATION_AUDIT.md) for the PARSEC source
mapping, supported-scope alignment status, and the C++/CuPy/NumPy decision for
every single-point stage.

Acceleration modules may depend on readable `parsec_python` components. Core
scientific modules do not import acceleration internals; only the public API
and canonical launcher select the optimized workflow. This keeps the physical
implementation independently inspectable and testable.

## Validation

From the repository root, make `src` importable and run the accelerated tests:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s src\parsec_python\acceleration\tests -p "test_*.py" -v
```

Performance claims should always state the input, grid, eigenstate/filter
settings, selected backend, dtype, device, thread count, and whether operator
profiling was enabled. Compare energies, eigenvalues, residual histories, and
densities against the unchanged `parsec_python` result before comparing
wall time. State also whether a symmetry cache was named: the workstation
figures below that speak of a cache hit, a warm cache or a cache load are of
runs with the cache that was then the default and now needs
`--symmetry-cache DIRECTORY`.

As one machine-specific implementation check, the canonical H2 grid
(`N=179,944`) with a 16-vector block and ten warmed applications measured
about 0.0363 s/application for SciPy and 0.00835 s/application for the
C++/OpenMP backend (about 4.35x faster, OpenMP maximum 32 threads). This is a
kernel microbenchmark, not a promise for total SCF speed: Hartree work,
projected dense algebra, problem size, memory bandwidth, and thread settings
also matter.

On the benzene grid (`N=268,096`), the initial spherical boundary/RHS stage
measured 11.92 s with the reference spherical-harmonic calls and 1.41 s with
the recurrence (8.46x). The complete initial Hartree solve measured 13.02 s
for the reference, 2.85 s for accelerated SciPy, 2.08 s for native with 24
OpenMP threads, and 4.52 s on the available laptop GPU. GPU CG is limited by
the scalar reductions required by every iteration on this case; the GPU path
is retained because its eigensolver can dominate larger CHEBFF workloads.
On that same benzene domain, construction of the 9,286,528-nonzero
finite-difference CSR matrix measured 0.620 s in Python/SciPy and 0.125 s in
the native builder (4.94x).

On the 523,984-point naphthalene benchmark, the cached native boundary/RHS
took 0.063 s versus 2.639 s for the NumPy recurrence path (41.9x), with
`max |delta b_eff| = 2.27e-13`. The compact native CG took 0.710 s for the
initial 276-iteration solve. Across the complete 11-solve SCF run, Hartree
fell from 55.77 s to 9.76 s: 0.56 s for all boundary/RHS work and 9.20 s for
all linear solves. Total accelerated Python wall time fell from 88.94 s to
31.66 s, while the final energy remained `-123.37042729 Ry` (the recorded
28-rank Fortran result is `-123.37042748 Ry`). These are workstation-specific
measurements, but they explain why the fastest default keeps the GPU
Hamiltonian/eigensolver and selects native Hartree on this machine.

With the stencil-major layout, fused Chebyshev recurrence, and diagnostic-only
later-SUBSPACE residual work removed, the same naphthalene input retained the
complete printed SCF energy sequence and final `-123.37042729 Ry` energy.
Diagonalization fell from 9.66 s to 7.10 s, SCF from 19.68 s to 16.11 s, and
the one-process wall time from 24.41 s to 21.16 s on the RTX 5070 Laptop GPU.
The recorded 28-rank Fortran wall time remains 19.63 s.

The recorded PARSEC job obtains that time using an eight-operation `D2h`
wedge (65,498 points rather than 523,984) and four concurrent seven-rank MPI
representation groups. After applying the same totally symmetric reduction
to Hartree, the Python Hartree subtotal fell from 7.36 s to 0.96 s and total
process wall time to 13.68 s. The bit-exact skip-ahead DLARNV generator then
reduced the latest run to 6.52 s diagonalization, 8.12 s SCF, and 11.43 s
best observed complete wall time (13.13 s in a repeated cold-start run), with
unchanged final `-123.37042729 Ry`.

With all eight `D2h` reflection representations active on the GPU, the same
input converged in ten SCF iterations to `-123.37042737 Ry`; the recorded
28-rank Fortran value is `-123.37042748 Ry`. The final representation label of
every one of the 30 printed states matched PARSEC. On the measured RTX 5070
Laptop GPU, repeated runs gave 4.06--4.21 s diagonalization, 5.68--5.86 s
SCF, and 12.42--12.55 s cold complete-process time. PARSEC-style global
sorting trimmed the final active sector counts to `9 8 8 9 7 9 9 9`. The
subsequent structural pass removed the unused full-grid CUDA allocation,
batched host projection, added exact-key operator caching, and shared the
invariant CUDA kernels. The current GPU-ready cache also shares sector
neighbor topology: its naphthalene entry is about 29 MB rather than 211 MB,
and a measured cache load fell from about 0.131 s to 0.035 s. Cache-hit runs
retained the same energy and representation labels. Keeping selected orbitals
on the wedge reduced the ten repeated occupation/density stages from about
0.025 s to 0.012--0.013 s and cut that repeated orbital workspace by eight;
the overall wall change is small because diagonalization dominates. CUDA-
stream schedules with 2 or 8 workers were measured and were slower, so they
remain an explicit profiling option rather than the default. Multi-GPU sector
distribution is implemented automatically when more than one device is
visible, but this one-GPU workstation cannot supply a scaling measurement.

The next general optimization pass retained the same ten-iteration energy
trajectory and final `-123.37042737 Ry`. On the same naphthalene run, fusing
the KB scatter, reducing SCF scalar algebra, and constructing the native
Hartree RHS directly on symmetry orbits reduced diagonalization from 4.326 s
to 3.868 s, Hartree from 1.031 s to 0.611 s, mixing/energy from 0.363 s to
0.173 s, and SCF wall time from 5.932 s to 4.825 s. A warm geometry/phase/
operator-cache process completed in 6.69 s versus 10.68 s for the prior v3
profile. These are measured workstation timings, not portable guarantees.

The latest exact pass added the custom canonical-order CUDA `B.T` projector,
persistent native Hartree geometry, fused native CG matrix-vector/dot work,
compact end-to-end scalar fields, cached CUDA discovery, small projected host
LAPACK, broader exact scalar symmetry, and multi-GPU sector assignment. In a
paired naphthalene comparison, replacing only cuSPARSE `B.T` with the custom
kernel reduced diagonalization from 3.842 s to 3.622 s and SCF from 4.585 s
to 4.403 s. The final default validation took 3.615 s diagonalization,
0.627 s Hartree, 0.063 s mixing/energy, and 4.380 s SCF. Every printed
ten-iteration energy was unchanged and the final value remained
`-123.37042737 Ry`. Complete-process wall time was 8.29 s in that cold Python
process; CUDA initialization makes this figure more variable than the SCF
subtotal.

The subsequent synchronization/structure audit removed cuSPARSE construction
from the production representation path, retained raw KB CSR factors, kept
each short Lanczos scalar recurrence on its CUDA stream, and batched its
single tiny host transfer. It also added audited complete-subspace MGS,
stable shared local-potential buffers, adaptive projector reductions, a
2,048-lane bit-exact DLARNV tile, and a lightweight exact atom-matching path
that avoids importing SciPy optimize. The final cache-hit naphthalene run
again reproduced all ten printed energies and `-123.37042737 Ry`, with
3.502 s diagonalization, 0.601 s Hartree, 4.235 s SCF, and 8.39 s internal
wall time (9.08 s measured complete process). The independent H2 full-
nonlocal case also converged with the new defaults. These timings vary with
CUDA initialization and system load; the unchanged SCF trajectory is the
acceptance criterion.

The 8.39 s one-shot total is **not** an end-to-end improvement over the saved
v12 value of 8.29 s, despite the 0.145 s reduction in its SCF subtotal. Phase
instrumentation showed that the difference is in pre-SCF CUDA/process setup,
not the DFT kernels. Repeated current-code warm-cache runs measured 7.06,
7.17, and 7.29 s internally (7.74, 7.82, and 7.93 s for the complete Python
process). Consequently, compare medians from interleaved runs with identical
cache, console, and power-state conditions; do not rank two implementations
from one `Total accelerated Python wall time` line. New reports expose both
`Pre-SCF setup/reporting wall time` and `Post-SCF finalization/reporting` to
make that distinction visible.

The startup-recovery pass overlaps CUDA driver/device discovery with the
independent CPU reference construction. The final default naphthalene runs
used automatic symmetry and the canonical-order custom CUDA projector. Their
internal totals were 7.99 s for the first cold-driver process, followed by
6.03 s and 6.39 s, giving a three-run median of **6.39 s**; complete-process
times were 8.62, 6.65, and 7.01 s. Every one of the 300 printed eigenvalue
rows, all ten total energies, the final `-123.37042737 Ry`, and sector counts
`9 8 8 9 7 9 9 9` exactly matched `parsec_v10_custom_valid.out`. Thus the
validated default recovered and surpassed the historical 6.54 s internal
target without changing the DFT or eigensolver trajectory. Set
`PARSEC_OVERLAP_CUDA_INITIALIZATION=0` to restore sequential initialization
for profiling; it is not the measured-fast default.

The subsequent fast-start pass removed full-grid Laplacian construction from
exact reduced-operator cache hits, replaced repeated full-buffer hashing with
validated upstream keys, reused the cached totally symmetric operator for
native Hartree CG, and memory-mapped the 55 MiB native multipole table. On the
same naphthalene case, operator hashing fell from 0.119 s to 0.0001 s and
Hartree cache restoration from 0.539 s to 0.050--0.078 s. A fresh process was
still dominated by a machine-variable 2.65 s CUDA context initialization.
Through the resident worker, backend resolution took 0.00001 s, pre-SCF setup
fell to 0.464 s, and the second complete internal run took **4.53 s** (4.71 s
client round trip), versus 6.69 s for the stored v4 cache-hit run. All ten SCF
energies, the final `-123.37042737 Ry`, and sector counts remained unchanged.

The next resident pass added exact in-process static-system/operator reuse,
an exact finite-difference-NNZ cache, workload-sized OpenMP teams for repeated
Hartree/CA-LDA vector loops, and one-thread OpenBLAS for the tiny projected
host eigensystems. Against `parsec_resident_optimized.out` at 5.00 s, three
warm naphthalene runs measured **4.21, 4.23, and 4.19 s**. The complete ten-
step energy trajectory was bit-identical at printed precision; all 300 printed
eigenvalues differed by at most `2.0e-10 Ry`, and the final energy and sector
counts remained `-123.37042737 Ry` and `9 8 8 9 7 9 9 9`.

The fixed-point symmetry and CHEBDAV/Hartree audit then used Si28H36 as the
large-sector acceptance case. Exact stabilizer selection produced sector
dimensions `182718 178378` from 361,096 full-grid points. Against the prior
25.17 s full-orbital-grid validation, the retained default measured 22.68 s
total, 19.164 s SCF, 16.731 s diagonalization, and 1.937 s Hartree. It
converged in 12 SCF steps to `-263.51147823 Ry`; the full-grid result was
`-263.51147864 Ry`, a `4.1e-7 Ry` difference, and the largest final printed
eigenvalue change was `1.11e-4 Ry` at the input's `1e-4` diagonal tolerance.
The retained changes are exact stabilizer-aware sectors, contiguous-workspace
CHEBDAV projection, and chronological Hartree initialization. Fortran-order
Davidson storage, an alternate in-place GEMM, single-GPU sector streams, and
PyAMG were measured slower or unstable and are not production defaults.

The cold-first-calculation pass retained three further arithmetic-preserving
changes. For naphthalene, exact integer symmetry maps reduced geometry and
representation construction from about `0.69 + 0.34 s` to `0.35 + 0.18 s`;
four-way cache-miss operator assembly reduced the operator build from
`1.88--1.91 s` to `1.59 s`. A six-run fresh-cache A/B comparison measured
median complete times of `7.66 s` with one worker and `7.31 s` with four.
The contiguous full-workspace CHEBDAV Ritz
projection reduced its fresh-cache median from `8.35 s` to `7.46 s`, with the
projected stage falling from about `1.06 s` to `0.106 s`. For Si28H36, that
Ritz change reduced projected work from about `3.00 s` to `2.39 s`, while the
fused active-prefix update reduced the initial eigensolver from `13.63 s` to
`13.16 s` and diagonalization from `16.11 s` to `15.63 s`. Parallel two-sector
assembly reduced that component, but its six-run complete-time median was
`22.97 s` versus `22.93 s` serial, so the adaptive production policy retains
one worker for at most two representations. Every retained Si run kept
the 12-step trajectory and final `-263.51147823 Ry`; every retained
naphthalene run kept all ten printed energies and `-123.37042737 Ry`.
Standalone totals still vary by roughly one to two seconds with Windows CUDA
driver startup, so retain/reject decisions used interleaved stage timings and
fresh molecule-specific caches. Audited one-prefix CGS, CholeskyQR2, a
column-major Davidson basis, a 12-vector Davidson block, and deferred Hartree
cache publication were slower and were removed.
