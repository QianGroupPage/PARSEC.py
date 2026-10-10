# GPU setup and memory profile

The launch settings under "Build and run" also include the subsequently
validated Ritz workspace and large-A100 streaming-boundary profile described in
[GPU_SFC_OPTIMIZATION.md](GPU_SFC_OPTIMIZATION.md). The measurements below
document the earlier setup-only release.

This profile combines FP64 GPU ionic fields, support-localized nonlocal
projectors, bounded native symmetry-operator assembly, and column-major
filter output. It builds on the existing GPU Hartree CG, device-resident
orbitals, and CUDA graph filtering. The Hamiltonian, functional, radial
cutoffs, Poisson tolerance, and SCF thresholds are unchanged.

## Build and run

Use the Python/CuPy environment already configured for the cluster. Rebuild
the native extension from this checkout; an older installed wheel cannot
provide `reduce_sector_csr`.

```bash
python -m pip install --no-deps /path/to/parsec_python
cd /path/to/input-directory
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PARSEC_IONIC_BACKEND=cupy PARSEC_IONIC_GPU_COUNT=auto
export PARSEC_NATIVE_PROJECTOR_LOOKUP=1 PARSEC_NATIVE_SECTOR_ASSEMBLY=1
export PARSEC_CUPY_FILTER_COLUMN_MAJOR=1 PARSEC_CUPY_FILTER_GRAPHS=1
export PARSEC_CUPY_DEVICE_RANDOM=1 PARSEC_CUPY_RITZ_ROTATION=reuse
export PARSEC_HARTREE_LINEAR_BACKEND=cupy PARSEC_HARTREE_BOUNDARY_BACKEND=auto
export PARSEC_CUPY_MIXED_FILTER=off PARSEC_CUPY_SECTOR_STATE_STORAGE=device
export PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR=pool
PARSEC_CUPY_DEVICES=0,1,2,3 \
  python /path/to/parsec_python/src/parsec_python/main.py --backend auto parsec.in --debug
```

Run on allocated compute resources. On Perlmutter the benchmark used one
exclusive node with four A100 80 GB GPUs and a single task bound to 32 CPUs,
with `OMP_NUM_THREADS=16` and one BLAS thread. Select `0`, `0,1`, or `0,1,2`
to restrict the GPU count. Device IDs are local to `CUDA_VISIBLE_DEVICES`.
The exports are the launch settings of the measurements in
[GPU_SFC_OPTIMIZATION.md](GPU_SFC_OPTIMIZATION.md), which continues this
document; [GPU_COMPACT_OPTIMIZATION.md](GPU_COMPACT_OPTIMIZATION.md) adds six.
The measurements below were made before two of them existed,
`PARSEC_HARTREE_BOUNDARY_BACKEND=auto` and `PARSEC_CUPY_RITZ_ROTATION=reuse`,
and with `PARSEC_IONIC_GPU_COUNT=1`; `auto` came later.
`scripts/run_multi_gpu_scf.sh`,
the launch script for one or several nodes, sets neither
`PARSEC_CUPY_MIXED_FILTER` nor the storage of the sector states and their
allocator and leaves them to the code.

For an existing Perlmutter allocation, an explicit step avoids inheriting
a GPU binding that exposes fewer devices than the calculation selects:

```bash
srun --exclusive -N1 -n1 -c32 --gpus-per-task=4 --gpu-bind=none \
  --cpu-bind=cores env PARSEC_CUPY_DEVICES=0,1,2,3 \
  python /path/to/parsec_python/src/parsec_python/main.py --backend auto parsec.in --debug
```

Here the 32 allowed logical CPUs are 16 physical cores with SMT on the
measured node. Four GPUs do not imply four times as many CPU cores; keep
the CPU allocation and thread settings fixed for a GPU scaling curve.

The benchmark uses no persistent symmetry cache and starts a new process for
each measurement. This exposes first-run setup costs. Its runs passed
`--no-symmetry-cache`; no cache has since become the default of the command
line and of the MPI runner, so the same run needs no flag. Repeated
calculations of one structure and grid may name the exact-key cache with
`--symmetry-cache DIRECTORY`; `README.md` gives what it costs and returns.

## Retained implementations

| Control | Optimized setting | Work performed |
|---|---|---|
| `PARSEC_IONIC_BACKEND` | `cupy` | Local ionic potential, initial valence density, and NLCC density on GPU |
| `PARSEC_IONIC_GPU_COUNT` | `auto` (the code default) | Every GPU of an MPI rank that solves symmetry sectors sums a slab of the grid; any other process uses one GPU, the former default. A number takes the first that many GPUs of the process |
| `PARSEC_NATIVE_PROJECTOR_LOOKUP` | `1` | Search only a conservative projector support box, then apply the exact native spherical cutoff |
| `PARSEC_NATIVE_SECTOR_ASSEMBLY` | `1` | Build reduced CSR rows in C++ with per-thread row scratch |
| `PARSEC_CUPY_FILTER_COLUMN_MAJOR` | `1` | Produce the layout already required by the large generalized Ritz projection |
| `PARSEC_OVERLAP_IONIC_SETUP` | `1` (default) | Build the GPU ionic fields and the ion-ion energy on a thread beside symmetry detection and the sector operators; `0` keeps them in line |
| `PARSEC_OVERLAP_CUDA_CONTEXTS` | `1` (default) | `benchmarks/mpi_full_scf.py` only: a rank creates its CUDA contexts on a thread beside its static preparation; `0` creates them before it |
| `PARSEC_CUDA_CONTEXT_CREATION` | `driver` (default) | That thread creates each context through the CUDA driver library, outside Python's interpreter lock; `runtime` leaves them to CuPy's first calls on each device, which hold the lock, as before |
| `PARSEC_SYMMETRY_FAST_MAPS` | `1` (default) | Symmetry detection and the representation phases without passes over three coordinates per grid point; `0` takes the former routes |
| `PARSEC_FAST_GRID` | `1` (default) | The cluster grid slab by slab from three axis vectors, without the coordinate triples of the bounding cube; `0` calls the builder of the reference package |

Ionic GPU threads own independent grid points and visit atoms in input order.
They retain the full Coulomb tail and the native interpolation/origin rules.
No atom-by-grid matrix or atomic reduction is introduced. Temporary setup
allocations bypass the orbital pool and are freed on their owning device.
The devices are among those of `PARSEC_CUPY_DEVICES`, each with one contiguous
slab of grid points. A point is summed by one thread whichever device holds
it, the kernel is compiled with `--fmad=false`, and its arithmetic is sums,
products, quotients and square roots only, so the fields do not depend on
the number of devices.

`auto` takes a device only where the calculation is known to use it. The
sums run before the symmetry of the structure is detected. An MPI rank that
solves symmetry sectors (the full-SCF runner, also with one rank) is given
sector groups that cover its devices, and the runner creates a CUDA context
on each, so it sums on all of them. With `PARSEC_OVERLAP_CUDA_CONTEXTS` (the
default, see below) the runner creates the contexts beside the preparation,
and a sum that reaches a device first waits for its context or creates it:
which thread creates a context is then decided by the CUDA runtime, its
creation can be part of `ionic_setup_seconds`, and the first memory
observation of the runner can include the arrays of the sums. A process
without a sector context
(`main.py`, the serial control of the runner) may go on to solve on the full
grid, on one device, and sums on one: on the others a sum would create
contexts that nothing uses afterwards (measured on A100: 426 MiB of device
memory each, and 0.7 to 1.9 s to create four, the first included). Name a
count to sum on several devices there; the
contexts are created inside the sums then, which was not measured.
`ionic_gpu_devices` in the backend report lists the devices the sums ran on
(`none` where no GPU sum ran in the preparation), next to
`ionic_gpu_count_requested`. Sums on their own thread (below) are reported
when the thread is joined, and read `PARSEC_CUPY_DEVICES=current` as the
thread that prepares does, not from their own.

Only the root rank of an MPI run (or the one process of a serial run) builds
the local ionic potential, the two densities and the ion-ion energy, and
nothing reads them before the XC evaluator is set up. With the GPU fields the
reference preparation therefore returns without them, and one thread
(`parsec-ionic-setup`) builds them with the same calls while the calling
thread detects the symmetry and builds the sector operators. The driver joins
the thread before the XC evaluator and the Poisson graph capture. The arrays
are the ones of the in-line stages bit for bit. The report gives the thread
time, the wait at the join and the hidden time (`ionic_setup*`); the stage
times of these fields in `preparation` are measured on the thread, and its
`total_seconds` is the wall time of the reference preparation without them
(`reference_preparation_seconds`), which the text report says under its setup
timings. A resident process (`PARSEC_ACCELERATED_RESIDENT=1`) and the
C++/OpenMP ionic builders keep the in-line stages.

The saving is bounded by the host work between the start of the thread and
its join. The kernels run on the default stream of the ionic device or
devices, where work of the calling thread waits behind a running kernel.
Measured together with the context thread and the symmetry maps described
below, the three switches on against off, in first runs on A100 nodes: 129.8
instead of 136.6 s for 23,768 electrons on 16 GPUs and 86.1 instead of 90.3 s
for 10,456 on 4, with the same energies to the last bit. Those runs summed on
one device and reduced the sector stencils from the full-grid matrix: the
thread took 5.11 s (kernels 2.75 + 1.90 s) in the 16-GPU run and nothing
waited at its join, because the symmetry stages and the sector reduction on
the host (5.70 s) came before the first sector upload. With the
defaults of this version those host stages are shorter (the maps below, the
sector stencils built from the grid) and the root sums on four devices.
First runs with all of it together, in one allocation: for 23,768 electrons
on 16 GPUs the thread took 2.64, 2.66 and 3.26 s in three runs and its join
waited 26 and 24 microseconds and 0.27 s; the time before the SCF was 6.8,
7.0 and 7.2 s, against 13.2 s with the three switches off (the sums in
line on four devices) and 25.8 s with every one of those defaults switched
back. With the sums on one device the thread took 5.18 s, its join waited
0.34 s and the time before the SCF was 9.2 s. For 10,456 electrons on 4
GPUs the thread took 0.83 to 1.53 s without a wait, and the time before the
SCF was 6.7 to 7.3 s against 8.9 and 18.5 s. In the 39 runs of that
allocation the first memory observation read 426 MiB on every device of
every rank, the context alone, with the overlap and without: the arrays of
the sums were in none of them. Where the host stages are shorter than the
thread, the rest is waited for at the join (`ionic_setup_wait_seconds`) or
in the sector upload to an ionic device (`initialization_seconds` of the
backend), so the saving of this overlap does not add to that of shorter
symmetry or sector stages. The comparison to make is the time of a rank
before its first SCF step with `PARSEC_OVERLAP_IONIC_SETUP=0` and `1`, which
was not run apart from the two other switches.

`benchmarks/mpi_full_scf.py` used to create the CUDA contexts of a rank one
after another before its preparation: 0.7-0.9 s per rank with four devices in
the 8- and 16-GPU runs, and 1.2, 1.3-1.5 and 1.7-2.1 s for one, two and four
devices in one-rank runs. By default a thread (`parsec-cuda-contexts`) now
creates them while the calling thread builds the grid, the finite differences
and the projectors, none of which needs a device; a thread that reaches a
device whose context is still being created waits for it inside the CUDA
runtime. `PARSEC_OVERLAP_CUDA_CONTEXTS=0` restores the former order. Each rank
records `cuda_context_initialization_overlapped`, the thread time and the wait
at its join. In both orders the `nvidia-smi` sampler is started once the
contexts exist (with the overlap, by that thread), so its start-up is beside
their creation in neither. The preparing thread still asks the driver for the
device count when it resolves the backend and waits there for whatever driver
initialization is under way; `backend_resolution_seconds` shows that wait. In
the measured runs named above the thread took 1.1 s on the root of the 16-GPU
run and 1.9 s in the 4-GPU run, and its join waited less than 0.1 ms. The
switch was not measured apart from the two others. The GPU ionic sums on
their own thread can reach a device before this thread does; see the note on
`auto` above.

Measured apart since, the overlap hid almost nothing. First runs of 3,480
electrons on A100 nodes, the default against `PARSEC_OVERLAP_CUDA_CONTEXTS=0`
in one allocation: the root prepared in 2.31 and 2.36 s against 2.25 and 2.35
s on 16 GPUs (the thread took 0.72 to 0.90 s on its four devices), in 2.76
and 2.59 against 2.55 and 2.60 s on 8, in 3.78 to 3.95 against 3.80 to 3.92 s
on 4 (1.73 to 1.82 s) and in 3.34 and 3.40 against 3.35 and 3.37 s on 1 (1.21
s). The reason is Python's interpreter lock. CuPy creates the context of a
device inside its first runtime call that needs one and holds the lock for
that call, so the thread that prepares ran between two devices only: on a
laptop GPU a pure-Python loop of the main thread stood still for 97 to 100 %
of the time in which another thread created a context that way.

By default the thread now creates each context through the CUDA driver
library first (`libcuda.so.1`, `nvcuda.dll` on Windows): `cuInit`, then per
device `cuDeviceGet`, `cuDeviceGetPCIBusId` and `cuDevicePrimaryCtxRetain`,
called with ctypes, which releases the lock around a foreign call. The
primary context of a device is the one the CUDA runtime and so CuPy use: the
calls of CuPy that follow in the thread find it, and the reference taken
here is kept for the life of the process, as the runtime keeps its own. A
context is created this way only where the driver gives the device of an
index the PCI address that CuPy reports for it; where the library cannot be
loaded, a call fails or the addresses differ, CuPy creates the context as
before and raises what it raised before.
`PARSEC_CUDA_CONTEXT_CREATION=runtime` restores the creation inside the
calls of CuPy. The contexts are the same ones either way, so no array,
kernel, stream or result changes; the first memory observation of a rank is
taken where it was. Without the overlap (`PARSEC_OVERLAP_CUDA_CONTEXTS=0`,
which the serial control names) nothing runs beside the contexts and CuPy
creates them, whatever this switch says. Each rank records
`cuda_context_driver_devices`, the devices whose context the thread created
through the library, and `cuda_context_driver_seconds`, its time inside
those calls.

What this creation can hide is bounded by the
seconds of the thread, 0.7 to 0.9 s of a rank with four devices in the 8-
and 16-GPU runs, and by the host work of the preparing thread before it
needs a device. A thread that selects a device through CuPy while its
context is being created still waits for it with the lock held: the GPU
ionic sums of the root start about half a second into the preparation of
3,480 electrons, after the backend resolution and the reference preparation,
and the device count that the backend resolution asks of the runtime may
wait for the start of the driver. On the laptop GPU with CuPy 14.0.1,
through `_prepare_on_devices` of the runner in a new process each, three per
setting, with a pure-Python loop as the preparation: with `driver` the
thread took 0.20 to 0.24 s, 0.19 to 0.22 s of them inside the driver calls,
and the loop never waited longer than 1.7 ms; with `runtime` the thread took
0.15 to 0.18 s and the loop stood still for 0.10 to 0.13 s of them. A later
probe there timed each call, three new processes per setting: the loop
stopped for 44 to 67 ms of the 63 to 110 ms of CuPy's two calls that start
the driver and create the context, and for 0.5 ms at most in the 116 to 158
ms of the library's load and calls. How much of CuPy's creation holds the
lock thus varies between runs; the driver calls do not hold it. The
device records were the same, and arrays and kernels of CuPy worked on the
context of either. Through the class alone, without CuPy: the driver started
in 0.04 s, a context took 0.11 to 0.14 s, and a wrong address left the
device without a context. The thread itself is thus a few hundredths of a
second longer there with the driver calls; what it costs on an A100 is in
the runs below.
The run to judge it by is 3,480 electrons on 16, 4 and 1 GPUs, default
against `runtime`, read at `preparation_seconds` of every rank and at the
time before the SCF. A pair decides only where
`cuda_context_driver_devices` lists every device of every rank: a rank with
fewer was left to CuPy, on Linux as anywhere. What a rank kept under the
lock is its thread
time less `cuda_context_driver_seconds`. Beside the times, the pair has to
agree in what no creation may change: the total energy, the peak and
`graph_captures` (51, 57, 69 and 69 captures of the root on 16, 8, 4 and 1
GPUs in one measured series, none repeated; a capture that was invalidated is
repeated and counted there, and what the creation of a context on another
device does to an open capture has not been measured). The first memory
observation of a device, 426 MiB on every device of that series, shows a
context that is not the runtime's own, unless the ionic sums reached the
device first.
Those pairs have run since: 3,480
electrons on 16, 8, 4 and 1 GPUs, two or three runs per setting, and 10,456
and 23,768 electrons on 16. With `driver` every rank listed every one of its
devices, and its thread spent all but 0.00 to 0.09 s of its 0.77 to 1.71 s
inside the driver calls. The preparation of the root took 1.62 to 1.77
against 2.57 to 2.70 s on 16 GPUs (the other ranks 1.45 to 1.58 against 2.46
to 2.59), 1.76 to 1.79 against 2.39 to 2.51 on 8, 2.83 to 3.03 against 3.70
to 4.44 on 4 and 2.65 to 2.74 against 3.22 to 3.26 on 1; 2.89 against 3.82 s
for 10,456 electrons and 5.48 against 6.33 s for 23,768 on 16 GPUs. The
total energies were the same to the last bit, the captures 51, 57, 69 and 69
(53 for 10,456 electrons) with none repeated, and the peaks equal on 8 and 1
GPUs, within 2 MiB on 16 and within 94 MiB of each other on 4, where they
vary from run to run under either setting.

Every rank builds the cluster grid before anything else: 1.57 s of every rank
in a 16-GPU run of 23,768 electrons and 0.29 s for 3,480. The builder of the
reference package writes three integers and three floats for every point of
the bounding cube (29.5 million points around a grid of 15.1 million), tests
them and copies the active rows out. With `PARSEC_FAST_GRID` (the default)
`Grid/cluster.py` builds the same arrays from three axis vectors. A slab of
constant x is a table of y and z positions: its active points, their integer
and physical coordinates and their lookup entries are written while the slab
is in the cache, and the only array of the size of the cube is one Boolean
per point. Each coordinate is the reference expression on one axis value. A
box is decided axis by axis by the reference test. For a sphere the sum of
squares of a slab decides every point farther than a relative 1e-9 from the
sphere, and the reference test itself, on the coordinates of the point,
decides the others: its sum can differ from another order of the same three
squares in the last place only. The reference preparation takes the builder
as `grid_builder`, like its other builders, and `grid_builder` in the backend
report says which one ran; `PARSEC_FAST_GRID=0` passes none. The tests
compare every array with the reference one (type, shape, layout, bytes) for
spheres and boxes with and without points on the surface, shifted and
unshifted, down to an empty grid, and for radii within one unit in the last
place of a grid point. On a workstation the grids of the inputs with 3,480,
19,392, 23,768, 29,576 and 39,368 electrons are the same bytes and take 0.07,
0.31, 0.44, 0.40 and 0.55 s instead of 0.37, 1.57, 2.07, 2.09 and 2.68 s
(best of five), and the transient memory above the 0.90 GiB of the grid of
23,768 electrons falls from 1.74 to 0.03 GiB. Those seconds are a
workstation's. On A100 nodes the switch ran on and off in one allocation
together with `PARSEC_SYMMETRY_SCF_BUFFERS` and
`PARSEC_CUPY_DENSITY_COLLECTION` (3,480 electrons on 4, 8 and 16 GPUs, 10,456
and 19,392 on 4, 23,768 on 16), with the same total energies to the last bit.

Every rank detects the symmetry and builds the representation phases before
it can build its sectors (3.43 + 1.22 s in the 16-GPU run of 23,768 electrons).
With `PARSEC_SYMMETRY_FAST_MAPS` (the default):

- the atoms of a species are grouped once, and a species whose transformed
  atoms have one candidate each inside the matching radius is decided by one
  k-d tree query, the distance formula of the dense construction and a test
  that the partners differ; a species with a second candidate inside the
  radius of an atom is decided by the augmenting paths as before;
- the row map of an operation is read out of the grid lookup table: a signed
  permutation of the lattice is a transposed, mirrored and shifted window of
  that table, and the rows are numbered along it, so the images of all rows
  are one masked read of the window. No coordinate of a point is read. A grid
  whose rows are numbered otherwise keeps the gather through the coordinates;
- the orbits are numbered by a scatter of the representatives instead of a
  cumulative count, and their smallest images are taken map by map;
- the phases of every representation are read from one operation label per
  row, after a check that every image of a representative lies in its own
  orbit; where no operation other than the identity fixes a representative,
  every representation admits every orbit and no orbit is tested again.

Every Boolean, map and phase equals the former one: the tests compare them
with the former constructions on grids with and without points on the
symmetry planes and on clusters with atoms on the axes and planes. On a
workstation, with the atoms and the grid of the 23,768-electron cluster, the
two stages take 0.37 + 0.44 s instead of 2.61 + 1.13 s, and 0.26 + 0.31
instead of 1.81 + 0.79 s for 14,680 electrons; the arrays are the same bytes.
All phases are still built on every rank: the array is read by the cache
writer and the orbital export, and it now costs about 0.1 s.

KB projector rows are sorted back into the original grid order. Species
radial tables are shared within one construction. The native operator path
retains stabilizer selection, character phases, multiplicity normalization,
duplicate coalescing, and the symmetry audit. It supports 32- and 64-bit CSR
indices and avoids large COO, mask, and normalization temporaries. The
outer representation loop is serial; each native row pass uses OpenMP.

`PARSEC_NATIVE_SECTOR_ASSEMBLY` now acts on `PARSEC_SECTOR_STENCIL=csr` only.
By default (`direct`) a rank builds the packed stencil of each of its sectors
from the grid and forms no full-grid matrix; see "Sector stencils from the
grid" below. The new arrays are those of the setting `1` of this table. The
variable is 0 when unset, so a run started without this profile used the
SciPy reduction, from which the default route now differs by round-off.

Column-major filtering removes an extra tall wavefunction conversion before
Rayleigh--Ritz. Filter arithmetic and recurrence order are unchanged. It is
a memory optimization first; small timing changes must be assessed from
repeated complete calculations.

## Sector stencils from the grid

On a first run every rank used to build the whole full-grid `-nabla_FD^2` as
CSR in one C++ thread, cut its sectors out of it, audit them and repack them
slot by slot. In one measured series this was 11.9 s of the 26.1 s before the
SCF for 23,768 valence electrons on 16 GPUs (finite difference 6.08 s,
operator build 5.78 s), and the matrix, held twice while it is built, was
11.2 GiB of the 13.1 GiB host high-water mark.

`Symmetry/sector_stencil.py` writes the packed arrays of a sector from the
grid lookup, the orbit and phase maps and the stencil coefficients. The
native kernel `build_sector_stencil` (`native/sector_stencil.cpp`) does it in
two OpenMP passes over the sector rows, in blocks of 4096 rows, and then
compares every entry with its transpose, the audit the former route made with
SciPy. Host tests compare the arrays with the former route for every sector,
on free orbits and with grid points and atoms on the symmetry planes and
axes, and for sectors of seven blocks on one to seven threads.

The NumPy builder serves an extension without the kernel. It works in blocks
of 16,384 rows and audits the transpose on coefficient codes (five bytes an
entry), one sector at a time while the other sectors go on building. Without
the lock four sectors of the larger grid took 7.6 s and 8.2 GiB; with it,
the figures of the table.

Host measurements on a 16-core workstation, `OMP_NUM_THREADS=16`, the grids
of the 3,480- and 23,768-electron clusters (2,710,168 and 15,147,328 points),
reference operators and
`load_or_build_reduced_operators` through the driver's functions, no GPU. The
operator build includes the projector reduction, which did not change:

| Grid, sectors built | Route | Finite difference + operator build | Peak working set |
|---|---|---|---|
| 3,480 electrons, 1 | `csr` | 0.94 + 1.00 s | 2.24 GiB |
| 3,480 electrons, 1 | `direct` (native kernel) | 0.014 + 0.09 s | 0.54 GiB |
| 3,480 electrons, 1 | `numpy` | 0.014 + 0.68 s | 0.65 GiB |
| 3,480 electrons, 4 | `csr` | 0.92 + 2.03 s | 2.74 GiB |
| 3,480 electrons, 4 | `direct` (native kernel) | 0.014 + 0.30 s | 0.68 GiB |
| 3,480 electrons, 4 | `numpy` | 0.014 + 1.52 s | 1.06 GiB |
| 23,768 electrons, 1 | `csr` | 8.93 + 5.89 s | 12.34 GiB |
| 23,768 electrons, 1 | `direct` (native kernel) | 0.13 + 0.65 s | 2.70 GiB |
| 23,768 electrons, 1 | `numpy` | 0.10 + 3.77 s | 3.29 GiB |
| 23,768 electrons, 2 | `csr` | 5.97 + 12.00 s | not recorded |
| 23,768 electrons, 2 | `direct` (native kernel) | 0.09 + 1.08 s | not recorded |
| 23,768 electrons, 4 | `direct` (native kernel) | 0.11 + 1.93 s | 3.60 GiB |
| 23,768 electrons, 4 | `numpy` | 0.10 + 8.7 to 10.7 s (three runs) | 5.42 GiB |

One run each unless stated; other work on the workstation moved single
figures by up to a half. The `csr` runs of the larger grid nearly exhausted
its memory (the 8.93 s matrix build was 5.97 s in the second run), and four
sectors of that grid on `csr` did not fit. The native kernel alone takes
0.33 s for a sector of the larger grid on 16 threads and 2.6 s on one.

A run whose extension lacks the kernel still forms no full-grid matrix. For
four sectors on one rank (one to four GPUs) of the smaller grid it takes
about half the time of `csr` and two fifths of its peak, where the kernel
takes a ninth and a quarter: rebuild the extension for the full gain.
`PARSEC_SYMMETRY_OPERATOR_WORKERS=1` lowers the NumPy peak further (0.90 GiB
for those four sectors) at 2.7 s.

`benchmarks/sector_stencil_routes.py` repeats this on any host: for each route
it prepares the sector operators of an input in a fresh process and prints the
stage times, the high-water mark and a SHA-256 of every sector's arrays. It
runs `csr` with `PARSEC_NATIVE_SECTOR_ASSEMBLY=1` and refuses the comparison
when the variable is set to anything else. On the larger grid `csr` and
`direct` gave the same digests for sectors 0 and 1, and `direct` and `numpy`
for all four:

```text
0 2de03a5337a9f3e289d2aa14040730dd0096f1561a603e0eb591ce3a159a71cf
1 f45615ca4673dba7b9a86136cc23e104f83bf3431bce992550eb6e5dea1f06b9
2 800980e2c053141fd3acdac97f22ddbd9c8c22d21bad53b08404dca7f1b79c2a
3 45cf4592a2aefce3ed345b280151b52f2ae790de4be64ed66c7c0361be9c8802
```

An MSVC build on Windows and a GCC 13.3 build under Linux (`-O3
-fno-fast-math -fopenmp`) gave these digests. Under Linux the high-water mark
of the process for one sector of the larger grid was 12.34 GiB on `csr`,
2.52 GiB on `direct` and 3.60 GiB on `numpy` (3.35 s), and for four sectors
3.67 GiB on `direct` and 5.81 GiB on `numpy` (7.7 s).

Complete SCF runs on one 8 GiB laptop GPU (C87H76 and C185H124, 424 and 864
valence electrons, the solver controls of the benchmark inputs) gave bitwise the same
energies, eigenvalues, density and Hartree potential on the three routes, with
and without a symmetry cache, the former route taken with
`PARSEC_NATIVE_SECTOR_ASSEMBLY=1`. The runs of a comparison need the same
`OMP_NUM_THREADS`: between 16 and 28 host threads the total energy of C87H76
moved by 4e-12 Ry on every route, the former one included.

First runs on A100 nodes, `direct` (the native kernel) against `csr` in one
allocation, with the same energies to the last bit and strict parity against
the serial references: 125.2 instead of 136.6 s for 23,768 electrons on 16
GPUs, with a host high-water mark of 5.6 instead of 13.1 GiB on the root rank
and 3.7 GiB on the others; 79.0 instead of 90.5 s and 3.6 instead of 9.6 GiB
for 10,456 electrons on 4 GPUs; 9.5 instead of 11.2 s for 3,480 electrons on
16. Those runs kept the slot-major stencil, the former symmetry maps and the
ionic fields in line. With the other defaults of this version beside it, in
a later allocation, `csr` switched back alone took 80.3 instead of 69.2 and
68.9 s for 10,456 electrons on 4 GPUs (root high-water mark 9.3 instead of
3.2 GiB), whose sectors pack affine tiles of 16 from the stencil of either
route, and 125.3 instead of 114.1 and 114.4 s for 23,768 on 16 (13.1
instead of 5.4 and 5.8 GiB), again with the same energies to the last bit.

## Validation and interpretation

Two unchanged nanodiamond geometries are used: 3,480 and 5,264 **valence
electrons**, calculated as `4 * number_of_C + number_of_H`. No geometry
generation or xTB optimization is performed by these changes.

Compare like-for-like fresh processes on the same node, with the same CPU
thread counts, inputs, FP64 policy, and convergence thresholds. Warm-up runs
are excluded. Program timing excludes Python imports/input parsing and NPZ
export; complete-process time is recorded separately. RSS is sampled every
0.1 s and GPU memory every 1 s, so short-lived peaks may be missed.

Numerical tests cover radial knots/cutoffs, Coulomb tails, NLCC, VCD and
non-VCD density, shifted grids, out-of-domain atoms, symmetry stabilizers,
32-/64-bit indices, and multi-device filtering. Complete SCFs independently
compare energy, density, eigenvalues, occupations, electron count, and
iteration count. Full-field validation also checks all archived potentials,
energy components, and the complete SCF trajectory.

Four-GPU ionic setup did not improve complete program time in the pilot,
so that profile retained one GPU for this stage. Larger cases decided for
all devices of a rank, which is now the default of the full-SCF runner (see
`auto` above): for 23,768 electrons on four nodes
(16 A100) the local ionic sum fell from 2.68 to 0.92 s, the initial density
from 1.77 to 0.98 s and the time before the SCF from 25.5 to 23.6 s, with
the same total energy to the last bit and the same sampled peak of device
memory; for 14,680 electrons on one node the sums fell from 1.21 to 0.91 s
and from 0.78 to 0.26 s and the program time did not change (160.8 and
161.7 s). ELPA 2026.02.002 was screened
separately on 442-, 665-, and 1500-dimensional standard dense problems with
residual/orthogonality checks. This single-process, host-interface experiment
does not measure MPI ELPA scaling. ELPA is not a dependency of this profile.

By default the whole small generalized Ritz problem is solved on the device
(`PARSEC_CUPY_RITZ_DENSE_BACKEND=device`, see `README.md`), which reads no
`PARSEC_CUPY_RITZ_EIGH_BACKEND`. That variable belongs to the host solve,
`PARSEC_CUPY_RITZ_DENSE_BACKEND=host`: there an optional `cupy` sends only
the whitened dense standard eigensystem to cuSOLVER. Cholesky whitening,
conditioning checks, and coefficient orthogonality audits remain on the
host, unchanged. Its default is `host`; the general execution profile does
not enable this option. The numbers of this passage were measured with the
host solve.
On the 5,264-electron case, three complete four-GPU runs reduced the median
program time from 137.23 to 135.10 s and SCF time from 108.83 to 106.96 s.
This is a small, workload-specific improvement, not the much larger speedup
of the isolated dense microbenchmark. On 3,480 electrons the 49.69 -> 49.44 s
difference overlaps run-to-run variation, so it did not justify changing
the default of the host solve. Both cases pass the strict physical parity
gate and retain their SCF iteration counts, but cuSOLVER results are not
bitwise identical to the host eigensolver.
This option is not a memory optimization: the large-case median sampled
GPU peak rose slightly from 21.68 to 21.80 GiB and host RSS from 5.50 to
5.62 GiB. Within the host solve, `host` remains preferable when minimizing
memory use.

```bash
# with the launch settings of "Build and run" in GPU_SETUP_OPTIMIZATION.md exported
PARSEC_CUPY_RITZ_DENSE_BACKEND=host PARSEC_CUPY_RITZ_EIGH_BACKEND=cupy \
  PARSEC_CUPY_DEVICES=0,1,2,3 \
  python /path/to/parsec_python/src/parsec_python/main.py --backend auto parsec.in --debug
```

The same-node four-GPU medians below use three measured runs per retained
variant, after a separate warm-up, with 16 OpenMP threads. Memory entries
are medians of sampled peaks, not exact allocator maxima.

| Valence electrons | Program seconds, previous -> retained | Host GiB | Largest per-GPU GiB |
|---:|---:|---:|---:|
| 3,480 | 70.37 -> 49.69 | 7.43 -> 3.91 | 12.15 -> 10.08 |
| 5,264 | 182.22 -> 137.23 | 10.83 -> 5.50 | 26.79 -> 21.68 |

Both systems preserve all 21 selected archived physical arrays and the
complete SCF history exactly in the independent four-GPU audit. Every
measured 1/2/3/4-GPU retained run also preserves energy, density, eigenvalues,
occupations, electron count, and SCF iteration count relative to the control.
These cases validate the tested inputs and settings, rather than every
possible pseudopotential, functional, or hardware configuration.

There are four independent symmetry sectors in these cases. On three GPUs
their assignment is 2+1+1; a threefold speedup should not be expected from
this scheduling scheme. Preparation and the shared Hartree work also limit
end-to-end scaling. Report SCF and complete-program speedups separately.

Tests assuming a CPU reference backend must unset externally forced GPU
Hartree settings before running.
