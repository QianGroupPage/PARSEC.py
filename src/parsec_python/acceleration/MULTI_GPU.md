# Multi-GPU SCF

The CuPy backend distributes exact symmetry representations across GPUs within
one process. Large orbital blocks remain on their assigned devices. The SCF
driver combines the eigenvalue spectrum and sector densities, so this is one
coupled physical calculation, not a collection of independent jobs. Across
nodes the same sectors are solved by one MPI rank per node
(`benchmarks/mpi_full_scf.py`), which gives a sector several devices where
its rank has more devices than sectors; the memory rules below cover both.

## Installation and launch

Install the native extension and a CuPy wheel matching the cluster CUDA
runtime, as described in the acceleration README. Build and run inside a Slurm
allocation. A representative two-H100 job (Lawrencium cluster, LBNL) is:

```bash
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=es2
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
set -euo pipefail
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0 cuda/12.8.0
source /path/to/venv/bin/activate
export OMP_NUM_THREADS=14
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PARSEC_CUPY_DEVICES=auto
export PARSEC_CUPY_MIXED_FILTER=off
export PARSEC_CUPY_STAGE_TIMING=1
python /path/to/parsec_python/src/parsec_python/main.py parsec.in --backend auto
```

Set the account and QoS from `sacctmgr show assoc user=$USER`. Follow the
[site's current examples](https://scienceit-docs.lbl.gov/hpc/running/script-examples/)
for CPU/GPU ratios and memory limits. Keep Slurm's `CUDA_VISIBLE_DEVICES`
unchanged. `PARSEC_CUPY_DEVICES=0`, `0,1`, or `0,1,2,3` selects a subset of
the GPUs allocated to this process. Use one Python task, not one driver per GPU.

The example holds native CPU work at 14 threads for comparable 1/2/4-GPU
tests. Additional CPUs allocated by Slurm are not automatically additional
workers. More OpenMP or BLAS threads can be slower; benchmark the chosen system.
`--backend auto` enables native Hartree with GPU orbitals when both backends
are installed. Check the reported backend and fallback reasons.

### Several nodes

`benchmarks/mpi_full_scf.py` runs one SCF calculation with one MPI rank per
node, each on the GPUs of its node: `--devices` names CuPy indices inside the
`CUDA_VISIBLE_DEVICES` of a rank (default `0,1,2,3`), and the runner sets
`PARSEC_CUPY_DEVICES` from it. It needs mpi4py built against the MPI of the
cluster (`experimental/README.md` describes the Cray MPICH environment of the
measurements) and stops where two ranks share a node. With one rank it also
ran the 1-, 2- and 4-GPU figures of the last measured series.

That series (first calculations on A100 nodes with four GPUs each, 3,480 to
39,368 electrons on 1 to 16 GPUs) was launched like this inside an allocation
of `N` nodes, with `-N 1 -n 1` and `--devices 0` or `0,1` for fewer than four
GPUs:

```bash
export PYTHONPATH=/path/to/checkout/src
export OMP_NUM_THREADS=16 OMP_PROC_BIND=false
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset OMP_PLACES
export PARSEC_IONIC_BACKEND=cupy PARSEC_HARTREE_LINEAR_BACKEND=cupy
export PARSEC_HARTREE_BOUNDARY_BACKEND=cupy
export PARSEC_NATIVE_PROJECTOR_LOOKUP=1 PARSEC_NATIVE_SECTOR_ASSEMBLY=1
export PARSEC_CUPY_DEVICE_RANDOM=1 PARSEC_CUPY_FILTER_GRAPHS=1
export PARSEC_CUPY_FILTER_COLUMN_MAJOR=1 PARSEC_CUPY_DISTRIBUTED_FILTER=1
export PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ=1 PARSEC_CUPY_RECYCLE_STATE=1
export PARSEC_CUPY_RITZ_ROTATION=reuse PARSEC_CUPY_RITZ_CONDITION=symmetric
export PARSEC_CUPY_RITZ_EIGH_BACKEND=cupy
export PARSEC_CUPY_RESIDENT_HARTREE=auto PARSEC_CUPY_RESIDENT_PREDICTOR=host
export PARSEC_CUPY_SECTOR_STATE_STORAGE=device
export PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR=pool
export PARSEC_CUPY_STAGE_TIMING=1
srun -N "$N" -n "$N" --ntasks-per-node=1 -c 32 --cpu-bind=cores \
  --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
  python -m parsec_python.acceleration.benchmarks.mpi_full_scf \
  --input parsec.in --output-dir NEW_DIRECTORY --devices 0,1,2,3
```

`scripts/run_multi_gpu_scf.sh` in the repository root holds these settings
and the `srun` line: `scripts/run_multi_gpu_scf.sh [--nodes N]
[--gpus-per-node G] parsec.in NEW_DIRECTORY`. It also sets
`MPICH_OFI_NIC_POLICY=GPU` for Cray MPICH, as the launch environment of the
series did, and leaves the storage of the sector states and their
allocator unset. Launched through it, six
configurations of the series, from 3,480 electrons on one GPU to 39,368 on
16, gave the totals of the measured runs to the last bit.

These are the settings of the measured runs, not the defaults of the code.
Unset, the ionic backend, the Hartree linear backend and the Hartree boundary
backend are `native`, the rotation is `allocate`, and the device random
numbers, the filter graphs, the column-major filter, the native sector
assembly, the generalized first Ritz step, the distributed filter, the
recycled state, the resident Hartree chain and the stage timing are off: a
run without them takes other routes than the measured ones.
`PARSEC_NATIVE_PROJECTOR_LOOKUP=1` and `PARSEC_CUPY_RESIDENT_PREDICTOR=host`
are the defaults, the Ritz condition follows the place of the small solve
when unset, which is `symmetric` for the default device solve, and
`PARSEC_CUPY_RITZ_EIGH_BACKEND` belongs to the host dense solve, which the
default device solve does not read. The storage of the sector states and
their allocator are `auto` when unset, which keeps a basis that does not fit
its device on the host instead of stopping (below); four runs of the series
left both unnamed. The inputs of the series set
`Eigensolver: chebff`, `FF_MaxIter: 4`, `Chebdav_Degree: 30` and
`Chebyshev_Degree: 15`; `tests/data/C795H300_parsec.in` of the reference
package is the smallest of them.

## Correctness and memory

Filtering is FP64 unless `PARSEC_CUPY_MIXED_FILTER` asks for the FP32 later
filter; `off`, as in the example, is the default. Run the
same input with one and multiple GPUs, checking convergence, final total
energy, density, electron count and occupations. Compare identical physical
settings: geometry, pseudopotentials, functional, grid, domain, states,
temperature and SCF tolerance. Degenerate eigenvectors can differ by rotations;
density and total energy are the meaningful invariants.

The domain of an input without `Boundary_Sphere_Radius` is chosen by the
default rule of the parser from the input and the pseudopotential files
alone, so it is the same on any number of devices and ranks; the
`PARSEC_HARTREE_*` switches of a launcher change the boundary of a run and
not its sphere. The MPI runner compares the radius and the Hartree tolerance
every rank resolved before any of them prepares and stops where they differ.
To compare a run with one made from an input that holds a radius, copy the
two lines `parsec.out` prints (`Boundary_Sphere_Radius` and
`Hartree_Boundary_Tolerance`): the radius alone leaves the fixed tolerance
of `1e-3 Ry` and, from 6,400 electrons up, a lower multipole order. The
root estimates after the SCF what the sphere adds to the energy and which
radius would meet the tolerance, on the host and inside the reported time
(0.3 to 1.2 s for 39,368 electrons on a workstation host);
`PARSEC_DOMAIN_REPORT=0` leaves that out for a run that is timed against one
of an earlier tree, without touching its results. This switch and the
others added with it, with their defaults, the values that restore the
former behaviour and what first runs on A100 nodes gave for each, are in one
table in `README.md`.

From an allocated two-GPU node:

```bash
PYTHONPATH=src python -m unittest parsec_python.acceleration.tests.test_multi_gpu -v
```

GPU-specific tests skip if fewer than two usable devices are available. A skip
does not establish that multi-GPU execution works. The tests cover source-device
density reduction of non-contiguous views, exact blocked orbital export,
unequal-sector scheduling, and cleanup on errors.

Each device retains its sector states and operator metadata. Concurrent sectors
on the same device can multiply temporary workspace, so the default scheduler
serializes them on that device. Explicit `PARSEC_CUPY_SECTOR_SCHEDULER=streams`
remains a profiling option. Density reduction transfers grid-length vectors,
not grid-by-state orbital blocks. Host wavefunction export avoids a full GPU
gather but cannot eliminate the final host matrix's memory requirement.

In a process with several devices the sectors that one of them solves one
after another have a CUDA stream each, and CuPy's memory pool keeps one free
list per stream: the Ritz buffer that a sector gave back (1 to 4 GiB) could
not serve the next sector, which took another, and both stayed until the
density step emptied the pool. With
two A100 for four sectors the sampled peak stood two such buffers above the
memory between steps (2.08 GiB for 3,480 electrons, 5.08 GiB for 10,456)
where four devices show one (1.04 and 2.55 GiB). A finished sector therefore
returns the unused pool blocks of its stream to the device, where the device
solves several sectors in turn on streams of their own
(`PARSEC_CUPY_SECTOR_POOL_RELEASE=1`, the default; `0` leaves them to the
density step, as before). It does so where the vectors of the sectors of
the process together reach `PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES` (512
MiB), the bytes from which the density step empties the pool: the blocks
would go back in the same SCF step anyway, and below that size they stay, so
that nothing is allocated again that the density step would have left in the
pool. That holds with the states on their devices. With the states spilled
to the host (`PARSEC_CUPY_SECTOR_STATE_STORAGE`) the density step empties no
device pool: there the release returns blocks that stayed cached before, so
that the device holds the blocks of one sector at a time, and every sector
takes its vectors and workspace from the driver again in each step. So it
does under the direct allocator, which a device that cannot hold the states
of its sectors is given together with the spill where both settings are
`auto` (below); the added allocations are those of a spill beside a pool
that still allocates (storage `host` on a device that can hold them, or
`PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR=pool` on one that cannot). No
arithmetic changes. `sector_pool_release` in the rank records of
`benchmarks/mpi_full_scf.py` counts the releases and the bytes per device.
On A100 nodes the release has run on two devices since, beside runs with
`PARSEC_CUPY_SECTOR_POOL_RELEASE=0` (3,480, 5,264 and 10,456 electrons;
`sector_pool_release` of a rank counts the releases); their peaks are not
given here. On one laptop GPU, with the four
sectors of a 424-electron cluster (164,882 rows, 403 MiB of vectors) given a
stream each and `PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES` lowered to 256 MiB,
so that the density step empties the pool as for a large system, the largest
reservation of the pool fell from 1,233 to 904 MiB (896 MiB with all four on
the default stream) with the same energies to the last bit and the same 418
blocks taken from the driver; at the default limit this cluster releases
nothing. With its states spilled to the host the largest reservation fell
from 1,425 to 794 MiB and the pool took 476 blocks from the driver in place
of 60. The blocks were cached, not needed: the pool returns them by itself
when an allocation fails, so the sampled peak falls and no larger system
fits.

On multiple devices, the next solve consumes the preceding global state's
sector-array snapshot; each sector solver still owns its exact restart state.
This releases obsolete matrices as sectors advance instead of pinning them
until all devices finish. The 1625-atom example then converged on two A40s
(46068 MiB CUDA-visible per device) with FP64, matching the one-A100 density
to relative L2 error below 1e-9.
The single-device lifetime is unchanged because early release did not show a
speed benefit there. Near-capacity success is input-dependent, not a blanket
guarantee that every problem fitting in the sum of device memories will run.

`PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES` controls an existing large-case cache
release policy. Raising it retains free workspace and may save allocations,
but increases cached GPU memory. Treat it as a measured tuning parameter, not
an unconditional recommendation.

Where the states of the sectors are kept between SCF steps, and which
allocator serves them, is set by `PARSEC_CUPY_SECTOR_STATE_STORAGE` (`device`,
`host` or `auto`) and `PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR` (`pool`, `direct`
or `auto`). A named value is taken as it is. `auto`, the default of both,
keeps the states on their devices and CuPy's pool unless the sectors of some
device cannot fit it; then the states are spilled to the host after every
solve and allocations bypass the pool. What is held against the memory of a
device, against `PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION` of it (0.978 by
default, see below), is a count of what its Ritz steps need: the vectors of
every sector whose basis lies on it and one slab buffer, that of its largest
sector, since its sectors are solved in turn; for a basis that the devices
of a group share, the blocks that the sharing rule counts on each of them.
Operators, graph buffers, small matrices and the CUDA context are left out,
and no memory in use is read: the count follows from the sizes and the
devices, so two launches of one calculation decide alike. For a basis on one
device it is a lower limit of what the device must hold. For a shared basis
it is not: the sharing rule counts a workspace for slabs as wide as their
budget allows, and the slabs that are cut can be narrower. The devices of
10,456 electrons on 16 GPUs count 9,945 MiB each and sampled as little as
10,005 MiB, 426 of them the CUDA context: 366 MiB less than the count
outside the context. So it was, by that or less, on 61 of the 208 devices
with shared blocks in the first runs of one series, all from 14,680
electrons down, while from 19,392 electrons up they sampled 0.5 to 5.0 GiB
more than the count and the context. For 19,392 electrons on four A100 the
count is 56.0 of the 79.25 GiB of a device (that run peaked at 58.8 GiB),
for 14,680 on two 75.5 and for 10,456 on one 77.7, which is above that share
of the device: they are spilled, and so are the four sectors of 14,680
electrons on one device, which count 147 GiB. 14,680 electrons on two
devices are the fullest device that has run: a first run on the code before
this rule, with both values named, took 239.4 s and peaked at 77.5 and 78.3
GiB (79,333 and 80,157 MiB), 0.97 GiB below the device, with the total
energy of the run on four devices to the last bit. A run that fits with
`device` and `pool` named therefore takes that route without naming them,
with the same two report lines (`orbital_sector_state_storage`,
`orbital_memory_allocator`); the runner records both for every rank, with
the bytes counted (`sector_state_fit_bytes`). The measured runs (except
those said below to name neither) name both values and may go on doing so;
`scripts/run_multi_gpu_scf.sh` names neither.

The rule before this one spilled and left the pool as soon as the vectors of
a device, the whole basis of a shared sector counted on its owner, reached
half of its memory. It dates from two or three arrays per sector. In a first
run on an A100 node with neither value named, 19,392 electrons on four
devices (52.0 GiB of vectors each) stopped with an out-of-memory error of
the root's solve command 81 s after their start, which is where the first
eigensolve ends (7.9 s of set-up and 69.7 s in the run with both named,
which took 184.1 s and 58.8 GiB). The stop had been derived from the code
beforehand: the download of a spill first made a second copy of the basis
on its device (`cupy.asnumpy` of a Fortran-ordered array, measured with CuPy
14.0.1 on a laptop GPU), so as it was written the spill could not move a
basis above half of a device at all. By the same rule the ranks of 8- and
16-GPU runs from 19,392 electrons up would have left the pool for a basis
that no spill moves: the report of 29,576 electrons on 8 GPUs counts 122 GB
on a device of 85. Naming `PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION` selects
that rule again, with the fraction it names; `0.5` gives the former
decisions and report lines.

The count leaves out what else a device holds (58.8 GiB were sampled where
it gives 56.0), so it is held against a share of the device and not all of
it: `PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION`, 0.978 by default; `1` holds it
against all of the device. In the first runs on A100 nodes every device
that held a basis of its own sampled at least 1.026 times its count: 39
devices of 15 runs with 3.2 to 75.5 GiB counted, the least of them the
device without the Hartree arrays in that run of 14,680 electrons on two
(77.5 GiB sampled, 75.5 counted), and from 50 GiB counted up the peak lay
1.8 to 2.8 GiB above the count. At 0.978 a device is therefore taken as too
small only where that ratio puts its peak above its memory, 1.74 GiB below
a device of 79.25 GiB. That run of 14,680 electrons, at
0.953 of its devices, keeps its states. 10,456 electrons on one A100 count
77.7 GiB, 0.981 of the device, and are spilled, as the former rule spilled
them: by that ratio they would peak at 79.8 GiB or more, and held against
all of the device they are told that they fit and are expected to stop for
memory, as with `device` named. Both have run since (below): the first
ended by the spill and the second stopped. A sampled peak includes
blocks that the pool only caches and returns when an allocation fails, so a
run can need less than it sampled: between 0.953 and 0.981 of a device no
run exists, and where in that band the share lies is a choice.
A basis that the devices of a group share is counted as the rule that shares
it counts it, which holds it to
`PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION` of a device (0.85): this share
decides nothing for it unless that one is raised above it or the sharing is
forced. For every first run of that series on A100 nodes, 3,480 to
39,368 electrons on 1 to 16 GPUs, the count lies below the sampled peak of
the fullest device, 0.5 to 5.5 GiB below and at 0.90 to 0.96 of it from
10,456 electrons up, and keeps the states where the named values had them;
the former rule would have named the spill and the direct allocator for the
eight of them from 19,392 electrons up, and for 14,680 electrons on two
devices. A sector on two devices is counted with slabs of 2 GiB since
(`PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES`), 4 GiB less than in that
series: 31.0, 47.0 and 62.0 GiB for 19,392, 23,768 and 29,576 electrons on 8
GPUs, whose fullest devices then peaked at 33,643, 50,859 and 67,031 MiB.
With the former limit named the count is the former one.

The download of a spill now asks for the order the array has
(`PARSEC_CUPY_SPILL_DOWNLOAD_ORDER=A`, the default; `C` is the call as it
was). `cupy.asnumpy` returns C order unless told otherwise and makes the
C-ordered copy of a Fortran-ordered array on the device, a second array of
the size of the basis, and the host array was then copied once more into
Fortran order. In its own order the basis of a sector, whole or as the
leading columns that a smaller state count leaves, takes no array on the
device and no second one on the host, and the host array is the same to the
last bit: on a laptop GPU (CuPy 14.0.1) a hook on the allocator counted no
allocation where the former call took the bytes of the basis, and a spilled
SCF gave the same bits either way. A spilled sector thus needs its basis
once on the device where it needed it twice. Derived: 10,456
electrons on one A100 about 24 GiB in place of 40 (24,465 MiB were sampled
since), and 14,680 on one, whose
two copies of 35.75 GiB stood at the edge of the device, about 43. The
upload of the next step is unchanged: it copies the host array through a
pinned buffer of its size (measured there as well), so the host holds every
spilled basis and one more, 75 and 19 GiB for 10,456 electrons on one
device. 19,392 electrons on one node would hold 208 GiB of spilled states
before any buffer of an upload, 52 GiB for each sector on its way back, on
a node of 256 GB: that case is at the limit of the host, on one, two or
four devices, whatever the device now allows. The rule does not ask what
the host can hold. The former rule by its fraction spills those 19,392
electrons on four devices too; only with `C` named beside it does that
launch stop for device memory after its first eigensolve, as it did.

On a laptop GPU (424 electrons, nine SCF steps, the launch settings of the
measured runs on one device) the run with neither value named gave the two
report lines and, to the last bit, the density, eigenvalues, Hartree
potential, next potential and the energies and residuals of every step of
the run with both named; it counted 0.53 of the 8.5 GB of the device. With
the fraction named at 0.04, which the 0.42 GB of vectors reach, the run
spilled and left the pool, with the report lines and the bits of the same
launch on the code before this rule, and with the bits of the run that kept
its states: the spill is exact. First runs on A100 nodes have decided by
this count since (on the code that introduced this rule). 19,392 electrons
with neither value named gave
the two report lines and the total energy of the run with both named to the
last bit: on 4 GPUs in 183.2 against 183.7 s with the same peak of 60,251
MiB, where the former rule had stopped that launch, on 8 GPUs in 109.5
against 108.6 s and on 16 in 66.3 against 66.1 s. 14,680 electrons on two
devices with neither value named were the named run again: the same energy,
242.5 s, 80,157 MiB. 10,456 electrons on one device ended by the spill, in
770.6 s with a peak of 24,465 MiB and the total energy of the run on four
devices to the last bit; with the share at `1` beside it the same launch
stopped in a solve command with an out-of-memory error at 81.9 GB
allocated. The download was run for 7,120 electrons spilled on one device:
254.1 s and a peak of 11,085 MiB in the order of the array, 458.9 s and
18,175 MiB with `PARSEC_CUPY_SPILL_DOWNLOAD_ORDER=C`, the same total energy.

Before that release every rank collects Python's cyclic garbage, so that a
device array held only by a reference cycle goes back with the cached blocks.
`PARSEC_CUPY_DENSITY_COLLECTION=full` collects every generation after every
density, as before. `changed`, the default, does so after the first two
densities and afterwards only when a pool of the released devices has other
bytes in use than the last full collection left there; otherwise it collects
the two young generations and releases the same pools. A cycle that keeps
device memory beside the live arrays shows as such a difference and is
collected at once. One that only keeps arrays which were live at the last
full collection does not, and waits for the next: compare
`gpu_peak_used_mib_max` between the two settings at the largest size. Only
arrays of the default pool are seen this way; pinned host memory and device
memory that a library holds outside the pool (a captured graph, a solver
workspace) are not. A process that does not allocate from the default pool
collects in full, and so does every density whose sector states are kept on
the host (`PARSEC_CUPY_SECTOR_STATE_STORAGE=host`, or `auto` where the
sectors of a device cannot fit it, see above): no device pool is released
then, and none is compared. The runner records `density_pool_release` for
every rank: the releases, the full and young collections, the unreachable
objects they found and the seconds of the collections, of the pool release
and of the whole sector-wise density.
In a complete SCF of 424 electrons on one workstation GPU (nine steps, the
threshold above at zero) the collections took 0.040 instead of 0.130 s, four
full and seven young instead of nine full, the density stage 0.20 instead of
0.32 s, with the same pool bytes after every step and the same energies to
the last bit. Those seconds are a workstation's. On A100 nodes the switch ran
on and off in one allocation together with `PARSEC_FAST_GRID` and
`PARSEC_SYMMETRY_SCF_BUFFERS` (3,480 electrons on 4, 8 and 16 GPUs, 10,456
and 19,392 on 4, 23,768 on 16), with the same total energies to the last bit.

The release itself is on the path of every SCF step: it runs inside the
density command, in which the ranks wait for the slowest, and before the
Hartree solve of the root. In first runs on A100 nodes (3,480 to 39,368
electrons on 1 to 16 GPUs) the rank that took longest over it spent 0.01 to
0.12 s of a step there, 0.10 to 1.24 s of a run: 0.1 to 2.8% of the program,
above 2% only for clusters of up to 7,120 electrons on 4 to 16 GPUs, and 0.4
to 1.7% on 8 and 16 GPUs from 14,680 electrons up. Those are the
sums over the devices of a rank, which were emptied one after another by the
thread that built the density. With `PARSEC_CUPY_DENSITY_RELEASE=threads`,
the default, a process of several devices hands the pool of each to the
thread of that device (the one that filters on it where a filter is spread
over devices, which lives as long as the process), empties the pinned pool
meanwhile and waits for the devices: they are emptied side by side. An MPI
rank waits only behind the collective calls of its density command, which
touch no device, so that the release also runs beside those; what the
density command took beyond the slowest density of a rank, those calls among
it, was 0.005 to 0.05 s of a step on 8 and 16 GPUs in the same runs. `serial`
selects the former release. A process that
releases one device does so in its own thread either way.
A release that fails in a device thread is raised by the wait. On an MPI rank
that is behind the collective calls of the command: the rank raises it alone,
as `MPISCFError` and with its context marked failed, so that a root sends no
`stop`, and the other ranks, which have their densities, learn of it by the
launcher's Abort. With `serial`, and with one device, a failed release is
reported through the command by every rank as before.
Three things were weighed against each other. Not emptying the pools saves
most, and a sector whose basis several devices share no longer empties them
(below). For a sector on one device the blocks that stay are not always
those that the next step asks for: with the release off, 14,680 electrons on
4 GPUs peaked at 45.3 instead of 42.1 GiB. Emptying them
later, beside the work of the root that follows the density, would take the
release off the path altogether, but the Hartree solve takes arrays on its
device at once, and that device is the fullest of the root for 14,680 and
19,392 electrons on 8 GPUs: whatever it took before its pool was empty would
come on top of the largest use of the step. So the pools are emptied where
they were, by more threads: every pool is empty before the thread that built
the density does anything else, the root's Hartree device included, and no
device holds more at any moment than it did. No graph capture is open in a
thread that releases (a thread of a device ends its captures within the task
that began them, and no sector is solved while a density is built), the
threads do not end with the release, no device is synchronized, and memory
that another thread returns leaves a capture valid
(`backends/cupy_capture.py`). The results are the same bit for bit.
At best it gives the longest device in place
of the sum, three quarters less on four devices, and that only if the driver
returns the memory of several devices at the same time. First runs on A100
nodes, `threads` against `serial` in one allocation, had a density stage of
2.70 against 2.94 s for 29,576 electrons on 8 GPUs, 2.12 against 2.37 for
14,680 on 4, 1.78 against 2.11 for 23,768 on 16, 1.43 against 1.78 for
14,680 on 16 and 0.58 against 0.62 for 5,264 on 8, with the same bits and
the same peaks but for 14,680 electrons on 4 GPUs (43,141 against 43,097
MiB). The device threads of the root took 1.92, 0.79, 3.15, 2.12 and 0.85 s
together in those runs where `serial` had released in 0.83, 0.66, 0.86, 0.81
and 0.32 s, 1.2 to 3.7 times as long, and the thread that built the density
spent 0.55, 0.45, 0.76, 0.56 and 0.29 s on the release: the devices slow
each other down, and less is gained than the longest device in place of the
sum. On a workstation, two pools of its one GPU standing for two devices
(CuPy 14.0.1 under Windows), the driver did not: 2 x 1.1 GiB went back in
11.5 to 12.2 ms side by side and in 12.1 to 17.4 ms one after another, the
two frees together taking 18 to 23 ms instead of 12 to 17, and through the
builder five releases of 2 x 0.6 GiB took 30 to 33 ms where `serial` took 29
to 31 (45 once, in the first round of a process).
That is what a driver that
frees one device at a time gives: no gain from the threads, and for the rank
that is slowest over its density only what the collective calls take. One
device, as on a workstation, takes the former path. `density_pool_release`
of a rank in `timing.json` names the
release and holds `release_seconds`, the seconds of the thread that built the
density with its wait for the device threads, beside
`device_release_seconds`, the sum of what those threads took. The first
below the second says that the devices were emptied at the same time, and
the second at or above the `release_seconds` of a `serial` run that the
driver freed one at a time or that they slowed each other down, as on the
workstation. What is gained is the first against the `release_seconds` of
`serial`: where it is not lower, `serial` is the setting to keep.
Between the builder's return and its wait a rank must use no device, free no
array of one and not compute in the interpreter; in the collective calls it
does none of the three. CuPy takes the interpreter lock again around every
call into the driver: beside a thread in a Python loop the workstation
returned 24 blocks of 48 MiB in 351 to 384 ms instead of 5 to 7, and the
calls of MPI give that lock up while they wait. A thread that frees an array
while the pool of its device is being emptied stands at the lock of that pool
until it is empty, and the array stays cached, where one freed just before
goes back to the driver. The builder therefore keeps what a density still
holds on a device, the sector vectors and a selection that is an array of its
own (selected columns that are no prefix), until the wait: they are freed
behind the release and stay cached, as where the builder waits itself, and
the builder is back at once (after 24 to 28 ms on the workstation where it
had stood 32 to 40 ms, until a pool of 1.4 GiB was empty). The route that
production takes selects leading columns, views that free nothing.

A sector whose basis several devices share (8 and 16 GPUs) returns nothing
after a density: the pools of its devices hold what its next step takes
again. Its blocks stay where the trial basis was created; a filter writes
into them, the Ritz step lays the rows into them and returns the rotated
columns in them, and a trim keeps their leading columns. What a step takes
beside them, the slab workspace, the exchange buffer and the Gram arrays, it
takes on the thread of each device under the stream of its group, the first
two in capacities that only grow so that the pool block of one pass serves
the next, and gives back to the free list of that stream when it ends. The
filter between two Ritz steps takes nothing from the pool: it works in the
blocks, and the buffers of its graphs are allocations of their own. So the
next step asks the same free lists for the same blocks, and the release in
between returned them to the driver only to have them taken from it again.
The largest step of a shared basis is its first solve, which precedes every
release: what it leaves in the pools was there at the peak. One block serves
no later request, the workspace of the passes before where a pass needs a
larger one, as a trim can bring about by moving the cut of the slabs: the
group then returns the free blocks of its stream on every device before it
takes the larger workspace, which is what the release did for that pass.
A process whose sectors are all shared therefore collects nothing and
empties no pool after a density, the pinned pool included; one that also
holds a sector as one array of a device empties the pool of that device as
before. Its cyclic garbage then waits for the collector of the interpreter,
which runs as it did, and the pinned pool keeps the blocks through which
host arrays go to a device: the process holds host memory that the release
returned after every density (below).
`PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE=1` empties the pools of
a shared basis after every density, as before. The results are the same bit
for bit either way.
First runs on A100 nodes with the release off for the whole process, all of
whose sectors are shared there (`PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES`
above every basis), against the default in one allocation: the density stage
took 0.58 instead of 1.70 s for 14,680 electrons on 16 GPUs and 0.89 instead
of 1.98 for 23,768, and on 8 GPUs 1.77 instead of 3.05 s for 29,576, 1.04
instead of 2.14 for 14,680 and 0.32 instead of 0.60 for 5,264 electrons. The
later solves, which found their workspace in the pool, took 0.82, 0.70,
0.84, 0.62 and 0.28 s less, and the Hartree solves of the root 0.2 to 0.5 s.
The fullest device of each run held 2, 2, 48, 2 and 8 MiB more (15.4, 28.8,
69.2, 28.0 and 6.4 GiB), where the peak of a default repeats to 2 MiB
between runs, and the total energies were the same to the last bit. The
three runs on 8 GPUs cut slabs of 4 GiB, as the code did then. With the
slabs of 2 GiB that a sector on two devices takes since, 29,576 electrons
peak at 65.5 GiB and 14,680 at 24.8, and both have run without the release
since (below). Of the later solves, the 0.84 s of 29,576 electrons are
within the spread of that run: the same default took 143.9 s in another
allocation, between the 144.6 and 143.8 s compared here.
The process keeps host memory that the release returned. The root rank of
those runs had a high-water mark of 4.72 instead of 4.10 GiB, 6.40 instead
of 5.35, 6.96 instead of 5.99, 4.83 instead of 4.20 and 3.25 instead of 2.91
GiB: 0.3 to 1.1 GiB more, about 8 to 11 arrays of the full grid in each of
the five (derived). The other ranks, whose mark dates from the preparation,
kept it to 0.02 GiB and held 0.1 to 0.4 GiB more when the run ended. The
collections that the release began with are not what frees it: over the
measured series they found 0 to 523 unreachable objects per rank and run
and none for 19,392 to 29,576 electrons, where the memory stays all the
same. On a workstation (864 electrons, one array per sector, the release
off the same way) the pinned pool held 17 free blocks at the end of the
run, 50 MiB or six arrays of the grid, which is what the working set of the
process had grown by (45 MiB); a full collection then found 16 objects and
returned nothing. So it is the pinned pool there; on A100 nodes the two have
not been separated. Nothing is done about it. Where it matters, emptying
the pinned pool alone is the smaller step back.
That is the rule as a setting for the whole process. The rule itself, by
sector, ran on A100 nodes since: first runs with the default against
`PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE=1` in one allocation, with slabs
of 2 GiB on two devices and the rounds of whole multiples on both sides. The
density stage took 0.88 instead of 1.56 s for 14,680 electrons on 8 GPUs,
1.85 instead of 2.64 for 23,768 and 1.68 instead of 2.70 for 29,576, and on
16 GPUs 0.80 instead of 1.23, 1.01 instead of 1.67 and 0.99 instead of 1.89
s; the SCF took 0.6 to 2.1 s less, and the total energies were the same to
the last bit. The fullest device held 2 MiB more in four of the six pairs
and 44 and 48 MiB more for 23,768 and 29,576 electrons on 8 GPUs (50,677 and
67,083 MiB). Every rank recorded `calls` 0 and as many `shared_kept` as
densities. The root rank had a high-water mark of host memory 0.7 to 1.1 GiB
higher (6.95 instead of 5.88 GiB for 29,576 electrons on 8 GPUs), the other
ranks the same to 0.01 GiB. The return of an outgrown workspace has run on
the pool of a workstation GPU only: the peaks of all these runs say that no
pass of theirs needed a larger workspace than the one before. With the one
GPU of a laptop named two and four times as the devices of every sector,
complete calculations of 176 and 864 electrons gave the same bits with the
pools kept as with `1` and recorded `calls` 0 and as many `shared_kept` as
densities, that of 176 electrons also with the sector states of the rank
spilled to the host; the block after the SCF had its density in each. With
one array per sector the setting for the whole process left the peak of
10,456 electrons on 4 GPUs at 23.0 GiB and raised that of 14,680 electrons
from 42.1 to 45.3. Such a sector takes its arrays in the sizes of
the step, with no capacity that is kept for the next, and what its first
solve leaves is not all taken again; which block stayed there has not been
looked for. Its pools are emptied as before.
`density_pool_release` of a rank says `shared`: `keep` or `release`, and
counts in `shared_kept` the densities after which the pools of shared
sectors were left; `calls` are then only the densities after which a pool
was emptied, 0 on 8 and 16 GPUs.

The release of a finished sector, that collection and the release after a
density act together as follows. A finished sector returns free blocks of
its own stream: the bytes in use on the device stay what they were, so the
collection rule compares the same numbers with the release as without it. The
density step then empties the free lists of every stream of its devices, the
blocks that its own sums left on the default stream and those of a sector
that released nothing included, so the next Ritz step starts from the vectors
alone. Both read one limit, `PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES`, and
hold against it the vectors of all sectors that the process solves, the
density step as it is handed them and a finished sector at the columns of the
solve in progress; a value that is no nonnegative integer is an error of
both, and of every rank when its solver and its density builder are built,
before the first eigensolve. Neither falls inside a graph capture of its own
thread: a
sector's thread has ended its captures before its solve returns, and no
sector is solved while a density is built, whichever threads empty the
pools then (`PARSEC_CUPY_DENSITY_RELEASE`). A capture that the thread of
another device has open meanwhile (two GPUs) was not measured across
devices; an invalidated capture is recorded again and counted
(`graph_captures`). With a shared basis (8 and 16 GPUs) no device returns
blocks at the end of a sector. The device that estimates the condition
number of the overlap there (`PARSEC_CUPY_RITZ_CONDITION_HELPER`) does so on
the thread that filters on it, which lives as long as the process, with the
stream of its group: its arrays are taken in the free list in which its Gram
arrays just lay, and from the driver what does not fit there, and go back to
that list; the estimate has ended before the small solve returns, and the
blocks stay in that list for the next step like the others of a shared basis
(the density step returned them before, and does with
`PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE=1`).

## Optional FP64 initialization and filter submission improvements

Two independent switches address host overhead without changing the physical
model, precision, filter degrees, or convergence tolerance:

```bash
export PARSEC_CUPY_DEVICE_RANDOM=1
export PARSEC_CUPY_FILTER_GRAPHS=1
```

The first generates the initial CHEBFF basis directly on its assigned GPU.
Unsigned 48-bit modular skip-ahead produces exactly the existing LAPACK DLARNV
values, array ordering, and final seed, including continuation by any subsequent
host replacement draws. It avoids the large temporary host basis and its upload.
It does not change the Lanczos random-number policy.

The second replays each FP64 Chebyshev block through a CUDA graph. The existing
six-column fused stencil/projector arithmetic and recurrence carry are preserved.
An uploaded coefficient table accommodates changing bounds without recapture.
Each sector owns three grid-by-block buffers, one projector buffer, and a small
coefficient table; graphs share these buffers on an ordered stream. They do not
retain old SCF orbital matrices. One graph is recorded per block width and
degree and kept: it reads a table of its own, into which the rows of a block are
copied on the stream before each launch, so a plan with other degrees records
only the widths and degrees that are new and keeps the buffers.
`PARSEC_CUPY_FILTER_GRAPH_REUSE=0` (default `1`) records one graph per block and
all of them again whenever the block plan changes, as before; the filtered
columns are the same to the last bit.
The bounded persistent graph buffers use direct CUDA allocations so that small
long-lived pieces cannot strand multi-gigabyte orbital blocks in CuPy's caching
pool. Temporary orbitals still use the configured allocator. These graph buffers
are additional to any limit applied only to the CuPy memory pool.
Mixed-precision, wider-block, and unsupported operator paths retain the existing
implementation. Both options default to off and can be selected independently.

On an allocated GPU node, validate these paths with:

```bash
PYTHONPATH=src python -m unittest \
  parsec_python.acceleration.tests.test_device_lapack_random \
  parsec_python.acceleration.tests.test_filter_graph
```

The tests check bitwise sequence/seed continuity, filtering equivalence,
changing bounds and potential, strided/remainder blocks, workspace lifetime,
nondefault streams, shared graphs against one graph per block over the plans of
a run, also with passes that nothing waits for on two streams, and independent
concurrent capture on two devices.
They also verify that graph workspaces do not retain or fragment the orbital pool.
Use complete converged SCF runs to select performance options; increasing batch
size or CPU/stream concurrency is not automatically beneficial. Compare total
SCF time separately from the eigensolver and program startup/preparation time.

## Interpretation and limits

This implementation parallelizes exact real one-dimensional symmetry sectors.
In one process (`main.py`) the useful GPU count is bounded by the number and
balance of those sectors; the nanodiamond examples have four. The MPI runner
(`benchmarks/mpi_full_scf.py`, one rank per node) divides the devices of a
rank among the sectors it owns, and the devices of a sector share its basis,
so that four sectors use 8 or 16 GPUs on two or four nodes of four. No sector
is spread over ranks, and through
`main.py` an asymmetric system uses one GPU.

Record the GPU model, active GPU count, allocated and active CPU counts,
precision, input hashes, code revision, iterations, and peak memory. Separate
SCF wall time from startup, preparation, archive writing, and optional forces.
The Fortran code computes forces that the Python single-point comparison omits;
compare SCF wall time primarily and report external elapsed time separately.

Warm CUDA's compilation cache before timing repeats. The symmetry cache is
another matter: no run reads or writes one unless `--symmetry-cache DIRECTORY`
names it, so a run with default flags outside the resident worker is the
first calculation of its structure, and `symmetry_cache_directory` in its
report says `disabled`. A resident worker reports the same and still reuses
what it holds in memory from an earlier request (`README.md`). Record the
directory of a run that had a cache. Preserve failed and
nonconverged runs but exclude them from performance claims. Sum-of-sector GPU
stage counters are accumulated work across devices; they are not parallel wall
time. Use SCF iteration timers for a stacked wall-time breakdown and label
device-stage breakdowns separately. A speedup over 40 MPI ranks does not by
itself establish equivalence to 50 or 100 CPUs.
