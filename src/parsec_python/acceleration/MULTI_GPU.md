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
python /path/to/parsec_python/src/parsec_python/main.py parsec.in --device gpu
```

Set the account and QoS from `sacctmgr show assoc user=$USER`. Follow the
[site's current examples](https://scienceit-docs.lbl.gov/hpc/running/script-examples/)
for CPU/GPU ratios and memory limits. Keep Slurm's `CUDA_VISIBLE_DEVICES`
unchanged. `PARSEC_CUPY_DEVICES=0`, `0,1`, or `0,1,2,3` selects a subset of
the GPUs allocated to this process, as `--gpu-ids 0,1` or `--gpus 2` does on
the command line; `--device gpu` makes the run stop, instead of taking the
CPU, where no GPU is usable. Use one Python task, not one driver per GPU.

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
Since then the root of such a run sums its GPU ionic fields on the symmetry
wedge where the images of the atoms are exact
(`PARSEC_IONIC_SUM_GRID=auto`, see `GPU_SETUP_OPTIMIZATION.md`), which
changes the fields by the round-off of one sum over the atoms and the
totals with them: on a laptop GPU those of small runs under these launch
settings (30 to 176 electrons) moved by 2e-13 to 1.5e-11 Ry with the same
number of SCF steps (`README.md`), and on A100 nodes that of 39,368
electrons on 16 GPUs by 3.2e-10 Ry. `PARSEC_IONIC_SUM_GRID=full` beside the
launch settings undoes that change. Other switches that came after the
series change totals by round-off too (`README.md`). With the switches that
came after the series at their former values, `PARSEC_IONIC_SUM_GRID=full`
and `PARSEC_CUPY_TALL_PRODUCT_SPLIT=0` among them, eight configurations on
A100 nodes gave the total energies of the series to the last bit: 3,480
electrons on 1, 4 and 16 GPUs, 14,680 on 4, 23,768 on 8 and 16, 29,576 on 8
and 39,368 on 16.

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

### Relaxations and forces on several nodes

The same runner and launch script take an input with `Minimization: bfgs` or
`simple`, or with `Skip_Force: false`. The output directory then also receives
`atom.cor`, `relax_restart.dat`, `iterate.dat` and `relaxation.xyz`, and
`timing.json` holds, under `result.relaxation`, the energy, largest force, SCF
iterations and wall time of every structure, with the time of the parts of
each ionic step (`step_timings`) and of each force evaluation
(`force_phase_seconds`).

Between two structures nothing that a single point keeps on the devices
leaves them. The root sends three more commands to the ranks: `geometry`
(every rank replaces the projectors of its sectors in place, also on the
devices that share a sector basis), `new_scf_loop` (every sector solver keeps
its subspace as the start of the next SCF loop) and `forces` (every rank
contracts the nonlocal force terms of its sectors on its devices; the root
adds them). The local forces are summed on the GPUs of the root node over the
symmetry wedge of the grid.

The ranks divide the symmetry sectors of the structure. A structure without
point-group symmetry, a relaxation whose `Move_Flag` breaks the symmetry, a
`manual` relaxation and molecular dynamics have one sector: one rank solves it
on the GPUs of one node, or one rank per GPU on several nodes
(`scripts/run_multi_gpu_scf.sh --rank-per-gpu`), whose ranks share its basis
(`Eigensolvers/mpi_shared_basis.py`; numbers in
`../FORCES_AND_RELAXATION.md`, "On GPUs and several nodes").

Measured for C795H300 on A100 nodes (seven structures to 0.01 Ry/bohr with
the inputs of the series above): 47 s on 4 GPUs, 35 s on 8 and 30 s on 16,
with identical trajectories; a later structure costs 2.3 s of SCF, 0.5 s of
forces and 0.26 s of rebuilding on 16 GPUs. The comparisons with Fortran
PARSEC and the other measurements are in
[FORCES_AND_RELAXATION.md](../FORCES_AND_RELAXATION.md).

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
of an earlier tree, without touching its results. The estimates of the
omitted potential that the search for that radius asks for, which were most
of those seconds, are made by a host thread of the root while the SCF runs
(`PARSEC_DOMAIN_REPORT_THREAD`, on by default; `0` makes them after the SCF,
as before): the thread works with NumPy on host arrays of its own, calls
nothing on a device, and is ended and joined before the root reports. These
switches and the
others added with them, with their defaults, the values that restore the
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
list per stream: the Ritz buffer that a sector gave back (1 to 4 GiB then, 2
at most since) could
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
device, against `PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION` of it (0.974 by
default, see below), is a count of what its Ritz steps need: the vectors of
every sector whose basis lies on it and one slab buffer, that of its largest
sector, since its sectors are solved in turn; for a basis that the devices
of a group share, the blocks that the sharing rule counts on each of them.
Operators, graph buffers, small matrices and the CUDA context are left out,
and no memory in use is read: the count follows from the sizes and the
devices, so two launches of one calculation decide alike. For a basis on one
device it is a lower limit of what the device must hold. For a shared basis
it is not: the sharing rule counts a workspace for slabs as wide as their
budget allows, and the slabs that are cut can be narrower. It also counts
the most that an exchange buffer can take which follows a slab, where no
chunk size is named: up to 128 MiB above the chunk of 1 GiB on four devices,
from 10.7 GiB of basis and all of it from 12 GiB
(`PARSEC_CUPY_EXCHANGE_CHUNK_BYTES` in `README.md`). The devices
of 10,456 electrons on 16 GPUs counted 9,945 MiB each with the chunk of 1 GiB
that they had (10,073 MiB where none is named) and sampled as little as
10,005 MiB, 426 of them the CUDA context: 366 MiB less than the count
outside the context. So it was, by that or less, on 61 of the 208 devices
with shared blocks in the first runs of one series, all from 14,680
electrons down, while from 19,392 electrons up they sampled 0.5 to 5.0 GiB
more than the count and the context. For 19,392 electrons on four A100 the
count is 54.0 of the 79.25 GiB of a device (a run peaked at 56.8 GiB),
for 14,680 on two 73.5 and for 10,456 on one 77.4, which is above that share
of the device: they are spilled, and so are the four sectors of 14,680
electrons on one device, which count 145 GiB. Those are the counts with the
buffer of at most 2 GiB that a sector on one device takes
(`PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES`, `README.md`); with the buffer of
at most 4 GiB that it took before they were 56.0, 75.5 and 77.7 GiB, and the
first of those runs peaked at 58.8 GiB. 14,680 electrons on two
devices are the fullest device that has run: a first run on the code before
this rule, with both values named, took 239.4 s and peaked at 77.5 and 78.3
GiB (79,333 and 80,157 MiB), 0.97 GiB below the device, with the total
energy of the run on four devices to the last bit; with the buffer of 2 GiB
it peaked at 75.5 and 76.3 GiB (77,285 and 78,109 MiB). A run that fits with
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

The count leaves out what else a device holds (56.8 GiB were sampled where
it gives 54.0), so it is held against a share of the device and not all of
it: `PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION`, 0.974 by default; `1` holds it
against all of the device. The share was 0.978 while the buffer of a sector
on one device took up to 4 GiB. In the first runs on A100 nodes with that
buffer every device
that held a basis of its own sampled at least 1.026 times its count: 39
devices of 15 runs with 3.2 to 75.5 GiB counted, the least of them the
device without the Hartree arrays in that run of 14,680 electrons on two
(77.5 GiB sampled, 75.5 counted), and from 50 GiB counted up the peak lay
1.8 to 2.8 GiB above the count. At 0.978 a device was therefore taken as too
small only where that ratio put its peak above its memory, 1.74 GiB below
a device of 79.25 GiB. That run of 14,680 electrons, at
0.953 of its devices, kept its states. 10,456 electrons on one A100 counted
77.7 GiB, 0.981 of the device, and were spilled, as the former rule spilled
them: by that ratio they would peak at 79.8 GiB or more, and held against
all of the device they were told that they fit and were expected to stop for
memory, as with `device` named. Both have run since (below): the first
ended by the spill and the second stopped. A sampled peak includes
blocks that the pool only caches and returns when an allocation fails, so a
run can need less than it sampled: between 0.953 and 0.981 of a device no
run existed, and where in that band the share lay was a choice.

With the buffer of at most 2 GiB the count of a device and its peak fall
together. The run of 14,680 electrons on two devices, with the budget set to
2 GiB beside the default of then in one allocation, peaked 2,048 MiB lower on
both devices where 2 GiB less are counted (73.5 GiB): 75.5 and 76.3 GiB
sampled, 1.027 and 1.038 times the count, and 2.0 and 2.8 GiB above it as
before. That run is now at 0.927 of its devices. 10,456 electrons on one
A100 count 77.4 GiB with the smaller buffer, 0.976 of the device: below the
former share, which would keep their states on the device. Nothing says that
they fit there. They stopped for memory with a buffer 0.36 GiB larger, and
the same four sectors in the smaller sphere of the default rule ran on one
device 2.5 GiB above their count (71.3 GiB counted, 73.8 sampled, 71.5 held
between the steps), which would be 79.9 GiB here. The share is therefore
0.974, the reciprocal of 1.027: a device is still taken as too small only
where the least ratio that was measured puts its peak above its memory,
2.06 GiB below a device of 79.25 GiB, and those sectors are spilled as they
were. A first run of those sectors on one A100 with neither value named and
`0.978` beside them has decided the share since: the rule kept their states
on the device, and the first solve stopped for memory (an allocation of
2,147,475,456 bytes failed with 81,874,197,504 allocated). With 0.974 the
same launch spilled them and completed (below), so the share stays 0.974.
Between 0.927 of a device, the fullest that kept its states, and 0.976,
which did not fit, no run exists. With that buffer a device of 79.25 GiB
keeps the states of sectors of 32 GiB and more up to 75.2 GiB of them,
where it kept them up to 73.5: no layout of the measured series lies
between. A device
whose count lies between 0.974 and 0.978 of it with sectors of up to 16 GiB,
whose buffer is what it was, kept its states and spills them now: none has
run, and the least ratio that was measured puts its peak above its memory.
`PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES=4294967296` with
`PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION=0.978` gives the former count and
the former decisions.
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
was, under `PARSEC_CUPY_SPILL_BUFFERS=fresh` since: with the host buffers
of a spill, the default of that switch (below), a Fortran-ordered basis is
written into its buffer whatever this says, and the order is read only for
vectors of another layout). `cupy.asnumpy` returns C order unless told
otherwise and makes the C-ordered copy of a Fortran-ordered array on the
device, a second array of
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
upload of the next step was left as it was then: it copied the host array
through a block of CuPy's pinned pool (measured there as well), so the host
held every spilled basis and that block. The pool rounds a request up to a
power of two: for 10,456 electrons on one device the block of a basis of
18.84 GiB is one of 32 GiB beside the 75 GiB of the four bases, as an A100
node measured since (a high-water mark of 110.0 GiB of host memory in that
run, 78.0 GiB when it ended). The spilled states have host buffers of their
own since, and no such block (below). 19,392 electrons on one node would
hold 208 GiB of spilled states, 52 GiB for each sector, before the block of
an upload that the former staging adds, on
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

A spilled sector moves its basis three times in an SCF step: to the device
before its solve, to the host behind it, and to the device once more for the
density. As it was, every download went into a new pageable array, whose
pages the system hands out again each time, every upload through a block of
CuPy's pinned pool, into which the host array is copied first, and the
density step emptied that pool, so that the first upload of the next step
page-locked the block again. In a first run of 10,456 electrons on one A100
with 5 ang of vacuum (four sectors of 1,924,792 rows and 1,314 states, 18.84
GiB each; 745.1 s) the solve commands took 575.4 s for 216.1 s of device
work, and the density stage 159.1 s, of which 64.2 s were the eleven returns
of the pinned pool: 517 s for moving 128 bases. One array of that size
between an A100 and its host (CuPy 14.2.0; every path returned the bits of
the source): `cupy.asarray` took 17.99 s with the pinned pool empty, 16.2 s
of it the page-locking of a block of 32 GiB, and 1.75 s (10.7 GiB/s) with
the block in the pool; emptying the pool took 5.82 to 5.89 s; `cupy.asnumpy`
into a new array took 2.62 s (7.2 GiB/s). A buffer of the size of the array
that is page-locked once (9.6 to 9.7 s) took 0.758 s up and 0.770 s down
(24.8 and 24.5 GiB/s) by `ndarray.set` and `ndarray.get(out=)`, which hand
the driver the host memory itself, and a pageable buffer that is kept took
1.40 and 1.52 s by the same calls (13.4 and 12.4 GiB/s).

Each spilled sector therefore has a host buffer of its own
(`PARSEC_CUPY_SPILL_BUFFERS=persistent`, the default; `fresh` is the staging
as it was). It is taken when the sector is first downloaded, at the size of
its basis, and kept: a later download writes over it, with as many columns
or, after a trim, with fewer in its leading bytes, and an upload reads it
where it lies, before a solve and for the density alike. Neither call takes
a staging block or a second array on the device (a hook on both allocators
counted none on a laptop GPU, CuPy 14.0.1), and after the first step no page
of the host is handed out or page-locked again. The host holds every
spilled basis once. The device holds what it held, one array for an upload
and none for a download, so the count of the rule above and what it spills
are unchanged. The host array of the states of a sector is a view of its
buffer and holds them until the next download of that sector: the density
step reads it before the next solve, and the solver uploads it first, which
is what the SCF loop already asked of the arrays of a sector (a later pass
under `PARSEC_CUPY_RECYCLE_STATE=1` filters its basis in place). A sector
that grows, as the bracket of the spectrum can ask in the first steps, takes
a larger buffer. Vectors that are not Fortran-ordered on their device are
downloaded by the former call, in the order that
`PARSEC_CUPY_SPILL_DOWNLOAD_ORDER` names, and laid into the buffer in
Fortran order, a copy on the host that went into a second new array before;
their uploads are direct like the others. On a laptop GPU the sectors of an
atom and of a molecule of 32 electrons left such vectors under the settings
of the hybrid driver itself, where the Ritz step of a small sector
orthogonalizes its basis first, and those of 176 electrons under launch
settings of the measured runs left Fortran-ordered ones, as the spilled runs
on A100 nodes did (the download order above made a second array there). For
a Fortran-ordered basis the buffers do not read that switch: it belongs to
`fresh`.

The buffers are page-locked within a bound, so that the driver copies
between them and the device without a copy of its own. What is page-locked
is out of reach of everything else on the host until it is given back: the
system cannot page it out to make room. The bound therefore follows from the
host. The buffers of a process are locked while their sum stays within half
of the memory that the host reports as available when the storage of the
states is decided (`MemAvailable` of `/proc/meminfo`, the available physical
memory under Windows), which leaves as much available as they take, and
within the limit of the system on locked memory where it has one
(`RLIMIT_MEMLOCK`). A host whose memory cannot be read locks nothing.
`PARSEC_CUPY_SPILL_PINNED_BYTES` names the bound instead and is taken as it
is; `0` locks nothing. The plan is made with the storage, from the bytes of
each basis at the state counts of the first solve and in the order of the
sectors: a buffer is to be locked if it still fits the bound beside those
before it, so that two launches on a host with the same memory available
lock the same buffers. Sectors whose operator was given a group of devices
come last in that order: a shared basis never leaves its devices and takes
no buffer, and whether one is shared is decided by the first solve, so the
plan reads the group and asks no device. A sector on one device beside
them is therefore not made pageable by the bytes of one that takes nothing,
and one of them that is not shared after all is still planned, locked if
the bound has room left. A buffer beyond the bound is pageable, and so is one
that the driver refuses to lock (the record says what it answered): both
move the same bytes by the same calls, at about half the rate, and a small
host runs as it did, without the staging block. The bound holds for what is
locked at any moment: a buffer that a sector has outgrown is counted until
the result that still reads it has gone. It is a bound of one process:
several processes that spill on one host each take half of what they find
available, and a process inside a memory limit of its control group reads
the memory of the host, so name the bytes there. And it bounds what is
locked, not what the buffers need: the host holds every spilled basis
whether its buffer is locked or not, as it did, and the rule that spills
still does not ask what the host can hold. For 10,456 electrons on one
device the four buffers are 75.4 GiB, which half of the memory of a node of
256 GB locks whole, where the former staging reached the high-water mark of
110.0 GiB with its block of 32 GiB; of the 208 GiB of 19,392 electrons the
same half locks two buffers of the four (both derived).

The density step leaves CuPy's pinned pool as it is where it was handed a
sector whose states are on the host (`PARSEC_CUPY_SPILL_PINNED_POOL=keep`,
the default; `release` empties it after every density, as before). With the
buffers no basis goes through that pool, which then holds the blocks of
smaller uploads. With `fresh` named beside the default the staging block
stays in the pool for the run: the eleven returns of 64.2 s and the
page-locking that each caused (15.9 to 16.3 s for the block on that node)
are what keeping it saves by itself, and the host then holds the block
between the steps as well, at the high-water mark it had, one block for
every power of two that a basis of the process rounds up to. A process whose
sectors all stay on their devices empties the pinned pool as it did under
either value, and the collection before the release is the same.

On a laptop GPU (CuPy 14.0.1 under Windows, which has no `RLIMIT_MEMLOCK`)
complete calculations with their states on the host gave with the buffers
every array of the archive and every line of `parsec.out` apart from its
times as the code before them did: 32 and 176 electrons through the hybrid
driver and through the MPI runner with one rank under the launch settings
of the measured runs, with host storage named beside CuPy's pool, and 32
electrons through that runner spilled by the rule itself, with the direct
allocator; each with the buffers page-locked by the rule of the host and
pageable (`PARSEC_CUPY_SPILL_PINNED_BYTES=0`), and through a serial control
of that runner with the buffers named. With `fresh` and `release` named
the backend details were those of that code as well, and so with `fresh`
alone. Every download of the 32 electrons was one
that the former call made, for C-ordered vectors; of the 176 electrons
under the launch settings 48 of 52 were direct, and with a CHEBFF first
solve and the generalized Ritz step named all 44. So it was with every
sector on a thread and a non-blocking stream of its own
(`PARSEC_CUPY_SECTOR_SCHEDULER=streams`), where the sectors of one device
move their states side by side, for 32 electrons through the hybrid driver
and for 176 through the runner. A shared basis with host storage named took
no buffer. Half of the available memory was 0.13 to 7.9 GB on that host
while other work ran on it, and every buffer was locked within it. The
seconds of that laptop say nothing: its GPU was shared.

Derived beforehand from the rates of the one array above for that run of
10,456 electrons: its 128 transfers, 2,412 GiB, take 98 s through
page-locked buffers and the page-locking of the four another 38 s (38.4 s
were measured for four blocks of that size one after another), where they
took 517 s, so the run about 365 s in place of 745; the solves on the
device are what they were. A first run on one A100 node has been made
since, with the storage left to the code, which spills those sectors, and
the defaults against `fresh` with `release` in one allocation: 396.9
against 746.4 s of SCF and 401.2 against 751.3 s of reported program, with
the same total energy to the last bit. The four buffers page-locked 80.93
GB in 40.0 s, inside the offloads, which took 73.9 against 116.3 s; the
restores took 46.6 against 220.4 s. The host high-water mark fell from
109.9 to 78.2 GiB, and the GPU peak was 23.61 against 23.53 GiB. That stays
above the run whose states fit: the same cluster in the sphere of the
default rule keeps its states on the device and took 205.3 s. The
page-locked memory goes back when the process ends, after the reported
program and outside every record of `timing.json` (13.9 s were measured for
four blocks of that size): by the log of the batch job the whole process,
start and teardown included, took 434 against 770 s, 32.8 against 18.7 s
beyond the reported program: the return of the buffers costs about 14 s
that no record holds, and a comparison of the two stagings reads that time
beside the records. Not
measured there at all: a buffer that is locked in the context of one device
and read by another, as the density step of two devices does (the CUDA
documentation says that memory from `cudaHostAlloc` is page-locked for every
context under unified addressing); pageable buffers in a complete run; each
switch by itself; and `RLIMIT_MEMLOCK`, which those nodes do not set.
`spill_staging` of a rank in `timing.json` says what ran: the staging, the
calls that moved the states of a sector to its device and back with their
wall seconds (a basis that the devices of its sector share is not moved and
not counted), and for the buffers the bound and where it comes from, the
bytes that are page-locked and pageable, the seconds of the page-locking
(inside the first download of each sector), what the driver answered where
it refused, the bytes that went up and down, and the downloads that the
former call made for vectors of another layout. `density_pool_release`
counts the uploads of the density step under `host_uploads` and says
whether the pinned pool is kept (`spill_pinned_pool`, `pinned_kept`). A run
whose sectors took buffers has the backend detail
`orbital_sector_spill_staging`: how many of them are page-locked and the
bytes of each kind. The bound is not in it, since it follows from what the
host has available, and two launches of one calculation would differ there;
`spill_staging` holds it. The detail is of the sectors of the process that
writes the result, the root rank under the MPI runner; `spill_staging` of
each rank says what the others did. A run that keeps its states on its
devices, shares every basis, or stages them as before has the details it
had.

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
The exchange buffer is treated alike: it grows where a later pass sends a
slab that it is to hold whole and the passes before sent none
(`PARSEC_CUPY_EXCHANGE_CHUNK_BYTES` in `README.md`), and a device that takes
a larger buffer than before first returns the free blocks of that stream.
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
`PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE=1`). By default (`auto`) that
device is never the one that holds the Hartree objects; `any` asks that one
too where a sector of two devices has no other and the sharing rule counts
the room for the estimate. `any` was the default for a while, until complete
first calculations on 8 A100 GPUs, two of each value in alternation, took
163.40 s of SCF on average with it against 163.51 s with `auto` for 23,768
electrons and 236.16 against 235.92 s for 29,576, with the same total
energies and the fullest device at 47.87 against 47.67 GiB and at 63.52 GiB
with either (2026-10-16): it would have stayed the default for at least
0.4 % less SCF time in either case (`README.md`).

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
Every recurrence step of a graph first forms the projector coefficients
`diag(signs) B.T X` of its block. Where the projector rows have at least 256
entries, a block of two to six columns forms them in one launch of a CUDA
block of 128 threads per projector (`PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS=6`,
the default), the six-column kernel as first measured; a wider block, as
the graphs of the distributed filter record them, in one launch of its group
kernel, a CUDA block per projector and group of six columns; a block of one
column takes the former kernel, a CUDA block per projector and column, which
`1` keeps for every width. All of them sum a column in the same order, so the
coefficients and the filtered columns are the same to the last bit. A graph
is recorded for one block width and replays the kernel of that width: an
operator keeps the kernels it was built with, the other devices of a shared
sector are given the columns per block of the sector owner, and a workspace
whose operator has been given a projection of another reduction records its
graphs again. On one A100 with the real factor of sector 0 (2026-10-16) the
six-column kernel formed six columns 1.71 and 1.80 times as fast as the
former kernel with 19,392 and 39,368 electrons; a single column takes the
former kernel since it was 0.62 and 0.67 times as fast there. One launch of
the group kernel for 126 columns was 1.12 times as fast as 21 launches of the
six-column kernel with 39,368 electrons, 1.23 times with 19,392 and 3.9 times
with 3,480, but slower per group from three to six columns with 39,368, 6 %
at six, so blocks of up to six columns keep the six-column kernel; their time
with the launches of now is still to be measured, and is expected to be that
of the six-column kernel.
Complete calculations on 1, 4 and 16 GPUs took 0.44 to 3.9 s less SCF than
with `1`, with the same totals, while every width took the six-column kernel
six columns per launch. `README.md` has the switch in its tables, with these
measurements.
The recurrence step itself is one launch of `stencil_major_chebyshev6` per
block. By default (`PARSEC_CUPY_STENCIL_KERNEL=pipelined`; `former` keeps the
one kernel of before) an operator holds two of them, both compiled when it is
built: one whose slot loop is a software pipeline of its loads, for the
launches of six columns of unit row stride, and the former kernel, for every
other launch. A graph recorded for a block of
six columns replays the first, one recorded for a narrower block (the last
block of a plan, two columns in a plan of 14) the second, and the other
devices of a shared sector build the kernels of their owner. The filtered
columns are the same to the last bit, NaN payloads included on an A100
(`README.md`). On A100 GPUs
(2026-10-20, each value on its own GPU of one node, sector 0 in tiles of 16
rows, a probe of the filter) a block of six
columns and degree 15 took 10.35 against 14.95 ms through the graphs with
19,392 electrons and 21.24 against 29.50 ms with 39,368, 0.69 and 0.72 of the
time, and blocks of one, two and five columns the time of the former kernel.
`README.md` describes the kernel with the switches.
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
  parsec_python.acceleration.tests.test_filter_graph \
  parsec_python.acceleration.tests.test_stencil_pipeline
```

The tests check bitwise sequence/seed continuity, filtering equivalence,
changing bounds and potential, strided/remainder blocks, workspace lifetime,
nondefault streams, shared graphs against one graph per block over the plans of
a run, also with passes that nothing waits for on two streams, six columns per
projection block against one over the same plans and through both filters, a
workspace that records again for another projection kernel, and independent
concurrent capture on two devices.
They also verify that graph workspaces do not retain or fragment the orbital pool.
The last module compares the pipelined recurrence kernel with the former one
word for word on every launch path (NaN payloads where two NaNs meet on
compute capability 8.0 alone, `README.md`), also for stencils of 2 to 61 slots in
every tile size, records which kernel each launch of a graph took, filters a
sector shared by two devices with either kernel, and checks the entry of the
backend details that names the kernel (`orbital_sector_stencil_kernel`).
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
What a rank spent its preparation on, and which stage the other ranks waited
for, is in `per_rank[].preparation_stages` of `timing.json`: every stage with
its thread, its start and its end (`GPU_SETUP_OPTIMIZATION.md`).
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
