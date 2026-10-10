# Compact GPU execution and follow-up experiments

This extends `GPU_SFC_OPTIMIZATION.md`. The reference for these comparisons
is the GPU version described there, not the older CPU/Fortran implementation.
The opt-in compact profile
combines affine stencil metadata, ownership-based orbital buffer recycling,
a Hartree interface that keeps full-grid fields on the GPU, and an SPD overlap
condition screen with the original SVD fallback. It retains FP64, the original grid and coefficients, pseudopotentials,
functional, spectral bounds, occupations, and stopping tolerances.

## Running the profile

Build the native extension from this source and activate an environment with
CuPy. Run inside an allocated GPU job, from the directory containing parsec.in,
with the launch settings listed under "Build and run" in
[GPU_SETUP_OPTIMIZATION.md](GPU_SETUP_OPTIMIZATION.md) and, exported after
them (the first replaces `PARSEC_HARTREE_BOUNDARY_BACKEND=auto`), the six of
the table below:

```bash
export PARSEC_HARTREE_BOUNDARY_BACKEND=cupy PARSEC_CUPY_IMPLICIT_TILE=16
export PARSEC_CUPY_RECYCLE_STATE=1 PARSEC_CUPY_RESIDENT_HARTREE=auto
export PARSEC_CUPY_RESIDENT_PREDICTOR=host PARSEC_CUPY_RITZ_CONDITION=symmetric
PARSEC_CUPY_DEVICES=0,1,2,3 \
  python /path/to/checkout/src/parsec_python/main.py --backend auto parsec.in --debug
```

The profile keeps the earlier settings, including
16 OpenMP threads, one BLAS thread, GPU ionic setup, and private Ritz workspace
reuse. CPU resources must be explicitly allocated by the batch script; setting
OMP_NUM_THREADS alone does not allocate CPU cores. The performance experiments
keep 16 physical CPU cores fixed while varying GPU count.

| Environment variable | Compact profile | Restore previous behavior |
|---|---|---|
| `PARSEC_HARTREE_BOUNDARY_BACKEND` | `cupy` | `auto` for the older size/architecture policy; `native` to force native |
| `PARSEC_CUPY_IMPLICIT_TILE` | `16` | `0` |
| `PARSEC_CUPY_RECYCLE_STATE` | `1` | `0` |
| `PARSEC_CUPY_RESIDENT_HARTREE` | `auto` | `0` |
| `PARSEC_CUPY_RESIDENT_PREDICTOR` | `host` | Keep `host` for validated arithmetic |
| `PARSEC_CUPY_RITZ_CONDITION` | `symmetric` | `svd` |

`auto` residency is eligible
only with the GPU boundary, GPU CG and the same scalar/Poisson symmetry wedge.
It leaves the original interface in place for native boundaries, asymmetric
cases and other ineligible layouts. Forced `1` rejects an ineligible layout
rather than mixing representations. The backend details record the selected
transfer policy and predictor. The earlier settings alone remain usable
for direct comparisons.

The compact profile explicitly selects the GPU boundary, matching both
nanodiamond benchmark sizes. The earlier settings use an automatic
size/architecture policy, which can select a different boundary implementation
for small cases. `PARSEC_HARTREE_BOUNDARY_BACKEND=auto` restores that policy
without forcing residency in an ineligible layout.

## Exact stencil metadata compression

Rows are grouped into 16-row tiles in their existing order. A tile whose
neighbor displacements and coefficient codes are constant stores one offset
and one code per stencil slot. Irregular tiles preserve all original indices
and codes. Missing neighbors, symmetry boundaries and partial tiles remain
explicit. No extra grid points are introduced, and the coefficient order,
floating operations, KB projectors and fused Chebyshev recurrence are unchanged.

Packing uses bounded vectorized chunks instead of a Python loop over every
tile. Up to four threads pack the chunks of a stencil side by side
(`PARSEC_CUPY_IMPLICIT_PACK_WORKERS`; `1` packs them one after another, as
before); the packed arrays are the same. A value that is no positive
integer is an error of every operator that is to pack tiles, and threads
that cannot be started are a failure of the tiles like any other (see the
end of this section). This still starts from the existing assembled metadata; it does not
eliminate CPU CSR or sector construction. Compressed descriptors are private
to each sector and must not be reported as shared neighbor tables.

On the captured 1,031,322-row sector of the 5,264-electron system, metadata fell
from 128,915,354 to 43,984,019 bytes. The six-column local-Hamiltonian kernel was
about 9% faster; the separate full four-GPU experiment improved whole-program
time from 126.575 to 124.200 s (two formal runs each). These are different
measurements, and the kernel speedup is not a whole-program speedup.

Later complete first calculations (no symmetry cache, four sectors on the
four A100 of one node, the current solver) measured where the tiles pay.
Every run passed strict parity, with the same total energy to the last bit
with and without tiles, and the sampled peak of device memory was 50 to 300
MiB lower with them. The first slot-major number of a row is from the earlier
measured series, the others from the allocations of the tile runs.

| Valence electrons | Rows per sector | Program s, slot-major | Tiles of 16 | SCF s, slot-major | Tiles of 16 |
|---:|---:|---:|---:|---:|---:|
| 3,480 | 677,542 | 18.0, 20.6 | 19.2 | 10.5, 11.9 | 10.4 |
| 5,264 | 1,031,322 | 30.4, 31.8 | 31.6 | 20.2, 20.6 | 19.5 |
| 7,120 | 1,194,720 | 42.2, 43.9 | 41.3 | 30.1, 31.2 | 28.2 |
| 10,456 | 1,924,792 | 89.0, 88.2, 91.5 | 86.3, 86.8 | 70.7, 70.4, 72.3 | 66.4, 66.4 |
| 14,680 | 2,604,846 | 162.3, 160.8 | 156.2 | 137.6, 137.3 | 129.4 |
| 19,392 | 2,873,400 | 226.6, 227.3 | 215.1 | 198.4, 199.0 | 185.0 |

The SCF part gains 4 to 9% from 1,031,322 rows on. Packing the tiles on the
host, then with one thread, cost about 0.2 microseconds per row and sector
before the SCF: 0.9 s at 1,031,322 rows and 1.5 to 2 s for the three largest. The program
therefore did not gain at 1,031,322 rows (31.6 s against 30.4 and 31.8)
and gained 2 to 6% from 1,194,720 rows on; the limit of `auto` was
1,100,000 rows.

With four threads the packing costs less than the two smaller sectors gain.
First runs with them (no symmetry cache, one allocation of one A100 node),
slot-major against `PARSEC_CUPY_IMPLICIT_TILE=16`, on one, two and four
devices (four, two and one sector on each); strict parity passed and the
two total energies of every pair are the same to the last bit:

| Rows per sector | Devices | Program s, slot-major | Tiles of 16 | Before the SCF s, slot-major | Tiles of 16 | Filter thread-s, slot-major | Tiles of 16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 677,542 | 1 | 29.75 | 28.35 | 2.78 | 2.99 | 20.66 | 19.28 |
| 677,542 | 2 | 18.70 | 17.31 | 2.89 | 3.08 | 20.70 | 19.31 |
| 677,542 | 4 | 12.57 | 12.25 | 3.49 | 3.55 | 20.75 | 19.38 |
| 1,031,322 | 1 | 64.10 | 59.96 | 3.42 | 3.51 | 45.68 | 41.74 |
| 1,031,322 | 2 | 36.31 | 35.06 | 3.65 | 3.96 | 45.72 | 41.78 |
| 1,031,322 | 4 | 22.05 | 21.45 | 3.94 | 4.20 | 45.91 | 41.91 |

The filter takes 6.6 to 6.7% and 8.6 to 8.7% fewer seconds whatever the
number of devices, the time before the SCF grows by 0.1 to 0.3 s, and the
sampled peak of device memory is 10 to 330 MiB lower. What a device saves
adds up over the sectors that it filters one after another, while the host
packs every stencil once: the program gains 1.4 and 4.1 s on one device and
0.3 and 0.6 s on four. The slot-major run of 677,542 rows on two devices
spent 0.65 s more in its sector solves than the earlier measured series did (13.50
against 12.85 s), so about 0.6 s of its 1.4 s is the gain of the tiles. The
same setting differed by up to 0.5 s between the earlier measured series and this
allocation, so the gain on four devices is within what two allocations
differ by; the filter seconds are not.

`PARSEC_CUPY_IMPLICIT_TILE=auto`, the default, therefore packs tiles of 16
for a sector of at least 650,000 rows
(`PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS`) whose basis one device filters, where
that device filters no other sector. Where it filters several, `auto`
divides the limit by the square root of their number: 459,620 rows for two
sectors on a device, 325,000 for four. The square root, not the number,
because a smaller sector saves less than in proportion: it has fewer
states, and fewer of its rows lie in regular tiles (75.5% at 1,924,792
rows, 70.5% at 1,031,322, 66.6% at 677,542, 61.2% at 407,478, 56.7% at
274,784; spheres at 0.2 angstrom, order 8). No sector below 677,542 rows
was run with tiles on an A100, so the limits for two and four sectors are
derived, not measured. Taking from the runs above the filter seconds as the
rows to the power 1.9, their saving as falling with the share of regular
tiles (10.6% at 1,924,792 rows) and the packing as proportional to the
rows, the tiles gain 0.1 to 0.2 s at each of the three limits and stop
paying at about 420,000, 310,000 and 250,000 rows. More than four sectors
on a device count as four: a process that solves eight packs eight
stencils, so that by the same arithmetic the tiles stop paying at the same
rows as with four, and none was run. Of the smaller benchmark clusters,
C459H204 (577,426 rows a sector in its sphere of 16.4 angstrom) lies
between the limits, with tiles on one and two devices and without on four,
and C185H124 (280,790 rows, 12.9 angstrom) below all three; neither has
been run on an A100 with tiles.
`PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS` at 1,100,000 for one sector on a
device, 1,555,634 for two or 2,200,000 for four gives the former limit of
1,100,000 rows there. Each value holds for its own number of sectors: a
process whose devices filter unequal numbers of them (three devices for
four sectors) has no setting that gives every device the former limit.

A sector whose basis or filter is given several devices has a row count of
its own (below); a sector that is to hold the
opt-in FP32 recurrence keeps the slot-major stencil, as does any stencil
outside symmetry sectors. `0` keeps
the slot-major stencil everywhere and a tile size packs every stencil; the
compact profile still names 16. Tiles were not measured on a sector with
many rows and few states, which pays the packing without much filter to gain.

The limits are calibrations on A100 with one device for the basis of a
sector: with a shared basis (8 and 16 devices) the sectors of 677,542 and
1,031,322 rows were run slot-major only. Another device need not gain. On
an RTX 5070 Laptop the recurrence alone took 248 to 258 ps per row and
vector with either layout at five sizes from 274,784 to 1,031,322 rows
(tiles 0.1% faster to 1.5% slower in the median of 21 runs, same bits),
where the whole filter of the runs above took 67 and 64 ps slot-major and
62 and 59 ps with tiles on an A100. On such a device the tiles only save
device memory, the host packing is paid for nothing, and
`PARSEC_CUPY_IMPLICIT_TILE=0` is the setting to use.

A sector whose basis or filter is given several devices (8 and 16 GPUs)
is packed once, by its owner. The other devices of the
group are given the packed arrays that the owner holds and pack nothing;
their operators are built from them or the calculation stops, since they
have no other layout to give way to. Every device then filters, and
applies `H` in the Ritz step, with the tile kernels; a group whose owner
holds the slot-major stencil reads that on every device, as before.
`auto` packs such a sector from 1,100,000 rows, by a setting of its own,
`PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS`, which the limit of one device
does not move. That row count is assumed, not measured against others;
complete calculations with a tiled group have run on 8 and 16 A100 GPUs
since.

What is measured is a sector on one device: later first calculations on
A100 with four packing threads, tiles of 16 against the slot-major stencil
in one allocation. The filter took 6.6 to 6.7% fewer seconds at 677,542
rows, 8.6 to 8.7% at 1,031,322, 10.6% at 1,924,792 and 11.5% at 2,873,400,
and `H` in the Ritz step 7.1, 9.3, 11.0 and 11.9%. The set-up of the
sector operators of a rank (`initialization_seconds`) grew by 0.02 to
0.03, 0.05 to 0.07, 0.06 and 0.07 s per sector. A group of `d` devices
filters a sector in a `d`-th of the time and packs it once. Those shares
of the filter and `H` seconds of a slot-major group of the earlier measured series
would be 0.4 s at 1,194,720 rows on 16 GPUs (of 16.2 s; about 9%,
interpolated) and 0.9 s on 8, 5.2 to 5.6 s at 3,786,832 (10.3 to 11.2 s
on 8) and 7.2 to 7.8 s at 4,130,510, where the share may exceed the 10.6
to 11.5% taken, since it grew with the size. Below the row count they
would be 0.1 s at 677,542 rows on 16 GPUs and 0.2 s on 8, 0.3 and 0.5 s at
1,031,322: more than the packing, but at the smallest size less than two
runs of one setting differed by (8.28 and 8.12 s). A run with a lower
`PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS`, or with a named tile size,
against the default shows whether a group gains there.

Complete calculations with a tiled group have run on 8 and 16 A100 GPUs
since (7,120 to 39,368 electrons; `stencil_storage` of their records reads
`implicit_affine_tile_16`). Their times and device memory against the
slot-major stencil of a group are not given here; what follows is from a
laptop with one GPU.
The arrays of the 3,786,832-row sector of the 23,768-electron system fall
from 473.4 to 119.0 MB per device and those of 4,130,510 rows from 516.3
to 127.2 MB. Packing either took 0.4 to 0.7 s with four threads there, and
handing the packed arrays to one more operator 0.03 s where the slot-major
arrays took 0.18 s. An operator built from the packed arrays of another
applied `H` and filtered bit for bit like it and like the slot-major
stencil, with both of these stencils. With four device numbers mapped to
that one GPU, a shared basis of 48 states on a 1,194,720-row stencil (two
numbers and 30 states for 3,786,832 rows) gave the eigenvalues of a first
solve and of two later passes and the final basis bit for bit as with the
slot-major stencil. `per_rank[].distributed_state[]` of
`timing.json` names the layout of a group (`stencil_storage`) and the
seconds in which the operators of its other devices were built
(`replica_seconds`), which lie in the first solve.

Tiles that cannot be built (the packed arrays outgrow int32 offsets, the
tile kernels do not compile, the device has no room for them) are handled
by who asked for them. Under `auto` the sector keeps the slot-major stencil,
the same kernels and results, and `compact_finite_difference_reason` of its
operator records why; `orbital_sector_finite_difference_storage` shows the
layout. A tile size named in the environment is an error then, as is a value
that is not a tile size, for every operator. Neither case falls back to the
CSR-order kernel, which has no fused recurrence and which the filter graphs
do not read.

## Orbital array ownership

After filtering consumes the old vectors, the old orbital array can serve as
the generalized Ritz HX/rotation workspace. The caller explicitly transfers
ownership with `consume_state=True`; public/default calls preserve their input.
Residual diagnostics retain the existing allocation path. The resulting state
does not retain an extra tall Ritz scratch array. Subsequent updates are covered
by device tests, in addition to complete SCF field/history comparisons.

This complements the earlier C-contiguous-transpose rotation optimization. It
does not distribute the coupled Ritz matrices or the owning sector's states
across GPUs. In the isolated four-GPU test it reduced the largest-device sampled
peak from a median 21.874 to 17.363 GiB. Whole-program time was effectively
unchanged/slightly higher, so it is a memory improvement, not a claimed speedup.

## Hartree transfer path

The SCF continues to use host-side physical values on the symmetry wedge. GPU
expansion, streaming multipoles, boundary-corrected full RHS, ordered orbit
projection and GPU CG are connected without transferring full-grid arrays to
the host between stages. Only compact wedge fields, and the existing small
boundary moments, cross that interface.

Ordered projection matches NumPy bincount's original per-orbit accumulation.
The validated host predictor preserves the original NumPy dot products and
secant update. The compact export kernel reproduces the repeated addition and
division formerly performed by full-grid expansion and orbit averaging. These
rounding details matter for reproducing a nonlinear SCF trajectory.

The earlier device-predictor variant passed the primary final-field tolerances
but differed in energy components by up to 5.31e-5 Ry. It is not selected by the
compact profile. The host-predictor/original-rounding variant completed two
formal four-GPU runs in a median 120.925 s versus 126.575 s for the reference;
the independently audited formal run matched every selected physical field
and the SCF history bit for bit. Complete combined-profile results are in the
table below.

## Symmetric overlap condition screen

`PARSEC_CUPY_RITZ_CONDITION=symmetric` estimates an SPD overlap condition number
from symmetric eigenvalues. It retains the original SVD within a factor of 100
of the guard, for an indefinite matrix or on eigensolver failure; Cholesky and
coefficient orthogonality checks are unchanged. The compact profile enables
the screen. Left unset, the variable follows the place of the small solve:
the device solve, which is the default, uses the screen, and the host solve
(`PARSEC_CUPY_RITZ_DENSE_BACKEND=host`) retains the original SVD.
Two formal runs took a median 124.370 s versus 126.575 s and matched all audited
physical fields/history exactly. This changes condition screening only, not
the eigensolver or the dense eigenpair calculation itself.

## Experiments not enabled by the profile

`PARSEC_CUPY_DISTRIBUTED_FILTER=1` splits blocks within a sector among GPUs,
replicates its stencil and KB projectors, then gathers filtered columns for
the existing owner-local Ritz solve. It is a filter prototype, not a complete
distributed eigensolver. In the tested three-GPU runs, program time increased
from 58.89 to 61.05 s for 3,480 electrons and from 187.26 to 194.53 s for 5,264
electrons. The extra copies also raise memory use. It remains disabled and
cannot be combined with compressed implicit metadata.

`PARSEC_CUPY_FILTER_DEGREE_CAP` is a screening override after the first five
updates. The existing code already adapts the global requested degree and
retains spectral-width lower limits. Caps 12 and 10 are no-ops when the input
degree is already 10; they provide no optimization evidence. Cap 8 did change
the work but failed to converge in the original 70-step budget, ending at a
weighted residual of 0.006765 Ry versus the 0.0002 Ry criterion. Its 126.10 s
elapsed time is not a successful time-to-solution and is excluded from speedups.

The AMG/PCG and geometric Galerkin experiments live under `benchmarks/` and are
not production Hartree backends. Initial uncaptured versions cut iterations
but made the solve slower. A separate capture-compatible SpMV experiment
checks whether CUDA launch overhead explains that result; it uses an isolated
memory pool to protect captured scratch addresses from live PCG state. Its
original fine-grid operator and independently recomputed residual are retained.
After replacing both the sparse calls and the tiny coarse inverse, capture
actually succeeded. The geometric graph solve still took 0.117--0.134 s versus
0.088--0.112 s for existing CG; the SA graph solve took 0.172--0.195 s. Build and
upload cost another 4.02 s and 9.06 s, respectively. These tested preconditioners
are not enabled. Iteration count alone is never used to select a faster solver.

## Complete combined-profile results

Same-node A100 80 GB results with cold persistent caches. Times below are
program seconds, not just GPU kernel time. The largest-device memory
number is the median sampled peak; warmups are excluded.

| Valence electrons | GPUs | Formal n old/new | Previous s | Compact s | Time reduction | GPU peak GiB old/new |
|---|---|---|---|---|---|---|
| 3,480 | 4 | 1/1 | 45.780 | 42.530 | 7.10% | 8.109/8.150 |
| 5,264 | 1 | 1/1 | 314.000 | 285.880 | 8.96% | 68.916/48.264 |
| 5,264 | 2 | 1/2 | 198.060 | 185.165 | 6.51% | 38.633/43.705 |
| 5,264 | 3 | 1/1 | 187.260 | 172.270 | 8.00% | 38.633/38.594 |
| 5,264 | 4 | 2/2 | 126.575 | 115.625 | 8.65% | 21.874/17.404 |

For 5,264 electrons on four GPUs, SCF time fell from 98.385 to 86.755 s;
the SCF Hartree component fell from 15.355 to 9.500 s. Every selected
physical field, energy component and the SCF trajectory in all formal
combined runs was bitwise identical to that reference, including 66 SCF iterations
for the larger case and 35 for the smaller one. This is a regression
comparison against that reference, not an independent physical benchmark.

Memory benefit is configuration-dependent: the single-GPU peak fell about
30%, the four-GPU largest-device peak about 20%, while three GPUs and the
smaller case were essentially unchanged. Two-GPU formal runs showed a
higher peak than the previous code; all samples, including the repeat
triggered by that observation, were kept. Pool allocation
and one-second sampling may contribute to variability, but the cause is
not established and no two-GPU memory improvement is claimed. Host RSS
for the larger four-GPU case was effectively unchanged at about 5.6 GiB.

Four-versus-one-GPU whole-program speedup is 2.47 for the compact profile,
versus 2.48 for the previous code; orbital eigensolver speedup is 3.54.
Thus this profile reduces absolute runtime without demonstrating better
whole-program strong-scaling efficiency. Three GPUs still assign the four
symmetry sectors as 2+1+1. Preparation/finalization takes about 29 s and
Hartree and other SCF work also limit whole-program scaling.

Validation also includes 32 passing GPU tests, 53 local regression tests
(one native-extension availability skip), native/GPU boundary fallback
checks, and a smoke run with the compact profile. Its one-iteration
fixture intentionally exits before convergence and is not a performance
or full-convergence sample.

## Measurement and precision limits

Measurements use the A100 SXM4 80 GB GPUs of one Perlmutter node, fresh processes,
disabled persistent geometry/operator caches, fixed CPU resources and excluded
warmups. Reported program time excludes imports, input parsing and NPZ output;
external process time is retained separately. GPU memory is sampled every
second, host RSS every 0.1 s. Peaks from different devices are not simultaneous.

CPU setup was measured with 16, 32 and 64 threads. The original `-c64
--hint=nomultithread` step actually exposed 32 physical cores, not 64. A separate
probe before OpenMP imports confirmed this and the C++ OpenMP thread settings.
The follow-up `-c128 --cpu-bind=cores` step exposed all 64 physical cores. Its
16/32/64-thread medians through operator construction were 25.268/24.752/25.047 s
(two formal runs each), versus 25.720/25.153/25.762 s on the earlier mask. This
is only a few percent, not evidence of proportional CPU scaling or a measured
whole-program speedup; the selected GPU profile keeps 16 threads.

Reusing the existing exact-key cache reduced setup-through-operators to 6.242 s.
That is an existing cache benefit for repeated structures, not a new GPU
algorithm. Early setup JSON recorded the calling thread's affinity after
OpenMP binding; those `[0,64]` masks do not mean the workers had only one core.
An independent probe recorded the actual
128-logical-CPU/64-physical-core mask and native worker settings.

No geometry optimization or new Fortran MPI reference was performed for this
profile. These are comparisons against the GPU version of
`GPU_SFC_OPTIMIZATION.md`. They do not remove the energy-component
absolute-difference floor against earlier GPU code that is documented there.
Performance and memory conclusions should not be generalized to other GPUs,
grids, symmetry groups or electronic structures without measurements.
