# GPU Hartree linear solve

The conjugate gradients of the cluster Hartree solve run on a GPU by default
since 2026-11-07, wherever the orbitals do with the native Hartree
components (`--backend auto` on a GPU, the native C++ extension and CuPy
installed). `PARSEC_HARTREE_LINEAR_BACKEND` chooses:

| Value | Conjugate gradients |
|---|---|
| unset or `auto` (default) | on the GPU where it applies; elsewhere the CG of the selected backend (C++/OpenMP on a CPU run, SciPy for `--backend scipy`, the device CSR CG of `--backend cupy`, the periodic solver of a cell). The C++/OpenMP CG takes over, on the same symmetry wedge, where the GPU CG cannot be made (an out-of-memory error, a CUDA runtime or driver error, a wedge whose stencil it cannot take) and, beside a full-grid eigensolver, where the GPU Hartree arrays would take more than 1/20 of the free device memory (below); `hartree_linear_backend` says which and why |
| `native` | the C++/OpenMP CG on the host, the default before 2026-11-07 (bit for bit the former runs) |
| `cupy` | the GPU CG by name; refused where it does not apply, and a failure to make it stops the run |

`hartree_backend` reports `native-boundary/cupy-linear` (or
`cupy-boundary/cupy-linear` with the GPU boundary below) and
`hartree_linear_backend` the choice and its reason. The older
`--backend cupy` path, with its own Poisson solver, is a different route.

Both CGs run one recurrence: no preconditioner, the stopping test
`||r_k|| <= rtol ||r_0|| + atol` (`Hartree_CG_Tolerance`, default `1e-7`,
`atol 1e-13`), the same matrix-vector budget, the chronological starting
vector of `NativePoissonSolver` and the explicit final residual. They differ
in the order of the floating-point sums only (fixed-order block reductions on
the GPU, so a GPU run repeats its own bits), so the Hartree potential moves
within the CG tolerance and the results by what the SCF tolerance passes on:
on A100 nodes, against `native`, the total energies of the example inputs
that run on this path in minutes (the small-molecule folders, all 56 inputs
of `0d_hybrid_functionals`, the 16 of `0_CH4_CF4`, the 14 SAD inputs of
`ml_initial_density` with one SCDP and one ChargE3Net start, the relaxations
and 11 of the 224 inputs of `0d_xps_core_levels`; 3 hybrids with
`--symmetry off`) moved by at most 1.3e-11 Ry at their own SCF tolerances,
the exact-exchange energies by 4e-11 Ry, forces and relaxation paths by
1.5e-11 Ry/bohr (1e-9 Ry/bohr with the CA potentials of CF4), refined B3LYP
energies by 6e-13 Ry, with the same number of SCF steps in every case.
Occupied levels moved by at most 1.4e-10 Ry except where the spin channels of
O2 are solved on the full grid: 1.1e-8 Ry for PBE (7e-10 Ry at an SCF
tolerance of 1e-9 Ry) and, through the different paths that the Fock loop then
takes to its tolerance, 5e-6 Ry for PBE0 at 1e-6 Ry (2e-7 Ry at 1e-9 Ry), with
totals within 2e-10 Ry (3e-13 Ry at 1e-9 Ry). That is less than the CG
tolerance itself leaves there: with `Hartree_CG_Tolerance: 1e-10` the two CGs
agree within 5e-10 Ry in every level and 5e-13 Ry in the total, and either of
them lies 1.2e-6 Ry in the occupied levels (and 1.6e-6 Ry in the exact
exchange) from the host run at the default `1e-7`. The extensive totals of the
nanodiamonds C185H124 and C459H204 on four GPUs moved by 6e-11 to 1.0e-9 Ry
at their tolerance of 2e-4 Ry and by 3e-11 to 5e-10 Ry at 1e-8 Ry. The pair
densities of
`Energy_Refinement` follow the CG of the SCF (`Hartree/pair_poisson.py`); the
Kronecker pair solves of the hybrid functionals are not this solver.

Program wall time in seconds, first runs in FP64 on A100 80 GB nodes, one GPU
of a node held whole (`--gpus 1`) unless four are named; the host CG with 16
OpenMP threads (in brackets: the launcher's default of 124 threads, which is
not faster for a grid of 1.4 million points); "GPU" is the default, the CG and
for a full grid the boundary on the GPU:

| Case (grid points, Hartree problem) | native | GPU | Hartree seconds |
|---|---:|---:|---|
| H2O PBE, `0d_hybrid_functionals` (1.57 M, C2v wedge) | 28.1 | 10.1 | 21.6 -> 4.0 |
| H2O PBE0 | 44.6 (39.1) | 18.9 (18.7) | 31.0 -> 5.4 |
| H2O PBE0, `--symmetry off` (full grid; one GPU of a shared node) | 449 | 59.5 | 414 -> 18.7 |
| C6H6 PBE (C2v wedge) | 26.2 | 12.4 | 16.4 -> 2.7 |
| C6H6 PBE0 | 59.3 (56.7) | 32.8 (33.3) | 31.3 -> 5.1 |
| O2 PBE, spin channels on the full grid (1.41 M, the semilocal default) | 101 (163) | 26.2 | 84.4 -> 8.1 |
| O2 PBE, `Spin_Symmetry_Sectors: auto` | 16.9 | 11.4 | 6.5 -> 1.2 |
| O2 PBE0, sectors (the hybrid default) | 38.6 | 25.5 | 15.7 -> 2.7 |
| O2 PBE0, full grid | 253 | 69.0 | 206 -> 19.8 |
| CF4 PBE, `0_CH4_CF4/python_pbe` (3.10 M, full grid) | 323 (312) | 41.9 | 298 -> 15.3 |
| CF4 PBE + B3LYP refinement, 136 pair solves (one GPU of a shared node) | 1105 | 108.5 | refinement 741 -> 61.8 |
| O atom, spin, `0d_xps_core_levels` (1.44 M, full grid) | 38.7 | 15.7 | 28.1 -> 3.3 |
| protonated water, B3LYP refinement (1.36 M, full grid) | 81.8 | 20.0 | 39.6 -> 5.1 |
| C185H124 PBE, four GPUs (D2 wedge) | 13.9 | 10.6 | 3.1 -> 0.6 |
| C185H124 PBE0, four GPUs | 37.3 | 31.9 | 5.7 -> 1.1 |
| C459H204 PBE, four GPUs | 49.6 | 34.1 | 17.1 -> 3.3 |
| C459H204 PBE0, four GPUs | 263.4 | 242.9 | 24.6 -> 4.4 |
| O2 PBE / PBE0 full grid, four GPUs (GPU CG, host boundary) | 95.2 / 241.6 | 33.3 / 73.0 | 76.0 -> 12.1 / 193 -> 26.4 |

Cases of a few thousand points (H2, the SiNa4 relaxation) take the same time
either way. The boundary of a full grid on the GPU, against the host
recurrences beside the GPU CG: O2 PBE 32.8 -> 26.2 s, CF4 46.2 -> 41.9 s, the
O atom 17.7 -> 15.7 s, protonated water 23.2 -> 20.0 s (totals within 2e-13
Ry).

## The resident chain on the wedge (default since 2026-11-07)

Where the orbitals are solved in symmetry sectors and the GPU CG solves the
wedge that the SCF fields use, the boundary of that wedge is built on the
Hartree device as well and every solve runs in the device-resident chain
(`Hartree/cupy_resident.py`): the density goes up as one value per orbit;
the multipoles, the boundary values, the normalized right-hand side, the
chronological starting vector and the CG stay on the device; the potential
and the right-hand side come back as one value per orbit. Before, the
native builder formed the right-hand side on the host from its orbit table,
which was uploaded, solved and downloaded in every solve.
`PARSEC_CUPY_RESIDENT_HARTREE` chooses:

| Value | Hartree solve on the wedge of a sector run |
|---|---|
| unset or `auto` (default) | the chain where the GPU CG solves the wedge of the SCF fields with a multipole boundary, on a grid of int32 indices; elsewhere the host interface. The boundary of the wedge then takes the wedge kernels on an axis-reflection group, whether the plan is engaged or not, and the kernels of the plan on another group |
| `0` | the host interface and, on a wedge, the orbit table on the host: the route before, bit for bit |
| `1` | the chain by name; refused where it does not apply, and a failure to build it stops the run |

`PARSEC_HARTREE_BOUNDARY_BACKEND=native` (or `PARSEC_HARTREE_BOUNDARY_KERNEL=full`
without a GPU boundary named) keeps the orbit table and so the route before
as well. `hartree_resident_chain` says whether the chain runs and why,
`hartree_full_grid_transfer_policy` is `device-resident` or `host-interface`,
`hartree_boundary_kernel` names the kernels and `hartree_resident_device_bytes`
what the chain adds on the device.

The chain changes results by round-off only: the wedge kernels form the same
expansion as the orbit table in another order of the sums, and the chain
runs the CG, the predictor and the projections of the host interface on the
device. On A100 nodes, against `0`, the 85 example inputs of the sweep of the
GPU CG (above) that take the chain (the small-molecule folders, the 56 hybrids, the O2 spin
sectors, the ML starting densities, the relaxations and the forces) moved
by at most 2.6e-11 Ry in the total (Si28H36), 1.4e-10 Ry in the exact
exchange (CO HSE06), 2.3e-10 Ry in an occupied level, 1.4e-11 Ry/bohr in the
forces and 1.1e-12 in the relaxation paths, with the same number of SCF
steps in every case; the 30 inputs that do not take it (the full grids of
the spin channels, CH4/CF4, the core holes, periodic cells, symmetry off)
gave the same bits. `0` and `PARSEC_HARTREE_BOUNDARY_BACKEND=native` gave
the bits of the code before (f007f03a) for H2O PBE0, CH4 HSE06, O2 PBE and
PBE0 in sectors, the water relaxation, H2O PBE forces and benzene. The
refined B3LYP totals of `Energy_Refinement` on the wedges of H2O and CH4
moved by 4e-12 Ry. With the SCF tightened to 1e-9 Ry, H2O PBE0 and O2 PBE0
in sectors took the same steps and moved by 4e-13 Ry; C6H6 PBE took 74
steps where `0` took 77 (total 6e-13 Ry apart, occupied levels 1.5e-9 Ry),
which is the floor that the CG tolerance leaves: with
`Hartree_CG_Tolerance: 1e-10` both took 51 steps and agreed within 5e-10 Ry
in every level. The extensive totals of the nanodiamonds on four GPUs moved
by 5e-12 to 9e-10 Ry (C185H124, C459H204) and by 2.3e-9 and 2.6e-9 Ry
(C795H300 PBE and PBE0) at their tolerance of 2e-4 Ry and, at 1e-8 Ry, by
2.0e-10 (C185H124 PBE), 1.4e-10 (its PBE0), 2.9e-10 (C459H204) and 2.0e-9 Ry
for C795H300 (9,417 Ry: 2e-13 of the total, the round-off of moment sums
over 22 million points in another order: the host recurrences of the
native full-grid builder, `PARSEC_HARTREE_BOUNDARY_BACKEND=native`, gave a
total 5.1e-9 Ry from `0` and 3.1e-9 Ry from the default, and the wedge
kernels through the host interface gave the default's bits). An MPI SCF over two
nodes (H2O PBE, `benchmarks/mpi_full_scf.py`, four GPUs each) ran the chain
on the Hartree device of the root and gave the total of `0` within 4e-14 Ry
in 44 steps either way.

Where `auto` cannot keep the chain it falls back to the route before. A GPU
boundary that cannot be built (an extension without the geometry exporter, an
out-of-memory error), a GPU CG that gives way to the host CG, and a chain
whose arrays cannot be made leave the boundary of that route (the orbit
table, or the GPU kernels where it has none); the results are then those of
`0` bit for bit. A device allocation that fails in a solve in the middle of
the SCF (CuPy's out-of-memory error, or a CUDA runtime or driver error of
status 2; any other error stops the run) ends the chain for the rest of the
calculation: its arrays and its GPU boundary builder are released, the host
builder of the wedge (the orbit table, or the host recurrences where it has
none) takes their place, and the host interface beside the GPU CG repeats
that solve and takes every later one, its chronological predictor continued
from the chain's history; `hartree_resident_chain` of the result says from
which solve on, and the boundary lines name the host builder.

The boundary builder of the chain lies on the device of its CG: the Hartree
device (`PARSEC_HARTREE_DEVICE`), or, with a single GPU or with that switch
`off`, the first device of the sectors, where the CG goes. A GPU that is not
device 0 (`--gpu-ids N`, `PARSEC_CUPY_DEVICES=N`) so holds the whole chain;
a builder and a CG found on two devices give the route before. A full-grid
density or starting potential, which the SCF does not pass, is reduced to
the wedge on the host, so a builder of the wedge keeps no full-grid map on
the device. The chain runs in MPI runs too, on the root rank, which alone
builds the Hartree objects (on its Hartree device, the last device of the
rank); the serial control of the MPI runner names `0`.

Program wall time in seconds (Hartree seconds in brackets) and the largest
sampled device peak, first runs in FP64 on A100 80 GB nodes, one GPU of a node
held whole unless four are named, the same code with `0` and by default:

| Case | `0` | default | peak MiB `0` / default |
|---|---:|---:|---|
| H2O PBE, `0d_hybrid_functionals` (C2v) | 10.3 (3.91) | 8.87 (2.56) | 1156 / 1216 |
| H2O PBE0 | 17.5 (4.84) | 15.6 (3.14) | 1690 / 1814 |
| C6H6 PBE | 12.0 (2.53) | 11.1 (1.85) | 1552 / 1610 |
| C6H6 PBE0 | 32.9 (5.09) | 30.8 (3.54) | 2810 / 2732 |
| CH4 PBE (D2) | 10.3 (3.62) | 8.72 (2.31) | 1454 / 1520 |
| CH4 PBE0 | 14.3 (4.13) | 13.0 (2.73) | 2004 / 2110 |
| O2 PBE, spin sectors | 11.9 (1.14) | 11.2 (0.78) | 1074 / 1102 |
| O2 PBE0, spin sectors | 26.2 (2.71) | 24.4 (1.91) | 1626 / 1640 |
| water relaxation, `0d_water_geoopt` (7 structures) | 7.06 (1.76) | 6.08 (0.80) | 600 / 612 |
| SiNa4 relaxation (4 structures) | 2.24 (0.10) | 2.08 (0.13) | 430 / 430 |
| C185H124 PBE, four GPUs (D2) | 10.8 (0.57) | 10.8 (0.55) | 3344 / 3516 |
| C185H124 PBE0, four GPUs | 32.1 (1.12) | 31.8 (0.96) | 5768 / 5828 |
| C459H204 PBE, four GPUs | 36.1 (3.25) | 34.4 (2.57) | 5450 / 5540 |
| C459H204 PBE0, four GPUs | 242.7 (4.31) | 240.8 (3.42) | 11616 / 11710 |
| C795H300 PBE, four GPUs | 60.1 (6.00) | 59.4 (3.50) | 9112 / 8984 |
| C795H300 PBE, four GPUs, SCF to 1e-8 Ry | 79.2 (10.4) | 76.2 (6.04) | 9112 / 8984 |
| C795H300 PBE0, four GPUs | 713.9 (8.34) | 709.7 (4.94) | 22142 / 21974 |
The wedge kernels are what gains: at f007f03a by switches, the full-grid GPU
kernels with the chain (the earlier opt-in route for PARSEC's boundary,
`PARSEC_HARTREE_BOUNDARY_BACKEND=cupy` and `PARSEC_CUPY_RESIDENT_HARTREE=auto`)
took 1.01 s of Hartree for O2 PBE in sectors and 2.91 s for H2O PBE, at peaks
130 to 380 MiB above the wedge kernels, and the wedge kernels through the host
interface ran within 1 % of the chain. The peaks rose by 0 to 124 MiB on one
GPU: the boundary of the wedge (40 to 47 MB for the small molecules,
`hartree_boundary_device_bytes`) and the chain (23 to 27 MB,
`hartree_resident_device_bytes`) on the Hartree device. C795H300 had no orbit
table before (its table exceeds 512 MiB) and took the full-grid GPU kernels
(258 MB); the wedge kernels hold 63 MB and its largest device peaked lower.

## Device memory and placement

The CG holds `rows x (5 w + 40)` bytes on the Hartree device for a stencil of
`w` entries per row (25 at `Expansion_Order: 8`, 37 at 12): 45 MB for the
wedge of C185H124, 95 MB for C459H204, 318 MB for the full grid of O2, 512 MB
for CF4 (`hartree_cg_device_bytes`). The sampled peak of that device rose by
40 to 230 MiB on a wedge (C459H204 PBE0 on four GPUs: 14,942 -> 14,988 MiB,
the largest device) and by 300 to 920 MiB on a full grid with its GPU boundary
(CF4: 12.0 -> 12.9 GiB).

Where the orbitals are solved in symmetry sectors, the GPU Hartree arrays (the
CG, the GPU boundary and the resident chain, `hartree_cg_device_bytes`,
`hartree_boundary_device_bytes` and `hartree_resident_device_bytes`; the
chain's count holds its maps and the working vectors of one solve, which it
was missing before 2026-11-07) are told to the memory rules of the sectors as
`resident_device_bytes` of their device, beside the device XC evaluator and the
exchange of a hybrid: the rule of the state storage and the rule that shares a
basis plan around them, and a basis that does not fit beside them is spilled
or not shared instead of failing later. A full-grid eigensolver has no such
rule. There `auto` plans the arrays up front (`Hartree/gpu_memory.py`: the CG
`rows x (5 w + 40)` bytes and 60 bytes per row of the streaming boundary)
against what CuPy may still allocate on their device (the driver's free
memory and the free blocks of the pool, within `CUPY_GPU_MEMORY_LIMIT`), and
keeps them on the GPU only where they take at most 1/20 of it: the full-grid
runs above took 10 to 17 times those arrays beside them. Under a pool cap of
7 GiB, O2 PBE0 with the spin channels on the full grid ran with the host CG
and stopped with an out-of-memory error at its first exchange build with the
GPU CG and boundary (they needed 7.5 GiB); with the rule the default plans
the arrays (380 MiB) against the less than 7 GiB that the cap leaves free,
takes the host CG and finishes as `native` does, bit for bit (111 SCF steps,
the same total). The pair solves of
`Energy_Refinement` plan a new full-grid CG in the same way, at most half of
the free memory, and take the host CG where it does not fit or cannot be
made (`Hartree/pair_poisson.py`). `PARSEC_HARTREE_LINEAR_BACKEND=native` keeps
every one of these arrays off the device.

The existing native multipole boundary/RHS, normalized symmetry projection,
finite-difference coefficients, FP64 arithmetic, chronological initial guess,
CG tolerance, and matrix-vector budget are preserved. Only the prepared
linear solve changes. The exact coefficient palette uses uint8/uint16 indices
without quantization. Slot-major storage coalesces row-wise stencil reads;
matvec/dot and vector update/residual norm are fused. When the orbitals use
symmetry sectors, the solver takes the slot-major stencil the totally
symmetric sector was already packed into: packing its CSR again would give
the same neighbors and the same coefficient in every slot, so the solve is
bitwise unchanged while preparation skips the CSR conversion and the second
packing. The CSR form is built only if `negative_laplacian` is read or the
native CPU solver, which copies CSR buffers, is used. Recurrence scalars and
work vectors remain on one CUDA device. CUDA graph groups (default 8 steps,
`PARSEC_CUPY_POISSON_GRAPH_STEPS=1..64`) reduce host launch/synchronization
overhead. Device-side convergence/breakdown flags stop numerical updates on
the first stopping iteration, including in the middle of a graph group.

Hartree uses one of the selected orbital GPUs (see `PARSEC_HARTREE_DEVICE`
below); it does not distribute a single Poisson solve across devices. This
removes CPU work from the serial portion
of multi-GPU SCF. Through the host interface the boundary/RHS construction,
the chronological predictor and the public SCF scalar fields remain on the
CPU, so vectors are transferred once per solve; in the resident chain of a
sector run (above) only the density and the returned fields cross. CUDA work
buffers and the operator are persistent. Metadata reports
the actual GPU, allocated solver bytes, graph size and boundary/linear timings.

The first GPU also holds the owner's share of a sector basis, so with the
Hartree objects it would set the memory peak of a multi-GPU run (the sector
index and expansion maps of the eigensolver stay on the host and no longer
add to it). `PARSEC_HARTREE_DEVICE` therefore places the Poisson CG arrays,
the GPU boundary geometry and the resident Hartree maps together on another
device of the process: `auto`, the default, takes the device with the
smallest share of sector bases (one that owns no sector if there is one,
otherwise the last listed), and a CuPy device index from
`PARSEC_CUPY_DEVICES` names one. `off` keeps the former placement: the CG on
the first GPU and the boundary geometry on the device current in the thread
that builds it. On 16 GPUs with 14,680 electrons `auto` lowered the peak from
31.2 to 30.7 GiB without a change in run time. Only these three objects
follow the variable: the Poisson solver of the older `--backend cupy` path
stays on
the current device, and a sector worker of an MPI run, which builds no
Hartree object, chooses no device. The kernels and their inputs are the same
on every device of one model, so the results do not change in any bit;
`hartree_cg_device` and `hartree_boundary_device` report the placement.

## Boundary values

The right-hand side of every Hartree solve is `8 pi rho - A_IB V_B` with the
boundary values `V_B` at the exterior stencil points. `V_B` is the multipole
expansion of the density, as in PARSEC, at the order of the boundary plan:
`Solver_Lpole`, or, where the estimate `e(Solver_Lpole)` of the potential
the atoms' expansion omits exceeds a tenth of the tolerance (default
`1e-3 Ry`), the smallest order up to 60 with `e(L)` within the tolerance.
Where the plan has one, the static atomic tail
`C_L(P) = 2 sum_a q_a/|P-R_a| - M_L[q](P)` of the valence point charges at
the nuclei is added (reference `README.md`, `Hartree/boundary.py`). The plan
is made from the geometry in the first phase of the reference preparation,
also on a rank or thread that completes the reference later, and its order
is written into `reference.input.hartree.multipole_order`, which every
builder reads. It is reported as
`hartree_boundary`, `hartree_multipole_order`, `hartree_boundary_tolerance`,
`hartree_boundary_estimate`, `hartree_atomic_tail`, `hartree_atomic_tail_max`
and `hartree_atomic_tail_seconds`.

The plan bounds the part of the atoms. Density that reaches the sphere leaves
an error that neither the order nor the tail removes; with 3 A of vacuum it
is 1e-4 Ry in the boundary values and 1e-7 to 3e-7 Ry per valence electron in
the energy, and below about 2 A the default boundary is no closer to the
exact one than PARSEC's (reference `README.md`).
`PARSEC_HARTREE_BOUNDARY_CHECK` measures the boundary values of a run.

The tail is built once, beside the boundary geometry (on the Hartree set-up
thread where that overlaps the GPU orbital set-up): the missing stencil
entries come from the surface shell of the grid, and the values at the unique
exterior points from an FP64 device kernel (`Hartree/cupy_atomic_tail.py`:
compensated Coulomb sum over the atoms minus the series, no fused
multiply-adds) or, without a device, from host threads.
`PARSEC_HARTREE_ATOMIC_TAIL_VALUES=host` takes the host threads on every
backend, also for the tail a wedge builder holds per exterior point; the
default, `auto`, is the kernel wherever the orbital backend is CuPy, and
`hartree_atomic_tail_values` reports which of the two ran. Their values
differ at round-off (1e-12 of the Coulomb sum in the tests). The host threads
are slow at a raised order: the tail of the full grid took 4.8, 7.3 and 33 s
for 3,480, 5,264 and 14,680 electrons (orders 16, 20 and 30; 0.33, 0.43 and
0.78 million exterior points) on a workstation. Every builder then
adds the same static rows to its right-hand side: the native full-grid
builder and the Python builder add the rows, the native wedge builder adds
`U.T` of them, and the GPU builder adds them on the device, so the resident
chain needs no change. Nothing is added per solve but one indexed add.

The GPU boundary has two sets of kernels (`Hartree/cupy_boundary.py`),
chosen by `PARSEC_HARTREE_BOUNDARY_KERNEL`; the second set runs on the wedge
of a group or, without one, on the full grid:

- `full`: the kernels of PARSEC's boundary, on the full grid: moments from
  every grid point, boundary values at every missing stencil entry. Their
  cost grows with `(L+1)(L+2)/2`.
- `wedge`: `CuPySymmetryMultipoleBoundaryBuilder`, for a Hartree problem
  reduced by an axis-reflection group. The moments are summed over one row
  per orbit with the parity classes of the group (for D2 only even `m`
  survive, real for even `l` and imaginary for odd `l`); the boundary values
  are evaluated once per unique exterior point of the stencils of those
  rows, the atomic tail is added to the value of the point, and the
  normalized wedge right-hand side is gathered from the values. The resident
  chain takes that right-hand side as it is and uploads neither of its two
  full-grid maps. Results differ from the full kernels at round-off
  (laptop tests: 3e-12 on the right-hand side); moments the group forbids
  are exact zeros.
- `auto`, the default: where the boundary plan is engaged (estimate above a
  tenth of the tolerance) and the boundary is built on a GPU, `wedge` if such
  a reduction exists, and otherwise the same kernels under the identity
  alone (`CuPyPointMultipoleBoundaryBuilder`, reported as `points`): no
  symmetry, a signed-permutation group, or symmetry switched off. The moments
  are then summed over every grid row and each boundary value is still
  evaluated once per unique exterior point, which four to five stencil
  entries share; the builder has the full-grid interface and the driver
  projects its right-hand side as before. The boundary of the resident
  chain (above) takes `wedge` on an axis-reflection group whether the plan
  is engaged or not (since 2026-11-07). `full` elsewhere, so a boundary
  that is PARSEC's keeps its kernels and its bits. `wedge` without such a
  reduction, or without a GPU boundary (named, or that of the resident
  chain), is an error.

What the raised order costs depends on these kernels (laptop GPU, 1.1 million
grid points, per right-hand side; no A100 figure is measured). The former
kernels at order 9 took 70 ms. On the wedge of D2 the raised order took 3 ms
at order 12 and 19 ms at order 34. Under the identity the angular work is not
reduced: the unique-point kernels took 11, 19, 49 and 138 ms at orders 9, 12,
20 and 34, against 118, 302 and 835 ms for the full-grid kernels at orders 12,
20 and 34. A system without an axis-reflection group therefore pays about
twice the former boundary at order 34, and less than it up to about order 20.

The boundary is built on a GPU where `PARSEC_HARTREE_BOUNDARY_BACKEND` is
`cupy`, or `auto` on the devices its policy names. Left unset (`native`), or
with `auto` elsewhere, the native builders run, and they need their orbit
table: without a reduction, or where the table of the raised order exceeds
512 MiB, they repeat the full-grid recurrences in every solve (2.3 million
points, order 16: 249 ms against 12 ms for the GPU wedge kernels and 21 ms for
the former boundary, laptop). An engaged plan takes the GPU kernels in that
case if the orbital backend is CuPy, and since 2026-11-07 so does any plan
whose conjugate gradients run on the GPU (the default above): the full-grid
GPU kernels for PARSEC's boundary, the unique-point kernels (on the wedge or
under the identity) for an engaged one, on the Hartree device. That covers every Hartree problem on the full
grid, the spin channels outside the sectors among them. A wedge keeps its
orbit table on the host through the host interface, where the GPU kernels
measured no faster; the wedge of the resident chain is built on the GPU
(above). `PARSEC_HARTREE_BOUNDARY_BACKEND=native` or
`PARSEC_HARTREE_BOUNDARY_KERNEL=full` by name keep the native route.

The serial control of the MPI runner (`benchmarks/mpi_full_scf.py
--serial-control`) sets `PARSEC_HARTREE_BOUNDARY_KERNEL=full`,
`PARSEC_HARTREE_ATOMIC_TAIL_VALUES=host` and, since 2026-11-07,
`PARSEC_HARTREE_LINEAR_BACKEND=native` and `PARSEC_CUPY_RESIDENT_HARTREE=0`
where the launcher names none of them: the host CG and host interface of the
code before the GPU defaults, and
the same boundary values through the full-grid kernels, with the tail in rows
that are built from the surface shell and host threads. A control on the
wedge kernels of the run would repeat its Hartree computation bit for bit and
could not see a defect of them, and one that asked the device kernel for the
tail would hold the values of the run in its rows. What the control does not
choose is the boundary itself: order, tolerance and tail come from the
launcher's switches and the plan made from the geometry, and have to be those
of the run it is compared with (`result.hartree_boundary` in `timing.json`
records them for both). The plan, the moments of the point charges and the
exterior points that the native builder exports are in both sides. The
control so pays for its independence at set-up (the host threads above) and in
every solve (the full-grid kernels at the raised order).

`hartree_boundary_kernel` reports the kernels that ran and
`hartree_boundary_device_bytes` their storage: nine doubles and a row pointer
of eight bytes per wedge row, twelve bytes per stencil entry of a wedge row,
seven doubles per unique exterior point (six without a tail) and `2 (L+1)^2`
doubles per 4096 wedge rows.

These arrays lie on the Hartree device, on the root rank of an MPI run the
last device of the rank (`PARSEC_HARTREE_DEVICE` above), with the arrays of
the Poisson solver and of the resident chain. With the former kernels that
was the fullest device of the calculation. The table counts
them for the grids of the large benchmark clusters with the expressions of
the builders (bytes, 5 A of vacuum, D2, the order of the default plan).
Nothing in it was read from a device; the counts of the former builder for
14,680 to 29,576 electrons are the `hartree_boundary_device_bytes` that its
runs reported:

| Electrons | Order | Wedge rows | Unique exterior points | Wedge kernels with the tail | Former kernels, order 9 | Full-grid kernels at the order, tail in rows | Resident chain, wedge / full-grid builder |
|---|---|---|---|---|---|---|---|
| 14,680 | 30 | 2,604,846 | 195,554 | 239,500,864 | 897,966,184 | 1,467,693,760 | 31,258,156 / 114,613,228 |
| 19,392 | 28 | 2,873,400 | 208,524 | 262,085,276 | 984,753,768 | 1,526,694,048 | 34,480,804 / 126,429,604 |
| 23,768 | 34 | 3,786,832 | 250,326 | 348,418,280 | 1,277,118,120 | 2,353,793,424 | 45,441,988 / 166,620,612 |
| 29,576 | 31 | 4,130,510 | 265,300 | 375,958,168 | 1,386,607,912 | 2,353,082,440 | 49,566,124 / 181,742,444 |
| 39,368 | 36 | 5,660,200 | 326,644 | 518,798,652 | 1,869,048,168 | 3,679,974,048 | 67,922,404 / 249,048,804 |

With the default plan the boundary and the resident chain so hold 0.39 GB on
that device for 23,768 electrons where they held 1.44 GB, and 0.59 instead of
2.12 GB for 39,368: 708, 777, 1,001, 1,090 and 1,460 MiB less for the five
rows. The peak of a run falls by that much only while that device stays its
fullest. In first runs of the code before this boundary, on 16 GPUs,
it stood 1,022, 854, 970 and 622 MiB above the device that owns a shared
sector, for 14,680, 19,392, 23,768 and 29,576 electrons (970 MiB for 23,768
on 8 GPUs too). The Hartree device is therefore expected to stay the fullest
up to 19,392 electrons and to come to lie just below the owner from 23,768:
there the peak of the run falls by about 970 and 620 MiB, and the dense stage
of the owner sets it. This is arithmetic on those records and on the table,
made before a run with this boundary on an A100; such runs have been made
since (`../README.md`). The rule that decides
whether a sector shares its basis counts no Hartree array: what it leaves of
a device, 15% by default, holds them beside the sector operators, the filter
buffers and the dense stage.

A serial control, and a run with `PARSEC_HARTREE_BOUNDARY_KERNEL=full`, hold
more than before instead: the partial moments of the full-grid kernels grow
with `(L+1)^2`. A control has every sector alone on a device and the Hartree
objects on the first. For 14,680 electrons on four devices the former
controls peaked there at 80,203 and 80,671 MiB of the 81,152 MiB of an A100,
and the full-grid kernels at order 30 add 543 MiB: such a control is not
expected to fit. A launcher that names `PARSEC_HARTREE_BOUNDARY_BACKEND=native`
gives a control its boundary on the host, where it holds nothing on a device
(856 MiB less than the former control at that size): the C++ builders with
the tail in rows from host threads, on the orbit table where that fits and
through the full-grid recurrences in every solve where it does not (for that
grid 2.2 to 2.6 s per solve at order 30 on 28 threads of a workstation, 0.25
to 0.29 s at order 9). No GPU kernel of the boundary is then in the control;
the plan, the moments of the point charges and the geometry of the native
builder stay in both sides. On a laptop GPU such a control and the default
run of 864 electrons were 6e-11 Ry apart on the orbit table (order 12) and
2e-10 Ry through the full-grid recurrences (order 16), nine steps on both
sides, where the control on the full-grid GPU kernels was at the printed
digits of the run and the pair with PARSEC's boundary 9e-11 Ry apart.

Which thread builds what: the plan is host work of the thread that prepares
the reference, on every rank, before its grid (0.09 to 0.17 s for 14,680 to
39,368 electrons on a workstation; `hartree_boundary_plan_seconds`). Geometry,
the table of unique exterior points and the kernel of the tail run on the
Hartree set-up thread of the root rank, on the Hartree device, beside the GPU
orbital set-up of the main thread, the thread of the ionic sums and, in the
MPI runner, the thread that creates the CUDA contexts. That thread is joined
after the Poisson solver has recorded its graph on the same device. It
uploads arrays, compiles and launches kernels, copies to the host and
synchronizes its own stream; it synchronizes no device and takes no cuBLAS
or cuSOLVER handle, so by the rules of `backends/cupy_capture.py` it leaves
that capture valid while it works and when it ends. With the set-up itself
as the event inside a capture this was measured on a laptop GPU only, not on
an A100. A sector worker builds none of this.

`PARSEC_HARTREE_BOUNDARY_CHECK=n` measures the boundary values after the SCF:
at `n` of the unique exterior points the values the right-hand side was built
from are compared with the direct Coulomb sum of the final density over the
whole grid, summed on the Hartree device from the wedge rows and their images
(FP64, compensated). The `n` points are a sample, the same rule in every GPU
builder: equal shares of the innermost points, of those with the largest
atomic tail and of those nearest to an atom, each from what the earlier shares
left, and a random share. `hartree_boundary_check_max` and `_rms` are the
errors of the boundary in use over that sample, not over all points, and
`hartree_boundary_check_sample` says how many of how many points it holds;
`hartree_boundary_check_legacy_max` is the error of the expansion at
`Solver_Lpole` alone, PARSEC's, for the same density on the same points. On
the wedge of C35H36 to C185H124 a sample of 512 reached the maximum over all points
(laptop GPU). The full-grid builders hold every image of a point under the
symmetry of the system, so their shares cover fewer distinct points of a
symmetric cluster (30% low for C185H124 with 3 A of vacuum): give them four times
the count under D2. `n` at or above the number of points checks every one,
which is the true maximum: `points x grid points` terms. The sum for a sample
is `n x grid points` terms, about 1.2e10 for 512 points at 22.6 million grid
points. The check changes no result. It runs before the driver returns from
the SCF: `hartree_boundary_check_seconds` are reported apart from the SCF wall
time by the command line and by the MPI runner, and its device arrays count
in a memory peak, so timed runs leave it off.

`PARSEC_HARTREE_BOUNDARY=legacy` gives PARSEC's boundary, bitwise the former
right-hand side; `PARSEC_HARTREE_BOUNDARY_TOLERANCE` (Ry, or `off`) and
`PARSEC_HARTREE_ATOMIC_TAIL` (`auto`, `on`, `off`) replace the two input
keywords. They are applied before the reference is prepared and are part of
the key of the resident reference cache.

### The tolerance of an input without a radius

An input that leaves `Boundary_Sphere_Radius` to the default rule (reference
`README.md`, `Hartree/domain.py`) gets the tolerance of the plan from the
same rule, in the parser: `min(1e-3, (B/2) / (K sqrt(N_e)))` Ry with `B`
the `Domain_Energy_Tolerance` of the input, cut to two digits, 1.0e-3 Ry up
to 6,400 electrons and 4.0e-4 Ry at 39,368. The radius is chosen so that
some order up to 60 meets it. The plan, its estimate and every builder are
what they are for an input that holds the two values; the parser prints
them as two input lines. Orders of the benchmark clusters at the rule's
radius: 19 (3,480 electrons), 24, 23, 29, 36, 37, 44, 43 and 50 (39,368),
against 16 to 36 at their inputs with 5 ang of vacuum. A high order is cheap
on the wedge kernels: for 23,768 electrons on 16 A100 with 3.5 ang of
vacuum, order 60 took 18 MiB and 0.5 s of Hartree time more than order 46.

The switches above act after the parsing and do not move the sphere. The
set-up block prints, from the plan in use, what the boundary values are
estimated to leave in the total energy: about `2.0e-3 sqrt(N_e) e(L)`, at
most `K sqrt(N_e) e(L)`, for an engaged plan with the tail, and at most
`5e-2 sqrt(N_e) e(Solver_Lpole)` for PARSEC's values, constants measured on
hydrogen-terminated clusters of 176 to 23,768 electrons. Under
`PARSEC_HARTREE_BOUNDARY=legacy`, a tolerance switched `off`, a tail
switched `off`, an order of 60 that misses the tolerance, or a tail switched
`on` over a plan that is not engaged (PARSEC's order with the tail: neither
PARSEC's values nor measured), the line gives no number, and a run whose
radius the rule chose is warned that `Domain_Energy_Tolerance` then covers
the wall only: with PARSEC's boundary C795H300 is 3.2e-2 Ry off at 5 ang of
vacuum. A `PARSEC_HARTREE_LPOLE` or
`Solver_Lpole` so large that its estimate is within a tenth of the tolerance
leaves the plan not engaged, without the tail and on the full-grid kernels
(4.4 GiB and 14 s more for 23,768 electrons on 16 A100); the rule
never raises the order that way.

After a converged SCF the reporter estimates, on the host arrays of the root
and outside the SCF wall time, the energy the sphere adds from the density
in shells 0.4 bohr thick near it: 0.156 s and 19 MiB of work arrays for
22.6 million grid points, 0.31 s for 45 million (workstation host, blocks
of 2^20 rows). No device is touched. The record is `result.domain` of
`timing.json` and the `domain_*` entries of the backend details.

The radius it prints underneath comes with the tolerance of the rule where
the input has no `Hartree_Boundary_Tolerance` line, and is raised where
order 60 would miss that tolerance. That search makes estimates of the
omitted potential on the host, each the cost of the plan's own: none for
3,480 electrons, where the series bound of the rule settles it; none or
one of 0.12 s for 23,768 electrons; one of 0.17 s for 39,368 while the wall
leaves 4 ang of vacuum, and four to seven where it leaves less, the
tolerance halves and order 60 misses it. For the atoms of C9449H1572 the
whole block took 0.33 s (one estimate) to 1.2 s (six) on 22.6 million points
on the workstation host; `result.domain.after_scf` records `estimates`,
`shell_seconds` and `radius_seconds`. A tolerance line in the input is not
the rule's to meet: no estimate, 0.16 s. On the CuPy backend a host thread
makes the estimates of that search while the SCF runs, for every radius the
fit of the final density can lead the search to, and the block reads them
(`PARSEC_DOMAIN_REPORT_THREAD`, on by default; `0` makes them after the SCF;
`../README.md`): the lines and the record are the same, and
`radius_seconds` then counts a search that had its estimates. A run that is measured for time and
has its radius may set `PARSEC_DOMAIN_REPORT=0` (in the last table of
switches in `../README.md`), which leaves out these
reports and their work (reference `README.md`); the results are the same
bits either way.

Tests: `python -m unittest parsec_python.acceleration.tests.test_cupy_prepared`.
Real-device tests compare with direct sparse solutions, cover normalized
wedge energy/potential, warm starts, the historical predictor, breakdown,
iteration budgets, and bitwise equivalence of graph and uncaptured recurrence.
Full SCF validation must additionally compare energy, density, electron count,
convergence, and repeat timings on identical inputs/nodes/CPU thread counts.
Small consumer-GPU results are not evidence of H100 scaling. Setup/JIT warmups
must be excluded from steady-state comparisons and setup cost reported apart.
