# Streaming Hartree, Ritz workspaces, and spatial-ordering experiments

For the subsequent affine-stencil, orbital-recycling and resident-Hartree
profile, see `GPU_COMPACT_OPTIMIZATION.md`. This document preserves the earlier
measurement cohort and its precision diagnostics.

This extends the setup profile in `GPU_SETUP_OPTIMIZATION.md`. The retained
production paths use FP64 and preserve the active grid, eighth-order stencil,
symmetry sectors, functional, pseudopotentials, Poisson tolerance and SCF
criterion. Spatial-ordering kernels are separate benchmarks until a complete
SCF measurement demonstrates a benefit.

## Run the optimized profile

Rebuild the native extension from this checkout: the GPU boundary requires
the new `MultipoleBoundaryBuilder.export_full_geometry` method. The exporter
copies the exact native source and missing-stencil geometry; it changes no
geometry or coefficients.

```bash
python -m pip install --no-deps /path/to/checkout
cd /path/to/input-directory
# with the launch settings of "Build and run" in GPU_SETUP_OPTIMIZATION.md exported
PARSEC_CUPY_DEVICES=0,1,2,3 \
  python /path/to/checkout/src/parsec_python/main.py --backend auto parsec.in --debug
```

Run inside an allocated GPU job, with the launch settings listed under
"Build and run" in [GPU_SETUP_OPTIMIZATION.md](GPU_SETUP_OPTIMIZATION.md).
The Perlmutter measurements use one task,
32 allowed logical CPUs (16 physical cores), 16 OpenMP threads and one BLAS
thread, keeping the CPU resources fixed across GPU counts. Select `0`, `0,1`
or `0,1,2` to use fewer GPUs. Native CPU-only execution is unchanged.

| Setting | Launch setting | Meaning |
|---|---|---|
| `PARSEC_CUPY_RITZ_ROTATION` | `reuse` | Consume the private filtered basis and swap it with the HX workspace after all stability checks |
| `PARSEC_HARTREE_BOUNDARY_BACKEND` | `auto` | Use GPU boundary evaluation on A100 when the native orbit coefficient table would exceed 512 MiB; retain native cached cases and other GPU models |
| `PARSEC_HARTREE_LINEAR_BACKEND` | `cupy` | Existing GPU CG; required by the automatic GPU-boundary policy |
| `PARSEC_CUPY_BOUND_SCHEDULE` | `inline` (solver default) | Keep bounds inside sector workers; `before_sectors` is an optional two-GPU scheduling variant |

`PARSEC_HARTREE_BOUNDARY_BACKEND=cupy` forces
the streaming boundary for an ablation or another GPU; `native` restores the
previous boundary. `PARSEC_CUPY_RITZ_ROTATION=allocate` restores the previous
rotation. Other GPU models need their own performance measurements before
automatic GPU-boundary selection is broadened.

## Changes and ownership

Ritz projection needs HX only until the small projected matrices and all
condition/orthogonality checks finish. With residual diagnostics disabled,
the filtered basis X is consumed and XC is written to HX; X becomes the next
scratch buffer. Public calls only take this route when `consume_basis=True`.
Residual diagnostics and the QR fallback retain their prior path. Aliasing
between the input and supplied scratch is rejected by allocating a distinct
workspace, so no matrix multiplication writes over its input.

CuPy 14.2 can allocate an internal full-size temporary for F-contiguous
`matmul(out=...)`. The retained implementation instead evaluates
`(XC).T = C.T @ X.T` into the C-contiguous transpose view of HX. Both transpose
views are free. The actual 1,031,322-by-665 test removed a 5.1098 GiB temporary
allocation. This is an allocation measurement, not a guarantee that every
whole-program GPU peak falls by that amount: initialization, live snapshots,
and memory-pool reservations also contribute.

The Hartree boundary evaluates normalized positive-m spherical harmonics in
registers, reduces fixed 256-point blocks without floating-point atomics,
then reduces their small partial moments. An exterior-stencil kernel forms
the Rydberg-unit RHS `8*pi*rho - A_IB V_B`. Geometry and partial buffers use
O(N) storage; no N-by-angular-momentum table is created. The temporary native
geometry builder and host export copies are released after GPU upload.
The existing host RHS interface, symmetry reduction and GPU CG remain in
place, so measured timings include host/device transfers.

The CUDA reduction and rotation can change floating-point summation order.
They are tolerance-equivalent, not bitwise-identical, to the prior code.

## Four-GPU acceptance measurements

One Perlmutter node with four A100 80 GB GPUs, FP64. Inputs and
geometries were unchanged; no new geometry optimization was performed. Each
measurement starts a fresh process and disables persistent symmetry caches.
Warmups are excluded. The baseline is the setup-only version of
`GPU_SETUP_OPTIMIZATION.md`; the measured combined implementation is the first
one with the Ritz workspace reuse and the GPU boundary, with subsequent
changes limited to diagnostics, automatic selection and documentation.

| Valence electrons | Prior program seconds (N=2) | Combined seconds (N=3) | Prior SCF seconds | Combined SCF seconds | SCF steps |
|---|---:|---:|---:|---:|---:|
| 3480 | 49.52 | 45.86 | 31.435 | 27.89 | 35 |
| 5264 | 136.925 | 126.94 | 108.865 | 98.68 | 66 |

Program timing excludes imports, input parsing and NPZ archival. External
process timing was also recorded. Four-GPU combined errors
relative to the previously validated GPU code were at most 5.36e-9 Ry in total
energy, 4.83e-10 in density relative L2, and 1.49e-8 Ry in eigenvalues, with
unchanged electron count and occupations to numerical precision. Independent
field/history audits and regression tests were made alongside. These
are comparisons with the previous GPU implementation, not a new independent
Fortran reference calculation.

The large-system combined profile took 315.865 / 220.91 / 188.25 / 126.94 s on
1 / 2 / 3 / 4 GPUs on this same node (respectively 2 / 1 / 1 / 3 measured runs,
after separate warmups). The four-GPU speedup is 2.49 for the whole program
and 3.58 for the orbital eigensolver. The single-sample two- and three-GPU
points have limited statistical precision. Preparation and Hartree still
limit whole-program scaling; whole-sector assignment also leaves three
GPUs with a 2+1+1 workload.

Whole-program memory needs a more careful interpretation than the rotation
microbenchmark. With one GPU, both formal combined runs peaked at 70,570 MiB
versus 70,200 MiB for the prior code: there was no measured peak reduction.
The lower combined warmup peak is excluded. For four GPUs, the corrected
rotation lowered the median sum of per-device sampled peaks from 86.41 to
71.50 GiB in the large case, but the largest device changed only from 21.89
to 21.27 GiB, with overlapping ranges. Per-device peaks are not simultaneous,
and the memory pool can retain unused blocks. No 5.11 GiB full-program
saving is inferred from the avoided temporary.

### Additional precision diagnostics

The original total-energy / density / eigenvalue / occupation / electron
count gates pass, as do the final-potential comparisons. A subsequently
added diagnostic requiring *every energy component* to differ by less than
1e-6 Ry does **not** pass at the benchmark SCF criterion of 2e-4 Ry. In the
large combined case the maximum component difference is 5.71e-5 Ry, while
the total-energy difference is 5.36e-9 Ry. The largest affected terms are
the electron-ion and Hartree contributions, of order millions of Ry.

Tightening the same SCF criterion in both versions to 2e-6 Ry gives 82
iterations in each, density relative L2 difference 7.85e-11 and maximum
component difference 3.86e-6 Ry; the latter still exceeds that diagnostic.
Independent differential integration of archived full-grid densities,
potentials and eigenpairs explains the component changes to about 2.3e-9 Ry.
This is a check of energy bookkeeping, not a replacement reference functional
evaluation. The failed absolute-component diagnostic is reported here for
that reason. Mixed-unit SCF-history arrays are reported column by column;
unweighted residual norms are not judged with an energy-unit tolerance.

A further separate pair at SCF criterion 2e-8 Ry converged in 119 iterations
in both versions. The total-energy difference was 1.05e-8 Ry, density relative
L2 difference 2.73e-11, maximum eigenvalue difference 6.05e-10 Ry, and maximum
potential difference 4.25e-9 Ry. The largest component difference remained
3.84e-6 Ry (electron-ion), so the strict absolute-component diagnostic still
fails; tighter SCF alone did not remove that floor. Its relative difference
in the roughly -2.9-million-Ry electron-ion term is about 1.3e-12. Independent
full-grid differential bookkeeping agrees to within 9.5e-9 Ry. These tighter
SCF pairs are precision studies and are excluded from performance medians.

The naive F-output reuse candidate was rejected: it retained the
temporary and was slower. cuSOLVER/ELPA policies, precision and orbital
decomposition were not changed here. No new Fortran timing was taken.

## Optional two-GPU bound scheduling

For the tested 5,264-electron, four-sector case on two A100 80 GB GPUs,
preparing each sector's unchanged Lanczos bound on its own stream before
submitting any large sector solve reduced program time. The paired experiment
used another node of the same kind, with the same fixed CPU resources, inputs,
GPU boundary, Ritz reuse and FP64 arithmetic in both arms. Two formal inline
runs took 199.59 and 206.08 s; two `before_sectors` runs took 189.32 and
189.37 s. Their medians differ by 6.65%. Both had a 39,560 MiB largest-device
peak. These timings must not be combined with the other node's scaling data.

```bash
# with the launch settings of "Build and run" in GPU_SETUP_OPTIMIZATION.md exported
PARSEC_CUPY_DEVICES=0,1 PARSEC_CUPY_BOUND_SCHEDULE=before_sectors \
  python /path/to/checkout/src/parsec_python/main.py --backend auto parsec.in --debug
```

The default stays `inline`: three- and four-GPU comparisons were slower
with bound preparation separated. Three-GPU medians (two formal runs each)
were 188.410 versus 189.795 s; the four-GPU comparison (one formal run each)
was 126.40 versus 128.87 s. The option has not been established as a general
speedup for other sizes, sector counts or hardware.

No seed, Lanczos step, tolerance, state-growth or eigensolver formula changes.
All 27 selected archived physical quantities, including the SCF history,
were bitwise identical between the inline and separated schedules in the
two-, three- and four-GPU measured pairs.
The new `eigensolver_bound_prepare_wall_seconds` counter records the separated
phase, and scheduler/SCF wall times include it. Per-sector CUDA event spans
overlap and can include launch delays; their sum is not elapsed wall time.
When explicitly selected, this phase replaces the optional collective
Lanczos scheduler, preserving each sector's owning device and stream.

## Spatial-ordering ablations

The standalone tools in `benchmarks/` capture an actual symmetry-sector
kinetic operator and compare original order, regular 2x2x2 blocks, point
Hilbert order, and Hilbert ordering between regular 2x2x2 blocks. Missing
points remain masked; no physical grid point is added. Original CSR summation
order and all coefficients are preserved. The Hilbert mapping checks complete
cubes for a unique, face-adjacent traversal.

```bash
python -m parsec_python.acceleration.benchmarks.capture_stencil \
  /path/to/sector.npz parsec.in --backend auto --no-symmetry-cache --no-archive
python -m parsec_python.acceleration.benchmarks.sfc_stencil \
  /path/to/sector.npz /path/to/stencil_results.json
```

The shared-halo variants test tiles of 64, 128 and 256 rows, both with
persistently permuted orbitals and with the original external layout.
Per-tile uint16 indices have an explicit overflow check. GPU timing warms
clocks and records 21 batches of ten applications. Setup time and metadata
bytes are separate from kernel time. This is a six-orbital kinetic-action
benchmark, not a full Hamiltonian, Chebyshev filter or SCF benchmark.

The reference SFC paper (DOI 10.1021/acs.jctc.1c00237) studied CPU MPI SpMV;
its speedup is not a GPU prediction. The SPARC GPU paper (DOI
10.1063/5.0147249) motivates GPU tiling and orbital distribution. Our existing
whole-sector assignment remains unchanged: three GPUs still receive 2+1+1
sectors. A spatial permutation alone cannot remove that load imbalance.

All 56 tested variants (two system sizes) preserved the six-orbital action
within 2e-12 absolute error. Original global-stencil medians were 0.2140 and
0.3152 ms; block-Hilbert global medians were 0.2120 and 0.3136 ms. These
approximately 0.5--1% differences do not justify production-wide reordering
and its setup/storage costs. Even the best shared-halo variants took 0.6483
and 0.9478 ms, about three times the existing global kernel. Consequently
none of the spatial-ordering variants is enabled in the production solver.
