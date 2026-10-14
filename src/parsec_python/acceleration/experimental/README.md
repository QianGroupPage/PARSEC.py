# Multi-node GPU experiments

With one exception these modules are **not an SCF backend** and are not
selected by the production driver. They replay fixed Hamiltonians and saved
orbital blocks to validate distribution and measure communication costs.
Passing these tests does not establish SCF convergence, total-energy parity,
or end-to-end speedup. The exception is `mpi_scf.py`, the MPI sector context
of the full-SCF runner `benchmarks/mpi_full_scf.py` (`../README.md`,
`../MULTI_GPU.md`).

## Representations

- `domain_partition.py`: axis, recursive geometric brick, Morton and block
  Hilbert ownership. The active grid, symmetry normalization, stencil slots and
  coefficients are preserved. No points are added to partially occupied blocks.
- `mpi_domain.py`: persistent row-distributed orbitals, unique stencil halos,
  KB coefficient reduction, reduced-matrix projection and local Ritz rotation.
  Host setup is replicated. The small generalized eigenproblem runs on rank 0.
- `mpi_domain_overlap.py`: optional interior/halo overlap and validated fixed
  widths. `collective_checks=False` removes repeated diagnostic collectives;
  every rank must still call operations in the same order with validated widths.
  The outer application must abort the MPI job on rank-local failure.
- `mpi_compact_projectors.py`: communicates only coefficients of projectors
  whose nonzero support intersects more than one rank. Wholly local and empty
  projectors do not need a reduction. Signs are applied after summation.
- `orbital_layout.py`: orbital-column ownership for communication-free local H
  applications, with packed `Alltoallv` conversions to and from row ownership.
  Static Hamiltonian data remain replicated. At one rank, valid F-contiguous
  FP64 inputs alias the output unless an explicit output buffer is supplied.
- `column_ritz.py`: complete column-layout generalized Ritz, including all
  three layout conversions, projection, root solve, rotation and residuals.
- `host_gather.py`: column-wise host input gathering for F-order captures.
  Contiguous selections can alias the source or read-only memory map; callers
  must treat the result as read-only input. This optimizes excluded input
  preparation, not GPU kernels.
- `mpi_scf.py`: root-controlled SCF with persistent symmetry-sector workers,
  used by `benchmarks/mpi_full_scf.py` and by the driver for the device groups
  of its sectors. Rank zero runs the production SCF loop, and every rank
  solves the sectors it owns on its own GPUs; MPI carries scalar potentials
  and densities and small Ritz data, never orbital matrices.
- `nccl_reductions.py`: optional NCCL sums for the CUDA replay; point-to-point
  halos still use MPI.

All inner products use the Euclidean metric of the normalized symmetry sector.
Do not multiply them by orbit multiplicities. The captured Hamiltonian is in Ry:
`T + diag(V) + B diag(signs) B.T`. In the replay modules spin, occupations,
charge-density expansion, mixing, state growth, restarts and production solver
fallback are not integrated.

## Run on allocated NERSC GPUs

Build mpi4py against the active Cray compiler wrapper in an isolated environment,
not against a bundled alternative MPI. The October 2026 tests used Cray MPICH
9.1 and CuPy with CUDA 12.9. That MPI installation additionally required its
CUDA 13 runtime library on the library search path. Record `ldd`, module versions
and the actual MPI library string; this is an environment-specific dependency.

```bash
export PARSEC_MPI_EXTRA_RUNTIME_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64
source scripts/nersc_multinode_env.sh
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
# Inside an existing allocation; four MPI ranks and four GPUs per node:
bash scripts/nersc_multinode_run.sh --nodes 2 -- python -m \
  parsec_python.acceleration.benchmarks.mpi_communication --output comm.json
```

The launcher uses 32 logical CPUs (16 physical cores on Perlmutter) per rank,
allocates GPUs at node level, and selects the GPU before MPI initialization.
Do not assume an arbitrary per-task GPU cgroup preserves CUDA IPC connectivity.
The scripts do not request an allocation themselves.

`benchmarks.capture_distributed` exports an actual sector after a chosen SCF
eigensolver call. It requires a run without a symmetry cache, which is the
default (no `--symmetry-cache`), and device-resident state.
Its intentional early exit is not a converged calculation. The manifest is
written last; incomplete directories must not be replayed.

`benchmarks.mpi_domain_replay --reference` builds an independent chunked CPU
Hamiltonian and Ritz reference. `--run` validates H actions, reduced matrices,
eigenvalues, residual norms and rotated-orbital orthogonality before timing.
Use `--implementation base|overlap|compact` and
`--collective-checks safe|fast`. Defaults remain base/safe.

`benchmarks.nccl_communication` compares identical FP64 Allreduces in MPI and
NCCL. On Perlmutter, load an NCCL module containing the OFI plugin. Save NCCL's
INIT/NET log and verify actual Libfabric/GDR selection; the requested backend
name alone is not evidence that the plugin was used.

`orbital_layout.py` and `benchmarks.mpi_filter_layout_replay` test the other
distribution: keep full grid rows and partition orbital columns during repeated
Hamiltonian applications, then use GPU `Alltoallv` to change layout. The timed
two-exchange round trip does not include a Ritz solve. The column path uses
production kernels and fused recurrence, whereas the row path uses experimental
halo kernels; that comparison does not isolate data layout alone. Column
filtering uses a uniform polynomial degree and fixed captured bounds. Production
adaptive block degrees and SCF updates are not reproduced.

`benchmarks.mpi_complete_cycle` measures a complete fixed-input column filter
plus `complete_column_ritz`: H of the filtered orbitals, both X and HX from
columns to rows, FP64 Gram reductions, root generalized solve and broadcast,
row rotation, global residual norms, then rotated X back to columns. All three
layout conversions are included; tall orbitals are never gathered. `--reference`
generates serial correctness data using the same implementation. Use a separate
**`--run` with one MPI rank** for the strong-scaling baseline, then `--run` at
larger rank counts with identical capture, width, degree and transport.

The Ritz primitive requires a positive-definite overlap with condition number
at most `1e8` by default, successful Cholesky factorization and
`max(abs(C.T @ S @ C - I)) <= 5e-10`. It fails collectively rather than attempting
a QR/TSQR fallback. Process/network failures still require the runner's MPI abort.
Before timing, the benchmark checks every eigenvalue, projected-matrix entry
and residual norm against its serial reference with finite positive tolerances
(default `atol=rtol=1e-8`). These checks do not establish full-SCF accuracy.

For example, inside an existing four-node allocation, after sourcing the
environment above and activating the environment containing CuPy and mpi4py:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
CAP=/path/to/capture   # a directory written by benchmarks.capture_distributed
REF="$PWD/complete_reference_${SLURM_JOB_ID}" # must be a new directory
BENCH=parsec_python.acceleration.benchmarks.mpi_complete_cycle
COMMON=(--capture "$CAP" --reference-dir "$REF" --width 96 --degree 8
        --transport cuda --warmups 2 --repeats 5)
# The launcher below uses four ranks/node, so launch one-rank cases directly.
for mode in reference run; do
  srun -N1 -n1 --ntasks-per-node=1 -c32 --cpu-bind=cores \
    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    python -m "$BENCH" "--$mode" "${COMMON[@]}" --output "complete_${mode}_1g.json"
done
for nodes in 1 2 4; do
  bash scripts/nersc_multinode_run.sh --nodes "$nodes" -- \
    python -m "$BENCH" --run "${COMMON[@]}" --output "complete_$((4*nodes))g.json"
done
```

The four-sector integration that followed assigns whole sectors to nodes (one
each on four nodes) and keeps the four-GPU filtering and projection traffic of
a sector node-local (`mpi_scf.py`, `../Eigensolvers/distributed_state.py`):
only scalar potentials and densities and small Ritz data cross nodes. The
replay cycle above is not that integration and implements none of its steps.

## Measurement limits

Use synchronized **external** maximum-rank wall times. Fast paths can enqueue
GPU work asynchronously, so internal enqueue times are not kernel wall times.
Preparation, disk I/O and hashing are reported separately. Memory observations
are CuPy pool high-water values and synchronized device-use samples, not a
continuously sampled process allocation peak. Compare the same capture, column
count, polynomial degree, transport, implementation and source version.

The motivation follows Sharma et al., JCP 158, 204117 (2023), DOI
10.1063/5.0147249, and Liou et al., JCTC 17, 4039–4048 (2021), DOI
10.1021/acs.jctc.1c00237. The measured SPARC GPU implementation used orbital
column distribution, not spatial domain decomposition. The PARSEC SFC paper
studied finite-difference CPU MPI domain decomposition. Neither establishes that
Hilbert ordering or domain decomposition will be faster for these GPU inputs.
