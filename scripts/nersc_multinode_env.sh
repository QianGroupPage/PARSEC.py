#!/usr/bin/env bash
# Source inside a NERSC shell; this script never allocates nodes or installs packages.
# Environment of the communication and replay benchmarks that
# src/parsec_python/acceleration/experimental/README.md describes.
# scripts/run_multi_gpu_scf.sh, the launch script of a complete SCF
# calculation, needs the module loads of this file only: it keeps a thread
# count that is already set, and the two BLAS counts set below are 1 where
# the measured SCF runs had 4 (unset OPENBLAS_NUM_THREADS and MKL_NUM_THREADS
# before it, or export 4).
# Prepare a separate venv, then build its mpi4py with the loaded Cray wrappers:
#   MPICC="cc -shared" python -m pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
# Verify the active interpreter also has a CuPy build compatible with this CUDA.
if ! type module >/dev/null 2>&1; then
    printf '%s\n' 'NERSC module command is unavailable; source this file in a NERSC login shell.' >&2
    return 1 2>/dev/null || exit 1
fi
module load PrgEnv-gnu "${PARSEC_MPI_MODULE:-cray-mpich/9.1.0}" craype-accel-nvidia80 || { return 1 2>/dev/null || exit 1; }
module load "${PARSEC_MPI_CUDA_MODULE:-cudatoolkit/12.9}" || { return 1 2>/dev/null || exit 1; }
# Some Cray MPI GTL builds need a second versioned CUDA runtime while CuPy
# continues using the toolkit above. Set this explicitly from an ldd audit.
if [[ -n ${PARSEC_MPI_EXTRA_RUNTIME_DIR:-} ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${PARSEC_MPI_EXTRA_RUNTIME_DIR}"
fi
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
export MPICH_OFI_NIC_POLICY="${MPICH_OFI_NIC_POLICY:-GPU}"
# Set to 2 during the first audit to record each rank's selected NIC.
export MPICH_OFI_NIC_VERBOSE="${MPICH_OFI_NIC_VERBOSE:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# Preserve CUDA_VISIBLE_DEVICES and all explicit MPI tuning overrides. In
# particular, asynchronous progress is an experiment, not an assumed benefit.
