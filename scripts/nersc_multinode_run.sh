#!/usr/bin/env bash
# Launch inside an EXISTING allocation. Never calls salloc or sbatch.
# For the communication and replay benchmarks of
# src/parsec_python/acceleration/benchmarks that
# src/parsec_python/acceleration/experimental/README.md describes (four ranks
# per node, one GPU each). A complete SCF calculation runs one rank per node:
# use scripts/run_multi_gpu_scf.sh for it.
# Usage: bash scripts/nersc_multinode_run.sh [--nodes N] [--] [command args...]
# With no command, runs the FP64 communication audit with PARSEC_MPI_PYTHON.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf '%s\n' 'Usage: nersc_multinode_run.sh [--nodes N] [--] [command args...]' \
        'Requires an existing NERSC GPU allocation; defaults to its full node count.' \
        'Runs 4 ranks/node, 1 GPU/rank, 32 logical CPUs/rank. Never allocates nodes.'
    exit 0
fi
: "${SLURM_JOB_ID:?Run this script inside an existing GPU allocation}"
allocated_nodes=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}
: "${allocated_nodes:?SLURM_JOB_NUM_NODES or SLURM_NNODES is required}"
nodes=$allocated_nodes
if [[ ${1:-} == --nodes ]]; then
    [[ $# -ge 2 ]] || { printf '%s\n' '--nodes requires a value' >&2; exit 2; }
    nodes=$2
    shift 2
fi
[[ ${1:-} != -- ]] || shift
[[ $nodes =~ ^[1-9][0-9]*$ && $allocated_nodes =~ ^[1-9][0-9]*$ ]] || {
    printf '%s\n' 'Node count must be a positive integer' >&2; exit 2;
}
(( nodes <= allocated_nodes )) || { printf '%s\n' 'Requested nodes exceed the existing allocation' >&2; exit 2; }
if [[ $# == 0 ]]; then
    set -- "${PARSEC_MPI_PYTHON:-python}" \
        "$repo_root/src/parsec_python/acceleration/benchmarks/mpi_communication.py" \
        --output "mpi_communication_${SLURM_JOB_ID}_${nodes}nodes.json"
fi
# Node-level GPU allocation avoids per-task cgroups that can inhibit CUDA IPC.
# Select from Slurm's existing visibility list, retaining UUIDs or physical IDs.
exec srun --nodes="$nodes" --ntasks="$((4 * nodes))" --ntasks-per-node=4 \
    --cpus-per-task=32 --cpu-bind=cores --gpus-per-node=4 --gpu-bind=none \
    --kill-on-bad-exit=1 bash -c '
        set -euo pipefail
        : "${SLURM_LOCALID:?Missing per-node rank}"
        export PARSEC_MPI_ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-<unset>}"
        if [[ -v CUDA_VISIBLE_DEVICES ]]; then
            [[ -n $CUDA_VISIBLE_DEVICES ]] || { echo "No visible GPU" >&2; exit 2; }
            IFS=, read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
            if (( ${#visible_gpus[@]} == 1 )); then
                export CUDA_VISIBLE_DEVICES="${visible_gpus[0]}"
            elif (( SLURM_LOCALID < ${#visible_gpus[@]} )); then
                export CUDA_VISIBLE_DEVICES="${visible_gpus[$SLURM_LOCALID]}"
            else
                echo "Rank cannot select a GPU from Slurm visibility" >&2; exit 2
            fi
        else
            export CUDA_VISIBLE_DEVICES="$SLURM_LOCALID"
        fi
        exec "$@"
    ' parsec-mpi-rank "$@"
