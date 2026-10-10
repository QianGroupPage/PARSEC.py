#!/usr/bin/env bash
# Complete SCF calculation of an isolated system (molecule or cluster) on the
# GPUs of one node or of several nodes.
#
# One MPI rank per node drives all GPUs of its node. This is the layout and
# these are the settings the multi-GPU solver was measured with (1 to 16
# A100 80 GB GPUs, four per node, double precision); see "Several nodes" in
# src/parsec_python/acceleration/MULTI_GPU.md. The runner refuses periodic
# cells.
#
# Call the script by its path in the checkout, from a batch script or from the
# shell of an interactive allocation, after loading the MPI and CUDA modules
# of the site and activating the Python environment: NumPy, SciPy, CuPy,
# mpi4py built with the MPI of the site, and the native extension of this
# repository. It finds the source tree beside itself, so do not hand the
# script itself to sbatch: Slurm would run a copy from another directory. It
# requests no nodes and installs nothing. Without Slurm it runs one rank on
# the GPUs of this machine.
#
# Usage:
#   scripts/run_multi_gpu_scf.sh [--nodes N] [--gpus-per-node G] [--cpus-per-rank C]
#                                INPUT OUTPUT_DIR [runner options]
#
#   INPUT        parsec.in, with its pseudopotential files beside it
#   OUTPUT_DIR   a new directory; it receives parsec.out, timing.json and the
#                result archive (the runner refuses a directory that holds
#                earlier results)
#   --nodes      nodes to use, one rank each (default: all of the allocation)
#   --gpus-per-node   GPUs each rank drives (default 4)
#   --cpus-per-rank   CPUs of each rank as srun -c counts them (default 32,
#                which is 16 cores with two hardware threads each)
#   runner options are passed on, for example --symmetry-cache DIR or --quiet
#
# Examples:
#   scripts/run_multi_gpu_scf.sh parsec.in out                    # every node, 4 GPUs each
#   scripts/run_multi_gpu_scf.sh --nodes 1 --gpus-per-node 2 parsec.in out
#
# Environment: a setting below that is already in the environment is left as
# it is, except the OpenMP binding, which is switched off (OMP_PROC_BIND=false,
# OMP_PLACES removed). OMP_NUM_THREADS is 16 unless set, whatever
# --cpus-per-rank says. PARSEC_PYTHON names the interpreter (default: python).
#
# Deliberately not set here, so that the rules and defaults of the code act:
# where the states are stored (PARSEC_CUPY_SECTOR_STATE_STORAGE), the device
# allocator (PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR), stencil tiles
# (PARSEC_CUPY_IMPLICIT_TILE), the devices of the ionic set-up
# (PARSEC_IONIC_GPU_COUNT), whether the GPUs of a symmetry sector share its
# basis (PARSEC_CUPY_DISTRIBUTED_STATE), and the mixed-precision filter
# (PARSEC_CUPY_MIXED_FILTER, off by default).
#
# The measured inputs use Eigensolver chebff with FF_MaxIter 4,
# Chebdav_Degree 30 and Chebyshev_Degree 15; these are input keywords, not
# settings of this script.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }
fail() { printf '%s\n' "$*" >&2; exit 2; }

nodes=
gpus=4
cpus=32
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes|--gpus-per-node|--cpus-per-rank)
            [[ $# -ge 2 ]] || fail "$1 needs a value"
            case $1 in
                --nodes) nodes=$2;;
                --gpus-per-node) gpus=$2;;
                --cpus-per-rank) cpus=$2;;
            esac
            shift 2;;
        -h|--help) usage; exit 0;;
        --) shift; break;;
        -*) fail "unknown option $1";;
        *) break;;
    esac
done
if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi
input=$1
output=$2
shift 2
number='^[1-9][0-9]*$'
[[ -f $input ]] || fail "no input file $input"
[[ $gpus =~ $number && $cpus =~ $number ]] || fail 'GPU and CPU counts must be positive integers'
[[ -z $nodes || $nodes =~ $number ]] || fail 'the number of nodes must be a positive integer'

repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ -d $repo_root/src/parsec_python ]] ||
    fail "no src/parsec_python beside $repo_root/scripts: call this script by its path in the checkout"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
python=${PARSEC_PYTHON:-python}

# Threads. The solver runs one thread per symmetry sector and per device; with
# an OpenMP binding in force they would all be pinned to one core.
export OMP_PROC_BIND=false
unset OMP_PLACES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Set-up and Hartree solver on the GPU, native kernels on the host.
export PARSEC_IONIC_BACKEND="${PARSEC_IONIC_BACKEND:-cupy}"
export PARSEC_HARTREE_LINEAR_BACKEND="${PARSEC_HARTREE_LINEAR_BACKEND:-cupy}"
export PARSEC_HARTREE_BOUNDARY_BACKEND="${PARSEC_HARTREE_BOUNDARY_BACKEND:-cupy}"
export PARSEC_NATIVE_PROJECTOR_LOOKUP="${PARSEC_NATIVE_PROJECTOR_LOOKUP:-1}"
export PARSEC_NATIVE_SECTOR_ASSEMBLY="${PARSEC_NATIVE_SECTOR_ASSEMBLY:-1}"

# Eigensolver: recorded filter graphs on column-major blocks, the filter of a
# sector spread over its GPUs, the first solve with the Cholesky-whitened Ritz
# step, the small Ritz problem solved on the GPU, states kept between steps.
export PARSEC_CUPY_FILTER_GRAPHS="${PARSEC_CUPY_FILTER_GRAPHS:-1}"
export PARSEC_CUPY_FILTER_COLUMN_MAJOR="${PARSEC_CUPY_FILTER_COLUMN_MAJOR:-1}"
export PARSEC_CUPY_DISTRIBUTED_FILTER="${PARSEC_CUPY_DISTRIBUTED_FILTER:-1}"
export PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ="${PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ:-1}"
export PARSEC_CUPY_DEVICE_RANDOM="${PARSEC_CUPY_DEVICE_RANDOM:-1}"
export PARSEC_CUPY_RITZ_CONDITION="${PARSEC_CUPY_RITZ_CONDITION:-symmetric}"
export PARSEC_CUPY_RITZ_EIGH_BACKEND="${PARSEC_CUPY_RITZ_EIGH_BACKEND:-cupy}"
export PARSEC_CUPY_RITZ_ROTATION="${PARSEC_CUPY_RITZ_ROTATION:-reuse}"
export PARSEC_CUPY_RECYCLE_STATE="${PARSEC_CUPY_RECYCLE_STATE:-1}"
export PARSEC_CUPY_RESIDENT_HARTREE="${PARSEC_CUPY_RESIDENT_HARTREE:-auto}"
export PARSEC_CUPY_RESIDENT_PREDICTOR="${PARSEC_CUPY_RESIDENT_PREDICTOR:-host}"

# Stage times in timing.json.
export PARSEC_CUPY_STAGE_TIMING="${PARSEC_CUPY_STAGE_TIMING:-1}"

# Cray MPICH: the network interface nearest to the GPUs (other MPI libraries
# ignore the variable). The runner itself switches MPICH_GPU_SUPPORT_ENABLED
# off: its ranks exchange host arrays only.
export MPICH_OFI_NIC_POLICY="${MPICH_OFI_NIC_POLICY:-GPU}"

devices=0
for ((device = 1; device < gpus; device++)); do devices+=,$device; done
runner=("$python" -m parsec_python.acceleration.benchmarks.mpi_full_scf
        --input "$input" --output-dir "$output" --devices "$devices" "$@")

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    [[ -z $nodes || $nodes == 1 ]] || fail 'more than one node needs a Slurm allocation'
    exec "${runner[@]}"
fi
allocated=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-$(squeue -h -j "$SLURM_JOB_ID" -o %D 2>/dev/null | head -n 1 || true)}}
if [[ -z $nodes ]]; then
    [[ $allocated =~ $number ]] || fail 'cannot tell how many nodes the allocation has: give --nodes'
    nodes=$allocated
elif [[ $allocated =~ $number ]] && (( nodes > allocated )); then
    fail "$nodes nodes asked for, the allocation has $allocated"
fi
exec srun -N "$nodes" -n "$nodes" --ntasks-per-node=1 -c "$cpus" --cpu-bind=cores \
    --gpus-per-node="$gpus" --gpu-bind=none --kill-on-bad-exit=1 "${runner[@]}"
