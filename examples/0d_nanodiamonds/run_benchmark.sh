#!/usr/bin/env bash
# Run nanodiamond cases of this folder and compare every run with its
# reference: total energy, SCF steps, program time and peak GPU memory.
#
# Usage:
#   examples/0d_nanodiamonds/run_benchmark.sh [options] [CASE ...]
#
#   CASE   k06 ... k14 from gpu_benchmark/ (default: k06), or k00 ... k06 from
#          fortran_comparison/ with --fortran-set (default: k00); "all" runs
#          every case of the set, smallest first
#
# Options:
#   --nodes N           nodes, one MPI rank each (default 1); more than one
#                       needs a Slurm allocation of that many nodes
#   --gpus-per-node G   GPUs of each node (default 4)
#   --cpus-per-rank C   CPUs of each rank as srun -c counts them (default 32)
#   --launcher L        runner (default): scripts/run_multi_gpu_scf.sh, the
#                       launch settings the A100 reference times were measured
#                       with; it needs mpi4py. main: one process of
#                       src/parsec_python/main.py with the code's defaults (one
#                       node; up to four GPUs are useful)
#   --fortran-set       take the cases from fortran_comparison/
#   --out DIR           folder for the runs (default ./nanodiamond_runs_<time>)
#   --python PY         Python interpreter (default: $PARSEC_PYTHON or python)
#   --no-warmup         skip the warm-up run. By default the smallest case of
#                       the set runs once first, on the same GPUs, outside the
#                       table: it compiles the CUDA kernels into CuPy's cache,
#                       as the warm-up runs before the reference times did
#   --check             check the environment, print the commands, run nothing
#
# Examples:
#   # the smallest benchmark case on one GPU
#   examples/0d_nanodiamonds/run_benchmark.sh --gpus-per-node 1 k06
#   # three cases on the four GPUs of one node
#   examples/0d_nanodiamonds/run_benchmark.sh k06 k07 k08
#   # inside a Slurm allocation of four nodes: the largest case on 16 GPUs
#   examples/0d_nanodiamonds/run_benchmark.sh --nodes 4 k14
#   # every Fortran comparison case on one GPU, one plain process each
#   examples/0d_nanodiamonds/run_benchmark.sh --fortran-set --launcher main \
#       --gpus-per-node 1 all
#
# Run it from any directory, by its path in the checkout. It needs the Python
# environment of the repository README (NumPy, SciPy, CuPy and the native
# extension) and, for the runner, mpi4py built with the MPI of the site; on a
# cluster load the MPI and CUDA modules first. The table goes to the screen
# and to DIR/summary.csv; each run keeps its parsec.out (and, with the runner,
# timing.json) in DIR.
#
# The A100 columns are the references: 4 x A100-SXM4-80GB per node, one MPI
# rank per node (NERSC Perlmutter), first calculation without symmetry cache,
# double precision. For gpu_benchmark/ they come from the series of
# 10 October 2026, measured with the runner; a blank means that the case did
# not fit on that many 80 GB GPUs. For fortran_comparison/ the reference is the
# run of parsec_reference.out (main.py, one A100). Program time is the line
# "Total accelerated Python wall time" of parsec.out: from the end of input
# parsing to the final report, CUDA initialization included. A total energy
# within 1e-5 Ry of the reference is reported as ok; a correct run is usually
# within 1e-6 Ry.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }
fail() { printf 'run_benchmark.sh: %s\n' "$*" >&2; exit 2; }

here=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(CDPATH= cd -- "$here/../.." && pwd)
[[ -d $repo/src/parsec_python ]] || fail "no src/parsec_python in $repo: run the script inside the checkout"
python=${PARSEC_PYTHON:-python}
nodes=1
gpus=4
cpus=32
launcher=runner
set_dir=gpu_benchmark
out=
check=0
warmup=1
cases=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes|--gpus-per-node|--cpus-per-rank|--launcher|--out|--python)
            [[ $# -ge 2 ]] || fail "$1 needs a value"
            case $1 in
                --nodes) nodes=$2;;
                --gpus-per-node) gpus=$2;;
                --cpus-per-rank) cpus=$2;;
                --launcher) launcher=$2;;
                --out) out=$2;;
                --python) python=$2;;
            esac
            shift 2;;
        --fortran-set) set_dir=fortran_comparison; shift;;
        --no-warmup) warmup=0; shift;;
        --check) check=1; shift;;
        -h|--help) usage; exit 0;;
        -*) fail "unknown option $1 (see --help)";;
        *) cases+=("$1"); shift;;
    esac
done
number='^[1-9][0-9]*$'
[[ $nodes =~ $number && $gpus =~ $number && $cpus =~ $number ]] || fail 'node, GPU and CPU counts must be positive integers'
[[ $launcher == runner || $launcher == main ]] || fail "--launcher is runner or main, not $launcher"
[[ $launcher == runner || $nodes == 1 ]] || fail '--launcher main runs one process on one node'
if [[ $nodes -gt 1 && -z ${SLURM_JOB_ID:-} ]]; then
    fail "$nodes nodes need a Slurm allocation: run the script inside salloc or sbatch"
fi
total=$((nodes * gpus))

# The cases, as folders of the set.
if [[ ${#cases[@]} -eq 0 ]]; then
    [[ $set_dir == gpu_benchmark ]] && cases=(k06) || cases=(k00)
fi
folders=()
for case in "${cases[@]}"; do
    if [[ $case == all ]]; then
        for folder in "$here/$set_dir"/k*_*/; do folders+=("${folder%/}"); done
        continue
    fi
    [[ $case =~ ^k[0-9][0-9]$ ]] || fail "a case is named k00 ... k14, not $case"
    match=("$here/$set_dir/${case}"_*)
    [[ -d ${match[0]} ]] || fail "$case is not in $set_dir/ (the set has $(cd "$here/$set_dir" && ls -d k*_* | cut -c1-3 | tr '\n' ' '))"
    folders+=("${match[0]}")
done

# The environment.
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
"$python" - "$launcher" <<'EOF' || fail 'the Python environment is not complete (see above and the repository README)'
import sys
missing = []
def version(name):
    try:
        module = __import__(name)
        return getattr(module, '__version__', 'present')
    except Exception as error:
        missing.append(f'{name} ({type(error).__name__}: {error})')
        return None
found = {name: version(name) for name in ('numpy', 'scipy', 'cupy', 'parsec_accelerated_native')}
if sys.argv[1] == 'runner':
    found['mpi4py'] = version('mpi4py')
print('Python', sys.version.split()[0], '|', ', '.join(f'{name} {value}' for name, value in found.items() if value))
if found.get('cupy'):
    import cupy
    try:
        count = cupy.cuda.runtime.getDeviceCount()
        names = []
        for index in range(count):
            properties = cupy.cuda.runtime.getDeviceProperties(index)
            name = properties['name'].decode() if isinstance(properties['name'], bytes) else properties['name']
            names.append(f'{name} {properties["totalGlobalMem"] / 2**30:.1f} GiB')
        print(f'GPUs visible here: {count}' + (': ' + '; '.join(names) if names else ''))
    except Exception as error:
        print(f'GPUs visible here: none ({error}); the runs need GPUs on the nodes they run on')
if missing:
    print('missing: ' + '; '.join(missing))
    sys.exit(1)
EOF

out=${out:-nanodiamond_runs_$(date +%Y%m%d_%H%M%S)}
if [[ $check == 1 ]]; then
    out=$(cd -- "$(dirname -- "$out")" 2> /dev/null && pwd)/$(basename -- "$out")
else
    mkdir -p "$out"
    out=$(CDPATH= cd -- "$out" && pwd)
fi
summary=$out/summary.csv
[[ $check == 1 || -f $summary ]] || echo 'case,formula,launcher,gpus,status,scf_steps,reference_steps,total_energy_ry,energy_difference_ry,program_s,a100_program_s,peak_gpu_gib,a100_peak_gpu_gib' > "$summary"
echo "Runs: $out ($launcher, $nodes node(s) x $gpus GPU(s), cases: $(for folder in "${folders[@]}"; do basename "$folder"; done | tr '\n' ' '))"

compare() {  # compare <case folder> <run folder> <exit status>
    "$python" - "$1" "$2" "$3" "$launcher" "$total" "$summary" <<'EOF'
import csv, json, re, sys
from pathlib import Path
case_folder, run, status, launcher, total, summary = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6]
# Series F9 on A100-SXM4-80GB: "GPUs:program seconds:peak GiB of the fullest GPU"
A100 = {
    'k06': '1:27.8:11.2 2:17.0:6.5 4:11.1:4.2 8:7.8:3.2 16:6.0:2.2',
    'k07': '1:58.1:23.1 2:32.7:12.5 4:20.4:7.3 8:13.3:6.1 16:9.5:4.1',
    'k08': '1:90.1:34.8 2:49.3:18.5 4:28.7:10.3 8:18.6:7.3 16:12.6:5.3',
    'k09': '1:745.1:23.9 2:117.3:42.1 4:65.0:23.0 8:39.7:15.6 16:25.3:10.0',
    'k10': '2:231.6:78.3 4:123.8:42.1 8:73.8:24.5 16:44.4:14.9',
    'k11': '4:180.1:58.8 8:103.8:32.0 16:61.4:19.6',
    'k12': '8:172.5:49.5 16:98.9:28.5',
    'k13': '8:249.2:65.5 16:140.2:36.1',
    'k14': '16:279.9:61.2',
}
def energies(path):
    return [float(value) for value in re.findall(r'Total Energy =\s+(-?[0-9.]+)', path.read_text(errors='replace'))] if path.exists() else []
def seconds(path):
    found = re.findall(r'Total accelerated Python wall time \[sec\] :\s+([0-9.]+)', path.read_text(errors='replace')) if path.exists() else []
    return float(found[-1]) if found else None
case, formula = case_folder.name.split('_', 1)
reference = case_folder / 'parsec_reference.out'
ours = run / 'parsec.out'
reference_energies, our_energies = energies(reference), energies(ours)
peak = None
timing = run / 'timing.json'
if timing.exists():
    record = json.loads(timing.read_text())
    peak = (record.get('gpu_peak_used_mib_max') or 0) / 1024 or None
samples = run / 'gpu_memory.csv'
if peak is None and samples.exists():
    values = [float(line.split(',')[1]) for line in samples.read_text().splitlines() if line.count(',') == 1 and line.split(',')[1].strip().replace('.', '', 1).isdigit()]
    peak = max(values) / 1024 if values else None
a100_seconds = a100_peak = None
if case_folder.parent.name == 'gpu_benchmark':
    for item in A100.get(case, '').split():
        count, value, memory = item.split(':')
        if int(count) == total:
            a100_seconds, a100_peak = float(value), float(memory)
elif total == 1:
    a100_seconds = seconds(reference)
energy = our_energies[-1] if our_energies else None
difference = energy - reference_energies[-1] if energy is not None and reference_energies else None
if status != 0 or energy is None:
    verdict = 'FAILED'
elif abs(difference) <= 1e-5:
    verdict = 'ok'
else:
    verdict = 'DIFFERS'
row = [case, formula, launcher, total, verdict, len(our_energies) or '', len(reference_energies), '' if energy is None else f'{energy:.8f}',
       '' if difference is None else f'{difference:.1e}', '' if seconds(ours) is None else f'{seconds(ours):.1f}',
       '' if a100_seconds is None else f'{a100_seconds:.1f}', '' if peak is None else f'{peak:.1f}', '' if a100_peak is None else f'{a100_peak:.1f}']
with open(summary, 'a', newline='') as handle:
    csv.writer(handle).writerow(row)
print(f'  {case} {formula}: {verdict}, {row[5] or "?"} steps (reference {row[6]}), E = {row[7] or "?"} Ry (difference {row[8] or "?"}), '
      f'{row[9] or "?"} s (A100 {row[10] or "n/a"}), peak {row[11] or "?"} GiB (A100 {row[12] or "n/a"})')
EOF
}

command_for() {  # command_for <case folder> <run folder>: fills the array "command"
    if [[ $launcher == runner ]]; then
        command=("$repo/scripts/run_multi_gpu_scf.sh" --nodes "$nodes" --gpus-per-node "$gpus" --cpus-per-rank "$cpus"
                 "$1/parsec.in" "$2" --pp-dir "$here")
    else
        command=("$python" "$repo/src/parsec_python/main.py" "$1/parsec.in" --pp-dir "$here" --device gpu
                 --gpus "$gpus" --no-archive --log "$2/parsec.out")
    fi
}

if [[ $warmup == 1 ]]; then
    smallest=("$here/$set_dir"/k*_*)
    name=$(basename "${smallest[0]}")
    command_for "${smallest[0]}" "$out/warmup_$name"
    if [[ $check == 1 ]]; then
        printf 'warm-up %s:' "$name"; printf ' %q' "${command[@]}"; printf '\n'
    elif [[ -e $out/warmup_$name ]]; then
        echo "warm-up: $out/warmup_$name exists already, skipped"
    else
        [[ $launcher == main ]] && mkdir -p "$out/warmup_$name"
        echo "warm-up: $name on the same GPUs, not in the table (compiles the CUDA kernels) $(date +%T)"
        PARSEC_PYTHON=$python "${command[@]}" > "$out/warmup_$name.log" 2>&1 ||
            echo "  the warm-up run failed (exit status $?), see $out/warmup_$name.log"
    fi
fi

failures=0
for folder in "${folders[@]}"; do
    name=$(basename "$folder")
    run=$out/${name}_${total}gpu_$launcher
    [[ -e $run ]] && fail "$run exists already: choose another --out"
    command_for "$folder" "$run"
    if [[ $check == 1 ]]; then
        printf '%s:' "$name"; printf ' %q' "${command[@]}"; printf '\n'
        continue
    fi
    echo "$name: started $(date +%T)"
    sampler=
    if [[ $launcher == main ]]; then
        mkdir -p "$run"
        if command -v nvidia-smi > /dev/null 2>&1; then
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 200 > "$run/gpu_memory.csv" 2> /dev/null &
            sampler=$!
        fi
    fi
    status=0
    PARSEC_PYTHON=$python "${command[@]}" > "$out/$name.log" 2>&1 || status=$?
    [[ -n $sampler ]] && { kill "$sampler" 2> /dev/null || true; wait "$sampler" 2> /dev/null || true; }
    [[ $status == 0 ]] || { failures=$((failures + 1)); echo "  exit status $status, see $out/$name.log"; }
    compare "$folder" "$run" "$status" || true
done
[[ $check == 1 ]] && exit 0

echo
echo "Summary ($summary):"
if command -v column > /dev/null 2>&1; then column -s, -t < "$summary"; else cat "$summary"; fi
if [[ $launcher == main && $set_dir == gpu_benchmark ]]; then
    echo "Note: the A100 times of gpu_benchmark/ were measured with the runner. main.py with the code's"
    echo "defaults keeps the Hartree solver and parts of the set-up on the CPU and is slower; compare"
    echo "times with the default launcher."
fi
exit $((failures > 0 ? 1 : 0))
