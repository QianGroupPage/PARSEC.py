"""Complete captured column filter+Ritz cycle; not an SCF benchmark.

Create --reference with one GPU, then --run at any rank count using the same
capture, width, degree and reference directory.  All three Ritz layout
changes are included in the directly measured cycle, including residuals.
"""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from parsec_python.acceleration.benchmarks import mpi_filter_layout_replay as support


def execute(args, cp, comm=None):
    import numpy as np
    from parsec_python.acceleration.experimental.orbital_layout import ColumnLayout, column_chebyshev_filter
    from parsec_python.acceleration.experimental.column_ritz import complete_column_ritz
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)
    load_started = time.perf_counter()
    helper, arrays, basis, metadata, projectors, capture, eigenvalues = support.load_capture(args)
    folder = Path(args.reference_dir)
    hashes = support.capture_hashes(helper, args.capture)
    if args.reference:
        if size != 1 or int(os.environ.get('SLURM_NTASKS', '1')) != 1:
            raise ValueError('reference requires exactly one task/GPU')
        if (folder/'complete_reference.json').exists():
            raise FileExistsError('use a new reference directory')
        bounds = support.conservative_bounds(arrays, projectors, eigenvalues)
    else:
        manifest = json.loads((folder/'complete_reference.json').read_text())
        if not manifest.get('complete') or manifest['capture_sha256'] != hashes:
            raise ValueError('reference capture hashes do not match')
        if any(manifest[k] != getattr(args, k) for k in ('width', 'degree', 'column_start')):
            raise ValueError('reference parameters do not match')
        if helper.digest(folder/'complete_reference.npz') != manifest['arrays_sha256']:
            raise ValueError('reference arrays checksum failed')
        bounds = manifest['bounds']
    load_seconds = time.perf_counter()-load_started
    setup_started = time.perf_counter()
    layout = ColumnLayout(metadata, arrays['local_potential'], projectors, arrays['signs'],
                          args.width, comm=comm, xp=cp, transport=args.transport)
    local = cp.asarray(basis[:, args.column_start+layout.column_start:
                                  args.column_start+layout.column_stop], order='F')
    cp.cuda.get_current_stream().synchronize()
    setup_seconds = time.perf_counter()-setup_started
    observed_memory = {}
    def observe_memory():
        cp.cuda.get_current_stream().synchronize()
        free, total = cp.cuda.runtime.memGetInfo()
        observations = dict(cupy_pool_reserved_high_water_bytes=cp.get_default_memory_pool().total_bytes(),
                            cupy_pool_used_observed_max_bytes=cp.get_default_memory_pool().used_bytes(),
                            device_memory_used_observed_max_bytes=total-free)
        for key, value in observations.items():
            observed_memory[key] = max(observed_memory.get(key, 0), int(value))
    observe_memory()
    parameters = dict(degree=args.degree, **bounds)
    def cycle():
        filtered = column_chebyshev_filter(layout, local, **parameters)
        return complete_column_ritz(layout, filtered, compute_residuals=True)
    initial = cycle()
    cp.cuda.get_current_stream().synchronize()
    observe_memory()
    compared = dict(eigenvalues=initial.eigenvalues, overlap=initial.overlap,
                    projected_hamiltonian=initial.projected_hamiltonian,
                    residual_norms=initial.residual_norms)
    checks = {}
    if args.reference:
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder/'complete_reference.npz', **compared)
        helper.atomic_json(folder/'complete_reference.json', dict(
            complete=True, kind='same_implementation_serial_complete_cycle_reference_not_scf',
            capture_sha256=hashes, arrays_sha256=helper.digest(folder/'complete_reference.npz'),
            width=args.width, degree=args.degree, column_start=args.column_start,
            bounds=bounds, orthogonality_error=initial.orthogonality_error))
    else:
        with np.load(folder/'complete_reference.npz', allow_pickle=False) as reference:
            for name, actual in compared.items():
                expected = reference[name]
                delta = np.abs(actual-expected)
                allowed = args.atol+args.rtol*np.abs(expected)
                passed = bool(np.all(np.isfinite(actual)) and np.all(delta <= allowed))
                checks[name] = dict(passed=passed, max_absolute_error=float(delta.max(initial=0)),
                                    max_tolerance_fraction=float((delta/allowed).max(initial=0)))
                if not passed:
                    raise AssertionError(f'{name} failed serial reference: {checks[name]}')
    if initial.orthogonality_error > 5e-10:
        raise AssertionError('generalized Ritz orthogonality audit failed')
    initial_orthogonality_error = initial.orthogonality_error
    initial_condition_number = initial.condition_number
    # Keep only small diagnostic matrices.  Do not pin the validation cycle's
    # tall output orbital array throughout warmups and measured iterations.
    del initial
    for _ in range(args.warmups):
        result = cycle()
        cp.cuda.get_current_stream().synchronize()
        observe_memory()
        del result
    before = dict(layout.stats)
    samples, phases = [], []
    for _ in range(args.repeats):
        cp.cuda.get_current_stream().synchronize()
        if comm is not None:
            comm.Barrier()
        start = time.perf_counter()
        result = cycle()
        cp.cuda.get_current_stream().synchronize()
        samples.append(time.perf_counter()-start)
        phases.append(result.timings)
        observe_memory()
        del result
    statistics = {name: layout.stats[name]-before[name] for name in before}
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    gpu_name = properties['name']
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode('utf-8', errors='replace')
    mpi_library = None
    if comm is not None:
        from mpi4py import MPI
        mpi_library = MPI.Get_library_version().strip()
    controls = ('SLURM_JOB_ID', 'SLURM_LOCALID', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'MPICH_GPU_SUPPORT_ENABLED',
                'MPICH_OFI_NIC_POLICY', 'MPICH_ASYNC_PROGRESS')
    environment = {key: os.environ.get(key) for key in controls}
    environment.update({key: value for key, value in os.environ.items()
                        if key.startswith('PARSEC_CUPY_')})
    payload = dict(rank=rank, hostname=socket.gethostname(),
                   gpu_pci_bus_id=cp.cuda.Device().pci_bus_id, samples_seconds=samples,
                   ritz_phase_seconds=phases, counters=statistics, checks=checks,
                   orthogonality_error=initial_orthogonality_error,
                   condition_number=initial_condition_number,
                   gpu_name=gpu_name, gpu_total_memory_bytes=int(properties['totalGlobalMem']),
                   cupy=cp.__version__, numpy=np.__version__, mpi_library=mpi_library,
                   cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                   environment=environment, load_hash_and_reference_seconds=load_seconds,
                   layout_and_input_setup_seconds=setup_seconds,
                   observed_memory=observed_memory)
    records = [payload] if comm is None else comm.gather(payload, root=0)
    layout.close()
    if rank == 0:
        critical = [max(r['samples_seconds'][i] for r in records) for i in range(args.repeats)]
        root = Path(__file__).resolve().parents[1]
        sources = support.source_hashes(helper)
        sources['experimental/column_ritz.py'] = helper.digest(root/'experimental/column_ritz.py')
        sources['benchmarks/mpi_complete_cycle.py'] = helper.digest(__file__)
        report = dict(kind='captured_complete_column_filter_ritz_cycle_not_scf', status='passed',
                      ranks=size, nodes=len({r['hostname'] for r in records}), capture=capture,
                      capture_sha256=hashes, source_sha256=sources, parameters=vars(args),
                      bounds=bounds, per_rank=records, cycle_seconds_max_rank=critical,
                      median_cycle_seconds=float(np.median(critical)),
                      notes=['Same implementation is used for serial and MPI reference comparison.',
                             'Cycle includes degree-N filtering, H(filtered X), X and HX forward layout changes,',
                             'Gram reductions, root generalized solve, coefficient broadcast, row rotation,',
                             'global residual norms and final backward layout change: all three Ritz transposes.',
                             'One-rank layout identities do not copy or communicate valid F-FP64 inputs.',
                             'Use a separate --run with one MPI rank as the MPI-consistent strong-scaling baseline; --reference is for correctness.',
                             'Capture loading/hashing, layout construction and input upload are excluded from cycle timing and reported separately.',
                             'The initial tall validation result is released before warmup/timing. Memory observations include validation, warmups and measured cycles; reserved pool memory may remain cached.',
                             'Memory is observed at synchronization points, not continuously sampled allocation peak.',
                             'No density, Hartree update or SCF convergence is included.',
                             'Residual/eigen/projection reference checks precede timing; no tall orbital gather.'])
        helper.atomic_json(args.output, report)
        print('COMPLETE_COLUMN_CYCLE_PASSED', args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--reference', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--capture', required=True)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--column-start', type=int, default=0)
    parser.add_argument('--degree', type=int, default=8)
    parser.add_argument('--transport', choices=('host', 'cuda'), default='cuda')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--atol', type=float, default=1e-8)
    parser.add_argument('--rtol', type=float, default=1e-8)
    args = parser.parse_args()
    if min(args.width, args.degree, args.repeats) < 1 or args.warmups < 0 or args.column_start < 0:
        parser.error('invalid counts')
    if not all(math.isfinite(value) and value > 0 for value in (args.atol, args.rtol)):
        parser.error('atol and rtol must be finite and positive')
    cp, _, _ = support.select_gpu()
    if args.reference:
        execute(args, cp)
        return
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
    if MPI.Is_initialized():
        raise RuntimeError('MPI initialized before GPU selection')
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    try:
        if 'cray' in MPI.Get_library_version().lower() and os.environ.get('MPICH_GPU_SUPPORT_ENABLED') != '1':
            raise RuntimeError('Cray GPU support is not enabled')
        devices = MPI.COMM_WORLD.allgather((socket.gethostname(), cp.cuda.Device().pci_bus_id))
        if len(set(devices)) != len(devices):
            raise RuntimeError('multiple ranks selected the same GPU')
        execute(args, cp, MPI.COMM_WORLD)
        MPI.COMM_WORLD.Barrier()
    except BaseException as exc:
        print(f'COMPLETE_CYCLE_FAILED rank={MPI.COMM_WORLD.rank}: {exc}', file=sys.stderr, flush=True)
        MPI.COMM_WORLD.Abort(1)
        raise
    finally:
        if not MPI.Is_finalized():
            MPI.Finalize()


if __name__ == '__main__':
    main()
