"""Compare NCCL and CUDA-aware MPI Allreduce on identical allocated GPUs.

Load NERSC's NCCL module (including the AWS OFI plugin) before launching.
This is a communication microbenchmark, not an SCF performance result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

from .mpi_communication import parse_bytes, summarize, visible_device_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sizes', nargs='+', type=parse_bytes,
                        default=[8192, 1048576, 8388608, 33554432])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=5)
    args = parser.parse_args()
    if min(args.repeats, args.iterations) < 1:
        parser.error('repeat and iteration counts must be positive')
    import cupy as cp
    from cupy.cuda import nccl
    local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
    device = visible_device_index(local_rank, cp.cuda.runtime.getDeviceCount())
    cp.cuda.Device(device).use()
    cp.cuda.runtime.free(0)
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
    if MPI.Is_initialized():
        raise RuntimeError('GPU must be selected before MPI initialization')
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    world = MPI.COMM_WORLD
    rank = world.rank
    try:
        if os.environ.get('MPICH_GPU_SUPPORT_ENABLED') != '1':
            raise RuntimeError('MPICH_GPU_SUPPORT_ENABLED=1 required')
        devices = world.allgather((socket.gethostname(), cp.cuda.Device(device).pci_bus_id))
        if len(set(devices)) != world.size:
            raise RuntimeError('duplicate GPU assignment')
        node = world.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
        lane = world.Split(node.rank, rank)
        records = []
        stream = cp.cuda.get_current_stream()
        for label, comm in [('world', world), ('within_node', node), ('across_nodes_lane', lane)]:
            world.Barrier()
            if comm.size < 2:
                continue
            uid = comm.bcast(nccl.get_unique_id() if comm.rank == 0 else None, root=0)
            communicator = nccl.NcclCommunicator(comm.size, uid, comm.rank)
            for size in args.sizes:
                count = size // 8
                send = cp.full(count, float(comm.rank + 1), dtype=cp.float64)
                recv = cp.empty_like(send)
                expected = comm.size * (comm.size + 1) // 2
                for transport in ['cuda_aware_mpi', 'nccl_ofi']:
                    def operation():
                        stream.synchronize()
                        if transport == 'nccl_ofi':
                            communicator.allReduce(send.data.ptr, recv.data.ptr, count,
                                                   nccl.NCCL_FLOAT64, nccl.NCCL_SUM, stream.ptr)
                        else:
                            comm.Allreduce(send, recv, op=MPI.SUM)
                        stream.synchronize()
                    recv.fill(float('nan'))
                    operation()
                    if not comm.allreduce(bool(cp.all(recv == expected).item()), op=MPI.LAND):
                        raise AssertionError('FP64 Allreduce validation failed')
                    for _ in range(2):
                        operation()
                    samples = []
                    for _ in range(args.repeats):
                        world.Barrier()
                        start = MPI.Wtime()
                        for _ in range(args.iterations):
                            operation()
                        samples.append((MPI.Wtime() - start) / args.iterations)
                    world.Barrier()
                    if not comm.allreduce(bool(cp.all(recv == expected).item()), op=MPI.LAND):
                        raise AssertionError('post-timing validation failed')
                    all_samples = world.allgather(samples)
                    if rank == 0:
                        records.append(dict(scope=label, transport=transport,
                            payload_bytes=size, validated_exact=True,
                            max_rank_seconds=summarize([max(x[i] for x in all_samples)
                                                       for i in range(args.repeats)]),
                            per_rank_seconds=all_samples))
                del send, recv
            communicator.destroy()
            world.Barrier()
        metadata = world.gather(dict(rank=rank, device=devices[rank], cupy=cp.__version__,
            nccl_version=nccl.get_version(), mpi_library=MPI.Get_library_version(),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            environment={k: v for k, v in os.environ.items()
                         if k.startswith(('NCCL_', 'FI_CXI_', 'MPICH_'))}), root=0)
        if rank == 0:
            data = dict(kind='nccl_mpi_allreduce_microbenchmark_not_scf', status='passed',
                        rank_count=world.size, parameters=vars(args), records=records,
                        metadata=metadata,
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(data, indent=2))
            print('NCCL_MPI_COMPARISON_PASSED', output, flush=True)
        lane.Free()
        node.Free()
        world.Barrier()
    except BaseException as exc:
        print(f'NCCL_COMPARISON_FAILED rank={rank}: {exc}', flush=True)
        world.Abort(1)
    finally:
        if not MPI.Is_finalized():
            MPI.Finalize()


if __name__ == '__main__':
    main()
