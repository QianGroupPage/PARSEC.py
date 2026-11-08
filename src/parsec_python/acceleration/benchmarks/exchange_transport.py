"""Bandwidth of the two moves of the exact exchange across MPI ranks: broadcasts and point-to-point sums.

The distributed exchange of a hybrid functional (``acceleration/SCF/hybrid_mpi.py``) broadcasts the
occupied orbitals of each symmetry sector from its owner rank and sends the parts of ``W = V_x Psi``
that other ranks formed back to the owner.  This measures both on the launch layout of the MPI runner
(one rank per node) with device buffers handed to MPI (Cray MPICH with ``MPICH_GPU_SUPPORT_ENABLED=1``)
and with host staging, in chunks of the size the exchange uses.

Usage (inside an allocation, one rank per node)::

    srun -N 2 -n 2 --ntasks-per-node=1 --gpus-per-node=4 python -m \
        parsec_python.acceleration.benchmarks.exchange_transport --transport cuda --gib 2

Prints one JSON line per measurement on rank 0.  A communication benchmark, not a DFT result.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("cuda", "host"), default="cuda")
    parser.add_argument("--gib", type=float, default=2.0, help="bytes moved per measurement")
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args(argv)
    os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "1" if arguments.transport == "cuda" else "0"
    os.environ["PARSEC_MPI_EXCHANGE_TRANSPORT"] = arguments.transport
    import cupy as cp
    import numpy as np
    import mpi4py

    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI

    MPI.Init_thread(MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    try:
        from parsec_python.acceleration.SCF.hybrid_mpi import ExchangeTransport

        transport = ExchangeTransport(comm)
        rows = 1 << 20
        columns = max(1, int(arguments.gib * 2**30 / (8 * rows)))
        with cp.cuda.Device(0):
            block = cp.asfortranarray(cp.full((rows, columns), float(comm.rank + 1)))
            cp.cuda.Device(0).synchronize()
        results = []
        for repeat in range(arguments.repeats):
            # Broadcast from every rank in turn, as the owners of the sectors do.
            comm.Barrier()
            started = time.perf_counter()
            for owner in range(comm.size):
                with cp.cuda.Device(0):
                    target = block if comm.rank == owner else cp.empty((rows, columns), order="F")
                transport.broadcast(target, owner)
            comm.Barrier()
            seconds = time.perf_counter() - started
            results.append(dict(kind="broadcast", repeat=repeat, ranks=comm.size, transport=transport.kind,
                                bytes_per_owner=int(block.nbytes), seconds=seconds,
                                gb_per_s_received=(comm.size - 1) * block.nbytes / seconds / 1e9))
            # Every rank sends its block to every other rank, which adds them up in rank order.
            comm.Barrier()
            started = time.perf_counter()
            with cp.cuda.Device(0):
                total = cp.zeros_like(block)
                transport.reduce_to_owners(
                    [(owner, block, np.arange(columns)) for owner in range(comm.size) if owner != comm.rank],
                    [(source, total, np.arange(columns)) for source in range(comm.size) if source != comm.rank],
                )
                cp.cuda.Device(0).synchronize()
            comm.Barrier()
            seconds = time.perf_counter() - started
            expected = sum(rank + 1 for rank in range(comm.size) if rank != comm.rank)
            with cp.cuda.Device(0):
                error = float(cp.abs(total - expected).max())
            results.append(dict(kind="sum_to_owner", repeat=repeat, ranks=comm.size, transport=transport.kind,
                                bytes_per_pair=int(block.nbytes), seconds=seconds, max_error=error,
                                gb_per_s_received=(comm.size - 1) * block.nbytes / seconds / 1e9))
        if comm.rank == 0:
            for item in results:
                print(json.dumps(item), flush=True)
        return 0
    finally:
        MPI.Finalize()


if __name__ == "__main__":
    raise SystemExit(main())
