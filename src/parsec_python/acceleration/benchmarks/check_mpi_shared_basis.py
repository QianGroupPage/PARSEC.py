"""MPI check of a sector basis shared by ranks (:mod:`..Eigensolvers.mpi_shared_basis`).

Run with one rank per GPU, CUDA_VISIBLE_DEVICES set per rank, e.g. on one node::

    srun -n 4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none bash -c \\
        'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID exec python -m \\
         parsec_python.acceleration.benchmarks.check_mpi_shared_basis'

Every rank holds its columns of one host matrix known to all.  Checked, for
the device and the host-staged transport and for one and several exchange
rounds: the row blocks equal the rows of the matrix and come back to the
same columns bit for bit; the summed Gram matrices of the row blocks equal
the product of the whole matrix; the trial basis of each rank equals its
columns of the single-device DLARNV basis bit for bit, and the generator
ends where the single-device one ends.  Prints one line per check on the
root and exits with 1 if any failed on any rank.
"""

from __future__ import annotations

import os
import sys

import numpy as np


class _Device:
    def __init__(self, cp):
        self.effective_potential = cp.zeros(1)


def _local_columns(layout, rank, matrix):
    parts = [matrix[:, column : column + count] for column, _offset, count in layout.spans(rank)]
    return np.concatenate(parts, axis=1) if parts else matrix[:, :0]


def main() -> int:
    from mpi4py import MPI
    import cupy as cp

    from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
    from parsec_python.acceleration.Eigensolvers.lapack_random import LapackRandom
    from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import MPIShardGroup, row_edges

    comm = MPI.COMM_WORLD
    rank = comm.rank
    cp.cuda.Device(0).use()
    failures = []

    def check(name, ok):
        bad = comm.allreduce(0 if ok else 1)
        if rank == 0:
            print(f"{'PASS' if not bad else 'FAIL'} {name}" + (f" ({bad} ranks)" if bad else ""), flush=True)
        if bad:
            failures.append(name)

    device = _Device(cp)
    rows, states = 10007, 203
    blocks = uniform_filter_blocks(states, 6, 10)
    matrix = np.random.default_rng(7).standard_normal((rows, states))
    for transport in ("cuda", "host"):
        if transport == "cuda" and os.environ.get("MPICH_GPU_SUPPORT_ENABLED") != "1":
            if rank == 0:
                print("SKIP cuda transport: MPICH_GPU_SUPPORT_ENABLED is not 1", flush=True)
            continue
        for per_round in (None, 7):
            if per_round is None:
                os.environ.pop("PARSEC_MPI_SHARED_EXCHANGE_BYTES", None)
            else:
                os.environ["PARSEC_MPI_SHARED_EXCHANGE_BYTES"] = str(8 * rows * per_round)
            label = f"{transport}, {'one round' if per_round is None else f'{per_round} columns per round'}"
            group = MPIShardGroup(comm, device, transport=transport)
            layout = group.column_layout(blocks)
            local = cp.asfortranarray(cp.asarray(_local_columns(layout, rank, matrix)))
            row_block = group.to_rows(local, layout, rows)
            edges = row_edges(rows, comm.size)
            expected_rows = matrix[edges[rank] : edges[rank + 1], :]
            check(f"rows of the basis [{label}]", np.array_equal(cp.asnumpy(row_block), expected_rows))
            back = group.to_columns(row_block, layout, rows)
            check(f"columns back [{label}]", np.array_equal(cp.asnumpy(back), cp.asnumpy(local)))
            gram = group._sum(cp.ascontiguousarray(row_block.T @ row_block)[None])[0]
            reference = matrix.T @ matrix
            error = float(np.max(np.abs(cp.asnumpy(gram) - reference)) / np.max(np.abs(reference)))
            check(f"summed Gram matrix [{label}], relative error {error:.1e}", error < 1e-12)
    os.environ.pop("PARSEC_MPI_SHARED_EXCHANGE_BYTES", None)
    for on_device in ("1", "0"):
        os.environ["PARSEC_CUPY_DEVICE_RANDOM"] = on_device
        group = MPIShardGroup(comm, device, transport="host")
        layout = group.column_layout(blocks)
        generator = LapackRandom()
        basis = group.random_basis(generator, rows, layout)
        single = LapackRandom()
        whole = single.uniform_minus_1_1((rows, states), column_major=True)
        check(
            f"trial basis = single-device columns (device random {on_device})",
            np.array_equal(cp.asnumpy(basis.local), _local_columns(layout, rank, whole))
            and tuple(generator.seed) == tuple(single.seed),
        )
        leading = basis[:, :100]
        check(
            f"leading columns of the trial basis (device random {on_device})",
            np.array_equal(cp.asnumpy(leading.local), _local_columns(leading.layout, rank, whole[:, :100])),
        )
    if rank == 0:
        print("ALL PASSED" if not failures else f"FAILED: {', '.join(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
