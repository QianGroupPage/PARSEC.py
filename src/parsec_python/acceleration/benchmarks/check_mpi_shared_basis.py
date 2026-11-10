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
ends where the single-device one ends; the Rayleigh--Ritz step in rounds
(``PARSEC_MPI_SHARED_RITZ=rounds``, one and several rounds) gives the Ritz
values of the two row blocks to round-off and their Ritz vectors up to signs;
the condition number of the small solve estimated on the next rank gives the
Ritz pairs of the root's own estimate bit for bit.  Prints one line per check
on the root and exits with 1 if any failed on any rank.
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
    # The Ritz step in rounds (PARSEC_MPI_SHARED_RITZ=rounds) against the two row blocks: the same Ritz values to
    # round-off, Ritz vectors of a simple spectrum equal up to their signs, every column back on its rank.
    import cupyx.scipy.sparse as csp
    import scipy.sparse as sp

    generator = np.random.default_rng(17)
    off = generator.uniform(-1.0, 1.0, rows - 1)
    hamiltonian = sp.diags((off, generator.uniform(-2.0, 6.0, rows), off), (-1, 0, 1), format="csr")

    class _Operator(_Device):
        def __init__(self):
            super().__init__(cp)
            self.hamiltonian = csp.csr_matrix(hamiltonian)

        def apply_into(self, block, out):
            out[...] = self.hamiltonian @ block

    operator = _Operator()

    def ritz_pairs(route, transport, per_round):
        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import SharedBasis

        os.environ["PARSEC_MPI_SHARED_RITZ"] = route
        if per_round is None:
            os.environ.pop("PARSEC_MPI_SHARED_EXCHANGE_BYTES", None)
        else:
            os.environ["PARSEC_MPI_SHARED_EXCHANGE_BYTES"] = str(8 * rows * per_round)
        group = MPIShardGroup(comm, operator, transport=transport)
        layout = group.column_layout(blocks)
        local = cp.asfortranarray(cp.asarray(_local_columns(layout, rank, matrix)))
        values, rotated, _whitened = group.ritz(operator, SharedBasis(group, local, layout), stages=False)
        whole = np.zeros((rows, states))
        for column, offset, count in layout.spans(rank):
            whole[:, column : column + count] = cp.asnumpy(rotated.local[:, offset : offset + count])
        comm.Allreduce(MPI.IN_PLACE, whole)
        return np.asarray(values), whole

    for transport in ("cuda", "host"):
        if transport == "cuda" and os.environ.get("MPICH_GPU_SUPPORT_ENABLED") != "1":
            continue
        reference_values, reference_basis = ritz_pairs("rows", transport, None)
        # The condition number on the next rank (the default) or on the root: the same bits.
        os.environ["PARSEC_CUPY_RITZ_CONDITION_HELPER"] = "off"
        alone_values, alone_basis = ritz_pairs("rows", transport, None)
        os.environ.pop("PARSEC_CUPY_RITZ_CONDITION_HELPER", None)
        check(f"condition number beside the root changes no bit [{transport}]",
              np.array_equal(alone_values, reference_values) and np.array_equal(alone_basis, reference_basis))
        for per_round in (None, 7):
            label = f"{transport}, {'one round' if per_round is None else f'{per_round} columns per round'}"
            values, basis = ritz_pairs("rounds", transport, per_round)
            scale = float(np.max(np.abs(reference_values)))
            value_error = float(np.max(np.abs(values - reference_values))) / scale
            signs = np.sign(np.sum(basis * reference_basis, axis=0))
            vector_error = float(np.max(np.abs(basis * signs - reference_basis)))
            check(f"Ritz values in rounds = two row blocks [{label}], relative {value_error:.1e}", value_error < 1e-12)
            check(f"Ritz vectors in rounds = two row blocks [{label}], {vector_error:.1e}", vector_error < 1e-9)
    os.environ.pop("PARSEC_MPI_SHARED_RITZ", None)
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
