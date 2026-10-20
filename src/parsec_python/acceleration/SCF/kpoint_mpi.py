"""The eigenproblems of a k-point job with the basis distributed over MPI ranks.

:class:`~.kpoint_parallel.PooledKPointSolver` puts every k-point (and spin channel) on one device.
A basis that does not fit one device (a two-component basis of thousands of states on millions of
grid points) is instead split by blocks of grid rows over the GPUs of several nodes: this solver.
Every rank holds rows ``[start, stop)`` of the Ritz subspace of every task, applies the Hamiltonian to
its rows (:mod:`..Hamiltonian.distributed_bloch`: halo rows from the neighbours, projector overlaps
summed over the ranks) and takes part in the orthonormalization, the Rayleigh-Ritz projection and
the residuals, whose reductions are ``count x count`` matrices
(:mod:`..Eigensolvers.chebff_spinor`).  The tasks are solved one after the other.

The SCF driver is replicated on every rank (see :mod:`..Parallel.mpi`): all ranks receive the same
eigenvalues, so the Fermi level and the occupations agree; the densities are formed from the local
rows and joined on every rank, so the Hartree and exchange-correlation steps see the whole grid.
Wavefunctions are not gathered (``vectors`` returns an empty block).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from parsec_python.SCF.kpoint_solver import KPointJob, residual_max

from ..Eigensolvers.chebff_spinor import solve_spinor_eigval
from ..Hamiltonian.device_bloch import NumpyArrayBackend
from ..Hamiltonian.distributed_bloch import DistributedBlochConverter
from ..Parallel import mpi
from ..Parallel.mpi import RowPartition


class MPIKPointSolver:
    """Solves a :class:`KPointJob` with the basis of every task distributed over the MPI ranks."""

    def __init__(self, job: KPointJob, *, problem_size: int | None = None, backend: str | None = None, **_ignored) -> None:
        from .kpoint_parallel import resolve_backend

        context = mpi.context()
        if context is None:
            raise RuntimeError("MPIKPointSolver needs PARSEC_MPI=1 (or --mpi) and mpi4py")
        self.context = context
        self.job = job
        plan = resolve_backend(backend)
        if plan is None:
            device_backend = NumpyArrayBackend()  # ranks on the CPU: the device code on the host
        else:
            device_backend = plan.converter(context.local_rank).backend
        self.backend = device_backend
        self.converter = DistributedBlochConverter(device_backend, context)
        self.states: dict[int, Any] = {}
        self.vectors_by_task: dict[int, Any] = {}
        self._stats: dict[str, float] = {}
        self._residual: dict[int, float] = {}
        self._partition: RowPartition | None = None
        self.n_workers = context.size
        self.backend_name = "mpi"

    # ------------------------------------------------------------------ steps
    def _tick(self, name: str, start: float) -> None:
        self.backend.synchronize()
        key = f"worker_{name}_seconds"
        self._stats[key] = self._stats.get(key, 0.0) + time.perf_counter() - start

    def solve(self, potentials: dict, filter_degree: int) -> list[np.ndarray]:
        """One eigenvalue array per task; the same on every rank."""
        # the drivers are replicated: stop with a message, not a deadlock, if they ever diverge
        checksum = float(sum(float(np.sum(np.asarray(v, dtype=float))) for v in potentials.values()))
        self.context.check_equal(checksum, "the effective potentials", 1.0e-9)
        settings = self.job.eigval_settings(filter_degree)
        backend = self.backend
        backend.use()
        eigenvalues = []
        for task in range(self.job.n_tasks):
            start = time.perf_counter()
            hamiltonian = self.job.make_hamiltonian(task, potentials)
            self._tick("hamiltonian", start)  # the Bloch projectors of a k-point on first use
            start = time.perf_counter()
            operator = self.converter.operator(hamiltonian)
            self._tick("upload", start)
            if self._partition is None:
                self._partition = RowPartition(operator.global_grid_size, self.context.size)
                self._check_memory(operator)
            state = self.states.pop(task, None)
            self.vectors_by_task.pop(task, None)  # release the previous basis before the next one is made
            start = time.perf_counter()
            result = solve_spinor_eigval(
                operator,
                self.job.number_of_states,
                settings=settings,
                state=state,
                xp=backend.xp,
                sync=backend.synchronize,
                converge_first=settings.initial_method == "chebdav",
            )
            self._tick("solve", start)
            for name, value in result.stats.items():
                self._stats[name] = self._stats.get(name, 0.0) + value
            self._residual[task] = residual_max(result.residual_norms)
            self.states[task] = result.state
            self.vectors_by_task[task] = result.state.vectors[:, : self.job.number_of_states]
            eigenvalues.append(np.asarray(result.eigenvalues, dtype=float))
        return eigenvalues

    def _check_memory(self, operator) -> None:
        """Stop with the number of ranks that is needed, not with an out-of-memory deep in a solve."""
        working = self.job.number_of_states + 6
        basis = operator.local_dimension * working * 16  # bytes of the local rows of one subspace
        # filtering holds the old and the new subspace; the Gram matrices and chunks are small
        per_task = 2 * basis + 8 * working * working * 16
        # tasks keep their subspace between the solves; one more is in flight
        needed = per_task + (self.job.n_tasks - 1) * basis
        if self.backend.name != "cupy":
            return
        backend = self.backend
        backend.use()
        free, total = backend.cp.cuda.runtime.memGetInfo()
        usable = 0.92 * total
        if needed > usable and self.context.is_root:
            total_needed = needed * self.context.size
            ranks = int(np.ceil(total_needed / usable))
            raise MemoryError(
                f"the distributed subspaces need about {needed / 1e9:.0f} GB per GPU on {self.context.size} "
                f"ranks ({basis / 1e9:.0f} GB for the local rows of one subspace), and a GPU has "
                f"{total / 1e9:.0f} GB: use at least {ranks} ranks"
            )
        if needed > usable:  # the other ranks fail with the same numbers: do not wait on a dead root
            raise MemoryError("the distributed subspaces do not fit the GPUs; see rank 0")

    def accumulate(self, occupations: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Sum of the tasks' density contributions over the whole grid, the same on every rank."""
        backend = self.backend
        backend.use()
        total: dict[str, np.ndarray] = {}
        start = time.perf_counter()
        for task in range(self.job.n_tasks):
            parts = self.job.accumulate(
                task, self.vectors_by_task[task], occupations[task], xp=backend.xp
            )
            for name, value in parts.items():
                host = np.array(backend.asnumpy(value), copy=True)
                total[name] = total[name] + host if name in total else host
        self._tick("density", start)
        start = time.perf_counter()
        joined = {
            name: self.context.allgather_last_axis(value, self._partition) for name, value in total.items()
        }
        self._tick("gather", start)
        return joined

    def extras(self) -> list[Any]:
        if self.job.extras is None:
            return [None] * self.job.n_tasks
        results = []
        for task in range(self.job.n_tasks):
            partial = np.ascontiguousarray(
                self.job.extras(task, self.vectors_by_task[task], xp=self.backend.xp), dtype=np.float64
            )
            results.append(self.context.allsum_host(partial))  # the extras are sums over the rows
        return results

    def vectors(self, task: int) -> np.ndarray:
        """Wavefunctions are not gathered: an empty block of the right number of rows."""
        return np.empty((self.job_dimension(), 0), dtype=np.complex128)

    def job_dimension(self) -> int:
        state = next(iter(self.states.values()), None)
        return 0 if state is None else int(state.operator_dimension)

    def stats(self) -> dict[str, float]:
        """Eigensolver counters and seconds of this rank (the same work on every rank)."""
        return dict(self._stats)

    def residuals(self) -> list[float]:
        return [self._residual.get(task, float("nan")) for task in range(self.job.n_tasks)]

    def close(self) -> None:
        self.states.clear()
        self.vectors_by_task.clear()

    def __enter__(self) -> "MPIKPointSolver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


__all__ = ["MPIKPointSolver"]
