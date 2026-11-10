"""The eigenproblems of a k-point job on MPI ranks: pools of ranks, the basis of a k-point shared by columns.

The SCF driver of a k-point run (:func:`parsec_python.SCF.kpoints.run_scf_kpoints`, the collinear spin
driver) is replicated on every rank (:mod:`..Parallel.mpi`); only the eigensolves and the sums of the
density are collective.  The ranks of ``COMM_WORLD`` form ``P`` *pools* (contiguous blocks of ranks,
of sizes that differ by at most one); the tasks (k-points, spin channels) are dealt to the pools in
turn, task ``t`` to pool ``t mod P``, and every pool solves its tasks one after another, all pools at
the same time.  A pool shares the basis of each task by columns
(:func:`..Eigensolvers.chebff_columns.solve_column_eigval`): every rank holds all grid rows of its
columns and a replica of the device Bloch operator (the local term in the kernel of
:mod:`..Hamiltonian.bloch_stencil` on an orthorhombic grid), filters without communication, and the
orthonormalization and Rayleigh-Ritz steps move the basis to blocks of rows and back.  A pool of one
rank runs the same solver without moves: the policy and numbers of the one-node solver
(:class:`.kpoint_parallel.PooledKPointSolver`) to round-off, with the Cholesky QR in place of its
Householder QR and the residuals from the rotated ``H X`` (diamond n = 4, 8 k-points on 4 A100, 8 SCF
iterations: 109 s against 134 s, 22.3 against 27.9 GB per GPU).

Every rank gets the eigenvalues of every task (gathered from the pool roots), so the Fermi level and the
occupations agree on all ranks; the density (and every other part of ``accumulate``, all sums over the
states) is formed by each rank from its columns and summed over ``COMM_WORLD`` (``Allreduce``).  The
orbitals are not gathered (``vectors`` is an empty block), which the drivers that need them refuse.

The number of pools.  ``PARSEC_KPOINT_POOLS`` (``--kpoint-pools``) sets it; by default it is the one
that balances the work best, the largest of the equally balanced counts (fewer ranks per pool, less
exchange), among the counts whose memory per GPU (:func:`pool_memory_bytes`) fits 90 % of the device:
``max_j tasks_j / ranks_j`` is the time of the slowest pool if the shared basis scaled perfectly.  The
pools are formed at the first solve (the eigensolver settings give the working subspace).

A pool with several tasks keeps the Ritz subspaces of some of the tasks it is not solving on the host
when they would not fit the device together (``PARSEC_KPOINT_STATE_STORAGE``: ``auto``, the default, as
the one-node solver; ``device``; ``host``): the first ``resident - 1`` tasks of the pool stay on the device
(as many as fit, :attr:`PoolPlan.resident`), the others move their share of the subspace to pageable host
memory between their solves and back for the solve and for the density, three copies of ``N x m / p``
complex numbers per task and SCF iteration over PCIe.  The host then holds the idle ones (about
``(tasks - resident) N m / p`` complex numbers per rank, times the ranks of a node).  A plan that needs no
such storage is preferred to one of the same balance that does.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from parsec_python.SCF.kpoint_solver import (
    KPointJob,
    device_solver_supports,
    job_eigval_settings,
    residual_max,
)

from ..Eigensolvers.chebff_columns import ColumnGroup, ColumnLayout, requested_spans, solve_column_eigval
from ..Eigensolvers.chebff_spinor import _filter_columns
from ..Eigensolvers.mpi_shared_basis import exchange_bytes
from ..Parallel import mpi

POOLS_ENV = "PARSEC_KPOINT_POOLS"
# The share of a device the estimate may take.  Measured on A100 (diamond, 2026-11-08): n = 4 to 6 used 83 to
# 100 % of the estimate (CuPy's pool, nvidia-smi); n = 6 in four pools of 16 A100, estimated at 69 GB, ran and
# was 15 % faster than two pools of eight ranks (the choice of 85 %).
_USABLE_FRACTION = 0.9
# Device bytes per grid point that do not depend on the number of states, for stencils up to order 12
# (37 entries of 12 bytes per row): the Laplacian of the Hartree conjugate gradients and about thirty
# fields of the SCF step (potentials, Hartree vectors, XC fields); per task the complex Bloch projectors
# with their adjoint (eight entries of 20 bytes per point each, a generous allowance: carbon at 0.37 bohr
# has fewer than two).  Where the local term is not the kernel of .bloch_stencil (``fused=False``) the
# operator holds the Laplacian too, and every task its pre-summed k.grad (36 entries).
_FIXED_BYTES_PER_POINT = 37 * 12 + 30 * 8
_TASK_BYTES_PER_POINT = 2 * 8 * 20
_SPARSE_FIXED_BYTES_PER_POINT = 37 * 12
_SPARSE_TASK_BYTES_PER_POINT = 36 * 12
_CONTEXT_BYTES = 3 << 29  # CUDA context, cuBLAS/cuSOLVER work spaces


def requested_pools() -> int | None:
    """``PARSEC_KPOINT_POOLS`` as a positive number, ``None`` (the default rule) when unset or ``auto``."""
    text = os.environ.get(POOLS_ENV, "").strip().lower()
    if text in ("", "auto"):
        return None
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"{POOLS_ENV} (--kpoint-pools) must be a positive number or auto, not {text!r}")
    return int(text)


def storage_mode() -> str:
    """``PARSEC_KPOINT_STATE_STORAGE`` (``auto``, ``device`` or ``host``), as the one-node solver reads it."""
    from .kpoint_parallel import _storage_mode

    return _storage_mode()


def pool_memory_bytes(
    dimension: int,
    working: int,
    tasks: int,
    ranks: int,
    block_size: int = 1,
    *,
    resident: int | None = None,
    fused: bool = True,
) -> int:
    """Device bytes one rank of a pool of ``ranks`` ranks needs for ``tasks`` tasks.

    The Ritz subspaces of the pool's tasks (this rank's columns of each, ``dimension x width``
    complex; ``resident`` of them on the device, all by default, one when the others stay on the
    host), the working copies of the task in hand (the rows, the columns and the rows of ``H X`` of a
    Rayleigh-Ritz step: three of its share in all; on one rank, where rows and columns are the same
    arrays, two), the exchange buffers of a round, the dense
    ``m x m`` matrices of the Rayleigh-Ritz step, the blocks of columns of the filter recurrence and of
    the products of the operator (eight blocks of ``PARSEC_SPINOR_FILTER_COLUMNS`` columns, 64 by default)
    and the static data (:data:`_FIXED_BYTES_PER_POINT`, :data:`_TASK_BYTES_PER_POINT`).
    """
    dimension, working, tasks, ranks = int(dimension), int(working), int(tasks), int(ranks)
    width = max(ColumnLayout(working, block_size, ranks).widths) if ranks > 1 else working
    share = dimension * width * 16
    resident = tasks if resident is None else max(1, min(int(resident), tasks))
    subspaces = (resident - 1) * share + (3 if ranks > 1 else 2) * share
    buffers = 2 * exchange_bytes() if ranks > 1 else 0
    dense = 6 * working * working * 16
    transient = 8 * dimension * min(_filter_columns(), working) * 16
    fixed = _FIXED_BYTES_PER_POINT + (0 if fused else _SPARSE_FIXED_BYTES_PER_POINT)
    per_task = _TASK_BYTES_PER_POINT + (0 if fused else _SPARSE_TASK_BYTES_PER_POINT)
    static = dimension * (fixed + tasks * per_task)
    return int(subspaces + buffers + dense + transient + static + _CONTEXT_BYTES)


@dataclass(frozen=True)
class PoolPlan:
    """``pools`` pools: pool ``j`` holds the ranks ``starts[j] : starts[j] + sizes[j]`` and the tasks ``t``
    with ``t % pools == j``; ``bytes_per_rank`` is the estimate of the fullest pool."""

    pools: int
    sizes: tuple[int, ...]
    tasks: int
    bytes_per_rank: int
    balance: float
    host_storage: bool = False
    resident: int = 0  # subspaces a rank of the fullest pool keeps on the device (0: all of its tasks)

    @property
    def starts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.concatenate(([0], np.cumsum(self.sizes[:-1]))))

    def pool_of_rank(self, rank: int) -> int:
        return int(np.searchsorted(np.cumsum(self.sizes), rank, side="right"))

    def tasks_of_pool(self, pool: int) -> list[int]:
        return list(range(pool, self.tasks, self.pools))


def plan_pools(
    n_tasks: int,
    n_ranks: int,
    bytes_of,
    usable: float,
    requested: int | None = None,
    storage: str = "auto",
    limit: float | None = None,
) -> PoolPlan:
    """The pools of ``n_tasks`` tasks on ``n_ranks`` ranks (see the module text).

    ``bytes_of(tasks, ranks, resident)`` is the device memory one rank of a pool needs with ``resident``
    subspaces on the device, ``usable`` the bytes a device offers; ``requested`` a number of pools to take
    (``--kpoint-pools``); ``storage`` the ``PARSEC_KPOINT_STATE_STORAGE`` mode (``auto``: the host for
    the subspaces not in hand only where the device cannot hold them).  A requested number of pools is
    refused only beyond ``limit`` (the whole device, default ``usable``): the estimate exceeded the peaks
    measured on A100 by up to 17 % (diamond n = 4 to 6), so a pool count the default rule leaves out for
    its 90 % may still run when asked for.
    """
    n_tasks, n_ranks = int(n_tasks), int(n_ranks)
    plans = []
    for count in range(1, min(n_tasks, n_ranks) + 1):
        sizes = tuple(n_ranks // count + (1 if j < n_ranks % count else 0) for j in range(count))
        loads = tuple(n_tasks // count + (1 if j < n_tasks % count else 0) for j in range(count))
        balance = max(load / size for load, size in zip(loads, sizes))
        on_device = max(int(bytes_of(load, size, load)) for load, size in zip(loads, sizes))
        resident = 1
        if storage == "auto":
            # as many subspaces on the device as fit, the fewest on the host
            while resident + 1 < loads[0] and max(
                int(bytes_of(load, size, resident + 1)) for load, size in zip(loads, sizes)
            ) <= usable:
                resident += 1
        on_host = max(int(bytes_of(load, size, resident)) for load, size in zip(loads, sizes))
        host = storage == "host" or (storage == "auto" and on_device > usable and on_host < on_device)
        plans.append(
            PoolPlan(count, sizes, n_tasks, on_host if host else on_device, balance, host, resident if host else 0)
        )
    if requested is not None:
        if requested > min(n_tasks, n_ranks):
            raise ValueError(
                f"--kpoint-pools {requested}: at most {min(n_tasks, n_ranks)} pools for {n_tasks} tasks "
                f"(k-points and spin channels) on {n_ranks} ranks"
            )
        plan = plans[requested - 1]
        ceiling = usable if limit is None else max(float(limit), usable)
        if plan.bytes_per_rank > ceiling:
            raise MemoryError(
                f"--kpoint-pools {requested}: about {plan.bytes_per_rank / 1e9:.1f} GB per GPU are needed and "
                f"the GPU has {ceiling / 1e9:.1f} GB; fewer pools share each basis over more ranks"
            )
        return plan
    fitting = [plan for plan in plans if plan.bytes_per_rank <= usable]
    if not fitting:
        fewest = plans[0]
        total = fewest.bytes_per_rank * n_ranks
        raise MemoryError(
            f"the k-point subspaces need about {fewest.bytes_per_rank / 1e9:.1f} GB per GPU on {n_ranks} ranks "
            f"in one pool, and {usable / 1e9:.1f} GB are usable: about {int(np.ceil(total / usable))} ranks "
            "(GPUs) are needed"
        )
    return min(fitting, key=lambda plan: (round(plan.balance, 12), plan.host_storage, -plan.pools))


def host_memory_bytes() -> tuple[float, float]:
    """``(resident, peak resident)`` host memory of this process in bytes (``VmRSS``, ``VmHWM`` of
    ``/proc/self/status``; zeros where there is none)."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            values = {}
            for line in status:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    name, amount, *_ = line.split()
                    values[name] = float(amount) * 1024.0
        return values.get("VmRSS:", 0.0), values.get("VmHWM:", 0.0)
    except OSError:
        return 0.0, 0.0


def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    return np.asarray(get() if get is not None else value)


class MPIPoolKPointSolver:
    """Solves a :class:`KPointJob` on pools of MPI ranks (see the module text).

    ``context`` is the MPI context (:func:`..Parallel.mpi.context` by default; the tests give a context
    of threads), ``pools`` a number of pools in place of ``PARSEC_KPOINT_POOLS``.
    """

    backend_name = "mpi-pools"

    def __init__(
        self,
        job: KPointJob,
        *,
        problem_size: int | None = None,
        backend: str | None = None,
        context=None,
        pools: int | None = None,
        **_ignored,
    ) -> None:
        from ..Hamiltonian.device_bloch import DeviceBlochConverter, NumpyArrayBackend
        from .kpoint_parallel import resolve_backend

        context = mpi.context() if context is None else context
        if context is None:
            raise RuntimeError("MPIPoolKPointSolver needs PARSEC_MPI=1 (or --mpi) and mpi4py")
        if job.extras is not None:
            raise NotImplementedError(
                "the k-point pools gather no per-state extras (forces, spin moments of spinors); "
                "their jobs run on the row-distributed MPI solver"
            )
        if problem_size is None:
            raise ValueError("the k-point pools need the grid size of the job (problem_size)")
        self.context = context
        self.job = job
        self.problem_size = int(problem_size)
        plan = resolve_backend(backend)
        if plan is None:
            self.converter = DeviceBlochConverter(NumpyArrayBackend())  # ranks on the CPU: the device code on the host
        else:
            self.converter = plan.converter(context.local_rank)
            # the local term in one kernel where the grid is an orthorhombic periodic one (.bloch_stencil)
            self.converter.fused_stencil = True
        self.backend = self.converter.backend
        self.xp = self.backend.xp
        self.n_workers = context.size
        self._requested_pools = requested_pools() if pools is None else int(pools)
        self.plan: PoolPlan | None = None
        self.pool = -1
        self.group: ColumnGroup | None = None
        self.tasks: list[int] = []
        self.states: dict[int, Any] = {}
        self._residual: dict[int, float] = {}
        self._stats: dict[str, float] = {}
        self._pool_comm = None
        self.static_bytes = 0

    # ------------------------------------------------------------------ set-up
    def _tick(self, name: str, start: float) -> None:
        self.backend.synchronize()
        key = f"worker_{name}_seconds"
        self._stats[key] = self._stats.get(key, 0.0) + time.perf_counter() - start

    def _usable_bytes(self) -> float:
        if self.backend.name != "cupy":
            return float("inf")
        self.backend.use()
        _, total = self.backend.cp.cuda.runtime.memGetInfo()
        return _USABLE_FRACTION * float(total)

    def _set_up(self, settings) -> None:
        """Form the pools (collective: every rank calls it at its first solve)."""
        context = self.context
        working = min(self.job.number_of_states + int(settings.safety_buffer), self.problem_size)
        block = int(settings.subspace.block_size)
        plan = None
        failure = None
        if context.is_root:
            try:
                usable = self._usable_bytes()
                plan = plan_pools(
                    self.job.n_tasks,
                    context.size,
                    lambda tasks, ranks, resident: pool_memory_bytes(
                        self.problem_size, working, tasks, ranks, block, resident=resident
                    ),
                    usable,
                    self._requested_pools,
                    storage_mode() if self.xp is not np else "device",
                    limit=usable / _USABLE_FRACTION,
                )
            except (MemoryError, ValueError) as error:
                failure = (type(error).__name__, str(error))
        plan, failure = context.bcast((plan, failure), root=0)
        if failure is not None:
            kind, message = failure
            raise (MemoryError if kind == "MemoryError" else ValueError)(message)
        self.plan = plan
        self.host_at_setup_bytes = host_memory_bytes()[0]  # after the preparation of the system
        self.pool = plan.pool_of_rank(context.rank)
        self.tasks = plan.tasks_of_pool(self.pool)
        comm = context.comm.Split(self.pool, context.rank)
        self._pool_comm = comm
        sync = self.backend.synchronize if self.xp is not np else None
        self.group = ColumnGroup(comm, self.xp, sync=sync, mpi=getattr(context, "MPI", None))

    # ------------------------------------------------------------------- steps
    def solve(self, potentials: dict, filter_degree: int, tolerance: float | None = None) -> list[np.ndarray]:
        """One eigenvalue array per task, the same on every rank.

        ``tolerance`` replaces ``Diag_Tolerance`` for this call (``Dynamic_Diag_Tol``).
        """
        settings = job_eigval_settings(self.job, filter_degree, tolerance)
        if not device_solver_supports(settings):
            raise NotImplementedError(
                f"Eigensolver: {settings.initial_method} is not implemented with the MPI k-point pools "
                "(chebff or chebdav), run without --mpi"
            )
        # the drivers are replicated: stop with a message, not a deadlock, if they ever diverge
        checksum = float(sum(float(np.sum(np.asarray(v, dtype=float))) for v in potentials.values()))
        self.context.check_equal(checksum, "the effective potentials", 1.0e-9)
        if self.plan is None:
            self._set_up(settings)
        backend = self.backend
        backend.use()
        found: dict[int, np.ndarray] = {}
        residuals: dict[int, float] = {}
        requested = self.job.number_of_states
        order = self._solve_order()
        for task in order:
            start = time.perf_counter()
            hamiltonian = self.job.make_hamiltonian(task, potentials)
            self._tick("hamiltonian", start)  # the Bloch projectors of a k-point on first use
            start = time.perf_counter()
            operator = self.converter.operator(hamiltonian)
            self._tick("upload", start)
            if not self.static_bytes:
                self.static_bytes = _static_device_bytes(operator)
            state = self.states.pop(task, None)
            self._resident(state)
            start = time.perf_counter()
            result = solve_column_eigval(
                operator,
                requested,
                settings=settings,
                group=self.group,
                state=state,
                xp=self.xp,
                sync=backend.synchronize,
                converge_first=settings.initial_method == "chebdav",
            )
            del state
            self._tick("solve", start)
            for name, value in result.stats.items():
                self._stats[name] = self._stats.get(name, 0.0) + value
            self.states[task] = result.state
            if self.group.is_root:
                found[task] = np.asarray(result.eigenvalues, dtype=np.float64)
                residuals[task] = residual_max(result.residual_norms)
            del result
            self._note_reserved()
            if task != order[-1]:
                self._spill(task, self.states[task])  # the last one stays for the density and the next solve
        self._kept = order[-1]
        start = time.perf_counter()
        gathered = self.context.comm.allgather((found, residuals))
        self._tick("pool_wait", start)  # the eigenvalues of the other pools: waiting for the slowest
        eigenvalues: dict[int, np.ndarray] = {}
        for values, norms in gathered:
            eigenvalues.update(values)
            self._residual.update(norms)
        missing = [task for task in range(self.job.n_tasks) if task not in eigenvalues]
        if missing:
            raise RuntimeError(f"no pool returned the eigenvalues of tasks {missing}")
        return [eigenvalues[task] for task in range(self.job.n_tasks)]

    def accumulate(self, occupations: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Sum of the tasks' density contributions over the whole grid, the same on every rank."""
        backend = self.backend
        backend.use()
        start = time.perf_counter()
        total: dict[str, np.ndarray] = {}

        def add(parts):
            for name, value in parts.items():
                host = np.array(_host(value), copy=True)
                total[name] = total[name] + host if name in total else host

        requested = self.job.number_of_states
        for task in self.tasks:
            state = self.states[task]
            vectors = self._device_vectors(state)
            values = np.asarray(occupations[task], dtype=np.float64)
            count = min(requested, values.size)
            if getattr(state, "layout", None) is None:
                add(self.job.accumulate(task, vectors[:, :count], values[:count], xp=self.xp))
            else:
                for first, local, width in requested_spans(state.layout, self.group.rank, count):
                    add(
                        self.job.accumulate(
                            task, vectors[:, local : local + width], values[first : first + width], xp=self.xp
                        )
                    )
            del vectors
            if task != getattr(self, "_kept", None):
                self._spill(task, state)
        self._tick("density", start)
        start = time.perf_counter()
        # every rank sums the same parts, those its tasks did not give as zeros
        shapes: dict[str, tuple[tuple, np.dtype]] = {}
        for part in self.context.comm.allgather({name: (value.shape, value.dtype.str) for name, value in total.items()}):
            for name, (shape, dtype) in part.items():
                known = shapes.get(name)
                if known is not None and known[0] != tuple(shape):
                    raise RuntimeError(f"the ranks disagree about the shape of the density part {name!r}")
                kind = np.dtype(dtype) if known is None else np.result_type(known[1], np.dtype(dtype))
                shapes[name] = (tuple(shape), kind)
        joined: dict[str, np.ndarray] = {}
        for name in sorted(shapes):
            shape, dtype = shapes[name]
            value = total.get(name)
            flat = np.zeros(int(np.prod(shape, dtype=np.int64)), dtype=dtype) if value is None else (
                np.array(value, dtype=dtype, copy=True).reshape(-1)
            )
            self.context.allsum_host(flat)
            joined[name] = flat.reshape(shape)
        self._tick("gather", start)
        return joined

    def _note_reserved(self) -> None:
        """The most device memory CuPy's pool has held so far (its blocks are kept until released)."""
        self._note_host()
        if self.xp is np:
            return
        reserved = float(self.xp.get_default_memory_pool().total_bytes())
        self.reserved_bytes = max(getattr(self, "reserved_bytes", 0.0), reserved)

    def _note_host(self) -> None:
        """The resident host memory of this process now and at its peak (Linux ``/proc``; 0 elsewhere)."""
        now, peak = host_memory_bytes()
        self.host_bytes = max(getattr(self, "host_bytes", 0.0), now)
        self.host_peak_bytes = max(getattr(self, "host_peak_bytes", 0.0), peak)

    # ---------------------------------------------------------- host storage
    @property
    def host_storage(self) -> bool:
        """Whether the subspaces of the tasks not in hand stay on the host (:attr:`PoolPlan.host_storage`)."""
        return bool(self.plan is not None and self.plan.host_storage and len(self.tasks) > 1 and self.xp is not np)

    def _solve_order(self) -> list[int]:
        """The tasks of this rank in the order of the next solve.

        With host storage the tasks that move start with the one the last solve left on the device (it
        stays there through the density, so that the host never holds more than ``tasks - resident``
        subspaces); otherwise the order of the pool.
        """
        if not self.host_storage:
            return list(self.tasks)
        fixed = [task for task in self.tasks if self._stays_on_device(task)]
        moving = [task for task in self.tasks if not self._stays_on_device(task)]
        kept = getattr(self, "_kept", None)
        if kept in moving:
            first = moving.index(kept)
            moving = moving[first:] + moving[:first]
        return fixed + moving

    def _stays_on_device(self, task: int) -> bool:
        """The first ``resident - 1`` tasks of the pool keep their subspaces on the device."""
        return self.tasks.index(task) < max(int(self.plan.resident) - 1, 0)

    def _spill(self, task: int, state) -> None:
        """Move the subspace of ``state`` to pageable host memory where the plan keeps it there.

        Page-locked copies of every idle subspace were measured (diamond n = 7 on 16 A100): with four
        ranks per node they took the node's memory; the pageable ones are freed as they come back.
        """
        if not self.host_storage or state is None or isinstance(state.vectors, np.ndarray):
            return
        if self._stays_on_device(task):
            return
        start = time.perf_counter()
        state.vectors = self.backend.asnumpy(state.vectors)
        self._tick("spill", start)
        self._note_host()

    def _resident(self, state) -> None:
        """Bring the subspace of ``state`` back to the device before its solve."""
        if state is None or not isinstance(state.vectors, np.ndarray) or self.xp is np:
            return
        start = time.perf_counter()
        state.vectors = self.backend.asarray(state.vectors)
        self._tick("restore", start)

    def _device_vectors(self, state):
        """The subspace of ``state`` on the device (a copy of a host one, released after use)."""
        if isinstance(state.vectors, np.ndarray) and self.xp is not np:
            start = time.perf_counter()
            vectors = self.backend.asarray(state.vectors)
            self._tick("restore", start)
            return vectors
        return state.vectors

    def density_symmetrizer(self, density, symmetry, volume_element: float):
        """``Symmetrize_Charge_Density`` of a field on the device of this rank: the space-group average of
        :class:`.periodic_symmetry.PeriodicDensitySymmetrizer`, the host's bit for bit, and its report.

        Every rank averages the same field (the replicated drivers), on its own device.
        """
        if self.xp is np:
            from parsec_python.Grid.point_group import symmetrize_density

            return symmetrize_density(density, symmetry, volume_element)
        from .periodic_symmetry import PeriodicDensitySymmetrizer

        cached = getattr(self, "_symmetrizer", None)
        if cached is None or cached[0] is not symmetry:
            self.backend.use()
            cached = (symmetry, PeriodicDensitySymmetrizer(symmetry, xp=self.xp))
            self._symmetrizer = cached
        start = time.perf_counter()
        self.backend.use()
        symmetric, report = cached[1].symmetrize_density(self.xp.asarray(density), volume_element)
        result = np.asarray(_host(symmetric), dtype=np.float64)
        self._tick("symmetrize", start)
        return result, report

    def projector_builder(self, grid, atoms, potentials, specifications, lattice_vectors=None, **keywords):
        """:func:`parsec_python.V_ion.build_nonlocal_projectors` with the atoms dealt over the ranks of the pool.

        Every rank of a pool holds the operator of the pool's k-points, so every rank needs the Bloch
        projectors of each: each builds those of its contiguous share of the atoms (the columns of an atom do
        not depend on the others) and the pool gathers them in atom order, the matrix of one whole build
        (diamond n = 7 at P = 1: 80 s when every rank of the 16-rank pool built all four k-points).
        """
        from parsec_python.V_ion import build_nonlocal_projectors
        from parsec_python.V_ion.ionic_potential import NonlocalProjectorOperator

        atoms = list(atoms)
        group = self.group
        if group is None or group.size == 1 or len(atoms) < group.size:
            return build_nonlocal_projectors(grid, atoms, potentials, specifications, lattice_vectors, **keywords)
        start = time.perf_counter()
        first = len(atoms) * group.rank // group.size
        last = len(atoms) * (group.rank + 1) // group.size
        part = build_nonlocal_projectors(grid, atoms[first:last], potentials, specifications, lattice_vectors, **keywords)
        pieces = self._pool_comm.allgather(
            (first, sp.csc_matrix(part.projectors), np.asarray(part.signs), tuple(part.labels))
        )
        matrices, signs, labels = [], [], []
        for offset, matrix, piece_signs, piece_labels in sorted(pieces, key=lambda piece: piece[0]):
            matrices.append(matrix)
            signs.append(piece_signs)
            labels.extend((int(atom) + offset, int(l), int(m)) for atom, l, m in piece_labels)
        self._tick("projector_build", start)
        return NonlocalProjectorOperator(
            sp.hstack(matrices, format="csc"), np.concatenate(signs), tuple(labels)
        )

    def extras(self) -> list[Any]:
        raise NotImplementedError("the k-point pools compute no per-task extras")

    def vectors(self, task: int) -> np.ndarray:
        """Orbitals are not gathered: an empty block of the right number of rows."""
        return np.empty((self.problem_size, 0), dtype=np.complex128)

    def export_states(self) -> dict[int, Any]:
        raise NotImplementedError(
            "the Ritz states of the MPI k-point pools stay on their ranks (a hybrid functional's Fock loop "
            "runs without --mpi)"
        )

    def import_states(self, states: dict[int, Any]) -> None:
        raise NotImplementedError(
            "the MPI k-point pools start from their own first solve (a hybrid functional's Fock loop runs "
            "without --mpi)"
        )

    def stats(self) -> dict[str, float]:
        """Eigensolver counters and seconds of this rank and its pool's exchanges; the plan."""
        stats = dict(self._stats)
        if self.plan is not None:
            stats["pools"] = float(self.plan.pools)
            stats["pool_ranks"] = float(self.group.size)
            stats["pool_tasks"] = float(len(self.tasks))
            stats["pool_estimate_bytes"] = float(self.plan.bytes_per_rank)
            stats["pool_static_bytes"] = float(self.static_bytes)
            stats["pool_host_storage"] = float(self.host_storage)
            stats["pool_resident"] = float(self.plan.resident or len(self.tasks))
            stats["pool_host_setup_bytes"] = float(getattr(self, "host_at_setup_bytes", 0.0))
            stats["pool_host_peak_bytes"] = float(getattr(self, "host_peak_bytes", 0.0))
            stats["pool_reserved_bytes"] = float(getattr(self, "reserved_bytes", 0.0))
        return stats

    def residuals(self) -> list[float]:
        return [self._residual.get(task, float("nan")) for task in range(self.job.n_tasks)]

    def describe_plan(self) -> str:
        """One line on the pools, for a log (empty before the first solve)."""
        if self.plan is None:
            return ""
        return (
            f"{self.plan.pools} pool(s) of {'/'.join(str(size) for size in sorted(set(self.plan.sizes), reverse=True))} "
            f"rank(s), {self.job.n_tasks} task(s); about {self.plan.bytes_per_rank / 1e9:.1f} GB per GPU"
        )

    def close(self) -> None:
        self.states.clear()
        comm, self._pool_comm = self._pool_comm, None
        free = getattr(comm, "Free", None)
        if free is not None:
            try:
                free()
            except Exception:  # noqa: BLE001 - a communicator of the tests, or MPI already finalized
                pass

    def __enter__(self) -> "MPIPoolKPointSolver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _static_device_bytes(operator) -> int:
    """Bytes of the device matrices and fields of one operator (stencils, projectors, potential)."""
    total = 0
    for name in ("laplacian", "k_gradient", "projector_forward", "projector_adjoint", "diagonal"):
        value = getattr(operator, name, None)
        if value is None:
            continue
        if hasattr(value, "indices"):  # CSR on either backend
            total += sum(int(getattr(value, part).nbytes) for part in ("data", "indices", "indptr"))
        elif hasattr(value, "nbytes"):
            total += int(value.nbytes)
    return total


__all__ = [
    "MPIPoolKPointSolver",
    "POOLS_ENV",
    "PoolPlan",
    "plan_pools",
    "pool_memory_bytes",
    "requested_pools",
]
