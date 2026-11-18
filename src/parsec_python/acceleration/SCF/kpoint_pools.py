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

Tasks of different sizes (the sectors of the k-points, :mod:`.kpoint_sectors`: ``rows_per_task`` and
``states_per_task`` of the job) are planned by their own sizes (:func:`plan_pools_by_size`): the tasks, the
largest work (rows times states squared) first, go each to the pool that would finish it first, among those
whose devices hold it (with the subspaces not in hand on the host where they must) and whose nodes hold what
their ranks keep on the host (a fraction of the memory the node has free when the pools are formed); the
balance is the time of the slowest pool in units of the largest task.  So a few large sectors get pools of
their own (diamond 9 x 9 x 9 with a 2 x 2 x 2 mesh: eight sectors of 1.06 million rows and 3,799 states, 65 GB
each, and 24 smaller ones), where dealing every task round-robin as if it were the largest had put two large
ones in the pools of 8 of the 16 ranks, their idle subspaces on the host, 260 GB of a node of 256.
``PARSEC_KPOINT_POOL_ASSIGN=cyclic`` restores that rule (the plan of the largest task, round-robin).
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
    resized_job,
    task_rows,
    task_states,
)

from ..Eigensolvers.chebff_columns import (
    ColumnGroup,
    POOL_FILTER_COLUMNS,
    ColumnLayout,
    kpoint_ritz_route,
    requested_spans,
    solve_column_eigval,
)
from ..Eigensolvers.chebff_spinor import _filter_columns
from ..Eigensolvers.mpi_shared_basis import exchange_bytes
from ..Parallel import mpi

POOLS_ENV = "PARSEC_KPOINT_POOLS"
ASSIGN_ENV = "PARSEC_KPOINT_POOL_ASSIGN"
# The share of the memory a node has free when the pools are formed (``MemAvailable``, after every rank has
# prepared its system) that the subspaces its ranks keep on the host may take.
_HOST_FRACTION = 0.8
# Balances of plans by size closer than this (relative) count as equal.
_BALANCE_TIE = 0.02
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


def _round_width(dimension: int) -> int:
    """Columns per rank in a round of the block route (:meth:`.chebff_columns.ColumnGroup.block_rounds`)."""
    return max(1, exchange_bytes() // (16 * max(int(dimension), 1)))


def requested_pools() -> int | None:
    """``PARSEC_KPOINT_POOLS`` as a positive number, ``None`` (the default rule) when unset or ``auto``."""
    text = os.environ.get(POOLS_ENV, "").strip().lower()
    if text in ("", "auto"):
        return None
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"{POOLS_ENV} (--kpoint-pools) must be a positive number or auto, not {text!r}")
    return int(text)


def assignment_mode() -> str:
    """``PARSEC_KPOINT_POOL_ASSIGN``: ``auto`` (tasks of different sizes planned by their sizes) or ``cyclic``."""
    text = os.environ.get(ASSIGN_ENV, "").strip().lower() or "auto"
    if text not in ("auto", "cyclic"):
        raise ValueError(f"{ASSIGN_ENV} must be auto or cyclic, not {text!r}")
    return text


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
    route: str | None = None,
) -> int:
    """Device bytes one rank of a pool of ``ranks`` ranks needs for ``tasks`` tasks.

    The Ritz subspaces of the pool's tasks (this rank's columns of each, ``dimension x width``
    complex; ``resident`` of them on the device, all by default, one when the others stay on the
    host), the working copies of the task in hand (under ``PARSEC_KPOINT_RITZ=block``, the default,
    its one share with a round of padding and the send, receive and ``H X`` buffers of a round;
    under ``rows`` the rows, the columns and the rows of ``H X`` of a Rayleigh-Ritz step: three of
    its share in all, on one rank, where rows and columns are the same arrays, two, and the exchange
    buffers of a round), the dense
    ``m x m`` matrices of the Rayleigh-Ritz step, the blocks of columns of the filter recurrence and of
    the products of the operator (eight blocks of ``PARSEC_SPINOR_FILTER_COLUMNS`` columns, 32 by default)
    and the static data (:data:`_FIXED_BYTES_PER_POINT`, :data:`_TASK_BYTES_PER_POINT`).
    """
    dimension, working, tasks, ranks = int(dimension), int(working), int(tasks), int(ranks)
    share = _share_bytes(dimension, working, ranks, block_size)
    resident = tasks if resident is None else max(1, min(int(resident), tasks))
    route = kpoint_ritz_route() if route is None else route
    hand = _hand_bytes(dimension, working, ranks, block_size, route)
    fixed = _FIXED_BYTES_PER_POINT + (0 if fused else _SPARSE_FIXED_BYTES_PER_POINT)
    per_task = _TASK_BYTES_PER_POINT + (0 if fused else _SPARSE_TASK_BYTES_PER_POINT)
    static = dimension * (fixed + tasks * per_task)
    return int((resident - 1) * share + hand + static + _CONTEXT_BYTES)


def _share_bytes(dimension: int, working: int, ranks: int, block_size: int) -> int:
    """Bytes of one rank's columns of a subspace of ``dimension`` rows and ``working`` columns."""
    width = max(ColumnLayout(working, block_size, ranks).widths) if ranks > 1 else working
    return int(dimension) * int(width) * 16


def _hand_bytes(dimension: int, working: int, ranks: int, block_size: int, route: str) -> int:
    """Device bytes of the task in hand on one rank: its subspace in the working arrays of the route, the
    exchange buffers, the dense ``m x m`` matrices and the blocks of the filter (:func:`pool_memory_bytes`)."""
    dimension, working = int(dimension), int(working)
    share = _share_bytes(dimension, working, ranks, block_size)
    if route == "block":
        # one share in hand (and a round of padding where the widths differ), the send, receive and
        # H X buffers of a round (PARSEC_KPOINT_RITZ=block)
        subspace = share + (dimension * 16 * _round_width(dimension) if ranks > 1 else 0)
        buffers = 3 * exchange_bytes() if ranks > 1 else 0
    else:
        subspace = (3 if ranks > 1 else 2) * share
        buffers = 2 * exchange_bytes() if ranks > 1 else 0
    dense = 6 * working * working * 16
    transient = 8 * dimension * min(_filter_columns(POOL_FILTER_COLUMNS), working) * 16
    return int(subspace + buffers + dense + transient)


def pool_task_bytes(
    tasks,
    ranks: int,
    block_size: int = 1,
    *,
    resident: int | None = None,
    grid_points: int | None = None,
    fused: bool = True,
    route: str | None = None,
) -> tuple[int, int]:
    """``(device, host)`` bytes of one rank of a pool of ``ranks`` ranks whose tasks have the sizes ``tasks``.

    ``tasks`` holds ``(rows, working columns)`` of each task in the order of the pool; the first
    ``resident - 1`` keep their subspaces on the device, the others move theirs to the host between their
    solves (all stay on the device where ``resident`` is ``None`` or covers them), as :class:`MPIPoolKPointSolver`
    does.  The device holds the subspaces kept, the task in hand (:func:`_hand_bytes`; while a kept one is in
    hand, the moving one the last solve left there as well), the fields of the grid of ``grid_points`` points
    and the operators of every task; the host the moving subspaces but one.  For tasks of one size it is
    :func:`pool_memory_bytes`.
    """
    sizes = [(int(rows), int(working)) for rows, working in tasks]
    count, ranks = len(sizes), int(ranks)
    resident = count if resident is None else max(1, min(int(resident), count))
    route = kpoint_ritz_route() if route is None else route
    shares = [_share_bytes(rows, working, ranks, block_size) for rows, working in sizes]
    hands = [_hand_bytes(rows, working, ranks, block_size, route) for rows, working in sizes]
    largest = max(rows for rows, _ in sizes)
    points = largest if grid_points is None else max(int(grid_points), largest)
    fixed = _FIXED_BYTES_PER_POINT + (0 if fused else _SPARSE_FIXED_BYTES_PER_POINT)
    per_task = _TASK_BYTES_PER_POINT + (0 if fused else _SPARSE_TASK_BYTES_PER_POINT)
    static = points * fixed + sum(rows for rows, _ in sizes) * per_task
    if resident >= count:
        total = sum(shares)
        device = max(total - share + hand for share, hand in zip(shares, hands))
        host = 0
    else:
        kept = sum(shares[: resident - 1])
        moving = shares[resident - 1 :]
        in_hand = max(hands[resident - 1 :])
        if resident > 1:
            in_hand = max(in_hand, max(hand - share for share, hand in zip(shares[: resident - 1], hands[: resident - 1])) + max(moving))
        device = kept + in_hand
        host = sum(moving) - min(moving)
    return int(device + static + _CONTEXT_BYTES), int(host)


# The share of a device above which the pools take the block route under PARSEC_KPOINT_RITZ=auto although
# the rows route fits as well: measured on 16 A100 (bulk diamond), rows was faster where it needed a third of
# a device (1,000 atoms, 6-8 %) and slower where it needed nearly all of it (1,728 atoms, 5 %).
_BLOCK_ABOVE = 0.5


def choose_route(rows_plan: "PoolPlan | None", block_plan: "PoolPlan", usable: float) -> str:
    """The route of the pools under ``PARSEC_KPOINT_RITZ=auto`` from the plans of both routes.

    ``block`` where ``rows`` has no plan that fits, a worse one (by the order of :func:`plan_pools`:
    balance, subspaces on the host, fewer pools) or one that needs more than half of ``usable``;
    ``rows`` otherwise (its four moves of the basis per pass against five).
    """
    if rows_plan is None:
        return "block"
    order = lambda plan: (round(plan.balance, 12), plan.host_storage, -plan.pools)  # noqa: E731
    if order(block_plan) < order(rows_plan):
        return "block"
    return "block" if rows_plan.bytes_per_rank > _BLOCK_ABOVE * float(usable) else "rows"


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
    # the tasks of every pool where they were planned by their sizes (plan_pools_by_size); None: round-robin
    assignment: tuple[tuple[int, ...], ...] | None = None
    residents: tuple[int, ...] | None = None  # the subspaces each pool keeps on the device (with assignment)
    host_bytes_per_node: float = 0.0  # the estimate of the node whose ranks keep the most on the host

    @property
    def starts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.concatenate(([0], np.cumsum(self.sizes[:-1]))))

    def pool_of_rank(self, rank: int) -> int:
        return int(np.searchsorted(np.cumsum(self.sizes), rank, side="right"))

    def tasks_of_pool(self, pool: int) -> list[int]:
        if self.assignment is not None:
            return list(self.assignment[pool])
        return list(range(pool, self.tasks, self.pools))

    def resident_of(self, pool: int) -> int:
        """The subspaces the ranks of ``pool`` keep on the device while they move the others to the host."""
        return int(self.resident if self.residents is None else self.residents[pool])

    def spills(self, pool: int) -> bool:
        """Whether ``pool`` moves the subspaces of the tasks not in hand to the host."""
        if not self.host_storage:
            return False
        if self.residents is None:
            return True
        return self.residents[pool] < len(self.tasks_of_pool(pool))


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


def plan_pools_by_size(
    tasks,
    n_ranks: int,
    bytes_of,
    usable: float,
    requested: int | None = None,
    storage: str = "auto",
    limit: float | None = None,
    nodes=None,
    host_budget=None,
) -> PoolPlan:
    """The pools of tasks of different sizes (see the module text).

    ``tasks`` holds ``(rows, working columns)`` of every task; ``bytes_of(sizes, ranks, resident)`` gives the
    ``(device, host)`` bytes of one rank of a pool of ``ranks`` ranks with tasks of the sizes ``sizes`` (in
    the order of the pool) and ``resident`` subspaces on the device (``None``: all), as
    :func:`pool_task_bytes`; ``nodes`` the node of every rank and ``host_budget`` the host bytes the ranks of
    each node may keep (no check where ``None``).  ``usable``, ``requested``, ``storage`` and ``limit`` as in
    :func:`plan_pools`.  Every pool gets a task (a count of pools that leaves one idle is not taken).
    """
    sizes = [(int(rows), int(working)) for rows, working in tasks]
    n_tasks, n_ranks = len(sizes), int(n_ranks)
    works = [float(rows) * float(working) * float(working) for rows, working in sizes]
    order = sorted(range(n_tasks), key=lambda task: (-works[task], task))
    largest = max(works)
    nodes = [0] * n_ranks if nodes is None else [int(node) for node in nodes]

    def fit(members, ranks, ceiling):
        """``(resident, device, host)`` of a pool of ``ranks`` ranks with ``members``, ``None`` if it cannot hold them."""
        shapes = [sizes[task] for task in members]
        device, host = bytes_of(shapes, ranks, None)
        if storage != "host" or len(members) == 1:
            if device <= ceiling:
                return len(members), device, host
            if storage == "device":
                return None
        for resident in range(len(members) - 1, 0, -1):  # as many on the device as fit, the fewest on the host
            if storage == "host" and resident > 1:
                continue
            device, host = bytes_of(shapes, ranks, resident)
            if device <= ceiling:
                return resident, device, host
        return None

    def assign(count, ceiling, check_host):
        pool_sizes = tuple(n_ranks // count + (1 if j < n_ranks % count else 0) for j in range(count))
        starts = np.concatenate(([0], np.cumsum(pool_sizes)))
        ranks_of = [range(int(starts[j]), int(starts[j + 1])) for j in range(count)]
        members: list[list[int]] = [[] for _ in range(count)]
        loads = [0.0] * count
        fits = [(0, 0, 0)] * count
        node_host: dict[int, float] = {}

        def host_after(pool, host):
            """The host bytes of every node of ``pool`` with ``host`` per rank of it."""
            change: dict[int, float] = {}
            for rank in ranks_of[pool]:
                change[nodes[rank]] = change.get(nodes[rank], 0.0) + float(host) - float(fits[pool][2])
            return {node: node_host.get(node, 0.0) + delta for node, delta in change.items()}

        for task in order:
            candidates = sorted(range(count), key=lambda j: ((loads[j] + works[task]) / pool_sizes[j], j))
            for pool in candidates:
                found = fit(members[pool] + [task], pool_sizes[pool], ceiling)
                if found is None:
                    continue
                after = host_after(pool, found[2])
                if check_host and host_budget is not None and any(
                    amount > float(host_budget.get(node, float("inf"))) for node, amount in after.items() if amount > 0
                ):
                    continue
                members[pool].append(task)
                loads[pool] += works[task]
                fits[pool] = found
                node_host.update(after)
                break
            else:
                return None
        if any(not tasks_of for tasks_of in members):
            return None
        residents = tuple(found[0] for found in fits)
        spill = any(resident < len(tasks_of) for resident, tasks_of in zip(residents, members))
        return PoolPlan(
            count,
            pool_sizes,
            n_tasks,
            max(found[1] for found in fits),
            max(load / size for load, size in zip(loads, pool_sizes)) / largest,
            spill,
            min(resident for resident, tasks_of in zip(residents, members) if resident < len(tasks_of)) if spill else 0,
            tuple(tuple(tasks_of) for tasks_of in members),
            residents,
            max(node_host.values(), default=0.0),
        )

    counts = range(1, min(n_tasks, n_ranks) + 1)
    if requested is not None:
        if requested > min(n_tasks, n_ranks):
            raise ValueError(
                f"--kpoint-pools {requested}: at most {min(n_tasks, n_ranks)} pools for {n_tasks} tasks "
                f"(k-points and spin channels) on {n_ranks} ranks"
            )
        ceiling = usable if limit is None else max(float(limit), usable)
        # a requested count is refused only where the devices cannot hold it (the host estimate steers only)
        plan = assign(requested, usable, True) or assign(requested, ceiling, True) or assign(requested, ceiling, False)
        if plan is None:
            raise MemoryError(
                f"--kpoint-pools {requested}: the tasks do not fit {ceiling / 1e9:.1f} GB per GPU in {requested} "
                "pools (or leave a pool without a task); fewer pools share each basis over more ranks"
            )
        return plan
    plans = [plan for plan in (assign(count, usable, True) for count in counts) if plan is not None]
    if not plans:
        unchecked = [plan for plan in (assign(count, usable, False) for count in counts) if plan is not None]
        if unchecked:
            least = min(plan.host_bytes_per_node for plan in unchecked)
            budget = min(float(value) for value in host_budget.values()) if host_budget else float("inf")
            raise MemoryError(
                f"the k-point subspaces kept on the host need about {least / 1e9:.1f} GB on a node, and "
                f"{budget / 1e9:.1f} GB are free there; more nodes (or --kpoint-pools to take a count anyway)"
            )
        raise MemoryError(
            f"the k-point tasks do not fit {usable / 1e9:.1f} GB per GPU on {n_ranks} ranks in any number of pools: "
            "more ranks (GPUs) are needed"
        )
    # balances within a few per cent are equal (tasks of nearly equal sizes): then no subspaces on the host,
    # then the most pools (fewer ranks per pool, less exchange)
    best = min(plan.balance for plan in plans)
    near = [plan for plan in plans if plan.balance <= best * (1.0 + _BALANCE_TIE)]
    return min(near, key=lambda plan: (plan.host_storage, -plan.pools, plan.balance))


def node_layout(context) -> tuple[list[int], dict[int, float]]:
    """The node of every rank of ``context`` and the host bytes the pools may keep on each node
    (:data:`_HOST_FRACTION` of the least ``MemAvailable`` its ranks see; collective)."""
    import socket

    gathered = context.comm.allgather((socket.gethostname(), available_host_bytes()))
    names: dict[str, int] = {}
    nodes = [names.setdefault(name, len(names)) for name, _ in gathered]
    budget: dict[int, float] = {}
    for node, (_, free) in zip(nodes, gathered):
        budget[node] = min(budget.get(node, float("inf")), _HOST_FRACTION * float(free))
    return nodes, budget


def available_host_bytes() -> float:
    """``MemAvailable`` of ``/proc/meminfo`` in bytes (infinite where there is none)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as info:
            for line in info:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    return float("inf")


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


def _column_major(array) -> bool:
    """Whether ``array`` is column-major and not also row-major (the bases of the block route)."""
    flags = getattr(array, "flags", None)
    return flags is not None and bool(flags.f_contiguous) and not bool(flags.c_contiguous)


def _download(backend, array) -> np.ndarray:
    """A host copy of a device array in its own layout.

    A copy to the host comes back row-major, so a column-major array (the basis of the block route) is first
    copied row-major on the device: a second basis on the device at the moment the first is moved off it (a
    sector of 1.06 million rows and 3805 states of diamond 9 x 9 x 9, 65 GB on an 80 GB A100, failed so).  Its
    transpose is row-major already: copied as it is, and transposed back on the host (a view).
    """
    if _column_major(array):
        return backend.asnumpy(array.T).T
    return backend.asnumpy(array)


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
        if job.extras is not None and not job.extras_by_columns:
            raise NotImplementedError(
                "the k-point pools gather no per-state extras (spin moments of spinors); "
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
        # the largest basis of a task (the sectors of the k-points have their own counts)
        most = max(task_states(self.job, task) for task in range(self.job.n_tasks))
        working = min(most + int(settings.safety_buffer), self.problem_size)
        block = int(settings.subspace.block_size)
        self._block, self._buffer = block, int(settings.safety_buffer)  # for resize_tasks
        sizes = [
            (rows, min(task_states(self.job, task) + int(settings.safety_buffer), rows))
            for task, rows in ((task, task_rows(self.job, task, self.problem_size)) for task in range(self.job.n_tasks))
        ]
        # tasks of different sizes are planned by their sizes (collective: the nodes of the ranks and their memory)
        by_size = len(set(sizes)) > 1 and assignment_mode() == "auto"
        nodes, host_budget = node_layout(context) if by_size else (None, None)
        if self.xp is np:
            host_budget = None  # ranks on the CPU keep no subspaces on the host
        plan = None
        failure = None
        route = None
        if context.is_root:
            try:
                usable = self._usable_bytes()

                def plan_of(name):
                    if by_size:
                        return plan_pools_by_size(
                            sizes,
                            context.size,
                            lambda shapes, ranks, resident: pool_task_bytes(
                                shapes, ranks, block, resident=resident, grid_points=self.job.grid_points, route=name
                            ),
                            usable,
                            self._requested_pools,
                            storage_mode() if self.xp is not np else "device",
                            limit=usable / _USABLE_FRACTION,
                            nodes=nodes,
                            host_budget=host_budget,
                        )
                    return plan_pools(
                        self.job.n_tasks,
                        context.size,
                        lambda tasks, ranks, resident: pool_memory_bytes(
                            self.problem_size, working, tasks, ranks, block, resident=resident, route=name
                        ),
                        usable,
                        self._requested_pools,
                        storage_mode() if self.xp is not np else "device",
                        limit=usable / _USABLE_FRACTION,
                    )

                setting = kpoint_ritz_route()
                if setting == "auto":
                    try:
                        rows_plan = plan_of("rows")
                    except MemoryError:
                        rows_plan = None
                    block_plan = plan_of("block")
                    route = choose_route(rows_plan, block_plan, usable)
                    plan = block_plan if route == "block" else rows_plan
                else:
                    route, plan = setting, plan_of(setting)
            except (MemoryError, ValueError) as error:
                failure = (type(error).__name__, str(error))
        plan, route, failure = context.bcast((plan, route, failure), root=0)
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
        self.route = route
        self.group.route = route

    # ------------------------------------------------------------------- steps
    def solve(
        self, potentials: dict, filter_degree: int, tolerance: float | None = None, tasks=None
    ) -> list[np.ndarray]:
        """One eigenvalue array per task, the same on every rank.

        ``tolerance`` replaces ``Diag_Tolerance`` for this call (``Dynamic_Diag_Tol``); ``tasks`` solves those only
        (collective: every rank names the same), the others keep the eigenvalues of their last solve.
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
        order = self._solve_order()
        if tasks is not None:
            wanted = {int(task) for task in tasks}
            order = [task for task in order if task in wanted]
            kept = getattr(self, "_kept", None)
            if kept is not None and kept not in wanted and kept in self.states:
                self._spill(kept, self.states[kept])  # host storage keeps one subspace on the device: the next
        for task in order:
            requested = task_states(self.job, task)
            start = time.perf_counter()
            hamiltonian = self.job.make_hamiltonian(task, potentials)
            self._tick("hamiltonian", start)  # the Bloch projectors of a k-point on first use
            start = time.perf_counter()
            operator = self.converter.operator(hamiltonian)
            self._tick("upload", start)
            if not self.static_bytes:
                self.static_bytes = _static_device_bytes(operator)
            state = self.states.pop(task, None)
            self._release_cached()
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
                route=self.route,
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
        if order:
            self._kept = order[-1]
        start = time.perf_counter()
        gathered = self.context.comm.allgather((found, residuals))
        self._tick("pool_wait", start)  # the eigenvalues of the other pools: waiting for the slowest
        eigenvalues = getattr(self, "_values", None)
        if eigenvalues is None:
            eigenvalues = self._values = {}
        for values, norms in gathered:
            eigenvalues.update(values)
            self._residual.update(norms)
        missing = [task for task in range(self.job.n_tasks) if task not in eigenvalues]
        if missing:
            raise RuntimeError(f"no pool returned the eigenvalues of tasks {missing}")
        return [eigenvalues[task] for task in range(self.job.n_tasks)]

    def resize_tasks(self, counts: dict) -> bool:
        """New numbers of states for the tasks of ``counts`` (task -> states), collective.

        The tasks keep their pools: where the plan still holds them on the devices (and the nodes) their Ritz
        states are dropped, those of the other tasks kept, and ``True`` is returned; ``False`` where it does not
        (the caller makes a new solver).
        """
        counts = {int(task): int(count) for task, count in counts.items()}
        job = resized_job(self.job, counts)
        if self.plan is not None:
            fits = None
            if self.context.is_root:
                fits = self._plan_holds(job)
            if not bool(self.context.bcast(fits, root=0)):
                return False
        self.job = job
        for task in counts:
            self.states.pop(task, None)
            if getattr(self, "_kept", None) == task:
                self._kept = None
        return True

    def _plan_holds(self, job) -> bool:
        """Whether the pools of :attr:`plan` hold the tasks of ``job`` (their new numbers of states)."""
        plan = self.plan
        block, buffer = self._block, self._buffer
        usable = self._usable_bytes()
        ceiling = max(float(plan.bytes_per_rank), usable)
        if plan.assignment is None:
            most = max(task_states(job, task) for task in range(job.n_tasks))
            working = min(most + buffer, self.problem_size)
            resident = plan.resident if plan.host_storage else None
            return max(
                pool_memory_bytes(
                    self.problem_size, working, len(plan.tasks_of_pool(pool)), plan.sizes[pool], block,
                    resident=resident, route=self.route,
                )
                for pool in range(plan.pools)
            ) <= ceiling
        for pool in range(plan.pools):
            shapes = []
            for task in plan.tasks_of_pool(pool):
                rows = task_rows(job, task, self.problem_size)
                shapes.append((rows, min(task_states(job, task) + buffer, rows)))
            resident = plan.resident_of(pool) if plan.spills(pool) else None
            device, _ = pool_task_bytes(
                shapes, plan.sizes[pool], block, resident=resident, grid_points=job.grid_points, route=self.route
            )
            if device > ceiling:
                return False
        return True

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

        for task in self.tasks:
            requested = task_states(self.job, task)
            state = self.states[task]
            self._release_cached()
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
        return bool(
            self.plan is not None
            and self.plan.host_storage
            and self.plan.spills(self.pool)
            and len(self.tasks) > 1
            and self.xp is not np
        )

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
        return self.tasks.index(task) < max(self.plan.resident_of(self.pool) - 1, 0)

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
        state.vectors = _download(self.backend, state.vectors)
        self._tick("spill", start)
        self._note_host()

    def _release_cached(self) -> None:
        """Return the unused blocks of CuPy's pool to the device before a subspace is laid out on it (host storage).

        A subspace moved to the host leaves its block in the pool; the arrays of the next task are cut out of it,
        and while they live the block cannot be given back, so a subspace as large coming back from the host needs
        a second block, which a subspace of more than half the device does not find.  Released before every solve
        and every density of a task, each subspace takes a block of its own (the same numbers).

        Only in a pool of one rank: with four ranks per pool, diamond 9 x 9 x 9 ran out of device memory after five
        to seven SCF steps (CuPy's pool held 43.8 GB, and 4 GB more were not to be had on the 80 GB device): the
        blocks given back did not return to the device, most likely because the ranks of a larger pool exchange
        device buffers that MPI keeps mapped between the ranks of a node (CUDA IPC).  Without the release that run
        took all its steps.
        """
        if self.xp is np or not self.host_storage or self.group is None or self.group.size > 1:
            return
        self.backend.cp.get_default_memory_pool().free_all_blocks()

    def _resident(self, state) -> None:
        """Bring the subspace of ``state`` back to the device before its solve.

        Under the block route the solve lays a host copy into an allocation with room for its rows itself.
        """
        if state is None or not isinstance(state.vectors, np.ndarray) or self.xp is np:
            return
        if getattr(self, "route", None) == "block":
            return
        start = time.perf_counter()
        state.vectors = self.backend.asarray(state.vectors)
        self._tick("restore", start)

    def _device_vectors(self, state):
        """The subspace of ``state`` on the device (a copy of a host one, released after use)."""
        if isinstance(state.vectors, np.ndarray) and self.xp is not np:
            start = time.perf_counter()
            host = state.vectors
            # a column-major host copy goes up as it is (no row-major copy of it on the host first)
            vectors = self.xp.asarray(host, order="F") if _column_major(host) else self.backend.asarray(host)
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
        """The job's extras of every task (``extras_by_columns``: the nonlocal forces), the same on every rank.

        Every rank sums its columns of the tasks of its pool (``extras(task, columns of this rank, columns=their
        states)``); the parts of all tasks are summed over ``COMM_WORLD``.
        """
        if self.job.extras is None or not self.job.extras_by_columns:
            raise NotImplementedError("the k-point pools compute extras only where they sum over the states")
        self.backend.use()
        start = time.perf_counter()
        found: dict[int, np.ndarray] = {}
        for task in self.tasks:
            requested = task_states(self.job, task)
            state = self.states[task]
            self._release_cached()
            vectors = self._device_vectors(state)
            count = int(requested)
            part = None
            if getattr(state, "layout", None) is None:
                spans = [(0, 0, count)]
            else:
                spans = list(requested_spans(state.layout, self.group.rank, count))
            for first, local, width in spans:
                if width <= 0:
                    continue
                piece = np.asarray(
                    _host(
                        self.job.extras(
                            task, vectors[:, local : local + width], xp=self.xp, columns=np.arange(first, first + width)
                        )
                    ),
                    dtype=np.float64,
                )
                part = piece if part is None else part + piece
            if part is not None:
                found[task] = part
            del vectors
            if task != getattr(self, "_kept", None):
                self._spill(task, state)
        self._tick("extras", start)
        shapes = {tuple(value.shape) for value in found.values()}
        shape = self.context.comm.allgather(next(iter(shapes)) if shapes else None)
        shape = next(value for value in shape if value is not None)
        stacked = np.zeros((self.job.n_tasks,) + tuple(shape), dtype=np.float64)
        for task, value in found.items():
            stacked[task] = value
        flat = stacked.reshape(-1)
        self.context.allsum_host(flat)
        stacked = flat.reshape(stacked.shape)
        return [stacked[task] for task in range(self.job.n_tasks)]

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
            stats["pool_resident"] = float(
                self.plan.resident_of(self.pool) if self.host_storage else len(self.tasks)
            )
            stats["pool_host_node_estimate_bytes"] = float(self.plan.host_bytes_per_node)
            stats["pool_assignment_by_size"] = float(self.plan.assignment is not None)
            stats["pool_host_setup_bytes"] = float(getattr(self, "host_at_setup_bytes", 0.0))
            stats["pool_host_peak_bytes"] = float(getattr(self, "host_peak_bytes", 0.0))
            stats["pool_reserved_bytes"] = float(getattr(self, "reserved_bytes", 0.0))
            stats["pool_block_route"] = float(getattr(self, "route", None) == "block")
        return stats

    def residuals(self) -> list[float]:
        return [self._residual.get(task, float("nan")) for task in range(self.job.n_tasks)]

    def describe_plan(self) -> str:
        """One line on the pools, for a log (empty before the first solve)."""
        if self.plan is None:
            return ""
        return (
            f"{self.plan.pools} pool(s) of {'/'.join(str(size) for size in sorted(set(self.plan.sizes), reverse=True))} "
            f"rank(s), {self.job.n_tasks} task(s); about {self.plan.bytes_per_rank / 1e9:.1f} GB per GPU; "
            f"{getattr(self, 'route', None) or 'rows'} route"
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
    "ASSIGN_ENV",
    "MPIPoolKPointSolver",
    "POOLS_ENV",
    "PoolPlan",
    "assignment_mode",
    "plan_pools",
    "plan_pools_by_size",
    "pool_memory_bytes",
    "pool_task_bytes",
    "requested_pools",
]
