"""Parallel solution of the per-k-point eigenproblems of a periodic SCF.

The performance counterpart of
:class:`parsec_python.SCF.kpoint_solver.SerialKPointSolver`, which is the
readable definition.  The diagonalizations of the k-points are independent, so
they are spread over workers (:mod:`parsec_python.acceleration.Parallel`):
threads on the CPU, one thread per CUDA device on a GPU.  A worker keeps what
belongs to its tasks: the Bloch-phase projectors, the eigenvectors and the Ritz
subspace the next SCF step starts from.  The driver sends the effective
potentials, receives the eigenvalues, sends back the occupations and receives
the summed partial densities.  The eigenvectors never leave the worker.

With one worker the inline backend runs this in the calling process.  Parallel
results equal the serial ones to round-off (the eigenvalues exactly: each task
is the same computation; the densities differ only by the order of the sum over
tasks).  Importing :mod:`parsec_python.acceleration` installs this solver and
:func:`map_tasks` as the parallel backend of the reference drivers
(:func:`parsec_python.SCF.kpoint_solver.set_parallel_backend`).
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from parsec_python.Eigensolvers import solve_eigval
from parsec_python.SCF.kpoint_solver import KPointJob, set_parallel_backend

from ..Parallel import WORKERS_ENV, make_pool

def resolve_backend(requested: str | None = None):
    """The device backend plan, or ``None`` for the CPU.

    The device follows ``PARSEC_DEVICE`` (what ``--device`` sets, default
    ``auto``): ``gpu`` and ``auto`` use the GPU path (one worker thread per
    CUDA device, ``PARSEC_CUPY_DEVICES`` selects them) when CuPy and a device
    work, ``auto`` falling back to the CPU and ``gpu`` raising without one;
    ``cpu`` never uses it.  ``requested`` or ``PARSEC_KPOINT_BACKEND``
    (``cpu``, ``cupy``, ``auto`` or ``numpy-device``, the device code on the
    host, for tests) overrides that.  The acceleration package is only
    imported when a device could be used.
    """
    name = (requested or os.environ.get("PARSEC_KPOINT_BACKEND", "")).strip().lower()
    if not name:
        device = os.environ.get("PARSEC_DEVICE", "auto").strip().lower() or "auto"
        name = {"cpu": "cpu", "gpu": "cupy", "auto": "auto"}.get(device)
        if name is None:
            raise ValueError(f"PARSEC_DEVICE must be auto, gpu or cpu, not {device!r}")
    if name == "cpu":
        return None
    from ..Hamiltonian.device_bloch import resolve_kpoint_backend

    plan = resolve_kpoint_backend(name)
    return plan if plan.uses_device else None


def device_worker_count(plan, n_tasks: int) -> int:
    """Worker threads of a device plan: one per device (host double: ``PARSEC_KPOINT_WORKERS``)."""
    if plan.name == "numpy-device":
        requested = os.environ.get(WORKERS_ENV, "1")
        count = int(requested) if requested.strip().isdigit() else 1
    else:
        count = len(plan.device_ids)
    return max(1, min(count, n_tasks))


class _SolveWorker:
    """Owns the k-point tasks ``tasks``: their Ritz states and eigenvectors."""

    def __init__(
        self, job: KPointJob, tasks: list[int], index: int, plan=None
    ) -> None:
        self.job = job
        self.tasks = tasks
        self.index = index
        self.states: dict[int, Any] = {}
        self.vectors: dict[int, np.ndarray] = {}
        # Built here, in the worker (thread), so the device is current for it.
        self.converter = plan.converter(index) if plan is not None else None

    def operator(self, hamiltonian):
        if self.converter is not None:
            return self.converter.operator(hamiltonian).as_linear_operator()
        return hamiltonian.as_linear_operator()

    def solve(self, potentials: dict, filter_degree: int):
        settings = self.job.eigval_settings(filter_degree)
        for task in self.tasks:
            operator = self.operator(self.job.make_hamiltonian(task, potentials))
            solution = solve_eigval(
                operator,
                self.job.number_of_states,
                settings=settings,
                state=self.states.get(task),
            )
            self.states[task] = solution.state
            self.vectors[task] = np.asarray(solution.vectors)
            yield task, np.asarray(solution.eigenvalues, dtype=float)

    def accumulate(self, occupations: dict[int, np.ndarray]):
        total: dict[str, np.ndarray] = {}
        for task in self.tasks:
            parts = self.job.accumulate(task, self.vectors[task], occupations[task])
            for name, value in parts.items():
                if name in total:
                    total[name] += value
                else:
                    total[name] = np.array(value, copy=True)
        yield None, total

    def extras(self):
        for task in self.tasks:
            yield task, self.job.extras(task, self.vectors[task])

    def vectors_of(self, task: int):
        if task in self.vectors:
            yield task, self.vectors[task]


class PooledKPointSolver:
    """Drives a pool of workers through the SCF steps of a k-point job."""

    def __init__(
        self,
        job: KPointJob,
        *,
        workers: int | str | None = None,
        problem_size: int | None = None,
        backend: str | None = None,
    ) -> None:
        self.job = job
        plan = resolve_backend(backend)
        if plan is not None:
            workers = device_worker_count(plan, job.n_tasks)
        self.backend = plan.name if plan is not None else "cpu"
        self.pool = make_pool(
            range(job.n_tasks),
            lambda tasks, index: _SolveWorker(job, tasks, index, plan),
            workers=workers,
            problem_size=problem_size,
        )

    @property
    def n_workers(self) -> int:
        return self.pool.n_workers

    def solve(self, potentials: dict, filter_degree: int) -> list[np.ndarray]:
        """One eigenvalue array per task (task order), vectors kept in the workers."""
        by_task = self.pool.by_task("solve", potentials, filter_degree)
        return [by_task[task] for task in range(self.job.n_tasks)]

    def accumulate(self, occupations: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Sum of every task's density contribution for the given occupations."""
        payload = {task: occupations[task] for task in range(self.job.n_tasks)}
        total: dict[str, np.ndarray] = {}
        for _, part in self.pool.results("accumulate", payload):
            for name, value in part.items():
                if name in total:
                    total[name] = total[name] + value
                else:
                    total[name] = value
        return total

    def extras(self) -> list[Any]:
        by_task = self.pool.by_task("extras")
        return [by_task[task] for task in range(self.job.n_tasks)]

    def vectors(self, task: int) -> np.ndarray:
        """The final eigenvectors of one task (fetched from its worker)."""
        for _, vectors in self.pool.results("vectors_of", task):
            return vectors
        raise KeyError(task)

    def close(self) -> None:
        self.pool.close()

    def __enter__(self) -> "PooledKPointSolver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _TaskWorker:
    """Runs ``solve_task(task, converter)`` for its tasks (band paths and the like)."""

    def __init__(self, solve_task, tasks, index, plan=None) -> None:
        self.solve_task = solve_task
        self.tasks = tasks
        # Built here, in the worker (thread), so the device is current for it.
        self.converter = plan.converter(index) if plan is not None else None

    def run(self):
        for task in self.tasks:
            yield task, self.solve_task(task, self.converter)


def map_tasks(solve_task, n_tasks: int, *, problem_size=None, workers=None, backend=None):
    """Yield ``(task, result)`` as the workers finish the independent tasks."""
    plan = resolve_backend(backend)
    if plan is not None:
        workers = device_worker_count(plan, n_tasks)
    with make_pool(
        range(n_tasks),
        lambda tasks, index: _TaskWorker(solve_task, tasks, index, plan),
        workers=workers,
        problem_size=problem_size,
    ) as pool:
        yield from pool.stream("run")


def install() -> None:
    """Make this the parallel backend of the reference k-point drivers and band paths."""
    set_parallel_backend(PooledKPointSolver, map_tasks)


__all__ = ["PooledKPointSolver", "device_worker_count", "install", "map_tasks", "resolve_backend"]
