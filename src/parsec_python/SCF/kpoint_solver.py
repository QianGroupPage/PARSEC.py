"""Per-k-point eigenproblems of a periodic SCF, and the hook that parallelizes them.

Every SCF step of a k-point-sampled calculation diagonalizes one Bloch
Hamiltonian per k-point (per spin channel for collinear spin) and then needs
only the pooled eigenvalues (one Fermi level) and the summed density.  A
*job* describes the physics of one driver:

``make_hamiltonian(task, potentials)``
    the Bloch Hamiltonian of task ``task`` for the potentials dict;
``accumulate(task, vectors, occupations)``
    the task's contribution to the density, as a ``{name: array}`` dict;
``extras(task, vectors)`` (optional)
    a small per-task quantity computed from the final vectors.

:class:`SerialKPointSolver` is the readable definition: it solves the tasks one
after another in the calling process and keeps each task's Ritz state and
eigenvectors for the next step.  The performance counterpart, which spreads the
tasks over worker threads and GPUs, lives in
:mod:`parsec_python.acceleration.SCF.kpoint_parallel` and installs itself here
(:func:`set_parallel_backend`) when the acceleration package is imported.  The
drivers call only :func:`make_kpoint_solver` and :func:`map_tasks`, so they do
not depend on the acceleration package.  The two paths are checked against each
other in ``acceleration/tests/test_kpoint_parallel.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np

from ..Eigensolvers import EigvalSettings, solve_eigval


@dataclass
class KPointJob:
    """The physics of one SCF driver."""

    n_tasks: int
    number_of_states: int
    eigval_settings: Callable[[int], EigvalSettings]
    make_hamiltonian: Callable[[int, dict], Any]
    accumulate: Callable[[int, np.ndarray, np.ndarray], dict[str, np.ndarray]]
    extras: Callable[[int, np.ndarray], Any] | None = None


class SerialKPointSolver:
    """Solves the tasks of a :class:`KPointJob` in order, in the calling process."""

    backend = "cpu"
    n_workers = 1

    def __init__(self, job: KPointJob, **_ignored) -> None:
        self.job = job
        self.states: dict[int, Any] = {}
        self._vectors: dict[int, np.ndarray] = {}
        self._residuals: dict[int, float] = {}

    def solve(self, potentials: dict, filter_degree: int) -> list[np.ndarray]:
        """One eigenvalue array per task (task order), vectors kept in the solver."""
        settings = self.job.eigval_settings(filter_degree)
        eigenvalues = []
        for task in range(self.job.n_tasks):
            operator = self.job.make_hamiltonian(task, potentials).as_linear_operator()
            solution = solve_eigval(
                operator,
                self.job.number_of_states,
                settings=settings,
                state=self.states.get(task),
            )
            self.states[task] = solution.state
            self._vectors[task] = np.asarray(solution.vectors)
            self._residuals[task] = residual_max(solution.residual_norms)
            eigenvalues.append(np.asarray(solution.eigenvalues, dtype=float))
        return eigenvalues

    def residuals(self) -> list[float]:
        """Largest eigenpair residual of each task's last solve (NaN where none was measured)."""
        return [self._residuals.get(task, float("nan")) for task in range(self.job.n_tasks)]

    def accumulate(self, occupations: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Sum of every task's density contribution for the given occupations."""
        total: dict[str, np.ndarray] = {}
        for task in range(self.job.n_tasks):
            parts = self.job.accumulate(task, self._vectors[task], occupations[task])
            for name, value in parts.items():
                if name in total:
                    total[name] = total[name] + value
                else:
                    total[name] = np.array(value, copy=True)
        return total

    def extras(self) -> list[Any]:
        return [self.job.extras(task, self._vectors[task]) for task in range(self.job.n_tasks)]

    def vectors(self, task: int) -> np.ndarray:
        """The final eigenvectors of one task."""
        return self._vectors[task]

    def close(self) -> None:
        pass

    def __enter__(self) -> "SerialKPointSolver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def residual_max(norms) -> float:
    """Largest residual norm, NaN if the solver reported none."""
    if norms is None:
        return float("nan")
    norms = np.asarray(norms, dtype=float)
    return float(np.max(norms)) if norms.size else float("nan")


def split_pooled(pooled: np.ndarray, per_task: list[np.ndarray]) -> list[np.ndarray]:
    """Cut a task-major pooled array back into one block per task."""
    blocks = []
    offset = 0
    for reference in per_task:
        blocks.append(pooled[offset : offset + reference.size])
        offset += reference.size
    return blocks


_parallel_solver: Callable[..., Any] | None = None
_parallel_mapper: Callable[..., Iterator[tuple[int, Any]]] | None = None
_uses_device: Callable[[], bool] | None = None
_describe: Callable[[], list[str]] | None = None


def set_parallel_backend(
    solver: Callable[..., Any] | None,
    mapper: Callable[..., Any] | None,
    uses_device: Callable[[], bool] | None = None,
    describe: Callable[[], list[str]] | None = None,
) -> None:
    """Install (or, with ``None``, remove) the parallel solver factory and task mapper.

    ``uses_device`` tells whether the installed solver would run on a device for the current run;
    ``describe`` returns the lines of ``parsec.out`` that say what it runs on.
    """
    global _parallel_solver, _parallel_mapper, _uses_device, _describe
    _parallel_solver, _parallel_mapper, _uses_device, _describe = solver, mapper, uses_device, describe


def parallel_backend_report() -> list[str]:
    """What the k-point eigenproblems run on, for the log; the serial solver on the host if none is installed."""
    if _describe is None:
        return [
            " Execution resources (k-point eigensolver):",
            " -------------------------------------------",
            " Device            = CPU (serial solver, no parallel backend installed)",
        ]
    return list(_describe())


def parallel_backend_uses_device() -> bool:
    """Whether the installed parallel backend would solve the k-points on a device."""
    return _uses_device is not None and bool(_uses_device())


def make_kpoint_solver(job: KPointJob, *, problem_size: int | None = None, **options):
    """The installed parallel solver for ``job``, else the serial one."""
    if _parallel_solver is None:
        return SerialKPointSolver(job)
    return _parallel_solver(job, problem_size=problem_size, **options)


def map_tasks(
    solve_task: Callable[[int, Any], Any],
    n_tasks: int,
    *,
    problem_size: int | None = None,
    **options,
) -> Iterator[tuple[int, Any]]:
    """Yield ``(task, solve_task(task, converter))`` for every task, in completion order.

    ``converter`` is the device operator converter of the worker that runs the
    task, ``None`` on the CPU.  Serially the tasks run in order in this process.
    """
    if _parallel_mapper is None:
        for task in range(n_tasks):
            yield task, solve_task(task, None)
        return
    yield from _parallel_mapper(solve_task, n_tasks, problem_size=problem_size, **options)


__all__ = [
    "KPointJob",
    "SerialKPointSolver",
    "make_kpoint_solver",
    "map_tasks",
    "parallel_backend_report",
    "parallel_backend_uses_device",
    "residual_max",
    "set_parallel_backend",
    "split_pooled",
]
