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
    # (filter degree) -> settings; with Dynamic_Diag_Tol also called as
    # (filter degree, tolerance).
    eigval_settings: Callable[..., EigvalSettings]
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

    def solve(
        self, potentials: dict, filter_degree: int, tolerance: float | None = None
    ) -> list[np.ndarray]:
        """One eigenvalue array per task (task order), vectors kept in the solver.

        ``tolerance`` replaces ``Diag_Tolerance`` for this call
        (``Dynamic_Diag_Tol``); the job's ``eigval_settings`` then takes it as
        a second argument.
        """
        settings = job_eigval_settings(self.job, filter_degree, tolerance)
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

    def export_states(self) -> dict[int, Any]:
        """The Ritz state of every task, on the host: what the SCF loop of the next structure of a relaxation or
        dynamics run filters instead of solving from random vectors."""
        return dict(self.states)

    def import_states(self, states: dict[int, Any]) -> None:
        """Start the tasks from the Ritz states ``states`` (by task) of an earlier loop."""
        self.states.update(states)

    def vectors(self, task: int) -> np.ndarray:
        """The final eigenvectors of one task."""
        return self._vectors[task]

    def close(self) -> None:
        pass

    def __enter__(self) -> "SerialKPointSolver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def job_eigval_settings(
    job: KPointJob, filter_degree: int, tolerance: float | None = None
) -> EigvalSettings:
    """The eigensolver settings of one SCF step: ``Diag_Tolerance`` unless the
    step gives its own ``tolerance`` (``Dynamic_Diag_Tol``)."""
    if tolerance is None:
        return job.eigval_settings(filter_degree)
    return job.eigval_settings(filter_degree, tolerance)


DEVICE_FIRST_SOLVERS = ("chebff", "chebdav")
"""First solvers the device-resident eigensolver
(:func:`parsec_python.acceleration.Eigensolvers.chebff_spinor.solve_spinor_eigval`)
implements: CHEBFF, and CHEBDAV as CHEBFF cycles to CHEBDAV's residual test,
each followed by SUBSPACE steps."""


def device_solver_supports(settings: EigvalSettings) -> bool:
    """Whether the device-resident eigensolver carries ``settings``.

    ARPACK and DIAGLA (``Eigensolver: arpack`` / ``diagla``) are host
    eigensolvers: a device worker runs them with the host route (the device
    Hamiltonian as a linear operator), as before the device solver existed.
    """
    return getattr(settings, "initial_method", "chebff") in DEVICE_FIRST_SOLVERS


def kept_vectors(solver, task: int, what: str) -> np.ndarray:
    """The final eigenvectors of ``task`` as a host array, for a driver that keeps them.

    The row-distributed MPI solver does not gather the orbitals (its
    ``vectors`` is an empty block): a driver that needs them (forces, the
    Ylm-projected DOS, Output_GW, spinor parsec.dat) stops here instead of
    computing with an empty set.
    """
    vectors = solver.vectors(task)
    host = getattr(vectors, "get", None)
    vectors = np.asarray(host() if host is not None else vectors)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise NotImplementedError(
            f"{what} needs the orbitals of the k-point eigenproblems, which the "
            "MPI-distributed k-point solver does not gather (run without --mpi)"
        )
    return vectors


def whole_orbitals(vectors, grid_size: int, what: str) -> np.ndarray:
    """``vectors`` as host orbitals of the whole grid (``grid_size`` rows, or twice that
    for spinors), for a density part that is not a sum over grid rows (the LDA+U
    occupation matrices of :mod:`parsec_python.SCF.kpoint_hubbard`)."""
    rows = int(vectors.shape[0])
    if rows not in (grid_size, 2 * grid_size):
        raise NotImplementedError(
            f"{what} needs the orbitals on every grid point; the MPI-distributed "
            "k-point solver holds a block of grid rows on each rank (run without --mpi)"
        )
    host = getattr(vectors, "get", None)
    return np.asarray(host() if host is not None else vectors)


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
    "DEVICE_FIRST_SOLVERS",
    "KPointJob",
    "SerialKPointSolver",
    "device_solver_supports",
    "job_eigval_settings",
    "kept_vectors",
    "make_kpoint_solver",
    "map_tasks",
    "parallel_backend_report",
    "parallel_backend_uses_device",
    "residual_max",
    "set_parallel_backend",
    "split_pooled",
    "whole_orbitals",
]
