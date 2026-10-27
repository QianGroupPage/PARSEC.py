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

On a device the worker keeps the Ritz subspace of every k-point *on the device* and runs the
SUBSPACE steps of the later SCF iterations there (:func:`..Eigensolvers.chebff_spinor.solve_spinor_eigval`:
filter, QR and Rayleigh-Ritz without leaving the GPU).  The first solve is CHEBFF on the device; for ``Eigensolver: chebdav`` it is CHEBFF followed by
more CHEBFF cycles until CHEBDAV's residual test is met (the device counterpart of
CHEBDAV; ``PARSEC_KPOINT_FIRST_SOLVE=chebdav`` runs the host CHEBDAV with the device filter instead and
moves its result to the device).  The eigenvectors come to the host only to form the density.  When the subspaces of all
the worker's k-points would not fit in device memory they are kept on the host between solves
(``PARSEC_KPOINT_STATE_STORAGE``).  ``PARSEC_KPOINT_SOLVER=host`` restores the former route: the host
eigensolver with only the Chebyshev filter on the device.

With one worker the inline backend runs this in the calling process.  Parallel
results equal the serial ones to round-off (the eigenvalues exactly: each task
is the same computation; the densities differ only by the order of the sum over
tasks).  Importing :mod:`parsec_python.acceleration` installs this solver and
:func:`map_tasks` as the parallel backend of the reference drivers
(:func:`parsec_python.SCF.kpoint_solver.set_parallel_backend`).
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np

from parsec_python.Eigensolvers import solve_eigval
from parsec_python.SCF.kpoint_solver import KPointJob, residual_max, set_parallel_backend

from ..Eigensolvers.chebff_spinor import SpinorEigvalState, solve_spinor_eigval
from ..Parallel import WORKERS_ENV, make_pool, mpi

SOLVER_ENV = "PARSEC_KPOINT_SOLVER"  # device (default) | host
STORAGE_ENV = "PARSEC_KPOINT_STATE_STORAGE"  # auto (default) | device | host
FIRST_SOLVE_ENV = "PARSEC_KPOINT_FIRST_SOLVE"  # refine (default) | chebdav
_AUTO_FRACTION = 0.75  # of the device memory the subspaces (and one transient copy set) may take


def _device_solver_requested() -> bool:
    value = os.environ.get(SOLVER_ENV, "device").strip().lower() or "device"
    if value not in ("device", "host"):
        raise ValueError(f"{SOLVER_ENV} must be device or host, not {value!r}")
    return value == "device"


def _first_solve_on_host() -> bool:
    """Whether ``Eigensolver: chebdav`` runs as the host CHEBDAV (``chebdav``) or is replaced on the
    device by CHEBFF with extra cycles until CHEBDAV's residual test is met (``refine``)."""
    value = os.environ.get(FIRST_SOLVE_ENV, "refine").strip().lower() or "refine"
    if value not in ("refine", "chebdav"):
        raise ValueError(f"{FIRST_SOLVE_ENV} must be refine or chebdav, not {value!r}")
    return value == "chebdav"


def _storage_mode() -> str:
    value = os.environ.get(STORAGE_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "device", "host"):
        raise ValueError(f"{STORAGE_ENV} must be auto, device or host, not {value!r}")
    return value


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
        self.vectors: dict[int, Any] = {}
        self.stats: dict[str, float] = {}
        self.residual: dict[int, float] = {}
        # Built here, in the worker (thread), so the device is current for it.
        self.converter = plan.converter(index) if plan is not None else None
        self.device_solver = self.converter is not None and _device_solver_requested()
        self._spill: bool | None = None

    def _tick(self, name: str, start: float) -> None:
        """Add the seconds since ``start`` to the worker's ``worker_<name>_seconds`` counter."""
        if self.converter is not None:
            self.converter.backend.synchronize()
        key = f"worker_{name}_seconds"
        self.stats[key] = self.stats.get(key, 0.0) + time.perf_counter() - start

    # ------------------------------------------------------------- host route
    def operator(self, hamiltonian):
        if self.converter is not None:
            return self.converter.operator(hamiltonian).as_linear_operator()
        return hamiltonian.as_linear_operator()

    # ----------------------------------------------------------- device route
    def _should_spill(self, dimension: int, working: int) -> bool:
        """Whether the subspaces of this worker's tasks stay on the host between solves."""
        mode = _storage_mode()
        if mode != "auto":
            return mode == "host"
        backend = self.converter.backend
        if backend.name != "cupy":
            return False
        backend.use()
        _, total = backend.cp.cuda.runtime.memGetInfo()
        needed = (len(self.tasks) + 6) * dimension * working * 16
        return needed > _AUTO_FRACTION * total

    def _to_host(self, array) -> np.ndarray:
        return array if isinstance(array, np.ndarray) else self.converter.backend.asnumpy(array)

    def _solve_on_device(self, task: int, hamiltonian, settings):
        backend = self.converter.backend
        backend.use()
        start = time.perf_counter()
        operator = self.converter.operator(hamiltonian)
        self._tick("upload", start)
        requested = self.job.number_of_states
        solve_start = time.perf_counter()
        state = self.states.pop(task, None)
        self.vectors.pop(task, None)  # release the previous basis before the next one is made
        if state is not None and isinstance(state.vectors, np.ndarray) and backend.xp is not np:
            state.vectors = backend.asarray(state.vectors)  # a spilled subspace comes back
        if state is None and settings.initial_method == "chebdav" and _first_solve_on_host():
            # CHEBDAV (the first solve of Eigensolver: chebdav) runs on the host with the device
            # filter; its subspace then continues on the device.
            host = solve_eigval(operator.as_linear_operator(), requested, settings=settings, state=None)
            saved = host.state.subspace
            new_state = SpinorEigvalState(
                vectors=backend.asarray(np.ascontiguousarray(saved.vectors)),
                eigenvalues=np.asarray(saved.eigenvalues, dtype=float),
                working_states=saved.working_states,
                operator_dimension=saved.operator_dimension,
                first_filter=True,
                filter_lower_bound=None,
                filters_completed=0,
            )
            eigenvalues = np.asarray(host.eigenvalues, dtype=float)
            self.residual[task] = residual_max(host.residual_norms)
        else:
            result = solve_spinor_eigval(
                operator,
                requested,
                settings=settings,
                state=state,
                xp=backend.xp,
                sync=backend.synchronize,
                # Eigensolver: chebdav converges its first solve; on the device that is CHEBFF plus
                # more cycles to the same residual test
                converge_first=settings.initial_method == "chebdav",
            )
            new_state = result.state
            eigenvalues = result.eigenvalues
            self.residual[task] = residual_max(result.residual_norms)
            for name, value in result.stats.items():
                self.stats[name] = self.stats.get(name, 0.0) + value
        self._tick("solve", solve_start)
        if self._spill is None:
            self._spill = self._should_spill(new_state.operator_dimension, new_state.working_states)
        if self._spill:
            spill_start = time.perf_counter()
            host_vectors = self._to_host(new_state.vectors)
            new_state.vectors = host_vectors
            self.vectors[task] = host_vectors[:, :requested]
            self._tick("spill", spill_start)
        else:
            self.vectors[task] = new_state.vectors[:, :requested]
        self.states[task] = new_state
        return eigenvalues

    def _host_vectors(self, task: int) -> np.ndarray:
        return np.asarray(self._to_host(self.vectors[task]))

    # ------------------------------------------------------------- SCF steps
    def solve(self, potentials: dict, filter_degree: int):
        settings = self.job.eigval_settings(filter_degree)
        for task in self.tasks:
            start = time.perf_counter()
            hamiltonian = self.job.make_hamiltonian(task, potentials)
            self._tick("hamiltonian", start)  # builds the Bloch projectors of a k-point on first use
            if self.device_solver:
                yield task, self._solve_on_device(task, hamiltonian, settings)
                continue
            start = time.perf_counter()
            operator = self.operator(hamiltonian)
            self._tick("upload", start)
            start = time.perf_counter()
            solution = solve_eigval(
                operator,
                self.job.number_of_states,
                settings=settings,
                state=self.states.get(task),
            )
            self._tick("solve", start)
            self.states[task] = solution.state
            self.vectors[task] = np.asarray(solution.vectors)
            self.residual[task] = residual_max(solution.residual_norms)
            yield task, np.asarray(solution.eigenvalues, dtype=float)

    def accumulate(self, occupations: dict[int, np.ndarray]):
        total: dict[str, np.ndarray] = {}
        for task in self.tasks:
            vectors = self.vectors[task]
            if self.converter is not None and not isinstance(vectors, np.ndarray):
                # form the density where the orbitals are: only the grid-length fields come back
                start = time.perf_counter()
                backend = self.converter.backend
                backend.use()
                parts = {
                    name: np.asarray(backend.asnumpy(value))
                    for name, value in self.job.accumulate(
                        task, vectors, occupations[task], xp=backend.xp
                    ).items()
                }
                self._tick("density", start)
            else:
                start = time.perf_counter()
                vectors = self._host_vectors(task)
                self._tick("download", start)
                start = time.perf_counter()
                parts = self.job.accumulate(task, vectors, occupations[task])
                self._tick("density", start)
            for name, value in parts.items():
                if name in total:
                    total[name] += value
                else:
                    total[name] = np.array(value, copy=True)
        yield None, total

    def extras(self):
        for task in self.tasks:
            vectors = self.vectors[task]
            if self.converter is not None and not isinstance(vectors, np.ndarray):
                self.converter.backend.use()
                yield task, self.job.extras(task, vectors, xp=self.converter.backend.xp)
            else:
                yield task, self.job.extras(task, self._host_vectors(task))

    def vectors_of(self, task: int):
        if task in self.vectors:
            yield task, self._host_vectors(task)

    def stats_of(self):
        yield None, dict(self.stats)

    def residuals_of(self):
        yield None, dict(self.residual)


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

    def stats(self) -> dict[str, float]:
        """Eigensolver counters and seconds summed over the workers (device route)."""
        total: dict[str, float] = {}
        for _, part in self.pool.results("stats_of"):
            for name, value in part.items():
                total[name] = total.get(name, 0.0) + value
        return total

    def residuals(self) -> list[float]:
        """Largest eigenpair residual of each task's last solve (NaN where none was measured)."""
        merged: dict[int, float] = {}
        for _, part in self.pool.results("residuals_of"):
            merged.update(part)
        return [merged.get(task, float("nan")) for task in range(self.job.n_tasks)]

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


def _mpi_map_tasks(context, solve_task, n_tasks: int, backend=None):
    """The independent tasks dealt out to the MPI ranks, each solved whole on the device of its rank;
    every rank gets all the results (in task order) and yields them."""
    plan = resolve_backend(backend)
    converter = plan.converter(context.local_rank) if plan is not None else None
    mine = [(task, solve_task(task, converter)) for task in range(context.rank, n_tasks, context.size)]
    gathered = {}
    for part in context.comm.allgather(mine):
        gathered.update(part)
    for task in range(n_tasks):
        yield task, gathered[task]


def map_tasks(solve_task, n_tasks: int, *, problem_size=None, workers=None, backend=None):
    """Yield ``(task, result)`` as the workers finish the independent tasks."""
    context = mpi.context()
    if context is not None and context.size > 1:
        yield from _mpi_map_tasks(context, solve_task, n_tasks, backend)
        return
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


def make_solver(job: KPointJob, **options):
    """The solver of a k-point job: row-distributed over the MPI ranks if the run is an MPI run
    (``PARSEC_MPI=1``), else the pool of threads and devices of this process."""
    if mpi.context() is not None:
        from .kpoint_mpi import MPIKPointSolver

        return MPIKPointSolver(job, **options)
    return PooledKPointSolver(job, **options)


def describe() -> list[str]:
    """The lines of ``parsec.out`` that say what the k-point eigenproblems run on."""
    lines = [" Execution resources (k-point eigensolver):", " -------------------------------------------"]
    context = mpi.context()
    try:
        plan = resolve_backend()
    except Exception as error:  # no GPU where one was requested, for instance
        return lines + [f" Device            = unavailable ({error})"]
    if plan is None:
        lines += [
            " Device            = CPU",
            " Eigensolver       = host, k-points and spin channels spread over worker threads",
        ]
    else:
        names = []
        if plan.name == "cupy":
            try:
                from ..backends.cupy import require_cupy

                cp, _ = require_cupy()
                for device_id in plan.device_ids[:1]:
                    names.append(cp.cuda.runtime.getDeviceProperties(device_id)["name"].decode())
            except Exception:
                pass
        kind = {"cupy": "GPU", "numpy-device": "CPU (device code on the host, a test double)"}[plan.name]
        ids = ",".join(str(i) for i in plan.device_ids)
        lines.append(f" Device            = {kind}" + (f" ({names[0]})" if names else ""))
        if plan.name == "cupy":
            if context is not None:
                lines.append(f" GPUs/node         = {len(plan.device_ids)} visible, index {ids}")
            else:
                lines.append(f" GPUs              = {len(plan.device_ids)} active, index {ids}")
        route = (
            "device-resident Ritz subspaces (filter, QR and Rayleigh-Ritz on the device)"
            if _device_solver_requested()
            else "host eigensolver, only the Chebyshev filter on the device"
        )
        lines.append(f" Eigensolver       = {route}")
    if context is not None:
        import socket

        device = None
        if plan is not None and plan.name == "cupy":
            device = plan.device_ids[context.local_rank % len(plan.device_ids)]
        # every rank takes part: the report says where all of them run, not only rank 0
        placement = context.comm.allgather((socket.gethostname(), device))
        nodes: dict[str, list] = {}
        for host, gpu in placement:
            nodes.setdefault(host, []).append(gpu)
        lines.append(
            f" MPI               = {context.size} ranks on {len(nodes)} node(s), {context.ranks_per_node} per node; "
            "the basis of every eigenproblem is split by blocks of grid rows over the ranks"
        )
        if device is not None:
            lines.append(
                " GPU-aware MPI     = "
                + (
                    "on: halo rows go between the GPUs and overlap the local product; the sums go "
                    + ("between the GPUs" if context.cuda_aware_reduce else "through the host")
                    if context.cuda_aware
                    else "off: halo rows and sums go through the host (PARSEC_MPI_CUDA_AWARE=1 for GPU buffers)"
                )
            )
            lines.append(
                f" GPUs              = {sum(len(g) for g in nodes.values())} active, one per rank, "
                f"on {len(nodes)} node(s):"
            )
            for host, gpus in nodes.items():
                lines.append(f"   node {host}: GPU index {','.join(str(g) for g in gpus)}")
    elif plan is not None and plan.name == "cupy":
        lines.append(" Workers           = one thread per GPU; the k-points and spin channels are dealt out to them")
    return lines


def install() -> None:
    """Make this the parallel backend of the reference k-point drivers and band paths."""
    set_parallel_backend(
        make_solver,
        map_tasks,
        uses_device=lambda: mpi.requested() or resolve_backend() is not None,
        describe=describe,
    )


__all__ = ["PooledKPointSolver", "device_worker_count", "install", "map_tasks", "resolve_backend"]
