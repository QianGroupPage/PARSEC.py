"""Checks of the k-point pools with ranks as threads (see test_kpoint_pools.py); prints ``RESULT {json}``.

    python -m parsec_python.acceleration.tests.pool_check eigensolver METHOD SIZES STEPS [EXCHANGE_BYTES]
    python -m parsec_python.acceleration.tests.pool_check scf DRIVER RANKS POOLS
    python -m parsec_python.acceleration.tests.pool_check sectors RANKS POOLS
    python -m parsec_python.acceleration.tests.pool_check guard RANKS SECTOR_POOLS PLAIN_POOLS

Run it with ``OPENBLAS_NUM_THREADS=1`` (test_kpoint_pools.py does): the package imports NumPy before
this module, so the settings below come too late for the BLAS of a ``-m`` run (:mod:`.thread_mpi`).  The
device code runs on the host (``numpy-device``).  ``eigensolver`` compares the shared-column eigensolver on each number of
ranks of ``SIZES`` (``1,2,3``) with the one-device solver over ``STEPS`` solves; ``scf`` runs the k-point
driver ``DRIVER`` of ``test_kpoint_parallel.DRIVERS`` on ``RANKS`` ranks in ``POOLS`` pools and in one
process, with one task per k-point (``PARSEC_KPOINT_SECTORS=off``); ``sectors`` the scalar SCF of the silicon cube
with the eight time-reversal invariant k-points split into the sectors of their little groups
(:mod:`..SCF.kpoint_sectors`), the same on the pools and in one process; ``guard`` the spin-polarized loop that
leaves the sectors of its k-points (the symmetry guard) and runs again on pools of another layout (:func:`check_guard`).
"""

from __future__ import annotations

import json
import os
import sys

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PARSEC_KPOINT_BACKEND"] = "numpy-device"
os.environ["PARSEC_KPOINT_WORKERS"] = "1"
os.environ.pop("PARSEC_MPI", None)
# the checks of the pools compare one task per k-point; ``sectors`` turns the split on for its runs
os.environ["PARSEC_KPOINT_SECTORS"] = "off"

import threading  # noqa: E402
from unittest import mock  # noqa: E402

import numpy as np  # noqa: E402

from parsec_python.acceleration.Eigensolvers.chebff_columns import ColumnGroup, solve_column_eigval  # noqa: E402
from parsec_python.acceleration.Eigensolvers.chebff_spinor import solve_spinor_eigval  # noqa: E402
from parsec_python.acceleration.SCF.kpoint_pools import MPIPoolKPointSolver  # noqa: E402
from parsec_python.acceleration.tests.thread_mpi import run_ranks  # noqa: E402
from parsec_python.Eigensolvers import ChebFFSettings, EigvalSettings, SubspaceSettings  # noqa: E402

_REQUESTED = 30


def _operator():
    from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter, NumpyArrayBackend
    from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
    from parsec_python.tests.unit.test_hamiltonian_fast_paths import _PeriodicFixture

    _PeriodicFixture.setUpClass()
    fx = _PeriodicFixture
    host = KPointKohnShamHamiltonian(fx.laplacian, fx.gradient, fx.potential, fx.nonlocal_k, fx.k_point)
    return DeviceBlochConverter(NumpyArrayBackend()).operator(host)


def _settings(method: str) -> EigvalSettings:
    """``chebff`` or ``chebdav``; ``-smart`` appended: the smart SUBSPACE lower bound (Smart_Subspace)."""
    method, _, rule = method.partition("-")
    return EigvalSettings(
        safety_buffer=6,
        initial_method=method,
        chebff=ChebFFSettings(polynomial_degree=20, filter_cycles=3),
        subspace=SubspaceSettings(
            polynomial_degree=12, degree_delta=3, block_size=6, lower_bound_rule=rule or "monotone"
        ),
    )


def _solves(solve, steps: int):
    state, out = None, []
    for _ in range(steps):
        result = solve(state)
        state = result.state
        out.append((result.eigenvalues.copy(), None if result.residual_norms is None else result.residual_norms.copy()))
    return out


def check_eigensolver(method: str, sizes: str, steps: int) -> dict:
    operator = _operator()
    settings = _settings(method)
    converge = method.startswith("chebdav")
    reference = _solves(
        lambda state: solve_spinor_eigval(
            operator, _REQUESTED, settings=settings, state=state, xp=np, converge_first=converge
        ),
        steps,
    )
    report = {}
    for size in (int(value) for value in sizes.split(",")):

        def rank(context):
            group = ColumnGroup(context.comm, np)
            return _solves(
                lambda state: solve_column_eigval(
                    operator, _REQUESTED, settings=settings, group=group, state=state, xp=np, converge_first=converge
                ),
                steps,
            )

        results = run_ranks(size, rank)
        equal = all(
            np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            for other in results[1:]
            for a, b in zip(other, results[0])
        )
        value_error = max(float(np.max(np.abs(a[0] - b[0]))) for a, b in zip(results[0], reference))
        norm_error = max(
            float(np.max(np.abs(a[1] - b[1]))) if a[1] is not None and b[1] is not None else (0.0 if a[1] is b[1] else 1.0)
            for a, b in zip(results[0], reference)
        )
        report[str(size)] = {"ranks_equal": bool(equal), "value_error": value_error, "norm_error": norm_error}
    return report


_LOCAL = threading.local()


def _thread_solver(job, **options):
    return MPIPoolKPointSolver(job, context=_LOCAL.context, pools=_LOCAL.pools, **options)


def _summary(result) -> dict:
    density = getattr(result, "density", None)
    if density is None:
        density = result.density_up
    return {
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "energy": float(result.energies.total),
        "fermi": float(result.fermi_level),
        "density": np.asarray(density, dtype=float).tolist(),
    }


def check_scf(name: str, ranks: int, pools: int) -> dict:
    from parsec_python.acceleration.tests import test_kpoint_parallel as fixtures

    driver, flags, soc = fixtures.DRIVERS[name]
    system = fixtures._system(soc=soc, criterion=1.0e-4, **flags)
    serial = _summary(driver(system, fixtures._K, fixtures._WEIGHTS))

    def rank(context):
        _LOCAL.context, _LOCAL.pools = context, pools
        return _summary(driver(system, fixtures._K, fixtures._WEIGHTS))

    with mock.patch("parsec_python.SCF.kpoints.make_kpoint_solver", _thread_solver):
        pooled = run_ranks(ranks, rank)
    return {"serial": serial, "pooled": pooled}


def check_sectors(ranks: int, pools: int) -> dict:
    """The scalar SCF of the silicon cube at the eight time-reversal invariant k-points, its k-points in the sectors
    of their little groups: the pools (tasks: the sectors) against one process."""
    from dataclasses import replace

    from parsec_python.acceleration.SCF import kpoint_sectors
    from parsec_python.acceleration.SCF.kpoint_sectors import sector_solver
    from parsec_python.acceleration.tests.test_periodic_point_group import silicon
    from parsec_python.SCF.kpoints import run_scf_kpoints
    from parsec_python.SCF.pbc import prepare_periodic_single_point

    os.environ["PARSEC_KPOINT_SECTORS"] = "auto"
    kpoint_sectors._MINIMUM_ENTRIES = 0  # the small cube is split all the same
    if os.environ.get("POOL_CHECK_GROW"):
        # no margin and no extra states: sectors miss levels, get more states and are solved again
        kpoint_sectors._EXTRA_STATES = 0
        os.environ["PARSEC_KPOINT_SECTOR_MARGIN"] = "1.0"
    base = silicon("pbe", spacing=0.855)
    base = replace(base, scf=replace(base.scf, convergence_criterion=1.0e-7))
    system = prepare_periodic_single_point(base)
    side = float(np.asarray(base.periodic_cell.lattice_vectors)[0, 0])
    k_points = np.array([np.array(bits, dtype=float) * np.pi / side for bits in np.ndindex(2, 2, 2)])
    weights = np.full(8, 0.125)
    def counters(found) -> dict:
        detail = found.timings.detail
        names = ("tasks", "regrowths", "regrown_tasks", "rebuilds")
        return {name: float(detail.get(f"kpoint_sector_{name}", 0.0)) for name in names}

    result = run_scf_kpoints(system, k_points, weights)
    serial = dict(_summary(result), **counters(result))

    def solver(job, **options):
        made = sector_solver(job, _thread_solver, **options)
        return _thread_solver(job, **options) if made is None else made

    def rank(context):
        _LOCAL.context, _LOCAL.pools = context, pools
        found = run_scf_kpoints(system, k_points, weights)
        return dict(_summary(found), **counters(found))

    with mock.patch("parsec_python.SCF.kpoints.make_kpoint_solver", solver):
        pooled = run_ranks(ranks, rank)
    return {"serial": serial, "pooled": pooled}


def check_forces(ranks: int, pools: int, sectors: str) -> dict:
    """The nonlocal forces of the scalar k-point SCF (one atom of the silicon cube moved, Gamma and X) on pools of
    ranks against one process: the k-points whole (``sectors=off``) or in the sectors of their little groups."""
    from dataclasses import replace

    from parsec_python.acceleration.SCF import kpoint_sectors
    from parsec_python.acceleration.SCF.kpoint_sectors import sector_solver
    from parsec_python.acceleration.tests.test_periodic_point_group import silicon
    from parsec_python.SCF.kpoints import run_scf_kpoints
    from parsec_python.SCF.pbc import prepare_periodic_single_point

    os.environ["PARSEC_KPOINT_SECTORS"] = sectors
    kpoint_sectors._MINIMUM_ENTRIES = 0
    base = silicon("pbe", spacing=0.855)
    atoms = list(base.atoms)
    atoms[0] = replace(atoms[0], position=np.asarray(atoms[0].position, dtype=float) + np.array([0.1, 0.0, 0.0]))
    base = replace(base, atoms=atoms, scf=replace(base.scf, convergence_criterion=1.0e-7))
    system = prepare_periodic_single_point(base)
    side = float(np.asarray(base.periodic_cell.lattice_vectors)[0, 0])
    k_points = np.array([np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0])])
    weights = np.array([0.5, 0.5])

    def summary(found) -> dict:
        return dict(
            _summary(found),
            forces=np.asarray(found.nonlocal_forces, dtype=float).tolist(),
            tasks=float(found.timings.detail.get("kpoint_sector_tasks", 0.0)),
        )

    serial = summary(run_scf_kpoints(system, k_points, weights, compute_forces=True))

    def solver(job, **options):
        made = sector_solver(job, _thread_solver, **options)
        return _thread_solver(job, **options) if made is None else made

    def rank(context):
        _LOCAL.context, _LOCAL.pools = context, pools
        return summary(run_scf_kpoints(system, k_points, weights, compute_forces=True))

    with mock.patch("parsec_python.SCF.kpoints.make_kpoint_solver", solver):
        pooled = run_ranks(ranks, rank)
    return {"serial": serial, "pooled": pooled}


def check_guard(ranks: int, sector_pools: int, plain_pools: int) -> dict:
    """The retry of the symmetry guard of the spin k-point sectors on pools of another layout, against one process.

    Two O atoms of a periodic cube (the origin and the centre, spin-polarized, 3000 K) at the Gamma point and at X:
    the loop in the sectors fails the guard after the start Hamiltonian and runs again with one task per k-point.
    The sectors are solved on ``sector_pools`` pools of one rank, pool ``j`` holding the sectors of the k-points
    ``k`` with ``k % sector_pools == j`` (each rank then caches the Bloch projectors of its own k-points, as the
    pools of one rank of a 16-rank run that holds one sector each); the retry runs on ``plain_pools`` pools, whose
    ranks build the projectors of a k-point together (the atoms dealt over them) where one of them lacks them.
    """
    import shutil
    import warnings

    from parsec_python.acceleration.SCF import kpoint_pools, kpoint_sectors
    from parsec_python.acceleration.SCF.kpoint_pools import PoolPlan
    from parsec_python.acceleration.SCF.kpoint_sectors import sector_solver
    from parsec_python.acceleration.tests import thread_mpi
    from parsec_python.acceleration.tests.test_kpoint_sector_guard import _oxygen
    from parsec_python.models import SpinSectorFallbackRecord
    from parsec_python.SCF.kpoints import run_scf_kpoints_spin_polarized
    from parsec_python.SCF.pbc import prepare_periodic_single_point

    os.environ["PARSEC_KPOINT_SECTORS"] = "auto"
    os.environ["PARSEC_KPOINT_SECTOR_GUARD"] = "auto"
    kpoint_sectors._MINIMUM_ENTRIES = 0
    thread_mpi._TIMEOUT = 120.0  # ranks that disagree about a collective fail in minutes, not ten
    directory, problem = _oxygen("3000", body_centre=True, criterion="1.0e-5", max_iter=60)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = prepare_periodic_single_point(problem)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
    k_points = np.array([np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0])])
    weights = np.array([0.5, 0.5])

    def summary(found, records) -> dict:
        notes = [record for record in records if isinstance(record, SpinSectorFallbackRecord)]
        return dict(
            _summary(found),
            notes=[(note.stage, note.route) for note in notes],
            abandoned=float(found.timings.detail.get("kpoint_sector_abandoned", 0.0)),
        )

    serial_records: list = []
    serial = summary(
        run_scf_kpoints_spin_polarized(system, k_points, weights, callback=serial_records.append), serial_records
    )

    def by_kpoint(sizes, n_ranks, *arguments, **keywords):
        # the sectors of the k-points dealt by k-point over pools of one rank (planned on the root)
        items = _LOCAL.sectors._items
        assignment = tuple(
            tuple(position for position, (task, _) in enumerate(items) if task % len(k_points) % sector_pools == pool)
            for pool in range(sector_pools)
        )
        return PoolPlan(
            pools=sector_pools, sizes=(1,) * sector_pools, tasks=len(items), bytes_per_rank=0, balance=1.0,
            assignment=assignment,
        )

    def inner(job, **options):
        return MPIPoolKPointSolver(job, context=_LOCAL.context, pools=sector_pools, **options)

    def solver(job, **options):
        made = sector_solver(job, inner, **options)
        if made is None:
            return MPIPoolKPointSolver(job, context=_LOCAL.context, pools=plain_pools, **options)
        _LOCAL.sectors = made
        return made

    def rank(context):
        _LOCAL.context = context
        records: list = []
        found = run_scf_kpoints_spin_polarized(system, k_points, weights, callback=records.append)
        return summary(found, records)

    with mock.patch("parsec_python.SCF.kpoints.make_kpoint_solver", solver), mock.patch.object(
        kpoint_pools, "plan_pools_by_size", by_kpoint
    ):
        pooled = run_ranks(ranks, rank)
    return {"serial": serial, "pooled": pooled}


def check_projectors(ranks: int) -> dict:
    """The Bloch projectors built with the atoms dealt over a pool of ``ranks`` ranks against one build."""
    import scipy.sparse as sp
    from types import SimpleNamespace

    from parsec_python.SCF.kpoint_solver import KPointJob
    from parsec_python.tests.unit.test_hamiltonian_fast_paths import _PeriodicFixture
    from parsec_python.V_ion import build_nonlocal_projectors

    _PeriodicFixture.setUpClass()
    fx = _PeriodicFixture
    arguments = (fx.grid, fx.atoms, fx.pots, fx.specs, fx.cell.lattice_vectors)
    whole = build_nonlocal_projectors(*arguments, k_point=fx.k_point)

    def rank(context):
        solver = MPIPoolKPointSolver(
            KPointJob(1, 4, lambda *a: None, lambda *a: None, lambda *a: None), problem_size=fx.grid.size, context=context
        )
        solver._pool_comm = context.comm
        solver.group = ColumnGroup(context.comm, np)
        built = solver.projector_builder(*arguments, k_point=fx.k_point)
        difference = abs(sp.csc_matrix(built.projectors) - whole.projectors)
        return {
            "error": float(difference.max()) if difference.nnz else 0.0,
            "labels": list(built.labels) == list(whole.labels),
            "signs": bool(np.array_equal(built.signs, whole.signs)),
            # the decision of the pool to build: where the last rank lacks a k-point's projectors, and where none does
            "any": [solver.pool_any(context.rank == ranks - 1), solver.pool_any(False)],
        }

    return {"ranks": run_ranks(ranks, rank)}


def main() -> None:
    # "route=<rows|block>" anywhere: the PARSEC_KPOINT_RITZ of the run
    for argument in [a for a in sys.argv[1:] if a.startswith("route=")]:
        os.environ["PARSEC_KPOINT_RITZ"] = argument.split("=", 1)[1]
        sys.argv.remove(argument)
    # "grow": sectors without margin (check_sectors), so that they get more states and are solved again
    if "grow" in sys.argv[1:]:
        os.environ["POOL_CHECK_GROW"] = "1"
        sys.argv.remove("grow")
    # "assign=<auto|cyclic>": the PARSEC_KPOINT_POOL_ASSIGN of the run (tasks of different sizes)
    for argument in [a for a in sys.argv[1:] if a.startswith("assign=")]:
        os.environ["PARSEC_KPOINT_POOL_ASSIGN"] = argument.split("=", 1)[1]
        sys.argv.remove(argument)
    mode = sys.argv[1]
    if mode == "eigensolver":
        exchange = sys.argv[5] if len(sys.argv) > 5 else None
        if exchange is not None:
            os.environ["PARSEC_MPI_SHARED_EXCHANGE_BYTES"] = exchange
        payload = check_eigensolver(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif mode == "scf":
        payload = check_scf(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "sectors":
        payload = check_sectors(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "forces":
        payload = check_forces(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif mode == "guard":
        payload = check_guard(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "projectors":
        payload = check_projectors(int(sys.argv[2]))
    else:
        raise SystemExit(f"unknown mode {mode}")
    print("RESULT " + json.dumps(payload))


if __name__ == "__main__":
    main()
