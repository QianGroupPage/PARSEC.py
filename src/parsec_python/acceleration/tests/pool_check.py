"""Checks of the k-point pools with ranks as threads (see test_kpoint_pools.py); prints ``RESULT {json}``.

    python -m parsec_python.acceleration.tests.pool_check eigensolver METHOD SIZES STEPS [EXCHANGE_BYTES]
    python -m parsec_python.acceleration.tests.pool_check scf DRIVER RANKS POOLS

Run it with ``OPENBLAS_NUM_THREADS=1`` (test_kpoint_pools.py does): the package imports NumPy before
this module, so the settings below come too late for the BLAS of a ``-m`` run (:mod:`.thread_mpi`).  The
device code runs on the host (``numpy-device``).  ``eigensolver`` compares the shared-column eigensolver on each number of
ranks of ``SIZES`` (``1,2,3``) with the one-device solver over ``STEPS`` solves; ``scf`` runs the k-point
driver ``DRIVER`` of ``test_kpoint_parallel.DRIVERS`` on ``RANKS`` ranks in ``POOLS`` pools and in one
process.
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
        }

    return {"ranks": run_ranks(ranks, rank)}


def main() -> None:
    mode = sys.argv[1]
    if mode == "eigensolver":
        exchange = sys.argv[5] if len(sys.argv) > 5 else None
        if exchange is not None:
            os.environ["PARSEC_MPI_SHARED_EXCHANGE_BYTES"] = exchange
        payload = check_eigensolver(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif mode == "scf":
        payload = check_scf(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "projectors":
        payload = check_projectors(int(sys.argv[2]))
    else:
        raise SystemExit(f"unknown mode {mode}")
    print("RESULT " + json.dumps(payload))


if __name__ == "__main__":
    main()
