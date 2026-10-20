"""Run under ``mpirun -n N``: checks of the row-distributed solver (see test_kpoint_mpi.py).

    mpirun -n 3 python mpi_check.py operators   # distributed H applied to a block == the serial one
    mpirun -n 3 python mpi_check.py scf NAME     # an SCF driver with the distributed solver; rank 0 prints JSON

The device code runs on the host (``PARSEC_KPOINT_BACKEND=numpy-device``).
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("PARSEC_MPI", "1")
os.environ.setdefault("PARSEC_KPOINT_BACKEND", "numpy-device")

import numpy as np

from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter, NumpyArrayBackend
from parsec_python.acceleration.Hamiltonian.distributed_bloch import DistributedBlochConverter
from parsec_python.acceleration.Parallel import mpi
from parsec_python.acceleration.tests import test_kpoint_parallel as fixtures
from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.Hamiltonian.kpoint_spinor_operator import KPointSpinorKohnShamHamiltonian
from parsec_python.Hamiltonian.spinor_operator import SpinorKohnShamHamiltonian
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.tests.unit.test_hamiltonian_fast_paths import _PeriodicFixture
from parsec_python.V_ion import build_nonlocal_projectors


def check_operators(context) -> None:
    _PeriodicFixture.setUpClass()
    fx = _PeriodicFixture
    rng = np.random.default_rng(11)
    n = fx.grid.size
    field = 0.2 * rng.normal(size=(3, n))
    delta = 0.1 * rng.normal(size=n)
    scalar = KPointKohnShamHamiltonian(fx.laplacian, fx.gradient, fx.potential, fx.nonlocal_k, fx.k_point)
    gamma = build_nonlocal_projectors(fx.grid, fx.atoms, fx.pots, fx.specs, fx.cell.lattice_vectors)
    cases = {
        "scalar k-point": (scalar, 1),
        "spinor field + soc": (KPointSpinorKohnShamHamiltonian(scalar, fx.soc, xc_field=field), 2),
        "spinor delta + soc": (KPointSpinorKohnShamHamiltonian(scalar, fx.soc, xc_delta=delta), 2),
        "isolated non-collinear": (SpinorKohnShamHamiltonian(fx.laplacian, fx.potential, gamma, (), xc_field=field), 2),
    }
    serial = DeviceBlochConverter(NumpyArrayBackend())
    distributed = DistributedBlochConverter(NumpyArrayBackend(), context)
    for name, (host, components) in cases.items():
        block = np.random.default_rng(5).normal(size=(components * n, 6)) + 1j * np.random.default_rng(6).normal(
            size=(components * n, 6)
        )
        expected = serial.operator(host).apply(np.ascontiguousarray(block))
        operator = distributed.operator(host)
        start, stop = operator.row_bounds
        local = np.concatenate([block[c * n + start : c * n + stop] for c in range(components)])
        result = operator.apply(np.ascontiguousarray(local))
        pieces = context.comm.gather((start, stop, result), root=0)
        if context.is_root:
            assembled = np.empty_like(expected)
            for lo, hi, piece in pieces:
                rows = hi - lo
                for c in range(components):
                    assembled[c * n + lo : c * n + hi] = piece[c * rows : (c + 1) * rows]
            error = float(np.max(np.abs(assembled - expected)))
            if error > 1e-10:
                raise SystemExit(f"{name}: distributed apply differs from the serial one by {error:.2e}")
            print(f"ok {name} ({context.size} ranks, halo {operator.halo is not None}) max error {error:.1e}")
    context.barrier()


def run_scf(context, name: str, criterion: float) -> None:
    driver, flags, soc = fixtures.DRIVERS[name]
    system = fixtures._system(soc=soc, criterion=criterion, **flags)
    result = driver(system, fixtures._K, fixtures._WEIGHTS)
    if context.is_root:
        eigenvalues = getattr(result, "eigenvalues", None)
        payload = {
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "energy": float(result.energies.total),
            "fermi": float(result.fermi_level),
            "ranks": context.size,
            "eigenvalues": None if eigenvalues is None else [float(v) for v in np.asarray(eigenvalues)[:12]],
            "workers": float(result.timings.detail["workers"]) if getattr(result.timings, "detail", None) else None,
        }
        print("RESULT " + json.dumps(payload))
    context.barrier()


def check_map(context) -> None:
    """Independent tasks (band-path points) are dealt out to the ranks and every rank gets every result."""
    from parsec_python.acceleration.SCF.kpoint_parallel import map_tasks

    seen = []

    def solve_task(task, converter):
        seen.append(task)
        return {"task": task, "rank": context.rank, "values": np.arange(task + 1.0)}

    results = dict(map_tasks(solve_task, 7))
    if sorted(results) != list(range(7)):
        raise SystemExit(f"rank {context.rank}: tasks {sorted(results)}")
    for task, result in results.items():
        if result["rank"] != task % context.size or result["values"].size != task + 1:
            raise SystemExit(f"rank {context.rank}: wrong result for task {task}: {result}")
    if seen != list(range(context.rank, 7, context.size)):
        raise SystemExit(f"rank {context.rank} solved {seen}")
    gathered = context.comm.gather(seen, root=0)
    if context.is_root:
        print(f"ok map_tasks {sorted(t for part in gathered for t in part)}")
    context.barrier()


def main() -> None:
    context = mpi.context()
    mode = sys.argv[1]
    if mode == "map":
        check_map(context)
        return
    if mode == "operators":
        check_operators(context)
    elif mode == "scf":
        run_scf(context, sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.0e-5)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
