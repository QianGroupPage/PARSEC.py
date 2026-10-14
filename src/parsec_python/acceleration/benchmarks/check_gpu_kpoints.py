#!/usr/bin/env python3
"""Check the GPU k-point / band path on a machine with CuPy and a CUDA device.

Runs on the GPU(s) and on the CPU and compares:

1. the device Bloch operator (scalar, and spinor with spin-orbit + Zeeman field)
   against the host operator, for every selected device;
2. the device Chebyshev filter against the CPU filter;
3. a short non-collinear spin-orbit SCF with 4 k-points: the
   chosen backend against the CPU (same energy, eigenvalues);
4. a short band path, GPU against CPU.

It prints the maximum differences, timings and a PASS/FAIL line per check, and
exits non-zero on a failure.  A small default cell is fast; raise ``--box`` (and
lower ``--spacing``) to time a realistic size, e.g. ``--box 14 --spacing 0.3``.

Usage:
    PYTHONPATH=src python -m parsec_python.acceleration.benchmarks.check_gpu_kpoints
        [--backend cupy] [--box 10] [--spacing 0.45] [--vectors 12] [--skip-scf]

``--backend numpy-device`` runs the same device code on the host (no GPU
needed), to check the script itself.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]

from parsec_python import (  # noqa: E402
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.acceleration.Hamiltonian.device_bloch import (  # noqa: E402
    CupyArrayBackend,
    DeviceBlochConverter,
    NumpyArrayBackend,
    resolve_kpoint_backend,
)
from parsec_python.Eigensolvers.chebyshev import chebyshev_filter  # noqa: E402
from parsec_python.models import BandSegment, BandStructureSettings  # noqa: E402
from parsec_python.SCF.bands import run_band_structure  # noqa: E402
from parsec_python.SCF.kpoints_soc import _k_point_operators, _spinor_hamiltonian  # noqa: E402
from parsec_python.SCF.noncollinear import run_self_consistent_noncollinear_kpoints  # noqa: E402
from parsec_python.SCF.pbc import prepare_periodic_single_point  # noqa: E402

AU_POTRE = REPO / "examples" / "0d_AuH" / "Au_POTRE.DAT"
failures: list[str] = []


def report(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def build_system(box: float, spacing: float, *, iterations: int = 40):
    problem = SinglePointInput(
        atoms=[
            Atom("Au", [0.35 * box, 0.5 * box, 0.5 * box], [0.0, 0.0, 1.0]),
            Atom("Au", [0.65 * box, 0.5 * box, 0.5 * box], [0.0, 0.0, 1.0]),
        ],
        pseudopotentials={
            "Au": SpeciesPotential(AU_POTRE, 0, spin_orbit=True, initial_spin_polarization=0.3)
        },
        grid=PeriodicGridSettings(spacing=spacing, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=box * np.eye(3)),
        scf=SCFSettings(
            max_iterations=iterations,
            number_of_states=36,
            convergence_criterion=2.0e-3,
            noncollinear=True,
            self_consistent_spin_orbit=True,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )
    return prepare_periodic_single_point(problem)


def make_converter(backend: str, device_id: int):
    if backend == "numpy-device":
        return DeviceBlochConverter(NumpyArrayBackend())
    return DeviceBlochConverter(CupyArrayBackend(device_id))


def check_operator_and_filter(system, backend: str, device_ids, vectors: int) -> None:
    rng = np.random.default_rng(0)
    n = system.grid.size
    k_point = np.array([0.21, -0.13, 0.08])
    nonlocal_k, soc_k = _k_point_operators(system, k_point[None, :])[0]
    potential = 0.3 * rng.normal(size=n)
    field = 0.2 * rng.normal(size=(3, n))
    host = _spinor_hamiltonian(system, k_point, nonlocal_k, soc_k, potential, xc_field=field)
    block = rng.normal(size=(2 * n, vectors)) + 1j * rng.normal(size=(2 * n, vectors))
    expected = host.apply(block)
    cpu_operator = host.as_linear_operator()
    arguments = dict(degree=15, lower_bound=-1.5, upper_bound=120.0, reference_eigenvalue=-1.5)

    start = time.perf_counter()
    cpu_filtered = chebyshev_filter(cpu_operator, block, **arguments)
    cpu_filter_seconds = time.perf_counter() - start
    start = time.perf_counter()
    host.apply(block)
    cpu_apply_seconds = time.perf_counter() - start

    for device_id in device_ids:
        converter = make_converter(backend, device_id)
        device = converter.operator(host)
        error = np.abs(device.matmat(block) - expected).max() / np.abs(expected).max()
        report(f"device {device_id}: spinor operator vs host", error < 1e-11, f"max rel error {error:.2e}")
        device_operator = device.as_linear_operator()
        chebyshev_filter(device_operator, block, **arguments)  # warm-up (kernel compile, allocator)
        device.backend.synchronize()
        start = time.perf_counter()
        filtered = chebyshev_filter(device_operator, block, **arguments)
        device.backend.synchronize()
        device_filter_seconds = time.perf_counter() - start
        scale = np.abs(cpu_filtered).max()
        error = np.abs(filtered - cpu_filtered).max() / scale
        report(
            f"device {device_id}: degree-15 Chebyshev filter vs CPU",
            error < 1e-9,
            f"max rel error {error:.2e}; CPU {cpu_filter_seconds:.2f} s, device "
            f"{device_filter_seconds:.2f} s ({cpu_filter_seconds / device_filter_seconds:.1f}x)",
        )
    print(f"        (one host apply of {vectors} spinors: {cpu_apply_seconds * 1e3:.0f} ms)")


def check_scf_and_bands(box: float, spacing: float, backend: str) -> None:
    k_points = 2.0 * np.pi / box * np.array(
        [[0.25, 0.25, 0.25], [-0.25, 0.25, 0.25], [0.25, -0.25, 0.25], [0.25, 0.25, -0.25]]
    )
    weights = np.full(4, 0.25)
    runs = {}
    for label, kpoint_backend in (("cpu", "cpu"), (backend, backend)):
        os.environ["PARSEC_KPOINT_BACKEND"] = kpoint_backend
        os.environ["PARSEC_KPOINT_WORKERS"] = "1" if kpoint_backend == "cpu" else "auto"
        system = build_system(box, spacing)
        start = time.perf_counter()
        result = run_self_consistent_noncollinear_kpoints(system, k_points, weights)
        seconds = time.perf_counter() - start
        runs[label] = (system, result, seconds)
    cpu = runs["cpu"][1]
    device = runs[backend][1]
    report(
        f"non-collinear SOC SCF, 4 k-points: {backend} vs CPU",
        abs(device.energies.total - cpu.energies.total) < 1e-6
        and np.abs(device.eigenvalues - cpu.eigenvalues).max() < 1e-5,
        f"energy difference {abs(device.energies.total - cpu.energies.total):.2e} Ry, "
        f"max eigenvalue difference {np.abs(device.eigenvalues - cpu.eigenvalues).max():.2e} Ry; "
        f"{cpu.iterations}/{device.iterations} iterations, CPU {runs['cpu'][2]:.1f} s, "
        f"{backend} {runs[backend][2]:.1f} s",
    )
    # bands along a short path from the CPU potentials
    system = runs["cpu"][0]
    potentials = dict(cpu.scf_potentials)
    path = BandStructureSettings(
        segments=(BandSegment((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), "G-X"),), points=3
    )
    bands = {}
    for label, kpoint_backend in (("cpu", "cpu"), (backend, backend)):
        start = time.perf_counter()
        bands[label] = (
            run_band_structure(
                system,
                path,
                mode="noncollinear",
                potentials=potentials,
                fermi_level=cpu.fermi_level,
                number_of_states=24,
                backend=kpoint_backend,
                workers=1 if kpoint_backend == "cpu" else None,
            ),
            time.perf_counter() - start,
        )
    difference = np.abs(bands[backend][0].eigenvalues - bands["cpu"][0].eigenvalues).max()
    report(
        f"bands along G-X: {backend} vs CPU",
        difference < 1e-5,
        f"max eigenvalue difference {difference:.2e} Ry; CPU {bands['cpu'][1]:.1f} s, "
        f"{backend} {bands[backend][1]:.1f} s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", default="cupy", choices=("cupy", "numpy-device"))
    parser.add_argument("--box", type=float, default=10.0, help="cubic cell edge [bohr]")
    parser.add_argument("--spacing", type=float, default=0.45, help="grid spacing [bohr]")
    parser.add_argument("--vectors", type=int, default=12, help="block size of the operator/filter checks")
    parser.add_argument("--skip-scf", action="store_true", help="only the operator and filter checks")
    arguments = parser.parse_args()

    if arguments.backend == "cupy":
        plan = resolve_kpoint_backend("cupy")  # raises with a clear message without CuPy/CUDA
        device_ids = plan.device_ids
        print(f"CuPy devices in use: {list(device_ids)}")
    else:
        device_ids = (0,)
    system = build_system(arguments.box, arguments.spacing)
    print(f"cell {arguments.box} bohr, {system.grid.size} grid points ({system.grid.shape})")
    check_operator_and_filter(system, arguments.backend, device_ids, arguments.vectors)
    if not arguments.skip_scf:
        check_scf_and_bands(arguments.box, arguments.spacing, arguments.backend)
    print("ALL CHECKS PASSED" if not failures else f"FAILED: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
