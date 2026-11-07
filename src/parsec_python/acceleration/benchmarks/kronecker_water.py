"""Water pair densities: the Kronecker solver against the conjugate-gradient pair solves.

    python -m parsec_python.acceleration.benchmarks.kronecker_water <work folder> [--domain sphere|box]

The water of ``examples/0d_water_geoopt`` (sphere of 12 bohr, h = 0.3, order
12) without its relaxation is converged with the accelerated driver; its
four occupied orbitals give ten pair densities ``psi_i psi_j``.  With
``--domain box`` the same orbitals are carried to a box domain of
21 x 20 x 19 bohr on the same lattice (zero at the corners outside the
sphere, renormalized): a self-consistent run on that box takes hours, since
its Hartree boundary is the host direct sum.  Printed, against a proxy of the
infinite-grid finite-difference potential (Kronecker solve on a box padded by
6 bohr with multipoles to order 20):

* the largest potential difference and the largest relative difference of the
  pair integrals ``(ij|ij)`` for the Kronecker solve (multipole order of the
  run, 4 and 20) and for ``reference_pair_poisson`` (sphere: multipoles of the
  run's order and of order 4, direct boundary sum; box: the direct sum of the
  run), with the exchange energies;
* ``exact_exchange_energy`` with both solvers and the time of each (one pair
  at a time and, for the Kronecker solver, all pairs in one batch);
* the screened exchange energies (erfc, erf at w = 0.11, 0.2, 0.4) by the
  split treatment, the split treatment on the padded box with order 20, and
  the spectral function on a box padded by 12 bohr (an independent boundary
  treatment);
* device times of the ten pairs: Kronecker (CuPy) against the prepared device
  conjugate gradients with host multipoles.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "0d_water_geoopt"


def device_direct_solver(grid, laplacian, settings):
    """``reference_pair_poisson`` with the direct boundary, its sums and CG on the device.

    The same discrete problem (boundary values ``2 sum rho h^3/|R - r|`` at the
    exterior stencil points, CG to the tolerances of ``settings`` from zero);
    the host NumPy sums take minutes per pair on a 270 000-point grid.
    Returns ``None`` without CuPy.
    """

    try:
        from parsec_python.acceleration.Hartree.cupy_direct_boundary import CuPyDirectCoulombBoundary
        from parsec_python.acceleration.Hartree.cupy_prepared import CuPyPreparedConjugateGradientBackend
    except Exception:
        return None
    try:
        boundary = CuPyDirectCoulombBoundary(grid)
        backend = CuPyPreparedConjugateGradientBackend(laplacian)
    except Exception:
        return None

    def solve(density):
        rhs = boundary.build(density, settings)[0]
        payload = backend.solve(
            rhs, np.zeros(grid.size), relative_tolerance=settings.relative_tolerance,
            absolute_tolerance=settings.absolute_tolerance, max_iterations=settings.max_iterations,
        )
        if not payload["converged"]:
            raise RuntimeError("device CG of a pair density did not converge")
        return np.asarray(payload["solution"], dtype=np.float64)

    return solve


BOX_LENGTHS = (21.0, 20.0, 19.0)


def to_box(grid, wavefunctions):
    """The box grid of :data:`BOX_LENGTHS` on the lattice of ``grid`` and the orbitals on it."""

    from parsec_python.Grid import build_cluster_grid

    settings = replace(grid.settings, domain_shape="box", box_lengths=BOX_LENGTHS)
    box = build_cluster_grid(settings)
    rows = grid.rows_for_integer_coordinates(box.integer_coordinates)
    moved = np.where((rows >= 0)[:, None], wavefunctions[np.maximum(rows, 0)], 0.0)
    norms = np.linalg.norm(moved, axis=0)
    moved /= np.where(norms > 0.0, norms, 1.0)
    return box, moved


def write_input(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("H_POTRE.DAT", "O_POTRE.DAT"):
        shutil.copy(EXAMPLE / name, folder / name)
    lines = [
        line
        for line in (EXAMPLE / "parsec.in").read_text().splitlines()
        if not line.startswith(("Minimization", "Force_Min"))
    ]
    (folder / "parsec.in").write_text("\n".join(lines) + "\n")
    return folder / "parsec.in"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--domain", choices=("sphere", "box"), default="sphere")
    parser.add_argument("--no-direct", action="store_true", help="skip the direct-boundary CG solves")
    arguments = parser.parse_args(argv)

    from parsec_python.acceleration.driver import prepare_single_point, run_scf
    from parsec_python.acceleration.SCF.single_point import materialize_wavefunctions
    from parsec_python.Hartree.kronecker import KroneckerPoisson
    from parsec_python.Input import parse_parsec_input
    from parsec_python.Laplacian import build_negative_laplacian
    from parsec_python.V_xc.exact_exchange import (
        exact_exchange_energy,
        occupied_columns,
        reference_pair_poisson,
    )

    def say(text):
        print(text, flush=True)

    path = write_input(arguments.folder / arguments.domain)
    problem = parse_parsec_input(path).problem
    started = perf_counter()
    system = prepare_single_point(problem)
    result = run_scf(system)
    materialize_wavefunctions(system, result)
    reference = getattr(system, "reference", system)
    grid = reference.grid
    psi = np.asarray(result.wavefunctions, dtype=np.float64)
    occupations = np.asarray(result.occupations, dtype=np.float64)
    hartree = reference.input.hartree
    order = hartree.multipole_order
    if arguments.domain == "box":
        grid, psi = to_box(grid, psi)
    say(f"{arguments.domain}: SCF {perf_counter() - started:.1f} s, total energy "
        f"{result.energies.total:.8f} Ry, grid {grid.size} points, multipole order {order}")

    columns = occupied_columns(occupations)
    volume = grid.volume_element
    orbitals = psi[:, columns]
    pairs = [(i, j) for i in range(len(columns)) for j in range(i + 1)]
    densities = np.column_stack([orbitals[:, i] * orbitals[:, j] / volume for i, j in pairs])
    weights = np.array(
        [(1.0 if i == j else 2.0) * occupations[columns[i]] * occupations[columns[j]] for i, j in pairs]
    )

    def integrals(potentials):
        return volume * np.einsum("ij,ij->j", densities, potentials)

    def energy(potentials):
        return -float(np.sum(weights * integrals(potentials)))

    started = perf_counter()
    truth = KroneckerPoisson(grid, multipole_order=20, padding=6.0)
    exact = truth.solve_batch(densities)
    del truth
    exact_integrals = integrals(exact)
    exact_energy = energy(exact)
    say(f"proxy: Kronecker, 6 bohr padding, order 20 ({perf_counter() - started:.1f} s): E_x {exact_energy:.10f} Ry")
    say("method | max |v - proxy| (Ry) | max rel. diff (ij|ij) | E_x (Ry) | E_x - proxy (Ry) | seconds (10 pairs)")

    def row(label, potentials, seconds):
        say(f"{label} | {np.abs(potentials - exact).max():.2e} | "
            f"{np.abs(integrals(potentials) / exact_integrals - 1.0).max():.2e} | {energy(potentials):.10f} | "
            f"{energy(potentials) - exact_energy:+.2e} | {seconds:.2f}")

    for multipole_order in sorted({order, 4, 20}):
        solver = KroneckerPoisson(grid, multipole_order=multipole_order)
        started = perf_counter()
        potentials = solver.solve_batch(densities)
        row(f"Kronecker L={multipole_order}", potentials, perf_counter() - started)
        del solver
    laplacian = build_negative_laplacian(grid)
    cases = []
    if grid.settings.domain_shape == "sphere":
        cases += [(f"CG multipole L={order}", replace(hartree, multipole_order=order, boundary_method="multipole")),
                  ("CG multipole L=4", replace(hartree, multipole_order=4, boundary_method="multipole"))]
        if not arguments.no_direct:
            cases.append(("CG direct sum", replace(hartree, boundary_method="direct")))
    elif not arguments.no_direct:
        cases.append(("CG direct sum (the run's)", hartree))
    for label, settings in cases:
        solve = reference_pair_poisson(grid, laplacian, settings)
        if settings.boundary_method == "direct" or (
            settings.boundary_method == "auto" and grid.settings.domain_shape != "sphere"
        ):
            fast = device_direct_solver(grid, laplacian, replace(settings, atomic_tail="off", boundary_tolerance=None))
            if fast is not None:
                solve = fast
                label += " (device sums and CG)"
        started = perf_counter()
        potentials = np.column_stack(
            [np.asarray(getattr(value, "potential", value)) for value in (solve(densities[:, k]) for k in range(densities.shape[1]))]
        )
        row(label, potentials, perf_counter() - started)

    cg = reference_pair_poisson(grid, laplacian, hartree)
    cg_label = "CG"
    if grid.settings.domain_shape != "sphere":
        fast = device_direct_solver(grid, laplacian, replace(hartree, atomic_tail="off", boundary_tolerance=None))
        if fast is not None:
            cg, cg_label = fast, "CG (direct boundary: device sums and CG)"
    started = perf_counter()
    by_cg = exact_exchange_energy(psi, occupations, volume, cg)
    cg_seconds = perf_counter() - started
    started = perf_counter()
    kronecker = KroneckerPoisson(grid, multipole_order=order)
    setup = perf_counter() - started
    started = perf_counter()
    by_kronecker = exact_exchange_energy(psi, occupations, volume, kronecker)
    one_by_one = perf_counter() - started
    started = perf_counter()
    batched = energy(kronecker.solve_batch(densities))
    batch_seconds = perf_counter() - started
    say(f"exact_exchange_energy: {cg_label} {by_cg.energy:.10f} Ry ({by_cg.poisson_iterations} CG iterations, "
        f"{cg_seconds:.2f} s); Kronecker {by_kronecker.energy:.10f} Ry ({one_by_one:.2f} s one by one, "
        f"batched {batched:.10f} in {batch_seconds:.2f} s, setup {setup:.2f} s); difference "
        f"{by_kronecker.energy - by_cg.energy:+.3e} Ry; speed-up {cg_seconds / one_by_one:.0f}x one by one, "
        f"{cg_seconds / batch_seconds:.0f}x batched")
    del kronecker

    for omega in (0.11, 0.2, 0.4):
        for kernel in ("erfc", "erf"):
            parts = [f"{kernel} w={omega}"]
            for label, keywords in (
                ("split", {"multipole_order": order}),
                ("split padded L=20", {"multipole_order": 20, "padding": 6.0}),
                ("spectral padded 12", {"multipole_order": order, "screening": "spectral", "padding": 12.0}),
            ):
                started = perf_counter()
                solver = KroneckerPoisson(grid, kernel=kernel, omega=omega, **keywords)
                built = perf_counter()
                value = energy(solver.solve_batch(densities))
                parts.append(f"{label} {value:.10f} ({built - started:.1f} + {perf_counter() - built:.2f} s)")
                del solver
            say(" | ".join(parts))

    try:
        import cupy as cp

        from parsec_python.acceleration.Hartree.cupy_prepared import CuPyPreparedConjugateGradientBackend
        from parsec_python.acceleration.Hartree.kronecker_device import DeviceKroneckerPoisson
        from parsec_python.acceleration.Hartree.poisson import build_hartree_problem
    except Exception as error:
        say(f"no device timing: {error}")
        return
    for kernel, omega in (("coulomb", None), ("erfc", 0.11)):
        device = DeviceKroneckerPoisson(grid, kernel=kernel, omega=omega, multipole_order=order)
        on_device = cp.asarray(densities)
        device.solve_batch(on_device)
        cp.cuda.Device().synchronize()
        started = perf_counter()
        potentials = device.solve_batch(on_device)
        cp.cuda.Device().synchronize()
        say(f"device Kronecker {kernel}: 10 pairs in {1e3 * (perf_counter() - started):.2f} ms, "
            f"E_x {energy(cp.asnumpy(potentials)):.10f} Ry")
        del device, on_device, potentials
        cp.get_default_memory_pool().free_all_blocks()
    backend = CuPyPreparedConjugateGradientBackend(laplacian)
    pair_settings = replace(hartree, atomic_tail="off", boundary_tolerance=None)
    if grid.settings.domain_shape != "sphere":
        say("device CG skipped on a box (its host direct-sum boundary dominates)")
        return
    rhs_seconds = solve_seconds = 0.0
    potentials = []
    for column in range(densities.shape[1]):
        started = perf_counter()
        rhs = build_hartree_problem(densities[:, column], grid, pair_settings, None)[0]
        rhs_seconds += perf_counter() - started
        started = perf_counter()
        payload = backend.solve(rhs, np.zeros(grid.size), relative_tolerance=hartree.relative_tolerance,
                                absolute_tolerance=hartree.absolute_tolerance,
                                max_iterations=hartree.max_iterations)
        cp.cuda.Device().synchronize()
        solve_seconds += perf_counter() - started
        potentials.append(np.asarray(payload["solution"]))
    say(f"device CG: 10 pairs in {1e3 * solve_seconds:.1f} ms on the device + {1e3 * rhs_seconds:.1f} ms host "
        f"right-hand sides, E_x {energy(np.column_stack(potentials)):.10f} Ry")


if __name__ == "__main__":
    sys.exit(main())
