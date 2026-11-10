"""Accuracy of the Kronecker-product Poisson solver against Gaussian charges (NumPy).

    python -m parsec_python.acceleration.benchmarks.kronecker_accuracy convergence
    python -m parsec_python.acceleration.benchmarks.kronecker_accuracy screening [--radius 12]
    python -m parsec_python.acceleration.benchmarks.kronecker_accuracy smearing
    python -m parsec_python.acceleration.benchmarks.kronecker_accuracy elongated

``smearing`` scans the cap ``beta h**2 <= c`` of the split screening against
the multipole order on small spheres; ``elongated`` compares unpadded and
automatically padded boxes on elongated domains (and, with CuPy, the device
time per density against the direct boundary sum alone).

Two densities with closed-form potentials for the three kernels: a unit
Gaussian charge ``g_1.2`` at ``(0.5, -0.3, 0.2)`` and a neutral pair
``g_1.0(r - c1) - g_1.0(r - c2)`` with a dipole (``c1 = (0.8, 0, 0)``,
``c2 = (-0.8, 0.3, 0)``).  ``1/r -> erf(sqrt(a) s)/s``,
``erf(w r)/r -> erf(sqrt(c) s)/s`` with ``1/c = 1/a + 1/w**2``,
``erfc = 1/r - erf`` (Hartree; the solver gives Rydberg).

``convergence`` prints the relative error of ``h**3 sum rho v`` (the pair
integral of exact exchange) and the largest pointwise error against ``h`` and
the finite-difference order.  ``screening`` compares the split treatment of
the screened kernels with the spectral function on padded boxes: errors, box
size and the time per density (NumPy, batch of the two densities).
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
from scipy.special import erf

from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree.kronecker import KroneckerPoisson
from parsec_python.models import GridSettings

CHARGE = ((1.0, (0.5, -0.3, 0.2), 1.2),)
DIPOLE = ((1.0, (0.8, 0.0, 0.0), 1.0), (-1.0, (-0.8, 0.3, 0.0), 1.0))


def _gaussian(points, center, exponent):
    offset = points - np.asarray(center)
    return (exponent / np.pi) ** 1.5 * np.exp(-exponent * np.einsum("ij,ij->i", offset, offset))


def _potential(points, center, exponent, kernel, omega):
    distance = np.linalg.norm(points - np.asarray(center), axis=1)
    coulomb = erf(np.sqrt(exponent) * distance) / distance
    if kernel == "coulomb":
        return 2.0 * coulomb
    combined = exponent * omega**2 / (exponent + omega**2)
    long_range = erf(np.sqrt(combined) * distance) / distance
    return 2.0 * (long_range if kernel == "erf" else coulomb - long_range)


def cases(points, kernel="coulomb", omega=None):
    densities, potentials = [], []
    for terms in (CHARGE, DIPOLE):
        densities.append(sum(w * _gaussian(points, c, a) for w, c, a in terms))
        potentials.append(sum(w * _potential(points, c, a, kernel, omega) for w, c, a in terms))
    return np.column_stack(densities), np.column_stack(potentials)


def errors(grid, solver, kernel, omega):
    density, exact = cases(grid.coordinates, kernel, omega)
    started = perf_counter()
    potential = solver.solve_batch(density)
    seconds = (perf_counter() - started) / density.shape[1]
    energy = np.einsum("ij,ij->j", density, potential)
    exact_energy = np.einsum("ij,ij->j", density, exact)
    return (
        np.abs(energy / exact_energy - 1.0),
        np.abs(potential - exact).max(axis=0),
        seconds,
    )


def convergence(arguments) -> None:
    print("kernel w h order box rel_err_E(charge) rel_err_E(dipole) max_err(charge) max_err(dipole)")
    for kernel, omega in (("coulomb", None), ("erfc", 0.11), ("erf", 0.11), ("erf", 0.4)):
        for spacing in arguments.spacings:
            for order in arguments.orders:
                grid = build_cluster_grid(
                    GridSettings(spacing=spacing, radius=arguments.radius, expansion_order=order)
                )
                solver = KroneckerPoisson(grid, kernel=kernel, omega=omega)
                energy, pointwise, _ = errors(grid, solver, kernel, omega)
                print(
                    f"{kernel} {omega or '-'} {spacing} {order} {solver.box_shape[0]} "
                    f"{energy[0]:.2e} {energy[1]:.2e} {pointwise[0]:.2e} {pointwise[1]:.2e}",
                    flush=True,
                )


def screening(arguments) -> None:
    grid = build_cluster_grid(
        GridSettings(spacing=arguments.spacing, radius=arguments.radius, expansion_order=arguments.order)
    )
    coulomb = KroneckerPoisson(grid)
    energy, pointwise, seconds = errors(grid, coulomb, "coulomb", None)
    print(f"sphere R={arguments.radius} h={arguments.spacing} order {arguments.order}")
    print("kernel w method padding box setup_s s_per_density rel_err_E(charge) rel_err_E(dipole) "
          "max_err(charge) max_err(dipole)")
    print(f"coulomb - - 0 {coulomb.box_shape[0]} - {seconds:.3f} {energy[0]:.1e} {energy[1]:.1e} "
          f"{pointwise[0]:.1e} {pointwise[1]:.1e}", flush=True)
    for omega in arguments.omegas:
        for kernel in ("erfc", "erf"):
            runs = [("split", 0.0)] + [("spectral", p) for p in arguments.paddings]
            for method, padding in runs:
                started = perf_counter()
                solver = KroneckerPoisson(
                    grid, kernel=kernel, omega=omega, screening=method, padding=padding
                )
                setup = perf_counter() - started
                energy, pointwise, seconds = errors(grid, solver, kernel, omega)
                print(
                    f"{kernel} {omega} {method} {padding:g} {solver.box_shape[0]} {setup:.1f} "
                    f"{seconds:.3f} {energy[0]:.1e} {energy[1]:.1e} {pointwise[0]:.1e} {pointwise[1]:.1e}",
                    flush=True,
                )
                del solver


def _terms_density(points, terms):
    return sum(w * _gaussian(points, c, a) for w, c, a in terms)


def _terms_potential(points, terms, kernel, omega):
    return sum(w * _potential(points, c, a, kernel, omega) for w, c, a in terms)


def smearing(arguments) -> None:
    """Error of the split erf/erfc potentials against the smearing cap ``beta h**2 <= c`` and ``L``."""

    terms = ((1.0, (1.5, -1.0, 0.8), 1.5), (-1.0, (-1.2, 1.3, -0.9), 1.5))
    for radius, spacing in ((5.0, 0.5), (5.0, 0.35), (8.0, 0.5)):
        grid = build_cluster_grid(GridSettings(spacing=spacing, radius=radius, expansion_order=8))
        points = grid.coordinates
        density = _terms_density(points, terms)
        for kernel, omega in (("erf", 0.4), ("erfc", 0.4)):
            exact = _terms_potential(points, terms, kernel, omega)
            print(f"sphere R={radius} h={spacing} order 8, {kernel} w={omega}: max |v - exact| (Ry) "
                  "[box, beta h^2, s per density]", flush=True)
            for order in arguments.multipole_orders:
                row = []
                for limit in arguments.limits:
                    solver = KroneckerPoisson(
                        grid, kernel=kernel, omega=omega, multipole_order=order, smearing_limit=limit
                    )
                    started = perf_counter()
                    potential = solver.solve_batch(density[:, None])[:, 0]
                    seconds = perf_counter() - started
                    row.append(
                        f"c={limit:g}: {np.abs(potential - exact).max():.1e} "
                        f"[{solver.box_shape[0]}, {solver.smearing_exponent * spacing**2:.3f}, {seconds:.2f}]"
                    )
                    del solver
                print(f"  L={order:2d} | " + " | ".join(row), flush=True)


def elongated(arguments) -> None:
    """Elongated domains and a cube: unpadded against automatically padded boxes (errors, sizes, times)."""

    cases = (
        ("box 26x10x10", GridSettings(spacing=0.5, radius=13.0, expansion_order=6, domain_shape="box",
                                      box_lengths=(26.0, 10.0, 10.0)), ((1.0, (-8.0, 0.0, 0.0), 1.0), (1.0, (8.0, 0.0, 0.0), 1.0))),
        ("cylinder r=5 L=26", GridSettings(spacing=0.5, radius=13.0, expansion_order=6, domain_shape="cylinder",
                                           domain_parameters=(5.0, 26.0, 1)), ((1.0, (-8.0, 0.0, 0.0), 1.0), (1.0, (8.0, 0.0, 0.0), 1.0))),
        ("box 24x8x8", GridSettings(spacing=0.5, radius=12.0, expansion_order=6, domain_shape="box",
                                    box_lengths=(24.0, 8.0, 8.0)), ((1.0, (-4.5, 0.0, 0.0), 0.8), (1.0, (4.5, 0.0, 0.0), 0.8))),
        ("cube 14^3", GridSettings(spacing=0.5, radius=7.0, expansion_order=6, domain_shape="box",
                                   box_lengths=(14.0, 14.0, 14.0)), ((1.0, (-3.0, 0.0, 0.0), 1.0), (1.0, (3.0, 0.0, 0.0), 1.0))),
    )
    for label, settings, terms in cases:
        grid = build_cluster_grid(settings)
        points = grid.coordinates
        density = _terms_density(points, terms)
        volume = grid.volume_element
        print(f"{label}: {grid.size} points", flush=True)
        for kernel, omega in (("coulomb", None), ("erfc", 0.11), ("erf", 0.4)):
            exact = _terms_potential(points, terms, kernel, omega)
            for order in (5, 9, 15):
                parts = []
                for automatic in (False, True):
                    solver = KroneckerPoisson(
                        grid, kernel=kernel, omega=omega, multipole_order=order, automatic_padding=automatic
                    )
                    started = perf_counter()
                    potential = solver.solve_batch(np.column_stack([density] * 4))[:, 0]
                    seconds = (perf_counter() - started) / 4
                    parts.append(
                        f"{'padded' if automatic else 'unpadded'} box {'x'.join(map(str, solver.box_shape))}: "
                        f"max err {np.abs(potential - exact).max():.1e}, dE {volume * density @ (potential - exact):+.1e}, "
                        f"{1e3 * seconds:.1f} ms"
                    )
                    del solver
                print(f"  {kernel:7s} L={order:2d} | " + " | ".join(parts), flush=True)
        _device_costs(grid, density)


def _device_costs(grid, density) -> None:
    """Device time per density: padded Kronecker (Coulomb, erfc) against the direct boundary sum alone."""

    try:
        import cupy as cp

        from parsec_python.acceleration.Hartree.cupy_direct_boundary import CuPyDirectCoulombBoundary
        from parsec_python.acceleration.Hartree.kronecker_device import DeviceKroneckerPoisson
    except Exception:
        return
    batch = 16
    block = cp.asarray(np.column_stack([density] * batch))
    line = []
    for kernel, omega in (("coulomb", None), ("erfc", 0.11)):
        for automatic in (False, True):
            solver = DeviceKroneckerPoisson(
                KroneckerPoisson(grid, kernel=kernel, omega=omega, automatic_padding=automatic)
            )
            solver.solve_batch(block)
            cp.cuda.Device().synchronize()
            started = perf_counter()
            solver.solve_batch(block)
            cp.cuda.Device().synchronize()
            line.append(f"{kernel} {'padded' if automatic else 'unpadded'} {1e3 * (perf_counter() - started) / batch:.3f}")
            del solver
    direct = CuPyDirectCoulombBoundary(grid)
    weights = density * grid.volume_element
    direct.boundary_values(weights)
    started = perf_counter()
    for _ in range(4):
        direct.boundary_values(weights)
    line.append(f"direct boundary sum alone ({direct.exterior_point_count} points) {1e3 * (perf_counter() - started) / 4:.3f}")
    cp.get_default_memory_pool().free_all_blocks()
    print("  device ms per density: " + ", ".join(line), flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("table", choices=("convergence", "screening", "smearing", "elongated"))
    parser.add_argument("--multipole-orders", type=int, nargs="+", default=[9, 16, 20, 25, 30])
    parser.add_argument("--limits", type=float, nargs="+", default=[1e9, 1.0, 0.5, 0.25, 0.15])
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--spacing", type=float, default=0.3)
    parser.add_argument("--order", type=int, default=12)
    parser.add_argument("--spacings", type=float, nargs="+", default=[0.5, 0.4, 0.3, 0.25])
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--omegas", type=float, nargs="+", default=[0.11, 0.2, 0.4])
    parser.add_argument("--paddings", type=float, nargs="+", default=[0.0, 4.0, 8.0, 12.0])
    arguments = parser.parse_args(argv)
    tables = {"convergence": convergence, "screening": screening, "smearing": smearing, "elongated": elongated}
    tables[arguments.table](arguments)


if __name__ == "__main__":
    main()
