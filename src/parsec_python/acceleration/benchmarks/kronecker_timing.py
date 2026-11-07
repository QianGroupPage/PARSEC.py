"""Time the CuPy Kronecker-product Poisson solver against the device CG pair solve.

    python -m parsec_python.acceleration.benchmarks.kronecker_timing \
        --sizes 40 60 80 --kernels coulomb erfc --spacing 0.3 --batch 32 [--cg] [--order 12]

For every box size ``n`` a sphere of radius ``n h / 2`` is built (its bounding
box has about ``n`` points per axis).  Pair-like densities (products of two
displaced Gaussian orbitals of s and p character) are solved in one batch,
after one warm-up batch; the time per density is the best of ``--repeat``
batches, measured with device synchronization.  ``--cg`` adds the device
conjugate-gradient solve of the existing pair solver
(:class:`~parsec_python.acceleration.Hartree.cupy_prepared.CuPyPreparedConjugateGradientBackend`
on the run's Laplacian, multipole right-hand side on the host, zero start,
the default Hartree tolerances) for a few densities of the same batch.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree.kronecker import KroneckerPoisson
from parsec_python.models import GridSettings, HartreeSettings


def pair_densities(grid, count, seed=0):
    rng = np.random.default_rng(seed)
    points = grid.coordinates
    columns = []
    for _ in range(count):
        first = rng.normal(scale=1.5, size=3)
        second = rng.normal(scale=1.5, size=3)
        a = points - first
        b = points - second
        orbital_a = np.exp(-0.6 * np.einsum("ij,ij->i", a, a))
        orbital_b = a[:, int(rng.integers(3))] * np.exp(-0.5 * np.einsum("ij,ij->i", b, b))
        columns.append(orbital_a * orbital_b)
    return np.column_stack(columns)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[40, 60, 80, 100])
    parser.add_argument("--kernels", nargs="+", default=["coulomb", "erfc"])
    parser.add_argument("--omega", type=float, default=0.11)
    parser.add_argument("--spacing", type=float, default=0.3)
    parser.add_argument("--order", type=int, default=12)
    parser.add_argument("--multipole-order", type=int, default=9)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cg", action="store_true")
    parser.add_argument("--cg-densities", type=int, default=3)
    parser.add_argument("--profile", action="store_true", help="time the stages of the Coulomb solve")
    parser.add_argument("--box", type=float, nargs=3, default=None,
                        help="a box domain of these side lengths (bohr) instead of the spheres of --sizes")
    parser.add_argument("--automatic-padding", type=int, nargs="+", default=[1],
                        help="1, 0 or both: with and without the automatic padding of the solver")
    arguments = parser.parse_args(argv)

    import cupy as cp

    from parsec_python.acceleration.Hartree.kronecker_device import DeviceKroneckerPoisson

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"].decode() if isinstance(properties["name"], bytes) else properties["name"]
    print(f"device {device.id}: {name}, spacing {arguments.spacing}, order {arguments.order}, "
          f"multipole order {arguments.multipole_order}", flush=True)
    print("n box n_active kernel setup_s upload_s plan_MB batch ms_per_density pool_peak_MB", flush=True)
    pool = cp.get_default_memory_pool()
    for size in arguments.sizes:
        radius = 0.5 * size * arguments.spacing
        if arguments.box is None:
            settings = GridSettings(spacing=arguments.spacing, radius=radius, expansion_order=arguments.order)
        else:
            settings = GridSettings(
                spacing=arguments.spacing, radius=0.5 * max(arguments.box), expansion_order=arguments.order,
                domain_shape="box", box_lengths=tuple(arguments.box),
            )
        grid = build_cluster_grid(settings)
        for kernel, automatic in [(k, a) for k in arguments.kernels for a in arguments.automatic_padding]:
            omega = None if kernel == "coulomb" else arguments.omega
            started = perf_counter()
            host = KroneckerPoisson(
                grid, kernel=kernel, omega=omega, multipole_order=arguments.multipole_order,
                automatic_padding=automatic,
            )
            setup = perf_counter() - started
            pool.free_all_blocks()
            solver = DeviceKroneckerPoisson(host)
            batch = min(arguments.batch, solver.batch_size)
            densities = cp.asarray(pair_densities(grid, batch))
            solver.solve_batch(densities, batch_size=batch)
            device.synchronize()
            best = np.inf
            for _ in range(arguments.repeat):
                device.synchronize()
                started = perf_counter()
                solver.solve_batch(densities, batch_size=batch)
                device.synchronize()
                best = min(best, perf_counter() - started)
            box = "x".join(str(v) for v in solver.box_shape)
            print(
                f"{size} {box} {grid.size} {kernel}{'' if automatic else '(unpadded)'} {setup:.2f} {solver.upload_seconds:.2f} "
                f"{solver.plan_bytes / 1e6:.0f} {batch} {1e3 * best / batch:.3f} "
                f"{pool.total_bytes() / 1e6:.0f}",
                flush=True,
            )
            if arguments.profile and kernel == "coulomb":
                stage_profile(solver.plan, densities, cp)
                print("  stages (ms per density):", stage_profile(solver.plan, densities, cp), flush=True)
            del solver, host, densities
            pool.free_all_blocks()
        if arguments.cg:
            cg_timing(grid, arguments, cp)


def stage_profile(plan, densities, cp) -> str:
    """Milliseconds per density of the stages of a Coulomb solve (same steps as ``solve_block``)."""

    import math

    from parsec_python.Hartree.kronecker import _separable

    nb = int(densities.shape[1])
    nx, ny, nz = plan.box_shape
    size = nx * ny * nz
    times = {}
    device = cp.cuda.Device()

    def stage(name, function):
        device.synchronize()
        started = perf_counter()
        value = function()
        device.synchronize()
        times[name] = times.get(name, 0.0) + perf_counter() - started
        return value

    def scatter():
        box = cp.zeros((nb, size), dtype=cp.float64)
        box[:, plan.box_index] = densities.T
        return box.reshape(nb, nx, ny, nz)

    box = stage("scatter", scatter)
    moments = stage("moments", lambda: _separable(box, *plan.monomials_t).reshape(nb, -1) @ plan.moment_map)

    def faces():
        rhs = (8.0 * math.pi) * box.reshape(nb, size)
        for rows, matrix in plan.faces:
            rhs[:, rows] += moments @ matrix
        return rhs

    rhs = stage("boundary", faces)
    spectral = stage("forward", lambda: _separable(rhs.reshape(nb, nx, ny, nz), *plan.eigenvectors_t))
    stage("scale", lambda: spectral.__imul__(plan.inverse_eigenvalues))
    potential = stage("inverse", lambda: _separable(spectral, *plan.eigenvectors))
    stage("gather", lambda: potential.reshape(nb, size)[:, plan.box_index])
    return " ".join(f"{name} {1e3 * value / nb:.3f}" for name, value in times.items())


def cg_timing(grid, arguments, cp) -> None:
    from parsec_python.acceleration.Hartree.cupy_prepared import CuPyPreparedConjugateGradientBackend
    from parsec_python.acceleration.Hartree.poisson import build_hartree_problem
    from parsec_python.Laplacian import build_negative_laplacian

    settings = HartreeSettings(multipole_order=arguments.multipole_order, atomic_tail="off",
                               boundary_tolerance=None)
    laplacian = build_negative_laplacian(grid)
    backend = CuPyPreparedConjugateGradientBackend(laplacian)
    builder = None
    try:
        from parsec_python.acceleration.Hartree.native_boundary import NativeMultipoleBoundaryBuilder

        builder = NativeMultipoleBoundaryBuilder(grid, settings.multipole_order)
    except Exception:  # no compiled extension
        builder = None
    densities = pair_densities(grid, arguments.cg_densities)
    rhs_seconds = solve_seconds = 0.0
    iterations = 0
    for column in range(densities.shape[1]):
        started = perf_counter()
        if builder is not None:
            rhs = builder.build(densities[:, column])[0]
        else:
            rhs = build_hartree_problem(densities[:, column], grid, settings, None)[0]
        rhs_seconds += perf_counter() - started
        started = perf_counter()
        payload = backend.solve(
            rhs, np.zeros(grid.size), relative_tolerance=settings.relative_tolerance,
            absolute_tolerance=settings.absolute_tolerance, max_iterations=settings.max_iterations,
        )
        cp.cuda.Device().synchronize()
        solve_seconds += perf_counter() - started
        iterations += int(payload["iterations"])
    count = densities.shape[1]
    print(
        f"  CG n_active {grid.size}: {iterations / count:.0f} iterations, device solve "
        f"{1e3 * solve_seconds / count:.2f} ms, host right-hand side {1e3 * rhs_seconds / count:.2f} ms "
        f"({'native' if builder is not None else 'NumPy'} multipoles) per density",
        flush=True,
    )


if __name__ == "__main__":
    main()
