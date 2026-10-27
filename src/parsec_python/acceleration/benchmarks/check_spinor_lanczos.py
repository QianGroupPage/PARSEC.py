#!/usr/bin/env python3
"""Why is the Lanczos bound of the device SOC solver slow?  A micro-benchmark.

In a full e176 run on an A100 the Lanczos bound took 17.6 s of 76 s: about 1.8 s per call
for 5 to 8 applications of the operator to ONE vector (0.3 s each), while an application
to a chunk of 64 columns averaged 40 ms.  This script builds the same device operator on
one GPU and separates the candidate causes without running an SCF:

1. **Width scaling.**  Time of ``operator.apply`` for 1, 2, 8, 16, 64 columns: the first
   call in the process (cold), the first call at each width, and the median of repeats.
   A per-call overhead shows up as ms per column falling steeply with the width.
2. **Parts of one application** (1 column and the widest block): the Laplacian product, the
   projector products, the spin-orbit term and the elementwise rest, each with a device
   barrier around it, to see which kernel is slow at width 1.
3. **The Lanczos routine itself**, several calls in a row (the first one is cold), against
   the applications it makes, with the cost of generating and uploading its start vector
   and of its scalar reductions measured separately.

4. **The SCF context.**  The full run's Lanczos calls were ~1.8 s each, far above the
   steady-state numbers of part 3, so part 4 repeats what the SCF does around them: the real
   eigensolver (:func:`solve_spinor_eigval`) for one first solve and ``--emulate-steps``
   SUBSPACE steps with the real settings of the input, a freshly built device operator before each
   (as the SCF loop does; ``--reuse-operator`` keeps one), the orbitals of the previous solve alive,
   and a heavy filter in between.  Per solve it prints the Lanczos/filter/QR/Ritz seconds, the CuPy
   memory-pool state, and afterwards a probe: one cold and one steady 1-column application and one
   Lanczos call.  ``--free-pool`` empties the pool before every solve (allocator hypothesis).
   If the in-solve Lanczos time is large while the probe is fast, the slowness belongs to the state
   the solver leaves or finds (pool, fresh operator); if the probe is slow too, the GPU or the
   allocator is in a slow state after heavy work.

Reading the result: if the first Lanczos call is much slower than the rest, it is warm-up
(kernel compilation, cuSPARSE set-up); if every call takes the same time, and a 1-column
application costs about as much as the 1-column line in part 1 times the step count, the
cost is steady and per application.  If a 1-column application is far slower than a
64-column one in part 1 or 2, that product is the problem.

Usage (from the repository root, on a GPU node):

    PYTHONPATH=src python -m parsec_python.acceleration.benchmarks.check_spinor_lanczos \\
        --input /path/to/cases/e176/soc/parsec.in

``--backend numpy-device`` runs the same code on the host, to check the script.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np


def _median_ms(function, backend, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        backend.synchronize()
        start = time.perf_counter()
        function()
        backend.synchronize()
        times.append(time.perf_counter() - start)
    return 1000.0 * statistics.median(times)


def _once_ms(function, backend) -> float:
    backend.synchronize()
    start = time.perf_counter()
    function()
    backend.synchronize()
    return 1000.0 * (time.perf_counter() - start)


def _random_block(xp, dimension: int, columns: int, seed: int):
    rng = np.random.default_rng(seed)
    host = rng.standard_normal((dimension, columns)) + 1j * rng.standard_normal((dimension, columns))
    return xp.asarray(np.ascontiguousarray(host))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="parsec.in of a spin-orbit case")
    parser.add_argument("--backend", default="cupy", choices=("cupy", "numpy-device"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--columns", default="1,2,8,16,64", help="block widths of part 1")
    parser.add_argument("--lanczos-calls", type=int, default=3)
    parser.add_argument("--emulate-steps", type=int, default=3,
                        help="SUBSPACE steps after the first solve in part 4 (0 skips part 4)")
    parser.add_argument("--reuse-operator", action="store_true",
                        help="part 4: keep one operator instead of building a new one per solve")
    parser.add_argument("--free-pool", action="store_true",
                        help="part 4: free the CuPy memory pool before every solve")
    arguments = parser.parse_args()
    widths = [int(value) for value in arguments.columns.split(",")]

    from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
    from parsec_python.Input import parse_parsec_input
    from parsec_python.SCF.single_point import prepare_single_point

    from ..Eigensolvers.chebff_spinor import _CountingOperator, _lanczos_upper_bound, _Stats
    from ..Hamiltonian.device_bloch import CupyArrayBackend, NumpyArrayBackend
    from ..SCF.self_consistent_soc import DeviceSpinorHamiltonianBuilder

    translation = parse_parsec_input(arguments.input)
    problem = translation.problem
    if not problem.scf.self_consistent_spin_orbit:
        raise SystemExit("the input is not a self-consistent spin-orbit case (SO_from_scratch)")
    t0 = time.perf_counter()
    system = prepare_single_point(problem)
    projectors = tuple(
        build_spin_orbit_projectors(system.grid, system.atoms, system.pseudopotentials, problem.pseudopotentials)
    )
    potential = system.ionic_potential + system.evaluate_xc(system.initial_density).potential
    print(f"case {arguments.input}: grid {system.grid.size} points, {len(projectors)} spin-orbit atoms "
          f"(host set-up {time.perf_counter() - t0:.1f} s)")

    backend = NumpyArrayBackend() if arguments.backend == "numpy-device" else CupyArrayBackend(arguments.device_id)
    backend.use()
    xp = backend.xp
    if arguments.backend == "cupy":
        properties = xp.cuda.runtime.getDeviceProperties(arguments.device_id)
        print(f"device {arguments.device_id}: {properties['name'].decode()}, CuPy {xp.__version__}")

    t0 = time.perf_counter()
    operator = DeviceSpinorHamiltonianBuilder(backend)(system.hamiltonian(potential), projectors)
    backend.synchronize()
    print(f"device operator built and uploaded in {time.perf_counter() - t0:.2f} s "
          f"(real projectors: {operator.projector_real}, real spin-orbit matrices: {operator.soc.real if operator.soc else None})")
    dimension = operator.dimension

    # ---------------------------------------------------------------- 1. width scaling
    print("\n1. operator.apply versus block width  (ms per call; ms per column)")
    print(f"   {'columns':>7s} {'cold/first':>11s} {'median':>10s} {'ms/column':>10s}")
    cold_reported = False
    scaling = {}
    for columns in widths:
        block = _random_block(xp, dimension, columns, columns)
        first = _once_ms(lambda: operator.apply(block), backend)
        median = _median_ms(lambda: operator.apply(block), backend, arguments.repeats)
        scaling[columns] = median
        label = f"{first:11.2f}" + ("  <- first call of the process" if not cold_reported else "")
        cold_reported = True
        print(f"   {columns:7d} {label} {median:10.2f} {median / columns:10.3f}")
        del block

    # ---------------------------------------------------------------- 2. parts of one apply
    print("\n2. parts of one application (median ms)")
    n = operator.grid_size
    for columns in sorted({widths[0], widths[-1]}):
        block = _random_block(xp, dimension, columns, 100 + columns)
        up = xp.ascontiguousarray(block[:n])
        parts = {
            "laplacian (real product)": lambda: operator._real_product(operator.laplacian, up),
            "diagonal * block": lambda: operator.diagonal[:, None] * up,
        }
        if operator.projector_forward is not None:
            if operator.projector_real:
                parts["projectors (adjoint, signs, forward)"] = lambda: operator._real_product(
                    operator.projector_forward,
                    operator._real_product(operator.projector_adjoint, up) * operator.projector_signs[:, None],
                )
            else:
                parts["projectors (adjoint, signs, forward)"] = lambda: operator.projector_forward @ (
                    (operator.projector_adjoint @ up) * operator.projector_signs[:, None]
                )
        if operator.soc is not None:
            parts["spin-orbit term (whole)"] = lambda: operator._spin_orbit(block)
            parts["  spin-orbit: adjoint product only"] = lambda: (
                operator._real_product(operator.soc.adjoint, xp.concatenate((up, up), axis=1))
                if operator.soc.real
                else operator.soc.adjoint @ xp.concatenate((up, up), axis=1)
            )
        parts["whole apply"] = lambda: operator.apply(block)
        print(f"   width {columns}")
        for name, function in parts.items():
            function()  # warm
            print(f"     {name:40s} {_median_ms(function, backend, arguments.repeats):10.3f}")
        del block, up

    # ---------------------------------------------------------------- 3. Lanczos
    print("\n3. the Lanczos bound")
    rng = np.random.default_rng(5)
    print(f"   start vector: random on the host {_once_ms(lambda: rng.random(dimension) + 1j * rng.random(dimension), backend):.2f} ms,"
          f" upload {_once_ms(lambda: xp.asarray(rng.random(dimension) + 1j * rng.random(dimension)), backend):.2f} ms (includes the draw)")
    vector = _random_block(xp, dimension, 1, 7)[:, 0]
    print(f"   one vdot {_median_ms(lambda: float(xp.real(xp.vdot(vector, vector))), backend, arguments.repeats):.3f} ms,"
          f" one norm {_median_ms(lambda: float(xp.linalg.norm(vector)), backend, arguments.repeats):.3f} ms")
    one_column = scaling.get(1)
    if one_column is None:
        block = _random_block(xp, dimension, 1, 1)
        one_column = _median_ms(lambda: operator.apply(block), backend, arguments.repeats)
    print(f"   {'steps':>5s} {'call':>5s} {'seconds':>9s} {'applications':>13s} {'ms/application':>15s}")
    for steps in (8, 5):
        for call in range(arguments.lanczos_calls):
            stats = _Stats(backend.synchronize)
            counted = _CountingOperator(operator, stats)
            seconds = _once_ms(
                lambda: _lanczos_upper_bound(xp, counted, dimension, steps, np.random.default_rng(100 + call)), backend
            ) / 1000.0
            applications = int(stats.values["applications"])
            print(f"   {steps:5d} {call + 1:5d} {seconds:9.3f} {applications:13d} {1000.0 * seconds / applications:15.2f}"
                  + ("   <- first call" if call == 0 and steps == 8 else ""))
    print(f"\n   reference: one 1-column application alone takes {one_column:.2f} ms (part 1).")
    print("   If the Lanczos ms/application is much larger than that, the time is not in the operator:")
    print("   look at the first-call line (warm-up) and the start-vector/vdot/norm lines above.")

    # ---------------------------------------------------------------- 4. SCF context
    if arguments.emulate_steps > 0:
        from parsec_python.SCF.self_consistent_soc import _eigval_settings

        from ..Eigensolvers.chebff_spinor import solve_spinor_eigval

        def pool_line() -> str:
            if arguments.backend != "cupy":
                return "pool n/a (host)"
            pool = xp.get_default_memory_pool()
            free, total = xp.cuda.runtime.memGetInfo()
            return (f"pool used {pool.used_bytes() / 2**30:5.2f} GiB, held {pool.total_bytes() / 2**30:5.2f} GiB,"
                    f" device free {free / 2**30:5.1f}/{total / 2**30:.1f} GiB")

        settings = _eigval_settings(system, system.input.eigensolver.filter_degree)
        requested = int(problem.scf.number_of_states)
        builder = DeviceSpinorHamiltonianBuilder(backend)
        print(f"\n4. the SCF context: {requested} states (+{settings.safety_buffer} buffer), "
              f"one first solve then {arguments.emulate_steps} SUBSPACE steps"
              f"{', one operator reused' if arguments.reuse_operator else ', a new operator per solve'}"
              f"{', pool freed before each solve' if arguments.free_pool else ''}")
        print(f"   {'solve':>8s} {'total s':>8s} {'lanczos':>8s} {'filter':>8s} {'QR':>8s} {'ritz':>8s} {'apps':>6s}"
              f" {'build s':>8s} | probe: {'1-col cold':>10s} {'1-col steady':>12s} {'lanczos s':>10s}")
        operator_k = None
        state = None
        probe_block = _random_block(xp, dimension, 1, 99)
        for step in range(arguments.emulate_steps + 1):
            build = 0.0
            if operator_k is None or not arguments.reuse_operator:
                start = time.perf_counter()
                operator_k = builder(system.hamiltonian(potential), projectors)
                backend.synchronize()
                build = time.perf_counter() - start
            if arguments.free_pool and arguments.backend == "cupy":
                xp.get_default_memory_pool().free_all_blocks()
            before = pool_line()
            backend.synchronize()
            start = time.perf_counter()
            result = solve_spinor_eigval(
                operator_k, requested, settings=settings, state=state, xp=xp, sync=backend.synchronize
            )
            backend.synchronize()
            total = time.perf_counter() - start
            state = result.state
            stats = result.stats
            # the probe: what a single vector costs right after the heavy work
            cold = _once_ms(lambda: operator_k.apply(probe_block), backend)
            steady = _median_ms(lambda: operator_k.apply(probe_block), backend, 3)
            counted = _CountingOperator(operator_k, _Stats(backend.synchronize))
            lanczos = _once_ms(
                lambda: _lanczos_upper_bound(xp, counted, dimension, 5, np.random.default_rng(7 + step)), backend
            ) / 1000.0
            print(f"   {'first' if step == 0 else 'subsp ' + str(step):>8s} {total:8.2f} {stats.get('lanczos_seconds', 0.0):8.2f}"
                  f" {stats.get('filter_seconds', 0.0):8.2f} {stats.get('orthonormalize_seconds', 0.0):8.2f}"
                  f" {stats.get('ritz_seconds', 0.0):8.2f} {int(stats.get('applications', 0)):6d} {build:8.2f} |"
                  f"        {cold:10.2f} {steady:12.2f} {lanczos:10.3f}")
            print(f"            before: {before}")
            print(f"            after : {pool_line()}")
        print("\n   Compare 'lanczos' (inside the solve) with the probe's 'lanczos s' (steady state, part 3).")
        print("   A large in-solve value with a fast probe means the slowness is tied to what the solver does or")
        print("   finds (a fresh operator, the pool); rerun with --reuse-operator and with --free-pool to split them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
