"""Profile a device ACE build of the nanodiamond grids, stage by stage.

    python -m parsec_python.acceleration.benchmarks.exchange_parity CASE \
        [--devices 0,1,2,3] [--functional pbe0|hse06] [--columns all|occupied] \
        [--layout C|F] [--stages] [--solve-stages] [--fraction 1.0]

``CASE`` is one of the sphere grids of ``examples/0d_nanodiamonds``
(``k03`` ... ``k06``: radius 10.8 ... 17.3 Angstrom, h = 0.20 Angstrom,
order 8) with its ``States_Num`` and occupied count.  The symmetry is that
of a tetrahedral molecule at the origin (the D2 subgroup the driver detects
for the nanodiamonds, four sectors); the sector orbitals are random
(fixed seed), one sector per device round-robin as the driver places them.
The build is :func:`~parsec_python.acceleration.Hamiltonian.exchange_device.apply_device_exchange`
of every column (``--columns all``, an ACE build) as the Fock loop calls it.

``--stages`` wraps the parts of the build with CUDA events on the stream
that runs them (column gathers and peer copies, unfold, solve, row update,
fold) and prints their GPU time per device; ``--solve-stages`` times the
steps of one Kronecker solve of ``Exx_Batch_Size`` densities (scatter,
moments, boundary, transforms, spectral factor, smoothing, gather).
``--fraction`` keeps that share of the occupied and empty states of every
sector (a shorter build with the same grid).  Times are wall clock of the
build after one warm-up row, and the GPU times of the events.
"""

from __future__ import annotations

import argparse
import math
from threading import Lock
from time import perf_counter

import numpy as np

ANGSTROM = 1.0 / 0.529177210903
# (radius in Angstrom, States_Num, occupied orbitals) of the nanodiamond inputs.
CASES = {
    "t1": (3.0, 40, 24),  # a small sphere for tests on a laptop
    "k03": (10.8, 297, 212),
    "k04": (12.8, 605, 432),
    "k05": (16.4, 1026, 1020),
    "k06": (17.3, 1746, 1740),
}
SPACING_ANGSTROM = 0.20
EXPANSION_ORDER = 8
MULTIPOLE_ORDER = 9
TERMS = {
    "pbe0": (("coulomb", 1.0, None),),
    "hse06": (("erfc", 1.0, 0.11),),
}


def nanodiamond_grid(case: str):
    from parsec_python.Grid import build_cluster_grid
    from parsec_python.models import GridSettings

    radius = CASES[case][0] * ANGSTROM
    return build_cluster_grid(
        GridSettings(spacing=SPACING_ANGSTROM * ANGSTROM, radius=radius, expansion_order=EXPANSION_ORDER)
    )


def tetrahedral_decomposition(grid):
    """The reflection decomposition of a tetrahedral molecule on ``grid`` (the driver's detection)."""

    from parsec_python.acceleration.Symmetry.axis_reflection import SignedPermutationReduction
    from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
    from parsec_python.models import Atom

    a = 1.18
    atoms = [Atom("C", (0.0, 0.0, 0.0))] + [
        Atom("H", (a * sx, a * sy, a * sx * sy)) for sx in (1, -1) for sy in (1, -1)
    ]
    reduction = SignedPermutationReduction.detect(grid, atoms)
    return ReflectionRepresentationDecomposition.build(grid, reduction)


def random_channel(grid, decomposition, states: int, occupied: int, devices, *, layout: str = "C",
                   fraction: float = 1.0, seed: int = 7):
    """:class:`ChannelOrbitals` of random sector orbitals, sector ``r`` on ``devices[r % len]``."""

    import cupy as cp

    from parsec_python.acceleration.Hamiltonian.exchange_device import (
        ChannelOrbitals,
        OrbitalSector,
        SectorExpansion,
    )

    count = decomposition.representation_count
    rng = np.random.default_rng(seed)
    occupied_counts = [occupied // count + (1 if r < occupied % count else 0) for r in range(count)]
    empty = states - occupied
    empty_counts = [empty // count + (1 if r < empty % count else 0) for r in range(count)]
    occupied_counts = [max(1, int(round(fraction * value))) for value in occupied_counts]
    empty_counts = [int(round(fraction * value)) for value in empty_counts]
    total = sum(occupied_counts) + sum(empty_counts)
    occupations = np.zeros(total)
    sectors = []
    next_state = 0
    occupied_states = []
    for r in range(count):
        occupied_states.extend(range(next_state, next_state + occupied_counts[r]))
        next_state += occupied_counts[r] + empty_counts[r]
    occupations[occupied_states] = 1.0
    next_state = 0
    multiplicities = decomposition.reduction.multiplicities
    for r in range(count):
        orbits = decomposition.sector_orbit_indices(r)
        scales = 1.0 / np.sqrt(multiplicities[orbits])
        columns = occupied_counts[r] + empty_counts[r]
        rows = int(orbits.size)
        device = int(devices[r % len(devices)])
        host = rng.standard_normal((rows, columns)) / math.sqrt(rows)
        with cp.cuda.Device(device):
            vectors = cp.asarray(np.ascontiguousarray(host) if layout == "C" else np.asfortranarray(host))
        sectors.append(
            OrbitalSector(
                representation=r,
                device=device,
                vectors=vectors,
                local=np.arange(columns, dtype=np.int64),
                states=np.arange(next_state, next_state + columns, dtype=np.int64),
                expansion=SectorExpansion(
                    decomposition.reduction.full_to_wedge,
                    decomposition.phases[r],
                    orbits,
                    scales,
                    decomposition.wedge_size,
                ),
            )
        )
        next_state += columns
    return ChannelOrbitals(tuple(sectors), occupations, 2.0, int(grid.size), float(grid.volume_element))


class StageTimer:
    """CUDA events around wrapped calls, summed per (stage, device)."""

    def __init__(self, cp) -> None:
        self.cp = cp
        self.events: list = []
        self.lock = Lock()

    def wrap(self, label: str, function):
        cp = self.cp

        def wrapped(*args, **kwargs):
            begin = cp.cuda.Event()
            begin.record()
            result = function(*args, **kwargs)
            end = cp.cuda.Event()
            end.record()
            with self.lock:
                self.events.append((label, int(cp.cuda.Device().id), begin, end))
            return result

        return wrapped

    def totals(self) -> dict:
        cp = self.cp
        result: dict = {}
        for label, device, begin, end in self.events:
            with cp.cuda.Device(device):
                end.synchronize()
                seconds = cp.cuda.get_elapsed_time(begin, end) * 1.0e-3
            key = (label, device)
            count, total = result.get(key, (0, 0.0))
            result[key] = (count + 1, total + seconds)
        self.events.clear()
        return result


def instrument(timer: StageTimer):
    """Wrap the parts of the build; returns a function that undoes it."""

    from parsec_python.acceleration.Hamiltonian import exchange_device as module

    saved = [
        (module.OrbitalSector, "columns", module.OrbitalSector.columns),
        (module.SectorExpansion, "unfold", module.SectorExpansion.unfold),
        (module.SectorExpansion, "fold", module.SectorExpansion.fold),
        (module.DevicePairSolvers, "solve", module.DevicePairSolvers.solve),
        (module, "_accumulate_row", module._accumulate_row),
    ]
    labels = {"columns": "gather/peer copy", "unfold": "unfold", "fold": "fold", "solve": "solve",
              "_accumulate_row": "row update", "solve_parity": "class solves"}
    labels["solve_parity_box"] = "class solves"
    for name in ("solve_parity", "solve_parity_box"):
        if hasattr(module.DevicePairSolvers, name):
            saved.append((module.DevicePairSolvers, name, getattr(module.DevicePairSolvers, name)))
    for owner, name, function in saved:
        setattr(owner, name, timer.wrap(labels[name], function))
    for name in ("parity_columns", "parity_fold", "parity_densities", "parity_box_densities", "parity_update"):
        function = getattr(module, name, None)
        if function is not None:
            saved.append((module, name, function))
            setattr(module, name, timer.wrap(name, function))

    def undo() -> None:
        for owner, name, function in saved:
            setattr(owner, name, function)

    return undo


def staged_solve(plan, densities, cp, record):
    """:func:`parsec_python.Hartree.kronecker.solve_block` with ``record(label)`` between its steps."""

    from parsec_python.Hartree.kronecker import _separable

    nb = int(densities.shape[1])
    nx, ny, nz = plan.box_shape
    box_size = nx * ny * nz
    record("start")
    box = cp.zeros((nb, box_size), dtype=cp.float64)
    box[:, plan.box_index] = densities.T
    box = box.reshape(nb, nx, ny, nz)
    record("scatter")
    cartesian = _separable(box, *plan.monomials_t)
    moments = cartesian.reshape(nb, -1) @ plan.moment_map
    record("moments")
    right_hand_side = (8.0 * math.pi) * box.reshape(nb, box_size)
    for rows, matrix in plan.faces:
        right_hand_side[:, rows] += moments @ matrix
    record("boundary")
    spectral = _separable(right_hand_side.reshape(nb, nx, ny, nz), *plan.eigenvectors_t)
    record("forward")
    spectral *= plan.inverse_eigenvalues
    record("scale")
    if plan.kernel == "coulomb":
        result = _separable(spectral, *plan.eigenvectors).reshape(nb, box_size)[:, plan.box_index]
        record("inverse+gather")
        return result
    reduced = _separable(spectral, *plan.gaussian_contract)
    smooth = _separable(reduced, *plan.gaussian_expand).reshape(nb, box_size)
    long_range = smooth[:, plan.box_index]
    record("smoothing")
    long_range += moments @ plan.exterior_tail
    record("tails")
    coulomb = _separable(spectral, *plan.eigenvectors).reshape(nb, box_size)[:, plan.box_index]
    record("inverse+gather")
    coulomb *= plan.coulomb_weight
    coulomb += plan.long_range_weight * long_range
    record("combine")
    return coulomb


def solve_stage_profile(solvers, grid, device: int, batch: int, repeat: int = 5) -> None:
    import cupy as cp

    with cp.cuda.Device(device):
        solver = solvers.solvers(device)[0]
        rng = cp.random.default_rng(3)
        densities = cp.asfortranarray(rng.standard_normal((grid.size, batch)))
        totals: dict = {}
        for trial in range(repeat + 1):
            marks = []

            def record(label):
                event = cp.cuda.Event()
                event.record()
                marks.append((label, event))

            staged_solve(solver.plan, densities, cp, record)
            marks[-1][1].synchronize()
            if trial == 0:
                continue
            for (_, before), (label, after) in zip(marks, marks[1:]):
                totals[label] = totals.get(label, 0.0) + cp.cuda.get_elapsed_time(before, after)
        text = ", ".join(f"{label} {value / (repeat * batch):.4f}" for label, value in totals.items())
        print(f"solve stages, ms per density (batch {batch}, box {solver.box_shape}): {text}; "
              f"total {sum(totals.values()) / (repeat * batch):.4f}", flush=True)


def class_solve_timing(host, device: int, batches, repeat: int = 3) -> None:
    """ms per density of the full-box solve and of each parity-class solve on the octant."""

    import cupy as cp

    from parsec_python.Hartree.kronecker import PARITY_CLASSES, solve_block
    from parsec_python.acceleration.Hartree.kronecker_device import _upload

    layout = host.parity_layout
    with cp.cuda.Device(device):
        full = host.plan.mapped(lambda array: _upload(cp, array))
        plans = [host.parity_plan(index).mapped(lambda array: _upload(cp, array)) for index in range(8)]
        rng = cp.random.default_rng(5)
        print(f"octant {layout.half_shape}, R {layout.size} points (active {host.size}, box {host.box_shape})",
              flush=True)

        def timed(plan, count, nb):
            densities = cp.asfortranarray(rng.standard_normal((count, nb)))
            solve_block(plan, densities, cp)
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record()
            for _ in range(repeat):
                solve_block(plan, densities, cp)
            end.record()
            end.synchronize()
            return cp.cuda.get_elapsed_time(begin, end) / (repeat * nb)

        for nb in batches:
            line = [f"batch {nb}: full {timed(full, host.size, min(nb, 32)):.4f}"]
            for index in (0, 7, 3):
                line.append(f"class {PARITY_CLASSES[index]} {timed(plans[index], layout.size, nb):.4f}")
            print("  ms/density " + ", ".join(line), flush=True)
            cp.get_default_memory_pool().free_all_blocks()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--class-solves", type=int, nargs="*", default=None,
                        help="time the parity-class solves for these batch sizes and stop")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--devices", default="0")
    parser.add_argument("--functional", choices=sorted(TERMS), default="pbe0")
    parser.add_argument("--columns", choices=("all", "occupied"), default="all")
    parser.add_argument("--layout", choices=("C", "F"), default="C")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--solve-stages", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--save", default=None, help="write E_x and checksums of W to this .npz")
    parser.add_argument("--compare", action="store_true",
                        help="build once on the full grid and once in parity classes, compare W and E_x")
    arguments = parser.parse_args(argv)

    import cupy as cp

    from parsec_python.acceleration.Hamiltonian.exchange_device import (
        DevicePairSolvers,
        apply_device_exchange,
    )

    devices = tuple(int(value) for value in arguments.devices.split(","))
    radius, states, occupied = CASES[arguments.case]
    started = perf_counter()
    grid = nanodiamond_grid(arguments.case)
    decomposition = tetrahedral_decomposition(grid)
    print(f"{arguments.case}: {grid.size} points, group order {decomposition.reduction.group_order}, "
          f"sectors {decomposition.sector_sizes}, setup {perf_counter() - started:.1f} s", flush=True)
    started = perf_counter()
    solvers = DevicePairSolvers(grid, TERMS[arguments.functional], MULTIPOLE_ORDER, arguments.batch)
    print(f"plan: {solvers.description}; {perf_counter() - started:.1f} s, "
          f"{solvers.plan_bytes() / 2**20:.0f} MB", flush=True)
    if arguments.class_solves is not None:
        started = perf_counter()
        for index in range(8):
            solvers.hosts[0].parity_plan(index)
        print(f"parity plans {perf_counter() - started:.1f} s", flush=True)
        class_solve_timing(solvers.hosts[0], devices[0], arguments.class_solves or [32, 64, 128, 256])
        return
    channel = random_channel(grid, decomposition, states, occupied, devices, layout=arguments.layout,
                             fraction=arguments.fraction)
    occupied_count = int(np.count_nonzero(channel.occupations > 0.5))
    print(f"states {channel.state_count}, occupied {occupied_count}, devices {devices}, layout {arguments.layout}",
          flush=True)
    for device in devices:
        solvers.solvers(device)
    if arguments.solve_stages:
        solve_stage_profile(solvers, grid, devices[0], arguments.batch)

    import os

    from parsec_python.acceleration.Hamiltonian.exchange_device import PARITY_SETTING

    if arguments.compare:
        # The same orbitals through the full-grid path and the parity path.
        results = {}
        for setting in ("off", "auto"):
            os.environ[PARITY_SETTING] = setting
            started = perf_counter()
            application = apply_device_exchange(channel, solvers, devices, columns=arguments.columns)
            for device in devices:
                cp.cuda.Device(device).synchronize()
            wall = perf_counter() - started
            blocks = []
            for index, sector in enumerate(channel.sectors):
                with cp.cuda.Device(sector.device):
                    blocks.append(cp.asnumpy(application.applied[index]))
            results[setting] = (application.energy, blocks, wall, application.poisson_seconds)
            print(f"{setting}: E_x {application.energy:.15f} Ry, {wall:.2f} s wall, "
                  f"solves {application.poisson_seconds:.2f} s", flush=True)
        (e0, w0, _, _), (e1, w1, _, _) = results["off"], results["auto"]
        scale = max(float(np.abs(block).max()) for block in w0)
        worst = max(float(np.abs(a - b).max()) for a, b in zip(w0, w1))
        print(f"compare: dE_x {e1 - e0:.3e} Ry (relative {abs(e1 - e0) / abs(e0):.2e}), "
              f"max|dW|/max|W| {worst / scale:.2e}", flush=True)
        os.environ.pop(PARITY_SETTING, None)
        return

    # Warm-up: a short build of a few columns (kernels, pools, plans on every device).
    warm = random_channel(grid, decomposition, 8 * len(devices) + 4, 4 * len(devices), devices,
                          layout=arguments.layout, fraction=1.0, seed=11)
    apply_device_exchange(warm, solvers, devices, columns="all")
    del warm
    pool_peaks = {}
    for trial in range(arguments.repeat):
        timer = StageTimer(cp)
        undo = instrument(timer) if arguments.stages else (lambda: None)
        for device in devices:
            with cp.cuda.Device(device):
                cp.get_default_memory_pool().free_all_blocks()
        try:
            started = perf_counter()
            application = apply_device_exchange(channel, solvers, devices, columns=arguments.columns)
            for device in devices:
                cp.cuda.Device(device).synchronize()
            wall = perf_counter() - started
        finally:
            undo()
        for device in devices:
            with cp.cuda.Device(device):
                pool_peaks[device] = cp.get_default_memory_pool().total_bytes() / 2**30
        print(f"build {trial}: {wall:.2f} s wall, {application.pair_solves} pair solves "
              f"({wall / max(1, application.pair_solves) * 1e3:.4f} ms/pair), solve GPU time "
              f"{application.poisson_seconds:.2f} s (max over devices), E_x {application.energy:.12f} Ry, "
              f"pool GiB {pool_peaks}", flush=True)
        if arguments.stages:
            totals = timer.totals()
            for device in devices:
                parts = {label: value for (label, owner), value in totals.items() if owner == device}
                text = ", ".join(f"{label} {seconds:.2f} s ({count})" for label, (count, seconds) in sorted(parts.items()))
                print(f"  device {device}: {text}", flush=True)
    if arguments.save:
        norms = []
        for index, sector in enumerate(channel.sectors):
            with cp.cuda.Device(sector.device):
                block = application.applied[index]
                norms.append(cp.asnumpy(cp.sqrt(cp.sum(block * block, axis=0))))
                checks = cp.asnumpy(block[:: max(1, block.shape[0] // 4096)])
                np.save(f"{arguments.save}_sector{index}.npy", checks)
        np.savez(arguments.save, energy=application.energy, norms=np.concatenate(norms))


if __name__ == "__main__":
    main()
