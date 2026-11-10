"""Time and measure the host preparation of one bulk-diamond cell, stage by stage, and compare two source trees.

Usage (from the repository root, src on the python path):
    python examples/3d_diamond_benchmark/profile_setup.py <n1> <n2> <n3> [--work DIR] [--route full|worker]
        [--forces] [--save FIELDS.npz] [--compare FIELDS.npz] [--reference-src SRC] [--json OUT.json]

The input is make_diamond.py's (with C_POTRE.DAT beside it). ``--route full`` runs the stages of
:func:`parsec_python.SCF.pbc.prepare_periodic_single_point` one after another through the functions of that module
(the preparation of the k-point route on every rank and of the Gamma route's root); ``--route worker`` runs
:func:`parsec_python.acceleration.driver._prepare_periodic_orbital_operators` (a rank that shares the basis). Every
stage prints its seconds, the resident memory after it and its own peak (Linux: the peak is reset before each stage
through /proc/self/clear_refs). ``--forces`` also times the Ewald ion-ion forces.

``--save`` writes the fields (local potential, densities, projectors, energies) to an npz file; ``--compare`` reads
such a file and prints the largest differences. ``--reference-src SRC`` runs this script a second time in a child
process with ``PYTHONPATH=SRC`` (another source tree, e.g. a copy of the base commit's src), in the same job so
that the timings of both share the machine, and compares its fields with these.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def _status_kib(field: str) -> float:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return float(line.split()[1])
    except OSError:
        pass
    return float("nan")


def _reset_peak() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
        return True
    except OSError:
        return False


def _memory_gib() -> tuple[float, float]:
    """(resident now, peak resident since the last reset) in GiB (Linux; NaN elsewhere)."""
    return _status_kib("VmRSS") / 2**20, _status_kib("VmHWM") / 2**20


class Stages:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def run(self, name: str, function, *args, **kwargs):
        _reset_peak()
        start = time.perf_counter()
        value = function(*args, **kwargs)
        seconds = time.perf_counter() - start
        now, peak = _memory_gib()
        self.records.append({"stage": name, "seconds": seconds, "rss_gib": now, "peak_gib": peak})
        print(f"  {name:<22s} {seconds:9.2f} s   rss {now:7.2f} GiB   stage peak {peak:7.2f} GiB", flush=True)
        return value


def _make_input(cells, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    target = work / "parsec.in"
    subprocess.run(
        [sys.executable, str(HERE / "make_diamond.py"), *map(str, cells), str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    shutil.copy(HERE / "C_POTRE.DAT", work / "C_POTRE.DAT")
    return target


def _full_route(problem, stages: Stages, forces: bool) -> dict:
    from parsec_python.SCF import pbc as periodic

    atoms = tuple(problem.atoms)
    lattice = problem.periodic_cell.lattice_vectors
    pseudopotentials = stages.run(
        "pseudopotentials", periodic.load_pseudopotentials, problem.pseudopotentials,
        xc_functional=problem.scf.xc_functional,
    )
    electrons = periodic.ionic_charge(atoms, pseudopotentials) - problem.scf.net_charge
    grid = stages.run("grid", periodic.build_periodic_grid, problem.periodic_cell, problem.grid)
    laplacian = stages.run("laplacian", periodic.build_negative_laplacian, grid)
    gradient = stages.run("gradient", periodic.build_gradient, grid)
    potential = stages.run(
        "local_ionic", periodic.ewald_local_ionic_potential, grid, atoms, pseudopotentials,
        problem.pseudopotentials, lattice,
    )
    projectors = stages.run(
        "projectors", periodic.build_nonlocal_projectors, grid, atoms, pseudopotentials, problem.pseudopotentials,
        lattice, periodic_dimensions=3,
    )
    density = stages.run(
        "initial_density", periodic.superpose_atomic_density, grid, atoms, pseudopotentials,
        problem.pseudopotentials, lattice, periodic_dimensions=3,
    )
    density = periodic.normalize_density(density, grid, electrons)
    core = stages.run(
        "core_density", periodic.superpose_atomic_density, grid, atoms, pseudopotentials,
        problem.pseudopotentials, lattice, core=True, periodic_dimensions=3,
    )
    ion_ion = stages.run("ion_ion", periodic.ewald_ion_ion_energy, atoms, pseudopotentials, lattice)
    alpha_z = periodic.ewald_alpha_z_energy(atoms, pseudopotentials, lattice, electrons)
    fields = {
        "ionic_potential": potential,
        "initial_density": density,
        "core_density": core,
        "projector_data": projectors.projectors.data,
        "projector_indices": projectors.projectors.indices,
        "projector_indptr": projectors.projectors.indptr,
        "projector_signs": projectors.signs,
        "laplacian_data": laplacian.data,
        "laplacian_indices": laplacian.indices,
        "laplacian_indptr": laplacian.indptr,
        "gradient_data": np.concatenate([matrix.data for matrix in gradient]),
        "gradient_indices": np.concatenate([matrix.indices for matrix in gradient]),
        "ion_ion_energy": np.array(ion_ion),
        "alpha_z_energy": np.array(alpha_z),
    }
    if forces:
        from parsec_python.Forces.ion_ion import ewald_ion_ion_forces

        fields["ion_ion_forces"] = stages.run("ion_ion_forces", ewald_ion_ion_forces, atoms, pseudopotentials, lattice)
    print(f"  grid {grid.points_per_axis} = {grid.size} points, {len(atoms)} atoms, E_ion-ion {ion_ion!r} Ry", flush=True)
    return fields


def _worker_route(problem, stages: Stages) -> dict:
    from parsec_python.acceleration.driver import _prepare_periodic_orbital_operators

    system = stages.run("worker_operators", _prepare_periodic_orbital_operators, problem)
    timings = system.timings
    print(
        f"  (grid {timings.grid_seconds:.2f} s, laplacian {timings.finite_difference_seconds:.2f} s, projectors "
        f"{timings.nonlocal_ionic_seconds:.2f} s)",
        flush=True,
    )
    projectors = system.nonlocal_operator.projectors
    return {
        "projector_data": projectors.data,
        "projector_indices": projectors.indices,
        "projector_indptr": projectors.indptr,
        "laplacian_data": system.negative_laplacian.data,
    }


def compare(mine: dict, other: dict) -> dict:
    """The largest differences between two sets of fields (absolute, and relative to the largest value)."""
    report = {}
    for name in sorted(set(mine) & set(other)):
        a = np.asarray(mine[name])
        b = np.asarray(other[name])
        if a.shape != b.shape:
            report[name] = f"shape {a.shape} vs {b.shape}"
            continue
        if a.dtype.kind in "iu":
            report[name] = "equal" if np.array_equal(a, b) else f"{int(np.count_nonzero(a != b))} entries differ"
            continue
        if a.size == 0:
            report[name] = "empty"
            continue
        difference = float(np.max(np.abs(a - b)))
        scale = float(np.max(np.abs(b))) or 1.0
        report[name] = f"max |diff| {difference:.3e} (relative {difference / scale:.3e})" + (
            " bit-identical" if np.array_equal(a, b) else ""
        )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cells", type=int, nargs=3)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--route", choices=("full", "worker"), default="full")
    parser.add_argument("--forces", action="store_true")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--reference-src", type=Path)
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args(argv)

    work = arguments.work or Path(tempfile.mkdtemp(prefix="profile_setup_"))
    reference_fields = None
    if arguments.reference_src is not None:
        # the other tree first (its fields to a file), then this one, in the same job
        reference_file = work / "reference_fields.npz"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(arguments.reference_src)
        command = [sys.executable, str(Path(__file__).resolve()), *map(str, arguments.cells),
                   "--work", str(work / "reference"), "--route", arguments.route, "--save", str(reference_file)]
        if arguments.forces:
            command.append("--forces")
        if arguments.json is not None:
            command += ["--json", str(arguments.json.with_name(arguments.json.stem + "_reference.json"))]
        print(f"reference tree {arguments.reference_src}:", flush=True)
        subprocess.run(command, env=environment, check=True)
        reference_fields = dict(np.load(reference_file))

    import parsec_python  # noqa: F401  (installs the fast periodic preparation)
    from parsec_python.Input import parse_parsec_input

    print(
        f"tree {Path(parsec_python.__file__).parent}; cells {arguments.cells}; route {arguments.route}; "
        f"OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS', 'unset')}, CPUs {os.cpu_count()}",
        flush=True,
    )
    path = _make_input(arguments.cells, work / "this")
    problem = parse_parsec_input(path).problem
    stages = Stages()
    started = time.perf_counter()
    if arguments.route == "full":
        fields = _full_route(problem, stages, arguments.forces)
    else:
        fields = _worker_route(problem, stages)
    total = time.perf_counter() - started
    now, _ = _memory_gib()
    peak_total = _status_kib("VmPeak") / 2**20
    print(f"  total {total:.2f} s; rss {now:.2f} GiB; peak of the stages "
          f"{max(record['peak_gib'] for record in stages.records):.2f} GiB (VmPeak virtual {peak_total:.2f} GiB)",
          flush=True)
    if arguments.save is not None:
        arguments.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(arguments.save, **fields)
    others = []
    if reference_fields is not None:
        others.append(("reference tree", reference_fields))
    if arguments.compare is not None:
        others.append((str(arguments.compare), dict(np.load(arguments.compare))))
    comparisons = {}
    for label, other in others:
        print(f"differences against {label}:", flush=True)
        comparisons[label] = compare(fields, other)
        for name, text in comparisons[label].items():
            print(f"  {name:<20s} {text}", flush=True)
        if "ion_ion_energy" in fields and "ion_ion_energy" in other:
            mine, theirs = float(fields["ion_ion_energy"]), float(other["ion_ion_energy"])
            print(f"  ion-ion {mine!r} vs {theirs!r}: relative {(mine - theirs) / abs(theirs):.3e}", flush=True)
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(
            {"cells": arguments.cells, "route": arguments.route, "stages": stages.records, "total_seconds": total,
             "comparisons": comparisons}, indent=1))
    if arguments.work is None:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
