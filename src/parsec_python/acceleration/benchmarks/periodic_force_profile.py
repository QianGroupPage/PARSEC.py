"""Where the time of one ionic step of a periodic cell goes (forces, relaxation, dynamics).

Writes small Si inputs (8 and 64 atoms, Gamma and k-points) into the output folder, then for each:
prepares the cell, runs the cold SCF loop, the forces (term by term), moves one atom, rebuilds the
cell, runs the warm-started SCF loop and the forces again.  The readable path is the one the
accelerated command line routes these inputs to (``Relaxation.periodic_backend``); with ``--accelerated``
the Gamma cases also run through the accelerated driver (``acceleration.driver``).

    python -m parsec_python.acceleration.benchmarks.periodic_force_profile OUT [case ...] [--accelerated]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SI_POTENTIAL = ROOT / "examples" / "3d_Si" / "Si_POTRE.DAT"

_DIAMOND = np.array(
    [
        [0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
    ]
)


def _input_text(
    cells: int,
    spacing: float,
    kpoints: str,
    states: int,
    criterion: float,
    extra: str = "",
    max_iterations: int = 60,
) -> str:
    side = 10.26 * cells
    fractions = []
    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                for site in _DIAMOND:
                    fractions.append((site + np.array([i, j, k])) / cells)
    fractions = np.array(fractions)
    fractions[0] += np.array([0.012, 0.007, -0.009]) / cells
    fractions[5] += np.array([-0.006, 0.010, 0.004]) / cells
    coordinates = "\n".join(f"  {x:.10f} {y:.10f} {z:.10f}" for x, y, z in np.mod(fractions, 1.0))
    return f"""Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  {side} {side} {side}
end Cell_Shape
Coordinate_Unit: lattice_vectors
Grid_Spacing: {spacing}
Expansion_Order: 12
Ignore_Symmetry: .true.
{kpoints}
States_Num: {states}
Fermi_Temp: 300.0
Max_Iter: {max_iterations}
Convergence_Criterion: {criterion}
Diag_Tolerance: {criterion}
Mixing_Method: Anderson
Mixing_Param: 0.3
Correlation_Type: ca
Atom_Types_Num: 1
Atom_Type: Si
Pseudopotential_Format: Martins_new
Local_Component: p
begin Atom_Coord
{coordinates}
end Atom_Coord
Skip_force: .false.
Output_Level: 1
{extra}
"""


_MP2 = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n  2 2 2\nend Monkhorst_Pack_Grid"
_TWO_K = (
    "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n"
    "  0.0 0.0 0.0 1.0\n  0.5 0.0 0.0 1.0\nend Kpoint_List"
)

CASES = {
    "si8_g": dict(cells=1, spacing=0.45, kpoints="Kpoint_Method: none", states=24, criterion=1.0e-6),
    "si8_k": dict(cells=1, spacing=0.45, kpoints=_MP2, states=24, criterion=1.0e-6),
    "si64_g": dict(cells=2, spacing=0.45, kpoints="Kpoint_Method: none", states=150, criterion=1.0e-5),
    "si64_k": dict(cells=2, spacing=0.45, kpoints=_TWO_K, states=150, criterion=1.0e-5),
    # Si216: criterion (and Diag_Tolerance) 1e-6 since 2026-10-30, with up to 150 steps.  At 1e-4 the
    # SCF stopped while the energy still moved by 2e-4 Ry per step (CHEBDAV) and 2e-2 Ry (CHEBFF), and
    # the totals of different first solves differed by up to 1.7 Ry.  Measured with this template at
    # h 0.5 (CHEBDAV / CHEBFF 4 x 30 first solves, error of the total against a run to 1e-9):
    #   Si64 (256 electrons): 1e-3 / 1.5e-3 Ry at 1e-4 (14 / 24 steps), 8e-5 / 8.5e-4 at 1e-5,
    #     3.5e-6 / 7.7e-6 at 1e-6 (35 / 64 steps; the two 4.3e-6 Ry apart), 9e-9 / 6e-7 at 1e-7
    #     (63 / 86 steps; CHEBFF 4 x 30 does not reach 1e-8 in 100 steps);
    #   Si8 (32 electrons): 1.7e-5 / 1.8e-5 Ry at 1e-4, 6.5e-7 / 3.4e-7 at 1e-6.
    # The error grows faster than the cell (x60-85 from Si8 to Si64 at 1e-4, x5-23 at 1e-6); with
    # Si216's 3.4 times Si64's electrons, 1e-6 leaves its totals within about 1e-5 to 1e-4 Ry (1e-4
    # left them within 1e-2 to 1 Ry).  1e-6 is the tightest criterion that both first solves reach in
    # a bounded number of steps (the report of the parity round).
    "si216_g": dict(
        cells=3, spacing=0.5, kpoints="Kpoint_Method: none", states=470, criterion=1.0e-6, max_iterations=150
    ),
    # The first CHEBDAV solve of the accelerated path stopped on its orthogonalization limit for
    # si216_g (before the fix of the literal fallback): the first-solve controls of the GPU benchmarks.
    # At 1e-6 these controls stop 7.0e-3 Ry above the converged total (A100, 69 steps): the density
    # criterion is met while the subspace still converges slowly (284 steps to 4e-8 Ry at 1e-8).
    # CHEBFF 9 x 40 stops within 2e-5 Ry in 65 steps; compare totals with that, not with this entry.
    "si216c_g": dict(
        cells=3,
        spacing=0.5,
        kpoints="Kpoint_Method: none",
        states=470,
        criterion=1.0e-6,
        max_iterations=150,
        extra="Eigensolver: chebff\nFF_MaxIter: 4\nChebdav_Degree: 30\n",
    ),
}


class Clock:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def wrap(self, module, name: str, label: str | None = None):
        label = label or name
        original = getattr(module, name)
        if getattr(original, "_profile_clock", None) is self:
            return original

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.seconds[label] = self.seconds.get(label, 0.0) + time.perf_counter() - started
                self.calls[label] = self.calls.get(label, 0) + 1

        timed._profile_clock = self
        setattr(module, name, timed)
        return original

    def take(self) -> dict[str, float]:
        taken = {name: round(value, 4) for name, value in self.seconds.items()}
        self.seconds.clear()
        self.calls.clear()
        return taken


@contextmanager
def stopwatch(record: dict, name: str):
    started = time.perf_counter()
    yield
    record[name] = round(time.perf_counter() - started, 4)


def _device_peak() -> float | None:
    try:
        import cupy as cp
    except Exception:
        return None
    pool = cp.get_default_memory_pool()
    return round(pool.total_bytes() / 1e9, 3)


def _wrap_terms(clock: Clock) -> None:
    from parsec_python.Forces import periodic as periodic_forces
    from parsec_python.Forces import total as total_forces
    from parsec_python.SCF import kpoints as kpoint_driver

    for name in (
        "ewald_ion_ion_forces",
        "periodic_local_forces",
        "build_periodic_projector_gradients",
        "periodic_nonlocal_forces",
        "build_nonlocal_projectors",
        "periodic_force_operations",
        "periodic_force_fields",
        "reduced_kpoint_operations",
    ):
        clock.wrap(periodic_forces, name)
    clock.wrap(total_forces, "correction_forces")
    clock.wrap(kpoint_driver, "kept_vectors")


def _stages(system) -> dict:
    timings = getattr(system, "timings", None)
    if timings is None:
        return {}
    return {key: round(value, 4) for key, value in vars(timings).items() if isinstance(value, float)}


def _ionic_step(record: dict, clock: Clock, system, backend) -> None:
    """Cold SCF, forces, one move, warm SCF, forces: seconds of each part into ``record``."""

    from dataclasses import replace

    clock.take()
    with stopwatch(record, "scf_cold"):
        result = backend.run_scf(system)
    record["scf_cold_iterations"] = int(result.iterations)
    record["scf_cold_inner"] = clock.take()
    record["energy_cold"] = float(result.energies.total)
    with stopwatch(record, "forces_cold"):
        forces = backend.compute_forces(system, result, symmetrize=True)
    record["forces_cold_terms"] = clock.take()
    record["forces_cold_phases"] = {
        key: round(float(value), 4) for key, value in (getattr(forces, "phase_seconds", None) or {}).items()
    }
    record["forces_cold_max"] = float(np.max(np.abs(forces.total)))
    record["forces_cold_total"] = np.asarray(forces.total).tolist()

    atoms = list(system.input.atoms)
    moved = np.asarray(atoms[1].position, dtype=float) + np.array([0.02, -0.01, 0.015])
    atoms[1] = replace(atoms[1], position=moved)
    with stopwatch(record, "update_geometry"):
        new_system = backend.update_geometry(system, tuple(atoms))
    record["update_stages"] = _stages(getattr(new_system, "reference", new_system))
    record["update_inner"] = clock.take()
    with stopwatch(record, "warm_start"):
        start = backend.warm_start(system, result, new_system)
    with stopwatch(record, "scf_warm"):
        warm = backend.run_scf(new_system, warm_start=start)
    record["scf_warm_iterations"] = int(warm.iterations)
    record["scf_warm_inner"] = clock.take()
    record["energy_warm"] = float(warm.energies.total)
    with stopwatch(record, "forces_warm"):
        forces = backend.compute_forces(new_system, warm, symmetrize=True)
    record["forces_warm_terms"] = clock.take()
    record["forces_warm_total"] = np.asarray(forces.total).tolist()
    record["device_pool_gb"] = _device_peak()


def profile_reference(directory: Path, clock: Clock) -> dict:
    """The readable periodic backend (the reference solver of the routed inputs)."""

    from parsec_python.Grid.kpoints import kpoint_sampling
    from parsec_python.Input import parse_parsec_input
    from parsec_python.Relaxation.backend import fortran_periodic_backend as periodic_backend
    from parsec_python.SCF.kpoint_solver import parallel_backend_report
    from parsec_python.SCF.pbc import prepare_periodic_single_point

    _wrap_terms(clock)
    translation = parse_parsec_input(directory / "parsec.in")
    record: dict = {"backend": parallel_backend_report()}
    with stopwatch(record, "prepare"):
        system = prepare_periodic_single_point(translation.problem)
    record["grid_points"] = int(system.grid.size)
    record["prepare_stages"] = _stages(system)
    k_sampling = kpoint_sampling(translation.problem)
    record["k_points"] = 1 if k_sampling is None else int(len(k_sampling[0]))
    _ionic_step(record, clock, system, periodic_backend(k_sampling))
    return record


def profile_accelerated(directory: Path, clock: Clock) -> dict:
    """The accelerated Gamma-point path (acceleration.driver with the periodic forces)."""

    from parsec_python.Input import parse_parsec_input
    from parsec_python.acceleration.driver import ionic_backend, prepare_single_point

    _wrap_terms(clock)
    translation = parse_parsec_input(directory / "parsec.in")
    record: dict = {}
    with stopwatch(record, "prepare"):
        system = prepare_single_point(translation.problem, symmetry="off")
    record["prepare_stages"] = _stages(system.reference)
    try:
        from parsec_python.acceleration.Forces.periodic_device import with_periodic_operations
    except ImportError:  # the tree before the periodic forces of the accelerated path
        record["device_pool_gb"] = _device_peak()
        with stopwatch(record, "scf_cold"):
            result = ionic_backend().run_scf(system)
        record["scf_cold_iterations"] = int(result.iterations)
        record["energy_cold"] = float(result.energies.total)
        return record
    _ionic_step(record, clock, system, with_periodic_operations(ionic_backend()))
    return record


def main(argv=None) -> int:
    from contextlib import nullcontext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("cases", nargs="*", default=["si8_g", "si8_k"])
    parser.add_argument("--accelerated", action="store_true", help="also the accelerated Gamma-point path")
    parser.add_argument("--device-scope", action="store_true", help="inside device_periodic_forces()")
    parser.add_argument("--skip-reference", action="store_true")
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)
    scope = nullcontext()
    if arguments.device_scope:
        from parsec_python.acceleration.Forces.periodic_device import device_periodic_forces

        scope = device_periodic_forces()
    clock = Clock()
    results = {}
    with scope:
        for case in arguments.cases:
            directory = arguments.out / case
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "parsec.in").write_text(_input_text(**CASES[case]))
            shutil.copy(SI_POTENTIAL, directory / "Si_POTRE.DAT")
            entry = {}
            if not arguments.skip_reference:
                entry["reference"] = profile_reference(directory, clock)
            if arguments.accelerated and "_g" in case:
                entry["accelerated"] = profile_accelerated(directory, clock)
            results[case] = entry
            print(case, json.dumps(entry, indent=1), flush=True)
    (arguments.out / "profile.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
