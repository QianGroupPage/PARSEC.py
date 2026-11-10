"""SCF runs of meta-GGAs on bulk Si (the cell of the QE + libxc references), and the cost of the orbital terms.

Writes the ``parsec.in`` of the fcc Si cell of the meta-GGA references
(``examples/3d_meta_gga``: a = 10.26 bohr, two atoms, the FHI98PP potential of ``src/pp_generation/NCPPs/PBE``
copied to ``examples/3d_meta_gga``, ``States_Num 8``, ``Fermi_Temp 10 K``) with
the given functional, grid and Gamma-centred mesh (``Kpoint_Method: mp``: the
irreducible wedge with the symmetrized density, as the ``_ibz`` inputs of the
references; ``--list`` writes the full mesh as an explicit list with k and -k
merged), runs the k-point loop in this process (on the GPUs where the
acceleration package finds them) and prints a JSON summary: energy, VBM/CBM and
gap (eV) on the mesh, iterations, wall times; with ``--term-timing`` the time of
one application of the device Bloch Hamiltonian to a block of vectors with and
without the meta-GGA terms; with ``--program`` also the wall time of
``main.py`` on the same input in a subprocess, run before the in-process loop (a first run).

Usage: ``python -m parsec_python.acceleration.benchmarks.meta_gga_periodic_check --functional scan --kmesh 4
--spacing 0.25 [--gradient density|orbital] [--term-timing] [--program] [--out DIR]``.
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

RYDBERG_EV = 13.605693122994

FUNCTIONALS = {
    "pbe": "Correlation_Type: pbe",
    "scan": "Correlation_Type: scan",
    "scan_libxc": "LibXC_Functional: mgga_x_scan mgga_c_scan",
    "r2scan": "LibXC_Functional: mgga_x_r2scan mgga_c_r2scan",
    "tpss": "LibXC_Functional: mgga_x_tpss mgga_c_tpss",
    "pbe_libxc": "LibXC_Functional: gga_x_pbe gga_c_pbe",
}

TEMPLATE = """# bulk Si, {functional}, {mesh}: the cell of the meta-GGA references (refs/mgga, QE 7.5 + libxc)
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  0.0000000000 5.1300000000 5.1300000000
  5.1300000000 0.0000000000 5.1300000000
  5.1300000000 5.1300000000 0.0000000000
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: {spacing}
Expansion_Order: {order}
{functional_line}
States_Num: 8
Fermi_Temp: 10 K
Convergence_Criterion: {criterion} Ry
Diag_Tolerance: 1e-6
Max_Iter: {max_iterations}
{kpoints}{extra}
Atom_Types_Num: 1

Atom_Type: Si
Pseudopotential_Format: fhipp
Potential_Num: 3
Local_Component: d
Core_Cutoff_Radius: 2.2
begin Electron_Per_Orbital
2 2 0
end Electron_Per_Orbital
Cubic_Spline: .true.
begin Atom_Coord
     0.0000000000    0.0000000000    0.0000000000
     2.5650000000    2.5650000000    2.5650000000
end Atom_Coord
"""


def kpoint_lines(mesh: int, explicit: bool) -> str:
    if mesh <= 1:
        return ""
    if not explicit:
        return (
            "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n"
            f"  {mesh} {mesh} {mesh}\nend Monkhorst_Pack_Grid\n"
            "begin Monkhorst_Pack_Shift\n  0.0 0.0 0.0\nend Monkhorst_Pack_Shift\n"
        )
    # The Gamma-centred mesh folded into (-1/2, 1/2], k and -k merged (weight 2).
    points: dict[tuple, float] = {}
    for i in range(mesh):
        for j in range(mesh):
            for k in range(mesh):
                point = tuple(((value / mesh + 0.5) % 1.0) - 0.5 for value in (i, j, k))
                point = tuple(0.5 if abs(value + 0.5) < 1e-12 else value for value in point)
                mirror = tuple(0.5 if abs(value - 0.5) < 1e-12 else -value for value in point)
                key = min(point, mirror)
                points[key] = points.get(key, 0.0) + 1.0
    rows = "\n".join(f"  {a:.10f} {b:.10f} {c:.10f} {weight:.1f}" for (a, b, c), weight in sorted(points.items()))
    return (
        "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n"
        f"{rows}\nend Kpoint_List\n"
    )


def write_input(directory: Path, arguments) -> Path:
    root = Path(__file__).resolve().parents[4]
    shutil.copy(root / "examples" / "3d_meta_gga" / "Si_FHIPP.DAT", directory / "Si_FHIPP.DAT")
    text = TEMPLATE.format(
        functional=arguments.functional,
        mesh=f"{arguments.kmesh}x{arguments.kmesh}x{arguments.kmesh}",
        spacing=arguments.spacing,
        order=arguments.order,
        functional_line=FUNCTIONALS[arguments.functional],
        criterion=arguments.criterion,
        max_iterations=arguments.max_iterations,
        kpoints=kpoint_lines(arguments.kmesh, arguments.list),
        extra="".join(line + "\n" for line in arguments.extra),
    )
    path = directory / "parsec.in"
    path.write_text(text)
    return path


def band_edges(result, n_kpoints: int) -> dict:
    eigenvalues = np.asarray(result.eigenvalues, dtype=np.float64).reshape(n_kpoints, -1)
    occupations = np.asarray(result.occupations, dtype=np.float64).reshape(n_kpoints, -1)
    occupied = occupations > 0.5
    valence = float(np.max(eigenvalues[occupied]))
    conduction = float(np.min(eigenvalues[~occupied])) if np.any(~occupied) else float("nan")
    lowest = float(np.min(eigenvalues))
    return {
        "vbm_eV": valence * RYDBERG_EV,
        "cbm_eV": conduction * RYDBERG_EV,
        "gap_eV": (conduction - valence) * RYDBERG_EV,
        "valence_width_eV": (valence - lowest) * RYDBERG_EV,
    }


def term_timing(system, result, k_point, columns: int, repeats: int) -> dict:
    """Seconds of one device application of the Bloch Hamiltonian with and without the meta-GGA terms."""

    from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
    from parsec_python.V_ion import build_nonlocal_projectors

    from ..Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter, NumpyArrayBackend

    try:
        backend = CupyArrayBackend(0)
    except Exception:
        backend = NumpyArrayBackend()
    converter = DeviceBlochConverter(backend)
    projectors = build_nonlocal_projectors(
        system.grid,
        system.cell_atoms,
        system.pseudopotentials,
        system.input.pseudopotentials,
        system.input.periodic_cell.lattice_vectors,
        k_point=k_point,
        periodic_dimensions=system.input.periodic_cell.dimensions,
    )
    record = result.meta_gga
    vtau = np.asarray(record.kinetic_potentials[0]) if record is not None else 0.5 * np.ones(system.grid.size)
    flux = tuple(0.01 * vtau for _ in range(3))
    potential = np.asarray(result.input_effective_potential, dtype=np.float64)
    variants = {
        "plain": {},
        "tau": {"kinetic_potential": vtau},
        "tau_and_gradient": {"kinetic_potential": vtau, "gradient_flux": flux},
    }
    rng = np.random.default_rng(0)
    host_block = rng.standard_normal((system.grid.size, columns)) + 1j * rng.standard_normal((system.grid.size, columns))
    block = backend.asarray(host_block)
    timings = {}
    for name, keywords in variants.items():
        hamiltonian = KPointKohnShamHamiltonian(
            system.negative_laplacian, system.gradient, potential, projectors, np.asarray(k_point), **keywords
        )
        operator = converter.operator(hamiltonian)
        operator.apply(block)
        backend.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            operator.apply(block)
        backend.synchronize()
        timings[name] = (time.perf_counter() - started) / repeats
    return {
        "backend": backend.name,
        "columns": columns,
        "seconds_per_apply": timings,
        "tau_term_fraction": (timings["tau"] - timings["plain"]) / timings["plain"],
        "gradient_term_fraction": (timings["tau_and_gradient"] - timings["tau"]) / timings["plain"],
    }


def run_program(path: Path) -> dict:
    root = Path(__file__).resolve().parents[3]
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(root / "parsec_python" / "main.py"), str(path)],
        cwd=str(path.parent),
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - started
    reported = None
    for line in completed.stdout.splitlines():
        if "Total Python wall time" in line:
            reported = float(line.split(":")[-1])
    return {"returncode": completed.returncode, "wall_seconds": wall, "reported_seconds": reported,
            "stderr_tail": completed.stderr[-2000:]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--functional", default="scan", help="one of FUNCTIONALS, or several separated by commas")
    parser.add_argument("--kmesh", default="2", help="n of the n x n x n mesh, or several separated by commas")
    parser.add_argument("--list", action="store_true", help="the full mesh as an explicit list (no symmetrization)")
    parser.add_argument("--spacing", default="0.25", help="bohr, or several separated by commas")
    parser.add_argument("--order", type=int, default=12)
    parser.add_argument("--criterion", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--gradient", default=None, help="density, orbital, or both separated by a comma")
    parser.add_argument("--extra", action="append", default=[], help="a further input line")
    parser.add_argument("--term-timing", action="store_true")
    parser.add_argument("--term-columns", type=int, default=16)
    parser.add_argument("--program", action="store_true", help="also time main.py on the same input")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--label", default=None)
    arguments = parser.parse_args(argv)
    import itertools

    import parsec_python.acceleration  # noqa: F401  (installs the device k-point solver)

    status = 0
    gradients = [None] if arguments.gradient is None else arguments.gradient.split(",")
    for functional, kmesh, spacing, gradient in itertools.product(
        arguments.functional.split(","), arguments.kmesh.split(","), arguments.spacing.split(","), gradients
    ):
        if functional not in FUNCTIONALS:
            raise SystemExit(f"unknown functional {functional!r}: {sorted(FUNCTIONALS)}")
        case = argparse.Namespace(**vars(arguments))
        case.functional, case.kmesh, case.spacing = functional, int(kmesh), float(spacing)
        case.gradient = None if functional in ("pbe", "pbe_libxc") else gradient
        if case.gradient is not None:
            os.environ["PARSEC_META_GGA_GRADIENT"] = case.gradient
        else:
            os.environ.pop("PARSEC_META_GGA_GRADIENT", None)
        status = max(status, run_case(case))
    return status


def run_case(arguments) -> int:
    from parsec_python.Grid import kpoint_sampling
    from parsec_python.Input import parse_parsec_input
    from parsec_python.SCF import prepare_periodic_single_point
    from parsec_python.SCF.kpoints import run_scf_kpoints

    label = (f"{arguments.label}_" if arguments.label else "") + (
        f"Si_{arguments.functional}_k{arguments.kmesh}{'list' if arguments.list else ''}_h{arguments.spacing}"
        + (f"_{arguments.gradient}" if arguments.gradient else "")
    )
    directory = Path(tempfile.mkdtemp(prefix="parsec_mgga_si_"))
    path = write_input(directory, arguments)
    # main.py first: its wall time is that of a first run in a new process (the in-process run follows)
    program = run_program(path) if arguments.program else None
    started = time.perf_counter()
    translation = parse_parsec_input(path)
    problem = translation.problem
    system = prepare_periodic_single_point(problem)
    prepared = time.perf_counter() - started
    sampling = kpoint_sampling(problem) or (np.zeros((1, 3)), np.ones(1))
    history = []

    def report(step) -> None:
        history.append((step.iteration, float(step.weighted_residual), float(step.energies.total)))
        print(f"{label} it {step.iteration:3d} sre {step.weighted_residual:.3e} E {step.energies.total:.10f}", flush=True)

    scf_started = time.perf_counter()
    result = run_scf_kpoints(system, sampling[0], sampling[1], callback=report)
    scf_seconds = time.perf_counter() - scf_started
    timings = result.timings
    record = getattr(result, "meta_gga", None)
    summary = {
        "label": label,
        "functional": arguments.functional,
        "gradient": None if record is None else record.gradient,
        "kmesh": arguments.kmesh,
        "explicit_list": arguments.list,
        "n_kpoints": int(len(sampling[0])),
        "spacing": arguments.spacing,
        "grid_points": int(system.grid.size),
        "grid": type(system.grid).__name__,
        "fortran_routines": bool(getattr(translation.routines, "fortran", False)),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "energy_Ry": float(result.energies.total),
        "energies": {name: float(value) for name, value in vars(result.energies).items()},
        **band_edges(result, len(sampling[0])),
        "prepare_seconds": prepared,
        "scf_seconds": scf_seconds,
        "stages": None
        if timings is None
        else {
            "diagonalization": timings.diagonalization_seconds,
            "density_and_tau": timings.occupations_density_seconds,
            "hartree": timings.hartree_seconds,
            "xc": timings.xc_seconds,
            "mixing_energy": timings.mixing_energy_seconds,
            "detail": dict(timings.detail),
        },
        "history": history,
        "meta_gga": None
        if record is None
        else {"vtau_residual": record.kinetic_residual, "clamped": record.fermi_hole_clamped},
    }
    if arguments.term_timing:
        summary["term_timing"] = term_timing(system, result, sampling[0][-1], arguments.term_columns, 20)
    if program is not None:
        summary["program"] = program
    text = json.dumps(summary, indent=1, default=float)
    print("SUMMARY " + json.dumps({key: summary[key] for key in summary if key not in ("history", "energies")}, default=float))
    if arguments.out is not None:
        arguments.out.mkdir(parents=True, exist_ok=True)
        (arguments.out / f"{label}.json").write_text(text)
    shutil.rmtree(directory, ignore_errors=True)
    return 0 if result.converged else 3


if __name__ == "__main__":
    raise SystemExit(main())
