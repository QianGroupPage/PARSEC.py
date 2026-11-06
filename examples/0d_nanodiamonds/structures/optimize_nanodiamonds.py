"""Optimize generated nanodiamonds without modifying their ideal inputs.

Small clusters are optimized directly with GFN2-xTB.  Clusters at or above a
configurable atom-count threshold first receive a GFN-FF geometry
preoptimization, followed by a tight GFN2-xTB optimization.  Every result is
written below the case-local ``optimized`` directory; the original XYZ,
POSCAR, audit, and PARSEC input remain untouched.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from ase import Atoms
from ase.io import read, write

import generate_nanodiamonds as generator


@dataclass(frozen=True)
class StageResult:
    name: str
    method: str
    opt_level: str
    input_path: Path
    output_path: Path
    log_path: Path
    command: tuple[str, ...]
    wall_seconds: float
    energy_hartree: float | None
    gradient_norm_hartree_per_bohr: float | None


def _xtb_environment(executable: Path, threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    environment["OMP_MAX_ACTIVE_LEVELS"] = "1"
    # The Windows xTB build otherwise inherits a small OpenMP worker stack.
    # GFN-FF's fragmented model-Hessian construction can exhaust it for a few
    # hundred atoms even though the actual arrays fit comfortably in memory.
    # These variables reserve virtual address space per worker; pages are only
    # committed when used and the electronic/force-field model is unchanged.
    environment.setdefault("OMP_STACKSIZE", "512M")
    environment.setdefault("KMP_STACKSIZE", "512M")
    if executable.is_file():
        installation_root = executable.parent.parent
        parameter_root = installation_root / "share" / "xtb"
        if parameter_root.is_dir():
            environment["XTBPATH"] = str(parameter_root)
        environment["PATH"] = (
            str(executable.parent)
            + os.pathsep
            + environment.get("PATH", "")
        )
    return environment


def _xtb_version(executable: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    match = re.search(r"xtb version\s+([^\s]+)(?:\s+\(([^)]+)\))?", result.stdout)
    if match is None:
        raise RuntimeError("could not parse xTB version")
    return " ".join(item for item in match.groups() if item)


def _last_number(log_text: str, label: str) -> float | None:
    pattern = rf"{re.escape(label)}\s+(-?[0-9]+(?:\.[0-9]*)?(?:[Ee][+-]?[0-9]+)?)"
    matches = re.findall(pattern, log_text, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else None


def _run_stage(
    *,
    stage_root: Path,
    source_geometry: Path,
    executable: Path,
    environment: dict[str, str],
    method: str,
    opt_level: str,
) -> StageResult:
    stage_root.mkdir(parents=True, exist_ok=False)
    input_atoms = read(source_geometry, format="xyz")
    input_path = stage_root / "input.xyz"
    write(input_path, input_atoms, format="xyz", comment=f"input for {method}")
    log_path = stage_root / "xtb.out"
    if method == "GFN-FF":
        method_arguments = ["--gfnff"]
    elif method == "GFN2-xTB":
        method_arguments = ["--gfn", "2"]
    else:
        raise ValueError(f"unsupported optimization method: {method}")
    command = (
        str(executable),
        "input.xyz",
        *method_arguments,
        "--chrg",
        "0",
        "--uhf",
        "0",
        "--opt",
        opt_level,
    )
    print(
        f"    {method} --opt {opt_level} ({len(input_atoms)} atoms) ...",
        flush=True,
    )
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=stage_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wall_seconds = time.perf_counter() - start
    log_path.write_text(result.stdout, encoding="utf-8")
    output_path = stage_root / "xtbopt.xyz"
    converged = "GEOMETRY OPTIMIZATION CONVERGED" in result.stdout
    normal = "normal termination of xtb" in result.stdout
    if result.returncode != 0 or not converged or not normal or not output_path.is_file():
        raise RuntimeError(
            f"{method} failed in {stage_root}; return code={result.returncode}, "
            f"converged={converged}, normal_termination={normal}. See {log_path}."
        )
    energy = _last_number(result.stdout, "TOTAL ENERGY")
    gradient = _last_number(result.stdout, "GRADIENT NORM")
    print(
        f"      converged in {wall_seconds:.2f} s; "
        f"E={energy if energy is not None else float('nan'):.12f} Eh",
        flush=True,
    )
    return StageResult(
        name=stage_root.name,
        method=method,
        opt_level=opt_level,
        input_path=input_path,
        output_path=output_path,
        log_path=log_path,
        command=command,
        wall_seconds=wall_seconds,
        energy_hartree=energy,
        gradient_norm_hartree_per_bohr=gradient,
    )


def _mass_weighted_alignment(reference: Atoms, mobile: Atoms) -> tuple[Atoms, float, float]:
    if reference.get_chemical_symbols() != mobile.get_chemical_symbols():
        raise ValueError("xTB changed atom ordering or chemical symbols")
    weights = np.asarray(reference.get_masses(), dtype=np.float64)
    reference_positions = np.asarray(reference.positions, dtype=np.float64)
    mobile_positions = np.asarray(mobile.positions, dtype=np.float64)
    reference_center = np.average(reference_positions, axis=0, weights=weights)
    mobile_center = np.average(mobile_positions, axis=0, weights=weights)
    reference_centered = reference_positions - reference_center
    mobile_centered = mobile_positions - mobile_center
    covariance = (mobile_centered * weights[:, None]).T @ reference_centered
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    aligned_positions = mobile_centered @ rotation
    # Match the isolated-cluster centering convention used by PARSEC.py.
    aligned_positions -= 0.5 * (
        aligned_positions.min(axis=0) + aligned_positions.max(axis=0)
    )
    reference_for_comparison = reference_centered @ np.eye(3)
    reference_for_comparison -= 0.5 * (
        reference_for_comparison.min(axis=0) + reference_for_comparison.max(axis=0)
    )
    displacements = aligned_positions - reference_for_comparison
    rmsd = float(
        math.sqrt(np.sum(weights[:, None] * displacements**2) / np.sum(weights))
    )
    maximum_displacement = float(np.max(np.linalg.norm(displacements, axis=1)))
    aligned = mobile.copy()
    aligned.positions = aligned_positions
    return aligned, rmsd, maximum_displacement


def _pair_distances(positions: np.ndarray) -> np.ndarray:
    displacement = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(displacement, axis=2)


def _bond_sets(atoms: Atoms) -> dict[str, set[tuple[int, int]]]:
    symbols = atoms.get_chemical_symbols()
    distances = _pair_distances(np.asarray(atoms.positions, dtype=np.float64))
    bonds: dict[str, set[tuple[int, int]]] = {"CC": set(), "CH": set(), "HH": set()}
    for atom_i in range(len(atoms)):
        for atom_j in range(atom_i + 1, len(atoms)):
            pair = "".join(sorted((symbols[atom_i], symbols[atom_j])))
            cutoff = {"CC": 1.85, "CH": 1.30, "HH": 0.95}[pair]
            if distances[atom_i, atom_j] <= cutoff:
                bonds[pair].add((atom_i, atom_j))
    return bonds


def _range_for_edges(atoms: Atoms, edges: set[tuple[int, int]]) -> list[float] | None:
    if not edges:
        return None
    positions = np.asarray(atoms.positions, dtype=np.float64)
    values = [float(np.linalg.norm(positions[j] - positions[i])) for i, j in edges]
    return [min(values), float(np.mean(values)), max(values)]


def _geometry_audit(initial: Atoms, final: Atoms) -> dict[str, object]:
    initial_bonds = _bond_sets(initial)
    final_bonds = _bond_sets(final)
    symbols = final.get_chemical_symbols()
    carbon_indices = [index for index, symbol in enumerate(symbols) if symbol == "C"]
    hydrogen_indices = [index for index, symbol in enumerate(symbols) if symbol == "H"]
    positions = np.asarray(final.positions, dtype=np.float64)
    distances = _pair_distances(positions)
    np.fill_diagonal(distances, np.inf)
    coordination = np.zeros(len(final), dtype=int)
    for pair_type in ("CC", "CH"):
        for atom_i, atom_j in final_bonds[pair_type]:
            coordination[atom_i] += 1
            coordination[atom_j] += 1
    audit: dict[str, object] = {
        "schema": "parsec_python.nanodiamond_optimized_geometry.v1",
        "n_atoms": len(final),
        "n_carbon": len(carbon_indices),
        "n_hydrogen": len(hydrogen_indices),
        "original_connectivity_preserved": all(
            initial_bonds[key] == final_bonds[key] for key in initial_bonds
        ),
        "initial_bond_counts": {key: len(value) for key, value in initial_bonds.items()},
        "final_bond_counts": {key: len(value) for key, value in final_bonds.items()},
        "broken_bonds": {
            key: sorted([list(edge) for edge in initial_bonds[key] - final_bonds[key]])
            for key in initial_bonds
        },
        "new_bonds": {
            key: sorted([list(edge) for edge in final_bonds[key] - initial_bonds[key]])
            for key in initial_bonds
        },
        "cc_bond_min_mean_max_angstrom": _range_for_edges(final, final_bonds["CC"]),
        "ch_bond_min_mean_max_angstrom": _range_for_edges(final, final_bonds["CH"]),
        "minimum_all_atom_distance_angstrom": float(np.min(distances)),
        "minimum_hydrogen_hydrogen_distance_angstrom": (
            float(np.min(distances[np.ix_(hydrogen_indices, hydrogen_indices)]))
            if len(hydrogen_indices) > 1
            else None
        ),
        "carbon_coordination_values": sorted(
            set(int(coordination[index]) for index in carbon_indices)
        ),
        "hydrogen_coordination_values": sorted(
            set(int(coordination[index]) for index in hydrogen_indices)
        ),
        "maximum_atomic_radius_angstrom": float(
            np.max(np.linalg.norm(positions, axis=1))
        ),
    }
    if not audit["original_connectivity_preserved"]:
        raise ValueError("optimization changed the intended C/H bond topology")
    if audit["carbon_coordination_values"] != [4]:
        raise ValueError(f"unexpected carbon coordination: {audit['carbon_coordination_values']}")
    if audit["hydrogen_coordination_values"] != [1]:
        raise ValueError(
            f"unexpected hydrogen coordination: {audit['hydrogen_coordination_values']}"
        )
    return audit


def _stage_dictionary(stage: StageResult, optimized_root: Path) -> dict[str, object]:
    return {
        "name": stage.name,
        "method": stage.method,
        "optimization_level": stage.opt_level,
        "input": str(stage.input_path.relative_to(optimized_root)),
        "output": str(stage.output_path.relative_to(optimized_root)),
        "log": str(stage.log_path.relative_to(optimized_root)),
        "command": list(stage.command[1:]),
        "wall_seconds": stage.wall_seconds,
        "final_energy_hartree": stage.energy_hartree,
        "final_gradient_norm_hartree_per_bohr": stage.gradient_norm_hartree_per_bohr,
        "converged": True,
    }


def _write_final_files(
    *,
    case_root: Path,
    initial: Atoms,
    optimized: Atoms,
    stage_results: list[StageResult],
    xtb_version: str,
    threads: int,
    vacuum_padding_angstrom: float,
) -> dict[str, object]:
    optimized_root = case_root / "optimized"
    aligned, rmsd, maximum_displacement = _mass_weighted_alignment(initial, optimized)
    audit = _geometry_audit(initial, aligned)
    maximum_radius = float(audit["maximum_atomic_radius_angstrom"])
    domain_radius = math.ceil(
        10.0 * (maximum_radius + vacuum_padding_angstrom)
    ) / 10.0
    box_length = 2.0 * domain_radius
    aligned.cell = [box_length] * 3
    aligned.pbc = False
    write(
        optimized_root / "structure_gfn2.xyz",
        aligned,
        format="xyz",
        comment=f"GFN2-xTB optimized {case_root.name}; charge=0; uhf=0",
    )
    write(optimized_root / "structure_gfn2.extxyz", aligned, format="extxyz")
    poscar = aligned.copy()
    poscar.positions += 0.5 * box_length
    write(optimized_root / "POSCAR", poscar, format="vasp", direct=False, sort=True)

    original_audit = json.loads(
        (case_root / "geometry_audit.json").read_text(encoding="utf-8")
    )
    n_carbon = int(original_audit["n_carbon"])
    carbon_positions = np.asarray(aligned.positions[:n_carbon], dtype=np.float64)
    hydrogen_positions = np.asarray(aligned.positions[n_carbon:], dtype=np.float64)
    template_cluster = generator.Cluster(
        multiplier=int(original_audit["radius_multiplier"]),
        cutoff_radius_angstrom=float(original_audit["cutoff_radius_angstrom"]),
        seed_n_carbon=int(original_audit["seed_n_carbon"]),
        surface_completion_passes=int(original_audit["surface_completion_passes"]),
        carbon_positions=carbon_positions,
        hydrogen_positions=hydrogen_positions,
        carbon_coordination=(),
        cc_edges=(),
        ch_edges=(),
    )
    parsec_text = generator._parsec_input(template_cluster, domain_radius)
    parsec_text = parsec_text.replace(
        "# Ideal unrelaxed hydrogen-terminated nanodiamond",
        "# GFN2-xTB optimized hydrogen-terminated nanodiamond",
        1,
    )
    (optimized_root / "parsec.in").write_text(parsec_text, encoding="utf-8")

    audit["mass_weighted_rmsd_from_ideal_angstrom"] = rmsd
    audit["maximum_displacement_from_ideal_angstrom"] = maximum_displacement
    audit["parsec_domain_radius_angstrom"] = domain_radius
    audit["display_box_length_angstrom"] = box_length
    (optimized_root / "geometry_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "parsec_python.nanodiamond_optimization.v1",
        "case": case_root.name,
        "xtb_version": xtb_version,
        "threads": threads,
        "charge": 0,
        "unpaired_electrons": 0,
        "stages": [
            _stage_dictionary(stage, optimized_root) for stage in stage_results
        ],
        "final_structure": "structure_gfn2.xyz",
        "parsec_input": "parsec.in",
        "original_files_modified": False,
        "geometry_audit_passed": True,
        "mass_weighted_rmsd_from_ideal_angstrom": rmsd,
        "maximum_displacement_from_ideal_angstrom": maximum_displacement,
        "total_wall_seconds": sum(stage.wall_seconds for stage in stage_results),
    }
    (optimized_root / "optimization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    final_stage = stage_results[-1]
    return {
        "case": case_root.name,
        "n_atoms": len(aligned),
        "workflow": "GFN-FF -> GFN2-xTB" if len(stage_results) == 2 else "GFN2-xTB",
        "gfn2_energy_hartree": f"{final_stage.energy_hartree:.12f}",
        "gfn2_gradient_hartree_per_bohr": (
            f"{final_stage.gradient_norm_hartree_per_bohr:.12e}"
        ),
        "rmsd_from_ideal_angstrom": f"{rmsd:.8f}",
        "maximum_displacement_angstrom": f"{maximum_displacement:.8f}",
        "domain_radius_angstrom": f"{domain_radius:.1f}",
        "wall_seconds": f"{summary['total_wall_seconds']:.3f}",
        "topology_preserved": True,
        "dry_run_validated": False,
    }


def optimize_case(
    *,
    case_root: Path,
    executable: Path,
    environment: dict[str, str],
    xtb_version: str,
    threads: int,
    preopt_min_atoms: int,
    gfnff_opt_level: str,
    gfn2_opt_level: str,
    vacuum_padding_angstrom: float,
) -> dict[str, object]:
    optimized_root = case_root / "optimized"
    if optimized_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing optimization directory: {optimized_root}"
        )
    initial_path = case_root / "structure.xyz"
    initial = read(initial_path, format="xyz")
    print(f"  {case_root.name}: {len(initial)} atoms", flush=True)
    stage_results: list[StageResult] = []
    source = initial_path
    if len(initial) >= preopt_min_atoms:
        gfnff = _run_stage(
            stage_root=optimized_root / "01_gfnff",
            source_geometry=source,
            executable=executable,
            environment=environment,
            method="GFN-FF",
            opt_level=gfnff_opt_level,
        )
        stage_results.append(gfnff)
        source = gfnff.output_path
    gfn2 = _run_stage(
        stage_root=optimized_root / ("02_gfn2" if stage_results else "01_gfn2"),
        source_geometry=source,
        executable=executable,
        environment=environment,
        method="GFN2-xTB",
        opt_level=gfn2_opt_level,
    )
    stage_results.append(gfn2)
    final = read(gfn2.output_path, format="xyz")
    return _write_final_files(
        case_root=case_root,
        initial=initial,
        optimized=final,
        stage_results=stage_results,
        xtb_version=xtb_version,
        threads=threads,
        vacuum_padding_angstrom=vacuum_padding_angstrom,
    )


def main() -> int:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xtb",
        type=Path,
        default=Path("xtb"),
        help="path to the xTB executable (default: xtb on PATH)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_root,
        help="folder holding the case folders and the manifest (default: this script's folder)",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="case directory name; repeat to select several (default: all)",
    )
    parser.add_argument(
        "--threads", type=int, default=max(1, (os.cpu_count() or 1) - 4)
    )
    parser.add_argument("--preopt-min-atoms", type=int, default=50)
    parser.add_argument("--gfnff-opt-level", default="normal")
    parser.add_argument("--gfn2-opt-level", default="tight")
    parser.add_argument(
        "--vacuum-padding",
        type=float,
        default=generator.DEFAULT_VACUUM_PADDING_ANGSTROM,
    )
    arguments = parser.parse_args()
    executable = arguments.xtb.expanduser().resolve() if arguments.xtb.is_file() else arguments.xtb
    environment = _xtb_environment(executable, arguments.threads)
    version = _xtb_version(executable, environment)
    root = arguments.root.resolve()
    cases = sorted(
        item
        for item in root.glob("k??_C*H*")
        if item.is_dir() and (item / "structure.xyz").is_file()
    )
    if arguments.case:
        requested = set(arguments.case)
        cases = [item for item in cases if item.name in requested]
        missing = requested - {item.name for item in cases}
        if missing:
            raise FileNotFoundError(f"unknown case(s): {', '.join(sorted(missing))}")
    if not cases:
        raise FileNotFoundError("no nanodiamond cases selected")
    print(f"xTB {version}; OpenMP threads={arguments.threads}", flush=True)
    rows = []
    for case_root in cases:
        rows.append(
            optimize_case(
                case_root=case_root,
                executable=executable,
                environment=environment,
                xtb_version=version,
                threads=arguments.threads,
                preopt_min_atoms=arguments.preopt_min_atoms,
                gfnff_opt_level=arguments.gfnff_opt_level,
                gfn2_opt_level=arguments.gfn2_opt_level,
                vacuum_padding_angstrom=arguments.vacuum_padding,
            )
        )
    manifest = root / "optimization_manifest.csv"
    merged_rows: dict[str, dict[str, object]] = {}
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as stream:
            for existing in csv.DictReader(stream):
                merged_rows[str(existing["case"])] = dict(existing)
    for row in rows:
        merged_rows[str(row["case"])] = row
    ordered_rows = [merged_rows[key] for key in sorted(merged_rows)]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(ordered_rows)
    print(f"Optimization manifest: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
