"""Project numerically symmetry-broken xTB geometries onto their exact point group.

The ideal diamond cut defines atom permutations for every signed Cartesian
permutation that preserves element labels.  Applying the corresponding group
projector to a tightly optimized geometry removes only optimizer round-off and
writes a separate ``optimized_symmetrized`` tree; raw xTB results are never
modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from scipy.spatial import cKDTree

import generate_nanodiamonds as generator
import optimize_nanodiamonds as optimizer


def _signed_permutations() -> tuple[np.ndarray, ...]:
    from itertools import permutations, product

    operations = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            operations.append(matrix)
    return tuple(operations)


def _ideal_group_mappings(ideal: Atoms, tolerance: float = 1.0e-7):
    positions = np.asarray(ideal.positions, dtype=np.float64)
    symbols = np.asarray(ideal.get_chemical_symbols())
    operations_and_mappings = []
    for operation in _signed_permutations():
        mapping = np.full(len(ideal), -1, dtype=np.int64)
        accepted = True
        for symbol in sorted(set(symbols)):
            indices = np.flatnonzero(symbols == symbol)
            tree = cKDTree(positions[indices])
            distances, local_rows = tree.query(positions[indices] @ operation.T, k=1)
            if float(np.max(distances, initial=0.0)) > tolerance:
                accepted = False
                break
            mapped = indices[np.asarray(local_rows, dtype=np.int64)]
            if np.unique(mapped).size != indices.size:
                accepted = False
                break
            mapping[indices] = mapped
        if accepted and np.all(mapping >= 0):
            operations_and_mappings.append((operation, mapping))
    if not operations_and_mappings:
        raise RuntimeError("the identity operation did not preserve the ideal geometry")
    return operations_and_mappings


def symmetrize(case_root: Path, vacuum_padding: float) -> dict[str, object]:
    source_root = case_root / "optimized"
    target_root = case_root / "optimized_symmetrized"
    if target_root.exists():
        raise FileExistsError(f"refusing to overwrite {target_root}")
    ideal = read(case_root / "structure.xyz", format="xyz")
    raw = read(source_root / "structure_gfn2.xyz", format="xyz")
    if ideal.get_chemical_symbols() != raw.get_chemical_symbols():
        raise ValueError(f"atom order changed in {case_root.name}")

    group = _ideal_group_mappings(ideal)
    raw_positions = np.asarray(raw.positions, dtype=np.float64)
    projected = np.zeros_like(raw_positions)
    before = 0.0
    for operation, mapping in group:
        transformed = raw_positions @ operation.T
        before = max(before, float(np.max(np.linalg.norm(transformed - raw_positions[mapping], axis=1))))
        projected[mapping] += transformed
    projected /= len(group)

    after = 0.0
    for operation, mapping in group:
        transformed = projected @ operation.T
        after = max(after, float(np.max(np.linalg.norm(transformed - projected[mapping], axis=1))))

    displacement = np.linalg.norm(projected - raw_positions, axis=1)
    symmetrized = raw.copy()
    symmetrized.positions[:] = projected
    audit = optimizer._geometry_audit(ideal, symmetrized)
    maximum_radius = float(np.max(np.linalg.norm(projected, axis=1)))
    domain_radius = math.ceil(10.0 * (maximum_radius + vacuum_padding) - 1.0e-12) / 10.0
    box_length = 2.0 * domain_radius
    symmetrized.cell = [box_length] * 3
    symmetrized.pbc = False

    target_root.mkdir(parents=True)
    write(
        target_root / "structure_gfn2.xyz",
        symmetrized,
        format="xyz",
        comment=f"symmetry-projected GFN2-xTB optimized {case_root.name}",
    )
    write(target_root / "structure_gfn2.extxyz", symmetrized, format="extxyz")
    poscar = symmetrized.copy()
    poscar.positions += 0.5 * box_length
    write(target_root / "POSCAR", poscar, format="vasp", direct=False, sort=True)

    original = json.loads((case_root / "geometry_audit.json").read_text(encoding="utf-8"))
    n_carbon = int(original["n_carbon"])
    cluster = generator.Cluster(
        multiplier=int(original["radius_multiplier"]),
        cutoff_radius_angstrom=float(original["cutoff_radius_angstrom"]),
        seed_n_carbon=int(original["seed_n_carbon"]),
        surface_completion_passes=int(original["surface_completion_passes"]),
        carbon_positions=np.asarray(projected[:n_carbon]),
        hydrogen_positions=np.asarray(projected[n_carbon:]),
        carbon_coordination=(),
        cc_edges=(),
        ch_edges=(),
    )
    parsec_text = generator._parsec_input(cluster, domain_radius).replace(
        "# Ideal unrelaxed hydrogen-terminated nanodiamond",
        "# Symmetry-projected GFN2-xTB optimized hydrogen-terminated nanodiamond",
        1,
    )
    (target_root / "parsec.in").write_text(parsec_text, encoding="utf-8")

    summary = {
        "schema": "parsec_python.nanodiamond_symmetry_projection.v1",
        "case": case_root.name,
        "source": "../optimized/structure_gfn2.xyz",
        "signed_permutation_group_order": len(group),
        "maximum_symmetry_violation_before_angstrom": before,
        "maximum_symmetry_violation_after_angstrom": after,
        "rms_projection_displacement_angstrom": float(np.sqrt(np.mean(displacement**2))),
        "maximum_projection_displacement_angstrom": float(np.max(displacement)),
        "maximum_atomic_radius_angstrom": maximum_radius,
        "parsec_domain_radius_angstrom": domain_radius,
        "topology_preserved": bool(audit["original_connectivity_preserved"]),
        "carbon_coordination_values": audit["carbon_coordination_values"],
        "hydrogen_coordination_values": audit["hydrogen_coordination_values"],
    }
    if after > 1.0e-10 or not summary["topology_preserved"]:
        raise ValueError(f"symmetry/topology audit failed for {case_root.name}")
    (target_root / "symmetrization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="folder holding the case folders (default: this script's folder)",
    )
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--vacuum-padding", type=float, default=5.0)
    args = parser.parse_args()
    root = args.root.resolve()
    for case_name in args.case:
        result = symmetrize(root / case_name, args.vacuum_padding)
        print(
            f"{case_name}: order={result['signed_permutation_group_order']}, "
            f"max shift={result['maximum_projection_displacement_angstrom']:.3e} A"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
