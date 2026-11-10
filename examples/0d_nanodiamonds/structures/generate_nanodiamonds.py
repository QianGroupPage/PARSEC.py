"""Generate ideal, hydrogen-terminated spherical diamond clusters.

The carbon skeleton is cut from an ASE bulk-diamond supercell about a carbon
site.  Candidate radii are integer multiples of the ideal nearest-neighbour
C--C distance.  Every bond from a retained carbon to a removed diamond-lattice
neighbour is terminated by one hydrogen along the missing-bond direction.

This script generates geometries, PARSEC input files, and geometry audits.  It
does not run a DFT calculation or relax the ideal structures.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from ase.neighborlist import neighbor_list


DEFAULT_LATTICE_CONSTANT_ANGSTROM = 3.567
DEFAULT_CH_BOND_ANGSTROM = 1.09
DEFAULT_GRID_SPACING_ANGSTROM = 0.20
DEFAULT_VACUUM_PADDING_ANGSTROM = 5.0
VALENCE_ELECTRONS = {"C": 4, "H": 1}


@dataclass(frozen=True)
class ReferenceLattice:
    atoms: Atoms
    relative_positions: np.ndarray
    adjacency: tuple[tuple[int, ...], ...]
    cc_bond_angstrom: float


@dataclass(frozen=True)
class Cluster:
    multiplier: int
    cutoff_radius_angstrom: float
    seed_n_carbon: int
    surface_completion_passes: int
    carbon_positions: np.ndarray
    hydrogen_positions: np.ndarray
    carbon_coordination: tuple[int, ...]
    cc_edges: tuple[tuple[int, int], ...]
    ch_edges: tuple[tuple[int, int], ...]

    @property
    def n_carbon(self) -> int:
        return int(self.carbon_positions.shape[0])

    @property
    def n_hydrogen(self) -> int:
        return int(self.hydrogen_positions.shape[0])

    @property
    def formula(self) -> str:
        return f"C{self.n_carbon}H{self.n_hydrogen}"


def _reference_lattice(
    lattice_constant_angstrom: float,
    maximum_multiplier: int,
) -> ReferenceLattice:
    """Build a finite bulk lattice large enough to surround every candidate.

    ASE's cubic diamond cell supplies the two interpenetrating FCC sublattices.
    The repeated reference is deliberately larger than the requested sphere by
    two C--C bonds so every retained surface carbon still has all four bulk
    neighbours available when dangling bonds are identified.
    """

    cc_bond = lattice_constant_angstrom * math.sqrt(3.0) / 4.0
    needed_half_width = (maximum_multiplier + 2) * cc_bond
    half_cells = math.ceil(needed_half_width / lattice_constant_angstrom)
    repeat = 2 * half_cells + 3

    atoms = bulk(
        "C", "diamond", a=lattice_constant_angstrom, cubic=True
    ).repeat((repeat, repeat, repeat))
    atoms.pbc = False
    positions = np.asarray(atoms.positions, dtype=np.float64)
    cell_center = 0.5 * np.asarray(atoms.cell.lengths(), dtype=np.float64)
    center_index = int(np.argmin(np.linalg.norm(positions - cell_center, axis=1)))
    relative_positions = positions - positions[center_index]

    # ASE returns both i->j and j->i for this request.  A 5% tolerance is far
    # below the diamond second-neighbour distance and robust to round-off.
    source, target = neighbor_list("ij", atoms, 1.05 * cc_bond)
    adjacency_lists: list[list[int]] = [[] for _ in range(len(atoms))]
    for atom_i, atom_j in zip(source, target, strict=True):
        adjacency_lists[int(atom_i)].append(int(atom_j))
    adjacency = tuple(tuple(sorted(items)) for items in adjacency_lists)

    maximum_radius = maximum_multiplier * cc_bond
    selected = np.flatnonzero(
        np.linalg.norm(relative_positions, axis=1) <= maximum_radius + 1.0e-9
    )
    if any(len(adjacency[int(index)]) != 4 for index in selected):
        raise RuntimeError("reference supercell is too small for the requested radius")
    return ReferenceLattice(atoms, relative_positions, adjacency, cc_bond)


def _build_cluster(
    lattice: ReferenceLattice,
    multiplier: int,
    ch_bond_angstrom: float,
) -> Cluster:
    """Crop one carbon sphere and saturate every missing tetrahedral bond."""

    cutoff = multiplier * lattice.cc_bond_angstrom
    radii = np.linalg.norm(lattice.relative_positions, axis=1)
    retained = np.flatnonzero(radii <= cutoff + 1.0e-9)
    retained_set = {int(index) for index in retained}
    seed_n_carbon = len(retained_set)

    # A raw Euclidean cut can leave one removed lattice site bonded to two or
    # more retained surface carbons.  Capping both bonds toward that same site
    # creates unphysical inward-facing hydrogens only ~0.74 A apart.  Complete
    # such concave sites with carbon, iterating because one addition can expose
    # another shared site.  This is a topology-only surface repair: it changes
    # neither the ideal lattice nor any bond direction.
    surface_completion_passes = 0
    while True:
        removed_neighbour_counts: dict[int, int] = {}
        for full_i in retained_set:
            for full_j in lattice.adjacency[full_i]:
                if full_j not in retained_set:
                    removed_neighbour_counts[full_j] = (
                        removed_neighbour_counts.get(full_j, 0) + 1
                    )
        additions = {
            full_j
            for full_j, count in removed_neighbour_counts.items()
            if count >= 2
        }
        if not additions:
            break
        retained_set.update(additions)
        surface_completion_passes += 1
    ordered = sorted(
        retained_set,
        key=lambda index: (
            round(float(radii[index]), 12),
            *(round(float(value), 12) for value in lattice.relative_positions[index]),
        ),
    )
    local_index = {full_index: index for index, full_index in enumerate(ordered)}

    carbon_positions = np.asarray(
        [lattice.relative_positions[index] for index in ordered], dtype=np.float64
    )
    hydrogen_positions: list[np.ndarray] = []
    carbon_coordination: list[int] = []
    cc_edges: list[tuple[int, int]] = []
    ch_edges: list[tuple[int, int]] = []

    for full_i in ordered:
        local_i = local_index[full_i]
        neighbours = lattice.adjacency[full_i]
        if len(neighbours) != 4:
            raise RuntimeError(f"carbon site {full_i} has {len(neighbours)} neighbours")
        retained_neighbours = [item for item in neighbours if item in retained_set]
        carbon_coordination.append(len(retained_neighbours))
        for full_j in retained_neighbours:
            local_j = local_index[full_j]
            if local_i < local_j:
                cc_edges.append((local_i, local_j))
        for full_j in neighbours:
            if full_j in retained_set:
                continue
            direction = (
                lattice.relative_positions[full_j]
                - lattice.relative_positions[full_i]
            )
            direction /= np.linalg.norm(direction)
            h_position = lattice.relative_positions[full_i] + ch_bond_angstrom * direction
            hydrogen_index = len(hydrogen_positions)
            hydrogen_positions.append(h_position)
            ch_edges.append((local_i, hydrogen_index))

    return Cluster(
        multiplier=multiplier,
        cutoff_radius_angstrom=cutoff,
        seed_n_carbon=seed_n_carbon,
        surface_completion_passes=surface_completion_passes,
        carbon_positions=carbon_positions,
        hydrogen_positions=np.asarray(hydrogen_positions, dtype=np.float64).reshape(-1, 3),
        carbon_coordination=tuple(carbon_coordination),
        cc_edges=tuple(cc_edges),
        ch_edges=tuple(ch_edges),
    )


def _minimum_pair_distance(positions: np.ndarray) -> float | None:
    if len(positions) < 2:
        return None
    displacements = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(displacements, axis=2)
    distances[np.diag_indices_from(distances)] = np.inf
    return float(np.min(distances))


def _connected_carbon_components(cluster: Cluster) -> int:
    neighbours: list[list[int]] = [[] for _ in range(cluster.n_carbon)]
    for atom_i, atom_j in cluster.cc_edges:
        neighbours[atom_i].append(atom_j)
        neighbours[atom_j].append(atom_i)
    visited: set[int] = set()
    components = 0
    for start in range(cluster.n_carbon):
        if start in visited:
            continue
        components += 1
        stack = [start]
        while stack:
            atom_i = stack.pop()
            if atom_i in visited:
                continue
            visited.add(atom_i)
            stack.extend(item for item in neighbours[atom_i] if item not in visited)
    return components


def _audit_cluster(
    cluster: Cluster,
    lattice_constant_angstrom: float,
    cc_bond_angstrom: float,
    ch_bond_angstrom: float,
    domain_radius_angstrom: float,
) -> dict[str, object]:
    positions = np.vstack((cluster.carbon_positions, cluster.hydrogen_positions))
    cc_lengths = [
        float(np.linalg.norm(cluster.carbon_positions[j] - cluster.carbon_positions[i]))
        for i, j in cluster.cc_edges
    ]
    ch_lengths = [
        float(np.linalg.norm(cluster.hydrogen_positions[j] - cluster.carbon_positions[i]))
        for i, j in cluster.ch_edges
    ]
    h_per_carbon = np.zeros(cluster.n_carbon, dtype=int)
    for carbon_i, _ in cluster.ch_edges:
        h_per_carbon[carbon_i] += 1
    valence = np.asarray(cluster.carbon_coordination) + h_per_carbon
    coordination_histogram = {
        str(coordination): int(cluster.carbon_coordination.count(coordination))
        for coordination in sorted(set(cluster.carbon_coordination))
    }

    audit: dict[str, object] = {
        "schema": "parsec_python.nanodiamond_geometry.v1",
        "formula": cluster.formula,
        "construction": (
            "atom-centered Euclidean seed sphere plus shared-dangling-site "
            "surface completion"
        ),
        "relaxed": False,
        "radius_multiplier": cluster.multiplier,
        "cutoff_radius_angstrom": cluster.cutoff_radius_angstrom,
        "seed_n_carbon": cluster.seed_n_carbon,
        "surface_completion_added_carbon": cluster.n_carbon - cluster.seed_n_carbon,
        "surface_completion_passes": cluster.surface_completion_passes,
        "lattice_constant_angstrom": lattice_constant_angstrom,
        "target_cc_bond_angstrom": cc_bond_angstrom,
        "target_ch_bond_angstrom": ch_bond_angstrom,
        "n_carbon": cluster.n_carbon,
        "n_hydrogen": cluster.n_hydrogen,
        "n_atoms": cluster.n_carbon + cluster.n_hydrogen,
        "valence_electrons": 4 * cluster.n_carbon + cluster.n_hydrogen,
        "carbon_carbon_bonds": len(cluster.cc_edges),
        "carbon_hydrogen_bonds": len(cluster.ch_edges),
        "carbon_coordination_histogram": coordination_histogram,
        "carbon_graph_components": _connected_carbon_components(cluster),
        "all_carbon_valences_after_h_termination": sorted(set(int(x) for x in valence)),
        "cc_bond_min_angstrom": min(cc_lengths) if cc_lengths else None,
        "cc_bond_max_angstrom": max(cc_lengths) if cc_lengths else None,
        "ch_bond_min_angstrom": min(ch_lengths) if ch_lengths else None,
        "ch_bond_max_angstrom": max(ch_lengths) if ch_lengths else None,
        "hydrogen_hydrogen_min_distance_angstrom": _minimum_pair_distance(
            cluster.hydrogen_positions
        ),
        "all_atom_min_distance_angstrom": _minimum_pair_distance(positions),
        "maximum_atomic_radius_angstrom": float(
            np.max(np.linalg.norm(positions, axis=1))
        ),
        "parsec_domain_radius_angstrom": domain_radius_angstrom,
    }

    if audit["carbon_graph_components"] != 1:
        raise ValueError(f"{cluster.formula}: disconnected carbon skeleton")
    if audit["all_carbon_valences_after_h_termination"] != [4]:
        raise ValueError(f"{cluster.formula}: carbon valence audit failed")
    if cc_lengths and not np.allclose(cc_lengths, cc_bond_angstrom, atol=1.0e-10):
        raise ValueError(f"{cluster.formula}: C--C bond audit failed")
    if ch_lengths and not np.allclose(ch_lengths, ch_bond_angstrom, atol=1.0e-10):
        raise ValueError(f"{cluster.formula}: C--H bond audit failed")
    hh_min = audit["hydrogen_hydrogen_min_distance_angstrom"]
    if hh_min is not None and float(hh_min) < 1.2:
        raise ValueError(f"{cluster.formula}: unphysical H--H contact ({hh_min:.6f} A)")
    return audit


def _parsec_input(cluster: Cluster, domain_radius_angstrom: float) -> str:
    """Return a PBE/SAD input template without launching a calculation."""

    electrons = 4 * cluster.n_carbon + cluster.n_hydrogen
    states = max(16, math.ceil(0.70 * electrons))
    lines = [
        "# Ideal unrelaxed hydrogen-terminated nanodiamond",
        f"# Formula: {cluster.formula}",
        f"# Initial carbon cutoff: {cluster.multiplier} * d_CC = "
        f"{cluster.cutoff_radius_angstrom:.10f} ang",
        f"# Shared-dangling-site completion: "
        f"+{cluster.n_carbon - cluster.seed_n_carbon} C",
        "",
        "Boundary_Conditions: cluster",
        "Cluster_Domain_Shape: sphere",
        f"Boundary_Sphere_Radius: {domain_radius_angstrom:.1f} ang",
        f"Grid_Spacing: {DEFAULT_GRID_SPACING_ANGSTROM:.2f} ang",
        "Expansion_Order: 8",
        "Coordinate_Unit: Cartesian_Ang",
        "Ignore_Symmetry: false",
        "",
        "Atom_Types_Num: 2",
        f"Total_Atom_Num: {cluster.n_carbon + cluster.n_hydrogen}",
        "",
        "Atom_Type: C",
        "Element_Symbol: C",
        "Pseudopotential_Format: martins_new",
        "Local_Component: p",
        "Read_VCD: true",
        "Cubic_Spline: true",
        "begin Atom_Coord",
    ]
    lines.extend(
        "  " + "  ".join(f"{value:.10f}" for value in position)
        for position in cluster.carbon_positions
    )
    lines.extend(
        [
            "end Atom_Coord",
            "",
            "Atom_Type: H",
            "Element_Symbol: H",
            "Pseudopotential_Format: martins_new",
            "Local_Component: s",
            "Read_VCD: true",
            "Cubic_Spline: true",
            "begin Atom_Coord",
        ]
    )
    lines.extend(
        "  " + "  ".join(f"{value:.10f}" for value in position)
        for position in cluster.hydrogen_positions
    )
    lines.extend(
        [
            "end Atom_Coord",
            "",
            "Correlation_Type: pbe",
            "Spin_Polarization: false",
            f"States_Num: {states}",
            "Net_Charges: 0 e",
            "Fermi_Temp: 500 K",
            "Max_Iter: 50",
            "Convergence_Criterion: 2e-4 Ry",
            "",
            "Eigensolver: chebdav",
            "Chebdav_Degree: 15",
            "FF_MaxIter: 2",
            "Matvec_Blocksize: 6",
            "Chebyshev_Degree: 10",
            "Chebyshev_Degree_Delta: 3",
            "Subspace_Buffer_Size: 6",
            "Diag_Tolerance: 1e-4",
            "",
            "Mixing_Method: Anderson",
            "Mixing_Param: 0.30",
            "Memory_Param: 4",
            "Restart_Mixing: 20",
            "Solver_Lpole: 9",
            "Full_Hartree: false",
            "",
            "Initial_Density: sad",
            "Normalize_Initial_Density: true",
            "",
            "Skip_Force: true",
            "Output_Level: 1",
            "Output_All_States: false",
            "",
        ]
    )
    return "\n".join(lines)


def _write_cluster(
    output_root: Path,
    cluster: Cluster,
    *,
    lattice_constant_angstrom: float,
    cc_bond_angstrom: float,
    ch_bond_angstrom: float,
    vacuum_padding_angstrom: float,
) -> dict[str, object]:
    positions = np.vstack((cluster.carbon_positions, cluster.hydrogen_positions))
    symbols = ["C"] * cluster.n_carbon + ["H"] * cluster.n_hydrogen
    maximum_radius = float(np.max(np.linalg.norm(positions, axis=1)))
    domain_radius = math.ceil(
        10.0 * (maximum_radius + vacuum_padding_angstrom)
    ) / 10.0
    box_length = 2.0 * domain_radius
    atoms = Atoms(symbols=symbols, positions=positions, cell=[box_length] * 3, pbc=False)
    name = f"k{cluster.multiplier:02d}_{cluster.formula}"
    directory = output_root / name
    directory.mkdir(parents=True, exist_ok=True)

    write(
        directory / "structure.xyz",
        atoms,
        format="xyz",
        comment=(
            f"ideal unrelaxed {cluster.formula}; atom-centered cutoff "
            f"R={cluster.multiplier}*d_CC={cluster.cutoff_radius_angstrom:.10f} ang; "
            f"surface completion +{cluster.n_carbon - cluster.seed_n_carbon} C"
        ),
    )
    write(directory / "structure.extxyz", atoms, format="extxyz")
    poscar_atoms = atoms.copy()
    poscar_atoms.positions += 0.5 * box_length
    write(directory / "POSCAR", poscar_atoms, format="vasp", direct=False, sort=True)
    (directory / "parsec.in").write_text(
        _parsec_input(cluster, domain_radius), encoding="utf-8"
    )

    audit = _audit_cluster(
        cluster,
        lattice_constant_angstrom,
        cc_bond_angstrom,
        ch_bond_angstrom,
        domain_radius,
    )
    audit["display_box_length_angstrom"] = box_length
    (directory / "geometry_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "directory": name,
        "radius_multiplier": cluster.multiplier,
        "cutoff_radius_angstrom": f"{cluster.cutoff_radius_angstrom:.10f}",
        "seed_n_carbon": cluster.seed_n_carbon,
        "surface_completion_added_carbon": cluster.n_carbon - cluster.seed_n_carbon,
        "formula": cluster.formula,
        "n_carbon": cluster.n_carbon,
        "n_hydrogen": cluster.n_hydrogen,
        "n_atoms": cluster.n_carbon + cluster.n_hydrogen,
        "valence_electrons": 4 * cluster.n_carbon + cluster.n_hydrogen,
        "states_num": max(16, math.ceil(0.70 * (4 * cluster.n_carbon + cluster.n_hydrogen))),
        "domain_radius_angstrom": f"{domain_radius:.1f}",
        "relaxed": False,
    }


def generate(
    output_root: Path,
    *,
    maximum_carbon_atoms: int,
    maximum_multiplier: int,
    lattice_constant_angstrom: float,
    ch_bond_angstrom: float,
    vacuum_padding_angstrom: float,
    include_first_at_or_above_limit: bool = False,
) -> list[dict[str, object]]:
    if maximum_carbon_atoms <= 1:
        raise ValueError("maximum_carbon_atoms must be greater than one")
    if maximum_multiplier < 1:
        raise ValueError("maximum_multiplier must be positive")
    lattice = _reference_lattice(lattice_constant_angstrom, maximum_multiplier)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    first_excluded: dict[str, object] | None = None

    for multiplier in range(maximum_multiplier + 1):
        cluster = _build_cluster(lattice, multiplier, ch_bond_angstrom)
        if cluster.n_carbon >= maximum_carbon_atoms:
            first_excluded = {
                "radius_multiplier": multiplier,
                "n_carbon": cluster.n_carbon,
                "formula": cluster.formula,
            }
            if include_first_at_or_above_limit:
                rows.append(
                    _write_cluster(
                        output_root,
                        cluster,
                        lattice_constant_angstrom=lattice_constant_angstrom,
                        cc_bond_angstrom=lattice.cc_bond_angstrom,
                        ch_bond_angstrom=ch_bond_angstrom,
                        vacuum_padding_angstrom=vacuum_padding_angstrom,
                    )
                )
            break
        rows.append(
            _write_cluster(
                output_root,
                cluster,
                lattice_constant_angstrom=lattice_constant_angstrom,
                cc_bond_angstrom=lattice.cc_bond_angstrom,
                ch_bond_angstrom=ch_bond_angstrom,
                vacuum_padding_angstrom=vacuum_padding_angstrom,
            )
        )
    if first_excluded is None:
        raise RuntimeError(
            "maximum_multiplier was reached before the carbon limit; "
            "increase --maximum-multiplier"
        )

    manifest_path = output_root / "cluster_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "parsec_python.nanodiamond_series.v1",
        "selection_rule": (
            "seed carbon if distance from center <= k*d_CC; then iteratively add "
            "each removed diamond site adjacent to at least two retained carbons"
        ),
        "maximum_carbon_atoms_exclusive": maximum_carbon_atoms,
        "first_at_or_above_limit_included": include_first_at_or_above_limit,
        "lattice_constant_angstrom": lattice_constant_angstrom,
        "cc_bond_angstrom": lattice.cc_bond_angstrom,
        "ch_bond_angstrom": ch_bond_angstrom,
        "vacuum_padding_angstrom": vacuum_padding_angstrom,
        "generated_directories": [row["directory"] for row in rows],
        "first_excluded_candidate": first_excluded,
        "scf_calculations_run": False,
    }
    (output_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="directory receiving the cluster folders and manifests",
    )
    parser.add_argument("--maximum-carbon-atoms", type=int, default=200)
    parser.add_argument("--maximum-multiplier", type=int, default=8)
    parser.add_argument(
        "--include-first-at-or-above-limit",
        action="store_true",
        help=(
            "retain the first radial shell whose carbon count reaches or exceeds "
            "--maximum-carbon-atoms"
        ),
    )
    parser.add_argument(
        "--lattice-constant", type=float, default=DEFAULT_LATTICE_CONSTANT_ANGSTROM
    )
    parser.add_argument("--ch-bond", type=float, default=DEFAULT_CH_BOND_ANGSTROM)
    parser.add_argument(
        "--vacuum-padding", type=float, default=DEFAULT_VACUUM_PADDING_ANGSTROM
    )
    arguments = parser.parse_args()
    rows = generate(
        arguments.output_root.resolve(),
        maximum_carbon_atoms=arguments.maximum_carbon_atoms,
        maximum_multiplier=arguments.maximum_multiplier,
        lattice_constant_angstrom=arguments.lattice_constant,
        ch_bond_angstrom=arguments.ch_bond,
        vacuum_padding_angstrom=arguments.vacuum_padding,
        include_first_at_or_above_limit=arguments.include_first_at_or_above_limit,
    )
    print("Generated geometry/input folders (no SCF calculations):")
    for row in rows:
        print(
            f"  {row['directory']}: {row['n_carbon']} C, "
            f"{row['n_hydrogen']} H, R={row['cutoff_radius_angstrom']} ang"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
