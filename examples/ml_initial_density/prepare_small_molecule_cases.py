"""Build matched SAD/SCDP/ChargE3Net regression cases from legacy results.

The legacy result tree contains dense model predictions in e/angstrom^3 on a
Cartesian lattice whose first voxel is at (0, 0, 0).  The current solver
recenters isolated molecules by their bounding-box midpoint.  This script
therefore stores that same translation in the affine origin of each portable
``.npz`` density; the density and nuclei then undergo exactly the same rigid
translation before the field is sampled on the authoritative PARSEC grid.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import shutil

import numpy as np


BOHR_TO_ANGSTROM = 0.529177210903
METHOD_SOURCE = {"charge3net": "c3n", "scdp": "scdp"}
VALENCE = {"H": 1, "C": 4, "N": 5, "O": 6}
LOCAL_COMPONENT = {"H": "s", "C": "p", "N": "p", "O": "p"}


def _parse_atoms(path: Path) -> list[tuple[str, np.ndarray]]:
    atoms: list[tuple[str, np.ndarray]] = []
    in_system = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.lower() == "$system":
            in_system = not in_system
            continue
        if not in_system or not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"unexpected legacy geometry line in {path}: {line!r}")
        atoms.append((fields[0], np.asarray(fields[1:], dtype=float)))
    if not atoms:
        raise ValueError(f"no atoms found in {path}")
    return atoms


def _source_directory(legacy_root: Path, molecule: str, method: str) -> Path:
    path = (
        legacy_root
        / molecule
        / "tol_2e-4"
        / "diagmeth3"
        / "hart_tol_1e-5"
        / method
    )
    if not path.is_dir():
        raise FileNotFoundError(f"legacy result directory is missing: {path}")
    return path


def _single(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern!r} below {path}, found {len(matches)}")
    return matches[0]


def _legacy_density(path: Path) -> Path:
    matches = [
        item
        for item in sorted(path.glob("*_init_rho.npy"))
        if not item.name.endswith(("_bohr3.npy", "_grid.npy"))
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one physical legacy density below {path}")
    return matches[0]


def _radius_from_name(path: Path) -> float:
    match = re.search(r"_([0-9]+(?:p[0-9]+)?)A_0p2A\.in$", path.name)
    if match is None:
        raise ValueError(f"cannot recover the legacy radius from {path.name}")
    return float(match.group(1).replace("p", "."))


def _write_portable_density(
    output: Path,
    source: Path,
    atoms: list[tuple[str, np.ndarray]],
    *,
    provider: str,
    molecule: str,
    source_label: str,
    spacing_angstrom: float,
) -> tuple[tuple[int, int, int], float]:
    density = np.load(source, allow_pickle=False)
    if density.ndim != 3:
        raise ValueError(f"legacy density must be 3D: {source}")
    positions = np.asarray([position for _, position in atoms], dtype=float)
    shift_angstrom = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    # This is the exact branch used by center_cluster_geometry (0.1 bohr).
    if np.linalg.norm(shift_angstrom / BOHR_TO_ANGSTROM) <= 0.1:
        shift_angstrom[:] = 0.0
    source_electrons = float(density.sum() * spacing_angstrom**3)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray("parsec_python.ml_density.v1"),
        density=np.asarray(density, dtype=np.float64),
        units=np.asarray("e_per_angstrom3"),
        origin_bohr=-shift_angstrom / BOHR_TO_ANGSTROM,
        voxel_vectors_bohr=np.eye(3) * spacing_angstrom / BOHR_TO_ANGSTROM,
        provider=np.asarray(provider),
        metadata_molecule=np.asarray(molecule),
        metadata_source=np.asarray(source_label),
        metadata_source_grid=np.asarray("legacy Cartesian grid; origin=(0,0,0) angstrom"),
    )
    return tuple(int(value) for value in density.shape), source_electrons


def _input_text(
    atoms: list[tuple[str, np.ndarray]],
    radius_angstrom: float,
    method: str,
) -> str:
    symbols: list[str] = []
    for symbol, _ in atoms:
        if symbol not in symbols:
            symbols.append(symbol)
    electrons = sum(VALENCE[symbol] for symbol, _ in atoms)
    states = max(16, round(0.7 * electrons + 0.5))
    lines = [
        "Boundary_Conditions: cluster",
        "Cluster_Domain_Shape: sphere",
        f"Boundary_Sphere_Radius: {radius_angstrom:.1f} ang",
        "Grid_Spacing: 0.2 ang",
        "Expansion_Order: 8",
        "Coordinate_Unit: Cartesian_Ang",
        "Ignore_Symmetry: false",
        "",
        f"Atom_Types_Num: {len(symbols)}",
        f"Total_Atom_Num: {len(atoms)}",
        "",
    ]
    for symbol in symbols:
        lines.extend(
            [
                f"Atom_Type: {symbol}",
                f"Element_Symbol: {symbol}",
                "Pseudopotential_Format: martins_new",
                f"Local_Component: {LOCAL_COMPONENT[symbol]}",
                "Read_VCD: true",
                "Cubic_Spline: true",
                "begin Atom_Coord",
            ]
        )
        for atom_symbol, position in atoms:
            if atom_symbol == symbol:
                lines.append("  " + "  ".join(f"{value:.10f}" for value in position))
        lines.extend(["end Atom_Coord", ""])
    lines.extend(
        [
            "Correlation_Type: pbe",
            "Spin_Polarization: false",
            f"States_Num: {states}",
            "Net_Charges: 0 e",
            "Fermi_Temp: 500 K",
            "Max_Iter: 50",
            "Convergence_Criterion: 2e-4 Ry",
            "",
            # CHEBDAV spends more work on the first Hamiltonian, producing a
            # substantially better starting eigensubspace.  Later nonlinear
            # steps use the same PARSEC SUBSPACE filter as CHEBFF.  Degree 15
            # is the smallest source-valid CHEBDAV degree and was faster than
            # degree 20 in the matched 42-case cold-start audit.
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
            f"Initial_Density: {method}",
        ]
    )
    if method in {"charge3net", "scdp"}:
        lines.extend(
            [
                f"ML_Density_File: ../densities/{method}.npz",
                "ML_Density_Units: auto",
                "ML_Density_Interpolation: linear",
                "ML_Density_Negative_Policy: clip",
                f"ML_Density_Model: {'qm9' if method == 'charge3net' else 'fast'}",
            ]
        )
    lines.extend(
        [
            "Normalize_Initial_Density: true",
            "",
            "Skip_Force: true",
            "Output_Level: 1",
            "Output_All_States: false",
            "",
        ]
    )
    return "\n".join(lines)


def prepare(legacy_root: Path, output_root: Path, pp_source: Path) -> None:
    molecules = sorted(
        item.name
        for item in legacy_root.iterdir()
        if item.is_dir() and item.name != "C60"
    )
    if not molecules:
        raise ValueError(f"no legacy molecule directories found below {legacy_root}")

    pp_output = output_root / "pseudopotentials"
    pp_output.mkdir(parents=True, exist_ok=True)
    for symbol in VALENCE:
        source = pp_source / symbol / "PARSEC" / f"{symbol}_POTRE.DAT"
        if not source.is_file():
            raise FileNotFoundError(f"PBE pseudopotential is missing: {source}")
        shutil.copy2(source, pp_output / source.name)

    manifest_rows: list[dict[str, object]] = []
    cases_root = output_root / "small_molecules"
    for molecule in molecules:
        geometry_dir = _source_directory(legacy_root, molecule, "c3n")
        geometry_input = _single(geometry_dir, "*.in")
        atoms = _parse_atoms(geometry_input)
        radius = _radius_from_name(geometry_input)
        expected_electrons = float(sum(VALENCE[symbol] for symbol, _ in atoms))
        case_root = cases_root / molecule
        density_root = case_root / "densities"

        for method, legacy_method in METHOD_SOURCE.items():
            legacy_dir = _source_directory(legacy_root, molecule, legacy_method)
            density_source = _legacy_density(legacy_dir)
            shape, source_electrons = _write_portable_density(
                density_root / f"{method}.npz",
                density_source,
                atoms,
                provider=method,
                molecule=molecule,
                source_label=str(density_source.relative_to(legacy_root)),
                spacing_angstrom=0.2,
            )
            if not np.isclose(source_electrons, expected_electrons, atol=1.0e-5):
                raise ValueError(
                    f"{molecule}/{method} integrates to {source_electrons}, "
                    f"expected {expected_electrons} electrons"
                )
            manifest_rows.append(
                {
                    "molecule": molecule,
                    "method": method,
                    "radius_angstrom": radius,
                    "spacing_angstrom": 0.2,
                    "source_shape": "x".join(str(value) for value in shape),
                    "source_electrons": f"{source_electrons:.8f}",
                    "legacy_density": density_source.relative_to(legacy_root),
                }
            )

        for method in ("sad", "scdp", "charge3net"):
            method_root = case_root / method
            method_root.mkdir(parents=True, exist_ok=True)
            (method_root / "parsec.in").write_text(
                _input_text(atoms, radius, method), encoding="utf-8"
            )

    manifest = output_root / "case_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Prepared {len(molecules) * 3} cases for {len(molecules)} molecules")
    print(f"Case root: {cases_root}")


def main() -> int:
    script_root = Path(__file__).resolve().parent
    repository = script_root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=repository / "tests" / "tests" / "small_molecules",
    )
    parser.add_argument("--output-root", type=Path, default=script_root)
    parser.add_argument(
        "--pp-source",
        type=Path,
        default=repository / "src" / "pp_generation" / "NCPPs",
    )
    arguments = parser.parse_args()
    prepare(
        arguments.legacy_root.resolve(),
        arguments.output_root.resolve(),
        arguments.pp_source.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
