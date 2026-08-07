"""Input parsing and preprocessing for the refactored RSDFT driver.

This module owns everything that happens before the physical domain is
constructed: loading element metadata, manual input prompts, file parsing,
unit conversion, optional ML-grid lookup, and normalization of path-like
settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .rsdft_models import A0_ANG, InputData, SolverSettings


def load_elements(base_dir: str | Path | None = None):
    """Load ``elements_new.csv`` and return ``(elem_dataframe, n_elements)``."""
    base_path = Path(base_dir or Path(__file__).resolve().parent)
    csv_path = base_path / "elements_new.csv"
    elem = pd.read_csv(csv_path, header=None)
    elem.columns = ["Element", "AtomicNumber", "Z", "Zvalue", "h", "r", "R"] + [
        f"Property{i}" for i in range(1, 5)
    ]
    return elem, elem.shape[0]


def element_exists(typ: str, elem) -> bool:
    """Return ``True`` when ``typ`` is present in the element table."""
    return typ in elem["Element"].values


def has_diagmeth_override(overrides: dict[str, Any] | None) -> bool:
    """Check whether the user explicitly selected a diagonalization method."""
    return any(str(key).lower() == "diagmeth" for key in (overrides or {}))


def _prompt_value_with_default(prompt_text: str, cast_func, current_value: Any):
    """Prompt once for an override and fall back to the current value on error."""
    raw = input(f"{prompt_text} [{current_value}]: ").strip()
    if raw == "":
        return None
    try:
        return cast_func(raw)
    except ValueError:
        print("  Invalid input, keeping current value.")
        return None


def prompt_for_settings_overrides(settings: SolverSettings) -> dict[str, Any]:
    """Interactively collect optional solver/grid overrides.

    Input:
        settings: Current default settings shown in the prompts.

    Output:
        A dictionary containing only the values the user chose to override.
    """
    print("\n ------------------------")
    choice = input("Override default solver/grid settings? (y/N): ").strip().lower()
    if choice not in ["y", "yes"]:
        return {}

    overrides = {
        "tol": _prompt_value_with_default("SCF tolerance", float, settings.tol),
        "maxits": _prompt_value_with_default("Max SCF iterations", int, settings.maxits),
        "fd_order": _prompt_value_with_default("Finite-difference order", int, settings.fd_order),
        "Fermi_temp": _prompt_value_with_default("Fermi temperature (K)", float, settings.fermi_temp),
        "poldeg": _prompt_value_with_default("Polynomial degree for Chebyshev", int, settings.poldeg),
        "diagmeth": _prompt_value_with_default("Diagonalization method (0-3)", int, settings.diagmeth),
        "CG_prec": _prompt_value_with_default("Use CG preconditioner? (0/1)", int, settings.cg_prec),
        "adaptiveScheme": _prompt_value_with_default(
            "Use adaptive scheme? (0/1)", int, settings.adaptive_scheme
        ),
        "use_gpu": _prompt_value_with_default("Use GPU backend when available? (0/1)", int, settings.use_gpu),
        "recenter_atoms": _prompt_value_with_default(
            "Recenter atoms for default SAD grid? (0/1)", int, settings.recenter_atoms
        ),
        "hartree_method": _prompt_value_with_default(
            "Hartree method (split/consistent_split/full)", str, settings.hartree_method
        ),
        "nev": _prompt_value_with_default("Number of eigenvalues (blank = auto)", int, "auto"),
        "grid_spacing": _prompt_value_with_default("Grid spacing h [Bohr] (blank = auto)", float, "auto"),
        "sphere_radius": _prompt_value_with_default("Sphere radius [Bohr] (blank = auto)", float, "auto"),
    }
    return {key: value for key, value in overrides.items() if value is not None}


def manual_input_species(elem) -> tuple[list[dict[str, Any]], list[int], float, str, str | None]:
    """Read an atomic system from interactive terminal prompts.

    Output tuple:
        atoms: list of ``{"typ": ..., "coord": ndarray}`` species blocks.
        n_atom: number of atoms for each species block.
        z_charge: total system charge convention used by the legacy code.
        density_method: ``sad`` or ``ml``.
        ml_file_path: optional ML density file path.
    """
    at = 0
    while at < 1:
        try:
            at = int(input("Input number of different atomic species: "))
            if at < 1:
                print("Please enter a number >= 1")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    atoms: list[dict[str, Any]] = []
    n_atom: list[int] = []
    for _ in range(at):
        typ = ""
        while not element_exists(typ, elem):
            typ = input("Element of species, e.g., Mg, C, O, only first 18 elements supported: ")
            if not element_exists(typ, elem):
                print("    Element not recognized")

        print("  Coordinates should be in atomic units ")
        print("  Example: atoms at (0,0,0) should be entered as 0 0 0 on each line ")
        print("  Terminate with /, i.e., 0 0 0 / for the last entry ")

        readxyz = ""
        while "/" not in readxyz:
            readxyz += " " + input("  Input coordinates: ")

        try:
            coordinates = [float(coord) for coord in readxyz.split("/")[-2].strip().split()]
        except ValueError as exc:
            raise ValueError("Invalid input detected. Please ensure all coordinates are numbers.") from exc

        if len(coordinates) == 0 or len(coordinates) % 3 != 0:
            raise ValueError("The number of coordinates must be nonzero and divisible by 3.")

        xyz = np.array(coordinates, dtype=float).reshape(-1, 3)
        n_atom.append(xyz.shape[0])
        atoms.append({"typ": typ, "coord": xyz})

    try:
        z_charge = float(input(" How many electrons should be added/removed from the system? "))
        z_charge = -z_charge
    except ValueError:
        print("Error: Invalid input detected. Use default Z_charge = 0")
        z_charge = 0.0

    print("\n ********************** ")
    print(" DENSITY INITIALIZATION METHOD")
    print(" ********************* ")
    print("------------------------")

    method_choice = 0
    while method_choice not in [1, 2]:
        try:
            print("Choose density initialization method:")
            print("1. Superposition of atomic densities (SAD)")
            print("2. Machine learning predicted density (.npy file)")
            method_choice = int(input("Enter your choice (1 or 2): "))
        except ValueError:
            print("Please enter 1 or 2")

    if method_choice == 1:
        density_method = "sad"
        ml_file_path = None
    else:
        density_method = "ml"
        ml_file_path = input("Enter path to ML density .npy file: ")

    return atoms, n_atom, z_charge, density_method, ml_file_path


def _parse_atoms_from_lines(lines: list[str]):
    """Parse ``TYPE x y z`` lines into the grouped atom representation."""
    grouped: dict[str, list[list[float]]] = {}
    order: list[str] = []
    for raw in lines:
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError("Atom line must be: TYPE x y z")

        typ = parts[0]
        try:
            coords = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError as exc:
            raise ValueError("Could not parse coordinates; must be numeric.") from exc

        if typ not in grouped:
            grouped[typ] = []
            order.append(typ)
        grouped[typ].append(coords)

    atoms: list[dict[str, Any]] = []
    n_atom: list[int] = []
    for typ in order:
        coord_arr = np.array(grouped[typ], dtype=float)
        atoms.append({"typ": typ, "coord": coord_arr})
        n_atom.append(coord_arr.shape[0])
    return atoms, n_atom


def _prune_empty(overrides_dict: dict[str, Any] | None) -> dict[str, Any]:
    """Drop ``None`` and empty-string values from a settings dictionary."""
    if not overrides_dict:
        return {}
    cleaned = {}
    for key, value in overrides_dict.items():
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        cleaned[key] = value
    return cleaned


def atoms_in_molecule_to_atoms_converter(atoms_in_molecule):
    """Convert the ``AtomsInMolecule`` structure into Python atoms/n_atom."""
    atoms: list[dict[str, Any]] = []
    n_atom: list[int] = []

    flat_atoms = np.array(atoms_in_molecule).ravel()
    for atom in flat_atoms:
        typ = None
        coord = None

        if isinstance(atom, dict):
            typ = atom.get("typ") or atom.get("type") or atom.get("element")
            coord = atom.get("coord") or atom.get("coords") or atom.get("coordinates")
        elif isinstance(atom, np.void) and atom.dtype.fields:
            if "typ" in atom.dtype.fields:
                typ = atom["typ"]
            if "coord" in atom.dtype.fields:
                coord = atom["coord"]
        else:
            try:
                typ = atom["typ"]
                coord = atom["coord"]
            except Exception:
                pass

        if typ is None or coord is None:
            raise ValueError("AtomsInMolecule entry missing typ or coord.")

        if isinstance(typ, np.ndarray):
            typ = typ.item()
        typ = str(typ)

        coord = np.asarray(coord, dtype=float)
        try:
            coord = np.reshape(coord, (-1, 3))
        except ValueError as exc:
            raise ValueError("Could not reshape coordinates to (N,3) in .mat file.") from exc

        n_atom.append(coord.shape[0])
        atoms.append({"typ": typ, "coord": coord})

    return atoms, n_atom


def _resolve_input_file(file_name: str, base_dir: Path) -> Path:
    """Resolve an input file against the legacy search locations."""
    raw = Path(os.path.expanduser(file_name))
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(
            [
                Path("SavedMolecules") / raw,
                base_dir / raw,
                base_dir / "SavedMolecules" / raw,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"File not found: tried {[str(path) for path in candidates]}")


def load_atomic_data_from_file(file_name: str | None, base_dir: str | Path | None = None) -> InputData:
    """Load geometry/settings from a supported input file.

    Supported formats are legacy ``.dat``/``.mat``/``.txt`` plus the newer
    ``.json`` and ``.in``/``.inp`` formats. The return value is an
    ``InputData`` object that still needs path resolution and unit handling.
    """
    base_path = Path(base_dir or Path(__file__).resolve().parent)
    if file_name is None:
        file_name = input("What is the name of the file to load from?: ").strip()
    else:
        file_name = str(file_name).strip()

    if len(file_name) <= 4:
        raise ValueError("File name is too short, must include file extension.")

    file_path = _resolve_input_file(file_name, base_path)
    file_extension = file_path.suffix.lower()

    atoms: list[dict[str, Any]] = []
    n_atom: list[int] = []
    z_charge = 0.0
    density_method = "sad"
    ml_file_path = None
    settings_overrides: dict[str, Any] = {}
    unit_setting = None

    if file_extension == ".dat":
        with open(file_path, "r", encoding="utf-8") as handle:
            at = int(handle.readline().strip())
            for _ in range(at):
                typ = handle.readline().strip()
                readxyz = np.array([float(x) for x in handle.readline().strip().split()], dtype=float)
                if len(readxyz) % 3 != 0:
                    raise ValueError("Coordinate count in .dat must be divisible by 3.")
                xyz = np.reshape(readxyz, (-1, 3))
                n_atom.append(xyz.shape[0])
                atoms.append({"typ": typ, "coord": xyz})

    elif file_extension == ".mat":
        mat_data = loadmat(file_path)
        if "AtomsInMolecule" not in mat_data:
            raise ValueError("File does not contain the correct variable: AtomsInMolecule")
        atoms, n_atom = atoms_in_molecule_to_atoms_converter(mat_data["AtomsInMolecule"])

    elif file_extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as handle:
            atoms, n_atom = _parse_atoms_from_lines(handle.readlines())

    elif file_extension in [".in", ".inp"]:
        with open(file_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()

        section = None
        atom_lines: list[str] = []
        settings_lines: list[str] = []
        for raw in lines:
            line = raw.strip()
            if line == "" or line.startswith("#"):
                continue

            token = line.lower()
            if token == "$system":
                section = "system" if section != "system" else None
                continue
            if token == "$settings":
                section = "settings" if section != "settings" else None
                continue

            if section == "system":
                atom_lines.append(line)
            elif section == "settings":
                settings_lines.append(line)

        atoms, n_atom = _parse_atoms_from_lines(atom_lines)
        for entry in settings_lines:
            if "=" not in entry:
                continue
            key, val = entry.split("=", 1)
            key = key.strip()
            val = val.strip()
            if val == "":
                continue

            key_lower = key.lower()
            if key_lower in ["z_charge", "charge"]:
                try:
                    z_charge = float(val)
                except ValueError:
                    print(f'Warning: could not parse Z_charge "{val}", using default 0.')
            elif key_lower in ["density_method", "density"]:
                density_method = val.lower()
            elif key_lower in ["ml_file_path", "density_file"]:
                ml_file_path = val
            elif key_lower == "unit":
                unit_setting = val
            else:
                settings_overrides[key] = val

        if ml_file_path:
            ml_candidate = Path(os.path.expanduser(ml_file_path))
            if not ml_candidate.is_absolute():
                ml_candidate = file_path.parent / ml_candidate
            ml_file_path = str(ml_candidate)
        settings_overrides = _prune_empty(settings_overrides)

    elif file_extension == ".json":
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        atoms_block = data.get("atoms") or data.get("Atoms")
        if not atoms_block:
            raise ValueError('JSON file missing "atoms" list.')

        if isinstance(atoms_block, str):
            atoms, n_atom = _parse_atoms_from_lines(atoms_block.strip().splitlines())
        elif isinstance(atoms_block, list) and atoms_block and isinstance(atoms_block[0], str):
            atoms, n_atom = _parse_atoms_from_lines(atoms_block)
        else:
            for entry in atoms_block:
                typ = entry.get("typ") or entry.get("type") or entry.get("element")
                coords = entry.get("coord") or entry.get("coords") or entry.get("coordinates")
                if typ is None or coords is None:
                    raise ValueError("Each atom entry must have typ and coord/coords.")

                coord_arr = np.asarray(coords, dtype=float).reshape(-1, 3)
                atoms.append({"typ": str(typ), "coord": coord_arr})
                n_atom.append(coord_arr.shape[0])

        try:
            z_charge = float(data.get("Z_charge", data.get("charge", 0.0)))
        except (TypeError, ValueError):
            print("Warning: invalid Z_charge in JSON, defaulting to 0.")
            z_charge = 0.0

        density_method = data.get("density_method", density_method)
        if isinstance(density_method, str):
            density_method = density_method.lower()
        else:
            density_method = "sad"

        ml_file_path = data.get("ml_file_path") or data.get("density_file")
        if ml_file_path:
            ml_candidate = Path(os.path.expanduser(str(ml_file_path)))
            if not ml_candidate.is_absolute():
                ml_candidate = file_path.parent / ml_candidate
            ml_file_path = str(ml_candidate)

        settings_overrides = data.get("settings", data.get("params", {})) or {}
        if isinstance(settings_overrides, dict):
            unit_setting = settings_overrides.get("unit", unit_setting)
        else:
            settings_overrides = {}

        for key in [
            "nev",
            "grid_spacing",
            "h",
            "sphere_radius",
            "radius",
            "grid_npy_path",
            "grid_poscar_path",
            "recenter_atoms",
        ]:
            if key in data and key not in settings_overrides:
                settings_overrides[key] = data[key]
        settings_overrides = _prune_empty(settings_overrides)
        if unit_setting:
            settings_overrides["unit"] = unit_setting

    else:
        raise ValueError("File extension not recognized (supported: .dat, .mat, .json, .txt, .in, .inp).")

    if density_method not in ["sad", "ml", "sad_ml_grid"]:
        density_method = "sad"
    if unit_setting:
        settings_overrides["unit"] = unit_setting

    return InputData(
        atoms=atoms,
        n_atom=n_atom,
        z_charge=z_charge,
        density_method=density_method,
        ml_file_path=ml_file_path,
        settings_overrides=settings_overrides,
        input_file_path=str(file_path),
    )


def get_input_data(elem, settings: SolverSettings, input_file_path: str | None = None) -> InputData:
    """Choose manual input or file input and return raw ``InputData``."""
    print(" ********************** ")
    print("  DATA INPUT FOR RSDFT  ")
    print(" ********************** ")
    print("------------------------")

    if input_file_path:
        print(f"Loading input file: {input_file_path}")
        return load_atomic_data_from_file(input_file_path)

    in_data = 0
    while in_data not in [1, 2]:
        try:
            in_data = int(input("Input data mode: 1 for manual, 2 for file: "))
        except ValueError:
            print("Please enter 1 or 2")

    if in_data == 1:
        atoms, n_atom, z_charge, density_method, ml_file_path = manual_input_species(elem)
        return InputData(
            atoms=atoms,
            n_atom=n_atom,
            z_charge=z_charge,
            density_method=density_method,
            ml_file_path=ml_file_path,
            settings_overrides=prompt_for_settings_overrides(settings),
            input_file_path=None,
        )

    return load_atomic_data_from_file(None)


def resolve_optional_path(
    raw_path: str | None,
    label: str,
    base_dir: str | Path | None = None,
    input_file_path: str | None = None,
) -> str | None:
    """Resolve a possibly-relative path against common RSDFT input locations."""
    if not raw_path:
        return None

    base_path = Path(base_dir or Path(__file__).resolve().parent)
    raw = Path(os.path.expanduser(str(raw_path)))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, base_path / raw])
        if input_file_path:
            candidates.append(Path(input_file_path).resolve().parent / raw)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    print(f"Warning: {label} {raw_path} not found. Tried: {[str(path) for path in candidates]}")
    return None


def read_poscar_cube_length(poscar_path: str) -> float:
    """Read the average cubic cell length from a POSCAR-like file in Angstrom."""
    with open(poscar_path, "r", encoding="utf-8") as handle:
        raw_lines = [line.strip() for line in handle if line.strip()]

    if len(raw_lines) < 5:
        raise ValueError("Incomplete POSCAR file for grid size.")

    scale = float(raw_lines[1].split()[0])
    lattice = []
    for index in range(2, 5):
        parts = raw_lines[index].split()
        if len(parts) < 3:
            raise ValueError(f'Invalid lattice vector line: "{raw_lines[index]}"')
        lattice.append(
            [
                float(parts[0]) * scale,
                float(parts[1]) * scale,
                float(parts[2]) * scale,
            ]
        )

    lengths = [np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) for vec in lattice]
    if not lengths:
        raise ValueError("Could not determine lattice size from POSCAR.")
    return sum(lengths) / 3.0


def load_grid_data_from_files(grid_npy: str, grid_poscar: str) -> dict[str, Any]:
    """Infer grid shape and spacing from an ML density file plus POSCAR."""
    density = np.load(grid_npy, mmap_mode="r")
    if density.ndim != 3:
        raise ValueError("Grid .npy file must be a 3D array.")

    nx, ny, nz = density.shape
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("Invalid grid shape in .npy file.")

    cube_length_ang = read_poscar_cube_length(grid_poscar)
    h_ang = cube_length_ang / float(nx)
    radius_ang = cube_length_ang / 2.0
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "h": h_ang / A0_ANG,
        "radius": radius_ang / A0_ANG,
    }


def convert_units_to_bohr(
    atoms: list[dict[str, Any]],
    settings_overrides: dict[str, Any] | None,
    input_file_path: str | None,
):
    """Convert atom coordinates and grid/radius overrides into Bohr units.

    Input:
        atoms: geometry as loaded from manual/file input.
        settings_overrides: user-provided overrides, potentially including
            unit-dependent values such as ``grid_spacing`` and ``radius``.
        input_file_path: used to decide the default unit convention.

    Output:
        ``(converted_atoms, converted_overrides, unit_used)``.
    """
    overrides = dict(settings_overrides or {})
    defaults_to_ang = input_file_path is not None
    unit_raw = overrides.get("unit")
    if unit_raw is None:
        unit_raw = "ang" if defaults_to_ang else "bohr"

    unit_norm = str(unit_raw).lower()
    is_ang = unit_norm in ["ang", "angstrom", "a"]
    is_bohr = unit_norm in ["bohr", "au", "a.u."]

    if is_ang:
        factor = 1.0 / A0_ANG
    elif is_bohr:
        factor = 1.0
    else:
        factor = 1.0 / A0_ANG if defaults_to_ang else 1.0

    converted_atoms = []
    for atom in atoms:
        converted_atoms.append(
            {
                "typ": atom["typ"],
                "coord": np.asarray(atom["coord"], dtype=float).copy() * factor,
            }
        )

    for key in ["grid_spacing", "h", "sphere_radius", "radius"]:
        if key in overrides:
            try:
                overrides[key] = float(overrides[key]) * factor
            except (TypeError, ValueError):
                pass

    overrides.pop("unit", None)
    return converted_atoms, overrides, unit_norm


def prepare_input_data(
    elem,
    settings: SolverSettings,
    cli_input_path: str | None = None,
    base_dir: str | Path | None = None,
) -> InputData:
    """Full input-preparation pipeline used by ``main_new.py``.

    This is the last step before numerical setup. It resolves auxiliary file
    paths, converts coordinates to Bohr, determines whether an ML grid is in
    use, and attaches any precomputed grid metadata to the returned
    ``InputData`` object.
    """
    input_data = get_input_data(elem, settings, cli_input_path)
    if not input_data.atoms:
        raise SystemExit("No atoms loaded, exiting RSDFT.")

    overrides = dict(input_data.settings_overrides or {})
    input_data.grid_npy_path = overrides.pop("grid_npy_path", None) or overrides.pop("grid_npy", None)
    input_data.grid_poscar_path = overrides.pop("grid_poscar_path", None) or overrides.pop("grid_poscar", None)
    input_data.settings_overrides = overrides

    input_data.ml_file_path = resolve_optional_path(
        input_data.ml_file_path,
        "ML density file",
        base_dir=base_dir,
        input_file_path=input_data.input_file_path,
    )
    input_data.grid_npy_path = resolve_optional_path(
        input_data.grid_npy_path,
        "Grid npy file",
        base_dir=base_dir,
        input_file_path=input_data.input_file_path,
    )
    input_data.grid_poscar_path = resolve_optional_path(
        input_data.grid_poscar_path,
        "Grid POSCAR file",
        base_dir=base_dir,
        input_file_path=input_data.input_file_path,
    )

    if input_data.density_method in ["ml", "sad_ml_grid"] and input_data.ml_file_path is None:
        raise SystemExit(
            "Density method requires ML .npy file but no ml_file_path was provided or found. "
            "Use the GUI generator to select a .npy file or add ml_file_path to the input file."
        )

    converted_atoms, converted_overrides, unit_used = convert_units_to_bohr(
        input_data.atoms,
        input_data.settings_overrides,
        input_data.input_file_path,
    )
    input_data.atoms = converted_atoms
    input_data.settings_overrides = converted_overrides
    input_data.unit_used = unit_used
    input_data.grid_mode = "ml" if input_data.density_method in ["ml", "sad_ml_grid"] else "default"

    if input_data.grid_mode == "ml":
        if input_data.grid_npy_path is None:
            input_data.grid_npy_path = input_data.ml_file_path
        if input_data.grid_npy_path is None:
            raise SystemExit("Grid .npy file is required to set grid structure.")
        if input_data.grid_poscar_path is None:
            raise SystemExit("Grid POSCAR file is required to set grid structure.")
        input_data.grid_data = load_grid_data_from_files(input_data.grid_npy_path, input_data.grid_poscar_path)

    return input_data
