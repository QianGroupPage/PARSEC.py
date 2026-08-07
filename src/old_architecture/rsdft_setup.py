"""Problem-setup helpers between input parsing and the actual SCF loop.

This module turns ``InputData`` into a ``PreparedSystem`` by choosing the
grid spacing, number of states, simulation domain, optional recentering, and
output-file names.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import numpy as np

from .rsdft_models import A0_ANG, InputData, PreparedSystem, RunPaths, SolverSettings


def calculate_grid_spacing(
    atoms: list[dict[str, Any]],
    elem,
    n_elements: int,
    n_atom: list[int],
    z_charge: float,
    h_override: float | None = None,
    nev_override: int | None = None,
    grid_mode: str = "default",
    grid_data: dict[str, Any] | None = None,
):
    """Choose the working grid spacing ``h`` and the number of states ``nev``.

    Output:
        ``(h, nev, zelec, ztest)`` where ``zelec`` is the neutral electron
        count from the atom list and ``ztest`` is the charge-adjusted count.
    """
    n_types = len(atoms)
    zelec = 0.0
    hmin = 100.0

    if grid_mode == "ml":
        if grid_data is None:
            raise SystemExit("Grid data is required to determine grid spacing.")
        hmin = grid_data["h"]

    for at_typ in range(n_types):
        typ = atoms[at_typ]["typ"]
        for index in range(n_elements):
            if typ == elem["Element"].iloc[index]:
                z_value = elem["Z"].iloc[index] * n_atom[at_typ]
                if grid_mode != "ml":
                    h_value = elem["h"].iloc[index]
                    if h_value < hmin:
                        hmin = h_value
                zelec += z_value

    ztest = zelec - z_charge
    if ztest < 0:
        raise SystemExit("Problem with charge state. Negative number of electrons.")

    if grid_mode == "ml":
        if h_override is not None:
            print("Warning: grid spacing override ignored when using ML grid.")
        h = hmin
    else:
        if h_override is not None:
            try:
                h = float(h_override)
            except (TypeError, ValueError):
                print("Warning: invalid grid spacing override, using auto value.")
                h = hmin
        else:
            h = hmin

    if nev_override is not None:
        try:
            nev = max(1, int(nev_override))
        except (TypeError, ValueError):
            print("Warning: invalid nev override, using auto value.")
            nev = max(16, round(0.7 * zelec + 0.5))
    else:
        nev = max(16, round(0.7 * zelec + 0.5))

    return h, nev, zelec, ztest


def estimate_radius_and_grid(
    atoms: list[dict[str, Any]],
    elem,
    n_elements: int,
    h: float,
    radius_override: float | None = None,
    grid_mode: str = "default",
    grid_data: dict[str, Any] | None = None,
):
    """Construct the domain dictionary ``{radius, nx, ny, nz, h}``.

    For default grids the radius is estimated from the farthest atom plus the
    tabulated atomic size. For ML grids the shape/radius come from the
    external grid files.
    """
    n_types = len(atoms)
    rmax = 0.0
    sph_rad = 0.0
    nx = ny = nz = 0

    if grid_mode == "ml":
        if grid_data is None:
            raise SystemExit("Grid data is required to determine domain radius and grid.")
        if radius_override is not None:
            print("Warning: radius override ignored when using ML grid.")
        return {
            "radius": grid_data["radius"],
            "radius_source": "ml_grid",
            "radius_auto": None,
            "radius_auto_raw": None,
            "radius_requested": radius_override,
            "nx": grid_data["nx"],
            "ny": grid_data["ny"],
            "nz": grid_data["nz"],
            "h": h,
        }, n_types

    for at_typ in range(n_types):
        typ = atoms[at_typ]["typ"]
        xyz = atoms[at_typ]["coord"]
        rsize = 0.0
        for index in range(n_elements):
            if typ == elem["Element"].iloc[index]:
                rsize = elem["r"].iloc[index]

        for atom_index in range(xyz.shape[0]):
            xx, yy, zz = xyz[atom_index, 0], xyz[atom_index, 1], xyz[atom_index, 2]
            rdis = np.sqrt(xx**2 + yy**2 + zz**2)
            rmax = max(rmax, rdis + rsize)

    sph_rad = rmax
    nx = int(2 * sph_rad / h) + 1
    nx = 2 * ((nx + 1) // 2)
    sph_rad = 0.5 * h * (nx - 1)
    auto_radius = sph_rad
    ny = nx
    nz = nx
    radius_source = "auto"
    radius_requested = None

    if radius_override is not None:
        try:
            radius_requested = float(radius_override)
            radius_source = "override"
            sph_rad = float(radius_override)
            nx = int(2 * sph_rad / h) + 1
            nx = 2 * ((nx + 1) // 2)
            sph_rad = 0.5 * h * (nx - 1)
            ny = nx
            nz = nx
        except (TypeError, ValueError):
            print("Warning: invalid radius override, keeping auto radius.")

    return {
        "radius": sph_rad,
        "radius_source": radius_source,
        "radius_auto": auto_radius,
        "radius_auto_raw": rmax,
        "radius_requested": radius_requested,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "h": h,
    }, n_types


def recenter_atoms(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a centered copy of the geometry for symmetric default SAD grids."""
    centered_atoms = deepcopy(atoms)
    all_coords = np.vstack([atom["coord"] for atom in centered_atoms])
    center = np.mean(all_coords, axis=0)
    for atom in centered_atoms:
        atom["coord"] -= center
    return centered_atoms


def build_formula_from_atoms(atoms: list[dict[str, Any]], n_atom: list[int], elem) -> str:
    """Build a compact chemical-formula label used in default output names."""
    counts: dict[str, int] = {}
    for atom, count in zip(atoms, n_atom):
        symbol = str(atom["typ"]).strip()
        counts[symbol] = counts.get(symbol, 0) + int(count)

    if not counts:
        return "system"

    order_map = {elem["Element"].iloc[index]: index for index in range(len(elem))}
    pieces = []
    for symbol in sorted(counts.keys(), key=lambda sym: (order_map.get(sym, 10**9), sym)):
        count = counts[symbol]
        pieces.append(f"{symbol}{count}" if count != 1 else symbol)
    return "".join(pieces)


def format_value_for_filename(value: float) -> str:
    """Format a float into a filesystem-friendly token such as ``1p25``."""
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    if formatted == "":
        formatted = "0"
    return formatted.replace(".", "p")


def density_method_tag(method: str) -> str:
    """Map internal density-mode names to shorter filename labels."""
    if method == "sad_ml_grid":
        return "sadwithml"
    return method


def build_default_output_basename(
    atoms: list[dict[str, Any]],
    n_atom: list[int],
    elem,
    density_method: str,
    diagmeth: int,
    radius_bohr: float,
    h_bohr: float,
) -> str:
    """Create the default basename used when the run is not file-driven."""
    formula = build_formula_from_atoms(atoms, n_atom, elem)
    method = density_method_tag(str(density_method).lower() if density_method else "sad")
    radius_ang = radius_bohr * A0_ANG
    h_ang = h_bohr * A0_ANG
    radius_str = format_value_for_filename(radius_ang)
    h_str = format_value_for_filename(h_ang)
    return f"{formula}_{method}_diagmeth{diagmeth}_{radius_str}A_{h_str}A"


def build_output_paths(
    input_data: InputData,
    settings: SolverSettings,
    elem,
    domain: dict[str, Any],
) -> RunPaths:
    """Create the output/log/wavefunction paths for the current run.

    For manual-input runs, outputs are placed inside ``./<base>/`` so the
    generated log, density files, and optional wavefunction file stay grouped
    together in one folder named after the run basename.
    """
    if input_data.input_file_path:
        base = os.path.splitext(os.path.basename(input_data.input_file_path))[0]
        out_dir = os.path.dirname(input_data.input_file_path) or "."
        output_file = os.path.join(out_dir, f"{base}.out")
    else:
        base = build_default_output_basename(
            input_data.atoms,
            input_data.n_atom,
            elem,
            input_data.density_method,
            settings.diagmeth,
            domain["radius"],
            domain["h"],
        )
        out_dir = os.path.join(".", base)
        output_file = os.path.join(out_dir, f"{base}.out")

    return RunPaths(
        output_file=output_file,
        wfn_file=f"{os.path.splitext(output_file)[0]}_wfn.dat",
        initial_density_base=f"{os.path.splitext(output_file)[0]}_init_rho",
        converged_density_base=f"{os.path.splitext(output_file)[0]}_conv_rho",
    )


def prepare_system(
    input_data: InputData,
    settings: SolverSettings,
    elem,
    n_elements: int,
) -> PreparedSystem:
    """Convert parsed input into the fully prepared solver problem.

    Input:
        input_data: geometry, charge, density mode, and parsed overrides.
        settings: normalized solver settings after applying overrides.

    Output:
        ``PreparedSystem`` containing the run geometry, domain, paths, and
        derived numerical quantities needed by the SCF solver.
    """
    nev_override = input_data.settings_overrides.get("nev")
    h_override = input_data.settings_overrides.get("grid_spacing", input_data.settings_overrides.get("h"))
    radius_override = input_data.settings_overrides.get(
        "sphere_radius", input_data.settings_overrides.get("radius")
    )

    h, nev, zelec, ztest = calculate_grid_spacing(
        input_data.atoms,
        elem,
        n_elements,
        input_data.n_atom,
        input_data.z_charge,
        h_override=h_override,
        nev_override=nev_override,
        grid_mode=input_data.grid_mode,
        grid_data=input_data.grid_data,
    )

    atoms_for_run = input_data.atoms
    if (
        input_data.density_method == "sad"
        and input_data.grid_mode == "default"
        and settings.recenter_atoms
    ):
        atoms_for_run = recenter_atoms(input_data.atoms)
        print("recenter atoms", atoms_for_run)

    domain, n_types = estimate_radius_and_grid(
        atoms_for_run,
        elem,
        n_elements,
        h,
        radius_override=radius_override,
        grid_mode=input_data.grid_mode,
        grid_data=input_data.grid_data,
    )

    prepared_input = InputData(
        atoms=atoms_for_run,
        n_atom=list(input_data.n_atom),
        z_charge=input_data.z_charge,
        density_method=input_data.density_method,
        ml_file_path=input_data.ml_file_path,
        settings_overrides=dict(input_data.settings_overrides),
        input_file_path=input_data.input_file_path,
        grid_npy_path=input_data.grid_npy_path,
        grid_poscar_path=input_data.grid_poscar_path,
        unit_used=input_data.unit_used,
        grid_mode=input_data.grid_mode,
        grid_data=deepcopy(input_data.grid_data),
    )
    paths = build_output_paths(prepared_input, settings, elem, domain)
    return PreparedSystem(
        input_data=prepared_input,
        settings=settings,
        domain=domain,
        h=h,
        nev=nev,
        zelec=zelec,
        ztest=ztest,
        n_types=n_types,
        paths=paths,
    )
