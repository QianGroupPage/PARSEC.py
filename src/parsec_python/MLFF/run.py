"""The initial structure of a DFT run from a machine-learned force field.

With ``Initial_Structure: mlff`` the coordinates of the input are relaxed
with a force field before the calculation is prepared, and the relaxed ones
replace them, as an XYZ restart file would: a single point, a relaxation or
molecular dynamics then starts from that structure.  The force field runs in
its own process (:mod:`.worker`), with the interpreter that has the model
package, and the files of the run receive

- ``mlff_structure.xyz``: the relaxed structure in angstrom, which a later
  run with the same input and settings takes again instead of relaxing, and
  which ``--restart-from`` accepts;
- ``mlff_relaxation.xyz``: every optimizer step with its energy and forces
  (eV, eV/angstrom);
- ``mlff.log``: the table of the ASE optimizer.

The point group the DFT run will use is kept: forces and positions of the
force-field relaxation are averaged over the operations of the centred input
structure that respect the movable flags, and atoms that ``Move_Flag`` fixes
stay where they are.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable

import numpy as np

from ..models import Atom
from .settings import MLFFSettings, interpreter

#: PARSEC's bohr in angstrom, as in the files of a relaxation.
BOHR_TO_ANGSTROM = 0.529177
STRUCTURE_FILE = "mlff_structure.xyz"
TRAJECTORY_FILE = "mlff_relaxation.xyz"
LOG_FILE = "mlff.log"
WORKER = Path(__file__).with_name("worker.py")


class MLFFError(RuntimeError):
    """The force field could not relax the structure."""


def _elements(problem) -> list[str]:
    elements = []
    for atom in problem.atoms:
        specification = problem.pseudopotentials.get(atom.symbol)
        element = getattr(specification, "element_symbol", None) or atom.symbol
        elements.append(element[:1].upper() + element[1:].lower())
    return elements


def _start(translation):
    """Centred positions (bohr) and the point group the DFT run keeps."""

    from ..Forces import atom_symmetry_operations
    from ..Forces.symmetry import operations_preserving
    from ..Relaxation.run import symmetry_decision
    from ..V_ion import center_cluster_geometry

    problem = translation.problem
    atoms = (
        center_cluster_geometry(problem.atoms)
        if problem.recenter_geometry
        else tuple(problem.atoms)
    )
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    use_symmetry, _note = symmetry_decision(translation)
    operations = ()
    if use_symmetry:
        movable = translation.movable or (True,) * len(atoms)
        operations = operations_preserving(atom_symmetry_operations(atoms), movable)
    return positions, operations


def _key(request: dict) -> str:
    """What the relaxed structure depends on, as a short digest."""

    fields = {
        name: request[name]
        for name in ("symbols", "fixed", "model", "task", "charge", "spin", "fmax", "steps", "optimizer")
    }
    fields["positions"] = np.round(np.asarray(request["positions"]), 9).tolist()
    fields["symmetry"] = None if request["symmetry"] is None else request["symmetry"]["images"]
    text = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_structure_file(path: Path) -> tuple[str | None, list[str], np.ndarray, dict]:
    """``(key, symbols, positions in angstrom, fields)`` of ``mlff_structure.xyz``."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    count = int(lines[0].split()[0])
    fields = {}
    for word in lines[1].split():
        name, separator, value = word.partition("=")
        if separator:
            fields[name] = value.strip('"')
    symbols, positions = [], []
    for line in lines[2 : 2 + count]:
        words = line.split()
        symbols.append(words[0])
        positions.append([float(value) for value in words[1:4]])
    return fields.get("parsec_mlff_key"), symbols, np.asarray(positions, dtype=np.float64), fields


def _write_structure_file(path: Path, symbols, positions, key: str, result: dict) -> None:
    energies = result.get("energies") or [float("nan")]
    maxima = result.get("fmax") or [float("nan")]
    lines = [
        str(len(symbols)),
        f"parsec_mlff_key={key} model=\"{result.get('model', '')}\" energy={energies[-1]:.10f} "
        f"fmax={maxima[-1]:.6f} steps={result.get('steps', 0)} converged={result.get('converged', False)} "
        f"charge={result.get('charge', 0)} spin={result.get('spin', 1)}",
    ]
    for symbol, position in zip(symbols, positions):
        lines.append(f"{symbol:<3s} {position[0]:16.10f} {position[1]:16.10f} {position[2]:16.10f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _worker_environment(python: str) -> dict:
    environment = dict(os.environ)
    if os.path.normcase(os.path.abspath(python)) != os.path.normcase(os.path.abspath(sys.executable)):
        # Paths of this interpreter would shadow the packages of the other one.
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


def run_worker(request: dict, settings: MLFFSettings, progress: Callable[[str], None] | None) -> dict:
    """Run :mod:`.worker` on ``request`` and return its result."""

    python = interpreter(settings)
    with tempfile.TemporaryDirectory(prefix="parsec_mlff_") as scratch:
        request = dict(request, result=str(Path(scratch) / "result.json"))
        request_path = Path(scratch) / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        errors: list[str] = []
        try:
            process = subprocess.Popen(
                [python, str(WORKER), str(request_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_worker_environment(python),
                cwd=scratch,
            )
        except OSError as error:
            raise MLFFError(f"MLFF_Python {python} cannot be started: {error}") from error
        with process:

            def drain_errors():
                for line in process.stderr:
                    errors.append(line.rstrip())

            reader = threading.Thread(target=drain_errors, daemon=True)
            reader.start()
            for line in process.stdout:
                if progress is not None:
                    progress(line.rstrip())
            code = process.wait()
            reader.join()
        result_path = Path(request["result"])
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    if "error" in result:
        raise MLFFError(result["error"])
    if code != 0 or "positions" not in result:
        tail = "\n".join(line for line in errors[-15:] if line.strip())
        raise MLFFError(f"the force field ended with exit status {code} ({python}):\n{tail}")
    return result


def relax_initial_structure(
    translation,
    directory: Path,
    *,
    restarted: bool = False,
    progress: Callable[[str], None] | None = None,
):
    """Apply ``Initial_Structure: mlff``; return ``(translation, notes)``.

    ``directory`` receives the files of the module text.  ``restarted``
    says that a restart file already replaced the coordinates of the input:
    the run then continues from it and the force field is not used.
    ``progress`` receives the lines the force field prints while it runs.
    """

    settings: MLFFSettings | None = getattr(translation, "mlff", None)
    if settings is None:
        return translation, []
    if restarted:
        return translation, [
            "NOTE: Initial_Structure: mlff is not used, the run continues from its restart structure."
        ]
    relaxation = translation.relaxation
    if relaxation is not None and relaxation.method == "manual":
        return translation, [
            "NOTE: Initial_Structure: mlff is not used in a manual run, whose structures are imposed."
        ]
    if translation.problem.periodic_cell is not None:
        raise MLFFError("Initial_Structure: mlff is implemented for isolated systems only")
    problem = translation.problem
    directory = Path(directory)
    positions, operations = _start(translation)
    symbols = _elements(problem)
    charge = settings.charge
    if charge is None:
        charge = int(round(float(problem.scf.net_charge)))
    movable = translation.movable or (True,) * len(symbols)
    request = {
        "symbols": symbols,
        "positions": (positions * BOHR_TO_ANGSTROM).tolist(),
        "fixed": [index for index, flag in enumerate(movable) if not flag],
        "symmetry": (
            {
                "matrices": [operation.matrix.tolist() for operation in operations],
                "images": [operation.image.tolist() for operation in operations],
            }
            if len(operations) > 1
            else None
        ),
        "model": settings.model,
        "task": settings.task,
        "charge": charge,
        "spin": settings.spin,
        "fmax": settings.fmax,
        "steps": settings.steps,
        "optimizer": settings.optimizer,
        "device": settings.device,
        "log": str(directory / LOG_FILE),
        "trajectory": str(directory / TRAJECTORY_FILE),
    }
    key = _key(request)
    structure_path = directory / STRUCTURE_FILE
    notes = []
    started = time.perf_counter()
    if settings.reuse and structure_path.is_file():
        try:
            stored_key, stored_symbols, stored, fields = read_structure_file(structure_path)
        except (OSError, ValueError, IndexError):
            stored_key = None
        if stored_key == key and stored_symbols == symbols:
            relaxed = stored / BOHR_TO_ANGSTROM
            notes.append(
                f" Initial structure from {STRUCTURE_FILE}, relaxed by an earlier run with the same input "
                f"and settings ({fields.get('model', settings.model)}, largest force "
                f"{float(fields.get('fmax', 'nan')):.4f} eV/ang); delete the file to relax again"
            )
            return _replaced(translation, relaxed), notes
    result = run_worker(request, settings, progress)
    relaxed_angstrom = np.asarray(result["positions"], dtype=np.float64)
    if relaxed_angstrom.shape != positions.shape or not np.all(np.isfinite(relaxed_angstrom)):
        raise MLFFError("the force field returned no finite structure of the input's atoms")
    # The digits of the file, so that a run that takes the file again starts
    # from the same numbers.
    relaxed_angstrom = np.vectorize(lambda value: float(f"{value:.10f}"))(relaxed_angstrom)
    relaxed = relaxed_angstrom / BOHR_TO_ANGSTROM
    displacement = float(np.linalg.norm(relaxed - positions, axis=1).max()) * BOHR_TO_ANGSTROM
    energies, maxima = result["energies"], result["fmax"]
    _require_sensible(translation, result, relaxed, settings)
    _write_structure_file(structure_path, symbols, relaxed_angstrom, key, result)
    notes += [
        f" Initial structure relaxed with the force field {result['model']} on {result['device']}",
        f"   charge {result['charge']}, spin multiplicity {result['spin']}, "
        + (f"{len(operations)} symmetry operations kept" if len(operations) > 1 else "no symmetry imposed")
        + (f", {len(request['fixed'])} atoms fixed" if request["fixed"] else ""),
        f"   {result['steps']} steps of {settings.optimizer.upper()}: energy {energies[0]:.6f} -> "
        f"{energies[-1]:.6f} eV, largest force {maxima[0]:.4f} -> {maxima[-1]:.4f} eV/ang, "
        f"largest displacement {displacement:.4f} ang",
        f"   model loaded in {result['seconds']['load']:.1f} s, relaxed in {result['seconds']['relax']:.1f} s "
        f"({time.perf_counter() - started:.1f} s in all); {STRUCTURE_FILE}, {TRAJECTORY_FILE}",
    ]
    if not result["converged"]:
        notes.append(
            f"WARNING: the force-field relaxation stopped after {result['steps']} steps (MLFF_Steps) "
            f"with a largest force of {maxima[-1]:.4f} eV/ang, above MLFF_Fmax {settings.fmax:g}"
        )
    return _replaced(translation, relaxed), notes


#: Largest force, in eV/angstrom, that a force field may report for a
#: structure it describes; a stretched bond gives a few eV/angstrom.
SENSIBLE_FORCE = 50.0


def _require_sensible(translation, result: dict, relaxed: np.ndarray, settings: MLFFSettings) -> None:
    """Refuse a structure the force field did not describe.

    A model outside its training (an element, a size, a bonding it has not
    seen) can return forces of thousands of eV/angstrom and push atoms
    anywhere; the DFT run would then start from that.  The structure is
    refused when a force is not finite or above ``SENSIBLE_FORCE``, when the
    energy rose, or when an atom left the domain of the input.
    """

    from ..SCF.single_point import _require_atoms_inside_domain
    from ..V_ion import center_cluster_geometry

    energies = np.asarray(result.get("energies") or [np.nan], dtype=np.float64)
    maxima = np.asarray(result.get("fmax") or [np.nan], dtype=np.float64)
    advice = (
        f"; the input structure is not changed. Choose another MLFF_Model (mace-off-medium for organic "
        f"molecules, uma-s-1p2 with MLFF_Task omol, mace-mp-medium-mpa-0, ...) or remove Initial_Structure: mlff. The steps are "
        f"in {TRAJECTORY_FILE}"
    )
    worst = max(float(maxima[0]), float(maxima[-1]))
    count = len(relaxed)
    if not (np.all(np.isfinite(energies)) and np.all(np.isfinite(maxima))) or worst > SENSIBLE_FORCE:
        raise MLFFError(
            f"the force field {result.get('model', settings.model)} does not describe this structure: "
            f"largest force {maxima[0]:.4g} eV/ang at the start and {maxima[-1]:.4g} after "
            f"{result.get('steps', 0)} steps" + advice
        )
    if energies[-1] - energies[0] > 1.0e-3 * count:
        raise MLFFError(
            f"the energy of the force field {result.get('model', settings.model)} rose from "
            f"{energies[0]:.6f} to {energies[-1]:.6f} eV during the relaxation" + advice
        )
    problem = translation.problem
    atoms = tuple(Atom(atom.symbol, position) for atom, position in zip(problem.atoms, relaxed))
    if problem.recenter_geometry:
        atoms = center_cluster_geometry(atoms)
    try:
        _require_atoms_inside_domain(problem.grid, atoms)
    except ValueError as error:
        raise MLFFError(
            f"the force field moved an atom out of the domain of the input ({error})" + advice
        ) from error


def _replaced(translation, positions_bohr: np.ndarray):
    problem = translation.problem
    atoms = tuple(
        Atom(atom.symbol, np.asarray(position, dtype=np.float64))
        for atom, position in zip(problem.atoms, positions_bohr)
    )
    return replace(translation, problem=replace(problem, atoms=atoms))


def with_positions(translation, positions_bohr: np.ndarray):
    """``translation`` with these coordinates, for the ranks that did not relax."""

    return _replaced(translation, positions_bohr)


def apply_mlff_option(translation, model: str | None):
    """The command-line ``--mlff [MODEL]``: ``Initial_Structure: mlff`` with that model."""

    settings = getattr(translation, "mlff", None) or MLFFSettings()
    if model:
        settings = replace(settings, model=model)
    return replace(translation, mlff=settings)


__all__ = [
    "LOG_FILE",
    "MLFFError",
    "STRUCTURE_FILE",
    "TRAJECTORY_FILE",
    "apply_mlff_option",
    "read_structure_file",
    "relax_initial_structure",
    "run_worker",
    "with_positions",
]
