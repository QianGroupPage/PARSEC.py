"""Command line of the force-field structures (``src/parsec_python/mlff.py``).

::

    python src/parsec_python/mlff.py relax calculation/parsec.in --model uma-s-1p2
    python src/parsec_python/mlff.py models --python /path/to/env/bin/python
    python src/parsec_python/mlff.py check uma-s-1p2

``relax`` relaxes the structure of an input as ``Initial_Structure: mlff``
does, without a DFT calculation, and writes ``mlff_structure.xyz`` (which a
run with the same settings then takes instead of relaxing again, and which
``--restart-from`` accepts).  ``models`` lists the models and whether an
interpreter can run them; ``check`` builds one model and evaluates a water
molecule.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

from .run import STRUCTURE_FILE, TRAJECTORY_FILE, WORKER, MLFFError, relax_initial_structure
from .settings import DEFAULT_MODEL, MLFFSettings, interpreter

#: Model families: names, package, notes.
MODELS = (
    (
        "uma-s-1p2 (default), uma-s-1p2p1, uma-s-1p1, uma-m-1p1 (UMA); esen-*, allscaip-* (fairchem)",
        "fairchem.core",
        "pip install fairchem-core; weights gated: accept the licence at "
        "https://huggingface.co/facebook/UMA, then `hf auth login`. MLFF_Task omol "
        "for molecules and clusters (charge and spin multiplicity are used).",
    ),
    (
        "mace-omol (MACE-OMOL-0, molecules; failed for clusters of a few hundred atoms)",
        "mace",
        "pip install mace-torch; Academic Software License.",
    ),
    (
        "mace-off-small|medium|large (MACE-OFF23, organic: H C N O F P S Cl Br I)",
        "mace",
        "pip install mace-torch; Academic Software License.",
    ),
    (
        "mace-mp-<name>, e.g. mace-mp-medium-mpa-0, mace-mp-small (materials)",
        "mace",
        "pip install mace-torch.",
    ),
    ("file.model (MACE), file.pt (fairchem checkpoint)", "", "a local file, relative to the input."),
    (
        "python:module:function, python:file.py:function",
        "",
        "any ASE calculator returned by your function(device, task, charge, spin).",
    ),
)


def _settings_for(translation, arguments) -> MLFFSettings:
    settings = getattr(translation, "mlff", None) or MLFFSettings()
    changes = {}
    for name in ("model", "task", "fmax", "steps", "optimizer", "device", "charge", "spin"):
        value = getattr(arguments, name, None)
        if value is not None:
            changes[name] = value
    if arguments.python is not None:
        changes["python"] = Path(arguments.python)
    if arguments.again:
        changes["reuse"] = False
    return replace(settings, **changes)


def _relax(arguments) -> int:
    from ..Input import parse_parsec_input

    translation = parse_parsec_input(arguments.input)
    translation = replace(translation, mlff=_settings_for(translation, arguments))
    directory = Path(arguments.output_dir) if arguments.output_dir else translation.source.parent
    directory.mkdir(parents=True, exist_ok=True)
    try:
        relaxed, notes = relax_initial_structure(translation, directory, progress=print)
    except MLFFError as error:
        print(f"MLFF error: {error}", file=sys.stderr)
        return 3
    for note in notes:
        print(note)
    print(f"Structure: {directory / STRUCTURE_FILE} (angstrom); steps: {directory / TRAJECTORY_FILE}")
    print(
        "Start a DFT run from it with `--restart-from "
        f"{directory / STRUCTURE_FILE}`, or with Initial_Structure: mlff and the same settings in the "
        "input, which takes the file instead of relaxing again."
    )
    return 0


def _probe(python: str, module: str) -> str:
    code = (
        "import importlib.util, sys; spec = importlib.util.find_spec(sys.argv[1].split('.')[0]); "
        "print('yes' if spec else 'no')"
    )
    try:
        completed = subprocess.run(
            [python, "-c", code, module], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"cannot run ({error})"
    return "installed" if completed.stdout.strip() == "yes" else "not installed"


def _models(arguments) -> int:
    python = interpreter(MLFFSettings(python=None if arguments.python is None else Path(arguments.python)))
    print(f"Interpreter: {python}")
    print(f"Default model: {DEFAULT_MODEL}\n")
    for names, module, note in MODELS:
        state = f" [{module.split('.')[0]}: {_probe(python, module)}]" if module else ""
        print(f"  {names}{state}\n      {note}")
    print(
        "\nName another interpreter with MLFF_Python in parsec.in, --python here, or the variable "
        "PARSEC_MLFF_PYTHON; it needs NumPy, ASE and the model package, not parsec_python."
    )
    return 0


def _check(arguments) -> int:
    python = interpreter(MLFFSettings(python=None if arguments.python is None else Path(arguments.python)))
    command = [python, str(WORKER), "--check", arguments.model_name, "--device", arguments.device or "auto"]
    return subprocess.run(command).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlff.py", description="Structures from machine-learned force fields for parsec_python."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    relax = commands.add_parser("relax", help="relax the structure of an input with a force field")
    relax.add_argument("input", type=Path, help="parsec.in")
    relax.add_argument("--model", help=f"MLFF_Model (default: the input's, else {DEFAULT_MODEL})")
    relax.add_argument("--task", help="UMA task (default omol)")
    relax.add_argument("--fmax", type=float, help="force criterion in eV/angstrom (default 0.05)")
    relax.add_argument("--steps", type=int, help="largest number of steps (default 500)")
    relax.add_argument("--optimizer", choices=("lbfgs", "bfgs", "fire"))
    relax.add_argument("--device", choices=("auto", "cuda", "cpu"))
    relax.add_argument("--charge", type=int, help="charge (default: Net_Charges of the input)")
    relax.add_argument("--spin", type=int, help="spin multiplicity 2S+1 (default 1 or 2)")
    relax.add_argument("--python", help="interpreter that has the model package")
    relax.add_argument("--output-dir", help="where the files go (default: beside the input)")
    relax.add_argument("--again", action="store_true", help="relax even if mlff_structure.xyz matches")
    models = commands.add_parser("models", help="list the models and what an interpreter can run")
    models.add_argument("--python", help="interpreter to look into")
    check = commands.add_parser("check", help="build a model (download its weights) and evaluate a water molecule")
    check.add_argument("model_name", metavar="MODEL", nargs="?", default=DEFAULT_MODEL,
                       help=f"default {DEFAULT_MODEL}")
    check.add_argument("--python", help="interpreter that has the model package")
    check.add_argument("--device", choices=("auto", "cuda", "cpu"))
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "relax":
        try:
            return _relax(arguments)
        except ValueError as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2
    if arguments.command == "models":
        return _models(arguments)
    return _check(arguments)


__all__ = ["MODELS", "build_parser", "main"]
