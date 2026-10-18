"""Command line of the ML initial-density interface.

    python src/parsec_python/ml_density.py models
    python src/parsec_python/ml_density.py check --provider charge3net
    python src/parsec_python/ml_density.py predict --input parsec.in
    python src/parsec_python/ml_density.py predict --xyz mol.xyz --vacuum 5 --spacing 0.2

``predict`` writes the density a calculation imports with
``ML_Density_File``; it does not run any DFT.  See ``MLDensity/README.md``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from .box import atomic_number, box_around_atoms, read_xyz
from .field import (
    BOHR_TO_ANGSTROM,
    dense_from_grid,
    save_box_density,
    save_point_density,
    write_cube,
)
from .providers import (
    _grid_box,
    electron_count_warning,
    physical_symbols,
    predict_density,
    resolve_provider,
)
from .registry import get_provider, provider_names


# A small molecule every built-in model knows, for ``check``.  Angstrom.
_CHECK_SYMBOLS = ["O", "H", "H"]
_CHECK_POSITIONS = np.asarray(
    [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
)
_CHECK_VALENCE = 8.0


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model (default: from --input, then discovery)")
    group.add_argument(
        "--provider",
        help=f"density model: {', '.join(provider_names())}",
    )
    group.add_argument("--model", help="ML_Density_Model, e.g. qm9 or fast")
    group.add_argument("--repository", type=Path, help="checkout of the model code")
    group.add_argument("--python", type=Path, help="interpreter of the model environment")
    group.add_argument("--checkpoint", type=Path, help="trained weights")
    group.add_argument("--bridge", type=Path, help="bridge script of --provider custom")
    group.add_argument("--device", choices=("auto", "cpu", "cuda"))
    group.add_argument("--chunk-size", type=int, help="points per model call (default 50000)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml_density.py",
        description=(
            "Predict machine-learned valence densities for PARSEC.py and check "
            "the model environments.  Lengths are in angstrom."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("models", help="list the density models and where their files are")

    check = commands.add_parser(
        "check", help="run each model once on a water molecule to test its environment"
    )
    _add_model_options(check)

    predict = commands.add_parser(
        "predict",
        help="predict a density on the grid of a parsec.in, or on a box",
        description=(
            "Predict a density and write it as .npz (and optionally .cube).  "
            "With --input and no box option the density is evaluated at the "
            "exact grid points of that calculation, the best start for it.  "
            "Box options (--box/--vacuum, --points, --spacing; give two) "
            "evaluate it on a regular box centred on the atoms instead."
        ),
    )
    atoms = predict.add_mutually_exclusive_group(required=True)
    atoms.add_argument("--input", type=Path, help="parsec.in: its atoms (and grid)")
    atoms.add_argument("--xyz", type=Path, help="XYZ file in angstrom")
    predict.add_argument("--pp-dir", type=Path, help="pseudopotential directory of --input")
    predict.add_argument(
        "--no-center",
        action="store_true",
        help="keep the XYZ coordinates (default: recentre as PARSEC.py does)",
    )
    box = predict.add_argument_group("box (two of: size, points, spacing)")
    box.add_argument("--box", type=float, nargs="+", metavar="L", help="edge lengths, 1 or 3 values")
    box.add_argument(
        "--vacuum", type=float, nargs="+", metavar="V",
        help="space added on each side of the atoms, 1 or 3 values (default 5)",
    )
    box.add_argument(
        "--points", "--grid", type=int, nargs="+", metavar="N",
        help="points per direction, 1 or 3 values",
    )
    box.add_argument(
        "--spacing", type=float, nargs="+", metavar="H",
        help="grid spacing, 1 or 3 values (default 0.2)",
    )
    predict.add_argument("-o", "--output", type=Path, help="density file to write (.npz)")
    predict.add_argument("--cube", type=Path, help="also write a Gaussian cube file")
    predict.add_argument("--quiet", action="store_true", help="no progress lines")
    _add_model_options(predict)
    return parser


def _model_options(args: argparse.Namespace, settings=None) -> dict:
    """Options of :func:`resolve_provider`, the input's below the command line's."""

    inherited = {}
    if settings is not None and settings.method == args.provider:
        inherited = {
            "model": settings.model,
            "repository": settings.repository,
            "python": settings.python_executable,
            "checkpoint": settings.checkpoint,
            "bridge": settings.bridge,
        }
    options = {}
    for name in ("model", "repository", "python", "checkpoint", "bridge"):
        value = getattr(args, name)
        options[name] = value if value is not None else inherited.get(name)
    return options


def _run_options(args: argparse.Namespace, settings=None) -> tuple[str, int]:
    inherited = settings is not None and settings.method == args.provider
    device = args.device or (settings.device if inherited else "auto")
    chunk = args.chunk_size or (settings.prediction_chunk_size if inherited else 50000)
    if chunk < 1:
        raise SystemExit("--chunk-size must be positive")
    return device, chunk


def _relative(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path, start)
    except ValueError:  # another drive on Windows
        return str(path)


def _summary(name: str, prediction, points: int, integral: float, electrons=None) -> None:
    density = prediction.density
    negative = int(np.count_nonzero(density < 0))
    device = prediction.metadata.get("device", "?")
    print(f"{name}: {points:,} points in {prediction.seconds:.1f} s on {device}")
    line = f"  electrons in the prediction: {integral:.4f}"
    if electrons is not None:
        line += f" (valence electrons of the calculation: {electrons:g})"
    print(line)
    message = None if electrons is None else electron_count_warning(integral, electrons, name)
    if message is not None:
        print(f"  WARNING: {message}")
    if negative:
        print(
            f"  {negative:,} negative values (min {density.min():.3e} e/bohr^3); "
            "the SCF import clips them and normalizes the integral"
        )


def _models() -> int:
    for name in provider_names():
        provider = get_provider(name)
        models = ", ".join(
            f"{model} (default)" if model == provider.default_model else model
            for model in provider.models
        ) or "any label"
        print(f"{name:11s} {provider.title}: {provider.summary}")
        print(f"            models: {models}")
        if provider.bridge is None:
            print("            needs ML_Density_Bridge (see bridges/template.py)")
            continue
        try:
            resolved = resolve_provider(name)
        except (FileNotFoundError, ValueError) as error:
            print(f"            not ready: {error}")
            continue
        for item in ("repository", "python", "checkpoint"):
            print(
                f"            {item:10s} {getattr(resolved, item)}  "
                f"[{resolved.origins[item]}]"
            )
    return 0


def _check(args: argparse.Namespace) -> int:
    if args.provider:
        names = [args.provider]
    else:
        names = [name for name in provider_names() if get_provider(name).bridge is not None]
        if args.bridge is not None:
            names.append("custom")
    positions = _CHECK_POSITIONS / BOHR_TO_ANGSTROM
    box = box_around_atoms(
        positions, vacuum_bohr=3.0 / BOHR_TO_ANGSTROM, spacing_bohr=0.25 / BOHR_TO_ANGSTROM
    )
    failures = 0
    for name in names:
        print(f"== {name}")
        try:
            # Paths and a model name belong to one provider: with several,
            # each is found by discovery (the bridge belongs to custom).
            options = (
                _model_options(args)
                if args.provider
                else {"bridge": args.bridge if name == "custom" else None}
            )
            resolved = resolve_provider(name, **options)
            for item in ("bridge", "repository", "python", "checkpoint"):
                print(f"  {item:10s} {getattr(resolved, item)}  [{resolved.origins[item]}]")
            device, chunk = _run_options(args)
            prediction = predict_density(
                resolved,
                _CHECK_SYMBOLS,
                positions,
                box.points(),
                device=device,
                chunk_size=chunk,
                lower_bohr=box.lower_bohr,
                cell_bohr=box.lengths_bohr,
                progress=lambda message: print(f"  {message}", flush=True),
            )
        except Exception as error:  # report every model, then fail
            failures += 1
            print(f"  FAILED: {error}")
            continue
        integral = box.integrate(prediction.density)
        _summary(f"  {resolved.provider.title} {resolved.model}", prediction, box.size, integral)
        if abs(integral - _CHECK_VALENCE) > 0.1 * _CHECK_VALENCE:
            print(
                f"  WARNING: water has {_CHECK_VALENCE:g} valence electrons; this "
                "model does not reproduce that (a custom toy bridge need not)"
            )
        else:
            print("  OK")
    return 1 if failures else 0


def _box_arguments(args: argparse.Namespace) -> dict | None:
    """Box options in bohr, with the defaults, or ``None`` for no box."""

    given = [args.box, args.vacuum, args.points, args.spacing]
    if all(item is None for item in given):
        return None
    to_bohr = lambda values: None if values is None else np.asarray(values) / BOHR_TO_ANGSTROM
    box, vacuum = to_bohr(args.box), to_bohr(args.vacuum)
    points, spacing = args.points, to_bohr(args.spacing)
    if box is None and vacuum is None and not (points is not None and spacing is not None):
        vacuum = 5.0 / BOHR_TO_ANGSTROM
    if (box is not None or vacuum is not None) and points is None and spacing is None:
        spacing = 0.2 / BOHR_TO_ANGSTROM
    return {"box_bohr": box, "vacuum_bohr": vacuum, "points": points, "spacing_bohr": spacing}


def _predict(args: argparse.Namespace) -> int:
    from ..models import Atom
    from ..V_ion import center_cluster_geometry

    progress = None if args.quiet else (lambda message: print(f"  {message}", flush=True))
    settings = None
    electrons = None
    grid = None
    if args.input is not None:
        from ..Grid import build_cluster_grid
        from ..Input import parse_parsec_input
        from ..V_ion import ionic_charge, load_pseudopotentials

        problem = parse_parsec_input(
            args.input, pseudopotential_directory=args.pp_dir
        ).problem
        if problem.periodic_cell is not None:
            raise SystemExit("ML densities are for isolated systems; this input is periodic")
        settings = problem.initial_density_settings
        if args.provider is None:
            if settings.method in {"sad", "file"}:
                raise SystemExit(
                    "the input has Initial_Density: "
                    f"{settings.method}; choose a model with --provider"
                )
            args.provider = settings.method
        atoms = (
            center_cluster_geometry(problem.atoms)
            if problem.recenter_geometry
            else tuple(problem.atoms)
        )
        symbols = physical_symbols(atoms, problem.pseudopotentials)
        positions = np.asarray([atom.position for atom in atoms], dtype=float)
        potentials = load_pseudopotentials(
            problem.pseudopotentials, xc_functional=problem.scf.xc_functional
        )
        electrons = ionic_charge(atoms, potentials) - problem.scf.net_charge
        recentred = problem.recenter_geometry
        base, stem = args.input.resolve().parent, None
    else:
        if args.provider is None:
            raise SystemExit("choose a model with --provider")
        symbols, positions_angstrom = read_xyz(args.xyz)
        positions = positions_angstrom / BOHR_TO_ANGSTROM
        if not args.no_center:
            positions = np.asarray(
                [
                    atom.position
                    for atom in center_cluster_geometry(
                        [Atom(symbol, position) for symbol, position in zip(symbols, positions)]
                    )
                ]
            )
        recentred = not args.no_center
        base, stem = args.xyz.resolve().parent, args.xyz.stem

    resolved = resolve_provider(args.provider, **_model_options(args, settings))
    device, chunk = _run_options(args, settings)
    box_options = _box_arguments(args)
    if box_options is None and args.input is None:
        box_options = _box_arguments(
            argparse.Namespace(box=None, vacuum=[5.0], points=None, spacing=None)
        )
    name = f"{resolved.provider.title} {resolved.model}"
    label = f"{resolved.name}_{resolved.model}"
    metadata = {"model": resolved.model}
    if resolved.checkpoint is not None:
        metadata["checkpoint"] = str(resolved.checkpoint)

    if box_options is None:
        grid = build_cluster_grid(problem.grid)
        lower, cell = _grid_box(grid)
        print(f"{name}: exact grid of {args.input.name}, {grid.size:,} points", flush=True)
        prediction = predict_density(
            resolved, symbols, positions, grid.coordinates,
            device=device, chunk_size=chunk, lower_bohr=lower, cell_bohr=cell,
            progress=progress,
        )
        integral = grid.integrate(prediction.density)
        output = args.output or base / f"{label}_density.npz"
        save_point_density(
            output, prediction.density, grid.coordinates,
            units="e_per_bohr3", provider=resolved.name,
            metadata={**prediction.metadata, **metadata},
            symbols=symbols, atom_positions_bohr=positions,
        )
        _summary(name, prediction, grid.size, integral, electrons)
        if args.cube is not None:
            dense, origin, vectors = dense_from_grid(prediction.density, grid)
            write_cube(
                args.cube, dense, origin, vectors,
                [atomic_number(symbol) for symbol in symbols], positions,
                comment=f"{name} density on the grid of {args.input.name}",
            )
    else:
        box = box_around_atoms(positions, **box_options)
        lengths = " x ".join(f"{value * BOHR_TO_ANGSTROM:.3f}" for value in box.lengths_bohr)
        steps = " x ".join(f"{value * BOHR_TO_ANGSTROM:.4f}" for value in box.spacing_bohr)
        print(
            f"{name}: box {lengths} A, "
            f"{' x '.join(str(value) for value in box.shape)} = {box.size:,} points, "
            f"spacing {steps} A",
            flush=True,
        )
        prediction = predict_density(
            resolved, symbols, positions, box.points(),
            device=device, chunk_size=chunk,
            lower_bohr=box.lower_bohr, cell_bohr=box.lengths_bohr,
            progress=progress,
        )
        integral = box.integrate(prediction.density)
        output = args.output or base / (
            f"{stem}_{label}_box.npz" if stem else f"{label}_box.npz"
        )
        dense = prediction.density.reshape(box.shape)
        save_box_density(
            output, dense, box.origin_bohr, box.voxel_vectors_bohr,
            units="e_per_bohr3", provider=resolved.name,
            metadata={**prediction.metadata, **metadata, "recentred": str(recentred)},
            symbols=symbols, atom_positions_bohr=positions,
        )
        _summary(name, prediction, box.size, integral, electrons)
        if args.cube is not None:
            write_cube(
                args.cube, dense, box.origin_bohr, box.voxel_vectors_bohr,
                [atomic_number(symbol) for symbol in symbols], positions,
                comment=f"{name} density",
            )
    print(f"  wrote {output}")
    if args.cube is not None:
        print(f"  wrote {args.cube}")
    target = args.input.resolve().parent if args.input is not None else None
    reference = _relative(Path(output).resolve(), target) if target else str(Path(output).resolve())
    print("To start an SCF from it, put in parsec.in:")
    print(f"  Initial_Density: {resolved.name}")
    print(f"  ML_Density_File: {reference}")
    if grid is None:
        print("  ML_Density_Interpolation: linear")
        if args.xyz is not None:
            print(
                "  (the box holds the atoms "
                + ("recentred as PARSEC.py does" if recentred else "at their XYZ coordinates")
                + "; the import refuses a calculation in another frame)"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "models":
        return _models()
    if args.command == "check":
        return _check(args)
    try:
        return _predict(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


__all__ = ["main"]
