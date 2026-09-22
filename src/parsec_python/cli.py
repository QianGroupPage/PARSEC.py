"""Command-line runner for PARSEC-style isolated single-point inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

from .driver import prepare_single_point, prepare_periodic_single_point, run_scf
from .Input import (
    ParsecInputError,
    parse_parsec_input,
    summarize_translation,
)
from .models import SinglePointResult
from .Output import ParsecTextReporter
from .V_ion import load_pseudopotentials


class _RunLog:
    def __init__(self, path: Path | None, quiet: bool = False) -> None:
        self.path = path
        self.quiet = quiet
        self._stream = None
        self.failure_reported = False

    def __enter__(self) -> "_RunLog":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        if _value is not None:
            if isinstance(_value, KeyboardInterrupt):
                message = "Calculation interrupted by user."
            else:
                message = (
                    f"Calculation failed: {type(_value).__name__}: {_value}"
                )
            try:
                self.write(message)
                self.failure_reported = True
            except OSError:
                pass
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def write(self, message: str = "") -> None:
        if not self.quiet:
            print(message, flush=True)
        if self._stream is not None:
            self._stream.write(message + "\n")
            self._stream.flush()


def _resolve_output(
    requested: Path | None,
    input_directory: Path,
    default_name: str,
) -> Path:
    if requested is None:
        return input_directory / default_name
    requested = requested.expanduser()
    if requested.is_absolute():
        return requested.resolve()
    return (input_directory / requested).resolve()


def _npz_path(path: Path) -> Path:
    if path.suffix.lower() == ".npz":
        return path
    return path.with_suffix(path.suffix + ".npz")


def save_result_archive(
    path: str | Path,
    result: SinglePointResult,
    *,
    include_wavefunctions: bool = False,
) -> Path:
    """Save a portable NumPy result archive without wavefunctions by default."""
    output = _npz_path(Path(path).expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    energy = asdict(result.energies)
    history = np.asarray(
        [
            [
                item.iteration,
                item.weighted_residual,
                item.plain_residual,
                item.eigen_residual_max,
                item.hartree_residual,
                item.energies.total,
            ]
            for item in result.history
        ],
        dtype=float,
    )
    timing_history = np.asarray(
        [
            [
                item.iteration,
                getattr(item, "hamiltonian_binding_seconds", 0.0),
                getattr(item, "diagonalization_seconds", 0.0),
                getattr(item, "occupations_density_seconds", 0.0),
                getattr(item, "hartree_seconds", 0.0),
                getattr(item, "xc_seconds", 0.0),
                getattr(item, "mixing_energy_seconds", 0.0),
                getattr(item, "total_seconds", 0.0),
            ]
            for item in result.history
        ],
        dtype=float,
    ).reshape(-1, 8)
    payload: dict[str, np.ndarray] = {
        "atom_symbols": np.asarray([atom.symbol for atom in result.atoms]),
        "atom_coordinates_bohr": np.asarray(
            [atom.position for atom in result.atoms], dtype=float
        ),
        "coordinates_bohr": result.grid.coordinates,
        "integer_coordinates": result.grid.integer_coordinates,
        "density_e_per_bohr3": result.density,
        "core_density_e_per_bohr3": result.core_density,
        "ionic_potential_ry": result.ionic_potential,
        "hartree_potential_ry": result.hartree_potential,
        "xc_potential_ry": result.xc_potential,
        "input_effective_potential_ry": result.input_effective_potential,
        "output_effective_potential_ry": result.output_effective_potential,
        "next_effective_potential_ry": result.next_effective_potential,
        "eigenvalues_ry": result.eigenvalues,
        "occupations": result.occupations,
        "representations": np.asarray(
            getattr(
                result,
                "representations",
                np.ones(np.asarray(result.eigenvalues).size, dtype=np.int32),
            ),
            dtype=np.int32,
        ),
        "fermi_level_ry": np.asarray(result.fermi_level),
        "electron_count": np.asarray(result.electron_count),
        "atomic_reference_correction_ry": np.asarray(
            getattr(result, "atomic_reference_correction", 0.0)
        ),
        "all_electron_total_ry": np.asarray(
            getattr(result, "all_electron_total", result.energies.total)
        ),
        "converged": np.asarray(result.converged),
        "iterations": np.asarray(result.iterations),
        "scf_history": history,
        "scf_history_columns": np.asarray(
            [
                "iteration",
                "weighted_sre_ry",
                "plain_sre",
                "eigen_residual",
                "hartree_residual",
                "total_energy_ry",
            ]
        ),
        "scf_timing_history": timing_history,
        "scf_timing_history_columns": np.asarray(
            [
                "iteration",
                "hamiltonian_binding_seconds",
                "diagonalization_seconds",
                "occupations_density_seconds",
                "hartree_seconds",
                "xc_seconds",
                "mixing_energy_seconds",
                "total_seconds",
            ]
        ),
    }
    for name, value in energy.items():
        payload[f"energy_{name}_ry"] = np.asarray(value)
    # Timing metadata is additive: archives from result-like objects created by
    # older callers still save successfully, while native results expose both
    # aggregate SCF and preparation values as named scalar keys.
    timings = getattr(result, "timings", None)
    if timings is not None:
        for name in (
            "initial_hartree_seconds",
            "initial_xc_seconds",
            "hamiltonian_binding_seconds",
            "diagonalization_seconds",
            "occupations_density_seconds",
            "hartree_seconds",
            "xc_seconds",
            "mixing_energy_seconds",
            "total_seconds",
        ):
            value = getattr(timings, name, None)
            if value is not None:
                payload[f"timing_scf_{name}"] = np.asarray(value)
        preparation_timings = getattr(timings, "preparation", None)
        if preparation_timings is not None:
            for name in (
                "pseudopotential_loading_seconds",
                "grid_seconds",
                "finite_difference_seconds",
                "local_ionic_seconds",
                "nonlocal_ionic_seconds",
                "initial_density_seconds",
                "core_density_seconds",
                "ion_ion_seconds",
                "total_seconds",
            ):
                value = getattr(preparation_timings, name, None)
                if value is not None:
                    payload[f"timing_preparation_{name}"] = np.asarray(value)
    # AcceleratedSinglePointResult deliberately presents the same numerical
    # interface as SinglePointResult.  When that optional wrapper is supplied,
    # retain backend provenance without importing the acceleration package or
    # changing ordinary reference archives.
    backend = getattr(result, "backend", None)
    if backend is not None:
        for name in (
            "requested",
            "selected",
            "dtype",
            "device",
            "implementation",
        ):
            value = getattr(backend, name, None)
            if value is not None:
                payload[f"backend_{name}"] = np.asarray(value)
        payload["backend_fallback_reasons"] = np.asarray(
            getattr(backend, "fallback_reasons", ()), dtype=str
        )
        details = tuple(getattr(backend, "details", ()))
        payload["backend_detail_keys"] = np.asarray(
            [key for key, _ in details], dtype=str
        )
        payload["backend_detail_values"] = np.asarray(
            [value for _, value in details], dtype=str
        )
    backend_statistics = getattr(result, "backend_statistics", None)
    if backend_statistics is not None:
        for name, value in vars(backend_statistics).items():
            if name == "component_profile_seconds":
                component_items = sorted(value.items())
                payload["backend_component_profile_names"] = np.asarray(
                    [key for key, _ in component_items], dtype=str
                )
                payload["backend_component_profile_seconds"] = np.asarray(
                    [seconds for _, seconds in component_items], dtype=float
                )
            elif isinstance(value, (int, float, np.integer, np.floating)):
                payload[f"backend_statistic_{name}"] = np.asarray(value)
    if include_wavefunctions:
        payload["wavefunctions"] = result.wavefunctions
    np.savez_compressed(output, **payload)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Run the modular isolated CA-LDA/PBE solver from a supported PARSEC "
            "parsec.in and adjacent *_POTRE.DAT files."
        ),
    )
    parser.add_argument("input", type=Path, help="PARSEC input file, usually parsec.in")
    parser.add_argument(
        "--pp-dir",
        type=Path,
        default=None,
        help="additional pseudopotential directory (searched before the input directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="result .npz path; relative paths are resolved beside parsec.in",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="text log path; defaults to parsec.out beside parsec.in",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="do not write the default parsec_python_results.npz archive",
    )
    parser.add_argument(
        "--save-wavefunctions",
        action="store_true",
        help=(
            "include wavefunctions even when Output_All_States is false "
            "(they can make the archive large)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse, validate, and print the translated input without building the grid",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress console progress (the text log is still written)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="re-raise runtime exceptions with a traceback",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        translation = parse_parsec_input(
            arguments.input,
            pseudopotential_directory=arguments.pp_dir,
        )
    except (ParsecInputError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2

    is_periodic = translation.problem.periodic_cell is not None

    symmetry_mode = (
        arguments.symmetry
        if arguments.symmetry is not None
        else ("off" if translation.ignore_symmetry else "auto")
    )

    summary = summarize_translation(translation)
    if arguments.dry_run:
        try:
            load_pseudopotentials(
                translation.problem.pseudopotentials,
                xc_functional=translation.problem.scf.xc_functional,
            )
        except (OSError, ValueError) as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2
        print(summary)
        for warning in translation.warnings:
            print(f"WARNING: {warning}")
        print("Dry run successful; no grid or SCF calculation was started.")
        return 0

    input_directory = translation.source.parent
    log_path = _resolve_output(arguments.log, input_directory, "parsec.out")
    archive_path = _resolve_output(
        arguments.output, input_directory, "parsec_python_results.npz"
    )
    archive_path = _npz_path(archive_path)

    protected_paths = {
        translation.source,
        *(
            potential.path.resolve()
            for potential in translation.problem.pseudopotentials.values()
        ),
    }
    if log_path in protected_paths:
        print(
            f"Input error: refusing to overwrite input/pseudopotential with log {log_path}",
            file=sys.stderr,
        )
        return 2
    if not arguments.no_archive:
        if archive_path in protected_paths:
            print(
                "Input error: refusing to overwrite input/pseudopotential with "
                f"archive {archive_path}",
                file=sys.stderr,
            )
            return 2
        if archive_path == log_path:
            print(
                f"Input error: --log and --output resolve to the same path {log_path}",
                file=sys.stderr,
            )
            return 2

    start = time.perf_counter()
    run_log = _RunLog(log_path, quiet=arguments.quiet)
    try:
        with run_log as log:
            reporter = ParsecTextReporter(log.write, translation)
            reporter.header()
            for warning in translation.warnings:
                log.write(f"WARNING: {warning}")
            log.write()
            if is_periodic:
                system = prepare_periodic_single_point(translation.problem)
            else:
                system = prepare_single_point(translation.problem)
            reporter.setup(system)

            scf_start = time.perf_counter()
            result = run_scf(system, callback=reporter.iteration)
            scf_elapsed = time.perf_counter() - scf_start
            reporter.finish(result, scf_elapsed)
            total_elapsed = time.perf_counter() - start
            log.write(f" Total Python wall time [sec] : {total_elapsed:11.2f}")

            if not arguments.no_archive:
                saved = save_result_archive(
                    archive_path,
                    result,
                    include_wavefunctions=(
                        arguments.save_wavefunctions
                        or translation.output_all_states
                    ),
                )
                log.write(f"Result archive: {saved}")
            log.write(f"Text log: {log_path}")
            return 0 if result.converged else 3
    except KeyboardInterrupt:
        if not getattr(run_log, "failure_reported", False) and not arguments.quiet:
            print("Calculation interrupted by user.", file=sys.stderr)
        return 130
    except Exception as error:
        if arguments.debug:
            raise
        message = f"Calculation failed: {type(error).__name__}: {error}"
        if not getattr(run_log, "failure_reported", False):
            print(message, file=sys.stderr)
        return 1


__all__ = ["main", "save_result_archive"]
