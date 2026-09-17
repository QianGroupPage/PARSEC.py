"""Command-line runner for the additive accelerated single-point package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Sequence

from parsec_python.Input import (
    ParsecInputError,
    parse_parsec_input,
    summarize_translation,
)
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.cli import save_result_archive

from .Output import AcceleratedTextReporter
from .backends.selection import resolve_backend
from .driver import (
    prepare_single_point,
    profile_hamiltonian_components,
    run_scf,
)
from .models import BackendUnavailableError


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

    def __exit__(self, _type, value, _traceback) -> None:
        if value is not None:
            message = (
                "Calculation interrupted by user."
                if isinstance(value, KeyboardInterrupt)
                else f"Calculation failed: {type(value).__name__}: {value}"
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
    return path if path.suffix.lower() == ".npz" else path.with_suffix(path.suffix + ".npz")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Run the modular isolated real-space DFT solver. By default, auto uses "
            "an accuracy-preserving hybrid when available: native C++ for "
            "finite-difference construction and Hartree, with CuPy for the "
            "Hamiltonian and eigensolver."
        ),
    )
    parser.add_argument("input", type=Path, help="PARSEC input file, usually parsec.in")
    parser.add_argument(
        "--backend",
        choices=("auto", "scipy", "native", "cupy"),
        default="auto",
        help=(
            "execution policy; auto may combine native setup/Hartree with "
            "CuPy Hamiltonian/eigensolver work and may fall back, while "
            "explicit scipy, native, and cupy are clean comparison modes; "
            "every selected component and fallback is recorded"
        ),
    )
    parser.add_argument(
        "--symmetry",
        choices=("auto", "on", "off"),
        default=None,
        help=(
            "symmetry policy: auto (default) detects exact supported "
            "operations and falls back safely, on requires a nontrivial "
            "usable symmetry, and off forces full-grid calculations; "
            "Ignore_Symmetry=true selects off unless this option is given"
        ),
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--symmetry-cache",
        type=Path,
        default=None,
        help=(
            "directory for exact-key cached representation operators; "
            "defaults to .parsec_cache/symmetry beside parsec.in"
        ),
    )
    cache_group.add_argument(
        "--no-symmetry-cache",
        action="store_true",
        help="disable the persistent representation-operator cache",
    )
    parser.add_argument(
        "--pp-dir",
        type=Path,
        default=None,
        help="additional pseudopotential directory searched before the input directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="result .npz path; defaults beside parsec.in",
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
        help="do not write parsec_python_results.npz",
    )
    parser.add_argument(
        "--save-wavefunctions",
        action="store_true",
        help="include wavefunctions in the result archive",
    )
    parser.add_argument(
        "--profile-operator",
        action="store_true",
        help=(
            "run an opt-in synchronized microprofile of finite-difference, "
            "local ionic, and nonlocal ionic Hamiltonian actions"
        ),
    )
    parser.add_argument(
        "--profile-repeats",
        type=int,
        default=3,
        help="component-profile repeats averaged when --profile-operator is used",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse/validate input and resolve backend without constructing the grid",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress console progress while retaining the text log",
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help=(
            "submit to the local warmed worker, auto-starting it if needed; "
            "handled by the folder-local launcher before ordinary parsing"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="re-raise runtime exceptions with a traceback",
    )
    parser.epilog = (
        "Resident controls: main.py --resident-start, --resident-status, "
        "or --resident-stop. Each request creates fresh DFT/SCF state while "
        "reusing only process and accelerator runtime state."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.profile_repeats < 1:
        print("Input error: --profile-repeats must be positive", file=sys.stderr)
        return 2
    try:
        translation = parse_parsec_input(
            arguments.input,
            pseudopotential_directory=arguments.pp_dir,
        )
    except (ParsecInputError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2

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
            selection = resolve_backend(arguments.backend, translation.problem)
        except (OSError, ValueError, BackendUnavailableError, NotImplementedError) as error:
            print(f"Validation error: {error}", file=sys.stderr)
            return 2
        print(summary)
        for warning in translation.warnings:
            print(f"WARNING: {warning}")
        print(
            f"Backend: requested={selection.requested}, "
            f"selected={selection.selected}"
        )
        print(
            "Finite-difference builder: "
            f"{selection.finite_difference_builder}"
        )
        print(f"Hartree backend: {selection.hartree_backend}")
        print(
            f"Symmetry: mode={symmetry_mode}; exact detection is deferred "
            "until grid construction"
        )
        for reason in selection.fallback_reasons:
            print(f"BACKEND FALLBACK: {reason}")
        print("Dry run successful; no grid or SCF calculation was started.")
        return 0

    input_directory = translation.source.parent
    symmetry_cache_directory = (
        None
        if arguments.no_symmetry_cache
        else (
            arguments.symmetry_cache.resolve()
            if arguments.symmetry_cache is not None
            else input_directory / ".parsec_cache" / "symmetry"
        )
    )
    log_path = _resolve_output(
        arguments.log, input_directory, "parsec.out"
    )
    archive_path = _npz_path(
        _resolve_output(
            arguments.output,
            input_directory,
            "parsec_python_results.npz",
        )
    )
    protected_paths = {
        translation.source,
        *(item.path.resolve() for item in translation.problem.pseudopotentials.values()),
    }
    if log_path in protected_paths:
        print(f"Input error: refusing to overwrite input with log {log_path}", file=sys.stderr)
        return 2
    if not arguments.no_archive:
        if archive_path in protected_paths:
            print(
                f"Input error: refusing to overwrite input with archive {archive_path}",
                file=sys.stderr,
            )
            return 2
        if archive_path == log_path:
            print("Input error: --log and --output resolve to the same path", file=sys.stderr)
            return 2

    started = time.perf_counter()
    run_log = _RunLog(log_path, quiet=arguments.quiet)
    try:
        with run_log as log:
            reporter = AcceleratedTextReporter(
                log.write,
                translation,
                symmetry_mode=symmetry_mode,
            )
            reporter.header()
            for warning in translation.warnings:
                log.write(f"WARNING: {warning}")
            log.write()

            system = prepare_single_point(
                translation.problem,
                backend=arguments.backend,
                symmetry=symmetry_mode,
                symmetry_cache_directory=symmetry_cache_directory,
            )
            # Wavefunctions are much larger than all scalar observables and
            # are absent from the default archive.  Keep symmetry-sector
            # orbitals resident/lazy unless the user explicitly requests
            # their public full-grid array.  This does not affect SCF density,
            # energies, eigenvalues, or any normal text output.
            system.materialize_final_wavefunctions = bool(
                not arguments.no_archive
                and (
                    arguments.save_wavefunctions
                    or translation.output_all_states
                )
            )
            reporter.setup(system)
            if arguments.profile_operator:
                component_times = profile_hamiltonian_components(
                    system,
                    repeats=arguments.profile_repeats,
                )
                log.write(" Hamiltonian component microprofile [sec/apply]:")
                for name, value in component_times.items():
                    log.write(f" {name:30s} = {value:12.6f}")
                log.write()

            scf_started = time.perf_counter()
            result = run_scf(system, callback=reporter.iteration)
            scf_completed = time.perf_counter()
            reporter.finish(result, scf_completed - scf_started)
            reporting_completed = time.perf_counter()
            log.write(
                f" Pre-SCF setup/reporting wall time [sec] : "
                f"{scf_started - started:11.6f}"
            )
            log.write(
                f" Post-SCF finalization/reporting [sec] : "
                f"{reporting_completed - scf_completed:11.6f}"
            )
            log.write(
                f" Total accelerated Python wall time [sec] : "
                f"{time.perf_counter() - started:11.2f}"
            )
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
        if not run_log.failure_reported and not arguments.quiet:
            print("Calculation interrupted by user.", file=sys.stderr)
        return 130
    except Exception as error:
        if arguments.debug:
            raise
        if not run_log.failure_reported:
            print(
                f"Calculation failed: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        return 1


__all__ = ["main"]
