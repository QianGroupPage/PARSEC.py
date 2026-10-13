"""Command-line runner for the additive accelerated single-point package."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import Sequence

from parsec_python.Input import (
    ParsecInputError,
    parse_parsec_input,
    summarize_translation,
)
from parsec_python.Output.parsec_output import domain_dry_run_lines
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.MolecularDynamics.run import (
    restart_structure as dynamics_restart_structure,
    run_recorded_molecular_dynamics,
)
from parsec_python.Relaxation.run import (
    relaxation_header_lines,
    restart_structure,
    run_recorded_relaxation,
    run_single_point_with_forces,
    symmetry_decision,
)
from parsec_python.cli import (
    refinement_progress,
    relaxation_exit_code,
    save_result_archive,
)

from .Output import AcceleratedTextReporter
from .backends.selection import resolve_backend
from .SCF.single_point import materialize_wavefunctions
from .driver import (
    hartree_boundary_check_seconds,
    ionic_backend,
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


def _symmetry_cache_directory(arguments: argparse.Namespace) -> Path | None:
    """Return the symmetry cache directory the command line named, if any.

    A first calculation of a structure is fastest without a cache, so none
    is read or written unless ``--symmetry-cache`` names its directory;
    ``--no-symmetry-cache`` spells that default out.  A relative directory
    is taken from the working directory, as it always was.
    """

    if arguments.symmetry_cache is None:
        return None
    return arguments.symmetry_cache.resolve()


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
        "--restart-from",
        type=Path,
        default=None,
        help=(
            "take the atom coordinates from this file instead of the input: a "
            "relax_restart.dat (its BFGS state continues a BFGS relaxation) or "
            "an .xyz file (last frame); Relax_Restart in the input reads "
            "relax_restart.dat beside it"
        ),
    )
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
        metavar="DIRECTORY",
        help=(
            "read and write exact-key symmetry maps, representation "
            "operators and Hartree boundary geometry in DIRECTORY; it pays "
            "only for repeated calculations of the same structure and grid, "
            "and no directory is read or written unless it is named here"
        ),
    )
    cache_group.add_argument(
        "--no-symmetry-cache",
        action="store_true",
        help="use no persistent symmetry cache (the default)",
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
        "--energy-refinement",
        choices=("b3lyp", "none"),
        default=None,
        help=(
            "after the SCF, evaluate the exchange-correlation energy of this "
            "hybrid functional once on the converged orbitals and report the "
            "refined total energy (PP-PBE(B3LYP) with a PBE input; ARES "
            "Lrefine); overrules Energy_Refinement of the input, none "
            "switches it off"
        ),
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
    if arguments.energy_refinement is not None:
        from parsec_python.Energy.refinement import apply_refinement_option

        try:
            translation = apply_refinement_option(
                translation, arguments.energy_refinement
            )
        except ValueError as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2

    try:
        translation, relax_restart, restart_notes = restart_structure(
            translation,
            translation.source.parent,
            None
            if arguments.restart_from is None
            else _resolve_output(arguments.restart_from, translation.source.parent, ""),
        )
        translation, dynamics_restart, dynamics_notes = dynamics_restart_structure(
            translation, translation.source.parent
        )
        restart_notes = [*restart_notes, *dynamics_notes]
        use_symmetry, symmetry_note = symmetry_decision(translation)
        # Whether the atoms stay in the point group of the first structure,
        # which Ignore_Symmetry does not tell.  The command line may overrule
        # that keyword, but not a run that leaves the group.
        keeps_group, group_note = symmetry_decision(
            replace(translation, ignore_symmetry=False)
        )
    except (OSError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    if arguments.symmetry == "on" and not keeps_group:
        print(
            "Input error: --symmetry on cannot be used here: "
            + (
                group_note
                or "manual movements and molecular dynamics use no symmetry"
            ),
            file=sys.stderr,
        )
        return 2
    symmetry_mode = (
        arguments.symmetry
        if arguments.symmetry is not None
        and (keeps_group or arguments.symmetry == "off")
        else ("auto" if use_symmetry else "off")
    )
    # Forces are symmetrized where the calculation itself uses the group.
    symmetrize_forces = use_symmetry and symmetry_mode != "off"

    summary = summarize_translation(translation)
    if arguments.dry_run:
        try:
            pseudopotentials = load_pseudopotentials(
                translation.problem.pseudopotentials,
                xc_functional=translation.problem.scf.xc_functional,
            )
            domain_lines = domain_dry_run_lines(translation, pseudopotentials)
            selection = resolve_backend(arguments.backend, translation.problem)
        except (OSError, ValueError, BackendUnavailableError, NotImplementedError) as error:
            print(f"Validation error: {error}", file=sys.stderr)
            return 2
        print(summary)
        for line in domain_lines:
            print(line)
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
    symmetry_cache_directory = _symmetry_cache_directory(arguments)
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

    # The default rule of the sphere ran in the parser, before this clock:
    # its seconds belong to the calculation.  None for an input with a radius.
    started = time.perf_counter() - translation.domain_rule_seconds
    run_log = _RunLog(log_path, quiet=arguments.quiet)
    reporter = None
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
            if symmetry_note is not None:
                log.write(f"WARNING: {symmetry_note}")
            for note in restart_notes:
                log.write(note)
            log.write()
            if translation.relaxation is not None:
                log.write("\n".join(relaxation_header_lines(translation)))

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
            wavefunctions_wanted = bool(
                not arguments.no_archive
                and (
                    arguments.save_wavefunctions
                    or translation.output_all_states
                )
            )
            # A relaxation exports the orbitals of its last structure only.
            moving = (
                translation.relaxation is not None
                or translation.molecular_dynamics is not None
            )
            system.materialize_final_wavefunctions = wavefunctions_wanted and not moving
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
            forces = None
            outcome = None
            dynamics = None
            check_seconds = 0.0
            if translation.relaxation is not None:
                outcome = run_recorded_relaxation(
                    translation,
                    system,
                    ionic_backend(),
                    reporter,
                    log.write,
                    directory=input_directory,
                    restart=relax_restart,
                    symmetrize=symmetrize_forces,
                )
                result = outcome.result
                forces = outcome.records[-1].forces
                if wavefunctions_wanted:
                    materialize_wavefunctions(outcome.system, result)
                scf_completed = time.perf_counter()
            elif translation.molecular_dynamics is not None:
                dynamics = run_recorded_molecular_dynamics(
                    translation,
                    system,
                    ionic_backend(),
                    reporter,
                    log.write,
                    directory=input_directory,
                    restart=dynamics_restart,
                )
                result, forces = dynamics.result, dynamics.forces
                if wavefunctions_wanted:
                    materialize_wavefunctions(dynamics.system, result)
                scf_completed = time.perf_counter()
            elif translation.compute_forces:
                result, forces = run_single_point_with_forces(
                    system, ionic_backend(), reporter, symmetrize=symmetrize_forces
                )
                scf_completed = time.perf_counter()
            else:
                result = run_scf(system, callback=reporter.iteration)
                scf_completed = time.perf_counter()
                # run_scf makes the optional check of the Hartree boundary
                # before it returns: it is finalization, not SCF.
                check_seconds = hartree_boundary_check_seconds(result)
                reporter.finish(result, scf_completed - scf_started - check_seconds)
            reporting_completed = time.perf_counter()
            refinement = None
            if getattr(translation, "energy_refinement", None) is not None:
                # One non-self-consistent evaluation on the converged
                # orbitals, after everything the SCF reports; of the last
                # structure where the atoms moved.
                from parsec_python.Energy.refinement import refinement_report_lines
                from parsec_python.Output import RYDBERG_TO_EV

                from .refinement import refine_energy

                announce, progress = refinement_progress(
                    log.write, translation.energy_refinement
                )
                announce()
                refinement = refine_energy(
                    system if outcome is None else outcome.system,
                    result,
                    translation.energy_refinement,
                    progress=progress,
                )
                log.write(
                    "\n".join(refinement_report_lines(refinement, RYDBERG_TO_EV))
                )
            log.write(
                f" Pre-SCF setup/reporting wall time [sec] : "
                f"{scf_started - started:11.6f}"
            )
            log.write(
                f" Post-SCF finalization/reporting [sec] : "
                f"{reporting_completed - scf_completed + check_seconds:11.6f}"
            )
            if refinement is not None:
                log.write(
                    f" Post-SCF energy refinement [sec] : "
                    f"{time.perf_counter() - reporting_completed:11.6f}"
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
                    # A plain single point keeps the call it always made.
                    **{
                        name: value
                        for name, value in (
                            ("forces", forces),
                            ("relaxation", outcome),
                            ("dynamics", dynamics),
                            ("refinement", refinement),
                        )
                        if value is not None
                    },
                )
                log.write(f"Result archive: {saved}")
            log.write(f"Text log: {log_path}")
            if dynamics is not None:
                return 3 if dynamics.scf_failed else 0
            return relaxation_exit_code(
                result,
                outcome,
                manual=(
                    translation.relaxation is not None
                    and translation.relaxation.method == "manual"
                ),
            )
    except KeyboardInterrupt:
        if not getattr(run_log, "failure_reported", False) and not arguments.quiet:
            print("Calculation interrupted by user.", file=sys.stderr)
        return 130
    except Exception as error:
        if arguments.debug:
            raise
        if not getattr(run_log, "failure_reported", False):
            print(
                f"Calculation failed: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        return 1
    finally:
        # The reporter may run a host thread beside the SCF: it ends here
        # where the calculation did not reach its report.
        if reporter is not None:
            reporter.close()


__all__ = ["main"]
