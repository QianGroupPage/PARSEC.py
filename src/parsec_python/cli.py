"""Command-line runner for PARSEC-style isolated single-point inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

from .driver import prepare_single_point, prepare_periodic_single_point, run_scf
from .MolecularDynamics.run import (
    restart_structure as dynamics_restart_structure,
    run_recorded_molecular_dynamics,
)
from .Relaxation.backend import reference_backend
from .Relaxation.run import (
    relaxation_header_lines,
    restart_structure,
    run_recorded_relaxation,
    run_single_point_with_forces,
    symmetry_decision,
)
from .Input import (
    ParsecInputError,
    parse_parsec_input,
    summarize_translation,
)
from .models import SinglePointResult
from .Output import ParsecTextReporter
from .Output.parsec_output import domain_dry_run_lines
from .V_ion import load_pseudopotentials


# Switches of the Hartree boundary values that only the accelerated driver
# reads.  This driver takes the values from the input keywords.
_ACCELERATED_BOUNDARY_SWITCHES = (
    "PARSEC_HARTREE_BOUNDARY",
    "PARSEC_HARTREE_BOUNDARY_TOLERANCE",
    "PARSEC_HARTREE_ATOMIC_TAIL",
    "PARSEC_HARTREE_LPOLE",
)


def ignored_boundary_switch_warnings() -> list[str]:
    """Name the boundary switches that are set and that this driver ignores.

    A run of the accelerated driver under one of them has another Hartree
    boundary than a reference run of the same input, which a comparison of
    the two would take for an error of either.
    """

    return [
        f"{name}={os.environ[name].strip()} is a switch of the accelerated "
        "driver and is ignored by the reference driver: the Hartree boundary "
        "of this run follows the input keywords Solver_Lpole, "
        "Hartree_Boundary_Tolerance and Hartree_Atomic_Tail"
        for name in _ACCELERATED_BOUNDARY_SWITCHES
        if os.environ.get(name, "").strip()
    ]


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
    forces=None,
    relaxation=None,
    dynamics=None,
    refinement=None,
) -> Path:
    """Save a portable NumPy result archive without wavefunctions by default.

    The members are stored uncompressed.  ``np.load`` reads the archive like
    a compressed one.  ``forces`` (a :class:`parsec_python.Forces.ForceResult`)
    adds the force tables of the structure under ``forces_*`` and
    ``relaxation`` (an :class:`parsec_python.Relaxation.IonicOutcome`) the
    positions, energies and forces of every structure under ``relaxation_*``;
    ``dynamics`` (a
    :class:`parsec_python.MolecularDynamics.run.MolecularDynamicsOutcome`)
    the structures, energies and temperatures of a molecular dynamics run
    under ``dynamics_*``; ``refinement`` (an
    :class:`parsec_python.Energy.refinement.EnergyRefinement`) the energies
    of a post-SCF refinement under ``refinement_*``.
    """
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
    # A spin-polarized result adds its two channels, up first.  The members
    # above then describe both together (SinglePointResult): the total
    # density, the averages of the potentials, and the states of the up
    # channel followed by those of the down channel.  Other archives keep
    # the members they had.
    channels = getattr(result, "spin", None)
    if channels is not None:
        payload["spin_polarized"] = np.asarray(True)
        payload["spin_electrons"] = np.asarray(
            [channel.electron_count for channel in channels]
        )
        payload["net_magnetic_moment"] = np.asarray(result.net_magnetic_moment)
        for name, member in (
            ("eigenvalues", "spin_eigenvalues_ry"),
            ("occupations", "spin_occupations"),
            ("representations", "spin_representations"),
            ("density", "spin_density_e_per_bohr3"),
            ("xc_potential", "spin_xc_potential_ry"),
            ("input_effective_potential", "spin_input_effective_potential_ry"),
            ("output_effective_potential", "spin_output_effective_potential_ry"),
            ("next_effective_potential", "spin_next_effective_potential_ry"),
        ):
            payload[member] = np.asarray(
                [getattr(channel, name) for channel in channels]
            )
        payload["scf_spin_history"] = np.asarray(
            [
                [
                    item.iteration,
                    *item.spin_weighted_residuals,
                    *item.spin_plain_residuals,
                    *item.spin_electrons,
                ]
                for item in result.history
            ],
            dtype=float,
        ).reshape(-1, 7)
        payload["scf_spin_history_columns"] = np.asarray(
            [
                "iteration",
                "weighted_sre_up_ry",
                "weighted_sre_down_ry",
                "plain_sre_up",
                "plain_sre_down",
                "electrons_up",
                "electrons_down",
            ]
        )
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
    if forces is not None:
        payload["forces_positions"] = np.asarray(forces.positions)
        payload["forces_total"] = np.asarray(forces.total)
        payload["forces_ion_ion"] = np.asarray(forces.ion_ion)
        payload["forces_local_ionic"] = np.asarray(forces.local.ionic)
        payload["forces_self_consistency"] = np.asarray(forces.local.self_consistency)
        payload["forces_core_correction"] = np.asarray(forces.local.core_correction)
        payload["forces_nonlocal"] = np.asarray(forces.nonlocal_pseudopotential)
        payload["forces_net"] = np.asarray(forces.net_force)
        payload["forces_torque"] = np.asarray(forces.torque)
    if relaxation is not None and relaxation.records:
        records = relaxation.records
        payload["relaxation_positions"] = np.asarray([item.positions for item in records])
        payload["relaxation_energies"] = np.asarray([item.energy for item in records])
        payload["relaxation_forces"] = np.asarray([item.forces.total for item in records])
        payload["relaxation_maximum_forces"] = np.asarray(
            [item.maximum_force for item in records]
        )
        payload["relaxation_scf_iterations"] = np.asarray(
            [item.scf_iterations for item in records]
        )
        payload["relaxation_final_positions"] = np.asarray(relaxation.positions)
        payload["relaxation_minimum_force_found"] = np.asarray(
            relaxation.minimum_force_found
        )
    if dynamics is not None and dynamics.records:
        records = dynamics.records
        payload["dynamics_positions"] = np.asarray([item.positions for item in records])
        payload["dynamics_energies"] = np.asarray([item.energy for item in records])
        steps = [item.step for item in records if item.step is not None]
        payload["dynamics_temperatures"] = np.asarray([item.temperature for item in steps])
        payload["dynamics_kinetic_energies"] = np.asarray(
            [item.kinetic_energy for item in steps]
        )
        payload["dynamics_conserved_energies"] = np.asarray(
            [item.conserved_energy for item in steps]
        )
        payload["dynamics_next_positions"] = np.asarray(dynamics.positions)
    if refinement is not None:
        payload["refinement_functional"] = np.asarray(refinement.functional)
        for name in (
            "exact_exchange",
            "slater_exchange",
            "b88_exchange",
            "vwn_correlation",
            "lyp_correlation",
            "exchange_correlation",
            "scf_exchange_correlation",
            "scf_total",
            "total",
            "all_electron_total",
        ):
            payload[f"refinement_{name}_ry"] = np.asarray(getattr(refinement, name))
        payload["refinement_pair_solves"] = np.asarray(refinement.exchange.pair_solves)
        payload["refinement_seconds"] = np.asarray(refinement.seconds)
    # Almost all bytes are full-grid coordinates and scalar fields.  Deflating
    # them runs zlib on one core (about 30 s for a 1.3 GB archive), so the
    # members are written as they are.
    np.savez(output, **payload)
    return output


def relaxation_exit_code(result, outcome, *, manual: bool = False) -> int:
    """Exit status of a run: 0 done, 3 an SCF loop did not converge, 4 not relaxed.

    A relaxation counts as done when the largest force fell below
    ``Force_Min``; a run of manual movements when every structure was
    computed.  ``outcome`` is ``None`` for a single point.
    """

    if outcome is None:
        return 0 if result.converged else 3
    if outcome.scf_failed:
        return 3
    if outcome.error:
        return 4
    return 0 if (manual or outcome.minimum_force_found) else 4


def refinement_progress(write, functional: str, interval: float = 60.0):
    """Return ``(announce, progress)`` for a post-SCF energy refinement.

    ``announce()`` writes that the refinement started; ``progress`` is the
    callback of :func:`parsec_python.Energy.refinement.refine_energy` and
    writes one line per ``interval`` seconds of pair-density Poisson solves,
    so that a long refinement is seen to advance and a short one adds none.
    """

    name = functional.upper()
    last = [0.0]

    def announce() -> None:
        write(
            f" Energy refinement: evaluating the {name} exchange-correlation "
            "energy of the converged orbitals"
        )

    def progress(done: int, total: int, seconds: float) -> None:
        if done < total and seconds - last[0] >= interval:
            last[0] = seconds
            write(
                f" Energy refinement: {done} of {total} pair-density Poisson "
                f"solves done after {seconds:.1f} s"
            )

    return announce, progress


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
    if arguments.energy_refinement is not None:
        from .Energy.refinement import apply_refinement_option

        try:
            translation = apply_refinement_option(
                translation, arguments.energy_refinement
            )
        except ValueError as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2

    is_periodic = translation.problem.periodic_cell is not None

    summary = summarize_translation(translation)
    if arguments.dry_run:
        try:
            pseudopotentials = load_pseudopotentials(
                translation.problem.pseudopotentials,
                xc_functional=translation.problem.scf.xc_functional,
            )
            domain_lines = domain_dry_run_lines(translation, pseudopotentials)
        except (OSError, ValueError) as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2
        print(summary)
        for line in domain_lines:
            print(line)
        for warning in translation.warnings:
            print(f"WARNING: {warning}")
        print("Dry run successful; no grid or SCF calculation was started.")
        return 0

    input_directory = translation.source.parent
    try:
        translation, relax_restart, restart_notes = restart_structure(
            translation,
            input_directory,
            None
            if arguments.restart_from is None
            else _resolve_output(arguments.restart_from, input_directory, ""),
        )
        translation, dynamics_restart, dynamics_notes = dynamics_restart_structure(
            translation, input_directory
        )
        restart_notes = [*restart_notes, *dynamics_notes]
        use_symmetry, symmetry_note = symmetry_decision(translation)
    except (OSError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
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

    # The default rule of the sphere ran in the parser, before this clock:
    # its seconds belong to the calculation.  None for an input with a radius.
    start = time.perf_counter() - translation.domain_rule_seconds
    run_log = _RunLog(log_path, quiet=arguments.quiet)
    try:
        with run_log as log:
            reporter = ParsecTextReporter(log.write, translation)
            reporter.header()
            # A periodic cell has no Hartree boundary values for a switch of
            # the accelerated driver to change.
            switch_warnings = () if is_periodic else ignored_boundary_switch_warnings()
            for warning in (*translation.warnings, *switch_warnings):
                log.write(f"WARNING: {warning}")
            if symmetry_note is not None:
                log.write(f"WARNING: {symmetry_note}")
            for note in restart_notes:
                log.write(note)
            log.write()
            if translation.relaxation is not None:
                log.write("\n".join(relaxation_header_lines(translation)))
            if is_periodic:
                system = prepare_periodic_single_point(translation.problem)
            else:
                system = prepare_single_point(translation.problem)
            reporter.setup(system)

            forces = None
            outcome = None
            dynamics = None
            if translation.relaxation is not None:
                outcome = run_recorded_relaxation(
                    translation,
                    system,
                    reference_backend(),
                    reporter,
                    log.write,
                    directory=input_directory,
                    restart=relax_restart,
                    symmetrize=use_symmetry,
                )
                result = outcome.result
                forces = outcome.records[-1].forces
            elif translation.molecular_dynamics is not None:
                dynamics = run_recorded_molecular_dynamics(
                    translation,
                    system,
                    reference_backend(),
                    reporter,
                    log.write,
                    directory=input_directory,
                    restart=dynamics_restart,
                )
                result, forces = dynamics.result, dynamics.forces
            elif translation.compute_forces:
                result, forces = run_single_point_with_forces(
                    system, reference_backend(), reporter, symmetrize=use_symmetry
                )
            else:
                scf_start = time.perf_counter()
                result = run_scf(system, callback=reporter.iteration)
                scf_elapsed = time.perf_counter() - scf_start
                reporter.finish(result, scf_elapsed)
            refinement = None
            if getattr(translation, "energy_refinement", None) is not None:
                # One non-self-consistent evaluation on the converged
                # orbitals, after everything the SCF reports; of the last
                # structure where the atoms moved.
                from .Energy.refinement import refine_energy, refinement_report_lines
                from .Output import RYDBERG_TO_EV

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
        message = f"Calculation failed: {type(error).__name__}: {error}"
        if not getattr(run_log, "failure_reported", False):
            print(message, file=sys.stderr)
        return 1


__all__ = ["main", "refinement_progress", "save_result_archive"]
