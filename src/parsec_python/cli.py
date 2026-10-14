"""Command-line runner for PARSEC-style isolated single-point inputs."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

from .driver import prepare_single_point, prepare_periodic_single_point, run_scf
from .Eigensolvers.perturbative_soc import (
    PerturbativeSpinOrbitResult,
    build_spin_orbit_projectors,
    perturbative_spin_orbit_correction,
)
from .Input import (
    ParsecInputError,
    parse_parsec_input,
    summarize_translation,
)
from .models import (
    SelfConsistentSOCResult,
    SinglePointResult,
    SpinPolarizedSinglePointResult,
)
from .Grid import kpoint_sampling
from .Output import ParsecTextReporter
from .Output.bands import save_band_structure_archive, write_bands_files
from .SCF.bands import (
    band_mode,
    band_potentials,
    load_scf_for_bands,
    run_band_structure,
)
from .SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
from .SCF.noncollinear import (
    run_self_consistent_noncollinear,
    run_self_consistent_noncollinear_kpoints,
)
from .SCF.kpoints_soc import (
    run_self_consistent_soc_kpoints,
    run_self_consistent_soc_kpoints_spin_polarized,
)
from .SCF.self_consistent_soc import run_self_consistent_soc
from .SCF.self_consistent_soc_spin_polarized import (
    run_self_consistent_soc_spin_polarized,
)
from .SCF.spin_polarized import run_scf_spin_polarized
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


def _k_point_payload(result) -> dict[str, np.ndarray]:
    """Archive entries describing the k-point sampling of a pooled result.

    ``eigenvalues_ry``/``occupations`` (and the spin-resolved or
    per-state-moment arrays) are k-major: reshape to
    ``(n_kpoints, n_states_per_kpoint)`` to index them by k-point.
    """
    k_points = getattr(result, "k_points", None)
    if k_points is None:
        return {}
    k_points = np.asarray(k_points, dtype=float)
    weights = getattr(result, "k_weights", None)
    if weights is None:
        weights = np.full(k_points.shape[0], 1.0 / k_points.shape[0])
    eigenvalues = getattr(result, "eigenvalues", None)
    if eigenvalues is None:
        eigenvalues = result.eigenvalues_up
    return {
        "k_points_cartesian_inv_bohr": k_points,
        "k_weights": np.asarray(weights, dtype=float),
        "n_kpoints": np.asarray(k_points.shape[0]),
        "n_states_per_kpoint": np.asarray(
            np.asarray(eigenvalues).size // k_points.shape[0]
        ),
    }


def _scf_potential_payload(result) -> dict[str, np.ndarray]:
    """Converged local potentials of a periodic run (inputs of a band calculation)."""
    potentials = getattr(result, "scf_potentials", None) or {}
    return {f"scf_potential_{name}": np.asarray(value) for name, value in potentials.items()}


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
    payload.update(_k_point_payload(result))
    payload.update(_scf_potential_payload(result))
    np.savez_compressed(output, **payload)
    return output


def save_spin_polarized_result_archive(
    path: str | Path,
    result: SpinPolarizedSinglePointResult,
    soc_result: PerturbativeSpinOrbitResult | None = None,
    *,
    include_wavefunctions: bool = False,
) -> Path:
    """Save a portable NumPy archive for a spin-polarized (LSDA) result.

    A leaner counterpart to :func:`save_result_archive`: the spin-polarized
    driver does not yet carry per-iteration history or preparation timings
    (see ``SCF.spin_polarized``'s module docstring), so this saves only the
    converged final state -- both channels' eigenpairs, energies, and the
    magnetic moment -- plus the perturbative spin-orbit correction's
    SOC-split eigenvalues/spinors when ``soc_result`` is supplied.
    """
    output = _npz_path(Path(path).expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    energy = asdict(result.energies)
    payload: dict[str, np.ndarray] = {
        "atom_symbols": np.asarray([atom.symbol for atom in result.atoms]),
        "atom_coordinates_bohr": np.asarray(
            [atom.position for atom in result.atoms], dtype=float
        ),
        "density_up_e_per_bohr3": result.density_up,
        "density_down_e_per_bohr3": result.density_down,
        "eigenvalues_up_ry": result.eigenvalues_up,
        "eigenvalues_down_ry": result.eigenvalues_down,
        "occupations_up": result.occupations_up,
        "occupations_down": result.occupations_down,
        "fermi_level_ry": np.asarray(result.fermi_level),
        "electron_count": np.asarray(result.electron_count),
        "magnetic_moment": np.asarray(result.magnetic_moment),
        "converged": np.asarray(result.converged),
        "iterations": np.asarray(result.iterations),
    }
    for name, value in energy.items():
        payload[f"energy_{name}_ry"] = np.asarray(value)
    if include_wavefunctions:
        payload["wavefunctions_up"] = result.wavefunctions_up
        payload["wavefunctions_down"] = result.wavefunctions_down
    if soc_result is not None:
        payload["soc_eigenvalues_ry"] = soc_result.eigenvalues
        payload["soc_magnetic_moment"] = soc_result.magnetic_moment
        if include_wavefunctions:
            payload["soc_spinor_coefficients_up"] = soc_result.spinor_coefficients_up
            payload["soc_spinor_coefficients_down"] = (
                soc_result.spinor_coefficients_down
            )
    payload.update(_k_point_payload(result))
    payload.update(_scf_potential_payload(result))
    np.savez_compressed(output, **payload)
    return output


def save_self_consistent_soc_result_archive(
    path: str | Path,
    result: SelfConsistentSOCResult,
    *,
    include_wavefunctions: bool = False,
) -> Path:
    """Save a portable NumPy archive for a self-consistent-SOC result."""
    output = _npz_path(Path(path).expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    energy = asdict(result.energies)
    payload: dict[str, np.ndarray] = {
        "atom_symbols": np.asarray([atom.symbol for atom in result.atoms]),
        "atom_coordinates_bohr": np.asarray(
            [atom.position for atom in result.atoms], dtype=float
        ),
        "density_e_per_bohr3": result.density,
        "eigenvalues_ry": result.eigenvalues,
        "occupations": result.occupations,
        "magnetic_moment_per_state": result.magnetic_moment_per_state,
        "fermi_level_ry": np.asarray(result.fermi_level),
        "electron_count": np.asarray(result.electron_count),
        "converged": np.asarray(result.converged),
        "iterations": np.asarray(result.iterations),
    }
    for name, value in energy.items():
        payload[f"energy_{name}_ry"] = np.asarray(value)
    if result.magnetic_moment is not None:
        payload["magnetic_moment"] = np.asarray(result.magnetic_moment)
    if result.magnetic_moment_vector is not None:
        payload["magnetic_moment_vector"] = np.asarray(result.magnetic_moment_vector)
    if result.magnetization is not None:
        payload["magnetization_mu_b_per_bohr3"] = result.magnetization
    if include_wavefunctions:
        payload["spinors"] = result.spinors
    payload.update(_k_point_payload(result))
    payload.update(_scf_potential_payload(result))
    np.savez_compressed(output, **payload)
    return output


def _band_progress(reporter):
    """One log line per finished k-point of the (slow) band calculation."""
    started = time.perf_counter()
    done = [0]

    def report(index: int, count: int, _eigenvalues, residual: float, attempts: int) -> None:
        done[0] += 1
        note = f", {attempts} CHEBDAV attempts" if attempts > 1 else ""
        reporter.write(
            f" band k-point {index + 1:4d} done ({done[0]:4d}/{count:<4d}) after "
            f"{time.perf_counter() - started:8.1f} s, max residual {residual:9.2e} Ry{note}"
        )

    return report


def _write_band_outputs(band_result, reporter, directory: Path, start: float) -> None:
    files = list(write_bands_files(directory, band_result))
    files.append(
        save_band_structure_archive(directory / "parsec_python_bands.npz", band_result)
    )
    reporter.band_structure(band_result, files, time.perf_counter() - start)


def _run_band_structure(system, translation, result, reporter, directory: Path) -> None:
    """Fortran ``calc_bands``: bands along the ``bandstruc`` path after the SCF."""
    bands = translation.problem.bands
    if bands is None:
        return
    mode = band_mode(translation.problem)
    if not result.converged:
        reporter.write(
            " WARNING: the SCF did not converge; the bands use its last potential."
        )
    eigenvalues = getattr(result, "eigenvalues", None)
    if eigenvalues is None:
        eigenvalues = result.eigenvalues_up
    k_points = getattr(result, "k_points", None)
    k_count = 1 if k_points is None else len(k_points)
    start = time.perf_counter()
    band_result = run_band_structure(
        system,
        bands,
        mode=mode,
        potentials=band_potentials(result, mode),
        fermi_level=result.fermi_level,
        number_of_states=np.asarray(eigenvalues).size // k_count,
        callback=_band_progress(reporter),
    )
    _write_band_outputs(band_result, reporter, directory, start)


def _run_bands_from_archive(system, translation, archive, reporter, directory: Path) -> None:
    """Bands from the potentials stored in a previous run's archive (no SCF)."""
    mode = band_mode(translation.problem)
    potentials, fermi_level, number_of_states = load_scf_for_bands(
        archive, translation.problem, system, mode
    )
    reporter.write(f" Band structure from the converged potential in {archive} (no SCF).")
    start = time.perf_counter()
    band_result = run_band_structure(
        system,
        translation.problem.bands,
        mode=mode,
        potentials=potentials,
        fermi_level=fermi_level,
        number_of_states=number_of_states,
        callback=_band_progress(reporter),
    )
    _write_band_outputs(band_result, reporter, directory, start)


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
        "--kpoint-workers",
        default=None,
        metavar="N|auto",
        help=(
            "worker threads for the k-points of a periodic SCF and for the band path "
            "(default: PARSEC_KPOINT_WORKERS, else auto = half the CPUs, serial for small "
            "problems)"
        ),
    )
    parser.add_argument(
        "--kpoint-backend",
        choices=("cpu", "cupy", "auto", "numpy-device"),
        default=None,
        help=(
            "where the k-point Hamiltonians are applied: cpu (default), cupy (one worker "
            "thread per CUDA device, PARSEC_CUPY_DEVICES selects them), auto (cupy if it "
            "works) or numpy-device (the GPU code on the host, for tests)"
        ),
    )
    parser.add_argument(
        "--bands-from",
        type=Path,
        default=None,
        metavar="ARCHIVE.npz",
        help=(
            "skip the SCF and compute the bandstruc bands from the converged potential "
            "stored in a previous run's result archive (written next to parsec.in as "
            "parsec_bands.out unless --log is given)"
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
    if arguments.kpoint_workers is not None:
        os.environ["PARSEC_KPOINT_WORKERS"] = str(arguments.kpoint_workers)
    if arguments.kpoint_backend is not None:
        os.environ["PARSEC_KPOINT_BACKEND"] = arguments.kpoint_backend
    try:
        translation = parse_parsec_input(
            arguments.input,
            pseudopotential_directory=arguments.pp_dir,
        )
    except (ParsecInputError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2

    is_periodic = translation.problem.periodic_cell is not None

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
    if arguments.bands_from is not None:
        if translation.problem.bands is None:
            print("Input error: --bands-from needs a bandstruc block in the input", file=sys.stderr)
            return 2
        if not arguments.bands_from.exists():
            print(f"Input error: {arguments.bands_from} does not exist", file=sys.stderr)
            return 2
    log_path = _resolve_output(
        arguments.log,
        input_directory,
        "parsec.out" if arguments.bands_from is None else "parsec_bands.out",
    )
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

    spin_polarized = translation.problem.scf.spin_polarized
    self_consistent_spin_orbit = translation.problem.scf.self_consistent_spin_orbit
    k_sampling = kpoint_sampling(translation.problem) if is_periodic else None
    has_soc_species = any(
        specification.spin_orbit
        for specification in translation.problem.pseudopotentials.values()
    )
    noncollinear = translation.problem.scf.noncollinear
    if self_consistent_spin_orbit and not has_soc_species and not noncollinear:
        print(
            "Input error: SO_from_scratch/SCF_SO=true requires at least one "
            "species with SO_PSP=true",
            file=sys.stderr,
        )
        return 2

    if translation.problem.bands is not None:
        try:
            band_mode(translation.problem)
        except ValueError as error:
            print(f"Input error: {error}", file=sys.stderr)
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

            if arguments.bands_from is not None:
                _run_bands_from_archive(
                    system, translation, arguments.bands_from, reporter, input_directory
                )
                log.write(f"Text log: {log_path}")
                return 0

            scf_start = time.perf_counter()
            if self_consistent_spin_orbit or noncollinear:
                if is_periodic:
                    if k_sampling is not None:
                        k_points, k_weights = k_sampling
                    else:
                        k_points = np.zeros((1, 3))
                        k_weights = np.ones(1)
                    if noncollinear:
                        result = run_self_consistent_noncollinear_kpoints(
                            system,
                            k_points,
                            k_weights,
                            callback=reporter.iteration_self_consistent_soc,
                        )
                    else:
                        periodic_soc_driver = (
                            run_self_consistent_soc_kpoints_spin_polarized
                            if spin_polarized
                            else run_self_consistent_soc_kpoints
                        )
                        result = periodic_soc_driver(
                            system,
                            k_points,
                            k_weights,
                            callback=reporter.iteration_self_consistent_soc,
                        )
                else:
                    soc_projectors = tuple(
                        build_spin_orbit_projectors(
                            system.grid,
                            system.atoms,
                            system.pseudopotentials,
                            translation.problem.pseudopotentials,
                        )
                    )
                    if noncollinear:
                        result = run_self_consistent_noncollinear(
                            system,
                            soc_projectors,
                            callback=reporter.iteration_self_consistent_soc,
                        )
                    elif spin_polarized:
                        result = run_self_consistent_soc_spin_polarized(
                            system,
                            soc_projectors,
                            callback=reporter.iteration_self_consistent_soc,
                        )
                    else:
                        result = run_self_consistent_soc(
                            system,
                            soc_projectors,
                            callback=reporter.iteration_self_consistent_soc,
                        )
                scf_elapsed = time.perf_counter() - scf_start
                reporter.finish_self_consistent_soc(result, scf_elapsed)
                if not arguments.no_archive:
                    saved = save_self_consistent_soc_result_archive(
                        archive_path,
                        result,
                        include_wavefunctions=(
                            arguments.save_wavefunctions
                            or translation.output_all_states
                        ),
                    )
                    log.write(f"Result archive: {saved}")
                # The SCF archive is written first: the band calculation can take a long time
                # and a crash there must not lose the converged SCF.
                _run_band_structure(
                    system, translation, result, reporter, input_directory
                )
                total_elapsed = time.perf_counter() - start
                log.write(f" Total Python wall time [sec] : {total_elapsed:11.2f}")
                log.write(f"Text log: {log_path}")
                return 0 if result.converged else 3

            if spin_polarized:
                if k_sampling is not None:
                    k_points, k_weights = k_sampling
                    result = run_scf_kpoints_spin_polarized(
                        system,
                        k_points,
                        k_weights,
                        callback=reporter.iteration_spin_polarized,
                    )
                else:
                    result = run_scf_spin_polarized(
                        system, callback=reporter.iteration_spin_polarized
                    )
                scf_elapsed = time.perf_counter() - scf_start
                reporter.finish_spin_polarized(result, scf_elapsed)

                soc_result = None
                if has_soc_species:
                    soc_result = perturbative_spin_orbit_correction(
                        system.grid,
                        result,
                        system.atoms,
                        system.pseudopotentials,
                        translation.problem.pseudopotentials,
                    )
                    reporter.spin_orbit_correction(soc_result)
                if not arguments.no_archive:
                    saved = save_spin_polarized_result_archive(
                        archive_path,
                        result,
                        soc_result,
                        include_wavefunctions=(
                            arguments.save_wavefunctions
                            or translation.output_all_states
                        ),
                    )
                    log.write(f"Result archive: {saved}")
                # The SCF archive is written first: the band calculation can take a long time
                # and a crash there must not lose the converged SCF.
                _run_band_structure(
                    system, translation, result, reporter, input_directory
                )
                total_elapsed = time.perf_counter() - start
                log.write(f" Total Python wall time [sec] : {total_elapsed:11.2f}")
                log.write(f"Text log: {log_path}")
                return 0 if result.converged else 3

            if k_sampling is not None:
                k_points, k_weights = k_sampling
                result = run_scf_kpoints(
                    system, k_points, k_weights, callback=reporter.iteration
                )
            else:
                result = run_scf(system, callback=reporter.iteration)
            scf_elapsed = time.perf_counter() - scf_start
            reporter.finish(result, scf_elapsed)
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
            # The SCF archive is written first so a long band calculation cannot lose it.
            _run_band_structure(system, translation, result, reporter, input_directory)
            total_elapsed = time.perf_counter() - start
            log.write(f" Total Python wall time [sec] : {total_elapsed:11.2f}")
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
