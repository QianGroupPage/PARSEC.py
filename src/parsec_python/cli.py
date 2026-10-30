"""Command-line runner for PARSEC-style isolated single-point inputs."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
import os
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
from .MolecularDynamics.run import (
    restart_structure as dynamics_restart_structure,
    run_recorded_molecular_dynamics,
)
from .Relaxation.backend import periodic_backend, reference_backend, spinor_backend
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
from .models import (
    SelfConsistentSOCResult,
    SinglePointResult,
    SpinPolarizedSinglePointResult,
)
from .Grid import kpoint_sampling
from .Output import ParsecTextReporter
from .Output.bands import save_band_structure_archive, write_bands_files, write_spin_file
from .SCF.bands import (
    band_mode,
    band_potentials,
    load_scf_for_bands,
    run_band_structure,
)
from .SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
from .SCF.kpoint_solver import parallel_backend_report, parallel_backend_uses_device
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


def _ionic_payload(forces, relaxation, dynamics) -> dict[str, np.ndarray]:
    """Archive entries of the forces of a structure, a relaxation and a molecular dynamics run."""

    payload: dict[str, np.ndarray] = {}
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
    return payload


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
    payload.update(_ionic_payload(forces, relaxation, dynamics))
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
    payload.update(_k_point_payload(result))
    payload.update(_scf_potential_payload(result))
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


class _SpinorReporter:
    """Presents the spin-orbit and non-collinear report methods under the names the ionic loops call.

    The loops of a relaxation and of a molecular dynamics run report each SCF iteration with ``iteration`` and each
    finished structure, with its forces, with ``finish``; a spinor result has its own tables for both.
    """

    def __init__(self, reporter) -> None:
        self._reporter = reporter

    def iteration(self, step) -> None:
        self._reporter.iteration_self_consistent_soc(step)

    def finish(self, result, elapsed_seconds: float, *, forces=None) -> None:
        self._reporter.finish_self_consistent_soc(result, elapsed_seconds, forces=forces)

    def __getattr__(self, name: str):
        return getattr(self._reporter, name)


def save_self_consistent_soc_result_archive(
    path: str | Path,
    result: SelfConsistentSOCResult,
    *,
    include_wavefunctions: bool = False,
    forces=None,
    relaxation=None,
    dynamics=None,
) -> Path:
    """Save a portable NumPy archive for a self-consistent-SOC result.

    ``forces``, ``relaxation`` and ``dynamics`` add the force tables, structures and energies as
    :func:`save_result_archive` does.
    """
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
    payload.update(_ionic_payload(forces, relaxation, dynamics))
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
    spin_file = write_spin_file(directory, band_result)  # spinor bands only
    if spin_file is not None:
        files.append(spin_file)
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


def _run_extended_drivers(
    system,
    translation,
    reporter,
    log,
    arguments,
    *,
    start: float,
    archive_path: Path,
    log_path: Path,
    input_directory: Path,
    k_sampling,
    is_periodic: bool,
    has_soc_species: bool,
) -> int:
    """SCF and bands of the drivers that main's ``run_scf`` does not cover.

    Periodic k-point sampling (scalar, collinear spin, spin-orbit, spin-orbit
    with collinear spin, non-collinear), the isolated and periodic
    spin-orbit/non-collinear drivers, the perturbative spin-orbit correction on
    a spin-polarized ground state, and periodic spin-polarized cells.  Returns
    the exit status.
    """

    spin_polarized = translation.problem.scf.spin_polarized
    self_consistent_spin_orbit = translation.problem.scf.self_consistent_spin_orbit
    noncollinear = translation.problem.scf.noncollinear
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

    if spin_polarized:  # periodic or with spin-orbit (isolated and plain: main's run_spin_scf)
        if (
            k_sampling is None
            and is_periodic
            and not has_soc_species
            and parallel_backend_uses_device()
        ):
            # A Gamma-only periodic spin run is the 1x1x1 k-point run.  Through the k-point driver
            # it runs on the device; the Gamma driver has no device path.
            k_sampling = (np.zeros((1, 3)), np.ones(1))
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


    # Scalar (spin-unpolarized, no spin-orbit) k-point sampling.
    k_points, k_weights = k_sampling
    result = run_scf_kpoints(system, k_points, k_weights, callback=reporter.iteration)
    reporter.finish(result, time.perf_counter() - scf_start)
    if not arguments.no_archive:
        saved = save_result_archive(
            archive_path,
            result,
            include_wavefunctions=(
                arguments.save_wavefunctions or translation.output_all_states
            ),
        )
        log.write(f"Result archive: {saved}")
    # The SCF archive is written first so a long band calculation cannot lose it.
    _run_band_structure(system, translation, result, reporter, input_directory)
    total_elapsed = time.perf_counter() - start
    log.write(f" Total Python wall time [sec] : {total_elapsed:11.2f}")
    log.write(f"Text log: {log_path}")
    return 0 if result.converged else 3


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
        "--mlff",
        nargs="?",
        const="",
        default=None,
        metavar="MODEL",
        help=(
            "relax the input structure with a machine-learned force field first "
            "(Initial_Structure: mlff); MODEL as MLFF_Model, for example uma-s-1p2, "
            "mace-mp-medium-mpa-0 or a file.model; without MODEL the input's or uma-s-1p2"
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
    if arguments.kpoint_workers is not None:
        os.environ["PARSEC_KPOINT_WORKERS"] = str(arguments.kpoint_workers)
    try:
        translation = parse_parsec_input(
            arguments.input,
            pseudopotential_directory=arguments.pp_dir,
        )
    except (ParsecInputError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    if arguments.mlff is not None:
        from .MLFF import apply_mlff_option

        translation = apply_mlff_option(translation, arguments.mlff)
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
    from .MLFF import MLFFError, relax_initial_structure

    try:
        problem_of_input = translation.problem
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
        # Initial_Structure: mlff, unless a restart file gave the structure.
        translation, mlff_notes = relax_initial_structure(
            translation,
            input_directory,
            restarted=translation.problem is not problem_of_input,
            progress=None if arguments.quiet else print,
        )
        restart_notes = [*restart_notes, *mlff_notes]
        use_symmetry, symmetry_note = symmetry_decision(translation)
    except MLFFError as error:
        print(f"MLFF error: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
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

    # Periodic spin polarization or k-points, spin-orbit coupling and non-collinear
    # magnetism run the drivers of parsec_python.SCF (k-point solver, spin-orbit,
    # non-collinear).  An isolated spin-polarized input without spin-orbit coupling
    # runs run_scf -> run_spin_scf, like every other isolated input, so it keeps
    # forces, relaxation, molecular dynamics and energy refinement; self-consistent
    # spin-orbit and non-collinear inputs keep all but the energy refinement (below).
    ionic_requested = bool(
        translation.relaxation is not None
        or translation.molecular_dynamics is not None
        or translation.compute_forces
    )
    # A periodic cell that asks for forces, a relaxation or molecular dynamics runs the ionic loops of the isolated path
    # with the periodic backend: scalar, at the Gamma point or with Monkhorst-Pack points, or collinear spin-polarized
    # at the Gamma point (no spin-orbit coupling, no non-collinear magnetism).
    gamma_only = k_sampling is None or (len(k_sampling[0]) == 1 and not np.any(k_sampling[0]))
    periodic_ionic = bool(
        is_periodic
        and ionic_requested
        and not (self_consistent_spin_orbit or noncollinear)
        and (not spin_polarized or (gamma_only and not has_soc_species))
    )
    # Self-consistent spin-orbit coupling and non-collinear magnetism, isolated or periodic (Gamma point or k-points),
    # run their own SCF drivers and, asked for forces, a relaxation or molecular dynamics, the ionic loops of the spinor
    # backend.
    spinor_ionic = bool(ionic_requested and (self_consistent_spin_orbit or noncollinear))
    uses_extended_drivers = not periodic_ionic and not spinor_ionic and bool(
        self_consistent_spin_orbit
        or noncollinear
        or k_sampling is not None
        or (spin_polarized and (is_periodic or has_soc_species))
    )
    if uses_extended_drivers and (
        translation.relaxation is not None
        or translation.molecular_dynamics is not None
        or translation.compute_forces
        or getattr(translation, "energy_refinement", None) is not None
    ):
        print(
            "Input error: Minimization, Molecular_Dynamics, Skip_Force: false and "
            "Energy_Refinement are not implemented together with the perturbative spin-orbit "
            "correction (use SO_from_scratch for forces with spin-orbit coupling) or a "
            "spin-polarized periodic calculation with k-points (a periodic cell without "
            "spin-orbit coupling has forces, relaxation and molecular dynamics: "
            "spin-unpolarized at the Gamma point or with k-points, spin-polarized at the "
            "Gamma point)",
            file=sys.stderr,
        )
        return 2
    if spinor_ionic and getattr(translation, "energy_refinement", None) is not None:
        print(
            "Input error: Energy_Refinement is not implemented together with spin-orbit coupling or "
            "non-collinear magnetism",
            file=sys.stderr,
        )
        return 2
    if spinor_ionic and not is_periodic and (noncollinear or spin_polarized):
        # a moment breaks the point group of the atoms: averaging the forces over it would be wrong
        if use_symmetry:
            symmetry_note = (
                "The forces of a spin-polarized or non-collinear spin-orbit calculation are not symmetrized "
                "over the point group of the atoms: the magnetization need not respect it."
            )
        use_symmetry = False

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
            if (
                noncollinear
                or (is_periodic and (self_consistent_spin_orbit or k_sampling is not None))
                or (
                    is_periodic
                    and spin_polarized
                    and not has_soc_species
                    and not periodic_ionic
                    and parallel_backend_uses_device()
                )
            ):
                # these drivers hand their eigenproblems to the k-point solver: say where it runs
                for line in parallel_backend_report():
                    log.write(line)
                log.write()
            if translation.relaxation is not None:
                log.write("\n".join(relaxation_header_lines(translation)))
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

            if uses_extended_drivers:
                return _run_extended_drivers(
                    system,
                    translation,
                    reporter,
                    log,
                    arguments,
                    start=start,
                    archive_path=archive_path,
                    log_path=log_path,
                    input_directory=input_directory,
                    k_sampling=k_sampling,
                    is_periodic=is_periodic,
                    has_soc_species=has_soc_species,
                )

            forces = None
            outcome = None
            dynamics = None
            ionic_reporter = reporter
            if spinor_ionic:
                ionic_reporter = _SpinorReporter(reporter)
                backend_factory = lambda extrapolation="none": spinor_backend(  # noqa: E731
                    *(k_sampling if k_sampling is not None else (None, None))
                )
            elif is_periodic:
                # the spin-polarized loop is the Gamma-point one, whatever Monkhorst-Pack grid of one point the input names
                backend_k_sampling = None if spin_polarized else k_sampling
                backend_factory = lambda extrapolation="none": periodic_backend(  # noqa: E731
                    *(backend_k_sampling if backend_k_sampling is not None else (None, None)), extrapolation
                )
            else:
                backend_factory = reference_backend
            if translation.relaxation is not None:
                outcome = run_recorded_relaxation(
                    translation,
                    system,
                    backend_factory(translation.potential_extrapolation),
                    ionic_reporter,
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
                    backend_factory(translation.potential_extrapolation),
                    ionic_reporter,
                    log.write,
                    directory=input_directory,
                    restart=dynamics_restart,
                )
                result, forces = dynamics.result, dynamics.forces
            elif translation.compute_forces:
                result, forces = run_single_point_with_forces(
                    system, backend_factory(), ionic_reporter, symmetrize=use_symmetry
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
            if not arguments.no_archive and spinor_ionic:
                saved = save_self_consistent_soc_result_archive(
                    archive_path,
                    result,
                    include_wavefunctions=(
                        arguments.save_wavefunctions or translation.output_all_states
                    ),
                    forces=forces,
                    relaxation=outcome,
                    dynamics=dynamics,
                )
                log.write(f"Result archive: {saved}")
            elif not arguments.no_archive:
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
            # The SCF archive is written first so a long band calculation cannot lose it.
            _run_band_structure(system, translation, result, reporter, input_directory)
            total_elapsed = time.perf_counter() - start
            log.write(f" Total Python wall time [sec] : {total_elapsed:11.2f}")
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
