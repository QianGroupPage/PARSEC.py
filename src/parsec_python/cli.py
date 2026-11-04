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
from .Relaxation.backend import (
    fortran_periodic_backend,
    fortran_spinor_backend,
    periodic_backend,
    reference_backend,
    spinor_backend,
    wire_slab_backend,
)
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
from .Output.output_files import ReporterWithFiles, RunOutputFiles, output_header_lines
from .Output.post_processing_output import post_processing_header_lines
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


def _isolated_dipole(result):
    """The :class:`parsec_python.Energy.DipoleMoment` of an isolated result, or ``None``.

    ``None`` for a periodic result and for one whose density is not on the
    full grid of the result.
    """

    from .Energy.dipole import cluster_dipole_moment
    from .models import GridSettings

    grid = getattr(result, "grid", None)
    if (
        not isinstance(getattr(grid, "settings", None), GridSettings)
        or getattr(result, "k_points", None) is not None
    ):
        return None
    density = getattr(result, "density", None)
    coordinates = getattr(grid, "coordinates", None)
    if density is None or coordinates is None:
        return None
    density = np.asarray(density)
    atoms = getattr(result, "atoms", None)
    pseudopotentials = getattr(result, "pseudopotentials", None)
    if (
        density.ndim != 1
        or density.shape[0] != np.asarray(coordinates).shape[0]
        or not atoms
        or not pseudopotentials
        or any(
            not hasattr(pseudopotentials.get(atom.symbol), "ionic_charge")
            for atom in atoms
        )
    ):
        return None
    return cluster_dipole_moment(density, grid, atoms, pseudopotentials)


def _dipole_payload(dipole=None, polarizability=None) -> dict[str, np.ndarray]:
    """Archive members of the dipole and of a polarizability run."""

    payload: dict[str, np.ndarray] = {}
    if dipole is not None:
        payload["dipole_charge_e"] = np.asarray(dipole.charge)
        payload["dipole_e_bohr"] = np.asarray(dipole.dipole)
        payload["dipole_debye"] = np.asarray(dipole.dipole_debye)
    if polarizability is not None:
        payload["polarizability_field"] = np.asarray(polarizability.field)
        payload["polarizability_directions"] = np.asarray(
            [run.direction for run in polarizability.runs]
        )
        payload["polarizability_energies_ry"] = np.asarray(
            [run.total_energy for run in polarizability.runs]
        )
        payload["polarizability_dipoles_e_bohr"] = polarizability.dipoles
        payload["polarizability_converged"] = np.asarray(
            [run.converged for run in polarizability.runs]
        )
        payload["polarizability_from_energy_ang3"] = polarizability.from_energy
        payload["polarizability_from_dipole_ang3"] = polarizability.from_dipole
    return payload


def _ionic_payload(forces=None, relaxation=None, dynamics=None) -> dict[str, np.ndarray]:
    """The ``forces_*``, ``relaxation_*`` and ``dynamics_*`` members of an archive."""

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
        spin_orbit = getattr(forces, "spin_orbit", None)
        if spin_orbit is not None:
            payload["forces_spin_orbit"] = np.asarray(spin_orbit)
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
    polarizability=None,
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
    of a post-SCF refinement under ``refinement_*``; ``polarizability`` (a
    :class:`parsec_python.SCF.polarizability.PolarizabilityResult`) the seven
    field runs and the polarizabilities under ``polarizability_*``.  The
    charge and dipole of the density of an isolated system (``dipole.f90``)
    are stored under ``dipole_*``.  The results of the spin-polarized
    k-point and spin-orbit drivers (with forces, a relaxation or dynamics)
    go to their own archives with the same ionic members.
    """
    if hasattr(result, "density_up") or hasattr(result, "spinors"):
        ionic = {"forces": forces, "relaxation": relaxation, "dynamics": dynamics}
        if hasattr(result, "density_up"):
            return save_spin_polarized_result_archive(
                path, result, include_wavefunctions=include_wavefunctions, **ionic
            )
        return save_self_consistent_soc_result_archive(
            path, result, include_wavefunctions=include_wavefunctions, **ionic
        )
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
    payload.update(_dipole_payload(_isolated_dipole(result), polarizability))
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
    dipole=None,
    forces=None,
    relaxation=None,
    dynamics=None,
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
    payload.update(_dipole_payload(dipole))
    payload.update(_k_point_payload(result))
    payload.update(_scf_potential_payload(result))
    payload.update(_ionic_payload(forces, relaxation, dynamics))
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
    dipole=None,
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
    payload.update(_dipole_payload(dipole))
    payload.update(_k_point_payload(result))
    payload.update(_scf_potential_payload(result))
    payload.update(_ionic_payload(forces, relaxation, dynamics))
    np.savez_compressed(output, **payload)
    return output


def planned_state_count(problem, electron_count: float) -> int:
    """The states of each channel the SCF loop will return (``States_Num``)."""

    requested = problem.scf.number_of_states
    if requested is None:
        requested = int(np.ceil(0.5 * electron_count)) + 6
    return int(requested)


def output_files_for(translation, directory: Path, k_sampling=None) -> RunOutputFiles:
    """The data files of a run (``parsec.dat``, ``eigen.dat``, cube files)."""

    return RunOutputFiles(
        translation.output_files,
        directory,
        write=lambda _line: None,
        kpoint_count=0 if k_sampling is None else len(k_sampling[0]),
    )


def write_qpoints_file(problem, directory: Path, write=None) -> Path | None:
    """``qpoints.dat`` of Fortran's k-point path (``irrbz.f90:186-209``).

    Written beside the input, as PARSEC writes it in its run directory, for
    an input with ``Monkhorst_Pack_Shift``; ``None`` otherwise.
    """

    from .Grid.kpoints import kpoint_sampling_details
    from .Output.kpoint_output import qpoints_dat_text

    if getattr(problem, "periodic_cell", None) is None:
        return None
    sampling = kpoint_sampling_details(problem)
    if sampling is None:
        return None
    path = Path(directory) / "qpoints.dat"
    path.write_text(qpoints_dat_text(sampling), encoding="ascii", newline="\n")
    if write is not None:
        write(f" qpoints.dat: {path} ({sampling.count} irreducible k-points)")
    return path


def output_file_conflict(files: RunOutputFiles, taken: set[Path]) -> Path | None:
    """The first data file of ``files`` that is one of the ``taken`` paths."""

    resolved = {Path(path).resolve() for path in taken}
    for path in files.paths():
        if path.resolve() in resolved:
            return path
    return None


def start_output_files(files: RunOutputFiles, translation, system, log, *, k_sampling=None) -> None:
    """Settle the wavefunctions of ``parsec.dat`` and open ``eigen.dat``."""

    files.write = log.write
    problem = translation.problem
    spinors = bool(problem.scf.self_consistent_spin_orbit or problem.scf.noncollinear)
    # Spinors (wfnsave.F90 with mxwd = 2) take two complex components per point.
    spin_count = 2 if problem.scf.spin_polarized or spinors else 1
    files.plan_wavefunctions(
        system.grid.size,
        planned_state_count(problem, system.electron_count),
        spin_count,
        complex_values=k_sampling is not None or spinors,
    )
    # Spinor runs and wires and slabs write
    # parsec.dat as every other run.
    files.start()


def load_restart_start(translation, system, log, directory: Path, *, k_sampling=None):
    """``Restart_Run``: the warm start of the first SCF loop, ``None`` without it."""

    if not translation.restart_run:
        return None
    if getattr(translation, "restart_from_wfndat", False):
        # Restart_From_Wfndat: restart.F90 reads wfn.dat, wavefunctions included.
        from .SCF.restart import load_wfndat_restart, wfndat_report_lines

        restart, warm_start = load_wfndat_restart(directory / "wfn.dat", system)
        log.write("\n".join(wfndat_report_lines(restart)))
        if getattr(translation.problem.eigensolver, "fixed_state_count", False):
            from .SCF.restart import check_fixed_state_count

            check_fixed_state_count(
                min(vectors.shape[1] for vectors in restart.wavefunctions),
                planned_state_count(translation.problem, system.electron_count),
            )
        return warm_start
    from .SCF.restart import load_restart, restart_report_lines

    scf = translation.problem.scf
    fields, warm_start = load_restart(
        directory / "parsec.dat",
        system,
        complex_wavefunctions=k_sampling is not None,
        # restart_run.F90 with stype(2) = 2: the two spin components.
        spinors=bool(scf.self_consistent_spin_orbit and not scf.noncollinear),
    )
    log.write("\n".join(restart_report_lines(fields)))
    if getattr(translation.problem.eigensolver, "fixed_state_count", False):
        from .SCF.restart import check_fixed_state_count

        check_fixed_state_count(
            fields.stored_states,
            planned_state_count(translation.problem, system.electron_count),
        )
    return warm_start


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


def save_nscf_archive(path: Path, result) -> Path:
    """The levels of a non-self-consistent run (:class:`parsec_python.SCF.nscf.NSCFResult`)."""

    output = _npz_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        nscf_channels=np.asarray(result.channels),
        nscf_k_points_bohr_inverse=result.k_points,
        nscf_k_weights=result.weights,
        nscf_eigenvalues_ry=result.eigenvalues,
        nscf_occupations=result.occupations,
        nscf_residual_norms_ry=result.residual_norms,
        fermi_level_ry=np.asarray(result.fermi_level),
        fermi_temperature_kelvin=np.asarray(result.temperature_kelvin),
    )
    return output


def _nscf_progress(reporter):
    """One log line per finished NSCF k-point (``eigval``'s "Working on k-point")."""

    started = time.perf_counter()
    done = [0]

    def report(index: int, count: int, _eigenvalues, residual: float, attempts: int) -> None:
        done[0] += 1
        note = f", {attempts} CHEBDAV attempts" if attempts > 1 else ""
        reporter.write(
            f"  ...Working on k-point {index + 1:12d} done ({done[0]:4d}/{count:<4d}) after "
            f"{time.perf_counter() - started:8.1f} s, max residual {residual:9.2e} Ry{note}"
        )

    return report


def _run_nscf(system, translation, reporter, log, arguments, *, start, archive_path, log_path, input_directory, k_sampling) -> int:
    """``nscf_kpoints``: ``calc_nscf`` on the potential of ``parsec.dat`` (no SCF).

    The potential of every spin channel and the Fermi level of the run that
    wrote the file come from ``parsec.dat`` (``Restart_Run``); see
    :mod:`parsec_python.SCF.nscf`.  No ``parsec.dat`` or ``eigen.dat`` is
    written, as in PARSEC; the levels go to ``parsec.out`` and the archive.
    """

    from .Eigensolvers import matvec_budget
    from .Output.parsec_dat import read_parsec_dat
    from .Output.post_processing_output import nscf_report_lines
    from .SCF.nscf import restart_fermi_level, run_nscf
    from .SCF.restart import RestartError, restart_fields, restart_report_lines

    problem = translation.problem
    settings = translation.nscf
    spin_count = 2 if problem.scf.spin_polarized else 1
    # A spin-orbit run restarts from the two spinor components of a
    # spin-orbit parsec.dat (restart_run.F90 with stype(2) = 2).
    spinors = bool(problem.scf.self_consistent_spin_orbit and not problem.scf.noncollinear)
    source = input_directory / "parsec.dat"
    if not source.is_file():
        raise RestartError(f"Restart_Run: {source} not found", 371)
    data = read_parsec_dat(source, wavefunctions=False)
    fields = restart_fields(
        data,
        system.grid,
        spin_count=spin_count,
        complex_wavefunctions=k_sampling is not None,
        spinors=spinors,
    )
    log.write("\n".join(restart_report_lines(fields)))
    temperature = float(problem.scf.fermi_temperature_kelvin)
    gw = getattr(translation, "gw", None)
    if gw is not None:
        # usrinputfile.F90 l.2535: Output_GW reads nscf_fermi_level into efermi.
        fermi_level = gw.fermi_level
        log.write(f" NSCF: the Fermi level is nscf_fermi_level: {fermi_level:.6f} Ry")
    else:
        fermi_level = restart_fermi_level(data, system.electron_count, temperature)
        log.write(
            " NSCF: the Fermi level is that of the SCF which wrote parsec.dat, from its "
            f"stored levels: {fermi_level:.6f} Ry (PARSEC without Output_GW uses 0 Ry)"
        )
    solve_start = time.perf_counter()
    result = run_nscf(
        system,
        settings.k_points,
        settings.weights,
        potentials=fields.potentials,
        number_of_states=settings.number_of_states,
        fermi_level=fermi_level,
        callback=_nscf_progress(reporter),
        **({"keep_wavefunctions": True} if gw is not None else {}),
        **({"spinors": True} if spinors else {}),
    )
    elapsed = time.perf_counter() - solve_start
    log.write(
        "\n".join(
            nscf_report_lines(
                result,
                problem.periodic_cell.lattice_vectors,
                matvec_budget=matvec_budget(
                    problem.eigensolver.maximum_matvec,
                    system.grid.size,
                    settings.number_of_states,
                ),
                diagonalization_seconds=elapsed,
            )
        )
    )
    if not arguments.no_archive:
        # The archive of the SCF that wrote parsec.dat stays: by default the
        # NSCF levels have a file of their own.
        target = (
            archive_path
            if arguments.output is not None
            else input_directory / "parsec_python_nscf.npz"
        )
        saved = save_nscf_archive(target, result)
        log.write(f"Result archive: {saved}")
    if gw is not None:
        _run_gw(
            system,
            translation,
            log.write,
            input_directory,
            k_points=result.k_points,
            weights=result.weights,
            energies=result.eigenvalues,
            occupations=result.occupations,
            orbitals=result.wavefunctions,
            densities=fields.densities,
        )
    log.write(" NSCF run: parsec.dat and eigen.dat are not written, as in PARSEC")
    total_elapsed = time.perf_counter() - start
    log.write(f" Total Python wall time [sec] : {total_elapsed:11.2f}")
    log.write(f"Text log: {log_path}")
    return 0


def _dos_eigenvalues(result) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Levels and occupations of every spin channel, ``(n_k, n_states)`` each."""

    k_points = getattr(result, "k_points", None)
    count = 1 if k_points is None else len(k_points)
    if hasattr(result, "eigenvalues_up"):
        values = [result.eigenvalues_up, result.eigenvalues_down]
        occupations = [result.occupations_up, result.occupations_down]
    elif getattr(result, "spin", None) is not None:
        values = [channel.eigenvalues for channel in result.spin]
        occupations = [channel.occupations for channel in result.spin]
    else:
        values = [result.eigenvalues]
        occupations = [result.occupations]
    shaped = [np.asarray(array, dtype=float).reshape(count, -1) for array in values]
    occupied = [np.asarray(array, dtype=float).reshape(count, -1) for array in occupations]
    return shaped, occupied


def _run_dos(system, translation, result, reporter, directory: Path) -> None:
    """``Create_DOS`` (``dos.F90``): the file ``DOS`` after the SCF."""

    settings = getattr(translation, "dos", None)
    if settings is None:
        return
    from .Output.dos import (
        dos_file_lines,
        dos_report_lines,
        fortran_cell_volume,
        monkhorst_pack_k_index,
        occupation_histogram,
        tetrahedron_dos,
        ylm_projection_weights,
    )

    started = time.perf_counter()
    eigenvalues, occupations = _dos_eigenvalues(result)
    problem = translation.problem
    if problem.periodic_cell is None:
        dos = occupation_histogram(
            [values[0] for values in eigenvalues],
            [values[0] for values in occupations],
            result.fermi_level,
            points=settings.points,
        )
    else:
        dimensions = settings.monkhorst_pack_grid
        projections = None
        if settings.ylm_l >= 0:
            wavefunctions = getattr(result, "k_wavefunctions", None)
            if wavefunctions is None:
                raise ValueError("the Ylm-projected DOS needs the orbitals of the k-point SCF")
            if not hasattr(result, "eigenvalues_up"):
                wavefunctions = (wavefunctions,)
            cutoffs = dict(settings.ylm_cutoffs)
            projections = ylm_projection_weights(
                system.grid,
                np.asarray([atom.position for atom in system.atoms], dtype=float),
                [cutoffs.get(atom.symbol, 1.0) for atom in system.atoms],
                wavefunctions,
                np.asarray(result.k_points, dtype=float),
                l_max=settings.ylm_l,
                fortran_projection=settings.fortran_projection,
            )
        # kptgrid: the identity on this port's unreduced grid; with
        # Monkhorst_Pack_Shift the SCF ran on Fortran's irreducible wedge, and
        # every full-grid point maps to its representative (irrbz.f90).
        from .Grid.kpoints import kpoint_sampling_details

        sampling = kpoint_sampling_details(problem)
        if sampling is None:
            k_index = monkhorst_pack_k_index(dimensions)
        else:
            if tuple(int(value) for value in sampling.dimensions) != tuple(dimensions):
                raise ValueError("the DOS grid is not the Monkhorst-Pack grid of the SCF")
            k_index = np.asarray(sampling.grid_map, dtype=np.int64)
        dos = tetrahedron_dos(
            eigenvalues,
            result.fermi_level,
            dimensions=dimensions,
            k_index=k_index,
            lattice_vectors=problem.periodic_cell.lattice_vectors,
            points=settings.points,
            projections=projections,
            cell_volume=fortran_cell_volume(system.grid),
        )
    path = directory / "DOS"
    lines = dos_file_lines(dos, atom_labels=[atom.symbol for atom in system.atoms])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reporter.write("\n".join(dos_report_lines(dos, time.perf_counter() - started)))
    reporter.write(f" DOS: {path}")


def _run_gw(
    system,
    translation,
    write,
    directory: Path,
    *,
    k_points,
    weights,
    energies,
    occupations,
    orbitals,
    densities,
) -> None:
    """``Output_GW`` (``parsec.F90`` l.2095-2147, ``gw_write.F90``): WFN, RHO, VXC.

    ``energies``/``occupations`` are ``(ns, nk, nbands)`` (Ry, 0..1),
    ``orbitals[s][k]`` the ``(n_grid, >= nbands)`` Bloch factors and
    ``densities`` the valence density of each spin channel (the total for one
    channel).  V_xc is evaluated anew from the density with the core charge
    and without PARSEC's potential shift, as ``parsec.F90`` does before
    ``gw_write``.  The files go beside the input.
    """

    settings = getattr(translation, "gw", None)
    if settings is None:
        return
    from .Output.berkeleygw import (
        band_coefficients,
        berkeleygw_date_time,
        berkeleygw_header,
        berkeleygw_report_lines,
        berkeleygw_setup,
        field_coefficients,
        write_berkeleygw_file,
    )
    from .Output.dos import fortran_cell_volume

    started = time.perf_counter()
    densities = [np.asarray(density, dtype=np.float64) for density in densities]
    spin_count = len(densities)
    if spin_count == 1:
        # zero_average=False: no potshift here also with Zero_Average_Vxc.
        vxc = [np.asarray(system.evaluate_xc(densities[0], zero_average=False).potential)]
    else:
        spin_xc = system.evaluate_spin_xc(densities[0], densities[1], zero_average=False)
        vxc = [np.asarray(spin_xc.potential_up), np.asarray(spin_xc.potential_down)]
    grid = system.grid
    group = settings.space_group
    setup = berkeleygw_setup(grid, group, k_points, cell_volume=fortran_cell_volume(grid))
    if setup.corner is not None:
        write(
            " NOTE: Output_GW: the FFT box corner of this grid is not a symmetric point of "
            "the space group; the coefficients are referred to the cell origin (PARSEC "
            "writes them in the frame of the corner, which contradicts the fractional "
            "translations of its header)"
        )
    rho_g = field_coefficients(setup, grid, densities, scale=setup.cell_volume)
    vxc_g = field_coefficients(setup, grid, vxc)
    energies = np.asarray(energies, dtype=np.float64)
    occupations = np.asarray(occupations, dtype=np.float64)
    band_count = energies.shape[2]
    atoms = translation.problem.atoms
    stamp = berkeleygw_date_time()
    common = dict(
        spin_count=spin_count,
        kgrid=settings.kgrid,
        kshift=settings.kshift,
        k_weights=weights,
        energies=energies,
        occupations=occupations,
        stamp=stamp,
    )
    paths = []
    for kind, data in (("VXC", vxc_g), ("RHO", rho_g)):
        header = berkeleygw_header(kind, setup, group, atoms, **common)
        paths.append(
            write_berkeleygw_file(directory / kind, header, setup.potential.vectors, data=data)
        )

    def k_blocks():
        for k_index, sphere in enumerate(setup.k_spheres):
            channels = [np.asarray(orbitals[s][k_index]) for s in range(spin_count)]
            yield sphere.vectors, (
                band_coefficients(setup, grid, k_index, [block[:, band] for block in channels])
                for band in range(band_count)
            )

    header = berkeleygw_header("WFN", setup, group, atoms, **common)
    paths.append(
        write_berkeleygw_file(directory / "WFN", header, setup.potential.vectors, k_blocks=k_blocks())
    )
    zero = setup.potential.zero_index
    write(
        "\n".join(
            berkeleygw_report_lines(
                setup, weights, complex(np.sum(rho_g[:, zero])), complex(vxc_g[0, zero])
            )
        )
    )
    write(
        f" BerkeleyGW files: {', '.join(str(path) for path in paths)} "
        f"({time.perf_counter() - started:.2f} s)"
    )


def _run_gw_after_scf(system, translation, result, write, directory: Path) -> None:
    """``Output_GW`` after a k-point SCF: the levels and orbitals of its last iteration.

    PARSEC writes the eigenpairs of the last ``eigval`` call and the density
    of the last ``newrho`` (``elec_st``), as the k-point drivers keep them
    (``keep_wavefunctions``).
    """

    k_points = np.asarray(result.k_points, dtype=np.float64)
    weights = np.asarray(result.k_weights, dtype=np.float64)
    count = k_points.shape[0]
    if hasattr(result, "eigenvalues_up"):
        values = [result.eigenvalues_up, result.eigenvalues_down]
        occupied = [result.occupations_up, result.occupations_down]
        orbitals = result.k_wavefunctions
        densities = [result.density_up, result.density_down]
    else:
        values = [result.eigenvalues]
        occupied = [result.occupations]
        orbitals = (result.k_wavefunctions,)
        densities = [result.density]
    energies = np.asarray([np.asarray(v, dtype=np.float64).reshape(count, -1) for v in values])
    occupations = np.asarray([np.asarray(v, dtype=np.float64).reshape(count, -1) for v in occupied])
    _run_gw(
        system,
        translation,
        write,
        directory,
        k_points=k_points,
        weights=weights,
        energies=energies,
        occupations=occupations,
        orbitals=orbitals,
        densities=densities,
    )


def _gamma_spin_on_kpoint_driver(translation, is_periodic: bool, has_soc_species: bool) -> bool:
    """Whether a Gamma-only periodic spin single point runs as the 1x1x1 k-point run.

    With a device k-point backend the k-point driver puts its eigenproblems on
    the GPU, while the Gamma driver (``run_scf_spin_polarized``) has no device
    path.  LDA+U goes along (the device workers apply its term and form its
    occupation matrices, ``SCF/kpoint_hubbard.py``).  A wire or slab takes the
    k-point driver too (its 1x1x1 run equals the Gamma loop to the SCF
    convergence, with LDA+U too); ``PARSEC_WIRE_SLAB_GAMMA_SPIN=gamma`` and an
    MPI run keep it on the Gamma driver.
    """
    if not is_periodic or has_soc_species:
        return False
    problem = translation.problem
    periodicity = getattr(problem.grid, "periodicity", 3)
    if periodicity in (1, 2) and (
        os.environ.get("PARSEC_WIRE_SLAB_GAMMA_SPIN", "kpoint").strip().lower() == "gamma"
        # an MPI run (--mpi) keeps a wire or slab on the Gamma driver, as before
        or os.environ.get("PARSEC_MPI", "").strip().lower() in ("1", "true", "on", "yes")
    ):
        return False
    return parallel_backend_uses_device()


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
    files: RunOutputFiles | None = None,
    restart=None,
    iteration_state_callback=None,
) -> int:
    """SCF and bands of the drivers that main's ``run_scf`` does not cover.

    Periodic k-point sampling (scalar, collinear spin, spin-orbit, spin-orbit
    with collinear spin, non-collinear), the isolated and periodic
    spin-orbit/non-collinear drivers, the perturbative spin-orbit correction on
    a spin-polarized ground state, and periodic spin-polarized cells.  Returns
    the exit status.  ``files`` writes the data files of the run;
    ``restart`` is the :class:`parsec_python.SCF.SCFWarmStart` of
    ``Restart_Run``, which only the scalar k-point loop takes.
    ``iteration_state_callback`` (``Save_Intermediate_*``) is handed to every
    SCF driver, which calls it after each iteration.
    """
    # Save_Intermediate_*: parsec.dat / eigen.dat after every iteration.
    state = (
        {}
        if iteration_state_callback is None
        else {"iteration_state_callback": iteration_state_callback}
    )

    def write_files(result) -> None:
        if files is not None:
            files.finish(result, system.grid, translation.problem)

    spin_polarized = translation.problem.scf.spin_polarized
    self_consistent_spin_orbit = translation.problem.scf.self_consistent_spin_orbit
    noncollinear = translation.problem.scf.noncollinear
    # The Ylm-projected DOS reads the orbitals of every k-point.
    dos = getattr(translation, "dos", None)
    keep = {"keep_wavefunctions": True} if dos is not None and dos.ylm_l >= 0 else {}
    # Output_GW writes the orbitals of every k-point.
    if getattr(translation, "gw", None) is not None:
        keep = {"keep_wavefunctions": True}
    scf_start = time.perf_counter()
    if self_consistent_spin_orbit or noncollinear:
        spinor_forces = None
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
                    **state,
                )
            else:
                periodic_soc_driver = (
                    run_self_consistent_soc_kpoints_spin_polarized
                    if spin_polarized
                    else run_self_consistent_soc_kpoints
                )
                restart_keywords = {}
                if restart is not None:
                    # Restart_Run from the parsec.dat of a spin-orbit run.
                    from .SCF.restart import spin_orbit_start

                    restart_keywords["initial_potential"] = spin_orbit_start(
                        restart, spin_polarized
                    )
                result = periodic_soc_driver(
                    system,
                    k_points,
                    k_weights,
                    callback=reporter.iteration_self_consistent_soc,
                    **restart_keywords,
                    **state,
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
            spinor_driver = (
                run_self_consistent_noncollinear
                if noncollinear
                else (
                    run_self_consistent_soc_spin_polarized
                    if spin_polarized
                    else run_self_consistent_soc
                )
            )
            if getattr(translation, "polarizability_field", None) is not None:
                # polar.f90 around the spinor SCF: seven loops, each reported as
                # a spin-orbit SCF; every loop after the first starts from the
                # last mixed potential of the one before (and, for the
                # spin-orbit loops, its spinors), as PARSEC keeps vold and the
                # wavefunctions; with Skip_Force: false the forces of the last
                # loop.
                from .Relaxation.backend import SpinorWarmStart

                def spinor_start(warm_start) -> dict:
                    if warm_start is None:
                        return {}
                    keywords = {"initial_potential": warm_start.potential}
                    if warm_start.eigensolver_state is not None and not noncollinear:
                        keywords["initial_eigensolver_state"] = warm_start.eigensolver_state
                    return keywords

                _, result, spinor_forces = _run_polarizability(
                    system,
                    translation,
                    reporter,
                    log,
                    input_directory,
                    scf_runner=lambda run_system, callback=None, warm_start=None: spinor_driver(
                        run_system,
                        soc_projectors,
                        callback=callback,
                        **spinor_start(warm_start),
                        **state,
                    ),
                    iteration_report=reporter.iteration_self_consistent_soc,
                    finish_report=reporter.finish_self_consistent_soc,
                    warm_start_builder=lambda field_result: SpinorWarmStart(
                        field_result.next_potential,
                        getattr(field_result, "eigensolver_state", None),
                    ),
                )
            else:
                restart_keywords = {}
                if restart is not None and not noncollinear:
                    # Restart_Run from the parsec.dat of a spin-orbit run.
                    from .SCF.restart import spin_orbit_start

                    restart_keywords["initial_potential"] = spin_orbit_start(
                        restart, spin_polarized
                    )
                result = spinor_driver(
                    system,
                    soc_projectors,
                    callback=reporter.iteration_self_consistent_soc,
                    **restart_keywords,
                    **state,
                )
        scf_elapsed = time.perf_counter() - scf_start
        if getattr(translation, "polarizability_field", None) is None:
            reporter.finish_self_consistent_soc(result, scf_elapsed)
        if not arguments.no_archive:
            saved = save_self_consistent_soc_result_archive(
                archive_path,
                result,
                include_wavefunctions=(
                    arguments.save_wavefunctions
                    or translation.output_all_states
                ),
                dipole=getattr(reporter, "last_dipole", None),
                **({} if spinor_forces is None else {"forces": spinor_forces}),
            )
            log.write(f"Result archive: {saved}")
        write_files(result)
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
        if k_sampling is None and _gamma_spin_on_kpoint_driver(
            translation, is_periodic, has_soc_species
        ):
            # A Gamma-only periodic spin run is the 1x1x1 k-point run.  Through the k-point driver
            # it runs on the device; the Gamma driver has no device path.
            k_sampling = (np.zeros((1, 3)), np.ones(1))
        # Single-point forces of a wire or slab with spin and k-points
        # (parsec_python.Forces.wire_slab) read the orbitals of the loop.
        spin_kpoint_forces = bool(
            translation.compute_forces
            and k_sampling is not None
            and getattr(system.grid, "is_confined_periodic", False)
        )
        if spin_kpoint_forces:
            keep = {"keep_wavefunctions": True}
        if k_sampling is not None:
            k_points, k_weights = k_sampling
            result = run_scf_kpoints_spin_polarized(
                system,
                k_points,
                k_weights,
                callback=reporter.iteration_spin_polarized,
                **keep,
                **state,
            )
        elif getattr(translation, "polarizability_field", None) is not None:
            # polar.f90 around the spin loop of a spin-orbit species (the
            # perturbative correction follows the last loop).
            _, result, _ = _run_polarizability(
                system,
                translation,
                reporter,
                log,
                input_directory,
                scf_runner=lambda run_system, callback=None: run_scf_spin_polarized(
                    run_system, callback=callback, **state
                ),
                iteration_report=reporter.iteration_spin_polarized,
                finish_report=reporter.finish_spin_polarized,
            )
        else:
            result = run_scf_spin_polarized(
                system, callback=reporter.iteration_spin_polarized, **state
            )
        scf_elapsed = time.perf_counter() - scf_start
        forces = None
        if spin_kpoint_forces:
            forces = reference_backend().compute_forces(system, result)
            reporter.finish_spin_polarized(result, scf_elapsed, forces=forces)
        elif getattr(translation, "polarizability_field", None) is None:
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
                dipole=getattr(reporter, "last_dipole", None),
                **({} if forces is None else {"forces": forces}),
            )
            log.write(f"Result archive: {saved}")
        write_files(result)
        # The SCF archive is written first: the band calculation can take a long time
        # and a crash there must not lose the converged SCF.
        _run_band_structure(
            system, translation, result, reporter, input_directory
        )
        _run_dos(system, translation, result, reporter, input_directory)
        if getattr(translation, "gw", None) is not None and result.k_wavefunctions is not None:
            _run_gw_after_scf(system, translation, result, reporter.write, input_directory)
        total_elapsed = time.perf_counter() - start
        log.write(f" Total Python wall time [sec] : {total_elapsed:11.2f}")
        log.write(f"Text log: {log_path}")
        return 0 if result.converged else 3


    # Scalar (spin-unpolarized, no spin-orbit) k-point sampling.
    k_points, k_weights = k_sampling
    result = run_scf_kpoints(
        system,
        k_points,
        k_weights,
        callback=reporter.iteration,
        **({} if restart is None else {"initial_potential": restart.input_potential}),
        **keep,
        **state,
    )
    scf_elapsed = time.perf_counter() - scf_start
    forces = None
    if translation.compute_forces and getattr(system.grid, "is_confined_periodic", False):
        # Single-point forces of a wire or slab with k-points.
        forces = reference_backend().compute_forces(system, result)
        reporter.finish(result, scf_elapsed, forces=forces)
    else:
        reporter.finish(result, scf_elapsed)
    if not arguments.no_archive:
        saved = save_result_archive(
            archive_path,
            result,
            include_wavefunctions=(
                arguments.save_wavefunctions or translation.output_all_states
            ),
            **({} if forces is None else {"forces": forces}),
        )
        log.write(f"Result archive: {saved}")
    write_files(result)
    # The SCF archive is written first so a long band calculation cannot lose it.
    _run_band_structure(system, translation, result, reporter, input_directory)
    _run_dos(system, translation, result, reporter, input_directory)
    if getattr(translation, "gw", None) is not None:
        _run_gw_after_scf(system, translation, result, reporter.write, input_directory)
    total_elapsed = time.perf_counter() - start
    log.write(f" Total Python wall time [sec] : {total_elapsed:11.2f}")
    log.write(f"Text log: {log_path}")
    return 0 if result.converged else 3


def _run_polarizability(
    system,
    translation,
    reporter,
    log,
    directory: Path,
    k_sampling=None,
    *,
    scf_runner=None,
    iteration_report=None,
    finish_report=None,
    iteration_state_callback=None,
    warm_start_builder=None,
    compute_forces=None,
):
    """The seven field loops of ``Polarizability``, reported, with ``polar.dat``.

    Returns the :class:`parsec_python.SCF.polarizability.PolarizabilityResult`,
    the result of the last loop and its forces where they are asked for
    (``parsec.F90`` computes them after the loop, with the field along -z
    in the ionic potential and no force of the field on the nuclei).
    ``scf_runner(system, callback=...)``, ``iteration_report`` and
    ``finish_report(result, seconds, forces=...)`` replace ``run_scf`` and
    its reports for the spin-orbit, non-collinear and spin-orbit
    spin-polarized drivers; ``warm_start_builder(result)`` turns the result
    of a loop into the ``warm_start`` the next call of ``scf_runner`` takes
    (``None``: every loop starts from the initial density).  The forces of
    the last loop of the spinor drivers are those of
    :mod:`parsec_python.Forces.spin_orbit`.  ``iteration_state_callback``
    (``Save_Intermediate_*``) goes to the SCF loops of the default runners
    (``run_scf`` and the k-point runner); a given ``scf_runner`` carries its
    own.  ``compute_forces(system, result, symmetrize=, operations=)``
    replaces the forces of the reference backend (the accelerated CLI gives
    those of its driver).
    """

    from .Output.polarizability_output import (
        polar_header,
        polar_run_line,
        polar_summary,
    )
    from .SCF.polarizability import run_polarizability

    field = translation.polarizability_field
    names = ("0", "+x", "-x", "+y", "-y", "+z", "-z")
    clock: dict[str, object] = {}
    path = directory / "polar.dat"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(polar_header(system.atoms, list(system.pseudopotentials), field))
        stream.flush()

        def before_run(index, direction):
            log.write(
                f" Polarizability run {index + 1} of 7: field "
                + ("zero" if index == 0 else f"{field:.5f} along {names[index]}")
            )
            log.write()
            clock["start"] = time.perf_counter()

        def after_run(index, run_system, result, record):
            elapsed = time.perf_counter() - clock["start"]
            forces = None
            if index == 6 and translation.compute_forces:
                # The forces of the last loop (Forces.compute_forces sends a
                # spinor result to Forces.spin_orbit).
                forces = (
                    reference_backend().compute_forces
                    if compute_forces is None
                    else compute_forces
                )(run_system, result, symmetrize=False, operations=None)
            clock["forces"] = forces
            if finish_report is not None:
                if forces is None:
                    finish_report(result, elapsed)
                else:
                    finish_report(result, elapsed, forces=forces)
            else:
                reporter.finish(result, elapsed, forces=forces)
            stream.write(
                polar_run_line(record.direction, record.total_energy, record.dipole.dipole_debye)
            )
            stream.flush()

        extra = {}
        if scf_runner is not None:
            extra = {
                "scf_runner": lambda run_system, callback=None, warm_start=None: scf_runner(
                    run_system,
                    callback=callback,
                    **({} if warm_start is None else {"warm_start": warm_start}),
                ),
                "warm_start_builder": warm_start_builder,
            }
        elif k_sampling is not None:
            # A wire or slab with k-points (usrinputfile.F90 allows any
            # pbc%per <= 2).
            from .SCF.polarizability import kpoint_scf_runner

            extra = {
                "scf_runner": kpoint_scf_runner(
                    *k_sampling, iteration_state_callback=iteration_state_callback
                )
            }
        else:
            if system.input.scf.spin_polarized and system.input.periodic_cell is not None:
                # A spin-polarized wire or slab: run_spin_scf from the periodic
                # starting densities.
                from .SCF.pbc import with_initial_spin_densities

                system = with_initial_spin_densities(system)
            if iteration_state_callback is not None:
                extra = {
                    "scf_runner": lambda run_system, callback=None, warm_start=None: run_scf(
                        run_system,
                        callback=callback,
                        warm_start=warm_start,
                        iteration_state_callback=iteration_state_callback,
                    )
                }
        polarization, _, result = run_polarizability(
            system,
            field,
            callback=reporter.iteration if iteration_report is None else iteration_report,
            before_run=before_run,
            after_run=after_run,
            **extra,
        )
        stream.write(polar_summary(polarization))
    log.write(f" Polarizability results: {path}")
    return polarization, result, clock.get("forces")


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
    if (
        getattr(translation, "polarizability_field", None) is not None
        and getattr(translation, "energy_refinement", None) is not None
    ):
        print(
            "Input error: Energy_Refinement is not implemented together with "
            "Polarizability",
            file=sys.stderr,
        )
        return 2

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
    # Output_File_Name names the report as in PARSEC; --log overrules it.
    log_path = _resolve_output(
        arguments.log,
        input_directory,
        translation.output_files.output_file_name
        if arguments.bands_from is None
        else "parsec_bands.out",
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

    # The routines of the run (parsec_python.Input.routines), decided by the parser.
    routines = translation.routines
    spin_polarized = translation.problem.scf.spin_polarized
    self_consistent_spin_orbit = translation.problem.scf.self_consistent_spin_orbit
    k_sampling = kpoint_sampling(translation.problem) if is_periodic else None
    # parsec.dat, eigen.dat and the cube files of Output_Grid_Data, beside
    # the input as PARSEC writes them in its run directory.
    files = output_files_for(translation, input_directory, k_sampling)
    taken = {*protected_paths, log_path}
    if not arguments.no_archive:
        taken.add(archive_path)
    conflict = output_file_conflict(files, taken)
    if conflict is not None and arguments.bands_from is None:
        print(
            f"Input error: refusing to overwrite {conflict}: it is an input, the log "
            "or the archive of this run",
            file=sys.stderr,
        )
        return 2
    restart_file = (
        "wfn.dat" if getattr(translation, "restart_from_wfndat", False) else "parsec.dat"
    )
    if (
        translation.restart_run
        and arguments.bands_from is None
        and not (input_directory / restart_file).is_file()
    ):
        print(
            f"Input error: Restart_Run: {input_directory / restart_file} not found "
            "(PARSEC error 371)",
            file=sys.stderr,
        )
        return 2
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

    # The routines of the run (parsec_python.Input.routines): the dispatch of
    # the Fortran parity round for the Fortran-derived ones, Andrew Grieder's for the
    # default (every input that reaches it is one his routines compute).
    if routines.fortran:
        # Periodic spin polarization or k-points, spin-orbit coupling and non-collinear
        # magnetism run the drivers of parsec_python.SCF (k-point solver, spin-orbit,
        # non-collinear).  An isolated spin-polarized input without spin-orbit coupling
        # runs run_scf -> run_spin_scf, like every other isolated input, so it keeps
        # forces, relaxation, molecular dynamics and energy refinement.
        uses_extended_drivers = bool(
            self_consistent_spin_orbit
            or noncollinear
            or k_sampling is not None
            or (spin_polarized and (is_periodic or has_soc_species))
        )
        # A wire or slab with k-points (scalar or collinear spin) computes its
        # single-point forces after the k-point loop (parsec_python.Forces.wire_slab).
        wire_or_slab = getattr(translation.problem.grid, "periodicity", 3) in (1, 2)
        atoms_move = (
            translation.relaxation is not None or translation.molecular_dynamics is not None
        )
        kpoint_wire_slab_forces = bool(
            translation.compute_forces
            and wire_or_slab
            and k_sampling is not None
            and not atoms_move
            and not self_consistent_spin_orbit
            and not noncollinear
            and not has_soc_species
        )
        # Forces, relaxation and molecular dynamics of a periodic cell run the
        # ionic loop with the periodic backend (Relaxation.periodic_backend, for
        # a wire or slab Relaxation.wire_slab_backend): at Gamma (also with
        # collinear spin, through run_spin_scf) and with k-points (scalar or
        # collinear spin).  The single-point forces of a wire or slab with
        # k-points (scalar or collinear spin) stay after the k-point loop.
        periodic_ionic = bool(
            is_periodic
            and (atoms_move or translation.compute_forces)
            and not kpoint_wire_slab_forces
            and getattr(translation, "energy_refinement", None) is None
            and not (self_consistent_spin_orbit or noncollinear or has_soc_species)
        )
        # Polarizability of a wire or slab (usrinputfile.F90: pbc%per <= 2): the
        # seven field loops of run_scf (also collinear spin through run_spin_scf)
        # or of the k-point loop (scalar or collinear spin).
        wire_slab_polarizability = bool(
            wire_or_slab
            and getattr(translation, "polarizability_field", None) is not None
            and not (self_consistent_spin_orbit or noncollinear or has_soc_species)
        )
        if wire_slab_polarizability:
            periodic_ionic = True
        # Forces, relaxation and molecular dynamics of the spin-orbit
        # (SO_from_scratch/SCF_SO, also with collinear spin) and non-collinear
        # drivers, isolated or periodic: the ionic loop with the spinor backend
        # (Relaxation.spinor_backend, Forces.spin_orbit).
        spinor_ionic = bool(
            (self_consistent_spin_orbit or noncollinear)
            and not wire_or_slab
            and (
                translation.relaxation is not None
                or translation.molecular_dynamics is not None
                or translation.compute_forces
            )
            and getattr(translation, "energy_refinement", None) is None
            # Polarizability of the spinor drivers: the seven field loops of
            # _run_extended_drivers, with the forces of the last one.
            and getattr(translation, "polarizability_field", None) is None
        )
        # parsec.F90: after the seven loops of polar.f90 the forces of the last
        # one (field along -z) are computed, for the spinor drivers as for the
        # scalar loop (forcnloc.f90 + soforc.F90); PARSEC moves no atom with
        # Polarizability (error 155).
        spinor_polarizability_forces = bool(
            (self_consistent_spin_orbit or noncollinear)
            and getattr(translation, "polarizability_field", None) is not None
            and translation.compute_forces
            and translation.relaxation is None
            and translation.molecular_dynamics is None
            and getattr(translation, "energy_refinement", None) is None
        )
    else:
        # Self-consistent spin-orbit and non-collinear inputs keep all but the energy refinement.
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
        # what the Fortran-derived routines add (wires and slabs, polarizability) does not reach them
        wire_or_slab = False
        kpoint_wire_slab_forces = False
        wire_slab_polarizability = False
        spinor_polarizability_forces = False
    if uses_extended_drivers and not (periodic_ionic or spinor_ionic) and (
        translation.relaxation is not None
        or translation.molecular_dynamics is not None
        or (
            translation.compute_forces
            and not kpoint_wire_slab_forces
            and not spinor_polarizability_forces
        )
        or getattr(translation, "energy_refinement", None) is not None
    ):
        if (
            has_soc_species
            and not (self_consistent_spin_orbit or noncollinear)
            and getattr(translation, "energy_refinement", None) is None
        ):
            # The perturbative correction (pls.F90) has spinor eigenvalues but
            # no spin-orbit total energy here; PARSEC's forces of that path
            # are flagged "may be inaccurate" and built on soforc.F90.
            print(
                "Input error: Minimization, Molecular_Dynamics and Skip_Force: "
                "false are not implemented with the perturbative spin-orbit "
                "correction (SO_PSP without SO_from_scratch); give "
                "SO_from_scratch: .true. for spin-orbit forces",
                file=sys.stderr,
            )
            return 2
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
    if uses_extended_drivers and is_periodic and not wire_slab_polarizability and (
        getattr(translation, "polarizability_field", None) is not None
    ):
        # The parser refuses a bulk Polarizability (PARSEC error 156) and the
        # wire/slab inputs that wire_slab_polarizability leaves out.
        print(
            "Input error: Polarizability is not implemented for periodic cells",
            file=sys.stderr,
        )
        return 2
    if not routines.fortran and spinor_ionic and getattr(translation, "energy_refinement", None) is not None:
        print(
            "Input error: Energy_Refinement is not implemented together with spin-orbit coupling or "
            "non-collinear magnetism",
            file=sys.stderr,
        )
        return 2
    if not routines.fortran and spinor_ionic and not is_periodic and (noncollinear or spin_polarized):
        # Andrew Grieder's routines: a moment breaks the point group of the atoms, averaging the forces over it
        # would be wrong
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
            # What usrinputfile.F90 prints about the output keywords the input gives.
            for line in output_header_lines(
                translation.output_files, kpoints=k_sampling is not None
            ):
                log.write(line)
            # usrinputfile.F90: Calc_Hartree_G, nscf_kpoints and Create_DOS.
            for line in post_processing_header_lines(translation):
                log.write(line)
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
                    spin_polarized
                    and _gamma_spin_on_kpoint_driver(translation, is_periodic, has_soc_species)
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
            if arguments.bands_from is None:
                write_qpoints_file(translation.problem, input_directory, log.write)

            if translation.problem.eigensolver.method == "none":
                # Eigensolver: none is PARSEC's TEST mode (parsec.F90:1085-1089):
                # the setup is reported and the run ends without an SCF.
                reporter.test_mode_complete()
                total_elapsed = time.perf_counter() - start
                log.write(f" Total Python wall time [sec] : {total_elapsed:11.2f}")
                log.write(f"Text log: {log_path}")
                return 0

            if arguments.bands_from is not None:
                _run_bands_from_archive(
                    system, translation, arguments.bands_from, reporter, input_directory
                )
                log.write(f"Text log: {log_path}")
                return 0

            if getattr(translation, "nscf", None) is not None:
                # parsec.F90 l.1098: no SCF, calc_nscf on the restart potential.
                return _run_nscf(
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
                )

            restart = load_restart_start(
                translation, system, log, input_directory, k_sampling=k_sampling
            )
            start_output_files(files, translation, system, log, k_sampling=k_sampling)
            reporter = ReporterWithFiles(reporter, files)

            # Save_Intermediate_*: parsec.dat (and the eigen.dat levels of a
            # k-point run) after every SCF iteration, in every SCF loop.
            iteration_state_callback = (
                (lambda state: files.iteration_fields(state, system.grid, translation.problem))
                if files.wants_iteration_fields
                else None
            )
            if uses_extended_drivers and not (periodic_ionic or spinor_ionic):
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
                    files=files,
                    restart=restart,
                    iteration_state_callback=iteration_state_callback,
                )

            # Save_Intermediate_*: parsec.dat after every iteration.
            scf_keywords = {}
            if iteration_state_callback is not None:
                scf_keywords["iteration_state_callback"] = iteration_state_callback

            def ionic_backend_of(extrapolation: str = "none"):
                if not routines.fortran and spinor_ionic:
                    # Andrew Grieder's spinor backend (Forces.compute_spinor_forces, warm starts)
                    backend = spinor_backend(
                        *(k_sampling if k_sampling is not None else (None, None)), extrapolation
                    )
                elif not routines.fortran and is_periodic:
                    # Andrew Grieder's periodic backend; the spin-polarized loop is the
                    # Gamma-point one, whatever Monkhorst-Pack grid of one point the input names
                    backend_k_sampling = None if spin_polarized else k_sampling
                    backend = periodic_backend(
                        *(backend_k_sampling if backend_k_sampling is not None else (None, None)),
                        extrapolation,
                    )
                elif spinor_ionic:
                    backend = fortran_spinor_backend(k_sampling, periodic=is_periodic)
                elif wire_or_slab:
                    # Before the bulk branch: a wire or slab has periodic_cell too.
                    backend = wire_slab_backend(k_sampling, extrapolation)
                elif is_periodic:
                    backend = fortran_periodic_backend(k_sampling, extrapolation)
                else:
                    backend = reference_backend(extrapolation)
                if scf_keywords:
                    from dataclasses import replace as _replace
                    from functools import partial

                    backend = _replace(backend, run_scf=partial(backend.run_scf, **scf_keywords))
                if restart is not None:
                    from .SCF.restart import with_first_warm_start

                    backend = with_first_warm_start(backend, restart)
                return backend

            forces = None
            outcome = None
            dynamics = None
            polarization = None
            # Andrew Grieder's spinor loops report each SCF iteration and structure
            # with the spin-orbit tables of the reporter.
            ionic_reporter = (
                _SpinorReporter(reporter) if spinor_ionic and not routines.fortran else reporter
            )
            if getattr(translation, "polarizability_field", None) is not None:
                polarization, result, forces = _run_polarizability(
                    system,
                    translation,
                    reporter,
                    log,
                    input_directory,
                    k_sampling=k_sampling,
                    iteration_state_callback=iteration_state_callback,
                )
            elif translation.relaxation is not None:
                outcome = run_recorded_relaxation(
                    translation,
                    system,
                    ionic_backend_of(translation.potential_extrapolation),
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
                    ionic_backend_of(translation.potential_extrapolation),
                    ionic_reporter,
                    log.write,
                    directory=input_directory,
                    restart=dynamics_restart,
                )
                result, forces = dynamics.result, dynamics.forces
            elif translation.compute_forces:
                result, forces = run_single_point_with_forces(
                    system, ionic_backend_of(), ionic_reporter, symmetrize=use_symmetry
                )
            else:
                scf_start = time.perf_counter()
                if restart is not None:
                    scf_keywords["warm_start"] = restart
                result = run_scf(system, callback=reporter.iteration, **scf_keywords)
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
            if not arguments.no_archive and spinor_ionic and not routines.fortran:
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
                            ("polarizability", polarization),
                        )
                        if value is not None
                    },
                )
                log.write(f"Result archive: {saved}")
            files.finish(result, system.grid, translation.problem)
            # The SCF archive is written first so a long band calculation cannot lose it.
            _run_band_structure(system, translation, result, reporter, input_directory)
            # Create_DOS (parsec.F90 l.2181-2189), of the last structure.
            _run_dos(system, translation, result, reporter, input_directory)
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
