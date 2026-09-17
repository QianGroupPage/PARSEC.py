"""PARSEC-style reporting additions for accelerated runs."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from parsec_python.Output import ParsecTextReporter

from ..models import AcceleratedSinglePointResult


# Keep execution placement and precision visible even in the short report.
# In particular, "Selected backend = cupy" alone does not distinguish the
# default hybrid from a pure-CuPy run. Everything remains in backend_info;
# this allowlist controls text presentation only.
_SUMMARY_DETAIL_KEYS = frozenset({
    "finite_difference_builder",
    "hartree_backend",
    "native_openmp_max_threads",
    "gpu_later_subspace_filter_precision",
    "orbital_sector_later_filter_precision",
    "orbital_symmetry",
    "hartree_symmetry",
})


class AcceleratedTextReporter:
    """Delegate physical reporting and append backend provenance/timings."""

    def __init__(
        self,
        write: Callable[[str], None],
        translation,
        *,
        symmetry_mode: str | None = None,
    ) -> None:
        self.write = write
        report_translation = (
            translation
            if symmetry_mode is None
            else replace(
                translation, ignore_symmetry=(symmetry_mode == "off")
            )
        )
        self.reference = ParsecTextReporter(write, report_translation)
        self.output_level = report_translation.output_level

    def header(self) -> None:
        self.reference.header()

    def setup(self, system) -> None:
        self.reference.setup(system)
        info = system.backend_info
        details = dict(info.details)
        lines = [
            " Acceleration backend:",
            " ---------------------",
            f" Requested backend = {info.requested}",
            f" Selected backend  = {info.selected}",
            f" Numeric dtype     = {info.dtype}",
            f" Device            = {info.device}",
        ]
        detailed = self.output_level >= 2
        if detailed:
            lines.append(f" Implementation    = {info.implementation}")
        if details.get("orbital_symmetry", "").startswith("CuPy real"):
            lines.insert(
                0,
                " The reference setup above describes the constructed full "
                "grid.  The GPU orbital solve is decomposed into the exact "
                "symmetry representations reported below.\n",
            )
        elif details.get("hartree_symmetry") not in {None, "full grid"}:
            lines.insert(
                0,
                " The full-grid statement above applies to orbitals; "
                "Hartree uses the proven symmetry wedge reported below.\n",
            )
        for key, value in info.details:
            if detailed or key in _SUMMARY_DETAIL_KEYS:
                lines.append(f" {key} = {value}")
        if not detailed:
            lines.append(" Detailed backend settings: set Output_Level: 2")
        for reason in info.fallback_reasons:
            lines.append(f" Backend fallback  = {reason}")
        lines.append(
            " Backend initialization [sec] = "
            f"{system.backend.statistics.initialization_seconds:12.6f}"
        )
        lines.append("")
        self.write("\n".join(lines))

    def iteration(self, step) -> None:
        self.reference.iteration(step)

    def finish(
        self,
        result: AcceleratedSinglePointResult,
        elapsed_seconds: float,
    ) -> None:
        self.reference.finish(result, elapsed_seconds)
        stats = result.backend_statistics
        average = (
            stats.apply_seconds / stats.applications
            if stats.applications
            else 0.0
        )
        lines = [
            "",
            " Accelerated Hamiltonian statistics:",
            " -----------------------------------",
            f" Backend = {result.backend.selected}",
            f" H applications = {stats.applications:12d}",
            f" Orbital vectors applied = {stats.vectors_applied:12d}",
            f" Total H application time [sec] = {stats.apply_seconds:12.6f}",
            f" Average H application [sec] = {average:12.6f}",
            f" Local-potential updates = {stats.local_updates:12d}",
            f" Local update time [sec] = {stats.local_update_seconds:12.6f}",
        ]
        if stats.applications and not stats.apply_seconds:
            lines.append(
                " Per-H device timing = disabled (would synchronize every recurrence)"
            )
        if stats.eigensolver_first_calls:
            lines.extend(
                [
                    f" GPU initial-eigensolver calls = {stats.eigensolver_first_calls:12d}",
                    (
                        " GPU initial-eigensolver synchronized time [sec] = "
                        f"{stats.eigensolver_first_seconds:12.6f}"
                    ),
                ]
            )
        if (
            stats.initial_bound_seconds
            or stats.initial_filter_seconds
            or stats.initial_orthogonalization_seconds
            or stats.initial_projection_seconds
            or stats.initial_rotation_seconds
            or stats.initial_residual_seconds
            or stats.initial_cleanup_seconds
        ):
            lines.extend(
                [
                    " GPU initial-eigensolver asynchronous stage profile [sec]:",
                    f"   spectral bound       = {stats.initial_bound_seconds:12.6f}",
                    f"   Chebyshev filtering  = {stats.initial_filter_seconds:12.6f}",
                    f"   orthogonalization    = {stats.initial_orthogonalization_seconds:12.6f}",
                    f"   projection/small eig = {stats.initial_projection_seconds:12.6f}",
                    f"   Ritz rotations       = {stats.initial_rotation_seconds:12.6f}",
                    f"   residual/locking     = {stats.initial_residual_seconds:12.6f}",
                    f"   final cleanup        = {stats.initial_cleanup_seconds:12.6f}",
                    (
                        "   block-orth audits    = "
                        f"{stats.initial_block_orth_calls:6d} calls, "
                        f"{stats.initial_block_orth_fallbacks:6d} fallbacks"
                    ),
                ]
            )
        if stats.eigensolver_subspace_calls:
            lines.extend(
                [
                    f" GPU SUBSPACE calls = {stats.eigensolver_subspace_calls:12d}",
                    (
                        " GPU SUBSPACE synchronized time [sec] = "
                        f"{stats.eigensolver_subspace_seconds:12.6f}"
                    ),
                ]
            )
        if (
            stats.subspace_bound_seconds
            or stats.subspace_filter_seconds
            or stats.subspace_orthogonalization_seconds
            or stats.subspace_ritz_seconds
        ):
            lines.extend(
                [
                    " GPU SUBSPACE asynchronous stage profile [sec]:",
                    f"   spectral bound       = {stats.subspace_bound_seconds:12.6f}",
                    f"   Chebyshev filtering  = {stats.subspace_filter_seconds:12.6f}",
                    f"   orthogonalization    = {stats.subspace_orthogonalization_seconds:12.6f}",
                    f"   Rayleigh--Ritz       = {stats.subspace_ritz_seconds:12.6f}",
                    f"     H applied basis    = {stats.subspace_ritz_hamiltonian_seconds:12.6f}",
                    f"     overlap/projection = {stats.subspace_ritz_projection_seconds:12.6f}",
                    f"     Ritz rotation      = {stats.subspace_ritz_rotation_seconds:12.6f}",
                ]
            )
        if stats.eigensolver_scheduler_batches:
            lines.extend(
                [
                    (
                        " GPU representation scheduler batches = "
                        f"{stats.eigensolver_scheduler_batches:12d}"
                    ),
                    (
                        " GPU representation scheduler wall time [sec] = "
                        f"{stats.eigensolver_scheduler_wall_seconds:12.6f}"
                    ),
                ]
            )
        final_sector_counts = dict(result.backend.details).get(
            "orbital_sector_final_state_counts"
        )
        if final_sector_counts is not None:
            lines.append(
                " Final active states by representation = "
                f"{final_sector_counts}"
            )
        if stats.eigensolver_download_seconds:
            lines.append(
                " Requested-eigenpair download [sec] = "
                f"{stats.eigensolver_download_seconds:12.6f}"
            )
        if stats.density_calls:
            lines.extend(
                [
                    f" GPU density builds = {stats.density_calls:12d}",
                    (
                        " GPU density build/download [sec] = "
                        f"{stats.density_seconds:12.6f}"
                    ),
                ]
            )
        if stats.final_wavefunction_download_seconds:
            lines.append(
                " Final wavefunction download [sec] = "
                f"{stats.final_wavefunction_download_seconds:12.6f}"
            )
        if stats.hartree_solve_calls:
            lines.extend(
                [
                    f" Accelerated Hartree solves = {stats.hartree_solve_calls:12d}",
                    (
                        " Accelerated Hartree total [sec] = "
                        f"{stats.hartree_total_seconds:12.6f}"
                    ),
                ]
            )
            if stats.hartree_rhs_seconds:
                lines.append(
                    " Hartree boundary/RHS [sec] = "
                    f"{stats.hartree_rhs_seconds:12.6f}"
                )
            if stats.hartree_linear_solve_seconds:
                lines.append(
                    " Hartree linear solve [sec] = "
                    f"{stats.hartree_linear_solve_seconds:12.6f}"
                )
            if stats.hartree_upload_seconds or stats.hartree_download_seconds:
                lines.extend(
                    [
                        f" Hartree upload [sec] = {stats.hartree_upload_seconds:12.6f}",
                        f" Hartree download [sec] = {stats.hartree_download_seconds:12.6f}",
                    ]
                )
        if stats.warmup_seconds:
            lines.append(f" Backend warmup time [sec] = {stats.warmup_seconds:12.6f}")
        if stats.host_to_device_seconds or stats.device_to_host_seconds:
            lines.extend(
                [
                    f" Host-to-device time [sec] = {stats.host_to_device_seconds:12.6f}",
                    f" Device kernel time [sec] = {stats.device_seconds:12.6f}",
                    f" Device-to-host time [sec] = {stats.device_to_host_seconds:12.6f}",
                ]
            )
        for name, value in sorted(stats.component_profile_seconds.items()):
            lines.append(f" Profile {name} [sec] = {value:12.6f}")
        self.write("\n".join(lines))


__all__ = ["AcceleratedTextReporter"]
