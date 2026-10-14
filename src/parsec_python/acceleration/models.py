"""Backend metadata and timing models for the accelerated single-point path.

The validated physical models remain in :mod:`parsec_python.models`.
This module contains only acceleration-specific state so the reference package
does not need to know about optional native or GPU runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


BackendName = Literal["auto", "scipy", "native", "cupy"]
SymmetryMode = Literal["auto", "on", "off"]


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly selected acceleration backend cannot run."""


@dataclass(frozen=True)
class BackendInfo:
    """Resolved execution backend recorded with each accelerated run."""

    requested: BackendName
    selected: str
    dtype: str = "float64"
    device: str = "CPU"
    implementation: str = ""
    fallback_reasons: tuple[str, ...] = ()
    details: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, str | tuple[str, ...]]:
        result: dict[str, str | tuple[str, ...]] = {
            "requested": self.requested,
            "selected": self.selected,
            "dtype": self.dtype,
            "device": self.device,
            "implementation": self.implementation,
            "fallback_reasons": self.fallback_reasons,
        }
        result.update(dict(self.details))
        return result


@dataclass
class BackendStatistics:
    """Coarse backend timings that avoid perturbing every Hamiltonian term.

    Production mode records one wall interval around each complete ``H @ Q``.
    Device backends may additionally fill transfer/device fields using their
    own synchronization or event mechanism.  Static component construction is
    recorded by ``parsec_python.PreparationTimings`` instead.
    """

    initialization_seconds: float = 0.0
    warmup_seconds: float = 0.0
    local_updates: int = 0
    local_update_seconds: float = 0.0
    applications: int = 0
    vectors_applied: int = 0
    apply_seconds: float = 0.0
    host_to_device_seconds: float = 0.0
    device_seconds: float = 0.0
    device_to_host_seconds: float = 0.0
    eigensolver_first_calls: int = 0
    eigensolver_first_seconds: float = 0.0
    eigensolver_subspace_calls: int = 0
    eigensolver_subspace_seconds: float = 0.0
    initial_bound_seconds: float = 0.0
    initial_filter_seconds: float = 0.0
    initial_orthogonalization_seconds: float = 0.0
    initial_projection_seconds: float = 0.0
    initial_rotation_seconds: float = 0.0
    initial_residual_seconds: float = 0.0
    initial_cleanup_seconds: float = 0.0
    initial_block_orth_calls: int = 0
    initial_block_orth_fallbacks: int = 0
    subspace_bound_seconds: float = 0.0
    subspace_filter_seconds: float = 0.0
    subspace_orthogonalization_seconds: float = 0.0
    subspace_ritz_seconds: float = 0.0
    subspace_ritz_hamiltonian_seconds: float = 0.0
    subspace_ritz_projection_seconds: float = 0.0
    subspace_ritz_rotation_seconds: float = 0.0
    eigensolver_download_seconds: float = 0.0
    eigensolver_scheduler_batches: int = 0
    eigensolver_scheduler_wall_seconds: float = 0.0
    eigensolver_bound_prepare_wall_seconds: float = 0.0
    density_calls: int = 0
    density_seconds: float = 0.0
    final_wavefunction_download_seconds: float = 0.0
    hartree_solve_calls: int = 0
    hartree_total_seconds: float = 0.0
    hartree_rhs_seconds: float = 0.0
    hartree_upload_seconds: float = 0.0
    hartree_linear_solve_seconds: float = 0.0
    hartree_download_seconds: float = 0.0
    component_profile_seconds: dict[str, float] = field(default_factory=dict)

    def snapshot(self) -> "BackendStatistics":
        return BackendStatistics(
            initialization_seconds=float(self.initialization_seconds),
            warmup_seconds=float(self.warmup_seconds),
            local_updates=int(self.local_updates),
            local_update_seconds=float(self.local_update_seconds),
            applications=int(self.applications),
            vectors_applied=int(self.vectors_applied),
            apply_seconds=float(self.apply_seconds),
            host_to_device_seconds=float(self.host_to_device_seconds),
            device_seconds=float(self.device_seconds),
            device_to_host_seconds=float(self.device_to_host_seconds),
            eigensolver_first_calls=int(self.eigensolver_first_calls),
            eigensolver_first_seconds=float(self.eigensolver_first_seconds),
            eigensolver_subspace_calls=int(self.eigensolver_subspace_calls),
            eigensolver_subspace_seconds=float(self.eigensolver_subspace_seconds),
            initial_bound_seconds=float(self.initial_bound_seconds),
            initial_filter_seconds=float(self.initial_filter_seconds),
            initial_orthogonalization_seconds=float(
                self.initial_orthogonalization_seconds
            ),
            initial_projection_seconds=float(self.initial_projection_seconds),
            initial_rotation_seconds=float(self.initial_rotation_seconds),
            initial_residual_seconds=float(self.initial_residual_seconds),
            initial_cleanup_seconds=float(self.initial_cleanup_seconds),
            initial_block_orth_calls=int(self.initial_block_orth_calls),
            initial_block_orth_fallbacks=int(self.initial_block_orth_fallbacks),
            subspace_bound_seconds=float(self.subspace_bound_seconds),
            subspace_filter_seconds=float(self.subspace_filter_seconds),
            subspace_orthogonalization_seconds=float(
                self.subspace_orthogonalization_seconds
            ),
            subspace_ritz_seconds=float(self.subspace_ritz_seconds),
            subspace_ritz_hamiltonian_seconds=float(
                self.subspace_ritz_hamiltonian_seconds
            ),
            subspace_ritz_projection_seconds=float(
                self.subspace_ritz_projection_seconds
            ),
            subspace_ritz_rotation_seconds=float(
                self.subspace_ritz_rotation_seconds
            ),
            eigensolver_download_seconds=float(self.eigensolver_download_seconds),
            eigensolver_scheduler_batches=int(self.eigensolver_scheduler_batches),
            eigensolver_scheduler_wall_seconds=float(
                self.eigensolver_scheduler_wall_seconds
            ),
            eigensolver_bound_prepare_wall_seconds=float(
                self.eigensolver_bound_prepare_wall_seconds
            ),
            density_calls=int(self.density_calls),
            density_seconds=float(self.density_seconds),
            final_wavefunction_download_seconds=float(
                self.final_wavefunction_download_seconds
            ),
            hartree_solve_calls=int(self.hartree_solve_calls),
            hartree_total_seconds=float(self.hartree_total_seconds),
            hartree_rhs_seconds=float(self.hartree_rhs_seconds),
            hartree_upload_seconds=float(self.hartree_upload_seconds),
            hartree_linear_solve_seconds=float(self.hartree_linear_solve_seconds),
            hartree_download_seconds=float(self.hartree_download_seconds),
            component_profile_seconds=dict(self.component_profile_seconds),
        )


@dataclass
class AcceleratedSinglePointResult:
    """Reference numerical result plus acceleration metadata.

    Unknown attributes are delegated to ``result`` so existing reporting and
    archive code can consume this wrapper as if it were ``SinglePointResult``.
    """

    result: object
    backend: BackendInfo
    backend_statistics: BackendStatistics

    def __getattr__(self, name: str):
        return getattr(self.result, name)


__all__ = [
    "AcceleratedSinglePointResult",
    "BackendInfo",
    "BackendName",
    "BackendStatistics",
    "BackendUnavailableError",
    "SymmetryMode",
]
