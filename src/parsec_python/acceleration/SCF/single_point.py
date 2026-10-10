"""SCF composition that swaps only the Hamiltonian execution backend."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from parsec_python.SCF.single_point import (
    PreparedSinglePointSystem,
    run_scf as run_reference_scf,
)
from parsec_python.models import SCFIteration

from ..backends.base import BoundHamiltonian, HamiltonianBackend
from ..models import AcceleratedSinglePointResult, BackendInfo


@dataclass
class AcceleratedPreparedSinglePointSystem:
    """Reference static physics plus a cached accelerated H-application backend.

    Attribute access delegates to ``reference``.  The reference SCF driver is
    deliberately reused unchanged; its only behavioral substitution is that
    ``hamiltonian(V_eff)`` returns a backend-bound operator.
    """

    reference: PreparedSinglePointSystem
    backend: HamiltonianBackend
    backend_info: BackendInfo
    eigenproblem_solver: Callable[..., object] | None = None
    hartree_solver: Callable[..., object] | None = None
    orbital_density_builder: Callable[..., object] | None = None
    xc_evaluator: Callable[..., object] | None = None
    mixer_factory: Callable[..., object] | None = None
    residual_metrics_evaluator: Callable[..., object] | None = None
    total_energy_evaluator: Callable[..., object] | None = None
    scalar_field_adapter: object | None = None
    materialize_final_wavefunctions: bool = True
    # ``max |C_L|`` of the atomic tail the substituted Hartree solver adds.
    boundary_tail_maximum: float | None = None

    def __getattr__(self, name: str):
        return getattr(self.reference, name)

    @property
    def hartree_boundary_tail_maximum(self) -> float | None:
        if self.boundary_tail_maximum is not None:
            return self.boundary_tail_maximum
        return getattr(self.reference, "hartree_boundary_tail_maximum", None)

    def hamiltonian(self, effective_potential) -> BoundHamiltonian:
        return self.backend.bind(effective_potential)

    def solve_hartree(self, *args, **kwargs):
        if self.hartree_solver is not None:
            return self.hartree_solver(*args, **kwargs)
        return self.reference.solve_hartree(*args, **kwargs)

    def evaluate_xc(self, *args, **kwargs):
        if self.xc_evaluator is not None:
            return self.xc_evaluator(*args, **kwargs)
        return self.reference.evaluate_xc(*args, **kwargs)


def run_scf(
    system: AcceleratedPreparedSinglePointSystem,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
) -> AcceleratedSinglePointResult:
    """Run the validated SCF algorithm through an accelerated H backend."""
    symmetry_eigensolver = getattr(
        system.backend, "symmetry_eigensolver", None
    )
    restore_allocator = getattr(
        symmetry_eigensolver, "restore_memory_allocator", None
    )
    try:
        result = run_reference_scf(
            system,
            callback=callback,
            eigenproblem_solver=system.eigenproblem_solver,
            orbital_density_builder=system.orbital_density_builder,
            mixer_factory=system.mixer_factory,
            residual_metrics_evaluator=system.residual_metrics_evaluator,
            total_energy_evaluator=system.total_energy_evaluator,
            scalar_field_adapter=system.scalar_field_adapter,
        )
        return _finalize_result(system, result, symmetry_eigensolver)
    finally:
        if callable(restore_allocator):
            restore_allocator()


def _finalize_result(system, result, symmetry_eigensolver):
    """Finalize output inside the allocator-restoration scope, including OOMs."""
    # The CuPy SCF downloads only density vectors during nonlinear iterations.
    # Materialize the requested orbitals once for the public final result.
    if (
        system.backend_info.selected == "cupy"
        and system.materialize_final_wavefunctions
    ):
        from time import perf_counter

        import numpy as np

        from ..backends.cupy import require_cupy, synchronize

        cp, _ = require_cupy()
        wavefunctions = result.wavefunctions
        materialize_host = getattr(wavefunctions, "to_full_host", None)
        materialize = getattr(wavefunctions, "to_full_device", None)
        if callable(materialize) or isinstance(wavefunctions, cp.ndarray):
            synchronize()
            started = perf_counter()
            if callable(materialize_host):
                result.wavefunctions = materialize_host()
            elif callable(materialize):
                wavefunctions = materialize()
                result.wavefunctions = np.asarray(cp.asnumpy(wavefunctions), dtype=np.float64)
            else:
                result.wavefunctions = np.asarray(cp.asnumpy(wavefunctions), dtype=np.float64)
            synchronize()
            timing_stats = getattr(system.backend, "timing_stats", None)
            if timing_stats is not None:
                timing_stats.final_wavefunction_download_seconds += (
                    perf_counter() - started
                )
    synchronize_statistics = getattr(
        system.backend, "synchronize_statistics", None
    )
    if synchronize_statistics is not None:
        synchronize_statistics()
    symmetry_state = getattr(symmetry_eigensolver, "state", None)
    if symmetry_state is not None:
        keys = {
            "orbital_sector_final_state_counts",
            "orbital_memory_allocator",
            "orbital_sector_state_storage",
        }
        # Written at preparation from what the sectors held and from the
        # switch of the filter graphs; now from the later passes that ran
        # and from the graphs that the filters of the sectors recorded.
        ran = {
            "orbital_sector_later_filter_precision": getattr(
                symmetry_eigensolver, "later_filter_precision", None
            ),
            "orbital_sector_filter_graphs": getattr(
                symmetry_eigensolver, "recorded_filter_graphs", None
            ),
        }
        details = tuple(
            (item[0], ran[item[0]]) if ran.get(item[0]) is not None else item
            for item in system.backend_info.details
            if item[0] not in keys
        ) + (
            (
                "orbital_sector_final_state_counts",
                " ".join(
                    str(value)
                    for value in symmetry_state.sector_state_counts
                ),
            ),
            (
                "orbital_memory_allocator",
                str(symmetry_eigensolver.memory_allocator_policy),
            ),
            (
                "orbital_sector_state_storage",
                str(symmetry_eigensolver.sector_state_storage),
            ),
        )
        system.backend_info = replace(
            system.backend_info, details=details
        )
        system.backend.info = system.backend_info
    return AcceleratedSinglePointResult(
        result=result,
        backend=system.backend_info,
        backend_statistics=system.backend.statistics.snapshot(),
    )


__all__ = ["AcceleratedPreparedSinglePointSystem", "run_scf"]
