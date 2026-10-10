"""Host/SCF adapter around the optional device-resident CuPy kernels."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator

from parsec_python.V_ion import NonlocalProjectorOperator

from ..Eigensolvers.eigval import CuPyEigvalSolver
from ..Occupations import CuPyDeviceDensityBuilder
from ..models import BackendInfo
from .base import HamiltonianBackend, as_vector_block
from .cupy import (
    CuPyHamiltonian,
    CuPyTimingStats,
    require_cupy,
    synchronize,
    synchronized_call,
)


@dataclass(frozen=True)
class _SCFEigvalResult:
    """Small structural adapter consumed by the reference SCF loop."""

    eigenvalues: np.ndarray
    vectors: Any
    residual_norms: np.ndarray | None
    state: object
    solver_path: str
    restarted: bool
    restart_reason: str | None


class CuPySCFEigensolver:
    """Callable retaining PARSEC's buffered eigensubspace on one GPU."""

    def __init__(self, operator: CuPyHamiltonian) -> None:
        self.operator = operator
        self.solver: CuPyEigvalSolver | None = None

    def __call__(
        self,
        operator: Any,
        requested_states: int,
        *,
        settings,
        state=None,
    ) -> _SCFEigvalResult:
        if operator is not self.operator:
            raise ValueError("CuPy SCF received a different Hamiltonian allocation")
        if self.solver is None:
            self.solver = CuPyEigvalSolver(
                operator,
                settings=settings,
                retain_vectors_on_device=True,
                # Later SUBSPACE residuals are diagnostics only in the SCF
                # driver.  Avoid a redundant N-by-state rotation/residual pass
                # in the fastest production route.
                compute_subspace_residuals=False,
            )
        elif state is None:
            # A new run_scf call on the same prepared system is an independent
            # nonlinear solve and must begin with the configured CHEBFF or
            # CHEBDAV solver, exactly like the CPU state=None policy.  Keep the
            # uploaded static Hamiltonian but discard only the buffered
            # eigensubspace.
            self.solver.reset()
        elif state is not self.solver.device_state:
            raise ValueError(
                "CuPy SCF state does not belong to this persistent eigensolver"
            )
        result = self.solver.solve(requested_states, settings=settings)
        return _SCFEigvalResult(
            eigenvalues=result.eigenvalues,
            vectors=result.vectors,
            residual_norms=result.residual_norms,
            state=self.solver.device_state,
            solver_path=result.solver_path,
            restarted=result.restarted,
            restart_reason=result.restart_reason,
        )


class CuPyBoundHamiltonian:
    """One local field bound to a persistent device Hamiltonian allocation."""

    def __init__(
        self,
        backend: "CuPyHamiltonianBackend",
        effective_potential: np.ndarray,
    ) -> None:
        self.backend = backend
        self.backend.update_local(effective_potential)

    @property
    def shape(self) -> tuple[int, int]:
        return self.backend.shape

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        return self.backend.apply(vectors)

    def apply_kinetic(self, vectors: np.ndarray) -> np.ndarray:
        return self.backend.apply_kinetic(vectors)

    def apply_local(self, vectors: np.ndarray) -> np.ndarray:
        return self.backend.apply_local(vectors)

    def apply_nonlocal(self, vectors: np.ndarray) -> np.ndarray:
        return self.backend.apply_nonlocal(vectors)

    def as_linear_operator(self) -> "LinearOperator":
        """Expose a host diagnostic operator; production SCF does not use it."""

        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(
            self.shape,
            matvec=self.apply,
            matmat=self.apply,
            rmatvec=self.apply,
            dtype=np.float64,
        )

    def as_eigensolver_operator(self) -> Any:
        """Return the persistent eigensolver identity without a host transfer."""

        return self.backend.eigensolver_operator

    def as_sparse(self) -> sp.csr_matrix:
        return self.backend.as_sparse()

    def __matmul__(self, vectors: Any) -> np.ndarray:
        return self.apply(vectors)


class CuPyHamiltonianBackend(HamiltonianBackend):
    """Float64 CUDA backend with static sparse terms uploaded exactly once."""

    def __init__(
        self,
        negative_laplacian: sp.spmatrix | object,
        nonlocal_operator: NonlocalProjectorOperator,
        *,
        requested: str = "cupy",
        fallback_reasons: tuple[str, ...] = (),
        defer_device_operator: bool = False,
    ) -> None:
        cp, _ = require_cupy()
        from ..Laplacian import DeferredNativeNegativeLaplacian

        lazy_kinetic = isinstance(
            negative_laplacian, DeferredNativeNegativeLaplacian
        )
        if lazy_kinetic and defer_device_operator:
            kinetic = negative_laplacian
            kinetic_shape = negative_laplacian.shape
            kinetic_nnz = int(negative_laplacian.nnz)
        else:
            from ..Laplacian import materialize_negative_laplacian

            kinetic = materialize_negative_laplacian(negative_laplacian)
            kinetic_shape = kinetic.shape
            kinetic_nnz = int(kinetic.nnz)
        projectors = sp.csc_matrix(
            nonlocal_operator.projectors, dtype=np.float64
        )
        projectors.sort_indices()
        if projectors.shape[0] != kinetic_shape[0]:
            raise ValueError(
                "projector row count must match the negative Laplacian"
            )
        super().__init__(kinetic_shape)
        self.host_negative_laplacian = kinetic
        self.host_projectors = projectors
        self.host_signs = np.asarray(nonlocal_operator.signs, dtype=np.float64)
        self.host_nonlocal_operator = NonlocalProjectorOperator(
            projectors=projectors,
            signs=self.host_signs.copy(),
            labels=tuple(nonlocal_operator.labels),
        )
        self.local_potential = np.zeros(kinetic_shape[0], dtype=np.float64)
        self.timing_stats = CuPyTimingStats()
        self._defer_device_operator = bool(defer_device_operator)
        self.device_operator: CuPyHamiltonian | None = None
        if not self._defer_device_operator:
            self.device_operator = self._make_device_operator()
        # The reference SCF loop uses object identity to ensure that a saved
        # eigensubspace belongs to the Hamiltonian allocation.  In symmetry
        # mode the full-grid allocation is deliberately absent, so retain a
        # stable opaque identity for the representation eigensolver instead.
        self.eigensolver_operator: Any = (
            self.device_operator if self.device_operator is not None else object()
        )
        self.eigenproblem_solver = (
            CuPySCFEigensolver(self.device_operator)
            if self.device_operator is not None
            else None
        )
        self.orbital_density_builder = CuPyDeviceDensityBuilder(
            cp, self.timing_stats
        )
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        raw_name = properties.get("name", b"CUDA device")
        device_name = (
            raw_name.decode(errors="replace")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        self.info = BackendInfo(
            requested=requested,
            selected="cupy",
            device=f"CUDA:{cp.cuda.Device().id} ({device_name})",
            implementation=(
                "CuPy float64 device-resident stencil-major/compact-CSR "
                "Hamiltonian with fused Chebyshev recurrence, "
                "CHEBFF/CHEBDAV/SUBSPACE eigensolver, and fused "
                "orbital-density construction"
            ),
            fallback_reasons=fallback_reasons,
            details=(
                ("laplacian_nnz", str(kinetic_nnz)),
                ("projector_columns", str(projectors.shape[1])),
                ("cupy_version", str(cp.__version__)),
                (
                    "gpu_finite_difference_storage",
                    (
                        self.device_operator.compact_finite_difference.storage_mode
                        if self.device_operator is not None
                        and self.device_operator.compact_finite_difference is not None
                        else (
                            "deferred_full_grid_symmetry_sectors"
                            if self._defer_device_operator
                            else "float64_csr"
                        )
                    ),
                ),
                (
                    "gpu_finite_difference_palette_size",
                    (
                        str(
                            self.device_operator.compact_finite_difference.palette_size
                        )
                        if self.device_operator is not None
                        and self.device_operator.compact_finite_difference is not None
                        else "0"
                    ),
                ),
                (
                    "gpu_finite_difference_fallback",
                    str(
                        (
                            self.device_operator.compact_finite_difference_reason
                            if self.device_operator is not None
                            else "full-grid operator intentionally deferred"
                        )
                        or "none"
                    ),
                ),
                (
                    "gpu_chebyshev_recurrence",
                    (
                        "fused_hamiltonian_recurrence"
                        if self.device_operator is not None
                        and hasattr(
                            self.device_operator.compact_finite_difference,
                            "chebyshev_recurrence",
                        )
                        else "separate_hamiltonian_and_elementwise_kernels"
                    ),
                ),
                (
                    "gpu_later_subspace_filter_precision",
                    (
                        "float32 stencil/projectors/recurrence; float64 Ritz and SCF"
                        if self.device_operator is not None
                        and self.device_operator.mixed_precision_recurrence
                        is not None
                        else (
                            "deferred_to_symmetry_sector_operators"
                            if self._defer_device_operator
                            else "float64"
                        )
                    ),
                ),
                (
                    "gpu_nonlocal_application",
                    (
                        "B.T projection plus KB scatter fused into CUDA stencil"
                        if self.device_operator is not None
                        and self.device_operator.fused_projector_scatter
                        else (
                            "deferred_to_symmetry_sector_operators"
                            if self._defer_device_operator
                            else "two sparse KB contractions"
                        )
                    ),
                ),
                (
                    "gpu_projector_reduction",
                    (
                        getattr(
                            self.device_operator.custom_projector_projection,
                            "reduction_mode",
                            "cuSPARSE",
                        )
                        if self.device_operator is not None
                        else "deferred_to_symmetry_sector_operators"
                    ),
                ),
                ("gpu_subspace_residuals", "disabled_diagnostics_only"),
            ),
        )
        self.statistics.initialization_seconds = (
            self.timing_stats.initialization_seconds
        )

    def _make_device_operator(self) -> CuPyHamiltonian:
        """Upload the full-grid operator only when a full-grid action is used."""

        from ..Laplacian import materialize_negative_laplacian

        kinetic = materialize_negative_laplacian(
            self.host_negative_laplacian
        )
        self.host_negative_laplacian = kinetic

        return CuPyHamiltonian(
            kinetic,
            self.local_potential,
            self.host_nonlocal_operator,
            timing_stats=self.timing_stats,
        )

    def ensure_device_operator(self) -> CuPyHamiltonian:
        """Materialize the optional full-grid diagnostic/operator allocation."""

        if self.device_operator is None:
            self.device_operator = self._make_device_operator()
            self.statistics.initialization_seconds = (
                self.timing_stats.initialization_seconds
            )
        return self.device_operator

    def bind(self, effective_potential: np.ndarray) -> CuPyBoundHamiltonian:
        return CuPyBoundHamiltonian(self, effective_potential)

    def update_local(self, effective_potential: np.ndarray) -> None:
        from ..SCF.symmetry_fields import SymmetryScalarField

        compact = isinstance(effective_potential, SymmetryScalarField)
        if compact:
            potential = effective_potential.values
        else:
            potential = np.asarray(effective_potential, dtype=np.float64)
            if potential.shape != (self.shape[0],):
                raise ValueError("effective potential does not match the Hamiltonian")
            if not np.all(np.isfinite(potential)):
                raise ValueError("effective potential contains nonfinite values")
        started = perf_counter()
        if self.device_operator is not None:
            full_potential = (
                np.ascontiguousarray(
                    potential[effective_potential.reduction.full_to_wedge]
                )
                if compact
                else potential
            )
            self.device_operator.update_local_potential(full_potential)
        self.local_potential = (
            effective_potential.copy() if compact else potential.copy()
        )
        self.statistics.local_updates += 1
        self.statistics.local_update_seconds += perf_counter() - started

    @staticmethod
    def _restore_rank(result: np.ndarray, was_vector: bool) -> np.ndarray:
        return result[:, 0] if was_vector else result

    def _host_apply(self, operation, vectors: np.ndarray) -> np.ndarray:
        cp, _ = require_cupy()
        block, was_vector = as_vector_block(vectors)
        device_block, upload_seconds = synchronized_call(
            cp.asarray, block, dtype=cp.float64
        )
        device_result, device_seconds = synchronized_call(operation, device_block)
        host_result, download_seconds = synchronized_call(cp.asnumpy, device_result)
        self.statistics.host_to_device_seconds += upload_seconds
        self.statistics.device_seconds += device_seconds
        self.statistics.device_to_host_seconds += download_seconds
        return self._restore_rank(np.asarray(host_result), was_vector)

    def apply_kinetic(self, vectors: np.ndarray) -> np.ndarray:
        return self._host_apply(
            self.ensure_device_operator().apply_kinetic, vectors
        )

    def apply_local(self, vectors: np.ndarray) -> np.ndarray:
        return self._host_apply(self.ensure_device_operator().apply_local, vectors)

    def apply_nonlocal(self, vectors: np.ndarray) -> np.ndarray:
        return self._host_apply(
            self.ensure_device_operator().apply_nonlocal, vectors
        )

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        started = perf_counter()
        block, was_vector = as_vector_block(vectors)
        result = self._host_apply(self.ensure_device_operator().apply, block)
        self.statistics.applications += 1
        self.statistics.vectors_applied += int(block.shape[1])
        self.statistics.apply_seconds += perf_counter() - started
        return self._restore_rank(np.asarray(result).reshape(block.shape), was_vector)

    def synchronize(self) -> None:
        synchronize()

    def profile_components(self, vectors: np.ndarray) -> dict[str, float]:
        """Time device kernels with one upload and coarse synchronization."""

        cp, _ = require_cupy()
        block, _ = as_vector_block(vectors)
        device_block, upload_seconds = synchronized_call(
            cp.asarray, block, dtype=cp.float64
        )
        self.statistics.host_to_device_seconds += upload_seconds
        device_operator = self.ensure_device_operator()
        timings: dict[str, float] = {}
        for name, operation in (
            ("finite_difference_apply", device_operator.apply_kinetic),
            ("local_potential_apply", device_operator.apply_local),
            ("nonlocal_potential_apply", device_operator.apply_nonlocal),
        ):
            _, elapsed = synchronized_call(operation, device_block)
            timings[name] = elapsed
        self.statistics.component_profile_seconds = dict(timings)
        return timings

    def as_sparse(self) -> sp.csr_matrix:
        from ..Laplacian import materialize_negative_laplacian

        kinetic = materialize_negative_laplacian(
            self.host_negative_laplacian
        )
        self.host_negative_laplacian = kinetic
        nonlocal_matrix = (
            self.host_projectors
            @ sp.diags(self.host_signs, format="csc")
            @ self.host_projectors.T
        )
        return (
            kinetic
            + sp.diags(self.local_potential, format="csr")
            + nonlocal_matrix
        ).tocsr()

    def synchronize_statistics(self) -> None:
        """Copy synchronized eigensolver totals into common report fields."""

        self.statistics.initialization_seconds = (
            self.timing_stats.initialization_seconds
        )
        self.statistics.applications = self.timing_stats.hamiltonian_applications
        self.statistics.vectors_applied = self.timing_stats.orbital_vectors_applied
        self.statistics.eigensolver_first_calls = self.timing_stats.first_solve_calls
        self.statistics.eigensolver_first_seconds = self.timing_stats.first_solve_seconds
        self.statistics.eigensolver_subspace_calls = (
            self.timing_stats.subspace_solve_calls
        )
        self.statistics.eigensolver_subspace_seconds = (
            self.timing_stats.subspace_solve_seconds
        )
        self.statistics.initial_bound_seconds = (
            self.timing_stats.initial_bound_seconds
        )
        self.statistics.initial_filter_seconds = (
            self.timing_stats.initial_filter_seconds
        )
        self.statistics.initial_orthogonalization_seconds = (
            self.timing_stats.initial_orthogonalization_seconds
        )
        self.statistics.initial_projection_seconds = (
            self.timing_stats.initial_projection_seconds
        )
        self.statistics.initial_rotation_seconds = (
            self.timing_stats.initial_rotation_seconds
        )
        self.statistics.initial_residual_seconds = (
            self.timing_stats.initial_residual_seconds
        )
        self.statistics.initial_cleanup_seconds = (
            self.timing_stats.initial_cleanup_seconds
        )
        self.statistics.initial_block_orth_calls = (
            self.timing_stats.initial_block_orth_calls
        )
        self.statistics.initial_block_orth_fallbacks = (
            self.timing_stats.initial_block_orth_fallbacks
        )
        self.statistics.eigensolver_bound_prepare_wall_seconds = (
            self.timing_stats.eigensolver_bound_prepare_wall_seconds
        )
        self.statistics.subspace_bound_seconds = (
            self.timing_stats.subspace_bound_seconds
        )
        self.statistics.subspace_filter_seconds = (
            self.timing_stats.subspace_filter_seconds
        )
        self.statistics.subspace_orthogonalization_seconds = (
            self.timing_stats.subspace_orthogonalization_seconds
        )
        self.statistics.subspace_ritz_seconds = (
            self.timing_stats.subspace_ritz_seconds
        )
        self.statistics.subspace_ritz_hamiltonian_seconds = (
            self.timing_stats.subspace_ritz_hamiltonian_seconds
        )
        self.statistics.subspace_ritz_projection_seconds = (
            self.timing_stats.subspace_ritz_projection_seconds
        )
        self.statistics.subspace_ritz_rotation_seconds = (
            self.timing_stats.subspace_ritz_rotation_seconds
        )
        self.statistics.eigensolver_download_seconds = (
            self.timing_stats.download_seconds
        )
        symmetry_eigensolver = getattr(self, "symmetry_eigensolver", None)
        if symmetry_eigensolver is not None:
            self.statistics.eigensolver_scheduler_batches = (
                symmetry_eigensolver.scheduler_batches
            )
            self.statistics.eigensolver_scheduler_wall_seconds = (
                symmetry_eigensolver.scheduler_wall_seconds
            )
        self.statistics.density_calls = self.timing_stats.density_calls
        self.statistics.density_seconds = self.timing_stats.density_seconds
        self.statistics.final_wavefunction_download_seconds = (
            self.timing_stats.final_wavefunction_download_seconds
        )
        self.statistics.device_seconds = (
            self.timing_stats.first_solve_seconds
            + self.timing_stats.subspace_solve_seconds
            + self.timing_stats.eigensolver_bound_prepare_wall_seconds
        )
        self.statistics.device_to_host_seconds = self.timing_stats.download_seconds
        poisson_solver = getattr(self, "poisson_solver", None)
        if poisson_solver is not None:
            poisson = poisson_solver.timings
            self.statistics.hartree_solve_calls = poisson.solve_calls
            self.statistics.hartree_total_seconds = poisson.total_seconds
            self.statistics.hartree_rhs_seconds = poisson.rhs_seconds
            self.statistics.hartree_upload_seconds = poisson.upload_seconds
            self.statistics.hartree_linear_solve_seconds = poisson.solve_seconds
            self.statistics.hartree_download_seconds = poisson.download_seconds


__all__ = [
    "CuPyBoundHamiltonian",
    "CuPyHamiltonianBackend",
    "CuPySCFEigensolver",
]
