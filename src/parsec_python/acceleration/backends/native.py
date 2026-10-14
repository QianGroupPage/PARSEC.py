"""Lazy C++/OpenMP backend for the accelerated single-point path.

The compiled extension is optional.  Importing this module never imports it;
selection code can therefore offer SciPy and CuPy backends in environments
where the native wheel has not been built.
"""

from __future__ import annotations

from functools import lru_cache
import importlib
from time import perf_counter
from types import ModuleType

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import RealSpaceGrid
from parsec_python.V_ion import NonlocalProjectorOperator

from .base import HamiltonianBackend, as_vector_block
from ..models import BackendInfo, BackendUnavailableError


@lru_cache(maxsize=1)
def _load_native() -> ModuleType:
    """Import the extension on first use and provide an actionable failure."""
    try:
        return importlib.import_module("parsec_accelerated_native")
    except (ImportError, OSError) as exc:
        raise BackendUnavailableError(
            "the native backend is not installed or could not be loaded; "
            "build it with `python -m pip install -v "
            "src/parsec_python/acceleration/native` from the repository root"
        ) from exc


def native_available() -> bool:
    """Return whether the optional extension imports successfully."""
    try:
        _load_native()
    except BackendUnavailableError:
        return False
    return True


def native_unavailable_reason() -> str | None:
    """Return the extension import error used by ``backend=auto`` reporting."""
    try:
        _load_native()
    except BackendUnavailableError as exc:
        cause = exc.__cause__
        return f"{exc}: {cause}" if cause is not None else str(exc)
    return None


def native_build_info() -> dict[str, object]:
    """Return compiler/runtime capabilities reported by the extension."""
    return dict(_load_native().build_info())


def _canonical_csr(matrix: sp.spmatrix) -> sp.csr_matrix:
    result = sp.csr_matrix(matrix, dtype=np.float64)
    result.sum_duplicates()
    result.sort_indices()
    if result.shape[0] != result.shape[1]:
        raise ValueError("negative_laplacian must be square")
    return result


def _canonical_csc(matrix: sp.spmatrix, row_count: int) -> sp.csc_matrix:
    result = sp.csc_matrix(matrix, dtype=np.float64)
    result.sum_duplicates()
    result.sort_indices()
    if result.shape[0] != row_count:
        raise ValueError("projector row count must match the negative Laplacian")
    return result


def _indices64(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def _values64(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def build_native_negative_laplacian(grid: RealSpaceGrid) -> sp.csr_matrix:
    """Build ``-nabla_FD^2`` with native compressed-grid lookup loops.

    The returned canonical CSR matrix has the same active-domain row ordering,
    centered coefficients, and zero exterior orbital boundary as
    ``parsec_python.Laplacian.build_negative_laplacian``.
    """
    native = _load_native()
    payload = native.build_negative_laplacian_buffers(
        _indices64(grid.integer_coordinates),
        _indices64(grid.index_min),
        _indices64(grid.lookup),
        int(grid.settings.expansion_order),
        float(grid.spacing),
    )
    shape = tuple(int(value) for value in payload["shape"])
    result = sp.csr_matrix(
        (
            _values64(payload["data"]),
            _indices64(payload["indices"]),
            _indices64(payload["indptr"]),
        ),
        shape=shape,
    )
    result.sum_duplicates()
    result.sort_indices()
    return result


class NativeConjugateGradientBackend:
    """Cached native CG solver for one static float64 CSR operator.

    This is the low-level modular interface.  Its right-hand side must already
    contain any finite-domain boundary correction; for Hartree Poisson use,
    that is ``8*pi*rho_I - A_IB*V_B``.  The extension copies the canonical CSR
    buffers once and returns the same work counters and residual diagnostics as
    ``parsec_python.Hartree.poisson._conjugate_gradient``.
    """

    def __init__(self, operator: sp.spmatrix) -> None:
        native = _load_native()
        self.operator = _canonical_csr(operator)
        self.shape = self.operator.shape
        self._native_solver = native.ConjugateGradientSolver(
            _indices64(self.operator.indptr),
            _indices64(self.operator.indices),
            _values64(self.operator.data),
        )
        self.storage_mode = str(self._native_solver.storage_mode)
        self.worker_count = int(self._native_solver.worker_count)
        self.coefficient_palette_size = int(
            self._native_solver.coefficient_palette_size
        )

    def solve(
        self,
        right_hand_side: np.ndarray,
        initial: np.ndarray,
        *,
        relative_tolerance: float,
        absolute_tolerance: float,
        max_iterations: int,
    ) -> dict[str, object]:
        """Solve ``A*x=b`` with PARSEC-port CG convergence semantics."""
        rhs = _values64(right_hand_side)
        starting_vector = _values64(initial)
        expected = (self.shape[0],)
        if rhs.shape != expected:
            raise ValueError("right_hand_side does not match the CG operator")
        if starting_vector.shape != expected:
            raise ValueError("initial vector does not match the CG operator")
        if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be finite and positive")
        if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
            raise ValueError("absolute_tolerance must be finite and nonnegative")
        iteration_limit = int(max_iterations)
        if iteration_limit != max_iterations or iteration_limit < 1:
            raise ValueError("max_iterations must be a positive integer")
        return dict(
            self._native_solver.solve(
                rhs,
                starting_vector,
                float(relative_tolerance),
                float(absolute_tolerance),
                iteration_limit,
            )
        )


class NativeHamiltonianBackend(HamiltonianBackend):
    """Cached float64 C++/OpenMP Hamiltonian action.

    SciPy CSR/CSC inputs are canonicalized once.  The extension copies those
    buffers into native-owned storage and also caches a row view of the KB
    projector factor.  Subsequent SCF iterations copy only the length-N local
    effective potential through :meth:`update_local`.
    """

    def __init__(
        self,
        negative_laplacian: sp.spmatrix,
        nonlocal_operator: NonlocalProjectorOperator,
        *,
        requested: str = "native",
        fallback_reasons: tuple[str, ...] = (),
    ) -> None:
        started = perf_counter()
        native = _load_native()
        kinetic = _canonical_csr(negative_laplacian)
        projectors = _canonical_csc(
            nonlocal_operator.projectors,
            kinetic.shape[0],
        )
        signs = _values64(nonlocal_operator.signs)
        if signs.shape != (projectors.shape[1],):
            raise ValueError("one nonlocal sign is required per projector column")

        super().__init__(kinetic.shape)
        self.negative_laplacian = kinetic
        self.projectors = projectors
        self.signs = signs
        self.local_potential = np.zeros(kinetic.shape[0], dtype=np.float64)
        self._native_operator = native.FusedHamiltonian(
            _indices64(kinetic.indptr),
            _indices64(kinetic.indices),
            _values64(kinetic.data),
            _indices64(projectors.indptr),
            _indices64(projectors.indices),
            _values64(projectors.data),
            signs,
            self.local_potential,
        )

        build = dict(native.build_info())
        details = (
            ("laplacian_nnz", str(kinetic.nnz)),
            ("projector_columns", str(projectors.shape[1])),
            ("projector_nnz", str(projectors.nnz)),
            ("openmp_enabled", str(bool(build.get("openmp_enabled", False)))),
            (
                "openmp_detected_processors",
                str(build.get("openmp_detected_processors", 1)),
            ),
            (
                "openmp_reserved_threads",
                str(build.get("openmp_reserved_threads", 0)),
            ),
            (
                "openmp_default_threads",
                str(build.get("openmp_default_threads", 1)),
            ),
            ("openmp_max_threads", str(build.get("openmp_max_threads", 1))),
            (
                "openmp_thread_source",
                str(build.get("openmp_thread_source", "unknown")),
            ),
            ("fixed_summation_order", str(build.get("fixed_summation_order", False))),
        )
        self.info = BackendInfo(
            requested=requested,
            selected="native",
            device="CPU",
            implementation="C++17/OpenMP fused float64 Hamiltonian action",
            fallback_reasons=fallback_reasons,
            details=details,
        )
        self.statistics.initialization_seconds = perf_counter() - started

    def replace_nonlocal_operator(self, nonlocal_operator) -> None:
        """Give the C++ operator the projectors of moved atoms.

        The extension owns copies of its buffers, so a new fused operator is
        made from the retained kinetic CSR, the new projectors and the
        current local potential.
        """

        native = _load_native()
        kinetic = self.negative_laplacian
        projectors = _canonical_csc(nonlocal_operator.projectors, kinetic.shape[0])
        signs = _values64(nonlocal_operator.signs)
        if signs.shape != (projectors.shape[1],):
            raise ValueError("one nonlocal sign is required per projector column")
        self.projectors = projectors
        self.signs = signs
        self._native_operator = native.FusedHamiltonian(
            _indices64(kinetic.indptr),
            _indices64(kinetic.indices),
            _values64(kinetic.data),
            _indices64(projectors.indptr),
            _indices64(projectors.indices),
            _values64(projectors.data),
            signs,
            self.local_potential,
        )

    @staticmethod
    def _restore_rank(result: np.ndarray, was_vector: bool) -> np.ndarray:
        return result[:, 0] if was_vector else result

    def update_local(self, effective_potential: np.ndarray) -> None:
        started = perf_counter()
        potential = _values64(effective_potential)
        if potential.shape != (self.shape[0],):
            raise ValueError("effective potential does not match the Hamiltonian")
        if not np.all(np.isfinite(potential)):
            raise ValueError("effective potential contains nonfinite values")
        self._native_operator.update_local(potential)
        self.local_potential = potential
        self.statistics.local_updates += 1
        self.statistics.local_update_seconds += perf_counter() - started

    def apply_kinetic(self, vectors: np.ndarray) -> np.ndarray:
        block, was_vector = as_vector_block(vectors)
        result = np.asarray(self.negative_laplacian @ block)
        return self._restore_rank(result, was_vector)

    def apply_local(self, vectors: np.ndarray) -> np.ndarray:
        block, was_vector = as_vector_block(vectors)
        result = self.local_potential[:, None] * block
        return self._restore_rank(result, was_vector)

    def apply_nonlocal(self, vectors: np.ndarray) -> np.ndarray:
        block, was_vector = as_vector_block(vectors)
        if self.projectors.shape[1] == 0:
            result = np.zeros_like(block)
        else:
            coefficients = np.asarray(self.projectors.T @ block)
            coefficients *= self.signs[:, None]
            result = np.asarray(self.projectors @ coefficients)
        return self._restore_rank(result, was_vector)

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        started = perf_counter()
        block, was_vector = as_vector_block(vectors)
        # The extension's buffer contract is row-major.  This is normally a
        # no-op for solver blocks and makes any unavoidable host copy visible
        # in the backend's complete apply timing.
        block = np.ascontiguousarray(block, dtype=np.float64)
        result = np.asarray(self._native_operator.apply(block))
        self.statistics.applications += 1
        self.statistics.vectors_applied += int(block.shape[1])
        self.statistics.apply_seconds += perf_counter() - started
        return self._restore_rank(result, was_vector)

    def as_sparse(self) -> sp.csr_matrix:
        nonlocal_matrix = (
            self.projectors
            @ sp.diags(self.signs, format="csc")
            @ self.projectors.T
        )
        return (
            self.negative_laplacian
            + sp.diags(self.local_potential, format="csr")
            + nonlocal_matrix
        ).tocsr()


__all__ = [
    "NativeConjugateGradientBackend",
    "NativeHamiltonianBackend",
    "build_native_negative_laplacian",
    "native_available",
    "native_build_info",
    "native_unavailable_reason",
]
