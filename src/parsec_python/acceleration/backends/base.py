"""Common host-facing interface for accelerated Hamiltonian backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator

from ..models import BackendInfo, BackendStatistics


def as_vector_block(vectors: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a finite float64 ``(n, block)`` array and its original rank."""
    array = np.asarray(vectors, dtype=np.float64)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("Hamiltonian input must be a vector or two-dimensional block")
    if not np.all(np.isfinite(array)):
        raise ValueError("Hamiltonian input contains nonfinite values")
    return array, was_vector


class HamiltonianBackend(ABC):
    """Mutable cached backend used by successive SCF Hamiltonians.

    Only the diagonal local field changes between nonlinear iterations.  The
    kinetic CSR matrix and Kleinman--Bylander projector factors are uploaded
    or copied once by a concrete backend.
    """

    info: BackendInfo
    statistics: BackendStatistics

    def __init__(self, shape: tuple[int, int]) -> None:
        self._shape = tuple(int(value) for value in shape)
        self.statistics = BackendStatistics()

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @abstractmethod
    def update_local(self, effective_potential: np.ndarray) -> None:
        """Replace the current diagonal local potential."""

    def replace_nonlocal_operator(self, nonlocal_operator) -> None:
        """Replace the Kleinman--Bylander projector factors of moved atoms.

        The kinetic operator and the grid are unchanged, so everything else
        the backend cached stays.  A backend that cannot do this says so.
        """

        raise NotImplementedError(
            f"{type(self).__name__} cannot replace its projector factors"
        )

    @abstractmethod
    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the complete Hamiltonian to a host vector/block."""

    @abstractmethod
    def apply_kinetic(self, vectors: np.ndarray) -> np.ndarray:
        """Apply only the finite-difference kinetic term."""

    @abstractmethod
    def apply_local(self, vectors: np.ndarray) -> np.ndarray:
        """Apply only the current diagonal local term."""

    @abstractmethod
    def apply_nonlocal(self, vectors: np.ndarray) -> np.ndarray:
        """Apply only the low-rank nonlocal ionic term."""

    @abstractmethod
    def as_sparse(self) -> sp.csr_matrix:
        """Materialize a diagnostic sparse Hamiltonian for small problems."""

    def synchronize(self) -> None:
        """Wait for asynchronous work; CPU backends have nothing to do."""

    def bind(self, effective_potential: np.ndarray) -> "BoundHamiltonian":
        return BoundHamiltonian(self, effective_potential)

    def profile_components(self, vectors: np.ndarray) -> dict[str, float]:
        """Time one representative component application outside production.

        The optional benchmark intentionally synchronizes between terms.  It
        must not be used inside every Chebyshev recurrence because that would
        distort the workload being measured.
        """
        block, _ = as_vector_block(vectors)
        timings: dict[str, float] = {}
        for name, operation in (
            ("finite_difference_apply", self.apply_kinetic),
            ("local_potential_apply", self.apply_local),
            ("nonlocal_potential_apply", self.apply_nonlocal),
        ):
            self.synchronize()
            started = perf_counter()
            operation(block)
            self.synchronize()
            timings[name] = perf_counter() - started
        self.statistics.component_profile_seconds = dict(timings)
        return timings


class BoundHamiltonian:
    """Lightweight matrix-free Hamiltonian bound to one SCF local field."""

    def __init__(
        self,
        backend: HamiltonianBackend,
        effective_potential: np.ndarray,
    ) -> None:
        self.backend = backend
        self.backend.update_local(effective_potential)

    @property
    def shape(self) -> tuple[int, int]:
        return self.backend.shape

    def apply_kinetic(self, vectors: np.ndarray) -> np.ndarray:
        return self.backend.apply_kinetic(vectors)

    def apply_local(self, vectors: np.ndarray) -> np.ndarray:
        return self.backend.apply_local(vectors)

    def apply_nonlocal(self, vectors: np.ndarray) -> np.ndarray:
        return self.backend.apply_nonlocal(vectors)

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        return self.backend.apply(vectors)

    def as_linear_operator(self) -> "LinearOperator":
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(
            self.shape,
            matvec=self.apply,
            matmat=self.apply,
            rmatvec=self.apply,
            dtype=np.float64,
        )

    def as_eigensolver_operator(self):
        """Return the operator consumed by the selected eigensolver.

        CPU backends use the ordinary SciPy ``LinearOperator``.  The CuPy
        bound object overrides this method and returns its persistent device
        operator, avoiding host transfers inside Chebyshev recurrences.
        """

        return self.as_linear_operator()

    def as_sparse(self) -> sp.csr_matrix:
        return self.backend.as_sparse()

    def __matmul__(self, vectors: Any) -> np.ndarray:
        return self.apply(vectors)


__all__ = [
    "BoundHamiltonian",
    "HamiltonianBackend",
    "as_vector_block",
]
