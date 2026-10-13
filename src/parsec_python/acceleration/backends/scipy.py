"""Allocation-reduced SciPy backend for the accelerated package."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.sparse as sp

from parsec_python.V_ion import NonlocalProjectorOperator

from .base import HamiltonianBackend, as_vector_block
from ..models import BackendInfo


class ScipyHamiltonianBackend(HamiltonianBackend):
    """Reference-compatible CPU backend with an in-place fused sum.

    SciPy already executes CSR/CSC products in compiled code.  This backend
    removes the reference operator's extra term arrays and additions while
    leaving every physical component and eigensolver decision unchanged.
    """

    def __init__(
        self,
        negative_laplacian: sp.spmatrix,
        nonlocal_operator: NonlocalProjectorOperator,
        *,
        requested: str = "scipy",
        fallback_reasons: tuple[str, ...] = (),
    ) -> None:
        started = perf_counter()
        kinetic = sp.csr_matrix(negative_laplacian, dtype=np.float64)
        kinetic.sort_indices()
        projectors = sp.csc_matrix(
            nonlocal_operator.projectors, dtype=np.float64
        )
        projectors.sort_indices()
        super().__init__(kinetic.shape)
        self.negative_laplacian = kinetic
        self.projectors = projectors
        self.signs = np.asarray(nonlocal_operator.signs, dtype=np.float64)
        self.local_potential = np.zeros(kinetic.shape[0], dtype=np.float64)
        self.info = BackendInfo(
            requested=requested,
            selected="scipy",
            device="CPU",
            implementation="SciPy CSR plus allocation-reduced fused host action",
            fallback_reasons=fallback_reasons,
            details=(
                ("laplacian_nnz", str(kinetic.nnz)),
                ("projector_columns", str(projectors.shape[1])),
            ),
        )
        self.statistics.initialization_seconds = perf_counter() - started

    def update_local(self, effective_potential: np.ndarray) -> None:
        started = perf_counter()
        potential = np.asarray(effective_potential, dtype=np.float64)
        if potential.shape != (self.shape[0],):
            raise ValueError("effective potential does not match the Hamiltonian")
        if not np.all(np.isfinite(potential)):
            raise ValueError("effective potential contains nonfinite values")
        self.local_potential = potential
        self.statistics.local_updates += 1
        self.statistics.local_update_seconds += perf_counter() - started

    def replace_nonlocal_operator(self, nonlocal_operator) -> None:
        projectors = sp.csc_matrix(nonlocal_operator.projectors, dtype=np.float64)
        projectors.sort_indices()
        if projectors.shape[0] != self.shape[0]:
            raise ValueError("projector row count must match the Hamiltonian")
        self.projectors = projectors
        self.signs = np.asarray(nonlocal_operator.signs, dtype=np.float64)

    @staticmethod
    def _restore_rank(result: np.ndarray, was_vector: bool) -> np.ndarray:
        return result[:, 0] if was_vector else result

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

        # Start with the dominant CSR SpMM result, then accumulate the local
        # and low-rank terms in place.  This avoids three N-by-block temporary
        # arrays and the two extra addition outputs in the reference operator.
        result = np.asarray(self.negative_laplacian @ block)
        result += self.local_potential[:, None] * block
        if self.projectors.shape[1]:
            coefficients = np.asarray(self.projectors.T @ block)
            coefficients *= self.signs[:, None]
            result += np.asarray(self.projectors @ coefficients)

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


__all__ = ["ScipyHamiltonianBackend"]
