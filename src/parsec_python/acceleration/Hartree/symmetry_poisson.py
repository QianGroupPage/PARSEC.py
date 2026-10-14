"""Totally symmetric wedge adapter for the native Hartree solver."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from parsec_python.models import HartreeSettings

from ..Symmetry import AxisReflectionReduction
from ..SCF.symmetry_fields import SymmetryScalarField
from ..backends.cupy_stencil_major import StencilMajorHostMetadata
from .native_poisson import NativePoissonResult, NativePoissonSolver


class SymmetryReducedPoissonSolver:
    """Solve a full-grid invariant Poisson problem on a normalized wedge.

    ``solve`` accepts a full RHS for compatibility and projects it.  Native
    symmetry-aware multipoles use ``solve_reduced`` to pass ``U.T b``
    directly, so neither RHS construction nor repeated CG needs a full-grid
    work vector.
    """

    def __init__(
        self,
        negative_laplacian: sp.spmatrix,
        reduction: AxisReflectionReduction,
        *,
        operator_is_reduced: bool = False,
        solver_factory=None,
    ) -> None:
        self.reduction = reduction
        if operator_is_reduced and isinstance(
            negative_laplacian, StencilMajorHostMetadata
        ):
            # The packed stencil of the totally symmetric orbital sector goes
            # to the solver as it is.  ``solver_factory`` must accept it; the
            # CSR form below is then formed only if it is asked for.
            if tuple(negative_laplacian.shape) != (
                reduction.wedge_size,
                reduction.wedge_size,
            ):
                raise ValueError(
                    "pre-reduced Hartree operator does not match the symmetry wedge"
                )
            reduced = negative_laplacian
            self._reduced_negative_laplacian = None
        elif operator_is_reduced:
            reduced = sp.csr_matrix(negative_laplacian, dtype=np.float64)
            if reduced.shape != (reduction.wedge_size, reduction.wedge_size):
                raise ValueError(
                    "pre-reduced Hartree operator does not match the symmetry wedge"
                )
            reduced.sum_duplicates()
            reduced.sort_indices()
            self._reduced_negative_laplacian = reduced
        else:
            reduced = reduction.reduce_operator(negative_laplacian)
            self._reduced_negative_laplacian = reduced
        factory = NativePoissonSolver if solver_factory is None else solver_factory
        self.solver = factory(reduced)
        self.shape = self.solver.shape
        self.storage_mode = self.solver.storage_mode
        self.worker_count = self.solver.worker_count
        self.coefficient_palette_size = self.solver.coefficient_palette_size

    @property
    def reduced_negative_laplacian(self) -> sp.csr_matrix:
        """Reduced CSR operator this adapter was built from.

        For a packed stencil it is the solver's matrix, formed on first use.
        """

        if self._reduced_negative_laplacian is None:
            self._reduced_negative_laplacian = self.solver.negative_laplacian
        return self._reduced_negative_laplacian

    @property
    def negative_laplacian(self) -> sp.csr_matrix:
        """Canonical reduced CSR operator held by the wrapped solver."""

        return self.solver.negative_laplacian

    def solve(
        self,
        full_right_hand_side: np.ndarray,
        initial_potential: np.ndarray | None = None,
        settings: HartreeSettings | None = None,
        **kwargs,
    ) -> NativePoissonResult:
        """Project the RHS/start, solve the wedge, and expand the potential."""

        rhs = np.asarray(full_right_hand_side, dtype=np.float64)
        if rhs.shape != (self.reduction.full_size,):
            raise ValueError("Hartree right-hand side does not match the full grid")
        reduced_rhs = self.reduction.reduce_vector(rhs)
        return self.solve_reduced(
            reduced_rhs,
            initial_potential,
            settings,
            **kwargs,
        )

    def solve_reduced(
        self,
        reduced_right_hand_side: np.ndarray,
        initial_potential: np.ndarray | None = None,
        settings: HartreeSettings | None = None,
        return_wedge: bool = False,
        **kwargs,
    ) -> NativePoissonResult:
        """Solve a preprojected normalized RHS and expand only the result."""

        reduced_rhs = np.asarray(reduced_right_hand_side, dtype=np.float64)
        if reduced_rhs.shape != (self.reduction.wedge_size,):
            raise ValueError("reduced Hartree RHS does not match the symmetry wedge")
        if initial_potential is None:
            reduced_initial = None
        elif isinstance(initial_potential, SymmetryScalarField):
            if initial_potential.reduction is self.reduction:
                reduced_initial = (
                    np.sqrt(self.reduction.multiplicities)
                    * initial_potential.values
                )
            else:
                full_initial = np.ascontiguousarray(
                    initial_potential.values[
                        initial_potential.reduction.full_to_wedge
                    ]
                )
                reduced_initial = self.reduction.reduce_vector(full_initial)
        else:
            reduced_initial = self.reduction.reduce_vector(initial_potential)
        reduced = self.solver.solve(
            reduced_rhs,
            reduced_initial,
            settings,
            **kwargs,
        )
        if return_wedge:
            roots = np.sqrt(self.reduction.multiplicities)
            potential = SymmetryScalarField(
                self.reduction, reduced.potential / roots
            )
            right_hand_side = SymmetryScalarField(
                self.reduction, reduced.right_hand_side / roots
            )
        else:
            potential = self.reduction.expand_vector(reduced.potential)
            right_hand_side = self.reduction.expand_vector(
                reduced.right_hand_side
            )
        return NativePoissonResult(
            potential=potential,
            right_hand_side=right_hand_side,
            converged=reduced.converged,
            iterations=reduced.iterations,
            matrix_vector_products=reduced.matrix_vector_products,
            residual_norm=reduced.residual_norm,
            initial_residual_norm=reduced.initial_residual_norm,
            tolerance=reduced.tolerance,
            breakdown=reduced.breakdown,
        )


__all__ = ["SymmetryReducedPoissonSolver"]
