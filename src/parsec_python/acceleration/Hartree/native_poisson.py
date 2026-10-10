"""Native C++/OpenMP solve for a prepared finite-domain Poisson system.

Boundary construction remains a separate, inspectable operation in the
validated implementation.  This module starts at the linear system

``A_II V_I = b_eff``,  where ``b_eff = 8*pi*rho_I - A_IB V_B``,

and accelerates only the repeated conjugate-gradient algebra.  Keeping this
boundary-corrected RHS interface makes the solver useful outside a complete
SCF calculation and avoids duplicating the physical boundary model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os

import numpy as np
import scipy.sparse as sp

from parsec_python.Hartree.poisson import (
    DirectCoulombBoundary,
    HartreeResult,
    MultipoleExpansion,
)
from parsec_python.models import HartreeSettings

from ..backends.native import NativeConjugateGradientBackend


@dataclass(frozen=True)
class NativePoissonResult:
    """Native solution plus reference-compatible CG diagnostics.

    ``potential``, ``right_hand_side``, ``converged``, ``iterations``,
    ``matrix_vector_products``, ``residual_norm``, and
    ``initial_residual_norm`` correspond directly to :class:`HartreeResult`.
    ``tolerance`` records ``rtol*||r0|| + atol`` and ``breakdown`` distinguishes
    an invalid/non-SPD direction from ordinary matrix-vector budget exhaustion.

    The boundary is intentionally not invented here because this solver
    accepts an already corrected RHS.  Use :meth:`as_hartree_result` when the
    boundary object is available from the physical setup stage.
    """

    potential: np.ndarray
    right_hand_side: np.ndarray
    converged: bool
    iterations: int
    matrix_vector_products: int
    residual_norm: float
    initial_residual_norm: float
    tolerance: float
    breakdown: bool

    def as_hartree_result(
        self,
        boundary: MultipoleExpansion | DirectCoulombBoundary,
    ) -> HartreeResult:
        """Attach the separately constructed boundary model."""
        return HartreeResult(
            potential=self.potential,
            right_hand_side=self.right_hand_side,
            boundary=boundary,
            converged=self.converged,
            iterations=self.iterations,
            matrix_vector_products=self.matrix_vector_products,
            residual_norm=self.residual_norm,
            initial_residual_norm=self.initial_residual_norm,
        )


class NativePoissonSolver:
    """Reusable native CG solver for one static negative Laplacian.

    Parameters
    ----------
    negative_laplacian
        Active-to-active finite-difference matrix ``A_II = -nabla_h**2``.
        It is converted to canonical float64 CSR and copied into C++ storage
        once, so subsequent density/SCF solves transfer vectors only.
    """

    def __init__(self, negative_laplacian: sp.spmatrix, *, backend_factory=None) -> None:
        # The prepared GPU solver reuses this exact warm-start policy and
        # result contract; its optional factory changes only linear algebra.
        factory = NativeConjugateGradientBackend if backend_factory is None else backend_factory
        self.backend = factory(negative_laplacian)
        self.shape = self.backend.shape
        self.storage_mode = self.backend.storage_mode
        self.worker_count = self.backend.worker_count
        self.coefficient_palette_size = self.backend.coefficient_palette_size
        self._previous_rhs: np.ndarray | None = None
        self._previous_previous_rhs: np.ndarray | None = None
        self._previous_solution: np.ndarray | None = None
        self._previous_previous_solution: np.ndarray | None = None
        self.chronological_prediction_calls = 0
        self.last_chronological_alpha: float | None = None

    @property
    def negative_laplacian(self) -> sp.csr_matrix:
        """Canonical CSR operator of the backend.

        Read on demand: a prepared GPU backend that was handed a packed
        stencil forms this matrix only when it is first asked for.
        """
        return self.backend.operator

    def solve(
        self,
        boundary_corrected_right_hand_side: np.ndarray,
        initial_potential: np.ndarray | None = None,
        settings: HartreeSettings | None = None,
        *,
        relative_tolerance: float | None = None,
        absolute_tolerance: float | None = None,
        max_iterations: int | None = None,
        raise_on_nonconvergence: bool = True,
    ) -> NativePoissonResult:
        """Solve the prepared Poisson system with a zero or warm start.

        A supplied :class:`HartreeSettings` provides the CG controls.  Any of
        the three scalar keyword controls overrides its corresponding value,
        which is convenient for focused solver benchmarks.
        """
        base = HartreeSettings() if settings is None else settings
        if not isinstance(base, HartreeSettings):
            raise TypeError("settings must be a HartreeSettings instance")
        controls = replace(
            base,
            relative_tolerance=(
                base.relative_tolerance
                if relative_tolerance is None
                else relative_tolerance
            ),
            absolute_tolerance=(
                base.absolute_tolerance
                if absolute_tolerance is None
                else absolute_tolerance
            ),
            max_iterations=(
                base.max_iterations if max_iterations is None else max_iterations
            ),
        )

        rhs = np.ascontiguousarray(
            boundary_corrected_right_hand_side, dtype=np.float64
        )
        expected = (self.shape[0],)
        if rhs.shape != expected:
            raise ValueError(
                "boundary-corrected right_hand_side does not match the operator"
            )
        if initial_potential is None:
            initial = np.zeros(self.shape[0], dtype=np.float64)
        else:
            initial = np.ascontiguousarray(initial_potential, dtype=np.float64)
            if initial.shape != expected:
                raise ValueError("initial Hartree potential does not match the operator")

        chronological = os.environ.get(
            "PARSEC_HARTREE_CHRONOLOGICAL_GUESS", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.last_chronological_alpha = None
        if (
            chronological
            and self._previous_rhs is not None
            and self._previous_previous_rhs is not None
            and self._previous_solution is not None
            and self._previous_previous_solution is not None
        ):
            rhs_direction = self._previous_rhs - self._previous_previous_rhs
            denominator = float(np.dot(rhs_direction, rhs_direction))
            if np.isfinite(denominator) and denominator > 0.0:
                alpha = float(
                    np.dot(rhs_direction, rhs - self._previous_rhs)
                    / denominator
                )
                # A distant extrapolation is more likely to amplify nonlinear
                # SCF transients than to improve the linear starting vector.
                # The interval still permits damping and a full secant step.
                alpha = float(np.clip(alpha, -0.5, 1.5))
                predicted = self._previous_solution + alpha * (
                    self._previous_solution
                    - self._previous_previous_solution
                )
                if np.all(np.isfinite(predicted)):
                    initial = np.ascontiguousarray(predicted, dtype=np.float64)
                    self.chronological_prediction_calls += 1
                    self.last_chronological_alpha = alpha

        payload = self.backend.solve(
            rhs,
            initial,
            relative_tolerance=controls.relative_tolerance,
            absolute_tolerance=controls.absolute_tolerance,
            max_iterations=controls.max_iterations,
        )
        result = NativePoissonResult(
            potential=np.asarray(payload["solution"], dtype=np.float64),
            right_hand_side=rhs,
            converged=bool(payload["converged"]),
            iterations=int(payload["iterations"]),
            matrix_vector_products=int(payload["matrix_vector_products"]),
            residual_norm=float(payload["residual_norm"]),
            initial_residual_norm=float(payload["initial_residual_norm"]),
            tolerance=float(payload["tolerance"]),
            breakdown=bool(payload["breakdown"]),
        )
        self._previous_previous_rhs = self._previous_rhs
        self._previous_rhs = rhs.copy()
        self._previous_previous_solution = self._previous_solution
        self._previous_solution = result.potential.copy()
        if not result.converged and raise_on_nonconvergence:
            raise RuntimeError(
                "Hartree conjugate-gradient solve did not converge: "
                f"residual={result.residual_norm:.3e}, "
                f"matvecs={result.matrix_vector_products}"
            )
        return result


def solve_native_poisson(
    negative_laplacian: sp.spmatrix,
    boundary_corrected_right_hand_side: np.ndarray,
    initial_potential: np.ndarray | None = None,
    settings: HartreeSettings | None = None,
    *,
    relative_tolerance: float | None = None,
    absolute_tolerance: float | None = None,
    max_iterations: int | None = None,
    raise_on_nonconvergence: bool = True,
) -> NativePoissonResult:
    """One-shot convenience wrapper around :class:`NativePoissonSolver`."""
    return NativePoissonSolver(negative_laplacian).solve(
        boundary_corrected_right_hand_side,
        initial_potential,
        settings,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        max_iterations=max_iterations,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )


__all__ = [
    "NativePoissonResult",
    "NativePoissonSolver",
    "solve_native_poisson",
]
