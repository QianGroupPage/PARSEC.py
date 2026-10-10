"""Optional float64 CuPy solver for the isolated Hartree Poisson problem.

The physical boundary construction deliberately remains on the host and
reuses the validated reference helpers.  Only the repeated conjugate-gradient
linear algebra runs on CUDA.  A :class:`CuPyPoissonSolver` is constructed from
the same :class:`~parsec_python.acceleration.backends.cupy.CuPyHamiltonian` used
by the GPU Kohn--Sham eigensolver, so its already-uploaded negative-Laplacian
CSR allocation is reused rather than copied to the device a second time.

CuPy is an optional dependency.  Importing this module never imports CuPy;
the runtime is requested only when a solver object is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from time import perf_counter
from typing import Any

import numpy as np

from parsec_python.Grid import RealSpaceGrid
from parsec_python.Hartree.poisson import (
    DirectCoulombBoundary,
    HartreeResult,
    MultipoleExpansion,
)
from parsec_python.models import HartreeSettings

from ..backends.cupy import (
    CuPyHamiltonian,
    require_cupy,
    synchronized_call,
)
from .poisson import build_hartree_problem


HartreeBoundary = MultipoleExpansion | DirectCoulombBoundary


@dataclass
class CuPyPoissonTimings:
    """Coarse wall timings for one or more GPU Poisson solves.

    ``rhs_seconds`` measures the host multipole/direct boundary and
    boundary-corrected right-hand-side construction.  ``upload_seconds``,
    ``solve_seconds``, and ``download_seconds`` are explicitly synchronized
    coarse stages.  The CG implementation does *not* insert timing events
    around individual sparse matrix-vector products; its scalar reductions
    already impose the synchronization required by the algorithm.

    :attr:`CuPyPoissonSolver.last_timings` contains one call, while
    :attr:`CuPyPoissonSolver.timings` accumulates all calls made by that
    reusable solver.
    """

    solve_calls: int = 0
    total_seconds: float = 0.0
    rhs_seconds: float = 0.0
    upload_seconds: float = 0.0
    solve_seconds: float = 0.0
    download_seconds: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        """Return a serializable timing snapshot."""

        return {item.name: getattr(self, item.name) for item in fields(self)}

    def snapshot(self) -> "CuPyPoissonTimings":
        """Return an independent copy of the current counters."""

        return CuPyPoissonTimings(
            solve_calls=int(self.solve_calls),
            total_seconds=float(self.total_seconds),
            rhs_seconds=float(self.rhs_seconds),
            upload_seconds=float(self.upload_seconds),
            solve_seconds=float(self.solve_seconds),
            download_seconds=float(self.download_seconds),
        )

    def add(self, other: "CuPyPoissonTimings") -> None:
        """Accumulate one timing record without changing ``other``."""

        self.solve_calls += int(other.solve_calls)
        self.total_seconds += float(other.total_seconds)
        self.rhs_seconds += float(other.rhs_seconds)
        self.upload_seconds += float(other.upload_seconds)
        self.solve_seconds += float(other.solve_seconds)
        self.download_seconds += float(other.download_seconds)


def build_boundary_corrected_rhs(
    density: np.ndarray,
    grid: RealSpaceGrid,
    settings: HartreeSettings = HartreeSettings(),
    boundary_tail=None,
) -> tuple[np.ndarray, HartreeBoundary]:
    """Build the exact reference Rydberg Poisson RHS on the host.

    The returned vector is

    ``b_eff = 8*pi*rho_I - A_IB*V_B``.

    ``auto`` chooses the convergent multipole boundary for an origin-centered
    sphere and the direct discrete Coulomb boundary for a box, exactly as
    :func:`parsec_python.Hartree.solve_hartree` does.  Keeping this stage on
    the host avoids duplicating the spherical-harmonic and irregular-boundary
    stencil logic in CUDA code.
    """

    return build_hartree_problem(density, grid, settings, boundary_tail)


def _device_scalar(value: Any) -> float:
    """Transfer one reduction scalar to the host.

    CG needs these values for stopping and recurrence coefficients, so their
    synchronization is algorithmic rather than profiling overhead.
    """

    item = getattr(value, "item", None)
    return float(item() if item is not None else value)


def _device_conjugate_gradient(
    operator: Any,
    rhs: Any,
    initial: Any,
    settings: HartreeSettings,
    cp: Any,
) -> tuple[Any, bool, int, int, float, float]:
    """Mirror the reference unpreconditioned CG recurrence on CUDA.

    The matrix-vector-product budget, warm-start-scaled tolerance, breakdown
    rule, early-exit behavior, and final true-residual recomputation match
    ``parsec_python.Hartree.poisson._conjugate_gradient`` exactly.
    """

    solution = initial
    residual = rhs - operator @ solution
    matrix_vector_products = 1
    initial_norm = _device_scalar(cp.linalg.norm(residual))
    tolerance = (
        settings.relative_tolerance * initial_norm
        + settings.absolute_tolerance
    )
    if initial_norm <= tolerance:
        return (
            solution,
            True,
            0,
            matrix_vector_products,
            initial_norm,
            initial_norm,
        )

    direction = residual.copy()
    residual_squared = _device_scalar(cp.vdot(residual, residual))
    iterations = 0
    converged = False
    while matrix_vector_products < settings.max_iterations:
        operator_direction = operator @ direction
        matrix_vector_products += 1
        denominator = _device_scalar(cp.vdot(direction, operator_direction))
        if denominator <= 0.0 or not np.isfinite(denominator):
            break

        alpha = residual_squared / denominator
        solution += alpha * direction
        residual -= alpha * operator_direction
        iterations += 1
        residual_norm = _device_scalar(cp.linalg.norm(residual))
        if residual_norm <= tolerance:
            converged = True
            break
        new_residual_squared = _device_scalar(cp.vdot(residual, residual))
        beta = new_residual_squared / residual_squared
        direction = residual + beta * direction
        residual_squared = new_residual_squared

    # As in the reference solver, this reported residual is recomputed from
    # b-A*x.  It does not retroactively alter ``converged``, which was decided
    # using the recursively updated residual above.
    residual_norm = _device_scalar(cp.linalg.norm(rhs - operator @ solution))
    matrix_vector_products += 1
    return (
        solution,
        converged,
        iterations,
        matrix_vector_products,
        residual_norm,
        initial_norm,
    )


class CuPyPoissonSolver:
    """Reusable device CG solver sharing a CuPy Hamiltonian's Laplacian.

    Parameters
    ----------
    hamiltonian
        The persistent GPU Hamiltonian used by the accelerated eigensolver.
        Its ``negative_laplacian`` CSR object is retained by identity; no
        second sparse upload or materialized Poisson matrix is created.

    Notes
    -----
    Reuse one instance across SCF iterations and pass the preceding Hartree
    potential through ``initial_potential``.  The returned object is the same
    :class:`parsec_python.Hartree.HartreeResult` used by the reference SCF
    path, and all large arrays in that result are ordinary host NumPy arrays.
    """

    def __init__(self, hamiltonian: CuPyHamiltonian) -> None:
        if not isinstance(hamiltonian, CuPyHamiltonian):
            raise TypeError("hamiltonian must be a CuPyHamiltonian")
        cp, _ = require_cupy()
        self._cupy = cp
        self.hamiltonian = hamiltonian
        # Critical integration invariant: share the existing device CSR.
        self.negative_laplacian = hamiltonian.negative_laplacian
        self.shape = tuple(int(value) for value in hamiltonian.shape)
        self.timings = CuPyPoissonTimings()
        self.last_timings = CuPyPoissonTimings()

    def solve(
        self,
        density: np.ndarray,
        grid: RealSpaceGrid,
        settings: HartreeSettings = HartreeSettings(),
        initial_potential: np.ndarray | None = None,
        *,
        raise_on_nonconvergence: bool = True,
        boundary_tail=None,
    ) -> HartreeResult:
        """Construct the host boundary problem and solve it on the GPU."""

        total_started = perf_counter()
        density = np.asarray(density, dtype=float)
        if density.shape != (grid.size,):
            raise ValueError("density does not match the active grid")
        if self.shape != (grid.size, grid.size):
            raise ValueError("negative_laplacian shape does not match the grid")

        rhs_started = perf_counter()
        rhs, boundary = build_boundary_corrected_rhs(
            density, grid, settings, boundary_tail
        )
        rhs_seconds = perf_counter() - rhs_started

        if initial_potential is None:
            initial = np.zeros(grid.size, dtype=float)
        else:
            initial = np.asarray(initial_potential, dtype=float)
            if initial.shape != (grid.size,):
                raise ValueError("initial Hartree potential does not match the grid")

        cp = self._cupy

        def upload() -> tuple[Any, Any]:
            device_rhs = cp.asarray(rhs, dtype=cp.float64)
            # The recurrence updates x in place, so it must own its buffer.
            device_initial = cp.array(initial, dtype=cp.float64, copy=True)
            return device_rhs, device_initial

        (device_rhs, device_initial), upload_seconds = synchronized_call(upload)
        device_result, solve_seconds = synchronized_call(
            _device_conjugate_gradient,
            self.negative_laplacian,
            device_rhs,
            device_initial,
            settings,
            cp,
        )
        (
            device_potential,
            converged,
            iterations,
            matrix_vector_products,
            residual_norm,
            initial_residual_norm,
        ) = device_result
        host_potential, download_seconds = synchronized_call(
            cp.asnumpy,
            device_potential,
        )

        call_timings = CuPyPoissonTimings(
            solve_calls=1,
            total_seconds=perf_counter() - total_started,
            rhs_seconds=rhs_seconds,
            upload_seconds=upload_seconds,
            solve_seconds=solve_seconds,
            download_seconds=download_seconds,
        )
        self.last_timings = call_timings
        self.timings.add(call_timings)

        if not converged and raise_on_nonconvergence:
            raise RuntimeError(
                "Hartree conjugate-gradient solve did not converge: "
                f"residual={residual_norm:.3e}, "
                f"matvecs={matrix_vector_products}"
            )
        return HartreeResult(
            potential=np.asarray(host_potential, dtype=float),
            right_hand_side=rhs,
            boundary=boundary,
            converged=converged,
            iterations=iterations,
            matrix_vector_products=matrix_vector_products,
            residual_norm=residual_norm,
            initial_residual_norm=initial_residual_norm,
            boundary_tail=boundary_tail,
        )


def solve_hartree_cupy(
    density: np.ndarray,
    grid: RealSpaceGrid,
    hamiltonian: CuPyHamiltonian,
    settings: HartreeSettings = HartreeSettings(),
    initial_potential: np.ndarray | None = None,
    *,
    raise_on_nonconvergence: bool = True,
    boundary_tail=None,
) -> HartreeResult:
    """One-shot convenience wrapper around :class:`CuPyPoissonSolver`.

    Repeated SCF calls should retain a ``CuPyPoissonSolver`` instance instead,
    both to accumulate timings and to make the shared device allocation
    explicit.
    """

    return CuPyPoissonSolver(hamiltonian).solve(
        density,
        grid,
        settings,
        initial_potential,
        raise_on_nonconvergence=raise_on_nonconvergence,
        boundary_tail=boundary_tail,
    )


__all__ = [
    "CuPyPoissonSolver",
    "CuPyPoissonTimings",
    "build_boundary_corrected_rhs",
    "solve_hartree_cupy",
]
