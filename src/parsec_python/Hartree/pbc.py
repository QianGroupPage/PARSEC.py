"""
Periodic (Gamma-point) Hartree Poisson solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..Grid.pbc import PeriodicRealSpaceGrid
from ..models import HartreeSettings
from .poisson import _conjugate_gradient


@dataclass(frozen=True)
class PeriodicHartreeResult:
    """Periodic Hartree solution and diagnostics.

    Modeled after HartreeResult in poisson.py but with no
    ``boundary`` field since a periodic cell has no exterior to model a boundary
    potential for. (May want to make a base class or move into poisson.py)

    ``potential``
        Active-grid Hartree potential in Rydberg.
    ``right_hand_side``
        ``8*pi*rho`` actually passed to CG.
    ``iterations`` and ``matrix_vector_products``
        CG work counters.  Matrix-vector products include initial/final
        residual evaluations as well as iteration products.
    ``residual_norm`` and ``initial_residual_norm``
        Euclidean norms of ``b-A*V`` before and after the solve.
    """

    potential: np.ndarray
    right_hand_side: np.ndarray
    converged: bool
    iterations: int
    matrix_vector_products: int
    residual_norm: float
    initial_residual_norm: float


def neutralize_density(
    density: np.ndarray, grid: PeriodicRealSpaceGrid
) -> np.ndarray:
    """
    Subtract the cell-averaged density.
    """
    density = np.asarray(density, dtype=np.float64)
    if density.shape != (grid.size,):
        raise ValueError("density does not match the active grid")
    return density - float(np.mean(density))


def solve_periodic_hartree(
    density: np.ndarray,
    grid: PeriodicRealSpaceGrid,
    negative_laplacian: sp.spmatrix,
    settings: HartreeSettings = HartreeSettings(),
    initial_potential: np.ndarray | None = None,
    *,
    raise_on_nonconvergence: bool = True,
) -> PeriodicHartreeResult:
    """Solve the periodic (Gamma-point) Hartree Poisson problem.

    Parameters mirror :func:`.poisson.solve_hartree`, minus the
    boundary-method settings (there is no boundary to construct).
    ``negative_laplacian`` must be built from the same ``grid`` (e.g. via
    ``build_negative_laplacian(grid)``); it is accepted rather than rebuilt
    so a caller solving many SCF iterations reuses one cached operator,
    matching ``solve_hartree``'s own convention.

    The returned potential has zero mean over the cell.  An additive
    constant is not determined by Poisson's equation (``V_H`` and
    ``V_H + c`` source the same charge density), so this fixes a definite
    gauge by construction rather than leaving it to whatever a given warm
    start happens to produce.
    """
    if negative_laplacian.shape != (grid.size, grid.size):
        raise ValueError("negative_laplacian shape does not match the grid")

    neutral_density = neutralize_density(density, grid)
    # Rydberg-unit Poisson source: -nabla**2 V_H = 8*pi*rho, same convention
    # as poisson.solve_hartree; there is no boundary term to add here.
    rhs = 8.0 * np.pi * neutral_density

    if initial_potential is None:
        initial = np.zeros(grid.size, dtype=np.float64)
    else:
        initial = np.asarray(initial_potential, dtype=np.float64)
        if initial.shape != (grid.size,):
            raise ValueError("initial Hartree potential does not match the grid")
        # Any mean component of a supplied warm start lies in A's null space
        # and would pass through CG completely unchanged, silently shifting
        # the result's gauge -- removed here instead.
        initial = initial - float(np.mean(initial))

    potential, converged, iterations, matvecs, residual, initial_residual = (
        _conjugate_gradient(negative_laplacian, rhs, initial, settings)
    )
    potential = potential - float(np.mean(potential))
    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            "periodic Hartree conjugate-gradient solve did not converge: "
            f"residual={residual:.3e}, matvecs={matvecs}"
        )
    return PeriodicHartreeResult(
        potential=potential,
        right_hand_side=rhs,
        converged=converged,
        iterations=iterations,
        matrix_vector_products=matvecs,
        residual_norm=residual,
        initial_residual_norm=initial_residual,
    )


__all__ = ["PeriodicHartreeResult", "neutralize_density", "solve_periodic_hartree"]
