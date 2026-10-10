"""Fast-boundary SciPy Hartree path shared by accelerated CPU backends."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import RealSpaceGrid
from parsec_python.Hartree import (
    DirectCoulombBoundary,
    HartreeResult,
    MultipoleExpansion,
)
from parsec_python.Hartree.poisson import (
    _check_boundary_tail,
    _conjugate_gradient,
)
from parsec_python.Laplacian import apply_negative_laplacian_boundary
from parsec_python.models import HartreeSettings

from .fast_multipole import density_multipoles_fast


def build_hartree_problem(
    density: np.ndarray,
    grid: RealSpaceGrid,
    settings: HartreeSettings,
    boundary_tail=None,
) -> tuple[np.ndarray, MultipoleExpansion | DirectCoulombBoundary]:
    """Return the identical isolated boundary-corrected Poisson right side.

    ``boundary_tail`` is the static atomic tail of the multipole boundary
    values, as in :func:`parsec_python.Hartree.solve_hartree`.
    """

    density = np.asarray(density, dtype=np.float64)
    if density.shape != (grid.size,):
        raise ValueError("density does not match the active grid")
    method = settings.boundary_method
    if method == "auto":
        method = "multipole" if grid.settings.domain_shape == "sphere" else "direct"
    if method == "multipole":
        if grid.settings.domain_shape != "sphere":
            raise ValueError(
                "an origin-centered multipole boundary is not convergent at "
                "general box faces; use boundary_method='direct' or 'auto'"
            )
        boundary: MultipoleExpansion | DirectCoulombBoundary = (
            density_multipoles_fast(density, grid, settings.multipole_order)
        )
    else:
        boundary = DirectCoulombBoundary(
            source_coordinates=grid.coordinates,
            source_weights=density * grid.volume_element,
            chunk_size=settings.direct_chunk_size,
        )
    source = 8.0 * np.pi * density
    right_hand_side = apply_negative_laplacian_boundary(
        source, grid, boundary.potential
    )
    if boundary_tail is not None:
        _check_boundary_tail(boundary_tail, method, settings)
        right_hand_side[boundary_tail.rows] += boundary_tail.values
    return right_hand_side, boundary


def solve_scipy_hartree(
    density: np.ndarray,
    grid: RealSpaceGrid,
    negative_laplacian: sp.spmatrix,
    settings: HartreeSettings = HartreeSettings(),
    initial_potential: np.ndarray | None = None,
    *,
    raise_on_nonconvergence: bool = True,
    boundary_tail=None,
) -> HartreeResult:
    """Solve Hartree with fast multipoles and reference-equivalent host CG."""

    if negative_laplacian.shape != (grid.size, grid.size):
        raise ValueError("negative_laplacian shape does not match the grid")
    right_hand_side, boundary = build_hartree_problem(
        density, grid, settings, boundary_tail
    )
    if initial_potential is None:
        initial = np.zeros(grid.size, dtype=np.float64)
    else:
        initial = np.asarray(initial_potential, dtype=np.float64)
        if initial.shape != (grid.size,):
            raise ValueError("initial Hartree potential does not match the grid")

    potential, converged, iterations, matvecs, residual, initial_residual = (
        _conjugate_gradient(
            negative_laplacian,
            right_hand_side,
            initial,
            settings,
        )
    )
    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            "Hartree conjugate-gradient solve did not converge: "
            f"residual={residual:.3e}, matvecs={matvecs}"
        )
    return HartreeResult(
        potential=potential,
        right_hand_side=right_hand_side,
        boundary=boundary,
        converged=converged,
        iterations=iterations,
        matrix_vector_products=matvecs,
        residual_norm=residual,
        initial_residual_norm=initial_residual,
        boundary_tail=boundary_tail,
    )


__all__ = ["build_hartree_problem", "solve_scipy_hartree"]
