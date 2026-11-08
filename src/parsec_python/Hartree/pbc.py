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


# A faster solver of the same equations (a CUDA device), or None.
_periodic_solver = None


def set_periodic_hartree_solver(solver=None) -> None:
    """Install (or, with ``None``, remove) a faster :func:`solve_periodic_hartree`.

    ``solver(density, grid, negative_laplacian, settings, initial_potential,
    raise_on_nonconvergence=...)`` returns a :class:`PeriodicHartreeResult` of
    the same equation (the spectral solve of ``Calc_Hartree_G`` or the
    finite-difference CG), or ``None`` for a solve it does not take (the code
    below then runs).  The performance counterpart lives in
    ``parsec_python.acceleration.Hartree.periodic_device``; importing
    :mod:`parsec_python.acceleration` installs it.
    """

    global _periodic_solver
    _periodic_solver = solver


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
    if _periodic_solver is not None:
        solved = _periodic_solver(
            density,
            grid,
            negative_laplacian,
            settings,
            initial_potential,
            raise_on_nonconvergence=raise_on_nonconvergence,
        )
        if solved is not None:
            return solved
    if getattr(settings, "periodic_g_space", False):
        return _solve_in_g_space(density, grid, initial_potential)
    if getattr(grid, "is_general_periodic", False):
        return _solve_by_fourier_transform(
            density,
            grid,
            negative_laplacian,
            settings,
            initial_potential,
            raise_on_nonconvergence=raise_on_nonconvergence,
        )

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


def _solve_in_g_space(
    density: np.ndarray,
    grid: PeriodicRealSpaceGrid,
    initial_potential: np.ndarray | None,
) -> PeriodicHartreeResult:
    """``Calc_Hartree_G``: the spectral solve of :mod:`.gspace`.

    The solve is exact for the spectral Poisson equation, so its residual is
    zero and it counts as converged; it uses neither the finite-difference
    operator nor a starting potential (``initial_potential`` is only
    checked).  The spectral ``V_H`` does not solve the finite-difference
    equation, so no finite-difference residual is reported.
    """

    from .gspace import solve_hartree_g_space

    if initial_potential is not None and np.shape(initial_potential) != (grid.size,):
        raise ValueError("initial Hartree potential does not match the grid")
    rhs = 8.0 * np.pi * neutralize_density(density, grid)
    potential = solve_hartree_g_space(density, grid)
    return PeriodicHartreeResult(
        potential=potential,
        right_hand_side=rhs,
        converged=True,
        iterations=0,
        matrix_vector_products=0,
        residual_norm=0.0,
        initial_residual_norm=float(np.linalg.norm(rhs)),
    )


def finite_difference_symbol(
    grid: PeriodicRealSpaceGrid, negative_laplacian: sp.spmatrix
) -> np.ndarray:
    """Eigenvalues of a wrapped finite-difference operator on its plane waves.

    On the full periodic grid the operator is circulant: row 0 holds the
    weight ``w_o`` of every offset ``o``, and the plane wave
    ``exp(2 pi i k.m/n)`` has the eigenvalue ``sum_o w_o exp(2 pi i k.o/n)``.
    For a symmetric operator that is ``real(fftn(row 0))``, returned on the
    ``grid.shape`` FFT mesh.
    """

    row = np.asarray(negative_laplacian.getrow(0).toarray()).reshape(grid.shape)
    return np.real(np.fft.fftn(row))


def _solve_by_fourier_transform(
    density: np.ndarray,
    grid: PeriodicRealSpaceGrid,
    negative_laplacian: sp.spmatrix,
    settings: HartreeSettings,
    initial_potential: np.ndarray | None,
    *,
    raise_on_nonconvergence: bool,
) -> PeriodicHartreeResult:
    """Exact solve of ``A V = 8 pi (rho - <rho>)`` by FFT (general periodic grid).

    ``A`` is the finite-difference ``-nabla^2`` of a
    :class:`~parsec_python.Grid.pbc.GeneralPeriodicGrid`, including its
    auxiliary directions; dividing by its symbol
    (:func:`finite_difference_symbol`) gives the solution of the same linear
    system the conjugate gradients of the orthorhombic path (and Fortran's
    ``hpotcg``) iterate on, with the ``G = 0`` component set to zero so that
    the potential has zero mean, as there.  The residual is reported for the
    diagnostics; ``initial_potential`` only enters the initial residual.
    """

    neutral_density = neutralize_density(density, grid)
    rhs = 8.0 * np.pi * neutral_density
    if initial_potential is None:
        initial = np.zeros(grid.size, dtype=np.float64)
    else:
        initial = np.asarray(initial_potential, dtype=np.float64)
        if initial.shape != (grid.size,):
            raise ValueError("initial Hartree potential does not match the grid")
        initial = initial - float(np.mean(initial))
    initial_residual = float(np.linalg.norm(rhs - negative_laplacian @ initial))

    symbol = finite_difference_symbol(grid, negative_laplacian)
    nonzero = np.ones(grid.shape, dtype=bool)
    nonzero[0, 0, 0] = False
    smallest = float(np.min(np.abs(symbol[nonzero]))) if np.any(nonzero) else 1.0
    if not smallest > 1.0e-12 * float(np.max(np.abs(symbol))):
        raise RuntimeError(
            "the finite-difference Laplacian of this periodic grid is singular "
            "beyond the constant mode; the Hartree potential is not defined"
        )
    transform = np.fft.fftn(rhs.reshape(grid.shape))
    symbol[0, 0, 0] = 1.0
    transform /= symbol
    transform[0, 0, 0] = 0.0
    potential = np.real(np.fft.ifftn(transform)).reshape(-1)
    potential = potential - float(np.mean(potential))
    residual = float(np.linalg.norm(rhs - negative_laplacian @ potential))
    converged = residual <= (
        settings.relative_tolerance * initial_residual + settings.absolute_tolerance
    ) or residual <= 1.0e-10 * float(np.linalg.norm(rhs))
    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            "periodic Hartree FFT solve left a residual of "
            f"{residual:.3e} (initial {initial_residual:.3e})"
        )
    return PeriodicHartreeResult(
        potential=potential,
        right_hand_side=rhs,
        converged=converged,
        iterations=0,
        matrix_vector_products=2,
        residual_norm=residual,
        initial_residual_norm=initial_residual,
    )


__all__ = [
    "PeriodicHartreeResult",
    "finite_difference_symbol",
    "neutralize_density",
    "set_periodic_hartree_solver",
    "solve_periodic_hartree",
]
