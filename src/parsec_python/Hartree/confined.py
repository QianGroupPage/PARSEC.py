"""Hartree potential of a wire or slab: boundary values plus the FD-CG solve.

As for a cluster (:mod:`parsec_python.Hartree.poisson`), the Rydberg Poisson
equation ``-nabla^2 V_H = 8 pi n`` is solved on the active points with the
finite-difference operator ``A_II`` of the grid (wrapped along the periodic
directions), the values ``V_B`` of the missing stencil neighbours beyond the
confined domain folded into the source,

``A_II V_I = 8 pi n_I - A_IB V_B``

(``rho_hart.F90:95-101``, ``hartset_wire.F90:262-333``,
``hartset_slab.F90:349-420``), and conjugate gradients from the previous
Hartree potential (``-V_ion`` on the first call, ``rho_hart.F90:164-175``).
``V_B`` comes from :mod:`.wire` (cylindrical multipoles) or :mod:`.slab`
(2-D Fourier sums per grid plane).  No average is removed: the boundary
values fix the potential's zero (Fortran's ``potshift`` acts in 3-D only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..Laplacian.finite_difference import second_derivative_coefficients
from ..Laplacian.nonorthogonal import grid_laplacian_coefficients
from ..models import HartreeSettings
from .poisson import _conjugate_gradient
from .slab_fortran import SlabBoundaryPlan, build_slab_boundary_plan
from .wire import WireBoundaryPlan, build_wire_boundary_plan


@dataclass(frozen=True)
class ExteriorStencil:
    """The missing neighbours of the finite-difference stencil.

    Entry ``e`` is the neighbour ``integer_points[e]`` (Cartesian
    ``coordinates[e]``) of active row ``rows[e]``, outside the confined
    domain, with the operator entry ``coefficients[e]`` of ``A_IB`` in
    ``bohr**-2``.  The order is that of the stencil terms, shells and signs of
    :func:`parsec_python.Laplacian.nonorthogonal.build_general_negative_laplacian`.
    """

    rows: np.ndarray
    coefficients: np.ndarray
    integer_points: np.ndarray
    coordinates: np.ndarray

    @property
    def size(self) -> int:
        return int(self.rows.size)


def exterior_stencil(grid) -> ExteriorStencil:
    """Collect every stencil neighbour of a wire/slab grid outside its domain."""

    coefficients = grid_laplacian_coefficients(grid)
    weights = second_derivative_coefficients(grid.settings.expansion_order)
    width = grid.settings.stencil_half_width
    base = grid.integer_coordinates
    rows, values, points = [], [], []
    for offset, coefficient, step in coefficients.stencil_terms(grid.spacings):
        for shell in range(1, width + 1):
            value = -coefficient * weights[width + shell] / step**2
            for sign in (-1, 1):
                displaced = base + sign * shell * offset
                missing = grid.rows_for_integer_coordinates(displaced) < 0
                if not np.any(missing):
                    continue
                rows.append(np.flatnonzero(missing))
                values.append(np.full(int(np.count_nonzero(missing)), value))
                points.append(displaced[missing])
    if not rows:
        empty = np.empty(0, dtype=np.int64)
        return ExteriorStencil(empty, np.empty(0), np.empty((0, 3), dtype=np.int64), np.empty((0, 3)))
    integer_points = np.concatenate(points)
    return ExteriorStencil(
        rows=np.concatenate(rows),
        coefficients=np.concatenate(values),
        integer_points=integer_points,
        coordinates=grid.physical_coordinates(integer_points),
    )


@dataclass(frozen=True)
class ConfinedHartreeResult:
    """Hartree solution of a wire or slab and its diagnostics.

    ``boundary_values`` are the Rydberg values at the exterior stencil
    points, ``right_hand_side`` the corrected source passed to CG; the CG
    counters are those of :class:`parsec_python.Hartree.poisson.HartreeResult`.
    """

    potential: np.ndarray
    right_hand_side: np.ndarray
    boundary_values: np.ndarray
    converged: bool
    iterations: int
    matrix_vector_products: int
    residual_norm: float
    initial_residual_norm: float


@dataclass(frozen=True)
class ConfinedHartreePlan:
    """Everything a wire or slab Hartree solve needs besides the density."""

    stencil: ExteriorStencil
    boundary: WireBoundaryPlan | SlabBoundaryPlan

    def boundary_values(self, density: np.ndarray) -> np.ndarray:
        return self.boundary.boundary_values(density)

    def right_hand_side(self, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``8 pi n - A_IB V_B`` and ``V_B`` for one density."""

        density = np.asarray(density, dtype=np.float64)
        values = self.boundary_values(density)
        correction = np.bincount(
            self.stencil.rows,
            weights=self.stencil.coefficients * values,
            minlength=density.size,
        )
        return 8.0 * np.pi * density - correction, values


def plan_confined_hartree(
    grid,
    multipole_order: int,
    *,
    fortran_quirks: frozenset[str] | set[str] = frozenset(),
) -> ConfinedHartreePlan:
    """Exterior stencil and boundary tables of a wire or slab grid.

    ``multipole_order`` is ``Solver_Lpole``.  ``fortran_quirks`` may hold
    ``"wire_axis_points"`` (:mod:`.wire`) and ``"slab_lowest_plane"``
    (:mod:`.slab`) to reproduce Fortran's boundary values exactly.
    """

    stencil = exterior_stencil(grid)
    if grid.periodicity == 1:
        boundary: WireBoundaryPlan | SlabBoundaryPlan = build_wire_boundary_plan(
            grid,
            stencil.coordinates,
            multipole_order,
            skip_axis_points="wire_axis_points" in fortran_quirks,
        )
    else:
        boundary = build_slab_boundary_plan(
            grid,
            stencil.integer_points,
            multipole_order,
            fortran_plane_quirk="slab_lowest_plane" in fortran_quirks,
        )
    return ConfinedHartreePlan(stencil=stencil, boundary=boundary)


# A faster solver of the same equations (a CUDA device), or None.
_confined_solver = None


def set_confined_hartree_solver(solver=None) -> None:
    """Install (or, with ``None``, remove) a faster :func:`solve_confined_hartree`.

    ``solver(density, plan, negative_laplacian, settings, initial_potential,
    raise_on_nonconvergence=...)`` returns a :class:`ConfinedHartreeResult` of
    the same equations and CG recurrence, or ``None`` for a solve it does not
    take (the conjugate gradients below then run).  The performance
    counterpart lives in ``parsec_python.acceleration.Hartree.periodic_device``;
    importing :mod:`parsec_python.acceleration` installs it.
    """

    global _confined_solver
    _confined_solver = solver


def solve_confined_hartree(
    density: np.ndarray,
    plan: ConfinedHartreePlan,
    negative_laplacian: sp.spmatrix,
    settings: HartreeSettings = HartreeSettings(),
    initial_potential: np.ndarray | None = None,
    *,
    raise_on_nonconvergence: bool = True,
) -> ConfinedHartreeResult:
    """Solve ``A_II V = 8 pi n - A_IB V_B`` for a wire or slab (Ry).

    ``negative_laplacian`` is the operator of the grid of ``plan``; the CG
    tolerances are those of ``settings`` (as for a cluster).
    """

    density = np.asarray(density, dtype=np.float64)
    if negative_laplacian.shape != (density.size, density.size):
        raise ValueError("negative_laplacian shape does not match the density")
    if initial_potential is not None and np.shape(initial_potential) != density.shape:
        raise ValueError("initial Hartree potential does not match the grid")
    if _confined_solver is not None:
        solved = _confined_solver(
            density,
            plan,
            negative_laplacian,
            settings,
            initial_potential,
            raise_on_nonconvergence=raise_on_nonconvergence,
        )
        if solved is not None:
            return solved
    rhs, values = plan.right_hand_side(density)
    if initial_potential is None:
        initial = np.zeros(density.size, dtype=np.float64)
    else:
        initial = np.asarray(initial_potential, dtype=np.float64)
        if initial.shape != density.shape:
            raise ValueError("initial Hartree potential does not match the grid")
    potential, converged, iterations, matvecs, residual, initial_residual = (
        _conjugate_gradient(negative_laplacian, rhs, initial, settings)
    )
    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            "wire/slab Hartree conjugate-gradient solve did not converge: "
            f"residual={residual:.3e}, matvecs={matvecs}"
        )
    return ConfinedHartreeResult(
        potential=potential,
        right_hand_side=rhs,
        boundary_values=values,
        converged=converged,
        iterations=iterations,
        matrix_vector_products=matvecs,
        residual_norm=residual,
        initial_residual_norm=initial_residual,
    )


__all__ = [
    "ConfinedHartreePlan",
    "ConfinedHartreeResult",
    "ExteriorStencil",
    "exterior_stencil",
    "plan_confined_hartree",
    "set_confined_hartree_solver",
    "solve_confined_hartree",
]
