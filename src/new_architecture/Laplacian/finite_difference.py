"""Centered finite differences used by PARSEC's real-space Hamiltonian.

The functions in this module construct the discrete *negative* Laplacian
``A = -nabla**2``.  This sign is useful for both applications in the
single-point calculation:

* the Rydberg-unit kinetic operator is ``T = -nabla**2 = A``;
* the Rydberg-unit Hartree equation is ``A V_H = 8*pi*rho``.

Only axial neighbors are used.  There are no mixed-derivative or diagonal
neighbors such as ``(i+1, j+1, k)``.
"""

from __future__ import annotations

from math import factorial
from typing import Iterator

import numpy as np
import scipy.sparse as sp

from ..Grid import RealSpaceGrid


def second_derivative_coefficients(expansion_order: int) -> np.ndarray:
    """Return centered coefficients for the positive derivative ``d²/dx²``.

    Let ``p = expansion_order`` and ``M = p/2``.  On a uniform grid with
    spacing ``h``, the one-dimensional approximation is

    ``f''(x_i) = h**(-2) * [c_0*f_i
                 + sum(c_j*(f_(i-j) + f_(i+j)), j=1,...,M)]
                 + O(h**p)``.

    For a centered stencil on a uniform grid, the nonzero-shell weights have
    the closed form

    ``c_j = 2*(-1)**(j+1)*(M!)**2
             / [j**2*(M-j)!*(M+j)!]``,

    and ``c_0 = -2*sum(c_j, j=1,...,M)``.  The latter identity makes the
    derivative of a constant exactly zero.

    The returned array is ordered as

    ``[c_-M, ..., c_-1, c_0, c_1, ..., c_M]``,

    with ``c_-j = c_j``.  These are dimensionless coefficients for the
    *positive* second derivative; the factor ``1/h**2`` and the minus sign
    required by ``-nabla**2`` are applied by
    :func:`build_negative_laplacian`.

    This closed-form expression is algebraically equivalent to PARSEC's
    centered Fornberg coefficients, but it is not an implementation of the
    recursive Fornberg weight-generation algorithm.
    """
    expansion_order = int(expansion_order)
    if expansion_order < 2 or expansion_order > 20 or expansion_order % 2:
        raise ValueError("expansion_order must be an even integer from 2 to 20")

    # PARSEC calls p the expansion order; M is the number of grid neighbors
    # used on each side of the center along one Cartesian axis.
    half_width = expansion_order // 2
    coeff = np.empty(2 * half_width + 1, dtype=float)
    center = half_width

    # Symmetry gives identical weights at distances -j and +j.
    for j in range(1, half_width + 1):
        value = (
            2.0
            * (-1.0) ** (j + 1)
            * factorial(half_width) ** 2
            / (j * j * factorial(half_width - j) * factorial(half_width + j))
        )
        coeff[center - j] = value
        coeff[center + j] = value

    # Enforce c_0 + 2*sum_{j=1}^M c_j = 0.
    coeff[center] = -2.0 * np.sum(coeff[center + 1 :])
    return coeff


def neighbor_shells(grid: RealSpaceGrid) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Enumerate every signed axial shell required by the centered stencil.

    If the stencil half-width is ``M``, this generator yields ``6*M`` items:
    the signed displacements ``-M,...,-1,1,...,M`` along each of the three
    Cartesian axes.

    Each yielded tuple contains:

    ``axis``
        Cartesian axis ``0``, ``1``, or ``2``.
    ``signed_shell``
        Signed displacement in integer grid units.  For example, ``-2`` on
        axis ``1`` means the candidate coordinate ``(i, j-2, k)``.
    ``neighbor_rows``
        An array with one entry for every active grid row.  Entry ``r`` is
        the active row of the displaced point, or ``-1`` when that point is
        outside the active domain.
    ``integer_points``
        The displaced integer coordinates themselves.  These are retained
        even for missing neighbors because the Hartree boundary correction
        must evaluate the prescribed potential at their physical positions.

    Looking up coordinates instead of assuming fixed flattened offsets lets
    the same stencil work on a compressed spherical or box-shaped domain.
    """
    width = grid.settings.stencil_half_width
    base = grid.integer_coordinates
    for axis in range(3):
        for signed_shell in range(-width, width + 1):
            if signed_shell == 0:
                continue
            points = base.copy()
            points[:, axis] += signed_shell
            yield axis, signed_shell, grid.rows_for_integer_coordinates(points), points


def build_negative_laplacian(grid: RealSpaceGrid) -> sp.csr_matrix:
    """Build the sparse negative Laplacian on the active real-space domain.

    If ``c_j`` are the coefficients returned by
    :func:`second_derivative_coefficients`, this function constructs

    ``(A u)_r = -h**(-2) * [3*c_0*u_r
                + sum_axis sum_{j=1}^M
                  c_j*(u_(r-j*e_axis) + u_(r+j*e_axis))]``,

    so that ``A`` approximates ``-nabla**2``.  The factor ``3*c_0`` appears
    because the center coefficient is contributed once by each of the
    ``x``, ``y``, and ``z`` second derivatives.

    The orbital boundary condition is homogeneous Dirichlet:

    ``u(point outside the active domain) = 0``.

    Consequently, the full center coefficient is retained at boundary rows,
    while entries for missing neighbors are omitted.  The stencil is not
    renormalized and is not replaced by a one-sided finite difference.

    The returned CSR matrix has at most ``1 + 6*M`` nonzeros per row.  It is
    assembled once and reused as the kinetic operator and as the interior
    Poisson operator.
    """
    coeff = second_derivative_coefficients(grid.settings.expansion_order)
    width = grid.settings.stencil_half_width
    inv_h2 = 1.0 / grid.spacing**2
    n = grid.size

    # Every one-dimensional second derivative contributes c_0 at the same
    # central point.  Negating their sum gives -3*c_0/h**2.
    row_parts: list[np.ndarray] = [np.arange(n, dtype=np.int64)]
    col_parts: list[np.ndarray] = [np.arange(n, dtype=np.int64)]
    value_parts: list[np.ndarray] = [
        np.full(n, -3.0 * coeff[width] * inv_h2, dtype=float)
    ]

    for _axis, signed_shell, neighbor_rows, _points in neighbor_shells(grid):
        # A negative row marks an exterior/inactive point.  Omitting that
        # matrix entry is equivalent to multiplying it by a zero exterior
        # wavefunction value.
        valid = neighbor_rows >= 0
        if not np.any(valid):
            continue
        shell = abs(signed_shell)
        row_parts.append(np.flatnonzero(valid))
        col_parts.append(neighbor_rows[valid])
        # coeff describes +d**2/dx**2; A=-nabla**2 therefore uses -coeff.
        value_parts.append(np.full(np.count_nonzero(valid), -coeff[width + shell] * inv_h2))

    # COO is convenient while appending neighbor contributions; CSR is more
    # efficient for the repeated matrix-vector products used by the solver.
    matrix = sp.coo_matrix(
        (np.concatenate(value_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(n, n),
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def apply_negative_laplacian_boundary(
    rhs: np.ndarray,
    grid: RealSpaceGrid,
    boundary_potential,
) -> np.ndarray:
    """Fold prescribed exterior values into a Poisson right-hand side.

    :func:`build_negative_laplacian` stores only the active-to-active block
    ``A_II`` of the negative Laplacian.  For a nonzero Dirichlet Hartree
    boundary, the full interior equation is

    ``A_II V_I + A_IB V_B = b_I``.

    Since ``V_B`` is known, the equation solved for the active values is

    ``A_II V_I = b_I - A_IB V_B``.

    This function performs exactly that right-hand-side update.  For every
    missing stencil neighbor, its matrix coefficient is ``-c_j/h**2`` and
    its contribution is subtracted from the corresponding interior entry.
    ``boundary_potential`` is called with an ``(n_missing, 3)`` array of
    physical Cartesian coordinates and must return the potential at those
    points.

    A copy of ``rhs`` is returned; the input array is not modified.  This
    function only applies the boundary correction--it does not solve the
    Poisson equation.
    """
    rhs = np.asarray(rhs, dtype=float).copy()
    if rhs.shape != (grid.size,):
        raise ValueError("Poisson right-hand side does not match the active grid")
    coeff = second_derivative_coefficients(grid.settings.expansion_order)
    width = grid.settings.stencil_half_width
    inv_h2 = 1.0 / grid.spacing**2

    for _axis, signed_shell, neighbor_rows, integer_points in neighbor_shells(grid):
        missing = neighbor_rows < 0
        if not np.any(missing):
            continue
        shell = abs(signed_shell)

        # This is the off-diagonal entry A_IB associated with the missing
        # point.  Moving A_IB*V_B to the RHS gives rhs -= A_IB*V_B.
        operator_coefficient = -coeff[width + shell] * inv_h2
        points = grid.physical_coordinates(integer_points[missing])
        rhs[missing] -= operator_coefficient * np.asarray(boundary_potential(points))
    return rhs


__all__ = [
    "apply_negative_laplacian_boundary",
    "build_negative_laplacian",
    "neighbor_shells",
    "second_derivative_coefficients",
]
