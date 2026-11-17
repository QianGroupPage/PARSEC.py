"""Centered finite differences used by PARSEC's real-space Hamiltonian.

The functions in this module construct the discrete *negative* Laplacian
``A = -nabla**2``.  This sign is useful for both applications in the
single-point calculation:

* the Rydberg-unit kinetic operator is ``T = -nabla**2 = A``;
* the Rydberg-unit Hartree equation is ``A V_H = 8*pi*rho``.

Only axial neighbors are used on a cluster grid and on an orthorhombic periodic cell.  There are no
mixed-derivative or diagonal neighbors such as ``(i+1, j+1, k)`` there.  An oblique periodic cell has
mixed derivatives, which are written as pure second derivatives along a few extra lattice directions
(:mod:`.lattice_metric`) and add those diagonal neighbors.
"""

from __future__ import annotations

from math import factorial
from typing import Iterator

import numpy as np
import scipy.sparse as sp

from ..Grid import RealSpaceGrid
from .lattice_metric import cartesian_derivative_matrix, second_derivative_plan


def _is_oblique(grid) -> bool:
    """A periodic grid whose axes are not the Cartesian axes: its derivatives carry the cell's metric."""
    return getattr(grid, "is_orthorhombic", True) is False


def first_derivative_coefficients(expansion_order: int) -> np.ndarray:
    """Return centered ``d/dx`` weights for a requested even order.

    With ``M=expansion_order/2`` and grid spacing ``h``,

    ``(D f)_i = h**(-1) * sum_j d_j f_(i+j) + O(h**(2*M))``,

    ``d_j = (-1)**(j+1) * (M!)**2 /
            (j * (M-j)! * (M+j)!)`` for ``j=1,...,M``,

    and ``d_-j=-d_j``.  The returned weights are dimensionless and ordered
    from ``-M`` through ``+M``.  Duplicated (not imported) from
    ``V_xc.pbe.first_derivative_coefficients``, which uses this same
    centered stencil for its density gradient, to avoid a
    ``Laplacian -> V_xc`` package dependency; the two are kept numerically
    identical.
    """

    expansion_order = int(expansion_order)
    if expansion_order < 2 or expansion_order > 20 or expansion_order % 2:
        raise ValueError("expansion_order must be an even integer from 2 to 20")
    width = expansion_order // 2
    weights = np.zeros(2 * width + 1, dtype=np.float64)
    for shell in range(1, width + 1):
        value = (
            (-1.0) ** (shell + 1)
            * factorial(width) ** 2
            / (shell * factorial(width - shell) * factorial(width + shell))
        )
        weights[width + shell] = value
        weights[width - shell] = -value
    return weights


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

    A :class:`~parsec_python.Grid.pbc.GeneralPeriodicGrid` (non-orthogonal
    cell) gets the operator with auxiliary directions of
    :func:`~.nonorthogonal.build_general_negative_laplacian` instead.
    """
    if getattr(grid, "is_general_periodic", False):
        from .nonorthogonal import build_general_negative_laplacian

        return build_general_negative_laplacian(grid)
    if _is_oblique(grid):
        return _build_oblique_negative_laplacian(grid)
    if _wrapped_box(grid):
        return _build_wrapped_negative_laplacian(grid)
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


def _wrapped_box(grid) -> bool:
    """Whether ``grid`` is an orthorhombic bulk cell whose rows are its whole box in row-major order (every
    neighbour wraps around and none is missing): the case of :func:`_build_wrapped_negative_laplacian`."""

    from ..Grid.pbc import PeriodicRealSpaceGrid

    if type(grid) is not PeriodicRealSpaceGrid or not grid.is_orthorhombic:
        return False
    if tuple(getattr(grid, "boundary_modes", ())) != ("wrap", "wrap", "wrap"):
        return False
    shape = tuple(int(value) for value in grid.points_per_axis)
    lookup = np.asarray(grid.lookup)
    width = grid.settings.stencil_half_width
    return (
        lookup.shape == shape
        and int(grid.size) == int(np.prod(shape))
        and min(shape) > 2 * width
        and int(lookup.reshape(-1)[0]) == 0
        and int(lookup.reshape(-1)[-1]) == int(grid.size) - 1
    )


_KINETIC_SWITCH = "PARSEC_KPOINT_KINETIC"


def kpoint_kinetic_setting() -> str:
    """``PARSEC_KPOINT_KINETIC``: the kinetic term of a k-point.  ``gradient`` (default): ``-nabla^2 - 2ik.nabla +
    k^2`` with the gradient discretized on its own; ``bloch``: the Bloch block of the finite-difference Laplacian
    (:func:`build_bloch_negative_laplacian`), whose mesh of ``m`` points along an axis gives the Gamma point of the
    supercell of ``m`` cells exactly; ``auto``: ``bloch`` where it applies, ``gradient`` elsewhere.

    Both forms converge to the same energies.  Neither is closer at every spacing: on silicon (2 x 2 x 2 and
    3 x 3 x 3 meshes, 12 to 28 points per cubic edge) the total energy of ``bloch`` is as close or closer to the
    converged one, but on a gold cell at 0.6 bohr its lowest band dips below Gamma at k = 1/4 (-1.2 mRy, +5.3
    with ``gradient``; +3.7 and +4.4 at 0.45 bohr, +10.4 and +10.9 at 0.35): the form of before stays the default.
    """

    import os

    value = os.environ.get(_KINETIC_SWITCH, "gradient").strip().lower() or "gradient"
    if value not in {"auto", "gradient", "bloch"}:
        raise ValueError(f"{_KINETIC_SWITCH} must be gradient, bloch or auto, not {value!r}")
    return value


def bloch_laplacian_or_none(grid, k_point, cache: dict | None = None, key=None, *, meta_gga: bool = False):
    """The u-gauge Bloch Laplacian of ``k_point`` (kept in ``cache`` by ``key``), or ``None`` for the gradient form.

    It applies to a scalar (not spinor) Hamiltonian without a meta-GGA on a bulk orthorhombic cell of the default
    routines (:func:`_wrapped_box`): ``auto`` takes it there and the gradient form elsewhere, ``bloch`` refuses
    the others, ``gradient`` never takes it.
    """

    setting = kpoint_kinetic_setting()
    if setting == "gradient":
        return None
    applies = _wrapped_box(grid) and not meta_gga
    if not applies:
        if setting == "auto":
            return None
        raise ValueError(
            "PARSEC_KPOINT_KINETIC=bloch needs a bulk orthorhombic cell of the default routines (an orthorhombic "
            "periodic box whose rows are its points) and a functional without a kinetic-energy density"
        )
    if cache is not None and key in cache:
        return cache[key]
    matrix = build_bloch_negative_laplacian(grid, k_point, "u")
    if cache is not None:
        cache[key] = matrix
    return matrix


def build_bloch_negative_laplacian(grid, k_point, gauge: str = "u") -> sp.csr_matrix:
    """The Bloch block of the finite-difference ``-nabla^2`` of a periodic box at ``k_point`` (Cartesian, 1/bohr).

    A supercell of ``m`` copies of the box along an axis has the finite-difference Laplacian of the box in each
    copy, and its eigenfunctions of the translation by the box are ``psi(r + L) = e^{ik.L} psi(r)``: on the box
    they solve the box's stencil with every wrapped neighbour taken times ``e^{ik.L}`` (``gauge="psi"``: the
    twisted Laplacian, real, ``+-1`` on the wrapped entries, at a time-reversal invariant ``k``).  The periodic
    factor ``u = e^{-ik.r} psi`` solves the stencil whose weight of the neighbour ``delta`` away is multiplied by
    ``e^{ik.delta}`` (``gauge="u"``, the default, complex): ``sum_delta c_delta e^{ik.delta} u(r + delta)``.
    Both are the exact Bloch block of the supercell's matrix, so a mesh of ``m`` points along an axis gives the
    eigenvalues of the supercell of ``m`` boxes at the Gamma point.  (``-nabla^2 - 2ik.nabla + k^2`` with the
    gradient discretized on its own, :class:`~parsec_python.Hamiltonian.KPointKohnShamHamiltonian` without
    ``bloch_laplacian``, is another discretization of the same operator.)  An orthorhombic box of the default
    routines (:func:`_wrapped_box`) only.
    """

    if not _wrapped_box(grid):
        raise ValueError("the Bloch Laplacian needs an orthorhombic periodic box whose rows are its points")
    if gauge not in {"u", "psi"}:
        raise ValueError(f"gauge must be 'u' or 'psi', not {gauge!r}")
    return _build_wrapped_negative_laplacian(grid, k_point=np.asarray(k_point, dtype=np.float64), gauge=gauge)


def _build_wrapped_negative_laplacian(grid, *, k_point=None, gauge: str = "u") -> sp.csr_matrix:
    """The matrix of :func:`build_negative_laplacian` for a periodic box, written directly in CSR.

    Every row holds its point and the ``2 M`` neighbours along each axis, all of them (they wrap), so the rows
    are of equal length and the matrix of the COO route is known entry by entry: the columns of a row in
    ascending order, the value of each its stencil weight.  Away from the faces the order is the same for every
    row (axis 0 backwards, axis 1 backwards, axis 2 backwards, the point, then forwards); only rows within ``M``
    of a face, whose wrapped neighbours change it, are sorted.  The same arrays as the COO route's (``tocsr``,
    ``sum_duplicates``: sorted columns, no duplicates since every axis holds more than ``2 M`` points) without
    its sort of all the entries.
    """

    coeff = second_derivative_coefficients(grid.settings.expansion_order)
    width = grid.settings.stencil_half_width
    inv_h2 = 1.0 / grid.spacing**2
    shape = tuple(int(value) for value in grid.points_per_axis)
    n = int(np.prod(shape))
    strides = (shape[1] * shape[2], shape[2], 1)
    # (axis, signed shift) in ascending column order for a row away from the faces
    order = (
        [(0, -s) for s in range(width, 0, -1)]
        + [(1, -s) for s in range(width, 0, -1)]
        + [(2, -s) for s in range(width, 0, -1)]
        + [(None, 0)]
        + [(2, s) for s in range(1, width + 1)]
        + [(1, s) for s in range(1, width + 1)]
        + [(0, s) for s in range(1, width + 1)]
    )
    count = len(order)
    index_type = np.int32 if n * count < 2**31 else np.int64
    weights = np.array(
        [-3.0 * coeff[width] * inv_h2 if axis is None else -coeff[width + abs(shift)] * inv_h2 for axis, shift in order]
    )
    rows = np.arange(n, dtype=index_type).reshape(shape)
    coordinates = [np.arange(extent, dtype=index_type) for extent in shape]
    columns = np.empty((n, count), dtype=index_type)
    for slot, (axis, shift) in enumerate(order):
        if axis is None:
            columns[:, slot] = rows.reshape(-1)
            continue
        # column of the neighbour (i + shift) mod n along the axis: the row plus the change of that index
        change = ((coordinates[axis] + shift) % shape[axis] - coordinates[axis]) * strides[axis]
        view = [1, 1, 1]
        view[axis] = shape[axis]
        columns[:, slot] = (rows + change.reshape(view)).reshape(-1)
    if k_point is None:
        data = np.broadcast_to(weights, (n, count)).copy()
    else:
        # the Bloch block (build_bloch_negative_laplacian): a phase per slot (u gauge) or per wrapped entry (psi)
        sides = np.diag(np.asarray(grid.cell.lattice_vectors, dtype=np.float64))
        steps = sides / np.asarray(shape, dtype=np.float64)
        if gauge == "u":
            phases = np.array(
                [1.0 if axis is None else np.exp(1j * k_point[axis] * shift * steps[axis]) for axis, shift in order]
            )
            data = np.broadcast_to(weights * phases, (n, count)).copy()
        else:
            data = np.empty((n, count), dtype=np.complex128)
            for slot, (axis, shift) in enumerate(order):
                if axis is None:
                    data[:, slot] = weights[slot]
                    continue
                wraps = (coordinates[axis] + shift) // shape[axis]
                phase = np.exp(1j * k_point[axis] * sides[axis] * wraps.astype(np.float64))
                view = [1, 1, 1]
                view[axis] = shape[axis]
                data[:, slot] = np.broadcast_to(weights[slot] * phase.reshape(view), shape).reshape(-1)
            if np.max(np.abs(data.imag)) <= 1.0e-12 * np.max(np.abs(data.real)):
                # a time-reversal invariant k: the twisted Laplacian is real
                data = np.ascontiguousarray(np.rint(data.real / weights[None, :]) * weights[None, :])
    near = np.zeros(shape, dtype=bool)
    for axis, extent in enumerate(shape):
        edge = (coordinates[axis] < width) | (coordinates[axis] >= extent - width)
        view = [1, 1, 1]
        view[axis] = extent
        near |= edge.reshape(view)
    near = np.flatnonzero(near.reshape(-1))
    if near.size:
        permutation = np.argsort(columns[near], axis=1, kind="stable")
        columns[near] = np.take_along_axis(columns[near], permutation, axis=1)
        if k_point is None:
            data[near] = weights[permutation]
        else:
            data[near] = np.take_along_axis(data[near], permutation, axis=1)
    indptr = np.arange(0, n * count + 1, count, dtype=index_type)
    matrix = sp.csr_matrix((data.reshape(-1), columns.reshape(-1), indptr), shape=(n, n))
    matrix.has_sorted_indices = True
    return matrix


def _build_oblique_negative_laplacian(grid) -> sp.csr_matrix:
    """``-nabla^2`` on a periodic grid with oblique axes: axial plus extra-direction second derivatives.

    ``-nabla^2 = -sum_i p_i d^2/du_i^2 - sum_k w_k d^2/dt^2 (along d_k)`` with the weights of
    :func:`~.lattice_metric.second_derivative_plan`; each second derivative is the centered stencil
    ``sum_j c_j f(u + j d)`` in index units (the steps are inside ``p`` and ``w``).
    """
    coeff = second_derivative_coefficients(grid.settings.expansion_order)
    width = grid.settings.stencil_half_width
    plan = second_derivative_plan(grid.step_vectors)
    n = grid.size
    base = grid.integer_coordinates

    total = float(np.sum(plan.axial)) + float(np.sum(plan.weights))
    row_parts: list[np.ndarray] = [np.arange(n, dtype=np.int64)]
    col_parts: list[np.ndarray] = [np.arange(n, dtype=np.int64)]
    value_parts: list[np.ndarray] = [np.full(n, -total * coeff[width], dtype=float)]

    unit_axes = [tuple(int(v) for v in row) for row in np.eye(3, dtype=int)]
    lines = [(axis, weight) for axis, weight in zip(unit_axes, plan.axial)]
    lines += list(zip(plan.directions, plan.weights))
    for direction, weight in lines:
        step = np.asarray(direction, dtype=np.int64)
        for shell in range(1, width + 1):
            for sign in (-1, 1):
                rows = grid.rows_for_integer_coordinates(base + sign * shell * step)
                valid = rows >= 0  # a neighbor outside a slab box is a zero wavefunction value
                if not np.any(valid):
                    continue
                row_parts.append(np.flatnonzero(valid))
                col_parts.append(rows[valid])
                value_parts.append(np.full(int(np.count_nonzero(valid)), -weight * coeff[width + shell], dtype=float))

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


def build_gradient(grid: RealSpaceGrid) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Build the three sparse centered first-derivative operators ``d/dx,y,z``.

    Needed for a periodic Bloch Hamiltonian's ``-2ik.grad`` kinetic term
    (:mod:`~parsec_python.Hamiltonian.kpoint_operator`): with
    ``psi_nk(r) = e^{ik.r}u_nk(r)``,
    ``e^{-ik.r}(-nabla^2)e^{ik.r} = -nabla^2 - 2ik.grad + |k|^2``, so the
    existing :func:`build_negative_laplacian` supplies the first and third
    terms and this function supplies the middle one.

    Unlike :func:`build_negative_laplacian`, there is no center-point
    contribution (``d/dx`` of a constant is zero: the stencil is exactly
    antisymmetric, ``d_-j = -d_j``), so each returned matrix has at most
    ``2*M`` nonzeros per row.  On the isolated cluster grid, a missing
    neighbor (outside the active domain) is dropped exactly as in
    :func:`build_negative_laplacian` -- appropriate for evaluating a
    density gradient there (as :mod:`~parsec_python.V_xc.pbe` already does
    with the same coefficients), but a k-point kinetic term is only
    physically meaningful on a periodic grid, where every neighbor lookup
    wraps around and none are ever missing.

    On a :class:`~parsec_python.Grid.pbc.GeneralPeriodicGrid` the three
    matrices are still Cartesian, built from the derivatives along the
    lattice directions (:func:`~.nonorthogonal.build_general_gradient`).
    """
    if getattr(grid, "is_general_periodic", False):
        from .nonorthogonal import build_general_gradient

        return build_general_gradient(grid)
    coeff = first_derivative_coefficients(grid.settings.expansion_order)
    width = grid.settings.stencil_half_width
    oblique = _is_oblique(grid)
    # An oblique cell differentiates along its grid axes (unit index steps) and mixes them afterwards.
    inv_h = 1.0 if oblique else 1.0 / grid.spacing
    n = grid.size

    axis_parts: list[list[np.ndarray]] = [[], [], []]
    for axis, signed_shell, neighbor_rows, _points in neighbor_shells(grid):
        valid = neighbor_rows >= 0
        if not np.any(valid):
            continue
        weight = coeff[width + signed_shell] * inv_h
        rows = np.flatnonzero(valid)
        cols = neighbor_rows[valid]
        values = np.full(rows.size, weight, dtype=float)
        axis_parts[axis].append((rows, cols, values))

    matrices = []
    for parts in axis_parts:
        if not parts:
            matrices.append(sp.csr_matrix((n, n)))
            continue
        rows = np.concatenate([part[0] for part in parts])
        cols = np.concatenate([part[1] for part in parts])
        values = np.concatenate([part[2] for part in parts])
        matrix = sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
        matrix.sum_duplicates()
        matrices.append(matrix)
    if oblique:
        # d/dx_k = sum_i C[k, i] d/du_i, C = S^{-1}
        mix = cartesian_derivative_matrix(grid.step_vectors)
        matrices = [sum(mix[k, i] * matrices[i] for i in range(3)).tocsr() for k in range(3)]
    return tuple(matrices)


__all__ = [
    "apply_negative_laplacian_boundary",
    "build_gradient",
    "build_negative_laplacian",
    "first_derivative_coefficients",
    "neighbor_shells",
    "second_derivative_coefficients",
]
