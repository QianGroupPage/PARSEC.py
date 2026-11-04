"""Finite differences on a non-orthogonal periodic grid (F ``pbc_grid_coefs.f90``).

On the grid of :class:`~parsec_python.Grid.pbc.GeneralPeriodicGrid` a point
is ``r = A_n u`` with ``u`` the coordinates along the unit lattice vectors
(the columns of ``A_n``).  With ``D = inv(A_n) inv(A_n)^T`` the Laplacian is

``nabla^2 = sum_i D_ii d_i^2 + sum_(i<j) 2 D_ij d_i d_j``,

``a_lap = (D11, D22, D33, 2 D12, 2 D13, 2 D23)``.  Following Natan et al.
(PRB 78, 075109, 2008), PARSEC removes the mixed derivatives with second
derivatives along up to three auxiliary grid directions ``w_k`` (integer
offsets ``o_k`` with entries in ``{-1, 0, 1}``):

``nabla^2 = sum_i b_i d_(u_i)^2 + sum_k b_(3+k) d_(w_k)^2``.

Each term is the ordinary centred second difference of order
``Expansion_Order`` along its own direction, with step ``h_i`` along the
lattice directions and ``|A_n (o_k1 h_1, o_k2 h_2, o_k3 h_3)|`` along
``w_k`` (``init_var.F90:393-420``, ``hpotcg.F90:lapmvs``).  The same operator
is the kinetic energy, the Hartree operator and the k-point kinetic
``-nabla^2``.

The gradient is the centred first difference along the three lattice
directions, ``d/du_i``, turned Cartesian by ``inv(A_n)^T``
(``grad_bvec_norm``, ``hpotcg.F90:gradmvs``); it serves PBE and the k-point
term ``-2i k.grad``.  No first derivative is taken along the auxiliary
directions, as in Fortran.

Deviations from Fortran, both chosen so that the operator is the Laplacian
(they change nothing for the cells of the Fortran examples):

* ``kw``: Fortran scales the third offset component by ``kw = h3/h2``
  (``pbc_grid_coefs.f90:198``) where the lattice displacement needs
  ``h3/h1``.  With ``h1 != h2`` its auxiliary steps and directions do not
  belong to the offsets used, and the operator is not exact for quadratic
  functions.  This module uses ``h3/h1``; ``step_ratio_quirk=True`` of
  :func:`pbc_grid_coefficients` reproduces Fortran's numbers.
* Compact indexing: Fortran stores the coefficient and step of the ``ii``-th
  *active* auxiliary direction in slot ``ii`` (``init_var.F90:413-415``), which
  mismatches the neighbours when an inactive direction precedes an active
  one.  This module pairs every direction with its own coefficient and step.
* ``gradmvs`` applies ``grad_bvec_norm`` only when auxiliary directions exist;
  this module always applies it (it is the identity for a diagonal cell).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import correlate1d

from .finite_difference import (
    first_derivative_coefficients,
    second_derivative_coefficients,
)

#: Mixed coefficients below this are zero (``init_var.F90:378`` passes 1e-9).
ORTHOGONALITY_TOLERANCE = 1.0e-9
#: ``mtrxin1``: a 3x3 matrix with ``|det|`` below this is singular.
_SINGULAR_DETERMINANT = 1.0e-5
#: An auxiliary direction with ``|b| <`` this is inactive (``pbc_grid_coefs.f90:329``).
_INACTIVE_COEFFICIENT = 1.0e-7


@dataclass(frozen=True)
class NonorthogonalLaplacianCoefficients:
    """Fortran's ``grid%b_lap``, ``lap_dir``, ``lap_neig`` and ``lap_dir_step``.

    ``original`` is ``a_lap`` (pure then mixed coefficients in ``u``
    coordinates).  ``coefficients`` is ``b_lap``: three for the lattice
    directions, then one per auxiliary direction.  ``directions[k]`` is the
    integer offset ``o_k`` of auxiliary direction ``k`` (Fortran
    ``lap_neig(:,k)``), ``active[k]`` its ``lap_dir`` flag (``0`` unused;
    ``1`` used; in the one-plane case ``1`` for ``u+v`` and ``-1`` for
    ``u-v``) and ``steps[k]`` its grid step in bohr.
    """

    original: np.ndarray
    coefficients: np.ndarray
    directions: np.ndarray
    active: tuple[int, int, int]
    steps: np.ndarray

    @property
    def number_of_directions(self) -> int:
        """``grid%lap_dir_num``."""
        return int(sum(abs(flag) for flag in self.active))

    def stencil_terms(
        self, spacings: np.ndarray
    ) -> list[tuple[np.ndarray, float, float]]:
        """``(offset, b, step)`` of every second difference of the operator.

        The three lattice directions come first, then the active auxiliary
        directions in Fortran's order (the neighbour order of ``setup.F90``).
        """

        spacings = np.asarray(spacings, dtype=float)
        terms = [
            (np.eye(3, dtype=np.int64)[axis], float(self.coefficients[axis]), float(spacings[axis]))
            for axis in range(3)
        ]
        for index in range(3):
            if self.active[index] != 0:
                terms.append(
                    (
                        np.asarray(self.directions[index], dtype=np.int64),
                        float(self.coefficients[3 + index]),
                        float(self.steps[index]),
                    )
                )
        return terms


def _normalized_columns(lattice_vectors: np.ndarray) -> list[list[float]]:
    """``avec_tmp``: Fortran's ``latt_vec`` (vectors as columns), unit columns.

    Written with scalar sums in Fortran's order (``sqrt(sum(a(:,i)**2))``) so
    that equal lengths stay equal and the neighbour ordering below breaks
    ties as Fortran does.
    """

    rows = np.asarray(lattice_vectors, dtype=float)
    matrix = [[0.0] * 3 for _ in range(3)]
    for column in range(3):
        vector = [float(value) for value in rows[column]]
        norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
        for row in range(3):
            matrix[row][column] = vector[row] / norm
    return matrix


def _cofactor_inverse(matrix: list[list[float]]) -> tuple[list[list[float]], float]:
    """``mtrxin``/``mtrxin1``: inverse by cofactors and the determinant."""

    m = matrix
    a = [[0.0] * 3 for _ in range(3)]
    a[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1]
    a[1][0] = -m[1][0] * m[2][2] + m[1][2] * m[2][0]
    a[2][0] = m[1][0] * m[2][1] - m[1][1] * m[2][0]
    a[0][1] = -m[0][1] * m[2][2] + m[0][2] * m[2][1]
    a[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0]
    a[2][1] = -m[0][0] * m[2][1] + m[0][1] * m[2][0]
    a[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1]
    a[1][2] = -m[0][0] * m[1][2] + m[0][2] * m[1][0]
    a[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    determinant = m[0][0] * a[0][0] + m[0][1] * a[1][0] + m[0][2] * a[2][0]
    if abs(determinant) < _SINGULAR_DETERMINANT:
        return [], determinant
    return [[a[i][j] / determinant for j in range(3)] for i in range(3)], determinant


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """``matvec3('N', ...)``: ``dot_product(mat(i,:), vec)`` for each row."""

    return [
        matrix[i][0] * vector[0] + matrix[i][1] * vector[1] + matrix[i][2] * vector[2]
        for i in range(3)
    ]


def _bubble_sort(values: list[float], indices: list[int]) -> None:
    """Fortran's in-place ``bubble_sort`` (a selection sort with swaps).

    Ties keep the order this exchange produces, which decides which of
    several equally short neighbours (e.g. in an fcc cell) are chosen.
    """

    n = len(values)
    for j in range(n - 1):
        minimum = values[j]
        for i in range(j + 1, n):
            if values[i] < minimum:
                index, value = indices[i], values[i]
                indices[i], values[i] = indices[j], values[j]
                indices[j], values[j] = index, value
                minimum = value


def _laplacian_metric(normalized: list[list[float]]) -> list[float]:
    """``a_lap`` from ``ainvdot = ainv ainv^T`` (``pbc_grid_coefs.f90:136-152``)."""

    inverse, _ = _cofactor_inverse(normalized)
    if not inverse:
        raise ValueError("the normalized lattice vectors are linearly dependent")
    dot = [
        [
            inverse[i][0] * inverse[j][0]
            + inverse[i][1] * inverse[j][1]
            + inverse[i][2] * inverse[j][2]
            for j in range(3)
        ]
        for i in range(3)
    ]
    return [
        dot[0][0],
        dot[1][1],
        dot[2][2],
        dot[0][1] + dot[1][0],
        dot[0][2] + dot[2][0],
        dot[1][2] + dot[2][1],
    ]


def _coefficients_one_plane(
    normalized: list[list[float]],
    a_lap: list[float],
    spacings: np.ndarray,
    plane: int,
) -> NonorthogonalLaplacianCoefficients:
    """``pbc_grid_coefs_2d``: one non-orthogonal pair, one auxiliary direction.

    ``plane`` is Fortran's ``plane_num`` (3: u-v, 2: u-w, 1: v-w).  For an
    acute pair the direction ``u - r v`` is used, for an obtuse one
    ``u + r v`` (``pbc_grid_coefs.f90:572-669``).
    """

    h1, h2, h3 = (float(value) for value in spacings)
    dflag = [0, 0, 0]
    mflag = [0, 0, 0]
    if plane == 3:
        dflag[0] = dflag[1] = 1
        mflag[0], mflag[1] = 1, -1
    elif plane == 2:
        dflag[0] = dflag[2] = 1
        mflag[0], mflag[2] = 1, -1
    else:
        dflag[1] = dflag[2] = 1
        mflag[1], mflag[2] = 1, -1
    columns = [[normalized[row][column] for row in range(3)] for column in range(3)]

    def dot(first: int, second: int) -> float:
        a, b = columns[first], columns[second]
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    cosines = [dot(0, 1), dot(0, 2), dot(1, 2)]
    a_steps = [h1, h1, h2]
    b_steps = [h2, h3, h3]
    pair = 3 - plane  # Fortran i = 3 - plane_num + 1, here 0-based
    a_step, b_step = a_steps[pair], b_steps[pair]
    ratio = b_step / a_step
    cosine = cosines[pair]
    mixed = a_lap[3 + pair]
    b_lap = [0.0] * 6
    if cosine > 0:
        flag = -1
        step = math.sqrt(a_step**2 + b_step**2 - 2 * a_step * b_step * cosine)
        kval = mixed * (1 + ratio**2 - 2 * ratio * cosine)
        b_lap[3] = -kval / 2 / ratio
    else:
        flag = 1
        step = math.sqrt(a_step**2 + b_step**2 + 2 * a_step * b_step * cosine)
        kval = mixed * (1 + ratio**2 + 2 * ratio * cosine)
        b_lap[3] = kval / 2 / ratio
    b_lap[0] = a_lap[0] - dflag[0] * (dflag[1] + dflag[2]) * flag * mixed / 2 / ratio
    b_lap[1] = (
        a_lap[1]
        - dflag[1] * dflag[0] * ratio * flag * mixed / 2
        - dflag[1] * dflag[2] * flag * mixed / 2 / ratio
    )
    b_lap[2] = a_lap[2] - dflag[2] * (dflag[0] + dflag[1]) * ratio * flag * mixed / 2
    directions = np.zeros((3, 3), dtype=np.int64)
    directions[0] = dflag if flag > 0 else mflag
    return NonorthogonalLaplacianCoefficients(
        original=np.asarray(a_lap, dtype=float),
        coefficients=np.asarray(b_lap, dtype=float),
        directions=directions,
        active=(flag, 0, 0),
        steps=np.array([step, 0.0, 0.0]),
    )


def _coefficients_general(
    normalized: list[list[float]],
    a_lap: list[float],
    spacings: np.ndarray,
    step_ratio_quirk: bool,
) -> NonorthogonalLaplacianCoefficients:
    """Three auxiliary directions among the nearest neighbours (l.197-351)."""

    h1, h2, h3 = (float(value) for value in spacings)
    kv = h2 / h1
    kw = h3 / h2 if step_ratio_quirk else h3 / h1
    offsets: list[tuple[int, int, int]] = []
    scaled: list[list[float]] = []
    displaced: list[list[float]] = []
    lengths: list[float] = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                offsets.append((i, j, k))
                opg = [float(i), kv * j, kw * k]
                opp = _matvec(normalized, opg)
                scaled.append(opg)
                displaced.append(opp)
                lengths.append(math.sqrt(opp[0] * opp[0] + opp[1] * opp[1] + opp[2] * opp[2]))
    order = list(range(27))
    _bubble_sort(lengths, order)

    chosen: list[int] = []
    bb = None
    uu = None
    steps = None
    for position in range(1, 27):
        candidate = order[position]
        offset = offsets[candidate]
        if sum(abs(value) for value in offset) > 1:
            opposite = any(
                sum((offsets[previous][axis] + offset[axis]) ** 2 for axis in range(3)) < 0.0001
                for previous in chosen
            )
            if not chosen or not opposite:
                chosen.append(candidate)
        if len(chosen) == 3:
            vectors = []
            steps = []
            for index in chosen:
                norm = math.sqrt(sum(value * value for value in displaced[index]))
                vectors.append([value / norm for value in scaled[index]])
                steps.append(norm * h1)
            uu = vectors  # uu[k][i]: component i of direction k
            mt = [
                [2 * uu[k][0] * uu[k][1] for k in range(3)],
                [2 * uu[k][0] * uu[k][2] for k in range(3)],
                [2 * uu[k][1] * uu[k][2] for k in range(3)],
            ]
            inverse, _ = _cofactor_inverse(mt)
            if not inverse:
                # "mt matrix is singular, replacing last neighbor"
                chosen.pop()
                continue
            bb = _matvec(inverse, a_lap[3:6])
            break
    if bb is None:
        raise ValueError(
            "no three auxiliary Laplacian directions among the nearest grid "
            "neighbours remove the mixed derivatives of this cell"
        )
    b_lap = [
        a_lap[i] - (bb[0] * uu[0][i] ** 2 + bb[1] * uu[1][i] ** 2 + bb[2] * uu[2][i] ** 2)
        for i in range(3)
    ] + list(bb)
    active = tuple(0 if abs(value) < _INACTIVE_COEFFICIENT else 1 for value in bb)
    return NonorthogonalLaplacianCoefficients(
        original=np.asarray(a_lap, dtype=float),
        coefficients=np.asarray(b_lap, dtype=float),
        directions=np.asarray([offsets[index] for index in chosen], dtype=np.int64),
        active=active,
        steps=np.asarray(steps, dtype=float),
    )


def pbc_grid_coefficients(
    lattice_vectors: np.ndarray,
    spacings: np.ndarray,
    *,
    tolerance: float = ORTHOGONALITY_TOLERANCE,
    step_ratio_quirk: bool = False,
) -> NonorthogonalLaplacianCoefficients:
    """Coefficients of the Laplacian on a general periodic grid.

    Translation of ``pbc_grid_coefs`` and ``pbc_grid_coefs_2d``
    (F ``pbc_grid_coefs.f90``).  ``lattice_vectors`` are rows in bohr,
    ``spacings`` the grid steps along them.  With every mixed coefficient
    below ``tolerance`` the cell is orthogonal and no auxiliary direction is
    used; with exactly one non-zero mixed coefficient one auxiliary
    direction in that plane; otherwise three among the 26 nearest neighbours,
    shortest first, skipping axis offsets, opposites of chosen ones and a
    third one that makes the system for their weights singular.

    ``step_ratio_quirk=True`` reproduces Fortran's ``kw = h3/h2``
    (see the module docstring); it exists to check the translation against
    Fortran's printed numbers.
    """

    spacings = np.asarray(spacings, dtype=float)
    normalized = _normalized_columns(lattice_vectors)
    a_lap = _laplacian_metric(normalized)
    zero_mixed = sum(1 for index in range(3) if abs(a_lap[3 + index]) < tolerance)
    if zero_mixed == 3:
        return NonorthogonalLaplacianCoefficients(
            original=np.asarray(a_lap, dtype=float),
            coefficients=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            directions=np.zeros((3, 3), dtype=np.int64),
            active=(0, 0, 0),
            steps=np.zeros(3),
        )
    if zero_mixed == 2:
        plane = 0
        for index in range(3):
            if abs(a_lap[3 + index]) > tolerance:
                plane = 3 - index  # Fortran lap_orth_dir = 3 - i + 1, i 1-based
        return _coefficients_one_plane(normalized, a_lap, spacings, plane)
    return _coefficients_general(normalized, a_lap, spacings, step_ratio_quirk)


def grid_laplacian_coefficients(grid) -> NonorthogonalLaplacianCoefficients:
    """:func:`pbc_grid_coefficients` of a :class:`~parsec_python.Grid.pbc.GeneralPeriodicGrid`."""

    return pbc_grid_coefficients(grid.cell.lattice_vectors, grid.spacings)


def build_general_negative_laplacian(
    grid, coefficients: NonorthogonalLaplacianCoefficients | None = None
) -> sp.csr_matrix:
    """Sparse ``-nabla^2`` on a general periodic grid (Fortran ``lapmvs``).

    Every term of :meth:`NonorthogonalLaplacianCoefficients.stencil_terms`
    contributes ``-b/step^2 * [c_0 u_r + sum_s c_s (u_(r-s o) + u_(r+s o))]``
    with the centred weights ``c_s`` of
    :func:`~.finite_difference.second_derivative_coefficients`; offsets wrap
    around the cell.  Entries that wrap onto the same point (a short cell)
    are summed.

    On the grid of a wire or slab
    (:class:`~parsec_python.Grid.confined.ConfinedPeriodicGrid`) a neighbour
    outside the domain has no entry: the orbitals vanish there, and the
    Hartree solver adds the boundary values to its right-hand side.
    """

    if coefficients is None:
        coefficients = grid_laplacian_coefficients(grid)
    weights = second_derivative_coefficients(grid.settings.expansion_order)
    width = grid.settings.stencil_half_width
    n = grid.size
    base = grid.integer_coordinates
    terms = coefficients.stencil_terms(grid.spacings)

    center = 0.0
    for _offset, coefficient, step in terms:
        center += -coefficient * weights[width] / step**2
    rows: list[np.ndarray] = [np.arange(n, dtype=np.int64)]
    columns: list[np.ndarray] = [np.arange(n, dtype=np.int64)]
    values: list[np.ndarray] = [np.full(n, center, dtype=float)]
    every_row = np.arange(n, dtype=np.int64)
    for offset, coefficient, step in terms:
        for shell in range(1, width + 1):
            value = -coefficient * weights[width + shell] / step**2
            for sign in (-1, 1):
                neighbours = grid.rows_for_integer_coordinates(base + sign * shell * offset)
                present = neighbours >= 0
                if not np.all(present):
                    # Only a wire or slab grid has neighbours outside.
                    rows.append(every_row[present])
                    columns.append(neighbours[present])
                    values.append(np.full(int(np.count_nonzero(present)), value, dtype=float))
                    continue
                rows.append(every_row)
                columns.append(neighbours)
                values.append(np.full(n, value, dtype=float))
    matrix = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(n, n),
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def build_lattice_derivatives(grid) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Centred first derivatives ``d/du_i`` along the three lattice directions."""

    weights = first_derivative_coefficients(grid.settings.expansion_order)
    width = grid.settings.stencil_half_width
    n = grid.size
    base = grid.integer_coordinates
    every_row = np.arange(n, dtype=np.int64)
    matrices = []
    for axis in range(3):
        rows, columns, values = [], [], []
        offset = np.eye(3, dtype=np.int64)[axis]
        for shell in range(-width, width + 1):
            if shell == 0:
                continue
            neighbours = grid.rows_for_integer_coordinates(base + shell * offset)
            present = neighbours >= 0
            if not np.all(present):
                # A wire or slab: no entry for a neighbour outside the domain.
                rows.append(every_row[present])
                columns.append(neighbours[present])
                values.append(
                    np.full(
                        int(np.count_nonzero(present)),
                        weights[width + shell] / grid.spacings[axis],
                    )
                )
                continue
            rows.append(every_row)
            columns.append(neighbours)
            values.append(np.full(n, weights[width + shell] / grid.spacings[axis]))
        matrix = sp.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(n, n),
        ).tocsr()
        matrix.sum_duplicates()
        matrices.append(matrix)
    return tuple(matrices)


def build_general_gradient(grid) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Cartesian ``(d/dx, d/dy, d/dz)`` on a general grid (``gradmvs``).

    ``d/dx_c = sum_i inv(A_n)^T[c, i] d/du_i``.  Used by the k-point kinetic
    term, ``-2i k.grad = -2i sum_i (inv(A_n) k)_i d/du_i``, which is
    Fortran's ``kecoe1`` (``init_var.F90:357-362``).
    """

    lattice = build_lattice_derivatives(grid)
    transform = np.asarray(grid.gradient_transform, dtype=float)
    n = grid.size
    result = []
    for component in range(3):
        matrix = sp.csr_matrix((n, n))
        for axis in range(3):
            if transform[component, axis] != 0.0:
                matrix = matrix + transform[component, axis] * lattice[axis]
        result.append(matrix.tocsr())
    return tuple(result)


def cartesian_gradient(values: np.ndarray, grid, weights: np.ndarray) -> np.ndarray:
    """Cartesian gradient ``(3, n)`` of a grid field on a general periodic grid.

    ``weights`` are the centred first-derivative weights ``d_-M .. d_M``
    (dimensionless); each lattice direction divides by its own step, the
    stencil wraps around the cell, and ``inv(A_n)^T`` makes the result
    Cartesian.  This is the GGA gradient of ``gradmvs``.

    On a wire or slab grid the field is zero outside the domain along the
    confined directions (the cluster convention) and wraps along the
    periodic ones.
    """

    if getattr(grid, "is_confined_periodic", False):
        return _confined_cartesian_gradient(values, grid, weights)
    box = np.asarray(values, dtype=np.float64).reshape(grid.shape)
    along = np.empty((3, grid.size), dtype=np.float64)
    for axis in range(3):
        along[axis] = correlate1d(
            box, weights / grid.spacings[axis], axis=axis, mode="wrap"
        ).reshape(-1)
    return np.asarray(grid.gradient_transform, dtype=float) @ along


def cartesian_gradient_transpose(fluxes: np.ndarray, grid, weights: np.ndarray) -> np.ndarray:
    """Exact transpose of :func:`cartesian_gradient` applied to ``(3, n)`` fluxes.

    ``D^T F = sum_i D_i^T (inv(A_n) F)_i`` with ``D_i^T = -D_i`` for the
    antisymmetric wrapped stencil, so ``-D^T F`` is the discrete divergence.
    """

    if getattr(grid, "is_confined_periodic", False):
        return _confined_cartesian_gradient_transpose(fluxes, grid, weights)
    along = np.asarray(grid.gradient_transform, dtype=float).T @ np.asarray(
        fluxes, dtype=np.float64
    )
    total = np.zeros(grid.size, dtype=np.float64)
    for axis in range(3):
        total += correlate1d(
            along[axis].reshape(grid.shape),
            -weights / grid.spacings[axis],
            axis=axis,
            mode="wrap",
        ).reshape(-1)
    return total


def _confined_cartesian_gradient(values: np.ndarray, grid, weights: np.ndarray) -> np.ndarray:
    """:func:`cartesian_gradient` on a wire or slab grid (zero outside the domain)."""

    box = grid.to_box(np.asarray(values, dtype=np.float64))
    modes = grid.box_boundary_modes
    along = np.empty((3, grid.size), dtype=np.float64)
    for axis in range(3):
        along[axis] = grid.from_box(
            correlate1d(
                box, weights / grid.spacings[axis], axis=axis, mode=modes[axis], cval=0.0
            )
        )
    return np.asarray(grid.gradient_transform, dtype=float) @ along


def _confined_cartesian_gradient_transpose(
    fluxes: np.ndarray, grid, weights: np.ndarray
) -> np.ndarray:
    """Exact transpose of :func:`_confined_cartesian_gradient`.

    The zero-padded stencil is antisymmetric on the index box, and gathering
    the active points is the transpose of embedding them, so the transpose
    applies the reversed weights to the embedded fluxes.
    """

    along = np.asarray(grid.gradient_transform, dtype=float).T @ np.asarray(
        fluxes, dtype=np.float64
    )
    modes = grid.box_boundary_modes
    total = np.zeros(grid.size, dtype=np.float64)
    for axis in range(3):
        total += grid.from_box(
            correlate1d(
                grid.to_box(along[axis]),
                -weights / grid.spacings[axis],
                axis=axis,
                mode=modes[axis],
                cval=0.0,
            )
        )
    return total


__all__ = [
    "NonorthogonalLaplacianCoefficients",
    "ORTHOGONALITY_TOLERANCE",
    "build_general_gradient",
    "build_general_negative_laplacian",
    "build_lattice_derivatives",
    "cartesian_gradient",
    "cartesian_gradient_transpose",
    "grid_laplacian_coefficients",
    "pbc_grid_coefficients",
]
