"""Point-group operations as permutations of the points of a cluster grid.

``symm_scalar`` of ``newrho.F90`` (lines 195-351) symmetrizes a scalar field,
the charge density of ``Symmetrize_Charge_Density``, over the full point
group of the system.  For each grid point ``r`` and operation ``g`` it adds
``f(r)`` to the point ``g r`` and divides by the number of operations:

``f_sym(r) = (1/|G|) sum_g f(g^-1 r) = (1/|G|) sum_g f(g r)``

(the group holds the inverse of each of its operations).  An operation of an
isolated system is one of the 48 signed permutations of the Cartesian axes
(:func:`parsec_python.Forces.symmetry.cubic_point_operations`).  It maps the
point ``(k + s) h`` to ``(R (k + s)) h``, which is the point of integer
index ``R (k + s) - s``: a grid point when the shift ``s`` is zero or one half
in every direction, as it is for the grids of this package, and inside the
domain because a sphere is invariant under every operation.

Fortran works on the irreducible wedge of the reduced (Abelian) group and
sums over the full group; this module works on the full grid, which needs no
wedge and gives the same field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridPointGroup:
    """Operations of a point group and the grid permutations they induce.

    ``matrices`` has shape ``(operations, 3, 3)``; ``images[g, i]`` is the
    row of the grid point ``matrices[g] @ r_i``.  The identity comes first.
    """

    matrices: np.ndarray
    images: np.ndarray

    @property
    def order(self) -> int:
        return int(self.matrices.shape[0])

    @property
    def reduced_order(self) -> int:
        """Operations with a diagonal matrix: PARSEC's Abelian subgroup.

        A cluster calculation of PARSEC stores its fields on the irreducible
        wedge of the operations that only change the signs of coordinates.
        """
        off_diagonal = self.matrices - np.einsum("gii->gi", self.matrices)[:, :, None] * np.eye(3)
        return int(np.count_nonzero(np.all(np.abs(off_diagonal) < 1.0e-12, axis=(1, 2))))


@dataclass(frozen=True)
class ScalarSymmetrization:
    """What ``symm_scalar`` prints about one symmetrized field.

    ``maximum_deviation_percent`` is ``max |f_sym - f| / sum f * 100`` as
    ``newrho.F90`` writes it.  Fortran's sum runs over the irreducible wedge
    of the reduced group, a sum over the full grid divided by the order of
    that group: the full-grid sum here is divided by
    :attr:`GridPointGroup.reduced_order` to print the same number.
    ``charge_before`` and ``charge_after`` are ``h^3 sum f`` before and
    after.
    """

    maximum_deviation_percent: float
    charge_before: float
    charge_after: float


def grid_point_group(grid, matrices: np.ndarray) -> GridPointGroup:
    """Map the operations ``matrices`` ``(operations, 3, 3)`` onto ``grid``.

    ``grid`` is a :class:`parsec_python.Grid.RealSpaceGrid`.  Raises
    ``ValueError`` where an operation does not map the grid onto itself.
    """

    matrices = np.asarray(matrices, dtype=np.float64).reshape(-1, 3, 3)
    shift = np.asarray(grid.settings.shift, dtype=np.float64)
    shifted = np.asarray(grid.integer_coordinates, dtype=np.float64) + shift
    images = np.empty((matrices.shape[0], grid.size), dtype=np.int64)
    for index, matrix in enumerate(matrices):
        target = shifted @ matrix.T - shift
        rounded = np.rint(target)
        if np.max(np.abs(target - rounded), initial=0.0) > 1.0e-9:
            raise ValueError(
                "a symmetry operation does not map the grid points onto grid "
                "points (the grid shift must be 0 or 1/2 in every direction)"
            )
        rows = grid.rows_for_integer_coordinates(rounded.astype(np.int64))
        if np.any(rows < 0):
            raise ValueError(
                "a symmetry operation maps grid points outside the domain; "
                "the density can be symmetrized on a sphere only"
            )
        images[index] = rows
    return GridPointGroup(matrices=matrices, images=images)


def symmetrize_scalar(values: np.ndarray, group: GridPointGroup) -> np.ndarray:
    """Return ``(1/|G|) sum_g f(g r)`` of ``values`` ``(points,)``."""

    values = np.asarray(values, dtype=np.float64)
    if group.order <= 1:
        return values.copy()
    total = np.zeros_like(values)
    for images in group.images:
        total += values[images]
    return total / float(group.order)


def symmetrize_density(
    density: np.ndarray, group: GridPointGroup, volume_element: float
) -> tuple[np.ndarray, ScalarSymmetrization]:
    """Symmetrize a density and report it as ``symm_scalar`` does."""

    density = np.asarray(density, dtype=np.float64)
    symmetric = symmetrize_scalar(density, group)
    before = float(np.sum(density))
    wedge_sum = before / group.reduced_order
    deviation = float(np.max(np.abs(symmetric - density), initial=0.0))
    return symmetric, ScalarSymmetrization(
        maximum_deviation_percent=(deviation / wedge_sum * 100.0 if before else 0.0),
        charge_before=before * float(volume_element),
        charge_after=float(np.sum(symmetric)) * float(volume_element),
    )


__all__ = [
    "GridPointGroup",
    "ScalarSymmetrization",
    "grid_point_group",
    "symmetrize_density",
    "symmetrize_scalar",
]
