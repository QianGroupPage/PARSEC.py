"""Geometry of a periodic cell: lattice metrics, super cells, atoms in the cell.

Lattice vectors are the *rows* of a ``(3, 3)`` array in bohr (the convention
of :class:`~parsec_python.models.PeriodicCell`); Fortran PARSEC stores them as
the columns of ``pbc%latt_vec``.  A position ``r`` has fractional (lattice)
coordinates ``f`` with ``r = f @ lattice_vectors``.

These are pure geometry helpers shared by the input translator (``Super_Cell``,
``Super_Cell_Vac``, ``Coordinate_Unit: lattice_vectors``) and the periodic grid
and SCF preparation (atoms outside the cell, image sums).  They do not print
or open files.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def is_diagonal_lattice(lattice_vectors: np.ndarray, *, atol: float = 1.0e-10) -> bool:
    """Whether the lattice vectors lie along the Cartesian axes.

    This is the test the orthorhombic code paths (grid, k-points, projector
    images) have always used; it is kept as one function so every module
    sends exactly the same cells to the general-lattice code.
    """

    vectors = np.asarray(lattice_vectors, dtype=float)
    off_diagonal = vectors - np.diag(np.diag(vectors))
    return bool(np.allclose(off_diagonal, 0.0, atol=atol))


def reciprocal_lattice(lattice_vectors: np.ndarray) -> np.ndarray:
    """Rows ``b_i`` with ``a_i . b_j = 2*pi*delta_ij`` (F ``inipbc.f90:64-71``)."""

    return 2.0 * np.pi * np.linalg.inv(np.asarray(lattice_vectors, dtype=float)).T


def lattice_plane_spacings(lattice_vectors: np.ndarray) -> np.ndarray:
    """Distance between neighbouring lattice planes of each family, ``2*pi/|b_i|``.

    For a diagonal lattice these are the side lengths.
    """

    return 2.0 * np.pi / np.linalg.norm(reciprocal_lattice(lattice_vectors), axis=1)


def cell_diagonal_length(lattice_vectors: np.ndarray) -> float:
    """Largest distance between two points of the cell (its longest diagonal).

    Two points with fractional coordinates in ``[0, 1)`` differ by
    ``d @ A`` with ``d`` in ``(-1, 1)^3``; the norm is convex, so its largest
    value is reached at a corner ``d = (+-1, +-1, +-1)``.
    """

    vectors = np.asarray(lattice_vectors, dtype=float)
    corners = np.array(
        [[i, j, k] for i in (-1.0, 1.0) for j in (-1.0, 1.0) for k in (-1.0, 1.0)]
    )
    return float(np.max(np.linalg.norm(corners @ vectors, axis=1)))


def fractional_coordinates(positions: np.ndarray, lattice_vectors: np.ndarray) -> np.ndarray:
    """Lattice coordinates ``f`` of Cartesian positions, ``r = f @ A``."""

    positions = np.asarray(positions, dtype=float)
    vectors = np.asarray(lattice_vectors, dtype=float)
    return np.linalg.solve(vectors.T, positions.T).T


def back_to_cell(position: np.ndarray, lattice_vectors: np.ndarray, periodicity: int = 3) -> np.ndarray:
    """Translate one position by lattice vectors to fractional coordinates about ``[-1/2, 1/2]`` (PARSEC 1.5).

    ``back_to_cell`` of ``scf/initial.f90:762-820``, which PARSEC 1.5 applies to every periodic input at start
    (``initial.f90:326``; the previous tree after moves only): ``u = A^-1 r``, ``u_i -= nint(u_i)`` along the
    ``periodicity`` periodic directions, ``r = A u``.  Here the translation is ``r - n_1 a_1 - n_2 a_2 - n_3 a_3``
    with ``n_i = floor(u_i + 1/2)`` (the arithmetic of :func:`parsec_python.SCF.wire_slab.wire_slab_back_to_cell`,
    which wires and slabs apply after moves), so every ``u`` lands in ``[-1/2, 1/2)``: Fortran's ``nint`` rounds
    halves away from zero and swaps ``u = 1/2`` and ``-1/2``, which would move an atom of a written input
    (``bak_complete_parsec.in``) once more when it is read back; the two are the same position modulo the lattice.
    A position with ``n = 0`` is returned unchanged (the same array), where Fortran's round trip ``A (A^-1 r)``
    changes its last bits: inside the cell no coordinate moves.  A translation by lattice vectors changes no
    periodic sum.
    """

    position = np.asarray(position, dtype=float)
    vectors = np.asarray(lattice_vectors, dtype=float)
    fraction = position @ np.linalg.inv(vectors)
    steps = [int(np.floor(float(fraction[axis]) + 0.5)) for axis in range(periodicity)]
    if not any(steps):
        return position
    moved = position.copy()
    for axis, step in enumerate(steps):
        if step:
            moved = moved - float(step) * vectors[axis]
    return moved


def wrap_into_cell(position: np.ndarray, lattice_vectors: np.ndarray) -> np.ndarray:
    """Translate one position by lattice vectors into the cell ``f in [0, 1)^3``.

    A position already inside the cell is returned unchanged (the same array
    values, not a round trip through fractional coordinates), so inputs whose
    atoms lie in the cell keep their exact coordinates.  For a diagonal
    lattice "inside" is the test ``0 <= x_i < L_i`` the periodic preparation
    has always applied; for a general lattice it is ``0 <= f_i < 1``.

    The translation is ``r - n @ A`` with integer ``n = floor(f)``.  All the
    periodic sums (Ewald, projector and density images) run over lattice
    images, so this changes no physical quantity; it only lets the image
    bounds assume atoms in the cell.
    """

    position = np.asarray(position, dtype=float)
    vectors = np.asarray(lattice_vectors, dtype=float)
    if is_diagonal_lattice(vectors):
        sides = np.diag(vectors)
        if np.all(position >= 0.0) and np.all(position < sides):
            return position
    fraction = fractional_coordinates(position[None, :], vectors)[0]
    if np.all(fraction >= 0.0) and np.all(fraction < 1.0):
        return position
    wrapped = position - np.floor(fraction) @ vectors
    # A coordinate a rounding error below a cell face lands on the face
    # itself: move it once more so that every fraction lies in [0, 1).
    fraction = fractional_coordinates(wrapped[None, :], vectors)[0]
    correction = np.where(fraction >= 1.0, 1.0, 0.0) - np.where(fraction < 0.0, 1.0, 0.0)
    if np.any(correction != 0.0):
        wrapped = wrapped - correction @ vectors
    return wrapped


def supercell_lattice(
    lattice_vectors: np.ndarray,
    replication: Sequence[int],
    vacuum: Sequence[int],
) -> np.ndarray:
    """Lattice vectors of PARSEC's ``Super_Cell`` and ``Super_Cell_Vac`` blocks.

    ``latt_vec(:,i) *= n_i`` and then ``latt_vec(:,i) *= (1 + vac_i)``
    (F ``usrinputfile.F90:743-745, 770-772``), in that order.
    """

    vectors = np.array(lattice_vectors, dtype=float, copy=True)
    for row, count in enumerate(replication):
        vectors[row] = vectors[row] * float(count)
    for row, extra in enumerate(vacuum):
        vectors[row] = vectors[row] * (1.0 + float(extra))
    return vectors


def primitive_lattice(
    lattice_vectors: np.ndarray,
    replication: Sequence[int],
    vacuum: Sequence[int],
) -> np.ndarray:
    """PARSEC's ``lattvec``: the input cell inside a super cell.

    ``lattvec(:,j) = latt_vec(:,j)/(1 + vac_j)/n_j``
    (F ``usrinputfile.F90:796-799``).  It is the unit of
    ``Coordinate_Unit: lattice_vectors``, of ``Kpoint_Unit:
    reciprocal_lattice_vectors`` and of the replication offsets.
    """

    vectors = np.array(lattice_vectors, dtype=float, copy=True)
    for row in range(3):
        vectors[row] = vectors[row] / (1.0 + float(vacuum[row])) / float(replication[row])
    return vectors


def replicate_positions(
    position: np.ndarray,
    primitive_vectors: np.ndarray,
    replication: Sequence[int],
) -> list[np.ndarray]:
    """Copies of one input atom in a ``Super_Cell``, in PARSEC's order.

    ``acoord = r + i1*a1 + i2*a2 + i3*a3`` for ``i_k = 0 .. n_k-1`` with
    ``i3`` fastest, written right after the input atom
    (F ``usrinputfile.F90:1964-2007``); ``a_k`` are the primitive vectors.
    """

    position = np.asarray(position, dtype=float)
    vectors = np.asarray(primitive_vectors, dtype=float)
    n1, n2, n3 = (int(value) for value in replication)
    copies = []
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                copies.append(
                    position
                    + float(i1) * vectors[0]
                    + float(i2) * vectors[1]
                    + float(i3) * vectors[2]
                )
    return copies


__all__ = [
    "back_to_cell",
    "cell_diagonal_length",
    "fractional_coordinates",
    "is_diagonal_lattice",
    "lattice_plane_spacings",
    "primitive_lattice",
    "reciprocal_lattice",
    "replicate_positions",
    "supercell_lattice",
    "wrap_into_cell",
]
