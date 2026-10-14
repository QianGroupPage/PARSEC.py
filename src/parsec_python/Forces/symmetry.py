"""Point-group symmetrization of atomic forces (``symm_forces.f90``).

A uniform Cartesian grid imposes at most cubic symmetry, so PARSEC looks for
the symmetry operations of an isolated system among the 48 operations of the
cube: the signed permutations of the Cartesian axes, acting about the origin.
An operation belongs to the system when it maps every atom onto an atom of
the same species.  Forces are then replaced by their average over the group,

``F_sym(a) = (1/|G|) * sum_g  U_g F(g^-1 a)``,

which removes the noise that would otherwise break the symmetry of the
structure during a relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..models import Atom


#: ``tol`` of ``symm_forces.f90``: an atom and its image coincide within it.
POSITION_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class AtomSymmetryOperation:
    """One orthogonal operation and the atom permutation it induces.

    ``matrix @ position[a]`` is the position of atom ``image[a]``.
    """

    matrix: np.ndarray
    image: np.ndarray


def cubic_point_operations() -> np.ndarray:
    """The 48 signed permutation matrices, identity first."""

    operations = []
    for order in permutations(range(3)):
        for signs in product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(order, signs)):
                matrix[row, column] = sign
            operations.append(matrix)
    return np.asarray(operations)


def atom_symmetry_operations(
    atoms: Sequence[Atom],
    *,
    tolerance: float = POSITION_TOLERANCE,
    candidates: np.ndarray | None = None,
) -> tuple[AtomSymmetryOperation, ...]:
    """Find the cubic operations that map the labeled atoms onto themselves.

    The identity is always returned first.  ``candidates`` replaces the 48
    cubic operations, for a caller that keeps a smaller group.
    """

    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    symbols = [atom.symbol for atom in atoms]
    matrices = cubic_point_operations() if candidates is None else np.asarray(candidates)
    by_species: dict[str, np.ndarray] = {}
    for symbol in dict.fromkeys(symbols):
        by_species[symbol] = np.flatnonzero([s == symbol for s in symbols])

    trees = {
        symbol: cKDTree(positions[members]) for symbol, members in by_species.items()
    }
    found: list[AtomSymmetryOperation] = []
    for matrix in matrices:
        rotated = positions @ matrix.T
        image = np.full(len(atoms), -1, dtype=np.int64)
        valid = True
        for symbol, members in by_species.items():
            distance, nearest = trees[symbol].query(rotated[members])
            # Every image lies on an atom of the species and no atom is the
            # image of two: the operation permutes the species.
            if np.any(distance > tolerance) or np.unique(nearest).size != members.size:
                valid = False
                break
            image[members] = members[nearest]
        if valid:
            found.append(AtomSymmetryOperation(matrix=matrix.copy(), image=image))
    if not found or not np.array_equal(found[0].matrix, np.eye(3)):
        raise RuntimeError("the identity must be a symmetry operation")
    return tuple(found)


def symmetrize_forces(
    forces: np.ndarray, operations: Sequence[AtomSymmetryOperation]
) -> np.ndarray:
    """Average ``(atoms, 3)`` forces over a group of operations."""

    forces = np.asarray(forces, dtype=np.float64)
    if len(operations) <= 1:
        return forces.copy()
    total = np.zeros_like(forces)
    for operation in operations:
        total[operation.image] += forces @ operation.matrix.T
    return total / float(len(operations))


def symmetrize_positions(
    positions: np.ndarray, operations: Sequence[AtomSymmetryOperation]
) -> np.ndarray:
    """Average ``(atoms, 3)`` positions over a group of operations.

    Positions transform like forces.  Averaging them removes round-off that
    accumulates over many ionic steps, so that the structure keeps the group
    it started with.
    """

    return symmetrize_forces(positions, operations)


def operations_preserving(
    operations: Sequence[AtomSymmetryOperation], flags: Sequence[bool]
) -> tuple[AtomSymmetryOperation, ...]:
    """The operations that map every flagged atom onto a flagged atom.

    A relaxation that moves only some atoms keeps a symmetry operation only
    if the operation never exchanges a movable atom with a fixed one.
    """

    mask = np.asarray(flags, dtype=bool)
    return tuple(
        operation
        for operation in operations
        if np.array_equal(mask[operation.image], mask)
    )


def remove_net_force(forces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the mean force; return the result and the net force."""

    forces = np.asarray(forces, dtype=np.float64)
    net = forces.sum(axis=0)
    return forces - net / float(forces.shape[0]), net


def total_torque(positions: np.ndarray, forces: np.ndarray) -> np.ndarray:
    """``sum_a R_a x F_a`` about the origin, in Ry."""

    return np.cross(np.asarray(positions), np.asarray(forces)).sum(axis=0)


__all__ = [
    "AtomSymmetryOperation",
    "POSITION_TOLERANCE",
    "atom_symmetry_operations",
    "cubic_point_operations",
    "operations_preserving",
    "remove_net_force",
    "symmetrize_forces",
    "symmetrize_positions",
    "total_torque",
]
