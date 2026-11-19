"""The operations of the full group of a point-group reduction, as operations on the atoms of the forces.

A :class:`~parsec_python.acceleration.Symmetry.axis_reflection.PointGroupReduction` solves the sectors of a subgroup
``H`` of the group ``G`` of the structure (``full_operations``, with the translations ``full_translations`` of a
periodic cell in grid steps); a sector of a weighted representation solves one partner of each degenerate level and
counts it with its weight.  The nonlocal force of such states is the force of the density that the SCF used only after
an average over ``G``: the states of a level span a representation of ``G``, and the sum of a term over the partners is
``|G|^-1 sum_g g`` of the term of one of them, counted once per partner (Schur).  The local and Ewald terms read the
density, which the SCF has averaged over ``G`` already, so the average of the total over ``G`` changes them by
round-off only and keeps a relaxation in the group.

An operation ``r -> R r + t`` (``t`` in grid steps along the axes of a cell) acts on the atoms as
``R (r_a - tau) = r_b`` modulo the lattice with ``tau = -R^T t``, the representation of
:class:`parsec_python.Forces.periodic.PeriodicAtomSymmetryOperation` (forces turn with ``R`` alone,
:func:`parsec_python.Forces.symmetrize_positions` uses ``tau``).
"""

from __future__ import annotations

import numpy as np

#: Distance within which an image of an atom is that atom (bohr), as for the orbits of the reduction.
ATOM_TOLERANCE = 1.0e-7

#: Distance (bohr) within which an image of a moved atom must fall on an atom of its species for the sectors of the
#: first structure to hold for the moved one (:func:`check_sector_atoms`).  An ionic loop that keeps the group (forces
#: and positions averaged over it, :func:`parsec_python.Forces.symmetrize_positions`) moves the atoms on it to the
#: round-off of the coordinates, about 1e-13 bohr per step in a cell of 100 bohr; the fold into the cell by lattice
#: vectors adds as much.  A displacement of 1e-6 bohr off the group changes the energy by ``|F| * 1e-6`` (below 1e-7
#: Ry) and the forces by ``k * 1e-6`` (about 1e-6 Ry/bohr for the bond stiffness of about 1 Ry/bohr^2), the level of
#: the force tolerance of the SCF loop, which the sectors would no longer resolve.  A real breaking of the group (a
#: manual move, molecular dynamics, an atom held by Move_Flag) is 1e-4 bohr and more.  The cluster path checks its
#: moved atoms to the same distance (:func:`.local_device.sector_atom_images`).  The operations of the forces of a
#: structure that this check accepts match its atoms to the same distance (:func:`sector_group_operations`).
MOVE_TOLERANCE = 1.0e-6


def sector_reduction(system):
    """The :class:`PointGroupReduction` of the sectors of ``system``, or ``None`` (no sectors, or an Abelian group)."""

    from ..Symmetry.axis_reflection import PointGroupReduction

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
    if not isinstance(reduction, PointGroupReduction) or getattr(reduction, "full_operations", None) is None:
        return None
    return reduction


def sector_group_operations(reduction, atoms, grid, lattice_vectors=None):
    """The operations of ``G`` of ``reduction`` acting on ``atoms``, the identity first.

    ``lattice_vectors`` (rows) for a periodic cell: :class:`PeriodicAtomSymmetryOperation` with the images modulo the
    lattice and ``tau = -R^T t``; a cluster gets :class:`parsec_python.Forces.AtomSymmetryOperation`.  Raises
    ``ValueError`` where an operation carries an atom onto no atom of its species (the structure has left the group).
    """

    from parsec_python.Forces import AtomSymmetryOperation
    from parsec_python.Forces.periodic import PeriodicAtomSymmetryOperation

    from ..Symmetry.axis_reflection import group_atom_image_table

    matrices = np.asarray(reduction.full_operations, dtype=np.float64).reshape(-1, 3, 3)
    shifts = getattr(reduction, "full_translations", None)
    shifts = None if shifts is None else np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
    # the rows of H and of the coset representatives, composed (raises where a row is not a permutation); matched to the
    # distance of the check of moved atoms, so that a structure check_sector_atoms accepts has its operations (the
    # nearest atoms, the same table for atoms on the group)
    images = group_atom_image_table(tuple(atoms), reduction, grid, max(ATOM_TOLERANCE, MOVE_TOLERANCE))
    if images is None:
        raise ValueError("an operation of the point group of the sectors carries an atom onto no atom of its species")
    identity = np.flatnonzero(
        [np.array_equal(m, np.eye(3)) and (shifts is None or not np.any(shifts[g])) for g, m in enumerate(matrices)]
    )
    if identity.size != 1:
        raise ValueError("the operations of the point group of the sectors hold no single identity")
    order = [int(identity[0])] + [g for g in range(len(matrices)) if g != identity[0]]
    if lattice_vectors is None:
        return tuple(AtomSymmetryOperation(matrix=matrices[g].copy(), image=images[g].copy()) for g in order)
    vectors = np.asarray(lattice_vectors, dtype=np.float64)
    steps = None
    if shifts is not None:
        steps = np.diag(vectors) / np.asarray(grid.points_per_axis, dtype=np.float64)
    operations = []
    for g in order:
        translation = None
        if shifts is not None and np.any(shifts[g]):
            translation = -matrices[g].T @ (shifts[g] * steps)
        operations.append(
            PeriodicAtomSymmetryOperation(
                matrix=matrices[g].copy(), image=images[g].copy(), lattice_vectors=vectors.copy(), translation=translation
            )
        )
    return tuple(operations)


def sector_force_operations(system):
    """``G`` of the sectors of ``system`` on its atoms (:func:`sector_group_operations`), or ``None``."""

    reduction = sector_reduction(system)
    if reduction is None:
        return None
    cell = getattr(system.input, "periodic_cell", None)
    grid = getattr(reduction, "grid", None)
    if grid is None:
        grid = system.grid
    return sector_group_operations(
        reduction, system.atoms, grid, None if cell is None else np.asarray(cell.lattice_vectors, dtype=np.float64)
    )


def abelian_cell_operations(system):
    """``H`` of the Abelian sectors of a periodic cell on its atoms, the identity first, or ``None``.

    A cell whose sectors are those of an Abelian group (an
    :class:`..Symmetry.axis_reflection.AxisReflectionReduction` or a
    :class:`..Symmetry.axis_reflection.SignedPermutationReduction` of order above one, not a
    :class:`..Symmetry.axis_reflection.PointGroupReduction`) solves every level, so its forces need no completion
    beyond that of the projectors of one atom per orbit (:func:`abelian_orbit_operations`, which takes these); but its
    moved atoms must keep ``H`` (:func:`check_sector_atoms`), and the round-off of its unsymmetrized forces on
    atoms that ``H`` fixes would move them (by about ``Min_Step`` in a steepest-descent step).  These operations
    average its forces and moved positions over ``H`` in the ionic loop (:func:`sector_symmetry_operations`).
    ``None`` for a cluster, a cell without sectors or with those of a point group.
    """

    from parsec_python.Forces.periodic import PeriodicAtomSymmetryOperation

    from ..Symmetry.axis_reflection import AxisReflectionReduction, PointGroupReduction, atom_image_table

    cell = getattr(getattr(system, "input", None), "periodic_cell", None)
    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
    if (
        cell is None
        or not isinstance(reduction, AxisReflectionReduction)
        or isinstance(reduction, PointGroupReduction)
        or int(getattr(reduction, "group_order", 1)) <= 1
    ):
        return None
    operations = getattr(reduction, "operations", None)
    if operations is None:
        operations = [np.diag(signs) for signs in np.asarray(reduction.signs)]
    matrices = np.asarray(operations, dtype=np.float64).reshape(-1, 3, 3)
    shifts = getattr(reduction, "translations", None)
    shifts = None if shifts is None else np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
    grid = getattr(reduction, "grid", None)
    if grid is None:
        grid = system.grid
    atoms = tuple(system.atoms)
    images = atom_image_table(atoms, matrices, shifts, grid, max(ATOM_TOLERANCE, MOVE_TOLERANCE))
    if images is None:
        raise ValueError("an operation of the group of the sectors carries an atom onto no atom of its species")
    identity = np.flatnonzero(
        [np.array_equal(m, np.eye(3)) and (shifts is None or not np.any(shifts[g])) for g, m in enumerate(matrices)]
    )
    if identity.size != 1:
        raise ValueError("the operations of the group of the sectors hold no single identity")
    vectors = np.asarray(cell.lattice_vectors, dtype=np.float64)
    steps = None if shifts is None else np.diag(vectors) / np.asarray(grid.points_per_axis, dtype=np.float64)
    order = [int(identity[0])] + [g for g in range(len(matrices)) if g != identity[0]]
    return tuple(
        PeriodicAtomSymmetryOperation(
            matrix=matrices[g].copy(),
            image=np.asarray(images[g], dtype=np.int64).copy(),
            lattice_vectors=vectors.copy(),
            translation=None if shifts is None or not np.any(shifts[g]) else -matrices[g].T @ (shifts[g] * steps),
        )
        for g in order
    )


def abelian_orbit_operations(system):
    """``H`` of Abelian sectors whose projectors hold one atom per orbit, on the atoms, the identity first, or ``None``.

    Such sectors (:func:`..Symmetry.axis_reflection.attach_abelian_atom_orbits`) contract the projector gradients of
    the representative of every orbit, each ``n_a`` times the term of one atom of its orbit, and none of the other
    atoms (:func:`..Symmetry.axis_reflection.compress_gradients_for_sectors`).  ``H`` is Abelian with real
    one-dimensional characters, so every level is solved and the density matrix of the solved states is invariant
    under it: the term of atom ``h a`` is ``R_h`` times that of ``a``, and that of ``a`` is invariant under the
    operations that fix ``a``.  The average over ``H`` (:func:`parsec_python.Forces.symmetry.symmetrize_forces`) then
    gives every atom of an orbit its own term exactly: ``|H| / n_a`` operations carry ``a`` onto each of them.
    :class:`parsec_python.Forces.AtomSymmetryOperation` for a cluster, those of :func:`abelian_cell_operations` for a
    periodic cell.  ``None`` for a point-group reduction (:func:`sector_force_operations` completes those), sectors
    without the orbits of their atoms (the projectors of every atom, nothing to complete), or
    ``PARSEC_SECTOR_PROJECTOR_ORBITS=off``.
    """

    from parsec_python.Forces import AtomSymmetryOperation

    from ..Symmetry.axis_reflection import (
        AxisReflectionReduction,
        PointGroupReduction,
        abelian_sector_operations,
        atom_image_table,
        sector_projector_orbits_setting,
    )

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
    if (
        not isinstance(reduction, AxisReflectionReduction)
        or isinstance(reduction, PointGroupReduction)
        or int(getattr(reduction, "group_order", 1)) <= 1
    ):
        return None
    sizes = getattr(reduction, "atom_orbit_sizes", None)
    if (
        getattr(reduction, "atom_representatives", None) is None
        or sizes is None
        or not np.any(np.asarray(sizes) > 1)
        or sector_projector_orbits_setting() == "off"
    ):
        return None
    if getattr(getattr(system, "input", None), "periodic_cell", None) is not None:
        return abelian_cell_operations(system)
    matrices = abelian_sector_operations(reduction).astype(np.float64)
    shifts = getattr(reduction, "translations", None)
    grid = getattr(reduction, "grid", None)
    if grid is None:
        grid = system.grid
    images = atom_image_table(
        tuple(system.atoms),
        matrices,
        None if shifts is None else np.asarray(shifts, dtype=np.int64).reshape(-1, 3),
        grid,
        max(ATOM_TOLERANCE, MOVE_TOLERANCE),
    )
    if images is None:
        raise ValueError("an operation of the group of the sectors carries an atom onto no atom of its species")
    identity = np.flatnonzero(
        [np.array_equal(m, np.eye(3)) and (shifts is None or not np.any(shifts[g])) for g, m in enumerate(matrices)]
    )
    if identity.size != 1:
        raise ValueError("the operations of the group of the sectors hold no single identity")
    order = [int(identity[0])] + [g for g in range(len(matrices)) if g != identity[0]]
    return tuple(
        AtomSymmetryOperation(matrix=matrices[g].copy(), image=np.asarray(images[g], dtype=np.int64).copy())
        for g in order
    )


def sector_symmetry_operations(system):
    """The operations that keep the forces and moved atoms of a periodic cell in its sectors on their group.

    ``G`` of a point-group reduction (:func:`sector_force_operations`), ``H`` of Abelian sectors
    (:func:`abelian_cell_operations`), ``None`` otherwise (no sectors, one sector of the identity, a cluster, whose
    ionic loop keeps the point group of its atoms).
    """

    operations = sector_force_operations(system)
    return operations if operations is not None else abelian_cell_operations(system)


def _operation_text(matrix: np.ndarray, shift) -> str:
    rows = "; ".join(" ".join(str(int(value)) for value in row) for row in np.asarray(matrix).reshape(3, 3))
    return f"[{rows}]" + ("" if shift is None or not np.any(shift) else f" + {np.asarray(shift).tolist()} grid steps")


def check_sector_atoms(reduction, atoms, grid, lattice_vectors=None, *, tolerance: float = MOVE_TOLERANCE) -> None:
    """Raise ``ValueError`` unless the sectors of ``reduction`` hold for ``atoms`` (moved atoms of a prepared system).

    The sectors, the weights of their states and the projectors of one atom per orbit
    (:func:`..Symmetry.axis_reflection.compress_projectors_for_sectors`) were made for the group of the first
    structure.  They hold for moved atoms where every operation of the group of the sectors ``H`` (``operations``
    with ``translations``) and of the full group ``G`` (for a :class:`..Symmetry.axis_reflection.PointGroupReduction`)
    carries every atom within ``tolerance`` (bohr, :data:`MOVE_TOLERANCE`) onto an atom of its species, as a
    permutation, modulo the lattice of a periodic grid, and where the orbits of the atoms under ``H`` are those of the
    reduction (``atom_representatives`` and ``atom_orbit_sizes``).  ``G`` is the union of the right cosets ``H c``,
    so ``H`` and the coset representatives (``coset_operations`` with ``coset_translations``) are tested, ``|H| +
    [G:H]`` tree queries in place of ``|G|``.  An Abelian reduction is tested on ``H``, which is all its sectors use.

    ``grid`` is the grid of the sectors (a periodic grid matches images modulo its cell); ``lattice_vectors`` (rows),
    where given, must be those of that cell.  The error names the operation, the atom and how far its image lies from
    the nearest atom of its species.
    """

    from ..Symmetry.axis_reflection import _QUERY_WORKERS, _folded, _lattice_period, _matching_tree

    atoms = tuple(atoms)
    if not atoms:
        return
    period = _lattice_period(grid)
    if lattice_vectors is not None:
        lattice = np.asarray(lattice_vectors, dtype=np.float64).reshape(3, 3)
        if period is None or not np.allclose(np.diag(lattice), period, rtol=0.0, atol=1.0e-10):
            raise ValueError("the lattice vectors are not those of the periodic grid of the sectors")
    operations = getattr(reduction, "operations", None)
    if operations is None:
        operations = [np.diag(signs) for signs in np.asarray(reduction.signs)]
    matrices = np.asarray(operations, dtype=np.float64).reshape(-1, 3, 3)
    shifts = getattr(reduction, "translations", None)
    steps = None
    if shifts is not None:
        steps = np.diag(np.asarray(grid.cell.lattice_vectors, dtype=np.float64)) / np.asarray(
            grid.points_per_axis, dtype=np.float64
        )
    positions = np.asarray([atom.position for atom in atoms], dtype=np.float64)
    symbols = np.asarray([atom.symbol for atom in atoms])
    species = [(symbol, np.flatnonzero(symbols == symbol)) for symbol in np.unique(symbols)]
    trees = {symbol: _matching_tree(positions[members], period) for symbol, members in species}
    order = int(matrices.shape[0])

    def images_of(matrix, shift, name: str) -> np.ndarray:
        images = np.empty(positions.shape[0], dtype=np.int64)
        for symbol, members in species:
            image = positions[members] @ matrix.T
            if shift is not None:
                image = image + np.asarray(shift, dtype=np.float64) * steps
            queries = image if period is None else _folded(image, period)
            distances, nearest = trees[symbol].query(queries, k=1, workers=_QUERY_WORKERS)
            worst = int(np.argmax(distances))
            if not distances[worst] <= tolerance:
                atom = int(members[worst])
                raise ValueError(
                    f"the moved atoms leave the group of the symmetry sectors: {name} {_operation_text(matrix, shift)} "
                    f"carries atom {atom + 1} ({symbol}) {float(distances[worst]):.3e} bohr from the nearest {symbol} "
                    f"atom (tolerance {tolerance:g} bohr). The sectors hold for the group of {order} operations of "
                    "the structure the calculation was prepared with: keep the atoms on that group, as a relaxation "
                    "with symmetrized forces does, or prepare the calculation without symmetry (symmetry='off', "
                    "PARSEC_PERIODIC_SYMMETRY=off for a periodic cell)"
                )
            images[members] = members[nearest]
        if np.unique(images).size != images.size:
            raise ValueError(
                f"the moved atoms leave the group of the symmetry sectors: {name} {_operation_text(matrix, shift)} "
                "carries two atoms onto one"
            )
        return images

    table = np.stack(
        [
            images_of(matrices[g], None if shifts is None else shifts[g], f"operation {g + 1} of the {order} of H")
            for g in range(order)
        ]
    )
    cosets = getattr(reduction, "coset_operations", None)
    if cosets is not None and getattr(reduction, "full_operations", None) is not None:
        coset_matrices = np.asarray(cosets, dtype=np.float64).reshape(-1, 3, 3)
        coset_shifts = getattr(reduction, "coset_translations", None)
        for c in range(coset_matrices.shape[0]):
            images_of(
                coset_matrices[c],
                None if coset_shifts is None else coset_shifts[c],
                f"coset representative {c + 1} of the {coset_matrices.shape[0]} of G",
            )
    representatives = getattr(reduction, "atom_representatives", None)
    sizes = getattr(reduction, "atom_orbit_sizes", None)
    if representatives is None or sizes is None:
        return
    found = table.min(axis=0)
    found_sizes = 1 + np.count_nonzero(np.diff(np.sort(table, axis=0), axis=0), axis=0)
    representatives = np.asarray(representatives, dtype=np.int64).reshape(-1)
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    if representatives.size != found.size or sizes.size != found.size:
        raise ValueError(
            f"the reduction knows the orbits of {representatives.size} atoms, the moved structure has {found.size}"
        )
    changed = np.flatnonzero((found != representatives) | (found_sizes != sizes))
    if changed.size:
        atom = int(changed[0])
        raise ValueError(
            "the moved atoms change the orbits of the atoms under the group of the symmetry sectors (atom "
            f"{atom + 1}: in the orbit of atom {int(found[atom]) + 1} with {int(found_sizes[atom])} atoms, prepared in "
            f"that of atom {int(representatives[atom]) + 1} with {int(sizes[atom])}): the projectors of one atom per "
            "orbit no longer hold; prepare the moved structure afresh"
        )


__all__ = [
    "ATOM_TOLERANCE",
    "MOVE_TOLERANCE",
    "abelian_cell_operations",
    "abelian_orbit_operations",
    "check_sector_atoms",
    "sector_force_operations",
    "sector_group_operations",
    "sector_reduction",
    "sector_symmetry_operations",
]
