"""A k-point mesh of a cell and the Gamma point of its supercell: the same calculation, either way.

With the Bloch block of the finite-difference Laplacian as the kinetic term of a k-point
(:func:`parsec_python.Laplacian.finite_difference.build_bloch_negative_laplacian`, ``PARSEC_KPOINT_KINETIC=bloch``),
the Gamma-centred mesh of ``m_1 x m_2 x m_3`` points of a cell has the eigenvalues of the
Gamma point of the supercell of ``m_1 m_2 m_3`` cells on the same grid: the total energy of the supercell is ``M``
times that of the cell (``M = m_1 m_2 m_3``), to round-off.  Two routes use it, each behind its switch:

* ``PARSEC_KPOINT_SUPERCELL=auto`` (:func:`supercell_of_trim_mesh`): a mesh of one or two points per axis (the
  time-reversal invariant k-points, ``Monkhorst_Pack_Shift`` zero) is solved as the Gamma point of the supercell,
  whose orbitals are real and take the sectors of its translations by half the cell with the point group
  (``parsec_python.acceleration.Symmetry.axis_reflection``): real arithmetic and the little groups of the k-points;
* ``PARSEC_SUPERCELL_KPOINTS=auto`` (:func:`kpoints_of_supercell`): a Gamma-point cell that repeats a smaller
  orthorhombic cell ``m_a`` times along each axis (a perfect crystal) is solved as that cell with the Gamma-centred
  mesh of ``m_1 x m_2 x m_3`` points: every translation of the crystal, through Bloch's theorem.

Both are ``off`` by default: the output of the other route reports its own quantities (the eigenvalues of the
supercell, or those of every k-point, and the total energy of what it solved), and a note gives the factor ``M``.
"""

from __future__ import annotations

from dataclasses import replace
import os

import numpy as np

TRIM_SWITCH = "PARSEC_KPOINT_SUPERCELL"
BLOCH_SWITCH = "PARSEC_SUPERCELL_KPOINTS"


def _setting(name: str) -> str:
    value = os.environ.get(name, "off").strip().lower() or "off"
    if value not in {"auto", "off"}:
        raise ValueError(f"{name} must be auto or off, not {value!r}")
    return value


def kpoint_supercell_setting() -> str:
    """``PARSEC_KPOINT_SUPERCELL``: ``auto`` solves a mesh of time-reversal invariant k-points as a supercell."""
    return _setting(TRIM_SWITCH)


def supercell_kpoints_setting() -> str:
    """``PARSEC_SUPERCELL_KPOINTS``: ``auto`` solves a repeating Gamma-point cell as its smallest cell with a mesh."""
    return _setting(BLOCH_SWITCH)


def _bloch_kinetic_applies(problem) -> bool:
    """Whether the cell takes the Bloch kinetic term (a bulk orthorhombic cell of the default routines)."""
    grid = problem.grid
    if getattr(grid, "fortran_routines", False) or int(getattr(grid, "periodicity", 3)) != 3:
        return False
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    return bool(np.allclose(lattice, np.diag(np.diag(lattice)), atol=1.0e-12) and np.all(np.diag(lattice) > 0.0))


def _plain_scalar(problem) -> bool:
    scf = problem.scf
    return not (
        scf.spin_polarized or scf.noncollinear or scf.self_consistent_spin_orbit or scf.is_hybrid or scf.is_meta_gga
        or getattr(scf, "fixed_occupations", None) is not None
    )


def _grid_counts(problem) -> np.ndarray:
    from parsec_python.Grid.pbc import build_periodic_grid

    grid = build_periodic_grid(problem.periodic_cell, problem.grid)
    return np.asarray(grid.points_per_axis, dtype=np.int64)


def _same_grid(cell_problem, super_problem, repeats) -> bool:
    """Whether the supercell's grid is the cell's grid repeated (the same steps and shift)."""
    return bool(np.array_equal(_grid_counts(super_problem), _grid_counts(cell_problem) * np.asarray(repeats)))


def supercell_of_trim_mesh(problem, refusal=None):
    """``(supercell problem, repeats)`` for a Gamma-centred mesh of one or two points per axis, else ``None``.

    The supercell repeats the cell ``m_a`` times along each axis (the atoms, the lattice, ``M`` times the states),
    on the cell's grid steps, without k-points.  ``None`` under ``PARSEC_KPOINT_SUPERCELL=off``, for an explicit
    k-point list, a shifted mesh, a mesh of more points, an oblique cell,
    the Fortran-derived routines, a spin-polarized, spin-orbit, hybrid or meta-GGA calculation, fixed occupations,
    or a supercell for which ``refusal(supercell)`` (the accelerated runners' ``driver.periodic_sector_refusal``)
    gives a reason.
    """

    if kpoint_supercell_setting() != "auto" or problem.periodic_cell is None:
        return None
    dims = getattr(problem, "monkhorst_pack_dimensions", None)
    shift = getattr(problem, "monkhorst_pack_shift", None)
    if dims is None or getattr(problem, "kpoints", None) is not None or shift is None:
        return None
    repeats = tuple(int(value) for value in dims)
    if any(value not in (1, 2) for value in repeats) or np.any(np.asarray(shift, dtype=np.float64) != 0.0):
        return None
    if not _bloch_kinetic_applies(problem) or not _plain_scalar(problem):
        return None
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    count = int(np.prod(repeats))
    atoms = [
        type(atom)(atom.symbol, tuple(np.asarray(atom.position, dtype=np.float64) + np.asarray(cell, float) @ lattice))
        for cell in np.ndindex(*repeats)
        for atom in problem.atoms
    ]
    # the cell's grid steps: the supercell's counts are the cell's times the repeats
    steps = np.diag(lattice) / _grid_counts(problem)
    supercell = replace(
        problem,
        atoms=atoms,
        periodic_cell=replace(problem.periodic_cell, lattice_vectors=lattice * np.asarray(repeats, float)[:, None]),
        grid=replace(problem.grid, spacing=float(np.min(steps))),
        # the sectors of the supercell's own group, not the density symmetrization of the k-point wedge with the
        # space group of the cell
        scf=replace(problem.scf, number_of_states=int(problem.scf.number_of_states) * count, symmetrize_density=False),
        space_group=None,
        monkhorst_pack_dimensions=None,
        monkhorst_pack_shift=None,
    )
    if not _same_grid(problem, supercell, repeats):
        return None
    if refusal is not None and refusal(supercell) is not None:
        return None
    return supercell, repeats


def _repeat_of_axis(positions_by_species, side: float, axis: int, points: int, tolerance: float) -> int:
    """The largest ``m`` dividing the grid's ``points`` along ``axis`` (and every species' atom count) whose
    translation ``side/m`` maps the atoms onto themselves, modulo the cell (1 where none does)."""
    from scipy.spatial import cKDTree

    def _has_tolerance_perfect_matching(moved, points_, tolerance_, period_):
        # a translation of a crystal maps every atom onto one atom of its species, a different one each
        tree = cKDTree(np.mod(points_, period_), boxsize=period_)
        distances, nearest = tree.query(np.mod(moved, period_), k=1, distance_upper_bound=2.0 * tolerance_)
        return bool(np.all(np.isfinite(distances)) and np.unique(nearest).size == len(points_))

    period = np.asarray(positions_by_species["__period__"])
    species = {name: points_ for name, points_ in positions_by_species.items() if name != "__period__"}
    for m in sorted((value for value in range(2, points + 1) if points % value == 0), reverse=True):
        if any(len(points_) % m for points_ in species.values()):
            continue
        shift = np.zeros(3)
        shift[axis] = side / m
        if all(_has_tolerance_perfect_matching(points_ + shift, points_, tolerance, period) for points_ in species.values()):
            return m
    return 1


def kpoints_of_supercell(problem, tolerance: float = 1.0e-7):
    """``(cell problem, repeats)`` for a Gamma-point cell that repeats a smaller one, else ``None``.

    The repeats ``m_a`` are the largest whole divisions of each axis whose translation maps the atoms (species by
    species, modulo the cell); the smaller cell keeps the atoms in it, ``1/M`` of the states (at least as many per
    k-point as the occupied ones need, rounded up), and the Gamma-centred mesh ``m_1 x m_2 x m_3``.  ``None``
    under ``PARSEC_SUPERCELL_KPOINTS=off``, with k-points already, without a repeat, where the cell's grid would not
    repeat into the supercell's, under an explicit ``PARSEC_KPOINT_KINETIC=gradient`` (the caller runs the mesh with
    ``bloch``, which reproduces the supercell), and for what :func:`supercell_of_trim_mesh` refuses besides the mesh.
    """

    if supercell_kpoints_setting() != "auto" or problem.periodic_cell is None:
        return None
    if getattr(problem, "has_kpoint_sampling", False):
        return None
    # the mesh reproduces the supercell with the Bloch kinetic term only (the caller sets it for the run)
    if (os.environ.get("PARSEC_KPOINT_KINETIC", "").strip().lower() or "unset") == "gradient":
        return None
    if not _bloch_kinetic_applies(problem) or not _plain_scalar(problem):
        return None
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    sides = np.diag(lattice)
    by_species: dict = {}
    for atom in problem.atoms:
        by_species.setdefault(atom.symbol, []).append(np.asarray(atom.position, dtype=np.float64))
    by_species = {name: np.asarray(points) for name, points in by_species.items()}
    by_species["__period__"] = sides
    counts = _grid_counts(problem)
    repeats = []
    for axis in range(3):
        repeats.append(_repeat_of_axis(by_species, float(sides[axis]), axis, int(counts[axis]), tolerance))
    count = int(np.prod(repeats))
    if count == 1:
        return None
    cell_sides = sides / np.asarray(repeats, dtype=np.float64)
    atoms = [
        atom for atom in problem.atoms
        if np.all(np.mod(np.asarray(atom.position, dtype=np.float64), sides) < cell_sides - 1.0e-9 * sides)
    ]
    if len(atoms) * count != len(problem.atoms):
        return None
    states = int(np.ceil(int(problem.scf.number_of_states) / count))
    cell = replace(
        problem,
        atoms=[type(atom)(atom.symbol, tuple(np.mod(np.asarray(atom.position, float), sides))) for atom in atoms],
        periodic_cell=replace(problem.periodic_cell, lattice_vectors=np.diag(cell_sides)),
        grid=replace(problem.grid, spacing=float(np.min(sides / counts))),
        scf=replace(problem.scf, number_of_states=states),
        # the space group of the supercell is not the cell's (the mesh takes time reversal alone)
        space_group=None,
        monkhorst_pack_dimensions=tuple(repeats),
        monkhorst_pack_shift=(0.0, 0.0, 0.0),
    )
    if not np.array_equal(_grid_counts(cell) * np.asarray(repeats), counts):
        return None
    return cell, tuple(repeats)


def describe(route: str, repeats) -> str:
    """The note of the output: what was solved and the factor between the two total energies."""
    m = tuple(int(value) for value in repeats)
    count = int(np.prod(m))
    if route == "supercell":
        return (
            f"the Gamma-centred k-point mesh {m[0]} x {m[1]} x {m[2]} of the cell is solved as the Gamma point of its "
            f"supercell of {count} cells (PARSEC_KPOINT_SUPERCELL=auto; the Bloch kinetic term): the total energy "
            f"reported is that of the supercell, {count} times the energy per cell of the mesh"
        )
    return (
        f"the cell repeats a smaller one {m[0]} x {m[1]} x {m[2]} times: it is solved as that cell with the "
        f"Gamma-centred k-point mesh {m[0]} x {m[1]} x {m[2]} (PARSEC_SUPERCELL_KPOINTS=auto; the Bloch kinetic "
        f"term): the total energy reported is that of the smaller cell, 1/{count} of the supercell's"
    )


__all__ = [
    "BLOCH_SWITCH",
    "TRIM_SWITCH",
    "describe",
    "kpoint_supercell_setting",
    "kpoints_of_supercell",
    "supercell_kpoints_setting",
    "supercell_of_trim_mesh",
]
