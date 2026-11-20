"""The k-points of a periodic SCF in the sectors of their little groups (``PARSEC_KPOINT_SECTORS``).

The Bloch Hamiltonian of a k-point acts on the periodic factor ``u_k`` (the u gauge),
``H_k = e^{-ik.r} H e^{ik.r}``.  An operation ``g: r -> R r + t`` of a bulk orthorhombic cell (a Cartesian
signed permutation ``R`` with a translation ``t`` of whole grid steps,
:func:`..Symmetry.axis_reflection._space_group_operations`) permutes the grid points, ``(P_g f)(r) = f(g^-1 r)``,
and commutes with ``H``.  It moves the phase, ``P_g e^{ik.r} P_g^-1 = e^{i (R k).(r - t)}``, so where ``R k = k``
it commutes with ``H_k`` as well (the constant ``e^{-ik.t}`` drops out).  ``R k = k`` exactly, not modulo a
reciprocal lattice vector: where ``R k = k + G``, ``P_g`` carries ``H_k`` into ``H_{k+G} = e^{-iG.r} H_k e^{iG.r}``,
so the operator that commutes with ``H_k`` is ``e^{iG.r} P_g``, a complex operator whose characters are not the
real ``+-1`` of the sector bases below.  (The Bloch form of ``PARSEC_KPOINT_KINETIC=bloch``, ``e^{-ik.r} L e^{ik.r}``
of the finite-difference Laplacian ``L``, obeys that identity exactly, since ``e^{iG.r}`` is periodic on the grid;
the default ``-nabla^2 - 2ik.grad + |k|^2`` of the u gauge only to its discretization, about 1e-8 Ry on the even
meshes of the tests.)  These operations form the little group ``G_k``.

On the grid each term commutes with ``P_g``: the Laplacian and the first derivatives of a uniform periodic grid
are invariant under its translations and turn as a scalar and a vector under ``R`` (the coefficients along the
axes that ``R`` exchanges agree, as their grid steps do where ``R`` maps the grid), so ``k.grad`` is invariant
where ``R^T k = k``.  The Bloch projectors of an atom go over into those of its image times one constant phase
(``e^{ik.(t - T)}`` for ``R tau_a + t = tau_b + T``, the same for every ``m``), which drops out of ``B D B^H``.
The local potential is invariant where the density is (below).

The largest subgroup ``H_k`` of ``G_k`` whose elements commute and are of order two (the half-cell translations,
the twofold axes and the mirrors that fix ``k``) has the real characters ``+-1``: the sector basis ``U`` of
:class:`..Symmetry.representations.ReflectionRepresentationDecomposition` is real, and the complex Hermitian
``H_k`` reduces as a real symmetric operator does, ``U^T H_k U`` from the representative rows
(:meth:`~..Symmetry.representations.ReflectionRepresentationDecomposition.reduce_hermitian_operators`).  The
spectrum of ``H_k`` is the union of the spectra of the sectors: each k-point becomes ``|H_k|`` eigenproblems of
about ``1/|H_k|`` of its rows and states, and the work of the eigensolver (rows times states squared) falls by
about ``|H_k|^2`` per k-point.  The projectors of one atom per orbit of ``H_k`` serve every sector
(:func:`..Symmetry.axis_reflection.compress_projectors_for_sectors`, which holds for the Bloch projectors as
well: their phases drop out of ``B D B^H``).

The potential is invariant under ``g`` where the density is: the density symmetrized over the space group of
the input (``Symmetrize_Charge_Density``, the wedge of a reduced mesh) under its operations; a density summed
over a set of k-points that ``R`` maps onto itself (with time reversal, modulo the reciprocal lattice, at equal
weights) under ``g`` at self-consistency; every ``rho_k`` under the half-cell translations.  An operation that
meets none of these stays out of ``G_k``.  The sector Hamiltonian takes the potential averaged over each orbit:
its invariant part, the potential itself up to the error of the eigensolver in the iterations before.

A sector of ``n_s`` of the ``N`` rows solves ``M_s = ceil(c M n_s / N) + 6`` states (``M`` the states of a
k-point, ``c`` = ``PARSEC_KPOINT_SECTOR_MARGIN``, 1.3 by default), at most ``n_s``; the k-point keeps the lowest
``M`` levels of the union.  After every solve, each sector with fewer states than rows must reach the ``M``-th
level of the union (``max e_s >= e_M``): every level it did not compute then lies above the levels kept.  A
sector that falls short gets 1.5 times its states and the step is solved again, from new starts.  The margin is
needed only until the levels are known: after the first solve of a loop each sector keeps the ``h_s`` levels it
holds among the ``M`` kept plus a tenth of them, at least six more (``PARSEC_KPOINT_SECTOR_REFIT=auto``, the
default; :meth:`SectorKPointSolver._refit`), its Ritz states cut to their lowest pairs at its next solve, and a sector
so cut that later falls short grows by its estimated deficit (the rule of the Gamma sectors).  A k-point
whose sectors would not halve its work (``sum n_s M_s^2 <= N M^2 / 2``) stays one task, and so do the k-points
of a basis of fewer than ``2^24`` entries (``N M``, 268 MB of complex vectors): their eigenproblems do not fill a
device, and more of them would add the cost of the tasks (the 4631 k-points of an 8-atom cube with a 21^3 mesh).

The density of a sector, ``|U v|^2``, is ``q_s / m_w`` on the rows of the orbit ``w`` of its column ``s``
(``q = sum_n f_n |v_n|^2``, ``m_w`` the orbit size): expanded on the host and summed by the job of the driver.

The k-point SCF of a bulk orthorhombic cell takes it, scalar or spin-polarized, with the exchange of a hybrid, forces
and kept orbitals, without meta-GGA, LDA+U, the fixed occupations of ``occup.in`` or refined levels (the driver then
gives the job its ``geometry``); ``PARSEC_KPOINT_SECTORS=off`` keeps one task per k-point, bit for bit as before.

The sectors hold the potential invariant under the operations of ``H_k``, also where only the self-consistent
solution is (a single Gamma point, a full mesh, the translations of a supercell: the operations the density is not
symmetrized over, ``KPointSectors.assumed``).  A spin-polarized state that breaks that symmetry (an open shell, a
Jahn-Teller or magnetic order) cannot be held, as on the sectors of a cluster (``Spin_Symmetry_Sectors``,
:mod:`.spin_sectors`), and the same three tests guard a spin-polarized loop (``PARSEC_KPOINT_SECTOR_GUARD=auto``):
no partly filled multiplet across the sectors of such a k-point in the start Hamiltonian, channel densities
invariant under the operations assumed (:class:`SectorGuardGroup`) at the end, and with Fermi-Dirac occupations no
fractional occupation of such a multiplet at the end, each weighted by its k-point (the electrons it adds to the
density).  A loop that fails one runs again with one task per k-point
(:func:`parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized`).  An unpolarized loop is not tested, as the
unpolarized sectors of a cluster are not.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from ...Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from ...SCF.kpoint_solver import KPointJob
from ...SCF.orbital_density import weighted_square_sum
from ..Symmetry.axis_reflection import (
    SignedPermutationReduction,
    _RowOrderTable,
    _affine_matrix,
    _atom_orbits,
    _fast_maps_requested,
    _grid_row_mapping_operation,
    _lattice_period,
    _orbit_labels,
    _periodic_row_major,
    _space_group_operations,
    compress_projectors_for_sectors,
)
from ..Symmetry.orbital_groups import multiplication_table
from ..Symmetry.representations import ReflectionRepresentationDecomposition

_SWITCH = "PARSEC_KPOINT_SECTORS"
# the key of the states of the sectors in what export_states hands to the next loop
_STATES_KEY = "kpoint_sectors"
# the key of the plans of the sectors in a caller's cache of the structure (the projector cache of the k-point loops)
_CACHE_KEY = "kpoint_sector_plans"
# states of a sector expanded to the whole grid at once for its extras (the forces)
_FORCE_COLUMNS = 32
_MARGIN_SWITCH = "PARSEC_KPOINT_SECTOR_MARGIN"
_REFIT_SWITCH = "PARSEC_KPOINT_SECTOR_REFIT"
# the buffer of a refitted sector over the levels it holds among those kept (at least _EXTRA_STATES states)
_REFIT_BUFFER = 0.1
_EXTRA_STATES = 6
_GROWTH = 1.5
_ATTEMPTS = 4
_WORK_FRACTION = 0.5
# the smallest basis of a k-point (rows times states) that is split
_MINIMUM_ENTRIES = 1 << 24


def kpoint_sectors_setting() -> str:
    """``PARSEC_KPOINT_SECTORS``: ``auto`` (default) splits the k-points of a scalar periodic SCF into the sectors
    of their little groups where that halves the work; ``off`` keeps one task per k-point, bit for bit as before."""

    value = os.environ.get(_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_SWITCH} must be auto or off, not {value!r}")
    return value


def sector_margin() -> float:
    """``PARSEC_KPOINT_SECTOR_MARGIN``: the states of a sector over its share of the states of the k-point (1.3)."""

    raw = os.environ.get(_MARGIN_SWITCH, "").strip()
    if not raw:
        return 1.3
    value = float(raw)
    if not value >= 1.0:
        raise ValueError(f"{_MARGIN_SWITCH} must be at least 1, not {raw!r}")
    return value


def sector_guard_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_GUARD``: ``auto`` (default) tests a spin-polarized loop in the sectors of its k-points
    for a state that breaks the symmetry the sectors assume and runs it again with one task per k-point where it
    does (:func:`parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized`); ``off`` keeps the sectors without the
    tests, as before."""

    value = os.environ.get("PARSEC_KPOINT_SECTOR_GUARD", "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"PARSEC_KPOINT_SECTOR_GUARD must be auto or off, not {value!r}")
    return value


def sector_refit_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_REFIT``: ``auto`` (default) cuts the states of every sector after the first solve of a
    loop to the levels it holds among those its k-point keeps plus a buffer (:meth:`SectorKPointSolver._refit`);
    ``off`` keeps the first counts (and their growth), bit for bit as before."""

    value = os.environ.get(_REFIT_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_REFIT_SWITCH} must be auto or off, not {value!r}")
    return value


def refitted_count(held: int, size: int, count: int) -> int:
    """The states of a sector that holds ``held`` of the levels its k-point keeps: ``held`` plus
    :data:`_REFIT_BUFFER` of them, at least :data:`_EXTRA_STATES` more, at most its rows and its ``count`` so far."""

    held = int(held)
    return min(int(size), int(count), held + max(_EXTRA_STATES, math.ceil(_REFIT_BUFFER * held)))


def _leading_columns(vectors, working: int):
    """A copy of the first ``working`` columns of ``vectors`` (host or device) in its memory order."""

    order = "F" if vectors.flags.f_contiguous and not vectors.flags.c_contiguous else "C"
    return vectors[:, : int(working)].copy(order=order)


def trimmed_state(state, requested: int, working: int, *, group=None, block_size: int | None = None):
    """The Ritz state ``state`` of an eigensolver cut to its lowest ``working`` Ritz pairs, for a solve of
    ``requested`` states that keeps it (``None`` where it cannot be cut: a state of ARPACK or DIAGLA).

    The Ritz vectors of a state are in the order of their values, so the first ``working`` columns are the lowest
    pairs.  The three kinds of states of the k-point solvers: that of the host eigensolver
    (:class:`parsec_python.Eigensolvers.eigval.EigvalState`), that of one device
    (:class:`..Eigensolvers.chebff_spinor.SpinorEigvalState`) and that of a basis the ranks of an MPI pool share by
    columns (:class:`..Eigensolvers.chebff_columns.ColumnEigvalState`, ``group`` its ranks, collective: the columns
    are laid out again over them, ``block_size`` that of the solves).  The lower bound of the next filter is the
    largest value kept (the filter of a larger subspace began above it).  A state with no more than ``working``
    columns comes back as it is.
    """

    from dataclasses import replace as _replace

    if state is None or int(state.working_states) <= int(working):
        return state
    working, requested = int(working), int(requested)
    subspace = getattr(state, "subspace", None)
    if subspace is not None:  # the host eigensolver
        if getattr(state, "window_vectors", None) is not None or getattr(state, "start_seed", None) is not None:
            return None
        values = np.asarray(subspace.eigenvalues, dtype=np.float64)[:working].copy()
        cut = _replace(
            subspace,
            working_states=working,
            eigenvalues=values,
            vectors=_leading_columns(np.asarray(subspace.vectors), working),
            filter_lower_bound=None if subspace.filter_lower_bound is None else float(np.max(values)),
        )
        return _replace(state, requested_states=requested, working_states=working, subspace=cut)
    values = np.asarray(_host(state.eigenvalues), dtype=np.float64)[:working].copy()
    bound = None if state.filter_lower_bound is None else float(np.max(values))
    layout = getattr(state, "layout", None)
    if layout is None:  # one device
        return _replace(
            state, vectors=_leading_columns(state.vectors, working), eigenvalues=values, working_states=working,
            filter_lower_bound=bound,
        )
    from ..Eigensolvers.chebff_columns import ColumnLayout

    fresh = ColumnLayout(working, layout.block_size if block_size is None else int(block_size), layout.size)
    if layout.size == 1:
        vectors = _leading_columns(state.vectors, working)
    else:
        dimension = int(state.operator_dimension)
        rows = group.to_rows(state.vectors, layout, dimension)
        vectors = group.to_columns(rows[:, :working], fresh, dimension)
        del rows
    return _replace(
        state, vectors=vectors, eigenvalues=values, working_states=working, filter_lower_bound=bound, layout=fresh
    )


def carried_state(state, requested: int, settings, dimension: int, *, group=None):
    """The Ritz state a k-point solver hands its next solve of ``requested`` states: ``state`` cut to the working
    states of that solve (:func:`trimmed_state`) where it holds more (a task whose count the refit of the sectors
    cut, ``trim_tasks``) and ``PARSEC_KPOINT_SECTOR_REFIT`` is ``auto``; else ``state`` as it is (a state the solve
    cannot take starts afresh, as before)."""

    if state is None or getattr(state, "operator_dimension", None) != int(dimension):
        return state
    working = min(int(requested) + int(settings.safety_buffer), int(dimension))
    if int(state.working_states) <= working or sector_refit_setting() == "off":
        return state
    cut = trimmed_state(state, requested, working, group=group, block_size=int(settings.subspace.block_size))
    return state if cut is None else cut


def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    return np.asarray(get() if get is not None else value)


@dataclass(frozen=True)
class LittleGroupReduction(SignedPermutationReduction):
    """The orbits of the commuting involutions ``H_k`` of a little group.

    ``operations`` and ``translations`` (grid steps) are the operations ``r -> R r + t`` of ``H_k``; operation
    ``p`` is the product of the generators ``j`` with bit ``j`` of ``p`` set (``generator_bits``).  The atom orbits
    under them give the projectors of one atom per orbit (:func:`..Symmetry.axis_reflection._atom_orbits`).
    """

    translations: np.ndarray = None
    atom_representatives: np.ndarray = None
    atom_orbit_sizes: np.ndarray = None


class CellGroup:
    """The operations ``r -> R r + t`` of a bulk orthorhombic cell (the identity first) and their group table.

    Those of the sectors of the Gamma point (:func:`..Symmetry.axis_reflection._space_group_operations`): with the
    operations whose translation is a fraction of the cell where they close into a group, else the half-cell
    translations alone.  ``table`` is ``None`` where the operations form no group.
    """

    def __init__(self, grid, atoms, *, atom_tolerance: float = 1.0e-7, lattice_tolerance: float = 1.0e-10) -> None:
        self.grid = grid
        self.atoms = tuple(atoms)
        self.atom_tolerance = float(atom_tolerance)
        self.lattice_tolerance = float(lattice_tolerance)
        self.counts = np.asarray(grid.points_per_axis, dtype=np.int64)
        self.matrices: list[np.ndarray] = []
        self.shifts: list[np.ndarray] = []
        self.table: np.ndarray | None = None
        for fractional in (True, False):
            matrices, shifts = _space_group_operations(
                grid, self.atoms, atom_tolerance, lattice_tolerance, fractional=fractional
            )
            shifts = [np.mod(np.asarray(shift, dtype=np.int64), self.counts) for shift in shifts]
            try:
                table = multiplication_table(
                    [_affine_matrix(matrix, shift) for matrix, shift in zip(matrices, shifts)], modulus=self.counts
                )
            except ValueError:
                continue
            self.matrices = [np.asarray(matrix, dtype=np.int8) for matrix in matrices]
            self.shifts = shifts
            self.table = table
            break

    @property
    def order(self) -> int:
        return len(self.matrices)

    def little_group(self, k_point, admissible: Callable[[int], bool]) -> list[int]:
        """The operations with ``R k = k`` (to round-off of ``k``) that ``admissible`` accepts, the identity first."""

        k = np.asarray(k_point, dtype=np.float64).reshape(3)
        tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(k))))
        members = []
        for index, matrix in enumerate(self.matrices):
            if float(np.max(np.abs(matrix.astype(np.float64) @ k - k))) > tolerance:
                continue
            if index and not admissible(index):
                continue
            members.append(index)
        return members

    def row_map(self, index: int, rows: np.ndarray | None = None, table=None) -> np.ndarray | None:
        """The grid rows of ``R r + t`` for the rows ``rows`` (all where ``None``), ``None`` where they miss."""

        return _grid_row_mapping_operation(
            self.grid,
            self.matrices[index],
            self.lattice_tolerance,
            rows,
            table=table,
            translation=self.shifts[index],
        )


def generated_subgroup(table: np.ndarray, members) -> list[int]:
    """The subgroup that ``members`` generate (ascending, the identity 0 first): the potential is invariant under
    the products of the operations it is invariant under (the space group of the density and the translations of
    the cell, say), and the products of operations that fix ``k`` fix it."""

    found = {0} | {int(index) for index in members}
    frontier = sorted(found)
    while frontier:
        fresh = set()
        for left in frontier:
            for right in sorted(found):
                for product_ in (int(table[left, right]), int(table[right, left])):
                    if product_ not in found:
                        fresh.add(product_)
        found |= fresh
        frontier = sorted(fresh)
    return sorted(found)


def largest_commuting_involutions(table: np.ndarray, members) -> tuple[list[int], list[int]]:
    """``(elements, generators)`` of the largest subgroup of ``members`` whose elements commute and square to the
    identity (index 0); element ``p`` is the product of the generators ``j`` with bit ``j`` of ``p`` set.

    The group a set of commuting involutions generates consists of commuting involutions, so the search adds one
    generator at a time among the involutions that commute with the group so far and lie outside it; a group of
    order ``2^r`` that grows past the best one so far needs ``2^(r+1)`` elements, all among those candidates, which
    bounds the search.  Products outside ``members`` (the operations the potential is invariant under) refuse a
    generator.  The first largest group found is kept (deterministic).
    """

    allowed = {int(index) for index in members}
    involutions = [int(g) for g in members if int(g) != 0 and int(table[g, g]) == 0]
    best: list[list[int]] = [[0], []]

    def search(group: list[int], generators: list[int], candidates: list[int]) -> None:
        if len(group) > len(best[0]):
            best[0], best[1] = group, generators
        for position, candidate in enumerate(candidates):
            if len(group) + len(candidates) - position < 2 * len(best[0]):
                return
            images = [int(table[candidate, element]) for element in group]
            if any(image not in allowed for image in images):
                continue
            grown = group + images
            inside = set(grown)
            rest = [
                other
                for other in candidates[position + 1 :]
                if int(table[candidate, other]) == int(table[other, candidate]) and other not in inside
            ]
            search(grown, generators + [candidate], rest)

    search([0], [], involutions)
    return best[0], best[1]


def little_group_reduction(cell: CellGroup, elements: list[int], generators: list[int]) -> LittleGroupReduction | None:
    """The orbits of the grid rows under ``elements`` (``None`` where one of them misses the grid)."""

    row_table = _RowOrderTable(cell.grid) if _fast_maps_requested() else None
    # The smallest image of every row under the group, map by map, as detect_point_group_reduction keeps it: the
    # maps are not held together (32 elements of a 20^3 diamond cell, 46.7 M points, are 12 GB); _orbit_labels of
    # that one array is _orbit_labels of all the maps.
    smallest = None
    for index in elements:
        mapping = cell.row_map(index, table=row_table)
        if mapping is None:
            return None
        if smallest is None:
            smallest = np.array(mapping, dtype=np.int64, copy=True)
        else:
            np.minimum(smallest, mapping, out=smallest)
        del mapping
    representative_rows, full_to_wedge, multiplicities = _orbit_labels([smallest])
    del smallest
    order, rank = len(elements), len(generators)
    if order != 1 << rank:
        raise RuntimeError("the commuting involutions do not form a group of order 2^rank")
    bits = np.array([[(p >> j) & 1 for j in range(rank)] for p in range(order)], dtype=np.int8).reshape(order, rank)
    operations = np.ascontiguousarray([cell.matrices[index] for index in elements], dtype=np.int8)
    translations = np.ascontiguousarray([cell.shifts[index] for index in elements], dtype=np.int64)
    orbits = _atom_orbits(cell.atoms, list(operations), list(translations), cell.grid, cell.atom_tolerance)
    return LittleGroupReduction(
        signs=np.ones((order, 3), dtype=np.int8),
        representative_rows=np.ascontiguousarray(representative_rows, dtype=np.int64),
        full_to_wedge=np.ascontiguousarray(full_to_wedge, dtype=np.int64),
        multiplicities=np.ascontiguousarray(multiplicities, dtype=np.int64),
        operations=operations,
        generator_bits=np.ascontiguousarray(bits),
        translations=translations,
        atom_representatives=None if orbits is None else orbits[0],
        atom_orbit_sizes=None if orbits is None else orbits[1],
    )


class PotentialInvariance:
    """Whether the potential of the loop is invariant under an operation of the cell (see the module text)."""

    _SAMPLES = 61

    def __init__(self, cell: CellGroup, density_symmetry, k_points, weights, lattice) -> None:
        self.cell = cell
        self.symmetry = density_symmetry
        self.k_points = np.asarray(k_points, dtype=np.float64).reshape(-1, 3)
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        self.lattice = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
        self.fractional = self.k_points @ self.lattice.T / (2.0 * np.pi)
        self._decided: dict[int, bool] = {}
        self._in_group: dict[int, bool] = {}
        self._row_table = None
        self._sample = None
        self._sampled = None
        images = getattr(density_symmetry, "images", None)
        if images is not None and np.ndim(images) == 2 and np.shape(images)[1] == cell.grid.size:
            size = int(cell.grid.size)
            self._sample = np.unique(np.linspace(0, size - 1, self._SAMPLES).astype(np.int64))
            self._sampled = np.asarray(images)[:, self._sample]

    def __call__(self, index: int) -> bool:
        decided = self._decided.get(index)
        if decided is None:
            decided = self._decided[index] = self._decide(index)
        return decided

    def _decide(self, index: int) -> bool:
        matrix = self.cell.matrices[index]
        if np.array_equal(matrix, np.eye(3, dtype=np.int8)):
            return True  # a translation of the cell: every rho_k is invariant under it
        return self.in_density_group(index) or self._maps_kpoints(matrix)

    def in_density_group(self, index: int) -> bool:
        """Whether the density is symmetrized over operation ``index`` (``Symmetrize_Charge_Density``, the wedge of a
        reduced mesh): then the loop without sectors makes the potential invariant under it as well."""

        if index == 0:
            return True
        found = self._in_group.get(index)
        if found is None:
            found = self._in_group[index] = self._in_density_group(index)
        return found

    def _in_density_group(self, index: int) -> bool:
        if self._sample is None:
            return False
        sampled = self.cell.row_map(index, self._sample)
        if sampled is None:
            return False
        hits = np.flatnonzero(np.all(self._sampled == sampled[None, :], axis=1))
        if hits.size == 0:
            return False
        if self._row_table is None and _fast_maps_requested():
            self._row_table = _RowOrderTable(self.cell.grid)
        whole = self.cell.row_map(index, table=self._row_table)
        if whole is None:
            return False
        images = np.asarray(self.symmetry.images)
        # the maps of a group are the same set in either direction (it holds the inverses)
        return any(np.array_equal(images[hit], whole) for hit in hits)

    def _maps_kpoints(self, matrix: np.ndarray) -> bool:
        rotated = (self.k_points @ matrix.astype(np.float64).T) @ self.lattice.T / (2.0 * np.pi)
        scale = max(float(np.max(np.abs(self.weights), initial=0.0)), 1.0e-300)
        for row, weight in zip(rotated, self.weights):
            same = np.abs(self.weights - weight) <= 1.0e-10 * scale
            direct = row[None, :] - self.fractional
            reversed_ = row[None, :] + self.fractional
            match = (
                np.all(np.abs(direct - np.rint(direct)) <= 1.0e-8, axis=1)
                | np.all(np.abs(reversed_ - np.rint(reversed_)) <= 1.0e-8, axis=1)
            ) & same
            if not np.any(match):
                return False
        return True


class SectorGuardGroup:
    """The operations of a cell against which the symmetry guard measures the densities of a loop in the sectors
    (test 2 of :meth:`SectorKPointSolver.density_symmetry_report`).

    Those the potential of the loop is taken to be invariant under (:class:`PotentialInvariance`), beyond those the
    density is symmetrized over (whose invariance holds by construction): a generating set of the subgroup they
    generate together, chosen in the order of the operations (the closure grows by right products, through the
    group table).  ``rows`` of an operation are the grid rows of its images ``R r + t``.
    """

    def __init__(self, cell: CellGroup, invariance: PotentialInvariance) -> None:
        self.cell = cell
        self.invariance = invariance
        self._images: list[tuple[int, np.ndarray]] | None = None

    def _close(self, closure: set[int], generators: list[int]) -> None:
        table = self.cell.table
        frontier = list(closure)
        while frontier:
            fresh = []
            for element in frontier:
                for generator in generators:
                    product_ = int(table[element, generator])
                    if product_ not in closure:
                        closure.add(product_)
                        fresh.append(product_)
            frontier = fresh

    def images(self) -> list[tuple[int, np.ndarray]]:
        """``(operation, rows)`` of the operations to test (built once)."""

        if self._images is not None:
            return self._images
        admitted = [index for index in range(1, self.cell.order) if self.invariance(index)]
        free = [index for index in admitted if self.invariance.in_density_group(index)]
        closure = {0}
        generators: list[int] = []
        for index in free:
            if index not in closure:
                generators.append(index)
                self._close(closure, generators)
        tested: list[tuple[int, np.ndarray]] = []
        table = _RowOrderTable(self.cell.grid) if _fast_maps_requested() else None
        for index in admitted:
            if index in closure:
                continue
            rows = self.cell.row_map(index, table=table)
            if rows is None:
                continue
            generators.append(index)
            self._close(closure, generators)
            tested.append((index, np.asarray(rows, dtype=np.int64)))
        self._images = tested
        return tested

    def operation_text(self, index: int) -> str:
        matrix = np.asarray(self.cell.matrices[index], dtype=int).tolist()
        shift = np.asarray(self.cell.shifts[index], dtype=int).tolist()
        return f"{matrix} + {shift} grid steps"


class SectorBlochHamiltonian(KPointKohnShamHamiltonian):
    """The Bloch Hamiltonian of one sector of a k-point: its complex matrices (the reduced kinetic term in the slot
    of the Bloch Laplacian, the reduced projectors) take the complex CSR kernel of the device
    (:mod:`..Hamiltonian.complex_csr_device`, ``device_csr``)."""

    device_csr = True


class KPointSectors:
    """The sectors of one k-point: their bases, states and static operators (built at the first Hamiltonian)."""

    def __init__(self, k_index: int, k_point, decomposition: ReflectionRepresentationDecomposition, counts) -> None:
        self.k_index = int(k_index)
        self.k_point = np.asarray(k_point, dtype=np.float64).reshape(3)
        self.decomposition = decomposition
        self.sizes = tuple(int(size) for size in decomposition.sector_sizes)
        self.counts = [int(count) for count in counts]
        self._multiplicities = np.asarray(decomposition.reduction.multiplicities, dtype=np.float64)
        self._lock = threading.Lock()
        # by sector: (kinetic, projectors, empty stand-in of the real stencils), built for the sectors asked for
        self._static: dict[int, tuple] = {}
        self._potential: tuple[Any, np.ndarray | None] = (None, None)
        self._exchange: dict[int, tuple] = {}  # by sector: (exchange term of the whole grid, its reduction)
        self._gathers: dict[int, sp.csr_matrix] = {}  # by sector: U^H (shared by the tasks of the k-point)
        self._expanders: dict[tuple, tuple] = {}  # by (sector, device): the rows and factors of U on a device
        # whether the sectors assume an invariance the density is not symmetrized over, and the operations of the
        # symmetry guard (set by plan_sectors)
        self.assumed = False
        self.guard_group: SectorGuardGroup | None = None

    @property
    def order(self) -> int:
        return self.decomposition.representation_count

    def twin(self, k_index: int) -> "KPointSectors":
        """The sectors of the same k-point for another task (the other channel of a spin-polarized run): the bases
        and the static operators shared, the numbers of states and the potential of its own."""

        other = object.__new__(KPointSectors)
        other.__dict__.update(self.__dict__)
        other.k_index = int(k_index)
        other.counts = list(self.counts)
        other._potential = (None, None)
        other._exchange = {}
        return other

    def _gather(self, sector: int) -> sp.csr_matrix:
        """``U^H`` of ``sector`` as a sparse matrix (its rows by the whole grid's), made once."""

        found = self._gathers.get(sector)
        if found is None:
            decomposition = self.decomposition
            rows = decomposition.reduction.full_to_wedge.shape[0]
            phases = decomposition.phases[sector].astype(np.float64)
            gather = sp.csr_matrix(
                (phases, (np.asarray(decomposition.reduction.full_to_wedge, dtype=np.int64), np.arange(rows))),
                shape=(decomposition.wedge_size, rows),
            )
            orbits = decomposition.sector_orbit_indices(sector)
            scale = sp.diags(1.0 / np.sqrt(self._multiplicities[orbits]))
            found = self._gathers[sector] = (scale @ gather[orbits]).tocsr().astype(np.complex128)
        return found

    def reduce(self, sector: int, block):
        """``U^H x`` on the rows of ``sector`` for the columns of ``block`` (rows of the whole grid): the adjoint
        of :meth:`orbitals`.  A block on a device (the exchange operator of a hybrid, built where its k-point is
        solved) is reduced there, and the result stays there."""

        gather = self._gather(sector)
        get = getattr(block, "get", None)
        if get is not None and not isinstance(block, np.ndarray):
            import cupy as cp
            import cupyx.scipy.sparse as csp

            with cp.cuda.Device(int(block.device.id)):
                device = csp.csr_matrix(gather)
                source = block[:, None] if block.ndim == 1 else block
                return cp.asfortranarray(device @ cp.ascontiguousarray(source, dtype=cp.complex128))
        block = np.asarray(block, dtype=np.complex128)
        block = block[:, None] if block.ndim == 1 else block
        return np.asarray(gather @ block)

    def _reduced_exchange(self, sector: int, exchange):
        """The compressed exchange ``-a_x xi xi^H`` of the whole grid on the rows of ``sector``: ``U^H xi`` (made
        once per term and sector)."""

        from parsec_python.V_xc.bloch_exchange import BlochCompressedExchange

        cached = self._exchange.get(sector)
        if cached is not None and cached[0] is exchange:
            return cached[1]
        projectors = getattr(exchange, "projectors", None)
        if projectors is None or getattr(exchange, "released", False):
            raise RuntimeError("the exchange operator of this Hamiltonian was released")
        reduced = BlochCompressedExchange(
            self.reduce(sector, projectors),
            fraction=float(exchange.fraction),
            asymmetry=float(getattr(exchange, "asymmetry", 0.0)),
        )
        self._exchange[sector] = (exchange, reduced)
        return reduced

    def _build_static(self, full, sectors) -> None:
        """The kinetic matrices (with ``|k|^2``), projectors and empty stand-ins of the real stencils of ``sectors``
        (one pass over the operator of the whole grid for all of them)."""

        sectors = sorted({int(sector) for sector in sectors} - set(self._static))
        if not sectors:
            return
        k = self.k_point
        bloch = getattr(full, "bloch_laplacian", None)
        if bloch is not None:
            matrix = sp.csr_matrix(bloch, dtype=np.complex128)
        else:
            matrix = sp.csr_matrix(full.negative_laplacian, dtype=np.complex128)
            # sum_c k_c G_c, here only (the driver's cache of it would keep a matrix of the whole grid per k-point)
            k_gradient = None
            for axis, gradient in enumerate(full.gradient):
                if k[axis] != 0.0:
                    term = float(k[axis]) * sp.csr_matrix(gradient, dtype=np.float64)
                    k_gradient = term if k_gradient is None else k_gradient + term
            if k_gradient is not None:
                matrix = matrix - 2.0j * k_gradient
            matrix = matrix + float(k @ k) * sp.identity(matrix.shape[0], dtype=np.complex128, format="csr")
        kinetic = self.decomposition.reduce_hermitian_operators(matrix, sectors)
        del matrix
        projectors = compress_projectors_for_sectors(full.nonlocal_operator, self.decomposition.reduction)
        reduced = self.decomposition.reduce_nonlocal_operators(projectors, sectors)
        for sector in sectors:
            empty = sp.csr_matrix((self.sizes[sector], self.sizes[sector]), dtype=np.float64)
            self._static[sector] = (kinetic[sector], reduced[sector], empty)

    def _orbit_means(self, potential: np.ndarray) -> np.ndarray:
        cached, means = self._potential
        if cached is not potential:
            means = self.decomposition.invariant_wedge_values(potential)
            # the array is held, so that its identity is not reused by another one while cached
            self._potential = (potential, means)
        return means

    def hamiltonian(self, sector: int, full, wanted=None) -> KPointKohnShamHamiltonian:
        """The Hamiltonian of ``sector`` from the Bloch Hamiltonian ``full`` of the whole grid.

        The static operators of the sectors in ``wanted`` (those this process solves; all where ``None``) are built
        with those of ``sector`` the first time.  The exact exchange of a hybrid functional
        (``KPointExchangeHamiltonian``: ``base`` and ``exchange``) joins as ``-a_x (U^H xi)(U^H xi)^H``.
        """

        exchange = getattr(full, "exchange", None)
        if exchange is not None and getattr(full, "base", None) is not None:
            from parsec_python.Hamiltonian.kpoint_operator import KPointExchangeHamiltonian

            base = self.hamiltonian(sector, full.base, wanted)
            with self._lock:
                reduced = self._reduced_exchange(sector, exchange)
            return KPointExchangeHamiltonian(base, reduced)
        with self._lock:
            if sector not in self._static:
                self._build_static(full, range(self.order) if wanted is None else set(wanted) | {sector})
            kinetic, projectors, empty = self._static[sector]
            means = self._orbit_means(full.effective_potential)
        orbits = self.decomposition.sector_orbit_indices(sector)
        return SectorBlochHamiltonian(
            empty,
            (empty,) * 3,
            np.ascontiguousarray(means[orbits]),
            projectors,
            self.k_point,
            bloch_laplacian=kinetic,
        )

    def density(self, sector: int, sums: np.ndarray) -> np.ndarray:
        """``|U v|^2`` on the whole grid from ``q = sum_n f_n |v_n|^2`` on the rows of ``sector``."""

        decomposition = self.decomposition
        orbits = decomposition.sector_orbit_indices(sector)
        wedge = np.zeros(decomposition.wedge_size, dtype=np.float64)
        wedge[orbits] = np.asarray(sums, dtype=np.float64) / self._multiplicities[orbits]
        return wedge[decomposition.reduction.full_to_wedge]

    def expand(self, sector: int, block, xp=np):
        """:meth:`orbitals` where ``block`` is: a device block becomes ``U v`` on its device (one gather with the
        phase and orbit factor of every row, made once per sector and device)."""

        if xp is np or isinstance(block, np.ndarray):
            return self.orbitals(sector, np.asarray(block))
        device = int(block.device.id)
        found = self._expanders.get((sector, device))
        if found is None:
            decomposition = self.decomposition
            orbits = decomposition.sector_orbit_indices(sector)
            place = np.full(decomposition.wedge_size, -1, dtype=np.int64)
            place[orbits] = np.arange(orbits.size, dtype=np.int64)
            to_wedge = np.asarray(decomposition.reduction.full_to_wedge, dtype=np.int64)
            rows = place[to_wedge]
            factor = decomposition.phases[sector].astype(np.float64) / np.sqrt(self._multiplicities[to_wedge])
            factor[rows < 0] = 0.0
            rows[rows < 0] = 0
            with xp.cuda.Device(device):
                found = (xp.asarray(rows), xp.asarray(factor))
            self._expanders[(sector, device)] = found
        rows, factor = found
        return xp.take(block, rows, axis=0) * factor[:, None]

    def orbitals(self, sector: int, block: np.ndarray) -> np.ndarray:
        """``U v`` on the whole grid for the columns of ``block`` (rows of ``sector``)."""

        decomposition = self.decomposition
        block = np.asarray(block, dtype=np.complex128).reshape(-1, np.shape(block)[-1] if np.ndim(block) > 1 else 1)
        orbits = decomposition.sector_orbit_indices(sector)
        wedge = np.zeros((decomposition.wedge_size, block.shape[1]), dtype=np.complex128)
        wedge[orbits] = block / np.sqrt(self._multiplicities[orbits])[:, None]
        phases = decomposition.phases[sector].astype(np.float64)
        return phases[:, None] * wedge[decomposition.reduction.full_to_wedge]


def plan_sectors(job: KPointJob) -> list[KPointSectors | None] | None:
    """The sectors of every k-point of ``job`` (``None`` for a k-point that stays one task), ``None`` for none."""

    geometry = job.geometry
    grid = geometry["grid"]
    try:
        if _lattice_period(grid) is None or not _periodic_row_major(grid):
            return None
    except ValueError:  # not a bulk orthorhombic cell
        return None
    k_points = np.asarray(geometry["k_points"], dtype=np.float64).reshape(-1, 3)
    # the k-point of every task (the channels of a spin-polarized run share the k-points: tasks spin * n_k + k)
    k_of_task = np.asarray(geometry.get("k_of_task", np.arange(k_points.shape[0])), dtype=np.int64).reshape(-1)
    if k_of_task.shape[0] != job.n_tasks or k_of_task.size == 0 or int(k_of_task.max()) >= k_points.shape[0]:
        return None
    states, rows = int(job.number_of_states), int(grid.size)
    if rows * states < _MINIMUM_ENTRIES:  # before any search of the group: a small cell stays as it was
        return None
    cell = CellGroup(grid, geometry["atoms"])
    if cell.table is None or cell.order < 2:
        return None
    invariance = PotentialInvariance(
        cell, geometry.get("density_symmetry"), k_points, geometry["weights"], geometry["lattice"]
    )
    margin = sector_margin()
    cache = geometry.get("cache")
    signature = (
        int(job.n_tasks), states, rows, margin, _EXTRA_STATES, tuple(int(value) for value in k_of_task),
        tuple(np.round(k_points, 12).ravel().tolist()),
    )
    if isinstance(cache, dict):
        found = cache.get(_CACHE_KEY)
        if found is not None and found[0] == signature:
            return found[1]
    decompositions: dict[tuple[int, ...], ReflectionRepresentationDecomposition | None] = {}
    plans: list[KPointSectors | None] = []
    for index, k_point in enumerate(k_points):
        members = generated_subgroup(cell.table, cell.little_group(k_point, invariance))
        elements, generators = largest_commuting_involutions(cell.table, members)
        if len(elements) < 2:
            plans.append(None)
            continue
        key = tuple(elements)
        if key not in decompositions:
            decomposition = None
            reduction = little_group_reduction(cell, elements, generators)
            if reduction is not None:
                try:
                    decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                except RuntimeError:  # a character that no orbit admits (a grid too small for the group)
                    decomposition = None
            decompositions[key] = decomposition
        decomposition = decompositions[key]
        if decomposition is None:
            plans.append(None)
            continue
        sizes = decomposition.sector_sizes
        counts = [min(size, math.ceil(margin * states * size / rows) + _EXTRA_STATES) for size in sizes]
        if sum(size * count * count for size, count in zip(sizes, counts)) > _WORK_FRACTION * rows * states * states:
            plans.append(None)
            continue
        plan = KPointSectors(index, k_point, decomposition, counts)
        # the operations of the sectors that the density is not symmetrized over: the sectors assume the invariance
        # of the potential under them (the symmetry guard, SectorKPointSolver.start_spectrum_report)
        plan.assumed = any(not invariance.in_density_group(element) for element in elements[1:])
        plans.append(plan)
    if not any(plan is not None for plan in plans):
        return None
    guard = SectorGuardGroup(cell, invariance)
    for plan in plans:
        if plan is not None:
            plan.guard_group = guard
    # the tasks of one k-point share its bases and static operators (KPointSectors.twin)
    by_task: list[KPointSectors | None] = []
    taken: set[int] = set()
    for task, k_index in enumerate(int(value) for value in k_of_task):
        plan = plans[k_index]
        if plan is None:
            by_task.append(None)
        elif k_index in taken:
            by_task.append(plan.twin(task))
        else:
            taken.add(k_index)
            plan.k_index = task
            by_task.append(plan)
    if isinstance(cache, dict):
        # the next loops of this structure (the Fock loop of a hybrid) take these plans and their static operators
        cache[_CACHE_KEY] = (signature, by_task)
    return by_task


class SectorKPointSolver:
    """The solver of a k-point job whose k-points are split into the sectors of their little groups.

    The tasks of the inner solver (made by ``make_solver``, the solver the job would have had) are the pairs
    ``(k-point, sector)``, ``(k-point, None)`` for a k-point that stays whole, in the order of their work (rows
    times states squared, the largest first: the MPI pools take them in this order, by their sizes).  The driver sees one
    task per k-point as before: the lowest levels of the union of its sectors, their occupations sent back to
    the sectors that hold them.
    """

    backend_name = "kpoint-sectors"

    def __init__(self, job: KPointJob, plans: list[KPointSectors | None], make_solver: Callable[..., Any], options) -> None:
        self.job = job
        self._plans = plans
        self._make_solver = make_solver
        self._options = dict(options)
        self._grid_size = int(job.geometry["grid"].size)
        items: list[tuple[int, int | None]] = []
        for k_index in range(job.n_tasks):
            plan = plans[k_index]
            if plan is None:
                items.append((k_index, None))
            else:
                items.extend((k_index, sector) for sector in range(plan.order))
        self._items = sorted(items, key=lambda item: -self._rows(item) * self._states(item) ** 2)
        self._position = {item: position for position, item in enumerate(self._items)}
        self._kept: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        self._returned: dict[int, int] = {}
        self._regrowths = 0
        self._regrown_tasks = 0  # sectors solved again alone (their solver kept the others)
        self._rebuilds = 0  # growths that made a new inner solver
        # PARSEC_KPOINT_SECTOR_REFIT: the counts cut after the first solve (_refit), once per solver
        self._refit_wanted = sector_refit_setting() == "auto"
        self._refitted = False
        self._trimmed: set[tuple[int, int]] = set()  # the sectors whose counts the refit cut
        self._refit_states = 0  # the states the refit took off all sectors
        self._inner = None
        self._inner = self._new_inner()

    # ------------------------------------------------------------------ tasks
    def _rows(self, item) -> int:
        k_index, sector = item
        return self._grid_size if sector is None else self._plans[k_index].sizes[sector]

    def _states(self, item) -> int:
        k_index, sector = item
        return int(self.job.number_of_states) if sector is None else self._plans[k_index].counts[sector]

    def _new_inner(self):
        counts = tuple(self._states(item) for item in self._items)
        inner_job = KPointJob(
            len(self._items),
            max(counts),
            self.job.eigval_settings,
            self._make_hamiltonian,
            self._accumulate,
            states_per_task=counts,
            rows_per_task=tuple(self._rows(item) for item in self._items),
            grid_points=self._grid_size,
            extras=None if self.job.extras is None else self._inner_extras,
            extras_by_columns=self.job.extras is not None,
        )
        options = dict(self._options)
        options["problem_size"] = max(self._rows(item) for item in self._items)
        return self._make_solver(inner_job, **options)

    def _make_hamiltonian(self, task: int, potentials: dict):
        k_index, sector = self._items[task]
        full = self.job.make_hamiltonian(k_index, potentials)
        if sector is None:
            return full
        return self._plans[k_index].hamiltonian(sector, full, self._owned_sectors(k_index))

    def _owned_sectors(self, k_index: int):
        """The sectors of ``k_index`` that this process solves: those of the tasks of its pool (MPI pools), all
        sectors otherwise (``None``)."""

        tasks = getattr(self._inner, "tasks", None)
        if not isinstance(tasks, (list, tuple)) or not tasks:
            return None
        return [
            sector for position, (k, sector) in enumerate(self._items)
            if k == k_index and sector is not None and position in tasks
        ]

    def _accumulate(self, task: int, vectors, occupations, xp=np) -> dict:
        k_index, sector = self._items[task]
        if sector is None:
            return self.job.accumulate(k_index, vectors, occupations, xp=xp)
        sums = _host(weighted_square_sum(vectors, occupations, xp))
        whole = self._plans[k_index].density(sector, sums)
        # the job's own density of the k-point (its weight and normalization) from one column |U v|; a job that sums
        # more (the exchange energy of a hybrid) gives its density part apart
        geometry = self.job.geometry
        parts = geometry.get("density_part", self.job.accumulate)(k_index, np.sqrt(whole)[:, None], np.ones(1))
        scale = geometry.get("exchange_cross_scale")
        if scale is not None:
            parts = dict(parts)
            parts["exchange_cross"] = self._exchange_cross(k_index, sector, vectors, occupations, scale)
        return parts

    def _exchange_cross(self, k_index: int, sector: int, vectors, occupations, scale) -> np.ndarray:
        """``(s/2) w_k sum_n f_n <v_n|U^H V_ACE U|v_n>`` of a sector (unscaled): the exchange term reduced to it
        (made with its Hamiltonian, :meth:`KPointSectors.hamiltonian`), on the host."""

        from parsec_python.V_xc.exact_exchange import OCCUPATION_THRESHOLD

        cached = self._plans[k_index]._exchange.get(sector)
        values = np.asarray(_host(occupations), dtype=np.float64).reshape(-1)
        occupied = np.flatnonzero(values > OCCUPATION_THRESHOLD)
        if cached is None or not occupied.size:
            return np.zeros(1)
        block = np.asarray(_host(vectors))[:, occupied]
        reduced = cached[1]
        projectors = reduced.projectors
        if isinstance(projectors, np.ndarray):
            expectation = reduced.expectation_values(block, scaled=False)
        else:  # the reduction of a device exchange stays on its device: the overlaps are formed there
            import cupy as cp

            with cp.cuda.Device(int(projectors.device.id)):
                overlaps = projectors.conj().T @ cp.asarray(block)
                expectation = cp.asnumpy(cp.sum(cp.abs(overlaps) ** 2, axis=0))
            expectation = -expectation
        return np.asarray([float(scale(k_index)) * float(np.dot(values[occupied], expectation))])

    # ------------------------------------------------------------------- steps
    def _short_sectors(self, values) -> list[tuple[int, int]]:
        states = int(self.job.number_of_states)
        short = []
        for k_index, plan in enumerate(self._plans):
            if plan is None:
                continue
            levels = [np.asarray(values[self._position[(k_index, sector)]], dtype=float) for sector in range(plan.order)]
            union = np.sort(np.concatenate(levels))
            if union.size < states:
                short.extend((k_index, sector) for sector in range(plan.order) if plan.counts[sector] < plan.sizes[sector])
                continue
            cut = union[states - 1]
            for sector, part in enumerate(levels):
                if plan.counts[sector] < plan.sizes[sector] and (part.size == 0 or float(np.max(part)) < cut):
                    short.append((k_index, sector))
        return short

    def _deficits(self, short, values) -> dict[tuple[int, int], int]:
        """The estimated levels that each sector of ``short`` the refit cut lacks below the cut of its k-point
        (:func:`..Eigensolvers.symmetry.bracket_deficits`, every sector counted once)."""

        cut = [item for item in short if item in self._trimmed]
        if not cut:
            return {}
        from ..Eigensolvers.symmetry import bracket_deficits

        found: dict[tuple[int, int], int] = {}
        for k_index in sorted({k for k, _ in cut}):
            plan = self._plans[k_index]
            levels = [np.asarray(values[self._position[(k_index, sector)]], dtype=float) for sector in range(plan.order)]
            if any(part.size == 0 for part in levels):
                continue
            failing = [sector for k, sector in cut if k == k_index]
            try:
                estimate = bracket_deficits(levels, [1] * plan.order, int(self.job.number_of_states), failing)
            except (ValueError, IndexError):
                continue
            found.update({(k_index, sector): int(value) for sector, value in estimate.items()})
        return found

    def _refit(self, values) -> None:
        """Cut the states of every sector to the levels it holds among those its k-point keeps, plus a buffer.

        The first counts carry the margin of :func:`sector_margin` over each sector's share of the rows, which a
        sector needs before its levels are known; after the first solve of the loop they are (the bracket of
        :meth:`_short_sectors` holds).  A sector that holds ``h`` of the ``M`` levels of its k-point keeps
        :func:`refitted_count` states.  The inner solver cuts their Ritz states at their next solves
        (``trim_tasks``: the lowest Ritz pairs are kept, nothing is solved again); a cut sector that later misses
        levels grows by its estimated deficit (:meth:`_deficits`).  Once per solver, where the inner solver can.
        """

        self._refitted = True
        trim = getattr(self._inner, "trim_tasks", None)
        if trim is None:
            return
        states = int(self.job.number_of_states)
        counts: dict[int, int] = {}
        cut: list[tuple[int, int, int]] = []
        for k_index, plan in enumerate(self._plans):
            if plan is None:
                continue
            parts = [np.asarray(values[self._position[(k_index, sector)]], dtype=float) for sector in range(plan.order)]
            levels = np.concatenate(parts)
            if levels.size < states:
                continue
            owners = np.concatenate([np.full(part.size, sector, dtype=np.int64) for sector, part in enumerate(parts)])
            held = np.bincount(owners[np.argsort(levels, kind="stable")[:states]], minlength=plan.order)
            for sector in range(plan.order):
                count = refitted_count(int(held[sector]), plan.sizes[sector], plan.counts[sector])
                if count < plan.counts[sector]:
                    cut.append((k_index, sector, count))
                    counts[self._position[(k_index, sector)]] = count
        if not counts or not trim(counts):
            return
        for k_index, sector, count in cut:
            self._refit_states += self._plans[k_index].counts[sector] - count
            self._plans[k_index].counts[sector] = count
            self._trimmed.add((k_index, sector))

    def _grow(self, short, potentials: dict, filter_degree: int, tolerance, values=None) -> list[np.ndarray]:
        """More states for the sectors ``short``, and the step solved again.

        Only those sectors are solved again where the inner solver can change the states of some tasks
        (``resize_tasks``: the others keep their Ritz states and levels); else a new inner solver solves every
        sector from the start.  ``values``: the levels of the solve that fell short (the deficits of the sectors the
        refit cut).
        """
        counts = {}
        deficits = self._deficits(short, values) if values is not None else {}
        for k_index, sector in short:
            plan = self._plans[k_index]
            count = plan.counts[sector]
            if (k_index, sector) in deficits:
                # a sector the refit cut grows by its estimated deficit (the Gamma route's rule, bracket_deficits)
                grown = count + _EXTRA_STATES + deficits[(k_index, sector)]
            else:
                grown = max(math.ceil(_GROWTH * count), count + _EXTRA_STATES)
            plan.counts[sector] = min(plan.sizes[sector], grown)
            counts[self._position[(k_index, sector)]] = plan.counts[sector]
        self._regrowths += 1
        resize = getattr(self._inner, "resize_tasks", None)
        if resize is not None and resize(counts):
            self._regrown_tasks += len(counts)
            return self._inner.solve(potentials, filter_degree, tolerance, tasks=sorted(counts))
        self._rebuilds += 1
        self._inner.close()
        self._inner = None
        self._inner = self._new_inner()
        return self._inner.solve(potentials, filter_degree, tolerance)

    def solve(self, potentials: dict, filter_degree: int, tolerance: float | None = None) -> list[np.ndarray]:
        """The lowest ``number_of_states`` levels of every k-point (task order)."""

        values = self._inner.solve(potentials, filter_degree, tolerance)
        for _ in range(_ATTEMPTS):
            short = self._short_sectors(values)
            if not short:
                union = self._union(values)
                if self._refit_wanted and not self._refitted:
                    self._refit(values)
                return union
            values = self._grow(short, potentials, filter_degree, tolerance, values)
        raise RuntimeError(
            f"the sectors of the k-points did not hold the lowest levels after {_ATTEMPTS} attempts; "
            f"{_SWITCH}=off solves each k-point whole"
        )

    def _union(self, values) -> list[np.ndarray]:
        states = int(self.job.number_of_states)
        result = []
        self._returned = {position: int(np.size(values[position])) for position in range(len(self._items))}
        for k_index in range(self.job.n_tasks):
            plan = self._plans[k_index]
            if plan is None:
                self._kept[k_index] = None
                result.append(np.asarray(values[self._position[(k_index, None)]], dtype=float))
                continue
            parts = [np.asarray(values[self._position[(k_index, sector)]], dtype=float) for sector in range(plan.order)]
            levels = np.concatenate(parts)
            owners = np.concatenate([np.full(part.size, sector, dtype=np.int64) for sector, part in enumerate(parts)])
            indices = np.concatenate([np.arange(part.size, dtype=np.int64) for part in parts])
            order = np.argsort(levels, kind="stable")[:states]
            self._kept[k_index] = (owners[order], indices[order])
            result.append(levels[order])
        return result

    def accumulate(self, occupations: list[np.ndarray]) -> dict[str, np.ndarray]:
        """The density of the occupations of the levels of every k-point (those of :meth:`solve`)."""

        inner: list[np.ndarray] = []
        for position, (k_index, sector) in enumerate(self._items):
            if sector is None:
                inner.append(np.asarray(occupations[k_index], dtype=np.float64))
            else:
                inner.append(np.zeros(self._returned[position], dtype=np.float64))
        for k_index, kept in self._kept.items():
            if kept is None:
                continue
            owners, indices = kept
            values = np.asarray(occupations[k_index], dtype=np.float64)
            for sector in range(self._plans[k_index].order):
                mask = owners == sector
                if np.any(mask):
                    inner[self._position[(k_index, sector)]][indices[mask]] = values[mask]
        return self._inner.accumulate(inner)

    def residuals(self) -> list[float]:
        inner = self._inner.residuals()
        result = []
        for k_index in range(self.job.n_tasks):
            plan = self._plans[k_index]
            items = [(k_index, None)] if plan is None else [(k_index, sector) for sector in range(plan.order)]
            norms = np.asarray([inner[self._position[item]] for item in items], dtype=float)
            finite = norms[np.isfinite(norms)]
            result.append(float(np.max(finite)) if finite.size else float("nan"))
        return result

    def vectors(self, task: int) -> np.ndarray:
        """The kept orbitals of k-point ``task`` on the whole grid (an empty block where a rank holds none)."""

        plan = self._plans[task]
        if plan is None:
            return self._inner.vectors(self._position[(task, None)])
        owners, indices = self._kept[task]
        columns = np.zeros((self._grid_size, owners.size), dtype=np.complex128)
        for sector in range(plan.order):
            mask = owners == sector
            if not np.any(mask):
                continue
            block = _host(self._inner.vectors(self._position[(task, sector)]))
            if block.ndim != 2 or block.shape[1] == 0:
                return np.zeros((0, 0), dtype=np.complex128)
            columns[:, mask] = plan.orbitals(sector, block[:, indices[mask]])
        return columns

    def extras(self) -> list[Any]:
        """The job's extras (the nonlocal forces) of every k-point: the sums of its sectors (:meth:`_inner_extras`,
        computed by the inner solver where the orbitals are: a process, or the ranks of the MPI pools)."""

        if self.job.extras is None:
            raise NotImplementedError("the job computes no per-state extras")
        inner = self._inner.extras()
        found: list[Any] = [None] * self.job.n_tasks
        for position, (k_index, _sector) in enumerate(self._items):
            value = np.asarray(_host(inner[position]), dtype=np.float64)
            found[k_index] = value if found[k_index] is None else found[k_index] + value
        return found

    def _inner_extras(self, position: int, vectors, xp=np, columns=None):
        """The extras of the states ``columns`` of task ``position`` (a sector): its kept states, ``U v`` on the
        whole grid a block of :data:`_FORCE_COLUMNS` at a time, with their places among the levels of the k-point
        (the occupations the job holds)."""

        k_index, sector = self._items[position]
        count = int(vectors.shape[1])
        columns = np.arange(count, dtype=np.int64) if columns is None else np.asarray(columns, dtype=np.int64)
        if sector is None:
            return self.job.extras(k_index, vectors, xp=xp, columns=columns)
        owners, indices = self._kept[k_index]
        mine = np.flatnonzero(owners == sector)
        size = int(max(int(indices[mine].max()) + 1 if mine.size else 0, int(columns.max()) + 1 if columns.size else 0))
        place = np.full(max(size, 1), -1, dtype=np.int64)
        place[indices[mine]] = mine
        wanted = place[columns] if columns.size else np.zeros(0, dtype=np.int64)
        keep = np.flatnonzero(wanted >= 0)
        plan = self._plans[k_index]
        total = None
        for c0 in range(0, int(keep.size), _FORCE_COLUMNS):
            chosen = keep[c0 : c0 + _FORCE_COLUMNS]
            block = vectors[:, chosen if xp is np else xp.asarray(chosen)]
            piece = np.asarray(
                _host(self.job.extras(k_index, plan.expand(sector, block, xp), xp=xp, columns=wanted[chosen])),
                dtype=np.float64,
            )
            total = piece if total is None else total + piece
        if total is None:
            total = np.zeros((len(self.job.geometry["atoms"]), 3), dtype=np.float64)
        return total

    # ----------------------------------------------------------- the symmetry guard
    def guarded_tasks(self) -> list[int]:
        """The tasks whose sectors assume an invariance of the potential that the density is not symmetrized over
        (a single Gamma point, a full mesh, the translations of a supercell): those the symmetry guard tests."""

        return [task for task, plan in enumerate(self._plans) if plan is not None and getattr(plan, "assumed", False)]

    def _task_weight(self, task: int) -> float:
        geometry = self.job.geometry
        k_of_task = np.asarray(geometry.get("k_of_task", np.arange(len(geometry["weights"]))), dtype=np.int64)
        return float(np.asarray(geometry["weights"], dtype=np.float64)[int(k_of_task[task])])

    def _guard_levels(self, task: int, levels, occupations, capacity: float):
        owners, _ = self._kept[task]
        values = np.asarray(levels[task], dtype=np.float64)
        filled = np.asarray(occupations[task], dtype=np.float64) / float(capacity)
        return values, filled, np.asarray(owners, dtype=np.int64)

    def start_spectrum_report(self, levels, occupations, *, capacity: float = 1.0):
        """Test 1 of the guard: the partly filled multiplets of the levels of the start Hamiltonian, ``None`` where
        there are none.

        ``levels`` and ``occupations`` are those of every task (the levels of :meth:`solve`), ``capacity`` the
        occupation of a full level (1 for a spin channel).  As for the sectors of a cluster
        (:func:`.spin_sectors.partly_filled_multiplets`): a multiplet of :func:`.spin_sectors.sector_multiplets`
        (levels in two or more sectors of a guarded task) of ``d`` levels holding ``n`` electrons is partly filled
        when ``w_k min(n, d - n)`` exceeds :data:`.spin_sectors.SPIN_SECTOR_OCCUPATION_TOLERANCE`, ``w_k`` the
        weight of its k-point (the electrons it adds to the density; 1 at a single Gamma point).
        """

        from .spin_sectors import SPIN_SECTOR_OCCUPATION_TOLERANCE, FractionalOccupations

        largest = 0.0
        lines: list[str] = []
        for part, line in self._start_multiplet_parts(levels, occupations, capacity):
            if part <= SPIN_SECTOR_OCCUPATION_TOLERANCE:
                continue
            largest = max(largest, part)
            lines.append(line())
        if not lines:
            return None
        return FractionalOccupations(
            largest=largest,
            tolerance=float(SPIN_SECTOR_OCCUPATION_TOLERANCE),
            lines=tuple(lines),
            reason="a partly filled multiplet",
            measure="largest w_k min(n, d - n) of d levels holding n electrons",
        )

    def _start_multiplet_parts(self, levels, occupations, capacity: float):
        """``(w_k min(n, d - n), line)`` of every multiplet of :meth:`start_spectrum_report`, below its tolerance as
        well; ``line()`` describes it."""

        from .spin_sectors import START_MULTIPLET_WIDTH, sector_multiplets

        for task in self.guarded_tasks():
            values, filled, sectors = self._guard_levels(task, levels, occupations, capacity)
            weight = self._task_weight(task)
            for group in sector_multiplets(values, sectors):
                electrons = float(np.sum(filled[group]))
                part = weight * min(electrons, len(group) - electrons)

                def line(task=task, weight=weight, group=group, electrons=electrons, values=values, sectors=sectors):
                    text = " ".join(f"{values[member] * 13.605693122994:.4f}" for member in group)
                    labels = " ".join(str(int(sectors[member])) for member in group)
                    return (
                        f"task {task} (k-point weight {weight:.4g}): {len(group)} levels within "
                        f"{START_MULTIPLET_WIDTH:g} Ry at {text} eV hold {electrons:.4f} electrons (sectors {labels})"
                    )

                yield part, line

    def start_spectrum_measure(self, levels, occupations, *, capacity: float = 1.0) -> float:
        """The value test 1 compares with its tolerance: the largest ``w_k min(n, d - n)`` of the multiplets of
        :meth:`start_spectrum_report`, 0 where there are none (for the report of the guard that passed)."""

        return max((part for part, _ in self._start_multiplet_parts(levels, occupations, capacity)), default=0.0)

    def density_symmetry_measure(self, densities, volume_element: float) -> float:
        """Test 2 of the guard: the largest ``int |rho(g r) - rho(r)|`` (electrons) of the channel ``densities``
        (whole grid) over the operations of :class:`SectorGuardGroup` (0 where there are none)."""

        return max((value for _, value in self._asymmetries(densities, volume_element)), default=0.0)

    def _asymmetries(self, densities, volume_element: float) -> list[tuple[str, float]]:
        guarded = self.guarded_tasks()
        if not guarded:
            return []
        group = self._plans[guarded[0]].guard_group
        if group is None:
            return []
        found = []
        for index, rows in group.images():
            for channel, density in enumerate(densities):
                values = np.asarray(density, dtype=np.float64)
                deviation = float(volume_element * np.sum(np.abs(values[rows] - values)))
                found.append((f"channel {channel + 1}, operation {group.operation_text(index)}", deviation))
        return found

    def density_symmetry_report(self, densities, volume_element: float, tolerance: float | None = None):
        """Test 2 as a report (:class:`.spin_sectors.FractionalOccupations`), ``None`` below ``tolerance``
        (:data:`.spin_sectors.DENSITY_SYMMETRY_TOLERANCE`)."""

        from .spin_sectors import DENSITY_SYMMETRY_TOLERANCE, FractionalOccupations

        tolerance = DENSITY_SYMMETRY_TOLERANCE if tolerance is None else float(tolerance)
        found = self._asymmetries(densities, volume_element)
        largest = max((value for _, value in found), default=0.0)
        if largest < tolerance:
            return None
        return FractionalOccupations(
            largest=largest,
            tolerance=tolerance,
            lines=tuple(
                f"{label}: the density differs by {value:.4f} electrons from its image" for label, value in found
                if value >= tolerance
            ),
            reason="a channel density that breaks the symmetry the k-point sectors assume",
            measure="largest int |rho(g r) - rho(r)| (electrons)",
        )

    def fractional_report(self, levels, occupations, *, capacity: float = 1.0):
        """Test 3 of the guard (Fermi-Dirac occupations, at the end of a loop): a level of a multiplet whose levels
        lie in two or more sectors of a guarded task holds ``w_k min(f, 1 - f)`` of at least
        :data:`.spin_sectors.SPIN_SECTOR_OCCUPATION_TOLERANCE` (:func:`.spin_sectors.fractional_occupations`), else
        ``None``."""

        from .spin_sectors import SPIN_SECTOR_OCCUPATION_TOLERANCE, FractionalOccupations

        largest = 0.0
        lines: list[str] = []
        for part, line in self._fractional_parts(levels, occupations, capacity):
            if part < SPIN_SECTOR_OCCUPATION_TOLERANCE:
                continue
            largest = max(largest, part)
            lines.append(line())
        if not lines:
            return None
        return FractionalOccupations(
            largest=largest, tolerance=float(SPIN_SECTOR_OCCUPATION_TOLERANCE), lines=tuple(lines),
            measure="largest w_k min(f, 1-f)",
        )

    def _fractional_parts(self, levels, occupations, capacity: float):
        """``(w_k max min(f, 1 - f), line)`` of every multiplet of :meth:`fractional_report`, below its tolerance as
        well; ``line()`` describes it."""

        from .spin_sectors import MULTIPLET_WIDTH, _multiplets, sector_multiplets

        for task in self.guarded_tasks():
            values, filled, sectors = self._guard_levels(task, levels, occupations, capacity)
            weight = self._task_weight(task)
            fractional = weight * np.minimum(filled, 1.0 - filled)
            groups = [group for group in _multiplets(values, MULTIPLET_WIDTH) if np.unique(sectors[group]).size >= 2]
            exact = {tuple(group) for group in groups}
            groups += [group for group in sector_multiplets(values, sectors) if tuple(group) not in exact]
            for group in groups:
                part = float(np.max(fractional[group]))

                def line(task=task, weight=weight, group=group, values=values, filled=filled, sectors=sectors):
                    text = " ".join(f"{values[member] * 13.605693122994:.4f}" for member in group)
                    shares = " ".join(f"{filled[member]:.4f}" for member in group)
                    labels = " ".join(str(int(sectors[member])) for member in group)
                    return (
                        f"task {task} (k-point weight {weight:.4g}): {len(group)} level(s) at {text} eV "
                        f"(occupations {shares}; sectors {labels})"
                    )

                yield part, line

    def fractional_measure(self, levels, occupations, *, capacity: float = 1.0) -> float:
        """The value test 3 compares with its tolerance: the largest ``w_k min(f, 1 - f)`` of the multiplets of
        :meth:`fractional_report`, 0 where there are none (for the report of the guard that passed)."""

        return max((part for part, _ in self._fractional_parts(levels, occupations, capacity)), default=0.0)

    def _layout(self) -> list[tuple[int, int, int]]:
        return [(int(k), -1 if sector is None else int(sector), self._rows((k, sector))) for k, sector in self._items]

    def export_states(self) -> dict:
        """The Ritz states of the sectors with their layout and numbers of states (for the next loop of the same
        structure or of a moved one: :meth:`import_states`)."""

        counts = [self._states(item) for item in self._items]
        return {_STATES_KEY: {"layout": self._layout(), "counts": counts, "states": self._inner.export_states()}}

    def import_states(self, states: dict) -> None:
        """Start from the states of :meth:`export_states`: those of sectors of the same layout (k-points, sectors,
        rows), with their numbers of states; anything else (the states of a solver without sectors, or of other
        sectors) is not taken, and the sectors start afresh."""

        carried = states.get(_STATES_KEY) if isinstance(states, dict) else None
        if not isinstance(carried, dict) or [tuple(item) for item in carried.get("layout", ())] != self._layout():
            return
        counts = [int(count) for count in carried["counts"]]
        if counts != [self._states(item) for item in self._items]:
            for (k_index, sector), count in zip(self._items, counts):
                if sector is not None:
                    self._plans[k_index].counts[sector] = count
            self._inner.close()
            self._inner = None
            self._inner = self._new_inner()
        self._inner.import_states(carried["states"])

    def device_vectors(self, task: int) -> np.ndarray:
        """The kept orbitals of k-point ``task`` on the whole grid (host), for a driver that asks for them where the
        solver keeps them (the Fock loop of a hybrid)."""

        return self.vectors(task)

    def stats(self) -> dict[str, float]:
        inner = getattr(self._inner, "stats", None)
        stats = dict(inner()) if inner is not None else {}
        rows, states = self._grid_size, int(self.job.number_of_states)
        split = [plan for plan in self._plans if plan is not None]
        whole_work = float(rows) * states * states * self.job.n_tasks
        work = sum(self._rows(item) * self._states(item) ** 2 for item in self._items)
        stats.update(
            {
                "kpoint_sector_tasks": float(len(self._items)),
                "kpoint_sector_kpoints": float(len(split)),
                "kpoint_sector_largest_group": float(max(plan.order for plan in split)),
                "kpoint_sector_work_fraction": float(work) / whole_work,
                "kpoint_sector_regrowths": float(self._regrowths),
                "kpoint_sector_regrown_tasks": float(self._regrown_tasks),
                "kpoint_sector_rebuilds": float(self._rebuilds),
                "kpoint_sector_refit_states": float(self._refit_states),
                "kpoint_sector_refit_sectors": float(len(self._trimmed)),
            }
        )
        return stats

    def describe_sectors(self) -> str:
        orders = [1 if plan is None else plan.order for plan in self._plans]
        return f"k-point sectors: {len(self._items)} tasks for {self.job.n_tasks} k-points (sectors per k-point {orders})"

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()

    def __enter__(self) -> "SectorKPointSolver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __getattr__(self, name: str):
        # what the drivers read of the solver they made (the projector builder and density symmetrizer of the MPI
        # pools, its backend, ...) is the inner solver's
        if name.startswith("__") or name in {"_inner", "job", "_plans", "_items"}:
            raise AttributeError(name)
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


def sector_solver(job: KPointJob, make_solver: Callable[..., Any], **options):
    """A :class:`SectorKPointSolver` for ``job`` where its k-points split, else ``None``.

    A job with extras takes them where they sum over the states (the forces of the scalar SCF,
    ``extras_by_columns``): each sector's part from its kept states, in one process or on the ranks of the MPI pools.
    """

    if (
        getattr(job, "geometry", None) is None
        or (job.extras is not None and not job.extras_by_columns)
        or job.states_per_task is not None
        or kpoint_sectors_setting() == "off"
    ):
        return None
    plans = plan_sectors(job)
    if plans is None:
        return None
    return SectorKPointSolver(job, plans, make_solver, options)


__all__ = [
    "CellGroup",
    "KPointSectors",
    "LittleGroupReduction",
    "PotentialInvariance",
    "SectorBlochHamiltonian",
    "SectorGuardGroup",
    "SectorKPointSolver",
    "kpoint_sectors_setting",
    "generated_subgroup",
    "largest_commuting_involutions",
    "little_group_reduction",
    "plan_sectors",
    "carried_state",
    "refitted_count",
    "sector_guard_setting",
    "sector_margin",
    "sector_refit_setting",
    "sector_solver",
    "trimmed_state",
]
