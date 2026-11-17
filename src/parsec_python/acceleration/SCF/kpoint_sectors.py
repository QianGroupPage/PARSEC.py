"""The k-points of a periodic SCF in the sectors of their little groups (``PARSEC_KPOINT_SECTORS``).

The Bloch Hamiltonian of a k-point acts on the periodic factor ``u_k`` (the u gauge),
``H_k = e^{-ik.r} H e^{ik.r}``.  An operation ``g: r -> R r + t`` of a bulk orthorhombic cell (a Cartesian
signed permutation ``R`` with a translation ``t`` of whole grid steps,
:func:`..Symmetry.axis_reflection._space_group_operations`) permutes the grid points, ``(P_g f)(r) = f(g^-1 r)``,
and commutes with ``H``.  It moves the phase, ``P_g e^{ik.r} P_g^-1 = e^{i (R k).(r - t)}``, so where ``R k = k``
it commutes with ``H_k`` as well (the constant ``e^{-ik.t}`` drops out).  ``R k = k`` exactly, not modulo a
reciprocal lattice vector: the finite-difference ``-nabla^2 - 2ik.grad + |k|^2`` of the u gauge (and the Bloch
Laplacian of ``PARSEC_KPOINT_KINETIC=bloch``) is not invariant under the change of gauge ``u -> e^{-iG.r} u``
that ``R k = k + G`` would need.  These operations form the little group ``G_k``.

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
sector that falls short gets 1.5 times its states and the step is solved again, from new starts.  A k-point
whose sectors would not halve its work (``sum n_s M_s^2 <= N M^2 / 2``) stays one task, and so do the k-points
of a basis of fewer than ``2^24`` entries (``N M``, 268 MB of complex vectors): their eigenproblems do not fill a
device, and more of them would add the cost of the tasks (the 4631 k-points of an 8-atom cube with a 21^3 mesh).

The density of a sector, ``|U v|^2``, is ``q_s / m_w`` on the rows of the orbit ``w`` of its column ``s``
(``q = sum_n f_n |v_n|^2``, ``m_w`` the orbit size): expanded on the host and summed by the job of the driver.

The scalar k-point SCF of a bulk orthorhombic cell takes it, without spin, hybrid, meta-GGA, LDA+U, forces or
kept orbitals (the driver then gives the job its ``geometry``); ``PARSEC_KPOINT_SECTORS=off`` keeps one task per
k-point, bit for bit as before.
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
_MARGIN_SWITCH = "PARSEC_KPOINT_SECTOR_MARGIN"
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
    mappings = []
    for index in elements:
        mapping = cell.row_map(index, table=row_table)
        if mapping is None:
            return None
        mappings.append(mapping)
    representative_rows, full_to_wedge, multiplicities = _orbit_labels(mappings)
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
        return self._in_density_group(index) or self._maps_kpoints(matrix)

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

    @property
    def order(self) -> int:
        return self.decomposition.representation_count

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
        with those of ``sector`` the first time.
        """

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
    if k_points.shape[0] != job.n_tasks:
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
        plans.append(KPointSectors(index, k_point, decomposition, counts))
    return plans if any(plan is not None for plan in plans) else None


class SectorKPointSolver:
    """The solver of a k-point job whose k-points are split into the sectors of their little groups.

    The tasks of the inner solver (made by ``make_solver``, the solver the job would have had) are the pairs
    ``(k-point, sector)``, ``(k-point, None)`` for a k-point that stays whole, in the order of their work (rows
    times states squared, the largest first: the MPI pools deal the tasks round-robin).  The driver sees one
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
        # the job's own density of the k-point (its weight and normalization) from one column |U v|
        return self.job.accumulate(k_index, np.sqrt(whole)[:, None], np.ones(1))

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

    def _grow(self, short) -> None:
        for k_index, sector in short:
            plan = self._plans[k_index]
            count = plan.counts[sector]
            plan.counts[sector] = min(plan.sizes[sector], max(math.ceil(_GROWTH * count), count + _EXTRA_STATES))
        self._regrowths += 1
        self._inner.close()
        self._inner = None
        self._inner = self._new_inner()

    def solve(self, potentials: dict, filter_degree: int, tolerance: float | None = None) -> list[np.ndarray]:
        """The lowest ``number_of_states`` levels of every k-point (task order)."""

        for _ in range(_ATTEMPTS):
            values = self._inner.solve(potentials, filter_degree, tolerance)
            short = self._short_sectors(values)
            if not short:
                return self._union(values)
            self._grow(short)
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
        raise NotImplementedError("the sectors of the k-points compute no per-state extras")

    def export_states(self) -> dict[int, Any]:
        return self._inner.export_states()

    def import_states(self, states: dict[int, Any]) -> None:
        self._inner.import_states(states)

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
    """A :class:`SectorKPointSolver` for ``job`` where its k-points split, else ``None``."""

    if (
        getattr(job, "geometry", None) is None
        or job.extras is not None
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
    "SectorKPointSolver",
    "kpoint_sectors_setting",
    "generated_subgroup",
    "largest_commuting_involutions",
    "little_group_reduction",
    "plan_sectors",
    "sector_margin",
    "sector_solver",
]
