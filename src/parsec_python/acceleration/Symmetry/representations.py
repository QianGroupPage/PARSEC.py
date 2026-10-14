"""Real one-dimensional representations of a commuting reflection group.

For an orbit ``O_w`` with stabilizer ``S_w``, a real character ``chi_Gamma``
contributes one normalized orbital degree of freedom exactly when

``chi_Gamma(s) = 1  for every s in S_w``.

On an admitted orbit the expansion is

``U_Gamma[i,w] = chi_Gamma(g_i) / sqrt(|O_w|)``,

where ``g_i`` maps the selected representative to full-grid row ``i``.  An
orbit rejected by the stabilizer rule contributes no column to that sector.
This is the standard finite-group projector, not a boundary approximation:

``P_Gamma = |G|^-1 sum_g chi_Gamma(g) R(g)``.

Free half-shifted grids are the special case ``S_w={e}``; every sector then
retains the former common wedge dimension and byte-for-byte phase convention.
Zero-shift planes and axes instead give different, mathematically required
sector dimensions (for example, odd reflection states vanish on a mirror
plane).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import product
import os
import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import RealSpaceGrid
from parsec_python.V_ion import NonlocalProjectorOperator

from .axis_reflection import (
    AxisReflectionReduction,
    SignedPermutationReduction,
    _RowOrderTable,
    _fast_maps_requested,
    _grid_row_mapping_operation,
)


# PARSEC's D2h order is Ag, B1g, B2g, B3g, Au, B1u, B2u, B3u.  For diagonal
# Cartesian signs these are respectively the scalar monomials
# 1, yz, xz, xy, xyz, x, y, z.  Deduplicating this order after restriction to
# a detected subgroup also gives deterministic labels for smaller groups.
_PARSEC_REFLECTION_PARITIES = (
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)


def operator_build_workers(representation_count: int) -> int:
    """Return the measured cold-setup worker count.

    A/B measurements show that a four-worker pool pays for itself when there
    are several independent representations, while creating a pool for only
    two representations does not improve complete first-calculation time.
    An explicit environment setting remains authoritative for profiling.
    """

    count = int(representation_count)
    if count < 1:
        raise ValueError("representation_count must be positive")
    raw = os.environ.get("PARSEC_SYMMETRY_OPERATOR_WORKERS")
    if raw is None:
        return 1 if count <= 2 else min(count, 4)
    raw = raw.strip()
    try:
        requested = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_SYMMETRY_OPERATOR_WORKERS must be a positive integer"
        ) from error
    if requested < 1:
        raise ValueError(
            "PARSEC_SYMMETRY_OPERATOR_WORKERS must be a positive integer"
        )
    return min(count, requested)


def _representative_images(
    grid: RealSpaceGrid,
    reduction: AxisReflectionReduction,
) -> np.ndarray:
    """Return the full row of every operation applied to every representative.

    The decomposition reads a row map only at the orbit representatives, and
    the detector has already accepted every operation on the whole grid, so
    only those rows are mapped and no grid-sized temporary is built.

    With ``PARSEC_SYMMETRY_FAST_MAPS`` the whole-grid map of an operation is
    read out of the lookup table in the order of the rows, which costs less
    than the coordinates of the representatives alone, and its entries at
    the representatives are kept.  An operation without a whole-grid map
    is still decided on the representatives, and so is every operation of a
    grid whose rows are not numbered along its table.
    """

    operations = (
        reduction.operations
        if isinstance(reduction, SignedPermutationReduction)
        else [np.diag(np.asarray(signs, dtype=np.int8)) for signs in reduction.signs]
    )
    images = np.empty(
        (reduction.group_order, reduction.wedge_size), dtype=np.int64
    )
    table = _RowOrderTable(grid) if _fast_maps_requested() else None
    if table is not None and not table.in_row_order:
        # Nothing to read off: a whole-grid map would be gathered through the
        # coordinates of every point instead of the representatives.
        table = None
    for index, operation in enumerate(operations):
        mapping = None
        # The identity is its own shortcut: the representatives themselves.
        if table is not None and not np.array_equal(operation, np.eye(3)):
            whole = _grid_row_mapping_operation(
                grid, operation, 1.0e-10, table=table
            )
            if whole is not None:
                mapping = whole[reduction.representative_rows]
        if mapping is None:
            mapping = _grid_row_mapping_operation(
                grid, operation, 1.0e-10, reduction.representative_rows
            )
        if mapping is None:
            raise RuntimeError("accepted symmetry operation no longer maps the grid")
        images[index] = mapping
    return images


@dataclass(frozen=True)
class ReflectionRepresentationDecomposition:
    """Character table, row phases, and stabilizer-filtered sector maps."""

    reduction: AxisReflectionReduction
    characters: np.ndarray
    phases: np.ndarray
    orbit_to_sector: np.ndarray
    # Orbit index list of each representation, kept after its first request.
    _sector_orbit_lists: dict[int, np.ndarray] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @property
    def representation_count(self) -> int:
        return int(self.characters.shape[0])

    @property
    def full_size(self) -> int:
        return self.reduction.full_size

    @property
    def wedge_size(self) -> int:
        """Return the scalar-field orbit count shared with Hartree and SCF."""

        return self.reduction.wedge_size

    @property
    def sector_sizes(self) -> tuple[int, ...]:
        """Return the exact orbital dimension of every representation."""

        return tuple(
            int(np.count_nonzero(mapping >= 0))
            for mapping in self.orbit_to_sector
        )

    def sector_size(self, representation: int) -> int:
        """Return one stabilizer-filtered representation dimension."""

        index = int(representation)
        if not 0 <= index < self.representation_count:
            raise IndexError("representation index is outside the character table")
        return int(np.count_nonzero(self.orbit_to_sector[index] >= 0))

    def sector_orbit_indices(self, representation: int) -> np.ndarray:
        """Map sector columns to the underlying scalar-field orbit indices.

        The list is a function of the immutable sector map alone, and the
        row selection of the operator assembly, the vector transforms and
        the eigensolver ask for it repeatedly.  It is therefore built once
        per representation, on first request; every caller receives the
        same read-only array.
        """

        index = int(representation)
        if not 0 <= index < self.representation_count:
            raise IndexError("representation index is outside the character table")
        orbits = self._sector_orbit_lists.get(index)
        if orbits is None:
            mapping = self.orbit_to_sector[index]
            admitted = np.flatnonzero(mapping >= 0)
            order = np.argsort(mapping[admitted], kind="stable")
            orbits = np.ascontiguousarray(admitted[order], dtype=np.int64)
            orbits.setflags(write=False)
            self._sector_orbit_lists[index] = orbits
        return orbits

    @classmethod
    def build(
        cls,
        grid: RealSpaceGrid,
        reduction: AxisReflectionReduction,
    ) -> "ReflectionRepresentationDecomposition":
        """Build all characters and apply the exact orbit-stabilizer rule."""

        order = reduction.group_order
        if order <= 1:
            raise ValueError("orbital decomposition requires nontrivial symmetry")
        character_rows: list[np.ndarray] = []
        seen: set[bytes] = set()
        if isinstance(reduction, SignedPermutationReduction):
            # For C2^k, chi_p(g_b)=(-1)^(p dot b).  These are all |G|
            # mutually orthogonal real one-dimensional characters.
            rank = int(reduction.generator_bits.shape[1])
            for parity in product((0, 1), repeat=rank):
                exponents = np.asarray(parity, dtype=np.int8)
                character = np.where(
                    (reduction.generator_bits @ exponents) % 2,
                    -1,
                    1,
                ).astype(np.int8)
                character_rows.append(character)
        else:
            # Diagonal sign groups are subgroups of C2^3.  Restricting the
            # PARSEC-ordered product characters preserves D2h labels.
            signs = reduction.signs.astype(np.int8, copy=False)
            for parity in _PARSEC_REFLECTION_PARITIES:
                character = np.ones(order, dtype=np.int8)
                for axis, exponent in enumerate(parity):
                    if exponent:
                        character *= signs[:, axis]
                key = character.tobytes()
                if key not in seen:
                    seen.add(key)
                    character_rows.append(character)
        characters = np.ascontiguousarray(np.vstack(character_rows), dtype=np.int8)
        if characters.shape != (order, order):
            raise RuntimeError("failed to construct the reflection character table")
        gram = characters.astype(np.int64) @ characters.astype(np.int64).T
        if not np.array_equal(gram, order * np.eye(order, dtype=np.int64)):
            raise RuntimeError("reflection characters are not orthogonal")

        representatives = reduction.representative_rows
        # ``images[g,w]`` is the full row obtained by applying operation g to
        # orbit representative w.  Duplicate images expose a stabilizer.
        images = _representative_images(grid, reduction)

        phases = np.zeros(
            (characters.shape[0], reduction.full_size), dtype=np.int8
        )
        orbit_to_sector = np.full(
            (characters.shape[0], reduction.wedge_size), -1, dtype=np.int64
        )
        fast = _fast_maps_requested()
        if fast:
            # An image counts where no earlier operation gave the same row:
            # the number of different images, without sorting every column.
            unique_image_counts = np.ones(reduction.wedge_size, dtype=np.int64)
            for operation in range(1, order):
                repeated = images[operation] == images[0]
                for earlier in range(1, operation):
                    repeated |= images[operation] == images[earlier]
                unique_image_counts += ~repeated
        else:
            sorted_images = np.sort(images, axis=0)
            unique_image_counts = 1 + np.count_nonzero(
                np.diff(sorted_images, axis=0), axis=0
            )
        if not np.array_equal(
            unique_image_counts, reduction.multiplicities
        ):
            raise RuntimeError("reflection operations do not reproduce every orbit")
        stabilizer = images == representatives[None, :]
        # Where every image lies in the orbit of its representative, a row
        # is reached from one orbit only, and the last operation that
        # reaches it is the same for every representation: one label per row
        # replaces a phase scatter per representation and operation.  The
        # images of an orbit are then as many different rows as its
        # multiplicity, so multiplicities that add up to the grid leave no
        # row without a label.
        row_operation = None
        orbit_numbers = np.arange(reduction.wedge_size, dtype=np.int64)
        if (
            fast
            and int(np.sum(reduction.multiplicities)) == reduction.full_size
            and all(
                np.array_equal(
                    reduction.full_to_wedge[images[operation]], orbit_numbers
                )
                for operation in range(order)
            )
        ):
            row_operation = np.empty(reduction.full_size, dtype=np.intp)
            for operation in range(order):
                row_operation[images[operation]] = operation
        # Only the identity has every character +1.  Where no other operation
        # fixes a representative, every representation admits every orbit.
        free = (
            row_operation is not None
            and reduction.wedge_size > 0
            and not np.any(stabilizer[np.any(characters != 1, axis=0)])
        )
        for representation, character in enumerate(characters):
            if free:
                orbit_to_sector[representation] = orbit_numbers
                phases[representation] = character[row_operation]
                continue
            # chi is trivial on S_w iff every operation fixing the selected
            # representative has character +1.  This is equivalent to the
            # former duplicate-image test, evaluated for every orbit at once.
            admitted = np.all(
                (~stabilizer) | (character[:, None] == 1), axis=0
            )
            admitted_orbits = np.flatnonzero(admitted)
            if admitted_orbits.size < 1:
                raise RuntimeError("a reflection representation has zero dimension")
            orbit_to_sector[representation, admitted_orbits] = np.arange(
                admitted_orbits.size, dtype=np.int64
            )
            if row_operation is not None:
                phases[representation] = character[row_operation]
                if admitted_orbits.size != admitted.size:
                    # Rows of a rejected orbit keep phase zero.
                    phases[representation, ~admitted[reduction.full_to_wedge]] = 0
                continue
            for operation in range(order):
                phases[
                    representation,
                    images[operation, admitted_orbits],
                ] = character[operation]

        # Identity maps every admitted representative to itself.  Rejected
        # representative rows deliberately carry phase zero.
        for representation in range(characters.shape[0]):
            admitted = orbit_to_sector[representation] >= 0
            if not np.all(phases[representation, representatives[admitted]] == 1):
                raise RuntimeError("invalid representation phase convention")
        return cls(
            reduction=reduction,
            characters=characters,
            phases=np.ascontiguousarray(phases),
            orbit_to_sector=np.ascontiguousarray(orbit_to_sector),
        )

    def reduce_operator(
        self,
        operator: sp.spmatrix,
        representation: int,
    ) -> sp.csr_matrix:
        """Construct ``U_Gamma.T A U_Gamma`` from representative rows."""

        index = int(representation)
        if not 0 <= index < self.representation_count:
            raise IndexError("representation index is outside the character table")
        matrix = sp.csr_matrix(operator, dtype=np.float64)
        if matrix.shape != (self.full_size, self.full_size):
            raise ValueError("operator does not match the representation grid")
        matrix.sum_duplicates()
        matrix.sort_indices()
        selected = matrix[
            self.reduction.representative_rows[
                self.sector_orbit_indices(index)
            ],
            :,
        ].tocoo(copy=False)
        return self._assemble_reduced_operator(selected, index)

    def _selected_representations(
        self,
        representations: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if representations is None:
            return tuple(range(self.representation_count))
        selected = tuple(sorted({int(index) for index in representations}))
        if not selected or selected[0] < 0 or selected[-1] >= self.representation_count:
            raise IndexError("representation index is outside the character table")
        return selected

    def reduce_operators(
        self,
        operator: sp.spmatrix,
        representations: Sequence[int] | None = None,
    ) -> tuple[sp.csr_matrix | None, ...]:
        """Construct every ``U_Gamma.T A U_Gamma`` in one sparse pass.

        The full operator is canonicalized and its representative rows are
        gathered only once.  Each representation differs solely by the
        character phase multiplying the gathered column.  This is exactly
        the same projection used by :meth:`reduce_operator`; the batched
        route only avoids repeating the expensive CSR slicing and COO
        conversion once per representation.

        ``representations`` restricts assembly to those sectors; the result
        keeps one slot per representation and leaves the others ``None``.
        A process that owns only some sectors need not assemble the rest.
        """

        wanted = self._selected_representations(representations)
        matrix = sp.csr_matrix(operator, dtype=np.float64)
        if matrix.shape != (self.full_size, self.full_size):
            raise ValueError("operator does not match the representation grid")
        matrix.sum_duplicates()
        matrix.sort_indices()
        # Gather every scalar-orbit representative row exactly once.  The
        # earlier implementation performed this large CSR slice and COO
        # conversion independently for every representation even though the
        # source rows differ only by stabilizer admission.  ``selected.row``
        # is now the scalar-orbit index; the exact representation row map is
        # applied inside ``_assemble_reduced_operator``.
        if os.environ.get("PARSEC_NATIVE_SECTOR_ASSEMBLY", "0") == "1":
            from ..backends.native import _load_native

            native = _load_native()
            if not hasattr(native, "reduce_sector_csr"):
                raise RuntimeError("PARSEC_NATIVE_SECTOR_ASSEMBLY requires rebuilding the native extension")
            sectors: list[sp.csr_matrix | None] = [None] * self.representation_count
            for index in wanted:
                payload = native.reduce_sector_csr(
                    matrix.indptr, matrix.indices, matrix.data,
                    self.reduction.representative_rows[self.sector_orbit_indices(index)],
                    self.reduction.full_to_wedge, self.orbit_to_sector[index],
                    self.reduction.multiplicities, self.phases[index],
                )
                size = self.sector_size(index)
                reduced = sp.csr_matrix((payload["data"], payload["indices"], payload["indptr"]),
                                        shape=(size, size))
                asymmetry = reduced - reduced.T
                maximum = float(np.max(np.abs(asymmetry.data), initial=0.0))
                scale = float(np.max(np.abs(reduced.data), initial=1.0))
                if maximum > 5.0e-13 * scale:
                    raise ValueError(f"representation operator is not symmetric: maximum asymmetry {maximum:.3e}")
                sectors[index] = reduced
            return tuple(sectors)

        selected = matrix[
            self.reduction.representative_rows,
            :,
        ].tocoo(copy=False)
        workers = operator_build_workers(len(wanted))

        def assemble(index: int) -> sp.csr_matrix:
            return self._assemble_reduced_operator(
                selected,
                index,
                rows_are_scalar_orbits=True,
            )

        assembled: list[sp.csr_matrix | None] = [None] * self.representation_count
        if workers == 1:
            for index in wanted:
                assembled[index] = assemble(index)
            return tuple(assembled)
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="parsec-representation-build",
        ) as executor:
            for index, reduced in zip(wanted, executor.map(assemble, wanted)):
                assembled[index] = reduced
        return tuple(assembled)

    def _assemble_reduced_operator(
        self,
        selected: sp.coo_matrix,
        representation: int,
        *,
        rows_are_scalar_orbits: bool = False,
    ) -> sp.csr_matrix:
        """Assemble one reduced matrix from already-selected wedge rows."""

        index = int(representation)
        if rows_are_scalar_orbits:
            # The sector map answers this route by itself, so a process that
            # assembles a sector it does not solve builds no orbit list for it.
            row_orbits = selected.row
            sector_rows = self.orbit_to_sector[index, row_orbits]
        else:
            row_orbits = self.sector_orbit_indices(index)[selected.row]
            sector_rows = selected.row
        column_orbits = self.reduction.full_to_wedge[selected.col]
        sector_columns = self.orbit_to_sector[index, column_orbits]
        keep = (sector_rows >= 0) & (sector_columns >= 0)
        if not np.any(keep):
            raise RuntimeError("representation projection produced an empty operator")
        kept_row_orbits = row_orbits[keep]
        kept_column_orbits = column_orbits[keep]
        normalization = np.sqrt(
            self.reduction.multiplicities[kept_row_orbits]
            / self.reduction.multiplicities[kept_column_orbits]
        )
        data = (
            selected.data[keep]
            * self.phases[index, selected.col[keep]]
            * normalization
        )
        sector_size = self.sector_size(index)
        reduced = sp.coo_matrix(
            (
                data,
                (sector_rows[keep], sector_columns[keep]),
            ),
            shape=(sector_size, sector_size),
        ).tocsr()
        reduced.sum_duplicates()
        reduced.eliminate_zeros()
        reduced.sort_indices()
        asymmetry = reduced - reduced.T
        maximum = float(np.max(np.abs(asymmetry.data))) if asymmetry.nnz else 0.0
        scale = float(np.max(np.abs(reduced.data), initial=1.0))
        if maximum > 5.0e-13 * scale:
            raise ValueError(
                "representation operator is not symmetric: "
                f"maximum asymmetry {maximum:.3e}"
            )
        return reduced

    def reduce_nonlocal_operator(
        self,
        operator: NonlocalProjectorOperator,
        representation: int,
    ) -> NonlocalProjectorOperator:
        """Project KB factors as ``B_Gamma=U_Gamma.T B``."""

        index = int(representation)
        if not 0 <= index < self.representation_count:
            raise IndexError("representation index is outside the character table")
        projectors = sp.coo_matrix(operator.projectors, dtype=np.float64)
        return self._assemble_reduced_nonlocal(operator, projectors, index)

    def reduce_nonlocal_operators(
        self,
        operator: NonlocalProjectorOperator,
        representations: Sequence[int] | None = None,
    ) -> tuple[NonlocalProjectorOperator | None, ...]:
        """Construct every ``U_Gamma.T B`` from one canonical COO buffer.

        PARSEC's Kleinman--Bylander operator remains ``B D B.T`` in each
        representation.  Only the basis transformation of ``B`` is batched;
        projector coefficients, signs, labels, and summation order are kept.
        ``representations`` restricts assembly as in :meth:`reduce_operators`.
        """

        wanted = self._selected_representations(representations)
        projectors = sp.coo_matrix(operator.projectors, dtype=np.float64)
        projectors.sum_duplicates()
        workers = operator_build_workers(len(wanted))

        def assemble(index: int) -> NonlocalProjectorOperator:
            return self._assemble_reduced_nonlocal(operator, projectors, index)

        assembled: list[NonlocalProjectorOperator | None] = [None] * self.representation_count
        if workers == 1:
            for index in wanted:
                assembled[index] = assemble(index)
            return tuple(assembled)
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="parsec-projector-build",
        ) as executor:
            for index, reduced in zip(wanted, executor.map(assemble, wanted)):
                assembled[index] = reduced
        return tuple(assembled)

    def _assemble_reduced_nonlocal(
        self,
        operator: NonlocalProjectorOperator,
        projectors: sp.coo_matrix,
        representation: int,
    ) -> NonlocalProjectorOperator:
        """Assemble one reduced projector matrix from shared COO buffers."""

        index = int(representation)
        row_orbits = self.reduction.full_to_wedge[projectors.row]
        sector_rows = self.orbit_to_sector[index, row_orbits]
        keep = sector_rows >= 0
        scale = 1.0 / np.sqrt(self.reduction.multiplicities[row_orbits[keep]])
        reduced = sp.coo_matrix(
            (
                projectors.data[keep]
                * self.phases[index, projectors.row[keep]]
                * scale,
                (
                    sector_rows[keep],
                    projectors.col[keep],
                ),
            ),
            shape=(self.sector_size(index), projectors.shape[1]),
        ).tocsc()
        reduced.sum_duplicates()
        reduced.eliminate_zeros()
        reduced.sort_indices()
        return NonlocalProjectorOperator(
            projectors=reduced,
            signs=np.asarray(operator.signs, dtype=np.float64).copy(),
            labels=tuple(operator.labels),
        )

    def invariant_wedge_values(self, values: np.ndarray) -> np.ndarray:
        """Orbit-average a symmetry-invariant local scalar field."""

        reduced = self.reduction.reduce_vector(values)
        return reduced / np.sqrt(self.reduction.multiplicities)

    def reduce_vector(
        self,
        values: np.ndarray,
        representation: int,
    ) -> np.ndarray:
        """Apply ``U_Gamma.T`` to one full-grid host vector."""

        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.full_size,):
            raise ValueError("full vector does not match the representation grid")
        weights = self.phases[int(representation)] * array
        sums = np.bincount(
            self.reduction.full_to_wedge,
            weights=weights,
            minlength=self.wedge_size,
        )
        sector_orbits = self.sector_orbit_indices(representation)
        return sums[sector_orbits] / np.sqrt(
            self.reduction.multiplicities[sector_orbits]
        )

    def expand_vector(
        self,
        values: np.ndarray,
        representation: int,
    ) -> np.ndarray:
        """Apply ``U_Gamma`` to one wedge vector on the host."""

        array = np.asarray(values, dtype=np.float64)
        index = int(representation)
        sector_orbits = self.sector_orbit_indices(index)
        if array.shape != (sector_orbits.size,):
            raise ValueError("sector vector does not match the representation")
        base_wedge = np.zeros(self.wedge_size, dtype=np.float64)
        base_wedge[sector_orbits] = array / np.sqrt(
            self.reduction.multiplicities[sector_orbits]
        )
        return np.ascontiguousarray(
            self.phases[index] * base_wedge[self.reduction.full_to_wedge]
        )


__all__ = ["ReflectionRepresentationDecomposition", "operator_build_workers"]
