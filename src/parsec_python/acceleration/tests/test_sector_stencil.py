"""Sector stencils built from the grid against the route through the full CSR."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from threading import Lock
from time import sleep
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import build_cluster_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.V_ion import NonlocalProjectorOperator
from parsec_python.models import Atom, GridSettings
from parsec_python.acceleration.Laplacian import DeferredNativeNegativeLaplacian
from parsec_python.acceleration.Symmetry import (
    AxisReflectionReduction,
    ReflectionRepresentationDecomposition,
    SignedPermutationReduction,
)
from parsec_python.acceleration.Symmetry import sector_stencil
from parsec_python.acceleration.Symmetry.operator_cache import (
    _MEMORY_BUNDLES,
    _MEMORY_BUNDLES_LOCK,
    load_or_build_reduced_operators,
)
from parsec_python.acceleration.Symmetry.sector_stencil import (
    build_sector_stencils,
    native_kernel_available,
    sector_stencil_route,
    stencil_values,
)
from parsec_python.acceleration.backends.cupy_stencil_major import (
    StencilMajorHostMetadata,
    build_stencil_major_metadata,
)
from parsec_python.acceleration.backends.implicit_stencil import (
    pack_affine_tiles,
    unpack_affine_tiles,
)
from parsec_python.acceleration.backends.native import (
    _load_native,
    build_native_negative_laplacian,
    native_available,
)


# A carbon with four hydrogens on alternate cube corners: the three twofold
# axes map it onto itself, no mirror plane of the axes does.
_D2_CLUSTER = (
    Atom("C", (0.0, 0.0, 0.0)),
    Atom("H", (1.1, 1.1, 1.1)),
    Atom("H", (1.1, -1.1, -1.1)),
    Atom("H", (-1.1, 1.1, -1.1)),
    Atom("H", (-1.1, -1.1, 1.1)),
)
_ATOM_AT_ORIGIN = (Atom("H", (0.0, 0.0, 0.0)),)
_HALF = (0.5, 0.5, 0.5)
_ZERO = (0.0, 0.0, 0.0)

# name: spacing, radius, expansion order, shift, atoms, reduction type.  The
# zero-shift grids have points, and with them atoms, on every symmetry plane
# and axis, so their sectors differ in size.
_CASES = {
    "d2 cluster, free orbits": (0.5, 4.6, 8, _HALF, _D2_CLUSTER, AxisReflectionReduction),
    "d2 cluster, points on the axes": (0.55, 3.6, 8, _ZERO, _D2_CLUSTER, AxisReflectionReduction),
    "d2h, free orbits": (0.65, 2.8, 8, _HALF, _ATOM_AT_ORIGIN, AxisReflectionReduction),
    "d2h, points on the planes": (0.65, 2.8, 8, _ZERO, _ATOM_AT_ORIGIN, AxisReflectionReduction),
    "signed permutations, points on the planes": (
        0.65, 2.8, 8, _ZERO, _ATOM_AT_ORIGIN, SignedPermutationReduction,
    ),
    "order 4": (0.8, 2.8, 4, _ZERO, _D2_CLUSTER, AxisReflectionReduction),
    "order 12": (0.5, 3.3, 12, _ZERO, _D2_CLUSTER, AxisReflectionReduction),
}


def _case(name):
    spacing, radius, order, shift, atoms, kind = _CASES[name]
    grid = build_cluster_grid(
        GridSettings(spacing=spacing, radius=radius, expansion_order=order, shift=shift)
    )
    reduction = kind.detect(grid, atoms)
    return grid, ReflectionRepresentationDecomposition.build(grid, reduction)


def _reduced_like_the_native_route(matrix, decomposition, representation):
    """``reduce_sector_csr`` written out row by row.

    The entries of a representative full-grid row are taken in ascending
    full-grid column, stably sorted by sector column, and equal sector columns
    are added from left to right; an exact zero is dropped.
    """

    reduction = decomposition.reduction
    orbits = decomposition.sector_orbit_indices(representation)
    to_sector = decomposition.orbit_to_sector[representation]
    phases = decomposition.phases[representation]
    multiplicity = reduction.multiplicities
    indptr, indices, data = [0], [], []
    for orbit in orbits:
        full_row = reduction.representative_rows[orbit]
        entries = []
        for position in range(matrix.indptr[full_row], matrix.indptr[full_row + 1]):
            column = matrix.indices[position]
            column_orbit = reduction.full_to_wedge[column]
            if to_sector[column_orbit] >= 0:
                entries.append((
                    int(to_sector[column_orbit]),
                    matrix.data[position]
                    * np.float64(phases[column])
                    * np.sqrt(np.float64(multiplicity[orbit]) / np.float64(multiplicity[column_orbit])),
                ))
        entries.sort(key=lambda entry: entry[0])
        position = 0
        while position < len(entries):
            column, value = entries[position]
            position += 1
            while position < len(entries) and entries[position][0] == column:
                value = value + entries[position][1]
                position += 1
            if value != 0.0:
                indices.append(column)
                data.append(value)
        indptr.append(len(indices))
    size = orbits.size
    return sp.csr_matrix(
        (np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int64), indptr),
        shape=(size, size),
    )


def _implementations():
    return ("numpy", "native") if native_kernel_available() else ("numpy",)


def _kernel_payload(grid, decomposition, representation, **changed):
    """What ``build_sector_stencil`` returns for one sector."""

    reduction = decomposition.reduction
    arguments = dict(
        integer_coordinates=grid.integer_coordinates,
        index_min=grid.index_min,
        lookup=grid.lookup,
        expansion_order=int(grid.settings.expansion_order),
        spacing=float(grid.spacing),
        representatives=reduction.representative_rows[
            decomposition.sector_orbit_indices(representation)
        ],
        full_to_orbit=reduction.full_to_wedge,
        orbit_to_sector=decomposition.orbit_to_sector[representation],
        multiplicities=reduction.multiplicities,
        phases=decomposition.phases[representation],
    )
    arguments.update(changed)
    return _load_native().build_sector_stencil(**arguments)


def _nonlocal(grid):
    values = np.linspace(-0.3, 0.7, grid.size)
    return NonlocalProjectorOperator(
        projectors=sp.csc_matrix(values[:, None]),
        signs=np.asarray((1.0,), dtype=np.float64),
        labels=((0, 0, 0),),
    )


class SectorStencilTests(unittest.TestCase):
    def assert_same_stencil(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.neighbors.dtype, np.int32)
        self.assertEqual(actual.coefficient_codes.dtype, np.uint8)
        np.testing.assert_array_equal(actual.neighbors, expected.neighbors)
        np.testing.assert_array_equal(
            actual.coefficient_codes, expected.coefficient_codes
        )
        # Bit patterns: a palette may hold a value and its negative.
        np.testing.assert_array_equal(
            actual.coefficient_palette.view(np.uint64),
            expected.coefficient_palette.view(np.uint64),
        )

    def test_every_sector_equals_the_reduced_and_packed_full_matrix(self) -> None:
        for name in _CASES:
            grid, decomposition = _case(name)
            # Bit for bit the native matrix up to order 18, and it needs no
            # extension.
            matrix = build_negative_laplacian(grid)
            expected = [
                build_stencil_major_metadata(
                    _reduced_like_the_native_route(matrix, decomposition, index)
                )
                for index in range(decomposition.representation_count)
            ]
            for implementation in _implementations():
                with self.subTest(case=name, implementation=implementation):
                    built = build_sector_stencils(
                        grid, decomposition, implementation=implementation
                    )
                    self.assertEqual(len(built), decomposition.representation_count)
                    for actual, wanted in zip(built, expected, strict=True):
                        self.assert_same_stencil(actual, wanted)
            # Every case must have what the summation order is about: stencil
            # points of one row that share a sector column.
            with self.subTest(case=name):
                rows = decomposition.reduction.representative_rows[
                    decomposition.sector_orbit_indices(0)
                ]
                self.assertLess(
                    int(np.count_nonzero(expected[0].neighbors >= 0)),
                    int(matrix[rows].nnz),
                )
        # Points on the axes leave sectors of unequal size.
        sizes = _case("d2 cluster, points on the axes")[1].sector_sizes
        self.assertGreater(max(sizes), min(sizes))

    def test_production_reduction_routes_give_the_same_arrays(self) -> None:
        native_reduction = native_available() and hasattr(
            _load_native(), "reduce_sector_csr"
        )
        for name in _CASES:
            grid, decomposition = _case(name)
            matrix = build_negative_laplacian(grid)
            built = build_sector_stencils(grid, decomposition, implementation="numpy")
            with patch.dict(os.environ, PARSEC_NATIVE_SECTOR_ASSEMBLY="0"):
                python_route = decomposition.reduce_operators(matrix)
            for index, reduced in enumerate(python_route):
                with self.subTest(case=name, route="python", sector=index):
                    packed = build_stencil_major_metadata(reduced)
                    np.testing.assert_array_equal(
                        built[index].neighbors, packed.neighbors
                    )
                    # SciPy adds repeated columns in the order its unstable
                    # row sort leaves them, so three or more can differ from
                    # the native order in the last bit; rows of at most 16
                    # entries are sorted stably everywhere.
                    actual = built[index].to_csr()
                    if grid.settings.expansion_order <= 4:
                        np.testing.assert_array_equal(actual.data, reduced.data)
                    else:
                        np.testing.assert_allclose(
                            actual.data, reduced.data, rtol=0.0, atol=2.0e-14
                        )
            if not native_reduction:
                continue
            with patch.dict(os.environ, PARSEC_NATIVE_SECTOR_ASSEMBLY="1"):
                native_route = decomposition.reduce_operators(matrix)
            for index, reduced in enumerate(native_route):
                with self.subTest(case=name, route="native", sector=index):
                    self.assert_same_stencil(
                        built[index], build_stencil_major_metadata(reduced)
                    )

    def test_tiles_packed_from_a_built_stencil_are_those_of_the_former_route(
        self,
    ) -> None:
        # Rows of at most 13 entries, which SciPy reduces in the native order
        # (see DeferredOperatorRouteTests), in sectors of more than one chunk
        # of the packer, with points on the symmetry axes.
        settings = GridSettings(
            spacing=0.5, radius=13.5, expansion_order=4, shift=_ZERO
        )
        grid = build_cluster_grid(settings)
        matrix = build_negative_laplacian(grid)
        tile = 16
        packed_by_maps = {}
        for fast_maps in ("1", "0"):
            with patch.dict(os.environ, PARSEC_SYMMETRY_FAST_MAPS=fast_maps):
                reduction = AxisReflectionReduction.detect(grid, _D2_CLUSTER)
                decomposition = ReflectionRepresentationDecomposition.build(
                    grid, reduction
                )
            self.assertGreater(min(decomposition.sector_sizes), 1024 * tile)
            with patch.dict(os.environ, PARSEC_NATIVE_SECTOR_ASSEMBLY="0"):
                former = [
                    build_stencil_major_metadata(reduced)
                    for reduced in decomposition.reduce_operators(matrix)
                ]
            for implementation in _implementations():
                built = build_sector_stencils(
                    grid, decomposition, implementation=implementation
                )
                for index, (stencil, wanted) in enumerate(
                    zip(built, former, strict=True)
                ):
                    with self.subTest(
                        fast_maps=fast_maps,
                        implementation=implementation,
                        sector=index,
                    ):
                        self.assert_same_stencil(stencil, wanted)
                        with patch.dict(
                            os.environ, PARSEC_CUPY_IMPLICIT_PACK_WORKERS="1"
                        ):
                            expected = pack_affine_tiles(wanted, tile)
                        for workers in ("1", "4"):
                            with patch.dict(
                                os.environ,
                                PARSEC_CUPY_IMPLICIT_PACK_WORKERS=workers,
                            ):
                                packed = pack_affine_tiles(stencil, tile)
                            for left, right in zip(packed[:2], expected[:2]):
                                self.assertEqual(left.dtype, right.dtype)
                                np.testing.assert_array_equal(left, right)
                            self.assertEqual(packed[2], expected[2])
                        slots, rows = stencil.neighbors.shape
                        # Both kinds of tile are there.
                        self.assertGreater(packed[2]["regular_rows"], 0)
                        self.assertLess(packed[2]["regular_rows"], rows)
                        neighbors, codes = unpack_affine_tiles(
                            packed[0], packed[1], rows, slots, tile
                        )
                        np.testing.assert_array_equal(neighbors, stencil.neighbors)
                        np.testing.assert_array_equal(
                            codes, stencil.coefficient_codes
                        )
                        packed_by_maps.setdefault(index, []).append(packed[:2])
        # The same tiles whichever route built the maps.
        for index, (first, *others) in packed_by_maps.items():
            for other in others:
                for left, right in zip(first, other, strict=True):
                    np.testing.assert_array_equal(left, right)

    def test_requested_sectors_only(self) -> None:
        grid, decomposition = _case("d2h, points on the planes")
        complete = build_sector_stencils(grid, decomposition, implementation="numpy")
        for implementation in _implementations():
            with self.subTest(implementation=implementation):
                partial = build_sector_stencils(
                    grid, decomposition, (5, 0, 5), implementation=implementation
                )
                self.assertEqual(
                    [index for index, item in enumerate(partial) if item is not None],
                    [0, 5],
                )
                for index in (0, 5):
                    self.assert_same_stencil(partial[index], complete[index])
        with self.assertRaises(IndexError):
            build_sector_stencils(grid, decomposition, (decomposition.representation_count,))
        with self.assertRaises(ValueError):
            build_sector_stencils(grid, decomposition, implementation="scipy")

    def test_blocks_do_not_change_the_result(self) -> None:
        grid, decomposition = _case("d2 cluster, points on the axes")
        expected = build_sector_stencils(grid, decomposition, implementation="numpy")
        with patch.object(sector_stencil, "_BLOCK_ROWS", 37):
            blocked = build_sector_stencils(grid, decomposition, implementation="numpy")
        with patch.dict(os.environ, PARSEC_SYMMETRY_OPERATOR_WORKERS="3"):
            threaded = build_sector_stencils(grid, decomposition, implementation="numpy")
        for index, wanted in enumerate(expected):
            self.assert_same_stencil(blocked[index], wanted)
            self.assert_same_stencil(threaded[index], wanted)

    @staticmethod
    def _with_phase(grid, decomposition, change):
        """Change the phase of sector 1 on one stencil point of a sector row.

        The point belongs to another orbit than the row and is not the
        representative of its own, so only that one row reads the phase.
        """

        reduction = decomposition.reduction
        matrix = build_negative_laplacian(grid)
        full_row = int(reduction.representative_rows[reduction.wedge_size // 2])
        columns = matrix.indices[matrix.indptr[full_row]:matrix.indptr[full_row + 1]]
        point = int(
            next(
                column
                for column in columns
                if reduction.full_to_wedge[column] != reduction.full_to_wedge[full_row]
                and reduction.representative_rows[reduction.full_to_wedge[column]] != column
            )
        )
        phases = decomposition.phases.copy()
        phases[1, point] = change(phases[1, point])
        return ReflectionRepresentationDecomposition(
            reduction=reduction,
            characters=decomposition.characters,
            phases=phases,
            orbit_to_sector=decomposition.orbit_to_sector,
        )

    def test_an_operator_that_is_not_symmetric_is_refused(self) -> None:
        grid, decomposition = _case("d2 cluster, free orbits")
        # The wrong character sign.
        broken = self._with_phase(grid, decomposition, lambda phase: -phase)
        for implementation in _implementations():
            with self.subTest(implementation=implementation):
                with self.assertRaisesRegex(ValueError, "not symmetric"):
                    build_sector_stencils(grid, broken, (1,), implementation=implementation)
                build_sector_stencils(grid, broken, (0,), implementation=implementation)

    def test_zero_phase_on_an_admitted_point(self) -> None:
        grid, decomposition = _case("d2 cluster, free orbits")
        broken = self._with_phase(grid, decomposition, lambda phase: 0)
        # The CSR route drops such an entry; the NumPy builder hands over.
        with self.assertRaises(sector_stencil.SectorStencilUnavailable):
            build_sector_stencils(grid, broken, (1,), implementation="numpy")
        if native_kernel_available():
            # The kernel drops it too, and the audit then finds the transpose.
            with self.assertRaisesRegex(ValueError, "not symmetric"):
                build_sector_stencils(grid, broken, (1,), implementation="native")

    def test_switch_values(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("PARSEC_SECTOR_STENCIL", None)
            self.assertEqual(sector_stencil_route(), "direct")
            for value in ("direct", "numpy", "csr", " CSR "):
                os.environ["PARSEC_SECTOR_STENCIL"] = value
                self.assertEqual(sector_stencil_route(), value.strip().lower())
            os.environ["PARSEC_SECTOR_STENCIL"] = "1"
            with self.assertRaises(ValueError):
                sector_stencil_route()

    @unittest.skipUnless(native_available(), "native extension required")
    def test_stencil_values_are_the_entries_of_the_native_matrix(self) -> None:
        for order in range(2, 22, 2):
            for spacing in (0.37794537555487107, 0.65, 0.3):
                with self.subTest(order=order, spacing=spacing):
                    width = order // 2
                    grid = build_cluster_grid(
                        GridSettings(
                            spacing=spacing,
                            radius=spacing * (width + 2.2),
                            expansion_order=order,
                        )
                    )
                    matrix = build_native_negative_laplacian(grid)
                    values = np.asarray(stencil_values(order, spacing))
                    self.assertEqual(values.size, width + 1)
                    np.testing.assert_array_equal(
                        np.unique(matrix.diagonal().view(np.uint64)),
                        values[:1].view(np.uint64),
                    )
                    np.testing.assert_array_equal(
                        np.unique(matrix.data.view(np.uint64)),
                        np.unique(values.view(np.uint64)),
                    )

    @unittest.skipUnless(native_kernel_available(), "rebuilt native extension required")
    def test_native_kernel_rejects_inconsistent_maps(self) -> None:
        grid, decomposition = _case("d2h, free orbits")
        rows = decomposition.reduction.representative_rows[
            decomposition.sector_orbit_indices(0)
        ]

        def build(**changed):
            return _kernel_payload(grid, decomposition, 0, **changed)

        payload = build()
        self.assertEqual(payload["neighbors"].shape[1], rows.size)
        self.assertEqual(payload["codes"].shape, payload["neighbors"].shape)
        self.assertLessEqual(payload["maximum_asymmetry"], 5.0e-13 * payload["scale"])
        with self.assertRaises(ValueError):
            build(representatives=rows[::-1].copy())
        with self.assertRaises(ValueError):
            build(phases=np.full(grid.size, 2, dtype=np.int8))
        with self.assertRaises(ValueError):
            build(lookup=np.roll(grid.lookup, 1, axis=2))
        with self.assertRaises(ValueError):
            build(expansion_order=7)
        with self.assertRaises(ValueError):
            build(threads=-1)


class TransposeAuditTests(unittest.TestCase):
    """The audit of the NumPy builder against ``A - A.T`` itself."""

    @staticmethod
    def _former(matrix):
        difference = matrix - matrix.T
        return (
            float(np.max(np.abs(difference.data), initial=0.0)),
            float(np.max(np.abs(matrix.data), initial=1.0)),
        )

    @staticmethod
    def _symmetric(values, size=80, seed=11):
        generator = np.random.default_rng(seed)
        upper = sp.triu(
            sp.random(
                size,
                size,
                density=0.1,
                random_state=generator,
                data_rvs=lambda count: generator.choice(values, count),
            ),
            1,
        )
        return (
            upper + upper.T + sp.diags(generator.choice(values, size))
        ).tolil()

    def _audited(self, matrix, reduced_csr):
        """Audit ``matrix``; ``reduced_csr`` says whether ``A - A.T`` is formed."""

        metadata = build_stencil_major_metadata(matrix.tocsr())
        with patch.object(
            StencilMajorHostMetadata,
            "to_csr",
            autospec=True,
            side_effect=StencilMajorHostMetadata.to_csr,
        ) as to_csr:
            result = sector_stencil._asymmetry(metadata)
        self.assertEqual(to_csr.called, reduced_csr)
        self.assertEqual(result, self._former(matrix.tocsr()))
        return result

    def test_symmetric_and_value_differences_come_from_the_packed_arrays(self) -> None:
        values = np.array([-2.5, -1.0, 0.25, 0.5, 3.0])
        matrix = self._symmetric(values)
        self.assertEqual(self._audited(matrix, False), (0.0, 3.0))
        # No value above one: the scale of the tolerance stays one.
        self.assertEqual(self._audited(self._symmetric(values / 8.0), False), (0.0, 1.0))
        row, column = next(
            (row, column)
            for row, column in zip(*sp.triu(matrix, 1).nonzero())
            if matrix[row, column] == 3.0
        )
        matrix[row, column] = -2.5
        self.assertEqual(self._audited(matrix, False), (5.5, 3.0))
        # One ulp between a value and its transpose is a difference too.
        matrix[row, column] = np.nextafter(3.0, 4.0)
        maximum, scale = self._audited(matrix, False)
        self.assertEqual(maximum, np.nextafter(3.0, 4.0) - 3.0)
        sector_stencil._check_symmetric(maximum, scale)

    def test_an_entry_without_a_transpose_counts_in_full(self) -> None:
        matrix = self._symmetric(np.array([-2.5, -1.0, 0.25, 0.5, 3.0]))
        row, column = map(int, next(zip(*sp.triu(matrix, 1).nonzero())))
        expected = abs(matrix[row, column])
        entries = matrix.nnz
        matrix[column, row] = 0.0
        self.assertEqual(matrix.nnz, entries - 1)
        maximum, scale = self._audited(matrix, True)
        self.assertEqual(maximum, expected)
        with self.assertRaisesRegex(ValueError, "not symmetric"):
            sector_stencil._check_symmetric(maximum, scale)
        # A sum that cancels on one side only is within the tolerance.
        matrix[row, column] = 1.0e-14
        maximum, scale = self._audited(matrix, True)
        self.assertEqual(maximum, 1.0e-14)
        sector_stencil._check_symmetric(maximum, scale)

    def test_builder_audits_symmetric_sectors_without_a_float64_matrix(self) -> None:
        for name in (
            "d2 cluster, points on the axes",
            "signed permutations, points on the planes",
        ):
            grid, decomposition = _case(name)
            with (
                self.subTest(case=name),
                patch.object(
                    StencilMajorHostMetadata,
                    "to_csr",
                    side_effect=AssertionError("the reduced CSR was formed"),
                ),
            ):
                build_sector_stencils(grid, decomposition, implementation="numpy")

    def test_one_sector_is_audited_at_a_time(self) -> None:
        grid, decomposition = _case("d2 cluster, points on the axes")
        guard = Lock()
        audits = {"running": 0, "most": 0, "count": 0}
        audit = sector_stencil._asymmetry

        def counted(metadata):
            with guard:
                audits["running"] += 1
                audits["count"] += 1
                audits["most"] = max(audits["most"], audits["running"])
            try:
                # Long enough for the other sectors to finish building.
                sleep(0.05)
                return audit(metadata)
            finally:
                with guard:
                    audits["running"] -= 1

        with (
            patch.object(sector_stencil, "_asymmetry", counted),
            patch.dict(os.environ, PARSEC_SYMMETRY_OPERATOR_WORKERS="4"),
        ):
            build_sector_stencils(grid, decomposition, implementation="numpy")
        self.assertEqual(audits["count"], decomposition.representation_count)
        self.assertEqual(audits["most"], 1)


@unittest.skipUnless(native_kernel_available(), "rebuilt native extension required")
class KernelBlockTests(unittest.TestCase):
    """Sectors of several kernel blocks, as every sector of a production run.

    The kernel takes the rows of a sector in blocks.  It numbers the
    coefficients within a block before it merges the palettes, takes the slot
    count from the widest block, and starts its OpenMP teams only for more
    than one block.  The sectors of the cases above are smaller than a block.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # A slab of eleven grid layers in x.  The rows of a sector begin with
        # its outermost layer, which is more than a block and whose rows have
        # no stencil point above them.
        cls.grid = build_cluster_grid(
            GridSettings(
                spacing=0.3,
                radius=1.0,
                expansion_order=8,
                shift=_ZERO,
                domain_shape="box",
                box_lengths=(3.2, 30.2, 30.2),
            )
        )
        reduction = AxisReflectionReduction.detect(cls.grid, _D2_CLUSTER)
        cls.decomposition = ReflectionRepresentationDecomposition.build(
            cls.grid, reduction
        )
        cls.expected = build_sector_stencils(
            cls.grid, cls.decomposition, implementation="numpy"
        )
        cls.block_rows = int(
            _kernel_payload(cls.grid, cls.decomposition, 0, threads=1)["block_rows"]
        )

    assert_same_stencil = SectorStencilTests.assert_same_stencil

    def _built(self, decomposition, representation, threads):
        payload = _kernel_payload(
            self.grid, decomposition, representation, threads=threads
        )
        size = decomposition.sector_size(representation)
        return payload, StencilMajorHostMetadata(
            shape=(size, size),
            neighbors=payload["neighbors"],
            coefficient_codes=payload["codes"],
            coefficient_palette=payload["palette"],
        )

    def test_the_case_spans_blocks_that_differ(self) -> None:
        block = self.block_rows
        self.assertEqual(self.decomposition.representation_count, 4)
        self.assertGreater(
            max(self.decomposition.sector_sizes), min(self.decomposition.sector_sizes)
        )
        for index, stencil in enumerate(self.expected):
            with self.subTest(sector=index):
                used = stencil.neighbors >= 0
                self.assertGreaterEqual(stencil.shape[0], 4 * block)
                # No row of the first block is as wide as the widest row, and
                # later blocks meet coefficients the first one does not.
                self.assertLess(
                    int(np.count_nonzero(used[:, :block], axis=0).max()),
                    stencil.neighbors.shape[0],
                )
                self.assertLess(
                    np.unique(stencil.coefficient_codes[:, :block][used[:, :block]]).size,
                    stencil.coefficient_palette.size,
                )

    def test_arrays_do_not_depend_on_blocks_or_threads(self) -> None:
        # 0 is the default team of the process.
        for threads in (1, 2, 3, 7, 0):
            for index, wanted in enumerate(self.expected):
                with self.subTest(threads=threads, sector=index):
                    payload, actual = self._built(self.decomposition, index, threads)
                    self.assert_same_stencil(actual, wanted)
                    self.assertLessEqual(
                        payload["maximum_asymmetry"], 5.0e-13 * payload["scale"]
                    )
        for actual, wanted in zip(
            build_sector_stencils(
                self.grid, self.decomposition, implementation="native"
            ),
            self.expected,
            strict=True,
        ):
            self.assert_same_stencil(actual, wanted)

    @unittest.skipUnless(
        native_available() and hasattr(_load_native(), "reduce_sector_csr"),
        "the former route needs the native sector reduction",
    )
    def test_arrays_equal_the_former_native_route(self) -> None:
        matrix = build_native_negative_laplacian(self.grid)
        with patch.dict(os.environ, PARSEC_NATIVE_SECTOR_ASSEMBLY="1"):
            reduced = self.decomposition.reduce_operators(matrix)
        for index, (former, wanted) in enumerate(
            zip(reduced, self.expected, strict=True)
        ):
            with self.subTest(sector=index):
                packed = build_stencil_major_metadata(former)
                self.assert_same_stencil(wanted, packed)
                self.assert_same_stencil(
                    self._built(self.decomposition, index, 3)[1], packed
                )

    def _phase_read_from(self, first_row):
        """A grid point whose phase in sector 1 only later rows read.

        The point is no representative.  Every sector row that has it in its
        stencil, and the row of its own orbit, is ``first_row`` or later, so
        a wrong phase there leaves the leading rows and their transposes as
        they were.
        """

        decomposition = self.decomposition
        reduction = decomposition.reduction
        to_sector = decomposition.orbit_to_sector[1]
        matrix = build_negative_laplacian(self.grid)

        def stencil(point):
            return matrix.indices[matrix.indptr[point]:matrix.indptr[point + 1]]

        def readers(point):
            points = stencil(point)
            orbits = reduction.full_to_wedge[points]
            rows = to_sector[orbits[reduction.representative_rows[orbits] == points]]
            return rows[rows >= 0]

        last = int(
            reduction.representative_rows[decomposition.sector_orbit_indices(1)[-1]]
        )
        for point in map(int, stencil(last)):
            orbit = reduction.full_to_wedge[point]
            if (
                reduction.representative_rows[orbit] != point
                and to_sector[orbit] >= first_row
                and readers(point).min() >= first_row
            ):
                return point, np.unique(readers(point))
        self.fail("no stencil point of the last row is read by later rows only")

    def test_asymmetry_outside_the_first_block_is_refused(self) -> None:
        decomposition = self.decomposition
        # The first row of the last block.
        first_row = (
            (decomposition.sector_size(1) - 1) // self.block_rows * self.block_rows
        )
        point, rows = self._phase_read_from(first_row)
        self.assertGreaterEqual(first_row, 3 * self.block_rows)
        self.assertGreaterEqual(int(rows.min()), first_row)
        phases = decomposition.phases.copy()
        phases[1, point] = -phases[1, point]
        broken = ReflectionRepresentationDecomposition(
            reduction=decomposition.reduction,
            characters=decomposition.characters,
            phases=phases,
            orbit_to_sector=decomposition.orbit_to_sector,
        )
        for threads in (1, 3, 0):
            with self.subTest(threads=threads):
                payload, actual = self._built(broken, 1, threads)
                self.assertGreater(
                    payload["maximum_asymmetry"], 5.0e-13 * payload["scale"]
                )
                # Only rows that read the phase differ; the codes of the
                # others are renumbered at most.
                intact = self.expected[1]
                changed = np.flatnonzero(
                    np.any(actual.neighbors != intact.neighbors, axis=0)
                    | np.any(
                        actual.coefficient_palette[actual.coefficient_codes]
                        != intact.coefficient_palette[intact.coefficient_codes],
                        axis=0,
                    )
                )
                self.assertGreater(changed.size, 0)
                self.assertTrue(set(changed.tolist()) <= set(rows.tolist()))
        for implementation in _implementations():
            with self.subTest(implementation=implementation):
                with self.assertRaisesRegex(ValueError, "not symmetric"):
                    build_sector_stencils(
                        self.grid, broken, (1,), implementation=implementation
                    )
                self.assert_same_stencil(
                    build_sector_stencils(
                        self.grid, broken, (0,), implementation=implementation
                    )[0],
                    self.expected[0],
                )


class DeferredOperatorRouteTests(unittest.TestCase):
    """``load_or_build_reduced_operators`` on a deferred native Laplacian."""

    @classmethod
    def setUpClass(cls) -> None:
        # Rows of at most 13 entries: SciPy then adds repeated columns in the
        # native order on every platform, and the routes agree bit for bit.
        cls.grid, cls.decomposition = _case("order 4")
        cls.matrix = build_negative_laplacian(cls.grid)
        cls.nonlocal_operator = _nonlocal(cls.grid)

    def setUp(self) -> None:
        with _MEMORY_BUNDLES_LOCK:
            _MEMORY_BUNDLES.clear()
        self.addCleanup(self._forget)
        # A test that names no route is about the default one, whatever the
        # shell that runs it has exported.
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("PARSEC_SECTOR_STENCIL", None)

    @staticmethod
    def _forget() -> None:
        with _MEMORY_BUNDLES_LOCK:
            _MEMORY_BUNDLES.clear()

    def _materialized(self):
        """Stand in for the C++ builder, whose matrix this one equals."""

        return patch.object(
            DeferredNativeNegativeLaplacian,
            "materialize",
            autospec=True,
            side_effect=lambda descriptor: self.matrix,
        )

    def _from_matrix(self, representations=None):
        with patch.dict(os.environ, PARSEC_NATIVE_SECTOR_ASSEMBLY="0"):
            return load_or_build_reduced_operators(
                self.decomposition,
                self.matrix,
                self.nonlocal_operator,
                cache_directory=None,
                representations=representations,
            )

    def assert_same_bundle(self, actual, expected):
        for left, right in zip(
            actual.stencil_metadata, expected.stencil_metadata, strict=True
        ):
            self.assertEqual(left is None, right is None)
            if left is None:
                continue
            np.testing.assert_array_equal(left.neighbors, right.neighbors)
            np.testing.assert_array_equal(
                left.coefficient_codes, right.coefficient_codes
            )
            np.testing.assert_array_equal(
                left.coefficient_palette.view(np.uint64),
                right.coefficient_palette.view(np.uint64),
            )
        for left, right in zip(
            actual.nonlocal_operators, expected.nonlocal_operators, strict=True
        ):
            self.assertEqual(left is None, right is None)
            if left is not None:
                np.testing.assert_array_equal(
                    left.projectors.toarray().view(np.uint64),
                    right.projectors.toarray().view(np.uint64),
                )

    def test_default_builds_from_the_grid_without_the_matrix(self) -> None:
        expected = self._from_matrix()
        self.assertEqual(expected.cache_info.stencil_builder, "csr")
        for route, builder in (
            ("direct", "direct-native" if native_kernel_available() else "direct-numpy"),
            ("numpy", "direct-numpy"),
        ):
            with self.subTest(route=route):
                descriptor = DeferredNativeNegativeLaplacian(self.grid)
                with (
                    patch.dict(os.environ, PARSEC_SECTOR_STENCIL=route),
                    patch.object(
                        DeferredNativeNegativeLaplacian,
                        "materialize",
                        side_effect=AssertionError("the full-grid matrix was built"),
                    ),
                ):
                    bundle = load_or_build_reduced_operators(
                        self.decomposition,
                        descriptor,
                        self.nonlocal_operator,
                        cache_directory=None,
                    )
                    partial = load_or_build_reduced_operators(
                        self.decomposition,
                        descriptor,
                        self.nonlocal_operator,
                        cache_directory=None,
                        representations=(2,),
                    )
                self.assertEqual(bundle.cache_info.stencil_builder, builder)
                self.assertEqual(bundle.cache_info.status, "disabled")
                self.assertIsNone(bundle.cache_info.key)
                # Nothing looked a key up, so none was hashed.
                self.assertIsNone(descriptor.hashed_cache_key)
                self.assert_same_bundle(bundle, expected)
                self.assertEqual(partial.cache_info.status, "disabled-partial")
                self.assertEqual(partial.cache_info.stencil_builder, builder)
                self.assert_same_bundle(partial, self._from_matrix((2,)))

    def test_csr_switch_restores_the_route_through_the_matrix(self) -> None:
        descriptor = DeferredNativeNegativeLaplacian(self.grid)
        with (
            patch.dict(
                os.environ,
                PARSEC_SECTOR_STENCIL="csr",
                PARSEC_NATIVE_SECTOR_ASSEMBLY="0",
            ),
            self._materialized() as materialize,
            patch.object(
                sector_stencil,
                "build_sector_stencils",
                side_effect=AssertionError("the direct builder ran"),
            ),
        ):
            bundle = load_or_build_reduced_operators(
                self.decomposition,
                descriptor,
                self.nonlocal_operator,
                cache_directory=None,
            )
        materialize.assert_called()
        self.assertEqual(bundle.cache_info.stencil_builder, "csr")
        self.assert_same_bundle(bundle, self._from_matrix())

    def test_input_the_direct_builder_does_not_cover_takes_the_matrix(self) -> None:
        descriptor = DeferredNativeNegativeLaplacian(self.grid)
        with (
            patch.dict(os.environ, PARSEC_NATIVE_SECTOR_ASSEMBLY="0"),
            self._materialized() as materialize,
            patch.object(
                sector_stencil,
                "build_sector_stencils",
                side_effect=sector_stencil.SectorStencilUnavailable("not covered"),
            ),
        ):
            bundle = load_or_build_reduced_operators(
                self.decomposition,
                descriptor,
                self.nonlocal_operator,
                cache_directory=None,
            )
        materialize.assert_called()
        self.assertEqual(bundle.cache_info.stencil_builder, "csr")
        self.assert_same_bundle(bundle, self._from_matrix())

    def test_cache_entries_of_both_routes_are_the_same_entry(self) -> None:
        directory = Path.cwd() / ".tmp" / f"sector-stencil-cache-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            written = DeferredNativeNegativeLaplacian(
                self.grid, cache_directory=directory
            )
            self.assertEqual(len(written.hashed_cache_key), 64)
            with patch.object(
                DeferredNativeNegativeLaplacian,
                "materialize",
                side_effect=AssertionError("the full-grid matrix was built"),
            ):
                first = load_or_build_reduced_operators(
                    self.decomposition,
                    written,
                    self.nonlocal_operator,
                    cache_directory=directory,
                    kinetic_key_seed=written.cache_key,
                    decomposition_key_seed="d" * 64,
                    representations=(1,),
                )
            self.assertEqual(first.cache_info.status, "miss-written")
            self.assertTrue(first.cache_info.stencil_builder.startswith("direct"))
            # A cache entry is always the complete bundle.
            self.assertTrue(all(item is not None for item in first.stencil_metadata))
            # The former route addresses the entry the direct one wrote; a
            # descriptor without a seed supplies the same key by itself.
            with (
                patch.dict(os.environ, PARSEC_SECTOR_STENCIL="csr"),
                patch.object(
                    DeferredNativeNegativeLaplacian,
                    "materialize",
                    side_effect=AssertionError("the full-grid matrix was built"),
                ),
            ):
                second = load_or_build_reduced_operators(
                    self.decomposition,
                    DeferredNativeNegativeLaplacian(
                        self.grid, cache_directory=directory
                    ),
                    self.nonlocal_operator,
                    cache_directory=directory,
                    decomposition_key_seed="d" * 64,
                )
            self.assertEqual(second.cache_info.status, "hit")
            self.assertEqual(second.cache_info.stencil_builder, "cached")
            self.assertEqual(second.cache_info.key, first.cache_info.key)
            self.assertEqual(second.cache_info.path, first.cache_info.path)
            self.assert_same_bundle(second, first)
            self.assert_same_bundle(first, self._from_matrix())

            # An entry written by the former route has that key and content.
            for generated in directory.iterdir():
                generated.unlink()
            with (
                patch.dict(
                    os.environ,
                    PARSEC_SECTOR_STENCIL="csr",
                    PARSEC_NATIVE_SECTOR_ASSEMBLY="0",
                ),
                self._materialized(),
            ):
                former = load_or_build_reduced_operators(
                    self.decomposition,
                    DeferredNativeNegativeLaplacian(
                        self.grid, cache_directory=directory
                    ),
                    self.nonlocal_operator,
                    cache_directory=directory,
                    decomposition_key_seed="d" * 64,
                )
            self.assertEqual(former.cache_info.status, "miss-written")
            self.assertEqual(former.cache_info.key, first.cache_info.key)
            self.assert_same_bundle(former, first)
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()

    def test_resident_bundle_key_comes_from_the_descriptor(self) -> None:
        descriptor = DeferredNativeNegativeLaplacian(self.grid)
        self.assertIsNone(descriptor.hashed_cache_key)
        with (
            patch.dict(
                os.environ,
                {
                    "PARSEC_ACCELERATED_RESIDENT": "1",
                    "PARSEC_RESIDENT_OPERATOR_CACHE_SIZE": "1",
                },
            ),
            patch.object(
                DeferredNativeNegativeLaplacian,
                "materialize",
                side_effect=AssertionError("the full-grid matrix was built"),
            ),
        ):
            first = load_or_build_reduced_operators(
                self.decomposition,
                descriptor,
                self.nonlocal_operator,
                cache_directory=None,
            )
            second = load_or_build_reduced_operators(
                self.decomposition,
                DeferredNativeNegativeLaplacian(self.grid),
                self.nonlocal_operator,
                cache_directory=None,
            )
        self.assertEqual(len(first.cache_info.key), 64)
        self.assertEqual(descriptor.hashed_cache_key, descriptor.cache_key)
        self.assertEqual(second.cache_info.status, "memory-hit")
        self.assertEqual(second.cache_info.stencil_builder, "cached")
        self.assertIs(second.stencil_metadata, first.stencil_metadata)

    def test_matrix_operator_keeps_the_csr_route(self) -> None:
        with patch.object(
            sector_stencil,
            "build_sector_stencils",
            side_effect=AssertionError("the direct builder ran"),
        ):
            bundle = self._from_matrix()
        self.assertEqual(bundle.cache_info.stencil_builder, "csr")


class DeferredDescriptorTests(unittest.TestCase):
    def test_entry_count_from_the_lookup_mask(self) -> None:
        settings = (
            GridSettings(spacing=0.8, radius=2.8, expansion_order=4, shift=_HALF),
            GridSettings(spacing=0.65, radius=2.8, expansion_order=8, shift=_ZERO),
            GridSettings(spacing=0.5, radius=3.3, expansion_order=12, shift=_HALF),
            # Wider than the domain: the stencil leaves the lookup table.
            GridSettings(spacing=0.9, radius=1.9, expansion_order=12, shift=_HALF),
            GridSettings(
                spacing=0.7,
                radius=3.0,
                expansion_order=6,
                shift=_HALF,
                domain_shape="box",
                box_lengths=(4.2, 2.9, 3.6),
            ),
        )
        for item in settings:
            with self.subTest(settings=item):
                grid = build_cluster_grid(item)
                descriptor = DeferredNativeNegativeLaplacian(grid)
                self.assertEqual(descriptor.nnz, build_negative_laplacian(grid).nnz)
                self.assertEqual(descriptor.shape, (grid.size, grid.size))

    def test_applying_the_descriptor_builds_the_matrix_on_first_use(self) -> None:
        from parsec_python.Hamiltonian import KohnShamHamiltonian

        grid = build_cluster_grid(
            GridSettings(spacing=0.8, radius=2.8, expansion_order=4, shift=_HALF)
        )
        matrix = build_negative_laplacian(grid)
        vectors = np.random.default_rng(3).standard_normal((grid.size, 2))
        descriptor = DeferredNativeNegativeLaplacian(grid)
        with patch.object(
            DeferredNativeNegativeLaplacian,
            "materialize",
            autospec=True,
            side_effect=lambda _descriptor: matrix,
        ) as materialize:
            # What ``system.reference.hamiltonian`` builds for a diagnostic.
            hamiltonian = KohnShamHamiltonian(
                descriptor, np.zeros(grid.size), _nonlocal(grid)
            )
            materialize.assert_not_called()
            np.testing.assert_array_equal(
                hamiltonian.apply_kinetic(vectors), matrix @ vectors
            )
            materialize.assert_called_once()
        if native_available():
            np.testing.assert_array_equal(descriptor @ vectors, matrix @ vectors)
            self.assertTrue(descriptor.materialized)

    def test_key_is_hashed_on_request_or_for_a_cache_directory(self) -> None:
        from parsec_python.acceleration.Laplacian import deferred

        grid = build_cluster_grid(
            GridSettings(spacing=0.8, radius=2.8, expansion_order=4, shift=_HALF)
        )
        with patch.object(
            deferred, "_operator_key", side_effect=AssertionError("hashed")
        ):
            descriptor = DeferredNativeNegativeLaplacian(grid)
        self.assertIsNone(descriptor.hashed_cache_key)
        self.assertEqual(descriptor.hash_seconds, 0.0)
        self.assertEqual(descriptor.nnz_cache_status, "disabled")
        key = descriptor.cache_key
        self.assertEqual(len(key), 64)
        self.assertEqual(descriptor.hashed_cache_key, key)
        with patch.object(
            deferred, "_operator_key", side_effect=AssertionError("hashed twice")
        ):
            self.assertEqual(descriptor.cache_key, key)

        directory = Path.cwd() / ".tmp" / f"sector-stencil-key-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            cached = DeferredNativeNegativeLaplacian(grid, cache_directory=directory)
            self.assertEqual(cached.hashed_cache_key, key)
            self.assertEqual(cached.nnz, descriptor.nnz)
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()


class RouteBenchmarkTests(unittest.TestCase):
    def test_host_benchmark_reports_the_same_arrays_for_every_route(self) -> None:
        from parsec_python.acceleration.benchmarks import sector_stencil_routes

        source = Path(__file__).resolve().parents[2] / "tests" / "data" / "H_cli_smoke.in"
        routes = ["direct", "numpy"]
        if native_available() and hasattr(_load_native(), "reduce_sector_csr"):
            routes.append("csr")
        rows = {
            route: sector_stencil_routes.measure(source, route, (0, 3))
            for route in routes
        }
        self.assertEqual(rows["numpy"]["stencil_builder"], "direct-numpy")
        self.assertEqual(rows["direct"]["sectors"], [0, 3])
        for route, row in rows.items():
            with self.subTest(route=route):
                self.assertEqual(row["route"], route)
                self.assertEqual(row["full_grid_matrix_built"], route == "csr")
                self.assertEqual(row["sha256"], rows["numpy"]["sha256"])
                self.assertEqual(row["laplacian_nnz"], rows["numpy"]["laplacian_nnz"])
                self.assertEqual(len(row["sha256"]["0"]), 64)
        self.assertNotEqual(
            rows["numpy"]["sha256"]["0"], rows["numpy"]["sha256"]["3"]
        )
        # The former route is run with its native reduction, whose order of
        # repeated columns the new builders reproduce.
        self.assertEqual(
            {route: row["native_sector_assembly"] for route, row in rows.items()},
            {route: "1" if route == "csr" else None for route in routes},
        )

    def test_command_line_compares_only_what_it_can(self) -> None:
        from parsec_python.acceleration.benchmarks import sector_stencil_routes

        source = Path(__file__).resolve().parents[2] / "tests" / "data" / "H_cli_smoke.in"

        def run(routes, **environment):
            printed, errors = io.StringIO(), io.StringIO()
            with (
                patch.dict(os.environ, environment),
                redirect_stdout(printed),
                redirect_stderr(errors),
            ):
                status = sector_stencil_routes.main(
                    ["--input", str(source), "--routes", routes, "--sectors", "0"]
                )
            return status, printed.getvalue().splitlines(), errors.getvalue()

        # One route is measured; there is nothing to call identical.
        status, lines, _ = run("numpy")
        self.assertEqual((status, lines[-1]), (0, "SECTOR_STENCILS_NOT_COMPARED"))
        row = json.loads(lines[0])
        self.assertEqual(row["stencil_builder"], "direct-numpy")
        if sys.platform in ("linux", "win32"):
            self.assertGreater(row["process_rss_high_water_bytes"], 0)
        # The SciPy reduction of the former route adds repeated columns in
        # another order: no bitwise comparison with it.
        status, lines, errors = run("csr,direct", PARSEC_NATIVE_SECTOR_ASSEMBLY="0")
        self.assertEqual((status, lines), (2, []))
        self.assertIn("PARSEC_NATIVE_SECTOR_ASSEMBLY", errors)


if __name__ == "__main__":
    unittest.main()
