"""Symmetry detection and representation build against whole-grid row maps.

The reference functions are the former constructions: every atom-preserving
operation is tested on the atoms and mapped on the whole grid, the identity
included, before a group is chosen; orbits are labelled by sorting; the
representation images are read from whole-grid maps; and every phase is
scattered per representation and operation.  The detector and the
representation build must return the same arrays bit for bit, with
``PARSEC_SYMMETRY_FAST_MAPS`` on (the first test class) and off (the second).
"""

from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations, product
import os
import unittest
from unittest import mock

import numpy as np

from parsec_python.acceleration.Symmetry import (
    AxisReflectionReduction,
    ReflectionRepresentationDecomposition,
    SignedPermutationReduction,
    axis_reflection,
    representations,
)
from parsec_python.Grid import RealSpaceGrid, build_cluster_grid
from parsec_python.models import Atom, GridSettings


def _whole_grid_row_mapping(grid, operation, lattice_tolerance):
    """Map every row under one signed permutation by the coordinate gather."""

    parameters = axis_reflection._integer_signed_permutation_parameters(
        grid, operation, lattice_tolerance
    )
    if parameters is None:
        return None
    sources, factors, offset = parameters
    transformed = grid.integer_coordinates[:, sources] * factors + offset
    rows = grid.rows_for_integer_coordinates(transformed)
    return None if np.any(rows < 0) else rows


def _sorted_orbit_labels(mappings):
    """Label orbits by sorting the smallest image of every row."""

    canonical_rows = np.minimum.reduce(mappings)
    representative_rows = np.unique(canonical_rows)
    full_to_wedge = np.searchsorted(representative_rows, canonical_rows)
    multiplicities = np.bincount(
        full_to_wedge, minlength=representative_rows.size
    ).astype(np.int64, copy=False)
    return dict(
        representative_rows=np.ascontiguousarray(representative_rows, dtype=np.int64),
        full_to_wedge=np.ascontiguousarray(full_to_wedge, dtype=np.int64),
        multiplicities=np.ascontiguousarray(multiplicities, dtype=np.int64),
    )


def _reference_axis_detect(grid, atoms, atom_tolerance=1.0e-7, lattice_tolerance=1.0e-10):
    """Former diagonal detection: a whole-grid map for every valid sign triple."""

    valid_signs, mappings = [], []
    for values in product((-1, 1), repeat=3):
        signs = np.asarray(values, dtype=np.int8)
        if not axis_reflection._atoms_are_invariant(atoms, signs, atom_tolerance):
            continue
        mapping = _whole_grid_row_mapping(grid, np.diag(signs), lattice_tolerance)
        if mapping is None:
            continue
        valid_signs.append(signs)
        mappings.append(mapping)
    if not mappings:
        raise ValueError("the identity operation did not preserve the active grid")
    labels = _sorted_orbit_labels(mappings)
    if np.any(labels["multiplicities"] <= 0):
        raise RuntimeError("invalid empty symmetry orbit")
    return AxisReflectionReduction(
        signs=np.ascontiguousarray(np.vstack(valid_signs), dtype=np.int8), **labels
    )


def _reference_signed_detect(grid, atoms, atom_tolerance=1.0e-7, lattice_tolerance=1.0e-10):
    """Former generalized detection: every atom-preserving involution is mapped first."""

    axis = _reference_axis_detect(grid, atoms, atom_tolerance, lattice_tolerance)
    valid = {}
    identity = np.eye(3, dtype=np.int8)
    for operation in axis_reflection._signed_permutation_operations():
        if not np.array_equal(operation @ operation, identity):
            continue
        if not axis_reflection._atoms_are_invariant_operation(atoms, operation, atom_tolerance):
            continue
        mapping = _whole_grid_row_mapping(grid, operation, lattice_tolerance)
        if mapping is not None:
            valid[operation.tobytes()] = (operation, mapping)

    candidates = [
        item[0]
        for key, item in sorted(valid.items(), key=lambda pair: pair[0])
        if not np.array_equal(item[0], identity)
    ]
    best = None
    for rank in range(3, 0, -1):
        for generators in combinations(candidates, rank):
            if any(
                not np.array_equal(left @ right, right @ left)
                for left, right in combinations(generators, 2)
            ):
                continue
            operations, bits, seen, usable = [], [], set(), True
            for exponents in product((0, 1), repeat=rank):
                operation = identity.copy()
                for enabled, generator in zip(exponents, generators):
                    if enabled:
                        operation = operation @ generator
                key = operation.astype(np.int8, copy=False).tobytes()
                if key in seen or key not in valid:
                    usable = False
                    break
                seen.add(key)
                operations.append(valid[key][0])
                bits.append(exponents)
            if usable and len(operations) > axis.group_order:
                best = (operations, np.asarray(bits, dtype=np.int8))
                break
        if best is not None:
            break
    if best is None:
        return axis

    operations, generator_bits = best
    labels = _sorted_orbit_labels(
        [valid[operation.tobytes()][1] for operation in operations]
    )
    return SignedPermutationReduction(
        signs=np.ones((len(operations), 3), dtype=np.int8),
        operations=np.ascontiguousarray(operations, dtype=np.int8),
        generator_bits=np.ascontiguousarray(generator_bits, dtype=np.int8),
        **labels,
    )


def _reference_representative_images(grid, reduction):
    """Former representation images: whole-grid maps read at the representatives."""

    generalized = isinstance(reduction, SignedPermutationReduction)
    images = np.empty((reduction.group_order, reduction.wedge_size), dtype=np.int64)
    for index, value in enumerate(reduction.operations if generalized else reduction.signs):
        mapping = _whole_grid_row_mapping(
            grid, value if generalized else np.diag(value), 1.0e-10
        )
        if mapping is None:
            raise RuntimeError("accepted symmetry operation no longer maps the grid")
        images[index] = mapping[reduction.representative_rows]
    return images


def _reference_decomposition(grid, reduction):
    """Former representation build: sorted image counts and one scatter per phase."""

    order = reduction.group_order
    rows = []
    if isinstance(reduction, SignedPermutationReduction):
        rank = int(reduction.generator_bits.shape[1])
        for parity in product((0, 1), repeat=rank):
            exponents = np.asarray(parity, dtype=np.int8)
            rows.append(
                np.where((reduction.generator_bits @ exponents) % 2, -1, 1).astype(np.int8)
            )
    else:
        seen = set()
        signs = reduction.signs.astype(np.int8, copy=False)
        for parity in representations._PARSEC_REFLECTION_PARITIES:
            character = np.ones(order, dtype=np.int8)
            for axis, exponent in enumerate(parity):
                if exponent:
                    character *= signs[:, axis]
            if character.tobytes() not in seen:
                seen.add(character.tobytes())
                rows.append(character)
    characters = np.ascontiguousarray(np.vstack(rows), dtype=np.int8)
    representatives = reduction.representative_rows
    images = _reference_representative_images(grid, reduction)
    phases = np.zeros((order, reduction.full_size), dtype=np.int8)
    orbit_to_sector = np.full((order, reduction.wedge_size), -1, dtype=np.int64)
    counts = 1 + np.count_nonzero(np.diff(np.sort(images, axis=0), axis=0), axis=0)
    if not np.array_equal(counts, reduction.multiplicities):
        raise RuntimeError("reflection operations do not reproduce every orbit")
    stabilizer = images == representatives[None, :]
    for representation, character in enumerate(characters):
        admitted = np.flatnonzero(
            np.all((~stabilizer) | (character[:, None] == 1), axis=0)
        )
        if admitted.size < 1:
            raise RuntimeError("a reflection representation has zero dimension")
        orbit_to_sector[representation, admitted] = np.arange(admitted.size, dtype=np.int64)
        for operation in range(order):
            phases[representation, images[operation, admitted]] = character[operation]
    for representation in range(order):
        admitted = orbit_to_sector[representation] >= 0
        if not np.all(phases[representation, representatives[admitted]] == 1):
            raise RuntimeError("invalid representation phase convention")
    return dict(characters=characters, phases=phases, orbit_to_sector=orbit_to_sector)


@contextmanager
def _looked_up_rows():
    """Record how many rows each call sends through the grid lookup table."""

    counts = []
    lookup = RealSpaceGrid.rows_for_integer_coordinates

    def counting(grid, points):
        counts.append(len(points))
        return lookup(grid, points)

    with mock.patch.object(RealSpaceGrid, "rows_for_integer_coordinates", counting):
        yield counts


@contextmanager
def _mapped_rows():
    """Record the rows gathered through the coordinates and the maps read off the table."""

    record = dict(gathered=[], table=0)
    read = axis_reflection._RowOrderTable.row_mapping

    def counting(table, *parameters):
        record["table"] += 1
        return read(table, *parameters)

    with (
        _looked_up_rows() as gathered,
        mock.patch.object(axis_reflection._RowOrderTable, "row_mapping", counting),
    ):
        record["gathered"] = gathered
        yield record


def _diamond_cluster(radius):
    """Atoms of a diamond lattice around one of them: the symmetry of a tetrahedron."""

    span = int(np.ceil(radius)) + 1
    cells = np.arange(-span, span + 1)
    sites = np.stack(np.meshgrid(cells, cells, cells, indexing="ij"), axis=-1).reshape(-1, 3)
    face_centred = sites[np.sum(sites, axis=1) % 2 == 0].astype(np.float64)
    positions = np.concatenate((face_centred, face_centred + 0.5))
    return positions[np.linalg.norm(positions, axis=1) <= radius]


class SymmetryRowMapTests(unittest.TestCase):
    """Detected groups, orbit maps and sector maps must equal the whole-grid ones."""

    fast_maps = "1"

    def setUp(self) -> None:
        switch = mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_FAST_MAPS": self.fast_maps})
        switch.start()
        self.addCleanup(switch.stop)

    @property
    def fast(self) -> bool:
        return self.fast_maps == "1"

    def _expected_maps(self, grid, whole, restricted=()):
        """Maps of a detection or a build: off the table, or gathered through the coordinates."""

        if self.fast:
            return dict(gathered=[], table=whole)
        return dict(gathered=[grid.size] * whole + list(restricted), table=0)

    @classmethod
    def setUpClass(cls) -> None:
        def grid(shift, **domain):
            return build_cluster_grid(
                GridSettings(spacing=0.7, radius=3.6, expansion_order=4, shift=shift, **domain)
            )

        cls.grids = {
            "half shift": grid((0.5, 0.5, 0.5)),
            "zero shift": grid((0.0, 0.0, 0.0)),
            # x <-> y does not preserve this lattice, nor the active points of this box.
            "shift in x": grid((0.5, 0.0, 0.0)),
            "long x box": grid(
                (0.5, 0.5, 0.5), domain_shape="box", box_lengths=(6.0, 4.4, 4.4)
            ),
        }
        corner = 0.9
        methane = (
            Atom("C", (0.0, 0.0, 0.0)),
            Atom("H", (corner, corner, corner)),
            Atom("H", (corner, -corner, -corner)),
            Atom("H", (-corner, corner, -corner)),
        )
        cls.atom_sets = {
            "centre": (Atom("C", (0.0, 0.0, 0.0)),),
            # Ten involutions preserve a tetrahedron; its largest commuting group is D2.
            "methane": methane + (Atom("H", (-corner, -corner, corner)),),
            # A stretched bond leaves the x <-> y mirror and no diagonal operation.
            "stretched methane": methane + (Atom("H", (-corner, -corner, corner + 0.05)),),
            "atom on x": (Atom("H", (0.3, 0.0, 0.0)),),
            "atom off every axis": (Atom("H", (0.3, 0.2, 0.1)),),
            "xy dimer": (Atom("H", (1.0, 0.0, 0.0)), Atom("H", (0.0, 1.0, 0.0))),
            "labelled xy dimer": (Atom("H", (1.0, 0.0, 0.0)), Atom("C", (0.0, 1.0, 0.0))),
            "yz dimer": (Atom("H", (0.0, 1.0, 0.0)), Atom("H", (0.0, 0.0, 1.0))),
            "diagonal dimer": (Atom("H", (1.0, 1.0, 0.0)), Atom("H", (-1.0, -1.0, 0.0))),
            # Three mirrors that do not commute: only one of them can be chosen.
            "trimer": tuple(Atom("H", position) for position in np.eye(3)),
        }
        # Species above the dense matching limit, with atoms on the twofold
        # axes and on the mirror planes: a tetrahedral cluster of two species
        # as in the nanodiamonds, the same with one atom moved, and a cube.
        diamond = _diamond_cluster(5.6)
        outer = np.linalg.norm(diamond, axis=1) > 5.35
        cls.large_atom_sets = {
            "diamond": tuple(
                Atom("H" if hydrogen else "C", position)
                for position, hydrogen in zip(diamond, outer)
            ),
            "diamond with a moved atom": tuple(
                Atom("C", position + (1.0e-3 if index == 11 else 0.0))
                for index, position in enumerate(diamond)
            ),
            "cube": tuple(
                Atom("C", position)
                for position in np.stack(
                    np.meshgrid(*(np.arange(-4.0, 5.0),) * 3, indexing="ij"), axis=-1
                ).reshape(-1, 3)
            ),
        }

    def _assert_same_arrays(self, actual, expected, names) -> None:
        for name in names:
            left, right = getattr(actual, name), getattr(expected, name)
            self.assertEqual(left.dtype, right.dtype, name)
            self.assertTrue(left.flags.c_contiguous, name)
            np.testing.assert_array_equal(left, right, err_msg=name)

    def _assert_same_reduction(self, actual, expected) -> None:
        self.assertIs(type(actual), type(expected))
        names = ["signs", "representative_rows", "full_to_wedge", "multiplicities"]
        if isinstance(expected, SignedPermutationReduction):
            names += ["operations", "generator_bits"]
        self._assert_same_arrays(actual, expected, names)

    def test_detected_groups_and_orbit_maps_equal_the_whole_grid_construction(self) -> None:
        detected = set()
        for (grid_name, grid), (atoms_name, atoms) in product(
            self.grids.items(), self.atom_sets.items()
        ):
            with self.subTest(grid=grid_name, atoms=atoms_name):
                expected = _reference_signed_detect(grid, atoms)
                actual = SignedPermutationReduction.detect(grid, atoms)
                self._assert_same_reduction(actual, expected)
                self._assert_same_reduction(
                    AxisReflectionReduction.detect(grid, atoms),
                    _reference_axis_detect(grid, atoms),
                )
                detected.add((type(actual).__name__, actual.group_order))
        # Identity only, every diagonal order, and generalized groups of rank one to three.
        self.assertEqual(
            detected,
            {("AxisReflectionReduction", order) for order in (1, 2, 4, 8)}
            | {("SignedPermutationReduction", order) for order in (2, 4, 8)},
        )

    def test_candidates_failing_on_the_grid_leave_the_diagonal_group(self) -> None:
        dimer = self.atom_sets["xy dimer"]
        swap = np.array(((0, 1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.int8)
        self.assertTrue(axis_reflection._atoms_are_invariant_operation(dimer, swap, 1.0e-7))
        self.assertIsInstance(
            SignedPermutationReduction.detect(self.grids["half shift"], dimer),
            SignedPermutationReduction,
        )
        # The lattice rejects the swap before any row is looked up; the box
        # rejects it only through an image outside the active points.
        for grid_name, lattice_preserved in (("shift in x", False), ("long x box", True)):
            grid = self.grids[grid_name]
            parameters = axis_reflection._integer_signed_permutation_parameters(
                grid, swap, 1.0e-10
            )
            self.assertEqual(parameters is not None, lattice_preserved)
            self.assertIsNone(_whole_grid_row_mapping(grid, swap, 1.0e-10))
            reduction = SignedPermutationReduction.detect(grid, dimer)
            self.assertIs(type(reduction), AxisReflectionReduction)
            self.assertEqual(reduction.group_order, 2)
            self._assert_same_reduction(reduction, _reference_signed_detect(grid, dimer))

        # The first mirror of the trimer in candidate order is x <-> z.  Where
        # it and x <-> y fail on the grid, the search must go on to y <-> z.
        trimer = self.atom_sets["trimer"]
        for grid_name, mirror in (
            ("half shift", ((0, 0, 1), (0, 1, 0), (1, 0, 0))),
            ("shift in x", ((1, 0, 0), (0, 0, 1), (0, 1, 0))),
            ("long x box", ((1, 0, 0), (0, 0, 1), (0, 1, 0))),
        ):
            reduction = SignedPermutationReduction.detect(self.grids[grid_name], trimer)
            self.assertIsInstance(reduction, SignedPermutationReduction)
            np.testing.assert_array_equal(reduction.operations, (np.eye(3), mirror))
            self._assert_same_reduction(
                reduction, _reference_signed_detect(self.grids[grid_name], trimer)
            )

    def test_discarded_involutions_are_never_mapped(self) -> None:
        grid, atoms = self.grids["half shift"], self.atom_sets["methane"]
        with _mapped_rows() as diagonal:
            AxisReflectionReduction.detect(grid, atoms)
        with _mapped_rows() as generalized:
            reduction = SignedPermutationReduction.detect(grid, atoms)
        with _looked_up_rows() as reference:
            _reference_signed_detect(grid, atoms)
        self.assertEqual(reduction.group_order, 4)
        # Four diagonal maps and ten involution maps, none of the latter used.
        self.assertEqual(reference, [grid.size] * 14)
        self.assertEqual(generalized, diagonal)
        # The three diagonal maps besides the identity, and no other.
        self.assertEqual(diagonal, self._expected_maps(grid, 3))

    def test_representation_build_equals_the_whole_grid_images(self) -> None:
        built = 0
        for (grid_name, grid), (atoms_name, atoms) in product(
            self.grids.items(), self.atom_sets.items()
        ):
            for detect in (AxisReflectionReduction.detect, SignedPermutationReduction.detect):
                reduction = detect(grid, atoms)
                if reduction.group_order <= 1:
                    continue
                with self.subTest(grid=grid_name, atoms=atoms_name, kind=type(reduction).__name__):
                    with _looked_up_rows() as reference_rows:
                        expected_images = _reference_representative_images(grid, reduction)
                    self.assertEqual(sum(reference_rows), reduction.group_order * grid.size)
                    with _mapped_rows() as rows:
                        actual_images = representations._representative_images(grid, reduction)
                    self.assertEqual(actual_images.dtype, expected_images.dtype)
                    np.testing.assert_array_equal(actual_images, expected_images)
                    # Every operation but the identity, on the representatives only
                    # or off the table.
                    self.assertEqual(
                        rows,
                        self._expected_maps(
                            grid,
                            0 if not self.fast else reduction.group_order - 1,
                            [reduction.wedge_size] * (reduction.group_order - 1),
                        ),
                    )

                    with mock.patch.object(
                        representations, "_representative_images", _reference_representative_images
                    ):
                        expected = ReflectionRepresentationDecomposition.build(grid, reduction)
                    actual = ReflectionRepresentationDecomposition.build(grid, reduction)
                    self._assert_same_arrays(
                        actual, expected, ("characters", "phases", "orbit_to_sector")
                    )
                    self.assertEqual(actual.sector_sizes, expected.sector_sizes)
                    # The former build, written out independently of the module.
                    former = _reference_decomposition(grid, reduction)
                    for name, values in former.items():
                        produced = getattr(actual, name)
                        self.assertEqual(produced.dtype, values.dtype, name)
                        self.assertEqual(produced.tobytes(), values.tobytes(), name)
                    built += 1
        self.assertGreater(built, 40)

    def test_representatives_leaving_the_grid_are_still_refused(self) -> None:
        # A group found on the sphere is not one of the box: x <-> y takes
        # representatives of the box beyond its short side.
        grid = self.grids["long x box"]
        accepted = SignedPermutationReduction.detect(
            self.grids["half shift"], self.atom_sets["xy dimer"]
        )
        own = AxisReflectionReduction.detect(grid, self.atom_sets["centre"])
        foreign = SignedPermutationReduction(
            signs=accepted.signs,
            representative_rows=own.representative_rows,
            full_to_wedge=own.full_to_wedge,
            multiplicities=own.multiplicities,
            operations=accepted.operations,
            generator_bits=accepted.generator_bits,
        )
        with self.assertRaisesRegex(RuntimeError, "no longer maps the grid"):
            representations._representative_images(grid, foreign)
        with self.assertRaisesRegex(RuntimeError, "no longer maps the grid"):
            ReflectionRepresentationDecomposition.build(grid, foreign)

    def _assert_sorted_labels(self, mappings, *, sorting: bool = True) -> None:
        expected = _sorted_orbit_labels(mappings)
        if sorting:
            actual = axis_reflection._orbit_labels(mappings)
        else:
            with mock.patch.object(np, "unique", side_effect=AssertionError("sorted")):
                actual = axis_reflection._orbit_labels(mappings)
        for values, name in zip(
            actual, ("representative_rows", "full_to_wedge", "multiplicities")
        ):
            np.testing.assert_array_equal(values, expected[name], err_msg=name)

    def test_orbit_labels_equal_the_sorted_labels(self) -> None:
        swap = np.array(((0, 1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.int8)
        for grid_name in ("half shift", "zero shift"):
            grid = self.grids[grid_name]
            maps = {
                signs: _whole_grid_row_mapping(grid, np.diag(signs), 1.0e-10)
                for signs in product((-1, 1), repeat=3)
            }
            identity, flip_x, flip_y = maps[1, 1, 1], maps[-1, 1, 1], maps[1, -1, 1]
            mirror = _whole_grid_row_mapping(grid, swap, 1.0e-10)
            # A group is labelled without sorting its smallest images.
            for group in (
                list(maps.values()),
                [identity, flip_x, flip_y, maps[-1, -1, 1]],
                [identity, mirror, maps[1, 1, -1], mirror[maps[1, 1, -1]]],
                [identity, mirror],
                [identity],
            ):
                self._assert_sorted_labels(group, sorting=False)

            # Two reflections without their product: the smallest image of a
            # row can itself be moved to a smaller row, and only the sort
            # returns the former labels.
            open_set = [identity, flip_x, flip_y]
            smallest = np.minimum.reduce(open_set)
            self.assertNotEqual(
                np.count_nonzero(smallest == np.arange(grid.size)),
                np.unique(smallest).size,
            )
            self._assert_sorted_labels(open_set)

        generator = np.random.default_rng(41)
        for count, rows in ((1, 50), (3, 400), (8, 400)):
            self._assert_sorted_labels(list(generator.integers(0, rows, size=(count, rows))))
            permutations = [generator.permutation(rows) for _ in range(count)]
            self._assert_sorted_labels([np.arange(rows)] + permutations)

    def test_operations_accepted_without_their_product_keep_the_former_labels(self) -> None:
        # Each reflection in x and in y moves these atoms by 0.8e-7, inside
        # the tolerance; their product moves them by 1.13e-7, outside it.
        step = 0.8e-7
        atoms = (
            Atom("H", (1.0, 1.0, 0.0)),
            Atom("H", (-1.0, 1.0 + step, 0.0)),
            Atom("H", (1.0 + step, -1.0, 0.0)),
            Atom("H", (-1.0 - step, -1.0 - step, 0.0)),
        )
        for grid_name in ("half shift", "zero shift"):
            grid = self.grids[grid_name]
            reduction = AxisReflectionReduction.detect(grid, atoms)
            self.assertEqual(reduction.group_order, 6)
            self._assert_same_reduction(reduction, _reference_axis_detect(grid, atoms))
            self._assert_same_reduction(
                SignedPermutationReduction.detect(grid, atoms),
                _reference_signed_detect(grid, atoms),
            )

    def test_diagonal_operations_are_decided_by_the_diagonal_detection(self) -> None:
        axis = np.arange(-4, 5, dtype=np.float64)
        lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        self.assertGreater(len(lattice), axis_reflection._DENSE_MATCHING_LIMIT)
        moved = lattice.copy()
        moved[7] += 1.0e-3
        atom_sets = dict(self.atom_sets)
        atom_sets["lattice"] = tuple(Atom("C", position) for position in lattice)
        atom_sets["lattice with a moved atom"] = tuple(Atom("C", position) for position in moved)
        # Displacements near half the tolerance leave every reflection of this
        # cube within a few percent of the threshold, on either side of it.
        half = np.arange(-1.5, 2.0)
        cube = np.stack(np.meshgrid(half, half, half, indexing="ij"), axis=-1).reshape(-1, 3)
        generator = np.random.default_rng(43)
        for trial in range(6):
            shaken = cube + generator.uniform(-0.45e-7, 0.45e-7, size=cube.shape)
            atom_sets[f"shaken {trial}"] = tuple(Atom("C", position) for position in shaken)
        decisions = set()
        for name, atoms in atom_sets.items():
            for values in product((-1, 1), repeat=3):
                signs = np.asarray(values, dtype=np.int8)
                by_signs = axis_reflection._atoms_are_invariant(atoms, signs, 1.0e-7)
                by_matrix = axis_reflection._atoms_are_invariant_operation(
                    atoms, np.diag(signs), 1.0e-7
                )
                self.assertEqual(by_signs, by_matrix, (name, values))
                if values != (1, 1, 1):
                    decisions.add((name.split()[0], by_signs))
        self.assertTrue({("lattice", True), ("lattice", False)} <= decisions)
        self.assertTrue({("shaken", True), ("shaken", False)} <= decisions)

        # The generalized detector then tests the twelve other involutions only.
        grid, atoms = self.grids["half shift"], self.atom_sets["methane"]
        with mock.patch.object(
            axis_reflection,
            "_atoms_are_invariant_operation",
            wraps=axis_reflection._atoms_are_invariant_operation,
        ) as tested:
            SignedPermutationReduction.detect(grid, atoms)
        self.assertEqual(tested.call_count, 12)
        for call in tested.call_args_list:
            self.assertFalse(np.array_equal(np.abs(call.args[1]), np.eye(3)))

    def test_identity_map_is_the_row_numbers_without_a_lookup(self) -> None:
        identity = np.eye(3, dtype=np.int8)
        for grid_name, grid in self.grids.items():
            with self.subTest(grid=grid_name):
                expected = _whole_grid_row_mapping(grid, identity, 1.0e-10)
                np.testing.assert_array_equal(expected, np.arange(grid.size))
                selected = np.arange(grid.size)[::-3]
                with _looked_up_rows() as rows:
                    whole = axis_reflection._grid_row_mapping_operation(grid, identity, 1.0e-10)
                    by_signs = axis_reflection._grid_row_mapping(
                        grid, np.ones(3, dtype=np.int8), 1.0e-10
                    )
                    part = axis_reflection._grid_row_mapping_operation(
                        grid, identity, 1.0e-10, selected
                    )
                self.assertEqual(rows, [])
                for actual, reference in (
                    (whole, expected),
                    (by_signs, expected),
                    (part, expected[selected]),
                ):
                    self.assertEqual(actual.dtype, reference.dtype)
                    np.testing.assert_array_equal(actual, reference)
                self.assertFalse(np.shares_memory(part, selected))

                # Every other diagonal operation is still read from the table.
                for values in product((-1, 1), repeat=3):
                    reference = _whole_grid_row_mapping(grid, np.diag(values), 1.0e-10)
                    actual = axis_reflection._grid_row_mapping(
                        grid, np.asarray(values, dtype=np.int8), 1.0e-10
                    )
                    if reference is None:
                        self.assertIsNone(actual)
                    else:
                        np.testing.assert_array_equal(actual, reference)

        grid, atoms = self.grids["half shift"], self.atom_sets["methane"]
        with _mapped_rows() as rows:
            reduction = SignedPermutationReduction.detect(grid, atoms)
        self.assertEqual(rows, self._expected_maps(grid, 3))
        with _mapped_rows() as rows:
            ReflectionRepresentationDecomposition.build(grid, reduction)
        self.assertEqual(
            rows,
            self._expected_maps(grid, 3 if self.fast else 0, [reduction.wedge_size] * 3),
        )

    def test_switch_is_on_unless_named_off(self) -> None:
        for value, expected in (
            (None, True),
            ("1", True),
            (" on ", True),
            ("0", False),
            (" Off ", False),
            ("false", False),
            ("no", False),
        ):
            with self.subTest(value=value), mock.patch.dict(os.environ):
                os.environ.pop("PARSEC_SYMMETRY_FAST_MAPS", None)
                if value is not None:
                    os.environ["PARSEC_SYMMETRY_FAST_MAPS"] = value
                self.assertIs(axis_reflection._fast_maps_requested(), expected)

    @staticmethod
    def _cropped(grid):
        """The same grid with the empty planes of its lookup table removed."""

        occupied = [
            np.flatnonzero(
                (grid.lookup >= 0).any(axis=tuple(other for other in range(3) if other != axis))
            )
            for axis in range(3)
        ]
        low = np.array([planes[0] for planes in occupied])
        high = np.array([planes[-1] for planes in occupied])
        return RealSpaceGrid(
            settings=grid.settings,
            integer_coordinates=grid.integer_coordinates,
            coordinates=grid.coordinates,
            index_min=grid.index_min + low,
            index_max=grid.index_min + high,
            lookup=np.ascontiguousarray(
                grid.lookup[low[0] : high[0] + 1, low[1] : high[1] + 1, low[2] : high[2] + 1]
            ),
        )

    def test_table_maps_equal_the_coordinate_gather_for_every_signed_permutation(self) -> None:
        def grid(shift, radius=3.6, **domain):
            return build_cluster_grid(
                GridSettings(
                    spacing=0.7, radius=radius, expansion_order=4, shift=shift, **domain
                )
            )

        grids = dict(self.grids)
        grids.update(
            {
                # A table of odd size: the mirrored window is moved by one plane,
                # as in the large nanodiamond grids.
                "half shift, odd table": grid((0.5, 0.5, 0.5), radius=3.9),
                "zero shift, odd table": grid((0.0, 0.0, 0.0), radius=3.9),
                "shift in x and y": grid((0.5, 0.5, 0.0)),
                "quarter shift": grid((0.25, 0.25, 0.25)),
                "flat box": grid((0.5, 0.5, 0.5), domain_shape="box", box_lengths=(6.0, 6.0, 2.2)),
            }
        )
        self.assertEqual(
            {grids[name].lookup.shape[0] % 2 for name in ("half shift", "half shift, odd table")},
            {0, 1},
        )
        # Tables that are not cubes: an axis exchange leaves them.
        grids["long x box, cropped"] = self._cropped(grids["long x box"])
        grids["flat box, cropped"] = self._cropped(grids["flat box"])
        self.assertEqual(len(set(grids["long x box, cropped"].lookup.shape)), 2)

        outcomes = set()
        for name, lattice in grids.items():
            table = axis_reflection._RowOrderTable(lattice)
            self.assertTrue(table.in_row_order, name)
            for operation in axis_reflection._signed_permutation_operations():
                with self.subTest(grid=name, operation=operation.tolist()):
                    expected = _whole_grid_row_mapping(lattice, operation, 1.0e-10)
                    with _mapped_rows() as rows:
                        actual = axis_reflection._grid_row_mapping_operation(
                            lattice, operation, 1.0e-10, table=table
                        )
                    self.assertEqual(rows["gathered"], [])
                    identity = np.array_equal(operation, np.eye(3))
                    preserved = (
                        axis_reflection._integer_signed_permutation_parameters(
                            lattice, operation, 1.0e-10
                        )
                        is not None
                    )
                    self.assertEqual(rows["table"], int(preserved and not identity))
                    if expected is None:
                        self.assertIsNone(actual)
                    else:
                        self.assertEqual(actual.dtype, np.int64)
                        self.assertTrue(actual.flags.c_contiguous)
                        np.testing.assert_array_equal(actual, expected)
                        # A bijection of the rows.
                        self.assertEqual(np.unique(actual).size, lattice.size)
                    outcomes.add((name.split(",")[0], expected is not None, preserved))
        # Accepted, refused on the lattice, and refused only at the active points.
        self.assertTrue({(True, True), (False, False), (False, True)} <= {o[1:] for o in outcomes})
        self.assertIn(("long x box", False, True), outcomes)

    def test_grid_numbered_in_another_order_keeps_the_coordinate_gather(self) -> None:
        source = self.grids["zero shift"]

        def renumbered(order, lookup=None):
            points = source.integer_coordinates[order]
            if lookup is None:
                lookup = np.full_like(source.lookup, -1)
                local = points - source.index_min
                lookup[local[:, 0], local[:, 1], local[:, 2]] = np.arange(len(points))
            return RealSpaceGrid(
                settings=source.settings,
                integer_coordinates=points,
                coordinates=source.coordinates[order],
                index_min=source.index_min,
                index_max=source.index_max,
                lookup=lookup,
            )

        rows = np.arange(source.size)
        self.assertTrue(axis_reflection._RowOrderTable(renumbered(rows)).in_row_order)
        exchanged = rows.copy()
        exchanged[[4, 5]] = exchanged[[5, 4]]
        grids = {
            "shuffled": renumbered(np.random.default_rng(3).permutation(source.size)),
            "two rows exchanged": renumbered(exchanged),
            "reversed": renumbered(rows[::-1]),
        }
        for name, grid in grids.items():
            table = axis_reflection._RowOrderTable(grid)
            self.assertFalse(table.in_row_order, name)
            for operation in axis_reflection._signed_permutation_operations():
                with self.subTest(grid=name, operation=operation.tolist()):
                    expected = _whole_grid_row_mapping(grid, operation, 1.0e-10)
                    with _mapped_rows() as mapped:
                        actual = axis_reflection._grid_row_mapping_operation(
                            grid, operation, 1.0e-10, table=table
                        )
                    self.assertEqual(mapped["table"], 0)
                    if expected is None:
                        self.assertIsNone(actual)
                    else:
                        np.testing.assert_array_equal(actual, expected)
            for atoms_name in ("centre", "methane", "xy dimer", "atom on x"):
                atoms = self.atom_sets[atoms_name]
                with self.subTest(grid=name, atoms=atoms_name):
                    reduction = SignedPermutationReduction.detect(grid, atoms)
                    self._assert_same_reduction(reduction, _reference_signed_detect(grid, atoms))
                    self.assertGreater(reduction.group_order, 1)
                    # No table to read, and no gather of the whole grid in its place: the
                    # representatives only, as with the switch off.
                    with _mapped_rows() as mapped:
                        images = representations._representative_images(grid, reduction)
                    self.assertEqual(
                        mapped,
                        dict(gathered=[reduction.wedge_size] * (reduction.group_order - 1), table=0),
                    )
                    np.testing.assert_array_equal(
                        images, _reference_representative_images(grid, reduction)
                    )
                    built = ReflectionRepresentationDecomposition.build(grid, reduction)
                    for field, values in _reference_decomposition(grid, reduction).items():
                        np.testing.assert_array_equal(getattr(built, field), values, err_msg=field)

        # A table that does not hold every row, and a grid without points.
        short = source.lookup.copy()
        short[short == source.size - 1] = -1
        self.assertFalse(axis_reflection._RowOrderTable(renumbered(rows, short)).in_row_order)
        empty = RealSpaceGrid(
            settings=source.settings,
            integer_coordinates=source.integer_coordinates[:0],
            coordinates=source.coordinates[:0],
            index_min=source.index_min,
            index_max=source.index_max,
            lookup=np.full_like(source.lookup, -1),
        )
        table = axis_reflection._RowOrderTable(empty)
        self.assertFalse(table.in_row_order)
        mirror = np.diag((-1, 1, 1)).astype(np.int8)
        self.assertEqual(
            axis_reflection._grid_row_mapping_operation(empty, mirror, 1.0e-10, table=table).size, 0
        )

    def test_single_partner_decision_is_the_one_of_the_augmenting_paths(self) -> None:
        tolerance = 1.0e-7
        axis = np.arange(-4.0, 5.0)
        lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        self.assertGreater(len(lattice), axis_reflection._DENSE_MATCHING_LIMIT)
        origin = int(np.flatnonzero(np.all(lattice == 0.0, axis=1))[0])

        def decide(transformed, candidates):
            """The decision by the augmenting paths, by the default route, and by one partner."""

            with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_FAST_MAPS": "0"}):
                with mock.patch.object(
                    axis_reflection,
                    "_single_partner_matching",
                    side_effect=AssertionError("the former route asked for one partner"),
                ):
                    former = axis_reflection._has_tolerance_perfect_matching(
                        transformed, candidates, tolerance
                    )
            with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_FAST_MAPS": "1"}):
                default = axis_reflection._has_tolerance_perfect_matching(
                    transformed, candidates, tolerance
                )
            single = axis_reflection._single_partner_matching(transformed, candidates, tolerance)
            self.assertIs(type(default), bool)
            self.assertEqual(default, former)
            return former, single

        signs = np.array((1.0, -1.0, -1.0))
        self.assertEqual(decide(lattice * signs, lattice), (True, True))
        # An atom without a partner.
        moved = lattice.copy()
        moved[7] += 1.0e-3
        self.assertEqual(decide(moved * signs, moved), (False, False))
        # Two atoms on one candidate and a candidate without an atom: every
        # atom has its one partner, and no matching exists.
        doubled = lattice.copy()
        doubled[5] = lattice[6]
        self.assertEqual(decide(doubled, lattice), (False, False))
        # At the tolerance and one unit in the last place either side of it.
        # The atom at the origin is displaced exactly.
        for scale, matched in (
            (1.0, True),
            (np.nextafter(1.0, 0.0), True),
            (np.nextafter(1.0, 2.0), False),
        ):
            displaced = lattice.copy()
            displaced[origin] = (tolerance * scale, 0.0, 0.0)
            self.assertEqual(
                float(np.linalg.norm(displaced[origin] - lattice[origin])), tolerance * scale
            )
            self.assertEqual(decide(displaced, lattice), (matched, matched), scale)
        # Two candidates inside the radius of one atom: one partner cannot
        # decide, the augmenting paths do.
        twin = np.concatenate((lattice, lattice[:1] + 0.3e-7))
        self.assertEqual(decide(twin.copy(), twin), (True, None))
        crowded = twin.copy()
        crowded[1] = twin[0]
        self.assertEqual(decide(crowded, twin), (False, None))

        # Displacements near half the tolerance put the reflections of a
        # lattice on either side of it.
        generator = np.random.default_rng(53)
        decisions = set()
        for _ in range(4):
            shaken = lattice + generator.uniform(-0.36e-7, 0.36e-7, size=lattice.shape)
            for values in product((-1.0, 1.0), repeat=3):
                former, single = decide(shaken * np.asarray(values), shaken)
                self.assertEqual(single, former)
                decisions.add(former)
        self.assertEqual(decisions, {True, False})

    def test_large_clusters_give_the_arrays_of_the_former_routes(self) -> None:
        orders = {}
        grids = {name: self.grids[name] for name in ("half shift", "zero shift")}
        for (grid_name, grid), (atoms_name, atoms) in product(
            grids.items(), self.large_atom_sets.items()
        ):
            largest = max(
                sum(atom.symbol == symbol for atom in atoms)
                for symbol in {atom.symbol for atom in atoms}
            )
            self.assertGreater(largest, axis_reflection._DENSE_MATCHING_LIMIT)
            with self.subTest(grid=grid_name, atoms=atoms_name):
                produced = {}
                for switch in ("1", "0"):
                    with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_FAST_MAPS": switch}):
                        reduction = SignedPermutationReduction.detect(grid, atoms)
                        decomposition = (
                            ReflectionRepresentationDecomposition.build(grid, reduction)
                            if reduction.group_order > 1
                            else None
                        )
                    produced[switch] = (reduction, decomposition)
                # The whole-grid construction decides the atoms by the augmenting paths.
                with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_FAST_MAPS": "0"}):
                    expected = _reference_signed_detect(grid, atoms)
                reduction, decomposition = produced["1"]
                self._assert_same_reduction(reduction, expected)
                self._assert_same_reduction(reduction, produced["0"][0])
                if decomposition is not None:
                    self._assert_same_arrays(
                        decomposition, produced["0"][1], ("characters", "phases", "orbit_to_sector")
                    )
                    for field, values in _reference_decomposition(grid, reduction).items():
                        self.assertEqual(getattr(decomposition, field).tobytes(), values.tobytes())
                orders[grid_name, atoms_name] = (type(reduction).__name__, reduction.group_order)
        # The twofold axes of the tetrahedron, as in the nanodiamonds; one moved
        # atom leaves a mirror through it; the cube keeps every reflection.
        self.assertEqual(orders["half shift", "diamond"], ("AxisReflectionReduction", 4))
        self.assertLess(orders["half shift", "diamond with a moved atom"][1], 4)
        self.assertEqual(orders["half shift", "cube"], ("AxisReflectionReduction", 8))
        self.assertEqual(orders["zero shift", "cube"], ("AxisReflectionReduction", 8))

    def test_images_outside_their_orbits_keep_the_scatter_per_representation(self) -> None:
        # Orbits of the reflections in y and z, with the reflections in x and
        # y as operations: every representative still has four images, but
        # they are not the rows of its orbit.
        for grid_name in ("half shift", "zero shift"):
            grid = self.grids[grid_name]
            own = AxisReflectionReduction.detect(grid, self.atom_sets["atom on x"])
            self.assertEqual(own.group_order, 4)
            np.testing.assert_array_equal(own.signs[:, 0], 1)
            foreign = AxisReflectionReduction(
                signs=np.ascontiguousarray(own.signs[:, [1, 2, 0]]),
                representative_rows=own.representative_rows,
                full_to_wedge=own.full_to_wedge,
                multiplicities=own.multiplicities,
            )
            images = _reference_representative_images(grid, foreign)
            self.assertFalse(
                all(
                    np.array_equal(own.full_to_wedge[row], np.arange(own.wedge_size))
                    for row in images
                )
            )

            def outcome(build):
                try:
                    return build()
                except RuntimeError as error:
                    return str(error)

            expected = outcome(lambda: _reference_decomposition(grid, foreign))
            with self.subTest(grid=grid_name):
                actual = outcome(
                    lambda: ReflectionRepresentationDecomposition.build(grid, foreign)
                )
                if isinstance(expected, str):
                    self.assertEqual(actual, expected)
                else:
                    for field, values in expected.items():
                        self.assertEqual(getattr(actual, field).tobytes(), values.tobytes())


class FormerRouteSymmetryRowMapTests(SymmetryRowMapTests):
    """The same arrays with ``PARSEC_SYMMETRY_FAST_MAPS=0``, the former routes."""

    fast_maps = "0"


if __name__ == "__main__":
    unittest.main()
