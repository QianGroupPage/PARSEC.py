"""Exactness checks for the conservative Hartree symmetry wedge."""

from __future__ import annotations

import unittest
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from parsec_python.acceleration.Symmetry import (
    AxisReflectionReduction,
    ReflectionRepresentationDecomposition,
    SignedPermutationReduction,
    operator_build_workers,
)
from parsec_python.acceleration.Symmetry.operator_cache import (
    load_or_build_reduced_operators,
)
from parsec_python.acceleration.Symmetry.geometry_cache import (
    load_or_build_reflection_decomposition,
    load_or_detect_reflection_reduction,
)
from parsec_python.acceleration.Eigensolvers import CuPySymmetrySCFEigensolver
from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetryOrbitals
from parsec_python.acceleration.Occupations import CuPySymmetryDensityBuilder
from parsec_python.acceleration.SCF import SymmetrySCFReducer, SymmetryScalarField
from parsec_python.Energy import total_energy
from parsec_python.Grid import build_cluster_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.Mixer import AndersonMixer, potential_residual_metrics
from parsec_python.V_ion import NonlocalProjectorOperator
from parsec_python.models import Atom, GridSettings, MixingSettings


class AxisReflectionReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(
            GridSettings(
                spacing=0.8,
                radius=2.5,
                expansion_order=4,
                shift=(0.5, 0.5, 0.5),
            )
        )
        cls.operator = build_negative_laplacian(cls.grid)
        cls.reduction = AxisReflectionReduction.detect(
            cls.grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )

    def test_half_shifted_sphere_has_eight_equal_images(self) -> None:
        reduction = self.reduction
        self.assertEqual(reduction.group_order, 8)
        self.assertEqual(reduction.full_size, 8 * reduction.wedge_size)
        np.testing.assert_array_equal(reduction.multiplicities, 8)

    def test_normalized_expansion_is_an_isometry(self) -> None:
        generator = np.random.default_rng(7)
        wedge = generator.standard_normal(self.reduction.wedge_size)
        full = self.reduction.expand_vector(wedge)
        np.testing.assert_allclose(
            self.reduction.reduce_vector(full), wedge, rtol=0.0, atol=2.0e-15
        )
        self.assertAlmostEqual(np.dot(full, full), np.dot(wedge, wedge), 13)

    def test_reduced_operator_is_exact_projection(self) -> None:
        reduced = self.reduction.reduce_operator(self.operator)
        generator = np.random.default_rng(11)
        wedge = generator.standard_normal(self.reduction.wedge_size)
        expected = self.reduction.reduce_vector(
            self.operator @ self.reduction.expand_vector(wedge)
        )
        np.testing.assert_allclose(reduced @ wedge, expected, atol=2.0e-13)
        np.testing.assert_allclose(reduced.toarray(), reduced.toarray().T)

    def test_reduced_poisson_solution_matches_full_invariant_solution(self) -> None:
        reduction = self.reduction
        reduced_operator = reduction.reduce_operator(self.operator)
        generator = np.random.default_rng(17)
        raw_rhs = generator.standard_normal(reduction.full_size)
        rhs = reduction.project_invariant(raw_rhs)
        full_solution = spla.spsolve(self.operator, rhs)
        wedge_solution = spla.spsolve(
            reduced_operator, reduction.reduce_vector(rhs)
        )
        expanded = reduction.expand_vector(wedge_solution)
        np.testing.assert_allclose(expanded, full_solution, rtol=2.0e-13, atol=2.0e-13)

    def test_wedge_density_equals_density_of_expanded_representations(self) -> None:
        decomposition = ReflectionRepresentationDecomposition.build(
            self.grid, self.reduction
        )
        generator = np.random.default_rng(23)
        state_count = 7
        representations = np.asarray((0, 2, 5, 1, 7, 3, 4), dtype=np.int32)
        wedge = generator.standard_normal(
            (decomposition.wedge_size, state_count)
        ) / np.sqrt(decomposition.reduction.group_order)
        occupations = generator.random(state_count)
        volume_element = self.grid.volume_element
        orbitals = CuPySymmetryOrbitals(
            scaled_wedge_vectors=wedge,
            representations=representations,
            full_to_wedge=decomposition.reduction.full_to_wedge,
            device_full_to_wedge=decomposition.reduction.full_to_wedge,
            phases=decomposition.phases,
            full_size=decomposition.full_size,
        )

        def density_builder(vectors, weights, volume):
            return (2.0 / volume) * np.sum(
                vectors * vectors * weights[None, :], axis=1
            )

        actual = CuPySymmetryDensityBuilder(density_builder)(
            orbitals, occupations, volume_element
        )
        expanded = np.empty((decomposition.full_size, state_count))
        for column, representation in enumerate(representations):
            expanded[:, column] = (
                wedge[decomposition.reduction.full_to_wedge, column]
                * decomposition.phases[representation]
            )
        expected = density_builder(expanded, occupations, volume_element)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)

    def test_sector_density_equals_packed_wedge_density(self) -> None:
        decomposition = ReflectionRepresentationDecomposition.build(
            self.grid, self.reduction
        )
        generator = np.random.default_rng(231)
        representations = np.asarray((0, 2, 1, 3, 0, 2), dtype=np.int32)
        occupations = generator.random(representations.size)
        volume_element = self.grid.volume_element
        sector_vectors = []
        sector_orbits = []
        sector_scales = []
        representation_columns = np.empty(representations.size, dtype=np.int32)
        packed = np.zeros((decomposition.wedge_size, representations.size))
        for representation in range(decomposition.representation_count):
            output_columns = np.flatnonzero(representations == representation)
            orbits = decomposition.sector_orbit_indices(representation)
            vectors = generator.standard_normal((orbits.size, output_columns.size))
            scales = 1.0 / np.sqrt(
                decomposition.reduction.multiplicities[orbits]
            )
            sector_vectors.append(vectors)
            sector_orbits.append(orbits)
            sector_scales.append(scales)
            representation_columns[output_columns] = np.arange(output_columns.size)
            packed[np.ix_(orbits, output_columns)] = vectors * scales[:, None]

        def density_builder(vectors, weights, volume):
            return (2.0 / volume) * np.sum(
                vectors * vectors * weights[None, :], axis=1
            )

        packed_orbitals = CuPySymmetryOrbitals(
            scaled_wedge_vectors=packed,
            representations=representations,
            full_to_wedge=decomposition.reduction.full_to_wedge,
            device_full_to_wedge=decomposition.reduction.full_to_wedge,
            phases=decomposition.phases,
            full_size=decomposition.full_size,
        )
        sector_orbitals = CuPySymmetryOrbitals(
            scaled_wedge_vectors=None,
            representations=representations,
            full_to_wedge=decomposition.reduction.full_to_wedge,
            device_full_to_wedge=decomposition.reduction.full_to_wedge,
            phases=decomposition.phases,
            full_size=decomposition.full_size,
            representation_columns=representation_columns,
            sector_vectors=tuple(sector_vectors),
            sector_orbits=tuple(sector_orbits),
            sector_scales=tuple(sector_scales),
            wedge_size=decomposition.wedge_size,
        )
        builder = CuPySymmetryDensityBuilder(density_builder)
        expected = builder(packed_orbitals, occupations, volume_element)
        actual = builder(sector_orbitals, occupations, volume_element)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)

    def test_only_atom_preserving_operations_are_retained(self) -> None:
        # Without the partner at -x, reflection in x and inversion are not
        # symmetries.  Reflections in y and z remain valid.
        reduction = AxisReflectionReduction.detect(
            self.grid, (Atom("H", (0.3, 0.0, 0.0)),)
        )
        self.assertEqual(reduction.group_order, 4)
        self.assertTrue(np.all(reduction.signs[:, 0] == 1))

    def test_signed_permutations_find_non_axis_exact_symmetry(self) -> None:
        # Interchanging x and y is an exact symmetry of this labeled dimer,
        # but it is outside the diagonal-sign group.  The generalized detector
        # must use it only because it strictly enlarges the accepted group.
        atoms = (
            Atom("H", (1.0, 0.0, 0.0)),
            Atom("H", (0.0, 1.0, 0.0)),
        )
        axis = AxisReflectionReduction.detect(self.grid, atoms)
        generalized = SignedPermutationReduction.detect(self.grid, atoms)
        self.assertIsInstance(generalized, SignedPermutationReduction)
        self.assertEqual(axis.group_order, 2)
        self.assertEqual(generalized.group_order, 4)
        self.assertLess(generalized.wedge_size, axis.wedge_size)

        reduced = generalized.reduce_operator(self.operator)
        wedge = np.random.default_rng(19).standard_normal(
            generalized.wedge_size
        )
        expected = generalized.reduce_vector(
            self.operator @ generalized.expand_vector(wedge)
        )
        np.testing.assert_allclose(reduced @ wedge, expected, atol=2.0e-13)

    def test_zero_shift_plane_points_use_normalized_unequal_orbits(self) -> None:
        grid = build_cluster_grid(
            GridSettings(
                spacing=1.0,
                radius=2.2,
                expansion_order=2,
                shift=(0.0, 0.0, 0.0),
            )
        )
        operator = build_negative_laplacian(grid)
        reduction = AxisReflectionReduction.detect(
            grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )
        self.assertEqual(reduction.group_order, 8)
        self.assertGreater(np.unique(reduction.multiplicities).size, 1)
        generator = np.random.default_rng(29)
        wedge = generator.standard_normal(reduction.wedge_size)
        full = reduction.expand_vector(wedge)
        np.testing.assert_allclose(reduction.reduce_vector(full), wedge)
        reduced = reduction.reduce_operator(operator)
        np.testing.assert_allclose(
            reduced @ wedge,
            reduction.reduce_vector(operator @ full),
            atol=2.0e-13,
        )

    def test_geometry_and_phase_caches_are_exact_and_content_addressed(self) -> None:
        directory = Path.cwd() / ".tmp" / f"geometry-cache-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        atoms = (Atom("H", (0.0, 0.0, 0.0)),)
        try:
            first, first_info = load_or_detect_reflection_reduction(
                self.grid, atoms, cache_directory=directory
            )
            second, second_info = load_or_detect_reflection_reduction(
                self.grid, atoms, cache_directory=directory
            )
            self.assertEqual(first_info.status, "miss-written")
            self.assertEqual(second_info.status, "hit")
            self.assertEqual(first_info.key, second_info.key)
            np.testing.assert_array_equal(first.signs, second.signs)
            np.testing.assert_array_equal(
                first.representative_rows, second.representative_rows
            )
            np.testing.assert_array_equal(first.full_to_wedge, second.full_to_wedge)
            np.testing.assert_array_equal(first.multiplicities, second.multiplicities)

            first_decomposition, first_phase_info = (
                load_or_build_reflection_decomposition(
                    self.grid,
                    first,
                    reduction_key=first_info.key,
                    cache_directory=directory,
                )
            )
            second_decomposition, second_phase_info = (
                load_or_build_reflection_decomposition(
                    self.grid,
                    second,
                    reduction_key=second_info.key,
                    cache_directory=directory,
                )
            )
            self.assertEqual(first_phase_info.status, "miss-written")
            self.assertEqual(second_phase_info.status, "hit")
            np.testing.assert_array_equal(
                first_decomposition.characters,
                second_decomposition.characters,
            )
            np.testing.assert_array_equal(
                first_decomposition.phases,
                second_decomposition.phases,
            )

            changed, changed_info = load_or_detect_reflection_reduction(
                self.grid,
                (Atom("H", (0.1, 0.0, 0.0)),),
                cache_directory=directory,
            )
            self.assertNotEqual(changed_info.key, first_info.key)
            self.assertEqual(changed_info.status, "miss-written")
            self.assertEqual(changed.group_order, 4)
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()

    def test_wedge_scf_algebra_matches_full_invariant_fields(self) -> None:
        reducer = SymmetrySCFReducer(self.reduction)
        generator = np.random.default_rng(31)

        def invariant() -> np.ndarray:
            return reducer.expand_values(
                generator.standard_normal(self.reduction.wedge_size)
            )

        input_potential = invariant()
        output_potential = invariant()
        density = np.abs(invariant()) + 0.1
        reference_metrics = potential_residual_metrics(
            input_potential, output_potential, density, 0.125, 10.0
        )
        reduced_metrics = reducer.potential_residual_metrics(
            input_potential, output_potential, density, 0.125, 10.0
        )
        self.assertAlmostEqual(reference_metrics.plain, reduced_metrics.plain, 13)
        self.assertAlmostEqual(
            reference_metrics.weighted, reduced_metrics.weighted, 13
        )
        np.testing.assert_allclose(
            reference_metrics.residual,
            reduced_metrics.residual,
            rtol=0.0,
            atol=5.0e-15,
        )

        settings = MixingSettings(
            parameter=0.37,
            memory=3,
            restart=6,
            safeguard=True,
        )
        full_mixer = AndersonMixer(settings)
        wedge_mixer = reducer.mixer(settings)
        current_full = input_potential
        current_wedge = input_potential
        for iteration in range(1, 6):
            target = invariant()
            current_full = full_mixer.mix(
                current_full, target, iteration=iteration
            )
            current_wedge = wedge_mixer.mix(
                current_wedge, target, iteration=iteration
            )
            np.testing.assert_allclose(
                current_wedge, current_full, rtol=2.0e-13, atol=2.0e-13
            )

        eigenvalues = np.sort(generator.standard_normal(5))
        occupations = generator.random(5)
        ionic = invariant()
        hartree = invariant()
        xc = invariant()
        full_energy = total_energy(
            eigenvalues,
            occupations,
            density,
            input_potential,
            ionic,
            hartree,
            xc,
            -3.2,
            4.1,
            0.125,
        )
        wedge_energy = reducer.total_energy(
            eigenvalues,
            occupations,
            density,
            input_potential,
            ionic,
            hartree,
            xc,
            -3.2,
            4.1,
            0.125,
        )
        for field in full_energy.__dataclass_fields__:
            self.assertAlmostEqual(
                getattr(full_energy, field), getattr(wedge_energy, field), 12
            )

    def test_compact_scalar_fields_stay_compact_and_match_full_algebra(self) -> None:
        reducer = SymmetrySCFReducer(self.reduction)
        generator = np.random.default_rng(37)

        def compact(*, positive: bool = False) -> SymmetryScalarField:
            values = generator.standard_normal(self.reduction.wedge_size)
            if positive:
                values = np.abs(values) + 0.2
            return reducer.field(values)

        input_potential = compact()
        output_potential = compact()
        density = compact(positive=True)
        compact_metrics = reducer.potential_residual_metrics(
            input_potential, output_potential, density, 0.125, 10.0
        )
        full_metrics = potential_residual_metrics(
            reducer.to_full(input_potential),
            reducer.to_full(output_potential),
            reducer.to_full(density),
            0.125,
            10.0,
        )
        self.assertAlmostEqual(compact_metrics.plain, full_metrics.plain, 13)
        self.assertAlmostEqual(compact_metrics.weighted, full_metrics.weighted, 13)

        mixer = reducer.mixer(MixingSettings(parameter=0.41, memory=3, restart=6))
        mixed = mixer.mix(input_potential, output_potential, iteration=1)
        self.assertIsInstance(mixed, SymmetryScalarField)
        np.testing.assert_allclose(
            reducer.to_full(mixed),
            reducer.to_full(input_potential)
            + 0.41
            * (
                reducer.to_full(output_potential)
                - reducer.to_full(input_potential)
            ),
            atol=2.0e-15,
        )

        eigenvalues = np.sort(generator.standard_normal(5))
        occupations = generator.random(5)
        ionic, hartree, xc = compact(), compact(), compact()
        compact_energy = reducer.total_energy(
            eigenvalues,
            occupations,
            density,
            input_potential,
            ionic,
            hartree,
            xc,
            -3.2,
            4.1,
            0.125,
        )
        full_energy = total_energy(
            eigenvalues,
            occupations,
            reducer.to_full(density),
            reducer.to_full(input_potential),
            reducer.to_full(ionic),
            reducer.to_full(hartree),
            reducer.to_full(xc),
            -3.2,
            4.1,
            0.125,
        )
        for name in full_energy.__dataclass_fields__:
            self.assertAlmostEqual(
                getattr(compact_energy, name), getattr(full_energy, name), 12
            )


class ReflectionRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(
            GridSettings(
                spacing=0.8,
                radius=2.5,
                expansion_order=4,
                shift=(0.5, 0.5, 0.5),
            )
        )
        cls.operator = build_negative_laplacian(cls.grid)
        cls.reduction = AxisReflectionReduction.detect(
            cls.grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )
        cls.decomposition = ReflectionRepresentationDecomposition.build(
            cls.grid, cls.reduction
        )

    def test_characters_are_complete_and_orthogonal(self) -> None:
        decomposition = self.decomposition
        self.assertEqual(decomposition.representation_count, 8)
        gram = decomposition.characters @ decomposition.characters.T
        np.testing.assert_array_equal(gram, 8 * np.eye(8, dtype=np.int8))

    def test_cold_operator_worker_policy_is_size_adaptive(self) -> None:
        original = os.environ.pop("PARSEC_SYMMETRY_OPERATOR_WORKERS", None)
        try:
            self.assertEqual(operator_build_workers(2), 1)
            self.assertEqual(operator_build_workers(8), 4)
            os.environ["PARSEC_SYMMETRY_OPERATOR_WORKERS"] = "3"
            self.assertEqual(operator_build_workers(2), 2)
            self.assertEqual(operator_build_workers(8), 3)
        finally:
            if original is None:
                os.environ.pop("PARSEC_SYMMETRY_OPERATOR_WORKERS", None)
            else:
                os.environ["PARSEC_SYMMETRY_OPERATOR_WORKERS"] = original

    def test_each_reduced_operator_matches_full_projected_action(self) -> None:
        generator = np.random.default_rng(41)
        batched = self.decomposition.reduce_operators(self.operator)
        for representation in range(self.decomposition.representation_count):
            with self.subTest(representation=representation):
                wedge = generator.standard_normal(self.decomposition.wedge_size)
                full = self.decomposition.expand_vector(wedge, representation)
                expected = self.decomposition.reduce_vector(
                    self.operator @ full, representation
                )
                reduced = self.decomposition.reduce_operator(
                    self.operator, representation
                )
                np.testing.assert_array_equal(
                    batched[representation].indptr, reduced.indptr
                )
                np.testing.assert_array_equal(
                    batched[representation].indices, reduced.indices
                )
                np.testing.assert_array_equal(
                    batched[representation].data, reduced.data
                )
                np.testing.assert_allclose(reduced @ wedge, expected, atol=2.0e-13)

    def test_sector_spectra_reconstruct_full_invariant_operator(self) -> None:
        radius_squared = np.einsum(
            "ij,ij->i", self.grid.coordinates, self.grid.coordinates
        )
        local = 0.03 * radius_squared
        full = self.operator.toarray() + np.diag(local)
        sector_values: list[np.ndarray] = []
        wedge_local = self.decomposition.invariant_wedge_values(local)
        for representation in range(self.decomposition.representation_count):
            reduced = self.decomposition.reduce_operator(
                self.operator, representation
            ).toarray()
            sector_values.append(np.linalg.eigvalsh(reduced + np.diag(wedge_local)))
        np.testing.assert_allclose(
            np.sort(np.concatenate(sector_values)),
            np.linalg.eigvalsh(full),
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def test_nonlocal_factors_match_full_projected_action(self) -> None:
        generator = np.random.default_rng(53)
        projectors = sp.csc_matrix(
            generator.standard_normal((self.grid.size, 5))
        )
        operator = NonlocalProjectorOperator(
            projectors=projectors,
            signs=np.asarray((1.0, -1.0, 1.0, -1.0, 1.0)),
            labels=tuple((0, 0, index) for index in range(5)),
        )
        batched = self.decomposition.reduce_nonlocal_operators(operator)
        for representation in range(self.decomposition.representation_count):
            with self.subTest(representation=representation):
                wedge = generator.standard_normal(self.decomposition.wedge_size)
                full = self.decomposition.expand_vector(wedge, representation)
                expected = self.decomposition.reduce_vector(
                    operator.apply(full), representation
                )
                reduced = self.decomposition.reduce_nonlocal_operator(
                    operator, representation
                )
                batched_projectors = batched[representation].projectors
                np.testing.assert_array_equal(
                    batched_projectors.indptr, reduced.projectors.indptr
                )
                np.testing.assert_array_equal(
                    batched_projectors.indices, reduced.projectors.indices
                )
                np.testing.assert_array_equal(
                    batched_projectors.data, reduced.projectors.data
                )
                np.testing.assert_allclose(
                    reduced.apply(wedge), expected, atol=3.0e-13
                )

    def test_reduced_operator_cache_is_exact_and_content_addressed(self) -> None:
        generator = np.random.default_rng(59)
        projectors = sp.csc_matrix(
            generator.standard_normal((self.grid.size, 3))
        )
        nonlocal_operator = NonlocalProjectorOperator(
            projectors=projectors,
            signs=np.asarray((1.0, -1.0, 1.0)),
            labels=((0, 0, 0), (0, 0, 1), (0, 0, 2)),
        )
        directory = Path.cwd() / ".tmp" / f"operator-cache-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            first = load_or_build_reduced_operators(
                self.decomposition,
                self.operator,
                nonlocal_operator,
                cache_directory=directory,
            )
            second = load_or_build_reduced_operators(
                self.decomposition,
                self.operator,
                nonlocal_operator,
                cache_directory=directory,
            )

            self.assertEqual(first.cache_info.status, "miss-written")
            self.assertEqual(second.cache_info.status, "hit")
            self.assertEqual(first.cache_info.key, second.cache_info.key)
            expected = self.decomposition.reduce_operators(self.operator)
            for index, (left, right) in enumerate(
                zip(
                    first.stencil_metadata,
                    second.stencil_metadata,
                    strict=True,
                )
            ):
                np.testing.assert_array_equal(left.neighbors, right.neighbors)
                np.testing.assert_array_equal(
                    left.coefficient_codes, right.coefficient_codes
                )
                np.testing.assert_array_equal(
                    left.coefficient_palette, right.coefficient_palette
                )
                reconstructed = left.to_csr()
                np.testing.assert_array_equal(
                    reconstructed.indptr, expected[index].indptr
                )
                np.testing.assert_array_equal(
                    reconstructed.indices, expected[index].indices
                )
                np.testing.assert_array_equal(
                    reconstructed.data, expected[index].data
                )
            for left, right in zip(
                first.nonlocal_operators,
                second.nonlocal_operators,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    left.projectors.data, right.projectors.data
                )

            changed = self.operator.copy()
            changed.data *= np.nextafter(1.0, 2.0)
            changed_bundle = load_or_build_reduced_operators(
                self.decomposition,
                changed,
                nonlocal_operator,
                cache_directory=directory,
            )
            self.assertNotEqual(
                changed_bundle.cache_info.key, first.cache_info.key
            )
            self.assertEqual(changed_bundle.cache_info.status, "miss-written")
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()

    def test_zero_shift_stabilizers_give_complete_variable_sectors(self) -> None:
        grid = build_cluster_grid(
            GridSettings(
                spacing=1.0,
                radius=2.2,
                expansion_order=2,
                shift=(0.0, 0.0, 0.0),
            )
        )
        reduction = AxisReflectionReduction.detect(
            grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )
        decomposition = ReflectionRepresentationDecomposition.build(
            grid, reduction
        )
        self.assertEqual(sum(decomposition.sector_sizes), grid.size)
        self.assertGreater(
            max(decomposition.sector_sizes), min(decomposition.sector_sizes)
        )

        full_values = np.linalg.eigvalsh(build_negative_laplacian(grid).toarray())
        sector_values = []
        operator = build_negative_laplacian(grid)
        generator = np.random.default_rng(20260820)
        for representation, sector_size in enumerate(
            decomposition.sector_sizes
        ):
            reduced = decomposition.reduce_operator(
                operator, representation
            )
            vector = generator.standard_normal(sector_size)
            full = decomposition.expand_vector(vector, representation)
            expected = decomposition.reduce_vector(
                operator @ full, representation
            )
            np.testing.assert_allclose(
                reduced @ vector, expected, rtol=2.0e-13, atol=2.0e-13
            )
            sector_values.append(np.linalg.eigvalsh(reduced.toarray()))
        np.testing.assert_allclose(
            np.sort(np.concatenate(sector_values)),
            full_values,
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def test_parsec_global_sort_trims_the_naphthalene_sector_counts(self) -> None:
        # Representation sequence of the lowest Nstate-1=29 values in the
        # recorded PARSEC naphthalene result.  eigen_sort adds at most nadd=6
        # later states per representation, yielding PARSEC's reported active
        # counts 9,8,8,9,7,9,9,9.
        first_representations = np.asarray(
            (
                1, 7, 6, 1, 4, 7, 6, 1, 7, 4,
                1, 1, 6, 7, 4, 8, 6, 7, 2, 4,
                1, 3, 8, 5, 2, 3, 8, 1, 7,
            ),
            dtype=np.int32,
        ) - 1
        results = []
        for representation in range(8):
            values = np.flatnonzero(
                first_representations == representation
            ).astype(np.float64)
            padding = 100.0 + representation + np.arange(9 - values.size) / 10.0
            results.append(
                SimpleNamespace(eigenvalues=np.concatenate((values, padding)))
            )

        class RecordingSolver:
            truncated_to: int | None = None

            def truncate_state(self, count: int) -> None:
                self.truncated_to = count

        wrapper = object.__new__(CuPySymmetrySCFEigensolver)
        wrapper.decomposition = SimpleNamespace(representation_count=8)
        wrapper._solvers = [RecordingSolver() for _ in range(8)]
        counts = [9] * 8
        wrapper._trim_sector_states_like_parsec(
            results, counts, requested_states=30, safety_buffer=6
        )

        self.assertEqual(counts, [9, 8, 8, 9, 7, 9, 9, 9])
        self.assertEqual(
            [solver.truncated_to for solver in wrapper._solvers],
            [None, 8, 8, None, 7, None, None, None],
        )

if __name__ == "__main__":
    unittest.main()
