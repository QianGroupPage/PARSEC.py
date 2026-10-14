"""Correctness tests for exact-key startup shortcuts."""

from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.Laplacian import (
    DeferredNativeNegativeLaplacian,
)
from parsec_python.acceleration.Symmetry import (
    AxisReflectionReduction,
    ReflectionRepresentationDecomposition,
)
from parsec_python.acceleration.Symmetry.geometry_cache import (
    load_or_build_reflection_decomposition,
    load_or_detect_reflection_reduction,
)
from parsec_python.acceleration.Symmetry.operator_cache import (
    _MEMORY_BUNDLES,
    _MEMORY_BUNDLES_LOCK,
    load_or_build_reduced_operators,
)
from parsec_python.acceleration.backends.native import native_available
from parsec_python.Grid import build_cluster_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.V_ion import NonlocalProjectorOperator
from parsec_python.models import Atom, GridSettings


class DeferredFiniteDifferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = build_cluster_grid(
            GridSettings(
                spacing=0.8,
                radius=2.8,
                expansion_order=4,
                shift=(0.5, 0.5, 0.5),
            )
        )

    def test_descriptor_count_and_key_match_discrete_operator_inputs(self) -> None:
        descriptor = DeferredNativeNegativeLaplacian(self.grid)
        reference = build_negative_laplacian(self.grid)
        repeated = DeferredNativeNegativeLaplacian(self.grid)

        self.assertEqual(descriptor.shape, reference.shape)
        self.assertEqual(descriptor.nnz, reference.nnz)
        self.assertEqual(descriptor.cache_key, repeated.cache_key)
        self.assertFalse(descriptor.materialized)

        changed_grid = build_cluster_grid(
            GridSettings(
                spacing=0.75,
                radius=2.8,
                expansion_order=4,
                shift=(0.5, 0.5, 0.5),
            )
        )
        changed = DeferredNativeNegativeLaplacian(changed_grid)
        self.assertNotEqual(descriptor.cache_key, changed.cache_key)

    def test_descriptor_reuses_exact_nnz_cache(self) -> None:
        directory = Path.cwd() / ".tmp" / f"nnz-cache-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            first = DeferredNativeNegativeLaplacian(
                self.grid,
                cache_directory=directory,
            )
            self.assertEqual(first.nnz_cache_status, "miss-written")
            self.assertIsNotNone(first.nnz_cache_path)
            self.assertTrue(first.nnz_cache_path.is_file())

            second = DeferredNativeNegativeLaplacian(
                self.grid,
                cache_directory=directory,
            )
            self.assertEqual(second.nnz, first.nnz)
            self.assertEqual(second.cache_key, first.cache_key)
            self.assertEqual(second.nnz_cache_status, "memory-hit")
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()

    @unittest.skipUnless(
        native_available(),
        "parsec_accelerated_native has not been built",
    )
    def test_materialization_is_exact_and_memoized(self) -> None:
        descriptor = DeferredNativeNegativeLaplacian(self.grid)
        expected = build_negative_laplacian(self.grid)

        first = descriptor.materialize()
        second = descriptor.materialize()

        self.assertIs(first, second)
        self.assertTrue(descriptor.materialized)
        np.testing.assert_array_equal(first.indptr, expected.indptr)
        np.testing.assert_array_equal(first.indices, expected.indices)
        np.testing.assert_array_equal(first.data, expected.data)

    def test_exact_seed_cache_hit_never_materializes_full_operator(self) -> None:
        reduction = AxisReflectionReduction.detect(
            self.grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )
        decomposition = ReflectionRepresentationDecomposition.build(
            self.grid, reduction
        )
        full = build_negative_laplacian(self.grid)
        values = np.linspace(-0.3, 0.7, self.grid.size)
        projectors = sp.csc_matrix(values[:, None])
        nonlocal_operator = NonlocalProjectorOperator(
            projectors=projectors,
            signs=np.asarray((1.0,), dtype=np.float64),
            labels=((0, 0, 0),),
        )
        directory = Path.cwd() / ".tmp" / f"startup-cache-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            first = load_or_build_reduced_operators(
                decomposition,
                full,
                nonlocal_operator,
                cache_directory=directory,
                kinetic_key_seed="a" * 64,
                decomposition_key_seed="b" * 64,
            )
            # An arbitrary sentinel cannot be converted to CSR.  A successful
            # second call proves the exact-key hit was accepted before any
            # full-grid materialization/canonicalization was attempted.
            sentinel = object()
            second = load_or_build_reduced_operators(
                decomposition,
                sentinel,
                nonlocal_operator,
                cache_directory=directory,
                kinetic_key_seed="a" * 64,
                decomposition_key_seed="b" * 64,
            )

            self.assertEqual(first.cache_info.status, "miss-written")
            self.assertEqual(second.cache_info.status, "hit")
            for left, right in zip(
                first.stencil_metadata,
                second.stencil_metadata,
                strict=True,
            ):
                np.testing.assert_array_equal(left.neighbors, right.neighbors)
                np.testing.assert_array_equal(
                    left.coefficient_codes, right.coefficient_codes
                )
                np.testing.assert_array_equal(
                    left.coefficient_palette, right.coefficient_palette
                )
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()

    def test_resident_operator_bundle_uses_bounded_memory_hit(self) -> None:
        reduction = AxisReflectionReduction.detect(
            self.grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )
        decomposition = ReflectionRepresentationDecomposition.build(
            self.grid, reduction
        )
        full = build_negative_laplacian(self.grid)
        values = np.linspace(-0.3, 0.7, self.grid.size)
        nonlocal_operator = NonlocalProjectorOperator(
            projectors=sp.csc_matrix(values[:, None]),
            signs=np.asarray((1.0,), dtype=np.float64),
            labels=((0, 0, 0),),
        )
        directory = Path.cwd() / ".tmp" / f"operator-memory-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "PARSEC_ACCELERATED_RESIDENT": "1",
                        "PARSEC_RESIDENT_OPERATOR_CACHE_SIZE": "1",
                    },
                ),
                _MEMORY_BUNDLES_LOCK,
            ):
                _MEMORY_BUNDLES.clear()
            with patch.dict(
                os.environ,
                {
                    "PARSEC_ACCELERATED_RESIDENT": "1",
                    "PARSEC_RESIDENT_OPERATOR_CACHE_SIZE": "1",
                },
            ):
                first = load_or_build_reduced_operators(
                    decomposition,
                    full,
                    nonlocal_operator,
                    cache_directory=directory,
                    kinetic_key_seed="c" * 64,
                    decomposition_key_seed="d" * 64,
                )
                second = load_or_build_reduced_operators(
                    decomposition,
                    object(),
                    nonlocal_operator,
                    cache_directory=directory,
                    kinetic_key_seed="c" * 64,
                    decomposition_key_seed="d" * 64,
                )
            self.assertEqual(first.cache_info.status, "miss-written")
            self.assertEqual(second.cache_info.status, "memory-hit")
            self.assertIs(second.stencil_metadata, first.stencil_metadata)
            self.assertIs(second.nonlocal_operators, first.nonlocal_operators)
        finally:
            with _MEMORY_BUNDLES_LOCK:
                _MEMORY_BUNDLES.clear()
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()


class DisabledCacheKeyTests(unittest.TestCase):
    """A key that no cache can look up is not hashed; the objects do not change."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(
            GridSettings(
                spacing=0.8,
                radius=2.8,
                expansion_order=4,
                shift=(0.5, 0.5, 0.5),
            )
        )
        cls.atoms = (Atom("H", (0.0, 0.0, 0.0)),)

    @staticmethod
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("a key nothing looks up was hashed")

    def test_symmetry_keys_are_hashed_only_for_a_cache_directory(self) -> None:
        from parsec_python.acceleration.Symmetry import geometry_cache

        with (
            patch.object(geometry_cache, "_geometry_key", self._forbidden),
            patch.object(geometry_cache, "_representation_key", self._forbidden),
        ):
            reduction, reduction_info = load_or_detect_reflection_reduction(
                self.grid, self.atoms, cache_directory=None
            )
            decomposition, phase_info = load_or_build_reflection_decomposition(
                self.grid,
                reduction,
                reduction_key=reduction_info.key,
                cache_directory=None,
            )
        for info in (reduction_info, phase_info):
            self.assertIsNone(info.key)
            self.assertIsNone(info.path)
            self.assertEqual(info.status, "disabled-built")
            self.assertEqual(info.hash_seconds, 0.0)

        directory = Path.cwd() / ".tmp" / f"disabled-key-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            hashed, hashed_info = load_or_detect_reflection_reduction(
                self.grid, self.atoms, cache_directory=directory
            )
            hashed_decomposition, hashed_phase_info = (
                load_or_build_reflection_decomposition(
                    self.grid,
                    hashed,
                    reduction_key=hashed_info.key,
                    cache_directory=directory,
                )
            )
            # A cache entry cannot be addressed without its upstream key.
            with self.assertRaises(ValueError):
                load_or_build_reflection_decomposition(
                    self.grid,
                    hashed,
                    reduction_key=None,
                    cache_directory=directory,
                )
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()
        for info in (hashed_info, hashed_phase_info):
            self.assertEqual(len(info.key), 64)
            self.assertEqual(info.status, "miss-written")
        self.assertIs(type(reduction), type(hashed))
        for field in fields(reduction):
            np.testing.assert_array_equal(
                getattr(reduction, field.name), getattr(hashed, field.name)
            )
        for name in ("characters", "phases", "orbit_to_sector"):
            np.testing.assert_array_equal(
                getattr(decomposition, name), getattr(hashed_decomposition, name)
            )

    def test_operator_key_is_hashed_only_for_a_disk_or_resident_cache(self) -> None:
        from parsec_python.acceleration.Symmetry import operator_cache

        reduction = AxisReflectionReduction.detect(self.grid, self.atoms)
        decomposition = ReflectionRepresentationDecomposition.build(
            self.grid, reduction
        )
        full = build_negative_laplacian(self.grid)
        values = np.linspace(-0.3, 0.7, self.grid.size)
        nonlocal_operator = NonlocalProjectorOperator(
            projectors=sp.csc_matrix(values[:, None]),
            signs=np.asarray((1.0,), dtype=np.float64),
            labels=((0, 0, 0),),
        )
        with (
            patch.dict(os.environ, {"PARSEC_ACCELERATED_RESIDENT": "0"}),
            patch.object(operator_cache, "_cache_key", self._forbidden),
        ):
            plain = load_or_build_reduced_operators(
                decomposition, full, nonlocal_operator, cache_directory=None
            )
            partial = load_or_build_reduced_operators(
                decomposition,
                full,
                nonlocal_operator,
                cache_directory=None,
                representations=(0, 3),
            )
        self.assertIsNone(plain.cache_info.key)
        self.assertIsNone(plain.cache_info.path)
        self.assertEqual(plain.cache_info.status, "disabled")
        self.assertIsNone(partial.cache_info.key)
        self.assertEqual(partial.cache_info.status, "disabled-partial")
        # A key that was not hashed took no hashing time.
        for bundle in (plain, partial):
            self.assertEqual(bundle.cache_info.hash_seconds, 0.0)

        # The operators themselves are those of the direct projection.
        kinetic = decomposition.reduce_operators(full)
        projectors = decomposition.reduce_nonlocal_operators(nonlocal_operator)
        for index, (stencil, reduced) in enumerate(
            zip(plain.stencil_metadata, plain.nonlocal_operators, strict=True)
        ):
            reconstructed = stencil.to_csr()
            np.testing.assert_array_equal(reconstructed.indptr, kinetic[index].indptr)
            np.testing.assert_array_equal(reconstructed.indices, kinetic[index].indices)
            np.testing.assert_array_equal(reconstructed.data, kinetic[index].data)
            expected = projectors[index].projectors
            np.testing.assert_array_equal(reduced.projectors.indptr, expected.indptr)
            np.testing.assert_array_equal(reduced.projectors.indices, expected.indices)
            np.testing.assert_array_equal(reduced.projectors.data, expected.data)

        # The resident bundle cache is addressed by the key.  No upstream key
        # exists without a cache directory, so every buffer is hashed here and
        # a different operator can never be answered from memory.
        changed = full.copy()
        changed.data *= np.nextafter(1.0, 2.0)
        with _MEMORY_BUNDLES_LOCK:
            _MEMORY_BUNDLES.clear()
        try:
            with patch.dict(
                os.environ,
                {
                    "PARSEC_ACCELERATED_RESIDENT": "1",
                    "PARSEC_RESIDENT_OPERATOR_CACHE_SIZE": "2",
                },
            ):
                first = load_or_build_reduced_operators(
                    decomposition, full, nonlocal_operator, cache_directory=None
                )
                second = load_or_build_reduced_operators(
                    decomposition, full, nonlocal_operator, cache_directory=None
                )
                third = load_or_build_reduced_operators(
                    decomposition, changed, nonlocal_operator, cache_directory=None
                )
        finally:
            with _MEMORY_BUNDLES_LOCK:
                _MEMORY_BUNDLES.clear()
        self.assertEqual(len(first.cache_info.key), 64)
        self.assertGreater(first.cache_info.hash_seconds, 0.0)
        self.assertEqual(first.cache_info.status, "disabled")
        self.assertEqual(second.cache_info.status, "memory-hit")
        self.assertEqual(second.cache_info.key, first.cache_info.key)
        self.assertIs(second.stencil_metadata, first.stencil_metadata)
        self.assertEqual(third.cache_info.status, "disabled")
        self.assertNotEqual(third.cache_info.key, first.cache_info.key)


class CoefficientPaletteTests(unittest.TestCase):
    """The search-based palette must equal the former full sort exactly."""

    @staticmethod
    def _sorted_reference(values):
        bits, inverse = np.unique(
            np.ascontiguousarray(values, dtype=np.float64).view(np.uint64).ravel(),
            return_inverse=True,
        )
        return bits, inverse.astype(np.uint8)

    def _assert_same(self, values) -> None:
        from parsec_python.acceleration.backends.cupy_compact import _coefficient_palette

        palette, codes = _coefficient_palette(values)
        bits, inverse = self._sorted_reference(values)
        np.testing.assert_array_equal(palette.view(np.uint64), bits)
        np.testing.assert_array_equal(codes.ravel(), inverse.ravel())
        self.assertEqual(codes.dtype, np.uint8)
        self.assertEqual(codes.shape, np.shape(values))

    def test_palette_and_codes_match_the_sorted_encoding(self) -> None:
        generator = np.random.default_rng(71)
        coefficients = np.concatenate(
            (generator.standard_normal(40), (0.0, -0.0, 1.0, -1.0))
        )
        self._assert_same(coefficients[generator.integers(0, 44, 3_000_000)])
        self._assert_same(np.array((0.0, -0.0, 0.0, -0.0, 2.0)))
        self._assert_same(np.full(7, 3.25))
        self._assert_same(np.arange(256.0)[generator.integers(0, 256, 200_000)])
        self._assert_same(coefficients[generator.integers(0, 44, (500, 25))])

    def test_values_missed_by_the_first_sample_are_still_encoded(self) -> None:
        generator = np.random.default_rng(73)
        values = generator.standard_normal(20)[generator.integers(0, 20, 5_000_000)]
        values[4_999_999] = 123.456
        values[3_333_333] = -7.5e-300
        values[17] = np.nextafter(1.0, 2.0)
        self._assert_same(values)

    def test_more_than_256_coefficients_are_rejected(self) -> None:
        from parsec_python.acceleration.backends.cupy_compact import _coefficient_palette

        self.assertIsNone(_coefficient_palette(np.arange(257.0)))
        hidden = np.concatenate((np.zeros(3_000_000), np.arange(1.0, 400.0)))
        self.assertIsNone(_coefficient_palette(hidden))
        palette, codes = _coefficient_palette(np.empty(0))
        self.assertEqual((palette.size, codes.size), (0, 0))


if __name__ == "__main__":
    unittest.main()
