"""End-to-end parity for the accelerated multipole/SciPy Poisson path."""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import parsec_python.acceleration.Hartree.native_boundary as native_boundary_module
from parsec_python.acceleration.Hartree.native_boundary import (
    NativeMultipoleBoundaryBuilder,
    NativeSymmetryMultipoleBoundaryBuilder,
)
from parsec_python.acceleration.Hartree.poisson import build_hartree_problem
from parsec_python.acceleration.Hartree.poisson import solve_scipy_hartree
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.Symmetry import AxisReflectionReduction
from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree import solve_hartree
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.models import Atom, GridSettings, HartreeSettings


class FastHartreeTests(unittest.TestCase):
    def test_sphere_solution_matches_reference_boundary_and_cg(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=0.65, radius=3.5, expansion_order=4)
        )
        operator = build_negative_laplacian(grid)
        radius_squared = np.sum(grid.coordinates * grid.coordinates, axis=1)
        density = np.exp(-0.7 * radius_squared) * (
            1.0 + 0.04 * grid.coordinates[:, 0]
        )
        density *= 2.0 / grid.integrate(density)
        settings = HartreeSettings(
            boundary_method="multipole",
            multipole_order=9,
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            max_iterations=1000,
        )
        reference = solve_hartree(density, grid, operator, settings)
        accelerated = solve_scipy_hartree(density, grid, operator, settings)
        np.testing.assert_allclose(
            accelerated.right_hand_side,
            reference.right_hand_side,
            rtol=3.0e-13,
            atol=3.0e-13,
        )
        np.testing.assert_allclose(
            accelerated.potential,
            reference.potential,
            rtol=3.0e-11,
            atol=3.0e-11,
        )
        self.assertEqual(accelerated.iterations, reference.iterations)
        self.assertEqual(
            accelerated.matrix_vector_products,
            reference.matrix_vector_products,
        )

    @unittest.skipUnless(
        native_available(),
        "parsec_accelerated_native has not been built",
    )
    def test_native_multipole_rhs_matches_python_boundary(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=0.65, radius=3.5, expansion_order=4)
        )
        radius_squared = np.sum(grid.coordinates * grid.coordinates, axis=1)
        density = np.exp(-0.7 * radius_squared) * (
            1.0 + 0.04 * grid.coordinates[:, 0]
        )
        density *= 2.0 / grid.integrate(density)
        settings = HartreeSettings(
            boundary_method="multipole",
            multipole_order=9,
        )

        reference_rhs, reference_boundary = build_hartree_problem(
            density, grid, settings
        )
        builder = NativeMultipoleBoundaryBuilder(grid, settings.multipole_order)
        native_rhs, native_boundary = builder.build(density)

        self.assertGreater(builder.boundary_term_count, 0)
        np.testing.assert_allclose(
            native_rhs,
            reference_rhs,
            rtol=3.0e-13,
            atol=3.0e-13,
        )
        for key, expected in reference_boundary.moments.items():
            self.assertAlmostEqual(native_boundary.moments[key], expected, places=12)
        exterior_points = 1.2 * grid.coordinates[:37]
        np.testing.assert_allclose(
            native_boundary.potential(exterior_points),
            reference_boundary.potential(exterior_points),
            rtol=3.0e-13,
            atol=3.0e-13,
        )

    @unittest.skipUnless(
        native_available(),
        "parsec_accelerated_native has not been built",
    )
    def test_native_symmetry_builder_returns_exact_projected_rhs(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=0.65, radius=3.5, expansion_order=4)
        )
        reduction = AxisReflectionReduction.detect(
            grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )
        radius_squared = np.sum(grid.coordinates * grid.coordinates, axis=1)
        density = np.exp(-0.7 * radius_squared)
        density *= 2.0 / grid.integrate(density)
        settings = HartreeSettings(
            boundary_method="multipole",
            multipole_order=9,
        )
        full_builder = NativeMultipoleBoundaryBuilder(
            grid, settings.multipole_order
        )
        full_rhs, full_boundary = full_builder.build(density)
        wedge_builder = NativeSymmetryMultipoleBoundaryBuilder(
            grid, reduction, settings.multipole_order
        )
        wedge_rhs, wedge_boundary = wedge_builder.build_reduced(density)

        np.testing.assert_allclose(
            wedge_rhs,
            reduction.reduce_vector(full_rhs),
            rtol=8.0e-13,
            atol=8.0e-13,
        )
        for key, expected in full_boundary.moments.items():
            self.assertAlmostEqual(wedge_boundary.moments[key], expected, places=11)

    @unittest.skipUnless(
        native_available(),
        "parsec_accelerated_native has not been built",
    )
    def test_native_symmetry_rhs_handles_reflection_plane_orbits(self) -> None:
        grid = build_cluster_grid(
            GridSettings(
                spacing=0.8,
                radius=3.2,
                expansion_order=4,
                shift=(0.0, 0.0, 0.0),
            )
        )
        reduction = AxisReflectionReduction.detect(
            grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )
        self.assertGreater(np.unique(reduction.multiplicities).size, 1)
        radius_squared = np.sum(grid.coordinates * grid.coordinates, axis=1)
        density = np.exp(-0.8 * radius_squared)
        density *= 2.0 / grid.integrate(density)
        full_rhs, _ = NativeMultipoleBoundaryBuilder(grid, 9).build(density)
        wedge_rhs, _ = NativeSymmetryMultipoleBoundaryBuilder(
            grid, reduction, 9
        ).build_reduced(density)
        np.testing.assert_allclose(
            wedge_rhs,
            reduction.reduce_vector(full_rhs),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    @unittest.skipUnless(
        native_available(),
        "parsec_accelerated_native has not been built",
    )
    def test_native_symmetry_geometry_cache_reloads_exact_buffers(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=0.7, radius=3.1, expansion_order=4)
        )
        reduction = AxisReflectionReduction.detect(
            grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )
        radius_squared = np.sum(grid.coordinates * grid.coordinates, axis=1)
        density = np.exp(-0.9 * radius_squared)
        density *= 2.0 / grid.integrate(density)
        directory = Path.cwd() / ".tmp" / f"hartree-cache-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            first = NativeSymmetryMultipoleBoundaryBuilder(
                grid,
                reduction,
                9,
                cache_directory=directory,
                cache_key_seed="1" * 64,
            )
            first_rhs, first_boundary = first.build_reduced(density)
            second = NativeSymmetryMultipoleBoundaryBuilder(
                grid,
                reduction,
                9,
                cache_directory=directory,
                cache_key_seed="1" * 64,
            )
            second_rhs, second_boundary = second.build_reduced(density)

            self.assertEqual(first.cache_info.status, "miss-written")
            self.assertEqual(second.cache_info.status, "hit")
            self.assertEqual(first.cache_info.key, second.cache_info.key)
            np.testing.assert_array_equal(second_rhs, first_rhs)
            for key, expected in first_boundary.moments.items():
                self.assertEqual(second_boundary.moments[key], expected)

            # Without a cache directory the key addresses nothing and is not
            # hashed; the builder itself is the one that was cached above.
            with patch.object(
                native_boundary_module,
                "_symmetry_boundary_key",
                side_effect=AssertionError("a disabled cache hashed its key"),
            ):
                plain = NativeSymmetryMultipoleBoundaryBuilder(grid, reduction, 9)
            plain_rhs, plain_boundary = plain.build_reduced(density)
            self.assertIsNone(plain.cache_info.key)
            self.assertIsNone(plain.cache_info.path)
            self.assertEqual(plain.cache_info.status, "disabled-built")
            self.assertEqual(plain.cache_info.hash_seconds, 0.0)
            np.testing.assert_array_equal(plain_rhs, first_rhs)
            for key, expected in first_boundary.moments.items():
                self.assertEqual(plain_boundary.moments[key], expected)
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
