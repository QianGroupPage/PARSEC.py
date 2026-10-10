"""The Hartree boundary plan and its atomic tail in the accelerated builders.

The classes named ``CuPy...`` need a CUDA device and are skipped on a host
without one.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, product
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import parsec_python.acceleration.driver as driver_module
from parsec_python.acceleration.backends.cupy import cupy_available, require_cupy
from parsec_python.acceleration.backends.native import _load_native, native_available
from parsec_python.acceleration.backends.selection import BackendSelection
from parsec_python.acceleration.Hartree.atomic_tail import (
    build_atomic_tail_fast,
    host_tail_values,
    missing_stencil_entries,
)
from parsec_python.acceleration.Hartree.fast_multipole import (
    FastMultipoleExpansion,
    density_multipoles_fast,
)
from parsec_python.acceleration.Hartree.poisson import (
    build_hartree_problem,
    solve_scipy_hartree,
)
from parsec_python.acceleration.Symmetry import AxisReflectionReduction
from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree import (
    build_atomic_tail,
    density_multipoles,
    point_charge_multipoles,
    point_charge_potential,
    solve_hartree,
)
from parsec_python.Hartree.harmonics import _normalization
from parsec_python.Laplacian import (
    apply_negative_laplacian_boundary,
    build_negative_laplacian,
    neighbor_shells,
)
from parsec_python.models import (
    Atom,
    GridSettings,
    HartreeSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.SCF.single_point import prepare_single_point as prepare_reference


DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
BOUNDARY_SWITCHES = (
    "PARSEC_HARTREE_BOUNDARY",
    "PARSEC_HARTREE_BOUNDARY_TOLERANCE",
    "PARSEC_HARTREE_ATOMIC_TAIL",
    "PARSEC_HARTREE_LPOLE",
    "PARSEC_HARTREE_BOUNDARY_KERNEL",
    "PARSEC_HARTREE_BOUNDARY_BACKEND",
    "PARSEC_HARTREE_BOUNDARY_CHECK",
    "PARSEC_HARTREE_ATOMIC_TAIL_VALUES",
)
# What the launch environment of a cluster names beside them.  A preparation
# of this module names its own backends.
LAUNCH_BACKENDS = (
    "PARSEC_HARTREE_LINEAR_BACKEND",
    "PARSEC_IONIC_BACKEND",
    "PARSEC_CUPY_RESIDENT_HARTREE",
)
# Charges placed symmetrically under the three axis reflections.
SYMMETRIC_POSITIONS = np.array([
    [sx * 2.1, sy * 1.3, sz * 0.9]
    for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)
])
SYMMETRIC_CHARGES = np.full(8, 0.75)


def native_boundary_ready() -> bool:
    """Whether the loaded extension has the boundary kernels of this tree."""

    if not native_available():
        return False
    builder = getattr(_load_native(), "MultipoleBoundaryBuilder", None)
    return hasattr(builder, "configure_symmetry") and hasattr(
        builder, "export_full_geometry"
    )


def native_order_ready(order: int) -> bool:
    """Whether the loaded extension stores the angular arrays of ``order``."""

    if not native_boundary_ready():
        return False
    return order <= int(dict(_load_native().build_info()).get(
        "maximum_multipole_order", 9))


def sphere_grid(shift=(0.5, 0.5, 0.5), spacing=0.65, radius=3.9, order=8):
    return build_cluster_grid(GridSettings(
        spacing=spacing, radius=radius, expansion_order=order, shift=shift))


def symmetric_density(grid) -> np.ndarray:
    xyz = grid.coordinates
    density = np.exp(-0.5 * np.sum(xyz * xyz, axis=1)) * (
        1.0 + 0.2 * xyz[:, 0] ** 2 + 0.1 * xyz[:, 2] ** 2
    )
    return density * (6.0 / grid.integrate(density))


def legacy_fast_right_hand_side(density, grid, order) -> np.ndarray:
    """The former fast right-hand side."""

    return apply_negative_laplacian_boundary(
        8.0 * np.pi * density, grid,
        density_multipoles_fast(density, grid, order).potential,
    )


def clean_environment() -> dict[str, str]:
    """The environment without the boundary switches and the launch backends."""

    return {
        name: value for name, value in os.environ.items()
        if name not in BOUNDARY_SWITCHES and name not in LAUNCH_BACKENDS
    }


def hydrogen_problem(**hartree) -> SinglePointInput:
    positions = np.array(
        [[2.8, 0.0, 0.0], [-1.1, 2.3, 0.6], [0.4, -1.7, 2.1], [-0.8, -0.6, -2.5]])
    return SinglePointInput(
        atoms=[Atom("H", position) for position in positions],
        pseudopotentials={"H": SpeciesPotential(DATA / "H_POTRE.DAT", 0)},
        grid=GridSettings(spacing=0.7, radius=6.0, expansion_order=8),
        scf=SCFSettings(max_iterations=1, number_of_states=6),
        hartree=HartreeSettings(**hartree),
        recenter_geometry=False,
    )


class FastAtomicTailTests(unittest.TestCase):
    def test_surface_shell_gives_the_rows_of_the_reference_walk(self) -> None:
        for shift, order in (((0.5, 0.5, 0.5), 9), ((0.0, 0.0, 0.0), 4)):
            with self.subTest(shift=shift, order=order):
                grid = sphere_grid(shift)
                reference = build_atomic_tail(
                    grid, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, order)
                fast = build_atomic_tail_fast(
                    grid, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, order)
                np.testing.assert_array_equal(fast.rows, reference.rows)
                np.testing.assert_allclose(
                    fast.values, reference.values, rtol=0.0,
                    atol=1.0e-13 * np.abs(reference.values).max())
                self.assertAlmostEqual(fast.maximum, reference.maximum, places=12)
                self.assertEqual(fast.order, order)

    def test_entries_are_those_of_the_stencil_walk(self) -> None:
        grid = sphere_grid((0.0, 0.0, 0.0))
        rows, coefficients, point_index, integer_points = missing_stencil_entries(grid)
        count = sum(
            int(np.count_nonzero(neighbor_rows < 0))
            for _axis, _shell, neighbor_rows, _points in neighbor_shells(grid)
        )
        self.assertEqual(rows.size, count)
        self.assertEqual(point_index.max() + 1, integer_points.shape[0])
        self.assertTrue((grid.rows_for_integer_coordinates(integer_points) < 0).all())
        self.assertTrue(np.all(coefficients != 0.0))

    def test_threads_do_not_change_the_values(self) -> None:
        points = 4.4 * np.random.default_rng(2).normal(size=(9000, 3))
        points = points[np.linalg.norm(points, axis=1) > 3.9]
        serial = host_tail_values(
            points, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, 6, workers=1)
        threaded = host_tail_values(
            points, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, 6, workers=4)
        np.testing.assert_array_equal(serial, threaded)
        expected = point_charge_potential(
            points, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES
        ) - point_charge_multipoles(
            SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, 6).potential(points)
        np.testing.assert_allclose(serial, expected, atol=1.0e-12)

    def test_fast_right_hand_side_carries_the_tail(self) -> None:
        grid = sphere_grid()
        density = symmetric_density(grid)
        settings = HartreeSettings(multipole_order=5)
        tail = build_atomic_tail_fast(grid, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, 5)
        laplacian = build_negative_laplacian(grid)
        reference = solve_hartree(
            density, grid, laplacian, settings,
            boundary_tail=build_atomic_tail(
                grid, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, 5))
        rhs, _ = build_hartree_problem(density, grid, settings, tail)
        np.testing.assert_allclose(
            rhs, reference.right_hand_side, rtol=3.0e-12, atol=3.0e-12)
        solved = solve_scipy_hartree(
            density, grid, laplacian, settings, boundary_tail=tail)
        np.testing.assert_array_equal(solved.right_hand_side, rhs)
        self.assertIs(solved.boundary_tail, tail)
        # Without a tail the right-hand side is bitwise the former one.
        plain, _ = build_hartree_problem(density, grid, settings)
        np.testing.assert_array_equal(
            plain, legacy_fast_right_hand_side(density, grid, 5))
        with self.assertRaisesRegex(ValueError, "multipole order 5"):
            build_hartree_problem(
                density, grid, HartreeSettings(multipole_order=4), tail)


class HighOrderTests(unittest.TestCase):
    def test_recurrences_match_the_reference_harmonics(self) -> None:
        grid = sphere_grid(spacing=0.8, radius=3.6)
        density = symmetric_density(grid) * (1.0 + 0.3 * np.sin(grid.coordinates[:, 1]))
        points = 1.3 * grid.coordinates[
            np.linalg.norm(grid.coordinates, axis=1) > 3.2][::9]
        monopole = 2.0 * grid.integrate(density) / np.linalg.norm(points, axis=1).min()
        for order in (20, 34):
            with self.subTest(order=order):
                reference = density_multipoles(density, grid, order)
                fast = density_multipoles_fast(density, grid, order)
                # A moment of degree l is a sum of terms of size w r**l.
                weight = np.abs(density) * grid.volume_element
                radius = np.linalg.norm(grid.coordinates, axis=1)
                for key, expected in reference.moments.items():
                    self.assertLess(
                        abs(fast.moments[key] - expected),
                        1.0e-12 * np.sum(weight * radius ** key[0]))
                np.testing.assert_allclose(
                    fast.potential(points), reference.potential(points),
                    rtol=0.0, atol=1.0e-11 * monopole)

    def test_order_60_stays_in_range_with_sources_at_70_bohr(self) -> None:
        self.assertGreater(_normalization(60, 60), 1.0e-200)
        rng = np.random.default_rng(8)
        directions = rng.normal(size=(40, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        positions = directions * rng.uniform(5.0, 70.0, size=40)[:, None]
        charges = rng.uniform(0.5, 4.0, size=40)
        outside = rng.normal(size=(25, 3))
        outside *= 82.0 / np.linalg.norm(outside, axis=1)[:, None]
        moments = point_charge_multipoles(positions, charges, 60).moments
        series = FastMultipoleExpansion(order=60, moments=moments).potential(outside)
        self.assertTrue(np.isfinite(series).all())
        # The same series from Legendre polynomials of the angle between a
        # source and a point, without harmonics.
        source_radius = np.linalg.norm(positions, axis=1)
        cosine = (outside / 82.0) @ (positions / source_radius[:, None]).T
        previous, current = np.ones_like(cosine), cosine.copy()
        legendre = (charges / 82.0) * previous + (
            charges * source_radius / 82.0**2) * current
        for degree in range(2, 61):
            previous, current = current, (
                (2 * degree - 1) * cosine * current - (degree - 1) * previous
            ) / degree
            legendre += charges * source_radius**degree / 82.0 ** (degree + 1) * current
        np.testing.assert_allclose(series, 2.0 * legendre.sum(axis=1), rtol=1.0e-11)

    def test_older_extension_is_refused_above_order_9(self) -> None:
        import parsec_python.acceleration.Hartree.native_boundary as module

        class Old:
            @staticmethod
            def build_info():
                return {"version": "0.5.0"}

        module._require_native_order(Old, 9)
        with self.assertRaisesRegex(RuntimeError, "0.6.0"):
            module._require_native_order(Old, 10)
        with patch.object(module, "_load_native", return_value=Old):
            with self.assertRaisesRegex(RuntimeError, "Rebuild"):
                module.NativeMultipoleBoundaryBuilder(sphere_grid(), 20)
            with self.assertRaisesRegex(ValueError, "between 0 and 60"):
                module.NativeMultipoleBoundaryBuilder(sphere_grid(), 61)

    @unittest.skipUnless(
        native_order_ready(34), "needs the native extension built from this tree")
    def test_exported_prefactors_keep_the_layout_of_older_trees_up_to_order_9(
        self,
    ) -> None:
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            _stride_normalization,
        )
        from parsec_python.acceleration.Hartree.native_boundary import (
            NativeMultipoleBoundaryBuilder,
        )

        grid = sphere_grid()
        for order in (0, 2, 8, 9, 10, 34):
            with self.subTest(order=order):
                table = np.asarray(
                    NativeMultipoleBoundaryBuilder(grid, order)
                    ._native_builder.export_full_geometry()["normalization"])
                expected = np.zeros((order + 1, order + 1))
                for l in range(order + 1):
                    for m in range(l + 1):
                        expected[l, m] = _normalization(l, m)
                if order <= 9:
                    # The GPU kernels of a Python tree from before extension
                    # 0.6.0, which accepts no higher order, read norm[l*10+m]
                    # whatever the order.
                    former = np.zeros((10, 10))
                    former[: order + 1, : order + 1] = expected
                    self.assertEqual(table.shape, (100,))
                    np.testing.assert_allclose(
                        table.reshape(10, 10), former, rtol=1.0e-13, atol=0.0)
                else:
                    self.assertEqual(table.shape, ((order + 1) ** 2,))
                # The kernels of this tree read l*(order+1)+m, from either.
                np.testing.assert_allclose(
                    _stride_normalization(table, order).reshape(order + 1, order + 1),
                    expected, rtol=1.0e-13, atol=0.0)

    @unittest.skipUnless(
        native_order_ready(34), "needs native extension 0.6.0 built from this tree")
    def test_three_builders_agree_at_orders_9_20_34(self) -> None:
        from parsec_python.acceleration.Hartree.native_boundary import (
            NativeMultipoleBoundaryBuilder,
            NativeSymmetryMultipoleBoundaryBuilder,
        )

        atoms = tuple(Atom("H", position) for position in SYMMETRIC_POSITIONS)
        for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)):
            grid = sphere_grid(shift)
            density = symmetric_density(grid)
            laplacian = build_negative_laplacian(grid)
            reduction = AxisReflectionReduction.detect(grid, atoms)
            for order in (9, 20, 34):
                with self.subTest(shift=shift, order=order):
                    settings = HartreeSettings(multipole_order=order)
                    tail = build_atomic_tail_fast(
                        grid, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, order)
                    fast, _ = build_hartree_problem(density, grid, settings, tail)
                    reference = solve_hartree(
                        density, grid, laplacian, settings,
                        boundary_tail=build_atomic_tail(
                            grid, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, order),
                    ).right_hand_side
                    np.testing.assert_allclose(fast, reference, rtol=3.0e-12, atol=3.0e-12)
                    full = NativeMultipoleBoundaryBuilder(grid, order)
                    payload = full._native_builder.build(density)
                    self.assertEqual(
                        np.asarray(payload["positive_m_moments"]).shape,
                        (order + 1, order + 1))
                    full.set_boundary_tail(tail)
                    native, boundary = full.build(density)
                    np.testing.assert_allclose(native, fast, rtol=3.0e-12, atol=3.0e-12)
                    self.assertEqual(len(boundary.moments), (order + 1) ** 2)
                    wedge = NativeSymmetryMultipoleBoundaryBuilder(grid, reduction, order)
                    wedge.set_boundary_tail(tail)
                    np.testing.assert_allclose(
                        wedge.build_reduced(density)[0], reduction.reduce_vector(fast),
                        rtol=3.0e-12, atol=3.0e-12)
        with self.assertRaises(ValueError):
            NativeMultipoleBoundaryBuilder(grid, 61)


@unittest.skipUnless(
    native_boundary_ready(),
    "needs the native extension built from this tree",
)
class NativeAtomicTailTests(unittest.TestCase):
    def test_full_and_wedge_builders_add_the_tail(self) -> None:
        from parsec_python.acceleration.Hartree.native_boundary import (
            NativeMultipoleBoundaryBuilder,
            NativeSymmetryMultipoleBoundaryBuilder,
        )

        atoms = tuple(Atom("H", position) for position in SYMMETRIC_POSITIONS)
        for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)):
            with self.subTest(shift=shift):
                grid = sphere_grid(shift)
                density = symmetric_density(grid)
                settings = HartreeSettings(multipole_order=9)
                tail = build_atomic_tail_fast(
                    grid, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, 9)
                expected, _ = build_hartree_problem(density, grid, settings, tail)

                full = NativeMultipoleBoundaryBuilder(grid, 9)
                before, _ = full.build(density)
                full.set_boundary_tail(tail)
                with_tail, _ = full.build(density)
                np.testing.assert_allclose(with_tail, expected, rtol=3.0e-12, atol=3.0e-12)
                added = before.copy()
                added[tail.rows] += tail.values
                np.testing.assert_array_equal(with_tail, added)
                full.set_boundary_tail(None)
                np.testing.assert_array_equal(full.build(density)[0], before)

                reduction = AxisReflectionReduction.detect(grid, atoms)
                self.assertGreater(reduction.group_order, 1)
                wedge = NativeSymmetryMultipoleBoundaryBuilder(grid, reduction, 9)
                reduced_before, _ = wedge.build_reduced(density)
                wedge.set_boundary_tail(tail)
                reduced, _ = wedge.build_reduced(density)
                np.testing.assert_allclose(
                    reduced, reduction.reduce_vector(expected),
                    rtol=3.0e-12, atol=3.0e-12)
                self.assertGreater(np.abs(reduced - reduced_before).max(), 1.0e-6)
                wedge.set_boundary_tail(None)
                np.testing.assert_array_equal(
                    wedge.build_reduced(density)[0], reduced_before)

    def test_tail_of_another_order_is_refused(self) -> None:
        from parsec_python.acceleration.Hartree.native_boundary import (
            NativeMultipoleBoundaryBuilder,
        )

        grid = sphere_grid()
        tail = build_atomic_tail_fast(grid, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, 4)
        with self.assertRaisesRegex(ValueError, "multipole order 4"):
            NativeMultipoleBoundaryBuilder(grid, 9).set_boundary_tail(tail)


class DriverBoundarySwitchTests(unittest.TestCase):
    def switched(self, **environment):
        with patch.dict(os.environ, {**clean_environment(), **environment}, clear=True):
            return driver_module._apply_hartree_boundary_switches(
                hydrogen_problem(multipole_order=2)).hartree

    def test_switches_replace_the_input_settings(self) -> None:
        problem = hydrogen_problem(multipole_order=2)
        with patch.dict(os.environ, clean_environment(), clear=True):
            self.assertIs(driver_module._apply_hartree_boundary_switches(problem), problem)
        legacy = self.switched(PARSEC_HARTREE_BOUNDARY="legacy")
        self.assertEqual((legacy.boundary_tolerance, legacy.atomic_tail), (None, "off"))
        self.assertEqual(legacy.multipole_order, 2)
        self.assertEqual(
            self.switched(PARSEC_HARTREE_BOUNDARY_TOLERANCE="2e-4").boundary_tolerance,
            2.0e-4)
        self.assertIsNone(
            self.switched(PARSEC_HARTREE_BOUNDARY_TOLERANCE="off").boundary_tolerance)
        self.assertEqual(self.switched(PARSEC_HARTREE_ATOMIC_TAIL="on").atomic_tail, "on")
        self.assertEqual(self.switched(PARSEC_HARTREE_ATOMIC_TAIL="off").atomic_tail, "off")
        self.assertEqual(self.switched(PARSEC_HARTREE_LPOLE="30").multipole_order, 30)
        fixed = self.switched(PARSEC_HARTREE_BOUNDARY="legacy", PARSEC_HARTREE_LPOLE="20")
        self.assertEqual(
            (fixed.multipole_order, fixed.boundary_tolerance, fixed.atomic_tail),
            (20, None, "off"))
        # The input of a prepared system keeps the Solver_Lpole it came
        # from; the switch replaces that too.
        resolved = replace(problem, hartree=replace(
            problem.hartree, multipole_order=8, minimum_multipole_order=2))
        with patch.dict(
            os.environ, {**clean_environment(), "PARSEC_HARTREE_LPOLE": "4"}, clear=True
        ):
            lowered = driver_module._apply_hartree_boundary_switches(resolved).hartree
        self.assertEqual(
            (lowered.multipole_order, lowered.minimum_multipole_order), (4, None))
        for bad in (
            {"PARSEC_HARTREE_BOUNDARY": "old"},
            {"PARSEC_HARTREE_BOUNDARY_TOLERANCE": "tight"},
            {"PARSEC_HARTREE_BOUNDARY_TOLERANCE": "-1"},
            {"PARSEC_HARTREE_ATOMIC_TAIL": "yes"},
            {"PARSEC_HARTREE_LPOLE": "nine"},
            {"PARSEC_HARTREE_LPOLE": "61"},
            {"PARSEC_HARTREE_BOUNDARY": "legacy", "PARSEC_HARTREE_ATOMIC_TAIL": "on"},
            {"PARSEC_HARTREE_BOUNDARY": "legacy",
             "PARSEC_HARTREE_BOUNDARY_TOLERANCE": "1e-3"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.switched(**bad)

    def test_switches_enter_the_reference_cache_key(self) -> None:
        selection = BackendSelection(
            requested="scipy", selected="scipy",
            finite_difference_builder="reference", hartree_backend="scipy")

        def key(**environment):
            with patch.dict(os.environ, {**clean_environment(), **environment}, clear=True):
                problem = driver_module._apply_hartree_boundary_switches(
                    hydrogen_problem(multipole_order=2))
                return driver_module._reference_cache_key(
                    problem, selection, defer_native_laplacian=False,
                    cache_directory=None)

        keys = {
            key(),
            key(PARSEC_HARTREE_BOUNDARY="legacy"),
            key(PARSEC_HARTREE_BOUNDARY_TOLERANCE="1e-4"),
            key(PARSEC_HARTREE_ATOMIC_TAIL="on"),
            key(PARSEC_HARTREE_LPOLE="12"),
        }
        self.assertEqual(len(keys), 5)

    def test_preparations_leave_the_backends_of_the_launch_out(self) -> None:
        # The environment of a cluster launch names GPU backends for the
        # Hartree solver and the ionic sums.  Left in, they stop the host
        # preparations of this module before anything is prepared.
        launch = {
            "PARSEC_HARTREE_LINEAR_BACKEND": "cupy",
            "PARSEC_HARTREE_BOUNDARY_BACKEND": "cupy",
            "PARSEC_IONIC_BACKEND": "cupy",
            "PARSEC_CUPY_RESIDENT_HARTREE": "auto",
        }
        with patch.dict(os.environ, launch):
            cleaned = clean_environment()
            self.assertFalse(set(launch) & set(cleaned))
            with patch.dict(os.environ, cleaned, clear=True):
                system = driver_module.prepare_single_point(
                    hydrogen_problem(multipole_order=2), backend="scipy", symmetry="off")
            self.assertEqual(
                dict(system.backend_info.details)["hartree_backend"], "scipy")
            # The shell has its names again.
            self.assertEqual(os.environ["PARSEC_HARTREE_LINEAR_BACKEND"], "cupy")

    def test_host_backend_solves_with_the_tail_of_the_plan(self) -> None:
        problem = hydrogen_problem(multipole_order=2)
        reference = prepare_reference(problem)
        self.assertTrue(reference.hartree_boundary.atomic_tail)
        density = reference.initial_density
        expected = reference.solve_hartree(density)
        with patch.dict(os.environ, clean_environment(), clear=True):
            system = driver_module.prepare_single_point(
                problem, backend="scipy", symmetry="off")
            result = system.solve_hartree(density)
            details = dict(system.backend_info.details)
            np.testing.assert_allclose(
                result.right_hand_side, expected.right_hand_side,
                rtol=3.0e-12, atol=3.0e-12)
            self.assertEqual(details["hartree_atomic_tail"], "applied")
            # No device in this backend: the values of the tail are the host's.
            self.assertEqual(details["hartree_atomic_tail_values"], "host threads")
            self.assertEqual(details["hartree_multipole_order"], "8 (Solver_Lpole 2)")
            self.assertEqual(system.input.hartree.multipole_order, 8)
            self.assertEqual(result.boundary.order, 8)
            self.assertIn("hartree_atomic_tail_max", details)
            self.assertIn("engaged", details["hartree_boundary_estimate"])
            # The seconds of the estimate, which every rank spends before its grid.
            self.assertEqual(
                details["hartree_boundary_plan_seconds"],
                f"{system.reference.hartree_boundary.seconds:.6f}")
            self.assertGreater(system.reference.hartree_boundary.seconds, 0.0)
            self.assertAlmostEqual(
                system.hartree_boundary_tail_maximum,
                reference.hartree_boundary_tail.maximum, places=12)

        # The legacy switch restores the former right-hand side bit for bit.
        with patch.dict(
            os.environ,
            {**clean_environment(), "PARSEC_HARTREE_BOUNDARY": "legacy"},
            clear=True,
        ):
            system = driver_module.prepare_single_point(
                problem, backend="scipy", symmetry="off")
            result = system.solve_hartree(density)
            details = dict(system.backend_info.details)
        np.testing.assert_array_equal(
            result.right_hand_side,
            legacy_fast_right_hand_side(density, system.grid, 2))
        self.assertIsNone(result.boundary_tail)
        self.assertEqual(details["hartree_multipole_order"], "2 (Solver_Lpole 2)")
        self.assertEqual(details["hartree_boundary"], "PARSEC multipole expansion")
        self.assertEqual(details["hartree_atomic_tail"], "not applied")
        self.assertNotIn("hartree_atomic_tail_values", details)
        self.assertEqual(details["hartree_boundary_tolerance"], "off")

    def test_report_says_what_evaluated_the_tail(self) -> None:
        from types import SimpleNamespace

        problem = hydrogen_problem(multipole_order=2)
        partial = prepare_reference(
            problem, orbital_operators_only=True, build_atomic_tail=False)
        asked = []

        def kernel(points, positions, charges, order, *, device_id=None):
            asked.append(device_id)
            return host_tail_values(points, positions, charges, order)

        with patch(
            "parsec_python.acceleration.Hartree.cupy_atomic_tail.device_tail_values",
            kernel,
        ):
            host = driver_module._hartree_boundary_tail(
                partial, on_device=False, device_id=None)
            device = driver_module._hartree_boundary_tail(
                partial, on_device=True, device_id=2)
        self.assertEqual(asked, [2])
        self.assertEqual(
            (host.values_from, device.values_from), ("host threads", "device kernel"))
        np.testing.assert_array_equal(device.values, host.values)
        # The entry is that of the tail which was built, whatever the switch
        # names when the report is written.
        for environment in ({}, {"PARSEC_HARTREE_ATOMIC_TAIL_VALUES": "host"}):
            with patch.dict(os.environ, {**clean_environment(), **environment}, clear=True):
                for tail in (host, device):
                    details = dict(driver_module._hartree_boundary_details(partial, tail))
                    self.assertEqual(
                        details["hartree_atomic_tail_values"], tail.values_from)
        # A builder that holds the tail per exterior point says it itself.
        for evaluator in ("device kernel", "host threads"):
            builder = SimpleNamespace(
                atomic_tail_maximum=1.0e-4, atomic_tail_seconds=0.5,
                atomic_tail_values=evaluator)
            details = dict(
                driver_module._hartree_boundary_details(partial, None, builder))
            self.assertEqual(details["hartree_atomic_tail_values"], evaluator)
            self.assertEqual(details["hartree_atomic_tail_max"], "1.000000e-04 Ry")
        # No tail on this rank, as on a sector worker: no entry.
        self.assertNotIn(
            "hartree_atomic_tail_values",
            dict(driver_module._hartree_boundary_details(partial, None)))

    def test_pure_cupy_poisson_takes_the_tail_from_what_the_switch_names(self) -> None:
        # The backend that keeps the Poisson solver on the device with the
        # fast multipole boundary of the host asks for the tail itself.
        # Doubles stand for the device; reference, plan and tail are real.
        from unittest.mock import MagicMock

        from parsec_python.acceleration.tests.test_hybrid_driver import (
            HybridHartreeWiringTests,
        )

        problem = hydrogen_problem(multipole_order=2)
        reference = prepare_reference(
            problem, orbital_operators_only=True, build_atomic_tail=False)
        selection = BackendSelection(
            requested="cupy", selected="cupy",
            finite_difference_builder="reference", hartree_backend="cupy")
        asked = []

        def kernel(points, positions, charges, order, *, device_id=None):
            asked.append(device_id)
            return host_tail_values(points, positions, charges, order)

        def prepare(**environment):
            del asked[:]
            solver = MagicMock(name="poisson_solver")
            with (
                patch.dict(
                    os.environ, {**clean_environment(), **environment}, clear=True),
                patch.object(driver_module, "resolve_backend", return_value=selection),
                patch.object(
                    driver_module, "_prepare_reference_physics", return_value=reference),
                patch.object(
                    driver_module, "_build_backend",
                    return_value=HybridHartreeWiringTests._cupy_backend_shell()),
                patch(
                    "parsec_python.acceleration.Hartree.cupy_poisson.CuPyPoissonSolver",
                    return_value=solver),
                patch(
                    "parsec_python.acceleration.Hartree.cupy_atomic_tail."
                    "device_tail_values", kernel),
            ):
                system = driver_module.prepare_single_point(
                    problem, backend="cupy", symmetry="off")
                system.solve_hartree(np.zeros(reference.grid.size))
            details = dict(system.backend_info.details)
            self.assertEqual(details["hartree_backend"], "cupy")
            return details, solver.solve.call_args.kwargs["boundary_tail"], list(asked)

        details, kernel_tail, calls = prepare()
        self.assertEqual(calls, [None])
        self.assertEqual(kernel_tail.values_from, "device kernel")
        self.assertEqual(details["hartree_atomic_tail_values"], "device kernel")
        details, host_tail, calls = prepare(PARSEC_HARTREE_ATOMIC_TAIL_VALUES="host")
        self.assertEqual(calls, [])
        self.assertEqual(host_tail.values_from, "host threads")
        self.assertEqual(details["hartree_atomic_tail_values"], "host threads")
        np.testing.assert_array_equal(host_tail.rows, kernel_tail.rows)
        self.assertEqual(host_tail.order, reference.hartree_boundary.order)

    def test_tail_is_built_for_a_reference_that_left_it_out(self) -> None:
        problem = hydrogen_problem(multipole_order=2)
        complete = prepare_reference(problem)
        partial = prepare_reference(
            problem, orbital_operators_only=True, build_atomic_tail=False)
        tail = driver_module._hartree_boundary_tail(
            partial, on_device=False, device_id=None)
        self.assertEqual(tail.order, complete.hartree_boundary.order)
        np.testing.assert_array_equal(tail.rows, complete.hartree_boundary_tail.rows)
        np.testing.assert_allclose(
            tail.values, complete.hartree_boundary_tail.values, rtol=0.0, atol=1.0e-12)
        off = prepare_reference(
            replace(problem, hartree=replace(problem.hartree, atomic_tail="off")),
            orbital_operators_only=True)
        self.assertIsNone(
            driver_module._hartree_boundary_tail(off, on_device=False, device_id=None))


def reflection_subgroups() -> list[tuple[tuple[int, int, int], ...]]:
    """The sixteen subgroups of the group of the three axis reflections."""

    elements = list(product((1, -1), repeat=3))
    found = set()
    for count in range(4):
        for generators in combinations(elements[1:], count):
            group = {(1, 1, 1)}
            grown = True
            while grown:
                grown = False
                for member in tuple(group):
                    for generator in generators:
                        image = tuple(a * b for a, b in zip(member, generator))
                        if image not in group:
                            group.add(image)
                            grown = True
            found.add(tuple(sorted(group)))
    return sorted(found, key=lambda group: (len(group), group))


def invariant_density(grid, reduction) -> np.ndarray:
    """A positive density with no symmetry beyond the orbits of ``reduction``."""

    rng = np.random.default_rng(11)
    xyz = grid.coordinates
    envelope = np.exp(-0.25 * np.sum(xyz * xyz, axis=1))
    values = rng.uniform(0.5, 1.5, size=reduction.wedge_size)
    density = envelope * values[reduction.full_to_wedge]
    # The envelope is even in every coordinate only to round-off.
    density = reduction.project_invariant(density)
    return density * (6.0 / grid.integrate(density))


# Atoms whose symmetry is D2h, D2 and the identity alone.
ARRANGEMENTS = {
    "D2h": SYMMETRIC_POSITIONS,
    "D2": np.array([[2.1, 1.3, 0.9], [-2.1, -1.3, 0.9], [-2.1, 1.3, -0.9], [2.1, -1.3, -0.9]]),
    "C1": np.array([[2.1, 1.3, 0.9], [-0.4, 0.8, -1.7]]),
}


class WedgeFormulaTests(unittest.TestCase):
    def test_class_table_reproduces_the_full_grid_moments(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            reflection_class_table,
        )
        from parsec_python.Hartree.harmonics import _positive_m_harmonic_rows

        order = 12
        subgroups = reflection_subgroups()
        self.assertEqual(sorted(len(group) for group in subgroups),
                         [1] + [2] * 7 + [4] * 7 + [8])
        seen = set()
        for shift in (0.5, 0.0):
            grid = sphere_grid((shift,) * 3, spacing=0.7, radius=4.3, order=4)
            integer = grid.integer_coordinates
            radius = np.linalg.norm(grid.coordinates, axis=1)
            base = np.random.default_rng(5).random(grid.size)
            for group in subgroups:
                maps = np.stack([
                    grid.rows_for_integer_coordinates(
                        np.where(np.array(signs) > 0, integer, -integer - int(2 * shift)))
                    for signs in group])
                self.assertTrue((maps >= 0).all())
                weight = base[maps].mean(axis=0) * grid.volume_element
                representatives = np.flatnonzero(maps.min(axis=0) == np.arange(grid.size))
                multiplicity = np.array([
                    np.unique(maps[:, row]).size for row in representatives])
                self.assertEqual(multiplicity.sum(), grid.size)
                seen.update(multiplicity.tolist())
                table = reflection_class_table(group)
                full = np.zeros((order + 1, order + 1), dtype=complex)
                for l, m, harmonic, r in _positive_m_harmonic_rows(grid.coordinates, order):
                    full[l, m] = np.sum(weight * r**l * np.conjugate(harmonic))
                wedge = np.zeros_like(full)
                for l, m, harmonic, r in _positive_m_harmonic_rows(
                    grid.coordinates[representatives], order
                ):
                    real_weight, imaginary_weight = table[l & 1, m & 1]
                    wedge[l, m] = np.sum(
                        weight[representatives] * multiplicity * r**l
                        * (real_weight * harmonic.real - 1j * imaginary_weight * harmonic.imag))
                for l in range(order + 1):
                    with self.subTest(shift=shift, group=group, l=l):
                        self.assertLess(
                            np.abs(full[l, : l + 1] - wedge[l, : l + 1]).max(),
                            1.0e-13 * np.sum(np.abs(weight) * radius**l))
        self.assertEqual(seen, {1, 2, 4, 8})

    def test_unique_rows_are_those_of_the_row_sort(self) -> None:
        import parsec_python.acceleration.Hartree.cupy_boundary as boundary_module

        generator = np.random.default_rng(8)
        base = generator.normal(size=(700, 5))
        base[:40, 2] = 0.0
        base[40:80, 4] = 1.0
        table = base[generator.integers(0, base.shape[0], size=6000)]
        # A point on the axis is reached with either sign of zero.
        negative = (table[:, 2] == 0.0) & (generator.random(table.shape[0]) < 0.5)
        table[negative, 2] = -0.0
        expected_rows, expected_inverse = np.unique(table, axis=0, return_inverse=True)
        rows, inverse = boundary_module.unique_rows(table)
        np.testing.assert_array_equal(rows, expected_rows)
        np.testing.assert_array_equal(inverse, expected_inverse.reshape(-1))
        np.testing.assert_array_equal(rows[inverse], table)
        self.assertLess(rows.shape[0], base.shape[0] + 1)
        # Rows that share a key without being equal go through the row sort.
        with patch.object(
            boundary_module, "_row_keys",
            side_effect=lambda rows: np.zeros(rows.shape[0], dtype=np.uint64),
        ) as keys:
            rows, inverse = boundary_module.unique_rows(table)
        keys.assert_called_once()
        np.testing.assert_array_equal(rows, expected_rows)
        np.testing.assert_array_equal(inverse, expected_inverse.reshape(-1))


class BoundaryCheckSwitchTests(unittest.TestCase):
    def count(self, **environment):
        with patch.dict(os.environ, {**clean_environment(), **environment}, clear=True):
            return driver_module._hartree_boundary_check_count()

    def test_switch_is_a_number_of_points(self) -> None:
        self.assertEqual(self.count(), 0)
        self.assertEqual(self.count(PARSEC_HARTREE_BOUNDARY_CHECK="512"), 512)
        for bad in ("many", "-1", "1.5"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.count(PARSEC_HARTREE_BOUNDARY_CHECK=bad)
        # A value the switch does not know is refused before any set-up.
        with patch.dict(
            os.environ,
            {**clean_environment(), "PARSEC_HARTREE_BOUNDARY_CHECK": "many"},
            clear=True,
        ), self.assertRaises(ValueError):
            driver_module.prepare_single_point(hydrogen_problem(), backend="scipy")

    def test_check_reports_the_errors_of_a_gpu_builder_only(self) -> None:
        from types import SimpleNamespace

        direct = np.array([10.0, 20.0, 30.0, 40.0])

        class CuPySymmetryMultipoleBoundaryBuilder:
            reduction = SimpleNamespace(representative_rows=np.array([0, 2]))
            total = 9

            def boundary_check(self, wedge_density, count, minimum_order=None, positions=None):
                self.called = (
                    wedge_density.tolist(), count, minimum_order, positions.tolist())
                return dict(
                    points=np.zeros((4, 3)), total=self.total, direct=direct,
                    tail=np.zeros(4),
                    boundary=direct + np.array([1e-5, -3e-5, 2e-5, 0.0]),
                    legacy=direct + np.array([0.1, -0.4, 0.2, 0.0]))

        builder = CuPySymmetryMultipoleBoundaryBuilder()
        system = SimpleNamespace(
            backend=SimpleNamespace(native_boundary_builder=builder),
            reference=SimpleNamespace(
                hartree_boundary=SimpleNamespace(minimum_order=9, atomic_tail=True),
                atoms=(Atom("H", (0.0, 0.0, 1.5)),),
                pseudopotentials={"H": SimpleNamespace(ionic_charge=1.0)},
                electron_count=1.0))
        result = SimpleNamespace(density=np.array([1.0, 2.0, 3.0, 4.0]))
        with patch.dict(os.environ, clean_environment(), clear=True):
            self.assertEqual(driver_module._hartree_boundary_check(system, result), ())
        with patch.dict(
            os.environ,
            {**clean_environment(), "PARSEC_HARTREE_BOUNDARY_CHECK": "4"}, clear=True,
        ):
            details = dict(driver_module._hartree_boundary_check(system, result))
            # The atoms go with the call: the sample takes the points nearest to one.
            self.assertEqual(builder.called, ([1.0, 3.0], 4, 9, [[0.0, 0.0, 1.5]]))
            self.assertEqual(details["hartree_boundary_check_points"], "4")
            # The maximum is that of a sample, and the report says of which.
            self.assertTrue(details["hartree_boundary_check_sample"].startswith(
                "4 of 9 unique exterior points of the symmetry wedge: the innermost"))
            self.assertEqual(details["hartree_boundary_check_max"], "3.000000e-05 Ry")
            self.assertEqual(details["hartree_boundary_check_legacy_max"], "4.000000e-01 Ry")
            self.assertAlmostEqual(
                float(details["hartree_boundary_check_rms"].split()[0]),
                np.sqrt((1 + 9 + 4) / 4) * 1e-5, places=10)
            # A caller that times the SCF around run_scf takes the check out.
            checked = SimpleNamespace(backend=SimpleNamespace(details=tuple(details.items())))
            self.assertEqual(
                driver_module.hartree_boundary_check_seconds(checked),
                float(details["hartree_boundary_check_seconds"]))
            self.assertEqual(driver_module.hartree_boundary_check_seconds(
                SimpleNamespace(backend=SimpleNamespace(details=()))), 0.0)
            builder.total = 4
            self.assertEqual(
                dict(driver_module._hartree_boundary_check(system, result))[
                    "hartree_boundary_check_sample"],
                "all 4 unique exterior points of the symmetry wedge")
            system.backend.native_boundary_builder = object()
            self.assertIn(
                "not run", dict(driver_module._hartree_boundary_check(system, result))[
                    "hartree_boundary_check"])

    def test_sample_takes_the_innermost_the_largest_tail_and_the_nearest(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_boundary import _check_points

        generator = np.random.default_rng(3)
        total = 400
        radius = generator.permutation(total).astype(float)
        tail = (generator.permutation(total) - 0.5 * total + 0.25) * 1.0e-4
        nearest = generator.permutation(total).astype(float)
        index = _check_points(40, radius, tail, nearest, seed=1)
        self.assertEqual((index.size, np.unique(index).size), (40, 40))
        np.testing.assert_array_equal(index[:10], np.argsort(radius)[:10])
        left = np.setdiff1d(np.arange(total), index[:10])
        np.testing.assert_array_equal(
            index[10:20], left[np.argsort(-np.abs(tail[left]), kind="stable")[:10]])
        left = np.setdiff1d(left, index[10:20])
        np.testing.assert_array_equal(
            index[20:30], left[np.argsort(nearest[left], kind="stable")[:10]])
        # The last share is random and follows the seed alone.
        np.testing.assert_array_equal(_check_points(40, radius, tail, nearest, seed=1), index)
        other = _check_points(40, radius, tail, nearest, seed=2)
        np.testing.assert_array_equal(other[:30], index[:30])
        self.assertFalse(np.array_equal(other[30:], index[30:]))
        # An error that peaks at the innermost point, in the direction of no
        # large tail, is in the sample; the largest tails alone miss it.
        error = 1.0 / (1.0 + radius)
        self.assertEqual(np.abs(error[index]).max(), error.max())
        self.assertLess(
            np.abs(error[np.argsort(-np.abs(tail))[:20]]).max(), 0.5 * error.max())
        # Without a tail and without atoms: the innermost half and a random half.
        plain = _check_points(40, radius)
        np.testing.assert_array_equal(plain[:20], np.argsort(radius)[:20])
        self.assertEqual(np.unique(plain).size, 40)
        # A count at or above the number of points takes them all.
        np.testing.assert_array_equal(
            np.sort(_check_points(1000, radius, tail, nearest)), np.arange(total))

class BoundaryKernelChoiceTests(unittest.TestCase):
    def choose(
        self, reduction, environment, *, gpu_kernels=False, orbit_table=True,
        gpu_failure=None, tail_on_device=False, **hartree
    ):
        """Build the boundary of a hydrogen reference under doubles of the builders.

        Returns the builder, whether it works on the symmetry wedge, and the
        doubles by name.  ``orbit_table=False`` makes the native wedge
        builder fail as it does where its table exceeds the storage limit,
        and ``gpu_failure`` is raised by the GPU builders.  ``tail_options``
        of the doubles holds the options of every tail in rows that was
        asked for.
        """

        from types import SimpleNamespace
        from unittest.mock import MagicMock

        reference = prepare_reference(
            hydrogen_problem(multipole_order=2, **hartree),
            orbital_operators_only=True, build_atomic_tail=False)
        doubles = SimpleNamespace(
            full=MagicMock(name="full", side_effect=gpu_failure),
            wedge=MagicMock(name="wedge", side_effect=gpu_failure),
            point=MagicMock(name="point", side_effect=gpu_failure),
            native_full=MagicMock(name="native_full"),
            native_wedge=MagicMock(
                name="native_wedge",
                side_effect=None if orbit_table else RuntimeError("orbit table limit")),
            tail=object(),
            tail_options=[],
        )

        def tail_in_rows(reference, **options):
            if not reference.hartree_boundary.atomic_tail:
                return None
            doubles.tail_options.append(options)
            return doubles.tail

        gpu = "parsec_python.acceleration.Hartree.cupy_boundary."
        native = "parsec_python.acceleration.Hartree.native_boundary."
        with (
            patch.dict(os.environ, {
                **clean_environment(), "PARSEC_HARTREE_BOUNDARY_BACKEND": "cupy",
                **environment}, clear=True),
            patch(gpu + "CuPyMultipoleBoundaryBuilder", doubles.full),
            patch(gpu + "CuPySymmetryMultipoleBoundaryBuilder", doubles.wedge),
            patch(gpu + "CuPyPointMultipoleBoundaryBuilder", doubles.point),
            patch(native + "NativeMultipoleBoundaryBuilder", doubles.native_full),
            patch(native + "NativeSymmetryMultipoleBoundaryBuilder", doubles.native_wedge),
            patch.object(
                driver_module, "_hartree_boundary_tail", side_effect=tail_in_rows),
        ):
            builder, symmetric, _cache, _seconds = (
                driver_module._build_native_boundary_builder(
                    reference, reduction, symmetry_cache_directory=None,
                    symmetry_geometry_cache_info=None, device_id=3,
                    gpu_kernels=gpu_kernels, tail_on_device=tail_on_device))
        return builder, symmetric, doubles

    def unsupported_reduction(self, reduction):
        """A reduction whose operations are no axis reflections."""

        from parsec_python.acceleration.Symmetry import SignedPermutationReduction

        return SignedPermutationReduction(
            signs=reduction.signs, representative_rows=reduction.representative_rows,
            full_to_wedge=reduction.full_to_wedge, multiplicities=reduction.multiplicities,
            operations=np.zeros((1, 3, 3), dtype=np.int8),
            generator_bits=np.zeros((1, 1), dtype=np.int8))

    def test_wedge_kernels_serve_the_engaged_boundary_only(self) -> None:
        grid = sphere_grid()
        reduction = AxisReflectionReduction.detect(grid, (Atom("H", (0.0, 0.0, 0.0)),))
        builder, symmetric, doubles = self.choose(reduction, {})
        self.assertIs(builder, doubles.wedge.return_value)
        self.assertTrue(symmetric)
        # The order of the plan, read from a reference that was prepared
        # without the fields of the SCF loop: 8 for Solver_Lpole 2.
        self.assertEqual(doubles.wedge.call_args.args[1:], (reduction, 8))
        self.assertEqual(doubles.wedge.call_args.kwargs, {"device_id": 3})
        positions, charges = builder.set_atomic_tail.call_args.args
        self.assertEqual((positions.shape, charges.shape), ((4, 3), (4,)))
        doubles.full.assert_not_called()
        doubles.point.assert_not_called()

        # Not engaged, or told so: the former full-grid kernels.
        for environment, hartree, tailed in (
            ({}, {"boundary_tolerance": None}, False),
            ({}, {"boundary_tolerance": None, "atomic_tail": "off"}, False),
            ({"PARSEC_HARTREE_BOUNDARY_KERNEL": "full"}, {}, True),
            ({}, {"boundary_tolerance": None, "atomic_tail": "on"}, True),
        ):
            for symmetry in (reduction, None, self.unsupported_reduction(reduction)):
                with self.subTest(
                    environment=environment, hartree=hartree, symmetry=type(symmetry).__name__
                ):
                    builder, symmetric, doubles = self.choose(
                        symmetry, environment, **hartree)
                    self.assertIs(builder, doubles.full.return_value)
                    self.assertFalse(symmetric)
                    doubles.wedge.assert_not_called()
                    doubles.point.assert_not_called()
                    self.assertEqual(
                        doubles.full.call_args.args[1],
                        2 if "boundary_tolerance" in hartree else 8)
                    if tailed:
                        builder.set_boundary_tail.assert_called_once_with(doubles.tail)
                    else:
                        builder.set_boundary_tail.assert_not_called()

        # The wedge kernels on request, without a tail to fold in.
        builder, symmetric, doubles = self.choose(
            reduction, {"PARSEC_HARTREE_BOUNDARY_KERNEL": "wedge"},
            boundary_tolerance=None)
        self.assertIs(builder, doubles.wedge.return_value)
        builder.set_atomic_tail.assert_not_called()
        with self.assertRaisesRegex(ValueError, "PARSEC_HARTREE_BOUNDARY_BACKEND=cupy"):
            self.choose(reduction, {
                "PARSEC_HARTREE_BOUNDARY_KERNEL": "wedge",
                "PARSEC_HARTREE_BOUNDARY_BACKEND": "native"})
        with self.assertRaisesRegex(ValueError, "auto, full, or wedge"):
            self.choose(reduction, {"PARSEC_HARTREE_BOUNDARY_KERNEL": "half"})

    def test_serial_control_of_the_runner_takes_the_full_grid_kernels(self) -> None:
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        reduction = AxisReflectionReduction.detect(
            sphere_grid(), (Atom("H", (0.0, 0.0, 0.0)),))
        # What the launcher of a control names, as the runner completes it.
        control = {"PARSEC_HARTREE_BOUNDARY_BACKEND": "cupy"}
        mpi_full_scf._keep_former_routes(control)
        for symmetry in (reduction, None):
            with self.subTest(symmetry=type(symmetry).__name__):
                # The run: the wedge of the group, or one value per point.
                builder, _symmetric, doubles = self.choose(symmetry, {})
                self.assertIs(
                    builder,
                    (doubles.wedge if symmetry is not None else doubles.point).return_value)
                # Its control: the same order and the same tail through
                # the full-grid kernels, with the tail in rows, and those
                # from host threads where the run asks the device kernel.
                builder, symmetric, doubles = self.choose(
                    symmetry, control, tail_on_device=True)
                self.assertIs(builder, doubles.full.return_value)
                self.assertFalse(symmetric)
                self.assertEqual(doubles.full.call_args.args[1], 8)
                builder.set_boundary_tail.assert_called_once_with(doubles.tail)
                self.assertEqual(
                    doubles.tail_options, [{"on_device": False, "device_id": 3}])
                doubles.wedge.assert_not_called()
                doubles.point.assert_not_called()
        # A launcher that names the kernels or the values keeps them.
        named = {
            "PARSEC_HARTREE_BOUNDARY_KERNEL": "wedge",
            "PARSEC_HARTREE_ATOMIC_TAIL_VALUES": "auto",
        }
        mpi_full_scf._keep_former_routes(named)
        self.assertEqual(named["PARSEC_HARTREE_BOUNDARY_KERNEL"], "wedge")
        self.assertEqual(named["PARSEC_HARTREE_ATOMIC_TAIL_VALUES"], "auto")

    def test_host_threads_evaluate_the_tail_where_they_are_asked_for(self) -> None:
        reduction = AxisReflectionReduction.detect(
            sphere_grid(), (Atom("H", (0.0, 0.0, 0.0)),))
        host = {"PARSEC_HARTREE_ATOMIC_TAIL_VALUES": "host"}
        full = {"PARSEC_HARTREE_BOUNDARY_KERNEL": "full"}
        native = {"PARSEC_HARTREE_BOUNDARY_BACKEND": "native"}
        # The tail in rows, of a GPU builder and of a native one: the device
        # kernel where the caller has a device, unless host threads are named.
        for route, environment in (("full", full), ("native_wedge", native)):
            for values, tail_on_device, on_device in (
                ({}, True, True),
                ({"PARSEC_HARTREE_ATOMIC_TAIL_VALUES": "auto"}, True, True),
                (host, True, False),
                ({}, False, False),
                (host, False, False),
            ):
                with self.subTest(route=route, values=values, device=tail_on_device):
                    builder, _symmetric, doubles = self.choose(
                        reduction, {**environment, **values},
                        tail_on_device=tail_on_device)
                    self.assertIs(builder, getattr(doubles, route).return_value)
                    self.assertEqual(
                        doubles.tail_options,
                        [{"on_device": on_device, "device_id": 3}])
        # The tail per unique exterior point of the wedge and of the full
        # grid: the kernel of the builder's device, or the host function.
        for symmetry, route in ((reduction, "wedge"), (None, "point")):
            with self.subTest(route=route):
                builder, _symmetric, doubles = self.choose(
                    symmetry, {}, tail_on_device=True)
                self.assertIs(builder, getattr(doubles, route).return_value)
                self.assertEqual(builder.set_atomic_tail.call_args.kwargs, {})
                builder, _symmetric, doubles = self.choose(
                    symmetry, host, tail_on_device=True)
                self.assertIs(builder, getattr(doubles, route).return_value)
                self.assertEqual(
                    builder.set_atomic_tail.call_args.kwargs,
                    {"tail_values": host_tail_values})
                positions, charges = builder.set_atomic_tail.call_args.args
                self.assertEqual((positions.shape, charges.shape), ((4, 3), (4,)))
                self.assertEqual(doubles.tail_options, [])
        with self.assertRaisesRegex(ValueError, "auto or host"):
            self.choose(reduction, {"PARSEC_HARTREE_ATOMIC_TAIL_VALUES": "device"})
        # Refused when a preparation starts, like the other switches.
        with patch.dict(os.environ, {
            **clean_environment(), "PARSEC_HARTREE_ATOMIC_TAIL_VALUES": "gpu"}, clear=True
        ), self.assertRaisesRegex(ValueError, "PARSEC_HARTREE_ATOMIC_TAIL_VALUES"):
            driver_module.prepare_single_point(
                hydrogen_problem(multipole_order=2), backend="scipy", symmetry="off")

    def test_engaged_boundary_without_reflections_takes_one_value_per_point(self) -> None:
        reduction = AxisReflectionReduction.detect(
            sphere_grid(), (Atom("H", (0.0, 0.0, 0.0)),))
        for unsupported in (None, self.unsupported_reduction(reduction)):
            with self.subTest(reduction=type(unsupported).__name__):
                builder, symmetric, doubles = self.choose(unsupported, {})
                # The kernels of the wedge builder under the identity alone,
                # behind the full-grid interface: the driver projects.
                self.assertIs(builder, doubles.point.return_value)
                self.assertFalse(symmetric)
                self.assertEqual(doubles.point.call_args.args[1:], (8,))
                self.assertEqual(doubles.point.call_args.kwargs, {"device_id": 3})
                positions, charges = builder.set_atomic_tail.call_args.args
                self.assertEqual((positions.shape, charges.shape), ((4, 3), (4,)))
                builder.set_boundary_tail.assert_not_called()
                doubles.full.assert_not_called()
                doubles.wedge.assert_not_called()
                # The wedge of a group is not to be had by asking.
                with self.assertRaisesRegex(ValueError, "axis-reflection"):
                    self.choose(unsupported, {"PARSEC_HARTREE_BOUNDARY_KERNEL": "wedge"})
        import parsec_python.acceleration.Hartree.cupy_boundary as boundary_module

        identity = boundary_module.identity_reduction(5)
        self.assertEqual((identity.group_order, identity.wedge_size, identity.full_size), (1, 5, 5))
        np.testing.assert_array_equal(
            identity.reduce_vector(np.arange(5.0)), np.arange(5.0))
        self.assertEqual(
            boundary_module.reflection_class_table(identity.signs).tolist(),
            [[[1.0, 1.0]] * 2] * 2)
        self.assertIn(
            "CuPyPointMultipoleBoundaryBuilder", driver_module._GPU_BOUNDARY_BUILDERS)

    def test_gpu_kernels_take_an_engaged_boundary_the_native_table_cannot(self) -> None:
        reduction = AxisReflectionReduction.detect(
            sphere_grid(), (Atom("H", (0.0, 0.0, 0.0)),))
        default = {"PARSEC_HARTREE_BOUNDARY_BACKEND": ""}

        def native_full(doubles, builder, symmetric, tailed=True):
            self.assertIs(builder, doubles.native_full.return_value)
            self.assertFalse(symmetric)
            if tailed:
                builder.set_boundary_tail.assert_called_once_with(doubles.tail)

        # The table of the raised order fits: the native wedge builder, as before.
        builder, symmetric, doubles = self.choose(reduction, default, gpu_kernels=True)
        self.assertIs(builder, doubles.native_wedge.return_value)
        self.assertTrue(symmetric)
        self.assertEqual(doubles.native_wedge.call_args.args[1:], (reduction, 8))
        doubles.wedge.assert_not_called()
        # It does not fit: the wedge kernels of the GPU instead of the
        # full-grid recurrences of the host in every solve.
        for environment in (default, {"PARSEC_HARTREE_BOUNDARY_BACKEND": "auto"}):
            with self.subTest(environment=environment):
                builder, symmetric, doubles = self.choose(
                    reduction, environment, gpu_kernels=True, orbit_table=False)
                self.assertIs(builder, doubles.wedge.return_value)
                self.assertTrue(symmetric)
                self.assertEqual(doubles.wedge.call_args.args[1:], (reduction, 8))
                builder.set_atomic_tail.assert_called_once()
                doubles.native_full.assert_not_called()
        # No reduction has no table at all.
        builder, symmetric, doubles = self.choose(None, default, gpu_kernels=True)
        self.assertIs(builder, doubles.point.return_value)
        self.assertFalse(symmetric)
        doubles.native_full.assert_not_called()

        # The former route stays: without a GPU for the orbitals, where
        # native or the full kernels are asked by name, where the plan is
        # not engaged, on an extension the GPU builders cannot read, and
        # on a device without the room for them.
        for options, hartree in (
            (dict(gpu_kernels=False), {}),
            (dict(gpu_kernels=True, environment={
                "PARSEC_HARTREE_BOUNDARY_BACKEND": "native"}), {}),
            (dict(gpu_kernels=True, environment={
                **default, "PARSEC_HARTREE_BOUNDARY_KERNEL": "full"}), {}),
            (dict(gpu_kernels=True, gpu_failure=RuntimeError("no exporter")), {}),
            (dict(gpu_kernels=True, gpu_failure=MemoryError("out of device memory")), {}),
            (dict(gpu_kernels=True), {"boundary_tolerance": None}),
        ):
            for symmetry in (reduction, None):
                with self.subTest(options=options, hartree=hartree, symmetry=symmetry is None):
                    environment = options.get("environment", default)
                    builder, symmetric, doubles = self.choose(
                        symmetry, environment, orbit_table=False,
                        **{name: value for name, value in options.items()
                           if name != "environment"},
                        **hartree)
                    native_full(doubles, builder, symmetric, tailed=not hartree)
                    self.assertEqual(
                        doubles.native_full.call_args.args[1], 2 if hartree else 8)


@unittest.skipUnless(
    cupy_available() and native_order_ready(34),
    "needs a CUDA device and native extension 0.6.0 built from this tree",
)
class CuPyWedgeBoundaryTests(unittest.TestCase):
    def reference_wedge_rhs(self, grid, reduction, density, order, tail):
        full, _ = build_hartree_problem(
            density, grid, HartreeSettings(multipole_order=order), tail)
        return reduction.reduce_vector(full)

    def test_wedge_builder_matches_native_and_projected_full_rhs(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            CuPySymmetryMultipoleBoundaryBuilder,
        )
        from parsec_python.acceleration.Hartree.native_boundary import (
            NativeSymmetryMultipoleBoundaryBuilder,
        )
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetryScalarField

        cp, _ = require_cupy()
        for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)):
            grid = sphere_grid(shift)
            for name, positions in ARRANGEMENTS.items():
                atoms = tuple(Atom("H", position) for position in positions)
                charges = np.linspace(0.6, 1.4, len(positions))[::-1].copy()
                if name != "C1":
                    charges[:] = 0.75
                reduction = AxisReflectionReduction.detect(grid, atoms)
                self.assertEqual(reduction.group_order, {"D2h": 8, "D2": 4, "C1": 1}[name])
                density = invariant_density(grid, reduction)
                for order in (9, 34):
                    with self.subTest(shift=shift, group=name, order=order):
                        tail = build_atomic_tail_fast(grid, positions, charges, order)
                        expected = self.reference_wedge_rhs(
                            grid, reduction, density, order, tail)
                        builder = CuPySymmetryMultipoleBoundaryBuilder(grid, reduction, order)
                        plain, boundary = builder.build_reduced(density)
                        np.testing.assert_allclose(
                            plain,
                            self.reference_wedge_rhs(grid, reduction, density, order, None),
                            rtol=3.0e-12, atol=3.0e-12)
                        bytes_before = builder.device_storage_bytes
                        builder.set_atomic_tail(positions, charges)
                        self.assertAlmostEqual(
                            builder.atomic_tail_maximum, tail.maximum, places=10)
                        actual, _ = builder.build_reduced(density)
                        np.testing.assert_allclose(actual, expected, rtol=3.0e-12, atol=3.0e-12)
                        native = NativeSymmetryMultipoleBoundaryBuilder(grid, reduction, order)
                        native.set_boundary_tail(tail)
                        native_rhs, native_boundary = native.build_reduced(density)
                        np.testing.assert_allclose(
                            actual, native_rhs, rtol=3.0e-12, atol=3.0e-12)
                        weight = np.abs(density) * grid.volume_element
                        radius = np.linalg.norm(grid.coordinates, axis=1)
                        for key, value in native_boundary.moments.items():
                            self.assertLess(
                                abs(boundary.moments[key] - value),
                                1.0e-11 * np.sum(weight * radius ** key[0]))
                        # The same bits from a second build, from a wedge
                        # field and from a device array.
                        again, _ = builder.build_reduced(density)
                        np.testing.assert_array_equal(again, actual)
                        field = SymmetryScalarField(
                            reduction, density[reduction.representative_rows])
                        from_field = builder.build_reduced(field)[0].copy()
                        np.testing.assert_allclose(
                            from_field, actual, rtol=1.0e-12, atol=1.0e-12)
                        device_rhs, _ = builder.build_reduced_device(
                            cp.asarray(density[reduction.representative_rows]))
                        np.testing.assert_array_equal(cp.asnumpy(device_rhs), from_field)
                        # Storage: the wedge, its stencil entries, the unique
                        # exterior points and the blocks of partial moments.
                        wedge, points = reduction.wedge_size, builder.exterior_point_count
                        self.assertLess(points, builder.boundary_term_count)
                        self.assertEqual(
                            builder.device_storage_bytes - bytes_before, 8 * points)
                        self.assertEqual(
                            builder.device_storage_bytes,
                            wedge * 9 * 8 + (wedge + 1) * 8 + builder.boundary_term_count * 12
                            + points * 56 + 2 * (order + 1) ** 2 * 8 * builder.blocks
                            + (order + 1) ** 2 * 8 + 4 * builder.m_list.size + 64)
                        builder.set_atomic_tail(None, None)
                        np.testing.assert_array_equal(
                            builder.build_reduced(density)[0], plain)
                        with self.assertRaises(ValueError):
                            builder.build_reduced_device(
                                np.full(reduction.wedge_size, np.nan))

    def test_boundary_check_measures_the_values_against_the_direct_sum(self) -> None:
        import parsec_python.acceleration.Hartree.cupy_atomic_tail as tail_module
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            CuPyMultipoleBoundaryBuilder,
            CuPyPointMultipoleBoundaryBuilder,
            CuPySymmetryMultipoleBoundaryBuilder,
        )
        from parsec_python.Hartree import DirectCoulombBoundary

        grid = sphere_grid((0.0, 0.0, 0.0))
        positions, charges = ARRANGEMENTS["D2"], np.full(4, 0.75)
        reduction = AxisReflectionReduction.detect(
            grid, tuple(Atom("H", position) for position in positions))
        density = invariant_density(grid, reduction)
        order = 12
        expansion = density_multipoles_fast(density, grid, order)
        wedge = CuPySymmetryMultipoleBoundaryBuilder(grid, reduction, order)
        wedge.set_atomic_tail(positions, charges)
        full = CuPyMultipoleBoundaryBuilder(grid, order)
        point = CuPyPointMultipoleBoundaryBuilder(grid, order)
        point.set_atomic_tail(positions, charges)
        # Several launches of the direct sum, as a large grid takes.
        with patch.object(tail_module, "_DIRECT_SOURCES_PER_LAUNCH", 37):
            checks = {
                "wedge": wedge.boundary_check(
                    density[reduction.representative_rows], 40, minimum_order=4,
                    positions=positions),
                "full": full.boundary_check(
                    density, 40, minimum_order=4, tail_charges=(positions, charges)),
                "point": point.boundary_check(
                    density, 40, minimum_order=4, positions=positions),
            }
        for name, check in checks.items():
            with self.subTest(builder=name):
                points = check["points"]
                self.assertEqual(points.shape, (40, 3))
                self.assertTrue(
                    (np.linalg.norm(points, axis=1) > grid.settings.radius - 1.0e-9).all())
                direct = DirectCoulombBoundary(
                    grid.coordinates, density * grid.volume_element).potential(points)
                np.testing.assert_allclose(check["direct"], direct, rtol=1.0e-12)
                tail = host_tail_values(points, positions, charges, order)
                np.testing.assert_allclose(check["tail"], tail, rtol=0, atol=1.0e-11)
                np.testing.assert_allclose(
                    check["boundary"], expansion.potential(points) + tail,
                    rtol=0, atol=1.0e-11)
                np.testing.assert_allclose(
                    check["legacy"],
                    density_multipoles_fast(density, grid, 4).potential(points),
                    rtol=0, atol=1.0e-11)
        # Each point of the full grid once, by the same rule in both
        # full-grid builders: the same sample, bit for bit.
        self.assertEqual(checks["full"]["total"], point.exterior_point_count)
        self.assertLess(checks["wedge"]["total"], checks["full"]["total"])
        self.assertEqual(checks["wedge"]["total"], wedge.exterior_point_count)
        np.testing.assert_array_equal(checks["full"]["points"], checks["point"]["points"])
        # A quarter of the sample is innermost, a quarter has the largest
        # tail of the rest, a quarter is nearest to an atom.
        radius = wedge.exterior[0].get()
        tail = np.abs(wedge.tail.get())
        innermost = np.argsort(radius, kind="stable")[:10]
        np.testing.assert_allclose(
            np.linalg.norm(checks["wedge"]["points"][:10], axis=1), radius[innermost],
            rtol=1.0e-14)
        tail[innermost] = 0.0
        np.testing.assert_array_equal(
            np.sort(np.abs(checks["wedge"]["tail"][10:20])), np.sort(tail)[-10:])
        distance = np.linalg.norm(
            checks["wedge"]["points"][:, None, :] - positions[None, :, :], axis=2).min(axis=1)
        self.assertLess(distance[20:30].max(), np.median(distance[30:]))
        # A count above the number of points takes every one of them.
        everywhere = wedge.boundary_check(
            density[reduction.representative_rows], 10**6, positions=positions)
        self.assertEqual(everywhere["points"].shape[0], wedge.exterior_point_count)
        self.assertGreaterEqual(
            np.abs(everywhere["boundary"] - everywhere["direct"]).max(),
            np.abs(checks["wedge"]["boundary"] - checks["wedge"]["direct"]).max())

    def test_point_builder_matches_the_full_grid_builders(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            CuPyMultipoleBoundaryBuilder,
            CuPyPointMultipoleBoundaryBuilder,
        )

        cp, _ = require_cupy()
        positions, charges = ARRANGEMENTS["C1"], np.array([1.4, 0.6])
        for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)):
            grid = sphere_grid(shift)
            xyz = grid.coordinates
            # No symmetry at all.
            density = symmetric_density(grid) * (
                1.0 + 0.2 * np.tanh(xyz[:, 0] + 0.5 * xyz[:, 1] - 0.3 * xyz[:, 2]))
            for order in (9, 34):
                with self.subTest(shift=shift, order=order):
                    settings = HartreeSettings(multipole_order=order)
                    tail = build_atomic_tail_fast(grid, positions, charges, order)
                    builder = CuPyPointMultipoleBoundaryBuilder(grid, order)
                    plain, _ = builder.build(density)
                    np.testing.assert_allclose(
                        plain, build_hartree_problem(density, grid, settings, None)[0],
                        rtol=3.0e-12, atol=3.0e-12)
                    builder.set_atomic_tail(positions, charges)
                    self.assertAlmostEqual(
                        builder.atomic_tail_maximum, tail.maximum, places=10)
                    actual, boundary = builder.build(density)
                    np.testing.assert_allclose(
                        actual, build_hartree_problem(density, grid, settings, tail)[0],
                        rtol=3.0e-12, atol=3.0e-12)
                    full = CuPyMultipoleBoundaryBuilder(grid, order)
                    full.set_boundary_tail(tail)
                    full_rhs, full_boundary = full.build(density)
                    np.testing.assert_allclose(actual, full_rhs, rtol=3.0e-12, atol=3.0e-12)
                    weight = np.abs(density) * grid.volume_element
                    radius = np.linalg.norm(xyz, axis=1)
                    for key, value in full_boundary.moments.items():
                        self.assertLess(
                            abs(boundary.moments[key] - value),
                            1.0e-11 * np.sum(weight * radius ** key[0]))
                    # The full-grid interface: the same bits from a device
                    # density, and no wedge entry for the resident chain.
                    device_rhs, _ = builder.build_device(cp.asarray(density))
                    np.testing.assert_array_equal(cp.asnumpy(device_rhs), actual)
                    self.assertFalse(hasattr(builder, "build_reduced_device"))
                    # A point is shared by several stencil entries.
                    self.assertLess(
                        3 * builder.exterior_point_count, builder.boundary_term_count)
                    if order == 34:
                        self.assertLess(
                            builder.device_storage_bytes, full.device_storage_bytes)

    def test_forbidden_moments_are_exact_zeros(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            CuPySymmetryMultipoleBoundaryBuilder,
        )

        grid = sphere_grid()
        atoms = tuple(Atom("H", position) for position in ARRANGEMENTS["D2"])
        reduction = AxisReflectionReduction.detect(grid, atoms)
        _rhs, boundary = CuPySymmetryMultipoleBoundaryBuilder(
            grid, reduction, 9).build_reduced(invariant_density(grid, reduction))
        for (l, m), value in boundary.moments.items():
            if m % 2:
                self.assertEqual(value, 0.0)
            elif l % 2:
                self.assertEqual(value.real, 0.0)
            else:
                self.assertEqual(value.imag, 0.0)
        self.assertNotEqual(boundary.moments[(3, 2)].imag, 0.0)

    def test_resident_chain_with_either_builder_matches_the_host_interface(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            CuPyMultipoleBoundaryBuilder,
            CuPyPointMultipoleBoundaryBuilder,
            CuPySymmetryMultipoleBoundaryBuilder,
        )
        from parsec_python.acceleration.Hartree.cupy_prepared import (
            CuPyPreparedPoissonSolver,
        )
        from parsec_python.acceleration.Hartree.cupy_resident import CuPyResidentHartree
        from parsec_python.acceleration.Hartree.symmetry_poisson import (
            SymmetryReducedPoissonSolver,
        )
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetryScalarField

        cp, _ = require_cupy()
        grid = sphere_grid((0.0, 0.0, 0.0), spacing=0.8, radius=4.1, order=4)
        positions, charges = ARRANGEMENTS["D2"], np.full(4, 0.75)
        reduction = AxisReflectionReduction.detect(
            grid, tuple(Atom("H", position) for position in positions))
        order = 12
        tail = build_atomic_tail_fast(grid, positions, charges, order)
        controls = HartreeSettings(
            multipole_order=order, relative_tolerance=1e-12, absolute_tolerance=1e-13)
        baseline = SymmetryReducedPoissonSolver(
            build_negative_laplacian(grid), reduction,
            solver_factory=CuPyPreparedPoissonSolver)
        full_builder = CuPyMultipoleBoundaryBuilder(grid, order)
        full_builder.set_boundary_tail(tail)
        wedge_builder = CuPySymmetryMultipoleBoundaryBuilder(grid, reduction, order)
        wedge_builder.set_atomic_tail(positions, charges)
        point_builder = CuPyPointMultipoleBoundaryBuilder(grid, order)
        point_builder.set_atomic_tail(positions, charges)
        density = invariant_density(grid, reduction)
        for builder in (full_builder, wedge_builder, point_builder):
            for predictor in ("host", "device"):
                with self.subTest(builder=type(builder).__name__, predictor=predictor):
                    solver = CuPyPreparedPoissonSolver(baseline.reduced_negative_laplacian)
                    with patch.dict(os.environ, {"PARSEC_CUPY_RESIDENT_PREDICTOR": predictor}):
                        resident = CuPyResidentHartree(
                            builder, reduction, solver.backend, controls)
                    self.assertEqual(resident.map is None, builder is wedge_builder)
                    initial = None
                    for factor in (1.0, 1.03, 1.05, 0.98):
                        rhs = self.reference_wedge_rhs(
                            grid, reduction, factor * density, order, tail)
                        expected = baseline.solve_reduced(
                            rhs, initial, controls, return_wedge=True)
                        field = SymmetryScalarField(
                            reduction, factor * density[reduction.representative_rows])
                        actual = resident.solve(field, initial)
                        self.assertTrue(actual.converged)
                        np.testing.assert_allclose(
                            actual.right_hand_side.values,
                            expected.right_hand_side.values, rtol=0, atol=3e-12)
                        np.testing.assert_allclose(
                            actual.potential.values, expected.potential.values,
                            rtol=0, atol=2e-10)
                        initial = expected.potential
                    if predictor == "device":
                        # The predictor extrapolates from the two last
                        # right-hand sides: they are two arrays, and neither
                        # is the buffer the wedge builder lends.
                        (older, _), (newer, _) = resident.history
                        self.assertNotEqual(older.data.ptr, newer.data.ptr)
                        self.assertGreater(float(cp.abs(newer - older).max()), 0.0)
                        if builder is wedge_builder:
                            self.assertNotEqual(newer.data.ptr, builder.rhs.data.ptr)
                    # A full-grid density takes the lazily uploaded maps.
                    full_density = resident.solve(1.01 * density, None)
                    wedge_density = resident.solve(SymmetryScalarField(
                        reduction, 1.01 * density[reduction.representative_rows]), None)
                    np.testing.assert_allclose(
                        full_density.right_hand_side.values,
                        wedge_density.right_hand_side.values, rtol=0, atol=3e-12)
                    self.assertIsNotNone(resident.members)


    def test_driver_runs_the_engaged_boundary_on_every_builder(self) -> None:
        problem = hydrogen_problem(multipole_order=2)
        problem = replace(
            problem,
            atoms=[Atom("H", 1.3 * position) for position in ARRANGEMENTS["D2"]],
            scf=replace(problem.scf, max_iterations=3),
        )
        gpu_hartree = {
            "PARSEC_HARTREE_LINEAR_BACKEND": "cupy",
            "PARSEC_HARTREE_BOUNDARY_BACKEND": "cupy",
            "PARSEC_CUPY_RESIDENT_HARTREE": "1",
        }

        def run(**environment):
            with patch.dict(
                os.environ, {**clean_environment(), **gpu_hartree, **environment},
                clear=True,
            ):
                system = driver_module.prepare_single_point(problem, backend="auto")
                details = dict(system.backend_info.details)
                return driver_module.run_scf(system), details

        wedge, wedge_details = run()
        self.assertTrue(wedge_details["hartree_boundary_kernel"].startswith("wedge"))
        self.assertNotIn("hartree_boundary_check_max", dict(wedge.backend.details))
        # The check after the SCF measures and changes nothing.
        checked, _ = run(PARSEC_HARTREE_BOUNDARY_CHECK="64")
        np.testing.assert_array_equal(checked.hartree_potential, wedge.hartree_potential)
        self.assertEqual(checked.energies.total, wedge.energies.total)
        measured = dict(checked.backend.details)
        self.assertEqual(measured["hartree_boundary_check_points"], "64")
        self.assertLess(
            float(measured["hartree_boundary_check_max"].split()[0]),
            0.1 * float(measured["hartree_boundary_check_legacy_max"].split()[0]))
        self.assertEqual(wedge_details["hartree_atomic_tail"], "applied")
        self.assertIn("hartree_atomic_tail_max", wedge_details)
        self.assertEqual(
            wedge_details["hartree_full_grid_transfer_policy"], "device-resident")
        full, full_details = run(
            PARSEC_HARTREE_BOUNDARY_KERNEL="full", PARSEC_HARTREE_BOUNDARY_CHECK="64")
        self.assertTrue(full_details["hartree_boundary_kernel"].startswith("full"))
        self.assertAlmostEqual(
            float(dict(full.backend.details)["hartree_boundary_check_legacy_max"].split()[0]),
            float(measured["hartree_boundary_check_legacy_max"].split()[0]), delta=0.5 * float(
                measured["hartree_boundary_check_legacy_max"].split()[0]))
        self.assertLess(
            int(wedge_details["hartree_boundary_device_bytes"]),
            int(full_details["hartree_boundary_device_bytes"]))
        host_interface, _ = run(PARSEC_CUPY_RESIDENT_HARTREE="0")
        native, native_details = run(
            PARSEC_HARTREE_BOUNDARY_BACKEND="native", PARSEC_CUPY_RESIDENT_HARTREE="0")
        self.assertTrue(native_details["hartree_boundary_kernel"].startswith("native"))
        self.assertAlmostEqual(
            float(native_details["hartree_atomic_tail_max"].split()[0]),
            float(wedge_details["hartree_atomic_tail_max"].split()[0]), places=10)
        for other in (full, host_interface, native):
            self.assertAlmostEqual(other.energies.total, wedge.energies.total, places=9)
            np.testing.assert_allclose(
                other.hartree_potential, wedge.hartree_potential, rtol=0, atol=1e-8)
        legacy, legacy_details = run(PARSEC_HARTREE_BOUNDARY="legacy")
        self.assertTrue(legacy_details["hartree_boundary_kernel"].startswith("full"))
        self.assertEqual(legacy_details["hartree_boundary"], "PARSEC multipole expansion")
        self.assertGreater(abs(legacy.energies.total - wedge.energies.total), 1e-5)
        self.assertNotIn("hartree_atomic_tail_values", legacy_details)
        # The tail from host threads, per point of the wedge and in the rows
        # that a serial control adds: the values of the kernel to round-off.
        self.assertEqual(wedge_details["hartree_atomic_tail_values"], "device kernel")
        for kernel in ("auto", "full"):
            with self.subTest(kernel=kernel):
                hosted, hosted_details = run(
                    PARSEC_HARTREE_ATOMIC_TAIL_VALUES="host",
                    PARSEC_HARTREE_BOUNDARY_KERNEL=kernel)
                self.assertEqual(
                    hosted_details["hartree_atomic_tail_values"], "host threads")
                self.assertTrue(hosted_details["hartree_boundary_kernel"].startswith(
                    "wedge" if kernel == "auto" else "full"))
                self.assertAlmostEqual(
                    float(hosted_details["hartree_atomic_tail_max"].split()[0]),
                    float(wedge_details["hartree_atomic_tail_max"].split()[0]), places=10)
                self.assertAlmostEqual(
                    hosted.energies.total, wedge.energies.total, places=9)
                np.testing.assert_allclose(
                    hosted.hartree_potential, wedge.hartree_potential, rtol=0, atol=1e-8)

    def test_wedge_builder_tail_and_chain_on_another_device_return_the_same_bits(
        self,
    ) -> None:
        # The root rank of an MPI run builds the boundary on its Hartree
        # device, the last of the rank, from a thread whose current device
        # is the first; the resident chain solves there.
        from functools import partial

        from parsec_python.acceleration.Hartree.cupy_atomic_tail import device_tail_values
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            CuPyPointMultipoleBoundaryBuilder,
            CuPySymmetryMultipoleBoundaryBuilder,
        )
        from parsec_python.acceleration.Hartree.cupy_prepared import (
            CuPyPreparedPoissonSolver,
        )
        from parsec_python.acceleration.Hartree.cupy_resident import CuPyResidentHartree
        from parsec_python.acceleration.Hartree.symmetry_poisson import (
            SymmetryReducedPoissonSolver,
        )
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetryScalarField

        cp, _ = require_cupy()
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 2:
            self.skipTest("requires at least two allocated GPUs")
        home = int(cp.cuda.Device().id)
        other = max(device for device in range(device_count) if device != home)
        grid = sphere_grid((0.0, 0.0, 0.0), spacing=0.8, radius=4.1, order=4)
        positions, charges = ARRANGEMENTS["D2"], np.full(4, 0.75)
        reduction = AxisReflectionReduction.detect(
            grid, tuple(Atom("H", position) for position in positions))
        order = 12
        density = invariant_density(grid, reduction)
        wedge_density = density[reduction.representative_rows]

        # The kernel of the tail in rows, on the device its caller names.
        points = grid.physical_coordinates(missing_stencil_entries(grid)[3])
        np.testing.assert_array_equal(
            device_tail_values(points, positions, charges, order, device_id=other),
            device_tail_values(points, positions, charges, order))
        self.assertEqual(int(cp.cuda.Device().id), home)

        # Without a device the builder takes the current one.
        first = CuPySymmetryMultipoleBoundaryBuilder(grid, reduction, order)
        moved = CuPySymmetryMultipoleBoundaryBuilder(
            grid, reduction, order, device_id=other)
        for builder in (first, moved):
            builder.set_atomic_tail(positions, charges)
        self.assertEqual((first.device_id, moved.device_id), (home, other))
        for array in (
            *moved.source, *moved.exterior, moved.weight, moved.root, moved.indptr,
            moved.coefficient, moved.point, moved.norm, moved.cls, moved.m_list,
            moved.partial, moved.density, moved.rhs, moved.value, moved.tail,
        ):
            self.assertEqual(int(array.device.id), other)
        self.assertEqual(moved.device_storage_bytes, first.device_storage_bytes)
        self.assertEqual(moved.atomic_tail_maximum, first.atomic_tail_maximum)
        self.assertEqual(int(cp.cuda.Device().id), home)
        with cp.cuda.Device(other):
            moved_tail = cp.asnumpy(moved.tail)
        np.testing.assert_array_equal(moved_tail, cp.asnumpy(first.tail))
        field = SymmetryScalarField(reduction, wedge_density)
        expected_rhs, expected_boundary = first.build_reduced(field)
        actual_rhs, actual_boundary = moved.build_reduced(field)
        np.testing.assert_array_equal(actual_rhs, expected_rhs)
        self.assertEqual(actual_boundary.moments, expected_boundary.moments)
        # A wedge density already on that device, handed over while another
        # device is current, as the resident chain passes it.
        with cp.cuda.Device(other):
            on_other = cp.asarray(wedge_density)
        device_rhs, device_boundary = moved.build_reduced_device(on_other)
        self.assertEqual(int(device_rhs.device.id), other)
        with cp.cuda.Device(other):
            np.testing.assert_array_equal(cp.asnumpy(device_rhs), expected_rhs)
        self.assertEqual(device_boundary.moments, expected_boundary.moments)
        self.assertEqual(int(cp.cuda.Device().id), home)

        # The same kernels under the identity alone.
        point_rhs = []
        for device in (None, other):
            point = CuPyPointMultipoleBoundaryBuilder(grid, order, device_id=device)
            point.set_atomic_tail(positions, charges)
            point_rhs.append(point.build(density)[0])
        np.testing.assert_array_equal(point_rhs[1], point_rhs[0])
        self.assertEqual(int(cp.cuda.Device().id), home)

        matrix = build_negative_laplacian(grid)
        controls = HartreeSettings(
            multipole_order=order, relative_tolerance=1e-12, absolute_tolerance=1e-13)

        def chain(builder, predictor):
            # CG arrays, boundary geometry and resident maps on one device.
            solver = SymmetryReducedPoissonSolver(
                matrix, reduction,
                solver_factory=partial(
                    CuPyPreparedPoissonSolver, device_id=builder.device_id))
            with patch.dict(os.environ, {"PARSEC_CUPY_RESIDENT_PREDICTOR": predictor}):
                return CuPyResidentHartree(
                    builder, reduction, solver.solver.backend, controls)

        for predictor in ("host", "device"):
            with self.subTest(predictor=predictor):
                expected_chain, moved_chain = chain(first, predictor), chain(moved, predictor)
                self.assertEqual(moved_chain.backend.device_id, other)
                # The wedge branch: neither full-grid map is uploaded.
                self.assertIsNone(moved_chain.map)
                for array in (moved_chain.roots, moved_chain.ptr):
                    self.assertEqual(int(array.device.id), other)
                initial = None
                for factor in (1.0, 1.03, 1.05, 0.98):
                    scaled = SymmetryScalarField(reduction, factor * wedge_density)
                    expected = expected_chain.solve(scaled, initial)
                    actual = moved_chain.solve(scaled, initial)
                    self.assertTrue(expected.converged)
                    np.testing.assert_array_equal(
                        actual.potential.values, expected.potential.values)
                    np.testing.assert_array_equal(
                        actual.right_hand_side.values, expected.right_hand_side.values)
                    for name in (
                        "converged", "iterations", "matrix_vector_products",
                        "residual_norm", "initial_residual_norm",
                    ):
                        self.assertEqual(getattr(actual, name), getattr(expected, name))
                    self.assertEqual(actual.boundary.moments, expected.boundary.moments)
                    self.assertEqual(int(cp.cuda.Device().id), home)
                    initial = expected.potential
                self.assertIsNone(moved_chain.members)

    def test_engaged_boundary_on_another_device_leaves_the_scf_bitwise_unchanged(
        self,
    ) -> None:
        cp, _ = require_cupy()
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 2:
            self.skipTest("requires at least two allocated GPUs")
        problem = hydrogen_problem(multipole_order=2)
        problem = replace(
            problem,
            atoms=[Atom("H", 1.3 * position) for position in ARRANGEMENTS["D2"]],
            scf=replace(problem.scf, max_iterations=3),
        )
        gpu_hartree = {
            "PARSEC_HARTREE_LINEAR_BACKEND": "cupy",
            "PARSEC_HARTREE_BOUNDARY_BACKEND": "cupy",
            "PARSEC_CUPY_RESIDENT_HARTREE": "1",
            "PARSEC_CUPY_DEVICES": ",".join(map(str, range(device_count))),
        }

        def run(device, **environment):
            with patch.dict(
                os.environ,
                {**clean_environment(), **gpu_hartree, "PARSEC_HARTREE_DEVICE": device,
                 **environment},
                clear=True,
            ):
                system = driver_module.prepare_single_point(problem, backend="auto")
                details = dict(system.backend_info.details)
                return driver_module.run_scf(system), details

        last = str(device_count - 1)
        for environment, kernel, values in (
            ({}, "wedge", "device kernel"),
            # What a serial control of the MPI runner names.
            ({"PARSEC_HARTREE_BOUNDARY_KERNEL": "full",
              "PARSEC_HARTREE_ATOMIC_TAIL_VALUES": "host"}, "full", "host threads"),
        ):
            with self.subTest(kernel=kernel):
                expected, former = run("0", **environment)
                actual, details = run(last, **environment)
                self.assertEqual(
                    (former["hartree_boundary_device"], details["hartree_boundary_device"]),
                    ("0", last))
                self.assertEqual(details["hartree_cg_device"], last)
                self.assertTrue(details["hartree_boundary_kernel"].startswith(kernel))
                self.assertEqual(details["hartree_atomic_tail_values"], values)
                self.assertEqual(
                    details["hartree_full_grid_transfer_policy"], "device-resident")
                for name in (
                    "hartree_boundary_device_bytes", "hartree_atomic_tail_max",
                    "hartree_multipole_order",
                ):
                    self.assertEqual(details[name], former[name])
                np.testing.assert_array_equal(
                    actual.hartree_potential, expected.hartree_potential)
                np.testing.assert_array_equal(actual.eigenvalues, expected.eigenvalues)
                np.testing.assert_array_equal(actual.density, expected.density)
                self.assertEqual(actual.energies, expected.energies)

    def test_set_up_on_its_thread_leaves_a_capture_of_the_device_valid(self) -> None:
        # The driver builds the boundary on a thread beside the GPU orbital
        # set-up and joins it after the Poisson solver has recorded its
        # graph, in the relaxed mode, on the same device.  The geometry of
        # the wedge, the kernel of the tail and the end of that thread fall
        # inside such a capture here; it is not repeated.
        from concurrent.futures import ThreadPoolExecutor

        from parsec_python.acceleration.backends.cupy_capture import (
            collector_paused,
            end_failed_capture,
        )
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            CuPySymmetryMultipoleBoundaryBuilder,
        )

        cp, _ = require_cupy()
        grid = sphere_grid()
        positions, charges = ARRANGEMENTS["D2"], np.full(4, 0.75)
        reduction = AxisReflectionReduction.detect(
            grid, tuple(Atom("H", position) for position in positions))
        density = invariant_density(grid, reduction)
        device = int(cp.cuda.Device().id)

        def set_up():
            builder = CuPySymmetryMultipoleBoundaryBuilder(
                grid, reduction, 34, device_id=device)
            builder.set_atomic_tail(positions, charges)
            return builder

        expected = set_up().build_reduced(density)[0]
        bump = cp.RawKernel(
            'extern "C" __global__ void bump(double* x, int n) {'
            " int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += 1.0; }",
            "bump")
        values = cp.zeros(1024)
        count = np.int32(values.size)
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            bump((4,), (256,), (values, count))
        stream.synchronize()
        graph = None
        with collector_paused(), stream:
            stream.begin_capture(mode=cp.cuda.runtime.streamCaptureModeRelaxed)
            try:
                bump((4,), (256,), (values, count))
                executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="parsec-hartree-setup")
                builder = executor.submit(set_up).result()
                # The thread ends here, as at the join of the driver.
                executor.shutdown(wait=True)
                bump((4,), (256,), (values, count))
                graph = stream.end_capture()
            finally:
                if graph is None:
                    end_failed_capture(stream)
            graph.launch(stream)
        stream.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(values), np.full(values.size, 3.0))
        np.testing.assert_array_equal(builder.build_reduced(density)[0], expected)

    def test_driver_runs_the_engaged_boundary_without_a_reflection_group(self) -> None:
        # Four atoms that no reflection maps onto each other.
        problem = hydrogen_problem(multipole_order=2)
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))
        gpu_hartree = {
            "PARSEC_HARTREE_LINEAR_BACKEND": "cupy",
            "PARSEC_HARTREE_BOUNDARY_BACKEND": "cupy",
        }

        def run(**environment):
            with patch.dict(
                os.environ, {**clean_environment(), **gpu_hartree, **environment},
                clear=True,
            ):
                system = driver_module.prepare_single_point(problem, backend="auto")
                details = dict(system.backend_info.details)
                return driver_module.run_scf(system), details

        point, point_details = run(PARSEC_HARTREE_BOUNDARY_CHECK="64")
        self.assertEqual(point_details["hartree_symmetry"], "full grid")
        self.assertTrue(point_details["hartree_boundary_kernel"].startswith("points"))
        self.assertEqual(point_details["hartree_atomic_tail"], "applied")
        self.assertIn("hartree_atomic_tail_max", point_details)
        full, full_details = run(
            PARSEC_HARTREE_BOUNDARY_KERNEL="full", PARSEC_HARTREE_BOUNDARY_CHECK="64")
        self.assertTrue(full_details["hartree_boundary_kernel"].startswith("full"))
        native, native_details = run(PARSEC_HARTREE_BOUNDARY_BACKEND="native")
        self.assertTrue(native_details["hartree_boundary_kernel"].startswith("native full"))
        for other in (full, native):
            self.assertAlmostEqual(other.energies.total, point.energies.total, places=9)
            np.testing.assert_allclose(
                other.hartree_potential, point.hartree_potential, rtol=0, atol=1e-8)
        # The two GPU builders check the same sample of the same points.
        measured, measured_full = dict(point.backend.details), dict(full.backend.details)
        self.assertEqual(
            measured["hartree_boundary_check_sample"],
            measured_full["hartree_boundary_check_sample"])
        self.assertIn("64 of ", measured["hartree_boundary_check_sample"])
        for name in ("hartree_boundary_check_max", "hartree_boundary_check_legacy_max"):
            self.assertAlmostEqual(
                float(measured[name].split()[0]), float(measured_full[name].split()[0]),
                delta=1.0e-9)
        self.assertLess(
            float(measured["hartree_boundary_check_max"].split()[0]),
            0.1 * float(measured["hartree_boundary_check_legacy_max"].split()[0]))
        # No boundary backend named: the native builder has no orbit table
        # without a reduction, and the GPU kernels take the raised order.
        unnamed, unnamed_details = run(PARSEC_HARTREE_BOUNDARY_BACKEND="")
        self.assertTrue(unnamed_details["hartree_boundary_kernel"].startswith("points"))
        np.testing.assert_array_equal(unnamed.hartree_potential, point.hartree_potential)
        legacy, legacy_details = run(
            PARSEC_HARTREE_BOUNDARY_BACKEND="", PARSEC_HARTREE_BOUNDARY="legacy")
        self.assertTrue(legacy_details["hartree_boundary_kernel"].startswith("native full"))
        self.assertGreater(abs(legacy.energies.total - point.energies.total), 1e-5)


@unittest.skipUnless(
    cupy_available() and native_boundary_ready(),
    "needs a CUDA device and the native extension built from this tree",
)
class CuPyAtomicTailTests(unittest.TestCase):
    def test_device_kernel_matches_the_host_values(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_atomic_tail import (
            device_coulomb_sum,
            device_tail_values,
        )

        points = 4.4 * np.random.default_rng(2).normal(size=(3000, 3))
        points = points[np.linalg.norm(points, axis=1) > 3.9]
        direct = point_charge_potential(points, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES)
        np.testing.assert_allclose(
            device_coulomb_sum(points, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES),
            direct, rtol=1.0e-12)
        for order in (0, 9):
            with self.subTest(order=order):
                np.testing.assert_allclose(
                    device_tail_values(
                        points, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, order),
                    host_tail_values(
                        points, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, order),
                    rtol=0.0, atol=1.0e-12 * np.abs(direct).max())

    def test_builders_per_exterior_point_take_the_tail_from_host_threads(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            CuPyPointMultipoleBoundaryBuilder,
            CuPySymmetryMultipoleBoundaryBuilder,
        )

        cp, _ = require_cupy()
        grid = sphere_grid()
        positions, charges = ARRANGEMENTS["D2"], np.full(4, 0.75)
        reduction = AxisReflectionReduction.detect(
            grid, tuple(Atom("H", position) for position in positions))
        density = invariant_density(grid, reduction)
        for builder in (
            CuPySymmetryMultipoleBoundaryBuilder(grid, reduction, 9),
            CuPyPointMultipoleBoundaryBuilder(grid, 9),
        ):
            with self.subTest(builder=type(builder).__name__):
                wedge = getattr(builder, "_wedge", builder)
                build = getattr(builder, "build_reduced", None) or builder.build
                self.assertIsNone(builder.atomic_tail_values)
                builder.set_atomic_tail(positions, charges)
                self.assertEqual(builder.atomic_tail_values, "device kernel")
                kernel_tail = cp.asnumpy(wedge.tail)
                kernel_bytes = builder.device_storage_bytes
                kernel_rhs = build(density)[0].copy()
                asked = []

                def host_values(points, *charges_and_order):
                    asked.append(points.shape)
                    return host_tail_values(points, *charges_and_order)

                builder.set_atomic_tail(positions, charges, host_values)
                self.assertEqual(asked, [(builder.exterior_point_count, 3)])
                self.assertEqual(builder.atomic_tail_values, "host threads")
                host_tail = cp.asnumpy(wedge.tail)
                with cp.cuda.Device(builder.device_id):
                    points = cp.asnumpy(wedge.exterior_points_device())
                scale = np.abs(point_charge_potential(points, positions, charges)).max()
                np.testing.assert_allclose(
                    host_tail, kernel_tail, rtol=0.0, atol=1.0e-12 * scale)
                self.assertEqual(builder.device_storage_bytes, kernel_bytes)
                self.assertAlmostEqual(
                    builder.atomic_tail_maximum, float(np.abs(host_tail).max()), places=14)
                np.testing.assert_allclose(
                    build(density)[0], kernel_rhs, rtol=3.0e-12, atol=3.0e-12)
                builder.set_atomic_tail(None, None)
                self.assertIsNone(builder.atomic_tail_values)

    def test_pure_cupy_poisson_solves_with_the_tail_of_the_plan(self) -> None:
        cp, _ = require_cupy()
        problem = hydrogen_problem(multipole_order=2)
        reference = prepare_reference(problem)
        self.assertTrue(reference.hartree_boundary.atomic_tail)
        density = reference.initial_density
        expected = reference.solve_hartree(density)

        def right_hand_side(**environment):
            with patch.dict(os.environ, {**clean_environment(), **environment}, clear=True):
                system = driver_module.prepare_single_point(
                    problem, backend="cupy", symmetry="off")
                details = dict(system.backend_info.details)
                self.assertEqual(details["hartree_backend"], "cupy")
                result = system.solve_hartree(density)
            return result, np.asarray(
                cp.asnumpy(result.right_hand_side)
                if isinstance(result.right_hand_side, cp.ndarray)
                else result.right_hand_side), details

        result, actual, details = right_hand_side()
        np.testing.assert_allclose(
            actual, expected.right_hand_side, rtol=3.0e-12, atol=3.0e-12)
        self.assertEqual(result.boundary.order, reference.hartree_boundary.order)
        self.assertEqual(details["hartree_atomic_tail_values"], "device kernel")
        self.assertEqual(result.boundary_tail.values_from, "device kernel")
        # Host threads where they are asked for: the kernel is not launched.
        with patch(
            "parsec_python.acceleration.Hartree.cupy_atomic_tail.device_tail_values",
            side_effect=AssertionError("the device kernel of the tail was asked"),
        ):
            hosted_result, hosted, details = right_hand_side(
                PARSEC_HARTREE_ATOMIC_TAIL_VALUES="host")
        self.assertEqual(details["hartree_atomic_tail_values"], "host threads")
        self.assertEqual(hosted_result.boundary_tail.values_from, "host threads")
        np.testing.assert_allclose(
            hosted, expected.right_hand_side, rtol=3.0e-12, atol=3.0e-12)
        np.testing.assert_allclose(
            hosted_result.boundary_tail.values, result.boundary_tail.values,
            rtol=0.0, atol=1.0e-11 * np.abs(result.boundary_tail.values).max())
        # The rows of the tail are far above that tolerance, and they are
        # what the legacy switch takes away.
        self.assertGreater(
            np.abs(reference.hartree_boundary_tail.values).max(),
            1.0e-6 * np.abs(expected.right_hand_side).max())
        _, legacy, _ = right_hand_side(PARSEC_HARTREE_BOUNDARY="legacy")
        np.testing.assert_allclose(
            legacy, legacy_fast_right_hand_side(density, reference.grid, 2),
            rtol=3.0e-12, atol=3.0e-12)

    def test_full_builder_and_resident_chain_add_the_tail(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            CuPyMultipoleBoundaryBuilder,
        )

        cp, _ = require_cupy()
        grid = sphere_grid()
        density = symmetric_density(grid)
        tail = build_atomic_tail_fast(grid, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, 9)
        builder = CuPyMultipoleBoundaryBuilder(grid, 9)
        before, _ = builder.build(density)
        bytes_before = builder.device_storage_bytes
        builder.set_boundary_tail(tail)
        self.assertGreater(builder.device_storage_bytes, bytes_before)
        with_tail, _ = builder.build(density)
        added = before.copy()
        added[tail.rows] += tail.values
        np.testing.assert_array_equal(with_tail, added)
        expected, _ = build_hartree_problem(
            density, grid, HartreeSettings(multipole_order=9), tail)
        np.testing.assert_allclose(with_tail, expected, rtol=3.0e-12, atol=3.0e-12)
        device_rhs, _ = builder.build_device(cp.asarray(density))
        np.testing.assert_array_equal(cp.asnumpy(device_rhs), with_tail)
        builder.set_boundary_tail(None)
        self.assertEqual(builder.device_storage_bytes, bytes_before)
        np.testing.assert_array_equal(builder.build(density)[0], before)
        with self.assertRaisesRegex(ValueError, "multipole order 4"):
            builder.set_boundary_tail(build_atomic_tail_fast(
                grid, SYMMETRIC_POSITIONS, SYMMETRIC_CHARGES, 4))


if __name__ == "__main__":
    unittest.main()
