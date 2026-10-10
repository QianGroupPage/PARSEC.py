"""PARSEC_FAST_GRID: the cluster grid slab by slab, against the builder of the reference package."""

from __future__ import annotations

import os
from pathlib import Path
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.Grid.cluster import _inside_domain
from parsec_python.acceleration.Grid import (
    build_cluster_grid_by_slabs,
    fast_grid_requested,
)
from parsec_python.acceleration.Grid import cluster as fast_cluster
from parsec_python.models import GridSettings

_ARRAYS = ("integer_coordinates", "coordinates", "index_min", "index_max", "lookup")
SMOKE_INPUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "H_cli_smoke.in"


class _ShellRecorder:
    """The reference test, with the points it was asked about counted."""

    def __init__(self, answer=None) -> None:
        self.points = 0
        self.answer = answer

    def __call__(self, settings, coordinates):
        self.points += coordinates.shape[0]
        decided = _inside_domain(settings, coordinates)
        return decided if self.answer is None else np.full_like(decided, self.answer)


class FastGridTests(unittest.TestCase):
    def assert_same_grid(self, settings: GridSettings):
        """Compare every array with the reference one: type, shape, layout and bytes."""

        reference = build_cluster_grid(settings)
        grid = build_cluster_grid_by_slabs(settings)
        self.assertIs(grid.settings, settings)
        for name in _ARRAYS:
            expected, actual = getattr(reference, name), getattr(grid, name)
            self.assertEqual(actual.dtype, expected.dtype, name)
            self.assertEqual(actual.shape, expected.shape, name)
            self.assertEqual(actual.strides, expected.strides, name)
            for flag in ("c_contiguous", "owndata", "writeable", "aligned"):
                self.assertEqual(getattr(actual.flags, flag), getattr(expected.flags, flag), (name, flag))
            self.assertEqual(actual.tobytes(), expected.tobytes(), name)
        return reference

    def test_switch_is_on_unless_named_off(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("PARSEC_FAST_GRID", None)
            self.assertTrue(fast_grid_requested())
            for value in ("0", "false", "No", " off "):
                os.environ["PARSEC_FAST_GRID"] = value
                self.assertFalse(fast_grid_requested())
            os.environ["PARSEC_FAST_GRID"] = "1"
            self.assertTrue(fast_grid_requested())

    def test_spheres_have_the_arrays_of_the_reference_builder(self) -> None:
        sizes = set()
        for spacing, radius in ((0.8, 3.3), (0.65, 2.8), (0.5, 4.0), (0.377945375554871, 7.3), (0.9, 0.5), (0.31, 6.02)):
            for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0.25, -0.25, 0.5), (0.1, 0.7, -0.3), (0.0, 0.5, 0.0)):
                with self.subTest(spacing=spacing, radius=radius, shift=shift):
                    settings = GridSettings(spacing=spacing, radius=radius, expansion_order=8, shift=shift)
                    sizes.add(self.assert_same_grid(settings).size)
        # Empty slabs at the faces of the cube, and grids from one point to tens of thousands.
        self.assertLessEqual(min(sizes), 8)
        self.assertGreater(max(sizes), 30_000)

    def test_boxes_have_the_arrays_of_the_reference_builder(self) -> None:
        sizes = set()
        for lengths in (None, (4.0, 4.0, 4.0), (3.0, 5.5, 2.25), (6.4, 1.6, 3.2), (0.4, 0.4, 0.4)):
            for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0.25, -0.25, 0.5)):
                # 0.8 puts faces of the 4.0, 6.4, 1.6 and 3.2 boxes on grid planes.
                for spacing in (0.8, 0.37):
                    with self.subTest(lengths=lengths, shift=shift, spacing=spacing):
                        settings = GridSettings(spacing=spacing, radius=2.6, expansion_order=4, shift=shift,
                                                domain_shape="box", box_lengths=lengths)
                        sizes.add(self.assert_same_grid(settings).size)
        # The smallest box holds no point of the coarse shifted grids.
        self.assertEqual(min(sizes), 0)
        self.assertGreater(max(sizes), 2_000)

    def test_points_on_the_sphere_are_decided_by_the_reference_test(self) -> None:
        # Without a shift and with a radius of eight spacings six points lie on the sphere itself.
        on_sphere = GridSettings(spacing=0.5, radius=4.0, expansion_order=4, shift=(0.0, 0.0, 0.0))
        recorder = _ShellRecorder()
        with patch.object(fast_cluster, "_inside_domain", recorder):
            reference = self.assert_same_grid(on_sphere)
        self.assertEqual(recorder.points, 6)
        self.assertTrue(np.all(reference.rows_for_integer_coordinates(8 * np.eye(3, dtype=int)) >= 0))
        # The slab builder asks the reference test there and takes its answer.
        for answer, change in ((False, -6), (True, 0)):
            with patch.object(fast_cluster, "_inside_domain", _ShellRecorder(answer)):
                self.assertEqual(build_cluster_grid_by_slabs(on_sphere).size, reference.size + change)
        # One unit in the last place less, and the six points are outside in both builders.
        inside = GridSettings(spacing=0.5, radius=float(np.nextafter(4.0, 0.0)), expansion_order=4,
                              shift=(0.0, 0.0, 0.0))
        self.assertEqual(self.assert_same_grid(inside).size, reference.size - 6)

    def test_spheres_through_a_grid_point_to_the_last_place(self) -> None:
        # Radii whose square is the sum of squares of one grid point, in every order of
        # the three terms, and its neighbours in the last place on either side.
        spacing, shift = 0.377945375554871, (0.5, 0.5, 0.5)
        rng = np.random.default_rng(12)
        asked = 0
        for _ in range(12):
            point = (rng.integers(-9, 9, size=3) + np.asarray(shift)) * spacing
            x, y, z = point * point
            for total in {(x + y) + z, (x + z) + y, (y + z) + x}:
                for limit in (np.nextafter(total, 0.0), total, np.nextafter(total, np.inf)):
                    radius = float(np.sqrt(limit))
                    for candidate in (np.nextafter(radius, 0.0), radius, np.nextafter(radius, np.inf)):
                        settings = GridSettings(spacing=spacing, radius=float(candidate), expansion_order=8,
                                                shift=shift)
                        recorder = _ShellRecorder()
                        with patch.object(fast_cluster, "_inside_domain", recorder):
                            self.assert_same_grid(settings)
                        asked += recorder.points
        # Every such sphere has grid points in the shell where the reference test decides.
        self.assertGreater(asked, 12 * 9)

    def test_no_array_of_the_bounding_cube_but_one_boolean_per_point(self) -> None:
        settings = GridSettings(spacing=0.25, radius=10.0, expansion_order=8)

        def peak(builder):
            tracemalloc.start()
            try:
                grid = builder(settings)
                return grid, tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        grid, slab_peak = peak(build_cluster_grid_by_slabs)
        kept = sum(getattr(grid, name).nbytes for name in _ARRAYS)
        cube = grid.lookup.size
        self.assertGreater(cube, 500_000)
        del grid
        # The arrays of the grid, one byte per cube point, and tables of one slab.
        self.assertLess(slab_peak, kept + 1.5 * cube)
        reference, reference_peak = peak(build_cluster_grid)
        # The reference holds three integers and three floats per cube point at least.
        self.assertGreater(reference_peak, kept + 48 * cube)

    def test_reference_preparation_calls_the_builder_it_is_given(self) -> None:
        from parsec_python.Input import parse_parsec_input
        from parsec_python.SCF.single_point import prepare_single_point

        problem = parse_parsec_input(SMOKE_INPUT).problem
        given = []

        def builder(settings):
            given.append(settings)
            return build_cluster_grid_by_slabs(settings)

        system = prepare_single_point(problem, grid_builder=builder)
        self.assertEqual(given, [problem.grid])
        reference = prepare_single_point(problem)
        for name in _ARRAYS:
            self.assertEqual(getattr(system.grid, name).tobytes(), getattr(reference.grid, name).tobytes())
        self.assertEqual(system.negative_laplacian.data.tobytes(), reference.negative_laplacian.data.tobytes())
        self.assertEqual(system.ionic_potential.tobytes(), reference.ionic_potential.tobytes())
        self.assertGreater(system.timings.grid_seconds, 0.0)

    def test_hartree_boundary_reads_the_same_grid_from_either_builder(self) -> None:
        # The plan of the boundary is made from the settings of the grid,
        # before either builder runs.  The atomic tail, the surface shell
        # and the right-hand side read the arrays of the grid.
        from parsec_python.Hartree import valence_point_charges
        from parsec_python.SCF.single_point import prepare_single_point
        from parsec_python.acceleration.Hartree.atomic_tail import (
            build_atomic_tail_fast,
            missing_stencil_entries,
        )
        from parsec_python.acceleration.Hartree.poisson import build_hartree_problem
        from parsec_python.acceleration.tests.test_hartree_boundary import hydrogen_problem

        problem = hydrogen_problem(multipole_order=2)
        order_of_calls = []

        def builder(settings):
            order_of_calls.append("grid")
            return build_cluster_grid_by_slabs(settings)

        def plan(*arguments):
            order_of_calls.append("plan")
            return plan_hartree_boundary(*arguments)

        import parsec_python.SCF.single_point as reference_module

        plan_hartree_boundary = reference_module.plan_hartree_boundary
        with patch.object(reference_module, "plan_hartree_boundary", plan):
            fast = prepare_single_point(problem, grid_builder=builder)
        self.assertEqual(order_of_calls, ["plan", "grid"])
        reference = prepare_single_point(problem)
        self.assertTrue(reference.hartree_boundary.atomic_tail)
        self.assertGreater(reference.hartree_boundary.order, 2)
        for name in ("order", "minimum_order", "engaged", "atomic_tail", "estimates"):
            self.assertEqual(
                getattr(fast.hartree_boundary, name), getattr(reference.hartree_boundary, name), name)
        self.assertEqual(fast.input.hartree, reference.input.hartree)
        charges = valence_point_charges(
            reference.atoms, reference.pseudopotentials, reference.electron_count)
        order = reference.hartree_boundary.order
        tails = [
            (system.hartree_boundary_tail, build_atomic_tail_fast(system.grid, *charges, order))
            for system in (fast, reference)
        ]
        for built, expected in zip(*tails):
            self.assertEqual(built.rows.tobytes(), expected.rows.tobytes())
            self.assertEqual(built.values.tobytes(), expected.values.tobytes())
            self.assertEqual(built.maximum, expected.maximum)
        for built, expected in zip(
            missing_stencil_entries(fast.grid), missing_stencil_entries(reference.grid)
        ):
            self.assertEqual(built.tobytes(), expected.tobytes())
        sides = [
            build_hartree_problem(
                system.initial_density, system.grid, system.input.hartree, tail)[0]
            for system, (_walked, tail) in zip((fast, reference), tails)
        ]
        self.assertEqual(sides[0].tobytes(), sides[1].tobytes())

    def test_accelerated_preparation_takes_the_builder_the_switch_names(self) -> None:
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.backends.selection import BackendSelection

        selection = BackendSelection(requested="cupy", selected="cupy",
                                     finite_difference_builder="reference", hartree_backend="cupy")
        problem = object()
        # The atomic tail of the Hartree boundary is left to the backend's
        # Hartree solver whichever builder makes the grid.
        tail = {"build_atomic_tail": False}
        for value, expected in ((None, {"grid_builder": build_cluster_grid_by_slabs, **tail}),
                                ("1", {"grid_builder": build_cluster_grid_by_slabs, **tail}),
                                ("0", tail)):
            with self.subTest(value=value), patch.dict(os.environ), \
                    patch.object(driver, "prepare_reference_single_point") as prepare:
                os.environ.pop("PARSEC_FAST_GRID", None)
                os.environ.pop("PARSEC_ACCELERATED_RESIDENT", None)
                if value is not None:
                    os.environ["PARSEC_FAST_GRID"] = value
                driver._prepare_reference_physics(problem, selection)
                prepare.assert_called_once_with(problem, **expected)
                # A rank that builds no ionic fields builds the same grid.
                prepare.reset_mock()
                driver._prepare_reference_physics(problem, selection, orbital_operators_only=True)
                prepare.assert_called_once_with(problem, orbital_operators_only=True, **expected)

    def test_backend_report_names_the_builder(self) -> None:
        from parsec_python.acceleration.tests.test_hybrid_driver import MPIRolePreparationTests

        harness = MPIRolePreparationTests("prepare")
        self.addCleanup(harness.tearDown)
        for value, expected in ((None, "slabs"), ("off", "reference"), ("1", "slabs")):
            with self.subTest(value=value), patch.dict(os.environ):
                os.environ.pop("PARSEC_FAST_GRID", None)
                prepared = harness.prepare(None, environment={} if value is None else {"PARSEC_FAST_GRID": value})
                self.assertEqual(dict(prepared.system.backend_info.details)["grid_builder"], expected)

    def test_serial_control_keeps_the_reference_builder(self) -> None:
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        with patch.dict(os.environ):
            for name in mpi_full_scf._SERIAL_CONTROL_SETTINGS:
                os.environ.pop(name, None)
            mpi_full_scf._keep_former_routes(os.environ)
            self.assertFalse(fast_grid_requested())
            # What the launcher names stays.
            os.environ["PARSEC_FAST_GRID"] = "1"
            mpi_full_scf._keep_former_routes(os.environ)
            self.assertTrue(fast_grid_requested())


if __name__ == "__main__":
    unittest.main()
