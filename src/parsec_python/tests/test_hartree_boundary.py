from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python import (
    Atom,
    GridSettings,
    HartreeSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
    build_cluster_grid,
    build_negative_laplacian,
    density_multipoles,
    solve_hartree,
)
from parsec_python.Hartree import (
    DirectCoulombBoundary,
    build_atomic_tail,
    estimate_omitted_potential,
    plan_hartree_boundary,
    point_charge_multipoles,
    point_charge_potential,
    valence_point_charges,
)
from parsec_python.Input.parsec_input import ParsecInputError, parse_parsec_input
from parsec_python.Laplacian import (
    apply_negative_laplacian_boundary,
    neighbor_shells,
)
from parsec_python.SCF.single_point import (
    complete_single_point,
    prepare_single_point,
)
from parsec_python.V_ion import (
    center_cluster_geometry,
    ionic_charge,
    load_pseudopotentials,
)


DATA = Path(__file__).parent / "data"
EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
ANGSTROM = 1.0 / 0.529177210903


def exterior_points(grid) -> np.ndarray:
    """Every exterior stencil point of the grid, once, in bohr."""

    found = [
        integer_points[rows < 0]
        for _axis, _shell, rows, integer_points in neighbor_shells(grid)
        if np.any(rows < 0)
    ]
    return grid.physical_coordinates(np.unique(np.concatenate(found), axis=0))


def legacy_right_hand_side(density, grid, order) -> np.ndarray:
    """The former right-hand side: the expansion of the density alone."""

    return apply_negative_laplacian_boundary(
        8.0 * np.pi * density, grid, density_multipoles(density, grid, order).potential
    )


def point_density(grid, rows, charges) -> np.ndarray:
    density = np.zeros(grid.size)
    density[rows] = np.asarray(charges) / grid.volume_element
    return density


def hydrogen_problem(radius, positions, spacing=0.7, **hartree) -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("H", position) for position in positions],
        pseudopotentials={"H": SpeciesPotential(DATA / "H_POTRE.DAT", 0)},
        grid=GridSettings(spacing=spacing, radius=radius, expansion_order=8),
        scf=SCFSettings(max_iterations=1, number_of_states=len(positions) + 2),
        hartree=HartreeSettings(**hartree),
        recenter_geometry=False,
    )


# Four hydrogen atoms up to 1.5 angstrom from the origin.
HYDROGEN_POSITIONS = ANGSTROM * np.array(
    [[1.5, 0.0, 0.0], [-0.6, 1.2, 0.3], [0.2, -0.9, 1.1], [-0.4, -0.3, -1.3]]
)


class EstimateTests(unittest.TestCase):
    def test_charge_on_the_axis_has_the_closed_form(self) -> None:
        # The omitted terms of one charge peak in its own direction, where
        # they are a geometric series.
        charge, distance, radius = 3.0, 8.1, 9.0
        estimate = estimate_omitted_potential(
            np.array([[0.0, 0.0, distance]]), np.array([charge]), radius
        )
        order = np.arange(estimate.size)
        expected = 2.0 * charge * (distance / radius) ** (order + 1) / (radius - distance)
        np.testing.assert_allclose(estimate[:21], expected[:21], rtol=1.0e-12)
        np.testing.assert_allclose(estimate, expected, rtol=1.0e-10)

    def test_estimate_sums_the_sample_points_in_blocks_that_fit_a_cache(self) -> None:
        from unittest.mock import patch

        import parsec_python.Hartree.boundary as boundary_module

        generator = np.random.default_rng(4)
        positions = generator.normal(size=(300, 3))
        positions *= (6.0 * generator.random(300) / np.linalg.norm(positions, axis=1))[:, None]
        charges = generator.uniform(1.0, 4.0, size=300)
        blocks = []

        def recorded(points, atoms, weights, *, chunk_pairs=2_000_000):
            blocks.append(chunk_pairs)
            return point_charge_potential(points, atoms, weights, chunk_pairs=chunk_pairs)

        with patch.object(boundary_module, "point_charge_potential", recorded):
            estimate = estimate_omitted_potential(positions, charges, 9.0)
        # 2,000 directions and more times 300 atoms, in blocks of 2**15 pairs.
        self.assertEqual(blocks, [1 << 15])
        # The blocks and the order of the sums over the atoms do not matter:
        # the sample points, summed whole and expanded by the reference.
        distance = np.linalg.norm(positions, axis=1)
        outer = distance > distance.max() - 2.0
        points = 9.0 * np.concatenate([
            boundary_module._fibonacci_directions(2000),
            positions[outer] / distance[outer][:, None]])
        exact = point_charge_potential(points, positions, charges)
        for order in (0, 9, 20):
            series = point_charge_multipoles(positions, charges, order).potential(points)
            self.assertAlmostEqual(
                estimate[order], np.abs(exact - series).max(), delta=1.0e-9 * estimate[0])

    def test_charge_at_the_origin_omits_nothing(self) -> None:
        positions, charges = np.zeros((1, 3)), np.array([4.0])
        estimate = estimate_omitted_potential(positions, charges, 6.0)
        self.assertLess(estimate.max(), 1.0e-14)
        plan = plan_hartree_boundary(
            HartreeSettings(multipole_order=0),
            GridSettings(spacing=0.5, radius=6.0),
            positions,
            charges,
        )
        self.assertFalse(plan.engaged)
        self.assertTrue(plan.legacy)

    def test_plan_follows_the_estimate_and_the_two_settings(self) -> None:
        positions = np.array([[0.0, 0.0, 7.2]])
        charges = np.array([1.0])
        sphere = GridSettings(spacing=0.5, radius=9.0)

        def plan(**settings):
            return plan_hartree_boundary(
                HartreeSettings(**settings), sphere, positions, charges
            )

        engaged = plan()
        self.assertTrue(engaged.engaged and engaged.atomic_tail)
        self.assertGreater(engaged.estimate_minimum, 1.0e-4)
        # e(L) = 2 * 0.8**(L+1) / 1.8 Ry first meets 1e-3 Ry at L = 31.
        self.assertEqual((engaged.order, engaged.minimum_order), (31, 9))
        self.assertLessEqual(engaged.estimate_order, 1.0e-3)
        self.assertGreater(engaged.estimates[30], 1.0e-3)
        self.assertFalse(engaged.cap_reached or engaged.legacy)
        untailed = plan(atomic_tail="off")
        self.assertEqual((untailed.order, untailed.atomic_tail), (31, False))
        self.assertFalse(untailed.legacy)
        self.assertEqual(plan(multipole_order=40).order, 40)
        # Between a tenth of the tolerance and the tolerance: the tail alone.
        tail_only = plan(boundary_tolerance=2.0 * engaged.estimate_minimum)
        self.assertEqual((tail_only.order, tail_only.atomic_tail), (9, True))
        # A tolerance ten times the estimate leaves PARSEC's boundary.
        loose = plan(boundary_tolerance=10.0 * engaged.estimate_minimum)
        self.assertFalse(loose.engaged or loose.atomic_tail)
        self.assertEqual(loose.order, 9)
        # No order up to 60 reaches 1e-6 Ry here.
        with self.assertWarnsRegex(RuntimeWarning, "largest order 60"):
            capped = plan(boundary_tolerance=1.0e-6)
        self.assertEqual((capped.order, capped.cap_reached), (60, True))
        self.assertGreater(capped.estimate_order, 1.0e-6)
        self.assertTrue(plan(
            boundary_tolerance=10.0 * engaged.estimate_minimum, atomic_tail="on"
        ).atomic_tail)
        off = plan(boundary_tolerance=None)
        self.assertTrue(off.legacy)
        self.assertEqual(off.order, 9)
        self.assertIsNone(off.estimates)
        self.assertTrue(plan(boundary_tolerance=None, atomic_tail="on").atomic_tail)
        # A box and a direct boundary are exact already.
        for settings, grid in (
            (HartreeSettings(atomic_tail="on"), GridSettings(
                spacing=0.5, radius=9.0, domain_shape="box")),
            (HartreeSettings(atomic_tail="on", boundary_method="direct"), sphere),
        ):
            direct = plan_hartree_boundary(settings, grid, positions, charges)
            self.assertTrue(direct.legacy)
            self.assertIsNone(direct.estimates)

    def test_settings_refuse_bad_values(self) -> None:
        for bad in ({"boundary_tolerance": 0.0}, {"boundary_tolerance": float("nan")},
                    {"atomic_tail": "yes"}, {"multipole_order": 61},
                    {"multipole_order": -1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                HartreeSettings(**bad)
        self.assertEqual(HartreeSettings(multipole_order=60).multipole_order, 60)


class AtomicTailTests(unittest.TestCase):
    def test_point_densities_get_the_direct_boundary_values(self) -> None:
        # Weights on a few grid points with "atoms" of the same charges on
        # them: the expansion plus the tail is the direct Coulomb sum.
        for radius in (7.0, 4.6):
            grid = build_cluster_grid(
                GridSettings(spacing=0.5, radius=radius, expansion_order=4)
            )
            distance = np.linalg.norm(grid.coordinates, axis=1)
            rows = np.argsort(np.abs(distance - 3.9), kind="stable")[[0, 7, 31, 90]]
            charges = np.array([4.0, 1.0, 2.5, 0.5])
            density = point_density(grid, rows, charges)
            positions = grid.coordinates[rows]
            points = exterior_points(grid)
            direct = DirectCoulombBoundary(
                grid.coordinates, density * grid.volume_element
            ).potential(points)
            for order in (0, 2, 9, 20):
                with self.subTest(radius=radius, order=order):
                    expansion = density_multipoles(density, grid, order).potential(points)
                    tail = point_charge_potential(
                        points, positions, charges
                    ) - point_charge_multipoles(positions, charges, order).potential(points)
                    np.testing.assert_allclose(expansion + tail, direct, atol=1.0e-12)
                    # The expansion alone misses by the tail.
                    if order <= 9:
                        self.assertGreater(np.abs(direct - expansion).max(), 1.0e-3)
                    built = build_atomic_tail(grid, positions, charges, order)
                    self.assertAlmostEqual(built.maximum, np.abs(tail).max(), places=12)
                    exact = apply_negative_laplacian_boundary(
                        8.0 * np.pi * density, grid,
                        DirectCoulombBoundary(
                            grid.coordinates, density * grid.volume_element
                        ).potential,
                    )
                    rhs = legacy_right_hand_side(density, grid, order)
                    rhs[built.rows] += built.values
                    np.testing.assert_allclose(rhs, exact, atol=1.0e-10)

    def test_solver_adds_the_rows_of_the_boundary_function(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.6, radius=5.0, expansion_order=6))
        rng = np.random.default_rng(4)
        density = rng.random(grid.size) * np.exp(-np.sum(grid.coordinates**2, axis=1) / 8.0)
        positions = np.array([[2.9, 0.4, -0.3], [-1.1, 2.2, 1.7], [0.0, 0.0, 0.0]])
        charges = np.array([2.0, 1.0, 3.0])
        settings = HartreeSettings(multipole_order=3)
        tail = build_atomic_tail(grid, positions, charges, 3)
        laplacian = build_negative_laplacian(grid)
        result = solve_hartree(density, grid, laplacian, settings, boundary_tail=tail)
        expansion = density_multipoles(density, grid, 3)
        atoms = point_charge_multipoles(positions, charges, 3)

        def boundary(points):
            return (
                expansion.potential(points)
                + point_charge_potential(points, positions, charges)
                - atoms.potential(points)
            )

        expected = apply_negative_laplacian_boundary(8.0 * np.pi * density, grid, boundary)
        np.testing.assert_allclose(
            result.right_hand_side, expected, rtol=0.0,
            atol=1.0e-13 * np.abs(expected).max(),
        )
        self.assertIs(result.boundary_tail, tail)
        # The tail lives on the rows with a missing neighbour, and only there.
        missing = np.zeros(grid.size, dtype=bool)
        for _axis, _shell, rows, _points in neighbor_shells(grid):
            missing |= rows < 0
        np.testing.assert_array_equal(tail.rows, np.flatnonzero(missing))
        plain = solve_hartree(density, grid, laplacian, settings)
        self.assertIsNone(plain.boundary_tail)
        changed = np.flatnonzero(plain.right_hand_side != result.right_hand_side)
        self.assertTrue(np.isin(changed, tail.rows).all())
        np.testing.assert_array_equal(
            plain.right_hand_side, legacy_right_hand_side(density, grid, 3)
        )

    def test_tail_of_another_boundary_is_refused(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.8, radius=4.0, expansion_order=4))
        density = np.full(grid.size, 0.01)
        laplacian = build_negative_laplacian(grid)
        tail = build_atomic_tail(grid, np.zeros((1, 3)), np.ones(1), 2)
        with self.assertRaisesRegex(ValueError, "multipole order 2"):
            solve_hartree(density, grid, laplacian, HartreeSettings(multipole_order=4),
                          boundary_tail=tail)
        with self.assertRaisesRegex(ValueError, "direct"):
            solve_hartree(
                density, grid, laplacian,
                HartreeSettings(multipole_order=2, boundary_method="direct"),
                boundary_tail=tail,
            )

    def test_blobs_at_the_atoms_leave_only_what_the_point_charges_miss(self) -> None:
        # Ten normalized Gaussian blobs at 0.7 and 0.42 of the sphere radius.
        grid = build_cluster_grid(GridSettings(
            spacing=0.6, radius=9.0, expansion_order=8, shift=(0.5, 0.5, 0.5)))
        rng = np.random.default_rng(3)
        directions = rng.normal(size=(9, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        positions = np.concatenate(
            [np.zeros((1, 3)), 6.3 * directions[:5], 0.6 * 6.3 * directions[5:]])
        charges = np.array([4.0, 1, 4, 1, 4, 1, 4, 1, 4, 1])
        density = np.zeros(grid.size)
        for position, charge in zip(positions, charges):
            blob = np.exp(-np.sum((grid.coordinates - position) ** 2, axis=1) / (2 * 0.55**2))
            density += charge * blob / (blob.sum() * grid.volume_element)
        # The order the plan takes for 1e-3 Ry.
        self.assertEqual(plan_hartree_boundary(
            HartreeSettings(), grid.settings, positions, charges).order, 22)
        points = exterior_points(grid)
        points = points[np.argsort(np.linalg.norm(points, axis=1), kind="stable")[::25]]
        direct = DirectCoulombBoundary(
            grid.coordinates, density * grid.volume_element).potential(points)
        for order in (4, 9, 16, 22):
            with self.subTest(order=order):
                error = direct - density_multipoles(density, grid, order).potential(points)
                tail = point_charge_potential(
                    points, positions, charges
                ) - point_charge_multipoles(positions, charges, order).potential(points)
                # The expansion alone misses by what the point charges miss,
                # at every order; with the tail little is left.
                if order <= 9:
                    self.assertGreater(np.abs(error).max(), 1.0e-2)
                self.assertLess(np.abs(error - tail).max(), 1.0e-4)


class PreparedSystemTests(unittest.TestCase):
    def boundary_error(self, system, order, tail) -> float:
        """max |V_B - direct| of the initial density at sampled exterior points."""

        grid, density = system.grid, system.initial_density
        points = exterior_points(grid)
        points = points[
            np.argsort(np.linalg.norm(points, axis=1), kind="stable")[
                :: max(1, points.shape[0] // 300)
            ]
        ]
        values = density_multipoles(density, grid, order).potential(points)
        if tail:
            positions, charges = valence_point_charges(
                system.atoms, system.pseudopotentials, system.electron_count)
            values += point_charge_potential(
                points, positions, charges
            ) - point_charge_multipoles(positions, charges, order).potential(points)
        direct = DirectCoulombBoundary(
            grid.coordinates, density * grid.volume_element).potential(points)
        return float(np.abs(values - direct).max())

    def test_superposed_atoms_get_their_exact_boundary(self) -> None:
        # Vacuum of 5 and of 3 angstrom beyond the outermost atom.  What is
        # left is what the grid makes of a spherical atom, so the spacing is
        # that of a production grid.
        for vacuum, bound in ((5.0, 1.0e-5), (3.0, 5.0e-4)):
            with self.subTest(vacuum=vacuum):
                system = prepare_single_point(hydrogen_problem(
                    (1.5 + vacuum) * ANGSTROM, HYDROGEN_POSITIONS, spacing=0.4,
                    multipole_order=2))
                plan = system.hartree_boundary
                self.assertTrue(plan.engaged and plan.atomic_tail)
                # The order is raised from Solver_Lpole 2 until the estimate
                # meets 1e-3 Ry, and every builder reads it from the input.
                self.assertEqual(
                    (plan.minimum_order, plan.order), (2, {5.0: 3, 3.0: 5}[vacuum]))
                self.assertLessEqual(plan.estimate_order, 1.0e-3)
                self.assertEqual(system.input.hartree.multipole_order, plan.order)
                self.assertEqual(system.hartree_boundary_tail.order, plan.order)
                legacy = self.boundary_error(system, 2, tail=False)
                fixed = self.boundary_error(system, plan.order, tail=True)
                self.assertLess(fixed, bound)
                self.assertLess(20.0 * fixed, legacy)
                # The prepared system solves with those boundary values.
                density = system.initial_density
                result = system.solve_hartree(density)
                expected = legacy_right_hand_side(density, system.grid, plan.order)
                expected[system.hartree_boundary_tail.rows] += (
                    system.hartree_boundary_tail.values)
                np.testing.assert_array_equal(result.right_hand_side, expected)

    def test_hartree_energy_is_that_of_the_exact_boundary(self) -> None:
        # An energy, not a maximum over boundary points: the Hartree energy
        # of the superposed atoms against the one of the direct Coulomb
        # boundary, which is exact for the grid density.
        def hartree_energy(vacuum, **hartree) -> float:
            system = prepare_single_point(hydrogen_problem(
                (1.5 + vacuum) * ANGSTROM, HYDROGEN_POSITIONS,
                relative_tolerance=1.0e-12, **hartree))
            density = system.initial_density
            return 0.5 * system.grid.integrate(
                density * system.solve_hartree(density).potential)

        # 3 angstrom of vacuum, Solver_Lpole 2: the order-2 expansion is
        # 7e-4 Ry off, the default (order 5 and the tail) 2e-8 Ry.
        exact = hartree_energy(3.0, boundary_method="direct")
        legacy = hartree_energy(3.0, multipole_order=2, boundary_tolerance=None)
        default = hartree_energy(3.0, multipole_order=2)
        self.assertGreater(abs(legacy - exact), 5.0e-4)
        self.assertLess(abs(default - exact), 1.0e-6)
        # 1.2 angstrom of vacuum, Solver_Lpole 9: density reaches the sphere,
        # and what it leaves is not the atoms' truncation error.  The
        # default (order 11 and the tail) is 2.3e-5 Ry off, no closer than
        # PARSEC's order 9 at 9e-6 Ry, and a hundred times the figure of
        # 3 angstrom; the exact boundary is the option there.
        exact = hartree_energy(1.2, boundary_method="direct")
        legacy = hartree_energy(1.2, boundary_tolerance=None)
        default = hartree_energy(1.2)
        self.assertLess(abs(default - exact), 5.0e-5)
        self.assertLess(abs(legacy - exact), 5.0e-5)
        self.assertGreater(abs(default - exact), 1.0e-6)

    def test_switched_off_the_boundary_is_bitwise_the_former_one(self) -> None:
        problem = hydrogen_problem(
            4.5 * ANGSTROM, HYDROGEN_POSITIONS, multipole_order=2,
            boundary_tolerance=None)
        system = prepare_single_point(problem)
        plan = system.hartree_boundary
        self.assertEqual((plan.order, plan.atomic_tail, plan.legacy), (2, False, True))
        self.assertIsNone(system.hartree_boundary_tail)
        density = system.initial_density
        np.testing.assert_array_equal(
            system.solve_hartree(density).right_hand_side,
            legacy_right_hand_side(density, system.grid, 2),
        )

    def test_atoms_near_the_sphere_raise_the_order(self) -> None:
        # Atoms at 0.8 of the sphere radius, the ratio of the large clusters.
        radius = 5.0
        positions = 0.8 * radius * np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-0.6, 0.0, -0.8]])
        system = prepare_single_point(
            hydrogen_problem(radius, positions), build_atomic_tail=False)
        plan = system.hartree_boundary
        self.assertEqual(plan.minimum_order, 9)
        self.assertGreater(plan.order, 20)
        self.assertTrue(plan.atomic_tail)
        self.assertEqual(system.input.hartree.multipole_order, plan.order)
        self.assertLessEqual(plan.estimate_order, 1.0e-3)
        self.assertGreater(plan.estimates[plan.order - 1], 1.0e-3)
        # The input that was asked for is not changed.
        fixed = prepare_single_point(
            hydrogen_problem(radius, positions, boundary_tolerance=None))
        self.assertEqual(fixed.input.hartree.multipole_order, 9)
        self.assertTrue(fixed.hartree_boundary.legacy)

    def test_prepared_input_is_planned_the_same_way_again(self) -> None:
        # A tetrahedron of atoms 2 bohr from the origin: e(3) is above the
        # tolerance and e(4) below a tenth of it.  Solver_Lpole 0 becomes
        # order 4 with the tail, and a plan that started at 4 would leave
        # PARSEC's boundary there.
        positions = (2.0 / np.sqrt(3.0)) * np.array(
            [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
        problem = hydrogen_problem(8.4, positions, multipole_order=0)
        system = prepare_single_point(problem, build_atomic_tail=False)
        plan = system.hartree_boundary
        self.assertEqual((plan.minimum_order, plan.order, plan.atomic_tail), (0, 4, True))
        self.assertGreater(plan.estimates[3], plan.tolerance)
        self.assertLess(plan.estimate_order, 0.1 * plan.tolerance)
        resolved = system.input.hartree
        self.assertEqual(
            (resolved.multipole_order, resolved.minimum_multipole_order), (4, 0))
        self.assertIsNone(problem.hartree.minimum_multipole_order)
        again = prepare_single_point(system.input, build_atomic_tail=False)
        self.assertEqual(
            replace(again.hartree_boundary, seconds=0.0), replace(plan, seconds=0.0))
        self.assertEqual(again.input.hartree, resolved)
        # The same order as an input of its own has no tail to add.
        alone = prepare_single_point(
            hydrogen_problem(8.4, positions, multipole_order=4))
        self.assertTrue(alone.hartree_boundary.legacy)
        # With the tolerance switched off the prepared input is back at
        # the Solver_Lpole it came from.
        off = prepare_single_point(replace(
            system.input, hartree=replace(resolved, boundary_tolerance=None)))
        self.assertEqual(off.input.hartree.multipole_order, 0)
        self.assertTrue(off.hartree_boundary.legacy)
        for bad in (5, -1, 1.5):
            with self.subTest(minimum=bad), self.assertRaises(ValueError):
                HartreeSettings(multipole_order=4, minimum_multipole_order=bad)

    def test_small_molecules_keep_the_former_boundary_by_default(self) -> None:
        # One atom at the origin: nothing is omitted at any order.
        problem = hydrogen_problem(4.0, np.zeros((1, 3)), multipole_order=2)
        system = prepare_single_point(problem)
        self.assertEqual(system.hartree_boundary.status, "estimate within a tenth of the tolerance")
        self.assertIsNone(system.hartree_boundary_tail)
        np.testing.assert_array_equal(
            system.solve_hartree(system.initial_density).right_hand_side,
            legacy_right_hand_side(system.initial_density, system.grid, 2),
        )
        # The shipped molecular inputs, from the geometry alone.
        for name in ("0d_benzene/parsec.in", "0d_naphthalene/parsec.in",
                     "0_CH4_CF4/python_pbe/CH4/IS/parsec.in",
                     "h2_full_nonlocal/parsec.in"):
            path = EXAMPLES / name
            if not path.is_file():
                continue
            with self.subTest(example=name):
                problem = parse_parsec_input(path).problem
                atoms = center_cluster_geometry(problem.atoms)
                potentials = load_pseudopotentials(
                    problem.pseudopotentials, xc_functional=problem.scf.xc_functional)
                electrons = ionic_charge(atoms, potentials) - problem.scf.net_charge
                plan = plan_hartree_boundary(
                    problem.hartree, problem.grid,
                    *valence_point_charges(atoms, potentials, electrons))
                self.assertFalse(plan.engaged)
                self.assertTrue(plan.legacy)
                self.assertEqual(plan.order, 9)
                self.assertLess(plan.estimate_minimum, 1.0e-4)

    def test_operator_only_preparation_has_the_plan_and_gets_the_tail(self) -> None:
        problem = hydrogen_problem(4.5 * ANGSTROM, HYDROGEN_POSITIONS, multipole_order=2)
        complete = prepare_single_point(problem)
        partial = prepare_single_point(problem, orbital_operators_only=True)
        self.assertIsNone(partial.hartree_boundary_tail)
        self.assertGreater(complete.hartree_boundary.order, 2)
        self.assertEqual(
            partial.input.hartree.multipole_order, complete.hartree_boundary.order)
        self.assertEqual(partial.input.hartree, complete.input.hartree)
        self.assertEqual(
            replace(partial.hartree_boundary, seconds=0.0),
            replace(complete.hartree_boundary, seconds=0.0),
        )
        completed = complete_single_point(partial)
        self.assertEqual(completed.hartree_boundary_tail.order, complete.hartree_boundary.order)
        np.testing.assert_array_equal(
            completed.hartree_boundary_tail.values, complete.hartree_boundary_tail.values)
        np.testing.assert_array_equal(
            completed.hartree_boundary_tail.rows, complete.hartree_boundary_tail.rows)
        # A caller that builds the tail for its own solver gets none here,
        # and the reference solver then refuses to run without it.
        bare = prepare_single_point(problem, build_atomic_tail=False)
        self.assertTrue(bare.hartree_boundary.atomic_tail)
        self.assertIsNone(bare.hartree_boundary_tail)
        with self.assertRaisesRegex(RuntimeError, "atomic tail"):
            bare.solve_hartree(bare.initial_density)


class ReportTests(unittest.TestCase):
    def test_reference_driver_says_which_boundary_switches_it_ignores(self) -> None:
        import os
        from unittest.mock import patch

        from parsec_python.cli import (
            _ACCELERATED_BOUNDARY_SWITCHES,
            ignored_boundary_switch_warnings,
        )

        clean = {
            name: value for name, value in os.environ.items()
            if name not in _ACCELERATED_BOUNDARY_SWITCHES
        }
        with patch.dict(os.environ, clean, clear=True):
            self.assertEqual(ignored_boundary_switch_warnings(), [])
        with patch.dict(os.environ, {
            **clean, "PARSEC_HARTREE_BOUNDARY": " legacy ", "PARSEC_HARTREE_LPOLE": "12",
            "PARSEC_HARTREE_ATOMIC_TAIL": "", "PARSEC_HARTREE_BOUNDARY_KERNEL": "full",
        }, clear=True):
            warnings = ignored_boundary_switch_warnings()
        # The two that change the boundary values; the kernels change none.
        self.assertEqual(len(warnings), 2)
        self.assertTrue(warnings[0].startswith("PARSEC_HARTREE_BOUNDARY=legacy is a switch"))
        self.assertTrue(warnings[1].startswith("PARSEC_HARTREE_LPOLE=12 is a switch"))
        self.assertIn("ignored by the reference driver", warnings[0])
        self.assertIn("Hartree_Boundary_Tolerance", warnings[0])

    def test_setup_lines_name_the_settings_that_replaced_those_of_the_input(self) -> None:
        from parsec_python.Output.parsec_output import hartree_boundary_lines

        problem = hydrogen_problem(4.5 * ANGSTROM, HYDROGEN_POSITIONS, multipole_order=2)
        system = prepare_single_point(problem, build_atomic_tail=False)
        lines = hartree_boundary_lines(system, problem.hartree)
        self.assertEqual(lines, hartree_boundary_lines(system))
        self.assertFalse(any("replace those of the input" in line for line in lines))
        # The same input prepared as the legacy switch of the accelerated
        # driver prepares it, with Solver_Lpole replaced as well.
        switched = prepare_single_point(replace(problem, hartree=replace(
            problem.hartree, multipole_order=4, boundary_tolerance=None,
            atomic_tail="off")))
        lines = hartree_boundary_lines(switched, problem.hartree)
        self.assertEqual(
            lines[-1],
            " Hartree boundary settings of this run replace those of the input "
            "(PARSEC_HARTREE_* switches): Solver_Lpole 2 -> 4, "
            "tolerance 1.000E-03 Ry -> off, atomic tail auto -> off")
        self.assertIn(" Hartree boundary values are PARSEC's multipole expansion", lines)
        # A raised order is the plan's doing and no replaced setting.
        self.assertGreater(system.hartree_boundary.order, 2)
        self.assertEqual(system.hartree_boundary.minimum_order, 2)


class PeriodicCellTests(unittest.TestCase):
    """A periodic cell has no Hartree boundary values to plan or to report."""

    @classmethod
    def setUpClass(cls) -> None:
        # One hydrogen atom in a cubic cell of 6 bohr.
        text = (
            "Periodic_System: .true.\nBoundary_Conditions: bulk\n"
            "begin Cell_Shape\n  6.0  6.0  6.0\nend Cell_Shape\n"
            "Grid_Spacing: 0.6 bohr\nExpansion_Order: 4\n"
            "Coordinate_Unit: Cartesian_Bohr\nStates_Num: 3\nMax_Iter: 40\n"
            "Convergence_Criterion: 1e-5 Ry\nEigensolver: chebff\n"
            "Mixing_Method: Anderson\nAtom_Types_Num: 1\nAtom_Type: H\n"
            "Local_Component: s\nbegin Atom_Coord\n  3.0  3.0  3.0\nend Atom_Coord\n"
            "Correlation_Type: ca\n"
        )
        cls.directory = tempfile.TemporaryDirectory()
        path = Path(cls.directory.name) / "parsec.in"
        path.write_text(text)
        (Path(cls.directory.name) / "H_POTRE.DAT").write_bytes(
            (DATA / "H_POTRE.DAT").read_bytes())
        cls.translation = parse_parsec_input(path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_periodic_system_is_prepared_without_a_plan_of_the_boundary(self) -> None:
        from unittest.mock import patch

        import parsec_python.Hartree.boundary as boundary_module
        from parsec_python.Hartree import PeriodicHartreeResult
        from parsec_python.Output.parsec_output import (
            domain_setup_lines,
            hartree_boundary_lines,
        )
        from parsec_python.SCF import prepare_periodic_single_point

        # Settings that raise the order and build the tail for a cluster.
        problem = replace(self.translation.problem, hartree=HartreeSettings(
            multipole_order=2, boundary_tolerance=1.0e-12, atomic_tail="on"))
        with (
            patch.object(boundary_module, "estimate_omitted_potential",
                         side_effect=AssertionError("the boundary was estimated")),
            patch.object(boundary_module, "build_atomic_tail",
                         side_effect=AssertionError("an atomic tail was built")),
        ):
            system = prepare_periodic_single_point(problem)
            hartree = system.solve_hartree(system.initial_density)
        self.assertIs(system.input, problem)
        self.assertEqual(system.input.hartree.multipole_order, 2)
        self.assertFalse(hasattr(system, "hartree_boundary"))
        self.assertFalse(hasattr(system, "hartree_boundary_tail"))
        self.assertEqual(system.timings.hartree_boundary_seconds, 0.0)
        self.assertIsInstance(hartree, PeriodicHartreeResult)
        self.assertTrue(hartree.converged)
        self.assertEqual(hartree_boundary_lines(system, problem.hartree), [])
        self.assertEqual(domain_setup_lines(system, self.translation), ([], None))
        # The same settings do plan the boundary of a cluster.
        cluster = prepare_single_point(hydrogen_problem(
            4.5 * ANGSTROM, HYDROGEN_POSITIONS, multipole_order=2,
            boundary_tolerance=1.0e-12, atomic_tail="on"))
        self.assertTrue(cluster.hartree_boundary.atomic_tail)
        self.assertGreater(cluster.hartree_boundary.order, 2)

    def test_report_of_a_periodic_run_has_no_line_of_the_boundary_or_the_sphere(self) -> None:
        import os
        from unittest.mock import patch

        from parsec_python.Output import ParsecTextReporter
        from parsec_python.SCF import prepare_periodic_single_point, run_scf

        messages: list[str] = []
        clean = {name: value for name, value in os.environ.items()
                 if name != "PARSEC_DOMAIN_REPORT"}
        with patch.dict(os.environ, clean, clear=True):
            reporter = ParsecTextReporter(messages.append, self.translation)
        reporter.header()
        system = prepare_periodic_single_point(self.translation.problem)
        reporter.setup(system)
        result = run_scf(system, callback=reporter.iteration)
        reporter.finish(result, 0.0)
        report = "\n".join(messages)
        self.assertTrue(result.converged)
        # Solver_Lpole is echoed for every input, as before.
        self.assertIn(" solver lpole is :            9", report)
        self.assertIn(" Ion-ion energy setup", report)
        for words in ("Hartree boundary", "Boundary_Sphere_Radius", "default rule",
                      "Energy the sphere adds", "Density at the sphere"):
            self.assertNotIn(words, report)
        self.assertIsNone(reporter.domain)


class InputKeywordTests(unittest.TestCase):
    def parse(self, *lines) -> HartreeSettings:
        text = (DATA / "H2_parsec.in").read_text() + "\n" + "\n".join(lines) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parsec.in"
            path.write_text(text)
            (Path(directory) / "H_POTRE.DAT").write_bytes(
                (DATA / "H_POTRE.DAT").read_bytes())
            return parse_parsec_input(path).problem.hartree

    def test_defaults_and_values(self) -> None:
        default = self.parse()
        self.assertEqual(
            (default.multipole_order, default.boundary_tolerance, default.atomic_tail),
            (9, 1.0e-3, "auto"),
        )
        self.assertIsNone(self.parse("Hartree_Boundary_Tolerance: off").boundary_tolerance)
        self.assertAlmostEqual(
            self.parse("Hartree_Boundary_Tolerance: 1e-4 Hartree").boundary_tolerance, 2.0e-4)
        self.assertEqual(self.parse("Hartree_Atomic_Tail: .true.").atomic_tail, "on")
        self.assertEqual(self.parse("Hartree_Atomic_Tail: false").atomic_tail, "off")
        self.assertEqual(self.parse("Hartree_Atomic_Tail: auto").atomic_tail, "auto")
        self.assertEqual(self.parse("Solver_Lpole: 4").multipole_order, 4)
        self.assertEqual(self.parse("Solver_Lpole: 30").multipole_order, 30)

    def test_bad_values_are_refused(self) -> None:
        for line in ("Hartree_Boundary_Tolerance: -1e-3", "Hartree_Boundary_Tolerance: 0",
                     "Hartree_Atomic_Tail: sometimes", "Solver_Lpole: -1",
                     "Solver_Lpole: 61"):
            with self.subTest(line=line), self.assertRaises(ParsecInputError):
                self.parse(line)


if __name__ == "__main__":
    unittest.main()
