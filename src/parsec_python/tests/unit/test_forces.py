"""Component tests of the atomic forces.

Each analytic derivative is compared with a finite difference of the quantity
it differentiates, on small grids and without an SCF loop.  The comparison of
complete forces with Fortran PARSEC is in ``test_forces_parsec_reference``.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from scipy.interpolate import CubicSpline

from parsec_python.Forces import (
    atom_symmetry_operations,
    build_nonlocal_projector_gradients,
    cubic_point_operations,
    ion_ion_forces,
    local_forces,
    nonlocal_forces,
    real_harmonics_with_gradients,
    remove_net_force,
    symmetrize_forces,
    total_torque,
)
from parsec_python.Grid import build_cluster_grid
from parsec_python.Pseudopotential import ParsecRadialSpline
from parsec_python.Pseudopotential.radial_derivatives import (
    FIRST_DERIVATIVE_WEIGHTS,
    evaluate_density_derivative,
    evaluate_local_potential_derivative,
    local_potential_derivative_table,
    log_grid_derivative,
    radial_projector_with_derivative,
    spline_derivative,
    tail_start_radius,
    valence_density_table,
)
from parsec_python.V_ion import (
    build_local_ionic_potential,
    build_nonlocal_projectors,
    ion_ion_energy,
    load_pseudopotentials,
    real_spherical_harmonics,
    superpose_atomic_density,
)
from parsec_python.models import Atom, GridSettings, SpeciesPotential


DATA = Path(__file__).resolve().parents[1] / "data"


def _species(use_spline: bool = False) -> dict[str, SpeciesPotential]:
    return {
        "O": SpeciesPotential(DATA / "water_O_POTRE.DAT", 1, use_spline=use_spline),
        "H": SpeciesPotential(DATA / "water_H_POTRE.DAT", 0, use_spline=use_spline),
    }


def _water() -> tuple[Atom, ...]:
    return (
        Atom("O", (0.05, -0.12, 0.08)),
        Atom("H", (1.55, 0.45, -0.20)),
        Atom("H", (-1.35, 0.75, 0.30)),
    )


def _moved(atoms, index, axis, step):
    shift = np.zeros(3)
    shift[axis] = step
    return tuple(
        Atom(atom.symbol, atom.position + shift) if position == index else atom
        for position, atom in enumerate(atoms)
    )


class RadialDerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specifications = _species()
        cls.potentials = load_pseudopotentials(cls.specifications)

    def test_first_derivative_weights_are_the_central_stencil(self) -> None:
        # Exact weights of the 18th-order central first derivative, which
        # fornberg.f90 stores with ten digits.
        from math import factorial

        m = FIRST_DERIVATIVE_WEIGHTS.size
        exact = [
            (-1) ** (k + 1) * factorial(m) ** 2 / (k * factorial(m - k) * factorial(m + k))
            for k in range(1, m + 1)
        ]
        np.testing.assert_allclose(FIRST_DERIVATIVE_WEIGHTS, exact, rtol=0, atol=6e-11)

    def test_log_grid_derivative_of_a_smooth_function(self) -> None:
        potential = self.potentials["O"]
        radii = potential.radii
        derivative = log_grid_derivative(potential, np.exp(-0.7 * radii) * radii**2)
        exact = np.exp(-0.7 * radii) * (2.0 * radii - 0.7 * radii**2)
        inner = slice(40, radii.size - 9)
        np.testing.assert_allclose(derivative[inner], exact[inner], rtol=2e-8, atol=1e-12)
        self.assertTrue(np.all(derivative[:7] == 0.0))
        self.assertTrue(np.all(derivative[radii.size - 9 :] == 0.0))

    def test_local_potential_derivative_table_is_the_slope_of_the_table(self) -> None:
        for symbol, local_l in (("O", 1), ("H", 0)):
            potential = self.potentials[symbol]
            radii = potential.radii
            table = local_potential_derivative_table(potential, local_l)
            # An independent derivative: that of a cubic spline through the
            # samples.  The two differ most where the pseudopotential has a
            # kink in its second derivative, at the core radius.
            numerical = CubicSpline(radii, potential.channel_potentials[local_l])(
                radii, 1
            )
            inner = (radii > 0.02) & (radii < 30.0)
            np.testing.assert_allclose(
                table[inner],
                numerical[inner],
                rtol=0,
                atol=1e-3 * np.abs(numerical[inner]).max(),
            )

    def test_local_potential_derivative_is_interpolated_between_samples(self) -> None:
        potential = self.potentials["O"]
        radii = potential.radii
        table = local_potential_derivative_table(potential, 1)
        index = np.array([300, 500, 700])
        middle = 0.5 * (radii[index] + radii[index + 1])
        expected = (
            0.5
            * (table[index] * radii[index] ** 2 + table[index + 1] * radii[index + 1] ** 2)
            / middle**2
        )
        np.testing.assert_allclose(
            evaluate_local_potential_derivative(potential, middle, 1),
            expected,
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            evaluate_local_potential_derivative(potential, radii[index], 1),
            table[index],
            rtol=1e-13,
        )

    def test_local_potential_derivative_rules_at_both_ends(self) -> None:
        potential = self.potentials["O"]
        table = local_potential_derivative_table(potential, 1)
        self.assertTrue(np.all(table[potential.radii <= 3.0e-4] == 0.0))
        tail = tail_start_radius(potential)
        radius = np.array([0.5 * potential.radii[0], tail, 2.0 * tail])
        values = evaluate_local_potential_derivative(potential, radius, 1)
        self.assertEqual(values[0], 0.0)
        np.testing.assert_allclose(
            values[1:], 2.0 * potential.ionic_charge / radius[1:] ** 2, rtol=1e-15
        )

    def test_spline_derivative_is_the_derivative_of_the_spline(self) -> None:
        potential = self.potentials["O"]
        spline = ParsecRadialSpline.from_positive_grid(
            potential.radii, potential.channel_potentials[1], 6
        )
        radius = np.linspace(0.02, 5.0, 300)
        step = 1.0e-6
        numerical = (spline(radius + step) - spline(radius - step)) / (2.0 * step)
        np.testing.assert_allclose(
            spline_derivative(spline, radius), numerical, rtol=0, atol=5e-7
        )

    def test_density_derivative_is_the_slope_of_the_density_table(self) -> None:
        potential = self.potentials["O"]
        radii = potential.radii
        density = valence_density_table(potential, read_from_file=False)
        numerical = CubicSpline(radii, density)(radii, 1)
        inner = np.flatnonzero((radii > 0.02) & (radii < 8.0))
        analytic = evaluate_density_derivative(potential, radii[inner], density)
        np.testing.assert_allclose(
            analytic,
            numerical[inner],
            rtol=0,
            atol=1e-3 * np.abs(numerical[inner]).max(),
        )
        outside = np.array([0.5 * radii[0], tail_start_radius(potential), 500.0])
        self.assertTrue(
            np.all(evaluate_density_derivative(potential, outside, density) == 0.0)
        )

    def test_projector_derivative_matches_the_projector(self) -> None:
        potential = self.potentials["O"]
        projector, derivative, sign = radial_projector_with_derivative(potential, 0, 1)
        self.assertIn(sign, (-1.0, 1.0))
        radii = potential.radii
        inner = slice(60, radii.size - 20)
        numerical = CubicSpline(radii[40:], projector[40:])(radii[inner], 1)
        np.testing.assert_allclose(
            derivative[inner],
            numerical,
            rtol=0,
            atol=1e-3 * np.abs(numerical).max(),
        )
        # The inner samples continue the derivative at Fortran index 2*mor.
        self.assertTrue(np.all(derivative[:16] == derivative[16]))
        self.assertTrue(np.all(derivative[radii.size - 9 :] == 0.0))


class HarmonicGradientTests(unittest.TestCase):
    def test_values_match_the_projector_harmonics(self) -> None:
        rng = np.random.default_rng(3)
        points = rng.normal(size=(40, 3))
        radius = np.linalg.norm(points, axis=1)
        unit = points / radius[:, None]
        for angular_momentum in range(4):
            values, _ = real_harmonics_with_gradients(angular_momentum, unit, 1.0 / radius)
            np.testing.assert_allclose(
                values,
                real_spherical_harmonics(angular_momentum, points),
                rtol=0,
                atol=1e-14,
            )

    def test_gradients_match_finite_differences(self) -> None:
        rng = np.random.default_rng(4)
        points = rng.normal(size=(25, 3)) + 0.3
        step = 1.0e-6
        for angular_momentum in range(4):
            radius = np.linalg.norm(points, axis=1)
            _, gradients = real_harmonics_with_gradients(
                angular_momentum, points / radius[:, None], 1.0 / radius
            )
            for axis in range(3):
                shift = np.zeros(3)
                shift[axis] = step
                numerical = (
                    real_spherical_harmonics(angular_momentum, points + shift)
                    - real_spherical_harmonics(angular_momentum, points - shift)
                ) / (2.0 * step)
                np.testing.assert_allclose(
                    gradients[:, :, axis], numerical, rtol=0, atol=1e-8
                )


class GridForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specifications = _species()
        cls.potentials = load_pseudopotentials(cls.specifications)
        cls.grid = build_cluster_grid(GridSettings(spacing=0.25, radius=5.0))
        cls.atoms = _water()

    def test_ion_ion_forces_are_the_gradient_of_the_energy(self) -> None:
        forces = ion_ion_forces(self.atoms, self.potentials)
        step = 1.0e-5
        for index in range(len(self.atoms)):
            for axis in range(3):
                numerical = -(
                    ion_ion_energy(_moved(self.atoms, index, axis, step), self.potentials)
                    - ion_ion_energy(_moved(self.atoms, index, axis, -step), self.potentials)
                ) / (2.0 * step)
                self.assertAlmostEqual(forces[index, axis], numerical, places=7)
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)

    def test_local_force_is_the_gradient_of_the_local_energy(self) -> None:
        # A fixed smooth density: the Hellmann-Feynman term must equal minus
        # the derivative of sum(rho * V_ion) * dV with respect to the nucleus.
        coordinates = self.grid.coordinates
        density = 0.3 * np.exp(-0.35 * np.einsum("ij,ij->i", coordinates, coordinates))
        zeros = np.zeros(self.grid.size)
        terms = local_forces(
            self.grid, self.atoms, self.potentials, self.specifications,
            density, zeros, zeros,
        )
        self.assertTrue(np.all(terms.self_consistency == 0.0))
        self.assertTrue(np.all(terms.core_correction == 0.0))
        step = 2.0e-3

        def energy(atoms):
            field = build_local_ionic_potential(
                self.grid, atoms, self.potentials, self.specifications
            )
            return self.grid.integrate(density * field)

        for index in range(len(self.atoms)):
            for axis in range(3):
                numerical = -(
                    energy(_moved(self.atoms, index, axis, step))
                    - energy(_moved(self.atoms, index, axis, -step))
                ) / (2.0 * step)
                # PARSEC integrates the tabulated slope of the potential,
                # which is not the slope of the piecewise-linear interpolant
                # the potential itself is sampled from: point by point the
                # two differ by about a percent.
                self.assertAlmostEqual(
                    terms.ionic[index, axis],
                    numerical,
                    delta=1e-2 + 3e-2 * abs(numerical),
                )

    def test_self_consistency_term_is_the_gradient_of_the_atomic_density(self) -> None:
        coordinates = self.grid.coordinates
        residual = 0.05 * np.cos(0.4 * coordinates[:, 0]) * np.exp(
            -0.1 * np.einsum("ij,ij->i", coordinates, coordinates)
        )
        zeros = np.zeros(self.grid.size)
        terms = local_forces(
            self.grid, self.atoms, self.potentials, self.specifications,
            zeros, residual, zeros,
        )
        step = 2.0e-3

        def functional(atoms):
            density = superpose_atomic_density(
                self.grid, atoms, self.potentials, self.specifications
            )
            return self.grid.integrate(density * residual)

        for index in range(len(self.atoms)):
            for axis in range(3):
                numerical = -(
                    functional(_moved(self.atoms, index, axis, step))
                    - functional(_moved(self.atoms, index, axis, -step))
                ) / (2.0 * step)
                self.assertAlmostEqual(
                    terms.self_consistency[index, axis],
                    numerical,
                    delta=5e-4 + 3e-2 * abs(numerical),
                )

    def test_nonlocal_force_is_the_gradient_of_the_nonlocal_energy(self) -> None:
        rng = np.random.default_rng(11)
        coordinates = self.grid.coordinates
        envelope = np.exp(-0.25 * np.einsum("ij,ij->i", coordinates, coordinates))
        orbitals = np.column_stack(
            [
                envelope * (1.0 + 0.3 * coordinates @ rng.normal(size=3))
                for _ in range(3)
            ]
        )
        orbitals /= np.linalg.norm(orbitals, axis=0)
        occupations = np.array([1.0, 0.6, 0.0])

        def energy(atoms):
            operator = build_nonlocal_projectors(
                self.grid, atoms, self.potentials, self.specifications
            )
            overlaps = operator.projectors.T @ orbitals
            return 2.0 * float(
                np.sum(operator.signs[:, None] * overlaps**2 * occupations[None, :])
            )

        operator = build_nonlocal_projectors(
            self.grid, self.atoms, self.potentials, self.specifications
        )
        gradients = build_nonlocal_projector_gradients(
            self.grid, self.atoms, self.potentials, self.specifications
        )
        self.assertEqual(gradients.labels, operator.labels)
        forces = nonlocal_forces(
            operator, gradients, orbitals, occupations, len(self.atoms)
        )
        # Hydrogen has no projector: its local channel is its only channel.
        self.assertTrue(np.all(forces[1:] == 0.0))
        step = 2.0e-3
        for axis in range(3):
            numerical = -(
                energy(_moved(self.atoms, 0, axis, step))
                - energy(_moved(self.atoms, 0, axis, -step))
            ) / (2.0 * step)
            self.assertAlmostEqual(
                forces[0, axis], numerical, delta=1e-2 + 6e-2 * abs(numerical)
            )


class SymmetryTests(unittest.TestCase):
    def test_cubic_operations_form_the_group_of_the_cube(self) -> None:
        operations = cubic_point_operations()
        self.assertEqual(operations.shape, (48, 3, 3))
        np.testing.assert_array_equal(operations[0], np.eye(3))
        for matrix in operations:
            np.testing.assert_array_equal(matrix @ matrix.T, np.eye(3))
        products = {tuple((a @ b).reshape(-1)) for a in operations[:6] for b in operations}
        self.assertEqual(len(products), 48)

    def test_operations_of_a_planar_molecule(self) -> None:
        atoms = (
            Atom("O", (0.0, -0.2, 0.0)),
            Atom("H", (1.6, 0.2, 0.0)),
            Atom("H", (-1.6, 0.2, 0.0)),
        )
        operations = atom_symmetry_operations(atoms)
        self.assertEqual(len(operations), 4)
        np.testing.assert_array_equal(operations[0].image, [0, 1, 2])
        swaps = sum(int(np.array_equal(op.image, [0, 2, 1])) for op in operations)
        self.assertEqual(swaps, 2)
        self.assertEqual(len(atom_symmetry_operations(_water())), 1)

    def test_symmetrization_projects_onto_invariant_forces(self) -> None:
        atoms = (
            Atom("O", (0.0, -0.2, 0.0)),
            Atom("H", (1.6, 0.2, 0.0)),
            Atom("H", (-1.6, 0.2, 0.0)),
        )
        operations = atom_symmetry_operations(atoms)
        rng = np.random.default_rng(5)
        forces = rng.normal(size=(3, 3))
        symmetric = symmetrize_forces(forces, operations)
        np.testing.assert_allclose(
            symmetrize_forces(symmetric, operations), symmetric, atol=1e-15
        )
        np.testing.assert_allclose(symmetric[:, 2], 0.0, atol=1e-15)
        self.assertAlmostEqual(symmetric[0, 0], 0.0, places=15)
        self.assertAlmostEqual(symmetric[1, 0], -symmetric[2, 0], places=15)
        self.assertAlmostEqual(symmetric[1, 1], symmetric[2, 1], places=15)
        # The mean force of each orbit along invariant directions is kept.
        self.assertAlmostEqual(symmetric[0, 1], forces[0, 1], places=15)
        self.assertAlmostEqual(
            symmetric[1, 1], 0.5 * (forces[1, 1] + forces[2, 1]), places=15
        )

    def test_net_force_and_torque(self) -> None:
        forces = np.array([[1.0, 0.0, 0.5], [0.0, 2.0, 0.0], [0.5, 1.0, -2.0]])
        balanced, net = remove_net_force(forces)
        np.testing.assert_allclose(net, [1.5, 3.0, -1.5])
        np.testing.assert_allclose(balanced.sum(axis=0), 0.0, atol=1e-15)
        positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(
            total_torque(positions, forces), np.cross(positions, forces).sum(axis=0)
        )


if __name__ == "__main__":
    unittest.main()
