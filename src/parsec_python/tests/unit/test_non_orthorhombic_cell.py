"""Periodic cells with oblique lattice vectors (fcc, bcc, hexagonal, triclinic ...).

The checks are independent of the stencil bookkeeping: plane waves and the exact eigenvalues of a smooth operator
for the derivatives, the same crystal in two descriptions for the energies, and brute-force sums for the images.
The Fortran code's results for the fcc diamond cell are the reference of the full calculation (see the
documentation of ``Laplacian.lattice_metric``).
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python.Grid import build_periodic_grid
from parsec_python.Grid.kpoints import monkhorst_pack_grid
from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.Laplacian import (
    build_gradient,
    build_negative_laplacian,
    cartesian_derivative_matrix,
    inverse_metric,
    second_derivative_plan,
)
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.SCF import prepare_periodic_single_point, run_scf
from parsec_python.V_ion import ewald, load_pseudopotentials
from parsec_python.V_xc.pbe import pbe, pbe_spin_polarized

_H_POTRE = Path(__file__).resolve().parents[1] / "data" / "H_POTRE.DAT"

A = 12.0
CELLS = {
    "fcc": np.array([[0, A / 2, A / 2], [A / 2, 0, A / 2], [A / 2, A / 2, 0]]),
    "bcc": np.array([[-A / 2, A / 2, A / 2], [A / 2, -A / 2, A / 2], [A / 2, A / 2, -A / 2]]),
    "hexagonal": np.array([[8.0, 0.0, 0.0], [-4.0, 6.928203230275509, 0.0], [0.0, 0.0, 9.0]]),
    "triclinic": np.array([[8.0, 0.0, 0.0], [2.5, 7.5, 0.0], [1.0, 0.7, 9.0]]),
    "sheared cubic": np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [8.0, 0.0, 8.0]]),
}


def _grid(lattice: np.ndarray, spacing: float = 0.5, order: int = 12):
    return build_periodic_grid(
        PeriodicCell(np.asarray(lattice, dtype=float)),
        PeriodicGridSettings(spacing=spacing, expansion_order=order),
    )


class ObliqueGridTests(unittest.TestCase):
    def test_points_follow_the_lattice_vectors_with_an_even_count_per_axis(self) -> None:
        for name, lattice in CELLS.items():
            with self.subTest(name):
                grid = _grid(lattice)
                self.assertFalse(grid.is_orthorhombic)
                self.assertTrue(np.all(grid.points_per_axis % 2 == 0))
                steps = lattice / grid.points_per_axis[:, None]
                np.testing.assert_allclose(grid.step_vectors, steps)
                first = grid.integer_coordinates[:5]
                np.testing.assert_allclose(grid.coordinates[:5], first @ steps, atol=1.0e-13)
                self.assertAlmostEqual(grid.volume_element * grid.size, abs(np.linalg.det(lattice)))
                # the wrap of an index by one cell is the identity
                shifted = grid.integer_coordinates + grid.points_per_axis * np.array([1, -2, 3])
                np.testing.assert_array_equal(grid.rows_for_integer_coordinates(shifted), np.arange(grid.size))
                np.testing.assert_allclose(
                    grid.physical_coordinates(grid.integer_coordinates[:7]), grid.coordinates[:7], atol=1.0e-13
                )

    def test_orthorhombic_cell_is_unchanged(self) -> None:
        grid = _grid(8.0 * np.eye(3), spacing=0.5)
        self.assertTrue(grid.is_orthorhombic)
        self.assertEqual(grid.shape, (16, 16, 16))
        self.assertAlmostEqual(grid.spacing, 0.5)

    def test_cell_too_small_for_the_stencil_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _grid(np.array([[3.0, 0, 0], [1.0, 3.0, 0], [0, 0, 3.0]]), spacing=0.5)


class SecondDerivativePlanTests(unittest.TestCase):
    def _reconstructed_metric(self, plan, steps):
        metric = np.diag(plan.axial).astype(float)
        for direction, weight in zip(plan.directions, plan.weights):
            d = np.asarray(direction, dtype=float)
            metric += weight * np.outer(d, d)
        return metric

    def test_plan_reproduces_the_inverse_metric(self) -> None:
        rng = np.random.default_rng(3)
        lattices = list(CELLS.values())
        for _ in range(6):
            base = np.eye(3) * 8.0 + rng.uniform(-2.5, 2.5, (3, 3))
            lattices.append(base)
        for lattice in lattices:
            steps = lattice / np.round(np.linalg.norm(lattice, axis=1) / 0.5)[:, None]
            plan = second_derivative_plan(steps)
            np.testing.assert_allclose(
                self._reconstructed_metric(plan, steps), inverse_metric(steps), rtol=1e-10, atol=1e-10
            )

    def test_known_direction_sets(self) -> None:
        orthogonal = second_derivative_plan(np.diag([0.5, 0.4, 0.6]))
        self.assertEqual(orthogonal.directions, ())
        np.testing.assert_allclose(orthogonal.axial, 1.0 / np.array([0.5, 0.4, 0.6]) ** 2)

        hexagonal = second_derivative_plan(CELLS["hexagonal"] / np.array([[16], [16], [18]]))
        self.assertEqual(hexagonal.directions, ((1, 1, 0),))  # 120 degrees: the short diagonal is u+v

        fcc = second_derivative_plan(CELLS["fcc"] / 19.0)
        self.assertEqual(set(fcc.directions), {(-1, 0, 1), (-1, 1, 0), (0, -1, 1)})
        np.testing.assert_allclose(fcc.weights, fcc.weights[0])

        bcc = second_derivative_plan(CELLS["bcc"] / 23.0)
        self.assertEqual(bcc.directions, ((-1, -1, -1),))

    def test_sheared_cubic_cell_needs_only_a_cartesian_axis(self) -> None:
        # a3 = a1 + a3_cubic: the direction u1 - u3 is the Cartesian z axis
        plan = second_derivative_plan(np.array([[0.5, 0, 0], [0, 0.5, 0], [0.5, 0, 0.5]]))
        self.assertEqual(plan.directions, ((1, 0, -1),))
        # nabla^2 = 4 d1^2 + 4 d2^2 + 4 (d3 - d1)^2: axial weights 4, 4, 0 and 4 on the direction
        np.testing.assert_allclose(plan.axial, [4.0, 4.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(plan.weights, [4.0], atol=1e-12)


class ObliqueOperatorTests(unittest.TestCase):
    def test_plane_waves(self) -> None:
        for name, lattice in CELLS.items():
            with self.subTest(name):
                grid = _grid(lattice, spacing=0.5 if name != "fcc" else 0.45)
                laplacian = build_negative_laplacian(grid)
                gradient = build_gradient(grid)
                reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
                for m in [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, -1, 1)]:
                    k = np.asarray(m) @ reciprocal
                    phase = grid.coordinates @ k + 0.3
                    values = np.cos(phase)
                    np.testing.assert_allclose(laplacian @ values, (k @ k) * values, atol=2.0e-6 * (k @ k))
                    for axis in range(3):
                        np.testing.assert_allclose(
                            gradient[axis] @ values, -k[axis] * np.sin(phase), atol=2.0e-6 * np.linalg.norm(k)
                        )

    def test_laplacian_is_symmetric_and_positive_semidefinite(self) -> None:
        grid = _grid(CELLS["triclinic"], spacing=0.6, order=8)
        laplacian = build_negative_laplacian(grid)
        self.assertEqual(abs(laplacian - laplacian.T).max(), 0.0)
        probe = np.random.default_rng(1).standard_normal((grid.size, 4))
        self.assertGreater(np.min(np.einsum("ij,ij->j", probe, laplacian @ probe)), 0.0)
        # the constant is its null vector
        np.testing.assert_allclose(laplacian @ np.ones(grid.size), 0.0, atol=1.0e-11)

    def test_gradient_matrices_are_antisymmetric(self) -> None:
        grid = _grid(CELLS["bcc"], spacing=0.6, order=8)
        for matrix in build_gradient(grid):
            self.assertLess(abs(matrix + matrix.T).max(), 1.0e-13)

    def test_rotated_orthorhombic_cell_gives_the_orthorhombic_operator(self) -> None:
        # a1, a2 exchanged: the same crystal and points, reached through the oblique code path
        rotated = _grid(np.array([[0, 8.0, 0], [8.0, 0, 0], [0, 0, 8.0]]))
        plain = _grid(8.0 * np.eye(3))
        self.assertFalse(rotated.is_orthorhombic)
        # row of the rotated grid -> row of the plain grid
        index = rotated.integer_coordinates
        rows = plain.rows_for_integer_coordinates(index[:, [1, 0, 2]])
        values = np.random.default_rng(5).standard_normal(plain.size)
        np.testing.assert_allclose(
            (build_negative_laplacian(rotated) @ values[rows]), (build_negative_laplacian(plain) @ values)[rows], atol=1.0e-11
        )
        gradient_rotated = build_gradient(rotated)
        gradient_plain = build_gradient(plain)
        for axis in range(3):
            np.testing.assert_allclose(
                gradient_rotated[axis] @ values[rows], (gradient_plain[axis] @ values)[rows], atol=1.0e-11
            )

    def test_same_grid_points_in_two_descriptions_give_the_same_operator(self) -> None:
        # cubic cell L and the cell with a3' = a3 + a1: the grid point sets coincide when n3' = n1
        n = 16
        length = 8.0
        cubic = _grid(length * np.eye(3), spacing=length / n)
        sheared_cell = PeriodicCell(np.array([[length, 0, 0], [0, length, 0], [length, 0, length]]))
        counts = np.array([n, n, n])
        mesh = np.meshgrid(*[np.arange(k) for k in counts], indexing="ij")
        indices = np.column_stack([m.reshape(-1) for m in mesh])
        from parsec_python.Grid.pbc import PeriodicRealSpaceGrid

        sheared = PeriodicRealSpaceGrid(
            cell=sheared_cell,
            settings=cubic.settings,
            points_per_axis=counts,
            integer_coordinates=indices,
            coordinates=indices.astype(float) @ (sheared_cell.lattice_vectors / counts[:, None]),
            lookup=np.arange(n**3).reshape(n, n, n),
        )
        rows = cubic.rows_for_integer_coordinates(
            np.column_stack([indices[:, 0] + indices[:, 2], indices[:, 1], indices[:, 2]])
        )
        values = np.random.default_rng(2).standard_normal(cubic.size)
        np.testing.assert_allclose(
            build_negative_laplacian(sheared) @ values[rows], (build_negative_laplacian(cubic) @ values)[rows], atol=1.0e-10
        )

    def test_derivative_matrix_is_the_inverse_of_the_steps(self) -> None:
        steps = CELLS["triclinic"] / 20.0
        np.testing.assert_allclose(cartesian_derivative_matrix(steps) @ steps, np.eye(3), atol=1e-12)


class ObliqueExchangeCorrelationTests(unittest.TestCase):
    def test_pbe_potential_is_the_derivative_of_the_energy(self) -> None:
        for name in ("fcc", "triclinic"):
            with self.subTest(name):
                lattice = CELLS[name]
                grid = _grid(lattice, spacing=0.6, order=8)
                reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
                density = 0.05 * (
                    1.2
                    + 0.4 * np.cos(grid.coordinates @ reciprocal[0])
                    + 0.3 * np.sin(grid.coordinates @ (reciprocal[1] + reciprocal[2]))
                )
                direction = np.random.default_rng(1).standard_normal(grid.size) * 5.0e-5
                step = 1.0e-4
                result = pbe(density, grid)
                numerical = (
                    pbe(density + step * direction, grid).total_energy
                    - pbe(density - step * direction, grid).total_energy
                ) / (2.0 * step)
                analytic = float(np.sum(result.potential * direction) * grid.volume_element)
                self.assertAlmostEqual(numerical / analytic, 1.0, places=5)

                up, down = 0.55 * density, 0.45 * density
                spin = pbe_spin_polarized(up, down, grid)
                numerical = (
                    pbe_spin_polarized(up + step * direction, down, grid).total_energy
                    - pbe_spin_polarized(up - step * direction, down, grid).total_energy
                ) / (2.0 * step)
                analytic = float(np.sum(spin.potential_up * direction) * grid.volume_element)
                self.assertAlmostEqual(numerical / analytic, 1.0, places=5)

    def test_pbe_of_an_orthogonal_description_matches_the_orthorhombic_cell(self) -> None:
        rotated = _grid(np.array([[0, 8.0, 0], [8.0, 0, 0], [0, 0, 8.0]]), spacing=0.5, order=8)
        plain = _grid(8.0 * np.eye(3), spacing=0.5, order=8)
        rows = plain.rows_for_integer_coordinates(rotated.integer_coordinates[:, [1, 0, 2]])
        k = 2.0 * np.pi / 8.0
        density = 0.04 * (1.3 + np.cos(k * plain.coordinates[:, 0]) * np.sin(2 * k * plain.coordinates[:, 1]))
        np.testing.assert_allclose(
            pbe(density[rows], rotated).potential, pbe(density, plain).potential[rows], atol=1.0e-12
        )


class ObliqueEwaldTests(unittest.TestCase):
    def setUp(self) -> None:
        specifications = {"H": SpeciesPotential(_H_POTRE, 0)}
        self.specifications = specifications
        self.potentials = load_pseudopotentials(specifications, xc_functional="ca")

    def _atoms_in(self, lattice, fractions):
        return [Atom("H", np.asarray(f) @ lattice) for f in fractions]

    def test_ion_ion_energy_does_not_depend_on_the_cell_chosen(self) -> None:
        # a simple cubic lattice of spacing 6 described by three different cells (1 and 4 atoms)
        a = 6.0
        primitive = a * np.eye(3)
        conventional2 = np.array([[2 * a, 0, 0], [0, a, 0], [0, 0, a]])
        sheared = np.array([[a, 0, 0], [0, a, 0], [a, a, a]])
        reference = ewald.ewald_ion_ion_energy([Atom("H", [0.0, 0.0, 0.0])], self.potentials, primitive)
        two = ewald.ewald_ion_ion_energy(
            [Atom("H", [0.0, 0.0, 0.0]), Atom("H", [a, 0.0, 0.0])], self.potentials, conventional2
        )
        one_sheared = ewald.ewald_ion_ion_energy([Atom("H", [0.0, 0.0, 0.0])], self.potentials, sheared)
        self.assertAlmostEqual(two / 2.0, reference, places=9)
        # the sheared cell has the same volume as the simple cubic one (det = a^3) and holds one atom
        self.assertAlmostEqual(one_sheared, reference, places=9)

    def test_local_potential_is_the_same_in_two_descriptions(self) -> None:
        length = 6.4
        n = 16
        cubic_cell = PeriodicCell(length * np.eye(3))
        cubic = build_periodic_grid(cubic_cell, PeriodicGridSettings(spacing=length / n, expansion_order=4))
        sheared_lattice = np.array([[length, 0, 0], [0, length, 0], [length, 0, length]])
        sheared_cell = PeriodicCell(sheared_lattice)
        counts = np.array([n, n, n])
        mesh = np.meshgrid(*[np.arange(k) for k in counts], indexing="ij")
        indices = np.column_stack([m.reshape(-1) for m in mesh])
        from parsec_python.Grid.pbc import PeriodicRealSpaceGrid

        sheared = PeriodicRealSpaceGrid(
            cell=sheared_cell,
            settings=cubic.settings,
            points_per_axis=counts,
            integer_coordinates=indices,
            coordinates=indices.astype(float) @ (sheared_lattice / counts[:, None]),
            lookup=np.arange(n**3).reshape(n, n, n),
        )
        rows = cubic.rows_for_integer_coordinates(
            np.column_stack([indices[:, 0] + indices[:, 2], indices[:, 1], indices[:, 2]])
        )
        position = np.array([1.3, 2.1, 4.4])
        fraction = (position @ np.linalg.inv(sheared_lattice))
        folded = (fraction - np.floor(fraction)) @ sheared_lattice
        v_cubic = ewald.ewald_local_ionic_potential(
            cubic, [Atom("H", position)], self.potentials, self.specifications, cubic_cell.lattice_vectors
        )
        v_sheared = ewald.ewald_local_ionic_potential(
            sheared, [Atom("H", folded)], self.potentials, self.specifications, sheared_lattice
        )
        np.testing.assert_allclose(v_sheared, v_cubic[rows], atol=1.0e-9)


class ObliqueKPointTests(unittest.TestCase):
    def test_monkhorst_pack_points_are_combinations_of_the_reciprocal_vectors(self) -> None:
        lattice = CELLS["fcc"]
        points, weights = monkhorst_pack_grid(PeriodicCell(lattice), (2, 2, 2))
        reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
        fractions = points @ np.linalg.inv(reciprocal)
        np.testing.assert_allclose(np.abs(fractions), 0.25, atol=1e-12)
        np.testing.assert_allclose(weights, 1.0 / 8.0)


def _hydrogen(lattice: np.ndarray, atoms_cartesian, spacing: float = 0.5) -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("H", p) for p in atoms_cartesian],
        pseudopotentials={"H": SpeciesPotential(_H_POTRE, 0)},
        grid=PeriodicGridSettings(spacing=spacing, expansion_order=8),
        periodic_cell=PeriodicCell(np.asarray(lattice, dtype=float)),
        scf=SCFSettings(max_iterations=60, number_of_states=3, convergence_criterion=1.0e-7),
        hartree=HartreeSettings(relative_tolerance=1.0e-9, absolute_tolerance=1.0e-12, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-7),
    )


class ObliqueSelfConsistentFieldTests(unittest.TestCase):
    def test_rotated_orthorhombic_cell_gives_the_orthorhombic_energy(self) -> None:
        plain = run_scf(prepare_periodic_single_point(_hydrogen(8.0 * np.eye(3), [[4.0, 4.0, 4.0]])))
        rotated = run_scf(
            prepare_periodic_single_point(
                _hydrogen(np.array([[0, 8.0, 0], [8.0, 0, 0], [0, 0, 8.0]]), [[4.0, 4.0, 4.0]])
            )
        )
        self.assertTrue(plain.converged and rotated.converged)
        self.assertAlmostEqual(rotated.energies.total, plain.energies.total, places=7)

    def test_two_descriptions_of_one_crystal_agree(self) -> None:
        # simple cubic H lattice (a = 7): the cubic cell and a cell with a3' = a3 + a1; the grids differ,
        # the energies agree to the discretization error
        a = 7.0
        cubic = run_scf(prepare_periodic_single_point(_hydrogen(a * np.eye(3), [[0.5 * a] * 3], spacing=0.4)))
        lattice = np.array([[a, 0, 0], [0, a, 0], [a, 0, a]])
        position = np.array([0.5 * a] * 3)
        fraction = position @ np.linalg.inv(lattice)
        folded = (fraction - np.floor(fraction)) @ lattice
        sheared = run_scf(prepare_periodic_single_point(_hydrogen(lattice, [folded], spacing=0.4)))
        self.assertTrue(cubic.converged and sheared.converged)
        self.assertAlmostEqual(sheared.energies.total, cubic.energies.total, delta=2.0e-3)

    def test_atom_outside_an_oblique_cell_is_rejected(self) -> None:
        lattice = CELLS["hexagonal"]
        problem = _hydrogen(lattice, [[-1.0, 0.5, 0.5]])
        with self.assertRaises(ValueError):
            prepare_periodic_single_point(problem)


_FCC_INPUT = """
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  0.0 0.5 0.5
  0.5 0.0 0.5
  0.5 0.5 0.0
end Cell_Shape
Lattice_Vector_Scale: 10.0 bohr
Coordinate_Unit: lattice_vectors
Grid_Spacing: 0.5 bohr
Expansion_Order: 4
Atom_Types_Num: 1
Atom_Type: H
Element_Symbol: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
  1.25 -0.5 0.25
end Atom_Coord
Correlation_Type: ca
States_Num: 3
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
2 2 2
end Monkhorst_Pack_Grid
"""


class ObliqueInputTests(unittest.TestCase):
    def test_fractional_coordinates_are_folded_into_the_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "parsec.in").write_text(_FCC_INPUT)
            (folder / "H_POTRE.DAT").write_bytes(_H_POTRE.read_bytes())
            problem = parse_parsec_input(folder / "parsec.in").problem
        lattice = problem.periodic_cell.lattice_vectors
        np.testing.assert_allclose(lattice, 10.0 * 0.5 * (np.ones((3, 3)) - np.eye(3)))
        fraction = problem.atoms[0].position @ np.linalg.inv(lattice)
        np.testing.assert_allclose(fraction, [0.25, 0.5, 0.25], atol=1e-12)
        self.assertEqual(problem.grid.box_lengths, tuple(np.linalg.norm(lattice, axis=1)))
        self.assertTrue(problem.has_kpoint_sampling)


if __name__ == "__main__":
    unittest.main()
