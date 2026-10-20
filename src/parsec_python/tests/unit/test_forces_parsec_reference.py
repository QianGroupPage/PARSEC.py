"""Forces of the readable path against Fortran PARSEC, end to end.

One small water molecule of low symmetry is solved to self-consistency on
the full grid and every force table is compared with the ``parsec.out`` of
Fortran PARSEC for the same input.  The accelerated package repeats the
comparison for more structures (core correction, p and d projectors, cubic
splines), where a calculation takes a second.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.Forces import compute_forces
from parsec_python.Input import parse_parsec_input
from parsec_python.SCF import (
    prepare_single_point,
    run_scf,
    update_geometry,
    warm_start_for_moved_atoms,
)
from parsec_python.SCF.single_point import (
    DensityHistory,
    SCFWarmStart,
    atomic_density_guess,
    extrapolated_warm_start,
)
from parsec_python.models import Atom
from parsec_python.tests.force_reference import (
    assert_matches_fortran,
    load_cases,
    write_case,
)


class ReadablePathForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = load_cases()["water_lda_micro"]
        with tempfile.TemporaryDirectory() as directory:
            translation = parse_parsec_input(write_case(cls.case, Path(directory)))
            cls.system = prepare_single_point(translation.problem)
        cls.result = run_scf(cls.system)
        cls.forces = compute_forces(cls.system, cls.result)

    def test_forces_match_fortran_parsec(self) -> None:
        self.assertTrue(self.result.converged)
        assert_matches_fortran(self, self.case, self.result.energies.total, self.forces)
        np.testing.assert_allclose(self.forces.total.sum(axis=0), 0.0, atol=1e-12)
        self.assertEqual(self.forces.symmetry_operations, 1)

    def test_moved_atoms_continue_from_the_previous_loop(self) -> None:
        shift = np.array([[0.0, 0.02, 0.0], [0.03, -0.01, 0.01], [-0.02, -0.01, 0.0]])
        atoms = tuple(
            Atom(atom.symbol, atom.position + delta)
            for atom, delta in zip(self.system.atoms, shift)
        )
        moved = update_geometry(self.system, atoms)
        self.assertIs(moved.grid, self.system.grid)
        self.assertIs(moved.negative_laplacian, self.system.negative_laplacian)
        warm = run_scf(
            moved,
            warm_start=warm_start_for_moved_atoms(self.system, self.result, moved),
        )
        cold = run_scf(moved)
        self.assertTrue(warm.converged and cold.converged)
        self.assertLess(warm.iterations, cold.iterations)
        self.assertAlmostEqual(warm.energies.total, cold.energies.total, delta=2e-8)
        np.testing.assert_allclose(
            compute_forces(moved, warm).total,
            compute_forces(moved, cold).total,
            rtol=0,
            atol=5e-7,
        )
        # The moved structure has a different energy: the loop did not just
        # return what it was handed.
        self.assertGreater(abs(warm.energies.total - self.result.energies.total), 1e-4)

    def test_extrapolated_starts_reach_the_same_structures(self) -> None:
        """Potential_Extrapolation changes the start of a loop, not its end.

        Two moves along a straight line: the first-order (atomic) and the
        second-order extrapolation converge to what a cold start gives, and
        the second move of second order uses the change of the first.
        """

        shift = np.array([[0.0, 0.02, 0.0], [0.03, -0.01, 0.01], [-0.02, -0.01, 0.0]])
        colds = {}
        for extrapolation in ("atomic", "second_order"):
            with self.subTest(extrapolation=extrapolation):
                history = DensityHistory()
                system, result = self.system, self.result
                for move in range(2):
                    atoms = tuple(
                        Atom(atom.symbol, atom.position + delta)
                        for atom, delta in zip(system.atoms, shift)
                    )
                    moved = update_geometry(system, atoms)
                    start = warm_start_for_moved_atoms(
                        system, result, moved, extrapolation=extrapolation, history=history
                    )
                    first_order = atomic_density_guess(system, result, moved)
                    self.assertIsNotNone(start.density_guess)
                    warm = run_scf(moved, warm_start=start)
                    if move not in colds:
                        cold = run_scf(moved)
                        self.assertTrue(cold.converged)
                        colds[move] = (cold.energies.total, compute_forces(moved, cold).total)
                    self.assertTrue(warm.converged)
                    self.assertAlmostEqual(warm.energies.total, colds[move][0], delta=2e-8)
                    np.testing.assert_allclose(
                        compute_forces(moved, warm).total, colds[move][1], rtol=0, atol=5e-7
                    )
                    system, result = moved, warm
                # The guess of the second move: of first order for "atomic",
                # with the change of the first move for "second_order".
                difference = np.abs(start.density_guess - first_order).max()
                if extrapolation == "atomic":
                    self.assertEqual(difference, 0.0)
                else:
                    self.assertGreater(difference, 1e-6)

    def test_atoms_must_stay_inside_the_domain(self) -> None:
        atoms = list(self.system.atoms)
        atoms[1] = Atom("H", (0.0, 0.0, self.system.grid.settings.radius + 0.1))
        with self.assertRaisesRegex(ValueError, "outside the domain"):
            update_geometry(self.system, atoms)
        with self.assertRaisesRegex(ValueError, "number, order and species"):
            update_geometry(self.system, atoms[:2])


def _structure(positions, atoms_density, **extra):
    return SimpleNamespace(
        atoms=tuple(Atom("H", position) for position in positions),
        initial_density=np.asarray(atoms_density, dtype=np.float64),
        input=SimpleNamespace(
            initial_density_settings=None,
            eigensolver=SimpleNamespace(filter_degree=10, first_filter_degree=30),
        ),
        **extra,
    )


class ExtrapolationFormulaTests(unittest.TestCase):
    """The guessed densities on four grid points."""

    def test_atomic_guess(self) -> None:
        old = _structure([[0.0, 0.0, 0.0]], [1.0, 1.0, 0.0, 0.0])
        new = _structure([[0.1, 0.0, 0.0]], [0.0, 1.0, 1.0, 0.0])
        result = SimpleNamespace(density=[1.5, 0.25, 0.25, 0.0])
        # rho_out - rho_atoms(old) + rho_atoms(new), not negative.
        np.testing.assert_array_equal(
            atomic_density_guess(old, result, new), [0.5, 0.25, 1.25, 0.0]
        )
        result = SimpleNamespace(density=[0.5, 1.0, 0.5, 0.0])
        np.testing.assert_array_equal(
            atomic_density_guess(old, result, new), [0.0, 1.0, 1.5, 0.0]
        )
        # The old superposition is scaled to the electron count of the new one.
        heavier = _structure([[0.1, 0.0, 0.0]], [0.0, 2.0, 2.0, 0.0])
        np.testing.assert_array_equal(
            atomic_density_guess(old, SimpleNamespace(density=[2.0, 2.0, 0.0, 0.0]), heavier),
            [0.0, 2.0, 2.0, 0.0],
        )

    def test_no_guess_without_superposed_atoms_or_with_spin(self) -> None:
        old = _structure([[0.0, 0.0, 0.0]], [1.0, 1.0, 0.0, 0.0])
        new = _structure([[0.1, 0.0, 0.0]], [0.0, 1.0, 1.0, 0.0])
        spin = SimpleNamespace(density=[1.0, 1.0, 0.0, 0.0], spin=(None, None))
        self.assertIsNone(atomic_density_guess(old, spin, new))
        imported = _structure([[0.0, 0.0, 0.0]], [1.0, 1.0, 0.0, 0.0])
        imported.input.initial_density_settings = SimpleNamespace(method="file")
        result = SimpleNamespace(density=[1.0, 1.0, 0.0, 0.0])
        self.assertIsNone(atomic_density_guess(imported, result, new))

    def test_second_order_follows_one_trajectory(self) -> None:
        first = _structure([[0.0, 0.0, 0.0]], [2.0, 0.0, 0.0, 0.0])
        second = _structure([[0.1, 0.0, 0.0]], [0.0, 2.0, 0.0, 0.0])
        third = _structure([[0.2, 0.0, 0.0]], [0.0, 0.0, 2.0, 0.0])
        history = DensityHistory()
        # First move: of first order; d(first) = [-0.5, 0.5, 0, 0].
        np.testing.assert_array_equal(
            atomic_density_guess(
                first, SimpleNamespace(density=[1.5, 0.5, 0.0, 0.0]), second, history
            ),
            [0.0, 2.5, 0.0, 0.0],
        )
        # Second move: rho_atoms(third) + 2 d(second) - d(first), with
        # d(second) = [0, -0.5, 0.5, 0].
        result = SimpleNamespace(density=[0.0, 1.5, 0.5, 0.0])
        np.testing.assert_array_equal(
            atomic_density_guess(second, result, third, history),
            [0.5, 0.0, 3.0, 0.0],
        )
        # A move that does not start where the last one ended is of first order.
        np.testing.assert_array_equal(
            atomic_density_guess(second, result, third, history),
            [0.0, 0.0, 2.5, 0.0],
        )

    def test_warm_start_of_each_extrapolation(self) -> None:
        old = _structure([[0.0, 0.0, 0.0]], [1.0, 1.0, 0.0, 0.0], ionic_potential=np.full(4, -1.0))
        new = _structure([[0.1, 0.0, 0.0]], [0.0, 1.0, 1.0, 0.0])
        result = SimpleNamespace(
            density=[1.0, 1.0, 0.0, 0.0], output_effective_potential=np.full(4, 0.5)
        )
        start = SimpleNamespace(input_potential=np.zeros(4))
        self.assertIs(extrapolated_warm_start(start, old, result, new, "none"), start)
        with self.assertRaisesRegex(ValueError, "Potential_Extrapolation"):
            extrapolated_warm_start(start, old, result, new, "linear")
        start = SCFWarmStart(input_potential=np.zeros(4), hartree_potential=np.zeros(4))
        atomic = extrapolated_warm_start(start, old, result, new, "atomic")
        np.testing.assert_array_equal(atomic.density_guess, [0.0, 1.0, 1.0, 0.0])
        # The Hartree and XC part of the last output potential.
        np.testing.assert_array_equal(atomic.hxc_reference, np.full(4, 1.5))
        # One filter of the first-solve degree after the move.
        self.assertEqual(atomic.first_iteration_degree, 30)


if __name__ == "__main__":
    unittest.main()
