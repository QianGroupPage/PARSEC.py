"""Fock exchange operator, its compressed form (ACE) and the exchange energy.

Every factor of :mod:`parsec_python.V_xc.exchange_operator` is checked
against an independent statement:

* the energy against :func:`parsec_python.V_xc.exact_exchange.exact_exchange_energy`
  (its own pair loop) and its spin form;
* ``V_x q_i`` against the gradient of the energy (central differences):
  ``dE_x/dq_i = 2 w f_i V_x q_i``;
* the compressed operator against the uncompressed one on the span of the
  orbitals it was built from, and the bound off the span;
* the unpolarized orbital set against two equal spin channels;
* the cached multipole pair solver against the conjugate gradients of
  :func:`parsec_python.Hartree.solve_hartree`.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.V_xc.exact_exchange import (
    exact_exchange_energy,
    reference_pair_poisson,
    spin_exact_exchange_energy,
)
from parsec_python.V_xc.exchange_operator import (
    CallablePairPoisson,
    MultipolePairPoisson,
    apply_exchange,
    apply_exchange_to_orbitals,
    cached_pair_solver,
    compress_exchange,
    exchange_band_energy,
    reference_pair_solver,
)
from parsec_python.models import GridSettings, HartreeSettings


class ExchangeOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(GridSettings(spacing=0.6, radius=6.0, expansion_order=8))
        cls.laplacian = build_negative_laplacian(cls.grid)
        cls.settings = HartreeSettings(
            multipole_order=6, relative_tolerance=1.0e-11, max_iterations=4000
        )
        cls.solver = MultipolePairPoisson(cls.grid, cls.laplacian, cls.settings)
        x = cls.grid.coordinates
        r2 = np.einsum("ij,ij->i", x, x)
        g = np.exp(-0.5 * r2)
        columns = [
            g,
            x[:, 0] * g,
            x[:, 1] * g,
            (x[:, 2] - 0.3) * np.exp(-0.3 * r2),
            np.exp(-0.2 * ((x[:, 0] - 1.0) ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)),
        ]
        cls.orbitals, _ = np.linalg.qr(np.column_stack(columns))
        # Two full, one fractional, two empty orbitals.
        cls.occupations = np.array([1.0, 1.0, 0.6, 0.0, 0.0])
        cls.volume = cls.grid.volume_element
        cls.application = apply_exchange_to_orbitals(
            cls.orbitals, cls.occupations, cls.volume, cls.solver, spin_weight=2.0, batch_size=4
        )
        cls.r2 = r2

    def test_energy_is_that_of_the_pair_loop(self) -> None:
        reference = exact_exchange_energy(
            self.orbitals,
            self.occupations,
            self.volume,
            lambda rho: self.solver.solve_batch(rho[:, None])[:, 0],
        )
        self.assertAlmostEqual(self.application.energy / reference.energy, 1.0, delta=1.0e-12)
        # Three occupied orbitals: 6 symmetric pairs, two empty targets: 6 more.
        self.assertEqual(self.application.pair_solves, 6 + 2 * 3)
        self.assertEqual(self.application.occupied, (0, 1, 2))

    def test_symmetric_pairs_equal_the_operator_on_arbitrary_vectors(self) -> None:
        applied = apply_exchange(self.orbitals, self.occupations, self.orbitals, self.volume, self.solver)
        scale = np.max(np.abs(applied))
        self.assertLess(np.max(np.abs(applied - self.application.applied)) / scale, 1.0e-13)
        single = apply_exchange(
            self.orbitals, self.occupations, self.orbitals[:, 3], self.volume, self.solver
        )
        self.assertLess(np.max(np.abs(single - applied[:, 3])) / scale, 1.0e-13)

    def test_occupied_columns_first_then_the_rest(self) -> None:
        occupied = apply_exchange_to_orbitals(
            self.orbitals, self.occupations, self.volume, self.solver, columns=[0, 1, 2]
        )
        self.assertAlmostEqual(occupied.energy, self.application.energy, places=12)
        self.assertEqual(occupied.pair_solves, 6)
        complete = apply_exchange_to_orbitals(
            self.orbitals, self.occupations, self.volume, self.solver, known=occupied
        )
        self.assertEqual(complete.pair_solves, 12)
        self.assertLess(
            np.max(np.abs(complete.applied - self.application.applied))
            / np.max(np.abs(self.application.applied)),
            1.0e-13,
        )
        partial = apply_exchange_to_orbitals(
            self.orbitals, self.occupations, self.volume, self.solver, columns=[0]
        )
        self.assertTrue(np.isnan(partial.energy))

    def test_operator_is_the_gradient_of_the_energy(self) -> None:
        rng = np.random.default_rng(7)
        step = rng.standard_normal(self.orbitals.shape) * np.exp(-0.2 * self.r2)[:, None]
        step /= np.linalg.norm(step)

        def energy(orbitals):
            return apply_exchange_to_orbitals(
                orbitals, self.occupations, self.volume, self.solver, columns=[0, 1, 2]
            ).energy

        epsilon = 1.0e-4
        numeric = (energy(self.orbitals + epsilon * step) - energy(self.orbitals - epsilon * step)) / (
            2.0 * epsilon
        )
        gradient = 2.0 * 2.0 * self.occupations[None, :] * self.application.applied
        analytic = float(np.sum(gradient * step))
        # Central differences of a quartic form with solves converged to 1e-11:
        # the noise of the stopping tests, not the step, sets this bound.
        self.assertAlmostEqual(numeric / analytic, 1.0, delta=1.0e-6)

    def test_compressed_operator_on_and_off_the_span(self) -> None:
        compressed = compress_exchange(self.orbitals, self.application, fraction=0.25)
        scale = np.max(np.abs(self.application.applied))
        self.assertLess(
            np.max(np.abs(compressed.apply(self.orbitals, scaled=False) - self.application.applied))
            / scale,
            1.0e-9,
        )
        self.assertLess(
            np.max(np.abs(compressed.apply(self.orbitals) - 0.25 * self.application.applied)) / scale,
            1.0e-9,
        )
        self.assertLess(compressed.asymmetry, 1.0e-6)
        # 2 E_x(P, P) through the operator: the band-energy double counting.
        self.assertAlmostEqual(
            exchange_band_energy(compressed, self.orbitals, self.occupations, scaled=False)
            / (2.0 * self.application.energy),
            1.0,
            delta=1.0e-9,
        )
        # Off the span: 0 >= <phi|V_ACE|phi> >= <phi|V_x|phi>.
        x = self.grid.coordinates
        outside = x[:, 0] * x[:, 1] * np.exp(-0.4 * self.r2)
        outside -= self.orbitals @ (self.orbitals.T @ outside)
        outside /= np.linalg.norm(outside)
        exact = float(
            outside @ apply_exchange(self.orbitals, self.occupations, outside, self.volume, self.solver)
        )
        compressed_value = float(compressed.expectation_values(outside, scaled=False)[0])
        self.assertLessEqual(compressed_value, 0.0)
        self.assertGreaterEqual(compressed_value, exact - 1.0e-12)
        self.assertEqual(compressed.shape, (self.grid.size, self.grid.size))
        self.assertEqual(compressed.rank, 5)

    def test_two_equal_spin_channels_are_the_unpolarized_set(self) -> None:
        channel = apply_exchange_to_orbitals(
            self.orbitals, self.occupations, self.volume, self.solver, spin_weight=1.0
        )
        self.assertAlmostEqual(2.0 * channel.energy, self.application.energy, places=12)
        self.assertLess(np.max(np.abs(channel.applied - self.application.applied)), 1.0e-14)
        spin = spin_exact_exchange_energy(
            [self.orbitals, self.orbitals],
            [self.occupations, self.occupations],
            self.volume,
            lambda rho: self.solver.solve_batch(rho[:, None])[:, 0],
        )
        self.assertAlmostEqual(2.0 * channel.energy / spin.energy, 1.0, delta=1.0e-12)
        compressed = compress_exchange(self.orbitals, channel, spin_weight=1.0)
        self.assertAlmostEqual(
            exchange_band_energy(compressed, self.orbitals, self.occupations, scaled=False),
            2.0 * channel.energy,
            places=9,
        )

    def test_cached_solver_reproduces_the_reference_conjugate_gradients(self) -> None:
        reference = reference_pair_solver(self.grid, self.laplacian, self.settings)
        densities = self.orbitals[:, [0, 1, 3]] * self.orbitals[:, [0, 2, 4]] / self.volume
        cached = self.solver.solve_batch(densities)
        expected = reference.solve_batch(densities)
        self.assertLess(np.max(np.abs(cached - expected)) / np.max(np.abs(expected)), 1.0e-9)
        self.assertEqual(reference.solves, 3)
        self.assertGreater(reference.iterations, 0)
        self.assertIsInstance(cached_pair_solver(self.grid, self.laplacian, self.settings), MultipolePairPoisson)
        box = build_cluster_grid(
            GridSettings(spacing=0.8, radius=3.0, expansion_order=4, domain_shape="box")
        )
        self.assertIsInstance(
            cached_pair_solver(box, build_negative_laplacian(box), self.settings), CallablePairPoisson
        )
        # The reference wrapper drops the atomic tail as the energy refinement does.
        solve = reference_pair_poisson(self.grid, self.laplacian, self.settings)
        single = np.asarray(solve(densities[:, 0]).potential)
        self.assertLess(np.max(np.abs(single - expected[:, 0])) / np.max(np.abs(single)), 1.0e-12)

    def test_kronecker_solver_gives_the_same_exchange(self) -> None:
        """The direct solver on the bounding box against the CG on the sphere.

        Same finite-difference Laplacian and multipole order; the boundary
        values sit at the faces of the box instead of just outside the
        sphere, which changes the energy at the level of the boundary error.
        """

        from parsec_python.Hartree.kronecker import KroneckerPoisson

        kronecker = KroneckerPoisson(self.grid, kernel="coulomb", multipole_order=6)
        application = apply_exchange_to_orbitals(
            self.orbitals, self.occupations, self.volume, kronecker, spin_weight=2.0
        )
        self.assertLess(abs(application.energy / self.application.energy - 1.0), 1.0e-5)
        compressed = compress_exchange(self.orbitals, application)
        self.assertLess(compressed.asymmetry, 1.0e-8)

    def test_what_cannot_be_compressed_or_applied_is_refused(self) -> None:
        partial = apply_exchange_to_orbitals(
            self.orbitals, self.occupations, self.volume, self.solver, columns=[0, 1, 2]
        )
        with self.assertRaisesRegex(ValueError, "every orbital"):
            compress_exchange(self.orbitals, partial)
        with self.assertRaises(NotImplementedError):
            apply_exchange_to_orbitals(
                self.orbitals.astype(complex), self.occupations, self.volume, self.solver
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            apply_exchange_to_orbitals(
                self.orbitals, 2.0 * self.occupations, self.volume, self.solver
            )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            apply_exchange_to_orbitals(
                self.orbitals, self.occupations, self.volume, self.solver, batch_size=0
            )
        # No occupied orbital: nothing to solve, no operator.
        empty = apply_exchange_to_orbitals(
            self.orbitals, np.zeros(5), self.volume, self.solver
        )
        self.assertEqual(empty.pair_solves, 0)
        self.assertEqual(empty.energy, 0.0)
        self.assertEqual(exchange_band_energy(None, self.orbitals, self.occupations), 0.0)


if __name__ == "__main__":
    unittest.main()
