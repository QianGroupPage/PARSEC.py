"""Kronecker-product Poisson solver of pair densities (Hartree/kronecker.py).

References used here:

* the run's own operators: the sparse negative Laplacian of a box domain and
  the conjugate-gradient solve with the same multipole boundary values;
* the complex multipole expansion of :mod:`parsec_python.Hartree.poisson`;
* closed forms for Gaussian charges ``g_a = (a/pi)**1.5 exp(-a r**2)``:
  ``1/r -> erf(sqrt(a) r)/r``, ``erf(w r)/r -> erf(sqrt(c) r)/r`` with
  ``1/c = 1/a + 1/w**2``, and ``erfc = 1/r - erf``.
"""

from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import erf

from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree import density_multipoles, solve_hartree
from parsec_python.Hartree.kronecker import (
    KroneckerPoisson,
    dirichlet_second_derivative_matrix,
    real_harmonics,
    solid_harmonic_coefficients,
    spectral_function,
)
from parsec_python.Laplacian import apply_negative_laplacian_boundary, build_negative_laplacian
from parsec_python.V_xc.exact_exchange import exact_exchange_energy, reference_pair_poisson
from parsec_python.models import GridSettings, HartreeSettings


def gaussian(points, center, exponent):
    offset = points - np.asarray(center)
    return (exponent / np.pi) ** 1.5 * np.exp(-exponent * np.einsum("ij,ij->i", offset, offset))


def gaussian_potential(points, center, exponent, kernel="coulomb", omega=None):
    """Rydberg potential of a unit Gaussian charge for the three kernels."""

    distance = np.linalg.norm(points - np.asarray(center), axis=1)
    coulomb = erf(np.sqrt(exponent) * distance) / distance
    if kernel == "coulomb":
        return 2.0 * coulomb
    combined = exponent * omega**2 / (exponent + omega**2)
    long_range = erf(np.sqrt(combined) * distance) / distance
    return 2.0 * (long_range if kernel == "erf" else coulomb - long_range)


# A charge (one electron) and a neutral pair density with a dipole.
CHARGE = ((1.0, (0.5, -0.3, 0.2), 1.2),)
DIPOLE = ((1.0, (0.8, 0.0, 0.0), 1.0), (-1.0, (-0.8, 0.3, 0.0), 1.0))


def densities_and_potentials(points, kernel="coulomb", omega=None):
    densities, potentials = [], []
    for terms in (CHARGE, DIPOLE):
        densities.append(sum(w * gaussian(points, c, a) for w, c, a in terms))
        potentials.append(
            sum(w * gaussian_potential(points, c, a, kernel, omega) for w, c, a in terms)
        )
    return np.column_stack(densities), np.column_stack(potentials)


class OperatorTests(unittest.TestCase):
    def test_kronecker_sum_is_the_negative_laplacian_of_a_box(self) -> None:
        grid = build_cluster_grid(
            GridSettings(
                spacing=0.5, radius=4.0, expansion_order=6, domain_shape="box",
                box_lengths=(7.0, 6.0, 5.0),
            )
        )
        solver = KroneckerPoisson(grid, automatic_padding=False)
        nx, ny, nz = solver.box_shape
        self.assertEqual(nx * ny * nz, grid.size)
        dx, dy, dz = (
            sp.csr_matrix(dirichlet_second_derivative_matrix(n, 6, 0.5)) for n in solver.box_shape
        )
        ix, iy, iz = (sp.identity(n, format="csr") for n in solver.box_shape)
        kronecker = -(
            sp.kron(sp.kron(dx, iy), iz) + sp.kron(sp.kron(ix, dy), iz) + sp.kron(sp.kron(ix, iy), dz)
        )
        vector = np.random.default_rng(3).normal(size=grid.size)
        box = np.zeros(nx * ny * nz)
        box[solver.box_index] = vector
        expected = build_negative_laplacian(grid) @ vector
        np.testing.assert_allclose((kronecker @ box)[solver.box_index], expected, atol=1e-11)

    def test_spectral_functions(self) -> None:
        values = np.array([1.0e-6, 0.3, 4.0, 90.0])
        np.testing.assert_allclose(spectral_function("coulomb", values), 1.0 / values)
        omega = 0.2
        erfc = spectral_function("erfc", values, omega)
        erf_part = spectral_function("erf", values, omega)
        np.testing.assert_allclose(erfc + erf_part, 1.0 / values, rtol=1e-14)
        # (1 - exp(-k^2/(4 w^2)))/k^2 -> 1/(4 w^2) as k -> 0: finite short range.
        self.assertAlmostEqual(erfc[0] * 4.0 * omega**2, 1.0, places=5)
        with self.assertRaises(ValueError):
            spectral_function("erfc", values)
        with self.assertRaises(ValueError):
            spectral_function("coulomb", np.array([0.0]))


class MultipoleTests(unittest.TestCase):
    def test_polynomial_coefficients_are_the_real_solid_harmonics(self) -> None:
        order = 9
        points = np.random.default_rng(1).normal(size=(64, 3))
        size = order + 1
        powers = np.arange(size)
        monomials = (
            (points[:, 0, None] ** powers)[:, :, None, None]
            * (points[:, 1, None] ** powers)[:, None, :, None]
            * (points[:, 2, None] ** powers)[:, None, None, :]
        ).reshape(len(points), -1)
        values = monomials @ solid_harmonic_coefficients(order).reshape(size * size, -1).T
        radius = np.linalg.norm(points, axis=1)
        degree = np.repeat(powers, 2 * powers + 1)
        expected = real_harmonics(points, order) * radius[:, None] ** degree
        np.testing.assert_allclose(values, expected, atol=1e-12 * np.abs(expected).max())

    def test_moments_and_exterior_values_match_the_complex_expansion(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.45, radius=6.0, expansion_order=8))
        density, _ = densities_and_potentials(grid.coordinates)
        solver = KroneckerPoisson(grid, multipole_order=7)
        box = np.zeros((2, solver.box_size))
        box[:, solver.box_index] = density.T
        moments = solver.scaled_moments(box.reshape(2, *solver.box_shape))
        far = np.random.default_rng(2).normal(size=(40, 3))
        far *= 8.0 / np.linalg.norm(far, axis=1)[:, None]
        for column in range(2):
            expansion = density_multipoles(density[:, column], grid, 7)
            np.testing.assert_allclose(
                solver._exterior_basis(far, None) @ moments[column],
                expansion.potential(far),
                rtol=1e-12, atol=1e-15,
            )


class CoulombTests(unittest.TestCase):
    def test_box_domain_equals_conjugate_gradients_with_the_same_boundary(self) -> None:
        grid = build_cluster_grid(
            GridSettings(
                spacing=0.4, radius=7.0, expansion_order=8, domain_shape="box",
                box_lengths=(12.0, 11.0, 10.0),
            )
        )
        density, _ = densities_and_potentials(grid.coordinates)
        # The bounding box itself: the discrete problem of the box domain.
        solver = KroneckerPoisson(grid, multipole_order=6, automatic_padding=False)
        potentials = solver.solve_batch(density)
        laplacian = build_negative_laplacian(grid)
        for column in range(2):
            expansion = density_multipoles(density[:, column], grid, 6)
            rhs = apply_negative_laplacian_boundary(
                8.0 * np.pi * density[:, column], grid, expansion.potential
            )
            reference, info = spla.cg(laplacian, rhs, rtol=1e-13, maxiter=20000)
            self.assertEqual(info, 0)
            np.testing.assert_allclose(potentials[:, column], reference, atol=1e-10)

    def test_sphere_agrees_with_the_hartree_solver_and_the_gaussian(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.4, radius=8.0, expansion_order=8))
        laplacian = build_negative_laplacian(grid)
        density, exact = densities_and_potentials(grid.coordinates)
        potentials = KroneckerPoisson(grid, multipole_order=9).solve_batch(density)
        settings = HartreeSettings(multipole_order=9, relative_tolerance=1e-12, max_iterations=10000)
        for column in range(2):
            reference = solve_hartree(density[:, column], grid, laplacian, settings).potential
            np.testing.assert_allclose(potentials[:, column], reference, atol=1e-7)
        # Finite-difference error of order 8 at h = 0.4 for these Gaussians.
        self.assertLess(np.abs(potentials - exact).max(), 4e-4)

    def test_convergence_with_spacing_and_order(self) -> None:
        errors = {}
        for spacing in (0.5, 0.35):
            for order in (4, 8):
                grid = build_cluster_grid(
                    GridSettings(spacing=spacing, radius=8.0, expansion_order=order)
                )
                density, exact = densities_and_potentials(grid.coordinates)
                potentials = KroneckerPoisson(grid).solve_batch(density)
                energy = np.einsum("ij,ij->j", density, potentials)
                exact_energy = np.einsum("ij,ij->j", density, exact)
                errors[spacing, order] = np.abs(energy / exact_energy - 1.0).max()
        self.assertLess(errors[0.35, 4], errors[0.5, 4] / 3.0)
        self.assertLess(errors[0.35, 8], errors[0.5, 8] / 5.0)
        self.assertLess(errors[0.35, 8], errors[0.35, 4] / 10.0)

    def test_batches_padding_and_single_densities(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=6.0, expansion_order=6))
        density, _ = densities_and_potentials(grid.coordinates)
        batch = np.column_stack([density, density[:, 0] - 2.0 * density[:, 1]])
        solver = KroneckerPoisson(grid)
        whole = solver.solve_batch(batch)
        np.testing.assert_allclose(whole[:, 2], whole[:, 0] - 2.0 * whole[:, 1], atol=1e-12)
        small = KroneckerPoisson(grid, batch_memory=1)
        self.assertEqual(small.batch_size, 1)
        np.testing.assert_allclose(small.solve_batch(batch), whole, atol=1e-13)
        np.testing.assert_allclose(solver(density[:, 1]), whole[:, 1], atol=1e-13)
        padded = KroneckerPoisson(grid, padding=1.2)
        self.assertEqual(padded.box_shape[0], solver.box_shape[0] + 6)
        np.testing.assert_allclose(padded.solve_batch(density), whole[:, :2], atol=1e-5)

    def test_exchange_energy_matches_the_conjugate_gradient_solver(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.4, radius=8.0, expansion_order=8))
        laplacian = build_negative_laplacian(grid)
        settings = HartreeSettings(multipole_order=6, relative_tolerance=1e-10, max_iterations=4000)
        points = grid.coordinates
        envelope = np.exp(-0.4 * np.einsum("ij,ij->i", points, points))
        orbitals = np.column_stack([envelope, points[:, 0] * envelope, points[:, 2] * envelope])
        orbitals /= np.linalg.norm(orbitals, axis=0)
        occupations = np.array([1.0, 1.0, 0.5])
        reference = exact_exchange_energy(
            orbitals, occupations, grid.volume_element,
            reference_pair_poisson(grid, laplacian, settings),
        )
        kronecker = exact_exchange_energy(
            orbitals, occupations, grid.volume_element, KroneckerPoisson(grid, multipole_order=6)
        )
        self.assertAlmostEqual(kronecker.energy / reference.energy, 1.0, delta=1e-7)
        self.assertEqual(kronecker.pair_solves, 6)


class ScreenedKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(GridSettings(spacing=0.4, radius=9.0, expansion_order=8))
        cls.density, cls.coulomb_exact = densities_and_potentials(cls.grid.coordinates)
        cls.coulomb = KroneckerPoisson(cls.grid).solve_batch(cls.density)

    def test_split_treatment_matches_the_gaussian_potentials(self) -> None:
        coulomb_error = np.abs(self.coulomb - self.coulomb_exact).max(axis=0)
        for omega in (0.11, 0.4):
            _, erf_exact = densities_and_potentials(self.grid.coordinates, "erf", omega)
            _, erfc_exact = densities_and_potentials(self.grid.coordinates, "erfc", omega)
            long_range = KroneckerPoisson(self.grid, kernel="erf", omega=omega).solve_batch(self.density)
            short_range = KroneckerPoisson(self.grid, kernel="erfc", omega=omega).solve_batch(self.density)
            # The smooth part has no finite-difference error to speak of at w = 0.11.
            tolerance = 1e-10 if omega < 0.2 else 1e-6
            self.assertLess(np.abs(long_range - erf_exact).max(), tolerance)
            # The short-range part carries the error of the Coulomb solve and nothing else.
            self.assertLess(
                np.abs(np.abs(short_range - erfc_exact).max(axis=0) - coulomb_error).max(), 2e-6
            )
            np.testing.assert_allclose(long_range + short_range, self.coulomb, atol=1e-11)

    def test_weighted_kernels_are_the_combinations_of_the_solves(self) -> None:
        omega = 0.2
        short = KroneckerPoisson(self.grid, kernel="erfc", omega=omega).solve_batch(self.density)
        long = KroneckerPoisson(self.grid, kernel="erf", omega=omega).solve_batch(self.density)
        # alpha/r + beta erfc(w r)/r (HSE-like and CAM-like weights) and a/r + b erf(w r)/r.
        for kernel, weights, expected in (
            ("erfc", (0.0, 0.25), 0.25 * short),
            ("erfc", (0.19, 0.46), 0.19 * self.coulomb + 0.46 * short),
            ("erf", (1.0, -1.0), short),
            ("erf", (0.2, 0.8), 0.2 * self.coulomb + 0.8 * long),
        ):
            solver = KroneckerPoisson(self.grid, kernel=kernel, omega=omega, weights=weights)
            np.testing.assert_allclose(solver.solve_batch(self.density), expected, atol=1e-11)
        with self.assertRaisesRegex(ValueError, "screened kernel"):
            KroneckerPoisson(self.grid, weights=(1.0, 0.5))
        with self.assertRaisesRegex(ValueError, "split"):
            KroneckerPoisson(self.grid, kernel="erf", omega=omega, screening="spectral", weights=(1.0, 0.5))

    def test_spectral_function_on_the_box_misses_the_isolated_boundary(self) -> None:
        _, erfc_exact = densities_and_potentials(self.grid.coordinates, "erfc", 0.11)
        spectral = KroneckerPoisson(
            self.grid, kernel="erfc", omega=0.11, screening="spectral"
        ).solve_batch(self.density)
        split = KroneckerPoisson(self.grid, kernel="erfc", omega=0.11).solve_batch(self.density)
        self.assertGreater(
            np.abs(spectral - erfc_exact).max(), 10.0 * np.abs(split - erfc_exact).max()
        )

    def test_screened_exchange_of_a_gaussian_orbital(self) -> None:
        # The pair density of the normalized orbital (2a/pi)**0.75 exp(-a r**2)
        # is the Gaussian g_p, p = 2a; (ss|ss) = (g_p * g_p * K)(0).
        grid = self.grid
        points = grid.coordinates
        exponent = 0.6
        orbital = np.exp(-exponent * np.einsum("ij,ij->i", points, points))
        orbital /= np.linalg.norm(orbital)
        omega = 0.11
        result = exact_exchange_energy(
            orbital[:, None], np.ones(1), grid.volume_element,
            KroneckerPoisson(grid, kernel="erfc", omega=omega),
        )
        pair = 2.0 * exponent
        combined = pair * omega**2 / (pair + omega**2)
        # Hartree units: 1/r gives 2 sqrt(p/(2 pi)); erf(w r)/r gives
        # 2 sqrt(e/pi) with 1/e = 2/p + 1/w**2 = 1/c + 1/p.
        coulomb = 2.0 * np.sqrt(pair / (2.0 * np.pi))
        long_range = 2.0 * np.sqrt(pair * combined / (np.pi * (pair + combined)))
        analytic = -2.0 * (coulomb - long_range)
        self.assertAlmostEqual(result.energy / analytic, 1.0, delta=2e-5)


class BoundaryConvergenceTests(unittest.TestCase):
    """Automatic padding: convergent exterior expansion and the smearing cap (added 2026-11-06)."""

    # Two unit Gaussians along the long axis of elongated domains: density
    # far beyond the side faces.
    PAIR = ((1.0, (-8.0, 0.0, 0.0), 1.0), (1.0, (8.0, 0.0, 0.0), 1.0))

    def _case(self, grid, kernel, omega, **keywords):
        points = grid.coordinates
        density = sum(w * gaussian(points, c, a) for w, c, a in self.PAIR)
        exact = sum(w * gaussian_potential(points, c, a, kernel, omega) for w, c, a in self.PAIR)
        solver = KroneckerPoisson(grid, kernel=kernel, omega=omega, multipole_order=9, **keywords)
        return solver, solver.solve_batch(density), exact, density

    def test_elongated_domains_are_padded_until_the_expansion_converges(self) -> None:
        domains = (
            GridSettings(spacing=0.5, radius=13.0, expansion_order=6, domain_shape="box",
                         box_lengths=(26.0, 10.0, 10.0)),
            GridSettings(spacing=0.5, radius=13.0, expansion_order=6, domain_shape="cylinder",
                         domain_parameters=(5.0, 26.0, 1)),
        )
        for settings in domains:
            grid = build_cluster_grid(settings)
            for kernel, omega, tolerance in (("coulomb", None, 2e-3), ("erfc", 0.11, 2e-3), ("erf", 0.4, 5e-4)):
                with self.subTest(shape=settings.domain_shape, kernel=kernel):
                    solver, potential, exact, density = self._case(grid, kernel, omega)
                    below, above = solver.automatic_padding
                    self.assertGreater(min(below[1:] + above[1:]), 10)
                    # Every exterior point farther out than every active point,
                    # and no more padding than that needs.
                    nearest = [np.abs(axis).min() for axis in solver.axes]
                    for axis in range(3):
                        point = list(nearest)
                        point[axis] = np.abs(solver.axes[axis]).max() + settings.spacing
                        self.assertGreater(np.linalg.norm(point), solver.source_radius)
                    corner = max(np.abs(axis).max() for axis in solver.axes)
                    self.assertLessEqual(corner, solver.source_radius + settings.spacing)
                    # Back to the finite-difference error (1.1e-3 Ry for the Coulomb kernel).
                    self.assertLess(np.abs(potential - exact).max(), tolerance)
                    if settings.domain_shape == "box" and kernel != "erfc":
                        # A box padded by six more bohr: the same within that error.
                        _, heavy, _, _ = self._case(grid, kernel, omega, padding=6.0)
                        self.assertLess(np.abs(potential - heavy).max(), tolerance)
            # The bounding box alone: the expansion diverges beyond the side faces.
            _, unpadded, exact, _ = self._case(grid, "coulomb", None, automatic_padding=False)
            self.assertGreater(np.abs(unpadded - exact).max(), 0.5)

    def test_elongated_exchange_matches_the_direct_boundary_solve(self) -> None:
        # Coulomb pair potentials against the conjugate gradients with the direct
        # boundary sum, the Hartree solver of a box domain.
        grid = build_cluster_grid(
            GridSettings(spacing=0.5, radius=12.0, expansion_order=6, domain_shape="box",
                         box_lengths=(24.0, 8.0, 8.0))
        )
        points = grid.coordinates
        first = gaussian(points, (-4.5, 0.0, 0.0), 0.8)
        second = gaussian(points, (4.5, 0.3, 0.0), 0.8)
        densities = np.column_stack([first + second, first - second])
        potentials = KroneckerPoisson(grid, multipole_order=9).solve_batch(densities)
        settings = HartreeSettings(
            multipole_order=9, relative_tolerance=1e-12, max_iterations=20000, boundary_method="direct"
        )
        laplacian = build_negative_laplacian(grid)
        for column in range(2):
            reference = solve_hartree(densities[:, column], grid, laplacian, settings).potential
            self.assertLess(np.abs(potentials[:, column] - reference).max(), 2e-5)

    def test_spheres_are_not_padded(self) -> None:
        # Coulomb: never.  Screened kernels: not while the radius exceeds
        # h sqrt(x_L / 0.25) (7.9 bohr at h = 0.5 and L = 9), the smearing cap.
        small = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6))
        self.assertEqual(KroneckerPoisson(small, multipole_order=20).automatic_padding, ((0, 0, 0), (0, 0, 0)))
        for settings in (
            GridSettings(spacing=0.5, radius=8.0, expansion_order=6),
            GridSettings(spacing=0.45, radius=7.5, expansion_order=8, shift=(0.0, 0.0, 0.0)),
            GridSettings(spacing=0.35, radius=6.0, expansion_order=8, shift=(0.5, 0.0, 0.25)),
        ):
            grid = build_cluster_grid(settings)
            density, _ = densities_and_potentials(grid.coordinates)
            for kernel, omega in (("coulomb", None), ("erfc", 0.11)):
                solver = KroneckerPoisson(grid, kernel=kernel, omega=omega)
                self.assertEqual(solver.automatic_padding, ((0, 0, 0), (0, 0, 0)))
                fixed = KroneckerPoisson(grid, kernel=kernel, omega=omega, automatic_padding=False)
                np.testing.assert_array_equal(solver.solve_batch(density), fixed.solve_batch(density))

    def test_smearing_is_capped_by_the_grid(self) -> None:
        # A neutral pair in a sphere of 5 bohr at
        # h = 0.5, erf at w = 0.4 with L = 20 (6.9e-4 Ry before the cap).
        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=5.0, expansion_order=8))
        points = grid.coordinates
        terms = ((1.0, (1.5, -1.0, 0.8), 1.5), (-1.0, (-1.2, 1.3, -0.9), 1.5))
        density = sum(w * gaussian(points, c, a) for w, c, a in terms)
        exact = sum(w * gaussian_potential(points, c, a, "erf", 0.4) for w, c, a in terms)
        solver = KroneckerPoisson(grid, kernel="erf", omega=0.4, multipole_order=20)
        self.assertLessEqual(solver.smearing_exponent * 0.5**2, 0.25 + 1e-12)
        self.assertGreater(max(solver.automatic_padding[0]), 0)
        self.assertLess(np.abs(solver.solve_batch(density) - exact).max(), 1e-6)
        # Without the cap the narrow smearing aliases on the lattice.
        uncapped = KroneckerPoisson(grid, kernel="erf", omega=0.4, multipole_order=20, smearing_limit=1e9)
        self.assertGreater(np.abs(uncapped.solve_batch(density) - exact).max(), 1e-4)


class RefusalTests(unittest.TestCase):
    def test_what_is_refused(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6))
        with self.assertRaisesRegex(ValueError, "screening parameter"):
            KroneckerPoisson(grid, kernel="erfc")
        with self.assertRaisesRegex(ValueError, "no screening parameter"):
            KroneckerPoisson(grid, omega=0.1)
        with self.assertRaisesRegex(ValueError, "unknown kernel"):
            KroneckerPoisson(grid, kernel="yukawa", omega=0.1)
        with self.assertRaisesRegex(ValueError, "unknown screening"):
            KroneckerPoisson(grid, kernel="erf", omega=0.1, screening="pad")
        with self.assertRaises(TypeError):
            KroneckerPoisson(object())
        solver = KroneckerPoisson(grid)
        with self.assertRaisesRegex(ValueError, "does not match"):
            solver(np.zeros(grid.size + 1))
        with self.assertRaisesRegex(ValueError, "shape"):
            solver.solve_batch(np.zeros((grid.size + 1, 2)))


if __name__ == "__main__":
    unittest.main()
